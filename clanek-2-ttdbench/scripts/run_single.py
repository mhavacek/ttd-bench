#!/usr/bin/env python3
"""Run a single matrix cell locally (smoke test / HPC array-job entry point).

Examples:
  # local smoke test with the synthetic scenario (auto-selects RTSP if
  # ffmpeg + mediamtx are available, otherwise falls back to direct mode):
  python scripts/run_single.py --synthetic

  # one specific cell of the matrix (used by the HPC array jobs):
  python scripts/run_single.py --cell-index 17

  # list the whole matrix:
  python scripts/run_single.py --list-cells
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # clanek-2-ttdbench/
ROOT = REPO.parent                                     # branch-A root
sys.path.insert(0, str(ROOT / "common" / "src"))

from ttdbench.config import ExperimentConfig  # noqa: E402
from ttdbench.metrics import confidence_trace, parse_run  # noqa: E402
from ttdbench.runner import run_cell  # noqa: E402


def rtsp_available(cfg: ExperimentConfig) -> bool:
    mediamtx = cfg.root / cfg.raw["replay"]["mediamtx_binary"]
    have_mtx = mediamtx.exists() or shutil.which("mediamtx") is not None
    return shutil.which("ffmpeg") is not None and have_mtx


def smoke_figure(log_path: Path, out_pdf: Path, cfg: ExperimentConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ttdbench.logio import read_log

    trace = confidence_trace(log_path)
    # a run may legitimately produce no usable trace (0 frames processed, or no
    # onset reached) — skip the debug figure instead of crashing the whole run
    # (this must never fail an HPC array job; the raw log is already written)
    if trace.empty or "t_rel_onset_ms" not in trace.columns \
            or trace["t_rel_onset_ms"].isna().all():
        print("figure  -> skipped (no plottable frames/onset in this run)")
        return

    t_alarm_rel = None
    onset_t = None
    for rec in read_log(log_path):
        if rec.get("type") == "onset_ref":
            onset_t = rec["t"]
        elif rec.get("type") == "alarm" and onset_t is not None:
            t_alarm_rel = (rec["t"] - onset_t) / 1e6

    thr = float(cfg.alarm["confidence_threshold"])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(trace["t_rel_onset_ms"], trace["max_conf"], linewidth=1.0,
            label="max weapon confidence")
    ax.axhline(thr, color="grey", linestyle=":", label=f"threshold {thr}")
    ax.axvline(0, color="green", linestyle="--", label="$t_{onset}$")
    if t_alarm_rel is not None:
        ax.axvline(t_alarm_rel, color="red", linestyle="--",
                   label=f"$t_{{alarm}}$ (TTD = {t_alarm_rel:.0f} ms)")
    ax.set_xlabel("time relative to onset [ms]")
    ax.set_ylabel("confidence")
    ax.legend(fontsize=8, frameon=False)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    print(f"figure  -> {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "experiment.yaml"))
    ap.add_argument("--cell-index", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="run the first synthetic-model cell (dev smoke test)")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="validate checkpoints and scenario videos for the whole "
                         "matrix without running anything (do this before qsub)")
    ap.add_argument("--replay-mode", choices=["rtsp", "direct", "auto"], default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None,
                    help="override the cell's inference device (e.g. cpu, mps, "
                         "cuda:0) — for local testing on non-CUDA machines")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    cells = cfg.enumerate_cells()

    if args.list_cells:
        for c in cells:
            print(f"{c.index:4d}  {c.run_id()}")
        print(f"total: {len(cells)} cells")
        return 0

    if args.preflight:
        from ttdbench.runner import preflight

        problems = preflight(cfg)
        print(f"config:    {args.config}")
        print(f"models:    {', '.join(cfg.active_models)}")
        print(f"scenarios: {len(cfg.active_scenarios)}")
        print(f"matrix:    {len(cells)} cells  ->  qsub -J 0-{len(cells) - 1}")
        if problems:
            print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("\npreflight OK — ready to submit")
        return 0

    if args.synthetic:
        candidates = [c for c in cells if c.model == "synthetic" and not c.remote]
        if not candidates:
            print("no synthetic cell in matrix — add 'synthetic' to active_models",
                  file=sys.stderr)
            return 2
        cell = candidates[0]
        video = cfg.root / cfg.scenario_spec(cell.scenario)["file"]
        if not video.exists():
            print("generating synthetic scenarios ...")
            import subprocess
            subprocess.run([sys.executable, str(REPO / "scripts" / "make_synthetic.py")],
                           check=True)
    elif args.cell_index is not None:
        cell = cfg.cell(args.cell_index)
    else:
        ap.error("one of --synthetic / --cell-index / --list-cells is required")
        return 2

    if args.device:
        import dataclasses
        cell = dataclasses.replace(cell, device=args.device)
        print(f"device override: {args.device}")

    mode = args.replay_mode
    if mode == "auto":
        mode = "rtsp" if rtsp_available(cfg) else "direct"
        print(f"replay mode: {mode} (auto)")

    # --out-dir governs the WHOLE output tree, not just the raw logs. It used
    # to move only those, while the csv and figure went to the config's
    # results_dir regardless -- so a repair run writing raw logs to
    # results-scvd-rerun would still have dropped its CSVs into the frozen
    # results-scvd/csv. RESULTS_SUBDIR in hpc/pbs_array.sh is expected to
    # isolate a run completely, and now does.
    if args.out_dir:
        out_dir = Path(args.out_dir)
        results_root = out_dir.parent        # conventionally <results-tree>/raw
    else:
        results_root = cfg.root / cfg.raw["results_dir"]
        out_dir = results_root / "raw"
    print(f"running cell {cell.index}: {cell.run_id()}")
    try:
        log_path = run_cell(cfg, cell, out_dir, replay_mode=mode)
    except FileNotFoundError as e:
        # missing checkpoint / scenario video: a configuration problem, not a
        # crash — report it plainly (an HPC array would otherwise emit hundreds
        # of identical tracebacks)
        print(f"\nCONFIGURATION ERROR\n{e}", file=sys.stderr)
        return 2
    print(f"raw log -> {log_path}")

    # post-hoc metrics for this one run
    m = parse_run(log_path, timeout_s=float(cfg.alarm["timeout_s"]),
                  alarm_cfg=cfg.alarm)
    import pandas as pd

    csv_dir = results_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{cell.run_id()}.csv"
    pd.DataFrame([m]).to_csv(csv_path, index=False)
    print(f"csv     -> {csv_path}")

    def _show(keys):
        for k in keys:
            v = m.get(k)
            print(f"  {k:24s} {v if not isinstance(v, float) else round(v, 2)}")

    # Per-frame latency decomposition: always populated (Phase-1 deliverable),
    # independent of whether an alarm fired.
    print("\n--- per-frame latency (all processed frames) ---")
    _show(("n_frames_processed", "drop_rate", "pf_dt_acq_ms", "pf_dt_transfer_ms",
           "pf_dt_infer_ms", "pf_dt_post_ms", "pf_latency_ms"))

    # Event-level TTD: only defined on a hit (needs a weapon detection + onset).
    print("\n--- event-level TTD ---")
    if m.get("hit"):
        _show(("ttd_ms", "dt_acq_ms", "dt_transfer_ms", "dt_infer_ms",
               "dt_post_ms", "dt_alarm_ms"))
    else:
        print("  hit                      False  (no alarm within timeout)")
        print("  -> event-level TTD/decomposition undefined for this run.")
        print("     Expected in Phase 1 (COCO has no 'weapon' class); the")
        print("     per-frame latency above is the Phase-1 result.")

    # the per-run debug figure is a convenience, never worth failing a run over
    fig_dir = results_root / "figures"
    try:
        smoke_figure(log_path, fig_dir / f"smoke_{cell.run_id()}.pdf", cfg)
    except Exception as e:
        print(f"figure  -> skipped ({type(e).__name__}: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

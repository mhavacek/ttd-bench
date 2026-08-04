#!/usr/bin/env python3
"""Produce the final paper assets from the master CSV.

  python scripts/make_paper_assets.py [--master results/csv/master.csv]

Outputs:
  results/tables/summary_ttd.csv / .tex        descriptives (mean±SD, CI, TTD50/90, miss rate)
  results/tables/decomposition.csv / .tex      mean component decomposition
  results/tables/pairwise_tests.csv / .tex     paired tests + Bonferroni + effect sizes
  results/figures/fig_ttd_decomposition.pdf    (a) stacked component bars
  results/figures/fig_ttd_violin.pdf           (b) box/violin per config × model
  results/figures/fig_ttd_cdf.pdf              (c) CDF with TTD50/TTD90
  results/figures/fig_missrate_heatmap.pdf     (d) miss-rate heatmap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent          # clanek-2-ttdbench/
ROOT = REPO.parent                                     # branch-A root
sys.path.insert(0, str(ROOT / "common" / "src"))

from ttdbench import report, stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(REPO / "results" / "csv" / "master.csv"))
    ap.add_argument("--out-root", default=str(REPO / "results"))
    ap.add_argument("--phase1", action="store_true",
                    help="produce per-frame latency-decomposition assets "
                         "(no event-level TTD / miss rate)")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    if df.empty:
        print("master CSV is empty", file=sys.stderr)
        return 1

    out_root = Path(args.out_root)
    tab_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)

    if args.phase1:
        return _phase1(df, tab_dir, fig_dir)

    # ---- descriptives -----------------------------------------------------
    summary = stats.summarize(df, metric="ttd_ms", group="config_label")
    summary.to_csv(tab_dir / "summary_ttd.csv", index=False)
    (tab_dir / "summary_ttd.tex").write_text(
        stats.summary_to_latex(
            summary,
            caption="Time-to-Detection per deployment configuration "
                    "(mean $\\pm$ SD, 95\\% CI, medians and 90th percentiles over "
                    "all hit events).",
            label="tab:ttd-summary",
        )
    )

    decomp = stats.decomposition_summary(df, group="config_label")
    decomp.to_csv(tab_dir / "decomposition.csv", index=False)
    dtex = decomp.copy()
    for c in dtex.columns[1:]:
        dtex[c] = dtex[c].map(lambda v: f"{v:.1f}")
    (tab_dir / "decomposition.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mean TTD decomposition per configuration [ms].}\n"
        "\\label{tab:ttd-decomposition}\n"
        + dtex.to_latex(index=False, escape=True)
        + "\\end{table}\n"
    )

    # ---- inferential ------------------------------------------------------
    tests = stats.pairwise_tests(df, metric="ttd_ms", group="config_label")
    tests.to_csv(tab_dir / "pairwise_tests.csv", index=False)
    (tab_dir / "pairwise_tests.tex").write_text(
        stats.tests_to_latex(
            tests,
            caption="Pairwise comparisons of TTD between deployment "
                    "configurations (paired by model $\\times$ scenario $\\times$ "
                    "repetition; Bonferroni-corrected).",
            label="tab:ttd-tests",
        )
    )

    # ---- figures ----------------------------------------------------------
    figs = report.make_all_figures(df, fig_dir)

    print("tables  ->", tab_dir)
    for f in sorted(tab_dir.glob("*")):
        print("   ", f.name)
    print("figures ->", fig_dir)
    for f in figs:
        print("   ", f.name)
    return 0


def _phase1(df: pd.DataFrame, tab_dir: Path, fig_dir: Path) -> int:
    """Per-frame latency-decomposition assets (Phase-1 pilot)."""
    import numpy as np

    pf_cols = ["pf_dt_acq_ms", "pf_dt_transfer_ms", "pf_dt_infer_ms",
               "pf_dt_post_ms", "pf_latency_ms", "drop_rate"]
    missing = [c for c in pf_cols if c not in df.columns]
    if missing:
        print(f"master CSV lacks per-frame columns {missing} — re-run "
              "scripts/aggregate.py with the updated metrics", file=sys.stderr)
        return 1

    # Deviation D1 (docs/ANALYSIS-PLAN-PHASE2.md): runs that processed zero
    # frames are infrastructure failures of the job, not slow pipelines. They
    # carry drop_rate = 0.0 rather than NaN, so leaving them in halves the
    # headline drop rate (edge-sim 0.82 -> 0.44, 4G 0.66 -> 0.34) and dilutes
    # dt_transfer. Rates must be computed on processed runs only.
    n_all = len(df)
    df = df[df["n_frames_processed"] > 0].copy()
    n_excl = n_all - len(df)
    if n_excl:
        print(f"D1: excluded {n_excl}/{n_all} runs ({100 * n_excl / n_all:.1f} %) "
              f"that processed zero frames; {len(df)} valid runs remain")
    if df.empty:
        print("no valid runs left after the D1 filter", file=sys.stderr)
        return 1

    g = df.groupby(["config_label", "model"], sort=True)
    summary = g[pf_cols].mean().reset_index()
    summary["n_runs"] = g.size().values
    summary["drop_rate_pct"] = summary["drop_rate"] * 100
    summary.to_csv(tab_dir / "phase1_perframe_summary.csv", index=False)

    # LaTeX table
    show = summary[["config_label", "model", "pf_dt_acq_ms", "pf_dt_transfer_ms",
                    "pf_dt_infer_ms", "pf_dt_post_ms", "pf_latency_ms",
                    "drop_rate_pct"]].copy()
    for c in show.columns[2:]:
        show[c] = show[c].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "--")
    show.columns = ["Config", "Model", r"$\Delta t_{\mathrm{acq}}$",
                    r"$\Delta t_{\mathrm{transfer}}$", r"$\Delta t_{\mathrm{infer}}$",
                    r"$\Delta t_{\mathrm{post}}$", "Total", "Drop \\%"]
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Phase-1 mean per-frame latency decomposition [ms] per "
        "deployment configuration and model (COCO-pretrained detectors, "
        "UCF-Crime CCTV footage).}\n\\label{tab:phase1-perframe}\n"
        + show.to_latex(index=False, escape=False,
                        column_format="ll" + "r" * (len(show.columns) - 2))
        + "\\end{table}\n"
    )
    (tab_dir / "phase1_perframe_summary.tex").write_text(tex)

    figs = report.make_phase1_figures(df, fig_dir)
    print("Phase-1 tables  ->", tab_dir)
    for f in ("phase1_perframe_summary.csv", "phase1_perframe_summary.tex"):
        print("   ", f)
    print("Phase-1 figures ->", fig_dir)
    for f in figs:
        print("   ", f.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

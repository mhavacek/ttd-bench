#!/usr/bin/env python3
"""Post-hoc confidence-threshold sweep and FP-matched model comparison.

  python scripts/threshold_sweep.py --build      # cache per-frame traces once
  python scripts/threshold_sweep.py              # sweep + matched comparison

WHY
---
The experiment logged every detection down to conf 0.1, while the alarm used
0.5. Comparing models at one fixed threshold confounds ARCHITECTURE with
CALIBRATION: a model whose scores are shifted upwards fires more often on
everything — weapons and parked cars alike. Any claim of the form "generation X
detects better than Y" therefore has to be made at a matched operating point.

This script re-derives, entirely post-hoc from the frozen logs, the alarm
outcome at a grid of thresholds, and then compares models at thresholds
calibrated to a COMMON pre-onset false-alarm rate.

Cache format (results-scvd/analysis/traces.parquet): one row per processed
frame per run — run_id, t_rel_onset_ms, max weapon-class confidence — plus a
per-run table with onset/follow-up. Rebuild with --build after new runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root
sys.path.insert(0, str(REPO / "src"))

WEAPON = {"weapon", "pistol", "knife", "gun", "rifle", "handgun"}
NS_PER_MS = 1e6


def build_cache(raw_dir: Path, out_dir: Path) -> None:
    frames_rows, run_rows = [], []
    logs = sorted(raw_dir.glob("*.jsonl"))
    for n, p in enumerate(logs, 1):
        if n % 250 == 0:
            print(f"  {n}/{len(logs)}", flush=True)
        onset = None
        recs = []
        header = None
        for line in open(p):
            d = json.loads(line)
            t = d.get("type")
            if t == "frame":
                confs = [x["conf"] for x in d["detections"]
                         if x["cls"].lower() in WEAPON]
                recs.append((d["t_post"], max(confs) if confs else 0.0))
            elif t == "onset_ref":
                onset = d["t"]
            elif t == "header":
                header = d
        if header is None or onset is None or not recs:
            continue
        rid = p.stem
        cell = header["cell"]
        recs.sort()
        last = recs[-1][0]
        run_rows.append({
            "run_id": rid, "deployment": cell["deployment"],
            "network_profile": cell["network_profile"] or "none",
            "model": cell["model"], "scenario": cell["scenario"],
            "repetition": cell["repetition"],
            "followup_ms": (last - onset) / NS_PER_MS,
            "n_frames": len(recs),
        })
        for t_post, c in recs:
            frames_rows.append((rid, (t_post - onset) / NS_PER_MS, c))

    fr = pd.DataFrame(frames_rows, columns=["run_id", "t_rel_onset_ms", "max_conf"])
    ru = pd.DataFrame(run_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(out_dir / "traces.parquet", index=False)
    ru.to_parquet(out_dir / "traces_runs.parquet", index=False)
    print(f"cached {len(ru)} runs / {len(fr)} frames -> {out_dir}/traces*.parquet")


def evaluate(fr: pd.DataFrame, ru: pd.DataFrame, thr: float, k: int = 3) -> pd.DataFrame:
    """Per-run outcome at threshold `thr` with the K-consecutive rule."""
    q = fr["max_conf"].to_numpy() >= thr
    rid = fr["run_id"].to_numpy()
    trel = fr["t_rel_onset_ms"].to_numpy()

    out = {}
    streak = 0
    prev = None
    fired_pre = {}
    ttd = {}
    for i in range(len(rid)):
        r = rid[i]
        if r != prev:
            streak = 0
            prev = r
        streak = streak + 1 if q[i] else 0
        if streak >= k:
            if trel[i] >= 0:
                if r not in ttd:
                    ttd[r] = trel[i]
            else:
                fired_pre[r] = True
    res = ru.copy()
    res["ttd_ms"] = res["run_id"].map(ttd)
    res["hit"] = res["ttd_ms"].notna()
    res["fp_pre"] = res["run_id"].map(fired_pre).fillna(False).astype(bool)
    res["threshold"] = thr
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--raw-dir",
                    default=str(ROOT / "results" / "scvd-phase2" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "scvd-phase2" / "analysis"))
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    if args.build:
        build_cache(Path(args.raw_dir), out_dir)
        return 0

    fr = pd.read_parquet(out_dir / "traces.parquet")
    ru = pd.read_parquet(out_dir / "traces_runs.parquet")
    ru["config_label"] = np.where(ru["network_profile"] == "none",
                                  ru["deployment"],
                                  ru["deployment"] + "-" + ru["network_profile"])
    fr = fr.sort_values(["run_id", "t_rel_onset_ms"], kind="stable")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    allres = []
    for thr in grid:
        allres.append(evaluate(fr, ru, thr, args.k))
    sweep = pd.concat(allres, ignore_index=True)
    sweep.to_parquet(out_dir / "threshold_sweep.parquet", index=False)

    print("=== threshold sweep, per model (all deployments pooled) ===")
    t = sweep.groupby(["model", "threshold"]).agg(
        hit_pct=("hit", lambda s: round(s.mean() * 100, 1)),
        fp_pre_pct=("fp_pre", lambda s: round(s.mean() * 100, 1)),
        ttd_med=("ttd_ms", lambda s: round(s.median(), 0)),
    )
    print(t.to_string())

    print("\n=== FP-MATCHED comparison: per-model threshold giving FP_pre ~= target ===")
    for target in (10.0, 20.0, 30.0):
        rows = []
        for m, g in sweep.groupby("model"):
            per = g.groupby("threshold").agg(fp=("fp_pre", "mean"),
                                             hit=("hit", "mean"),
                                             ttd=("ttd_ms", "median"))
            per["fp"] *= 100
            per["hit"] *= 100
            i = (per["fp"] - target).abs().idxmin()
            rows.append({"model": m, "thr": i, "FP_pre%": round(per.loc[i, "fp"], 1),
                         "hit%": round(per.loc[i, "hit"], 1),
                         "TTD_med_ms": round(per.loc[i, "ttd"], 0)})
        print(f"\n  target FP_pre = {target:.0f} %")
        print(pd.DataFrame(rows).sort_values("hit%", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

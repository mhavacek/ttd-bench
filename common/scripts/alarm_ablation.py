#!/usr/bin/env python3
"""Alarm-rule ablation (Table 7) and residual decomposition (Table 8).

Both tables re-score the same frozen logs; nothing here is a new measurement.

Table 7 compares three alarm rules on identical runs:
  K = 3 consecutive   the operational rule (meas_hit)
  K = 3 within 1 s    a windowed variant (win_hit)
  K = 1               the policy-free bound (any qualifying post-onset frame)
The point is whether the deployment effect is an artifact of a rule that slow
pipelines cannot satisfy. If it were, it would vanish at K = 1.

Table 8 splits the observed TTD residual on runs where the alarm fired:
  t_first_det_ms   onset -> first qualifying detection (a detector property)
  t_accum_ms       first detection -> K-th confirmation (a throughput property)

Usage:
  alarm_ablation.py [--master ...] [--out-dir ...] [--exclude-cell MODEL:CONFIG]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
CFG_ORDER = ["local-gpu", "remote-lan", "edge-sim", "remote-wifi",
             "remote-cellular-4g"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "results/scvd-phase2/csv/master.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "clanek-2-ttdbench/results-d10"))
    ap.add_argument("--exclude-cell", action="append", metavar="MODEL:CONFIG",
                    help="drop a model x configuration cell (D10). Repeatable.")
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    for spec in args.exclude_cell or []:
        model, cfg = spec.split(":", 1)
        hit = (df["model"] == model) & (df["config_label"] == cfg)
        if not hit.any():
            raise SystemExit(f"--exclude-cell {spec!r} matches no run")
        df = df[~hit]
        print(f"excluded cell {spec}: {int(hit.sum())} runs dropped")

    # D1: every rate is computed after the validity filter, never before
    d = df[df["n_frames_processed"] > 0].copy()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Table 7: alarm-rule ablation ----
    d["k1_hit"] = d["n_qualifying_post_onset"] > 0
    rows = []
    for cfg in CFG_ORDER:
        g = d[d["config_label"] == cfg]
        rows.append({
            "config": cfg,
            "n": len(g),
            "K=3 consecutive [%]": round(g["meas_hit"].mean() * 100, 1),
            "K=1 bound [%]": round(g["k1_hit"].mean() * 100, 1),
            "K=3 in 1 s [%]": round(g["win_hit"].mean() * 100, 1),
        })
    t7 = pd.DataFrame(rows)
    t7.to_csv(out_dir / "table7_alarm_ablation.csv", index=False)

    # ---- Table 8: residual decomposition, on runs where the alarm fired ----
    h = d[d["meas_hit"].astype(bool)]
    rows = []
    for cfg in CFG_ORDER:
        g = h[h["config_label"] == cfg]
        rows.append({
            "config": cfg,
            "n_hits": len(g),
            "onset->first detection [ms]": round(g["t_first_det_ms"].median(), 1),
            "IQR_lo": round(g["t_first_det_ms"].quantile(.25), 1),
            "IQR_hi": round(g["t_first_det_ms"].quantile(.75), 1),
            "accumulation to K-th [ms]": round(g["t_accum_ms"].median(), 1),
        })
    t8 = pd.DataFrame(rows)
    t8.to_csv(out_dir / "table8_residual.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Table 7: alarm-rule ablation ===")
    print(t7.to_string(index=False))
    gap = lambda c: t7.set_index("config").loc["local-gpu", c] - \
                    t7.set_index("config").loc["remote-cellular-4g", c]
    print(f"\nlocal - 4G gap: {gap('K=3 consecutive [%]'):.1f} pp at K=3, "
          f"{gap('K=1 bound [%]'):.1f} pp at the policy-free bound K=1")
    print("(if the network effect were an artifact of an unsatisfiable rule, "
          "it would vanish at K=1)")

    print("\n=== Table 8: residual decomposition (runs with an alarm) ===")
    print(t8.to_string(index=False))
    fd = t8["onset->first detection [ms]"]
    print(f"\nonset->first detection spans {fd.min():.1f}-{fd.max():.1f} ms "
          f"across configurations (spread {fd.max() - fd.min():.1f} ms)")
    print(f"accumulation spans "
          f"{t8['accumulation to K-th [ms]'].min():.1f}-"
          f"{t8['accumulation to K-th [ms]'].max():.1f} ms")
    print(f"\nwrote {out_dir}/table7_alarm_ablation.csv, table8_residual.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

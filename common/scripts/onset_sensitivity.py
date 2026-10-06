#!/usr/bin/env python3
"""Pre-registered onset sensitivity analysis (paper section 3.7.6).

Onset is the one input to TTD that a human places rather than a clock records,
so its uncertainty is tested rather than assumed. The blind double annotation
of the 17-clip subset (results/annotation-agreement/) measured the typical
disagreement between two annotators; every onset is then shifted uniformly by
-delta and +delta, the alarms are re-scored from the frozen frame traces, and
the survival quantities are recomputed.

DELTA, fixed before any result below was computed (2026-09-21):
  primary  delta = 25 frames = 833.3 ms  -- median |difference|, robust to the
                                            single 158-frame outlier
  stress   delta = 38 frames = 1266.7 ms -- mean |difference|
All SCVD clips run at 30 fps.

Why the alarm is re-scored and not simply shifted: the measurement definition
counts the first K = 3 qualifying streak whose decision falls at or after the
onset. Move the onset later and an alarm that fired in between stops counting;
move it earlier and streaks that were pre-onset start to. So each run is
re-scored from its frames with the onset moved, using the same machinery as
the calibration (far_reanalysis.evaluate), at the frozen operating point
(confidence 0.5) that section 4.4 reports.

Decision rule, pre-registered: a conclusion is robust if every configuration
hazard ratio keeps its sign (above or below 1) and its significance at
alpha = 0.05 under BOTH shifts. Anything that flips is reported as sensitive to
onset placement, in the results.

The zero shift must reproduce the published section 4.4 hazard ratios; the
script stops if it does not, because then it would be testing a different
quantity from the one the paper reports.

  onset_sensitivity.py [--out-dir results/onset-sensitivity]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import far_reanalysis as fr_mod  # noqa: E402  (reuse the exact re-scoring)

ROOT = Path(__file__).resolve().parent.parent.parent
FPS = 30.0
DELTAS = {"primary": 25, "stress": 38}          # frames, fixed in advance
THRESHOLD = 0.5                                 # frozen operating point, sec 4.4
ALPHA = 0.05
PUBLISHED = ROOT / "clanek-2-ttdbench/results-d10/analysis/cox.csv"
CONFIGS = ["edge-sim", "remote-cellular-4g", "remote-lan", "remote-wifi"]


def rescore(frames: pd.DataFrame, runs: pd.DataFrame, shift_ms: float) -> pd.DataFrame:
    """Move every onset by shift_ms (positive = later) and re-score."""
    f = frames.copy()
    r = runs.copy()
    f["t_rel_onset_ms"] = f["t_rel_onset_ms"] - shift_ms
    r["followup_ms"] = r["followup_ms"] - shift_ms
    return fr_mod.evaluate(f, r, lambda m, c: THRESHOLD)


def km_by_config(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in res.groupby("config_label"):
        rows.append({"config": c, **fr_mod.km_stats(g)})
    return pd.DataFrame(rows).set_index("config")


def cox_configs(res: pd.DataFrame) -> pd.DataFrame:
    cx = fr_mod.cox_hr(res)
    keep = [i for i in cx.index if str(i).startswith("config_label_")]
    cx = cx.loc[keep].copy()
    cx.index = [str(i).replace("config_label_", "") for i in cx.index]
    return cx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results/onset-sensitivity"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, runs = fr_mod.load_traces()   # applies the D10 exclusion

    shifts = {"baseline": 0.0}
    for name, d in DELTAS.items():
        ms = d * 1000.0 / FPS
        shifts[f"{name}_earlier(-{d}f)"] = -ms
        shifts[f"{name}_later(+{d}f)"] = +ms

    cox_all, km_all, dropped = {}, {}, []
    for label, s in shifts.items():
        res = rescore(frames, runs, s)
        cox_all[label] = cox_configs(res)
        km_all[label] = km_by_config(res)
        print(f"{label:<24} shift {s:+8.1f} ms  runs {len(res):>5}  "
              f"alarms {int(res['hit'].sum()):>5}")
        # a later onset can leave a run with no follow-up at all; record who
        # leaves, so the attrition is a result and not a line of stdout
        if label == "baseline":
            base_res = res
        else:
            gone = base_res[~base_res["run_id"].isin(res["run_id"])]
            dropped.append(gone[["run_id", "scenario", "config_label", "model",
                                 "hit"]].assign(shift=label))

    # --- the zero shift must reproduce the published hazard ratios ---
    pub = pd.read_csv(PUBLISHED, index_col=0)
    pub.index = [str(i).replace("config_label_", "") for i in pub.index]
    base = cox_all["baseline"]
    print("\n=== baseline vs published section 4.4 (config HRs) ===")
    worst = 0.0
    for c in CONFIGS:
        a, b = float(base.loc[c, "HR"]), float(pub.loc[c, "HR"])
        worst = max(worst, abs(a - b))
        print(f"  {c:<20} recomputed {a:.4f}   published {b:.4f}")
    if worst > 0.01:
        print(f"\nSTOP: baseline differs from the published HRs by {worst:.4f}. "
              "This would test a different quantity from the one reported.")
        return 1
    print(f"  reproduced within {worst:.4f}")

    # --- decision rule ---
    rows = []
    for label, cx in cox_all.items():
        for c in CONFIGS:
            hr, p = float(cx.loc[c, "HR"]), float(cx.loc[c, "p"])
            rows.append({"shift": label, "config": c, "HR": round(hr, 4),
                         "CI_lo": round(float(cx.loc[c, "CI_lo"]), 4),
                         "CI_hi": round(float(cx.loc[c, "CI_hi"]), 4),
                         "p": p, "below_1": hr < 1, "significant": p < ALPHA})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "cox_by_shift.csv", index=False)

    b = tab[tab["shift"] == "baseline"].set_index("config")
    verdict = []
    print("\n=== robustness, pre-registered rule: sign AND significance unchanged ===")
    for c in CONFIGS:
        robust = {}
        for name in DELTAS:
            sub = tab[tab["shift"].str.startswith(name) & (tab["config"] == c)]
            robust[name] = all(
                (r.below_1 == b.loc[c, "below_1"]) and
                (r.significant == b.loc[c, "significant"])
                for r in sub.itertuples())
        verdict.append({"config": c, "baseline_HR": b.loc[c, "HR"],
                        "baseline_significant": b.loc[c, "significant"],
                        "robust_primary_delta": robust["primary"],
                        "robust_stress_delta": robust["stress"]})
        print(f"  {c:<20} HR {b.loc[c, 'HR']:.4f} "
              f"(sig {b.loc[c, 'significant']!s:<5})  "
              f"robust at delta=25f: {robust['primary']!s:<5}  "
              f"at delta=38f: {robust['stress']!s}")
    pd.DataFrame(verdict).to_csv(out / "verdict.csv", index=False)

    km = pd.concat({k: v for k, v in km_all.items()}, names=["shift", "config"])
    km.to_csv(out / "km_by_shift.csv")
    pd.concat(dropped).rename(columns={"hit": "hit_at_baseline"})[
        ["shift", "run_id", "scenario", "config_label", "model", "hit_at_baseline"]
    ].to_csv(out / "dropped_runs.csv", index=False)
    print(f"\nwrote {out}/cox_by_shift.csv, verdict.csv, km_by_shift.csv, "
          "dropped_runs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

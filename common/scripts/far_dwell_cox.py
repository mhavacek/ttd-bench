#!/usr/bin/env python3
"""Clustered Cox hazard ratios of the deployment configurations, plus pooled
Kaplan-Meier summaries per configuration, under the dwell-time rule
(far_dwell.py) at the calibrated operating points (variants a and b).

Same Cox model as far_reanalysis.cox_hr (detector as covariate, Efron ties,
robust SE clustered on scenario), so fixed3 reproduces Table 3 of the
manuscript. Outputs: phase2_far_dwell_cox.csv, phase2_far_dwell_pooled.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from far_dwell import (CONFIGS, MODES, OUT, evaluate, k_of, km_stats,  # noqa: E402
                       load_events, realised_dwell_s)
from far_reanalysis import cox_hr  # noqa: E402

TARGETS = [60.0, 120.0, 300.0, 600.0]


def main() -> int:
    th = pd.read_csv(OUT / "thresholds_dwell.csv")
    fr, ru = load_events()
    cox_rows, pool_rows = [], []
    for mode, t in MODES.items():
        kf = lambda c, m=mode: k_of(m, c)
        for tgt in TARGETS:
            for var in ("a", "b"):
                sub = th[(th["mode"] == mode) & (th["variant"] == var) &
                         (th["target_far_h"] == tgt)]
                if var == "a":
                    tm = sub.drop_duplicates("model").set_index("model")["threshold"]
                    thr_of = lambda m, c, t=tm: t[m]
                else:
                    tm = sub.set_index(["model", "config"])["threshold"]
                    if len(tm) < 19:
                        continue
                    thr_of = lambda m, c, t=tm: t[(m, c)]
                res = evaluate(fr, ru, thr_of, kf)
                cx = cox_hr(res)
                cx = cx[cx.index.str.startswith("config_label")].copy()
                cx.insert(0, "mode", mode)
                cx.insert(1, "t_requested_s", t)
                cx.insert(2, "variant", var)
                cx.insert(3, "target_far_h", tgt)
                cx["n_runs"] = len(res)
                cox_rows.append(cx)
                far = sub.groupby("config")["achieved_far_h"].mean()
                for c, g in res.groupby("config_label"):
                    pool_rows.append({"mode": mode, "t_requested_s": t,
                                      "variant": var, "target_far_h": tgt,
                                      "config": c, "K": kf(c),
                                      "realised_dwell_s": realised_dwell_s(mode, c),
                                      "mean_derived_far_h": far.get(c),
                                      **km_stats(g)})
    cox = pd.concat(cox_rows)
    cox.to_csv(OUT / "phase2_far_dwell_cox.csv")
    pd.DataFrame(pool_rows).to_csv(OUT / "phase2_far_dwell_pooled.csv", index=False)
    pd.set_option("display.width", 250)
    print(cox[(cox.variant == "a") & (cox.target_far_h == 120.0)]
          [["mode", "HR", "CI_lo", "CI_hi", "p"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

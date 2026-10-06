#!/usr/bin/env python3
"""Statistics Scientific Reports asks for that the stored results do not hold.

Approved by the user 2026-09-21 for the Scientific Reports version. Nothing is
re-measured: every quantity is recomputed from the frozen frame traces with the
same machinery that produced the published numbers, and the script stops if it
does not reproduce them first.

  cox_exact.csv       the fixed-threshold Cox model (section 4.4.3/4.4.4) with
                      unrounded P values, z, standard errors, n, events and
                      clusters. The stored cox.csv rounds to four decimals, so
                      every P below 5e-5 was stored as 0.0.
  km_ci.csv           Kaplan-Meier P(alarm <= t) with 95 % pointwise confidence
                      intervals (lifelines, exponential Greenwood) at 0.5, 1, 2
                      and 3 s per configuration.
  km_curves.csv       the full curves with their bands, for the figure.
  onset_sign_test.txt exact two-sided sign test on the direction of the 13
                      onset disagreements (ties dropped). The text claimed a
                      consistent direction; this says how much 13 clips support.

  scirep_stats.py [--out-dir results/scirep-stats]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import far_reanalysis as fr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
D10 = ROOT / "clanek-2-ttdbench/results-d10/analysis"
AGREE = ROOT / "results/annotation-agreement/compare.txt"
THRESHOLD = 0.5
HORIZONS_MS = (500, 1000, 2000, 3000)


def cox_exact(res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from lifelines import CoxPHFitter
    x = res[["duration_ms", "hit", "config_label", "model", "scenario"]].copy()
    x["config_label"] = pd.Categorical(
        x["config_label"], categories=["local-gpu"] + sorted(
            c for c in x["config_label"].unique() if c != "local-gpu"))
    x["model"] = pd.Categorical(x["model"], categories=fr.MODEL_ORDER)
    des = pd.get_dummies(x[["config_label", "model"]], drop_first=True, dtype=float)
    des["duration_ms"] = x["duration_ms"].values
    des["event"] = x["hit"].astype(int).values
    des["scenario"] = x["scenario"].values
    cph = CoxPHFitter()
    cph.fit(des, duration_col="duration_ms", event_col="event",
            cluster_col="scenario", robust=True)
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%",
                     "se(coef)", "z", "p"]].rename(columns={
                         "exp(coef)": "HR", "exp(coef) lower 95%": "CI_lo",
                         "exp(coef) upper 95%": "CI_hi", "se(coef)": "robust_se"})
    meta = {"n_runs": len(des), "n_events": int(des["event"].sum()),
            "n_clusters": int(des["scenario"].nunique()),
            "ties": "Efron (lifelines default)",
            "test": "two-sided Wald test on the log hazard ratio, "
                    "robust sandwich variance clustered on scenario"}
    return s, meta


def km(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lifelines import KaplanMeierFitter
    rows, curves = [], []
    for cfg, g in res.groupby("config_label"):
        k = KaplanMeierFitter().fit(g["duration_ms"], g["hit"])
        ci = k.confidence_interval_cumulative_density_
        cdf = k.cumulative_density_
        for h in HORIZONS_MS:
            p = float(1 - k.predict(h))
            idx = cdf.index[cdf.index <= h].max()
            lo, hi = (float(v) for v in ci.loc[idx].values)
            rows.append({"config": cfg, "t_ms": h, "P_alarm": round(p, 3),
                         "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
                         "n": len(g), "events": int(g["hit"].sum())})
        c = pd.DataFrame({"t_ms": cdf.index, "P_alarm": cdf.iloc[:, 0].values,
                          "CI_lo": ci.iloc[:, 0].values, "CI_hi": ci.iloc[:, 1].values})
        c["config"] = cfg
        curves.append(c)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def sign_test() -> str:
    from scipy.stats import binomtest
    d = [int(v) for v in re.findall(r"^\s+scvd-\S+\s+\d+\s+\d+\s+([-+]\d+)",
                                    AGREE.read_text(), re.M)]
    earlier, later, ties = sum(v < 0 for v in d), sum(v > 0 for v in d), sum(v == 0 for v in d)
    r = binomtest(earlier, earlier + later, 0.5, alternative="two-sided")
    return (f"onset disagreements: {len(d)} clips; second reader earlier in {earlier}, "
            f"later in {later}, equal in {ties}\n"
            f"exact two-sided sign test (ties dropped, n = {earlier + later}): "
            f"P = {r.pvalue:.4f}\n"
            "reading: 13 clips cannot distinguish a directional bias from chance\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results/scirep-stats"))
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, runs = fr.load_traces()
    res = fr.evaluate(frames, runs, lambda m, c: THRESHOLD)

    cox, meta = cox_exact(res)
    pub = pd.read_csv(D10 / "cox.csv", index_col=0)
    worst = max(abs(cox.loc[k, "HR"] - pub.loc[k, "HR"]) for k in pub.index)
    if worst > 0.00005:
        print(f"STOP: recomputed HRs differ from the published ones by {worst}")
        return 1
    cox.to_csv(out / "cox_exact.csv")
    (out / "cox_meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")

    kmt, curves = km(res)
    pubkm = pd.read_csv(D10 / "km_table.csv", index_col=0)
    for _, r in kmt.iterrows():
        want = float(pubkm.loc[r["config"], f"P(alarm<={r['t_ms']}ms)"])
        if abs(want - r["P_alarm"]) > 0.0005:
            print(f"STOP: KM {r['config']} at {r['t_ms']} ms is {r['P_alarm']}, "
                  f"published {want}")
            return 1
    kmt.to_csv(out / "km_ci.csv", index=False)
    curves.to_csv(out / "km_curves.csv", index=False)

    (out / "onset_sign_test.txt").write_text(sign_test())
    print(f"HRs reproduced within {worst:.1e}; KM reproduced; wrote {out}")
    print(cox[["HR", "CI_lo", "CI_hi", "p"]].to_string())
    print(meta)
    print((out / "onset_sign_test.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

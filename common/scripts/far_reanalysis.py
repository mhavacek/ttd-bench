#!/usr/bin/env python3
"""Phase-2 re-analysis at FAR-calibrated thresholds (pure re-scoring, no GPU).

Reads the frozen Phase-2 traces (results/scvd-phase2/analysis/traces*.parquet,
detections logged from conf 0.1) and the calibrated thresholds from
far_sweep.py, re-scores every valid run under the K = 3 consecutive rule at
each calibrated operating point, and recomputes the published quantities:
hit rate, Kaplan-Meier median TTD, P(alarm <= 1 s), and Cox HRs (robust SE
clustered on scenario). The frozen data are read-only; everything lands in
results-far-calib/.

Also reproduces the OLD operating point (threshold per model matched to 30 %
pre-onset false-positive rate on the 34 event clips) with identical machinery,
so old and new numbers sit side by side in one table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN = ROOT / "results/scvd-phase2/analysis"
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
K = 3
MODEL_ORDER = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon",
               "yolov26m-weapon"]
# The primary corpus is the deduplicated one; --keep-duplicates re-scores
# against the thresholds calibrated with UCF-Crime's repeated files left in and
# writes a parallel set of outputs, so the robustness run never overwrites the
# primary one.
SFX = "_withdup" if "--keep-duplicates" in sys.argv else ""
if "--v8s416" in sys.argv:          # D11: thresholds from far_sweep.py --v8s416
    SFX += "_v8s416"


# Data defect found during this re-analysis: the yolov26m ONNX export used by
# edge-sim logs max_conf = 1.000 on every detection frame (all other model x
# backend combinations agree across configurations). Saturated scores cannot
# be re-thresholded, so these runs are excluded from every threshold-swept
# quantity and reported separately. NOTE: the frozen fixed-0.5 analysis and
# the old FP-matched comparison silently included them.
DEFECT = [("yolov26m-weapon", "edge-sim")]


def load_traces():
    fr = pd.read_parquet(FROZEN / "traces.parquet")
    ru = pd.read_parquet(FROZEN / "traces_runs.parquet")
    ru["config_label"] = np.where(ru["network_profile"] == "none",
                                  ru["deployment"],
                                  ru["deployment"] + "-" + ru["network_profile"])
    n0 = len(ru)
    for m, c in DEFECT:
        ru = ru[~((ru["model"] == m) & (ru["config_label"] == c))]
    print(f"defect exclusion (saturated ONNX scores): "
          f"{n0 - len(ru)} runs of {DEFECT} dropped, {len(ru)} remain")
    fr = fr.sort_values(["run_id", "t_rel_onset_ms"], kind="stable")
    fr = fr.merge(ru[["run_id", "model", "config_label"]], on="run_id")
    return fr, ru


def evaluate(fr: pd.DataFrame, ru: pd.DataFrame, thr_of) -> pd.DataFrame:
    """Per-run alarm outcome; thr_of(model, config_label) -> threshold."""
    thr = np.array([thr_of(m, c) for m, c in
                    zip(fr["model"].to_numpy(), fr["config_label"].to_numpy())])
    q = fr["max_conf"].to_numpy() >= thr
    rid = fr["run_id"].to_numpy()
    trel = fr["t_rel_onset_ms"].to_numpy()
    ttd, fired_pre = {}, {}
    streak, prev = 0, None
    for i in range(len(rid)):
        if rid[i] != prev:
            streak, prev = 0, rid[i]
        streak = streak + 1 if q[i] else 0
        if streak >= K:
            if trel[i] >= 0:
                ttd.setdefault(rid[i], trel[i])
            else:
                fired_pre[rid[i]] = True
    res = ru.copy()
    res["ttd_ms"] = res["run_id"].map(ttd)
    res["hit"] = res["ttd_ms"].notna()
    res["fp_pre"] = res["run_id"].map(fired_pre).fillna(False).astype(bool)
    res["duration_ms"] = np.where(res["hit"], res["ttd_ms"], res["followup_ms"])
    res = res[res["duration_ms"] > 0]
    return res


def km_stats(g: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(g["duration_ms"], g["hit"])
    med = kmf.median_survival_time_
    return {"n": len(g), "hit_pct": round(g["hit"].mean() * 100, 1),
            "P_alarm_1s": round(float(1 - kmf.predict(1000)), 3),
            "km_median_ms": round(med, 0) if np.isfinite(med) else np.inf}


def cox_hr(d: pd.DataFrame) -> pd.DataFrame:
    from lifelines import CoxPHFitter
    x = d[["duration_ms", "hit", "config_label", "model", "scenario"]].copy()
    x["config_label"] = pd.Categorical(
        x["config_label"], categories=["local-gpu"] + sorted(
            c for c in x["config_label"].unique() if c != "local-gpu"))
    x["model"] = pd.Categorical(
        x["model"], categories=MODEL_ORDER)
    des = pd.get_dummies(x[["config_label", "model"]], drop_first=True,
                         dtype=float)
    des["duration_ms"] = x["duration_ms"].values
    des["event"] = x["hit"].astype(int).values
    des["scenario"] = x["scenario"].values
    # At the lowest false-alarm targets the detections become so sparse that
    # whole model x configuration cells contain zero events, the design matrix
    # is singular, and no proportional-hazards model exists. That is a result
    # about the operating point, not a bug, so it is recorded and the sweep
    # continues instead of dying on the first failure.
    n_ev = int(des["event"].sum())
    cph = CoxPHFitter()
    try:
        cph.fit(des, duration_col="duration_ms", event_col="event",
                cluster_col="scenario", robust=True)
    except Exception as exc:
        return pd.DataFrame([{
            "HR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan, "p": np.nan,
            "n_events": n_ev,
            "note": f"Cox did not converge ({type(exc).__name__}); "
                    f"{n_ev} events over {len(des)} runs"}],
            index=["<model not estimable>"])
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                     "exp(coef) upper 95%", "p"]]
    s = s.assign(n_events=n_ev, note="")
    return s.rename(columns={"exp(coef)": "HR",
                             "exp(coef) lower 95%": "CI_lo",
                             "exp(coef) upper 95%": "CI_hi"}).round(4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="use thresholds_withdup.csv and write *_withdup "
                         "outputs (robustness run)")
    ap.add_argument("--v8s416", action="store_true",
                    help="D11: use thresholds_v8s416.csv, write *_v8s416 outputs")
    ap.parse_args()

    fr, ru = load_traces()
    th = pd.read_csv(OUT / f"thresholds{SFX}.csv")

    summaries, coxes = [], []

    # ---- new FAR-calibrated operating points ----
    for (variant, tgt), sub in th.groupby(["variant", "target_far_h"]):
        if variant == "a":
            tmap = sub.drop_duplicates("model").set_index("model")["threshold"]
            if len(tmap) < 4:
                continue
            thr_of = lambda m, c, t=tmap: t[m]
        else:
            tmap = sub.set_index(["model", "config"])["threshold"]
            if len(tmap) < 20:
                continue          # some config x model missed this target
            thr_of = lambda m, c, t=tmap: t[(m, c)]
        res = evaluate(fr, ru, thr_of)
        for (m, c), g in res.groupby(["model", "config_label"]):
            summaries.append({"operating_point": f"far{tgt:g}", "variant": variant,
                              "model": m, "config": c, **km_stats(g)})
        for m, g in res.groupby("model"):
            summaries.append({"operating_point": f"far{tgt:g}", "variant": variant,
                              "model": m, "config": "ALL", **km_stats(g)})
        cx = cox_hr(res)
        cx["operating_point"] = f"far{tgt:g}"
        cx["variant"] = variant
        coxes.append(cx)

    # ---- published operating point: fixed conf 0.5 for every model ----
    res = evaluate(fr, ru, lambda m, c: 0.5)
    for (m, c), g in res.groupby(["model", "config_label"]):
        summaries.append({"operating_point": "old_conf0.5", "variant": "-",
                          "model": m, "config": c, **km_stats(g)})
    for m, g in res.groupby("model"):
        summaries.append({"operating_point": "old_conf0.5", "variant": "-",
                          "model": m, "config": "ALL", **km_stats(g)})
    for c, g in res.groupby("config_label"):
        summaries.append({"operating_point": "old_conf0.5", "variant": "-",
                          "model": "ALL", "config": c, **km_stats(g)})
    cx = cox_hr(res)
    cx["operating_point"] = "old_conf0.5"
    cx["variant"] = "-"
    coxes.append(cx)

    # ---- old operating point: FP_pre ~= 30 % on the event clips ----
    grid = np.round(np.arange(0.10, 0.951, 0.005), 3)
    old_thr = {}
    for m in MODEL_ORDER:
        best, bdiff = None, 1e9
        for thr in grid:
            res = evaluate(fr[fr.model == m], ru[ru.model == m],
                           lambda _m, _c, t=thr: t)
            fp = res["fp_pre"].mean() * 100
            if abs(fp - 30.0) < bdiff:
                best, bdiff, bfp = thr, abs(fp - 30.0), fp
        old_thr[m] = best
        print(f"old operating point {m}: thr={best} (FP_pre={bfp:.1f} %)")
    res = evaluate(fr, ru, lambda m, c: old_thr[m])
    for (m, c), g in res.groupby(["model", "config_label"]):
        summaries.append({"operating_point": "old_fp30", "variant": "-",
                          "model": m, "config": c, **km_stats(g)})
    for m, g in res.groupby("model"):
        summaries.append({"operating_point": "old_fp30", "variant": "-",
                          "model": m, "config": "ALL", **km_stats(g)})
    cx = cox_hr(res)
    cx["operating_point"] = "old_fp30"
    cx["variant"] = "-"
    coxes.append(cx)
    pd.Series(old_thr).to_csv(OUT / f"old_fp30_thresholds{SFX}.csv")

    sm = pd.DataFrame(summaries)
    sm.to_csv(OUT / f"phase2_far_summary{SFX}.csv", index=False)
    cxall = pd.concat(coxes)
    cxall.to_csv(OUT / f"phase2_far_cox{SFX}.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== model ranking by hit % (pooled over configs) per operating point ===")
    pool = sm[sm.config == "ALL"].copy()
    for (op, var), g in pool.groupby(["operating_point", "variant"]):
        order = g.sort_values("hit_pct", ascending=False)
        rank = " > ".join(f"{r.model.replace('-weapon','').replace('yolov','v')}"
                          f"({r.hit_pct}%)" for r in order.itertuples())
        print(f"{op:>9} [{var}]: {rank}")
    print("\n=== Cox HRs (model terms, ref yolov8s; config terms, ref local-gpu) ===")
    print(cxall.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Proportional-hazards checks for the Cox models of the paper (reviewer M3).

For the primary model (fixed confidence 0.5, K = 3) and the calibrated
models of Table 3 (variant a at 60/120/300/600 per hour, variant b at
120/h) this script

1. refits the published model (configuration and detector as covariates,
   Efron ties, robust SE clustered on scenario) and checks that the
   configuration hazard ratios reproduce the stored ones;
2. runs the Grambsch-Therneau test on scaled Schoenfeld residuals
   (lifelines.statistics.proportional_hazard_test, rank time transform),
   per covariate and globally;
3. fits the same model stratified by scenario, which uses the paired design
   (every scenario is replayed under every configuration) without assuming a
   common baseline hazard across scenarios;
4. reports the restricted mean time to alarm up to 3 s, RMST(3 s), per
   configuration with a scenario-level cluster bootstrap, a summary that
   does not rely on proportional hazards.

Outputs in results/scirep-stats/: ph_tests.csv, cox_stratified.csv,
rmst_3s.csv. Pure re-analysis of the frozen traces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import far_reanalysis as fr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results/scirep-stats"
CAL = ROOT / "clanek-2-ttdbench/results-far-calib"
TAU_MS = 3000
N_BOOT = 1000


def design(res: pd.DataFrame) -> pd.DataFrame:
    x = res[["duration_ms", "hit", "config_label", "model", "scenario"]].copy()
    x["config_label"] = pd.Categorical(
        x["config_label"], categories=["local-gpu"] + sorted(
            c for c in x["config_label"].unique() if c != "local-gpu"))
    x["model"] = pd.Categorical(x["model"], categories=fr.MODEL_ORDER)
    des = pd.get_dummies(x[["config_label", "model"]], drop_first=True, dtype=float)
    des["duration_ms"] = x["duration_ms"].values
    des["event"] = x["hit"].astype(int).values
    des["scenario"] = x["scenario"].values
    return des


def ph(des: pd.DataFrame, label: str) -> tuple[list[dict], list[dict]]:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    cph = CoxPHFitter().fit(des, duration_col="duration_ms", event_col="event",
                            cluster_col="scenario", robust=True)
    t = proportional_hazard_test(cph, des,
                                 time_transform="rank")
    rows = [{"model_fit": label, "covariate": k, "chi2": float(r["test_statistic"]),
             "p": float(r["p"])} for k, r in t.summary.iterrows()]
    # global test: sum of the per-covariate statistics is not the global
    # Grambsch-Therneau statistic, so report the minimum p with Holm instead
    ps = np.array([r["p"] for r in rows])
    order = np.argsort(ps)
    holm = np.minimum(1, np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[order]))
    for i, j in enumerate(order):
        rows[j]["p_holm"] = float(holm[i])
    st = CoxPHFitter().fit(des, duration_col="duration_ms", event_col="event",
                           strata=["scenario"])
    s = st.summary
    strat = [{"model_fit": label, "covariate": k, "HR": float(r["exp(coef)"]),
              "CI_lo": float(r["exp(coef) lower 95%"]),
              "CI_hi": float(r["exp(coef) upper 95%"]), "p": float(r["p"])}
             for k, r in s.iterrows() if k.startswith("config_label")]
    return rows, strat


def rmst(res: pd.DataFrame, label: str, rng) -> list[dict]:
    """Restricted mean time to alarm up to TAU, per configuration, with a
    scenario-cluster bootstrap. Runs that end before TAU without an alarm are
    censored; KM is integrated up to TAU."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time as rm

    def one(g):
        k = KaplanMeierFitter().fit(g["duration_ms"], g["hit"])
        return float(rm(k, t=TAU_MS))
    out = []
    scen = res["scenario"].unique()
    by = {s: g for s, g in res.groupby("scenario")}
    for c, g in res.groupby("config_label"):
        est = one(g)
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(scen, size=len(scen), replace=True)
            b = pd.concat([by[s][by[s]["config_label"] == c] for s in pick])
            bs.append(one(b))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"model_fit": label, "config": c, "rmst_ms": est,
                    "boot_lo": lo, "boot_hi": hi, "n": len(g)})
    return out


def main() -> int:
    frames, runs = fr.load_traces()
    th = pd.read_csv(CAL / "thresholds.csv")
    fits = {"fixed0.5": lambda m, c: 0.5}
    for tgt in (60.0, 120.0, 300.0, 600.0):
        tm = th[(th.variant == "a") & (th.target_far_h == tgt)].drop_duplicates(
            "model").set_index("model")["threshold"]
        fits[f"a{tgt:g}"] = (lambda t: (lambda m, c: t[m]))(tm)
    tb = th[(th.variant == "b") & (th.target_far_h == 120.0)].set_index(
        ["model", "config"])["threshold"]
    fits["b120"] = lambda m, c, t=tb: t[(m, c)]

    pub = pd.read_csv(CAL / "phase2_far_cox.csv", index_col=0)
    rng = np.random.default_rng(42)
    ph_rows, strat_rows, rm_rows = [], [], []
    for label, thr_of in fits.items():
        res = fr.evaluate(frames, runs, thr_of)
        des = design(res)
        r, s = ph(des, label)
        ph_rows += r
        strat_rows += s
        if label in ("fixed0.5", "a120"):
            rm_rows += rmst(res, label, rng)
        # reproduction check against the stored clustered model
        op, var = ("old_conf0.5", "-") if label == "fixed0.5" else (
            f"far{label[1:]}", label[0])
        stored = pub[(pub.operating_point == op) & (pub.variant == var)]
        from lifelines import CoxPHFitter
        cph = CoxPHFitter().fit(des, duration_col="duration_ms", event_col="event",
                                cluster_col="scenario", robust=True)
        for k in stored.index:
            if k.startswith("config_label"):
                d = abs(cph.summary.loc[k, "exp(coef)"] - stored.loc[k, "HR"])
                if d > 5e-4:
                    raise SystemExit(f"{label} {k}: refit HR differs by {d}")
    pd.DataFrame(ph_rows).to_csv(OUT / "ph_tests.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(OUT / "cox_stratified.csv", index=False)
    pd.DataFrame(rm_rows).to_csv(OUT / "rmst_3s.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(ph_rows).round(4).to_string(index=False))
    print(pd.DataFrame(strat_rows).round(4).to_string(index=False))
    print(pd.DataFrame(rm_rows).round(1).to_string(index=False))
    return 0


def tables() -> None:
    """SI tables from the stored CSVs -> paper/scirep/tables/s_ph.tex, s_rmst.tex."""
    sc = ROOT / "clanek-2-ttdbench/paper/scirep/tables"
    hdr = "% generated by common/scripts/ph_check.py -- do not edit\n"
    lab = {"fixed0.5": "fixed 0.5", "a60": "(a) 60/h", "a120": "(a) 120/h",
           "a300": "(a) 300/h", "a600": "(a) 600/h", "b120": "(b) 120/h"}
    cfg = ["remote-lan", "remote-wifi", "remote-cellular-4g", "edge-sim"]
    ph_ = pd.read_csv(OUT / "ph_tests.csv")
    st = pd.read_csv(OUT / "cox_stratified.csv")
    L = [hdr + r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Model & remote-lan & remote-wifi & remote-cellular-4g & edge-sim \\", r"\midrule"]
    for m in lab:
        g = ph_[ph_.model_fit == m].set_index("covariate")
        h = st[st.model_fit == m].set_index("covariate")
        L.append(f"{lab[m]}: PH test, Holm $P$ & " + " & ".join(
            f"{g.loc['config_label_' + c, 'p_holm']:.3f}" for c in cfg) + r" \\")
        # an upper CI bound below 0.001 is quasi-separation (almost no events in
        # the configuration within the strata), not an estimate
        def ne(c):
            return h.loc["config_label_" + c, "CI_hi"] < 1e-3
        L.append(r"\quad stratified HR & " + " & ".join(
            "n.e." if ne(c) else f"{h.loc['config_label_' + c, 'HR']:.3f}"
            for c in cfg) + r" \\")
        L.append(r"\quad 95\% CI & " + " & ".join(
            "" if ne(c) else
            f"{h.loc['config_label_' + c, 'CI_lo']:.3f}--{h.loc['config_label_' + c, 'CI_hi']:.3f}"
            for c in cfg) + r" \\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (sc / "s_ph.tex").write_text("\n".join(L) + "\n")
    r = pd.read_csv(OUT / "rmst_3s.csv")
    order = ["local-gpu", "remote-lan", "remote-wifi", "remote-cellular-4g", "edge-sim"]
    L = [hdr + r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Operating point & " + " & ".join(order) + r" \\", r"\midrule"]
    for m in ("fixed0.5", "a120"):
        g = r[r.model_fit == m].set_index("config")
        L.append(f"{lab[m]} & " + " & ".join(f"{g.loc[c, 'rmst_ms']:.0f}" for c in order) + r" \\")
        L.append(r"\quad 95\% CI & " + " & ".join(
            f"{g.loc[c, 'boot_lo']:.0f}--{g.loc[c, 'boot_hi']:.0f}" for c in order) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (sc / "s_rmst.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    if "--tables" in sys.argv:
        tables()
        sys.exit(0)
    rc = main()
    tables()
    sys.exit(rc)

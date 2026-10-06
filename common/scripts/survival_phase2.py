#!/usr/bin/env python3
"""Phase-2 time-to-event analysis of TTD (survival / censored formulation).

  python scripts/survival_phase2.py --master results-scvd/csv/master.csv

WHY SURVIVAL AND NOT MEDIAN TTD
-------------------------------
TTD is only observed on runs where an alarm fired, and the alarm rate differs
strongly by deployment (45 % local vs 28 % 4G). Comparing medians of the
observed TTDs therefore compares *different subsets of scenarios* — the harder
scenarios drop out of the slower configurations (survivorship bias), which
biases the slow configurations' median DOWNWARDS. Detection rate and detection
latency are two projections of one quantity: the distribution of time from
onset to alarm, with non-detections as right-censored observations.

Censoring is ADMINISTRATIVE and set by the clip, not by the nominal 30 s
timeout: SCVD clips end a median of ~3.4 s after the onset. A run with no
alarm is censored at `followup_ms` (onset -> last processed frame), i.e. it
contributes "no alarm up to here", not "no alarm within 30 s".

Runs that processed zero frames are infrastructure failures, not
non-detections, and are excluded (reported separately).

Dependence: 5 repetitions of the same scenario are not independent, so Cox
models use robust standard errors clustered on scenario. Log-rank p-values
(which assume independence) are reported for reference and are anticonservative;
the clustered Cox model is the inferential statement.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root


def load(master: Path, exclude_cells: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(master)
    df["valid"] = df["n_frames_processed"] > 0
    for spec in exclude_cells or []:
        model, cfg = spec.split(":", 1)
        hit = (df["model"] == model) & (df["config_label"] == cfg)
        if not hit.any():
            raise SystemExit(f"--exclude-cell {spec!r} matches no run")
        df = df[~hit]
        print(f"excluded cell {spec}: {int(hit.sum())} runs dropped")
    return df


def build_surv(df: pd.DataFrame, event_col: str, time_col: str) -> pd.DataFrame:
    """Event/censoring table: duration = TTD on a hit, follow-up on a miss."""
    d = df[df["valid"] & df["followup_ms"].notna()].copy()
    d["event"] = d[event_col].astype(bool)
    d["duration_ms"] = np.where(d["event"], d[time_col], d["followup_ms"])
    # a hit can never post-date the last processed frame; guard anyway
    d = d[d["duration_ms"] > 0]
    return d


def km_table(d: pd.DataFrame, horizons=(500, 1000, 2000, 3000)) -> pd.DataFrame:
    from lifelines import KaplanMeierFitter

    rows = []
    for cfg, g in d.groupby("config_label"):
        kmf = KaplanMeierFitter().fit(g["duration_ms"], g["event"])
        # P(alarm by t) = 1 - S(t)
        r = {"n": len(g), "events": int(g["event"].sum())}
        for h in horizons:
            r[f"P(alarm<={h}ms)"] = round(float(1 - kmf.predict(h)), 3)
        med = kmf.median_survival_time_
        r["KM median TTD"] = round(med, 1) if np.isfinite(med) else np.inf
        r["naive median TTD"] = round(g.loc[g["event"], "duration_ms"].median(), 1)
        rows.append(pd.Series(r, name=cfg))
    return pd.DataFrame(rows)


def logrank_pairs(d: pd.DataFrame) -> pd.DataFrame:
    from lifelines.statistics import logrank_test

    cfgs = sorted(d["config_label"].unique())
    rows = []
    for a, b in itertools.combinations(cfgs, 2):
        ga, gb = d[d["config_label"] == a], d[d["config_label"] == b]
        res = logrank_test(ga["duration_ms"], gb["duration_ms"],
                           ga["event"], gb["event"])
        rows.append({"pair": f"{a} vs {b}", "chi2": round(res.test_statistic, 2),
                     "p": res.p_value})
    out = pd.DataFrame(rows)
    n = len(out)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)      # Bonferroni over all pairs
    out["sig_0.05"] = out["p_bonf"] < 0.05
    out["p"] = out["p"].map(lambda x: f"{x:.3g}")
    out["p_bonf"] = out["p_bonf"].map(lambda x: f"{x:.3g}")
    return out


def cox(d: pd.DataFrame, ref_cfg: str = "local-gpu") -> pd.DataFrame:
    from lifelines import CoxPHFitter

    x = d[["duration_ms", "event", "config_label", "model", "scenario"]].copy()
    x["config_label"] = pd.Categorical(
        x["config_label"],
        categories=[ref_cfg] + [c for c in sorted(x["config_label"].unique())
                                if c != ref_cfg])
    x["model"] = pd.Categorical(
        x["model"],
        categories=["yolov8s-weapon"] + [m for m in sorted(x["model"].unique())
                                         if m != "yolov8s-weapon"])
    des = pd.get_dummies(x[["config_label", "model"]], drop_first=True, dtype=float)
    des["duration_ms"] = x["duration_ms"].values
    des["event"] = x["event"].astype(int).values
    des["scenario"] = x["scenario"].values

    cph = CoxPHFitter()
    cph.fit(des, duration_col="duration_ms", event_col="event",
            cluster_col="scenario", robust=True)
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s = s.rename(columns={"exp(coef)": "HR", "exp(coef) lower 95%": "CI_lo",
                          "exp(coef) upper 95%": "CI_hi"})
    return s.round(4)


def figure(d: pd.DataFrame, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for cfg in sorted(d["config_label"].unique()):
        g = d[d["config_label"] == cfg]
        kmf = KaplanMeierFitter().fit(g["duration_ms"], g["event"], label=cfg)
        # plot cumulative incidence P(alarm by t) = 1 - S(t)
        ci = 1 - kmf.survival_function_[kmf._label]
        ax.step(ci.index / 1000.0, ci.values, where="post", label=cfg, linewidth=1.4)
    ax.set_xlim(0, 5)
    ax.set_xlabel("time since weapon onset [s]")
    ax.set_ylabel(r"P(alarm raised by $t$)")
    ax.set_title("Time-to-detection, censored at clip end (SCVD, all models)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    print(f"\nfigure -> {out_pdf}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "results/scvd-phase2/csv/master.csv"))
    ap.add_argument("--event", default="meas_hit",
                    choices=["meas_hit", "win_hit", "hit"])
    ap.add_argument("--time", default=None,
                    help="time column (default: matches --event)")
    ap.add_argument("--out-dir", default=str(ROOT / "results/scvd-phase2/analysis"))
    ap.add_argument("--exclude-cell", action="append", metavar="MODEL:CONFIG",
                    help="drop a model x configuration cell before any measure "
                         "(D10: yolov26m-weapon:edge-sim). Repeatable.")
    args = ap.parse_args()

    time_col = args.time or {"meas_hit": "ttd_meas_ms", "win_hit": "ttd_win_ms",
                             "hit": "ttd_ms"}[args.event]
    df = load(Path(args.master), args.exclude_cell)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    n_bad = int((~df["valid"]).sum())
    print(f"runs: {len(df)} total, {n_bad} zero-frame infrastructure failures "
          f"({n_bad / len(df) * 100:.1f} %) excluded")
    print("infrastructure failure rate by configuration [%]:")
    print((100 - df.groupby("config_label")["valid"].mean() * 100).round(1).to_string())

    d = build_surv(df, args.event, time_col)
    print(f"\nanalysed: {len(d)} runs, {int(d['event'].sum())} events "
          f"({d['event'].mean() * 100:.1f} %), "
          f"median follow-up {d['followup_ms'].median() / 1000:.2f} s")

    print(f"\n=== Kaplan-Meier: P(alarm by t), event = {args.event} ===")
    kt = km_table(d)
    print(kt.to_string())
    kt.to_csv(out_dir / "km_table.csv")

    print("\n=== pairwise log-rank (Bonferroni over all 10 config pairs) ===")
    lr = logrank_pairs(d)
    print(lr.to_string(index=False))
    lr.to_csv(out_dir / "logrank_pairs.csv", index=False)

    print("\n=== Cox PH, robust SE clustered on scenario "
          "(HR > 1 = alarm sooner/more often; ref = local-gpu, yolov8s) ===")
    cx = cox(d)
    print(cx.to_string())
    cx.to_csv(out_dir / "cox.csv")

    figure(d, out_dir / f"km_{args.event}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())

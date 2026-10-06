#!/usr/bin/env python3
"""Persist the three Phase-1 numbers that exist in the text but in no artifact.

The paper's rule is that a number may only come from a stored result. Three
claims about Phase 1 broke it — they lived as prose in the outline and in
PHASE1-FINAL.md and had never been recomputed since:

  O3  per-cell 95 % CI half-widths, quoted as 0.6-7.9 % and 0.5-5.1 %
  O4  ANOVA of drop_rate on model within edge-sim and 4G, quoted as F = 8401
      and F = 42.6
  4.3 the serial-client model, quoted as predicting the drop rate "within
      1 percentage point"

The frozen aggregate is read-only; outputs land in results-phase1-derived/.

Runs that never emitted a frame are excluded throughout. In the frozen
aggregate such a run carries drop_rate = 0.0 — a failure encoded as a flawless
run — so every statistic here filters n_frames_emitted > 0 first. That leaves
131 of 225 runs, the same count PHASE1-FINAL.md reports.

  phase1_validation.py   -> o3_cell_ci.csv, o4_anova.csv, serial_client.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN = ROOT / "results/phase1-final/csv/master.csv"
OUT = ROOT / "clanek-2-ttdbench/results-phase1-derived"
FPS = 30.0
PERIOD_MS = 1000.0 / FPS
METRICS = ["pf_dt_infer_ms", "pf_latency_ms", "drop_rate"]


def load() -> pd.DataFrame:
    df = pd.read_csv(FROZEN)
    n_all = len(df)
    df = df[df["n_frames_emitted"] > 0].copy()
    print(f"valid runs: {len(df)} of {n_all} "
          f"({n_all - len(df)} emitted no frame and are excluded)")
    return df


def o3_cell_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Half-width of the 95 % CI per cell, pooled over the three scenarios.

    O3 defends running 2-5 repetitions instead of the pre-registered 5 by
    arguing that a latency decomposition is a low-variance measurement. That
    defence is only as good as the interval it quotes, so the interval is
    computed here rather than remembered.
    """
    rows = []
    for (cfg, model), g in df.groupby(["config_label", "model"]):
        for met in METRICS:
            v = g[met].dropna().to_numpy()
            if len(v) < 2:
                continue
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            rows.append({"config": cfg, "model": model, "metric": met,
                         "n": len(v), "mean": round(float(v.mean()), 4),
                         "ci_half_width": round(float(half), 4),
                         "ci_half_width_rel_pct": (round(float(100 * half / v.mean()), 2)
                                                   if v.mean() else None)})
    return pd.DataFrame(rows)


def o4_anova(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA of drop_rate on model, within each deployment regime.

    O4 replaced the pilot's claim that the 4G drop is "model-independent" with
    a contrast of effect sizes: both regimes are significant, and they differ by
    an order of magnitude.
    """
    rows = []
    for cfg, g in df.groupby("config_label"):
        groups = [x["drop_rate"].dropna().to_numpy() for _, x in g.groupby("model")]
        groups = [x for x in groups if len(x) > 1]
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        means = g.groupby("model")["drop_rate"].mean()
        rows.append({"config": cfg, "k_models": len(groups),
                     "n": sum(len(x) for x in groups),
                     "F": round(float(f), 1), "p": float(p),
                     "drop_min": round(float(means.min()), 4),
                     "drop_max": round(float(means.max()), 4),
                     "range_pp": round(float(100 * (means.max() - means.min())), 1),
                     "sd_within": round(float(np.mean([x.std(ddof=1) for x in groups])), 4)})
    return pd.DataFrame(rows).sort_values("F", ascending=False)


def serial_client(df: pd.DataFrame) -> pd.DataFrame:
    """Does drop = 1 - period/cycle explain the measured frame loss?

    The client is serial with a one-frame queue, so while it is busy for `cycle`
    milliseconds every frame arriving inside that window is discarded. The cycle
    is the work the client does per frame — round trip to the server, inference,
    post-processing — and NOT dt_acq, which is pipeline delay rather than
    occupancy.

    The model is deterministic: it uses a mean cycle time and therefore cannot
    see variance. That is why the residual is the interesting column.
    """
    d = df.copy()
    d["cycle_ms"] = (d["pf_dt_transfer_ms"].fillna(0) + d["pf_dt_infer_ms"].fillna(0)
                     + d["pf_dt_post_ms"].fillna(0))
    d["predicted_drop"] = np.where(d["cycle_ms"] > PERIOD_MS,
                                   1 - PERIOD_MS / d["cycle_ms"], 0.0)
    rows = []
    for (cfg, model), g in d.groupby(["config_label", "model"]):
        # Cells are summarised by the mean, matching 4.2; the median is carried
        # alongside because a structural claim that depends on which average
        # was taken is not a structural claim. Here they agree.
        cyc = float(g["cycle_ms"].mean())
        pred, meas = float(g["predicted_drop"].mean()), float(g["drop_rate"].mean())
        pred_md = float(g["predicted_drop"].median())
        meas_md = float(g["drop_rate"].median())
        rows.append({"config": cfg, "model": model, "n": len(g),
                     "cycle_ms": round(cyc, 3),
                     "frame_period_ms": round(PERIOD_MS, 3),
                     "cycle_over_period": round(cyc / PERIOD_MS, 3),
                     "predicted_drop": round(pred, 4),
                     "measured_drop": round(meas, 4),
                     "residual_pp": round(100 * (meas - pred), 2),
                     "residual_pp_median": round(100 * (meas_md - pred_md), 2)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    o3 = o3_cell_ci(df)
    o3.to_csv(OUT / "o3_cell_ci.csv", index=False)
    print("\n--- O3: relative 95 % CI half-width per cell ---")
    for met in METRICS:
        s = o3[o3["metric"] == met]
        rel, ab = s["ci_half_width_rel_pct"], s["ci_half_width"]
        print(f"  {met:<16} relative {rel.min():.1f}-{rel.max():.1f} %   "
              f"absolute {ab.min():.4f}-{ab.max():.4f}   ({len(s)} cells, "
              f"n = {s['n'].min()}-{s['n'].max()})")

    o4 = o4_anova(df)
    o4.to_csv(OUT / "o4_anova.csv", index=False)
    print("\n--- O4: ANOVA drop_rate ~ model, within regime ---")
    print(o4.to_string(index=False))

    sc = serial_client(df)
    sc.to_csv(OUT / "serial_client.csv", index=False)
    print("\n--- 4.3: serial-client model ---")
    print(sc.to_string(index=False))
    within1 = int((sc["residual_pp"].abs() <= 1.0).sum())
    under = int((sc["residual_pp"] > 0).sum())
    print(f"\n  cells predicted within 1 pp: {within1} of {len(sc)}")
    print(f"  measured drop exceeds the prediction in {under} of {len(sc)} cells: "
          f"the deterministic model under-predicts, it never over-predicts by "
          f"more than {abs(sc['residual_pp'].min()):.2f} pp")
    # The model uses a mean cycle, so it is blind to variance - and variance
    # only matters when the cycle sits near the frame period, where a few ms of
    # jitter decides whether a frame is dropped at all. Far from the period, in
    # either direction, the outcome is determined regardless of jitter.
    near = sc[(sc["cycle_over_period"] - 1).abs() <= 0.15]
    far = sc[(sc["cycle_over_period"] - 1).abs() > 0.15]
    rho, p_rho = stats.spearmanr((sc["cycle_over_period"] - 1).abs(),
                                 sc["residual_pp"])
    print(f"  residual where the cycle is within 15 % of the frame period "
          f"(n = {len(near)}): {near['residual_pp'].min():.2f}-"
          f"{near['residual_pp'].max():.2f} pp")
    print(f"  residual elsewhere (n = {len(far)}): "
          f"{far['residual_pp'].min():.2f}-{far['residual_pp'].max():.2f} pp")
    print(f"  Spearman |cycle/period - 1| vs residual: "
          f"rho = {rho:.3f}, p = {p_rho:.4f}")
    rho_md, p_md = stats.spearmanr((sc["cycle_over_period"] - 1).abs(),
                                   sc["residual_pp_median"])
    print(f"  same on medians instead of means: rho = {rho_md:.3f}, "
          f"p = {p_md:.4f}, within 1 pp "
          f"{int(sc['residual_pp_median'].abs().le(1).sum())} of {len(sc)}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

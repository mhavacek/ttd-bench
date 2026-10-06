#!/usr/bin/env python3
"""Figures + side-by-side table for the FAR calibration deviation (D9).

Outputs in results-far-calib/:
  fig_far_curves.pdf     FAR(threshold) per model, full rate and edge rate
  fig_hit_vs_far.pdf     pooled hit %% vs target FAR, variants (a) and (b)
  old_vs_new.csv         published operating points vs FAR-calibrated ones
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
SHORT = {"yolov8s-weapon": "v8s", "yolov8m-weapon": "v8m",
         "yolov12m-weapon": "v12m", "yolov26m-weapon": "v26m"}


def main() -> int:
    cv = pd.read_parquet(OUT / "far_curves.parquet")
    sm = pd.read_csv(OUT / "phase2_far_summary.csv")

    # --- fig 1: FAR(threshold), full rate (solid) vs edge 1.76 fps (dashed)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = {}
    for m, g in cv[cv.config == "local-gpu"].groupby("model"):
        g = g.sort_values("threshold")
        ln, = ax.plot(g.threshold, g.far_h, label=f"{SHORT[m]} @30 fps", lw=1.5)
        colors[m] = ln.get_color()
    for m, g in cv[cv.config == "edge-sim"].groupby("model"):
        g = g.sort_values("threshold")
        ax.plot(g.threshold, g.far_h, ls="--", color=colors[m], lw=1.2,
                label=f"{SHORT[m]} @1.76 fps")
    # Resolution floors of the 94.03 h corpus (Garwood, 95 % one-sided): below
    # 0.032/h a zero count only supports an upper bound, and an estimate is not
    # +-50 % until 0.170/h. Anything plotted under those lines is a bound, not
    # a measurement, and the figure has to say so.
    ax.axhline(0.170, color="grey", lw=0.8, ls="--")
    ax.text(0.11, 0.20, "+-50 % precision floor (0.170/h)",
            fontsize=6.5, color="grey")
    ax.axhline(0.032, color="grey", lw=0.8, ls=":")
    ax.text(0.11, 0.036, "zero-count bound (0.032/h)",
            fontsize=6.5, color="grey")
    ax.set_yscale("symlog", linthresh=0.03)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("false alarms per camera-hour")
    ax.set_title("FAR vs threshold on the alarm-free corpus (K=3 consecutive)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.savefig(OUT / "fig_far_curves.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- fig 2: pooled hit% vs target FAR
    pool = sm[(sm.config == "ALL") & sm.operating_point.str.startswith("far")].copy()
    pool["tgt"] = pool.operating_point.str.replace("far", "").astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for axi, var, title in [(axes[0], "a", "(a) one threshold @ full rate"),
                            (axes[1], "b", "(b) threshold per configuration")]:
        for m, g in pool[pool.variant == var].groupby("model"):
            g = g.sort_values("tgt")
            axi.plot(g.tgt, g.hit_pct, "o-", lw=1.4, ms=3.5,
                     color=colors.get(m), label=SHORT[m])
        axi.set_xscale("log")
        axi.set_xlabel("target FAR [alarms/h]")
        axi.set_title(title, fontsize=9)
        axi.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("hit rate [%] (pooled, defect cell excluded)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Phase-2 hit rate at FAR-matched operating points", fontsize=10)
    fig.savefig(OUT / "fig_hit_vs_far.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- old vs new side-by-side (pooled per model)
    rows = []
    for m in SHORT:
        r = {"model": SHORT[m]}
        # Variant (a) is the primary threshold variant (one threshold at full
        # frame rate); (b) is secondary analysis, so the side-by-side table
        # that the paper quotes is built from (a).
        for op, var, tag in [("old_conf0.5", "-", "old0.5"),
                             ("old_fp30", "-", "oldFP30"),
                             ("far600", "a", "far600a"),
                             ("far120", "a", "far120a"),
                             ("far30", "a", "far30a"),
                             ("far3", "a", "far3a"),
                             ("far1", "a", "far1a")]:
            g = sm[(sm.operating_point == op) & (sm.variant == var) &
                   (sm.model == m) & (sm.config == "ALL")]
            if g.empty:
                continue
            g = g.iloc[0]
            r[f"{tag}_hit%"] = g.hit_pct
            r[f"{tag}_P1s"] = g.P_alarm_1s
        rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(OUT / "old_vs_new.csv", index=False)
    print(ov.to_string(index=False))
    print(f"\nfigures -> {OUT}/fig_far_curves.pdf, fig_hit_vs_far.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""TTD against false alarms per camera-hour, one curve per configuration.

Reads results-far-dwell/phase2_far_dwell_summary.csv (far_dwell.py). Variant a:
the detector threshold is fixed per detector and the false-alarm rate of every
configuration is derived, so the x position of a point is a MEASURED quantity
and the curves of different configurations are not forced onto a common x.

Top row: Kaplan-Meier P(alarm within 3 s of onset). Bottom row: time by which
the KM estimate reaches 25 % detected (KM quantile of TTD); absent where the
estimate never reaches 25 % within follow-up. Points with no false alarm on
the calibration corpus have no finite rate on a log axis and are omitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# house style of the Scientific Reports figures (fonts, TrueType embedding, colours)
from scirep_figures import COLOR, MM, plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-dwell"
MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]
LABEL = {"yolov8s-weapon": "YOLOv8s", "yolov8m-weapon": "YOLOv8m",
         "yolov12m-weapon": "YOLOv12m", "yolov26m-weapon": "YOLO26m"}
CONFIGS = ["local-gpu", "remote-lan", "remote-wifi", "remote-cellular-4g",
           "edge-sim"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="t0.568")
    ap.add_argument("--variant", default="a")
    args = ap.parse_args()
    sm = pd.read_csv(OUT / "phase2_far_dwell_summary.csv")
    sm = sm[(sm["mode"] == args.mode) & (sm["variant"] == args.variant) &
            (sm["achieved_far_h"] > 0)]
    fig, ax = plt.subplots(2, 4, figsize=(183 * MM, 95 * MM), sharex=True, sharey="row")
    for j, m in enumerate(MODELS):
        for c in CONFIGS:
            g = sm[(sm["model"] == m) & (sm["config"] == c)].sort_values(
                "achieved_far_h")
            if g.empty:
                continue
            ax[0, j].plot(g["achieved_far_h"], g["P_alarm_3s"], "-o", ms=2,
                          color=COLOR[c], label=c)
            q = g[np.isfinite(g["km_q25_ms"])]
            ax[1, j].plot(q["achieved_far_h"], q["km_q25_ms"] / 1000, "-o", ms=2,
                          color=COLOR[c])
        ax[0, j].set_title(LABEL[m])
        ax[0, j].set_ylim(0, 1)
        ax[1, j].set_xscale("log")
        ax[1, j].set_xlabel("False alarms per\ncamera-hour (derived)")
    ax[0, 0].set_ylabel("P(alarm within 3 s of onset)")
    ax[1, 0].set_ylabel("Time to 25 % detected (s)")
    ax[0, 0].legend(frameon=False)
    for a in ax.ravel():
        a.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    f = OUT / f"fig_ttd_vs_far_{args.mode}_{args.variant}.pdf"
    fig.savefig(f)
    fig.savefig(f.with_suffix(".png"), dpi=150)
    print("wrote", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CONSORT-style run-disposition diagram for the Phase-2 experiment.

  python scripts/consort_figure.py --master results-scvd/csv/master.csv

Clinical trials must report what happened to every enrolled subject; replay
benchmarks almost never report what happened to every submitted run. This
figure does: submitted -> completed -> excluded (with reasons) -> analysed,
with the per-configuration attrition that motivates the exclusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "results/scvd-phase2/csv/master.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/scvd-phase2/analysis/consort.pdf"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    df = pd.read_csv(args.master)
    df["valid"] = df["n_frames_processed"] > 0
    n_sub = len(df)
    n_zero = int((~df["valid"]).sum())
    n_ok = int(df["valid"].sum())
    n_onset = int((df["valid"] & df["followup_ms"].notna()).sum())
    n_noonset = n_ok - n_onset

    per = df.groupby("config_label").agg(
        submitted=("valid", "size"), analysed=("valid", "sum"))
    per["excluded"] = per["submitted"] - per["analysed"]
    per["attrition"] = (per["excluded"] / per["submitted"] * 100).round(1)
    per = per.sort_values("attrition")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4", ec="#333"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12", linewidth=0.9,
            facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            linewidth=0.9, color="#333"))

    box(0.6, 8.3, 5.2, 1.2, f"Submitted matrix cells\nn = {n_sub}"
                            "\n(5 deployments x 4 models x 34 scenarios x 5 reps)")
    arrow(3.2, 8.3, 3.2, 6.9)
    box(0.6, 5.7, 5.2, 1.2, f"Produced a raw log with >= 1 processed frame\n"
                            f"n = {n_ok}")
    arrow(3.2, 5.7, 3.2, 4.3)
    box(0.6, 3.1, 5.2, 1.2, f"Reached the annotated weapon onset\nn = {n_onset}")
    arrow(3.2, 3.1, 3.2, 1.7)
    box(0.6, 0.5, 5.2, 1.2, f"ANALYSED (time-to-event)\nn = {n_onset}",
        fc="#e3eef7")

    # exclusion side boxes
    arrow(3.2, 7.6, 6.2, 7.0)
    box(6.3, 6.4, 3.4, 1.2,
        f"Excluded: zero frames\nn = {n_zero} ({n_zero / n_sub * 100:.1f} %)\n"
        "infrastructure failure", fc="#fbeeee")
    arrow(3.2, 5.0, 6.2, 4.4)
    box(6.3, 3.8, 3.4, 1.2,
        f"Excluded: onset not reached\nn = {n_noonset}", fc="#fbeeee")

    ax.text(0.6, 9.75, "Run disposition, Phase 2 (SCVD)", fontsize=10, weight="bold")

    # right panel: attrition by configuration
    y = range(len(per))
    ax2.barh(list(y), per["attrition"], color="#c0504d", height=0.55)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(per.index, fontsize=8)
    ax2.set_xlabel("cells excluded [%]", fontsize=8.5)
    ax2.set_title("Attrition by deployment configuration", fontsize=9)
    ax2.tick_params(labelsize=8)
    for i, (v, e, s) in enumerate(zip(per["attrition"], per["excluded"],
                                      per["submitted"])):
        ax2.text(v + 0.7, i, f"{v:.1f} %  ({e}/{s})", va="center", fontsize=7.5)
    ax2.set_xlim(0, max(per["attrition"]) * 1.42)
    ax2.grid(axis="x", alpha=0.3, linewidth=0.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"submitted {n_sub} | zero-frame {n_zero} | no onset {n_noonset} | "
          f"analysed {n_onset}")
    print(per.to_string())
    print(f"figure -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

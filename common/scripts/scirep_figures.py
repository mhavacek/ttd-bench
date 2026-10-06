#!/usr/bin/env python3
"""Figures of the Scientific Reports version, in Nature style.

Redrawn, not recomputed: every value comes from a stored result that the
IEEE version already quotes (and check_section_numbers.py verified), except
the Kaplan-Meier confidence bands, which come from results/scirep-stats/
(scirep_stats.py). Style follows the Scientific Reports figure guidelines:
sans-serif lettering (Arial), no titles inside figures, panels labelled with
bold lower-case letters, a single space between number and unit, widths of
89 mm (one column) or 183 mm (two columns), TrueType fonts in the PDF.

  scirep_figures.py   -> clanek-2-ttdbench/paper/scirep/figures/fig{2..5}.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/paper/scirep/figures"
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
CONFIGS = ["local-gpu", "remote-lan", "remote-wifi", "remote-cellular-4g", "edge-sim"]
COLOR = {"local-gpu": "#1b6ca8", "remote-lan": "#6baed6", "remote-wifi": "#2ca25f",
         "remote-cellular-4g": "#e6a019", "edge-sim": "#c0392b"}
MODEL = {"yolov8s-weapon": "YOLOv8s", "yolov8m-weapon": "YOLOv8m",
         "yolov12m-weapon": "YOLOv12m", "yolov26m-weapon": "YOLO26m",
         "yolov8n": "YOLOv8n", "yolov8m": "YOLOv8m", "rtdetr": "RT-DETR"}


def panel(ax, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def fig2_phase1() -> None:
    """Per-frame latency components (a) and frame drop rate (b), Phase 1."""
    m = pd.read_csv(ROOT / "results/phase1-final/csv/master.csv")
    m = m[m["n_frames_emitted"] > 0]
    comp = ["pf_dt_acq_ms", "pf_dt_transfer_ms", "pf_dt_infer_ms", "pf_dt_post_ms"]
    g = m.groupby(["config_label", "model"])[comp + ["drop_rate"]].mean()
    # the values Table VIII / Supplementary Table S1 quotes
    assert round(g.loc[("local-gpu", "yolov8n"), "pf_dt_acq_ms"], 2) == 35.38
    assert round(g.loc[("edge-sim", "rtdetr"), "pf_dt_infer_ms"], 2) == 818.68
    assert round(g.loc[("remote-cellular-4g", "rtdetr"), "drop_rate"], 3) == 0.702
    cells = [(c, mo) for c in CONFIGS for mo in ["yolov8n", "yolov8m", "rtdetr"]]
    labels = [f"{MODEL[mo]}" for _, mo in cells]
    x = np.arange(len(cells)) + np.repeat(np.arange(len(CONFIGS)) * 0.6, 3)
    fig, (a, b) = plt.subplots(2, 1, figsize=(183 * MM, 110 * MM), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})
    names = {"pf_dt_acq_ms": "Acquisition", "pf_dt_transfer_ms": "Transfer",
             "pf_dt_infer_ms": "Inference", "pf_dt_post_ms": "Post-processing"}
    cols = ["#9e9e9e", "#e6a019", "#1b6ca8", "#2ca25f"]
    bottom = np.zeros(len(cells))
    for c, col in zip(comp, cols):
        v = np.array([g.loc[k, c] if not np.isnan(g.loc[k, c]) else 0.0 for k in cells])
        a.bar(x, v, 0.8, bottom=bottom, color=col, label=names[c], linewidth=0)
        bottom += v
    k = cells.index(("edge-sim", "rtdetr"))
    a.text(x[k], bottom[k] + 12, "†", ha="center", fontsize=7)
    a.set_ylabel("Mean per-frame latency (ms)")
    a.legend(frameon=False, ncol=2, loc="upper left")
    panel(a, "a")
    b.bar(x, [g.loc[k2, "drop_rate"] * 100 for k2 in cells], 0.8, color="#555555", linewidth=0)
    b.set_ylabel("Frames dropped (%)")
    b.set_ylim(0, 100)
    panel(b, "b")
    b.set_xticks(x)
    b.set_xticklabels(labels, rotation=90)
    for i, cfg in enumerate(CONFIGS):
        b.text(x[3 * i + 1], -44, cfg, ha="center", va="top", fontsize=6.5,
               transform=b.transData, clip_on=False)
    fig.subplots_adjust(bottom=0.21, left=0.08, right=0.99, top=0.95, hspace=0.12)
    fig.savefig(OUT / "fig2_phase1_latency_drop.pdf")
    plt.close(fig)


def fig3_km() -> None:
    """Kaplan-Meier probability of an alarm by time t, with 95 % bands."""
    c = pd.read_csv(ROOT / "results/scirep-stats/km_curves.csv")
    t = pd.read_csv(ROOT / "results/scirep-stats/km_ci.csv")
    n = t.groupby("config")["n"].first()
    fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))
    for cfg in CONFIGS:
        d = c[c["config"] == cfg].sort_values("t_ms")
        s = d["t_ms"] / 1000
        ax.step(s, d["P_alarm"], where="post", color=COLOR[cfg],
                label=f"{cfg} (n = {int(n[cfg])})")
        ax.fill_between(s, d["CI_lo"], d["CI_hi"], step="post", color=COLOR[cfg],
                        alpha=0.15, linewidth=0)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Time since weapon onset (s)")
    ax.set_ylabel("Probability of an alarm by time t")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_km_detection.pdf")
    plt.close(fig)


def fig4_ablation() -> None:
    t = pd.read_csv(ROOT / "clanek-2-ttdbench/results-d10/table7_alarm_ablation.csv").set_index("config")
    rules = [("K=3 consecutive [%]", "3 consecutive frames", "#1b6ca8"),
             ("K=3 in 1 s [%]", "3 frames within 1 s", "#e6a019"),
             ("K=1 bound [%]", "1 frame (upper bound)", "#9e9e9e")]
    x = np.arange(len(CONFIGS))
    fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
    for i, (col, lab, colr) in enumerate(rules):
        ax.bar(x + (i - 1) * 0.26, [t.loc[c, col] for c in CONFIGS], 0.26,
               label=lab, color=colr, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["local-gpu", "remote-lan", "remote-wifi", "remote-\ncellular-4g",
                        "edge-sim"])
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_alarm_rule.pdf")
    plt.close(fig)


def fig5_hit_vs_far() -> None:
    s = pd.read_csv(ROOT / "clanek-2-ttdbench/results-far-calib/phase2_far_summary.csv")
    s = s[(s["config"] == "ALL") & s["operating_point"].str.startswith("far")]
    s = s.assign(target=s["operating_point"].str.replace("far", "").astype(float))
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM), sharey=True)
    colors = {"yolov8s-weapon": "#c0392b", "yolov8m-weapon": "#2ca25f",
              "yolov12m-weapon": "#1b6ca8", "yolov26m-weapon": "#e6a019"}
    # chance level: hit rate that false alarms alone would produce (far_chance.py)
    ch = pd.read_csv(ROOT / "clanek-2-ttdbench/results-far-calib/chance_level.csv")
    ch = ch[(ch["config"] == "ALL") & ch["operating_point"].str.startswith("far")]
    ch = ch.assign(target=ch["operating_point"].str.replace("far", "").astype(float))
    for ax, var, letter in zip(axes, ["a", "b"], ["a", "b"]):
        d = s[s["variant"] == var]
        dc = ch[ch["variant"] == var]
        for m, col in colors.items():
            e = d[d["model"] == m].sort_values("target")
            ax.plot(e["target"], e["hit_pct"], marker="o", markersize=2.5, color=col,
                    label=MODEL[m])
            f = dc[dc["model"] == m].sort_values("target")
            ax.plot(f["target"], f["chance_hit_pct"], linestyle="--", linewidth=0.7,
                    color=col, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 100, 600])
        ax.set_xticklabels(["0.1", "1", "10", "100", "600"])
        ax.minorticks_off()
        ax.set_xlabel("Target false-alarm rate (alarms per camera-hour)")
        panel(ax, letter)
    axes[0].set_ylabel("Hit rate, pooled over configurations (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_hit_vs_far.pdf")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig2_phase1()
    fig3_km()
    fig4_ablation()
    fig5_hit_vs_far()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

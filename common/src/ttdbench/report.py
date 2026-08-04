"""Publication figures and tables.

Four figure types (matplotlib only, serif fonts, vector PDF @ 300 dpi):
  (a) stacked bar chart of the mean TTD decomposition per configuration
  (b) box/violin plots of TTD per configuration × model
  (c) empirical CDF of TTD with TTD50 / TTD90 marked
  (d) heatmap of miss rate, scenario × configuration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed TrueType (journal requirement)
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COMPONENTS = [
    ("dt_acq_ms", r"$\Delta t_{\mathrm{acq}}$", "#4878A8"),
    ("dt_transfer_ms", r"$\Delta t_{\mathrm{transfer}}$", "#E8A03C"),
    ("dt_infer_ms", r"$\Delta t_{\mathrm{infer}}$", "#B85450"),
    ("dt_post_ms", r"$\Delta t_{\mathrm{post}}$", "#82A85C"),
    ("dt_alarm_ms", r"$\Delta t_{\mathrm{alarm}}$", "#8E6FAD"),
]


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ figure a
def fig_decomposition_stacked(df: pd.DataFrame, out: Path) -> Path:
    with plt.rc_context(RC):
        hits = df[df["hit"]]
        means = hits.groupby("config_label", sort=True)[
            [c for c, _, _ in COMPONENTS]
        ].mean()
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        bottom = np.zeros(len(means))
        for col, label, color in COMPONENTS:
            vals = means[col].to_numpy(float)
            ax.bar(means.index, vals, bottom=bottom, label=label, color=color,
                   width=0.65, edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_ylabel("Mean TTD [ms]")
        ax.set_xlabel("Deployment configuration")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend(frameon=False, ncol=2, loc="upper left")
        return _save(fig, out)


# ------------------------------------------------------------------ figure b
def fig_ttd_violin(df: pd.DataFrame, out: Path) -> Path:
    with plt.rc_context(RC):
        hits = df[df["hit"]]
        configs = sorted(hits["config_label"].unique())
        models = sorted(hits["model"].unique())
        fig, ax = plt.subplots(figsize=(max(3.5, 1.1 * len(configs) * len(models)), 2.8))
        pos, labels, data = [], [], []
        p = 0
        for c in configs:
            for m in models:
                vals = hits[(hits["config_label"] == c) & (hits["model"] == m)][
                    "ttd_ms"
                ].dropna().to_numpy(float)
                if len(vals):
                    pos.append(p)
                    labels.append(f"{c}\n{m}" if len(models) > 1 else c)
                    data.append(vals)
                    p += 1
            p += 0.5  # gap between configurations
        parts = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#4878A8")
            body.set_alpha(0.35)
        ax.boxplot(
            data, positions=pos, widths=0.25, showfliers=True,
            flierprops=dict(marker=".", markersize=3),
            medianprops=dict(color="#B85450", linewidth=1.2),
        )
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("TTD [ms]")
        return _save(fig, out)


# ------------------------------------------------------------------ figure c
def fig_ttd_cdf(df: pd.DataFrame, out: Path) -> Path:
    with plt.rc_context(RC):
        hits = df[df["hit"]]
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        cmap = plt.get_cmap("tab10")
        for i, (label, g) in enumerate(sorted(hits.groupby("config_label"))):
            vals = np.sort(g["ttd_ms"].dropna().to_numpy(float))
            if not len(vals):
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            color = cmap(i % 10)
            ax.step(vals, y, where="post", label=label, color=color, linewidth=1.2)
            t50, t90 = np.median(vals), np.percentile(vals, 90)
            ax.plot([t50], [0.5], marker="o", ms=3.5, color=color)
            ax.plot([t90], [0.9], marker="s", ms=3.5, color=color)
        ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
        ax.axhline(0.9, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlabel("TTD [ms]")
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="lower right",
                  title=r"$\bullet$ TTD$_{50}$  $\blacksquare$ TTD$_{90}$")
        return _save(fig, out)


# ------------------------------------------------------------------ figure d
def fig_missrate_heatmap(df: pd.DataFrame, out: Path) -> Path:
    with plt.rc_context(RC):
        pivot = df.pivot_table(
            index="scenario", columns="config_label", values="miss",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(
            figsize=(1.0 + 0.75 * len(pivot.columns), 0.8 + 0.45 * len(pivot.index))
        )
        im = ax.imshow(pivot.to_numpy(float), cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Miss rate", fraction=0.046, pad=0.04)
        return _save(fig, out)


def make_all_figures(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        fig_decomposition_stacked(df, out_dir / "fig_ttd_decomposition.pdf"),
        fig_ttd_violin(df, out_dir / "fig_ttd_violin.pdf"),
        fig_ttd_cdf(df, out_dir / "fig_ttd_cdf.pdf"),
        fig_missrate_heatmap(df, out_dir / "fig_missrate_heatmap.pdf"),
    ]


# ============================================================================
# Phase-1 figures: per-frame latency decomposition (alarm-independent)
# ============================================================================
PF_COMPONENTS = [
    ("pf_dt_acq_ms", r"$\Delta t_{\mathrm{acq}}$", "#4878A8"),
    ("pf_dt_transfer_ms", r"$\Delta t_{\mathrm{transfer}}$", "#E8A03C"),
    ("pf_dt_infer_ms", r"$\Delta t_{\mathrm{infer}}$", "#B85450"),
    ("pf_dt_post_ms", r"$\Delta t_{\mathrm{post}}$", "#82A85C"),
]


def fig_perframe_decomposition(df: pd.DataFrame, out: Path) -> Path:
    """Stacked bar of mean per-frame latency components per config × model."""
    with plt.rc_context(RC):
        d = df.copy()
        d["grp"] = d["config_label"] + "\n" + d["model"].astype(str)
        means = d.groupby("grp", sort=True)[[c for c, _, _ in PF_COMPONENTS]].mean()
        fig, ax = plt.subplots(figsize=(max(3.5, 0.7 * len(means)), 2.8))
        bottom = np.zeros(len(means))
        for col, label, color in PF_COMPONENTS:
            vals = means[col].to_numpy(float)
            ax.bar(means.index, vals, bottom=bottom, label=label, color=color,
                   width=0.7, edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_ylabel("Mean per-frame latency [ms]")
        ax.set_xlabel("Configuration × model")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.legend(frameon=False, ncol=2, loc="upper left")
        return _save(fig, out)


def fig_perframe_infer_violin(df: pd.DataFrame, out: Path) -> Path:
    """Distribution of per-frame inference latency per config × model."""
    with plt.rc_context(RC):
        d = df.dropna(subset=["pf_dt_infer_ms"]).copy()
        configs = sorted(d["config_label"].unique())
        models = sorted(d["model"].astype(str).unique())
        fig, ax = plt.subplots(figsize=(max(3.5, 1.0 * len(configs) * len(models)), 2.8))
        pos, labels, data = [], [], []
        p = 0
        for c in configs:
            for m in models:
                vals = d[(d["config_label"] == c) & (d["model"].astype(str) == m)][
                    "pf_dt_infer_ms"
                ].to_numpy(float)
                if len(vals):
                    pos.append(p)
                    labels.append(f"{c}\n{m}")
                    data.append(vals)
                    p += 1
            p += 0.5
        if data:
            parts = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#B85450")
                body.set_alpha(0.35)
            ax.boxplot(data, positions=pos, widths=0.25, showfliers=False,
                       medianprops=dict(color="#4878A8", linewidth=1.2))
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(r"Per-frame $\Delta t_{\mathrm{infer}}$ [ms]")
        return _save(fig, out)


def fig_droprate_bars(df: pd.DataFrame, out: Path) -> Path:
    """Mean frame drop rate per config × model (operational significance)."""
    with plt.rc_context(RC):
        d = df.copy()
        d["grp"] = d["config_label"] + "\n" + d["model"].astype(str)
        means = d.groupby("grp", sort=True)["drop_rate"].mean()
        fig, ax = plt.subplots(figsize=(max(3.5, 0.7 * len(means)), 2.6))
        ax.bar(means.index, means.to_numpy(float) * 100, color="#8E6FAD", width=0.7)
        ax.set_ylabel("Mean frame drop rate [%]")
        ax.set_xlabel("Configuration × model")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        return _save(fig, out)


def make_phase1_figures(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        fig_perframe_decomposition(df, out_dir / "fig_pf_decomposition.pdf"),
        fig_perframe_infer_violin(df, out_dir / "fig_pf_infer_violin.pdf"),
        fig_droprate_bars(df, out_dir / "fig_pf_droprate.pdf"),
    ]

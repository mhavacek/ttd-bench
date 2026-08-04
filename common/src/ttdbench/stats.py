"""Statistical framework (§3 of the experimental design).

- Descriptives per configuration: mean ± SD, 95% CI (Student t), median
  (TTD50), 90th percentile (TTD90), miss rate.
- Pairwise configuration comparisons on PAIRED samples (paired by
  model × scenario × repetition):
    Shapiro–Wilk on the differences (α = 0.05)
      -> normal: paired two-sided t-test, effect size Cohen's d_z
      -> else  : Wilcoxon signed-rank,   effect size r = |Z| / sqrt(N)
  Bonferroni correction across all pairwise comparisons.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

ALPHA = 0.05

PAIR_KEYS = ["model", "scenario", "repetition"]


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def _ci95_halfwidth(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    return float(sps.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n))


def summarize(
    df: pd.DataFrame,
    metric: str = "ttd_ms",
    group: str = "config_label",
) -> pd.DataFrame:
    """Per-configuration descriptives; misses excluded from latency stats but
    counted in miss_rate."""
    rows = []
    for label, g in df.groupby(group, sort=True):
        vals = g[metric].dropna().to_numpy(dtype=float)
        row = {
            group: label,
            "n_events": len(g),
            "n_hits": int(g["hit"].sum()),
            "miss_rate": float(g["miss"].mean()),
        }
        if len(vals):
            row.update(
                mean=float(np.mean(vals)),
                sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                ci95=_ci95_halfwidth(vals),
                median_ttd50=float(np.median(vals)),
                p90_ttd90=float(np.percentile(vals, 90)),
                min=float(np.min(vals)),
                max=float(np.max(vals)),
            )
        else:
            row.update(mean=np.nan, sd=np.nan, ci95=np.nan, median_ttd50=np.nan,
                       p90_ttd90=np.nan, min=np.nan, max=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition_summary(df: pd.DataFrame, group: str = "config_label") -> pd.DataFrame:
    """Mean of each TTD component per configuration (hits only)."""
    comps = ["dt_acq_ms", "dt_transfer_ms", "dt_infer_ms", "dt_post_ms", "dt_alarm_ms"]
    hits = df[df["hit"]]
    out = hits.groupby(group, sort=True)[comps + ["ttd_ms"]].mean().reset_index()
    return out


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def _paired_vectors(
    df: pd.DataFrame, a: str, b: str, metric: str, group: str
) -> tuple[np.ndarray, np.ndarray]:
    da = df[(df[group] == a) & df["hit"]].set_index(PAIR_KEYS)[metric]
    db = df[(df[group] == b) & df["hit"]].set_index(PAIR_KEYS)[metric]
    joined = pd.concat({"a": da, "b": db}, axis=1, join="inner").dropna()
    return joined["a"].to_numpy(float), joined["b"].to_numpy(float)


def cohens_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else float("inf")


def pairwise_tests(
    df: pd.DataFrame,
    metric: str = "ttd_ms",
    group: str = "config_label",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """All pairwise configuration comparisons with Bonferroni correction."""
    labels = sorted(df[group].dropna().unique())
    pairs = list(combinations(labels, 2))
    m = len(pairs)
    rows = []
    for a, b in pairs:
        xa, xb = _paired_vectors(df, a, b, metric, group)
        n = len(xa)
        row: dict = {"config_a": a, "config_b": b, "n_pairs": n, "metric": metric}
        if n < 3:
            row.update(test="insufficient-pairs", shapiro_p=np.nan, statistic=np.nan,
                       p_value=np.nan, p_bonferroni=np.nan, significant=False,
                       effect_size=np.nan, effect_size_type="",
                       mean_diff=np.nan)
            rows.append(row)
            continue
        diff = xa - xb
        row["mean_diff"] = float(np.mean(diff))
        if np.allclose(diff, diff[0]):
            shapiro_p = 0.0  # constant differences: Shapiro undefined -> nonparametric
        else:
            shapiro_p = float(sps.shapiro(diff).pvalue)
        row["shapiro_p"] = shapiro_p
        if shapiro_p > alpha:
            t_stat, p = sps.ttest_rel(xa, xb)
            row.update(test="paired-t", statistic=float(t_stat), p_value=float(p),
                       effect_size=cohens_dz(diff), effect_size_type="cohen_dz")
        else:
            try:
                w_stat, p = sps.wilcoxon(xa, xb)
            except ValueError:  # all differences zero
                w_stat, p = np.nan, 1.0
            # effect size r = |Z|/sqrt(N) via normal approximation of W
            n_nz = int(np.sum(diff != 0))
            if n_nz > 0 and not np.isnan(w_stat):
                mu_w = n_nz * (n_nz + 1) / 4
                sigma_w = np.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24)
                z = (w_stat - mu_w) / sigma_w if sigma_w > 0 else 0.0
                r = abs(z) / np.sqrt(n_nz)
            else:
                r = 0.0
            row.update(test="wilcoxon", statistic=float(w_stat), p_value=float(p),
                       effect_size=float(r), effect_size_type="r")
        row["p_bonferroni"] = min(1.0, row["p_value"] * m)
        row["significant"] = bool(row["p_bonferroni"] < alpha)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def summary_to_latex(summary: pd.DataFrame, caption: str, label: str) -> str:
    df = summary.copy()
    cols = {
        "config_label": "Configuration",
        "mean": r"Mean [ms]",
        "sd": r"SD [ms]",
        "ci95": r"95\% CI $\pm$ [ms]",
        "median_ttd50": r"TTD$_{50}$ [ms]",
        "p90_ttd90": r"TTD$_{90}$ [ms]",
        "miss_rate": "Miss rate",
        "n_events": "$n$",
    }
    df = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "--")
    body = df.to_latex(index=False, escape=False, column_format="l" + "r" * (len(df.columns) - 1))
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        + body
        + "\\end{table}\n"
    )


def tests_to_latex(tests: pd.DataFrame, caption: str, label: str) -> str:
    df = tests.copy()
    if df.empty:
        return "% no pairwise tests\n"

    def fmt_p(p):
        if pd.isna(p):
            return "--"
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['config_a']} vs.\\ {r['config_b']} & {r['n_pairs']} & {r['test']} & "
            f"{fmt_p(r['p_value'])} & {fmt_p(r['p_bonferroni'])} & "
            f"{'--' if pd.isna(r['effect_size']) else f'{r.effect_size:.2f}'} "
            f"({r['effect_size_type'] or '--'}) & "
            f"{'yes' if r['significant'] else 'no'} \\\\"
        )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{llllllc}\n\\toprule\n"
        "Comparison & $n$ & Test & $p$ & $p_{\\mathrm{Bonf}}$ & Effect size & Sig. \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

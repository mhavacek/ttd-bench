"""Statistical framework: descriptives, paired tests, LaTeX rendering."""

import numpy as np
import pandas as pd
import pytest

from ttdbench import stats


def make_df(configs=("local-gpu", "remote-lan"), n_rep=6, shift=50.0, seed=0):
    """Paired synthetic events: config B systematically slower by `shift` ms."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_rep):
        for scenario in ("s1", "s2"):
            base = 200 + rng.normal(0, 10)
            for i, c in enumerate(configs):
                rows.append(
                    {
                        "config_label": c, "model": "m", "scenario": scenario,
                        "repetition": rep, "hit": True, "miss": False,
                        "ttd_ms": base + i * shift + rng.normal(0, 5),
                        "dt_acq_ms": 20.0, "dt_transfer_ms": 10.0 * i,
                        "dt_infer_ms": 15.0, "dt_post_ms": 2.0,
                        "dt_alarm_ms": 80.0,
                    }
                )
    return pd.DataFrame(rows)


def test_summarize_columns():
    s = stats.summarize(make_df())
    assert set(["mean", "sd", "ci95", "median_ttd50", "p90_ttd90", "miss_rate"]) <= set(s.columns)
    assert len(s) == 2


def test_ci_shrinks_with_n():
    small = stats.summarize(make_df(n_rep=5))
    large = stats.summarize(make_df(n_rep=50))
    assert large["ci95"].iloc[0] < small["ci95"].iloc[0]


def test_paired_test_detects_shift():
    t = stats.pairwise_tests(make_df(shift=50.0, n_rep=8))
    assert len(t) == 1
    row = t.iloc[0]
    assert row["p_bonferroni"] < 0.05
    assert row["significant"]
    assert abs(row["mean_diff"] + 50) < 10  # a - b ~ -50
    assert row["test"] in ("paired-t", "wilcoxon")
    assert abs(row["effect_size"]) > 0.8 or row["effect_size_type"] == "r"


def test_no_difference_not_significant():
    t = stats.pairwise_tests(make_df(shift=0.0, n_rep=8, seed=3))
    assert not t.iloc[0]["significant"]


def test_bonferroni_scales_with_pairs():
    df = make_df(configs=("a", "b", "c"), n_rep=6)
    t = stats.pairwise_tests(df)
    assert len(t) == 3  # C(3,2)
    assert (t["p_bonferroni"] >= t["p_value"] - 1e-12).all()
    assert (t["p_bonferroni"] <= 1.0).all()


def test_miss_rate_counted():
    df = make_df()
    df.loc[df.index[:4], ["hit", "miss"]] = [False, True]
    df.loc[df.index[:4], "ttd_ms"] = None
    s = stats.summarize(df)
    assert s["miss_rate"].sum() > 0


def test_decomposition_summary():
    d = stats.decomposition_summary(make_df())
    assert "dt_transfer_ms" in d.columns
    remote = d[d["config_label"] == "remote-lan"]["dt_transfer_ms"].iloc[0]
    assert remote == pytest.approx(10.0)


def test_latex_output_wellformed():
    df = make_df()
    tex1 = stats.summary_to_latex(stats.summarize(df), "cap", "tab:x")
    tex2 = stats.tests_to_latex(stats.pairwise_tests(df), "cap", "tab:y")
    for tex in (tex1, tex2):
        assert tex.count("\\begin{table}") == 1
        assert tex.count("\\end{table}") == 1
        assert "\\caption{cap}" in tex

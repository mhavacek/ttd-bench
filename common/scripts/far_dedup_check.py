#!/usr/bin/env python3
"""How much does the corpus's own duplication move the calibrated thresholds?

UCF-Crime ships 20 clips twice (the same bytes under a different name in a
different split), so 0.886 h of the 94.03 h corpus is counted twice and its
false alarms are counted twice with it. That is not extra evidence. This
script answers whether it matters, without re-running the full sweep: the
duplicated clips' episode counts are recomputed for every (model, config,
threshold) and subtracted from far_curves.parquet along with their duration,
which yields the exact deduplicated FAR curve. Thresholds are then recalibrated
against it and compared with the published ones.

If the thresholds do not move, the primary numbers stand as they are and the
duplication is a corpus fact to report, not a defect to fix. If they do move,
the sweep has to be re-run on the deduplicated corpus.

  far_dedup_check.py            -> results-far-calib/dedup_check.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from far_sweep import OUT, STRIDE, TARGETS, THR_GRID, episodes  # noqa: E402


def dropped_clips(man: pd.DataFrame) -> list[str]:
    """Second and later occurrence of each MD5, ordered as build_report does."""
    seen, drop = set(), []
    for _, r in man.sort_values(["source", "clip"]).iterrows():
        if r["md5"] in seen:
            drop.append(r["clip"])
        else:
            seen.add(r["md5"])
    return drop


def calibrate(sub: pd.DataFrame, target: float, col: str) -> float | None:
    """Smallest threshold whose estimated FAR <= target (same rule as the sweep)."""
    ok = sub[sub[col] <= target]
    return None if ok.empty else float(ok["threshold"].min())


def independence_test(man: pd.DataFrame, tr: pd.DataFrame) -> list[str]:
    """Is the second copy of a clip a second observation, or the same one twice?

    This is the question that decides whether the corpus is 94.03 h or 93.14 h,
    and it is empirical rather than a matter of taste. The duplicate pairs are
    byte-identical files, so if scoring were deterministic their traces would be
    identical and the second copy would carry no information at all. Both the
    raw scores and the alarm decisions they produce are compared.
    """
    pairs = [tuple(g.sort_values(["source", "clip"])["clip"])
             for _, g in man.groupby("md5") if len(g) == 2]
    names = {c for p in pairs for c in p}
    seq = {(m, c): g.sort_values("frame_idx")["max_conf"].to_numpy()
           for (m, c), g in tr[tr["clip"].isin(names)].groupby(["model", "clip"])}

    same = tot = 0
    worst = 0.0
    for a, b in pairs:
        for m in sorted({k[0] for k in seq}):
            if (m, a) not in seq or (m, b) not in seq:
                continue
            va, vb = seq[(m, a)], seq[(m, b)]
            tot += 1
            if len(va) == len(vb) and np.array_equal(va, vb):
                same += 1
            elif len(va) == len(vb):
                worst = max(worst, float(np.abs(va - vb).max()))

    out = ["", "independence of the duplicate copies:",
           f"  identical traces: {same} of {tot} (model x pair)",
           f"  largest score difference among the rest: {worst:.4f}",
           "  alarm decisions at the calibrated thresholds, variant (a):",
           f"    {'target/h':>9}{'comparisons':>13}{'disagreeing':>13}"]
    th = pd.read_csv(OUT / "thresholds.csv")
    th = th[(th["variant"] == "a") & (th["config"] == "local-gpu")]
    for tgt in TARGETS:
        diff = ncmp = 0
        for _, r in th[th["target_far_h"] == tgt].iterrows():
            for a, b in pairs:
                if (r["model"], a) not in seq or (r["model"], b) not in seq:
                    continue
                for s in STRIDE.values():
                    ncmp += 1
                    diff += (episodes(seq[(r["model"], a)][::s], r["threshold"])
                             != episodes(seq[(r["model"], b)][::s], r["threshold"]))
        if ncmp:
            out.append(f"    {tgt:>9}{ncmp:>13}{diff:>13}")
    out.append("  A pair that never disagrees is one observation recorded twice, "
               "not two hours of camera time.")
    return out


def compare_summaries() -> list[str]:
    """Diff the published hit rates against the deduplicated ones, if present."""
    a_p, b_p = OUT / "phase2_far_summary_withdup.csv", OUT / "phase2_far_summary.csv"
    if not b_p.exists():
        return ["", "phase2_far_summary_withdup.csv not present: run "
                    "far_reanalysis.py --keep-duplicates to complete the check."]
    a, b = pd.read_csv(a_p), pd.read_csv(b_p)
    key = ["operating_point", "variant", "model", "config"]
    m = a.merge(b, on=key, suffixes=("", "_d"))
    out = ["", f"downstream comparison over {len(m)} summary cells "
               f"(hit %, P(alarm<=1s), KM median):"]
    for col in ["hit_pct", "P_alarm_1s", "km_median_ms"]:
        d = m[m[col].fillna(-1) != m[f"{col}_d"].fillna(-1)]
        out.append(f"  {col}: {len(d)} cells differ")
        for _, r in d.iterrows():
            out.append(f"    {r['operating_point']:<12} var {r['variant']} "
                       f"{r['model']:<17} {r['config']:<19} "
                       f"{r[col]} -> {r[f'{col}_d']}")
    return out


def main() -> int:
    man = pd.read_csv(OUT / "corpus_manifest.csv")
    drop = dropped_clips(man)
    t_total_h = man["duration_s"].sum() / 3600
    t_drop_h = man[man["clip"].isin(drop)]["duration_s"].sum() / 3600

    # Both sweeps exist, so the curves are compared directly rather than
    # reconstructed by subtracting the duplicates' episodes from the full run.
    cv = pd.read_parquet(OUT / "far_curves.parquet")                 # primary
    wd = pd.read_parquet(OUT / "far_curves_withdup.parquet")         # robustness
    cv = wd.merge(cv, on=["model", "config", "threshold"],
                  suffixes=("_withdup", ""))

    lines = [
        f"corpus with duplicates {t_total_h:.4f} h, duplicated {t_drop_h:.4f} h "
        f"({100 * t_drop_h / t_total_h:.2f} %), "
        f"primary corpus {t_total_h - t_drop_h:.4f} h",
        f"duplicate clips dropped: {len(drop)}",
        "",
        "calibrated threshold, with duplicates kept vs primary (deduplicated):",
        f"{'model':<17}{'target/h':>9}{'thr_withdup':>13}{'thr':>8}"
        f"{'far_withdup':>13}{'far':>11}{'d_thr':>8}",
    ]
    full = cv[cv["config"] == "local-gpu"]
    moved = 0
    for model, g in full.groupby("model"):
        g = g.sort_values("threshold")
        for tgt in TARGETS:
            a = calibrate(g, tgt, "far_h_withdup")
            b = calibrate(g, tgt, "far_h")
            if a is None and b is None:
                continue
            fa = float(g[g["threshold"] == a]["far_h_withdup"].iloc[0]) if a else float("nan")
            fb = float(g[g["threshold"] == b]["far_h"].iloc[0]) if b else float("nan")
            d = "-" if (a is None or b is None) else f"{b - a:+.3f}"
            if a != b:
                moved += 1
            lines.append(f"{model:<17}{tgt:>9}{a if a else float('nan'):>13.3f}"
                         f"{b if b else float('nan'):>8.3f}{fa:>13.4f}{fb:>11.4f}"
                         f"{d:>8}")
    lines += ["", f"thresholds that move: {moved} of "
                  f"{len(full.groupby('model')) * len(TARGETS)}"]

    # Whether a threshold COULD move is not the question; whether a reported
    # quantity does is. The independence test says why the duplicates were
    # dropped, the summary diff says what dropping them cost.
    lines += independence_test(man, pd.read_parquet(OUT / "calib_traces.parquet"))
    lines += compare_summaries()

    # Bootstrap width: duplicates buy precision the corpus never earned, so the
    # primary intervals must come out WIDER than the robustness run's.
    m = cv[(cv["config"] == "local-gpu") & (cv["n_alarms"] > 0) &
           (cv["n_alarms_withdup"] > 0)].copy()
    m["w_wd"] = (m["boot_hi_withdup"] - m["boot_lo_withdup"]) / m["far_h_withdup"]
    m["w"] = (m["boot_hi"] - m["boot_lo"]) / m["far_h"]
    lo = m[m["far_h"] <= 30]
    lines += [
        "",
        "clip-level bootstrap width, primary vs with duplicates "
        f"({len(m)} thresholds):",
        f"  median relative width {m['w_wd'].median():.4f} -> {m['w'].median():.4f} "
        f"({100 * (m['w'] / m['w_wd'] - 1).median():+.2f} %)",
        f"  wider at {100 * (m['w'] > m['w_wd']).mean():.1f} % of thresholds; "
        f"below 30/h {100 * (lo['w'] / lo['w_wd'] - 1).median():+.2f} % and "
        f"{100 * (lo['w'] > lo['w_wd']).mean():.1f} %",
    ]
    txt = "\n".join(lines) + "\n"
    (OUT / "dedup_check.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

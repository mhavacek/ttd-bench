#!/usr/bin/env python3
"""Chance level of a "hit" at each calibrated operating point.

A run counts as a hit when any qualifying streak occurs after the annotated
onset; there are no bounding boxes, so a false alarm that happens to fall in
the post-onset window also counts. This script gives, for every operating
point, the hit rate that false alarms alone would produce:

    p0(run) = 1 - exp(-lambda * w)

with lambda the false-alarm rate of the run's detector and configuration on
the alarm-free corpus (derived per configuration in variant a, matched in
variant b, from thresholds.csv / far_curves.parquet) and w the run's
post-onset follow-up. The chance level of a cell is the mean of p0 over its
runs. It assumes that the corpus rate transfers to the event clips and that
episodes are Poisson; it is a reference level, not a correction.

Output: results-far-calib/chance_level.csv (same cells as
phase2_far_summary.csv: per model x configuration and pooled over
configurations, "ALL").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CAL = ROOT / "clanek-2-ttdbench/results-far-calib"
FROZEN = ROOT / "results/scvd-phase2/analysis"
DEFECT = [("yolov26m-weapon", "edge-sim")]


def runs() -> pd.DataFrame:
    ru = pd.read_parquet(FROZEN / "traces_runs.parquet")
    ru["config"] = np.where(ru["network_profile"] == "none", ru["deployment"],
                            ru["deployment"] + "-" + ru["network_profile"])
    for m, c in DEFECT:
        ru = ru[~((ru["model"] == m) & (ru["config"] == c))]
    # same validity filter as far_reanalysis.evaluate: positive follow-up
    return ru[ru["followup_ms"] > 0].copy()


def cells(ru: pd.DataFrame, lam: pd.Series, op: str, variant: str) -> list[dict]:
    """lam indexed by (model, config) -> false alarms per hour."""
    d = ru.join(lam.rename("lam"), on=["model", "config"]).dropna(subset=["lam"])
    d["p0"] = 1 - np.exp(-d["lam"] * d["followup_ms"] / 3.6e6)
    out = [{"operating_point": op, "variant": variant, "model": m, "config": c,
            "n": len(g), "far_h": float(g["lam"].iloc[0]),
            "chance_hit_pct": round(g["p0"].mean() * 100, 1)}
           for (m, c), g in d.groupby(["model", "config"])]
    out += [{"operating_point": op, "variant": variant, "model": m, "config": "ALL",
             "n": len(g), "far_h": np.nan,
             "chance_hit_pct": round(g["p0"].mean() * 100, 1)}
            for m, g in d.groupby("model")]
    return out


def main() -> int:
    ru = runs()
    th = pd.read_csv(CAL / "thresholds.csv")
    rows = []
    for (variant, tgt), sub in th.groupby(["variant", "target_far_h"]):
        lam = sub.set_index(["model", "config"])["achieved_far_h"]
        rows += cells(ru, lam, f"far{tgt:g}", variant)
    cv = pd.read_parquet(CAL / "far_curves.parquet")
    c05 = cv[cv["threshold"].round(3) == 0.5].set_index(["model", "config"])["far_h"]
    rows += cells(ru, c05, "old_conf0.5", "-")
    out = pd.DataFrame(rows)
    out.to_csv(CAL / "chance_level.csv", index=False)

    s = pd.read_csv(CAL / "phase2_far_summary.csv")
    j = s.merge(out, on=["operating_point", "variant", "model", "config"],
                suffixes=("", "_chance"))
    if (j["n"] != j["n_chance"]).any():
        bad = j[j["n"] != j["n_chance"]]
        print(bad[["operating_point", "variant", "model", "config", "n", "n_chance"]])
        raise SystemExit("run counts differ from phase2_far_summary.csv")
    pd.set_option("display.width", 200)
    a = j[(j["variant"] == "a") & (j["config"] == "local-gpu")]
    print("local-gpu, variant a: observed hit % (chance %)")
    print(a.assign(cell=a["hit_pct"].astype(str) + " (" + a["chance_hit_pct"].astype(str) + ")")
          .pivot_table(index="operating_point", columns="model", values="cell", aggfunc="first")
          .to_string())
    print(f"\nwrote {CAL / 'chance_level.csv'} ({len(out)} cells); run counts match the summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())

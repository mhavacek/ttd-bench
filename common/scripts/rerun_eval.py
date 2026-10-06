#!/usr/bin/env python3
"""Equivalence evaluation of the repair-run probe against the frozen Phase-2 run.

Implements RERUN-PREREGISTRACE.md §4 over two probe layers:

  - probe:    50 control cells (10 per configuration), CELLS_PER_JOB=10,
              results/probe/csv — the cells ran back to back inside chunks.
  - edgetest: the same 10 edge-sim cells re-run one cell per job (idle gaps
              like the frozen run), results/edgetest/csv — isolates the
              chunking hypothesis for the edge-sim deviation seen in probe.

For every cell, d = probe − frozen is standardized by the pre-registered
per-configuration SD (§4, fixed 2026-08-03) and TOST with margin ±1 SD,
α = 0.05 is run per configuration and pooled. The pooled test is reported
to document that it is blind to a one-configuration systematic shift; the
per-configuration tests carry the verdict.

Frozen data are read-only. Outputs land in results/rerun-eval/:
  percell.csv   one row per (layer, cell): frozen/new values, z per metric
  summary.csv   per (layer, configuration): mean z, TOST p, pass/fail
  verdict.txt   the numbers behind the rerun verdict, printed and saved
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_MASTER = ROOT / "results/scvd-phase2/csv/master.csv"
LAYERS = {
    "probe": ROOT / "results/probe/csv",
    "edgetest": ROOT / "results/edgetest/csv",
}
OUT = ROOT / "results/rerun-eval"

# Pre-registered equivalence SDs (RERUN-PREREGISTRACE.md §4, fixed 2026-08-03):
# pooled between-repetition SD within (cfg x model x scenario) groups.
SD = {
    "local-gpu": {"pf_latency_ms": 1.633, "pf_dt_infer_ms": 0.947, "drop_rate": 0.008},
    "edge-sim": {"pf_latency_ms": 10.517, "pf_dt_infer_ms": 10.276, "drop_rate": 0.003},
    "remote-lan": {"pf_latency_ms": 1.170, "pf_dt_infer_ms": 0.624, "drop_rate": 0.019},
    "remote-wifi": {"pf_latency_ms": 1.544, "pf_dt_infer_ms": 0.806, "drop_rate": 0.022},
    "remote-cellular-4g": {"pf_latency_ms": 7.187, "pf_dt_infer_ms": 1.171, "drop_rate": 0.015},
}
PRIMARY = ["pf_latency_ms", "pf_dt_infer_ms"]
MARGIN = 1.0  # +- 1 SD
ALPHA = 0.05


def config_label(row) -> str:
    if row["network_profile"] == "none":
        return row["deployment"]
    return row["deployment"] + "-" + row["network_profile"]


def tost(z: np.ndarray) -> dict:
    """TOST on standardized differences, H1: |mean| < MARGIN."""
    n = len(z)
    mean, sd = z.mean(), z.std(ddof=1)
    se = sd / np.sqrt(n)
    t_lo = (mean + MARGIN) / se  # H0: mean <= -MARGIN
    t_hi = (mean - MARGIN) / se  # H0: mean >= +MARGIN
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    return {"n": n, "mean_z": mean, "sd_z": sd, "tost_p": p,
            "equivalent": p < ALPHA}


def main() -> int:
    frozen = pd.read_csv(FROZEN_MASTER).set_index("run_id")

    rows = []
    for layer, csv_dir in LAYERS.items():
        for f in sorted(csv_dir.glob("*.csv")):
            new = pd.read_csv(f).iloc[0]
            rid = new["run_id"]
            if rid not in frozen.index:
                print(f"WARN {layer}/{rid}: not in frozen master, skipped")
                continue
            fro = frozen.loc[rid]
            cfg = config_label(new)
            if int(new["n_frames_processed"]) == 0 or int(fro["n_frames_processed"]) == 0:
                print(f"WARN {layer}/{rid}: zero-frame run, skipped")
                continue
            row = {"layer": layer, "run_id": rid, "config": cfg,
                   "model": new["model"], "scenario": new["scenario"]}
            for m in PRIMARY + ["drop_rate"]:
                row[f"frozen_{m}"] = fro[m]
                row[f"new_{m}"] = new[m]
                row[f"d_{m}"] = new[m] - fro[m]
                row[f"z_{m}"] = (new[m] - fro[m]) / SD[cfg][m]
            rows.append(row)
    percell = pd.DataFrame(rows)

    summaries = []
    for (layer, cfg), g in percell.groupby(["layer", "config"]):
        for m in PRIMARY:
            summaries.append({"layer": layer, "config": cfg, "metric": m,
                              **tost(g[f"z_{m}"].to_numpy()),
                              "mean_d_drop_rate": g["d_drop_rate"].mean()})
    for layer, g in percell.groupby("layer"):
        for m in PRIMARY:
            summaries.append({"layer": layer, "config": "POOLED", "metric": m,
                              **tost(g[f"z_{m}"].to_numpy()),
                              "mean_d_drop_rate": g["d_drop_rate"].mean()})
    # Best-case pooled set: the four non-edge configurations from the probe
    # plus edge-sim from edgetest (chunking removed). Documents that the
    # pooled TOST is blind to a one-configuration systematic shift: the
    # deviations cancel to ~0 while edge-sim alone sits ~1.6 SD off.
    best = pd.concat([
        percell[(percell["layer"] == "probe") & (percell["config"] != "edge-sim")],
        percell[percell["layer"] == "edgetest"],
    ])
    for m in PRIMARY:
        summaries.append({"layer": "probe+edgetest", "config": "POOLED", "metric": m,
                          **tost(best[f"z_{m}"].to_numpy()),
                          "mean_d_drop_rate": best["d_drop_rate"].mean()})
    summary = pd.DataFrame(summaries)

    OUT.mkdir(parents=True, exist_ok=True)
    percell.to_csv(OUT / "percell.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)

    lines = []

    def say(s=""):
        lines.append(s)
        print(s)

    say("== RERUN-PREREGISTRACE §4: TOST, margin +-1 SD, alpha 0.05 ==")
    for layer in list(LAYERS) + ["probe+edgetest"]:
        say(f"\n-- layer: {layer} --")
        sub = summary[summary["layer"] == layer]
        for _, r in sub.iterrows():
            say(f"  {r['config']:<20} {r['metric']:<16} n={r['n']:>2} "
                f"mean_z={r['mean_z']:+.2f} tost_p={r['tost_p']:.4f} "
                f"{'EQUIVALENT' if r['equivalent'] else 'NOT equivalent'}")

    edge_p = percell[(percell["layer"] == "probe") & (percell["config"] == "edge-sim")]
    edge_e = percell[percell["layer"] == "edgetest"]
    paired = edge_p.merge(edge_e, on="run_id", suffixes=("_chunk", "_solo"))
    say("\n-- edge-sim: chunked (probe) vs solo (edgetest), same 10 cells --")
    for m in PRIMARY:
        zc = paired[f"z_{m}_chunk"]
        zs = paired[f"z_{m}_solo"]
        say(f"  {m:<16} chunked mean_z={zc.mean():+.2f}  "
            f"solo mean_z={zs.mean():+.2f}  "
            f"chunking share of deviation={(zc.mean() - zs.mean()) / zc.mean():.0%}")
    say(f"  solo runs slower than frozen: "
        f"{(paired['d_pf_latency_ms_solo'] > 0).sum()}/10 cells")

    (OUT / "verdict.txt").write_text("\n".join(lines) + "\n")
    say(f"\nwrote {OUT / 'percell.csv'}, summary.csv, verdict.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

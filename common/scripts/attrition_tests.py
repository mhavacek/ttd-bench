#!/usr/bin/env python3
"""Run disposition and content-independence tests (paper Table 3).

The numbers behind section 4.1 -- how many runs failed, and whether failure
depends on anything the experiment was measuring -- existed only as prose in
docs/DIAGNOSTIKA-VYPADKU.md. A number in the paper needs an artifact that
recomputes it, so this script is that artifact for both phases.

WHAT IS BEING TESTED, AND WHY IT NEEDS CONDITIONING
---------------------------------------------------
Zero-frame runs are infrastructure failures: mediamtx binds a fixed UDP port,
a second concurrent job on the node loses the race for it, and the job dies
before the first frame is decoded. Failure therefore cannot depend on scene
content -- but it does depend on WHEN a cell ran, because a node only fails
once the scheduler puts a second job on it. Cell enumeration is lexicographic,
so configuration is the slowest-varying factor and is aliased with time. Tests
are therefore reported twice: marginally, and conditioned on the concurrency
regime.

The regime indicator is a PROXY, and a crude one: a node is treated as
2-slot from its first observed failure onward (nodes may have oscillated, and
we have no qstat/accounting record to do better). It is honest only because it
is stated -- see the residual configuration effect it leaves in phase 2.

  attrition_tests.py [--out-dir results/attrition]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
PHASES = {
    "phase2-scvd": ROOT / "results/scvd-phase2",
    "phase1-final": ROOT / "results/phase1-final",
}


def load_headers(raw_dir: Path) -> pd.DataFrame:
    """run_id, hostname, start time — only the first line of each log."""
    rows = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        with open(f) as fh:
            h = json.loads(fh.readline())
        c = h.get("cell", {})
        rows.append({
            "run_id": f.stem,
            "hostname": h.get("hostname"),
            "wallclock_utc": h.get("wallclock_utc"),
            "cell_index": c.get("index"),
            "model": c.get("model"),
            "scenario": c.get("scenario"),
            "repetition": c.get("repetition"),
            "deployment": c.get("deployment"),
            "network_profile": c.get("network_profile"),
        })
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["wallclock_utc"], format="ISO8601", utc=True)
    return df


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy: a node is '2-slot' from its first observed failure onward."""
    df = df.sort_values(["hostname", "start"], kind="stable").copy()
    df["regime"] = "1-slot"
    for host, g in df.groupby("hostname"):
        fails = g.index[g["failed"]]
        if len(fails):
            first = g["start"].loc[fails].min()
            df.loc[(df["hostname"] == host) & (df["start"] >= first), "regime"] = "2-slot"
    return df


def chi2_failure_by(df: pd.DataFrame, col: str) -> dict:
    tab = pd.crosstab(df[col], df["failed"])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"factor": col, "n": len(df), "chi2": np.nan, "df": np.nan,
                "p": np.nan, "note": "no variation"}
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return {"factor": col, "n": len(df), "chi2": round(chi2, 2), "df": dof,
            "p": p, "note": ""}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = stats.binomtest(k, n).proportion_ci(method="wilson")
    return (round(lo * 100, 1), round(hi * 100, 1))


def analyse(name: str, base: Path, out_dir: Path) -> pd.DataFrame:
    master = pd.read_csv(base / "csv/master.csv")
    hdr = load_headers(base / "raw")
    df = hdr.merge(master[["run_id", "n_frames_processed"]], on="run_id", how="left")
    df["failed"] = df["n_frames_processed"].fillna(0) <= 0
    df["config_label"] = np.where(df["network_profile"].isna()
                                  | (df["network_profile"] == "none"),
                                  df["deployment"],
                                  df["deployment"] + "-" + df["network_profile"].astype(str))
    df = add_regime(df)

    lines = [f"===== {name} =====",
             f"runs: {len(df)}, zero-frame failures: {int(df['failed'].sum())} "
             f"({df['failed'].mean() * 100:.1f} %), valid: {int((~df['failed']).sum())}"]

    lines.append("\nfailure rate by configuration [%]:")
    by_cfg = df.groupby("config_label")["failed"].agg(["size", "sum", "mean"])
    for cfg, r in by_cfg.iterrows():
        lo, hi = wilson(int(r["sum"]), int(r["size"]))
        lines.append(f"  {cfg:<20} {r['mean'] * 100:5.1f}  "
                     f"({int(r['sum'])}/{int(r['size'])}, Wilson [{lo}, {hi}])")

    lines.append("\nconcurrency regime (proxy: from a node's first failure onward):")
    for reg, g in df.groupby("regime"):
        lines.append(f"  {reg}: {int(g['failed'].sum())}/{len(g)} failed "
                     f"({g['failed'].mean() * 100:.1f} %)")

    lines.append("\nper node: failures before/after the transition")
    for host, g in df.groupby("hostname"):
        g1, g2 = g[g["regime"] == "1-slot"], g[g["regime"] == "2-slot"]
        lines.append(f"  {host:<22} 1-slot {int(g1['failed'].sum())}/{len(g1):<5} "
                     f"2-slot {int(g2['failed'].sum())}/{len(g2)}")

    rows = []
    two = df[df["regime"] == "2-slot"]
    for col in ["scenario", "model", "repetition", "config_label"]:
        m = chi2_failure_by(df, col); m["set"] = "all runs"; rows.append(m)
        c = chi2_failure_by(two, col); c["set"] = "within 2-slot"; rows.append(c)

    # the one configuration that carried an unexplained scenario effect
    if "remote-lan" in set(df["config_label"]):
        lan = df[df["config_label"] == "remote-lan"]
        m = chi2_failure_by(lan, "scenario"); m["set"] = "remote-lan only"
        m["factor"] = "scenario|remote-lan"; rows.append(m)
        c = chi2_failure_by(lan[lan["regime"] == "2-slot"], "scenario")
        c["set"] = "remote-lan, within 2-slot"; c["factor"] = "scenario|remote-lan"
        rows.append(c)

    tests = pd.DataFrame(rows)[["factor", "set", "n", "chi2", "df", "p", "note"]]
    lines.append("\nindependence tests (failure vs factor):")
    lines.append(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # run order within a repetition series: does failure creep up over time?
    rho, prho = stats.spearmanr(df["cell_index"], df["failed"].astype(int))
    lines.append(f"\nfailure vs cell index (run order): Spearman rho = {rho:+.3f}, "
                 f"p = {prho:.3g}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tests.insert(0, "phase", name)
    tests.to_csv(out_dir / f"table3_{name}.csv", index=False)
    by_cfg.to_csv(out_dir / f"failure_by_config_{name}.csv")
    (out_dir / f"report_{name}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    return tests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results/attrition"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    all_tests = [analyse(n, b, out) for n, b in PHASES.items() if b.exists()]
    pd.concat(all_tests).to_csv(out / "table3_all.csv", index=False)
    print(f"wrote {out}/table3_all.csv and per-phase reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())

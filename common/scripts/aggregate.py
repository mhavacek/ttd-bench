#!/usr/bin/env python3
"""Collect all raw JSONL logs into the master event-level CSV.

  python scripts/aggregate.py [--raw-dir results/raw] [--out results/csv/master.csv]

Idempotent: recomputes everything post-hoc from the raw logs (the source of
truth); safe to re-run after adding runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root
sys.path.insert(0, str(REPO / "src"))

from ttdbench.config import ExperimentConfig  # noqa: E402
from ttdbench.metrics import runs_to_dataframe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "clanek-2-ttdbench" / "configs" / "experiment.yaml"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.root / cfg.raw["results_dir"] / "raw"
    out = Path(args.out) if args.out else cfg.root / cfg.raw["results_dir"] / "csv" / "master.csv"

    logs = sorted(raw_dir.glob("*.jsonl"))
    if not logs:
        print(f"no raw logs found in {raw_dir}", file=sys.stderr)
        return 1

    df = runs_to_dataframe(logs, timeout_s=float(cfg.alarm["timeout_s"]),
                           alarm_cfg=cfg.alarm)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} runs from {raw_dir} -> {out}")

    n_hits = int(df["hit"].sum()) if "hit" in df else 0
    if n_hits:
        # event-level view (Phase 2 / synthetic): TTD per configuration
        summ = df.groupby("config_label")["ttd_ms"].describe()
        cols = [c for c in ("count", "mean", "50%") if c in summ.columns]
        print(summ[cols])
    else:
        # latency-only view (Phase 1: no alarms fired, no event-level TTD)
        print(f"no alarm events ({n_hits} hits) — showing per-frame latency:")
        pf = ["pf_dt_infer_ms", "pf_dt_transfer_ms", "pf_latency_ms", "drop_rate"]
        pf = [c for c in pf if c in df.columns]
        print(df.groupby("config_label")[pf].mean().round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Select the control cells for the phase-2 repair run.

  python common/scripts/select_control_cells.py \
      --raw results/scvd-phase2/raw \
      --out clanek-2-ttdbench/hpc/rerun/cells_control.txt

A control cell must satisfy two conditions:

  1. it SUCCEEDED in the frozen run, so a frozen value exists to compare against;
  2. it ran while its node was still single-slot, i.e. no sibling job was
     present at all.

Condition 2 matters because the repair run's comparability test must isolate
the code change. 39 % of the frozen run's valid cells had a sibling that
loaded a model onto the GPU and then died, which measurably shifts timing
(see docs/RERUN-PREREGISTRACE.md §3). Including those cells would mix that
shift into the equivalence test.

The single-slot indicator is a proxy: a node is treated as single-slot until
its first zero-frame run. It cannot be exact without PBS accounting records,
so it is deliberately conservative — it only ever marks cells as 2-slot too
early, never too late, so the selected set is clean.

Selection is deterministic (evenly spaced over the sorted index list within
each configuration, no RNG), so re-running reproduces the same file.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent


def load(raw_dir: str) -> list[dict]:
    runs = []
    for p in glob.glob(f"{raw_dir}/*.jsonl"):
        with open(p) as f:
            lines = f.readlines()
        h = json.loads(lines[0])
        c = h["cell"]
        runs.append(dict(
            idx=c["index"],
            host=h["hostname"],
            t=dt.datetime.strptime(h["wallclock_utc"], "%Y-%m-%dT%H:%M:%SZ"),
            ok=any('"type":"frame"' in l for l in lines),
            cfg=c["deployment"] if not c["network_profile"]
                else f'{c["deployment"]}-{c["network_profile"]}',
        ))
    runs.sort(key=lambda r: r["t"])
    by_host = collections.defaultdict(list)
    for r in runs:
        by_host[r["host"]].append(r)
    first_fail = {h: next((s["t"] for s in v if not s["ok"]), None)
                  for h, v in by_host.items()}
    for r in runs:
        ff = first_fail[r["host"]]
        r["slot2"] = ff is not None and r["t"] >= ff
    return runs


def evenly_spaced(items: list[int], k: int) -> list[int]:
    """k items spread across the sorted list, deterministic, no RNG."""
    if k >= len(items):
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results/scvd-phase2/raw"))
    ap.add_argument("--out", default=str(ROOT / "clanek-2-ttdbench/hpc/rerun/cells_control.txt"))
    ap.add_argument("--per-config", type=int, default=10)
    args = ap.parse_args()

    runs = load(args.raw)
    eligible = collections.defaultdict(list)
    for r in runs:
        if r["ok"] and not r["slot2"]:
            eligible[r["cfg"]].append(r["idx"])

    chosen: list[int] = []
    print(f"{'konfigurace':22s} {'k dispozici':>12s} {'vybrano':>9s}")
    for cfg in sorted(eligible):
        pool = sorted(eligible[cfg])
        pick = evenly_spaced(pool, args.per_config)
        chosen += pick
        print(f"{cfg:22s} {len(pool):12d} {len(pick):9d}")
        if len(pool) < args.per_config:
            print(f"  !! {cfg}: jen {len(pool)} zpusobilych bunek", file=sys.stderr)

    chosen = sorted(set(chosen))
    Path(args.out).write_text("\n".join(str(c) for c in chosen) + "\n")
    print(f"\n{len(chosen)} kontrolnich bunek -> {args.out}")

    # every selected cell must be verifiably valid + single-slot
    by_idx = {r["idx"]: r for r in runs}
    bad = [c for c in chosen if not (by_idx[c]["ok"] and not by_idx[c]["slot2"])]
    print("kontrola:", "vsechny zpusobile" if not bad else f"!! NEZPUSOBILE: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the cell lists for re-running failed cells + a validation control set.

  python scripts/rerun_failed.py --raw-dir ../results-scvd/raw \
      --config configs/experiment-scvd.yaml --out-dir hpc/rerun

Writes:
  cells_failed.txt    every cell whose log has no processed frame (or no log)
  cells_control.txt   a random sample of cells that SUCCEEDED in the frozen run

The control set exists to test the rerun itself: re-running cells that already
worked, with the fixed harness, and comparing their latency distributions
against the frozen run shows whether the fix perturbed the measurement path.
If the control distributions match, repaired and frozen cells may be combined;
if they do not, the whole experiment has to be re-run rather than patched.

Submit each list as one contiguous array (PBS needs >= 2 elements):

  qsub -q gpu -J 0-<n-1> \
    -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb:os=debian13:cl_konos=True \
    -v EXPERIMENT_CONFIG=configs/experiment-scvd.yaml,\
RESULTS_SUBDIR=results-scvd-rerun,CELL_LIST=$PWD/hpc/rerun/cells_failed.txt \
    hpc/pbs_array.sh

RESULTS_SUBDIR deliberately differs from the frozen run: repaired runs land in
their own directory and are only merged after the control check passes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # common/
ROOT = REPO.parent                                     # branch-A root
sys.path.insert(0, str(REPO / "src"))

from ttdbench.config import ExperimentConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "clanek-2-ttdbench" / "configs" / "experiment-scvd.yaml"))
    ap.add_argument("--raw-dir", default=str(ROOT / "results" / "scvd-phase2" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "clanek-2-ttdbench" / "hpc" / "rerun"))
    ap.add_argument("--control-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    cells = cfg.enumerate_cells()
    raw = Path(args.raw_dir)

    by_runid = {c.run_id(): c for c in cells}
    failed, ok, missing = [], [], []
    for rid, cell in by_runid.items():
        p = raw / f"{rid}.jsonl"
        if not p.exists():
            missing.append(cell.index)
            continue
        n_frames = 0
        with open(p) as f:
            for line in f:
                if '"type": "frame"' in line or '"type":"frame"' in line:
                    n_frames += 1
                    break
        (ok if n_frames else failed).append(cell.index)

    failed_all = sorted(failed + missing)
    rng = random.Random(args.seed)
    control = sorted(rng.sample(sorted(ok), min(args.control_n, len(ok))))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells_failed.txt").write_text("\n".join(map(str, failed_all)) + "\n")
    (out / "cells_control.txt").write_text("\n".join(map(str, control)) + "\n")

    print(f"matrix         {len(cells)} cells")
    print(f"succeeded      {len(ok)}")
    print(f"zero-frame     {len(failed)}")
    print(f"log missing    {len(missing)}")
    print(f"-> {out}/cells_failed.txt   ({len(failed_all)} cells)  "
          f"qsub -J 0-{len(failed_all) - 1}")
    print(f"-> {out}/cells_control.txt  ({len(control)} cells)  "
          f"qsub -J 0-{len(control) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

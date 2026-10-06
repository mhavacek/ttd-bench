#!/usr/bin/env python3
"""Compare the modern-GPU column (A100) with the frozen GTX 1080 Ti local-gpu cells.

Phase 1's most attackable result is that RT-DETR misses the frame period even on
a co-located GPU, dropping 12 % of frames. That reads as a property of the
model. It is a property of the card: the column re-runs the identical software
stack on an A100 and asks which parts of the decomposition move.

Runs that never emitted a frame are excluded, and that is not a detail. In the
frozen aggregate a failed run carries drop_rate = 0.0 - a failure encoded as a
perfect score - so any statistic taken over master.csv without filtering
n_frames_emitted > 0 is silently optimistic. Six of the 45 frozen local-gpu
runs are such rows, four of them RT-DETR; none of the 45 modern-GPU runs is.
Both mean and median are reported because they disagree here by half again.

The frozen run is read-only; everything lands in results-phase1-moderngpu/csv/.

  moderngpu_compare.py   -> moderngpu_vs_frozen.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN = ROOT / "results/phase1-final/csv/master.csv"
MODERN = ROOT / "clanek-2-ttdbench/results-phase1-moderngpu/csv/master.csv"
OUT = MODERN.parent / "moderngpu_vs_frozen.csv"
METRICS = ["pf_dt_acq_ms", "pf_dt_infer_ms", "pf_dt_post_ms", "pf_latency_ms",
           "drop_rate"]


def main() -> int:
    fr = pd.read_csv(FROZEN)
    mo = pd.read_csv(MODERN)
    # The frozen matrix crosses five deployments; only the co-located GPU cell
    # is comparable, since the modern column runs no network emulation.
    fr = fr[fr["deployment"] == "local-gpu"]
    if fr.empty:
        print("no local-gpu rows in the frozen aggregate", file=sys.stderr)
        return 1
    n_fr_all, n_mo_all = len(fr), len(mo)
    fr = fr[fr["n_frames_emitted"] > 0]
    mo = mo[mo["n_frames_emitted"] > 0]
    print(f"runs that emitted no frames, excluded: "
          f"{n_fr_all - len(fr)} of {n_fr_all} (GTX 1080 Ti), "
          f"{n_mo_all - len(mo)} of {n_mo_all} (A100)\n")

    rows = []
    for model in sorted(set(fr["model"]) & set(mo["model"])):
        a = fr[fr["model"] == model]
        b = mo[mo["model"] == model]
        for met in METRICS:
            if met not in a or met not in b:
                continue
            rows.append({"model": model, "metric": met,
                         "gtx1080ti_median": round(float(a[met].median()), 4),
                         "a100_median": round(float(b[met].median()), 4),
                         "gtx1080ti_mean": round(float(a[met].mean()), 4),
                         "a100_mean": round(float(b[met].mean()), 4),
                         "ratio_median": (round(float(a[met].median() / b[met].median()), 3)
                                          if b[met].median() else None),
                         "n_1080ti": int(a[met].notna().sum()),
                         "n_a100": int(b[met].notna().sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\n-> {OUT}")

    missing = sorted(set(fr["model"]) ^ set(mo["model"]))
    if missing:
        print(f"models present in only one run: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

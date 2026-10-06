#!/usr/bin/env python3
"""Platform check: cluster CUDA scores vs the Mac MPS scores of the same clips.

The calibration moved to the cluster so the whole corpus is scored on one
platform, on the same inference path as the frozen Phase-2 runs (ultralytics
on CUDA) -- same backend, but at imgsz 640 for every model, whereas replay ran
the yolov8s .pt at its stored 416 (D11, docs/IMGSZ-AUDIT-2026-09-27.md).
Before the Mac results are discarded, the two platforms have to be shown to
agree on clips both have scored — otherwise "one platform" would just
be trading an unknown for a different unknown.

Reference: results-far-calib/calib_traces.parquet (Mac, MPS, imgsz 640) from
the original 0.2728 h SCVD corpus.
Candidate: the per-clip JSONL traces written on the cluster.

Criterion (pre-set): max |delta conf| < 0.01 per model over all shared frames.

  far_platform_check.py --traces <dir> [--n-clips 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
CRITERION = 0.01


def load_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    with open(p) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") == "header":
                clip = d["clip"]
                continue
            rows.append((clip, d["model"], d["frame_idx"], d["max_conf"]))
    return pd.DataFrame(rows, columns=["clip", "model", "frame_idx", "max_conf"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="dir of cluster JSONL traces")
    ap.add_argument("--n-clips", type=int, default=20)
    args = ap.parse_args()

    ref = pd.read_parquet(OUT / "calib_traces.parquet")
    files = sorted(Path(args.traces).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no JSONL traces in {args.traces}")

    ref_clips = set(ref["clip"].unique())
    shared = [f for f in files
              if any(c.startswith(f.stem) for c in ref_clips)][: args.n_clips]
    if not shared:
        raise SystemExit("no clip is present in both the reference and the traces")

    cand = pd.concat([load_jsonl(f) for f in shared], ignore_index=True)
    merged = cand.merge(ref, on=["clip", "model", "frame_idx"],
                        suffixes=("_cluster", "_mac"))
    if merged.empty:
        raise SystemExit("clips matched by name but no frames joined")

    merged["delta"] = (merged["max_conf_cluster"] - merged["max_conf_mac"]).abs()
    rep = (merged.groupby("model")["delta"]
           .agg(frames="size", mean_abs="mean", max_abs="max").round(6))
    print(f"clips compared: {merged['clip'].nunique()}, "
          f"frames: {len(merged)}\n")
    print(rep.to_string())

    worst = float(merged["delta"].max())
    ok = worst < CRITERION
    print(f"\nmax |delta conf| = {worst:.6f}  (criterion < {CRITERION})")
    print("PLATFORMS AGREE" if ok else "PLATFORMS DISAGREE — STOP, do not mix")
    (OUT / "platform_check.txt").write_text(
        f"clips {merged['clip'].nunique()}, frames {len(merged)}\n"
        f"{rep.to_string()}\n\nmax |delta| = {worst:.6f} "
        f"(criterion < {CRITERION}): {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Merge per-clip JSONL traces from the cluster into calib_traces.parquet.

The scoring array writes one JSONL per clip so that a killed job costs one clip.
This folds them back into the single parquet the sweep reads, and refuses to do
so quietly if anything is missing: a corpus that is 3 % short would shift every
per-hour rate without any visible symptom.

  far_traces_merge.py --traces <dir> [--manifest ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--manifest", default=str(OUT / "corpus_manifest.csv"))
    ap.add_argument("--out", default=str(OUT / "calib_traces.parquet"))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--models", default="",
                    help="comma-separated models the traces must contain "
                         "(default: all four; D11 re-score: yolov8s-weapon)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or MODELS
    man = pd.read_csv(args.manifest)
    files = sorted(Path(args.traces).glob("*.jsonl"))
    print(f"manifest clips: {len(man)}, trace files: {len(files)}")

    frames, headers = [], []
    for f in files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("type") == "header":
                    clip = d["clip"]
                    headers.append(d)
                    continue
                frames.append((d["model"], clip, d["frame_idx"], d["t_s"],
                               d["max_conf"]))
    df = pd.DataFrame(frames, columns=["model", "clip", "frame_idx", "t_s",
                                       "max_conf"])
    hdr = pd.DataFrame(headers)
    extra = set(df["model"]) - set(models)
    if extra:
        raise SystemExit(f"traces contain unexpected models: {sorted(extra)}")

    # completeness: every manifest clip present, every model per clip, and the
    # frame count matching what the manifest probed
    missing = set(man["clip"]) - set(df["clip"])
    per = df.groupby("clip")["model"].nunique()
    short_models = per[per < len(models)]
    counts = df.groupby(["clip", "model"]).size().groupby("clip").max()
    expect = man.set_index("clip")["n_frames"]
    short_frames = {c: (int(counts[c]), int(expect[c])) for c in counts.index
                    if c in expect.index and counts[c] < expect[c] * 0.99}

    print(f"clips missing entirely: {len(missing)}")
    print(f"clips with fewer than {len(models)} models: {len(short_models)}")
    print(f"clips short on frames (<99 % of probe): {len(short_frames)}")
    if hdr is not None and not hdr.empty and "gpu" in hdr:
        print(f"GPUs used: {sorted(hdr['gpu'].unique())}")
        print(f"git hashes: {sorted(hdr['git'].unique())}")
        print(f"imgsz: {sorted(hdr['imgsz'].unique())}")

    ok = not missing and short_models.empty and not short_frames
    if not ok and not args.allow_incomplete:
        for c in sorted(missing)[:10]:
            print(f"  MISSING {c}")
        for c, (got, exp) in list(short_frames.items())[:10]:
            print(f"  SHORT {c}: {got}/{exp} frames")
        print("\nrefusing to write an incomplete corpus "
              "(pass --allow-incomplete to override)")
        return 1

    df.to_parquet(args.out, index=False)
    hours = man[man["clip"].isin(df["clip"])]["duration_s"].sum() / 3600
    print(f"\nwrote {args.out}: {len(df)} rows, "
          f"{df['clip'].nunique()} clips, {hours:.4f} h of footage")
    print(f"  zero episodes supports FAR < {3 / hours:.3f}/h")
    print(f"  +-50 % precision needs FAR >= {16 / hours:.2f}/h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

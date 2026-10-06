#!/usr/bin/env python3
"""Offline detector inference over the FAR-calibration corpus.

This is score collection, not replay: no mediamtx, no RTSP, no network
emulation, no latency measurement. Every decodable frame of every corpus clip
is scored by each active Phase-2 model and the maximum weapon-class confidence
is logged from conf 0.1 upwards — the same floor as the frozen Phase-2 logs,
so any threshold >= 0.1 can be evaluated post hoc.

Backend: ultralytics .pt checkpoints (the Phase-2 local-gpu backend). The
edge-sim ONNX backend is spot-checked separately (far_backend_check.py).

Output: results-far-calib/calib_traces.parquet
  columns: model, clip, split, frame_idx, t_s, max_conf
  (frames with no weapon detection at conf >= 0.1 get max_conf = 0.0)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
CKPT = ROOT / "clanek-2-ttdbench/checkpoints"

MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]
WEAPON = {"weapon", "pistol", "knife", "gun", "rifle", "handgun"}
CONF_FLOOR = 0.1
IMGSZ = 640  # replay imgsz of v8m/v12m/v26m and edge-sim v8s; replay .pt v8s ran at 416 (D11)


def main() -> int:
    import argparse

    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="scvd,ucf",
                    help="comma-separated corpus sources to score")
    ap.add_argument("--resume", action="store_true",
                    help="skip models whose per-model shard already exists")
    args = ap.parse_args()
    want = {s.strip() for s in args.sources.split(",")}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = pd.read_csv(OUT / "corpus_manifest.csv")
    if "source" in manifest.columns:
        manifest = manifest[manifest["source"].isin(want)]
    hours = manifest["duration_s"].sum() / 3600
    print(f"corpus: {len(manifest)} clips, {hours:.4f} h, sources={sorted(want)}",
          flush=True)

    # Per-model shards: scoring the whole corpus takes tens of hours on one
    # machine, and losing all of it to a crash at model four is not acceptable.
    shard_dir = OUT / "traces_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for mname in MODELS:
        shard = shard_dir / f"{mname}.parquet"
        if args.resume and shard.exists():
            print(f"{mname}: shard exists, skipped", flush=True)
            continue
        rows = []
        model = YOLO(CKPT / f"{mname}.pt")
        weapon_ids = {i for i, n in model.names.items() if n.lower() in WEAPON}
        assert weapon_ids, f"{mname}: no weapon classes in {model.names}"
        t0 = time.time()
        nf = 0
        for _, clip in manifest.iterrows():
            fps = clip["fps"]
            for i, r in enumerate(model.predict(
                    source=clip["path"], stream=True, conf=CONF_FLOOR,
                    imgsz=IMGSZ, device=device, verbose=False)):
                b = r.boxes
                mc = 0.0
                if b is not None and len(b):
                    sel = [j for j in range(len(b))
                           if int(b.cls[j]) in weapon_ids]
                    if sel:
                        mc = float(max(b.conf[j] for j in sel))
                rows.append((mname, clip["clip"], clip["split"], i,
                             i / fps, mc))
                nf += 1
        dt = time.time() - t0
        pd.DataFrame(rows, columns=["model", "clip", "split", "frame_idx",
                                    "t_s", "max_conf"]).to_parquet(shard,
                                                                   index=False)
        print(f"{mname}: {nf} frames in {dt:.0f} s ({nf / dt:.1f} fps) "
              f"-> {shard.name}", flush=True)

    shards = sorted(shard_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df.to_parquet(OUT / "calib_traces.parquet", index=False)
    print(f"-> {OUT / 'calib_traces.parquet'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

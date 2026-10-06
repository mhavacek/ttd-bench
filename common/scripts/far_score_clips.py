#!/usr/bin/env python3
"""Score calibration-corpus clips for FAR calibration — one clip per invocation.

Designed for a PBS array: the array index selects a slice of the manifest, each
clip is cached as its own JSONL, and an already-cached clip is skipped. A killed
job therefore costs at most one clip, and resubmitting the same array is safe.

  far_score_clips.py --manifest corpus_manifest.csv --out-dir traces \\
                     --index 7 --stride 64            # clips 448..511
  far_score_clips.py ... --all                        # everything, no array

INFERENCE PATH — CORRECTION 2026-09-27 (deviation D11). This docstring used to
claim that replay calls `model.predict(frame, device=..., conf=...)` with no
`imgsz` and therefore runs at the ultralytics default 640 for every model. The
first half is true, the conclusion is not: for a .pt checkpoint ultralytics
carries the checkpoint's own training `imgsz` into predict
(`Model._reset_ckpt_args` keeps `imgsz`), and 640 is used only when the
checkpoint stores none. In the frozen Phase-2 replay yolov8s (trained at 416)
therefore ran at 416 on local-gpu and remote, and at 640 only on edge-sim,
whose ONNX export has a fixed 640 input; yolov8m/v12m/v26m ran at 640
everywhere. This script scores every checkpoint at IMGSZ = 640 (unchanged), so
for yolov8s the calibration matches edge-sim but NOT local-gpu/remote. The
mismatch is reported as a limitation in the paper (option c of
clanek-2-ttdbench/docs/IMGSZ-AUDIT-2026-09-27.md); re-scoring yolov8s at 416
(option a) is pending the author's decision. Behaviour is unchanged.

The .pt checkpoints are scored, never the ONNX exports: .pt on CUDA is the
canonical path shared with replay local-gpu (same backend; for yolov8s not the
same imgsz, see above). Agreement between the two backends
after the D10 fix is documented separately (results-far-calib/backend_check.csv
and D10 in docs/ANALYSIS-PLAN-PHASE2.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]
WEAPON = {"weapon", "pistol", "knife", "gun", "rifle", "handgun"}
CONF_FLOOR = 0.1
IMGSZ = 640  # see module docstring: replay's imgsz for v8m/v12m/v26m and edge-sim v8s;
#               replay local-gpu/remote v8s ran at 416 (D11)


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        return "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", type=int, default=None, help="PBS array index")
    ap.add_argument("--stride", type=int, default=None, help="array size")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sources", default="", help="filter manifest by source")
    # D11 follow-up: re-score selected checkpoints at their DEPLOYED input size.
    # Defaults reproduce the original behaviour exactly (all four models, 640).
    ap.add_argument("--models", default="",
                    help="comma-separated subset of MODELS, e.g. yolov8s-weapon")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help="inference input size (default 640; yolov8s deployed at 416)")
    # The corpus is extremely skewed: median clip 55 s, longest 32 550 s, and
    # ten clips of 1196 hold a third of the footage. Splitting an array by clip
    # COUNT therefore hands one job nine hours of video and another nine
    # minutes. These bounds let the work be split into tiers of comparable
    # duration instead, each submitted with a walltime that fits it.
    ap.add_argument("--min-duration", type=float, default=None,
                    help="only clips at least this many seconds long")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="only clips shorter than this many seconds")
    args = ap.parse_args()

    from ultralytics import YOLO

    man = pd.read_csv(args.manifest)
    if args.sources:
        want = {s.strip() for s in args.sources.split(",")}
        man = man[man["source"].isin(want)]
    if args.min_duration is not None:
        man = man[man["duration_s"] >= args.min_duration]
    if args.max_duration is not None:
        man = man[man["duration_s"] < args.max_duration]
    man = man.sort_values("clip").reset_index(drop=True)

    if not args.all:
        if args.index is None or args.stride is None:
            raise SystemExit("give --all, or both --index and --stride")
        man = man.iloc[args.index::args.stride]

    use_models = [m.strip() for m in args.models.split(",") if m.strip()] or MODELS
    unknown = set(use_models) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = Path(args.ckpt_dir)

    meta = {"gpu": gpu_name(), "git": git_hash(Path(__file__).resolve().parents[2]),
            "host": os.uname().nodename, "imgsz": args.imgsz, "conf_floor": CONF_FLOOR,
            "device": args.device}
    print(f"meta: {json.dumps(meta)}", flush=True)
    print(f"clips assigned: {len(man)} "
          f"({man['duration_s'].sum() / 3600:.2f} h of footage)", flush=True)

    models = {}
    done = skipped = 0
    for _, clip in man.iterrows():
        dest = out_dir / f"{Path(clip['clip']).stem}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        # Temp name must be unique PER PROCESS, not per clip. PBS sends SIGKILL
        # on walltime but the doomed process can still hold its file descriptor
        # and keep writing at its own offset while a later job truncates the
        # same path and writes at zero; the result is two runs interleaved
        # mid-line in one file. That produced two corrupt traces before this
        # was fixed, detectable only because a JSON line began mid-object.
        tmp = dest.with_suffix(f".jsonl.{os.getpid()}.part")
        with open(tmp, "w") as fh:
            fh.write(json.dumps({"type": "header", "clip": clip["clip"],
                                 "source": clip["source"], "fps": clip["fps"],
                                 "n_frames": int(clip["n_frames"]),
                                 "md5": clip["md5"], **meta}) + "\n")
            for mname in use_models:
                if mname not in models:
                    models[mname] = YOLO(ck / f"{mname}.pt")
                model = models[mname]
                wid = {i for i, n in model.names.items() if n.lower() in WEAPON}
                if not wid:
                    raise SystemExit(f"{mname}: no weapon class in {model.names}")
                for i, r in enumerate(model.predict(
                        source=clip["path"], stream=True, conf=CONF_FLOOR,
                        imgsz=args.imgsz, device=args.device, verbose=False)):
                    b = r.boxes
                    mc = 0.0
                    if b is not None and len(b):
                        sel = [j for j in range(len(b)) if int(b.cls[j]) in wid]
                        if sel:
                            mc = float(max(b.conf[j] for j in sel))
                    fh.write(json.dumps({"model": mname, "frame_idx": i,
                                         "t_s": round(i / clip["fps"], 4),
                                         "max_conf": round(mc, 4)}) + "\n")
        tmp.rename(dest)
        done += 1
        print(f"  {clip['clip']}: done ({done}/{len(man)})", flush=True)

    print(f"finished: {done} scored, {skipped} already cached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

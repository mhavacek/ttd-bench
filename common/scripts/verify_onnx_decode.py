#!/usr/bin/env python3
"""Verify the D10 fix on real alarm-free footage, through our own backend.

The unit tests cover both layouts on synthetic arrays. This checks the thing
that actually failed in Phase 2: running the real ONNX exports over Normal-class
clips that contain no weapon, and asserting that no model reports a detection at
the saturated confidence 1.000 any more.

Before the fix, yolov26m (an end2end export, output (1, 300, 6)) reported a
phantom Knife/Handgun at conf exactly 1.000 on 11 of 20 such frames, while
yolov8m (raw head, (1, 6, 8400)) reported 0 of 20.

Usage: verify_onnx_decode.py [--frames N] [--clips N]
Exit code 1 if any saturated detection survives.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "clanek-2-ttdbench/checkpoints"
MANIFEST = ROOT / "clanek-2-ttdbench/results-far-calib/corpus_manifest.csv"
MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]


def main() -> int:
    sys.path.insert(0, str(ROOT / "common/src"))
    from ttdbench.inference import OnnxBackend, is_end2end_output

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20, help="frames per clip")
    ap.add_argument("--clips", type=int, default=3)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    rng = np.random.default_rng(42)
    clips = man.iloc[rng.choice(len(man), args.clips, replace=False)]

    failed = False
    for mname in MODELS:
        backend = OnnxBackend(CKPT / f"{mname}.onnx", cpu_threads=2, conf=0.10)
        shape = backend.session.get_outputs()[0].shape
        layout = "end2end" if is_end2end_output(tuple(shape)) else "raw head"

        n_frames = n_sat = n_det = 0
        for _, clip in clips.iterrows():
            cap = cv2.VideoCapture(clip["path"])
            for _ in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                dets = backend.infer(frame)["detections"]
                n_frames += 1
                n_det += len(dets)
                n_sat += sum(1 for d in dets if d["conf"] >= 0.999)
            cap.release()

        status = "OK" if n_sat == 0 else "FAIL"
        if n_sat:
            failed = True
        print(f"{mname:<18} {str(shape):<16} {layout:<9} "
              f"frames={n_frames:>3} detections={n_det:>4} "
              f"saturated={n_sat:>3}  {status}")

    if failed:
        print("\nFAIL: a model still reports conf 1.000 on weapon-free footage.")
        return 1
    print("\nOK: no saturated detections on alarm-free footage for any model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

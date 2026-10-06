#!/usr/bin/env python3
"""Spot-check: are calibration scores transferable across inference backends?

The FAR calibration runs the .pt checkpoints (ultralytics, the Phase-2
local-gpu backend). Phase-2 edge-sim used the ONNX exports on CPU. If the two
backends produced materially different confidence scores, one calibration
could not serve both. This scores a sample of corpus clips with the ONNX
exports and compares per-frame max weapon-class confidence against the .pt
scores already cached in calib_traces.parquet.

Expected outcome given D10: yolov8s/8m/12m agree closely; yolov26m ONNX is
degenerate (saturated scores).

NOTE (2026-09-27, D11): both sides are scored at IMGSZ = 640. That is the
replay pair for v8m/v12m/v26m, but not for yolov8s: its .pt ran at 416 in
replay local-gpu/remote (ultralytics keeps the checkpoint's training imgsz),
its ONNX at 640 on edge-sim. The v8s row therefore compares .pt@640 with
ONNX@640, not the deployed .pt@416 vs ONNX@640 pair. See
clanek-2-ttdbench/docs/IMGSZ-AUDIT-2026-09-27.md. Behaviour unchanged.

Output: results-far-calib/backend_check.csv (+ stdout summary)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"
CKPT = ROOT / "clanek-2-ttdbench/checkpoints"
MODELS = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon", "yolov26m-weapon"]
WEAPON = {"weapon", "pistol", "knife", "gun", "rifle", "handgun"}
N_CLIPS = 6
CONF_FLOOR = 0.1
IMGSZ = 640  # both backends; not the deployed imgsz of yolov8s .pt (416), see D11


def main() -> int:
    from ultralytics import YOLO

    man = pd.read_csv(OUT / "corpus_manifest.csv")
    tr = pd.read_parquet(OUT / "calib_traces.parquet")
    rng = np.random.default_rng(42)
    sample = man.iloc[rng.choice(len(man), N_CLIPS, replace=False)]

    rows = []
    for mname in MODELS:
        onnx = YOLO(CKPT / f"{mname}.onnx", task="detect")
        wid = {i for i, n in onnx.names.items() if n.lower() in WEAPON}
        for _, clip in sample.iterrows():
            b_scores = []
            for r in onnx.predict(source=clip["path"], stream=True,
                                  conf=CONF_FLOOR, imgsz=IMGSZ, device="cpu",
                                  verbose=False):
                bx = r.boxes
                sel = [float(bx.conf[j]) for j in range(len(bx or []))
                       if int(bx.cls[j]) in wid]
                b_scores.append(max(sel) if sel else 0.0)
            # NB: tr["clip"], not tr.clip — the latter is DataFrame.clip()
            a = tr[(tr["model"] == mname) & (tr["clip"] == clip["clip"])] \
                .sort_values("frame_idx")["max_conf"].to_numpy()
            n = min(len(a), len(b_scores))
            if n == 0:
                print(f"  WARN {mname}/{clip['clip']}: pt={len(a)} "
                      f"onnx={len(b_scores)} frames — skipped", flush=True)
                continue
            a, b = a[:n], np.array(b_scores[:n])
            rows.append({"model": mname, "clip": clip["clip"], "frames": n,
                         "mean_abs_diff": float(np.abs(a - b).mean()),
                         "max_abs_diff": float(np.abs(a - b).max()),
                         "onnx_frac_saturated": float((b >= 0.999).mean()),
                         "agree_at_0.5": float(((a >= .5) == (b >= .5)).mean()),
                         "agree_at_0.3": float(((a >= .3) == (b >= .3)).mean())})
        print(f"{mname} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "backend_check.csv", index=False)
    print(df.groupby("model")[["mean_abs_diff", "max_abs_diff",
                               "onnx_frac_saturated",
                               "agree_at_0.5", "agree_at_0.3"]]
          .mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fetch COCO-pretrained checkpoints and export ONNX for the Phase-1 pilot.

Phase 1 measures the deployment-configuration effect on the per-frame latency
decomposition (dt_acq / dt_transfer / dt_infer / dt_post, drop rate). These
latencies depend on the ARCHITECTURE, not on the trained classes, so the
COCO-pretrained weights give final-quality timing numbers that will carry over
to the weapon-fine-tuned checkpoints in Phase 2. (Event-level TTD / miss rate
still needs the weapon class + verified onsets — Phase 2.)

  python scripts/fetch_coco_checkpoints.py

Downloads yolov8n.pt, yolov8m.pt, rtdetr-l.pt into checkpoints/ (ultralytics
auto-download) and exports matching ONNX (opset 12, fixed 640) for the
onnxruntime edge-sim backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints"

# (ultralytics model name, local basename used in configs/experiment-phase1.yaml)
MODELS = [
    ("yolov8n.pt", "yolov8n"),
    ("yolov8m.pt", "yolov8m"),
    ("rtdetr-l.pt", "rtdetr"),
]


def main() -> int:
    try:
        from ultralytics import YOLO, RTDETR
    except ImportError:
        print("ultralytics not installed — pip install 'ultralytics>=8.2,<9'",
              file=sys.stderr)
        return 1

    CKPT.mkdir(parents=True, exist_ok=True)
    for src, base in MODELS:
        pt = CKPT / f"{base}.pt"
        onnx = CKPT / f"{base}.onnx"
        Model = RTDETR if src.startswith("rtdetr") else YOLO
        print(f"== {base} ==")
        # ultralytics downloads the weight file to CWD by name; load then save
        model = Model(src)
        model.save(str(pt))
        print(f"  weights -> {pt.relative_to(REPO)}")
        if not onnx.exists():
            # opset 17: RT-DETR needs >=16 (grid_sampler); YOLO exports fine too
            exported = model.export(format="onnx", opset=17, imgsz=640, dynamic=False)
            Path(exported).replace(onnx)
            print(f"  onnx    -> {onnx.relative_to(REPO)}")
        else:
            print(f"  onnx    -> exists")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

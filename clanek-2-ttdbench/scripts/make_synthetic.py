#!/usr/bin/env python3
"""Generate the synthetic test scenarios declared in configs/scenarios.yaml.

Each video: dark noisy background ("CCTV at night") in which a bright square
("weapon") appears at the annotated onset frame and drifts slowly. The
SyntheticBackend detects it by intensity thresholding, so the entire pipeline
— replay, RTSP, ingest, inference, alarm, metrics, stats — can be exercised
end-to-end with a deterministic, dependency-free detector and a priori known
ground truth.

Deterministic: seeded per scenario id.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

STYLES = {
    # scenario-tag driven appearance
    "day": {"bg": 70, "obj": 235, "noise": 12},
    "dusk": {"bg": 35, "obj": 215, "noise": 8},
    "near": {"size": 64},
    "far": {"size": 42},
}


def make_video(spec: dict, out_path: Path) -> None:
    fps = int(spec["fps"])
    n_frames = int(spec["duration_s"] * fps)
    onset = int(spec["t_onset_frame"])
    tags = spec.get("tags", {})
    lighting = STYLES.get(tags.get("lighting", "day"), STYLES["day"])
    size = STYLES.get(tags.get("distance", "near"), STYLES["near"])["size"]

    w, h = 640, 480
    seed = int(hashlib.sha256(spec["id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v)")

    x0, y0 = w // 2 - size // 2, h // 2 - size // 2
    for i in range(n_frames):
        frame = np.full((h, w), lighting["bg"], dtype=np.uint8)
        noise = rng.normal(0, lighting["noise"], (h, w))
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 150).astype(np.uint8)
        if i >= onset:
            dx = int(10 * np.sin((i - onset) / fps))  # slow drift
            dy = int(6 * np.cos((i - onset) / fps))
            x, y = x0 + dx, y0 + dy
            frame[y : y + size, x : x + size] = lighting["obj"]
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()
    print(f"  {out_path.name}: {n_frames} frames @ {fps} fps, onset at frame {onset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=REPO / "configs" / "scenarios.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(args.scenarios) as f:
        scen = yaml.safe_load(f)

    made = 0
    for spec in scen["scenarios"]:
        if not spec["id"].startswith("synthetic"):
            continue
        out = REPO / spec["file"]
        if out.exists() and not args.force:
            print(f"  {out.name}: exists, skipping (--force to regenerate)")
            continue
        make_video(spec, out)
        made += 1
    print(f"done ({made} generated)")


if __name__ == "__main__":
    sys.exit(main())

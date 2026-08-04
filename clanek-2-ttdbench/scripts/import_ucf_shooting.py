#!/usr/bin/env python3
"""Import UCF-Crime Shooting videos as TTD-Bench scenarios.

Reads the temporal anomaly annotations (start frame of the anomaly = candidate
t_onset_frame), probes each video with OpenCV (fps, frame count), and emits a
scenarios YAML fragment ready to be merged into configs/scenarios.yaml.

  python scripts/import_ucf_shooting.py \
      --ucf-dir ~/Datasets/ttd-bench/ucf-crime \
      --out configs/scenarios-ucf.yaml

IMPORTANT — before a production run, verify each t_onset_frame manually:
the UCF annotation marks the start of the *anomaly* (shooting event), which
is not necessarily the first frame in which the weapon is visible. Step
through each video (e.g. `ffplay -vf "drawtext=text=%{n}"` or any frame
stepper) and correct the onset. Entries are emitted with tag
`onset_verified: false` to make the pending check explicit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ANNOT_RELPATH = (
    "Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/"
    "Temporal_Anomaly_Annotation.txt"
)


def parse_annotations(path: Path, category: str) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] == category:
            rows.append(
                {"file": parts[0], "start": int(parts[2]), "end": int(parts[3])}
            )
    return rows


def find_video(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-dir", required=True, type=Path)
    ap.add_argument("--category", default="Shooting")
    ap.add_argument("--out", default="configs/scenarios-ucf.yaml", type=Path)
    ap.add_argument("--min-onset-frame", type=int, default=50,
                    help="skip videos where the anomaly starts almost immediately "
                         "(ingest needs a pre-onset lead-in to join the stream)")
    args = ap.parse_args()

    ucf = args.ucf_dir.expanduser()
    annot = ucf / ANNOT_RELPATH
    if not annot.exists():
        print(f"annotations not found: {annot}", file=sys.stderr)
        return 1

    scenarios = []
    skipped = []
    for row in parse_annotations(annot, args.category):
        video = find_video(ucf, row["file"])
        if video is None:
            skipped.append((row["file"], "video file not found (unzip Part-3?)"))
            continue
        if row["start"] < args.min_onset_frame:
            skipped.append((row["file"], f"onset too early ({row['start']})"))
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            skipped.append((row["file"], "cannot open"))
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # CAP_PROP_FRAME_COUNT is metadata and often overstates the number of
        # decodable frames in UCF-Crime files; count by actually decoding, so
        # the replay is guaranteed to reach the onset frame.
        n_frames = 0
        while cap.grab():
            n_frames += 1
        cap.release()
        # the alarm needs headroom after onset (K frames + processing); require
        # at least ~2 s of decodable video beyond the annotated onset
        if row["start"] >= n_frames - int(2 * fps):
            skipped.append(
                (row["file"],
                 f"onset {row['start']} too close to end of decodable stream "
                 f"({n_frames} frames)")
            )
            continue

        sid = video.stem.replace("_x264", "").lower()
        scenarios.append(
            {
                "id": f"ucf-{sid}",
                "file": str(video),
                "fps": round(float(fps), 3),
                "t_onset_frame": row["start"],
                "duration_s": round(n_frames / fps, 1),
                "tags": {
                    "lighting": "unknown",   # fill in manually per video
                    "distance": "unknown",
                    "occlusion": "unknown",
                    "source": "ucf-crime",
                    "onset_verified": False,  # anomaly start, not weapon-visible
                },
            }
        )

    out = {
        "scenarios": scenarios,
        "active_scenarios": [s["id"] for s in scenarios],
    }
    args.out.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"{len(scenarios)} scenarios -> {args.out}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Visual frame-index marker.

The replay engine stamps every outgoing frame with a machine-readable strip
along the top edge encoding the frame index. The ingest side decodes it to
re-identify frames after the lossy encode -> RTSP -> decode chain. This makes
frame correspondence robust to encoder frame drops/reordering and is the basis
for computing dt_acq = t_capture(frame k) - t_emit(frame k) post-hoc.

Layout (strip height MARKER_H px):
  [sync white][sync black][b23 .. b0 of frame_idx][c7 .. c0 XOR checksum]
  = 2 + 24 + 8 = 34 equal-width blocks; white = 1, black = 0.

High-contrast blocks >= ~10 px wide survive H.264 at the bitrates used here;
decoding samples the central region of each block and thresholds at the
midpoint of the observed strip intensity range.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MARKER_H = 16
N_BLOCKS = 34  # 2 sync + 24 index bits + 8 checksum bits
MAX_INDEX = (1 << 24) - 1


def _checksum(idx: int) -> int:
    b = [(idx >> 16) & 0xFF, (idx >> 8) & 0xFF, idx & 0xFF]
    return b[0] ^ b[1] ^ b[2] ^ 0xA5


def stamp(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """Draw the marker strip in place (top MARKER_H rows) and return frame."""
    if not 0 <= frame_idx <= MAX_INDEX:
        raise ValueError(f"frame_idx {frame_idx} exceeds 24-bit marker range")
    h, w = frame.shape[:2]
    bw = w // N_BLOCKS
    if bw < 4:
        raise ValueError(f"frame width {w} too small for marker ({N_BLOCKS} blocks)")

    bits = [1, 0]  # sync
    bits += [(frame_idx >> (23 - i)) & 1 for i in range(24)]
    cs = _checksum(frame_idx)
    bits += [(cs >> (7 - i)) & 1 for i in range(8)]

    strip = np.zeros((MARKER_H, w) + frame.shape[2:], dtype=frame.dtype)
    for i, bit in enumerate(bits):
        x0 = i * bw
        x1 = w if i == N_BLOCKS - 1 else (i + 1) * bw
        strip[:, x0:x1] = 255 if bit else 0
    frame[:MARKER_H] = strip
    return frame


def decode(frame: np.ndarray) -> Optional[int]:
    """Decode the frame index from the marker strip; None if unreadable."""
    h, w = frame.shape[:2]
    if h <= MARKER_H:
        return None
    strip = frame[:MARKER_H]
    if strip.ndim == 3:
        strip = strip.mean(axis=2)
    bw = w // N_BLOCKS

    # sample central 50% of each block, central rows only
    vals = []
    y0, y1 = MARKER_H // 4, MARKER_H - MARKER_H // 4
    for i in range(N_BLOCKS):
        x0 = i * bw
        x1 = w if i == N_BLOCKS - 1 else (i + 1) * bw
        cx0 = x0 + (x1 - x0) // 4
        cx1 = x1 - (x1 - x0) // 4
        vals.append(float(strip[y0:y1, cx0:cx1].mean()))
    vals_arr = np.array(vals)
    lo, hi = vals_arr.min(), vals_arr.max()
    if hi - lo < 40:  # no contrast -> no marker present
        return None
    thresh = (lo + hi) / 2.0
    bits = (vals_arr > thresh).astype(int).tolist()

    if bits[0] != 1 or bits[1] != 0:  # sync check
        return None
    idx = 0
    for b in bits[2:26]:
        idx = (idx << 1) | b
    cs = 0
    for b in bits[26:34]:
        cs = (cs << 1) | b
    if cs != _checksum(idx):
        return None
    return idx

"""Visual frame-index marker: roundtrip, corruption resistance."""

import numpy as np
import pytest

from ttdbench import marker


def frame(w=640, h=480):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("idx", [0, 1, 74, 255, 65535, marker.MAX_INDEX])
def test_roundtrip(idx):
    f = marker.stamp(frame(), idx)
    assert marker.decode(f) == idx


def test_roundtrip_survives_mild_compression_noise():
    f = marker.stamp(frame(), 12345).astype(np.int16)
    noisy = np.clip(f + np.random.default_rng(1).normal(0, 12, f.shape), 0, 255).astype(np.uint8)
    assert marker.decode(noisy) == 12345


def test_no_marker_returns_none():
    f = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert marker.decode(f) is None


def test_corrupted_checksum_rejected():
    f = marker.stamp(frame(), 999)
    # invert a data block in the strip -> checksum must fail
    bw = 640 // marker.N_BLOCKS
    f[: marker.MARKER_H, 5 * bw : 6 * bw] = 255 - f[: marker.MARKER_H, 5 * bw : 6 * bw]
    assert marker.decode(f) is None


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        marker.stamp(frame(), marker.MAX_INDEX + 1)

"""The vectorised alarm-episode counter must equal the frame-by-frame rule.

far_sweep.episodes() was rewritten from an explicit streak loop into run-length
arithmetic to make the 94 h sweep tractable. The two must agree exactly: a
silent divergence would move every calibrated threshold and therefore every
false-alarm rate in the paper, with no visible symptom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from far_sweep import episodes, episodes_ref  # noqa: E402


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_matches_reference_on_random_sequences(k: int) -> None:
    rng = np.random.default_rng(20260908)
    for _ in range(300):
        conf = np.round(rng.random(int(rng.integers(0, 80))), 3)
        thr = float(rng.choice([0.0, 0.25, 0.5, 0.75, 0.9, 0.995]))
        assert episodes(conf, thr, k) == episodes_ref(conf, thr, k)


@pytest.mark.parametrize("conf,k,expected", [
    ([], 3, 0),
    ([1.0, 1.0], 3, 0),                       # streak too short
    ([1.0, 1.0, 1.0], 3, 1),                  # exactly k
    ([1.0] * 10, 3, 1),                       # one long run is one alarm
    ([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 3, 2),   # re-arms after a low frame
    ([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], 3, 1),        # second run too short
    ([0.0] * 5, 3, 0),
    ([1.0], 1, 1),                            # k = 1 fires immediately
])
def test_known_cases(conf: list[float], k: int, expected: int) -> None:
    arr = np.array(conf, dtype=float)
    assert episodes(arr, 0.5, k) == expected
    assert episodes_ref(arr, 0.5, k) == expected


def test_threshold_is_inclusive() -> None:
    """A frame exactly at the threshold qualifies, in both implementations."""
    arr = np.array([0.5, 0.5, 0.5])
    assert episodes(arr, 0.5, 3) == 1
    assert episodes_ref(arr, 0.5, 3) == 1
    assert episodes(arr, 0.5001, 3) == 0

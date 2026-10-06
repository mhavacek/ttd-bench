import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from far_dwell import k_of, streak_fire  # noqa: E402


def test_k_of_rounds_before_ceiling():
    # 0.1 * 30 = 3.0000000000000004 must give 3, not 4
    assert [k_of("t0.1", c) for c in
            ("local-gpu", "remote-wifi", "remote-cellular-4g", "edge-sim")] == [3, 2, 1, 1]
    assert k_of("fixed3", "edge-sim") == 3


def test_streak_fire_matches_frame_loop():
    rng = np.random.default_rng(0)
    q = rng.random(500) < 0.6
    rid = np.repeat(np.arange(10), 50)
    for k in (1, 2, 3, 5):
        ref = np.zeros(500, bool)
        streak = 0
        for i in range(500):
            if i and rid[i] != rid[i - 1]:
                streak = 0
            streak = streak + 1 if q[i] else 0
            ref[i] = streak >= k
        assert (streak_fire(q, rid, np.full(500, k)) == ref).all()

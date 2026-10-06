"""Alarm logic: K-consecutive detections above threshold."""

from ttdbench.alarm import AlarmLogic


def det(conf, cls="weapon"):
    return [{"cls": cls, "conf": conf, "xyxy": [0, 0, 10, 10]}]


def make_alarm(k=3, thr=0.5):
    return AlarmLogic(["weapon", "pistol"], confidence_threshold=thr, k_consecutive=k)


def test_fires_after_k_consecutive():
    a = make_alarm(k=3)
    assert a.update(0, det(0.9)) is None
    assert a.update(1, det(0.9)) is None
    t = a.update(2, det(0.9))
    assert t is not None
    assert a.alarm_frame_idx == 2


def test_single_frame_fp_suppressed():
    a = make_alarm(k=3)
    assert a.update(0, det(0.9)) is None
    assert a.update(1, []) is None          # streak reset
    assert a.update(2, det(0.9)) is None
    assert a.update(3, det(0.9)) is None
    assert a.update(4, det(0.9)) is not None


def test_below_threshold_resets():
    a = make_alarm(k=2, thr=0.5)
    assert a.update(0, det(0.49)) is None
    assert a.update(1, det(0.51)) is None
    assert a.update(2, det(0.51)) is not None


def test_class_filter():
    a = make_alarm(k=1)
    assert a.update(0, det(0.9, cls="person")) is None
    assert a.update(1, det(0.9, cls="pistol")) is not None


def test_class_case_insensitive():
    a = make_alarm(k=1)
    assert a.update(0, det(0.9, cls="Weapon")) is not None


def test_fires_only_once():
    a = make_alarm(k=1)
    assert a.update(0, det(0.9)) is not None
    assert a.update(1, det(0.9)) is None
    assert a.fired


def test_k1_fires_immediately():
    a = make_alarm(k=1)
    assert a.update(0, det(0.5)) is not None  # >= threshold is inclusive

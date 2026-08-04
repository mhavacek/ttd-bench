"""Synthetic backend determinism and the remote inference protocol."""

import threading
import time

import numpy as np
import pytest

from ttdbench.inference import RemoteBackendClient, SyntheticBackend, serve
from ttdbench.marker import MARKER_H


def frame_with_object(size=64, val=235):
    f = np.full((480, 640, 3), 60, dtype=np.uint8)
    f[200 : 200 + size, 300 : 300 + size] = val
    return f


def frame_empty():
    return np.full((480, 640, 3), 60, dtype=np.uint8)


def test_synthetic_detects_object():
    b = SyntheticBackend()
    r = b.infer(frame_with_object())
    assert len(r["detections"]) == 1
    d = r["detections"][0]
    assert d["cls"] == "weapon"
    assert d["conf"] >= 0.5
    assert r["t_infer_end"] >= r["t_infer_start"]
    assert r["t_post"] >= r["t_infer_end"]


def test_synthetic_no_object():
    assert SyntheticBackend().infer(frame_empty())["detections"] == []


def test_synthetic_ignores_marker_strip():
    f = frame_empty()
    f[:MARKER_H] = 255  # bright marker strip must not trigger detections
    assert SyntheticBackend().infer(f)["detections"] == []


def test_synthetic_small_blob_low_conf():
    r = SyntheticBackend().infer(frame_with_object(size=15))
    assert all(d["conf"] < 0.5 for d in r["detections"])


def test_remote_protocol_roundtrip():
    port = 18970
    t = threading.Thread(
        target=serve, args=(SyntheticBackend(), port), daemon=True
    )
    t.start()
    time.sleep(0.3)
    client = RemoteBackendClient("127.0.0.1", port)
    r = client.infer(frame_with_object())
    # transfer timestamps present and ordered within one clock domain
    assert r["t_send"] <= r["t_recv_server"] <= r["t_resp_send"] <= r["t_resp_recv"]
    assert len(r["detections"]) == 1
    assert r["detections"][0]["cls"] == "weapon"
    client.shutdown_server()
    t.join(timeout=5)
    assert not t.is_alive()

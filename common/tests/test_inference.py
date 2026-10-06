"""Synthetic backend determinism, ONNX output decoding, remote protocol."""

import threading
import time

import numpy as np
import pytest

from ttdbench.inference import (
    RemoteBackendClient,
    SyntheticBackend,
    decode_onnx_output,
    is_end2end_output,
    serve,
)
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


def raw_output(n_anchors=8400, nc=2):
    """(1, 4+nc, N) raw YOLO head with one strong box at anchor 0."""
    out = np.zeros((1, 4 + nc, n_anchors), dtype=np.float32)
    out[0, :4, 0] = [320.0, 240.0, 64.0, 64.0]  # xywh, letterboxed coords
    out[0, 4, 0] = 0.83                          # class 0 score
    return out


def end2end_output(n_max=300):
    """(1, N, 6) export with NMS baked in: [x1, y1, x2, y2, conf, cls]."""
    out = np.zeros((1, n_max, 6), dtype=np.float32)
    out[0, 0] = [288.0, 208.0, 352.0, 272.0, 0.42, 1.0]
    return out


def test_layout_discriminator_handles_nc2_ambiguity():
    # with nc = 2 the raw layout is (1, 6, 8400): trailing 6 is not enough
    assert not is_end2end_output(raw_output().shape)
    assert is_end2end_output(end2end_output().shape)


def test_decode_raw_head():
    xyxy, cls_ids, confs = decode_onnx_output(raw_output(), 0.10)
    assert len(confs) == 1
    assert cls_ids[0] == 0
    assert confs[0] == pytest.approx(0.83, abs=1e-4)
    assert xyxy[0] == pytest.approx([288.0, 208.0, 352.0, 272.0], abs=1e-3)


def test_decode_end2end_export():
    xyxy, cls_ids, confs = decode_onnx_output(end2end_output(), 0.10)
    assert len(confs) == 1
    assert cls_ids[0] == 1
    assert confs[0] == pytest.approx(0.42, abs=1e-4)
    assert xyxy[0] == pytest.approx([288.0, 208.0, 352.0, 272.0], abs=1e-3)


def test_end2end_never_reports_phantom_confidence_one():
    """Regression for D10.

    The old code transposed unconditionally, so the six columns of an end2end
    export became six detections and the class-id row produced a detection at
    confidence exactly 1.000 on every frame with any detection at all.
    """
    _, _, confs = decode_onnx_output(end2end_output(), 0.10)
    assert not np.any(confs >= 0.999)


def test_end2end_empty_frame_yields_nothing():
    out = np.zeros((1, 300, 6), dtype=np.float32)  # all-zero conf column
    xyxy, cls_ids, confs = decode_onnx_output(out, 0.10)
    assert len(confs) == 0 and len(xyxy) == 0 and len(cls_ids) == 0


def test_raw_empty_frame_yields_nothing():
    out = np.zeros((1, 6, 8400), dtype=np.float32)
    xyxy, cls_ids, confs = decode_onnx_output(out, 0.10)
    assert len(confs) == 0 and len(xyxy) == 0 and len(cls_ids) == 0


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

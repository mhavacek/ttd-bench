"""Inference backends and the remote inference server/client.

Backends (selected per deployment configuration):
  synthetic    — deterministic brightness-blob detector for the programmatic
                 test scenario; zero ML dependencies. Validates the timing
                 chain, not detection quality.
  ultralytics  — YOLO/RT-DETR .pt checkpoints via the Ultralytics API (GPU).
  onnxruntime  — ONNX checkpoints on CPU with a hard intra-op thread limit
                 (edge-device simulation).

Each backend returns a *timed result* dict:
  {t_pre, t_infer_start, t_infer_end, t_post, detections:[{cls, conf, xyxy}]}
with all timestamps from time.monotonic_ns().

Remote deployment: the backend runs in a SEPARATE PROCESS on the SAME node
(`python -m ttdbench.inference --serve ...`); the client talks to it through
the userspace network-emulation proxy (ttdbench.netproxy) over localhost.
Because both processes share the node's monotonic clock, their timestamps are
directly comparable (single clock domain — see README).

Wire protocol (length-prefixed, TCP):
  request : u32 header_len | JSON{frame_idx, t_send} | u32 jpeg_len | JPEG
  response: u32 header_len | JSON{t_recv_server, t_pre, t_infer_start,
                                  t_infer_end, t_post, t_resp_send, detections}
"""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .logio import now_ns
from .marker import MARKER_H

Detection = dict[str, Any]  # {"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2]}


# ============================================================================
# Backends
# ============================================================================

class SyntheticBackend:
    """Detect the bright synthetic object by thresholding (deterministic).

    The synthetic scenario contains a single high-intensity square on a dark
    background (below the marker strip). Confidence is proportional to blob
    area, saturating at 0.99 — large enough objects clear the 0.5 alarm
    threshold, sub-pixel noise does not.
    """

    name = "synthetic"

    def __init__(self, infer_delay_ms: float = 0.0):
        self.infer_delay_ms = infer_delay_ms

    def infer(self, frame: np.ndarray) -> dict[str, Any]:
        t0 = now_ns()
        gray = cv2.cvtColor(frame[MARKER_H:], cv2.COLOR_BGR2GRAY)
        t_pre = now_ns()

        t_infer_start = now_ns()
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if self.infer_delay_ms > 0:
            deadline = t_infer_start + int(self.infer_delay_ms * 1e6)
            while now_ns() < deadline:
                pass
        t_infer_end = now_ns()

        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 100:
                continue
            conf = min(0.99, area / 2000.0)
            detections.append(
                {
                    "cls": "weapon",
                    "conf": round(float(conf), 4),
                    "xyxy": [int(x), int(y + MARKER_H), int(x + w), int(y + h + MARKER_H)],
                }
            )
        t_post = now_ns()
        return {
            "t_pre": t_pre,
            "t_infer_start": t_infer_start,
            "t_infer_end": t_infer_end,
            "t_post": t_post,
            "detections": detections,
            "_t0": t0,
        }


class UltralyticsBackend:
    """YOLO / RT-DETR inference via the Ultralytics API (.pt checkpoints)."""

    name = "ultralytics"

    def __init__(self, weights: str | Path, device: str = "cuda:0", conf: float = 0.10):
        from ultralytics import YOLO  # heavy import, deferred

        self.device = device
        self.conf = conf
        self.model = YOLO(str(weights))
        # warm-up (first CUDA call compiles kernels; must not pollute timings)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)

    def infer(self, frame: np.ndarray) -> dict[str, Any]:
        t_pre = now_ns()  # ultralytics fuses pre/infer/post; boundaries from its profiler
        t_infer_start = now_ns()
        results = self.model.predict(
            frame, device=self.device, conf=self.conf, verbose=False
        )
        t_infer_end = now_ns()

        r = results[0]
        detections: list[Detection] = []
        names = r.names
        if r.boxes is not None:
            for b in r.boxes:
                detections.append(
                    {
                        "cls": str(names[int(b.cls.item())]),
                        "conf": round(float(b.conf.item()), 4),
                        "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                    }
                )
        t_post = now_ns()

        # Refine boundaries with ultralytics' own per-stage speeds (ms) when
        # available: 'preprocess', 'inference', 'postprocess'. The outer
        # monotonic stamps bound the total; the split follows the profiler.
        speed = getattr(r, "speed", None)
        if speed and all(k in speed for k in ("preprocess", "inference", "postprocess")):
            pre_ns = int(speed["preprocess"] * 1e6)
            inf_ns = int(speed["inference"] * 1e6)
            t_infer_start = t_pre + pre_ns
            t_infer_end = t_infer_start + inf_ns
        return {
            "t_pre": t_pre,
            "t_infer_start": t_infer_start,
            "t_infer_end": t_infer_end,
            "t_post": t_post,
            "detections": detections,
        }


class OnnxBackend:
    """ONNX Runtime CPU inference with a hard thread limit (edge simulation).

    Assumes Ultralytics-exported YOLO ONNX: input (1,3,H,W) float32 0..1,
    output (1, 4+nc, N). Class names are read from ONNX metadata when present.
    """

    name = "onnxruntime"

    def __init__(self, onnx_path: str | Path, cpu_threads: int = 2, conf: float = 0.10):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = cpu_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.conf = conf
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # e.g. [1, 3, 640, 640]
        self.in_h = int(shape[2]) if isinstance(shape[2], int) else 640
        self.in_w = int(shape[3]) if isinstance(shape[3], int) else 640

        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            self.names = eval(meta["names"], {"__builtins__": {}})  # ultralytics dict-literal
        except Exception:
            self.names = {}

        # warm-up
        self.session.run(
            None, {self.input_name: np.zeros((1, 3, self.in_h, self.in_w), np.float32)}
        )

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = img.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        top, left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, r, (left, top)

    def infer(self, frame: np.ndarray) -> dict[str, Any]:
        img, r, (dx, dy) = self._letterbox(frame)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        t_pre = now_ns()

        t_infer_start = now_ns()
        out = self.session.run(None, {self.input_name: blob})[0]
        t_infer_end = now_ns()

        # (1, 4+nc, N) -> (N, 4+nc)
        pred = out[0].T
        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs >= self.conf
        boxes_xywh, cls_ids, confs = boxes_xywh[keep], cls_ids[keep], confs[keep]

        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        idxs = cv2.dnn.NMSBoxes(
            [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy],
            confs.astype(float).tolist(),
            self.conf,
            0.45,
        )
        detections: list[Detection] = []
        for i in np.array(idxs).flatten():
            x1, y1, x2, y2 = xyxy[i]
            detections.append(
                {
                    "cls": str(self.names.get(int(cls_ids[i]), f"class{int(cls_ids[i])}")),
                    "conf": round(float(confs[i]), 4),
                    "xyxy": [
                        round(float((x1 - dx) / r), 1),
                        round(float((y1 - dy) / r), 1),
                        round(float((x2 - dx) / r), 1),
                        round(float((y2 - dy) / r), 1),
                    ],
                }
            )
        t_post = now_ns()
        return {
            "t_pre": t_pre,
            "t_infer_start": t_infer_start,
            "t_infer_end": t_infer_end,
            "t_post": t_post,
            "detections": detections,
        }


def _require_checkpoint(path: Path, model_name: str, kind: str) -> Path:
    """Fail fast with an actionable message instead of a deep torch traceback.

    A missing checkpoint is the single most common misconfiguration (and on an
    HPC array it would otherwise produce hundreds of cryptic stack traces).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} checkpoint for model '{model_name}' not found:\n"
            f"    {path}\n"
            f"Put the file there (or fix `models:` in the experiment config).\n"
            f"Note: COCO-pretrained weights do NOT contain a 'weapon' class — "
            f"event-level TTD needs weapon-fine-tuned checkpoints "
            f"(see docs/RUN-SCVD.md step 0). For a latency-only run without "
            f"weapon classes use configs/experiment-phase1.yaml."
        )
    return path


def build_backend(
    backend: str,
    model_spec: dict[str, Any],
    device: str = "cpu",
    cpu_threads: Optional[int] = None,
    repo_root: Path = Path("."),
) -> Any:
    name = model_spec.get("name", "?")
    if backend == "synthetic" or name == "synthetic":
        return SyntheticBackend()
    if backend == "ultralytics":
        weights = _require_checkpoint(
            repo_root / model_spec["weights"], name, "PyTorch (.pt)"
        )
        return UltralyticsBackend(weights, device=device)
    if backend == "onnxruntime":
        onnx_path = _require_checkpoint(
            repo_root / model_spec.get("onnx", model_spec["weights"]), name, "ONNX"
        )
        return OnnxBackend(onnx_path, cpu_threads=cpu_threads or 2)
    raise ValueError(f"unknown backend {backend!r}")


# ============================================================================
# Remote server / client
# ============================================================================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf += chunk
    return buf


def _send_msg(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    hj = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(struct.pack("!I", len(hj)) + hj + struct.pack("!I", len(payload)) + payload)


def _recv_msg(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (hlen,) = struct.unpack("!I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, hlen))
    (plen,) = struct.unpack("!I", _recv_exact(sock, 4))
    payload = _recv_exact(sock, plen) if plen else b""
    return header, payload


def serve(backend: Any, port: int, host: str = "127.0.0.1") -> None:
    """Blocking single-client inference server (one experiment = one client)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[inference-server] listening on {host}:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                header, payload = _recv_msg(conn)
                t_recv_server = now_ns()
                if header.get("cmd") == "ping":
                    _send_msg(conn, {"ok": True})
                    continue
                if header.get("cmd") == "shutdown":
                    _send_msg(conn, {"ok": True})
                    conn.close()
                    srv.close()
                    return
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                result = backend.infer(frame)
                resp = {
                    "frame_idx": header["frame_idx"],
                    "t_recv_server": t_recv_server,
                    "t_pre": result["t_pre"],
                    "t_infer_start": result["t_infer_start"],
                    "t_infer_end": result["t_infer_end"],
                    "t_post": result["t_post"],
                    "detections": result["detections"],
                    "t_resp_send": now_ns(),
                }
                _send_msg(conn, resp)
        except (ConnectionError, OSError):
            conn.close()
            continue


class RemoteBackendClient:
    """Client side of the remote deployment; connects through the netproxy.

    infer() returns the same timed-result dict as local backends, extended
    with transfer timestamps: t_send, t_recv_server, t_resp_send, t_resp_recv.
    JPEG encoding happens on the client (as a real edge camera would) and is
    bounded by [call start, t_send]; it is part of the preprocessing cost.
    """

    name = "remote"

    def __init__(self, host: str, port: int, jpeg_quality: int = 90,
                 connect_timeout_s: float = 300.0):
        """connect_timeout_s bounds the WHOLE handshake, including waiting for
        the inference-server process to become ready. On HPC nodes a cold
        server start (torch import from NFS + model load + CUDA init) can take
        minutes — far beyond a plain TCP connect timeout. The proxy accepts
        our connection immediately but closes it if its own upstream retry
        window expires, so a single connect+ping attempt is NOT enough: we
        retry the full connect->ping->pong cycle until the deadline.
        """
        import time as _time

        self.jpeg_quality = jpeg_quality
        deadline = _time.monotonic() + connect_timeout_s
        last_err: Optional[Exception] = None
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise ConnectionError(
                    f"inference server not ready via proxy {host}:{port} within "
                    f"{connect_timeout_s:.0f}s: {last_err}"
                )
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # ping may block while the server is still loading the model
                self.sock.settimeout(min(remaining, 60.0))
                _send_msg(self.sock, {"cmd": "ping"})
                _recv_msg(self.sock)
                break  # full path client -> proxy -> server verified
            except (OSError, ConnectionError) as e:
                last_err = e
                try:
                    self.sock.close()
                except Exception:
                    pass
                _time.sleep(1.0)
        self.sock.settimeout(30)

    def infer(self, frame: np.ndarray) -> dict[str, Any]:
        ok, jpeg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        t_send = now_ns()
        _send_msg(self.sock, {"frame_idx": -1, "t_send": t_send}, jpeg.tobytes())
        resp, _ = _recv_msg(self.sock)
        t_resp_recv = now_ns()
        resp.update({"t_send": t_send, "t_resp_recv": t_resp_recv})
        return resp

    def shutdown_server(self) -> None:
        try:
            _send_msg(self.sock, {"cmd": "shutdown"})
            _recv_msg(self.sock)
        except (ConnectionError, OSError):
            pass
        finally:
            self.sock.close()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ============================================================================
# CLI: run the inference server (remote deployment, separate process)
# ============================================================================

def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="TTD-Bench inference server")
    p.add_argument("--serve", action="store_true", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--backend", required=True, choices=["synthetic", "ultralytics", "onnxruntime"])
    p.add_argument("--model-name", default="synthetic")
    p.add_argument("--weights", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument("--repo-root", default=".")
    args = p.parse_args()

    spec: dict[str, Any] = {"name": args.model_name}
    if args.weights:
        spec["weights"] = args.weights
    if args.onnx:
        spec["onnx"] = args.onnx
    backend = build_backend(
        args.backend, spec, device=args.device, cpu_threads=args.cpu_threads,
        repo_root=Path(args.repo_root),
    )
    serve(backend, args.port)


if __name__ == "__main__":
    _main()

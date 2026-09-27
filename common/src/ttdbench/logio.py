"""Raw JSONL log writing/reading — the single source of truth.

One run produces one JSONL file. Line types:

  {"type": "header", ...}                          # provenance (first line)
  {"type": "emit",  "frame_idx": i, "t": ns}       # replay: frame handed to encoder
  {"type": "onset_ref", "frame_idx": i, "t": ns}   # replay: annotated onset frame emitted
  {"type": "frame", "frame_idx": i, "t_capture": ns, "t_decode": ns, "t_pre": ns,
   "t_send": ns|null, "t_recv_server": ns|null, "t_resp_send": ns|null,
   "t_resp_recv": ns|null,
   "t_infer_start": ns, "t_infer_end": ns, "t_post": ns,
   "detections": [{"cls": str, "conf": float, "xyxy": [..]}]}
  {"type": "drop",  "frame_idx": i, "t": ns}       # backpressure drop (drop-oldest)
  {"type": "alarm", "frame_idx": i, "t": ns, "k": K}
  {"type": "end",   "t": ns, "reason": str}

All timestamps are time.monotonic_ns() on one physical node. Derived metrics
are computed POST-HOC from these logs only (ttdbench.metrics), never online.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator


def now_ns() -> int:
    """The single sanctioned clock. Never use time.time() for latencies."""
    return time.monotonic_ns()


def _git_hash(cwd: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _gpu_model() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out.splitlines()[0] if out else "none"
    except Exception:
        return "none"


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for lib in ("numpy", "cv2", "scipy", "pandas", "ultralytics", "onnxruntime"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not-installed"
    return versions


def provenance_header(cell: dict[str, Any], config_hash: str, repo_root: Path) -> dict[str, Any]:
    return {
        "type": "header",
        "schema": 1,
        "cell": cell,
        "config_hash": config_hash,
        "git_hash": _git_hash(repo_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": _gpu_model(),
        "pid": os.getpid(),
        "libs": _lib_versions(),
        "wallclock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "t_monotonic_at_header": now_ns(),
    }


class RawLogWriter:
    """Thread-safe append-only JSONL writer (replay + pipeline share one file)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line buffered
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __enter__(self) -> "RawLogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_log(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

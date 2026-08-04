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


def _git_dirty(cwd: Path) -> bool | None:
    """Whether tracked files differ from HEAD when the run started.

    `git_hash` alone cannot distinguish "this commit ran" from "this commit
    plus uncommitted edits ran", which is exactly the question a reviewer asks.
    On the cluster the answer is normally True and expected: the dataset paths
    in the scenario YAMLs are sed'ed to cluster paths after checkout. What the
    run actually used is pinned by `config_hash`, which is computed from the
    loaded config; this flag covers the source side.

    --untracked-files=no keeps it cheap (the untracked scan is the slow part on
    a shared filesystem). None means the question could not be answered.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def gpu_compute_apps() -> list[dict[str, Any]] | None:
    """Processes holding the GPU right now, from nvidia-smi.

    Recorded at the start and the end of every run so that node contention is a
    measured per-run covariate instead of an assumption. A sibling job that
    loads a model onto the GPU measurably shifts timing (+1-2 % of the median,
    see docs/RERUN-PREREGISTRACE.md §3), and exclusive placement cannot be
    relied on being scheduled — so the analysis stratifies on this instead.

    Returns None when nvidia-smi is unavailable (CPU-only node), which is
    distinct from an empty list ("GPU present, nobody on it").
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    apps: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        pid, _, mem = line.partition(",")
        try:
            apps.append({"pid": int(pid.strip()), "used_mib": int(mem.strip())})
        except ValueError:
            continue  # nvidia-smi prints "[N/A]" for memory under some drivers
    return apps


def _gpu_driver() -> str:
    """Driver and CUDA runtime the node exposes.

    Not recorded until 2026-08-04, so neither frozen phase can tell you which
    driver produced it. That matters: a driver upgrade changes kernel
    scheduling and is a candidate explanation for timing drift between
    sessions that is otherwise indistinguishable from a code change.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out.splitlines()[0] if out else "none"
    except Exception:
        return "none"


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    # torch/torchvision were missing here until 2026-08-03, so the phase-1 and
    # phase-2 frozen logs cannot tell you which torch produced them -- and torch
    # is the engine ultralytics runs on. Recorded from now on.
    for lib in ("numpy", "cv2", "scipy", "pandas", "ultralytics", "onnxruntime",
                "torch", "torchvision"):
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
        "git_dirty": _git_dirty(repo_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": _gpu_model(),
        "gpu_driver": _gpu_driver(),
        "gpu_compute_apps": gpu_compute_apps(),   # node contention at start
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

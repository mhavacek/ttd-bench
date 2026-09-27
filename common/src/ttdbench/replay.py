"""Replay engine: paced 1x real-time re-broadcast of an annotated scenario.

Production mode (`rtsp`):
    mediamtx (RTSP server, unprivileged port, plain subprocess — no root)
      <- ffmpeg (H.264 encode, zerolatency) <- this process piping raw frames
    The pipeline under test consumes rtsp://127.0.0.1:<port>/<path> exactly
    like a live IP camera.

Dev fallback (`direct`):
    Frames are handed to the consumer through an in-process queue at the same
    paced rate. Used where ffmpeg/mediamtx are unavailable; identical log
    semantics, but without the encode/stream/decode chain (dt_acq ~ 0).

Pacing: frame k is emitted at t0 + k/fps, enforced by this process (we do NOT
rely on `ffmpeg -re`), so the reference timestamp t_emit is taken in the same
clock domain (time.monotonic_ns) at the exact moment the frame enters the
encoder. The annotated onset frame additionally produces an `onset_ref` event.
Every frame is stamped with a visual index marker (ttdbench.marker) so the
ingest side can re-identify frames after the lossy streaming chain.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from . import marker
from .logio import RawLogWriter, now_ns

MEDIAMTX_CONFIG = """\
logLevel: warn
rtspAddress: :{port}
hls: no
rtmp: no
webrtc: no
srt: no
api: no
metrics: no
pprof: no
playback: no
paths:
  all:
"""


def _sleep_until(t_target_ns: int) -> None:
    """Hybrid sleep/spin for ~sub-ms pacing accuracy without burning CPU."""
    while True:
        remaining = t_target_ns - now_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:  # > 2 ms: coarse sleep
            time.sleep((remaining - 1_500_000) / 1e9)
        else:  # busy-wait the last stretch
            pass


class MediaMTX:
    """Manage a mediamtx RTSP server as an unprivileged subprocess.

    HPC hardening (observed on MetaCentrum): (a) under NFS/IO load the binary
    can take well over 10 s to start, (b) the deterministic port can be taken
    by an unrelated process of another user on a shared node. start() therefore
    waits up to `start_timeout_s` per attempt and retries on OS-assigned free
    ports; the port actually used is exposed as `.port` (the caller builds the
    RTSP URL only after start() succeeds) and recorded in the raw-log header
    provenance, so determinism of the *measurement* is unaffected.
    """

    def __init__(self, binary: str | Path, port: int, workdir: Path,
                 start_timeout_s: float = 60.0, attempts: int = 3):
        self.binary = Path(binary)
        self.port = port
        self.workdir = workdir
        self.start_timeout_s = start_timeout_s
        self.attempts = attempts
        self.proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None

    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _try_start(self, port: int) -> bool:
        import socket

        cfg = self.workdir / f"mediamtx-{port}.yml"
        cfg.write_text(MEDIAMTX_CONFIG.format(port=port))
        self._log_path = self.workdir / f"mediamtx-{port}.log"
        log_fh = open(self._log_path, "w")
        self.proc = subprocess.Popen(
            [str(self.binary), str(cfg)], stdout=log_fh, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    self.port = port
                    return True
            except OSError:
                if self.proc.poll() is not None:
                    return False  # exited (e.g. port taken by another user)
                time.sleep(0.1)
        self.stop()  # timed out: kill this instance before the next attempt
        return False

    def _last_output(self) -> str:
        try:
            return self._log_path.read_text()[-500:] if self._log_path else ""
        except OSError:
            return ""

    def start(self) -> None:
        if not self.binary.exists():
            # fall back to a PATH-installed binary (e.g. inside the container)
            import shutil

            found = shutil.which("mediamtx")
            if found:
                self.binary = Path(found)
            else:
                raise FileNotFoundError(
                    f"mediamtx binary not found at {self.binary} and not on "
                    "PATH — run scripts/fetch_mediamtx.sh first"
                )
        self.workdir.mkdir(parents=True, exist_ok=True)

        tried = []
        for attempt in range(self.attempts):
            # first attempt: the deterministic per-cell port; retries: an
            # OS-assigned free port (immune to squatters on shared nodes)
            port = self.port if attempt == 0 else self._free_port()
            tried.append(port)
            if self._try_start(port):
                return
        raise RuntimeError(
            f"mediamtx failed to start after {self.attempts} attempts "
            f"(ports {tried}); last output: {self._last_output()!r}"
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class RTSPReplay:
    """Stream a scenario video as a live RTSP camera at 1x speed."""

    def __init__(
        self,
        video_path: str | Path,
        fps: float,
        t_onset_frame: int,
        rtsp_url: str,
        log: RawLogWriter,
        encoder: str = "libx264",
        preset: str = "ultrafast",
        tune: str = "zerolatency",
        lead_in_s: float = 1.0,
    ):
        self.video_path = Path(video_path)
        self.fps = fps
        self.t_onset_frame = t_onset_frame
        self.rtsp_url = rtsp_url
        self.log = log
        self.encoder = encoder
        self.preset = preset
        self.tune = tune
        self.lead_in_s = lead_in_s
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self.finished = threading.Event()
        # Open the source file HERE (constructor, main thread): the ingest
        # side sets process-global OPENCV_FFMPEG_CAPTURE_OPTIONS tuned for
        # low-latency RTSP, and those options break plain file demuxing
        # (premature EOF on imperfect streams). Options are read at open
        # time, so an already-open capture is immune.
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self.video_path}")

    def _ffmpeg_cmd(self, w: int, h: int) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self.encoder,
            "-preset", self.preset,
            "-tune", self.tune,
            "-g", str(max(1, int(self.fps))),   # 1 s GOP: bounded decoder join latency
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]

    def _run(self) -> None:
        cap = self._cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ff = subprocess.Popen(
            self._ffmpeg_cmd(w, h),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert ff.stdin is not None

        frame_interval_ns = int(1e9 / self.fps)
        # lead-in: give the encoder pipeline and the ingest client time to
        # establish the RTSP session before frame 0 is due
        t0 = now_ns() + int(self.lead_in_s * 1e9)
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                marker.stamp(frame, frame_idx)
                _sleep_until(t0 + frame_idx * frame_interval_ns)
                t_emit = now_ns()
                ff.stdin.write(frame.tobytes())
                self.log.write({"type": "emit", "frame_idx": frame_idx, "t": t_emit})
                if frame_idx == self.t_onset_frame:
                    self.log.write(
                        {"type": "onset_ref", "frame_idx": frame_idx, "t": t_emit}
                    )
                frame_idx += 1
        finally:
            cap.release()
            try:
                ff.stdin.close()
            except Exception:
                pass
            try:
                ff.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ff.kill()
            if ff.returncode not in (0, None):
                err = ff.stderr.read().decode(errors="replace") if ff.stderr else ""
                if err.strip():
                    self._error = RuntimeError(f"ffmpeg replay error: {err[-2000:]}")
            self.finished.set()

    def start(self) -> None:
        def target():
            try:
                self._run()
            except BaseException as e:  # surfaced via join()
                self._error = e
                self.finished.set()

        self._thread = threading.Thread(target=target, name="rtsp-replay", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)
        if self._error:
            raise self._error


class DirectReplay:
    """In-process paced replay (dev fallback, no RTSP chain).

    Consumer side: iterate .frames_queue like an ingest source. Emitted items:
    (frame_idx, frame, t_emit); None terminates.
    """

    def __init__(
        self,
        video_path: str | Path,
        fps: float,
        t_onset_frame: int,
        log: RawLogWriter,
        queue_size: int = 4,
    ):
        self.video_path = Path(video_path)
        self.fps = fps
        self.t_onset_frame = t_onset_frame
        self.log = log
        self.frames_queue: "queue.Queue[Optional[tuple[int, np.ndarray, int]]]" = (
            queue.Queue(maxsize=queue_size)
        )
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self.finished = threading.Event()
        # open eagerly, before any RTSP capture options pollute the process
        # environment (see RTSPReplay.__init__)
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self.video_path}")

    def _run(self) -> None:
        cap = self._cap
        frame_interval_ns = int(1e9 / self.fps)
        t0 = now_ns() + frame_interval_ns
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                marker.stamp(frame, frame_idx)
                _sleep_until(t0 + frame_idx * frame_interval_ns)
                t_emit = now_ns()
                self.log.write({"type": "emit", "frame_idx": frame_idx, "t": t_emit})
                if frame_idx == self.t_onset_frame:
                    self.log.write(
                        {"type": "onset_ref", "frame_idx": frame_idx, "t": t_emit}
                    )
                self.frames_queue.put((frame_idx, frame, t_emit))
                frame_idx += 1
        finally:
            cap.release()
            self.frames_queue.put(None)
            self.finished.set()

    def start(self) -> None:
        def target():
            try:
                self._run()
            except BaseException as e:
                self._error = e
                self.frames_queue.put(None)
                self.finished.set()

        self._thread = threading.Thread(target=target, name="direct-replay", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)
        if self._error:
            raise self._error

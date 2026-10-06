"""Stream ingest: consume the RTSP stream like a live camera.

A reader thread pulls frames as fast as the stream delivers them, decodes the
visual index marker and pushes (frame_idx, frame, t_capture, t_decode) into a
bounded queue. When the downstream worker cannot keep up, the OLDEST queued
frame is dropped (drop-oldest backpressure) and the drop is logged — the drop
rate is an operationally significant metric and is reported per run.

Timestamps:
  t_capture — monotonic_ns immediately after VideoCapture.read() returns
              (frame fully received & decoded by FFmpeg backend)
  t_decode  — monotonic_ns after marker decode (frame identity known)

dt_acq for frame k is computed post-hoc as t_capture(k) - t_emit(k) using the
replay-side emit log; it covers encode + RTSP transport + decode of the
camera chain.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from . import marker
from .logio import RawLogWriter, now_ns

FrameItem = tuple[int, np.ndarray, int, int]  # frame_idx, frame, t_capture, t_decode


class RTSPIngest:
    def __init__(
        self,
        rtsp_url: str,
        log: RawLogWriter,
        queue_size: int = 1,
        open_timeout_s: float = 15.0,
        read_timeout_s: float = 5.0,
    ):
        self.rtsp_url = rtsp_url
        self.log = log
        self.queue: "queue.Queue[Optional[FrameItem]]" = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self.n_received = 0
        self.n_dropped = 0
        self.n_unreadable = 0

    def _open(self) -> cv2.VideoCapture:
        # Force TCP transport for deterministic delivery and disable receiver
        # buffering (nobuffer/low_delay/max_delay=0) — with default options the
        # FFmpeg demuxer holds several hundred ms of frames, which would be
        # misattributed to dt_acq. Overridable via the environment.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
            "|analyzeduration;500000|probesize;65536",
        )
        # Single-threaded decoding is essential for latency validity: FFmpeg's
        # default frame-threaded H.264 decoder pipelines ~n_cores frames, which
        # would add n_cores/fps of constant, machine-dependent latency to
        # dt_acq (measured +360 ms on a 10-core node at 25 fps).
        params = (
            [cv2.CAP_PROP_N_THREADS, 1] if hasattr(cv2, "CAP_PROP_N_THREADS") else []
        )
        import time as _time

        deadline = _time.monotonic() + self.open_timeout_s
        while _time.monotonic() < deadline and not self._stop.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG, params)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
            _time.sleep(0.25)
        raise TimeoutError(f"could not open stream {self.rtsp_url}")

    def _put_drop_oldest(self, item: FrameItem) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = self.queue.get_nowait()
                    if dropped is not None:
                        self.n_dropped += 1
                        self.log.write(
                            {"type": "drop", "frame_idx": dropped[0], "t": now_ns()}
                        )
                except queue.Empty:
                    pass

    def _run(self) -> None:
        cap = self._open()
        try:
            consecutive_failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                t_capture = now_ns()
                if not ok:
                    consecutive_failures += 1
                    # stream ended (replay finished tearing down) after enough misses
                    if consecutive_failures >= 5:
                        break
                    continue
                consecutive_failures = 0
                idx = marker.decode(frame)
                t_decode = now_ns()
                if idx is None:
                    self.n_unreadable += 1
                    continue
                self.n_received += 1
                self._put_drop_oldest((idx, frame, t_capture, t_decode))
        finally:
            cap.release()
            self._put_drop_oldest_none()

    def _put_drop_oldest_none(self) -> None:
        while True:
            try:
                self.queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    dropped = self.queue.get_nowait()
                    if dropped is not None:
                        self.n_dropped += 1
                        self.log.write(
                            {"type": "drop", "frame_idx": dropped[0], "t": now_ns()}
                        )
                except queue.Empty:
                    pass

    def start(self) -> None:
        def target():
            try:
                self._run()
            except BaseException as e:
                self._error = e
                self._put_drop_oldest_none()

        self._thread = threading.Thread(target=target, name="rtsp-ingest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)
        if self._error:
            raise self._error


class DirectIngest:
    """Adapter giving DirectReplay's queue the same interface as RTSPIngest.

    In direct mode the 'capture' is the queue handoff itself, so t_capture is
    taken at pop time and dt_acq measures only the in-process handoff (~0).
    """

    def __init__(self, replay_queue: "queue.Queue", log: RawLogWriter, queue_size: int = 1):
        self._src = replay_queue
        self.log = log
        self.queue: "queue.Queue[Optional[FrameItem]]" = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._thread: Optional[threading.Thread] = None
        self.n_received = 0
        self.n_dropped = 0
        self.n_unreadable = 0

    def _run(self) -> None:
        while True:
            item = self._src.get()
            if item is None:
                break
            frame_idx, frame, _t_emit = item
            t_capture = now_ns()
            self.n_received += 1
            entry: FrameItem = (frame_idx, frame, t_capture, t_capture)
            while True:
                try:
                    self.queue.put_nowait(entry)
                    break
                except queue.Full:
                    try:
                        dropped = self.queue.get_nowait()
                        if dropped is not None:
                            self.n_dropped += 1
                            self.log.write(
                                {"type": "drop", "frame_idx": dropped[0], "t": now_ns()}
                            )
                    except queue.Empty:
                        pass
        self.queue.put(None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="direct-ingest", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        pass

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

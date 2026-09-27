"""Run one cell of the experiment matrix (one deployment x model x scenario x rep).

Sequence:
  1. resolve cell, seed RNGs (seed = seed_base + repetition)
  2. open raw JSONL log, write provenance header
  3. [remote only] spawn inference-server process + userspace netproxy
  4. start RTSP replay (mediamtx + ffmpeg) and ingest, or direct fallback
  5. worker loop: frame -> inference -> alarm; one JSONL record per frame
  6. teardown; the raw log is the sole output (metrics are post-hoc)

Everything runs on ONE physical node; all timestamps are time.monotonic_ns()
in a single clock domain (see README).
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .alarm import AlarmLogic
from .config import Cell, ExperimentConfig
from .inference import RemoteBackendClient, _require_checkpoint, build_backend
from .ingest import DirectIngest, RTSPIngest
from .logio import RawLogWriter, now_ns, provenance_header
from .netproxy import NetProxy
from .replay import DirectReplay, MediaMTX, RTSPReplay


def _port_offset(cell: Cell) -> int:
    """Deterministic per-cell port offset so concurrent runs on one node
    (e.g. array sub-jobs co-scheduled by PBS) never collide.

    Modulus 2000 keeps offsets unique for matrices up to 2000 cells (the old
    %100 collided for cells i and i+100 on shared nodes). Offsets are
    multiples of 5 and the three base ports (8554, 8760, 8761) lie in
    distinct residues mod 5, so the rtsp/server/proxy ranges can never
    cross-collide either; the highest port stays < 20000 (unprivileged).
    """
    return (cell.index % 2000) * 5


def check_model_files(cfg: ExperimentConfig, model: str, backend: str) -> None:
    """Raise a clear error if the checkpoint a cell needs is missing."""
    spec = cfg.model_spec(model)
    if model == "synthetic" or backend == "synthetic":
        return
    if backend == "ultralytics":
        _require_checkpoint(cfg.root / spec["weights"], model, "PyTorch (.pt)")
    elif backend == "onnxruntime":
        _require_checkpoint(
            cfg.root / spec.get("onnx", spec["weights"]), model, "ONNX"
        )


def preflight(cfg: ExperimentConfig) -> list[str]:
    """Validate the whole configuration without running anything.

    Checks every active model's checkpoints (per backend actually used) and
    every active scenario's video file. Returns a list of problem strings;
    empty means the matrix is ready to submit.
    """
    problems: list[str] = []

    backends = {d["backend"] for d in cfg.raw["deployments"]}
    for model in cfg.active_models:
        for backend in sorted(backends):
            try:
                check_model_files(cfg, model, backend)
            except FileNotFoundError as e:
                path = str(e).splitlines()[1].strip()
                problems.append(
                    f"model '{model}' ({backend}): missing checkpoint {path}"
                )
            except KeyError as e:
                problems.append(f"model '{model}': missing config key {e}")

    for sid in cfg.active_scenarios:
        spec = cfg.scenario_spec(sid)
        video = cfg.root / spec["file"]
        if not video.exists():
            problems.append(f"scenario {sid}: video not found: {video}")
        onset = spec.get("t_onset_frame")
        if onset is None:
            problems.append(f"scenario {sid}: t_onset_frame is null")

    return problems


def run_cell(
    cfg: ExperimentConfig,
    cell: Cell,
    out_dir: str | Path,
    replay_mode: Optional[str] = None,
    infer_delay_ms: float = 0.0,
) -> Path:
    """Execute one run; returns the raw-log path."""
    random.seed(cell.seed)
    np.random.seed(cell.seed % (2**32))

    # Prefer the bundled static ffmpeg (third_party/ffmpeg) if present: some HPC
    # ffmpeg modules lack libx264, which the replay's H.264 encode needs. The
    # static build (scripts/fetch_ffmpeg.sh) ships it. Prepending here makes
    # both manual runs and array jobs pick it up without exporting PATH.
    tp = cfg.root / "third_party"
    if (tp / "ffmpeg").exists():
        os.environ["PATH"] = str(tp) + os.pathsep + os.environ.get("PATH", "")

    scenario = cfg.scenario_spec(cell.scenario)
    video_path = cfg.root / scenario["file"]
    if not video_path.exists():
        raise FileNotFoundError(
            f"scenario video {video_path} missing — run scripts/make_synthetic.py "
            "for the synthetic scenarios"
        )
    fps = float(scenario["fps"])
    t_onset_frame = int(scenario["t_onset_frame"])

    replay_mode = replay_mode or cfg.raw["replay"].get("mode", "rtsp")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{cell.run_id()}.jsonl"
    if log_path.exists():
        log_path.unlink()  # a rerun replaces the previous raw log

    log = RawLogWriter(log_path)
    log.write(provenance_header(cell.to_dict(), cfg.config_hash(), cfg.root))

    alarm_cfg = cfg.alarm
    alarm = AlarmLogic(
        weapon_class_names=alarm_cfg["weapon_class_names"],
        confidence_threshold=float(alarm_cfg["confidence_threshold"]),
        k_consecutive=int(alarm_cfg["k_consecutive"]),
    )

    # ------------------------------------------------------------- inference
    model_spec = cfg.model_spec(cell.model)
    server_proc: Optional[subprocess.Popen] = None
    proxy: Optional[NetProxy] = None
    backend: Any

    if cell.remote:
        off = _port_offset(cell)
        server_port = int(cfg.raw["remote"]["server_port"]) + off
        proxy_port = int(cfg.raw["remote"]["proxy_port"]) + off
        # validate the checkpoint here: a failure inside the server subprocess
        # would only surface as an opaque connection error on the client side
        check_model_files(cfg, cell.model, cell.backend)
        server_cmd = [
            sys.executable, "-m", "ttdbench.inference",
            "--serve", "--port", str(server_port),
            "--backend", cell.backend,
            "--model-name", cell.model,
            "--device", cell.device,
            "--repo-root", str(cfg.root),
        ]
        if model_spec.get("weights"):
            server_cmd += ["--weights", model_spec["weights"]]
        if model_spec.get("onnx"):
            server_cmd += ["--onnx", model_spec["onnx"]]
        if cell.cpu_threads:
            server_cmd += ["--cpu-threads", str(cell.cpu_threads)]
        env = os.environ.copy()  # src layout: make ttdbench importable
        env["PYTHONPATH"] = str(cfg.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        server_proc = subprocess.Popen(server_cmd, cwd=cfg.root, env=env)

        profile = cfg.network_profiles[cell.network_profile]
        proxy = NetProxy.from_profile(proxy_port, server_port, profile, seed=cell.seed)
        proxy.start()
        backend = RemoteBackendClient(
            "127.0.0.1", proxy_port,
            jpeg_quality=int(cfg.raw["remote"].get("jpeg_quality", 90)),
            connect_timeout_s=float(
                cfg.raw["remote"].get("server_startup_timeout_s", 300)
            ),
        )
    else:
        backend = build_backend(
            cell.backend, model_spec, device=cell.device,
            cpu_threads=cell.cpu_threads, repo_root=cfg.root,
        )
        if hasattr(backend, "infer_delay_ms"):
            backend.infer_delay_ms = infer_delay_ms

    # ------------------------------------------------------- replay + ingest
    mtx: Optional[MediaMTX] = None
    if replay_mode == "rtsp":
        rcfg = cfg.raw["replay"]
        rtsp_port = int(rcfg["rtsp_port"]) + _port_offset(cell)
        mtx = MediaMTX(
            cfg.root / rcfg["mediamtx_binary"], rtsp_port, out_dir / ".mediamtx"
        )
        mtx.start()
        # build the URL from mtx.port AFTER start(): on a busy shared node the
        # deterministic port may be taken and mediamtx retries on a free one
        rtsp_url = f"rtsp://127.0.0.1:{mtx.port}/{rcfg['rtsp_path']}"
        replay = RTSPReplay(
            video_path, fps, t_onset_frame, rtsp_url, log,
            encoder=rcfg.get("encoder", "libx264"),
            preset=rcfg.get("encoder_preset", "ultrafast"),
            tune=rcfg.get("encoder_tune", "zerolatency"),
        )
        ingest = RTSPIngest(
            rtsp_url, log, queue_size=int(cfg.raw["ingest"].get("queue_size", 1))
        )
        # ingest first: it retries opening until the publisher registers, so it
        # joins the stream during the replay lead-in, before frame 0 is emitted
        ingest.start()
        replay.start()
    elif replay_mode == "direct":
        replay = DirectReplay(video_path, fps, t_onset_frame, log)
        ingest = DirectIngest(
            replay.frames_queue, log,
            queue_size=int(cfg.raw["ingest"].get("queue_size", 1)),
        )
        replay.start()
        ingest.start()
    else:
        raise ValueError(f"unknown replay mode {replay_mode!r}")

    # ------------------------------------------------------------ worker loop
    hard_deadline = time.monotonic() + float(scenario.get("duration_s", 60)) + 90.0
    end_reason = "stream_end"
    try:
        while True:
            if time.monotonic() > hard_deadline:
                end_reason = "hard_deadline"
                break
            try:
                item = ingest.queue.get(timeout=2.0)
            except Exception:
                if replay.finished.is_set():
                    end_reason = "stream_end"
                    break
                continue
            if item is None:
                break
            frame_idx, frame, t_capture, t_decode = item
            result = backend.infer(frame)
            record: dict[str, Any] = {
                "type": "frame",
                "frame_idx": frame_idx,
                "t_capture": t_capture,
                "t_decode": t_decode,
                "t_pre": result.get("t_pre"),
                "t_send": result.get("t_send"),
                "t_recv_server": result.get("t_recv_server"),
                "t_resp_send": result.get("t_resp_send"),
                "t_resp_recv": result.get("t_resp_recv"),
                "t_infer_start": result["t_infer_start"],
                "t_infer_end": result["t_infer_end"],
                "t_post": result["t_post"],
                "detections": result["detections"],
            }
            log.write(record)
            t_alarm = alarm.update(frame_idx, result["detections"])
            if t_alarm is not None:
                log.write(
                    {"type": "alarm", "frame_idx": frame_idx, "t": t_alarm,
                     "k": alarm.k}
                )
                # keep consuming to log the full confidence trace until stream end
    finally:
        log.write({"type": "end", "t": now_ns(), "reason": end_reason,
                   "n_received": getattr(ingest, "n_received", None),
                   "n_dropped": getattr(ingest, "n_dropped", None),
                   "n_unreadable": getattr(ingest, "n_unreadable", None)})
        try:
            ingest.stop()
            replay.join(timeout=15)
            ingest.join(timeout=15)
        except Exception as e:
            print(f"[runner] teardown warning: {e}", file=sys.stderr)
        if isinstance(backend, RemoteBackendClient):
            backend.shutdown_server()
        if proxy:
            proxy.stop()
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if mtx:
            mtx.stop()
        log.close()

    return log_path

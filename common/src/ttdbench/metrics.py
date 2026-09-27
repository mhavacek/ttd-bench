"""Post-hoc computation of TTD and its decomposition from raw JSONL logs.

NOTHING here is measured online — every quantity below is derived from the
immutable raw log, so any analysis change is a re-run of this module, never a
re-run of the experiment.

Definitions (ns timestamps from a single monotonic clock domain):

  t_onset  = t_emit(onset frame)                     [replay 'onset_ref' event]
  t_alarm  = decision timestamp after the K-th consecutive qualifying frame
  TTD      = t_alarm - t_onset

Decomposition — measured on the ALARM-TRIGGERING frame f_K (the K-th frame of
the qualifying streak, whose processing fired the alarm):

  dt_acq      = t_capture(f_K) - t_emit(f_K)     encode + RTSP transport + decode
  dt_transfer = (t_recv_server - t_send) + (t_resp_recv - t_resp_send)   [remote]
                0                                                        [local]
  dt_infer    = t_infer_end(f_K) - t_infer_start(f_K)
  dt_post     = t_post(f_K) - t_infer_end(f_K)
  dt_alarm    = TTD - (dt_acq + dt_transfer + dt_infer + dt_post)   [residual]

The residual dt_alarm is the cost of the ALARM POLICY itself: it is dominated
by the K-frame confirmation window (t_emit(f_K) - t_onset >= (K-1)/fps plus
any backpressure-drop waiting) plus the decision overhead after t_post. Both
subterms are reported separately (dt_confirm_window, dt_decision) so the
residual is fully accounted for. The identity TTD == sum(5 components) holds
exactly by construction and is unit-tested.

A run is a MISS if no alarm fired within timeout_s after t_onset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .logio import read_log

NS_PER_MS = 1e6


def parse_run(
    log_path: str | Path,
    timeout_s: float = 30.0,
    alarm_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract per-run (per-event) metrics from one raw log.

    Two event-level definitions are computed from the SAME log:

    OPERATIONAL (single-shot; the online alarm as designed in §2): the alarm
      fires once, at the first K-consecutive qualification anywhere in the
      stream. A pre-onset (false) firing therefore makes the event a miss —
      this measures operational usability (fields: hit/miss/ttd_ms/dt_*).

    MEASUREMENT (post-hoc; requires `alarm_cfg` with weapon_class_names,
      confidence_threshold, k_consecutive): the first satisfaction of the
      SAME alarm condition whose decision time is >= t_onset, re-derived from
      the per-frame detections. Pre-onset false alarms do not consume it, so
      TTD is defined for every scenario and paired model comparisons keep the
      full scenario set (fields: meas_hit/ttd_meas_ms/dt_*_meas_ms). The
      decision timestamp of a post-hoc alarm is t_post of the K-th frame
      (the online decision overhead, dt_decision, is ~1 ms and is reported
      by the operational definition where available).
    """
    header: Optional[dict[str, Any]] = None
    emits: dict[int, int] = {}
    frames: dict[int, dict[str, Any]] = {}
    onset: Optional[dict[str, Any]] = None
    alarm: Optional[dict[str, Any]] = None
    end: Optional[dict[str, Any]] = None
    n_drops = 0

    for rec in read_log(log_path):
        t = rec.get("type")
        if t == "header":
            header = rec
        elif t == "emit":
            emits[rec["frame_idx"]] = rec["t"]
        elif t == "onset_ref":
            onset = rec
        elif t == "frame":
            frames[rec["frame_idx"]] = rec
        elif t == "drop":
            n_drops += 1
        elif t == "alarm":
            if alarm is None:  # first alarm only
                alarm = rec
        elif t == "end":
            end = rec

    if header is None:
        raise ValueError(f"{log_path}: missing header record")
    # onset_ref may be absent for a Phase-1 latency-only run (no verified onset)
    # or if the replay ended before the annotated onset frame; event-level TTD
    # is then undefined (miss) but per-frame latency metrics are still computed.
    # An alarm without an onset reference is a genuine inconsistency, though.
    if onset is None and alarm is not None:
        raise ValueError(f"{log_path}: alarm present but onset_ref missing")

    cell = header["cell"]
    t_onset = onset["t"] if onset is not None else None

    out: dict[str, Any] = {
        "run_id": Path(log_path).stem,
        "deployment": cell["deployment"],
        "network_profile": cell["network_profile"] or "none",
        "model": cell["model"],
        "scenario": cell["scenario"],
        "repetition": cell["repetition"],
        "seed": cell["seed"],
        "hostname": header.get("hostname"),
        "git_hash": header.get("git_hash"),
        "config_hash": header.get("config_hash"),
        "n_frames_emitted": len(emits),
        "n_frames_processed": len(frames),
        "n_drops": n_drops,
        "drop_rate": n_drops / max(1, len(frames) + n_drops),
    }

    hit = (
        alarm is not None
        and t_onset is not None
        and (alarm["t"] - t_onset) / 1e9 <= timeout_s
        and alarm["t"] >= t_onset
    )
    out["hit"] = bool(hit)
    out["miss"] = not hit
    # Operationally distinct miss cause: the K-consecutive alarm fired BEFORE
    # the weapon entered the field of view — a (sustained) false alarm. With
    # such detectors the event is not a detection failure but an FP failure;
    # report it separately so miss rate can be decomposed post-hoc.
    out["false_alarm_pre_onset"] = bool(
        alarm is not None and t_onset is not None and alarm["t"] < t_onset
    )

    if not hit:
        out.update(
            ttd_ms=None, dt_acq_ms=None, dt_transfer_ms=None, dt_infer_ms=None,
            dt_post_ms=None, dt_alarm_ms=None, dt_confirm_window_ms=None,
            dt_decision_ms=None,
        )
    else:
        assert alarm is not None
        f_k = alarm["frame_idx"]
        fr = frames.get(f_k)
        if fr is None:
            raise ValueError(f"{log_path}: alarm frame {f_k} has no frame record")
        t_emit_fk = emits.get(f_k)
        if t_emit_fk is None:
            raise ValueError(f"{log_path}: alarm frame {f_k} has no emit record")

        ttd_ns = alarm["t"] - t_onset
        dt_acq = fr["t_capture"] - t_emit_fk
        if fr.get("t_send") is not None and fr.get("t_recv_server") is not None:
            dt_transfer = (fr["t_recv_server"] - fr["t_send"]) + (
                fr["t_resp_recv"] - fr["t_resp_send"]
            )
        else:
            dt_transfer = 0
        dt_infer = fr["t_infer_end"] - fr["t_infer_start"]
        dt_post = fr["t_post"] - fr["t_infer_end"]
        dt_alarm = ttd_ns - (dt_acq + dt_transfer + dt_infer + dt_post)

        out.update(
            ttd_ms=ttd_ns / NS_PER_MS,
            dt_acq_ms=dt_acq / NS_PER_MS,
            dt_transfer_ms=dt_transfer / NS_PER_MS,
            dt_infer_ms=dt_infer / NS_PER_MS,
            dt_post_ms=dt_post / NS_PER_MS,
            dt_alarm_ms=dt_alarm / NS_PER_MS,
            # residual sub-terms (transparency; dt_alarm ~= confirm + decision)
            dt_confirm_window_ms=(t_emit_fk - t_onset) / NS_PER_MS,
            dt_decision_ms=(alarm["t"] - fr["t_post"]) / NS_PER_MS,
        )

    # ---- MEASUREMENT definition: first alarm-condition satisfaction with
    # decision time >= t_onset, re-derived post-hoc from the detections ------
    out["meas_hit"] = False
    out.update(
        ttd_meas_ms=None, dt_acq_meas_ms=None, dt_transfer_meas_ms=None,
        dt_infer_meas_ms=None, dt_post_meas_ms=None, dt_alarm_meas_ms=None,
    )
    if alarm_cfg is not None and t_onset is not None and frames:
        wclasses = {c.lower() for c in alarm_cfg["weapon_class_names"]}
        thr = float(alarm_cfg["confidence_threshold"])
        k = int(alarm_cfg["k_consecutive"])
        streak = 0
        meas_fk: Optional[int] = None
        for i in sorted(frames):
            f = frames[i]
            qualifies = any(
                d["cls"].lower() in wclasses and d["conf"] >= thr
                for d in f["detections"]
            )
            streak = streak + 1 if qualifies else 0
            # the same automaton as online, but pre-onset firings do not
            # consume it: take the first K-streak whose decision (t_post)
            # falls at or after the onset
            if streak >= k and f["t_post"] >= t_onset:
                meas_fk = i
                break
        if meas_fk is not None:
            fr = frames[meas_fk]
            t_emit_fk = emits.get(meas_fk)
            ttd_ns = fr["t_post"] - t_onset
            if (ttd_ns / 1e9) <= timeout_s and t_emit_fk is not None:
                dt_acq = fr["t_capture"] - t_emit_fk
                if fr.get("t_send") is not None and fr.get("t_recv_server") is not None:
                    dt_transfer = (fr["t_recv_server"] - fr["t_send"]) + (
                        fr["t_resp_recv"] - fr["t_resp_send"]
                    )
                else:
                    dt_transfer = 0
                dt_infer = fr["t_infer_end"] - fr["t_infer_start"]
                dt_post = fr["t_post"] - fr["t_infer_end"]
                out["meas_hit"] = True
                out.update(
                    ttd_meas_ms=ttd_ns / NS_PER_MS,
                    dt_acq_meas_ms=dt_acq / NS_PER_MS,
                    dt_transfer_meas_ms=dt_transfer / NS_PER_MS,
                    dt_infer_meas_ms=dt_infer / NS_PER_MS,
                    dt_post_meas_ms=dt_post / NS_PER_MS,
                    dt_alarm_meas_ms=(
                        ttd_ns - (dt_acq + dt_transfer + dt_infer + dt_post)
                    ) / NS_PER_MS,
                )

    # ---- per-frame latency decomposition (alarm-independent) --------------
    # Averaged over EVERY processed frame, so it is defined even when no alarm
    # fires (e.g. a COCO model that lacks the weapon class — Phase 1). This is
    # the pipeline latency profile that differs by deployment configuration;
    # it does not depend on t_onset or the alarm policy and therefore neither
    # changes nor relies on the event-level TTD definition (§2).
    acq, tr, inf, post = [], [], [], []
    for i, f in frames.items():
        t_emit_i = emits.get(i)
        if t_emit_i is not None:
            acq.append((f["t_capture"] - t_emit_i) / NS_PER_MS)
        if f.get("t_send") is not None and f.get("t_recv_server") is not None:
            tr.append(
                ((f["t_recv_server"] - f["t_send"]) + (f["t_resp_recv"] - f["t_resp_send"]))
                / NS_PER_MS
            )
        inf.append((f["t_infer_end"] - f["t_infer_start"]) / NS_PER_MS)
        post.append((f["t_post"] - f["t_infer_end"]) / NS_PER_MS)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def _median(xs):
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    out["pf_dt_acq_ms"] = _mean(acq)
    out["pf_dt_transfer_ms"] = _mean(tr) if tr else 0.0
    out["pf_dt_infer_ms"] = _mean(inf)
    out["pf_dt_post_ms"] = _mean(post)
    out["pf_dt_infer_median_ms"] = _median(inf)
    out["pf_latency_ms"] = (  # mean total per-frame processing latency
        (out["pf_dt_acq_ms"] or 0) + (out["pf_dt_transfer_ms"] or 0)
        + (out["pf_dt_infer_ms"] or 0) + (out["pf_dt_post_ms"] or 0)
    ) if inf else None
    out["mean_frame_infer_ms"] = out["pf_dt_infer_ms"]  # kept for compatibility
    out["end_reason"] = end.get("reason") if end else "unknown"
    return out


def confidence_trace(log_path: str | Path, weapon_classes: Optional[set[str]] = None) -> pd.DataFrame:
    """Per-frame max weapon confidence relative to onset (for trace plots)."""
    onset_t = None
    rows = []
    frames = []
    for rec in read_log(log_path):
        if rec.get("type") == "onset_ref":
            onset_t = rec["t"]
        elif rec.get("type") == "frame":
            frames.append(rec)
    for fr in frames:
        confs = [
            d["conf"] for d in fr["detections"]
            if weapon_classes is None or d["cls"].lower() in weapon_classes
        ]
        rows.append(
            {
                "frame_idx": fr["frame_idx"],
                "t_rel_onset_ms": (fr["t_post"] - onset_t) / NS_PER_MS if onset_t else None,
                "max_conf": max(confs) if confs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def runs_to_dataframe(
    log_paths: list[str | Path], timeout_s: float = 30.0, strict: bool = False,
    alarm_cfg: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Parse many raw logs into the master event-level DataFrame.

    Non-strict mode skips unparsable logs (e.g. a run still in progress or
    aborted before onset) with a warning — aggregation stays re-runnable
    while an experiment batch is underway.
    """
    rows = []
    for p in log_paths:
        try:
            rows.append(parse_run(p, timeout_s=timeout_s, alarm_cfg=alarm_cfg))
        except (ValueError, KeyError) as e:
            if strict:
                raise
            import sys

            print(f"WARNING: skipping {p}: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["config_label"] = df.apply(
            lambda r: r["deployment"]
            if r["network_profile"] in ("none", None)
            else f"{r['deployment']}-{r['network_profile']}",
            axis=1,
        )
    return df

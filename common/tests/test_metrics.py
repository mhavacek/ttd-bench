"""TTD computation on a synthetic raw log with hand-computed expected values."""

import json

import pytest

from ttdbench.metrics import parse_run, runs_to_dataframe

MS = 1_000_000  # ns per ms


def write_log(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def header(deployment="local-gpu", profile=None):
    return {
        "type": "header", "schema": 1,
        "cell": {
            "index": 0, "deployment": deployment, "backend": "synthetic",
            "device": "cpu", "cpu_threads": None, "remote": profile is not None,
            "network_profile": profile, "model": "synthetic",
            "scenario": "s1", "repetition": 0, "seed": 42,
        },
        "config_hash": "x", "git_hash": "y", "hostname": "test",
    }


def frame_rec(idx, t_capture, dt_infer=10 * MS, dt_post=2 * MS, conf=0.9,
              transfer=None):
    """Frame record; processing chain starts at t_capture."""
    rec = {
        "type": "frame", "frame_idx": idx,
        "t_capture": t_capture,
        "t_decode": t_capture + MS,
        "t_pre": t_capture + 2 * MS,
        "t_send": None, "t_recv_server": None,
        "t_resp_send": None, "t_resp_recv": None,
        "t_infer_start": t_capture + 3 * MS,
        "t_infer_end": t_capture + 3 * MS + dt_infer,
        "t_post": t_capture + 3 * MS + dt_infer + dt_post,
        "detections": [{"cls": "weapon", "conf": conf, "xyxy": [0, 0, 5, 5]}],
    }
    if transfer is not None:
        up, down = transfer
        rec["t_send"] = t_capture + 2 * MS
        rec["t_recv_server"] = rec["t_send"] + up
        rec["t_resp_send"] = rec["t_infer_end"] + MS
        rec["t_resp_recv"] = rec["t_resp_send"] + down
    return rec


def build_known_log(tmp_path, k=3, fps=25):
    """Onset at frame 10 (t=1000 ms); frames 10,11,12 detect; alarm on 12.

    Hand-computed expectation:
      t_onset          = 1000 ms (emit of frame 10)
      frame 12 emit    = 1080 ms  (40 ms period)
      t_capture(12)    = emit + 50 ms acq = 1130 ms
      dt_infer         = 10 ms, dt_post = 2 ms
      t_post(12)       = 1130 + 3 + 10 + 2 = 1145 ms
      t_alarm          = t_post(12) + 1 ms = 1146 ms
      TTD              = 146 ms
      dt_acq           = 50 ms; dt_transfer = 0
      dt_alarm (resid) = 146 - 50 - 0 - 10 - 2 = 84 ms
      dt_confirm_window= 80 ms; dt_decision = 1 ms
    """
    period = MS * 1000 // fps  # 40 ms
    records = [header()]
    t0 = 1000 * MS - 10 * period  # so that emit(10) == 1000 ms exactly
    alarm_t = None
    for i in range(20):
        t_emit = t0 + i * period
        records.append({"type": "emit", "frame_idx": i, "t": t_emit})
        if i == 10:
            records.append({"type": "onset_ref", "frame_idx": i, "t": t_emit})
    for i in range(20):
        t_emit = t0 + i * period
        conf = 0.9 if i >= 10 else 0.0
        rec = frame_rec(i, t_emit + 50 * MS, conf=conf)
        records.append(rec)
        if i == 12:
            alarm_t = rec["t_post"] + MS
            records.append({"type": "alarm", "frame_idx": 12, "t": alarm_t, "k": k})
    records.append({"type": "end", "t": alarm_t + 100 * MS, "reason": "stream_end"})
    p = tmp_path / "run.jsonl"
    write_log(p, records)
    return p


def test_ttd_known_values(tmp_path):
    p = build_known_log(tmp_path)
    m = parse_run(p, timeout_s=30.0)
    assert m["hit"] is True
    assert m["ttd_ms"] == pytest.approx(146.0)
    assert m["dt_acq_ms"] == pytest.approx(50.0)
    assert m["dt_transfer_ms"] == pytest.approx(0.0)
    assert m["dt_infer_ms"] == pytest.approx(10.0)
    assert m["dt_post_ms"] == pytest.approx(2.0)
    assert m["dt_alarm_ms"] == pytest.approx(84.0)
    assert m["dt_confirm_window_ms"] == pytest.approx(80.0)
    assert m["dt_decision_ms"] == pytest.approx(1.0)


def test_decomposition_identity(tmp_path):
    """TTD == sum of the five components, exactly by construction."""
    p = build_known_log(tmp_path)
    m = parse_run(p)
    total = (m["dt_acq_ms"] + m["dt_transfer_ms"] + m["dt_infer_ms"]
             + m["dt_post_ms"] + m["dt_alarm_ms"])
    assert total == pytest.approx(m["ttd_ms"], abs=1e-9)


def test_transfer_component_remote(tmp_path):
    records = [header(deployment="remote", profile="wifi")]
    t_emit = 1000 * MS
    records.append({"type": "emit", "frame_idx": 0, "t": t_emit})
    records.append({"type": "onset_ref", "frame_idx": 0, "t": t_emit})
    rec = frame_rec(0, t_emit + 20 * MS, transfer=(15 * MS, 10 * MS))
    records.append(rec)
    records.append({"type": "alarm", "frame_idx": 0, "t": rec["t_post"] + MS, "k": 1})
    p = tmp_path / "remote.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["hit"]
    assert m["dt_transfer_ms"] == pytest.approx(25.0)


def test_miss_on_timeout(tmp_path):
    records = [header()]
    t_emit = 1000 * MS
    records.append({"type": "emit", "frame_idx": 0, "t": t_emit})
    records.append({"type": "onset_ref", "frame_idx": 0, "t": t_emit})
    rec = frame_rec(0, t_emit + 5 * MS)
    records.append(rec)
    # alarm arrives 31 s after onset -> miss under 30 s timeout
    records.append({"type": "alarm", "frame_idx": 0, "t": t_emit + 31_000 * MS, "k": 3})
    p = tmp_path / "late.jsonl"
    write_log(p, records)
    m = parse_run(p, timeout_s=30.0)
    assert m["miss"] is True
    assert m["ttd_ms"] is None


def test_miss_without_alarm(tmp_path):
    records = [header()]
    records.append({"type": "emit", "frame_idx": 0, "t": 0})
    records.append({"type": "onset_ref", "frame_idx": 0, "t": 0})
    records.append(frame_rec(0, 5 * MS, conf=0.1))
    p = tmp_path / "noalarm.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["miss"] is True


def test_drop_rate(tmp_path):
    records = [header()]
    records.append({"type": "emit", "frame_idx": 0, "t": 0})
    records.append({"type": "onset_ref", "frame_idx": 0, "t": 0})
    records.append(frame_rec(0, 5 * MS))
    records.append({"type": "drop", "frame_idx": 1, "t": 10 * MS})
    records.append({"type": "drop", "frame_idx": 2, "t": 20 * MS})
    records.append(frame_rec(3, 30 * MS))
    p = tmp_path / "drops.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["n_drops"] == 2
    assert m["drop_rate"] == pytest.approx(0.5)


def test_dataframe_config_label(tmp_path):
    p = build_known_log(tmp_path)
    df = runs_to_dataframe([p])
    assert df.iloc[0]["config_label"] == "local-gpu"
    assert len(df) == 1


# -------------------------- per-frame decomposition (Phase 1) ----------------

def test_perframe_decomposition_values(tmp_path):
    """pf_* are averaged over ALL frames, independent of the alarm."""
    p = build_known_log(tmp_path)  # frames have dt_infer=10, dt_post=2, acq=50
    m = parse_run(p)
    assert m["pf_dt_acq_ms"] == pytest.approx(50.0)
    assert m["pf_dt_infer_ms"] == pytest.approx(10.0)
    assert m["pf_dt_post_ms"] == pytest.approx(2.0)
    assert m["pf_dt_transfer_ms"] == pytest.approx(0.0)
    assert m["pf_latency_ms"] == pytest.approx(62.0)


def test_phase1_no_alarm_still_has_perframe(tmp_path):
    """COCO-style run: frames present, no weapon detection, no alarm.

    Event-level TTD is a miss, but per-frame latency is fully computed.
    """
    records = [header()]
    t0 = 1000 * MS
    for i in range(5):
        records.append({"type": "emit", "frame_idx": i, "t": t0 + i * 40 * MS})
    records.append({"type": "onset_ref", "frame_idx": 2, "t": t0 + 2 * 40 * MS})
    for i in range(5):
        records.append(frame_rec(i, t0 + i * 40 * MS + 30 * MS, conf=0.0))  # no det
    records.append({"type": "end", "t": t0 + 10 ** 9, "reason": "stream_end"})
    p = tmp_path / "coco.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["miss"] is True
    assert m["ttd_ms"] is None
    assert m["pf_dt_infer_ms"] == pytest.approx(10.0)
    assert m["n_frames_processed"] == 5


def test_missing_onset_ref_is_miss_not_error(tmp_path):
    """Replay ended before the onset frame: parse as miss, keep per-frame."""
    records = [header()]
    for i in range(3):
        records.append({"type": "emit", "frame_idx": i, "t": i * 40 * MS})
        records.append(frame_rec(i, i * 40 * MS + 30 * MS, conf=0.0))
    records.append({"type": "end", "t": 10 ** 9, "reason": "stream_end"})
    p = tmp_path / "no_onset.jsonl"
    write_log(p, records)
    m = parse_run(p)  # must not raise
    assert m["miss"] is True
    assert m["pf_dt_infer_ms"] == pytest.approx(10.0)


def test_alarm_without_onset_raises(tmp_path):
    """An alarm with no onset reference is a genuine inconsistency."""
    records = [header()]
    records.append({"type": "emit", "frame_idx": 0, "t": 0})
    records.append(frame_rec(0, 5 * MS))
    records.append({"type": "alarm", "frame_idx": 0, "t": 100 * MS, "k": 3})
    p = tmp_path / "bad.jsonl"
    write_log(p, records)
    with pytest.raises(ValueError):
        parse_run(p)


def test_perframe_transfer_remote(tmp_path):
    """Per-frame transfer averages the round-trip components for remote runs."""
    records = [header(deployment="remote", profile="wifi")]
    for i in range(3):
        records.append({"type": "emit", "frame_idx": i, "t": i * 40 * MS})
        records.append(frame_rec(i, i * 40 * MS + 20 * MS,
                                 conf=0.0, transfer=(15 * MS, 10 * MS)))
    records.append({"type": "end", "t": 10 ** 9, "reason": "stream_end"})
    p = tmp_path / "remote_pf.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["pf_dt_transfer_ms"] == pytest.approx(25.0)


def test_pre_onset_false_alarm_flag(tmp_path):
    """Alarm before onset -> miss with false_alarm_pre_onset=True."""
    records = [header()]
    t0 = 1000 * MS
    records.append({"type": "emit", "frame_idx": 0, "t": t0})
    rec = frame_rec(0, t0 + 5 * MS)
    records.append(rec)
    records.append({"type": "alarm", "frame_idx": 0, "t": rec["t_post"] + MS, "k": 3})
    records.append({"type": "emit", "frame_idx": 10, "t": t0 + 400 * MS})
    records.append({"type": "onset_ref", "frame_idx": 10, "t": t0 + 400 * MS})
    records.append({"type": "end", "t": t0 + 10 ** 9, "reason": "stream_end"})
    p = tmp_path / "fp.jsonl"
    write_log(p, records)
    m = parse_run(p)
    assert m["miss"] is True
    assert m["false_alarm_pre_onset"] is True


def test_normal_hit_not_flagged_as_false_alarm(tmp_path):
    p = build_known_log(tmp_path)
    m = parse_run(p)
    assert m["hit"] is True
    assert m["false_alarm_pre_onset"] is False


# ------------------- measurement vs operational alarm definition -------------

ALARM_CFG = {"weapon_class_names": ["weapon"], "confidence_threshold": 0.5,
             "k_consecutive": 3}


def test_measurement_ttd_survives_pre_onset_false_alarm(tmp_path):
    """Pre-onset FP fires the single-shot alarm (operational miss), but the
    measurement definition still yields TTD from the post-onset streak."""
    period = 40 * MS
    records = [header()]
    t0 = 1000 * MS
    onset_i = 10
    for i in range(20):
        records.append({"type": "emit", "frame_idx": i, "t": t0 + i * period})
        if i == onset_i:
            records.append({"type": "onset_ref", "frame_idx": i, "t": t0 + i * period})
    alarm_logged = False
    for i in range(20):
        # frames 0-4 qualify (persistent FP on a car), 5-9 not, 10+ qualify (weapon)
        conf = 0.9 if (i < 5 or i >= onset_i) else 0.0
        rec = frame_rec(i, t0 + i * period + 50 * MS, conf=conf)
        records.append(rec)
        if i == 2 and not alarm_logged:  # online single-shot fired on the FP
            records.append({"type": "alarm", "frame_idx": 2, "t": rec["t_post"] + MS, "k": 3})
            alarm_logged = True
    records.append({"type": "end", "t": t0 + 10 ** 9, "reason": "stream_end"})
    p = tmp_path / "fp_then_weapon.jsonl"
    write_log(p, records)

    m = parse_run(p, alarm_cfg=ALARM_CFG)
    # operational: miss due to pre-onset false alarm
    assert m["hit"] is False
    assert m["false_alarm_pre_onset"] is True
    # measurement: first K-streak decision >= onset is frame 12
    #   ttd_meas = t_post(12) - t_onset = (2*40 + 50 + 3 + 10 + 2) = 145 ms
    assert m["meas_hit"] is True
    assert m["ttd_meas_ms"] == pytest.approx(145.0)
    assert m["dt_acq_meas_ms"] == pytest.approx(50.0)
    total = (m["dt_acq_meas_ms"] + m["dt_transfer_meas_ms"] + m["dt_infer_meas_ms"]
             + m["dt_post_meas_ms"] + m["dt_alarm_meas_ms"])
    assert total == pytest.approx(m["ttd_meas_ms"], abs=1e-9)


def test_measurement_matches_operational_on_clean_run(tmp_path):
    """No FP: the two definitions pick the same frame; they differ only by the
    online decision overhead (alarm.t - t_post = 1 ms in the fixture)."""
    p = build_known_log(tmp_path)
    m = parse_run(p, alarm_cfg=ALARM_CFG)
    assert m["hit"] is True and m["meas_hit"] is True
    assert m["ttd_ms"] - m["ttd_meas_ms"] == pytest.approx(m["dt_decision_ms"])


def test_measurement_requires_alarm_cfg(tmp_path):
    p = build_known_log(tmp_path)
    m = parse_run(p)  # no alarm_cfg -> measurement fields inert
    assert m["meas_hit"] is False and m["ttd_meas_ms"] is None


def _sparse_run(tmp_path, period_ms, qualifying, onset_i=5, n=20):
    """Log with one processed frame every `period_ms`; `qualifying` = set of
    frame indices whose detection clears the confidence threshold."""
    period = period_ms * MS
    t0 = 1000 * MS
    records = [header()]
    for i in range(n):
        records.append({"type": "emit", "frame_idx": i, "t": t0 + i * period})
        if i == onset_i:
            records.append({"type": "onset_ref", "frame_idx": i, "t": t0 + i * period})
    for i in range(n):
        records.append(frame_rec(i, t0 + i * period + 50 * MS,
                                 conf=0.9 if i in qualifying else 0.0))
    records.append({"type": "end", "t": t0 + n * period, "reason": "stream_end"})
    p = tmp_path / f"sparse_{period_ms}.jsonl"
    write_log(p, records)
    return p


def test_windowed_alarm_fires_when_detections_are_not_consecutive(tmp_path):
    """K=3 non-consecutive detections inside the window: the consecutive rule
    misses, the windowed rule fires (the ablation's whole point)."""
    p = _sparse_run(tmp_path, period_ms=40, qualifying={6, 8, 10})
    cfg = dict(ALARM_CFG, window_ms=1000.0)
    m = parse_run(p, alarm_cfg=cfg)
    assert m["meas_hit"] is False          # never 3 in a row
    assert m["win_hit"] is True
    # fires on frame 10: t_post = onset + 5*40 + 50 + 3 + 10 + 2 = 265 ms
    assert m["ttd_win_ms"] == pytest.approx(265.0)
    assert m["win_ms"] == 1000.0


def test_windowed_alarm_penalises_a_slow_pipeline(tmp_path):
    """Same 3 consecutive detections, but the pipeline runs at ~2 fps: the
    consecutive rule fires, the 1 s window cannot be met. This is why the
    windowed rule is not uniformly more permissive (edge-sim result)."""
    p = _sparse_run(tmp_path, period_ms=600, qualifying={6, 7, 8})
    cfg = dict(ALARM_CFG, window_ms=1000.0)
    m = parse_run(p, alarm_cfg=cfg)
    assert m["meas_hit"] is True
    assert m["win_hit"] is False           # 3 frames span 1200 ms > 1000 ms window
    wide = parse_run(p, alarm_cfg=dict(ALARM_CFG, window_ms=3000.0))
    assert wide["win_hit"] is True


def test_evidence_policy_split_and_followup(tmp_path):
    """t_first_det + t_accum == ttd_meas, and follow-up is the observation
    window actually available after the onset (clip end, not the timeout)."""
    p = _sparse_run(tmp_path, period_ms=40, qualifying={7, 8, 9, 10})
    m = parse_run(p, alarm_cfg=ALARM_CFG)
    assert m["meas_hit"] is True
    # first qualifying frame after onset is 7: t_post = 2*40 + 65 = 145 ms
    assert m["t_first_det_ms"] == pytest.approx(145.0)
    assert m["t_first_det_ms"] + m["t_accum_ms"] == pytest.approx(m["ttd_meas_ms"])
    # last processed frame is 19: (19-5)*40 + 65 = 625 ms after onset
    assert m["followup_ms"] == pytest.approx(625.0)
    # frames 4..19: frame 4 is CAPTURED before the onset but DECIDED after it
    # (t_post = onset + 25 ms), and a decision after the onset is what counts
    # as an observation opportunity.
    assert m["n_frames_post_onset"] == 16
    assert m["n_qualifying_post_onset"] == 4   # frames 7,8,9,10

#!/usr/bin/env python3
"""Dwell-time alarm rule: K = ceil(t * f) per configuration (pure re-scoring).

Motivation
----------
The published rule requires K = 3 consecutive qualifying frames. A deployment
that processes 1.8 frames per second needs about 1.7 s of video to satisfy it,
one that processes 30 fps needs 0.1 s, so the same K means a different
persistence requirement in each configuration. Here the requirement is a
dwell of t seconds and K follows from the achieved processed frame rate f of
each configuration:

    K(config) = max(1, ceil(t * f_config))

with f_config = CONFIG_FPS from far_sweep.py (the median processed rate of the
frozen Phase-2 logs, the same figure that sets the subsampling stride).

Calibration follows the two variants of far_sweep.py:
  (a) the detector threshold is calibrated ONCE at full frame rate with
      K(full rate); the false-alarm rate of each configuration (stride and
      K of that configuration) is a derived quantity. This is the primary
      analysis: the deployment penalty stays in the measurement.
  (b) one threshold per detector and configuration, matched to the same FAR.
      Used for comparing detectors WITHIN a configuration.

Modes: "fixed3" reproduces the published K = 3 rule and is the validation of
this script against thresholds.csv / phase2_far_summary.csv; "t<x>" are requested
dwell times in seconds (T_GRID). t = 0.1 s equals K = 3 at 30 fps, so it
leaves the full-rate results unchanged.

Everything is re-scored from the frozen traces; no new measurement.
Outputs in clanek-2-ttdbench/results-far-dwell/.
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from far_sweep import (AT416, CONFIG_FPS, FULL, STRIDE, TARGETS,  # noqa: E402
                       THR_GRID, V8S, episodes, garwood)

ROOT = Path(__file__).resolve().parent.parent.parent
CAL = ROOT / "clanek-2-ttdbench/results-far-calib"
FROZEN = ROOT / "results/scvd-phase2/analysis"
# D11: with --v8s416 yolov8s is scored at 416 px for the configurations that ran
# it at 416 and validated against far_sweep/far_reanalysis --v8s416 outputs.
V416 = "--v8s416" in sys.argv
SFX = "_v8s416" if V416 else ""
OUT = ROOT / "clanek-2-ttdbench/results-far-dwell" / ("v8s416" if V416 else "")
MODEL_ORDER = ["yolov8s-weapon", "yolov8m-weapon", "yolov12m-weapon",
               "yolov26m-weapon"]
DEFECT = [("yolov26m-weapon", "edge-sim")]
CONFIGS = list(STRIDE)
# Requested dwell times (s). 0.568 s = 1 / f_min, f_min = 1.76 fps of edge-sim:
# below it K cannot fall under one, so edge-sim enforces K / f > t and the
# comparison between deployments is not interpretable (not enforceable).
T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.568, 0.7, 1.0, 1.5]
F_MIN = min(CONFIG_FPS.values())
MODES = {"fixed3": None, **{f"t{t:g}": t for t in T_GRID}}


def k_of(mode: str, cfg: str) -> int:
    """Frames required in configuration cfg under the given rule."""
    t = MODES[mode]
    if t is None:
        return 3
    # rounding guards against 0.1 * 30 = 3.0000000000000004 -> ceil = 4
    return max(1, math.ceil(round(t * CONFIG_FPS[cfg], 6)))


def realised_dwell_s(mode: str, cfg: str) -> float:
    """Dwell actually enforced: K / f (equals t only when t * f is an integer)."""
    return k_of(mode, cfg) / CONFIG_FPS[cfg]


def pairs() -> list[tuple[int, int]]:
    return sorted({(STRIDE[c], k_of(m, c)) for m in MODES for c in CONFIGS})


def corpus():
    man = pd.read_csv(CAL / "corpus_manifest.csv")
    man = man.sort_values(["source", "clip"]).drop_duplicates("md5", keep="first")
    return man.set_index("clip")


def count_model(model: str) -> tuple[str, dict]:
    """Episode counts, shape (n_thr, n_clips), per (stride, K) for one model."""
    man = corpus()
    clips = list(man.index)
    # read only this model's rows: four workers each holding the full
    # 40 M-row table (plus the 416 px yolov8s traces) pushed the machine into swap
    tr = pd.read_parquet(CAL / "calib_traces.parquet",
                         columns=["clip", "frame_idx", "max_conf"],
                         filters=[("model", "==", model)])
    tr = tr[tr["clip"].isin(set(clips))]
    seqs = {c: g.sort_values("frame_idx")["max_conf"].to_numpy()
            for c, g in tr.groupby("clip")}
    seqs416 = {}
    if V416 and model == V8S:
        t4 = pd.read_parquet(CAL / "calib_traces_v8s416.parquet",
                             columns=["clip", "frame_idx", "max_conf"])
        seqs416 = {c: g.sort_values("frame_idx")["max_conf"].to_numpy()
                   for c, g in t4[t4["clip"].isin(set(clips))].groupby("clip")}
    strides416 = {STRIDE[c] for c in AT416}
    assert STRIDE["edge-sim"] not in strides416
    out = {}
    for stride, k in pairs():
        cnt = np.zeros((len(THR_GRID), len(clips)), dtype=np.int32)
        src = seqs416 if (seqs416 and stride in strides416) else seqs
        for j, c in enumerate(clips):
            sub = src[c][::stride]
            for i, thr in enumerate(THR_GRID):
                cnt[i, j] = episodes(sub, thr, k)
        out[(stride, k)] = cnt
    return model, out


def streak_fire(q: np.ndarray, rid: np.ndarray, k: np.ndarray) -> np.ndarray:
    """True where a run of >= k[i] consecutive qualifying rows ends at row i.

    Vectorised equivalent of the frame loop in far_reanalysis.evaluate; rows
    must be sorted by (run_id, time).
    """
    n = len(q)
    pos = np.arange(n)
    start = np.r_[True, rid[1:] != rid[:-1]]
    start_pos = np.maximum.accumulate(np.where(start, pos, 0))
    last_false = np.maximum.accumulate(np.where(~q, pos, -1))
    last_false = np.maximum(last_false, start_pos - 1)
    streak = np.where(q, pos - last_false, 0)
    return streak >= k


def load_events():
    fr = pd.read_parquet(FROZEN / "traces.parquet")
    ru = pd.read_parquet(FROZEN / "traces_runs.parquet")
    ru["config_label"] = np.where(ru["network_profile"] == "none",
                                  ru["deployment"],
                                  ru["deployment"] + "-" + ru["network_profile"])
    for m, c in DEFECT:
        ru = ru[~((ru["model"] == m) & (ru["config_label"] == c))]
    fr = fr.sort_values(["run_id", "t_rel_onset_ms"], kind="stable")
    fr = fr.merge(ru[["run_id", "model", "config_label"]], on="run_id")
    return fr, ru


def evaluate(fr, ru, thr_of, k_of_cfg):
    cfg = fr["config_label"].to_numpy()
    mod = fr["model"].to_numpy()
    thr = np.array([thr_of(m, c) for m, c in zip(mod, cfg)])
    k = np.array([k_of_cfg(c) for c in cfg])
    rid = fr["run_id"].to_numpy()
    trel = fr["t_rel_onset_ms"].to_numpy()
    fire = streak_fire(fr["max_conf"].to_numpy() >= thr, rid, k)
    post = fire & (trel >= 0)
    ttd = pd.Series(trel[post], index=rid[post])
    ttd = ttd[~ttd.index.duplicated(keep="first")]
    res = ru.copy()
    res["ttd_ms"] = res["run_id"].map(ttd)
    res["hit"] = res["ttd_ms"].notna()
    res["duration_ms"] = np.where(res["hit"], res["ttd_ms"], res["followup_ms"])
    return res[res["duration_ms"] > 0]


def km_stats(g: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import qth_survival_time
    if len(g) == 0:
        return {"n": 0}
    kmf = KaplanMeierFitter().fit(g["duration_ms"], g["hit"])
    sf = kmf.survival_function_
    med = kmf.median_survival_time_
    q25 = qth_survival_time(0.75, sf)
    return {"n": len(g), "hit_pct": round(g["hit"].mean() * 100, 1),
            "P_alarm_1s": float(1 - kmf.predict(1000)),
            "P_alarm_3s": float(1 - kmf.predict(3000)),
            "km_median_ms": float(med) if np.isfinite(med) else np.inf,
            "km_q25_ms": float(q25) if np.isfinite(q25) else np.inf}


def far_row(counts, i, dur_h, boot_idx):
    cnt = counts[i]
    n = int(cnt.sum())
    t_h = dur_h.sum()
    lo, hi = garwood(n, t_h)
    rates = cnt[boot_idx].sum(axis=1) / dur_h[boot_idx].sum(axis=1)
    blo, bhi = np.percentile(rates, [2.5, 97.5])
    return {"n_alarms": n, "achieved_far_h": n / t_h,
            "gar_lo": lo, "gar_hi": hi, "boot_lo": blo, "boot_hi": bhi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--v8s416", action="store_true",
                    help="D11: yolov8s at 416 px where it was deployed at 416")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse counts.npz from a previous run")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    man = corpus()
    dur_h = man["duration_s"].to_numpy() / 3600
    t_total_h = dur_h.sum()
    print(f"corpus {len(man)} clips, {t_total_h:.4f} h; (stride, K) pairs: {pairs()}")

    npz = OUT / "counts.npz"
    if args.reuse and npz.exists():
        z = np.load(npz)
        counts = {}
        for key in z.files:
            m, s, k = key.rsplit("|", 2)
            counts.setdefault(m, {})[(int(s), int(k))] = z[key]
    else:
        with Pool(args.workers) as p:
            counts = dict(p.map(count_model, MODEL_ORDER))
        np.savez_compressed(npz, **{f"{m}|{s}|{k}": v for m, d in counts.items()
                                    for (s, k), v in d.items()})
    rng = np.random.default_rng(42)
    boot_idx = rng.integers(0, len(man), size=(1000, len(man)))

    # ---- thresholds and derived FARs, both variants, every mode ----
    thr_rows = []
    for mode in MODES:
        kf = {c: k_of(mode, c) for c in CONFIGS}
        for model in MODEL_ORDER:
            cs = counts[model]
            full = cs[(STRIDE[FULL], kf[FULL])]
            far_full = full.sum(axis=1) / t_total_h
            for tgt in TARGETS:
                ok = np.flatnonzero(far_full <= tgt)
                if len(ok):                                   # variant a
                    i = ok[0]
                    for cfg in CONFIGS:
                        c = cs[(STRIDE[cfg], kf[cfg])]
                        thr_rows.append({"mode": mode, "variant": "a",
                                         "model": model, "target_far_h": tgt,
                                         "config": cfg, "K": kf[cfg],
                                         "threshold": THR_GRID[i],
                                         **far_row(c, i, dur_h, boot_idx)})
                for cfg in CONFIGS:                           # variant b
                    c = cs[(STRIDE[cfg], kf[cfg])]
                    far = c.sum(axis=1) / t_total_h
                    ok = np.flatnonzero(far <= tgt)
                    if len(ok):
                        i = ok[0]
                        thr_rows.append({"mode": mode, "variant": "b",
                                         "model": model, "target_far_h": tgt,
                                         "config": cfg, "K": kf[cfg],
                                         "threshold": THR_GRID[i],
                                         **far_row(c, i, dur_h, boot_idx)})
    th = pd.DataFrame(thr_rows)
    th.to_csv(OUT / "thresholds_dwell.csv", index=False)

    # ---- validation: fixed3 must reproduce the published calibration ----
    pub = pd.read_csv(CAL / f"thresholds{SFX}.csv")
    mine = th[th["mode"] == "fixed3"].drop(columns=["mode", "K"])
    key = ["variant", "model", "target_far_h", "config"]
    mg = pub.merge(mine, on=key, suffixes=("_pub", "_new"))
    bad = mg[(mg["threshold_pub"] != mg["threshold_new"]) |
             (mg["n_alarms_pub"] != mg["n_alarms_new"])]
    print(f"validation fixed3 vs thresholds.csv: {len(mg)} rows compared, "
          f"{len(bad)} differ")
    if len(bad):
        print(bad.head(10).to_string())
        return 1

    # ---- event re-scoring ----
    fr, ru = load_events()
    summaries = []
    for mode in MODES:
        kf = {c: k_of(mode, c) for c in CONFIGS}
        kfun = lambda c, kf=kf: kf[c]
        for (variant, tgt), sub in th[th["mode"] == mode].groupby(
                ["variant", "target_far_h"]):
            if variant == "a":
                tmap = sub.drop_duplicates("model").set_index("model")["threshold"]
                if len(tmap) < 4:
                    continue
                thr_of = lambda m, c, t=tmap: t[m]
            else:
                tmap = sub.set_index(["model", "config"])["threshold"]
                if len(tmap) < 19:      # 4 models x 5 configs, minus the defect
                    continue
                thr_of = lambda m, c, t=tmap: t[(m, c)]
            res = evaluate(fr, ru, thr_of, kfun)
            far = sub.set_index(["model", "config"])
            for (m, c), g in res.groupby(["model", "config_label"]):
                if (m, c) not in far.index:
                    continue
                summaries.append({"mode": mode, "variant": variant,
                                  "target_far_h": tgt, "model": m, "config": c,
                                  "K": kf[c],
                                  "achieved_far_h": far.loc[(m, c), "achieved_far_h"],
                                  "n_alarms_calib": far.loc[(m, c), "n_alarms"],
                                  **km_stats(g)})
    sm = pd.DataFrame(summaries)
    sm.to_csv(OUT / "phase2_far_dwell_summary.csv", index=False)

    # ---- validation of the event side against the published summary ----
    pubs = pd.read_csv(CAL / f"phase2_far_summary{SFX}.csv")
    pubs = pubs[(pubs["variant"] == "a") & (pubs["config"] != "ALL")].copy()
    pubs["target_far_h"] = pubs["operating_point"].str.replace("far", "").astype(float)
    mv = pubs.merge(sm[sm["mode"] == "fixed3"].query("variant == 'a'"),
                    on=["model", "config", "target_far_h"], suffixes=("_pub", "_new"))
    d = (mv["hit_pct_pub"] - mv["hit_pct_new"]).abs()
    print(f"validation fixed3 vs phase2_far_summary.csv (variant a): "
          f"{len(mv)} cells, max |hit% diff| = {d.max():.2f}")
    if d.max() > 0.05:
        print(mv.loc[d > 0.05, ["model", "config", "target_far_h",
                                "hit_pct_pub", "hit_pct_new"]].head(10).to_string())
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

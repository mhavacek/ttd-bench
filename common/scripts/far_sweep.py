#!/usr/bin/env python3
"""FAR calibration: thresholds matched to false-alarms-per-hour targets.

Unit and rule
-------------
FAR = alarm EPISODES per camera-hour of alarm-free footage. An episode starts
when K = 3 consecutive processed frames reach the threshold and re-arms on the
first frame below it (one continuous burst = one alarm, matching how an alert
reaches an operator). Streaks never cross clip boundaries.

Deployment configurations process different frame rates (edge ~2 fps vs
local 30 fps), so the SAME threshold yields a DIFFERENT FAR per configuration.
Both calibration variants are produced:
  (a) one threshold per model, calibrated at full frame rate; the FAR of that
      threshold under each configuration's frame rate is reported as a derived
      quantity;
  (b) one threshold per model AND configuration, calibrated so each
      configuration hits the same FAR target.
Frame rates are emulated by uniform subsampling of the 30 fps corpus at the
median processed rate of the Phase-2 logs (stride in {1, 2, 4, 17}). Real
frame loss is bursty rather than uniform; this is noted as a limitation.

Uncertainty: Garwood exact Poisson CI on the episode count, plus a
clip-level bootstrap (alarms cluster within clips, so the Poisson CI is
anticonservative; the bootstrap is the honest one).

Outputs in results-far-calib/:
  far_curves.parquet      FAR(threshold) per model x config stride
  thresholds.csv          calibrated thresholds, variants (a) and (b)
  achievability.txt       what the corpus can and cannot resolve, per target
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "clanek-2-ttdbench/results-far-calib"

K = 3
# median processed fps per Phase-2 configuration (from traces.parquet)
CONFIG_FPS = {"local-gpu": 30.0, "remote-lan": 30.0, "remote-wifi": 15.0,
              "remote-cellular-4g": 7.5, "edge-sim": 1.76}
STRIDE = {"local-gpu": 1, "remote-lan": 1, "remote-wifi": 2,
          "remote-cellular-4g": 4, "edge-sim": 17}
FULL = "local-gpu"                        # stride 1 = full frame rate
# D11: yolov8s ran at its checkpoint's 416 px wherever ultralytics served it
# (local-gpu and the three remote configurations); edge-sim's ONNX export has a
# fixed 640 px input. With --v8s416 its calibration follows that split.
V8S = "yolov8s-weapon"
AT416 = {"local-gpu", "remote-lan", "remote-wifi", "remote-cellular-4g"}
# Target grid. The low end became reachable only when the corpus grew from
# 0.2728 h to 94.03 h: 0.1/h needs ~30 h for a zero-count bound alone, and 1/h
# needs ~16 h for a +-50 % estimate. The high end is kept so the published
# fixed-0.5 and D6 operating points remain on the same axis for comparison.
TARGETS = [0.1, 0.3, 1.0, 3.0, 12.0, 30.0, 60.0, 120.0, 300.0, 600.0]  # alarms/h
# Ceiling raised from 0.95 to 0.995 once the low targets became reachable:
# at 0.95 three of four models could not hit 0.1-1/h, which would have been a
# statement about the grid rather than about the detectors. Confidences run to
# 1.0, so thresholds above 0.95 are representable and legitimate.
THR_GRID = np.round(np.arange(0.10, 0.9951, 0.005), 3)


def episodes_ref(conf: np.ndarray, thr: float, k: int = K) -> int:
    """Reference implementation of the alarm rule, frame by frame.

    Kept because it is the readable statement of the rule and the oracle the
    vectorised version is tested against; the sweep itself calls episodes().
    """
    n_alarms, streak, armed = 0, 0, True
    for c in conf:
        if c >= thr:
            streak += 1
            if streak >= k and armed:
                n_alarms += 1
                armed = False
        else:
            streak = 0
            armed = True
    return n_alarms


def episodes(conf: np.ndarray, thr: float, k: int = K) -> int:
    """Alarm episodes in one clip's (subsampled) frame sequence.

    The rule fires once per maximal run of >= k consecutive qualifying frames
    (it disarms on qualification and re-arms on the first frame below the
    threshold), so the count is exactly the number of such runs. Finding them
    from the run boundaries is equivalent to the frame-by-frame reference above
    and roughly two orders of magnitude faster, which is what makes sweeping
    180 thresholds over 1196 clips a minutes-long job rather than an hours-long
    one. Equivalence is covered by a test against episodes_ref.
    """
    m = (conf >= thr).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(np.count_nonzero(ends - starts >= k))


def garwood(n: int, t_h: float, alpha: float = 0.05) -> tuple[float, float]:
    lo = chi2.ppf(alpha / 2, 2 * n) / 2 / t_h if n > 0 else 0.0
    hi = chi2.ppf(1 - alpha / 2, 2 * n + 2) / 2 / t_h
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep the 20 clips UCF-Crime ships twice and write the "
                         "outputs with a _withdup suffix; this is the "
                         "robustness run, not the primary one")
    ap.add_argument("--v8s416", action="store_true",
                    help="D11: score yolov8s at 416 px (calib_traces_v8s416.parquet) "
                         "for the configurations that ran it at 416 (every stride "
                         "except edge-sim's); outputs get a _v8s416 suffix")
    args = ap.parse_args()
    sfx = "_withdup" if args.keep_duplicates else ""
    if args.v8s416:
        sfx += "_v8s416"

    tr = pd.read_parquet(OUT / "calib_traces.parquet")
    tr416 = None
    if args.v8s416:
        tr416 = pd.read_parquet(OUT / "calib_traces_v8s416.parquet")
        assert set(tr416["model"]) == {V8S}, set(tr416["model"])
    man = pd.read_csv(OUT / "corpus_manifest.csv")
    if not args.keep_duplicates:
        # A clip present twice contributes its false alarms twice while adding
        # no independent evidence: the two copies are byte-identical files whose
        # traces come back identical and whose alarm decisions never disagree
        # below 120/h (far_dedup_check.py). Its duration and its episodes
        # therefore leave together. Order by (source, clip) so the kept copy is
        # deterministic and matches the duplicate list in the corpus report.
        before = len(man)
        man = man.sort_values(["source", "clip"]).drop_duplicates("md5",
                                                                 keep="first")
        print(f"dedup: {before} -> {len(man)} clips "
              f"({before - len(man)} duplicates dropped)")
        tr = tr[tr["clip"].isin(set(man["clip"]))]
        if tr416 is not None:
            tr416 = tr416[tr416["clip"].isin(set(man["clip"]))]
    man = man.set_index("clip")
    seq416 = {} if tr416 is None else {
        c: g.sort_values("frame_idx")["max_conf"].to_numpy()
        for c, g in tr416.groupby("clip")}
    t_total_h = man["duration_s"].sum() / 3600

    # per model x stride x clip: conf sequences once, then sweep thresholds
    curves = []
    per_clip = {}          # (model, cfg, thr) -> np.array of per-clip counts
    clips = list(man.index)
    dur_h = man["duration_s"].to_numpy() / 3600
    for (model, clip), g in tr.groupby(["model", "clip"], sort=True):
        conf = g.sort_values("frame_idx")["max_conf"].to_numpy()
        for cfg, s in STRIDE.items():
            src = conf
            if model == V8S and seq416 and cfg in AT416:
                src = seq416[clip]
            sub = src[::s]
            for thr in THR_GRID:
                key = (model, cfg, thr)
                if key not in per_clip:
                    per_clip[key] = {}
                n = episodes(sub, thr)
                if n:
                    per_clip[key][clip] = n

    rng = np.random.default_rng(42)
    boot_idx = rng.integers(0, len(clips), size=(1000, len(clips)))
    for (model, cfg, thr), d in per_clip.items():
        cnt = np.array([d.get(c, 0) for c in clips])
        n = int(cnt.sum())
        far = n / t_total_h
        glo, ghi = garwood(n, t_total_h)
        rates = cnt[boot_idx].sum(axis=1) / dur_h[boot_idx].sum(axis=1) * 1  # /h
        blo, bhi = np.percentile(rates, [2.5, 97.5])
        curves.append({"model": model, "config": cfg, "stride": STRIDE[cfg],
                       "threshold": thr, "n_alarms": n, "far_h": far,
                       "gar_lo": glo, "gar_hi": ghi,
                       "boot_lo": blo, "boot_hi": bhi})
    cv = pd.DataFrame(curves)
    cv.to_parquet(OUT / f"far_curves{sfx}.parquet", index=False)

    def calibrate(sub: pd.DataFrame, target: float) -> pd.Series | None:
        """Smallest threshold whose estimated FAR <= target."""
        ok = sub[sub["far_h"] <= target]
        if ok.empty:
            return None
        return ok.sort_values("threshold").iloc[0]

    rows = []
    for model in sorted(tr["model"].unique()):
        # variant (a): calibrate once at full frame rate
        full = cv[(cv.model == model) & (cv.config == FULL)]
        for tgt in TARGETS:
            r = calibrate(full, tgt)
            if r is None:
                continue
            for cfg in STRIDE:
                d = cv[(cv.model == model) & (cv.config == cfg) &
                       (cv.threshold == r["threshold"])].iloc[0]
                rows.append({"variant": "a", "model": model, "target_far_h": tgt,
                             "config": cfg, "threshold": r["threshold"],
                             "achieved_far_h": d["far_h"],
                             "n_alarms": d["n_alarms"],
                             "boot_lo": d["boot_lo"], "boot_hi": d["boot_hi"]})
        # variant (b): calibrate per configuration
        for cfg in STRIDE:
            sub = cv[(cv.model == model) & (cv.config == cfg)]
            for tgt in TARGETS:
                r = calibrate(sub, tgt)
                if r is None:
                    continue
                rows.append({"variant": "b", "model": model, "target_far_h": tgt,
                             "config": cfg, "threshold": r["threshold"],
                             "achieved_far_h": r["far_h"],
                             "n_alarms": r["n_alarms"],
                             "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"]})
    th = pd.DataFrame(rows)
    th.to_csv(OUT / f"thresholds{sfx}.csv", index=False)

    # --- achievability of the corpus ---
    zero_ub = -np.log(0.05) / t_total_h
    lines = [
        f"corpus: {t_total_h:.4f} h alarm-free footage ({len(clips)} clips)",
        f"zero-alarm bound: a threshold with 0 observed alarms only supports "
        f"'FAR < {zero_ub:.3f}/h' (95% one-sided); nothing below that is "
        f"resolvable",
        f"+-50% precision (N>=16): needs FAR >= {16 / t_total_h:.3f}/h",
        "",
        "reachability of each target on THIS corpus "
        "(zero-count bound needs 3/target hours, +-50 % estimate ~16/target):",
        *[f"  {t:>6.1f}/h  bound {'yes' if t >= zero_ub else 'NO '}"
          f"  estimate {'yes' if t >= 16 / t_total_h else 'NO '}"
          f"   (needs {3 / t:.2f} h / {16 / t:.1f} h)"
          for t in TARGETS],
    ]
    (OUT / f"achievability{sfx}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    pd.set_option("display.width", 250)
    print("\n=== FAR(thr) at full frame rate (selected thresholds) ===")
    sel = cv[(cv.config == FULL) &
             (cv.threshold.isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))]
    print(sel.pivot_table(index="threshold", columns="model",
                          values="far_h").round(1).to_string())
    print("\n=== calibrated thresholds (variant a, full rate) ===")
    print(th[(th.variant == "a") & (th.config == FULL)]
          .drop(columns=["variant", "config"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

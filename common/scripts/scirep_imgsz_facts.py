#!/usr/bin/env python3
"""Indicative local impact of the YOLOv8s input-size mismatch (deviation D11).

In Phase 2 the replay harness called `predict()` without `imgsz`, so Ultralytics
took the input size stored in each checkpoint: YOLOv8s ran at 416 pixels in
local-gpu and the three remote configurations, and at 640 in edge-sim (the ONNX
export has a fixed 640 input). The other three detectors ran at 640 throughout,
and the FAR calibration scored all four at 640. Full account:
clanek-2-ttdbench/docs/IMGSZ-AUDIT-2026-09-27.md.

Handled as a limitation (option c, approved 2026-09-27): nothing frozen is
re-scored and no published number changes. This script only recomputes, from
the audit's stored local intermediates, the indicative numbers the Sci Rep SI
quotes, and writes them to results/scirep-stats/imgsz_impact.txt, which
scirep_build.py audits the manuscript against. Every quantity is LOCAL and
INDICATIVE: YOLOv8s only, scored on an Apple MPS device, on 34 event clips and
a 0.483 h random subset of the 93.14 h calibration corpus.

Inputs (clanek-2-ttdbench/docs/imgsz-audit-2026-09-27/, written by the audit
scripts in the same folder):
  event_scores.parquet         YOLOv8s max weapon confidence at 416 and 640 on
                               every frame of the 34 event clips (score_events.py)
  replay_matched.parquet       frozen Phase-2 YOLOv8s replay frames matched to
                               those offline scores (match_replay.py)
  calib_subset.csv,            32-clip random subset of the corpus and its 416
  calib_subset_scores.parquet  scores (calib416.py); 640 scores are the cluster's,
                               from results-far-calib/calib_traces.parquet
and the frozen results-scvd/analysis/traces*.parquet and
results-far-calib/thresholds.csv and far_curves.parquet (full-corpus rates).

  scirep_imgsz_facts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "clanek-2-ttdbench"
AUD = P2 / "docs/imgsz-audit-2026-09-27"
OUT = ROOT / "results/scirep-stats/imgsz_impact.txt"
V8S = "yolov8s-weapon"
K = 3


def episodes(conf: np.ndarray, thr: float, k: int = K) -> int:
    """Alarm episodes: K consecutive frames at/above thr, re-arming below it."""
    m = (conf >= thr).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return int(np.count_nonzero(np.flatnonzero(d == -1) - np.flatnonzero(d == 1) >= k))


def hit(g: pd.DataFrame, thr: float, k: int = K) -> bool:
    """Measurement definition: K consecutive qualifying frames, alarm at/after onset."""
    g = g.sort_values("t_rel_onset_ms")
    run = 0
    for q, post in zip(g.max_conf.values >= thr, g.t_rel_onset_ms.values >= 0):
        run = run + 1 if q else 0
        if run >= k and post:
            return True
    return False


def main() -> None:
    L = ["# Indicative, local: YOLOv8s only, Apple MPS, see the docstring of",
         "# common/scripts/scirep_imgsz_facts.py and docs/IMGSZ-AUDIT-2026-09-27.md", ""]

    # 1. replay traces against offline 416 / 640 scores
    m = pd.read_parquet(AUD / "replay_matched.parquet")
    L.append("[replay traces vs offline scores; frame offset -2]")
    L.append("deployment\tn_frames\tr_416\tr_640\tagree05_416\tagree05_640")
    for dep in ("local-gpu", "remote", "edge-sim"):
        g = m[m.deployment == dep]
        r4 = np.corrcoef(g.max_conf, g["416"])[0, 1]
        r6 = np.corrcoef(g.max_conf, g["640"])[0, 1]
        a4 = ((g.max_conf >= .5) == (g["416"] >= .5)).mean()
        a6 = ((g.max_conf >= .5) == (g["640"] >= .5)).mean()
        L.append(f"{dep}\t{len(g)}\t{r4:.3f}\t{r6:.3f}\t{a4:.3f}\t{a6:.3f}")

    # 2. event clips: per-frame decision disagreement 416 vs 640 at 0.5
    e = pd.read_parquet(AUD / "event_scores.parquet")
    p = e.pivot_table(index=["scenario", "frame_idx", "onset"], columns="imgsz",
                      values="max_conf").reset_index()
    dis = ((p[416] >= .5) != (p[640] >= .5)).mean()
    L += ["", "[event clips, offline]",
          f"n_clips\t{p.scenario.nunique()}", f"n_frames\t{len(p)}",
          f"disagreement_at_0.5_pct\t{100 * dis:.1f}",
          f"mean_abs_diff\t{(p[416] - p[640]).abs().mean():.3f}"]

    # 3. calibration subset: FAR at the published YOLOv8s thresholds, 416 vs 640
    sub = pd.read_csv(AUD / "calib_subset.csv")
    s = pd.read_parquet(AUD / "calib_subset_scores.parquet")
    s = s[s.imgsz == 416]
    ct = pd.read_parquet(P2 / "results-far-calib/calib_traces.parquet",
                         filters=[("model", "==", V8S), ("clip", "in", list(sub["clip"]))])
    c = s.merge(ct[["clip", "frame_idx", "max_conf"]], on=["clip", "frame_idx"],
                suffixes=("_416", "_640")).sort_values(["clip", "frame_idx"])
    hours = sub.duration_s.sum() / 3600
    th = pd.read_csv(P2 / "results-far-calib/thresholds.csv")
    th = th[(th.variant == "a") & (th.model == V8S) & (th.config == "local-gpu")]
    L += ["", "[calibration subset]", f"n_clips\t{len(sub)}", f"hours\t{hours:.3f}",
          f"n_frames\t{len(c)}",
          "threshold\ttarget\tfar_640\tfar_416\tratio_416_640\tachieved_full_corpus_640"]
    fc = pd.read_parquet(P2 / "results-far-calib/far_curves.parquet")
    fc = fc[(fc.model == V8S) & (fc.config == "local-gpu") & (fc.stride == 1)]
    ratios = []
    for thr, tg, ach in [(0.5, "fixed", fc.loc[(fc.threshold - 0.5).abs().idxmin(), "far_h"])] + [(r.threshold, f"{r.target_far_h:g}", r.achieved_far_h)
                                                    for r in th.itertuples() if r.target_far_h >= 12]:
        a = sum(episodes(g.max_conf_640.values, thr) for _, g in c.groupby("clip")) / hours
        b = sum(episodes(g.max_conf_416.values, thr) for _, g in c.groupby("clip")) / hours
        ratios.append(b / a)
        L.append(f"{thr:.3f}\t{tg}\t{a:.1f}\t{b:.1f}\t{b / a:.2f}\t{ach:.1f}")
    L.append(f"ratio_range\t{min(ratios):.2f}\t{max(ratios):.1f}")

    # 4. local-gpu YOLOv8s hit rate at 416-matched thresholds, 90 % clip bootstrap
    t = pd.read_parquet(P2 / "results-scvd/analysis/traces.parquet").merge(
        pd.read_parquet(P2 / "results-scvd/analysis/traces_runs.parquet"), on="run_id")
    runs = list(t[(t.model == V8S) & (t.deployment == "local-gpu")].groupby("run_id"))
    grid = np.round(np.arange(0.5, 0.9951, 0.005), 3)
    H = {thr: 100 * np.mean([hit(g, thr) for _, g in runs]) for thr in grid}
    clips = list(sub["clip"])
    dur = dict(zip(sub["clip"], sub.duration_s))
    E = {k: (np.array([episodes(g.max_conf_640.values, x) for x in grid]),
             np.array([episodes(g.max_conf_416.values, x) for x in grid]))
         for k, g in c.groupby("clip")}
    rng = np.random.default_rng(1)
    L += ["", "[local-gpu YOLOv8s hit rate, published vs 416-matched threshold;"
          " 300 clip-bootstrap replicates, 5th-95th percentile]",
          f"n_runs\t{len(runs)}",
          "target\tpub_thr\tpub_hit\tthr416_med\tthr416_lo\tthr416_hi\thit_med\thit_lo\thit_hi"]
    for tg in (120, 300, 600):
        tp = float(th[th.target_far_h == tg].threshold.iloc[0])
        i = int(np.argmin(abs(grid - tp)))
        res = []
        for _ in range(300):
            bs = rng.choice(clips, len(clips), replace=True)
            h = sum(dur[x] for x in bs) / 3600
            f640 = sum(E[x][0] for x in bs) / h
            f416 = sum(E[x][1] for x in bs) / h
            ok = np.flatnonzero(f416 <= f640[i])
            t2 = grid[ok.min()] if len(ok) else np.nan
            res.append((t2, H.get(t2, np.nan)))
        r = np.array(res)
        q = lambda a, x: np.nanpercentile(a, x)
        L.append(f"{tg}\t{tp:.3f}\t{H[grid[i]]:.1f}\t{np.nanmedian(r[:, 0]):.3f}\t{q(r[:, 0], 5):.3f}"
                 f"\t{q(r[:, 0], 95):.3f}\t{np.nanmedian(r[:, 1]):.1f}\t{q(r[:, 1], 5):.1f}"
                 f"\t{q(r[:, 1], 95):.1f}")
    OUT.write_text("\n".join(L) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()

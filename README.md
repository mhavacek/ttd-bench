# TTD-Bench

Replay-based hardware-in-the-loop benchmark that decomposes the end-to-end
**time-to-detection (TTD)** of CCTV weapon-detection pipelines into five
exactly-accounted components:

    TTD = dt_acq + dt_transfer + dt_infer + dt_post + dt_alarm

measured across deployment configurations (local GPU, emulated edge, remote
inference behind LAN / WiFi / 4G network emulation). The identity holds exactly
by construction and is unit-tested. All event-level quantities are derived
**post-hoc** from immutable raw JSONL logs, so an analysis change is a re-run
of the aggregation scripts, never a re-run of the experiment.

This repository accompanies the TTD-Bench paper (article reference will be
added upon publication).

## Code here, data on request, videos at the source

| what | where |
|---|---|
| benchmark package, analysis scripts, frozen configs, protocol docs | **this repository** |
| raw measurement logs (3 625 runs), derived CSVs, survival analysis, onset annotations | available from the corresponding author (martin.havacek@vsb.cz) on reasonable request |
| video data | original distributors — **UCF-Crime** (Sultani, Chen, Shah: *Real-World Anomaly Detection in Surveillance Videos*, CVPR 2018) and **SCVD** (Aremu et al., 2023); not redistributed |

## Layout

```
common/                     the `ttdbench` package + shared analysis scripts
  src/ttdbench/             replay, ingest, inference, alarm, metrics, stats …
  scripts/                  aggregate.py, survival_phase2.py, threshold_sweep.py,
                            consort_figure.py, annotate_onsets.py,
                            annotation_agreement.py, far_*.py (false-alarm
                            calibration, dwell-time analysis), onset_sensitivity.py,
                            alarm_ablation.py, scirep_*.py (paper tables/figures), …
  configs/                  scenario / network-profile / experiment YAMLs
  tests/                    unit tests (64)
clanek-2-ttdbench/          paper-2 experiment layer
  configs/                  frozen experiment configs (phase 1 final, SCVD phase 2)
    cluster-side/           path-rewritten variants that reproduce the logged
                            config_hash — see below
  scripts/                  run_single.py, checkpoint/asset fetchers,
                            make_paper_assets.py
  hpc/                      PBS/SLURM array scripts, venv setup, frozen env record
  docs/                     protocol & pre-registration documents (in Czech)
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "common/[dev]"      # analysis + tests; no ML backends
pip install -e "common/[ml]"       # optional: ultralytics + onnxruntime,
                                   # needed only to re-run measurements
pytest common/tests                # 64 tests
```

## Reproducing the published numbers

Obtain the measurement data from the corresponding author and unpack the archives into the repository root so that `results/…` sits next
to `common/`. Then:

```bash
# 1) raw JSONL -> per-run CSVs + master.csv (both phases)
python common/scripts/aggregate.py \
  --config clanek-2-ttdbench/configs/experiment-phase1-final.yaml \
  --raw-dir results/phase1-final/raw --out results/phase1-final/csv
python common/scripts/aggregate.py \
  --config clanek-2-ttdbench/configs/experiment-scvd.yaml \
  --raw-dir results/scvd-phase2/raw --out results/scvd-phase2/csv

# 2) phase-2 survival analysis (Kaplan-Meier, Cox PH, log-rank)
python common/scripts/survival_phase2.py \
  --master results/scvd-phase2/csv/master.csv --event meas_hit \
  --out-dir results/scvd-phase2/analysis

# 3) matched-FP model comparison (thresholds re-derived from conf>=0.1 logs)
python common/scripts/threshold_sweep.py --build \
  --raw-dir results/scvd-phase2/raw --out-dir results/scvd-phase2/analysis

# 4) CONSORT run-accounting figure
python common/scripts/consort_figure.py \
  --master results/scvd-phase2/csv/master.csv \
  --out results/scvd-phase2/analysis/consort.pdf

# 5) phase-1 per-frame summary tables + figures
python clanek-2-ttdbench/scripts/make_paper_assets.py --phase1
```

### False-alarm calibration, onset sensitivity and the dwell-time analysis

These steps need, in addition, the per-frame scores of the 93.14 h alarm-free
calibration corpus (`clanek-2-ttdbench/results-far-calib/calib_traces.parquet`
and `calib_traces_v8s416.parquet`, also available on request) and the frozen
Phase-2 frame traces (`results/scvd-phase2/analysis/traces*.parquet`).

```bash
# 6) corpus manifest + disjointness proof; scoring on the cluster (PBS array)
python common/scripts/far_corpus.py
qsub -J 0-63 -v CORPUS_DIR=...,STRIDE=64 clanek-2-ttdbench/hpc/pbs_far_scoring.sh
#    yolov8s at its deployed 416 px (deviation D11):
qsub -J 0-63 -v CORPUS_DIR=...,OUT_DIR=...,STRIDE=64,MODELS=yolov8s-weapon,IMGSZ=416 \
     clanek-2-ttdbench/hpc/pbs_far_scoring.sh
python common/scripts/far_traces_merge.py --traces <dir>
python common/scripts/far_traces_merge.py --traces <dir416> --models yolov8s-weapon \
       --out clanek-2-ttdbench/results-far-calib/calib_traces_v8s416.parquet

# 7) thresholds per false-alarm target and Phase-2 re-scoring (published = --v8s416)
python common/scripts/far_sweep.py --v8s416
python common/scripts/far_reanalysis.py --v8s416

# 8) alarm-rule ablation and the pre-registered onset-shift sensitivity analysis
python common/scripts/alarm_ablation.py --exclude-cell yolov26m-weapon:edge-sim
python common/scripts/onset_sensitivity.py

# 9) post hoc dwell-time rule K = ceil(t * f); the K = 3 mode is re-run as a
#    control and must reproduce step 7 exactly
python common/scripts/far_dwell.py --v8s416
python common/scripts/far_dwell_cox.py --v8s416

# 10) chance level of a hit (false alarms alone, Poisson) and proportional-
#     hazards checks: Grambsch-Therneau tests, scenario-stratified Cox, RMST(3 s)
python common/scripts/far_chance.py
python common/scripts/ph_check.py

# 11) tables and figures of the Scientific Reports version
python common/scripts/scirep_stats.py && python common/scripts/scirep_tables.py
python common/scripts/scirep_figures.py
python common/scripts/far_dwell_table.py && python common/scripts/far_dwell_hr_figure.py
```

**Deviation D11.** The replay harness left the inference input size to each
checkpoint, so yolov8s ran at 416 px on local-gpu and the remote
configurations and at 640 px on edge-sim (fixed ONNX input). The calibration
corpus was first scored at 640 px for every model; it was re-scored for
yolov8s at 416 px, and `--v8s416` uses those scores wherever yolov8s ran at
416 px. Running steps 7 and 9 without the flag reproduces the superseded
640 px calibration. Details: `clanek-2-ttdbench/docs/IMGSZ-AUDIT-2026-09-27.md`.

**Always filter `n_frames_processed > 0` before computing any rate** — see
Known limitations. The JSONL schema and every CSV column are documented in the
README that accompanies the data; formal definitions live in
`common/src/ttdbench/metrics.py`.

### Re-running the measurements themselves

Requires (a) model checkpoints — phase-1 COCO checkpoints via
`clanek-2-ttdbench/scripts/fetch_coco_checkpoints.py`; the phase-2 weapon
checkpoints are not part of this repository; the weights and training outputs
are on Zenodo ([10.5281/zenodo.19372252](https://doi.org/10.5281/zenodo.19372252)), (b) `mediamtx` and a static `ffmpeg` in
`clanek-2-ttdbench/third_party/` via `scripts/fetch_mediamtx.sh` and
`scripts/fetch_ffmpeg.sh`, (c) the videos (see above). HPC submission scripts
and the deployment protocol are in `clanek-2-ttdbench/hpc/` and
`clanek-2-ttdbench/docs/`.

Both frozen campaigns ran the runtime at tag `campaign-356842d` (branch
`campaign`), which holds `common/src/ttdbench/` exactly as it was at commit
356842d of the working repository. `main` adds later fixes used only by the
repair run: mediamtx bound to TCP only, logging of GPU sibling processes and
dispatch on the ONNX output layout.

## config_hash and the frozen configs

Every raw log records a `config_hash` — SHA-256 (first 16 hex chars) of the
canonical JSON of the loaded experiment config, scenarios, and network
profiles (`common/src/ttdbench/config.py`). Because scenario YAMLs embed the
**absolute dataset paths**, the hash depends on where the videos live:

| experiment config | repo (as committed) | logged (cluster-side) |
|---|---|---|
| experiment-phase1-final.yaml | `33da055d392b98d7` | `48c0b912a4f390ef` |
| experiment-scvd.yaml | `9de4fd9774457d25` | `20bdb7a3775a1c3d` |

`clanek-2-ttdbench/configs/cluster-side/` contains the scenario files exactly
as loaded on the cluster; hashing against them reproduces the logged values
(verified — see the README in that directory).

**Known limitation:** the frozen scenario YAMLs
(`scenarios-scvd.yaml`, `scenarios-ucf-phase1.yaml`, and the dev-only
`scenarios-ucf.yaml`) carry the original absolute dataset prefix
(`/Users/macbook/Datasets/ttd-bench`). They are hashed evidence and **must not
be edited** — rewriting the paths would silently change `config_hash` and
break verification against the logs. To run on your machine, apply the same
one-line substitution used for the frozen cluster deployment:

```bash
sed -i "s|/Users/macbook/Datasets/ttd-bench|/path/to/your/data|g" \
    common/configs/scenarios-*.yaml     # changes config_hash, by design
```

(The annotation *template* `scenarios-scvd-template.yaml` is not hashed by any
log and uses `--data-root`-relative paths, matching
`common/scripts/annotate_onsets.py`.)

## Known limitations

* **869 zero-frame runs** (775/3400 in phase 2, 94/225 in phase 1 final) are
  infrastructure failures, not detection misses: a port collision between
  concurrent jobs kills the run before the first frame is decoded. Diagnosis,
  evidence chain, and the measurement-neutral fix are documented in
  `clanek-2-ttdbench/docs/DIAGNOSTIKA-VYPADKU.md`; the CONSORT figure in the
  data deposit accounts for every run. Filter `n_frames_processed > 0`.
* `config_hash` in the logs is the **cluster-side** value — verify against
  `clanek-2-ttdbench/configs/cluster-side/`, not the committed mac-side YAMLs
  (table above).
* The `results*` symlinks in `clanek-2-ttdbench/` dangle until you create /
  download `results/` at the repository root (they are the `results_dir`
  targets of the frozen experiment configs).
* Protocol documents in `clanek-2-ttdbench/docs/` are written in Czech (they
  are frozen records of the pre-registered protocol and deviations).

## License and citation

Code is MIT-licensed (see `LICENSE`). Video
datasets remain under their original distributors' terms.

To cite, use the metadata in `CITATION.cff` (GitHub's "Cite this repository"
button).

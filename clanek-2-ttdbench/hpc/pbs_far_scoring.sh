#!/bin/bash
# ============================================================================
# TTD-Bench — PBS Pro array job: FAR-calibration scoring
# ============================================================================
# One array sub-job scores a slice of the calibration corpus (clips i, i+N,
# i+2N, ...). Each clip is cached as its own JSONL, so a killed sub-job costs
# one clip and resubmitting the identical array is safe and idempotent.
#
# GPU type is deliberately unconstrained: measured on identical software, an
# L40 and a GTX 1080 Ti disagree on 0.03 % of alarm decisions at threshold 0.5
# and 0.0009 % at 0.9 (results-far-calib/README.md). No place=excl, no cl_*,
# no scratch_local.
#
# BUT os=debian13 IS REQUIRED and is not a scientific constraint. The venv is
# built against the frontend python 3.13; a node running a different OS has a
# different python minor, site-packages does not resolve, and every task dies
# on "ModuleNotFoundError: No module named pandas". This cost 116 of 150 tasks
# in the first submission. Pin the OS, not the card.
#
# WALLTIME must match the TIER being submitted. Clip durations span 0.8 s to
# 32 550 s and ten clips of 1196 hold a third of the corpus, so an array split
# by clip count gives one task nine hours of video and another nine minutes.
# Use --min-duration / --max-duration to submit tiers of comparable size.
#
# Submit from the repo root:
#   qsub -q gpu -J 0-63 -v CORPUS_DIR=/storage/.../far-corpus hpc/pbs_far_scoring.sh
# ============================================================================

#PBS -N ttd-far
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:os=debian13
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:?submit from the repo root}}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/../.venv}"
CORPUS_DIR="${CORPUS_DIR:?set CORPUS_DIR (holds corpus_manifest.csv)}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints}"
OUT_DIR="${OUT_DIR:-$CORPUS_DIR/traces}"
SOURCES="${SOURCES:-}"
MIN_DUR="${MIN_DUR:-}"                 # tier lower bound, seconds
MAX_DUR="${MAX_DUR:-}"                 # tier upper bound, seconds
MODELS="${MODELS:-}"                   # subset, e.g. yolov8s-weapon (default: all four)
IMGSZ="${IMGSZ:-}"                     # inference size (default 640)

# Array geometry: PBS gives the index, the submit line gives the size.
IDX="${PBS_ARRAY_INDEX:-0}"
STRIDE="${STRIDE:?set STRIDE to the array size, e.g. 64 for -J 0-63}"

echo "== FAR scoring: index $IDX of $STRIDE on $(hostname) =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
git -C "$REPO_DIR" rev-parse HEAD || true

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python3 "$REPO_DIR/../common/scripts/far_score_clips.py" \
    --manifest "$CORPUS_DIR/corpus_manifest.csv" \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$OUT_DIR" \
    --index "$IDX" --stride "$STRIDE" \
    ${SOURCES:+--sources "$SOURCES"} \
    ${MIN_DUR:+--min-duration "$MIN_DUR"} \
    ${MAX_DUR:+--max-duration "$MAX_DUR"} \
    ${MODELS:+--models "$MODELS"} \
    ${IMGSZ:+--imgsz "$IMGSZ"}

echo "== done index $IDX =="

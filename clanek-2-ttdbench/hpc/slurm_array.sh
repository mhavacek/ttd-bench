#!/bin/bash
# ============================================================================
# TTD-Bench — SLURM array job (IT4Innovations / e-INFRA CZ)
# ============================================================================
# Same logic as hpc/pbs_array.sh: one array task = one matrix cell, mapped
# through `run_single.py --cell-index N`.
#
# BEFORE SUBMITTING — fill in the placeholders below (ACCOUNT, PARTITION,
# REPO_DIR), check the matrix size:
#   python3 scripts/run_single.py --list-cells | tail -1
# and submit, e.g. for 50 cells:
#   sbatch --array=0-49 hpc/slurm_array.sh
#
# INTENDED USE ON KAROLINA — the modern-GPU column, 45 cells:
#   EXPERIMENT_CONFIG=configs/experiment-phase1-moderngpu.yaml
#   RESULTS_SUBDIR=results-phase1-moderngpu
#   sbatch --array=0-44 -A <ACCOUNT> -p qgpu hpc/slurm_array.sh
# Pilot one cell first, in the short queue, which costs almost nothing:
#   sbatch --array=0-0 -A <ACCOUNT> -p qgpu_exp hpc/slurm_array.sh
#
# These results are a SEPARATE COLUMN, never merged with the frozen 1080 Ti
# campaign: the deployment is named local-gpu-modern precisely so run ids and
# config labels cannot collide. Measurement equivalence across environments was
# tested once and did not hold (docs/RERUN-PREREGISTRACE.md section 6);
# assuming it across two GPU generations would be worse.
#
# ENVIRONMENT — do NOT copy the MetaCentrum venv. It is built for sm_61
# (GTX 1080 Ti), where CUDA-13 wheels dropped Pascal support; the A100 is
# sm_80 and needs its own build. Verify with torch.cuda.get_device_capability(),
# which must report (8, 0). Full procedure: docs/DEPLOY-IT4I.md
#
# UNVERIFIED PLACEHOLDERS — check against current IT4I documentation before
# submitting, do not guess: partition name, account format, module names.
# ============================================================================

#SBATCH --job-name=ttd-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ttd-bench_%A_%a.out

# ---- user configuration (EDIT THESE) ---------------------------------------
ACCOUNT=""                          # project id from the approved allocation; sbatch -A
PARTITION="${PARTITION:-qgpu_exp}"  # qgpu_exp = 1 h cap, good for the pilot;
                                    # qgpu_free does not consume the allocation;
                                    # qgpu is the production queue (24/48 h)
# NOT /scratch: on Karolina the user scratch quota is 0 B and unusable. Work
# lives on the project filesystem (proj1, 20 TB for FTA-26-92).
REPO_DIR="${REPO_DIR:-/mnt/proj1/fta-26-92/vetev-a/clanek-2-ttdbench}"
CONTAINER="${REPO_DIR}/ttd-bench.sif"
# 0 = venv (what Karolina uses; no .sif is built there). Override to 1 only if
# a container image actually exists at ${CONTAINER}.
USE_CONTAINER="${USE_CONTAINER:-0}"
# setup_venv.sh builds into the BRANCH-A root, one level above this paper's
# directory -- REPO_DIR points at clanek-2-ttdbench, so the venv is its parent's.
VENV_DIR="${VENV_DIR:-${REPO_DIR}/../.venv}"
DATA_DIR="${DATA_DIR:-/mnt/proj1/fta-26-92/ttd-bench-data}"
# Ultralytics writes settings into $HOME/.config, which is not writable from a
# compute node; without this every task silently falls back to /tmp and the
# settings differ per node.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/mnt/proj1/fta-26-92/.ultralytics}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiment-phase1-moderngpu.yaml}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results-phase1-moderngpu}"
# -----------------------------------------------------------------------------

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?run via sbatch --array=<range>}"
CELL_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "== TTD-Bench cell ${CELL_INDEX} on $(hostname) =="
echo "account=${ACCOUNT:-<none>} partition=${PARTITION:-<default>}"

OUT_DIR="${REPO_DIR}/${RESULTS_SUBDIR}/raw"
mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

if [[ "${USE_CONTAINER}" == "1" ]]; then
    ml Apptainer 2>/dev/null || ml apptainer 2>/dev/null || true
    BINDS="${REPO_DIR}:${REPO_DIR}"
    [[ -n "${DATA_DIR}" ]] && BINDS="${BINDS},${DATA_DIR}:${DATA_DIR}"
    apptainer exec --nv \
        --bind "${BINDS}" \
        "${CONTAINER}" \
        python3 scripts/run_single.py \
            --config "${EXPERIMENT_CONFIG}" \
            --cell-index "${CELL_INDEX}" \
            --replay-mode rtsp \
            --out-dir "${OUT_DIR}"
else
    # Pin the module: replay encodes with libx264, and a bare `ml FFmpeg`
    # can resolve to a build against a different GCCcore than the venv's
    # Python. Verified present on Karolina 2026-08-27.
    ml "${FFMPEG_MODULE:-FFmpeg/7.1.2-GCCcore-14.3.0}" 2>/dev/null \
        || ml FFmpeg 2>/dev/null \
        || echo "WARNING: ffmpeg module not found; adjust FFMPEG_MODULE"
    "${VENV_DIR}/bin/python" scripts/run_single.py \
        --config "${EXPERIMENT_CONFIG}" \
        --cell-index "${CELL_INDEX}" \
        --replay-mode rtsp \
        --out-dir "${OUT_DIR}"
fi

echo "== cell ${CELL_INDEX} done =="

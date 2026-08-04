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
ACCOUNT=""                          # e.g. "OPEN-XX-XX"; pass via sbatch -A if preferred
PARTITION=""                        # e.g. "qgpu"; pass via sbatch -p if preferred
REPO_DIR="/scratch/project/CHANGE_ME/ttd-bench"
CONTAINER="${REPO_DIR}/ttd-bench.sif"
USE_CONTAINER=1
VENV_DIR="${REPO_DIR}/.venv"
DATA_DIR=""                         # dataset root OUTSIDE the repo (bind in)
EXPERIMENT_CONFIG="configs/experiment.yaml"   # Phase 1: experiment-phase1.yaml
RESULTS_SUBDIR="results"            # Phase 1: results-phase1 (matches config)
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
    ml FFmpeg 2>/dev/null || ml ffmpeg 2>/dev/null \
        || echo "WARNING: ffmpeg module not found; adjust module name"
    "${VENV_DIR}/bin/python" scripts/run_single.py \
        --config "${EXPERIMENT_CONFIG}" \
        --cell-index "${CELL_INDEX}" \
        --replay-mode rtsp \
        --out-dir "${OUT_DIR}"
fi

echo "== cell ${CELL_INDEX} done =="

#!/bin/bash
# ============================================================================
# TTD-Bench — PBS Pro array job (MetaCentrum, e-INFRA CZ)
# ============================================================================
# One array sub-job = one cell of the experiment matrix (deployment × network
# profile × model × scenario × repetition). The array index maps to the cell
# deterministically via `run_single.py --cell-index N` (enumeration order is
# fixed by configs/experiment.yaml — see ttdbench.config).
#
# BEFORE SUBMITTING — fill in the placeholders below:
#   PROJECT_ID  your MetaCentrum project/group for accounting (or leave empty)
#   QUEUE       target queue, e.g. gpu / default (check current queue list)
#   REPO_DIR    absolute path to the cloned ttd-bench repository on a storage
#               visible from compute nodes (e.g. /storage/brno2/home/$USER/...)
#
# Get the matrix size first (on the frontend):
#   python3 scripts/run_single.py --list-cells | tail -1
# then submit with the matching range, e.g. for 50 cells:
#   qsub -J 0-49 hpc/pbs_array.sh
#
# Resources below are conservative for a single replay+inference run on one
# node. Runs never span nodes (single clock domain by design).
# ============================================================================

#PBS -N ttd-bench
# os=debian13: venv mode needs every node to run the same OS image as the
# frontend the venv was built on (a different python minor on mixed-OS nodes
# breaks site-packages resolution -> ModuleNotFoundError). Drop or adjust the
# constraint when using the container instead.
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb:os=debian13
#PBS -l walltime=02:00:00
#PBS -j oe

# ---- user configuration -----------------------------------------------------
# Every value below can be overridden at submit time WITHOUT editing this file:
#   qsub -q gpu -J 0-134 -v EXPERIMENT_CONFIG=configs/experiment-phase1.yaml,RESULTS_SUBDIR=results-phase1 hpc/pbs_array.sh
# REPO_DIR defaults to the directory you submitted from (PBS_O_WORKDIR), i.e.
# submit from the repo root and no path needs configuring at all — this keeps
# the script working even after a fresh re-sync overwrites local edits.
PROJECT_ID="${PROJECT_ID:-}"        # e.g. "OPEN-XX-XX"; empty = default group
QUEUE="${QUEUE:-gpu}"               # informational; pass to qsub via -q
REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:?submit from the repo root (PBS_O_WORKDIR unset)}}"
DATA_DIR="${DATA_DIR:-}"            # dataset root OUTSIDE the repo (bind into
                                    # the container); empty if videos live
                                    # under REPO_DIR
CONTAINER="${CONTAINER:-${REPO_DIR}/ttd-bench.sif}"   # built from hpc/apptainer.def
# container if the .sif exists, venv otherwise (override with USE_CONTAINER=0/1)
if [[ -z "${USE_CONTAINER:-}" ]]; then
    [[ -f "${CONTAINER}" ]] && USE_CONTAINER=1 || USE_CONTAINER=0
fi
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"     # only used when USE_CONTAINER=0
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiment.yaml}"   # Phase 1: experiment-phase1.yaml
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results}"   # Phase 1: results-phase1 (matches config)
CELLS_PER_JOB="${CELLS_PER_JOB:-1}"   # only with CELL_LIST; see the block below
# -----------------------------------------------------------------------------

set -euo pipefail

: "${PBS_ARRAY_INDEX:?run via qsub -J <range> (array job)}"

# CELL_LIST (optional): a file of cell indices, one per line. The array index
# then selects the N-th LINE instead of being the cell index itself, so a
# scattered set of cells (e.g. a rerun of failed cells) can be submitted as one
# contiguous array. Generate it with scripts/rerun_failed.py.
if [[ -n "${CELL_LIST:-}" ]]; then
    [[ -f "${CELL_LIST}" ]] || { echo "CELL_LIST not found: ${CELL_LIST}" >&2; exit 2; }
    # CELLS_PER_JOB > 1 packs a contiguous slice of the list into one job, run
    # sequentially. Two reasons: PBS schedules ~80 jobs far better than ~800
    # (per-job overhead measured at ~4.5 min against a ~54 s median cell), and
    # sequential execution means a job can never collide with itself.
    FIRST_LINE=$((PBS_ARRAY_INDEX * CELLS_PER_JOB + 1))
    LAST_LINE=$((FIRST_LINE + CELLS_PER_JOB - 1))
    # read loop rather than mapfile: mapfile needs bash 4+, and this script has
    # to be verifiable on the machine it is written on (macOS ships bash 3.2)
    CELL_INDICES=()
    while IFS= read -r _line; do
        [[ -n "${_line}" ]] && CELL_INDICES+=( "${_line}" )
    done < <(sed -n "${FIRST_LINE},${LAST_LINE}p" "${CELL_LIST}")
    [[ ${#CELL_INDICES[@]} -gt 0 ]] || {
        echo "CELL_LIST ${CELL_LIST} has no lines ${FIRST_LINE}-${LAST_LINE}" >&2; exit 2; }
    echo "cell list: ${CELL_LIST} lines ${FIRST_LINE}-${LAST_LINE} -> cells ${CELL_INDICES[*]}"
else
    CELL_INDICES=( "${PBS_ARRAY_INDEX}" )
fi

echo "== TTD-Bench ${#CELL_INDICES[@]} cell(s) on $(hostname) =="
echo "project=${PROJECT_ID:-<none>} queue=${QUEUE}"

# results go to shared storage; raw logs are the source of truth
OUT_DIR="${REPO_DIR}/${RESULTS_SUBDIR}/raw"
mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

FAILED_CELLS=()
for CELL_INDEX in "${CELL_INDICES[@]}"; do
echo "===== cell ${CELL_INDEX} ====="
# One bad cell must not abandon the rest of the chunk; failures are collected
# and re-reported at the end so the job still exits non-zero.
set +e
if [[ "${USE_CONTAINER}" == "1" ]]; then
    # MetaCentrum: Apptainer available on all nodes; --nv exposes the GPU
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
    # venv fallback (no container): create once on the frontend with
    #   python3 -m venv --without-pip .venv   # (MetaCentrum system python lacks
    #   curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python  # ensurepip)
    #   .venv/bin/pip install -e '.[ml,dev]'
    #   bash scripts/fetch_mediamtx.sh && bash scripts/fetch_ffmpeg.sh
    # The bundled static ffmpeg (third_party/ffmpeg) is put first on PATH: the
    # MetaCentrum `ffmpeg` module (spack build) has NO libx264, so the replay's
    # H.264 encode fails with it. The static build ships libx264. mediamtx is
    # in the same dir. No `module add` needed.
    export PATH="${REPO_DIR}/third_party:${PATH}"
    "${VENV_DIR}/bin/python" scripts/run_single.py \
        --config "${EXPERIMENT_CONFIG}" \
        --cell-index "${CELL_INDEX}" \
        --replay-mode rtsp \
        --out-dir "${OUT_DIR}"
fi
RC=$?
set -e
if [[ ${RC} -ne 0 ]]; then
    echo "== cell ${CELL_INDEX} FAILED (rc=${RC}) =="
    FAILED_CELLS+=( "${CELL_INDEX}" )
else
    echo "== cell ${CELL_INDEX} done =="
fi
done

if [[ ${#FAILED_CELLS[@]} -gt 0 ]]; then
    echo "== ${#FAILED_CELLS[@]}/${#CELL_INDICES[@]} cells FAILED: ${FAILED_CELLS[*]} =="
    exit 1
fi
echo "== all ${#CELL_INDICES[@]} cell(s) done =="

#!/bin/bash
# Validate the mediamtx fix under the condition it was written for: two cells
# running AT THE SAME TIME on ONE node, each starting its own mediamtx.
#
#   qsub -q gpu -l select=1:ncpus=8:ngpus=1:mem=32gb:os=debian13:cl_konos=True \
#        -l walltime=00:30:00 \
#        -v VENV_DIR=$STORAGE/vetev-a/.venv \
#        clanek-2-ttdbench/hpc/smoke_concurrent.sh
#
# Deliberately NOT a job array. Two array elements are two independent jobs and
# the scheduler is free to put them on different nodes, in which case nothing
# collides and a green result would mean nothing. Backgrounding both cells
# inside one job makes the overlap certain.
#
# Before the fix roughly one of the two died on
#   RuntimeError: mediamtx failed to start ... bind: address already in use
# because the config template parameterised only rtspAddress while mediamtx
# also binds UDP :8000/:8001. After it both must finish with frames.
#PBS -N ttd-smoke
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:os=debian13
#PBS -l walltime=00:30:00
#PBS -j oe
set -uo pipefail

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:?submit from the repo root}}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
CONFIG="${EXPERIMENT_CONFIG:-configs/experiment-scvd.yaml}"
OUT_DIR="${REPO_DIR}/clanek-2-ttdbench/${RESULTS_SUBDIR:-results-smoke-fix}/raw"
CELL_A="${CELL_A:-0}"
CELL_B="${CELL_B:-1}"

cd "${REPO_DIR}/clanek-2-ttdbench"
export PATH="${REPO_DIR}/clanek-2-ttdbench/third_party:${PATH}"
mkdir -p "${OUT_DIR}"

echo "== concurrent smoke on $(hostname): cells ${CELL_A} and ${CELL_B} =="

run_cell() {
    "${VENV_DIR}/bin/python" scripts/run_single.py \
        --config "${CONFIG}" --cell-index "$1" \
        --replay-mode rtsp --out-dir "${OUT_DIR}" 2>&1 | sed "s/^/[cell $1] /"
    return "${PIPESTATUS[0]}"
}

run_cell "${CELL_A}" & PID_A=$!
run_cell "${CELL_B}" & PID_B=$!
wait "${PID_A}"; RC_A=$?
wait "${PID_B}"; RC_B=$?

echo "== exit codes: cell ${CELL_A}=${RC_A}  cell ${CELL_B}=${RC_B} =="
if [[ ${RC_A} -ne 0 || ${RC_B} -ne 0 ]]; then
    echo "!! SMOKE FAILED -- at least one cell did not complete" >&2
    exit 1
fi
echo "== both cells completed =="

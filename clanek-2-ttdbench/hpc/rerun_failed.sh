#!/bin/bash
#PBS -N ttd-rerun
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=20gb:os=debian13
#PBS -l walltime=02:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR}"
for i in 1 9 12 38 39 64 74 75 104 128 134; do
    echo "===== cell $i ====="
    .venv/bin/python scripts/run_single.py \
        --config configs/experiment-phase1.yaml \
        --cell-index "$i" \
        --out-dir results-phase1/raw || echo "cell $i FAILED"
done

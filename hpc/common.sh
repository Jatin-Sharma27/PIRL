#!/bin/bash
# Sourced at the top of every PIRL Slurm job.
#   - CPU threads capped to what the job booked; GPU selection left to Slurm (CUDA_VISIBLE_DEVICES)
#   - per-job temp files on scratch, deleted when the job ends
set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_DIR="/scratch/$USER"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export TMPDIR="$SCRATCH_DIR/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

source "$SCRATCH_DIR/miniforge3/etc/profile.d/conda.sh"
conda activate pirl
cd "$PROJECT_DIR"
mkdir -p results/logs logs
echo "== job ${SLURM_JOB_ID:-manual} on $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  $(date)"
nvidia-smi -L 2>/dev/null || true

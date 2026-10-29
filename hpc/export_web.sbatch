#!/bin/bash
# Data for the interactive browser simulation (pendulum LUTs + cart-pole replays).
#   sbatch --dependency=afterany:<finalize jobid> hpc/export_web.sbatch
#SBATCH --job-name=pirl
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/hpc/common.sh"
python -u scripts/export_web.py
echo "== done $(date)"

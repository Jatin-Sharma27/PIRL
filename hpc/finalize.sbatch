#!/bin/bash
# Tables, figures, GIFs, robustness sweeps (MPPI on GPU) and the ESP32 LUT export.
#   sbatch --dependency=afterany:<jobids> hpc/finalize.sbatch
#SBATCH --job-name=pirl
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/hpc/common.sh"
python -u scripts/compare.py --system pendulum
python -u scripts/compare.py --system cartpole
python -u scripts/deploy_lut.py --export PPO
echo "== done $(date)"

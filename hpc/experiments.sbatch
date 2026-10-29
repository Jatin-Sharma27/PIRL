#!/bin/bash
# One GPU job per plant: physics-informed / MPC methods on the GPU while the SB3 agents
# train on the job's CPU cores at the same time.
#   SYSTEM=cartpole METHODS="PIRL XTFC-PIRL" SB3="DDPG TD3 SAC" sbatch hpc/experiments.sbatch
# Methods with spaces in their names are separated by ';'  e.g. METHODS="PIRL;IRL (no physics)".
#SBATCH --job-name=pirl
#SBATCH --partition=rtx
#SBATCH --account=courses
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
# course QoS: max 1-12:00:00 wall time, 1 GPU per user (jobs share the name "pirl" + singleton)
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source "$SLURM_SUBMIT_DIR/hpc/common.sh"
SYSTEM="${SYSTEM:-pendulum}"

pids=()
for algo in ${SB3:-}; do
    OMP_NUM_THREADS=2 python -u scripts/train_sb3.py --system "$SYSTEM" --algo "$algo" --device cpu \
        > "results/logs/${SYSTEM}_${algo}.log" 2>&1 &
    pids+=($!)
done

if [ -n "${METHODS:-}" ]; then
    IFS=';' read -ra M <<< "$METHODS"
    OMP_NUM_THREADS=4 python -u scripts/run_experiments.py --system "$SYSTEM" --methods "${M[@]}" \
        2>&1 | tee -a "results/logs/run_${SYSTEM}_hpc.log"
fi

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "== done $(date)  sb3 rc=$rc"

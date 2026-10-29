#!/usr/bin/env bash
# Launch all 10 SB3 runs in parallel (5 algorithms x 2 systems).  Logs -> results/logs/
# PY can point at a CPU-only torch install: SB3 MLPs here are faster on CPU, and each
# CUDA-torch process costs ~1.5 GB of commit memory.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
for sys in pendulum cartpole; do
  for algo in DDPG TD3 SAC PPO A2C; do
    $PY -u scripts/train_sb3.py --system $sys --algo $algo --device cpu > results/logs/${sys}_${algo}.log 2>&1 &
  done
done
wait

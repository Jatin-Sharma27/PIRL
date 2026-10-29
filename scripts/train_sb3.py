"""Train a Stable-Baselines3 agent (DDPG / TD3 / SAC / PPO / A2C) on a swing-up env.

Every `eval_every` steps the deterministic policy is scored with the SAME batched
evaluator used for the HJB/MPC controllers, giving a learning curve in task cost.

    python scripts/train_sb3.py --system pendulum --algo SAC --steps 100000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from pirl.controllers.classical import Policy
from pirl.envs import SwingUpEnv
from pirl.evaluate import evaluate
from pirl.systems import make_system

ALGOS = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO, "A2C": A2C}
OFF_POLICY = {"DDPG", "TD3", "SAC"}


class SB3Policy(Policy):
    def __init__(self, sys, model, name):
        self.sys, self.model, self.name = sys, model, name

    def act(self, X):
        obs = self.sys.features(X).astype(np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        return a.astype(np.float64).reshape(-1, 1)


class EvalCallback(BaseCallback):
    def __init__(self, sys, every, log_path, name, best_path):
        super().__init__()
        self.sys, self.every, self.log_path, self.name = sys, every, log_path, name
        self.best_path, self.best = best_path, None
        self.curve, self.last, self.t0 = [], 0, time.time()

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            m, _ = evaluate(self.sys, SB3Policy(self.sys, self.model, self.name))
            m["timesteps"] = self.num_timesteps
            m["wall_s"] = time.time() - self.t0
            self.curve.append(m)
            # keep the best checkpoint (success first, then cost): off-policy agents on the
            # cart-pole can collapse late in training, so the last iterate is not the best one
            score = (m["success_rate"], -(m["mean_cost"] if m["mean_cost"] is not None else 1e9))
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(self.best_path)
            print(f"[{self.name}] {self.num_timesteps:>8d} steps  succ={m['success_rate']:.2f} "
                  f"nomJ={m['nominal_cost']}  meanJ={m['mean_cost']}", flush=True)
            with open(self.log_path, "w") as f:
                json.dump(self.curve, f, indent=1)
        return True


def build(algo, env, sys_name, seed, device):
    pk = dict(net_arch=[128, 128])
    common = dict(gamma=0.995, seed=seed, device=device, verbose=0)
    if algo in OFF_POLICY:
        kw = dict(learning_rate=3e-4, buffer_size=500_000, batch_size=256, learning_starts=5000,
                  train_freq=1, gradient_steps=1, policy_kwargs=pk, **common)
        if algo == "DDPG":
            kw["action_noise"] = NormalActionNoise(np.zeros(1), 0.3 * np.ones(1))
            kw["learning_rate"] = 1e-3 if sys_name == "pendulum" else 3e-4
        if algo == "TD3":
            kw["action_noise"] = NormalActionNoise(np.zeros(1), 0.2 * np.ones(1))
        return ALGOS[algo]("MlpPolicy", env, **kw)
    pk = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-0.5)
    if algo == "PPO":
        return PPO("MlpPolicy", env, n_steps=512, batch_size=2048, n_epochs=10, gae_lambda=0.95,
                   learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, policy_kwargs=pk, **common)
    return A2C("MlpPolicy", env, n_steps=32, learning_rate=3e-4, gae_lambda=0.95, ent_coef=0.0,
               use_rms_prop=False, normalize_advantage=True, max_grad_norm=0.5, policy_kwargs=pk, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pendulum")
    ap.add_argument("--algo", default="SAC")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(2)
    algo = a.algo.upper()
    off = algo in OFF_POLICY
    steps = a.steps or {("pendulum", True): 80_000, ("pendulum", False): 2_000_000,
                        ("cartpole", True): 600_000, ("cartpole", False): 5_000_000}[(a.system, off)]
    device = a.device or ("cuda" if off and torch.cuda.is_available() else "cpu")
    env_kw = dict(system=a.system, reward_scale=10.0)
    env = make_vec_env(SwingUpEnv, n_envs=1 if off else 16, seed=a.seed, env_kwargs=env_kw)
    model = build(algo, env, a.system, a.seed, device)
    out = os.path.join(a.out, a.system, "sb3")
    os.makedirs(out, exist_ok=True)
    name = f"{algo}_s{a.seed}"
    best_path = os.path.join(out, name)
    cb = EvalCallback(make_system(a.system), every=max(steps // 40, 2000), name=name,
                      log_path=os.path.join(out, f"{name}_curve.json"), best_path=best_path)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    model.save(os.path.join(out, name + "_last"))
    model = ALGOS[algo].load(best_path, device="cpu")
    m, _ = evaluate(make_system(a.system), SB3Policy(make_system(a.system), model, algo))
    m["selected"] = "best checkpoint by evaluation"
    m.update(train_wall_s=time.time() - t0, timesteps=steps, device=device)
    with open(os.path.join(out, f"{name}_final.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"[{name}] DONE {m}", flush=True)


if __name__ == "__main__":
    main()

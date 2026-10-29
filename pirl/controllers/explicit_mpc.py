"""Approximate explicit MPC: distil the MPPI controller into a small network.

Slide 4, item 4.  An exact explicit MPC only exists for linear/convex problems.
Instead we sample closed-loop MPPI trajectories and regress the policy on them.
Coverage uses DART-style execution noise: the plant gets a noisy action, but
the label is the clean MPPI action, so the network also learns how to recover
from off-nominal states.  The result is a ~10 us feedforward policy, small
enough for a microcontroller.
"""
import numpy as np
import torch

from ..evaluate import test_initial_states
from ..hjb import Actor, ActorPolicy, DEV, T
from ..systems import rk4_step, wrap, angle_index
from .mppi import MPPI


def collect_mppi_data(sys, n_envs=512, T_s=6.0, exec_noise=0.25, seed=0, K=512, log=print):
    rng = np.random.default_rng(seed)
    ai = angle_index(sys)
    X = sys.sample_states(n_envs, rng)
    X[:, ai + 1] *= 0.3
    if sys.name == "cartpole":
        X[:, 0] *= 0.5; X[:, 1] *= 0.3
    half = n_envs // 2
    X[:half] = sys.down_state() + rng.normal(0, 0.05, (half, sys.n))  # half start hanging
    mpc = MPPI(sys, K=K, seed=seed)
    mpc.reset(n_envs)
    xs, us = [], []
    for k in range(int(T_s / sys.dt)):
        U = mpc(X)
        xs.append(X.copy()); us.append(U.copy())
        Ue = np.clip(U + rng.normal(0, exec_noise, U.shape), -1, 1)
        X = rk4_step(sys, X, Ue)
        X[:, ai] = wrap(X[:, ai])
        if sys.name == "cartpole":  # respawn carts that hit the stops
            bad = sys.out_of_bounds(X)
            X[bad] = sys.down_state()
        if k % 50 == 0:
            log(f"  MPPI data step {k}")
    return np.concatenate(xs), np.concatenate(us)


def train_explicit_mpc(sys, X, U, iters=8000, seed=0, log=print):
    torch.manual_seed(seed)
    actor = Actor(sys, hidden=128, layers=3).to(DEV)
    opt = torch.optim.Adam(actor.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    Xt, Ut = T(X), T(U)
    for it in range(iters):
        idx = torch.randint(0, len(Xt), (4096,), device=DEV)
        loss = ((actor(Xt[idx]) - Ut[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0:
            log(f"  explicit-MPC fit it {it}: mse={loss.item():.4f}")
    return actor, ActorPolicy(actor, "Explicit-MPC (NN)")

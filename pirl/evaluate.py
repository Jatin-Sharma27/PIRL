"""Batched closed-loop rollouts and metrics, the same for every controller."""
import json
import math
import time

import numpy as np

from .systems import rk4_step, wrap, angle_index


def test_initial_states(sys, n_random=19, seed=123):
    """Nominal hanging-down start plus a fixed set of random starts."""
    rng = np.random.default_rng(seed)
    X = [sys.down_state()]
    for _ in range(n_random):
        x = sys.down_state().copy()
        ai = angle_index(sys)
        x[ai] = rng.uniform(-math.pi, math.pi)
        x[ai + 1] = rng.uniform(-2.0, 2.0)
        if sys.name == "cartpole":
            x[0] = rng.uniform(-0.15, 0.15)
        X.append(x)
    return np.array(X)


def rollout(sys, policy, X0, T=None, act_noise=0.0, obs_fn=None, seed=0):
    """Simulate N trajectories in parallel.  Returns dict of arrays (N, steps+1, ...)."""
    T = T or (10.0 if sys.name == "pendulum" else 12.0)
    steps = int(round(T / sys.dt))
    rng = np.random.default_rng(seed)
    X = np.array(X0, dtype=np.float64)
    N = X.shape[0]
    ai = angle_index(sys)
    xs = np.zeros((N, steps + 1, sys.n)); us = np.zeros((N, steps, 1)); cs = np.zeros((N, steps))
    alive = np.ones(N, bool)
    xs[:, 0] = X
    if hasattr(policy, "reset"):
        policy.reset(N)
    t0 = time.perf_counter()
    for k in range(steps):
        Xobs = obs_fn(X, rng) if obs_fn else X
        U = np.clip(policy(Xobs), -1, 1).reshape(N, 1)
        if act_noise:
            U = np.clip(U + rng.normal(0, act_noise, U.shape), -1, 1)
        cs[:, k] = sys.cost(X, U) * sys.dt
        Xn = rk4_step(sys, X, U)
        Xn[:, ai] = wrap(Xn[:, ai])
        oob = sys.out_of_bounds(Xn)
        alive &= ~oob
        Xn = np.where(alive[:, None], Xn, X)  # freeze crashed carts at the wall
        X = Xn
        xs[:, k + 1] = X; us[:, k] = U
    wall = time.perf_counter() - t0
    return {"x": xs, "u": us, "c": cs, "alive": alive, "dt": sys.dt, "wall": wall / steps}


def metrics(sys, traj, hold_s=1.5):
    xs, alive, dt = traj["x"], traj["alive"], traj["dt"]
    up = sys.is_upright(xs)  # (N, steps+1)
    N, S = up.shape
    hold = int(hold_s / dt)
    success = up[:, -hold:].all(1) & alive
    # time after which it stays upright
    t_up = np.full(N, np.nan)
    for i in range(N):
        if success[i]:
            bad = np.where(~up[i])[0]
            t_up[i] = (bad[-1] + 1) * dt if len(bad) else 0.0
    J = traj["c"].sum(1)
    J = np.where(alive, J, np.nan)
    return {
        "success_rate": float(success.mean()),
        "nominal_success": bool(success[0]),
        "nominal_t_up": float(t_up[0]) if success[0] else None,
        "mean_t_up": float(np.nanmean(t_up)) if success.any() else None,
        "nominal_cost": float(J[0]) if alive[0] else None,
        "mean_cost": float(np.nanmean(J)) if alive.any() else None,
        "mean_abs_u": float(np.abs(traj["u"]).mean()),
        "saturation_frac": float((np.abs(traj["u"]) > 0.98).mean()),
        "ms_per_step": 1000 * traj["wall"] / N,
    }


def evaluate(sys, policy, X0=None, **kw):
    X0 = test_initial_states(sys) if X0 is None else X0
    tr = rollout(sys, policy, X0, **kw)
    return metrics(sys, tr), tr


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

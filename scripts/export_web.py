"""Export data for the interactive browser simulation (results/web_data.json).

Pendulum: int8 lookup tables of each trained policy over (theta, omega). These are the same tables
the ESP32 runs (pirl/deploy.py), so the browser closes the loop live with a JS integrator and the
motor strength can be changed.
Cart-pole: the 4-D policies can't be tabulated compactly, so each method's closed-loop run from
hanging rest (plus two other starts) is recorded for replay.

    python scripts/export_web.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pirl.controllers.classical import EnergyLQR
from pirl.deploy import build_lut
from pirl.evaluate import evaluate, test_initial_states
from pirl.systems import make_system
from scripts.compare import load_policies, gather_metrics

PEND = ["DP", "PIRL", "XTFC-PIRL", "XTFC-PI", "PINN-AC", "DDPG", "SAC", "TD3", "PPO", "Explicit-MPC"]
CART = ["Energy+LQR", "MPPI-MPC", "Explicit-MPC", "SAC", "PPO", "PIRL (no anchor)", "PIRL", "XTFC-PIRL",
        "DDPG", "TD3", "A2C", "LQR"]


def r(a, d=4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def main():
    out = {}
    # ---------------- pendulum ----------------
    s = make_system("pendulum")
    pols = load_policies(s, os.path.join("results", "pendulum"))
    M = gather_metrics(os.path.join("results", "pendulum"))
    luts = {}
    for n in PEND:
        if n in pols:
            t = build_lut(pols[n], n_th=128, n_w=64, w_lim=12.0)
            luts[n] = {"lut": t["lut"].astype(int).ravel().tolist(), "metrics": M.get(n)}
            print("pendulum LUT", n, flush=True)
    out["pendulum"] = {
        "params": {k: getattr(s, k) for k in ["m", "l", "g", "J_rotor", "tau_stall", "w_noload", "b", "tau_c",
                                              "q_th", "q_w", "r", "dt", "substeps"]},
        "lut_shape": [128, 64], "w_lim": 12.0, "policies": luts,
        "energy_lqr": {"K": EnergyLQR(s).K.ravel().tolist(), "k_e": EnergyLQR(s).k_e, "catch": EnergyLQR(s).catch},
        "metrics": M,
    }
    # ---------------- cart-pole ----------------
    s = make_system("cartpole")
    pols = load_policies(s, os.path.join("results", "cartpole"))
    M = gather_metrics(os.path.join("results", "cartpole"))
    X0 = test_initial_states(s)[[0, 3, 7]]
    trajs = {}
    for n in CART:
        if n not in pols:
            continue
        if hasattr(pols[n], "reset"):
            pols[n].U = None
        _, tr = evaluate(s, pols[n], X0=X0, T=10.0)
        trajs[n] = {"x": [r(tr["x"][i, :, [0, 2]].T, 3) for i in range(len(X0))],
                    "u": [r(tr["u"][i, :, 0], 3) for i in range(len(X0))],
                    "alive": tr["alive"].tolist(), "metrics": M.get(n)}
        print("cartpole traj", n, flush=True)
    out["cartpole"] = {
        "params": {k: getattr(s, k) for k in ["M", "m", "l", "x_lim", "F_stall", "v_noload", "dt"]},
        "starts": r(X0, 3), "trajs": trajs, "metrics": M,
    }
    with open(os.path.join("results", "web_data.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote results/web_data.json", os.path.getsize(os.path.join("results", "web_data.json")) // 1024, "KB")


if __name__ == "__main__":
    main()

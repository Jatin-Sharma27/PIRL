"""Train + evaluate the model-based / physics-informed controllers on one system.

    python scripts/run_experiments.py --system pendulum
    python scripts/run_experiments.py --system cartpole --methods PIRL XTFC-PIRL

Each trained controller is saved to results/<system>/controllers/<name>.pt
(scripts/compare.py reloads them for plots, GIFs, robustness and LUT export).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pirl import hjb
from pirl.controllers.classical import EnergyLQR, LQROnly
from pirl.controllers.explicit_mpc import collect_mppi_data, train_explicit_mpc
from pirl.controllers.mppi import MPPI
from pirl.dp import DPPolicy, GridValue
from pirl.evaluate import evaluate
from pirl.systems import make_system

# Every critic uses the LQR-anchored constrained expression unless its name says "no anchor".
ALL = ["LQR", "Energy+LQR", "DP", "MPPI-MPC", "Explicit-MPC", "PINN-HJB", "PINN-AC",
       "XTFC-PI", "XTFC-PIRL", "IRL (no physics)", "PIRL",
       "XTFC-PIRL (no anchor)", "PIRL (no anchor)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pendulum")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    s = make_system(a.system)
    out = os.path.join("results", a.system)
    cdir = os.path.join(out, "controllers"); os.makedirs(cdir, exist_ok=True)
    mpath = os.path.join(out, "metrics_model_based.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    hpath = os.path.join(out, "training_history.json")
    history = json.load(open(hpath)) if os.path.exists(hpath) else {}
    methods = a.methods or [m for m in ALL if not (a.system == "cartpole" and m == "DP")]
    q = 0.25 if a.quick else 1.0
    pend = a.system == "pendulum"
    init = EnergyLQR(s)
    logf = open(os.path.join(out, "train_log.txt"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    for name in methods:
        log(f"=== {a.system}: {name} ===")
        t0 = time.time(); hist = None; obj = None
        if name == "LQR":
            pol = LQROnly(s)
        elif name == "Energy+LQR":
            pol = init
        elif name == "DP":
            g = GridValue(s); g.solve(verbose=False)
            pol = DPPolicy(g); obj = {"V": g.V.cpu(), "th": g.th.cpu(), "w": g.w.cpu()}
        elif name == "MPPI-MPC":
            pol = MPPI(s)
        elif name == "Explicit-MPC":
            X, U = collect_mppi_data(s, n_envs=512, T_s=6.0 if pend else 8.0, log=log)
            actor, pol = train_explicit_mpc(s, X, U, iters=int(8000 * q), log=log)
            obj = actor
        elif name == "PINN-HJB":
            V, hist = hjb.train_pinn_hjb(s, iters=int(20000 * q), anchor=True, log=log)
            pol = hjb.CriticPolicy(s, V, name); obj = V
        elif name == "PINN-AC":
            V, A, hist = hjb.train_pinn_ac(s, init, n_pi=int(12 * q) or 2, critic_iters=1500,
                                           actor_iters=500, anchor=True, log=log)
            pol = hjb.ActorPolicy(A, name); obj = {"critic": V, "actor": A}
        elif name.startswith("XTFC"):
            L = 1500 if pend else 2500
            elm, hist = hjb.train_xtfc(s, init, n_pi=int(12 * q) or 2, L=L, irl="PIRL" in name,
                                       lam_phys=0.1, alpha=1e-6 if pend else 1e-5,
                                       n_col=30000 if pend else 60000,
                                       anchor="no anchor" not in name, log=log)
            pol = hjb.ELMPolicy(elm, name); obj = elm
        elif name.startswith("PIRL") or name.startswith("IRL"):
            lam = 0.0 if name.startswith("IRL") else 0.1
            V, hist = hjb.train_pirl(s, init, n_pi=int(10 * q) or 2, iters=2000, lam_phys=lam,
                                     n_starts=4096 if pend else 8192,
                                     anchor="no anchor" not in name, log=log)
            pol = hjb.CriticPolicy(s, V, name); obj = V
        else:
            raise ValueError(name)
        train_s = time.time() - t0
        m, tr = evaluate(s, pol)
        m["train_wall_s"] = train_s
        # re-read before writing: several runner processes may share these files
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        history = json.load(open(hpath)) if os.path.exists(hpath) else {}
        metrics[name] = m
        if hist is not None:
            history[name] = hist
        log(f"  -> {json.dumps(m)}")
        if obj is not None:
            torch.save(obj, os.path.join(cdir, name.replace(" ", "_").replace("(", "").replace(")", "") + ".pt"))
        np.savez_compressed(os.path.join(cdir, name.replace(" ", "_").replace("(", "").replace(")", "") + "_traj.npz"),
                            x=tr["x"], u=tr["u"], c=tr["c"], alive=tr["alive"])
        json.dump(metrics, open(mpath, "w"), indent=1)
        json.dump(history, open(hpath, "w"), indent=1)


if __name__ == "__main__":
    main()

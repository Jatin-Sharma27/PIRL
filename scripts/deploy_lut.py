"""Turn trained pendulum policies into ESP32 lookup tables and test them in a hardware-in-the-loop sim.

    python scripts/deploy_lut.py                       # all available policies
    python scripts/deploy_lut.py --export DDPG         # policy written to firmware/.../policy_lut.h
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pirl.deploy import LUTPolicy, build_lut, export_header, hardware_rollout
from pirl.evaluate import metrics, rollout, test_initial_states
from pirl.systems import make_system
from scripts.compare import load_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="DDPG", help="policy whose LUT goes into the firmware header")
    ap.add_argument("--n_th", type=int, default=128)
    ap.add_argument("--n_w", type=int, default=64)
    a = ap.parse_args()
    s = make_system("pendulum")
    root = os.path.join("results", "pendulum")
    pols = load_policies(s, root)
    X0 = test_initial_states(s)
    names = [n for n in ["DDPG", "SAC", "TD3", "PPO", "PIRL", "XTFC-PIRL", "XTFC-PI", "DP", "Energy+LQR"] if n in pols]
    R = {}
    for n in names:
        tab = build_lut(pols[n], a.n_th, a.n_w)
        lut = LUTPolicy(tab, n + " LUT")
        r = {}
        r["float policy, ideal sensing"] = metrics(s, rollout(s, pols[n], X0))
        r["int8 LUT, ideal sensing"] = metrics(s, rollout(s, lut, X0))
        r["int8 LUT, encoder+EMA+delay+PWM"] = metrics(s, hardware_rollout(s, lut, X0))
        r["int8 LUT, HIL + 5% torque noise"] = metrics(s, hardware_rollout(s, lut, X0, torque_noise=0.05))
        r["lut_bytes"] = int(tab["lut"].size)
        R[n] = r
        print(n, {k: (v["success_rate"], round(v["mean_cost"] or -1, 3)) for k, v in r.items() if isinstance(v, dict)}, flush=True)
        if n == a.export:
            hdr = os.path.join("firmware", "esp32_pendulum", "policy_lut.h")
            export_header(tab, hdr, n)
            print(f"  wrote {hdr} ({tab['lut'].size} bytes)")
    json.dump(R, open(os.path.join(root, "deployment_hil.json"), "w"), indent=1)
    with open(os.path.join(root, "deployment_hil.md"), "w") as f:
        cols = ["float policy, ideal sensing", "int8 LUT, ideal sensing", "int8 LUT, encoder+EMA+delay+PWM",
                "int8 LUT, HIL + 5% torque noise"]
        f.write("| Policy | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n")
        for n, r in R.items():
            f.write(f"| {n} | " + " | ".join(f"{r[c]['success_rate']*100:.0f}% / J={r[c]['mean_cost']:.2f}"
                                             if r[c]['mean_cost'] else f"{r[c]['success_rate']*100:.0f}%" for c in cols) + " |\n")


if __name__ == "__main__":
    main()

"""Aggregate every controller (model-based + SB3) into tables, figures, GIFs and a robustness study.

    python scripts/compare.py --system pendulum
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pirl import hjb
from pirl.animate import animate
from pirl.controllers.classical import EnergyLQR, LQROnly
from pirl.controllers.mppi import MPPI
from pirl.dp import DPPolicy, GridValue
from pirl.evaluate import evaluate, test_initial_states
from pirl.systems import make_system, angle_index

# validated categorical slots (dataviz reference palette, light mode), fixed order by FAMILY
FAMILY_COLOR = {"Classical": "#2a78d6", "MPC": "#eb6834", "Physics-informed": "#1baf7a",
                "Model-free RL": "#6250d6", "Optimal (DP)": "#52514e"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY = {"LQR": "Classical", "Energy+LQR": "Classical", "DP": "Optimal (DP)",
          "MPPI-MPC": "MPC", "Explicit-MPC": "MPC", "PINN-HJB": "Physics-informed",
          "PINN-AC": "Physics-informed", "XTFC-PI": "Physics-informed", "XTFC-PIRL": "Physics-informed",
          "PIRL": "Physics-informed", "IRL (no physics)": "Physics-informed",
          "PIRL (no anchor)": "Physics-informed", "XTFC-PIRL (no anchor)": "Physics-informed",
          "DDPG": "Model-free RL", "TD3": "Model-free RL", "SAC": "Model-free RL",
          "PPO": "Model-free RL", "A2C": "Model-free RL"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.dpi": 130})


def fname(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


class SB3Policy:
    def __init__(self, sys, model, name):
        self.sys, self.model, self.name = sys, model, name

    def __call__(self, X):
        X = np.atleast_2d(X)
        a, _ = self.model.predict(self.sys.features(X).astype(np.float32), deterministic=True)
        return np.clip(a.astype(np.float64).reshape(-1, 1), -1, 1)


def load_policies(s, root):
    cdir = os.path.join(root, "controllers")
    pols = {"LQR": LQROnly(s), "Energy+LQR": EnergyLQR(s), "MPPI-MPC": MPPI(s)}
    def ld(n):
        p = os.path.join(cdir, fname(n) + ".pt")
        return torch.load(p, weights_only=False, map_location=hjb.DEV) if os.path.exists(p) else None
    o = ld("DP")
    if o is not None:
        g = GridValue(s); g.V = o["V"].to(g.dev); pols["DP"] = DPPolicy(g)
    o = ld("Explicit-MPC")
    if o is not None:
        pols["Explicit-MPC"] = hjb.ActorPolicy(o, "Explicit-MPC")
    for n in ["PINN-HJB", "PIRL", "IRL (no physics)", "PIRL (no anchor)"]:
        o = ld(n)
        if o is not None:
            pols[n] = hjb.CriticPolicy(s, o, n)
    o = ld("PINN-AC")
    if o is not None:
        pols["PINN-AC"] = hjb.ActorPolicy(o["actor"], "PINN-AC")
    for n in ["XTFC-PI", "XTFC-PIRL", "XTFC-PIRL (no anchor)"]:
        o = ld(n)
        if o is not None:
            pols[n] = hjb.ELMPolicy(o, n)
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    C = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO, "A2C": A2C}
    for n, cls in C.items():
        p = os.path.join(root, "sb3", f"{n}_s0.zip")
        if os.path.exists(p):
            pols[n] = SB3Policy(s, cls.load(p, device="cpu"), n)
    return pols


def gather_metrics(root):
    M = {}
    p = os.path.join(root, "metrics_model_based.json")
    if os.path.exists(p):
        M.update(json.load(open(p)))
    for f in glob.glob(os.path.join(root, "sb3", "*_s0_final.json")):
        M[os.path.basename(f).split("_")[0]] = json.load(open(f))
    return M


def table(M, s):
    order = sorted(M, key=lambda k: (-M[k]["success_rate"], M[k]["mean_cost"] if M[k]["mean_cost"] else 1e9))
    lines = ["| Method | Family | Success (20 starts) | Swing-up from rest | t_up nominal [s] | Cost nominal | Cost mean | mean abs u | train time [s] |",
             "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    for k in order:
        m = M[k]
        lines.append(f"| {k} | {FAMILY.get(k, 'Physics-informed')} | {m['success_rate']*100:.0f}% | {'yes' if m['nominal_success'] else 'no'} | "
                     f"{f(m['nominal_t_up'])} | {f(m['nominal_cost'])} | {f(m['mean_cost'] if m['success_rate'] >= 0.5 else None)} | {f(m['mean_abs_u'])} | "
                     f"{f(m.get('train_wall_s'), 0)} |")
    return "\n".join(lines), order


def fig_costs(M, order, path, title):
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        m = M[k]
        c = m["mean_cost"] if m["success_rate"] > 0 else None
        val = c if c is not None else 0
        ax.barh(yi, val, height=0.62, color=FAMILY_COLOR[FAMILY.get(k, "Physics-informed")], edgecolor="white", linewidth=2)
        lab = f"{val:.2f}  ·  {m['success_rate']*100:.0f}% success" if c is not None else "fails (0% success)"
        ax.text(val + 0.05, yi, lab, va="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("mean closed-loop cost over 20 starts (lower is better)")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f_, c in FAMILY_COLOR.items()]
    ax.legend(handles, FAMILY_COLOR.keys(), frameon=False, loc="lower right", fontsize=8)
    ax.set_xlim(0, max([M[k]["mean_cost"] or 0 for k in order if M[k]["success_rate"] > 0] + [1]) * 1.55)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_timeseries(s, trajs, path):
    ai = angle_index(s)
    rows = 3 if s.name == "pendulum" else 4
    fig, axs = plt.subplots(rows, 1, figsize=(7.5, 1.7 * rows + 0.6), sharex=True)
    for i, (k, (xs, us)) in enumerate(trajs.items()):
        t = np.arange(len(xs)) * s.dt
        c = SERIES[i % len(SERIES)]
        axs[0].plot(t, np.abs(np.angle(np.exp(1j * xs[:, ai]))), lw=1.6, color=c, label=k)
        axs[1].plot(t, xs[:, ai + 1], lw=1.4, color=c)
        axs[2].plot(t[:-1], us[:, 0], lw=1.2, color=c)
        if rows == 4:
            axs[3].plot(t, xs[:, 0], lw=1.4, color=c)
    axs[0].set_ylabel("|angle from upright| [rad]"); axs[1].set_ylabel("angular rate [rad/s]")
    axs[2].set_ylabel("motor command u"); axs[2].set_ylim(-1.1, 1.1)
    if rows == 4:
        axs[3].set_ylabel("cart position [m]")
        for yv in (-s.x_lim, s.x_lim):
            axs[3].axhline(yv, color="#8a8984", ls="--", lw=0.8)
    axs[-1].set_xlabel("time [s]")
    axs[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axs[0].set_title("Swing-up from hanging rest (weak motor: stall torque below what holding the pole horizontal needs)", loc="left",
                     fontsize=9, color="#0b0b0b")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_learning(root, M, path, s):
    curves = {}
    for f in sorted(glob.glob(os.path.join(root, "sb3", "*_s0_curve.json"))):
        curves[os.path.basename(f).split("_")[0]] = json.load(open(f))
    if not curves:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, (k, cv) in enumerate(curves.items()):
        ts = [c["timesteps"] for c in cv]
        axs[0].plot(ts, [c["mean_cost"] if c["mean_cost"] else np.nan for c in cv], lw=2, color=SERIES[i], label=k)
        axs[1].plot(ts, [c["success_rate"] * 100 for c in cv], lw=2, color=SERIES[i], label=k)
    for ref, ls in (("DP", "-"), ("PIRL", "--"), ("Energy+LQR", ":")):
        if ref in M and M[ref]["mean_cost"]:
            axs[0].axhline(M[ref]["mean_cost"], color="#52514e", ls=ls, lw=1)
            axs[0].annotate(f"{ref}", (1.0, M[ref]["mean_cost"]), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7, color="#52514e")
    axs[0].set_xscale("log"); axs[1].set_xscale("log")
    axs[0].set_ylabel("mean cost (20 starts)"); axs[1].set_ylabel("success rate [%]")
    for a in axs:
        a.set_xlabel("environment steps")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title(f"Model-free RL learning curves - {s.name}", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_value(s, root, pols, path):
    """Pendulum only: learned critics vs DP ground-truth value function."""
    cdir = os.path.join(root, "controllers")
    p = os.path.join(cdir, "DP.pt")
    if not os.path.exists(p):
        return None
    g = GridValue(s); g.V = torch.load(p, weights_only=False)["V"].to(g.dev)
    th, w = np.meshgrid(np.linspace(-np.pi, np.pi, 161), np.linspace(-8, 8, 121))
    X = np.stack([th.ravel(), w.ravel()], -1)
    Vt = g.value(X).cpu().numpy()
    crit = {}
    for n in ["PIRL", "XTFC-PIRL", "XTFC-PI", "IRL (no physics)", "PINN-HJB"]:
        q = os.path.join(cdir, fname(n) + ".pt")
        if os.path.exists(q):
            crit[n] = torch.load(q, weights_only=False, map_location=hjb.DEV)
    q = os.path.join(cdir, "PINN-AC.pt")
    if os.path.exists(q):
        crit["PINN-AC"] = torch.load(q, weights_only=False, map_location=hjb.DEV)["critic"]
    k = len(crit) + 1
    fig, axs = plt.subplots(1, k, figsize=(2.6 * k, 2.7), sharey=True)
    vmax = np.percentile(Vt, 99)
    im = axs[0].pcolormesh(th, w, Vt.reshape(th.shape), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
    axs[0].set_title("DP ground truth V*", fontsize=9)
    errs = {}
    for a, (n, c) in zip(axs[1:], crit.items()):
        v = hjb.critic_value_np(c, X)
        errs[n] = float(np.sqrt(((v - Vt) ** 2).mean()))
        a.pcolormesh(th, w, v.reshape(th.shape), cmap="Blues", vmin=0, vmax=vmax, shading="auto")
        a.set_title(f"{n}\nRMSE vs DP = {errs[n]:.2f}", fontsize=8)
    for a in axs:
        a.set_xlabel("angle [rad] (0 = up)"); a.grid(False)
    axs[0].set_ylabel("angular rate [rad/s]")
    fig.colorbar(im, ax=axs, shrink=0.8, label="cost-to-go V(x)")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return errs


def robustness(s_name, pols, names, path_json, path_png):
    """Success rate vs. motor weakness and vs. pole-mass mismatch (controllers designed for nominal)."""
    base = make_system(s_name)
    torque_key = "tau_stall" if s_name == "pendulum" else "F_stall"
    mass_key = "m"
    scales = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    masses = [0.7, 1.0, 1.3, 1.6]
    R = {"motor_scale": scales, "mass_scale": masses, "motor": {}, "mass": {}}
    X0 = test_initial_states(base)
    for n in names:
        if n not in pols:
            continue
        R["motor"][n] = []; R["mass"][n] = []
        for sc in scales:
            sp = make_system(s_name, **{torque_key: getattr(base, torque_key) * sc})
            if hasattr(pols[n], "reset"):
                pols[n].U = None
            m, _ = evaluate(sp, pols[n], X0=X0)
            R["motor"][n].append(m["success_rate"])
        for ms in masses:
            sp = make_system(s_name, **{mass_key: getattr(base, mass_key) * ms})
            if hasattr(pols[n], "reset"):
                pols[n].U = None
            m, _ = evaluate(sp, pols[n], X0=X0)
            R["mass"][n].append(m["success_rate"])
        print(f"  robustness {n}: motor {R['motor'][n]}  mass {R['mass'][n]}", flush=True)
    json.dump(R, open(path_json, "w"), indent=1)
    plot_robustness(R, path_png)
    return R


def plot_robustness(R, path_png):
    """Success-rate grid (method x perturbation). Most methods sit at exactly 100 %, so lines would hide each other."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("succ", ["#e34948", "#f3efe6", "#1baf7a"])
    names = list(R["motor"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.42 * len(names) + 1.6),
                            gridspec_kw={"width_ratios": [len(R["motor_scale"]), len(R["mass_scale"])]})
    for ax, key, xs, title, xl in ((axs[0], "motor", R["motor_scale"], "How weak can the motor get?", "motor strength / nominal"),
                                   (axs[1], "mass", R["mass_scale"], "Unmodelled pole mass", "pole mass / nominal")):
        Z = np.array([R[key][n] for n in names]) * 100
        ax.imshow(Z, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=8, color="#0b0b0b")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}x" for x in xs])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names if key == "motor" else [])
        ax.set_xlabel(xl); ax.set_title(title, loc="left", fontsize=10); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if key == "motor" and 1.0 in xs:
            ax.axvline(xs.index(1.0), color="#52514e", lw=0)  # nominal column labelled below
    fig.suptitle("Success rate [%] over 20 starts; controllers designed for 1x (nominal)", x=0.01, ha="left", fontsize=9, color="#52514e")
    fig.tight_layout(); fig.savefig(path_png); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pendulum")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    a = ap.parse_args()
    s = make_system(a.system)
    root = os.path.join("results", a.system)
    fdir = os.path.join(root, "figures"); os.makedirs(fdir, exist_ok=True)
    M = gather_metrics(root)
    md, order = table(M, s)
    pols = load_policies(s, root)
    print(md)
    fig_costs(M, order, os.path.join(fdir, "cost_comparison.png"),
              f"{a.system}: swing-up + balance with a weak motor, every method on one cost")
    fig_learning(root, M, os.path.join(fdir, "rl_learning_curves.png"), s)
    errs = fig_value(s, root, pols, os.path.join(fdir, "value_functions.png")) if a.system == "pendulum" else None

    # nominal trajectories (hanging at rest)
    x0 = s.down_state()[None]
    ok = lambda k: k in pols and M.get(k, {}).get("success_rate", 0) >= 0.5
    # on each plant show the better of the anchored / unanchored PIRL critic
    pirl = max([k for k in ("PIRL", "PIRL (no anchor)") if ok(k)], key=lambda k: M[k]["success_rate"], default=None)
    show = [k for k in ["DP", pirl, "XTFC-PIRL", "Energy+LQR", "MPPI-MPC", "SAC", "DDPG", "PPO"] if k and ok(k)]
    trajs = {}
    for k in show:
        if hasattr(pols[k], "reset"):
            pols[k].U = None
        _, tr = evaluate(s, pols[k], X0=x0)
        trajs[k] = (tr["x"][0], tr["u"][0])
    fig_timeseries(s, {k: trajs[k] for k in show[:6]}, os.path.join(fdir, "swingup_timeseries.png"))
    if not a.no_gif:
        gdir = os.path.join(root, "gifs"); os.makedirs(gdir, exist_ok=True)
        T = 8.0 if a.system == "pendulum" else 10.0
        animate(s, [trajs[k] for k in show], show, os.path.join(gdir, "comparison.gif"), max_s=T,
                suptitle=f"{a.system} swing-up with a weak motor - same start, different controllers")
        best = pirl if pirl in trajs else show[0]
        animate(s, [trajs[best]], [f"{best}: swing-up + balance"], os.path.join(gdir, f"{fname(best)}.gif"), max_s=T)
        for k in show:
            if k != best:
                animate(s, [trajs[k]], [k], os.path.join(gdir, f"{fname(k)}.gif"), max_s=T)
    R = None
    if not a.no_robust:
        names = [k for k in ["Energy+LQR", "MPPI-MPC", "DP", pirl, "XTFC-PIRL", "XTFC-PI", "SAC", "DDPG", "TD3", "PPO"] if k and ok(k)]
        R = robustness(a.system, pols, names, os.path.join(root, "robustness.json"), os.path.join(fdir, "robustness.png"))
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write(f"# {a.system} results\n\n{md}\n")
        if errs:
            f.write("\n## Critic accuracy vs DP ground truth (RMSE of V over |w|<8)\n\n")
            for k, v in errs.items():
                f.write(f"- {k}: {v:.3f}\n")
    json.dump({"metrics": M, "value_rmse": errs}, open(os.path.join(root, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

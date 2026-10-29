"""Matplotlib animations of swing-up trajectories (GIF), single or side-by-side."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle


def _draw_setup(ax, sys, title):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10)
    if sys.name == "pendulum":
        L = sys.l
        ax.set_xlim(-1.35 * L, 1.35 * L); ax.set_ylim(-1.35 * L, 1.35 * L)
        ax.add_patch(Circle((0, 0), 0.08 * L, color="0.3", zorder=3))  # motor
        rod, = ax.plot([], [], lw=3, color="#2b6cb0", solid_capstyle="round")
        bob = Circle((0, 0), 0.07 * L, color="#c53030", zorder=4); ax.add_patch(bob)
        bar = Rectangle((0, -1.3 * L), 0, 0.06 * L, color="#dd6b20"); ax.add_patch(bar)
        ax.plot([0, 0], [-1.33 * L, -1.2 * L], color="k", lw=0.8)
        ax.plot([0, 0], [0, 1.3 * L], ls=":", color="0.7", lw=0.8)
        return dict(rod=rod, bob=bob, bar=bar, L=L)
    L2 = 2 * sys.l
    ax.set_xlim(-sys.x_lim - 0.15, sys.x_lim + 0.15); ax.set_ylim(-L2 - 0.1, L2 + 0.15)
    ax.plot([-sys.x_lim - 0.06, sys.x_lim + 0.06], [0, 0], color="0.4", lw=2)
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * (sys.x_lim + 0.06) - 0.01, -0.06), 0.02, 0.12, color="0.2"))
    cart = Rectangle((0, 0), 0.12, 0.06, color="0.3", zorder=3); ax.add_patch(cart)
    rod, = ax.plot([], [], lw=3, color="#2b6cb0", solid_capstyle="round", zorder=4)
    bob = Circle((0, 0), 0.025, color="#c53030", zorder=5); ax.add_patch(bob)
    bar = Rectangle((0, -L2 - 0.05), 0, 0.03, color="#dd6b20"); ax.add_patch(bar)
    ax.plot([0, 0], [-L2 - 0.06, -L2 - 0.01], color="k", lw=0.8)
    return dict(rod=rod, bob=bob, bar=bar, cart=cart, L=L2)


def _update(art, sys, x, u):
    if sys.name == "pendulum":
        th = x[0]; L = art["L"]
        px, py = L * math.sin(th), L * math.cos(th)
        art["rod"].set_data([0, px], [0, py]); art["bob"].center = (px, py)
        art["bar"].set_width(1.1 * L * u)
    else:
        p, th = x[0], x[2]; L = art["L"]
        art["cart"].set_xy((p - 0.06, -0.03))
        px, py = p + L * math.sin(th), L * math.cos(th)
        art["rod"].set_data([p, px], [0, py]); art["bob"].center = (px, py)
        art["bar"].set_width(0.35 * u)


def animate(sys, trajs, titles, path, fps=25, max_s=None, suptitle=None):
    """trajs: list of (xs (S+1,n), us (S,1)).  Writes a GIF with one panel per controller."""
    k = len(trajs)
    cols = min(k, 4); rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 0.4), squeeze=False)
    arts = []
    for i, ax in enumerate(axes.ravel()):
        if i < k:
            arts.append(_draw_setup(ax, sys, titles[i]))
        else:
            ax.axis("off")
    txt = fig.text(0.5, 0.01, "", ha="center", fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    stride = max(1, int(round(1 / (fps * sys.dt))))
    S = min(len(t[0]) for t in trajs)
    if max_s:
        S = min(S, int(max_s / sys.dt))
    frames = range(0, S - 1, stride)

    def step(j):
        for a, (xs, us) in zip(arts, trajs):
            _update(a, sys, xs[j], float(us[min(j, len(us) - 1), 0]))
        txt.set_text(f"t = {j * sys.dt:4.1f} s     orange bar = motor command u in [-1, 1]")
        return []

    fig.tight_layout(rect=(0, 0.04, 1, 0.95 if suptitle else 1))
    anim = FuncAnimation(fig, step, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)

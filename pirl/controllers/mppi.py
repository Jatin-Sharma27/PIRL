"""GPU sampling-based nonlinear MPC (MPPI, Williams et al. 2017).

This is the "MPC / TrajOpt as a benchmark" item from the plan.  It runs many
closed loops at once: every state in the batch gets its own K sampled control
sequences, and all of them roll out in parallel on the GPU.
"""
import math

import numpy as np
import torch

from .classical import Policy
from ..systems import wrap, angle_index


class MPPI(Policy):
    name = "MPPI-MPC"

    def __init__(self, sys, horizon_s=1.6, K=1024, sigma=0.4, lam=None, device=None,
                 terminal_w=3.0, iters=1, seed=0):
        self.sys = sys
        self.H = int(round(horizon_s / sys.dt))
        self.K, self.sigma, self.iters = K, sigma, iters
        self.lam = lam if lam is not None else 0.05
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.terminal_w = terminal_w
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.disc = torch.exp(-sys.rho * sys.dt * torch.arange(self.H, device=self.dev))
        self.U = None

    def reset(self, N=1):
        self.U = torch.zeros(N, self.H, device=self.dev)

    def _step(self, X, u):
        sys = self.sys
        h = sys.dt / max(1, sys.substeps // 2)
        for _ in range(max(1, sys.substeps // 2)):
            k1 = sys.xdot(X, u)
            k2 = sys.xdot(X + 0.5 * h * k1, u)
            k3 = sys.xdot(X + 0.5 * h * k2, u)
            k4 = sys.xdot(X + h * k3, u)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def _extra_cost(self, X):
        if self.sys.name == "cartpole":
            over = torch.relu(X[..., 0].abs() - 0.85 * self.sys.x_lim)
            return 2000.0 * over ** 2 + 50.0 * (X[..., 0].abs() > self.sys.x_lim).float()
        return torch.zeros_like(X[..., 0])

    def _update(self, X, U, eps):
        """One MPPI iteration: sample, roll out, reweight.  Pure tensor ops -> CUDA-graphable."""
        sys = self.sys
        N = X.shape[0]
        Us = (U[:, None] + eps).clamp(-1, 1)
        x = X[:, None].expand(N, self.K, sys.n).contiguous()
        S = torch.zeros(N, self.K, device=self.dev)
        for k in range(self.H):
            u = Us[:, :, k:k + 1]
            S = S + self.disc[k] * (sys.cost(x, u) + self._extra_cost(x)) * sys.dt
            x = self._step(x, u)
        S = S + self.disc[-1] * self.terminal_w * (sys.state_cost(x) + self._extra_cost(x))
        S = S - S.min(1, keepdim=True).values
        w = torch.softmax(-S / self.lam, dim=1)
        return (w[:, :, None] * Us).sum(1)

    def _build_graph(self, N):
        # Windows/WDDM kernel-launch overhead dominates the ~6k tiny kernels of a
        # rollout; capturing them once in a CUDA graph gives a large speed-up.
        self._gX = torch.zeros(N, self.sys.n, device=self.dev)
        self._gU = torch.zeros(N, self.H, device=self.dev)
        self._gE = torch.zeros(N, self.K, self.H, device=self.dev)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._update(self._gX, self._gU, self._gE)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._gOut = self._update(self._gX, self._gU, self._gE)
        self._gN = N

    @torch.no_grad()
    def plan(self, X):
        N = X.shape[0]
        if self.U is None or self.U.shape[0] != N:
            self.reset(N)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        use_graph = self.dev.startswith("cuda")
        if use_graph and getattr(self, "_gN", None) != N:
            self._build_graph(N)
        for _ in range(self.iters):
            eps = torch.randn(N, self.K, self.H, generator=self.gen, device=self.dev) * self.sigma
            eps[:, 0] = 0.0  # keep the current mean sequence as one sample
            if use_graph:
                self._gX.copy_(X); self._gU.copy_(self.U); self._gE.copy_(eps)
                self._graph.replay()
                self.U = self._gOut.clone()
            else:
                self.U = self._update(X, self.U, eps)
        return self.U

    def act(self, X):
        U = self.plan(X)
        u0 = U[:, 0].clone()
        self.U = torch.roll(U, -1, dims=1); self.U[:, -1] = 0.0
        return u0.cpu().numpy()[:, None].astype(np.float64)

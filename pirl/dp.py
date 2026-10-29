"""Offline dynamic programming (semi-Lagrangian value iteration) on a dense grid.

The pendulum has a 2-D state, so it can be solved almost exactly on the GPU.
We use this as GROUND TRUTH for the value function V*(x) when scoring the
PINN / X-TFC / PIRL critics.  This is the "Offline DP and PI" item on slide 4.
"""
import math

import numpy as np
import torch

from .controllers.classical import Policy
from .systems import rk4_step


class GridValue:
    def __init__(self, sys, n_th=201, n_w=201, n_a=41, device=None):
        self.sys = sys
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.th = torch.linspace(-math.pi, math.pi, n_th + 1, device=self.dev)[:-1]
        self.w = torch.linspace(-sys.w_max, sys.w_max, n_w, device=self.dev)
        self.a = torch.linspace(-1, 1, n_a, device=self.dev)
        self.dth = 2 * math.pi / n_th
        self.dw = 2 * sys.w_max / (n_w - 1)
        self.gamma = math.exp(-sys.rho * sys.dt)
        self.V = torch.zeros(n_th, n_w, device=self.dev)

    def interp(self, V, x):
        """Bilinear interpolation, periodic in theta, clamped in omega."""
        nth, nw = V.shape
        ti = (x[..., 0] + math.pi) / self.dth
        wi = ((x[..., 1] + self.sys.w_max) / self.dw).clamp(0, nw - 1 - 1e-4)
        t0 = torch.floor(ti); ft = ti - t0; t0 = t0.long() % nth; t1 = (t0 + 1) % nth
        w0 = torch.floor(wi); fw = wi - w0; w0 = w0.long(); w1 = (w0 + 1).clamp(max=nw - 1)
        return ((1 - ft) * (1 - fw) * V[t0, w0] + ft * (1 - fw) * V[t1, w0]
                + (1 - ft) * fw * V[t0, w1] + ft * fw * V[t1, w1])

    @torch.no_grad()
    def solve(self, iters=4000, tol=1e-6, verbose=True):
        sys = self.sys
        TH, W, A = torch.meshgrid(self.th, self.w, self.a, indexing="ij")
        X = torch.stack([TH, W], -1)
        U = A[..., None]
        self.Xn = rk4_step(sys, X, U)
        # leaving the omega box: extrapolate cost (rarely matters, w_max >> reachable speeds)
        self.C = sys.cost(X, U) * sys.dt
        V = self.V
        for it in range(iters):
            Q = self.C + self.gamma * self.interp(V, self.Xn)
            Vn = Q.min(-1).values
            err = (Vn - V).abs().max().item()
            V = Vn
            if verbose and it % 500 == 0:
                print(f"  VI iter {it}: |dV|={err:.2e}")
            if err < tol:
                break
        self.V = V
        return V

    def value(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        return self.interp(self.V, X)


class DPPolicy(Policy):
    name = "DP (ground truth)"

    def __init__(self, grid, n_a=81):
        self.g = grid
        self.a = torch.linspace(-1, 1, n_a, device=grid.dev)

    @torch.no_grad()
    def act(self, X):
        g, sys = self.g, self.g.sys
        X = torch.as_tensor(X, dtype=torch.float32, device=g.dev)
        N = X.shape[0]
        Xa = X[:, None].expand(N, len(self.a), 2)
        U = self.a[None, :, None].expand(N, len(self.a), 1)
        Q = sys.cost(Xa, U) * sys.dt + g.gamma * g.interp(g.V, rk4_step(sys, Xa, U))
        return self.a[Q.argmin(1)].cpu().numpy()[:, None].astype(np.float64)

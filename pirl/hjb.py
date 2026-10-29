"""Physics-informed solvers for the discounted, input-constrained HJB equation

    rho V(x) = min_{|u|<=1}  l(x,u) + dV/dx . (f(x) + B(x) u)

Because the system is control-affine and the input cost is the separable
quadratic r u^2, the minimiser is known in closed form (Srinivasa & Kumar 2025):

    u*(x) = clip( -(1/2r) B(x)^T dV/dx , -1, 1 )

Every critic here uses the Theory-of-Functional-Connections constrained expression

    V(x) = eta(phi(x)) - eta(phi(0)),     phi(x) = (cos th, sin th, w/8, ...)

so V(0) = 0 and 2*pi periodicity in the angle hold exactly by construction,
for any network weights.  The network only has to learn the rest.

LQR-anchored constrained expression (anchor=True, our extension):

    V(x) = psi^T P psi + |phi(x)-phi(0)|^2 (eta(phi(x)) - eta(phi(0))),   psi = (sin th, 1-cos th, w, ...)

where P solves the discounted Riccati equation of the linearisation (optionally
saturated as kappa*tanh(psi^T P psi / kappa), which changes nothing up to 2nd
order but keeps the anchor bounded where the true discounted V is bounded).  This
pins V(0)=0, grad V(0)=0 AND Hess V(0)=2P for any weights, so the greedy
policy is exactly the LQR law near the top.  Without the anchor, a smooth
critic is slightly too flat there: the policy then balances gravity at an
offset angle (~0.2 rad on the weak-motor pendulum) instead of at vertical.

Methods
-------
PINN-HJB  : minimise the HJB residual directly (Adam).
PINN-AC   : model-based actor-critic policy iteration.  The critic solves the
            policy-evaluation PDE; the actor minimises the Hamiltonian.
XTFC-PI   : X-TFC/ELM critic.  Policy evaluation is LINEAR in the output
            weights, so each PI step is one least-squares solve
            (Newton-Kleinman-style PI instead of the paper's Adam + L-BFGS).
PIRL      : integral-RL policy iteration (Vrabie & Lewis; Wang & Wu 2024).
            The critic fits the integral Bellman equation
                V(x_t) = int_t^{t+T} e^{-rho s} l ds + e^{-rho T} V(x_{t+T})
            on simulator data; this needs no knowledge of the drift f.
            It adds a physics-informed PDE residual at collocation points as
            a regulariser.  Setting lambda_phys = 0 gives pure model-free IRL.
XTFC-PIRL : the same integral-Bellman + physics rows, solved in closed form
            with the ELM critic.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .controllers.classical import Policy
from .systems import rk4_step, wrap, angle_index

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def T(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEV)


def sample_domain(sys, n, rng):
    return T(sys.sample_states(n, rng))


def origin(sys):
    return torch.zeros(1, sys.n, device=DEV)


def greedy_u(sys, x, dVdx):
    Bx = sys.B(x)[..., 0]
    return (-(Bx * dVdx).sum(-1, keepdim=True) / (2 * sys.r)).clamp(-1, 1)


def pe_residual(sys, x, V, dVdx, u):
    """Policy-evaluation / HJB residual  rho V - l - dV/dx . xdot."""
    return sys.rho * V - sys.cost(x, u) - (dVdx * sys.xdot(x, u)).sum(-1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def mlp(n_in, n_out, hidden=128, layers=3, act=Swish):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


# --------------------------------------------------------------------------
# critics
# --------------------------------------------------------------------------
def local_P(sys):
    """Discounted LQR value matrix at the upright equilibrium, embedded in psi-coordinates."""
    from scipy.linalg import solve_continuous_are
    A, B = sys.linearize()
    P = solve_continuous_are(A - 0.5 * sys.rho * np.eye(sys.n), B, sys.Q_cost(), sys.R())
    m = len(sys.anchor_coords(np.zeros(sys.n)))
    Pb = np.zeros((m, m))
    idx = np.array(sys.anchor_lin_idx)
    Pb[np.ix_(idx, idx)] = P
    return Pb


def anchor_quadratic(sys, x, Pb):
    psi = sys.anchor_coords(x)
    q = ((psi @ Pb) * psi).sum(-1)
    k = getattr(sys, "anchor_kappa", None)
    return q if k is None else k * torch.tanh(q / k)


class TFCCritic(nn.Module):
    """V(x) = eta(phi(x)) - eta(phi(0))   [+ LQR anchor, see module docstring]."""

    def __init__(self, sys, hidden=128, layers=3, anchor=False):
        super().__init__()
        self.sys = sys
        self.anchor = anchor
        nf = len(sys.features(np.zeros(sys.n)))
        self.eta = mlp(nf, 1, hidden, layers)
        if anchor:
            self.register_buffer("Pb", torch.tensor(local_P(sys), dtype=torch.float32))

    def forward(self, x):
        phi = self.sys.features(x)
        phi0 = self.sys.features(origin(self.sys))
        n = (self.eta(phi) - self.eta(phi0))[..., 0]
        if not self.anchor:
            return n
        return anchor_quadratic(self.sys, x, self.Pb) + ((phi - phi0) ** 2).sum(-1) * n

    def v_and_grad(self, x, create_graph=True):
        x = x.detach().requires_grad_(True)
        V = self(x)
        g, = torch.autograd.grad(V.sum(), x, create_graph=create_graph)
        return x, V, g


class Actor(nn.Module):
    def __init__(self, sys, hidden=128, layers=3):
        super().__init__()
        self.sys = sys
        nf = len(sys.features(np.zeros(sys.n)))
        self.net = mlp(nf, 1, hidden, layers)

    def forward(self, x):
        return torch.tanh(self.net(self.sys.features(x)))


class CriticPolicy(Policy):
    """Greedy policy  u = clip(-(1/2r) B^T dV/dx)  derived from any differentiable critic."""

    def __init__(self, sys, critic, name):
        self.sys, self.critic, self.name = sys, critic, name

    def act(self, X):
        x = T(X)
        with torch.enable_grad():
            x, V, g = self.critic.v_and_grad(x, create_graph=False)
        return greedy_u(self.sys, x, g).detach().cpu().numpy().astype(np.float64)


class ActorPolicy(Policy):
    def __init__(self, actor, name):
        self.actor, self.name = actor, name

    @torch.no_grad()
    def act(self, X):
        return self.actor(T(X)).cpu().numpy().astype(np.float64)


def torch_policy(policy):
    """Make any Policy callable on torch tensors (for evaluating it at collocation points)."""
    if isinstance(policy, ActorPolicy):
        return lambda x: policy.actor(x)
    return lambda x: T(np.clip(policy(x.detach().cpu().numpy().astype(np.float64)), -1, 1))


def behaviour_clone(sys, teacher, n=200_000, iters=3000, seed=0, log=print):
    rng = np.random.default_rng(seed)
    actor = Actor(sys).to(DEV)
    opt = torch.optim.Adam(actor.parameters(), 1e-3)
    X = sys.sample_states(n, rng)
    Y = T(teacher(X))
    X = T(X)
    for it in range(iters):
        idx = torch.randint(0, n, (4096,), device=DEV)
        loss = ((actor(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  BC of {teacher.name}: mse={loss.item():.4f}")
    return actor


# --------------------------------------------------------------------------
# 1) PINN-HJB: direct residual minimisation
# --------------------------------------------------------------------------
def train_pinn_hjb(sys, iters=20000, batch=4096, lr=1e-3, seed=0, anchor=False, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    V = TFCCritic(sys, anchor=anchor).to(DEV)
    opt = torch.optim.Adam(V.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    hist = []
    for it in range(iters):
        x, v, g = V.v_and_grad(sample_domain(sys, batch, rng))
        u = greedy_u(sys, x, g)
        loss = (pe_residual(sys, x, v, g, u) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(V.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 1000 == 0:
            hist.append((it, loss.item())); log(f"  PINN-HJB it {it}: residual^2={loss.item():.3e}")
    return V, hist


# --------------------------------------------------------------------------
# 2) PINN-AC: model-based actor-critic policy iteration
# --------------------------------------------------------------------------
def train_pinn_ac(sys, init_policy, n_pi=12, critic_iters=1500, actor_iters=500, batch=4096,
                  seed=0, anchor=False, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    actor = behaviour_clone(sys, init_policy, seed=seed, log=log)  # admissible start
    V = TFCCritic(sys, anchor=anchor).to(DEV)
    optV = torch.optim.Adam(V.parameters(), 1e-3)
    optA = torch.optim.Adam(actor.parameters(), 3e-4)
    hist = []
    for k in range(n_pi):
        for it in range(critic_iters):          # policy evaluation (linear PDE)
            x, v, g = V.v_and_grad(sample_domain(sys, batch, rng))
            with torch.no_grad():
                u = actor(x)
            loss = (pe_residual(sys, x, v, g, u) ** 2).mean()
            optV.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(V.parameters(), 1.0); optV.step()
        for it in range(actor_iters):           # policy improvement: argmin Hamiltonian
            x, v, g = V.v_and_grad(sample_domain(sys, batch, rng), create_graph=False)
            u = actor(x)
            Ham = sys.cost(x, u) + (g.detach() * sys.xdot(x, u)).sum(-1)
            optA.zero_grad(); Ham.mean().backward(); optA.step()
        hist.append((k, loss.item(), Ham.mean().item()))
        log(f"  PINN-AC PI {k}: PE residual^2={loss.item():.3e}  mean H={Ham.mean().item():.3f}")
    return V, actor, hist


# --------------------------------------------------------------------------
# data: integral-reinforcement segments from the simulator (black box)
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_segments(sys, policy_t, n_starts=4096, horizon_s=1.0, seg_s=0.2, seed=0,
                     start_states=None):
    """Roll out the current policy and cut the trajectories into integral-RL tuples
    (x_t, x_{t+T}, int_t^{t+T} e^{-rho s} l ds).  The simulator is used only as a black box."""
    rng = np.random.default_rng(seed)
    X = T(start_states if start_states is not None else sys.sample_states(n_starts, rng))
    steps, seg = int(horizon_s / sys.dt), int(seg_s / sys.dt)
    ai = angle_index(sys)
    xs, cs = [X], []
    for k in range(steps):
        u = policy_t(X)
        cs.append(sys.cost(X, u) * sys.dt * math.exp(0))
        X = rk4_step(sys, X, u)
        X = torch.cat([X[:, :ai], wrap(X[:, ai:ai + 1]), X[:, ai + 1:]], 1)
        xs.append(X)
    xs = torch.stack(xs, 1); cs = torch.stack(cs, 1)
    gam = torch.exp(-sys.rho * sys.dt * torch.arange(seg, device=DEV))
    x0, x1, c = [], [], []
    for s in range(0, steps - seg + 1, max(1, seg // 2)):
        x0.append(xs[:, s]); x1.append(xs[:, s + seg]); c.append((cs[:, s:s + seg] * gam).sum(1))
    x0, x1, c = torch.cat(x0), torch.cat(x1), torch.cat(c)
    keep = ~sys.out_of_bounds(x1) & (x1[:, ai + 1].abs() < 1.5 * sys.w_max)
    return x0[keep], x1[keep], c[keep], math.exp(-sys.rho * seg * sys.dt)


# --------------------------------------------------------------------------
# 3) PIRL: integral-RL policy iteration with a physics-informed regulariser
# --------------------------------------------------------------------------
def train_pirl(sys, init_policy, n_pi=10, iters=2000, batch=4096, lam_phys=0.1, seed=0,
               n_starts=4096, anchor=False, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    V = TFCCritic(sys, anchor=anchor).to(DEV)
    opt = torch.optim.Adam(V.parameters(), 1e-3)
    pol_t = torch_policy(init_policy)
    hist = []
    for k in range(n_pi):
        x0, x1, c, gT = collect_segments(sys, pol_t, n_starts=n_starts, seed=seed + k)
        prev = copy.deepcopy(V)
        for it in range(iters):
            idx = torch.randint(0, len(c), (batch,), device=DEV)
            l_irl = ((V(x0[idx]) - gT * V(x1[idx]) - c[idx]) ** 2).mean()
            loss = l_irl
            if lam_phys > 0:
                x, v, g = V.v_and_grad(sample_domain(sys, batch, rng))
                with torch.no_grad():
                    u = pol_t(x)
                l_phys = (pe_residual(sys, x, v, g, u) ** 2).mean()
                loss = loss + lam_phys * l_phys
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(V.parameters(), 1.0); opt.step()
        # policy improvement only needs B(x) (the motor constant / inertia), not f(x)
        frozen = copy.deepcopy(V)
        def pol_t(x, frozen=frozen):
            with torch.enable_grad():
                x, v, g = frozen.v_and_grad(x, create_graph=False)
            return greedy_u(sys, x, g).detach()
        hist.append((k, l_irl.item(), len(c)))
        log(f"  PIRL(lam={lam_phys}) PI {k}: IRL residual^2={l_irl.item():.3e}  samples={len(c)}")
    return V, hist


# --------------------------------------------------------------------------
# 4) X-TFC (ELM) critic with least-squares policy iteration
# --------------------------------------------------------------------------
class ELMCritic:
    """X-TFC critic.  V(x) = Vq(x) + H(x) beta, with W, b random and fixed:
         plain   : Vq = 0,            H = tanh(W phi(x) + b) - tanh(W phi(0) + b)
         anchored: Vq = psi^T P psi,  H = |phi(x)-phi(0)|^2 [tanh(W phi(x) + b) - tanh(W phi(0) + b)]
       Either way V is LINEAR in beta, so policy evaluation is one least-squares solve."""

    def __init__(self, sys, L=1500, scale=2.0, seed=0, anchor=False):
        g = torch.Generator(device="cpu").manual_seed(seed)
        nf = len(sys.features(np.zeros(sys.n)))
        self.sys, self.anchor = sys, anchor
        self.W = ((torch.rand(L, nf, generator=g) * 2 - 1) * scale).to(DEV, torch.float64)
        self.b = ((torch.rand(L, generator=g) * 2 - 1) * scale).to(DEV, torch.float64)
        self.beta = torch.zeros(L, device=DEV, dtype=torch.float64)
        x0 = torch.zeros(1, sys.n, device=DEV, dtype=torch.float64)
        self.phi0 = sys.features(x0)
        self.h0 = torch.tanh(self.phi0 @ self.W.T + self.b)
        if anchor:
            self.Pb = torch.tensor(local_P(sys), device=DEV, dtype=torch.float64)

    def H(self, x):
        phi = self.sys.features(x)
        h = torch.tanh(phi @ self.W.T + self.b) - self.h0
        if self.anchor:
            h = h * ((phi - self.phi0) ** 2).sum(-1, keepdim=True)
        return h

    def Vq(self, x):
        if not self.anchor:
            return torch.zeros_like(x[..., 0])
        return anchor_quadratic(self.sys, x, self.Pb)

    def dH_along(self, x, v):
        """(dH/dx) v  via a forward-mode JVP."""
        return torch.func.jvp(self.H, (x,), (v,))[1]

    def dVq_along(self, x, v):
        return torch.func.jvp(self.Vq, (x,), (v,))[1]

    def value(self, x):
        return self.Vq(x) + self.H(x) @ self.beta

    def grad_dot(self, x, v):
        return self.dVq_along(x, v) + self.dH_along(x, v) @ self.beta

    def greedy(self, x):
        Bx = self.sys.B(x)[..., 0]
        return (-self.grad_dot(x, Bx) / (2 * self.sys.r)).clamp(-1, 1)[:, None]

    # make it look like a TFCCritic for evaluation utilities
    def v_and_grad(self, x, create_graph=False):
        x = x.to(torch.float64)
        E = torch.eye(self.sys.n, device=DEV, dtype=torch.float64)
        g = torch.stack([self.grad_dot(x, E[i].expand_as(x)) for i in range(self.sys.n)], -1)
        return x, self.value(x), g


class NormalEq:
    """Accumulates A^T A, A^T y, y^T y chunk by chunk: memory O(L^2), independent of #rows."""

    def __init__(self, L):
        self.AtA = torch.zeros(L, L, device=DEV, dtype=torch.float64)
        self.Aty = torch.zeros(L, device=DEV, dtype=torch.float64)
        self.yty, self.n = 0.0, 0

    def add(self, A, y):
        self.AtA += A.T @ A; self.Aty += A.T @ y
        self.yty += float(y @ y); self.n += len(y)

    def solve(self, alpha):
        M = self.AtA.clone()
        M.diagonal().add_(alpha * M.diagonal().mean())
        beta = torch.linalg.solve(M, self.Aty)
        sse = self.yty - 2 * float(beta @ self.Aty) + float(beta @ (self.AtA @ beta))
        return beta, math.sqrt(max(sse, 0.0) / self.n)


def train_xtfc(sys, init_policy, n_pi=12, n_col=30000, L=1500, scale=2.0, alpha=1e-6,
               irl=False, lam_phys=1.0, n_starts=4096, seed=0, anchor=False, chunk=8192, log=print):
    """irl=False: model-based X-TFC PI (physics rows only).
       irl=True : X-TFC-PIRL, integral-Bellman data rows + lam_phys * physics rows."""
    rng = np.random.default_rng(seed)
    elm = ELMCritic(sys, L=L, scale=scale, seed=seed, anchor=anchor)
    pol_t = torch_policy(init_policy)
    hist = []
    for k in range(n_pi):
        ne = NormalEq(L)
        if not irl or lam_phys > 0:
            w = 1.0 if not irl else math.sqrt(lam_phys)
            Xc = T(sys.sample_states(n_col, rng), torch.float64)
            for x in Xc.split(chunk):
                u = pol_t(x.float()).to(torch.float64)
                xd = sys.xdot(x, u)
                # rho V - dV/dx . xdot = l,  with V = Vq + H beta
                A = sys.rho * elm.H(x) - elm.dH_along(x, xd)
                y = sys.cost(x, u) - (sys.rho * elm.Vq(x) - elm.dVq_along(x, xd))
                ne.add(w * A, w * y)
        if irl:
            x0, x1, c, gT = collect_segments(sys, pol_t, n_starts=n_starts, seed=seed + k)
            sel = torch.randperm(len(c), device=DEV)[:60000]
            sc = 1.0 / (sys.dt * 10)  # put data rows on the same per-second scale as the PDE rows
            for idx in sel.split(chunk):
                a0, a1, cc = x0[idx].double(), x1[idx].double(), c[idx].double()
                ne.add(sc * (elm.H(a0) - gT * elm.H(a1)), sc * (cc - (elm.Vq(a0) - gT * elm.Vq(a1))))
        beta, res = ne.solve(alpha)
        dbeta = (beta - elm.beta).norm().item() / (beta.norm().item() + 1e-12)
        elm.beta = beta
        def pol_t(x, elm=elm):
            return elm.greedy(x.to(torch.float64)).float()
        hist.append((k, res, dbeta))
        log(f"  XTFC{'-PIRL' if irl else '-PI'} {k}: rms residual={res:.3e}  |dbeta|/|beta|={dbeta:.3e}")
    return elm, hist


class ELMPolicy(Policy):
    def __init__(self, elm, name):
        self.elm, self.name = elm, name

    @torch.no_grad()
    def act(self, X):
        return self.elm.greedy(T(X, torch.float64)).cpu().numpy()


# --------------------------------------------------------------------------
def critic_value_np(critic, X):
    x = T(X)
    if isinstance(critic, ELMCritic):
        with torch.no_grad():
            return critic.value(x.double()).cpu().numpy()
    with torch.no_grad():
        return critic(x).cpu().numpy()

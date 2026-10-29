"""Physics of the two plants, written once and usable from NumPy *and* PyTorch.

Both plants are driven by a weak, geared DC motor.  The control input is the
normalised armature voltage u in [-1, 1].  Motor torque is
    tau = tau_stall * u - (tau_stall / w_noload) * w_motor
so the motor gets weaker as it spins up (back-EMF).  The back-EMF term only
depends on the state, so it goes into the drift f(x).  The system stays
control-affine:  x_dot = f(x) + B(x) u.  That is the form the HJB / IRL methods need.

Angle convention: theta = 0 is UPRIGHT and theta = +-pi is hanging down.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import torch


def _lib(x):
    return torch if isinstance(x, torch.Tensor) else np


def wrap(th):
    """Wrap angle to [-pi, pi)."""
    lib = _lib(th)
    return lib.remainder(th + math.pi, 2 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Pendulum on a motor shaft
# --------------------------------------------------------------------------
@dataclass
class Pendulum:
    m: float = 0.20          # bob mass [kg]
    l: float = 0.30          # rod length [m]
    g: float = 9.81
    J_rotor: float = 0.002   # reflected rotor + gearbox inertia [kg m^2]
    tau_stall: float = 0.25  # motor stall torque at output [N m]  (~42 % of m g l)
    w_noload: float = 20.0   # no-load output speed [rad/s]
    b: float = 0.001         # viscous bearing friction [N m s]
    tau_c: float = 0.005     # Coulomb friction [N m]
    # cost  l(x,u) = q_th (1 - cos th) + q_w w^2 + r u^2
    q_th: float = 1.0
    q_w: float = 0.01
    r: float = 0.05
    rho: float = 0.25        # continuous-time discount [1/s]  (gamma = exp(-rho dt))
    dt: float = 0.02         # control period [s]  (50 Hz)
    substeps: int = 2
    n: int = 2
    m_u: int = 1
    name: str = "pendulum"
    # state-space box used for collocation / grids (theta is periodic).
    # The weak motor can't spin the pendulum much above sqrt(4 m g l / J) ~ 11 rad/s.
    w_max: float = 12.0

    @property
    def J(self):
        return self.m * self.l ** 2 + self.J_rotor

    @property
    def mgl(self):
        return self.m * self.g * self.l

    @property
    def b_eff(self):
        return self.b + self.tau_stall / self.w_noload

    def f(self, x):
        lib = _lib(x)
        th, w = x[..., 0], x[..., 1]
        dw = (self.mgl * lib.sin(th) - self.b_eff * w - self.tau_c * lib.tanh(w / 0.05)) / self.J
        return lib.stack([w, dw], -1)

    def B(self, x):
        lib = _lib(x)
        z = lib.zeros_like(x[..., 0])
        return lib.stack([z, z + self.tau_stall / self.J], -1)[..., None]  # (...,n,1)

    def xdot(self, x, u):
        return self.f(x) + (self.B(x) @ u[..., None])[..., 0]

    def xdot_scalar(self, x, u):
        """Pure-Python fast path for single-environment stepping (gym)."""
        th, w = x
        dw = (self.mgl * math.sin(th) - self.b_eff * w - self.tau_c * math.tanh(w / 0.05)
              + self.tau_stall * u) / self.J
        return (w, dw)

    def cost(self, x, u):
        lib = _lib(x)
        th, w = x[..., 0], x[..., 1]
        return self.q_th * (1 - lib.cos(th)) + self.q_w * w ** 2 + self.r * (u ** 2).sum(-1)

    def state_cost(self, x):
        lib = _lib(x)
        return self.q_th * (1 - lib.cos(x[..., 0])) + self.q_w * x[..., 1] ** 2

    def R(self):
        return np.array([[self.r]])

    def energy(self, x):
        lib = _lib(x)
        return 0.5 * self.J * x[..., 1] ** 2 + self.mgl * (lib.cos(x[..., 0]) - 1.0)  # 0 at upright rest

    def features(self, x):
        """Periodic embedding used by every learned critic/actor."""
        lib = _lib(x)
        th, w = x[..., 0], x[..., 1]
        return lib.stack([lib.cos(th), lib.sin(th), w / 8.0], -1)

    def sample_states(self, n, rng):
        th = rng.uniform(-math.pi, math.pi, n)
        w = rng.uniform(-self.w_max, self.w_max, n)
        return np.stack([th, w], -1)

    def down_state(self):
        return np.array([math.pi, 0.0])

    def is_upright(self, x):
        lib = _lib(x)
        return (lib.abs(wrap(x[..., 0])) < 0.15) & (lib.abs(x[..., 1]) < 1.0)

    def out_of_bounds(self, x):
        lib = _lib(x)
        return lib.zeros_like(x[..., 0]) > 1  # never

    def linearize(self):
        """A, B at the upright equilibrium (theta=0)."""
        A = np.array([[0, 1], [self.mgl / self.J, -(self.b_eff + self.tau_c / 0.05) / self.J]])
        Bm = np.array([[0], [self.tau_stall / self.J]])
        return A, Bm

    def Q_cost(self):
        """Half the Hessian of the state cost at the upright equilibrium."""
        return np.diag([self.q_th / 2, self.q_w])

    def anchor_coords(self, x):
        """psi(x): locally (th, w) to first order, periodic, and non-zero at the bottom."""
        lib = _lib(x)
        th, w = x[..., 0], x[..., 1]
        return lib.stack([lib.sin(th), 1 - lib.cos(th), w], -1)

    anchor_lin_idx = (0, 2)  # entries of psi that equal the state coordinates to 1st order
    anchor_kappa = None      # pendulum: the plain quadratic anchor stays small (< ~1.3) on the box


# --------------------------------------------------------------------------
# Cart-pole on a belt-driven linear rail (Pal & Nestorovic style rig)
# --------------------------------------------------------------------------
@dataclass
class CartPole:
    M: float = 0.50          # cart + reflected belt/pulley mass [kg]
    m: float = 0.15          # pole mass [kg]
    l: float = 0.25          # pivot -> pole COM [m] (pole length 0.5 m)
    g: float = 9.81
    F_stall: float = 4.0     # motor stall force at belt [N]
    v_noload: float = 2.0    # no-load cart speed [m/s]
    b_c: float = 0.1         # cart viscous friction
    b_p: float = 0.0005      # pivot friction
    x_lim: float = 0.40      # half rail length [m]
    q_th: float = 1.0
    q_w: float = 0.01
    q_p: float = 1.0
    q_v: float = 0.02
    r: float = 0.05
    rho: float = 0.25
    dt: float = 0.02
    substeps: int = 4
    n: int = 4
    m_u: int = 1
    name: str = "cartpole"
    v_max: float = 2.5
    w_max: float = 15.0

    @property
    def Ic(self):
        return self.m * (2 * self.l) ** 2 / 12.0

    @property
    def c_eff(self):
        return self.b_c + self.F_stall / self.v_noload

    def _terms(self, x):
        lib = _lib(x)
        p, v, th, w = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        s, c = lib.sin(th), lib.cos(th)
        ml = self.m * self.l
        Jp = self.Ic + self.m * self.l ** 2
        D = (self.M + self.m) * Jp - (ml * c) ** 2
        return lib, p, v, th, w, s, c, ml, Jp, D

    def f(self, x):
        lib, p, v, th, w, s, c, ml, Jp, D = self._terms(x)
        F0 = -self.c_eff * v
        tq = ml * self.g * s - self.b_p * w
        a = (Jp * (F0 + ml * s * w ** 2) - ml * c * tq) / D
        al = ((self.M + self.m) * tq - ml * c * (F0 + ml * s * w ** 2)) / D
        return lib.stack([v, a, w, al], -1)

    def B(self, x):
        lib, p, v, th, w, s, c, ml, Jp, D = self._terms(x)
        z = lib.zeros_like(p)
        return lib.stack([z, self.F_stall * Jp / D, z, -self.F_stall * ml * c / D], -1)[..., None]

    def xdot(self, x, u):
        return self.f(x) + (self.B(x) @ u[..., None])[..., 0]

    def xdot_scalar(self, x, u):
        p, v, th, w = x
        s, c = math.sin(th), math.cos(th)
        ml = self.m * self.l
        Jp = self.Ic + self.m * self.l ** 2
        D = (self.M + self.m) * Jp - (ml * c) ** 2
        F = self.F_stall * u - self.c_eff * v
        tq = ml * self.g * s - self.b_p * w
        a = (Jp * (F + ml * s * w * w) - ml * c * tq) / D
        al = ((self.M + self.m) * tq - ml * c * (F + ml * s * w * w)) / D
        return (v, a, w, al)

    q_wall: float = 200.0    # soft barrier that starts 10 cm before the end stops

    def state_cost(self, x):
        lib = _lib(x)
        over = lib.maximum(lib.abs(x[..., 0]) - (self.x_lim - 0.1), 0 * x[..., 0])
        return (self.q_p * x[..., 0] ** 2 + self.q_v * x[..., 1] ** 2
                + self.q_th * (1 - lib.cos(x[..., 2])) + self.q_w * x[..., 3] ** 2
                + self.q_wall * over ** 2)

    def cost(self, x, u):
        return self.state_cost(x) + self.r * (u ** 2).sum(-1)

    def R(self):
        return np.array([[self.r]])

    def energy(self, x):
        """Pole energy relative to upright rest (cart motion excluded)."""
        lib = _lib(x)
        Jpiv = self.Ic + self.m * self.l ** 2
        return 0.5 * Jpiv * x[..., 3] ** 2 + self.m * self.g * self.l * (lib.cos(x[..., 2]) - 1.0)

    def features(self, x):
        lib = _lib(x)
        p, v, th, w = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        return lib.stack([p / self.x_lim, v / 2.0, lib.cos(th), lib.sin(th), w / 8.0], -1)

    def sample_states(self, n, rng):
        p = rng.uniform(-self.x_lim, self.x_lim, n)
        v = rng.uniform(-self.v_max, self.v_max, n)
        th = rng.uniform(-math.pi, math.pi, n)
        w = rng.uniform(-self.w_max, self.w_max, n)
        return np.stack([p, v, th, w], -1)

    def down_state(self):
        return np.array([0.0, 0.0, math.pi, 0.0])

    def is_upright(self, x):
        lib = _lib(x)
        return (lib.abs(wrap(x[..., 2])) < 0.15) & (lib.abs(x[..., 3]) < 1.0) & (lib.abs(x[..., 0]) < self.x_lim)

    def out_of_bounds(self, x):
        lib = _lib(x)
        return lib.abs(x[..., 0]) > self.x_lim

    def linearize(self):
        eps = 1e-6
        x0 = np.zeros(4)
        A = np.zeros((4, 4))
        for i in range(4):
            dx = np.zeros(4); dx[i] = eps
            A[:, i] = (self.f(x0 + dx) - self.f(x0 - dx)) / (2 * eps)
        return A, self.B(x0)

    def Q_cost(self):
        return np.diag([self.q_p, self.q_v, self.q_th / 2, self.q_w])

    def anchor_coords(self, x):
        lib = _lib(x)
        p, v, th, w = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        return lib.stack([p, v, lib.sin(th), 1 - lib.cos(th), w], -1)

    anchor_lin_idx = (0, 1, 2, 4)
    anchor_kappa = 3.0   # saturate the LQR anchor: the global quadratic is far too large on a 4-D box


# --------------------------------------------------------------------------
def rk4_step(sys, x, u, dt=None, substeps=None):
    """Zero-order-hold integration over one control period."""
    dt = sys.dt if dt is None else dt
    k = sys.substeps if substeps is None else substeps
    h = dt / k
    for _ in range(k):
        k1 = sys.xdot(x, u)
        k2 = sys.xdot(x + 0.5 * h * k1, u)
        k3 = sys.xdot(x + 0.5 * h * k2, u)
        k4 = sys.xdot(x + h * k3, u)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def rk4_step_scalar(sys, x, u):
    """Same integrator as rk4_step for one state as a tuple of floats (~20x faster than NumPy)."""
    h = sys.dt / sys.substeps
    n = len(x)
    for _ in range(sys.substeps):
        k1 = sys.xdot_scalar(x, u)
        k2 = sys.xdot_scalar(tuple(x[i] + 0.5 * h * k1[i] for i in range(n)), u)
        k3 = sys.xdot_scalar(tuple(x[i] + 0.5 * h * k2[i] for i in range(n)), u)
        k4 = sys.xdot_scalar(tuple(x[i] + h * k3[i] for i in range(n)), u)
        x = tuple(x[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n))
    return x


def make_system(name, **kw):
    return {"pendulum": Pendulum, "cartpole": CartPole}[name](**kw)


def angle_index(sys):
    return 0 if sys.name == "pendulum" else 2

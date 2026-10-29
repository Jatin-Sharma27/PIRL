"""Classical baseline: energy-shaping swing-up (Astrom & Furuta 2000) + LQR catch.

It also serves as the *admissible initial policy* for policy-iteration methods.
PI theory needs a stabilising starting policy, and this controller swings up
and balances from almost everywhere even with the weak motor.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

from ..systems import wrap


class Policy:
    """Interface: act(X) with X of shape (N, n) -> U of shape (N, 1), in [-1, 1]."""
    name = "policy"

    def act(self, X):
        raise NotImplementedError

    def __call__(self, X):
        X = np.atleast_2d(X)
        return np.clip(self.act(X), -1.0, 1.0)


def lqr(sys):
    A, B = sys.linearize()
    if sys.name == "pendulum":
        Q = np.diag([sys.q_th / 2 * 20, sys.q_w * 5])
    else:
        Q = np.diag([sys.q_p * 10, sys.q_v, sys.q_th / 2 * 20, sys.q_w * 5])
    R = sys.R()
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


class EnergyLQR(Policy):
    name = "Energy+LQR"

    def __init__(self, sys, k_energy=None, catch=None, a_max=8.0, kp=20.0, kv=5.0):
        # cart-pole gains from a grid search over (k_e, a_max, kp, kv, catch)
        self.sys = sys
        self.K, self.P = lqr(sys)
        self.a_max, self.kp, self.kv = a_max, kp, kv
        if sys.name == "pendulum":
            self.k_e = k_energy or 40.0
            self.catch = catch or 0.45
        else:
            self.k_e = k_energy or 20.0
            self.catch = catch or 0.3

    def _err(self, X):
        Xe = X.copy()
        ai = 0 if self.sys.name == "pendulum" else 2
        Xe[:, ai] = wrap(Xe[:, ai])
        return Xe, ai

    def act(self, X):
        sys = self.sys
        Xe, ai = self._err(X)
        u_lqr = -(Xe @ self.K.T)
        E = sys.energy(Xe)
        th, w = Xe[:, ai], Xe[:, ai + 1]
        if sys.name == "pendulum":
            u_sw = np.clip(self.k_e * (-E) * w, -1, 1) + 0.0 * th
            # if at rest exactly at the bottom, kick
            u_sw = np.where(np.abs(w) < 1e-3, 1.0, u_sw)
        else:
            p, v = Xe[:, 0], Xe[:, 1]
            # desired cart acceleration: pump energy, keep cart centred
            a_pump = np.clip(self.k_e * E * w * np.cos(th), -self.a_max, self.a_max)
            a_des = a_pump - self.kp * p - self.kv * v
            a_des = np.where(np.abs(w) < 1e-3, 3.0, a_des)
            f = sys.f(Xe); B = sys.B(Xe)[:, :, 0]
            u_sw = (a_des - f[:, 1]) / B[:, 1]
        near = (np.abs(th) < self.catch) & (np.abs(w) < 4.0)
        if sys.name == "cartpole":
            near &= np.abs(Xe[:, 0]) < sys.x_lim * 0.9
        return np.where(near, u_lqr[:, 0], u_sw)[:, None]


class LQROnly(Policy):
    name = "LQR"

    def __init__(self, sys):
        self.sys = sys
        self.K, _ = lqr(sys)

    def act(self, X):
        Xe = X.copy()
        ai = 0 if self.sys.name == "pendulum" else 2
        Xe[:, ai] = wrap(Xe[:, ai])
        return -(Xe @ self.K.T)

"""Gymnasium environments for the swing-up tasks.

The reward is the negative running cost:  r = -l(x,u) dt.  The HJB / DP / MPC
methods minimise the same cost, so all methods are compared on one objective.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .systems import make_system, rk4_step_scalar, wrap


class SwingUpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, system="pendulum", episode_s=None, random_init=0.5, sys_kwargs=None,
                 obs_noise=0.0, reward_scale=1.0):
        super().__init__()
        self.sys = make_system(system, **(sys_kwargs or {}))
        self.episode_s = episode_s or (8.0 if system == "pendulum" else 10.0)
        self.max_steps = int(round(self.episode_s / self.sys.dt))
        self.random_init = random_init
        self.obs_noise = obs_noise
        self.reward_scale = reward_scale  # RL-only rescaling; does not change the optimal policy
        nf = len(self.sys.features(np.zeros(self.sys.n)))
        self.observation_space = spaces.Box(-np.inf, np.inf, (nf,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        # terminal penalty for leaving the rail = cost of hanging for the rest of the episode
        self._hang_cost = float(self.sys.state_cost(self.sys.down_state())) * self.sys.dt

    def _obs(self):
        o = self.sys.features(self.x).astype(np.float32)
        if self.obs_noise:
            o = o + self.np_random.normal(0, self.obs_noise, o.shape).astype(np.float32)
        return o

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "x0" in options:
            self.x = np.asarray(options["x0"], dtype=np.float64).copy()
        elif self.np_random.random() < self.random_init:
            self.x = self.sys.sample_states(1, self.np_random)[0]
            self.x[-1] *= 0.3  # don't start spinning wildly
            if self.sys.name == "cartpole":
                self.x[0] *= 0.5; self.x[1] *= 0.3
        else:
            self.x = self.sys.down_state() + self.np_random.normal(0, 0.05, self.sys.n)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        u = np.clip(np.asarray(action, dtype=np.float64).reshape(1), -1.0, 1.0)
        c = float(self.sys.cost(self.x, u)) * self.sys.dt
        self.x = np.array(rk4_step_scalar(self.sys, tuple(self.x), float(u[0])))
        ai = 0 if self.sys.name == "pendulum" else 2
        self.x[ai] = wrap(self.x[ai])
        self.t += 1
        reward = -c
        terminated = bool(self.sys.out_of_bounds(self.x))
        if terminated:
            reward -= self._hang_cost * (self.max_steps - self.t) * 1.5
        truncated = self.t >= self.max_steps
        return self._obs(), reward * self.reward_scale, terminated, truncated, {"x": self.x.copy()}


gym.register("PendulumSwingUp-v0", entry_point=lambda **kw: SwingUpEnv("pendulum", **kw))
gym.register("CartPoleSwingUp-v0", entry_point=lambda **kw: SwingUpEnv("cartpole", **kw))

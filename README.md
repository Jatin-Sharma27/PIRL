# PIRL — Optimal swing-up control with a weak motor: HJB / physics-informed RL vs. deep RL vs. MPC

Reproduction and extension of the *Optimal Control of CT Systems using RL* project
(team Pinball Wizards: mid-sem presentation `RL-MidSemPresentattion.pdf`). It follows the resume entry
*"Solving Infinite-Horizon Optimal Control Problems using Physics-Informed Neural Networks & Deep RL"*.

Two plants, each driven by an **under-powered geared DC motor**, must **swing up from hanging and balance**:

| Plant | State | Actuator | Why it's hard |
|---|---|---|---|
| Pendulum on a motor shaft | (θ, ω), θ=0 upright | stall torque 0.25 N·m = **42 % of m·g·l**; torque falls with speed (back-EMF, no-load 20 rad/s); Coulomb + viscous friction | Can't lift the pole directly. It must pump energy over several swings. |
| Cart-pole on a belt-driven rail | (p, v, θ, ω) | stall force 4 N, no-load speed 2 m/s (back-EMF damping) | Only ±0.40 m of rail with hard end stops. Energy must be pumped without hitting the walls. |

The input is the normalised motor voltage u ∈ [−1, 1]. Back-EMF only depends on the state, so it goes into
the drift and both plants stay **control-affine**: ẋ = f(x) + B(x)u. **Every method optimises the same
discounted cost** ∫ e^{−ρt} [q_θ(1−cos θ) + q_ω ω² (+ q_p p² + q_v v² + rail barrier) + r u²] dt with
ρ = 0.25 s⁻¹. RL agents see r = −cost·dt and γ = e^{−ρ·dt} = 0.995, so the methods are scored on one objective.

## Methods (everything in the slides + resume)

| Family | Method | Where | Notes |
|---|---|---|---|
| Classical | LQR | `pirl/controllers/classical.py` | linearised about upright; can't swing up |
| Classical | Energy shaping + LQR catch | same | Åström–Furuta pumping; also used as the **admissible initial policy** for every PI method |
| Optimal (ground truth) | Offline DP (value iteration) | `pirl/dp.py` | 201×201 semi-Lagrangian grid on GPU. Gives the true V* for the 2-D pendulum (slide 4, item 1) |
| MPC benchmark | MPPI nonlinear MPC | `pirl/controllers/mppi.py` | GPU, 1024 samples × 1.6 s horizon, captured in a **CUDA graph** (≈4× faster on Windows) |
| MPC benchmark | Approximate explicit MPC | `pirl/controllers/explicit_mpc.py` | MPPI distilled into an MLP with DART noise (slide 4, item 4) |
| Physics-informed | PINN-HJB | `pirl/hjb.py` | minimises the HJB residual directly |
| Physics-informed | PINN actor-critic PI | same | "Model-based PI using actor-critic NN" (slide 6) |
| Physics-informed | X-TFC PI | same | ELM critic (Srinivasa & Kumar 2025). PI makes every step **one linear least-squares solve** |
| Physics-informed | **PIRL** | same | integral-RL policy iteration (Wang & Wu 2024, Vrabie–Lewis): V(x_t) = ∫e^{−ρs}ℓ + e^{−ρT}V(x_{t+T}) on simulator data, plus a physics (HJB) residual |
| Physics-informed | X-TFC-PIRL | same | integral-Bellman + physics rows, solved in closed form |
| Ablations | IRL (no physics), "no anchor" variants | same | isolate the contribution of each ingredient |
| Model-free deep RL | DDPG, TD3, SAC, PPO, A2C | `scripts/train_sb3.py` | Stable-Baselines3 on custom Gymnasium envs (`pirl/envs.py`) |
| Deployment | ESP32 int8 lookup table | `pirl/deploy.py`, `firmware/` | flash-resident LUT, x4 quadrature ISR, EMA velocity, LEDC PWM; hardware-in-the-loop sim |

### What is new here (beyond the papers)

1. **LQR-anchored constrained expression.** X-TFC enforces V(0)=0 exactly. We go one order further:

   V(x) = ψᵀPψ + ‖φ(x)−φ(0)‖² · (η(φ(x)) − η(φ(0))),  ψ = (sin θ, 1−cos θ, ω, …),

   where P solves the *discounted* Riccati equation of the linearisation. For **any** weights this gives
   V(0)=0, ∇V(0)=0 and ∇²V(0)=2P, and it is periodic in θ. The greedy policy is therefore *exactly* LQR near
   the top. Without the anchor, every learned critic was slightly too flat there: the greedy policy
   balanced gravity at a ~0.2 rad offset and failed the balance test. The anchor keeps X-TFC linear in β.
2. **Newton–Kleinman X-TFC.** Srinivasa & Kumar train the ELM output layer with Adam + L-BFGS on the
   nonlinear HJB. Inside policy iteration the policy-evaluation PDE is linear in β. So is the integral
   Bellman equation. Each PI step is then a single ridge solve, accumulated as normal equations in chunks
   (O(L²) memory). One PI step takes about a second.
3. **Actuator-aware HJB.** Back-EMF and friction live in f(x). The weak motor is a hard box on u, so the
   minimiser is u* = clip(−B(x)ᵀ∇V / 2r, −1, 1) (exact for separable quadratic input cost).
4. **One cost, one evaluator, many methods.** A DP ground truth on the 2-D plant scores both the value
   functions (RMSE vs V*) and the closed-loop costs of every method.
5. **Sim-to-real check before flashing.** Encoder quantisation, EMA velocity, one-sample delay, 8-bit PWM
   and torque noise are simulated on the exact int8 table the ESP32 runs.

## Results

Interactive simulation: `results/PIRL_sim.html`. It runs the pendulum live with every trained controller
(as its ESP32 lookup table) and a motor-strength slider, and replays the cart-pole runs.

All numbers: 20 fixed starts (hanging at rest + 19 random). Success = upright for the final 1.5 s.
Cost = Σ ℓ·dt, the same objective for every method. Full tables: `results/<plant>/summary.md`.

### Pendulum (stall torque = 42 % of m·g·l)

| Method | Family | Success | Up from rest [s] | Cost from rest | Mean cost | Train |
|---|---|---|---|---|---|---|
| DP (exact optimum) | ground truth | 100 % | 2.08 | **3.34** | **1.79** | 4 s |
| DDPG | deep RL | 100 % | 2.14 | 3.38 | 1.88 | 42 min |
| SAC | deep RL | 100 % | 2.16 | 3.42 | 1.97 | 54 min |
| PPO | deep RL | 100 % | 2.16 | 3.44 | 1.99 | 29 min |
| **PIRL (anchored)** | physics-informed | 100 % | 2.08 | 3.46 | 1.96 | 45 min |
| PINN actor-critic (anchored) | physics-informed | 100 % | 2.18 | 3.48 | 1.90 | 17 min |
| **X-TFC-PI (anchored)** | physics-informed | 100 % | 2.14 | 3.50 | 1.90 | **19 s** |
| **X-TFC-PIRL (anchored)** | physics-informed | 100 % | 2.12 | 3.53 | 1.85 | 2.3 min |
| TD3 | deep RL | 100 % | 2.16 | 3.54 | 1.85 | 42 min |
| Energy shaping + LQR | classical | 100 % | 2.24 | 3.63 | 1.89 | – |
| X-TFC-PIRL, no anchor | ablation | 100 % | 2.16 | 3.97 | 2.27 | 11 s |
| IRL, no physics term | ablation | 100 % | 3.48 | 4.69 | 2.15 | 8 min |
| MPPI-MPC (1.6 s horizon) | MPC | 100 % | 3.10 | 5.22 | 3.12 | online |
| Explicit MPC (NN) | MPC | 100 % | 4.12 | 6.85 | 2.98 | 4 min |
| PIRL, no anchor | ablation | 0 % (parks at ≈0.2 rad) | – | – | – | 2 min |
| PINN-HJB (direct residual) | physics-informed | 15 % | – | – | – | 17 min |
| LQR only | classical | 15 % | – | – | – | – |
| A2C | deep RL | 0 % | – | – | – | 24 min |

Value-function error vs DP (RMSE over |ω|<8): PIRL 0.19, X-TFC-PIRL 0.24, X-TFC-PI 0.42, PINN-AC 0.53,
PINN-HJB 1.92 (spurious solution, ≈0 everywhere), IRL 0.08. The last shows that a more accurate V does not
by itself give a better greedy policy: the policy depends on ∇V near the switching curve.

### Cart-pole (4 N motor, ±0.40 m rail)

| Method | Success | Up from rest [s] | Cost from rest | Mean cost | Train |
|---|---|---|---|---|---|
| SAC (600k steps) | 100 % | 2.54 | 4.14 | **2.22** | 48 min (HPC) |
| PPO (5M steps) | 100 % | 1.84 | **3.16** | 2.31 | 58 min |
| DDPG (600k steps) | 100 % | 2.60 | 4.31 | 2.38 | 29 min (HPC) |
| Explicit MPC (NN) | 100 % | 3.84 | 6.13 | 3.00 | 12 min |
| MPPI-MPC | 100 % | 2.80 | 4.94 | 3.16 | online |
| Energy shaping + LQR | 100 % | 2.68 | 4.23 | 3.40 | – |
| **PIRL, no anchor** | 95 % | 1.98 | 3.78 | 2.44 | **2.5 min (HPC)** |
| X-TFC-PIRL / X-TFC-PI | 10 % / 0 % | – | – | – | < 30 s |
| PIRL anchored / IRL / PINN-AC / PINN-HJB | 0 % | – | – | – | 2–3 min |
| TD3 (600k) / A2C (5M) | 0 % | – | – | – | |
| LQR only | 10 % | – | – | – | – |

### Robustness (success % when the plant differs from the design model)

Pendulum, motor at 0.5× (stall torque = 21 % of m·g·l): PIRL, X-TFC-PIRL, X-TFC-PI, DP, DDPG, TD3 100 %;
PPO 90 %; SAC 0 %; Energy+LQR 10 % (it fails below 0.8×). All methods tolerate 0.7–1.6× pole mass.
Cart-pole: DDPG and SAC ≥95 % everywhere; PIRL 85–95 % with the motor at 0.5–1.2× and 75–95 % with the
pole mass at 0.7–1.6×; PPO falls to 10 % at 0.5× motor; Energy+LQR fails below 0.8× motor and above
1.3× pole mass. Grids: `results/<plant>/figures/robustness.png`.

### Deployment check (int8 lookup table + simulated encoder, EMA velocity, 1-step delay, 8-bit PWM)

| Pendulum policy | float, ideal | int8 LUT, ideal | LUT + sensing/delay/PWM | + 5 % torque noise |
|---|---|---|---|---|
| PPO | 100 % | 100 % | **100 %** | **100 %** |
| SAC | 100 % | 100 % | 100 % | 100 % |
| X-TFC-PIRL | 100 % | 95 % | 95 % | 100 % |
| PIRL | 100 % | 95 % | 95 % | 100 % |
| DP | 100 % | 95 % | 95 % | 100 % |
| DDPG | 100 % | 100 % | **45 %** | 45 % |
| TD3 | 100 % | 100 % | 45 % | 40 % |
| Energy+LQR | 100 % | 95 % | 30 % | 35 % |

DDPG and TD3 tables lose half their swing-ups once the filtered, delayed velocity estimate replaces the
true one. The firmware header (`firmware/esp32_pendulum/policy_lut.h`) therefore ships the **PPO** table.

### Figures and simulations

- `results/<plant>/figures/cost_comparison.png`: every method on one cost
- `results/<plant>/figures/swingup_timeseries.png`: angle, rate, motor command (and cart position) from rest
- `results/<plant>/figures/rl_learning_curves.png`: SB3 learning curves against the DP / PIRL / Energy+LQR levels
- `results/<plant>/figures/robustness.png`: success grid vs motor strength and pole mass
- `results/pendulum/figures/value_functions.png`: learned critics vs the DP value function
- `results/<plant>/gifs/comparison.gif` + one GIF per controller

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements.txt

python scripts/run_experiments.py --system pendulum      # classical, DP, MPC, PINN/X-TFC/PIRL (GPU)
python scripts/run_experiments.py --system cartpole
bash   scripts/launch_sb3.sh                              # DDPG/TD3/SAC/PPO/A2C on both plants (CPU)
python scripts/compare.py --system pendulum               # tables, figures, GIFs, robustness
python scripts/compare.py --system cartpole
python scripts/deploy_lut.py --export PPO                 # LUT + HIL test + firmware/esp32_pendulum/policy_lut.h
python scripts/export_web.py && python scripts/build_web.py  # interactive page results/PIRL_sim.html
```

`python scripts/run_experiments.py --system pendulum --quick` gives a 4× shorter smoke run.

## Layout

```
pirl/systems.py        plant physics (NumPy + PyTorch, same code), motor model, costs
pirl/envs.py           Gymnasium envs (PendulumSwingUp-v0, CartPoleSwingUp-v0)
pirl/hjb.py            PINN-HJB, PINN-AC, X-TFC PI, PIRL, X-TFC-PIRL, anchored critics
pirl/dp.py             value-iteration ground truth
pirl/controllers/      LQR, energy+LQR, MPPI (CUDA graph), explicit MPC
pirl/evaluate.py       batched closed-loop evaluation (20 fixed starts, incl. hanging at rest)
pirl/animate.py        GIF rendering
pirl/deploy.py         LUT build, hardware-in-the-loop rollout, C header export
firmware/esp32_pendulum/   Arduino-ESP32 sketch + generated policy_lut.h
scripts/               experiment runners, SB3 trainer, comparison, deployment
results/<system>/      metrics, summary.md, figures/, gifs/, controllers/, sb3/
```

## References

- Y. Wang, Z. Wu, *Physics-informed reinforcement learning for optimal control of nonlinear systems*, AIChE J. 70(10), 2024.
- T. R. Srinivasa, S. Kumar et al., *Solving Infinite-Horizon Optimal Control Problems using the Extreme Theory of Functional Connections*, arXiv:2510.27187, 2025.
- B. Kiumarsi, K. Vamvoudakis, H. Modares, F. Lewis, *Optimal and Autonomous Control Using RL: A Survey*, IEEE TNNLS 29(6), 2018.
- B. Recht, *A Tour of Reinforcement Learning: The View from Continuous Control*, 2018.
- R. Reiter et al., *Synthesis of Model Predictive Control and Reinforcement Learning: Survey and Classification*, arXiv:2502.02133, 2025.
- A. K. Pal, T. Nestorović, *Swing Up and Balance of an Inverted Pendulum Using RL Coupled With a PID Controller*, ICECCME 2022.
- D. Vrabie, F. Lewis, *Neural network approach to continuous-time direct adaptive optimal control for partially unknown nonlinear systems*, Neural Networks 22, 2009.
- K. J. Åström, K. Furuta, *Swinging up a pendulum by energy control*, Automatica 36, 2000.
- G. Williams et al., *Information-theoretic MPC for model-based RL (MPPI)*, ICRA 2017.

# cartpole results

| Method | Family | Success (20 starts) | Swing-up from rest | t_up nominal [s] | Cost nominal | Cost mean | mean abs u | train time [s] |
|---|---|---|---|---|---|---|---|---|
| SAC | Model-free RL | 100% | yes | 2.54 | 4.14 | 2.22 | 0.08 | 2892 |
| PPO | Model-free RL | 100% | yes | 1.84 | 3.16 | 2.31 | 0.10 | 3473 |
| DDPG | Model-free RL | 100% | yes | 2.60 | 4.31 | 2.38 | 0.11 | 1749 |
| Explicit-MPC | MPC | 100% | yes | 3.84 | 6.13 | 3.00 | 0.07 | 717 |
| MPPI-MPC | MPC | 100% | yes | 2.80 | 4.94 | 3.16 | 0.46 | 0 |
| Energy+LQR | Classical | 100% | yes | 2.68 | 4.23 | 3.40 | 0.12 | 0 |
| PIRL (no anchor) | Physics-informed | 95% | yes | 1.98 | 3.78 | 2.44 | 0.88 | 147 |
| XTFC-PIRL | Physics-informed | 10% | no | - | - | - | 0.69 | 26 |
| LQR | Classical | 10% | no | - | 25.11 | - | 0.90 | 0 |
| A2C | Model-free RL | 0% | no | - | 16.51 | - | 0.84 | 2957 |
| IRL (no physics) | Physics-informed | 0% | no | - | - | - | 0.94 | 56 |
| XTFC-PI | Physics-informed | 0% | no | - | - | - | 0.71 | 17 |
| PIRL | Physics-informed | 0% | no | - | 21.48 | - | 0.91 | 185 |
| TD3 | Model-free RL | 0% | no | - | 24.01 | - | 0.52 | 1757 |
| PINN-AC | Physics-informed | 0% | no | - | - | - | 0.65 | 134 |
| PINN-HJB | Physics-informed | 0% | no | - | - | - | 0.97 | 120 |
| XTFC-PIRL (no anchor) | Physics-informed | 0% | no | - | - | - | 0.95 | 335 |

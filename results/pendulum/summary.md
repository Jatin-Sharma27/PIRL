# pendulum results

| Method | Family | Success (20 starts) | Swing-up from rest | t_up nominal [s] | Cost nominal | Cost mean | mean abs u | train time [s] |
|---|---|---|---|---|---|---|---|---|
| DP | Optimal (DP) | 100% | yes | 2.08 | 3.34 | 1.79 | 0.13 | 4 |
| TD3 | Model-free RL | 100% | yes | 2.16 | 3.54 | 1.85 | 0.20 | 2522 |
| XTFC-PIRL | Physics-informed | 100% | yes | 2.12 | 3.53 | 1.85 | 0.13 | 138 |
| DDPG | Model-free RL | 100% | yes | 2.14 | 3.38 | 1.88 | 0.15 | 2530 |
| Energy+LQR | Classical | 100% | yes | 2.24 | 3.63 | 1.89 | 0.14 | 0 |
| XTFC-PI | Physics-informed | 100% | yes | 2.14 | 3.50 | 1.90 | 0.14 | 19 |
| PINN-AC | Physics-informed | 100% | yes | 2.18 | 3.48 | 1.90 | 0.16 | 1024 |
| PIRL | Physics-informed | 100% | yes | 2.08 | 3.46 | 1.96 | 0.14 | 2708 |
| SAC | Model-free RL | 100% | yes | 2.16 | 3.42 | 1.97 | 0.32 | 3249 |
| PPO | Model-free RL | 100% | yes | 2.16 | 3.44 | 1.99 | 0.16 | 1747 |
| IRL (no physics) | Physics-informed | 100% | yes | 3.48 | 4.69 | 2.15 | 0.17 | 485 |
| XTFC-PIRL (no anchor) | Physics-informed | 100% | yes | 2.16 | 3.97 | 2.27 | 0.96 | 11 |
| Explicit-MPC | MPC | 100% | yes | 4.12 | 6.85 | 2.98 | 0.12 | 265 |
| MPPI-MPC | MPC | 100% | yes | 3.10 | 5.22 | 3.12 | 0.45 | 0 |
| LQR | Classical | 15% | no | - | 19.53 | - | 0.85 | 0 |
| PINN-HJB | Physics-informed | 15% | no | - | 20.50 | - | 0.85 | 1002 |
| PIRL (no anchor) | Physics-informed | 0% | no | - | 3.75 | - | 0.61 | 139 |
| A2C | Model-free RL | 0% | no | - | 19.92 | - | 0.70 | 1412 |

## Critic accuracy vs DP ground truth (RMSE of V over |w|<8)

- PIRL: 0.186
- XTFC-PIRL: 0.238
- XTFC-PI: 0.417
- IRL (no physics): 0.078
- PINN-HJB: 1.922
- PINN-AC: 0.531

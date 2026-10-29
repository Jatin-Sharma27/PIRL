# PIRL project log

Running log of everything done so far: decisions, parameters, results, every error hit and how it
was fixed, and the HPC setup. Status as of **2026-10-29** (all local runs stopped by request; work
resumes on Plaksha HPC).

---

## 0. Where we stopped (read this first)

**Done**
- Full codebase: plants, Gymnasium envs, 11 model-based/physics-informed methods, 5 SB3 agents,
  evaluation, plotting/GIFs, robustness sweep, ESP32 LUT export + firmware, HPC job scripts.
- Pendulum: every main method trained and evaluated (table in §6).
- Cart-pole: classical, MPC, PPO and A2C final. Physics-informed methods were run with the *unsaturated*
  anchor (bad on cart-pole, see §7 E14). The *saturated* anchor fix is written but **not yet run**.

**Remaining (in order)**
1. Finish the HPC env (`~/PIRL/logs/setup_env.log` should end with `OK torch ...`). Re-run
   `bash hpc/setup_env.sh` if it didn't finish (it is idempotent).
2. Sync the latest local code to HPC: `hpc/`, `pirl/`, `scripts/` were already uploaded once. Re-upload
   if anything changed locally.
3. Cart-pole job (saturated anchor + missing ablation + off-policy RL):
   ```bash
   cd ~/PIRL
   SYSTEM=cartpole \
   METHODS="XTFC-PI;XTFC-PIRL;PIRL;IRL (no physics);PINN-AC;PINN-HJB;PIRL (no anchor)" \
   SB3="DDPG TD3 SAC" sbatch --dependency=singleton hpc/experiments.sbatch
   ```
4. Pendulum ablations (cut off locally):
   ```bash
   SYSTEM=pendulum METHODS="XTFC-PIRL (no anchor);PIRL (no anchor)" sbatch --dependency=singleton hpc/experiments.sbatch
   ```
5. `sbatch --dependency=singleton hpc/finalize.sbatch` → compare.py (tables, figures, GIFs,
   robustness) for both plants + `deploy_lut.py --export DDPG` (writes `firmware/esp32_pendulum/policy_lut.h`).
6. Copy `~/PIRL/results` back, fill `RESULTS_PLACEHOLDER` in `README.md` from `results/*/summary.md`.

**Stale artefacts to know about**
- `results/cartpole/metrics_model_based.json`: the entries XTFC-PI, XTFC-PIRL, PIRL, IRL, PINN-AC
  and PINN-HJB come from the *unsaturated* anchor. Their `.pt` files will now load with the saturated
  `anchor_kappa=3`, which doesn't match how they were trained. They must be re-trained (step 3).
- The cart-pole unanchored PIRL result (**95 % success**, nominal cost 3.77, from `results/logs/run_cartpole2.log`)
  was overwritten in the metrics file when the anchored run wrote "PIRL". It is re-run as "PIRL (no anchor)".
- `results/cartpole/sb3/{DDPG,TD3,SAC}_s0_final.json` are from the earlier 250k-step run (no
  best-checkpoint selection). The matching `.zip` may since have been overwritten by the killed 600k
  runs' best checkpoints. Re-run (step 3) makes them consistent.

---

## 1. Source material

- `RL-MidSemPresentattion.pdf` (team Pinball Wizards: Arnav Kapoor, Meher Sidhu, Suraj Dayma, Jatin Sharma).
  Covers: optimal regulation ↔ RL equivalence, quadratic cost vs Gym reward, value search (DP/PI, HJB,
  off-policy IRL) vs policy search (explicit / approximate explicit MPC), plan = PIRL (Wang & Wu 2024)
  + MPC benchmark + SB3 comparison + hardware.
- `Jatin_Resume.pdf`, project entry: HJB via PINN & X-TFC, stabilisation over ±π where LQR fails,
  DDPG/TD3/SAC/PPO benchmark on a custom Gymnasium env, DDPG on ESP32 as a flash LUT with
  interrupt quadrature decoding, EMA velocity and PWM.
- Papers looked up:
  - Wang & Wu 2024 (AIChE J., doi 10.1002/aic.18542). The publisher page returned **HTTP 403**, so
    the method was implemented from the slide description (integral Bellman equation, model-free PI).
  - Srinivasa & Kumar et al., arXiv:2510.27187 (X-TFC for infinite-horizon HJB, Plaksha). Read via the
    arXiv HTML. Took from it: the constrained expression V = η(x) − η(0), the ELM critic, and the
    closed-form box-constrained control law (clip, not tanh).

## 2. Design decisions

| Item | Choice | Why |
|---|---|---|
| Angle | θ=0 upright, ±π hanging | matches the slides' HJB about the origin |
| Input | normalised voltage u∈[−1,1] | the motor is a voltage-driven DC gearmotor |
| Motor | τ = τ_stall·u − (τ_stall/ω_noload)·ω | back-EMF goes into f(x), so the plant stays control-affine |
| Pendulum | m=0.2 kg, l=0.3 m, J_rotor=0.002, τ_stall=0.25 N·m (**42 % of mgl**), ω_noload=20 rad/s, b=0.001, Coulomb 0.005 (tanh) | weak: needs several pumps |
| Cart-pole | M=0.5, m=0.15, l=0.25 (to COM), F_stall=4 N, v_noload=2 m/s, rail ±0.40 m | belt-driven rig like Pal & Nestorović |
| Cost | q_θ(1−cosθ)+q_ω ω²(+q_p p²+q_v v²+200·relu(|p|−0.3)²)+r u², q_θ=1, q_ω=0.01, q_p=1, q_v=0.02, r=0.05 | one cost for all methods |
| Discount | ρ=0.25 s⁻¹ ⇔ γ=0.995 at dt=0.02 | same objective for HJB/DP and RL |
| Control rate | 50 Hz, RK4 (2 substeps pendulum, 4 cart-pole) | |
| Control law | u* = clip(−Bᵀ∇V/2r, −1, 1) | exact for separable quadratic input cost (X-TFC paper) |
| Eval | 20 fixed starts (hanging rest + 19 random), 10 s / 12 s | success = upright (|θ|<0.15, |ω|<1, in rail) for the final 1.5 s |
| RL reward | −ℓ·dt ×10 (scale only) | SAC's entropy term swamped the tiny raw reward |

## 3. Code map

- `pirl/systems.py` physics (NumPy + Torch, same code), `xdot_scalar` / `rk4_step_scalar` fast path,
  `anchor_coords`, `Q_cost`, `anchor_kappa`
- `pirl/envs.py` Gymnasium envs
- `pirl/controllers/classical.py` LQR, Energy+LQR (cart-pole gains from a grid search:
  k_e=20, a_max=8, kp=20, kv=5, catch=0.3)
- `pirl/controllers/mppi.py` MPPI (CUDA graph), `explicit_mpc.py` (MPPI→MLP with DART noise)
- `pirl/dp.py` value iteration (201×201×41 grid, GPU)
- `pirl/hjb.py` PINN-HJB, PINN-AC, X-TFC PI, PIRL, X-TFC-PIRL, anchored critics, `NormalEq`
- `pirl/evaluate.py`, `pirl/animate.py`, `pirl/deploy.py`
- `scripts/run_experiments.py`, `train_sb3.py`, `launch_sb3.sh`, `compare.py`, `deploy_lut.py`
- `firmware/esp32_pendulum/esp32_pendulum.ino`
- `hpc/setup_env.sh`, `common.sh`, `experiments.sbatch`, `finalize.sbatch`

## 4. Contributions beyond the papers

1. **LQR-anchored TFC critic:** V = V_q(x) + ‖φ(x)−φ(0)‖²(η(φ(x))−η(φ(0))), with V_q = ψᵀPψ and P the
   discounted-Riccati solution (A−ρ/2·I). For any weights this pins V(0)=0, ∇V(0)=0 and ∇²V(0)=2P, so
   the greedy policy equals LQR at the top. Verified numerically: V(0)=0 and ∇V(0)=0 exactly, for both
   critics and both plants.
2. **Saturated anchor** V_q = κ·tanh(ψᵀPψ/κ) (cart-pole κ=3; pendulum unsaturated). Identical to second
   order, but bounded where the true discounted V is bounded.
3. **Newton–Kleinman X-TFC:** policy evaluation and the integral Bellman equation are linear in β. Each
   PI step is one ridge solve, accumulated chunk by chunk as normal equations (O(L²) memory).
4. **PIRL = IRL data loss + λ·physics residual** (λ=0.1), with the energy controller as the admissible
   initial policy and policy improvement needing only B(x).
5. **CUDA-graph MPPI** (~4× faster on Windows). HIL sim of the int8 LUT (encoder, EMA, delay, PWM, noise).

## 5. Timeline of work

1. Read the PDFs. Searched for the papers (Wang & Wu: 403; X-TFC: read on arXiv).
2. Local `.venv` (Python 3.10). Torch CPU first, then **CUDA 12.6 torch** (RTX 4050 Laptop) on request.
3. Wrote the plants, envs and classical controllers. Energy+LQR: pendulum 100 %, cart-pole 35 % → grid search → 100 %.
4. DP: converged in ~500 iterations (3.3 s on GPU). Pendulum V*(down) = 2.77.
5. MPPI: sweep over horizon {1.6, 2.5} s, λ {0.01, 0.05}, σ {0.4, 0.8}. Picked 1.6 s, λ=0.05, σ=0.4.
6. HJB smoke tests (pendulum): X-TFC-PI 100 % (V-RMSE vs DP 0.37), X-TFC-PIRL 100 % (RMSE 0.18),
   PIRL 100 % (RMSE 0.17, nominal J 3.50), PINN-HJB fails (V(down) = −1.8, a spurious solution),
   PINN-AC 0 %.
7. Full runs on both plants + 10 SB3 runs in parallel (memory problems, §7).
8. Found the PIRL steady-state offset → designed the LQR anchor → fixed the scaling → pendulum methods to 100 %.
9. Cart-pole: anchor hurt → saturated anchor (written, not yet run).
10. Wrote the ESP32 LUT pipeline and firmware, README draft, HPC scripts. Copied the project to HPC and started env setup.
11. User asked to stop all local work → killed all local runs. This log written.

## 6. Results so far

Cost = discounted-free sum of ℓ·dt over the episode. "nominal" = start hanging at rest. Mean = over 20 starts.

### Pendulum (τ_stall = 42 % of mgl)

| Method | Success | Nominal swing-up | t_up [s] | J nominal | J mean | train [s] |
|---|---|---|---|---|---|---|
| **DP (optimum)** | 100 % | yes | 2.08 | **3.34** | **1.79** | 3.7 |
| Energy+LQR | 100 % | yes | 2.24 | 3.63 | 1.89 | 0 |
| LQR | 15 % | no | – | 19.53 | 16.47 | 0 |
| MPPI-MPC | 100 % | yes | 3.10 | 5.22 | 3.12 | – |
| Explicit-MPC (NN) | 100 % | yes | 4.12 | 6.85 | 2.98 | 265 |
| PINN-HJB (direct, anchored) | 15 % | no | – | 20.50 | 17.34 | 1002 |
| PINN-AC (anchored) | 100 % | yes | 2.18 | 3.48 | 1.90 | 1024 |
| X-TFC-PI (anchored) | 100 % | yes | 2.14 | 3.50 | 1.90 | **19** |
| X-TFC-PIRL (anchored) | 100 % | yes | 2.12 | 3.53 | **1.85** | 138 |
| PIRL (anchored) | 100 % | yes | 2.08 | 3.46 | 1.96 | 2708 |
| IRL, no physics (anchored) | 100 % | yes | 3.48 | 4.69 | 2.15 | 485 |
| DDPG | 100 % | yes | 2.14 | **3.38** | 1.88 | 2530 |
| TD3 | 100 % | yes | 2.16 | 3.54 | 1.85 | 2522 |
| SAC | 100 % | yes | 2.16 | 3.42 | 1.97 | 3249 |
| PPO | 100 % | yes | 2.16 | 3.44 | 1.99 | 1747 |
| A2C | 0 % | no | – | 19.92 | 15.73 | 1412 |

Anchor ablation (pendulum, before → after): PIRL 0 % → 100 %; PINN-AC 0 % → 100 %;
X-TFC-PI nominal J 3.90 → 3.50; X-TFC-PIRL 3.97 → 3.53. Unanchored PIRL/IRL caught the pole but
parked it at |θ| ≈ 0.19–0.32 rad.

### Cart-pole (F_stall = 4 N, rail ±0.4 m)

| Method | Success | Nominal | t_up | J nominal | J mean | note |
|---|---|---|---|---|---|---|
| Energy+LQR | 100 % | yes | 2.68 | 4.23 | 3.40 | |
| MPPI-MPC | 100 % | yes | 2.80 | 4.94 | 3.16 | |
| Explicit-MPC | 100 % | yes | 3.84 | 6.13 | 3.00 | |
| PPO (5M steps) | 100 % | yes | 1.84 | **3.16** | **2.31** | best overall |
| PIRL (no anchor) | 95 % | yes | 1.98 | 3.77 | 2.44 | earlier run, re-run pending |
| PIRL (unsaturated anchor) | 0 % | no | – | 23.42 | 22.59 | superseded |
| X-TFC-PIRL (unsat. anchor) | 20 % | no | – | – | 1.54 | superseded |
| X-TFC-PI / IRL / PINN-AC / PINN-HJB (unsat. anchor) | 0 % | | | | | superseded |
| X-TFC-PIRL (no anchor) | 0 % | | | | | final |
| DDPG / SAC / TD3 (250k) | 5 / 0 / 0 % | | | | | re-run at 600k pending (DDPG hit 75 % at 210k, then collapsed) |
| A2C (5M) | 0 % | | | | | final |
| LQR | 10 % | | | | | |

## 7. Errors encountered and fixes

| # | Symptom | Cause | Fix |
|---|---|---|---|
| E1 | `pip list` showed no torch / gymnasium / SB3 | fresh machine | created project `.venv`, installed CPU torch, then **CUDA 12.6** torch for the GPU |
| E2 | Cart-pole Energy+LQR only 35 % success, carts hit the end stops | unbounded pumping acceleration | clipped pumping acceleration (a_max) + centring PD. Grid search over 324 gain sets → 100 % |
| E3 | MPPI 535 ms per batched step | Windows WDDM kernel-launch overhead (~6k tiny kernels per rollout) | captured the rollout in a **CUDA graph** → 35 ms (≈4–15×) |
| E4 | Background smoke test hung for 600 s with no output | my command `cat > "$TMPDIR/smoke.py" 2>/dev/null` waited on stdin | removed it; scripts written to the scratchpad dir with heredocs |
| E5 | DP value function had a bias | "relative VI" anchoring at the nearest grid point (no exact θ=0 node) | plain discounted VI (γ^4000 ≈ 2e-9; converges in ~500 iterations anyway) |
| E6 | SAC 40–50 ms per update (CPU and GPU) | env step 0.5 ms (NumPy overhead on 1 state) + 256-wide MLPs on 2 threads + WDDM launch cost | pure-Python `rk4_step_scalar` (env step 80 µs, bit-identical to the batched integrator), nets 256→128, SB3 on CPU |
| E7 | 4 of 10 SB3 runs crashed: `OSError [WinError 1455] paging file too small` loading `cufft64_11.dll` | each CUDA-torch process commits ~1.5 GB, 12 processes on a 16 GB laptop | second **CPU-only torch venv** for SB3 (on `D:\PIRL_envs\venv-cpu`, see E8) |
| E8 | `pip install` failed: `[Errno 28] No space left on device`; C: had **0.12 GB** free | full system drive (this also blocked page-file growth) | `pip cache purge` (3.1 GB, my downloads) → 3.24 GB free. CPU venv put on **D:** |
| E9 | CPU venv install: `No matching distribution for flit_core` | old pip + `--index-url` restricted to the torch index | upgraded pip; `--extra-index-url https://pypi.org/simple` |
| E10 | Cart-pole runner: `torch.OutOfMemoryError ... 2 MiB` with 4.3 GB free | host commit memory exhausted (E7/E8), not the GPU | relaunched from the failed method once memory was freed |
| E11 | PPO cart-pole `DefaultCPUAllocator: not enough memory` | same memory crunch | relaunched with the CPU venv |
| E12 | Pendulum PIRL / IRL / DDPG "0 % success" even though they swung up (J ≈ 2.1) | learned critic too flat at the top → greedy torque balances gravity at **θ ≈ 0.2 rad** (steady-state offset) | **LQR-anchored constrained expression** (§4.1) |
| E13 | First anchor made things worse (X-TFC 45 %, PIRL 15 %, no swing-up) | multiplier ‖ψ‖² used raw ω (≈144 at 12 rad/s), which swamped the network term | multiplier → ‖φ(x)−φ(0)‖² (normalised features). X-TFC anchored → 100 %, J 3.49 |
| E14 | On cart-pole the anchor broke PIRL (95 % → 0 %, IRL residual 10× higher) | the global quadratic ψᵀPψ is huge on a 4-D box (|ω|≤15, p, v), while the true discounted V is bounded | **saturated anchor** κ·tanh(ψᵀPψ/κ), κ=3 for cart-pole (pendulum keeps κ=None). Not yet re-run |
| E15 | Cart-pole X-TFC job committed **15 GB** RAM / 5.6 GB working set | L=3000 × 60k rows fp64 + JVPs > 6 GB GPU → spilled into shared system memory | `NormalEq`: accumulate AᵀA, Aᵀy in 8192-row chunks (O(L²) memory). L 3000 → 2500 |
| E16 | Risk of lost results with two runners writing `metrics_model_based.json` | each process loaded the file once at start and overwrote it | re-read + merge right before every write |
| E17 | SB3 off-policy agents collapsed late (cart-pole DDPG 75 % at 210k → 5 % at 250k) | DDPG/TD3 instability | evaluation callback keeps the **best checkpoint** (success, then cost); final metrics use it. Last iterate saved as `*_last.zip` |
| E18 | Pendulum DDPG first run 0 % | same offset/instability | best-checkpoint selection → **100 %, J 3.38** (≈ DP 3.34) |
| E19 | A Bash heredoc patch failed (`unexpected EOF while looking for matching '`) | `'''` inside a bash `-c` string | patches written as files in the scratchpad and run with Python |
| E20 | My background "wait" shell was killed by Claude Code | the system was critically low on memory while idle | not restarted. Used the Monitor tool on log lines instead |
| E21 | A wait loop fired early | grep `PIRL (no anchor) ===` also matched `XTFC-PIRL (no anchor) ===` | anchored awk pattern `^=== cartpole: PIRL \(no anchor\) ===` |
| E22 | First HPC upload: `gzip: unexpected end of file` | in `tar -x && ... && nohup ... &`, the trailing `&` backgrounded the whole chain, so ssh closed stdin mid-transfer | upload and setup as two separate ssh calls |
| E23 | Paper page (Wiley) **HTTP 403** | paywall/bot block | worked from the slides. X-TFC details from arXiv HTML |
| E24 | Run under memory pressure | laptop limits (16 GB RAM, C: almost full) | user decided to move all compute to the HPC. All local runs stopped |

## 8. HPC (Plaksha) setup

- Access: ssh alias `plaksha` → `hpc.plaksha.edu.in`, user `jatin.sharma.ug23`, key `~/.ssh/plaksha_hpc`
  (already configured). The warning about non-post-quantum key exchange is harmless and comes from the server.
- Cluster: Slurm, no `module` command. Partitions: `rtx` (cn02/cn04: 2× MIG 2g.48gb; cn03: 4× 1g.24gb),
  `h200` (cn01, 2× H200 NVL). No CPU-only partition.
- **Rules we follow (user requirement):** only `--account=courses --qos=course` (1 GPU per user, max
  1-12:00:00 per job), partition `rtx`. Jobs share `--job-name=pirl` and are submitted with
  `--dependency=singleton`, so only one runs at a time. **Never** use `general` / `scavenger` / `h200`
  to get more GPUs. No compute on the login node; only the one-time env install ran there, as in
  project-atlas' setup.
- Storage: code + results in `~/PIRL` (home quota 9 GB / 90k files; using ~2.5 GB / 4.2k). Conda env on
  scratch: `/scratch/$USER/miniforge3/envs/pirl` (Python 3.11, torch cu128, SB3, gymnasium). Per-job
  temp in `/scratch/$USER/tmp/<jobid>`, removed on exit.
- Lustre quirk (learned in project-atlas): big Python writes to /home and /scratch can fail with
  `OSError [Errno 14] Bad address`. `hpc/setup_env.sh` installs pip packages into `/tmp` first and copies
  them with `cp -a` (`pip_local`), and extracts conda packages in `/tmp`.
- Files: `hpc/setup_env.sh` (idempotent), `hpc/common.sh` (thread caps, temp dir, conda activate),
  `hpc/experiments.sbatch` (GPU methods + SB3 on CPU cores in one job; env vars `SYSTEM`, `METHODS`
  `;`-separated, `SB3`), `hpc/finalize.sbatch` (compare + LUT export).
- State at stop: project copied to `~/PIRL`. `setup_env.sh` started on the login node (last seen:
  conda env creation executing). **No Slurm job submitted.**
- Useful commands: `squeue -u $USER`, `tail -f ~/PIRL/logs/pirl-<jobid>.out`,
  `scancel <jobid>`, copy back: `ssh plaksha 'cd ~/PIRL && tar czf - results' | tar xzf -`.

## 9. Local environment notes

- `.venv` (CUDA torch 2.14.1+cu126) for GPU work. `D:\PIRL_envs\venv-cpu` (CPU torch) for SB3.
- C: drive is nearly full (0.95–3 GB free during the session). Keep large installs and caches off C:.
- All local training processes were stopped on 2026-10-29 at the user's request.

---

## 10. Session 2026-10-30

- First ssh attempt failed: `Could not resolve hostname hpc.plaksha.edu.in`. The name is campus-internal
  and the laptop was off campus with Cloudflare WARP handling DNS (public DNS: "Non-existent domain").
  Fixed by the user reconnecting to the campus network / VPN. Logged as **E25**.
- HPC env confirmed: `OK torch 2.11.0+cu128 cuda 12.8 | sb3 2.9.2` (conda env `pirl`, Python 3.11).
- Synced the latest `hpc/ pirl/ scripts/ firmware/ results/ docs/` to `~/PIRL`.
- Submitted (all `--account=courses --qos=course`, partition rtx, 1 GPU, chained so only one runs at a time):
  - **7732** cart-pole: XTFC-PI, XTFC-PIRL, PIRL, IRL (no physics), PINN-AC, PINN-HJB (saturated
    anchor κ=3) + PIRL (no anchor) on the GPU. SB3 DDPG/TD3/SAC (600k steps, best checkpoint) on CPU cores.
    Started on cn02. X-TFC-PI finished in 17 s (94 s on the laptop) and still fails on the 4-D
    cart-pole (0 %) with the saturated anchor too, so the ELM critic doesn't scale to 4-D swing-up.
  - **7733** pendulum ablations: XTFC-PIRL (no anchor), PIRL (no anchor). `afterany:7732`.
  - **7734** finalize (compare both plants + LUT export). `afterany:7733`.
- Note: `--dependency=singleton` alone doesn't order two pending jobs, so explicit `afterany` chaining
  was set with `scontrol update`.

### Session 2026-10-30, continued: project completed

**Jobs (all `courses`/`course`, rtx, 1 GPU, run one after another)**

| Job | Content | Result |
|---|---|---|
| 7732 | cart-pole: anchored (κ=3) physics-informed + PIRL (no anchor) + DDPG/TD3/SAC 600k | done 04:28 |
| 7733 | pendulum ablations (no anchor) | done 04:31 |
| 7734 | finalize | **failed**: `KeyError: 'XTFC-PIRL (no anchor)'` (E26) |
| 7736 | web export | done (but missing the no-anchor PIRL replay, E27) |
| 7773 | finalize resubmit | cancelled by me to include the E27 fix |
| 7774 | finalize (tables, figures, GIFs, robustness, LUT/HIL) | done 04:45 |
| 7775 | web export | done 04:46 (`results/web_data.json`, 709 KB) |

**Final key results** (full tables in README "Results"):
- Pendulum: DP optimum 3.34 from rest. DDPG 3.38, SAC 3.42, PPO 3.44, PIRL 3.46, PINN-AC 3.48, X-TFC-PI 3.50
  (19 s to train), X-TFC-PIRL 3.53, Energy+LQR 3.63, MPPI 5.22.
- The anchor on the pendulum: PIRL 0 % → 100 %, X-TFC-PIRL 3.97 → 3.53.
- Cart-pole: SAC/PPO/DDPG 100 % (0.6–5M steps). PIRL (no anchor) 95 % after 2.5 min, cost from rest 3.78
  (Energy+LQR 4.23, MPPI 4.94). The anchor (even saturated) and the X-TFC critics fail in 4-D. TD3, A2C fail.
- Robustness: pendulum at 0.5× motor: physics-informed + DDPG/TD3 100 %, Energy+LQR 10 %.
- HIL deployment: DDPG/TD3 LUTs drop to 45 % with encoder + EMA + delay + PWM; PPO/SAC stay at 100 %.
  The firmware header now ships **PPO** (`hpc/finalize.sbatch` updated to `--export PPO`).

**Deliverables**
- Interactive page `results/PIRL_sim.html`, published privately: https://claude.ai/artifact/QWhPSQLibbVq4GPmB4qxwn
  (live pendulum with all LUT controllers + motor slider; cart-pole replays; results tables/bars).
  Built by `scripts/export_web.py` (HPC) + `scripts/build_web.py` from `web/sim_template.html` + `web/findings.html`.
- Figures and GIFs in `results/<plant>/figures`, `results/<plant>/gifs`.
- README results section filled.

**New problems and fixes**

| # | Symptom | Cause | Fix |
|---|---|---|---|
| E25 | `Could not resolve hostname hpc.plaksha.edu.in` | off campus; Cloudflare WARP DNS can't see the campus-internal name | user reconnected to the campus network / VPN |
| E26 | finalize crashed: `KeyError: 'XTFC-PIRL (no anchor)'` in `fig_costs` | `FAMILY` map lacked the ablation names | added them + `FAMILY.get(k, "Physics-informed")` |
| E27 | no-anchor PIRL (the best cart-pole PIRL) missing from GIFs/robustness/web | `load_policies` didn't load "no anchor" critics | load them; on each plant show the better of anchored/unanchored PIRL |
| E28 | X-TFC-PIRL cart-pole "mean cost 0.03" | mean taken over the few non-crashed runs only | mean cost shown only when success ≥ 50 % (table + page) |
| E29 | robustness plot unreadable | 10 lines, most exactly at 100 %, repeated colours | redrawn as a method × perturbation success grid (`plot_robustness`) |
| E30 | `UnicodeDecodeError 0x97` building the page | a Python patch wrote "—" with the Windows cp1252 default | repaired the one stray byte (surrogateescape); always pass `encoding="utf-8"` |
| E31 | `--dependency=singleton` didn't order pending jobs | singleton only limits concurrency | explicit `afterany` chains via `scontrol update` |
| E32 | the log watcher's ssh dropped intermittently | brief campus-network drops | direct checks + a single background `until` loop instead |
| E33 | DDPG LUT (the resume's deployed policy) 45 % in the HIL sim | DDPG/TD3 policies are sensitive to the filtered, delayed velocity | export the PPO table (100 % in HIL). For hardware use PPO/SAC, or physics-informed tables (95–100 %) |

**Status: complete.** Possible next steps: real-hardware trial of the PPO LUT; multi-seed runs for error
bars; a physics-informed critic that scales to 4-D (e.g. anchor only in a local region).

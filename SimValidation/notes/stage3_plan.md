# Immediate task (2026-11-19): put the project under git and push it for the GPU machine

Remote: https://github.com/GuanQinG-GitHub/ECE792-2026Fall-AdaptiveSafeMPPI.git. `ls-remote` returns nothing, so it is reachable and empty.

Steps:
1. **Copy this plan into the repo.** Copy it to `SimValidation/notes/stage3_plan.md` so the other device has it.
2. **Initialise the repo.**
   - Repo root: `/Users/xinlei/Documents/ECE792_Project`, which holds `AdaptiveSF/` (60 MB) and `SimValidation/` (3.9 MB).
   - Command: `git init -b main`.
3. **Add `.gitignore`:**
   - `.DS_Store`, `__pycache__/`, `*.pyc`
   - `AdaptiveSF/output/`
   - `SimValidation/external/` (the cloned log-MPPI_ros reference, fetched on demand)
   - `*.mp4` is not ignored. Videos are small and wanted for review.
4. **Add a short top-level `README.md`.**
   - Layout of the two folders.
   - Pointer to `SimValidation/index.html` as the hub and to `notes/stage3_plan.md`.
   - Setup on the GPU machine: pip deps, pycuda, ffmpeg, and `--device cuda`.
5. **Commit everything,** including the trained models.
   - `sdf_vae_latent64_penalty.pt` is 53 MB. That is under GitHub's 100 MB hard limit but triggers a >50 MB warning.
   - git-lfs is not installed, so commit it as a normal file.
   - Commit message ends with the Co-Authored-By line.
6. **Push.** `git remote add origin <url>`, then `git push -u origin main`. This needs the user's GitHub credentials on this Mac. If push auth fails, report it and suggest `! gh auth login` or an SSH remote (`github-personal` host alias in `~/.ssh/config`).
7. **Update the hub.**
   - Session-log entry in `SimValidation/index.html`: Stage 3 planned and paused for the GPU machine, plus the repo URL. Include it in the commit.
   - Stage-1 open questions relabelled as deferred notes.
   - Update the memory note with the repo URL.
8. **Verify** with `git log --oneline` and `git ls-remote origin`, which should show `main`.

---

# Stage 3 — Nominal MPPI + log-MPPI collision baseline (reproduce log-MPPI_ros first)

## Context
Stage 1 (static forest + value-function check) is done; its five review questions get parked as "notes to revisit" in the hub.
Stage 3 builds the controllers the safety filter (Stage 4) will be compared against:
- **Nominal MPPI**: drives to the goal and ignores obstacles. Later this is what the adaptive SF filters.
- **Baseline log-MPPI**: the same core with the repo's 1e7 crash penalty on a local costmap.

The user asked to follow https://github.com/IhabMohamed/log-MPPI_ros and first **reproduce its result**. That repo is ROS + Gazebo + PyCUDA (NVIDIA only) and cannot run on this Mac. So "reproduce" means a faithful re-implementation of its MPPI/log-MPPI core in PyTorch (MPS) and a re-run of the paper's **Table II** forest experiment on the repo's own Gazebo forests (`forest0.world`: 250 trees; `forest1.world`: 500 trees).

User decisions (2026-11-19):
- The baseline uses the **same 5 m ego window** as the SF.
- The baseline is **log-MPPI only**.
- Our version runs at **50 Hz** like the repo.

## What the repo does (verified from source)
- **Dynamics:** `mppi.py`, `jackal.py`, `mppi_params_sim.yaml`, `mppi_launcher.launch`. Unicycle kinematics at dt = 1/50 s, horizon 6 s (N = 300; the paper says 5 s), K = 2496 rollouts. The first rollout is noise-free.
- **Control clamp quirk:** if |v| > v_max it becomes 0.6·v_max (same for ω). Limits: v_max = 2, ω_max = 1.5, and negative v is allowed.
- **Noise:**
  - MPPI: δu = std_n·N(0,1), std_n = [0.152, 0.166], λ = 0.572.
  - log-MPPI: δu = std_n·N(0,1)·exp(N(μ_ln, σ_ln)), std_n = [0.045, 0.047], λ = 0.169.
  - μ_ln and σ_ln come from `Normal2LogN(0, mean(std_n))`, and pycuda's `gen_log_normal` takes the parameters of the underlying normal.
- **Cost per step:**
  - State cost: Σ w_i (s_i − target_i)², w = [2.5, 2.5, 2]. With `without_heading`, the heading target is the current heading until within 0.4 m of the goal.
  - Crash penalty: +1e7 if any of 4 footprint points (±0.22 m along the body axes) lands on a costmap cell > 0. Costmap: 240×240 at 0.05 m, inflation 0.23 m. Only applied when the centre is inside the map.
  - Control cost: 0.5(1 − 1/ν)R δu² + R δu u + 0.5 R u², with R = λ/std_n and ν = 1200.
  - The sum is multiplied by dt.
- **Update:** w = exp(−(S − S_min)/λ), normalised. U ← Σ w·(U + δu), then Savitzky–Golay (window 51, order 3, mode='mirror'). Execute U[0], then shift U by one step and append 0.
- **Paper Table II targets** (10 trials each):

  | | Forest #1 MPPI | Forest #1 log-MPPI | Forest #2 MPPI | Forest #2 log-MPPI |
  |---|---|---|---|---|
  | Crashes | 2 | 0 | 7 | 1 |
  | l_av (m) | 158.71 ± 1.54 | 157.91 ± 0.54 | 76.12 ± 3.31 | 72.64 ± 0.80 |

- **Missions:**
  - Forest #1 (`forest0.world`, v_max = 2): waypoints (20,20) → (−18,2) → (20,−21) → (20,20) → (0,0).
  - Forest #2 (`forest1.world`, v_max = 1.5): (0,0) → (20,20) → (−18,2).
  - A waypoint is reached when the position error is below 0.3 m.

## Status / handoff (2026-11-19)
**Paused before implementation.** The user will continue on an NVIDIA GPU machine. This Mac has an Apple M3 Pro with no CUDA, so PyCUDA cannot run here.

Additional requirements from the user (they override the torch/MPS parts below):
- **PyCUDA + GPU for the parallel sampling.**
  - Port the repo's Jinja-templated CUDA rollout kernel (`mppi.py: cuda_rollouts`, `jackal.py: cuda_kinematics / cuda_state_cost`) as-is.
  - Batch multiple tasks by adding a task index to the grid: blockIdx.y = task, with per-task state, U, target and costmap.
  - Our ego-window crash cost is a second `__device__` cost function in the same template.
  - Wherever this plan says "torch/MPS", read "PyCUDA". Torch is still used only for the AdaptiveSF models (device `cuda`).
- **MP4 for the reproduction** (for review): one representative trial per forest × {MPPI, log-MPPI}.
  - World view with the robot and its trail.
  - Inset of the local costmap.
  - A subsample of the sampled rollouts (grey), the optimal predicted trajectory, and waypoints.
  - Rendered with matplotlib + ffmpeg.
- **MP4 for our forest** once the log-MPPI baseline works: same layout, plus the 5 m ego window. Render the nominal MPPI too.

Setup on the GPU machine:
- Copy `ECE792_Project/{AdaptiveSF,SimValidation}`, keeping the sibling layout because `simenv/sensing.py` imports `../AdaptiveSF`.
- `pip install pycuda jinja2 torch numpy scipy matplotlib`, and make sure `ffmpeg` is installed.
- Stage 1 scripts default to `--device mps`; pass `--device cuda`.
- This plan file lives at `~/.claude/plans/good-done-now-let-s-sorted-blum.md` on the Mac. Copy it to `SimValidation/notes/stage3_plan.md` so the new session can read it.

## Implementation

### 0. Reference copy
`git clone` the repo into `SimValidation/external/log-MPPI_ros/` and record the commit hash. Only the world files and the `Tree*.dae` meshes are used; no ROS code runs.

### 1. `SimValidation/mppi/core.py`: batched MPPI in torch
- Class `MPPI(cfg)` holds U[B, N, 2] for B tasks running in lockstep.
- Noise types `normal` and `nln`. Shapes are [B, K, N, 2] on MPS, with a float32 fallback to CPU.
- `step(state[B,3], cost_fn) -> u[B,2]` implements the exact repo update: noise-free sample 0, control cost, min-baseline softmax, weighted average of U + δu, SG filter, shift.
- Clamp mode is a config switch: `repo_quirk` (×0.6) or `clip` (ours: v ∈ [0, 1], ω ∈ [−1, 1], the value-function bounds).
- The SG filter uses precomputed `scipy.signal.savgol_coeffs` applied as a 1D convolution with mirror padding. A unit check compares it against `scipy.signal.savgol_filter(mode='mirror')`.
- Configs as dataclasses: `REPO_MPPI`, `REPO_LOGMPPI`, `OURS_NOMINAL`, `OURS_BASELINE`.

### 2. `SimValidation/mppi/costs.py`
- `goal_cost(s, target, w, without_heading)`.
- `CrashRepo`: 4-point footprint check against a robot-centred 12 m × 12 m inflated grid. The grid is a perfect-sensing crop of `World.sdf ≤ 0.23` on a 0.05 m global raster, refreshed every tick.
- `CrashEgoWindow` (ours): the same ego 5 m window the SF sees, from `simenv/sensing.local_occupancy`.
  - Re-sensed every tick.
  - Crash means the rollout centre's SDF in the window is ≤ r + inflation margin. A config value `inflation`, default 0.10 m, is added to r = 0.25.
  - Out-of-window rollout points cost nothing, as in the repo (unknown space is free).
  - The window SDF comes from `sensing.sdf_from_occupancy`. Lookup is bilinear or nearest in ego coordinates via `world_to_ego`.

### 3. `SimValidation/simenv/gazebo_forest.py`
- Parse `forestX.world`: per-tree pose, yaw, scale and mesh name.
- Parse each `TreeN.dae` (minimal COLLADA reader: positions plus node transforms). Slice the vertices in the LiDAR band z ∈ [0.3, 0.7] m (from the costmap params), take the convex hull, scale and rotate by the tree pose, and emit a `Polygon`.
- If the mesh parse fails, fall back to a `Circle` with the trunk radius.
- Shift coordinates by +25 m so the map is [0, 50]², and shift the waypoints the same way.
- Returns a `World` (reuses `simenv/world.py`, `shapes.py`).

### 4. `SimValidation/simenv/robot.py`
- Plant: unicycle kinematics at the control dt, with an optional bounded disturbance using the same bounds as the value function. Off by default in Stage 3.
- Collision checks:
  - Repro: Jackal rectangle 0.508 × 0.430 m, sampled perimeter points with exact `World.sdf ≤ 0`.
  - Ours: disc, `World.sdf(centre) ≤ 0.25`.
- Mission/episode bookkeeping per batch element: waypoint index, done, crash, timeout, path length, speed, min clearance, compute time per tick.

### 5. `scripts/s3a_reproduce_logmppi.py`
- Runs {forest0, forest1} × {MPPI, log-MPPI} × 10 trials, batched B = 10, with different RNG seeds per trial.
- A crash ends that trial, which counts towards N_crash.
- Outputs to `results/stage3/`:
  - `repro_table.json` and a paper-vs-ours table.
  - `repro_forest{0,1}.png`: world, all trajectories, crash markers.
  - Timing.
- Acceptance:
  - Same ordering as the paper (log-MPPI ≤ MPPI crashes, more crashes in the denser forest).
  - l_av within about 5 % of the paper's.
  - Exact crash counts are not expected because the plant is kinematic rather than Gazebo and sensing is perfect rather than LiDAR ray-tracing. These deviations get listed in the hub.

### 6. `scripts/s3b_nominal_and_baseline.py`: our forest
- World: `worlds/forest_seed0.json`.
- Tasks: the original start (3,3) → goal (47,47), plus 49 random start/goal pairs. Each pair is ≥ 30 m apart, both ends have clearance > 0.6 m, and the pair is connected (reuse `world.path_exists`). Pairs are saved to `worlds/forest_seed0_tasks.json` for reuse in Stages 4–5.
- Controllers:
  - `nominal`: Gaussian MPPI, goal cost only.
  - `baseline`: log-MPPI plus `CrashEgoWindow`.
- Shared settings: 50 Hz, 6 s horizon, K = 2496, v ∈ [0, 1], ω ∈ [−1, 1], Q = [5, 5, 2] (the repo's v ≤ 1 setting), repo λ and std.
- Timeout: 3 × straight-line distance / 1 m/s.
- The nominal is expected to crash often; it is the reference that Stage 4 filters. Optional: also log nominal "virtual" runs where collisions are recorded but the run continues, to count obstacle contacts.
- Metrics per controller: success rate, crash rate, timeout rate, path length, mean speed, time to goal, min clearance, ms per tick.
- Figures: map with all trajectories per controller, a metric bar/box chart, and one example run with the 5 m window drawn.

### 7. Hub update (`SimValidation/index.html`)
- Stage 1 open questions are relabelled "Notes to revisit (deferred)".
- Stage cards: Stage 3 becomes "in review"; Stage 2 stays planned.
- New Stage 3 section:
  - Repo summary and the parameters actually used.
  - Paper-vs-ours reproduction table plus figures.
  - Deviation list (kinematic plant, perfect sensing, 6 s vs 5 s horizon, tree footprint from the mesh slice).
  - Our nominal and baseline setup, metrics and figures, and findings.
- File index and session log (2026-11-19) updated.
- The memory note gets an updated status line.

## Verification
1. `python -m mppi.selftest`:
   - The SG convolution matches scipy to 1e-5.
   - The NLN sample std matches the analytic `NLN()` variance.
   - In an empty world, both configs reach a goal 20 m away with no crash.
2. Run `s3a` and compare against the Table II targets above. Check the plots for sane trajectories.
3. Run `s3b`. Check that the nominal goes straight through obstacles and the baseline avoids most of them. Spot-check one crash by inspecting the window at the crash tick.
4. Open `index.html` and confirm the figures render and the numbers match the JSON files.

## Runtime estimate
K = 2496 × N = 300 sequential steps per tick, batched over 10–50 tasks on MPS. That is roughly 50–150 ms per tick, so about 10–30 min per (world, controller) batch. Long runs go in the background.

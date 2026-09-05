# The roll-over stack — manual

This is the working manual for the roll-over experiment in this fork. It documents
`mimoEnv/envs/roll_over.py`, its callbacks, `mimoEnv/illustrations.py` and
`mimoEnv/eval_rollover.py`. [`../README.md`](../README.md) covers installation and the two
commands you need to run something; everything here assumes you got that far.

It is a standalone document, not part of the Sphinx build in `docs/source/`.

Two conventions used throughout:

- Every warning carries the measurement that produced it. Numbers taken from code comments are
  attributed to the file that holds them; numbers marked *(measured here)* were re-run while
  writing this document and the command is given. A warning without evidence gets deleted in the
  next refactor, so if you add one, add its evidence.
- "How it is" and "how it was meant" are kept apart. Where the code does not do the obvious thing,
  §8 states what it actually does, and names the intention separately only when that intention is
  recoverable from the comments — never smoothed over.

**Contents**

1. [The task and what ρ measures](#1-the-task-and-what-ρ-measures)
2. [The environment](#2-the-environment)
3. [Reward and goal configuration](#3-reward-and-goal-configuration)
4. [The GoalEnv / HER stack](#4-the-goalenv--her-stack)
5. [Training loop, callbacks and TensorBoard tags](#5-training-loop-callbacks-and-tensorboard-tags)
6. [Evaluation protocol](#6-evaluation-protocol)
7. [`data.yml` round-tripping](#7-datayml-round-tripping)
8. [Divergences: how it is vs. how it was meant](#8-divergences-how-it-is-vs-how-it-was-meant)
9. [CLI flag reference](#9-cli-flag-reference)
10. [Analysis scripts and cluster scripts](#10-analysis-scripts-and-cluster-scripts)

---

## 1. The task and what ρ measures

MIMo is dropped onto a flat floor either prone (face down) or supine (face up) and has to roll
onto the other side. The episode is a fixed 500 steps (`max_episode_steps=500` in
`mimoEnv/__init__.py:75`), `frame_skip=2`, `dt = 0.01 s`, so an episode is 5 simulated seconds.

The scalar progress measure — written **ρ** (rho) in the logs, the callbacks and this document —
is `get_achieved_goal_cos_mean()` (`roll_over.py:863`):

```
d       = data.body(b).xmat[2, 0]         # dot product of the body's local x axis with global z
d      *= -1                              # only when starting supine
rho_b   = (d + 1) / 2
rho     = (rho_hip + rho_chest) / 2       # b in {"hip", "chest"}
```

MIMo's local x axis points from back to belly. Prone it points at the floor (`d = -1`), supine at
the ceiling (`d = +1`). The `*= -1` for supine makes ρ **task progress relative to the starting
posture**: ρ ≈ 0 at reset and ρ ≈ 1 on a completed roll, in *both* postures.

Consequences you need to hold on to:

- **ρ cannot tell prone from supine.** For absolute posture use the raw
  `get_dot_local_x_to_global_z("hip"/"chest")`: +1 supine, −1 prone. Any analysis script that
  compares postures must use the raw dot product, not ρ.
- ρ is linear in the cosine, not in the angle. ρ = 0.5 is a 90° roll ("side lying"); ρ = 0.25 is
  a 60° roll, **not** 45° despite the `info` key being called `45_deg`
  (`roll_over.py:982`; cos 120° = −0.5 → ρ = 0.25). The key name is wrong, the threshold is what
  it is.
- Both hip and chest enter the average. This is deliberate: with hip alone MIMo learned to
  twist the pelvis without bringing the torso over (`compute_reward_v1` docstring,
  `roll_over.py:1018`).

Two derived quantities appear in logs and rendering:

| Function | Value | Used by |
|---|---|---|
| `get_rotation_degrees_to_goal_z_axis(body)` (`:775`) | 180° at reset → 0° at goal | internal |
| `get_achieved_rotation_degrees(body)` (`:813`) | 0° at reset → 180° at goal | `info['hip_deg']`, `info['chest_deg']`, `rollout/ep_end_*_deg_mean` |

There are three goal-achievement functions; `cos` is the default and the only one all recent runs
use. `angle` is **currently broken** — see [§8](#angle-is-inverted-and-succeeds-at-reset).

---

## 2. The environment

`MIMoRollOverEnv` (`mimoEnv/envs/roll_over.py:127`) subclasses `MIMoEnv`
(`mimoEnv/envs/mimo_env.py:198`), which is a `MujocoEnv` with a **Dict** observation space
assembled from independently configurable sensor modules. Registered as `MIMoRollOver-v0`; call
`import mimoEnv` before `gym.make`.

### 2.1 Scenes and ages

Ages are **pre-generated scenes, not runtime growth.** The constructor picks

```
mimoEnv/assets/roll_over/prone/scene_act_<physio_age>_body_<morph_age>.xml
```

for `physio_age, morph_age ∈ AGES = [1, 3, 6, 9]` (`roll_over.py:35`, `:246`) and then passes
`age=None` up to `MIMoEnv` (`:326`), which skips `mimoGrowth.adjust_mimo_to_age` entirely
(`mimo_env.py:378`). `mimoGrowth` writes a *temporary* scene XML and deletes it after loading
(`mimo_env.py:391`); parallel runs on the cluster raced on creating and deleting the same
temporary file, which is why roll-over bypasses it. All 16 combinations exist as checked-in files.

Adding an age means adding `mimoEnv/assets/mimo/age/{act,body}/*_mo.xml` and the corresponding
scene files, and extending `AGES` in **both** `roll_over.py:35` and
`mimoEnv/envs/morphological_curriculum.py:4`. It does not mean touching `mimoGrowth`.

The two ages are independent by design, and the split is visible in the scene file: each scene
includes `../../mimo/age/act/act_<physio>_mo.xml` (actuators — physiological age) and, inside the
`mimo_location` body, `../../mimo/age/body/body_<morph>_mo.xml` (kinematic tree, geometry, masses
— morphological age). So an actuation-9-month MIMo can be put in a 1-month body.

### 2.2 Both postures use the `prone/` scene directory

There is no `supine/` directory, and the scene contains nothing posture-specific — a floor plane,
two lights, five cameras and MIMo, whose root body carries `euler="0 90 0"`. Supine is produced at
reset instead: `get_starting_quat()` (`:558`) sets the y Euler angle to `+π/2` for prone and
`−π/2` for supine. The one thing that has to be patched afterwards is the `top` camera, which the
scene rotates by −90° about z; `fix_top_camera_rotation_supine` (`:350`) overwrites its quaternion
with the identity for supine runs, otherwise MIMo's head ends up at the bottom of the frame. The
directory name is vestigial.

### 2.3 Reset

`reset_model()` (`:628`) runs, in this order:

1. flip `starting_position` if `--roll_over_starting_position=alternating`;
2. clear the running ρ maximum of the finished episode;
3. `self.goal = self.sample_goal()` — **per episode**, see [§4.4](#44-goal-sampling-and-the-rng);
4. clear `_prev_achieved_goal`;
5. `put_in_starting_position()` (`:582`):
   - joint noise: `qpos[7:] += U(−0.01, +0.01)` from `self.np_random`, or a fixed array if
     `deterministic_initial_state_sampling` (DISS) is set;
   - root orientation from `get_starting_quat()`;
   - drop to the floor: `mj_forward`, then offset z by `get_minimal_z_coordinate` plus a 1 mm
     safety margin;
   - zero all velocities;
   - `steps_after_reset` null-action physics steps — **default 30** (`:170`). The comment beside
     it says 20; 30 is what runs. The reason for it is the vestibular signal, which does not
     settle within the first ~20 steps.
6. reset `pbrs_last_state_potential` to 0.

`reset()` (`:953`) then adds `chest_deg`, `hip_deg`, `side_lying`, `45_deg`, `ctrl_cost=0` to the
info dict, because the evaluation frame-capture logic reads those keys on the very first frame.

### 2.4 Observation layout

Roll-over enables proprioception (default components) and vestibular; vision is off and touch is
off unless `--touch`. Measured on the default configuration (`9M/9M`, no touch,
`achieved_goal_in_observation=True`):

| Key | Shape | Note |
|---|---|---|
| `observation` | (305,) | proprioception |
| `vestibular` | (6,) | 3 accelerometer + 3 gyroscope |
| `achieved_goal` | (1,) or (7,) | only if `--achieved_goal_in_observation` (forced on by `--her`) |
| `desired_goal` | (1,) or (7,) | always — `goals_in_observation=True` is hard-coded (`roll_over.py:333`) |

Both goal keys are flat `Box` spaces of shape (`goal_dim`,): **(1,) under
`--goal_achievement_function=cos`**, the scalar ρ, and **(2,) under `gravity`** (§3.5), one entry
per body in `GRAVITY_GOAL_BODIES`. The third vector goal, `intrinsic` (§3.4), was removed on
26.08.2026.

**Proprioception is a sorted-key concatenation, so `qpos` does not start at index 0.**
`mimoProprioception/proprio.py:177` concatenates `sensor_outputs` in sorted key order, which for
the default component set gives

```
actuation(92) | limits(49) | qpos(49) | qvel(49) | torques(66)   = 305
                            ^ joint angles begin at index 141
```

*(measured here: build the env and print `env.proprioception.sensor_outputs`.)*
`observation[:49]` reads actuator commands, not joint angles.

**The `qpos` block excludes MIMo's 7-DoF root free joint** (`data.qpos` has 56 entries, 49 of
which are MIMo's named `robot:` joints). Proprioception therefore carries **no body orientation at
all**. Anything defined by orientation — which is the entire roll-over task — has to come from the
vestibular block or from `achieved_goal`.

`--proprio_config` selects the components (`position|velocity|torque|limits|actuation`, joined by
`|`), so any layout assumption in downstream code has to be re-derived, not hard-coded. Locate the
block and probe it against `data.qpos` rather than trusting an offset.

### 2.5 Action space

Derived from the XML: every actuator whose name starts with `act:` (`mimo_env.py:455`), with
bounds taken from `model.actuator_forcerange` (`mimoActuation/actuation.py:142`). For the 9M/9M
roll-over scene that is **46 actuators**, and the bounds are **not all [−1, 1]**: the age meta-XMLs
contain asymmetric ranges (`grep -o 'forcerange="[^"]*"' mimoEnv/assets/mimo/age/act/act_9_mo.xml`
gives 46 entries, of which 22 are `-1 1` and 24 are asymmetric — `-1 .3`, `-1 .765`, `-.65 1`,
`-.6875 1` and so on; MIMo is weaker in one direction of those joints). Code that
assumes a symmetric [−1, 1] box — e.g. when sampling exploration noise — is wrong at the edges.
Use `env.action_space.low/high`.

### 2.6 ISR and DISS

**ISR** (Initial State Randomization, `--isr`) replaces the fixed starting roll with a random one:
`euler[0] = Beta(1, 3) · π`, i.e. 0–180° biased towards small angles (`roll_over.py:571`). `ISRCallback`
switches it off via `env_method("disable_isr")` at 75 % of training so the final policy is
comparable with non-ISR runs (`mimoEnv/envs/isr_callback.py`), and both `illustrations.test()` and
`eval_rollover.py` disable it outright.

> **ISR inflates ρ_max and must be off for any reported number.** Training logs showing ρ_max ≈
> 0.94 turned out to be pure ISR artefacts — some episodes simply *start* nearly rolled. The same
> policies read 0.26–0.36 with ISR off. (`mimoEnv/eval_rollover.py:7`.) This invalidated a whole
> round of results once.

> **ISR is not reproducible from `env.reset(seed=…)`.** `get_starting_quat` draws from the global
> `np.random`, not from `self.np_random` (`roll_over.py:571`), so seeding the environment does not
> pin the starting angle. `GaussianNoiseObsWrapper` has the same property
> (`gaussiannoiseobswrapper.py`, `np.random.normal`). The joint noise and the goal sampling *do*
> use `self.np_random` and are seeded. Related, from the same family of traps:
> `env.reset(seed=s)` never seeds the action space — call `env.action_space.seed(s)` explicitly if
> a rollout uses `action_space.sample()`.

**DISS** (`deterministic_initial_state_sampling`, `:321`) is not a CLI flag: set the attribute to
an array of shape `qpos[7:].shape` and the per-joint reset noise is pinned to it instead of being
sampled. Used for reproducible evaluation; `results/diss/` holds the analyses that used it.

### 2.7 Hot-swapping the embodiment

`set_embodiment(morph_age, physio_age)` (`:395`) rebuilds `self.model`/`self.data` from a
different scene XML, re-runs `initialize()`, resets MuJoCo and calls `reset()`. It is driven by
the morphological growth curriculum (`--mgc`, `mimoEnv/envs/morphological_curriculum.py`):

| `--mgc` | Behaviour |
|---|---|
| `growth` | 1M → 3M → 6M → 9M, 250 000 steps per phase |
| `inverse` | 9M → 6M → 3M → 1M, same phase length |
| `stochastic` | uniform random age from `AGES` every `--mgc_stochastic_interval` steps (default 20 000) |
| `none` | no callback at all (baseline) |

All three call `set_embodiment(age, age)` — morphological and physiological age are always swapped
**together**, even though the method takes them separately. The swap only fires on episode
boundaries (`any(self.locals["dones"])`), because replacing the model mid-episode corrupts the
simulation state.

---

## 3. Reward and goal configuration

The reward is configurable along six axes that interact. `compute_reward`
(`roll_over.py:1048`) is, in order:

```
success   = achieved_goal >= desired_goal

sparse_reward:      r = 0 if success else -1
else, pbrs:         r = pbrs_w * (P(ag, dg) - P(prev_ag, dg))     with P(a, d) = -|d - a|
else:               r = P(ag, dg)
                    r = reward_success (=500) where success            # not in the sparse branch
r  -= ctrl_cost                                                        # 0 if --nopen
```

with `ctrl_cost = pen_factor * sum(data.ctrl²)` (`:1015`).

### 3.1 The axes

| Flag | Effect | Interaction to know |
|---|---|---|
| `--pbrs` (+ `--pbrs_w`, default 100) | potential *difference* shaping instead of the raw potential | forbidden with `--no_done_active`; pointless with `--sparse_reward` (the sparse branch is checked first) |
| `--sparse_reward` | {0, −1} only | overrides `--pbrs` silently; the control cost still applies unless `--nopen` |
| `--pen_factor` (default 0.02), `--nopen` | weight of the quadratic control cost | goal-*independent*, which is why it travels through `info` under HER (§4.2) |
| `--goal_low` / `--goal_high` | sample the target per episode instead of a fixed 0.95 | both or neither, else `ValueError`; makes `rollout/success_rate` unreadable (§5) |
| `--side_lying` | success at ρ ≥ 0.5 instead of 0.95, implemented by `sample_goal` returning 0.5 | ignored when `--goal_low`/`--goal_high` are set — the sampled range wins (`:722` is checked before `:734`) |

`--pbrs_w` defaults to 100 because the raw potential difference between two consecutive steps is
tiny; without a large weight the shaping signal does not drive learning at all
(`roll_over.py:168`).

### 3.2 Success

Success is `achieved_goal >= desired_goal`, evaluated by `is_success` (`:508`). The threshold
lives **in the goal**, not in the check: `sample_goal` (`:687`) returns

- `0.95` by default,
- `0.5` with `success_at_side_lying` (`--side_lying`),
- `U(goal_low, effective_goal_high)` when a range is configured.

Under `--goal_tolerance` the threshold becomes a band, `|achieved − desired| ≤ tolerance`
(§3.6). Both rules live in `_success_mask`; `is_success` is a thin wrapper that returns a plain
`bool` for a single goal and a `(N,)` array for a batch.

Moving the threshold out of `is_success` is what makes `is_success` a pure function of its
arguments, which is the precondition for HER (§4.1).

`_is_done` (`mimo_env.py:902`) terminates the episode when `done_active` and `is_success` — so
under a sampled goal an episode ends as soon as the *sampled* target is met, which for
`--goal_low=0.25` can be a 60° roll.

### 3.3 Combinations that are refused

Both guards live in `illustrations.py:main`, **not in the environment**, so a script that builds
the env directly (`results/utils.py`, `eval_rollover.py`, `goalenv_check.py`) can construct these
configurations without complaint:

- **`--pbrs` with `--no_done_active`** (`illustrations.py:762`) — `get_potential()` jumps to
  `+reward_success` inside the goal region. That is only safe while reaching the goal ends the
  episode. Measured before the guard existed: potential 500.0 inside the goal, −0.01 just
  outside, i.e. a single step leaving the goal region paid **−50001.0**, and the critic loss
  reached ~2.8e7 within 1000 updates.
There used to be a second guard, **`--her` with `--goal_achievement_function=intrinsic`**. Both
it and that goal function are gone (§3.4).

The environment validates its own arguments — unknown posture, unknown goal function, an age
outside `AGES`, a `goal_low`/`goal_high` mismatch, a non-positive `goal_tolerance` or
`gravity_goal_eps`, and `--goal_tolerance` combined with `gravity` (which is a point goal with a
radius already) — but it knows nothing about the combination above.

### 3.4 The `intrinsic` goal function — a non-scalar, non-extrinsic goal that did not work

**Removed from `roll_over.py` on 26.08.2026**, together with `--intrinsic_goal_joints`,
`--intrinsic_acc_axes` and `--intrinsic_acc_w`. It never worked (the measurements below say why),
and it is recorded here as a negative result. Runs trained with it can no longer be loaded against
the current environment. (`--intrinsic_goal_eps` and `--intrinsic_reference_samples` survive as
aliases of `--gravity_goal_eps`/`--gravity_reference_samples`, which is where those two knobs are
actually read — §3.5.)

Rewritten 19.08.2026. Before that date `intrinsic` meant "the whole observation dict is the goal",
selected by a `--intrinsic_goal` sub-mode (`all`, `vesti`, `vesti_acc`, `sparse_proprio`) and
weighted per modality by `--proprio_w`/`--vesti_w`. Those goals were dicts, which SB3 cannot use
as a goal space and HER cannot relabel; only 3 of 539 stored runs ever used them, two of those
with a value that was no longer in `choices`. The sub-modes and both weight flags are **gone**;
`load_model_yaml` drops the three retired keys with a printed notice (§7).

**The motivation.** `cos` and `angle` are both computed from the first 7 entries of `data.qpos`,
i.e. from MIMo's root free joint — his absolute orientation in the world. Proprioception only
reports joints named `robot:*`, which **excludes** the free joint, so the quantity those goal
functions optimise is not in the observation at all. The `intrinsic` goal is built only from
things MIMo does sense:

| Dimension | Source | Note |
|---|---|---|
| `head_swivel`, `head_tilt_side`, `head_tilt`, `hip_lean1`, `hip_rot1`, `hip_bend1` | `data.qpos[jnt_qposadr[id]]`, mapped from each joint's own `jnt_range` to [−1, 1] | all 1-DoF hinges; the same scalar a **Joint** slider in `mujoco.viewer` writes, and the same number already sitting in the `qpos` block of `obs['observation']` |
| `vestibular_acc_x` | `obs['vestibular'][0] / 9.81 * --intrinsic_acc_w` | the only dimension that separates prone from supine |

Deliberately **not** the actuator state (`actuation_model.observations()`, the `actuation(92)`
block of the observation): that is the motor *command*, not the sensed configuration, so a goal
defined on it could be reached by emitting the right control signal while lying perfectly still.

The joint list is `INTRINSIC_GOAL_JOINTS` (`roll_over.py:96`), overridable with
`--intrinsic_goal_joints` (comma-separated, `robot:` prefix optional). The environment raises if a
named joint is not a hinge or has no usable range (`initialize`, `:358`) — the goal normalises by
range, and a ball or free joint has no single scalar to normalise.

> **The accelerometer axis is x, not z.** *(measured here)* The MuJoCo `accelerometer` sensor
> reports in the **site's local frame**, and the `vestibular` site sits on the head with its local
> x axis pointing along world +z — the same convention as this module's own
> `get_dot_local_x_to_global_z` (§1). At reset:
>
> ```
> prone   acc = [-9.74, -0.02, +0.37]
> supine  acc = [+9.58, -0.10, -0.48]
> ```
>
> Component z is within noise of zero in **both** postures. A goal built on acc-z cannot tell
> prone from supine at all. `--intrinsic_acc_axes` therefore defaults to `x`; it accepts any
> subset of `xyz`, and `''` drops the accelerometer entirely.

**Normalisation is not cosmetic.** Raw units are radians (`hip_rot1` spans ±0.31) against m/s²
(acc spans ±9.81); an unweighted euclidean distance over the raw values would be ~97 %
accelerometer. Each joint is mapped to [−1, 1] by its own range and the accelerometer is divided
by gravity, so every dimension is O(1). `--intrinsic_acc_w` then trades the two blocks against
each other deliberately. The weight is folded into the goal *vector* rather than applied in the
distance, which is what keeps `_potential` a plain norm of its two arguments and therefore pure.

**The reference posture** is recorded once, in the constructor
(`create_prone_and_supine_intrinsic_goal`, `:412`): one vector per posture, and the target for an
episode is the vector recorded in the *opposite* posture. Two details matter:

- It is **averaged over `--intrinsic_reference_samples` resets (default 20) with ISR forced off**.
  A single reset would pin the goal to whatever `head_swivel` happened to settle at under the
  initial joint noise, and MIMo would then be scored on reproducing that draw rather than the
  posture.
- The **RNG state is saved and restored** around the recording. Those resets draw from
  `self.np_random`, the same generator that produces every training episode's initial joint noise,
  so without this the number of samples taken here would silently shift which episodes the run
  sees — the same failure as the `sample_goal` draw in §4.4.

The per-dimension mean and standard deviation are printed at construction and kept in
`intrinsic_goal_std`.

> **The joints carry ~4.6 % of the signal.** *(measured here, n = 10, age 9, normalised units)*
>
> | | head_swivel | head_tilt_side | head_tilt | hip_lean1 | hip_rot1 | hip_bend1 | acc_x |
> |---|---|---|---|---|---|---|---|
> | prone | +0.001 | +0.002 | −0.057 | −0.008 | +0.003 | −0.316 | −0.99 |
> | supine | +0.000 | −0.001 | −0.083 | −0.011 | +0.002 | −0.244 | +0.93 |
>
> Total prone↔supine separation is **1.923**, of which the accelerometer alone contributes
> **1.921**; the six joints contribute ≈0.088, and their per-reset spread (sd up to 0.024) is a
> third of that. In practice this is an accelerometer goal with six near-constant passengers.
> Consequences: `--intrinsic_acc_axes=''` (joints only) is not a viable ablation, and
> `--intrinsic_goal_eps` has to be read against a scale of ~1.9, not ~1.0.

**`eps` is the task definition under a sparse reward.** A continuous vector goal is never matched
exactly, so success is a ball of radius `--intrinsic_goal_eps` (default 0.15) around the
reference. Too large and reset already counts as success; too small and the sparse reward is never
earned. The default has **not** been calibrated against a trained policy, and given the finding
below it may not be calibratable at all: a policy that never rolls still reaches `d ≈ 0.44`, so
the gap between "rolled" and "faked the accelerometer" is much narrower than the 1.923 separation
suggests.

> ### The accelerometer is forgeable — this goal does not currently work
>
> *(measured 19.08.2026; two 1M-step PPO runs, supine, age 9/9, matched to
> `26-08-19_supine_age9_ep250` in every other respect)*
>
> | | intrinsic + PBRS | intrinsic + distance | baseline `cos` + PBRS |
> |---|---|---|---|
> | `ep_rho_max_mean` start → end | 0.012 → **0.019** | 0.012 → **0.038** | 0.085 → **0.951** |
> | `side_lying_success_rate` | **0** | **0** | 0.991 |
> | `ep_end_hip_deg_mean` | 5.3° → 5.2° | 5.9° → **4.6°** | 36° → 161° |
> | `train/value_loss` | 869 → **59** | 761 → **12** | 0.94 → 333 |
>
> Neither variant rolls, and the collapsing value loss says this is **not** an optimisation
> failure: PPO learned the objective well. Rolling out `model_4.zip` deterministically
> (10 episodes) shows what it learned instead:
>
> | | PBRS | distance |
> |---|---|---|
> | accelerometer gap, reset → final | 1.916 → **0.435** (77 % closed) | 1.914 → **0.726** |
> | joint gap, reset → final | 0.089 → **1.075** (12× *worse*) | 0.088 → **0.951** |
> | raw `acc_x` at episode end | **−5.49 m/s²** | −2.61 m/s² |
> | ρ_max | **0.0045** | 0.038 |
>
> A MuJoCo `accelerometer` reports **specific force — gravity plus self-acceleration**. Lying
> supine the head site statically reads `acc_x ≈ +9.6`; the prone reference is `−9.7`. The policy
> drives it to **−5.49 without ever rolling** — roughly 15 m/s² of self-generated acceleration,
> forging 77 % of prone's gravity signature by shaking. It simultaneously moves the six joints
> *away* from the target, because they are worth almost nothing in the distance.
>
> **No hyperparameter fixes this.** A smaller `--intrinsic_acc_w` shrinks signal and exploit
> equally, and the joints span only 0.088 of the 1.923 separation, so they cannot carry the goal
> alone. The two candidate fixes, both still within "only what MIMo can sense":
>
> 1. **Low-pass the accelerometer** across the episode — gravity is DC, self-acceleration is not.
>    A running mean over ~50 steps suppresses the forgery and keeps the posture signal. Cheapest,
>    and stays inside the sensor list the note specifies.
> 2. **Integrate the gyroscope into an orientation estimate**, as a real vestibular system does.
>    Closer to the biology, considerably more work, drift-prone.
>
> **The joint dimensions are anti-correlated with the task**, which is worse than being merely
> uninformative. *(measured 20.08.2026, cos baseline that rolls 12/12, mean rho_max 0.996)*
>
> | dimension | desired | @reset | \|d\| | @roll (rho>=0.95) | \|d\| |
> |---|---|---|---|---|---|
> | head_swivel | -0.001 | 0.001 | **0.002** | 0.161 | 0.162 |
> | head_tilt_side | -0.002 | -0.001 | **0.001** | 0.015 | 0.017 |
> | head_tilt | -0.059 | -0.087 | **0.028** | -0.061 | 0.002 |
> | hip_lean1 | -0.005 | -0.007 | **0.002** | 0.966 | **0.971** |
> | hip_rot1 | 0.006 | 0.002 | **0.003** | 0.164 | 0.159 |
> | hip_bend1 | -0.313 | -0.242 | **0.070** | -0.624 | 0.311 |
> | acc_x | -0.993 | 0.921 | 1.914 | -0.344 | 0.649 |
> | | **joint-part** | **acc-part** | **total d** | | |
> | reset | 0.081 | 1.914 | 1.916 | | |
> | at roll | **1.076** | 0.892 | **1.434** | | |
>
> At reset the six joints are already *at* the goal -- 0.081 of a 1.916 total distance. In both
> postures MIMo lies relaxed and the joints settle into their spring-driven rest position, which
> is the same lying either way. The joint half of the goal therefore reads "lie still", and it is
> solved at t = 0 by doing nothing.
>
> Rolling makes it **worse**: the joint distance rises from 0.081 to 1.076, `hip_lean1` alone
> going from |d| = 0.002 to 0.971. That is not an artefact -- rolling *is* actively bending the
> trunk, and the target is the relaxed posture afterwards.
>
> **So `eps` cannot be calibrated at all.** A genuine roll sits at d = 1.434; the non-rolling
> intrinsic policy of the runs above reached d_min = 0.38-0.74. The policy that does not roll gets
> *closer* to the goal than the one that does, so the window
> `max(d_min over real rolls) <= eps < min(d at reset)` is not merely empty -- it is inverted. No
> threshold separates rolled from not-rolled.
>
> Until one of them is in, treat `--goal_achievement_function=intrinsic` as **not working**, and
> expect the same failure under `--sparse_reward` + `--her`: the exploit is a property of the
> goal, not of the reward shaping. Note that fix (1) -- low-passing the accelerometer -- addresses
> only the forgery; the joint dimensions would still be anti-correlated, so the reference posture
> itself needs rethinking (a posture recorded *mid-roll* rather than at rest, or dropping the
> joints from the goal entirely).

**HER works with this goal**, and needs no `--goal_low`/`--goal_high` range
(§4.5). Those exist for the scalar goals because a constant `desired_goal` gives the policy no
reason to condition on it. Here every relabelled goal is an achieved posture vector from the
trajectory, so goal variation is automatic. `--no_done_active` is still required (§4.6).



### 3.5 The `gravity` goal function

Added 20.08.2026 after the measurements in 3.4 showed `intrinsic` to be unfixable by tuning,
**removed on 26.08.2026, and readded the same day.** It is a supported goal function again:
`--goal_achievement_function=gravity`, with `--gravity_goal_eps` (previously
`--intrinsic_goal_eps`) as the success radius. Everything below stands, including the runs.

> **Why it came back.** The removal argument was that `gravity`'s *mean over the two bodies* is
> the same quantity rho measures, so it made the same claim about sensing at the cost of a second
> training configuration -- and that claim is now made without training at all, by
> `results/intrinsic/intrinsic_rho_check.py` (see the end of this section). What the argument
> missed is that **HER does not see the mean, it sees the vector.** The goal is two-dimensional
> and its success criterion is a ball rather than a threshold, and under a sparse reward those
> are exactly the two properties that decide what a relabelled transition is worth: `gravity`
> trains without `--goal_low`/`--goal_high` where `cos` needs them, 14/16 seeds against 3/16
> (3.6). `--goal_tolerance` gives `cos` the ball criterion alone; it cannot give it the second
> dimension, and `(hip -1, chest +1)` is indistinguishable from `(hip 0, chest 0)` once averaged.

The idea is to stop reading a *posture* off an *acceleration* sensor. The accelerometer can only
do that at rest; the gyroscope measures rotation directly and is structurally untouched by linear
acceleration. So the gravity direction is **integrated**, not measured:

```
t = 0, MIMo demonstrably at rest:  g_site <- normalize(accelerometer)
every step, gyroscope only:        g_site <- rotate(g_site, -omega * dt)     # Rodrigues
goal:                              [ (R_body^T R_site @ g_site)[0]
                                     for body in GRAVITY_GOAL_BODIES ]       # hip, chest
```

+1 supine, -1 prone per body -- the same scale as `get_dot_local_x_to_global_z(body)`, but
reconstructed from what MIMo senses. `R_body^T R_site` depends only on the joints between that
body and the head, so the root free joint cancels (measured residual 0.017 deg) and the goal stays
non-extrinsic. It is configured by `--goal_achievement_function=gravity`, with
`--gravity_goal_eps` as the success radius, where `eps = 2*(1 - rho_target)` per body.
`--intrinsic_goal_eps` still works as an alias, and `load_model_yaml` renames the stored key, so
runs from 20.-22.08.2026 reload with their own radius rather than the default.

Why each defect of `intrinsic` disappears: shaking has no purchase because linear acceleration
never enters the integration; head turning cancels as an *identity* rather than being
counterweighted (the gyro rotates `g_site` by exactly the amount that leaves `R_rel @ g_site`
fixed); and the joints stop being goal dimensions, so the anti-correlation of 3.4 is gone.

**Validated offline before implementation**, replaying recorded rollouts: correlation with the
truth 0.9991 (min 0.9978) on a policy that rolls, and 0.004 drift over a full 250-step episode on
the policy that games the old goal. Re-measured on the trained policy afterwards: mean |error|
0.0355 over 6x250 steps with no accumulation (0.036 / 0.040 / 0.045 / 0.036 / 0.022 per 50-step
block). **Sensing is not the bottleneck.**

#### Results

*(supine, age 9/9, PPO + PBRS(100), pen 0.02, 250-step episodes, matched to
`26-08-19_supine_age9_ep250` in everything but the goal function)*

| run | steps | rho_max | success | side | hip | chest | gap |
|---|---|---|---|---|---|---|---|
| `intrinsic` (3.4) | 1M | 0.019 | 0.014 | 0.00 | 5.2 | 10.3 | -5.1 |
| `gravity`, hip only | 1M | 0.385 | 0.000 | 0.00 | 101.9 | 42.5 | **59.4** |
| `gravity`, hip+chest | 1M | 0.482 | 0.012 | 0.19 | 107.4 | 60.5 | 46.9 |
| `cos` baseline | 1M | 0.952 | 0.984 | 0.99 | 161.0 | 151.0 | 10.0 |

rho per 200k steps:

```
gravity hip+chest   0.162  0.369  0.387  0.397  0.444   <- still rising at 1M
gravity hip only    0.185  0.372  0.384  0.386  0.384   <- plateau from 400k
cos baseline        0.212  0.826  0.933  0.948  0.952
```

Two findings worth keeping:

- **The hip-only version reproduced a failure this fork already knew about.** Measuring only the
  hip made MIMo twist the pelvis and leave the torso lying -- a 59 deg gap. That is exactly why
  `cos` averages hip *and* chest; see the `compute_reward_v1` docstring. Adding the chest closed a
  third of the gap and lifted rho by 25%.
- **Two dimensions, not their average.** An average cannot distinguish (hip -1, chest +1) from
  (hip 0, chest 0); the vector distance penalises the gap directly.

The `gravity` hip+chest run above was deleted by accident on 22.08.2026 and only these summary
numbers survive -- the event file is gone, so it cannot be re-plotted.

#### It works: the non-extrinsic goal matches the extrinsic baseline

Once the potential is normalised (below), `gravity` solves the task.

| | rho_max | success | side | hip | chest | gap |
|---|---|---|---|---|---|---|
| **`gravity` hip+chest, normalised, 2M** | **0.938** | **0.997** | **1.00** | 165.9 | 141.3 | 24.6 |
| `cos` baseline, 1M | 0.952 | 0.984 | 0.99 | 161.0 | 151.0 | 10.0 |

Under the fixed protocol of section 6 (`eval_rollover.py`, 50 deterministic episodes, ISR off):

| | full roll | rho_max mean/min | steps to roll |
|---|---|---|---|
| **`gravity`** | **100 %** | 0.995 / 0.985 | **41.2 +- 8.7** |
| `cos` baseline | 100 % | 0.995 / 0.988 | 51.3 +- 8.9 |

**MIMo learns to roll from a goal built only out of what he can sense** -- no privileged access to
the root free joint -- as reliably as from the hand-designed extrinsic one, and in ~20 % fewer
steps.

rho per 250k steps shows the normalisation, not the extra million steps, was the fix:

```
gravity normalised   0.229  0.696  0.928  0.930  0.928  0.932  0.938  0.938
cos baseline         0.308  0.897  0.944  0.952  0.958    -      -      -
```

It is at 0.928 by **750k** steps, where the un-normalised version stood at 0.482 after a full 1M.
The second million bought nothing; 1M is enough for this configuration.

Caveats, because none of these are covered yet:

- **One seed per configuration.** The fork's convention elsewhere is 6-18 seeds (`*_run_N`). The
  20 % speed advantage in particular rests on a single pair of runs.
- **PPO + PBRS only.** The sparse `{0,-1}` + SAC + HER configuration that motivated the whole
  intrinsic-goal line has not been run against `gravity`.
- **`--obs_noise` untested.** `GaussianNoiseObsWrapper` perturbs the observation, not
  `data.sensordata`, so the goal computation inside the env does not currently see any noise at
  all. Whether it should is a design decision nobody has taken.

#### The potential is normalised

`cos` measures progress in [0, 1]; `gravity` runs from +1 to -1 per body, so with two bodies a
reset sits at distance 2.83. With `--pbrs_w=100` and `reward_success=500` unchanged, the shaping
term was 2.83x larger than in the baseline while the terminal bonus was not -- i.e. the +500
attractor was relatively 2.83x weaker, which fits a policy that farms shaping reward to rho 0.48
and does not pay for the last part of the roll. `_potential_scale` divides by the reset distance
`2*sqrt(n_bodies)`, so the two goal functions are comparable. It deliberately does **not** scale
`--gravity_goal_eps`, which stays in the readable +-1 units of the goal.

So `_potential_scale` returns `1/(2*sqrt(n_bodies))` for `gravity` and 1.0 for `cos`. It is a
property rather than an inlined constant, because the failure above is exactly what the next goal
function on a different scale would walk into.

#### What carries the sensing claim

The claim the goal function was originally built to make is a claim about *sensing*, not about
training: the quantity rho is defined on -- `data.body(b).xmat[2,0]`, read off the root free
joint -- is reconstructible from the vestibular sensors and the joint angles. A training
configuration is an expensive way to state that, which is what motivated the (reverted) removal.

`results/intrinsic/intrinsic_rho_check.py` states it directly instead, and remains the citation
for it: the goal function's own job is the HER behaviour above, not this. It reimplements the
estimator standalone -- nothing in it imports the environment's goal code -- and measures it
against the truth step by step along real rollouts:

* `g_site` is seeded from the accelerometer once, at reset, and afterwards integrated from the
  gyroscope alone (Rodrigues), exactly as above;
* `R_b^T R_site` comes from **forward kinematics on the joint angles alone**: `mj_kinematics` on a
  scratch `MjData` whose root free joint is pinned to the identity pose. `qpos[7:]` is the
  proprioceptive qpos block; `qpos[:7]` is overwritten and never read. This is stronger than the
  original implementation, which took the same rotation out of the live simulation and argued that
  the root cancels -- here it cannot enter at all.

Measured 26.08.2026, 10 episodes per policy, `--episodes=10`:

| policy | posture | steps | hip mean \|err\| | chest mean \|err\| | rho mean \|err\| | rho corr | rho_max mean \|err\| | same success call |
|---|---|---|---|---|---|---|---|---|
| `cos`, PPO+PBRS (`26-08-19_supine_age9_ep250_run_0`) | supine | 2510 | 0.0227 | 0.0154 | 0.0060 | 0.9998 | 0.0013 | 10/10 |
| `cos`, SAC+HER sparse (`26-08-24_prone_sac_her_ep200_run_0`, `model_5`) | prone | 2010 | 0.0094 | 0.0095 | 0.0046 | 0.9998 | 0.0007 | 10/10 |
| `gravity`, PPO+PBRS (`26-08-22_supine_gravity2n_ppo_pbrs`, `model_8`) | supine | 2510 | 0.0173 | 0.0095 | 0.0051 | 0.9997 | 0.0009 | 10/10 |

`d_b` spans 2.0 from prone to supine, so a mean error of 0.02 is 1 % of the range. On rho the
error is 0.005 and the worst single step across all three policies is 0.040. **The two signals
make the same success call at the 0.95 roll threshold in every episode**, which is the only
comparison that matters if the reconstruction is to stand in for the goal. Error does not
accumulate: the last decile of each episode is no worse than the first (0.0041 vs 0.0107 supine,
0.0015 vs 0.0018 prone). The two `cos` policies never saw the intrinsic signal in training, so
this is not a policy that learned to make its own estimator look good.

The script also runs the root-invariance check every step, comparing its forward-kinematics frames
against the same relative rotation taken from the simulation: **max 0.0000 deg** over all three
runs. Nothing about where MIMo lies in the world enters the estimate.

> **Trap this measurement walked into first, worth knowing for any script here that reads
> `data.xmat` after a step.** `mj_step` evaluates the forward dynamics at the state it is given
> and integrates `qpos` afterwards, so on return `data.xmat`, the site frames and `data.sensordata`
> describe the state *before* the last integration step while `data.qpos` describes the state
> after it. Reading the truth from `xmat` and the estimate from `qpos` therefore compares two
> instants one control step apart: it showed up as a 4.01 deg root-invariance residual that looked
> exactly like a leak of root orientation, and it is almost certainly the source of the 0.017 deg
> figure quoted above and of most of the 0.0355 mean error re-measured on the trained policy. One
> `mj_forward` after each step -- a pure recomputation of derived quantities, which `mj_step` does
> again at the start of the next step -- takes the residual to 4e-6 deg.

### 3.6 `--goal_tolerance` -- the scalar success test as a band

Added 25.08.2026, to test *why* `gravity` trains under SAC+HER without `--goal_low`/`--goal_high`
while `cos` does not (14/16 seeds against 3/16, `26-08-22 ... sac_her_gravity` vs
`26-08-23 ... sac_her_ep200_nolohi`).

**The measurement it comes out of.** Roll out a policy, rebuild HER's `future` relabelling by
hand, and count how many relabelled transitions score 0 (`_success_mask` is pure, so this needs no
training):

| | achieved-goal span | relabelled transitions scoring 0 |
|---|---|---|
| `cos`, random policy | rho 0 .. 0.007 | **41 %** (49 % at gap < 10 steps, 24 % at gap > 70) |
| `cos`, 200k checkpoint | rho 0 .. 0.18 | 54 % |
| `gravity`, random policy | 0.978 .. 1.0 per body | **100 %** |
| `gravity`, 200k checkpoint | 0.95 .. 1.0 per body | **100 %** |
| `gravity`, 1M checkpoint (rolls) | -1.0 .. 1.0 | 44 % |

So HER's relabelling is *blind* for `gravity` until the policy already moves further than
`gravity_goal_eps` -- and that run reaches rho 0.99 anyway. The relabelled batch is not where
its advantage comes from, which kills the obvious explanation ("the vector goal gives HER
automatic goal variation"; it does, but the variation carries no reward contrast).

**The hypothesis that survives.** With 80 % of the batch relabelled onto goals near the current
posture and scoring 0, and the remaining 20 % at the distant real goal scoring -1, the only
feature separating the two classes is the distance to the goal -- so the critic has to represent
it, and the actor climbing that estimate gets a dense, potential-like gradient out of a sparse
reward. `cos` cannot build the same estimate: `achieved >= desired` is a knife edge sitting inside
the jitter band of rho, so its near-goal labels are ~half 0 and half -1 and "close" does not look
better than "far".

`--goal_tolerance FLOAT` turns the scalar success test into `|achieved - desired| <= tolerance`
and moves the fixed full-roll goal from 0.95 to **1.0**. At the real goal that is a no-op --
rho is capped at 1.0, so `|rho - 1| <= 0.05` *is* `rho >= 0.95` -- and only the relabelled goals
see a different rule, which is what makes it a controlled A/B. 0.05 is the matched radius:
`gravity` uses 0.15 over two bodies, i.e. 0.106 per body in its +-1 units, and rho is that
quantity halved.

Under `--side_lying` the goal stays 0.5 but then means "stop at side lying" rather than "reach
at least side lying". `_potential` is untouched -- it was a distance already.

The experiment this exists for:

```bash
# 6 seeds, cos, sparse + HER, NO goal range -- against 26-08-23_supine_sac_her_ep200_nolohi
python mimoEnv/illustrations.py --algorithm=SAC --her --sparse_reward --no_done_active \
    --goal_achievement_function=cos --goal_tolerance=0.05 --episode_steps=200 ...
```

If it trains, the mechanism above is confirmed. The counter-test is `gravity` with
`--gravity_goal_eps=0.02`, which should then collapse; it is runnable again since the goal
function came back (3.5). Note the two groups compared above also differ in horizon (100 vs 200
steps), so a `cos` run at 100 steps is still needed to rule that out.

The reported numbers stay comparable either way: `eval_rollover.py` scores `rho_max >= 0.95`
measured off the simulation and never calls `is_success` (`:168`). It does pin `desired_goal` to
1.0 for a run trained with a tolerance, because the policy is conditioned on that input.

---

## 4. The GoalEnv / HER stack

Added 08.2026. It makes `MIMoRollOverEnv` a usable Gymnasium GoalEnv so off-policy algorithms can
train with Hindsight Experience Replay, which answers the research question: **can MIMo learn to
roll from a sparse {0, −1} reward, i.e. without the hand-designed rotation shaping?**

The headline result recorded in the repository: sparse + HER reaches 100 % full roll, against
94 % for the SAC + PBRS baseline (`run_her_sparse.sh:5`, `goalenv_check.py:115`). The write-up
lives outside this repository.

### 4.1 The purity contract — the thing that breaks silently

> **`compute_reward`, `is_success` and `_potential` must be pure functions of
> `(achieved_goal, desired_goal, info)` and must be vectorized.**

Since 19.08.2026 this holds for **every** goal function. All three reshape their arguments to
`(N, goal_dim)` through `_as_goal_batch` (`roll_over.py:480`), so the scalar goals (`goal_dim = 1`) and the
intrinsic posture goal (`goal_dim = 7`) share one code path; the only difference is the success
test in `_success_mask`. The `intrinsic` branch previously had its own live-state implementation,
`_compute_reward_intrinsic`, which is exactly the bug described below — it read `get_potential()`
and ignored its arguments. It has been deleted.

HER rewrites the `desired_goal` of a stored transition and recomputes its reward by calling
`env.compute_reward` on a batch. If any of those functions reads live simulation state
(`self.data`, `self.goal`, `self.get_achieved_goal()`), the recomputed reward is the reward of the
*real* transition, and **relabelling becomes a no-op**: training runs, the loss curves look
plausible, and HER does nothing.

This was a real pre-existing bug, not a hypothetical. `compute_reward` ignored its arguments and
returned the same value for every goal pair (`goalenv_check.py:13`):

```
compute_reward(ag=0.0017, dg=0.00)    = -0.224019
compute_reward(ag=0.0017, dg=0.95)    = -0.224019
compute_reward(FAKE ag=0.99, dg=0.95) = -0.224019
```

**SB3's `check_env` passes on that bug.** It only verifies that
`reward == compute_reward(achieved_goal, desired_goal, info)` for the transition that just
happened, which a function reading the live state satisfies trivially. `mimoEnv/goalenv_check.py`
exists precisely to call `compute_reward` with a goal that differs from the live state. Run it
after any change here:

```bash
MUJOCO_GL=osmesa python mimoEnv/goalenv_check.py
```

Its twelve sections are: SB3 `check_env`; purity (reward varies with each argument, and is
invariant under stepping the sim); vectorization (batch matches elementwise); PBRS regression
(the step reward still equals the original formula, so the 94 % baseline stays comparable);
sparse reward; goal sampling; the `info` contract; end-to-end relabelling; PBRS boundedness under
relabelling; `--goal_tolerance` (§3.6); the episode horizon; and the `gravity` goal function
(§3.5).

**Run them one per process** — `goalenv_check.py --list` prints the names. One env is ~3.8 GB RSS
and `close()` does not return all of it, so running every section in one process OOM-kills a
16 GB machine.

> **Known gap, now closed for `gravity`.** Until 26.08.2026 every section exercised the scalar
> goal only, so nothing in the gate would have caught a regression in a vector goal.
> `test_gravity_goal` covers the branches that differ: the width of the goal space, the ball
> success criterion, the fixed reference goal, the estimate against the extrinsic direction
> cosine it reconstructs, `compute_reward` batched at `goal_dim > 1`, the potential rescale, and
> the RNG restore around the reference recording.

### 4.2 Goal-independent terms travel through `info`

Two reward terms cannot be recomputed from `(achieved_goal, desired_goal)`:

- the **control cost**, which depends on `data.ctrl` and not on the goal at all;
- the **previous achieved goal**, which the PBRS difference needs because a potential difference
  is a function of two consecutive states.

Both are written into `info` during `step()` (`roll_over.py:1004`) and read back by
`_info_column` (`:1154`), which handles a single dict, a sequence of dicts (HER's batched form)
and a missing key, falling back to the live value so ordinary stepping is unchanged.

The previous achieved goal is read back by `_info_block`, the `(N, goal_dim)` counterpart of
`_info_column`. It is kept in that shape although `goal_dim` is 1 -- `_info_column` collapses a
batched fallback to its first element, which was wrong when the goal was a vector and would be
wrong again for the next one.

This is why `HerReplayBuffer` **must** be constructed with `copy_info_dict=True`
(`illustrations.py:917`). Without it the penalty silently drops to zero in every virtual
transition and PBRS cannot be reconstructed at all.

### 4.3 `_potential` is continuous, `get_potential` is not

There are two potentials, and the difference is the point:

- `get_potential()` (`:929`) reads live state and **jumps to `+reward_success` inside the goal
  region**. No reward path uses it any more: its last consumer, `_compute_reward_intrinsic`, was
  deleted on 19.08.2026, together with the `pbrs_last_state_potential` cache that `step()` used to
  maintain for it. It survives only as a diagnostic, read by `goalenv_check.py` and
  `results/collect_observation_util.py`.
- `_potential(achieved, desired)` (`:1127`) is `-‖desired − achieved‖` over the goal dimensions —
  for a scalar goal simply `-|desired − achieved|` — pure and continuous. This is what
  `compute_reward` uses for every goal function.

The jump is unreachable while episodes terminate on success: the previous state of a step can
never be a goal state, and a current goal state is handled by the success branch of
`compute_reward` first. Under HER it *is* reachable, because HER relabels onto rotations MIMo
actually reached mid-trajectory and then drifted back out of. With the jump in place such a
transition paid **−50002.0** (goal 0.40, drift 0.45 → 0.38) and drove SAC's critic loss to
**3.85e7 within 4000 steps** (`roll_over.py:1127`, `goalenv_check.py:216`). Terminating episodes do
not help; they only ever terminate on the *real* goal.

### 4.4 Goal sampling and the RNG

`sample_goal` is called once per episode from `reset_model()`. It must **not** consume a random
draw when the range is degenerate (`roll_over.py:730`):

> The environment draws its initial joint noise from the same generator. A wasted draw shifts
> every later draw, so the same policy under the same seeds sees different start states depending
> on whether the goal was pinned via `--goal_low=--goal_high` or left at the default. Measured:
> **94 % vs 98 %** success for the same policy and the same seeds.

That is also why `eval_rollover.py` pins the goal by passing `goal_low = goal_high = goal` rather
than by any other mechanism — it hits the no-draw path.

### 4.5 The goal-response cliff

A goal-conditioned policy trained under HER is only trustworthy inside the goal range it was
actually trained on, and outside it it is worse than useless:

> **HER only ever relabels onto goals that were actually reached.** A run plateauing at ρ ≈ 0.6
> has no relabelled transition anywhere above 0.6. With `n_sampled_goal=4`, four of five sampled
> transitions are relabelled, so the region above the plateau is trained almost entirely on
> original transitions carrying −1 — the policy does not merely fail there, it learns something
> actively wrong. Measured on the third E3b seed (2026-08-14, deterministic, 30 episodes per row):
>
> | `desired_goal` fed to the policy | resulting ρ_max |
> |---|---|
> | 0.25 | 0.546 |
> | 0.50 | 0.786 |
> | 0.75 | 0.091 |
> | 0.95 | 0.092 |
>
> A policy that ignored `desired_goal` entirely would score 0.786 everywhere. Conditioning on an
> out-of-distribution goal is therefore **worse than not conditioning at all**. The full sweep
> (recorded in the project notes rather than in the code) was monotone up to ≈0.60 and fell off a
> cliff between 0.65 and 0.70.

Reproduce the table with `eval_rollover.py --policy_goal_sweep=0.25:0.95:0.05` (§6.2).

This is the practical reason evaluation always pins the goal (§6): scoring a run at a goal it was
never asked for measures the cliff, not the policy.

> **A goal curriculum used to live here.** `--goal_curriculum` moved the upper end of the sampled
> range along with recent achievement (`quantile` mode), and `--goal_curriculum_mode=alp` added an
> absolute-learning-progress bandit over the range. Both were **removed on 25.08.2026** — they
> carried a lot of code, state and logging for a mechanism that `--goal_tolerance` (§3.6) attacks
> more directly. Runs trained before that date have the flags in their `data.yml` and a
> `rollout/goal_high_effective` curve in their event file; `parser.set_defaults` ignores unknown
> keys, so they still reload.

### 4.6 Known gap: SB3 does not relabel `dones`

`HerReplayBuffer` recomputes `rewards` through `env_method("compute_reward", …)` but takes `dones`
verbatim from the stored transition
(`stable_baselines3/her/her_replay_buffer.py`, in `_get_virtual_samples`: `dones` comes straight
out of `self.dones[batch_indices, env_indices]` while `rewards` is recomputed).
`grep -rn "compute_terminated" stable_baselines3/` returns nothing *(verified against
stable-baselines3 2.5.0 in the `mimo` env)*, even though the Farama GoalEnv API defines exactly
that method for this purpose.

HER's own paper assumes fixed-length episodes where `done` is constant, so the algorithm is fine;
the SB3 implementation silently is not. **That is why `--no_done_active` belongs with `--her`.**
Lifting the restriction would mean implementing `compute_terminated` and subclassing
`HerReplayBuffer`.

Note that this is *advice, not enforcement*: `illustrations.py:736` only prints a warning when
`--her` is used without `--no_done_active`, and `rbi_autorun.sh` currently launches HER runs
without it.

### 4.7 The `info` contract

`step()` writes (`roll_over.py:963`–`:1008`), on top of `is_success`/`is_failure`/`achieved_goal`
from the base class:

| Key | Meaning | Read by |
|---|---|---|
| `chest_deg`, `hip_deg` | achieved rotation in degrees, 0 → 180 | `RollOverCallback`, plots |
| `side_lying` | 1.0 if ρ ≥ 0.5 **at this step** | `RollOverCallback`, `illustrations.test()` |
| `45_deg` | 1.0 if ρ ≥ 0.25 at this step (a 60° roll, see §1) | `illustrations.test()` frame capture |
| `rolled_over` | 1.0 if ρ ≥ 0.95 at this step | analysis; goal-independent success |
| `raw_ctrl_cost` | `sum(action²)`, unweighted | `rollout/raw_ctrl_cost` |
| `ctrl_cost` | `pen_factor · sum(data.ctrl²)` | `compute_reward` under HER |
| `prev_achieved_goal` | ρ before this step | `compute_reward` under HER (PBRS) |
| `episode_rho_max` | running episode maximum of ρ | `rollout/ep_rho_max_mean` |

`reset()` writes `chest_deg`, `hip_deg`, `side_lying`, `45_deg`, `ctrl_cost=0`.

Do not rename any of these without grepping `mimoEnv/envs/*_callback.py`, `illustrations.py`,
`eval_rollover.py` and `results/`.

`rolled_over` exists because `is_success` measures against whatever goal was *sampled or
relabelled*. Under `--goal_low=0.25` `is_success` reports a healthy rate while no real roll ever
happens. And note the disagreement by construction: `side_lying`/`rolled_over` describe the
**final** step of an episode, while `episode_rho_max` (and the evaluation) describe the episode
**maximum** — they diverge whenever MIMo rolls and rolls back.

### 4.8 HER flags

| Flag | Note |
|---|---|
| `--her` | Attaches `HerReplayBuffer(copy_info_dict=True)`. Requires an off-policy algorithm (`OFF_POLICY_ALGORITHMS = ('SAC', 'TD3', 'DDPG')`, `illustrations.py:58`). Forces `--achieved_goal_in_observation` on, since HER reads `next_obs['achieved_goal']`. |
| `--sparse_reward` | use this with HER, not `--pbrs` |
| `--goal_low` / `--goal_high` | HER needs goal variation, otherwise the policy never learns to condition on the goal and the relabelled transitions describe goals nobody ever asks for |
| `--no_done_active` | §4.6 |
| `--buffer_size` (300 000), `--train_freq` (1) | off-policy knobs. `gradient_steps=1`, `n_sampled_goal=4`, `goal_selection_strategy='future'` and `learning_starts=100` are hardcoded since 26.08.2026; under `--her` the last is raised above the episode horizon automatically, because `HerReplayBuffer.sample` needs at least one finished episode |

`--algorithm=HER` does not exist. It used to be in `choices` but was never dispatched, so it hit
the `else: raise RuntimeError` branch at runtime. Since SB3 1.1, **HER is a replay buffer, not an
algorithm** — the flag was removed and the help text of `--algorithm` records why.

`--buffer_size` defaults to 300 000 rather than SB3's 1e6 for a measured reason: the observation
is stored twice (obs and next_obs), and at 1e6 transitions that is ~6 GB, more than the machine
has free (`illustrations.py:490`). 300k holds 600 episodes.

---

## 5. Training loop, callbacks and TensorBoard tags

`illustrations.train()` (`:222`) loops `model.learn(...)` in chunks of `--save_every` steps and
writes `model_<n>.zip` after each chunk, `n` counting from 1. It composes up to three callbacks:

- **`RollOverCallback`** (`mimoEnv/envs/roll_over_callback.py`) — always. Aggregates over SB3's
  stats window and logs, on every episode end:

  | Tag | Meaning |
  |---|---|
  | `rollout/ep_end_hip_deg_mean`, `rollout/ep_end_chest_deg_mean` | mean final rotation in degrees |
  | `rollout/side_lying_success_rate` | fraction of episodes whose **final step** had ρ ≥ 0.5 |
  | `rollout/raw_ctrl_cost` | per-episode **sum** of `sum(action²)`, unweighted by `pen_factor` |
  | `rollout/ep_rho_max_mean` | mean over episodes of the episode **maximum** ρ — the quantity comparable to `eval_rollover.py` |

  `rollout/raw_ctrl_cost` is a per-episode sum, not a mean: the mean rewarded long episodes, so
  a policy that rolled and then ran out the clock logged a *lower* control cost than one that
  rolled in a short episode. Do not compare the tag across runs with different `--episode_steps`.

  With `--save_intermediate` it writes `model_intermediate_90.zip` the first time the windowed
  side-lying rate exceeds 90 %.

- **`ISRCallback`** — only when `--isr`. Disables ISR at 75 % of `--train_for` (§2.6).
- **the MGC callback** — only when `--mgc != none` (§2.7).

> **`rollout/success_rate` is SB3's own tag and does not mean "rolled over".** SB3 fills it from
> `info["is_success"]` at episode end (`stable_baselines3/common/base_class.py:454`), and
> `is_success` scores against the *sampled* goal, which is as low as 0.25 under `--goal_low`. A
> run logging 0.26 there can be at 0 % real rolls. Read `rollout/ep_rho_max_mean` instead, and
> compare runs only through `eval_rollover.py`.

TensorBoard output goes to `<save_dir>/<ALGO>_<n>/`, e.g. `PPO_0/` or `SAC_0/`.

---

## 6. Evaluation protocol

### 6.1 `mimoEnv/eval_rollover.py`

Use this, not `illustrations.py --test`, for any number that goes into a write-up. `--test` rolls
out exactly one episode and exists to render video and the four key frames.

```bash
MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
    --model=<run_dir>/model_5.zip --starting_position=supine --episodes=50
```

It reads the run's `data.yml` to rebuild a matching environment, then **overrides** the following,
each for a measured reason (`eval_rollover.py:5`, `:58`):

| Rule | Why |
|---|---|
| **ISR off** | Beta(1,3) start angles inflate ρ_max: logs of ≈0.94 were artefacts; the same policies read 0.26–0.36 without ISR (§2.6) |
| **Goal pinned** (`goal_low = goal_high = goal`) | runs trained with a sampled range saw easy targets; scoring against those reports success at zero real rolls. Pinning via the equal-bounds path also avoids consuming an RNG draw (§4.4) |
| **`done_active=False`** | every episode gets the same 500 steps, so episodes are comparable |
| **Deterministic actions** | the stochastic policy is a training device |
| **Success = per-episode `ρ_max ≥ 0.95`**, not `terminated` | with `--no_done_active` nothing terminates, so `terminated` would read a constant 0 % |
| **`success_at_side_lying=False`** | the threshold comes from `--goal` instead, so one protocol covers both kinds of run |

The default goal is 0.95, or 0.5 if `data.yml` says the run was trained with `--side_lying`.
**The policy is conditioned on the goal**: a policy trained at a fixed 0.5 scores ~10 % when
queried at 0.95 and ~66 % at 0.5 (recorded in the project notes; `eval_rollover.py:10` states the
effect without the numbers). A low score at 0.95 for such a run is an out-of-distribution query,
not a failure to roll.

Episodes are seeded deterministically as `1000 + episode_index` (`eval_rollover.py:103`), so two
evaluations of the same checkpoint see the same start states — which holds precisely because ISR,
the one source of unseeded randomness in reset, is off (§2.6).

Output: full-roll rate, side-lying rate, ρ_max mean/min, and mean steps to the first success.
Recorded runtime ≈40 s per 50 episodes (not re-timed for this document).

#### Whole batches: `--group`

A sweep is launched as `<save_path>_run_0 … _run_n`, and a single run of it says nothing — the
question is always how many of the seeds learned to roll. `--group` evaluates the batch:

```bash
MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
    --group=models/roll_over/26-08-23/supine/26-08-23_supine_sac_her_ep500 \
    --episodes=40 --csv=26-08-23_supine_sac_her_ep500_test_success_rate.csv
```

The argument is the run directories' shared prefix (the save path without the `_run_<i>` tail), a
directory holding them, or a glob. Every rule of the single-model protocol still applies, per run
and from that run's own `data.yml`; two more rules are specific to the batch:

| Rule | Why |
|---|---|
| **The last checkpoint** (highest-numbered `model_<n>.zip`), not `model_best.zip` | `best` is the `EvalCallback`'s pick under its own protocol, so a table of best checkpoints measures the checkpoint selection as much as the runs. `--checkpoint=best` or `--checkpoint=model_3.zip` when that is what you want |
| **A run is successful when it rolls in more than 75 % of its episodes** (`--success_threshold`) | the convention used throughout the thesis; 40 episodes is its default sample size |
| One environment for the whole batch | a MIMo env is ~3.6 GB RSS, so group mode holds exactly one alive and rebuilds it only if a run's `data.yml` actually changes the environment |
| Identical episode seeds for every run | the comparison between seeds is paired |

Alongside the per-run table it prints the >90 % / <10 % / in-between banding that
`results/success_after_training_plot.py` draws, and `--csv` writes that script's input file
(`Run,Success_Rate`) directly — so the stacked-bar figure no longer needs
`results/success_after_training.py`, which loads `model_1.zip` with a hardcoded `PPO` and a
hardcoded PBRS environment rather than each run's own configuration.

`--policy_goal_sweep` is refused in group mode: it varies the fed goal for one model, which is a
different question from how many seeds succeeded.

### 6.2 `--policy_goal` and `--policy_goal_sweep`

`--policy_goal FLOAT` feeds the policy a *constant* `desired_goal` while the environment keeps the
real one. Only the policy's input is replaced; `achieved_goal`, every sensor reading and the
measured ρ_max stay real, so the score is honest even though the input is not. With a constant
input the behaviour cannot depend on what was asked for — that is a goal-agnostic policy built out
of a goal-conditioned one, with no retraining.

`--policy_goal_sweep` takes `low:high:step` (`0.25:0.95:0.05`) or a comma-separated list
(`0.05,0.5,2.0`) and prints one row per value. This is what produced the response curve in §4.5;
`results/plot_her_goal_response.py` draws it.

### 6.3 Loading HER-saved models

A model saved with a `HerReplayBuffer` **cannot** be loaded with `env=None` — SB3 asserts, because
the buffer needs `env.compute_reward` to relabel. Pass the env and shrink the buffer instead of
reallocating a training-sized one (`eval_rollover.py:90`):

```python
cls.load(path, env=env, custom_objects={'buffer_size': 1, 'learning_starts': 0})
```

### 6.4 What `eval_rollover.py` does *not* restore

`build_env` only reads the keys it knows about. These are silently **not** reproduced:

- **starting posture** — not stored in `data.yml` at all, so `config.get('roll_over_starting_position', 'supine')`
  always falls through to `supine`. **Always pass `--starting_position` explicitly.**
- **actuation model** — `--use_muscle` is not stored either, so a muscle-actuated run is evaluated
  with the spring-damper model.
- **observation noise** — `--obs_noise` is stored in `data.yml` but `build_env` never applies
  `GaussianNoiseObsWrapper`, so noise-trained policies are evaluated noise-free. That may be what
  you want; it is not what the run saw.
- **`--proprio_config`** — see §7; the observation layout can differ from training.

---

## 7. `data.yml` round-tripping

Every run writes `data.yml` next to its checkpoints (`illustrations.py:1007`). `--load_model` reads
it through `mimoEnv.utils.load_model_yaml` (`utils.py:955`) and pushes the values into argparse
defaults, so reloading a model reconstructs the environment it was trained in.

**Consequence: any new experiment-defining flag must be added to the `yaml_data` dict**, or
reloading will silently evaluate the model under different settings than it was trained with.

`load_model_yaml` also carries back-compat shims for renamed and retired flags — the old single
`age` is expanded into `morph_age` + `physio_age`, `proprio_only_qpos` / `no_proprio` are
translated into `proprio_config`, `intrinsic_goal_eps` / `intrinsic_reference_samples` are renamed
to `gravity_goal_eps` / `gravity_reference_samples` (§3.5), and the retired `intrinsic_goal` /
`proprio_w` / `vesti_w` / `intrinsic_goal_joints` / `intrinsic_acc_axes` / `intrinsic_acc_w` are
dropped with a printed notice (§3.4) so they do not land on the argparse namespace as settings
nothing reads. Note the rename has to happen here and not only as an argparse alias:
`set_defaults` keys on the *dest*, so a stored `intrinsic_goal_eps` would otherwise be a dead
attribute and the run would evaluate at the default radius instead of its own. Follow that
pattern when renaming, and keep the `# Previously: --old_name` comments the fork uses
(`--pen_factor` was `--pen_fac`).

Deliberately **not** stored, because they describe the invocation rather than the model:
`save_model`, `save_every`, `test`, `render_video`, `use_muscle`, `roll_over_starting_position`.

Not stored, but arguably should be — treat these as gaps rather than decisions:

- **`--proprio_config`.** `data.yml` stores the resulting `proprio_params` dict, but nothing reads
  it back: `illustrations.py` rebuilds `proprio_params` from `--proprio_config`, whose default is
  the full component list. Reloading a model trained on a reduced proprioception therefore
  reconstructs the full 305-dim observation. SB3 checks observation spaces on load, so this
  normally surfaces as a space-mismatch error rather than a wrong number — but you have to know to
  re-pass `--proprio_config` by hand. The legacy `proprio_only_qpos` shim still works; the
  `no_proprio` one does not (§8).
- **`--mgc` / `--mgc_stochastic_interval`.** A curriculum run's `data.yml` does not record that
  the embodiment was swapped during training; `morph_age`/`physio_age` record only the values the
  env was constructed with.

`train_for` is stored as `num_train` and is deliberately not read back into `--train_for`.

---

## 8. Divergences: how it is vs. how it was meant

### `angle` was inverted and succeeded at reset

**Removed 26.08.2026 along with the `goal_function` switch.** Recorded because it was the default
for a while, so anything measured with it before then is suspect. It was **unusable as it stood.**
`_get_standardized_rotation` (`roll_over.py:820`) returns `abs(angle_deg) / 180`, where
`angle_deg` is 180° at reset and 0° at the goal. Its own comment two lines above says the intended
scaling is "180° = 0 … 0° = 1". The code produces the opposite: `achieved_goal` starts near 1 and
*decreases* as MIMo rolls.

Measured here:

```bash
# prone, goal_function='angle', done_active=True, one null action
after 1 step: terminated=True  reward=500.0  is_success=True  achieved=[0.975]
```

`cos` in the identical setup reads `achieved = 0.0014` at reset and does not terminate. So with
`angle`, the episode ends as a "success" on step 1 with the full `reward_success`, and the PBRS
potential rewards *not* rolling.

How it got here is recoverable: before the purity rewrite, `is_success` ignored its arguments and
read `get_achieved_goal_cos()` internally, so the success check silently used
the `cos` convention no matter which goal function was configured, and the inversion only affected
the shaping term. Making `is_success` pure — required for HER — exposed it. Anything measured with
`--goal_achievement_function=angle` before that change was measured under a different success
rule than the one the code implements now.

`cos` is the `illustrations.py` default and what every recent run uses, so this does not affect
the reported results; but do not reach for `angle` as a variant without fixing the sign first
(and adding a `goalenv_check.py` case for it).

### `_reset_simulation` is dead code

`MIMoEnv._reset_simulation` (`mimo_env.py:744`) is never called. Under gymnasium 1.0.0,
`MujocoEnv.reset` goes straight to `mj_resetData` + `reset_model`, and the base class has no
`_reset_simulation` at all *(verified: `hasattr(MujocoEnv, '_reset_simulation') == False` on
gymnasium 1.0.0)*. So the actuation-model reset, the renderer re-initialization and the
sensorimotor-delay history clearing in that method do not happen.

The consequence that mattered: goal sampling used to live there, so `self.goal` was set exactly
once — in `initialize()` at construction — and never again. Invisible while the goal was the
constant 0.95; fatal for goal sampling and HER. It now lives in `reset_model` (`roll_over.py:664`).
**Put per-episode work in `reset_model`.**

### HER is a buffer, not an algorithm

`--algorithm=HER` was listed in `choices` but never dispatched, so it fell through to
`raise RuntimeError("Algorithm not defined")`. Removed. Use `--algorithm=SAC --her`.

### `--no_done_active` is recommended, not enforced

Despite being required for HER to be sound (§4.6), `illustrations.py:736` only prints a warning.
Some of the `rbi_autorun*.sh` sweeps launched `--her --sparse_reward` **without**
`--no_done_active`, while `run_her_sparse.sh` includes it. Check which script produced a run before
comparing them.

### Resolved: the `intrinsic` goal function was measured against a different goal than it optimised

Until 19.08.2026, `is_success` under `--goal_achievement_function=intrinsic` ignored its arguments
and returned `get_achieved_goal_cos() >= 0.95` — the **extrinsic** rotation — no matter which
observation-space goal had been configured. So the reward shaped the distance to a sensor-space
target while success was scored on the world-frame roll, and `compute_reward` took a separate
live-state path (`_compute_reward_intrinsic`) that HER could not relabel.

Both are gone. Success is now `‖achieved − desired‖ ≤ --intrinsic_goal_eps`, on the same goal the
reward shapes, through the same pure path as the scalar goals (§3.4). The extrinsic rotation is
still reported — as `info['rolled_over']` and `info['episode_rho_max']` — so training curves and
`eval_rollover.py` numbers stay comparable across goal functions. That is the intended division:
the goal is what MIMo can sense, ρ is how *we* score him.

Anything measured with `--goal_achievement_function=intrinsic` before this change was scored under
a different success rule than the one the code implements now. Three stored runs are affected.

### Fixed: `RollOverCallback` crashed at the first episode end

The switch of `rollout/raw_ctrl_cost` from a per-episode mean to a per-episode sum removed
`episode_stp_cnt` from `_on_training_start` and `_on_step`, but left the `if self.episode_stp_cnt
> 0:` guard that read it. Every training run raised `AttributeError` at the first episode end,
regardless of goal function. The guard was vacuous once the metric became a sum, and is gone.

### `--log_actuations` does nothing

`MIMoRollOverWrapper` (`mimoEnv/envs/roll_over_wrapper.py`) logs every actuator to CSV, but the
code that would install it is commented out (`illustrations.py:864`). The flag parses and is
ignored.

### `--freeze_arm` / `--freeze_leg` and operator precedence

`mimo_env.py:436` reads

```python
if self.freeze_leg or self.freeze_arm and self.actuation_model == SpringDamperModel:
```

which parses as `freeze_leg or (freeze_arm and model == SpringDamperModel)`. So `--freeze_leg`
substitutes `SpringDamperModel_Stationary_Limbs` regardless of the configured actuation model —
including with `--use_muscle`, where the muscle model is dropped without a word.

### Off-policy `--load_model` loses the TensorBoard log

`illustrations.py:921` loads off-policy models with `RL.load(load_model, env, buffer_size=…)` and
does **not** pass `tensorboard_log=save_dir`, unlike the PPO branch two blocks above (`:891`).
Continued training of a SAC/TD3/DDPG run therefore writes no TensorBoard events.

### `--roll_over_model_path_auto` help text is stale

The help string promises `<date>_<starting_position>_<reward_function>_<suffix>`. The code builds
`<date>_<starting_position>_<suffix>` (`illustrations.py:807`); the reward function is not in the
name.

### `--proprio_config=""` does not disable proprioception

`illustrations.py:770` only overrides the components when the string is non-empty:

```python
if len(args.proprio_config) > 0:
    proprio_params["components"] = parse_proprio(args.proprio_config)
```

so an empty value leaves `DEFAULT_PROPRIOCEPTION_PARAMS` — the **full** component list — in place.
The back-compat shim for the retired `--no_proprio` flag (`utils.py:989`) sets `proprio_config`
to exactly that empty string, so reloading an old "no proprioception" run restores full
proprioception instead of none. (`parse_proprio("")` would raise, which is why the guard is there;
the guard just picks the wrong fallback.)

### Smaller ones

- `steps_after_reset` defaults to **30** while the comment beside it says 20 (`roll_over.py:112`).
- `--render_frames` writes **PDFs** named `frame_1` … `frame_4`, not the `frame_{1-5}.png` its help
  text promises. Frames 2 and 3 (60° and side lying) are written on every `--test` run whether or
  not `--render_frames` is set; only frames 1 and 4 are gated by the flag
  (`illustrations.py:144`, `:177`, `:182`, `:190`).
- `info['45_deg']` fires at ρ ≥ 0.25, which is a **60°** roll, not 45° (§1).
- `illustrations.py:768` assigns `proprio_params = DEFAULT_PROPRIOCEPTION_PARAMS` and then mutates
  it in place, so the module-level default dict in `mimo_env.py` is modified for the process.
- `--side_lying` is ignored whenever `--goal_low`/`--goal_high` are given: the range branch of
  `sample_goal` returns first.

---

## 9. CLI flag reference

All 58 flags of `mimoEnv/illustrations.py`, generated from its `argparse` block and grouped by
purpose. "yaml" marks the flags that round-trip through `data.yml` (§7).

### Run control

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--env` | choice `reach\|standup\|selfbody\|catch\|roll_over`, default `roll_over` | — | non-roll-over envs take almost none of the flags below |
| `--train_for` | int, `0` | as `num_train` | 0 = no training |
| `--save_every` | int, `100000` | — | one `model_<n>.zip` per chunk |
| `--algorithm` | choice `PPO\|SAC\|TD3\|DDPG\|A2C`, default `PPO` | ✓ | off-policy = SAC, TD3, DDPG |
| `--lr` | float, `3e-4` | ✓ | applied on the PPO branch |
| `--load_model` | path, `False` | — | reads the run's `data.yml` for defaults |
| `--save_model` | str, `model` | — | directory name / suffix |
| `--roll_over_model_path_auto` | flag | — | `models/roll_over/<yy-mm-dd>/<posture>/<yy-mm-dd>_<posture>_<save_model>/` |
| `--test` | flag | — | one episode, rendering only — not a measurement |

### Task setup

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--roll_over_starting_position` | choice `supine\|prone\|alternating`, default `prone` | — | `alternating` flips every reset |
| `--morph_age` | int, `9` | ✓ | body age; must be in `AGES = [1,3,6,9]` |
| `--physio_age` | int, `9` | ✓ | actuation age; same restriction |
| `--use_muscle` | flag | — | muscle instead of spring-damper actuation |
| `--freeze_arm`, `--freeze_leg` | flag | ✓ | substitutes `SpringDamperModel_Stationary_Limbs` (see §8) |
| `--isr` | flag | ✓ | Initial State Randomization, off at 75 % of training |
| `--mgc` | choice `growth\|inverse\|stochastic\|none`, default `none` | — | morphological growth curriculum |
| `--mgc_stochastic_interval` | int, `20000` | — | only for `--mgc=stochastic` |

### Observation

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--proprio_config` | str, `position\|velocity\|torque\|limits\|actuation` | as `proprio_params` (not read back) | an empty string is a **no-op**, not "off" (§8) |
| `--touch` | flag, `False` | ✓ | enables `TOUCH_PARAMS` from `roll_over.py:44` |
| `--achieved_goal_in_observation` | flag, `False` | ✓ | forced on by `--her` |
| `--obs_noise` | float, `0.0` | ✓ | N(0, σ) on every key except the goals, clipped to the space |
| `--obs_norm` | flag, `False` | ✓ | loads `mimoEnv/envs/normalization/obs_stats_*.npz` |

### Reward and goal

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--pbrs` | flag | ✓ | potential-difference shaping |
| `--pbrs_w` | float, `100` | ✓ | needs to be large; see §3.1 |
| `--sparse_reward` | flag | ✓ | {0, −1}; takes precedence over `--pbrs` |
| `--pen_factor` | float, `0.02` | ✓ | weight of `sum(data.ctrl²)` |
| `--nopen` | flag | ✓ | control cost off |
| `--side_lying` | flag, `False` | ✓ | success at ρ ≥ 0.5; ignored under a goal range |
| `--goal_low`, `--goal_high` | float, `None` | ✓ | both or neither |
| `--goal_tolerance` | float, `None` | ✓ | band instead of threshold, §3.6 |
| `--no_done_active` | flag | ✓ | never terminate early |

### Off-policy and HER

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--her` | flag | ✓ | needs SAC/TD3/DDPG |
| `--buffer_size` | int, `300000` | ✓ | 1e6 OOMs, see §4.8 |
| `--train_freq` | int, `1` | ✓ | env steps between updates |

Hardcoded since 26.08.2026, previously flags: `n_sampled_goal=4`, `goal_selection_strategy='future'`,
`gradient_steps=1`, `learning_starts=100` (raised above the episode horizon under `--her`, and
written to `data.yml` as such). Also removed: `--lr_schedule`, `--lr_decay_start`,
`--target_entropy`, `--stop_at_roll_rate`, `--stop_patience`.

### Rendering and testing

| Flag | Type / default | yaml | Note |
|---|---|---|---|
| `--render_video` | flag | — | one `.avi` per test episode |
| `--render_frames` | flag, `False` | — | four PDFs: start, 60°, side lying, final |
| `--render_actuations` | flag | — | actuation plot overlay; render height 720 |
| `--log_actuations` | flag | — | **no effect** (§8) |
| `--save_intermediate` | flag | — | `model_intermediate_90.zip` at 90 % side-lying rate |

---

## 10. Analysis scripts and cluster scripts

### `results/`

Standalone scripts, run as `python results/<script>.py`, that identify a batch of runs by
`--date=<yy-mm-dd> --suffix=<save_model>` plus `--haltung=prone|supine` (German for posture) and
`--age_physio` / `--age_morph` — the same components `--roll_over_model_path_auto` builds the save
path from. `results/utils.py` holds the shared `make_env` and the `%y-%m-%d` date parsing.

Note that `results/utils.py:make_env` hard-codes `goal_function='cos'`, `pbrs=True`, `isr=False`
and `achieved_goal_in_observation=False` — it does **not** read `data.yml`, so it is not a
substitute for the evaluation protocol in §6.

Roughly:

| Script(s) | Purpose |
|---|---|
| `tb_plot_*.py`, `training_plot*.py`, `gemini_plot*.py` | read TensorBoard event files |
| `plot_her_goal_response.py` | goal-response curves from `eval_rollover.py --policy_goal_sweep` |
| `collect_run_statistics.py`, `success_after_training.py` | re-run trained policies for success rate and roll duration |
| `roll_time_stats*.py`, `laterality_index.py`, `actuation_plot*.py` | derived measures |
| `results/intrinsic/intrinsic_rho_check.py` | the standalone proof that ρ is reconstructible from vestibular + proprioception (§3.5). Takes `--model` (or `--random`), `--episodes`, `--plot`, `--json`; the estimator is implemented in that file and imports nothing from the environment's goal code |
| `results/cee/` | cross-embodiment evaluation (train at one age, evaluate at another) |
| `results/kobayashi/` | limb velocities against Kobayashi et al. data |
| `results/diss/`, `results/ctrl_cost/`, `results/proprio_ablations/` | the corresponding ablations |

Any script here that collects a random rollout must call `env.action_space.seed(s)` in addition to
`env.reset(seed=s)` — gymnasium keeps the two generators separate. Any script that reads
`data.xmat`, a site frame or `data.sensordata` after `mj_step`/`env.step` and compares it against
`data.qpos` must call `mj_forward` first: `mj_step` integrates `qpos` *after* evaluating the
dynamics, so the two describe instants one control step apart (§3.5). And pin
`torch.set_num_threads(n)` if the numbers need to be reproducible across machines: a fixed thread
count is bit-identical, different counts diverge because the matmul reduction order changes and
training epochs compound it.

### `rbi_*.sh`

These `ssh` into named hosts at `*.rbi.cs.uni-frankfurt.de`, one run per host, executing
`conda activate mimo && cd MIMo && python …` in the background. They are the record of which
sweeps were actually run: `rbi_autorun*.sh` (train), `rbi_loadmodel.sh` (continue training),
`rbi_testmodel.sh` (evaluate), `rbi_crossembodiment_statistics.sh` and `rbi_dcee_success.sh`
(the 4×4 age grid), `rbi_killallpy.sh` (abort). They hard-code paths and dates — read before
reuse. `run_her_sparse.sh` is the local, sequential equivalent for the HER configuration.

### `mimoComposer/`

A second, independent training stack (hierarchical COMPOSER over body regions) that does not go
through `illustrations.py`. It **did not work** — 0 % roll across 8 configurations — and is kept
for the negative result. It is excluded by `.gitignore`, so it exists only in a working copy. Two
of its findings are folded into this document because they apply to any code reading MIMo
observations: the proprioception layout (§2.4) and the two seeding traps above. Its README claims
`osmesa` is broken and `egl` required; that is inverted — use `MUJOCO_GL=osmesa` here as
everywhere else in this repository.

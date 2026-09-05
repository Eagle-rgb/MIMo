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
- **ρ is always the `mean` of the two, whatever the goal is pooled with.** `--cos_goal_pool`
  (§3.4) changes what the *goal* is computed from; ρ as reported by `info`, the callbacks and
  `eval_rollover.py` comes from `get_achieved_goal_cos_mean()` and is unaffected, so every run
  stays comparable to every other. Do not report `get_achieved_goal_cos()` — under a non-mean
  pool that is a different number.

Two derived quantities appear in logs and rendering:

| Function | Value | Used by |
|---|---|---|
| `get_rotation_degrees_to_goal_z_axis(body)` (`:775`) | 180° at reset → 0° at goal | internal |
| `get_achieved_rotation_degrees(body)` (`:813`) | 0° at reset → 180° at goal | `info['hip_deg']`, `info['chest_deg']`, `rollout/ep_end_*_deg_mean` |

There are two goal-achievement functions, `cos` (default, §3.4) and `gravity` (§3.5), and `cos`
carries a pooling choice on top.

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
| `achieved_goal` | (1,) or (2,) | only if `--achieved_goal_in_observation` (forced on by `--her`) |
| `desired_goal` | (1,) or (2,) | always — `goals_in_observation=True` is hard-coded (`roll_over.py:333`) |

Both goal keys are flat `Box` spaces of shape (`goal_dim`,): **(1,)** for the scalar `cos` pools
(§3.4) and **(2,)** for `--cos_goal_pool=none` and for `gravity` (§3.5), one entry per body. The
width is fixed at construction, so a run cannot be reloaded under a configuration that changes
it.

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

### 2.8 Missing limbs

`--missing_limb` (one of `MISSING_LIMBS`) removes a limb. `--missing_limb_mode` decides how, and
the two modes are for different questions:

| mode | what goes | spaces | use for |
|---|---|---|---|
| `cut` (default) | the body subtree, its joints, actuators and sensors | shrink (46 → 38 actuators, 305 → 253 proprioception values for one limb) | **training** a policy that never had the limb |
| `ghost` | only the physics — mass, inertia, collisions | unchanged | **transfer**: loading an intact-trained policy onto a body missing a limb |

A `cut` model cannot take an intact policy (SB3: "Observation spaces do not match"), which is the
whole reason `ghost` exists. Under `ghost`, `--ghost_obs` says what proprioception reports for the
missing limb: `rest` (default) the values measured on the intact body at rest, averaged over
`--ghost_reference_samples` ISR-free resets, or `zero`. Use `rest` — `robot:left_knee` rests at
−0.115 rad, not 0, so `zero` makes a zero-shot transfer also measure a distribution shift.
Measured: an intact PPO policy on `missing_limb=left_arm` reaches 30 % full roll with `rest`
against 10 % with `zero`.

Cut scenes are pre-generated by `mimoEnv/assets/roll_over/generate_amputated_scenes.py` —
regenerate rather than hand-edit. `--missing_limb` does not combine with `--touch`, and
`--missing_limb_mode=cut` does not combine with `--freeze_arm`/`--freeze_leg`; both raise.

### 2.9 The floor

Three flags make the support surface a variable of the experiment. All default to `None`, which
leaves the floor exactly as the scenes compile it, so every stored run reloads unchanged. They are
written onto the compiled model in `_apply_floor_properties` (called from `initialize()`), not into
the XMLs, because the age scenes are pre-generated and `set_embodiment` swaps between them.

| Flag | Meaning | Baseline |
|---|---|---|
| `--floor_softness` | contact time constant `solref[0]` in seconds; larger is softer (0.05 → ~1.2 mm sink, 0.10 → ~4.5 mm) | mixture 0.0125, ~0.11 mm |
| `--floor_friction` | sliding friction. Independent of softness — MuJoCo's Coulomb friction is area-independent, so sinking in buys no grip | 1.0 |
| `--floor_solimp_width` | penetration depth over which the contact ramps soft → firm (`solimp[2]`); turns the linear spring into a foam-like response | 0.001 |

**`geom_priority` is the part that is easy to get wrong.** At equal priority MuJoCo *mixes* the two
geoms' contact parameters, and MIMo's skin is `solref="0.005 1"`, so asking for 0.10 without
raising the floor's priority yields an effective 0.0525 and a fifth of the intended give. The env
sets `geom_priority=1` and pins the untouched parameters to the baseline mixture, so the three
flags stay independent.

Two results, and they say different things:

- **Passively, a soft floor removes free rolls.** On the rigid floor MIMo reaches ρ 0.85 from an
  imposed 45° and 0.97 from 75° with **zero action**, on landing energy alone. At
  `--floor_softness=0.1` nothing below 90° rolls and everything above does. This is the mechanism
  behind the ISR artefact of §2.6.
- **For a trained policy it is not harder.** Two rigid-floor-trained HER seeds, 25 paired
  episodes: 96 %/92 % rolls on rigid against 100 %/100 % at softness ≥ 0.10, and the ρ_max
  *minimum* rises from 0.81/0.83 to 0.95–0.99. So the compliant floor is not established as a
  difficulty manipulation; what it demonstrably does is remove bounce-assisted rolls and reduce
  outcome variance. Whether it is harder to *learn* on is open — zero-shot transfer says nothing
  about learnability.

The plane does not deform, MIMo penetrates it, so this is a compliant *contact* and not a visible
mat. Cost of the contact change alone is 0.10 → 0.36 ms/step against 1.7 ms for the env step.

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

with `ctrl_cost = pen_factor * raw_ctrl_cost`, and `raw_ctrl_cost = sum(action²)` by default or
the actuation model's `cost()` under `--pen_metabolic` (§3.7).

### 3.1 The axes

| Flag | Effect | Interaction to know |
|---|---|---|
| `--pbrs` (+ `--pbrs_w`, default 100) | potential *difference* shaping instead of the raw potential | forbidden with `--no_done_active`; pointless with `--sparse_reward` (the sparse branch is checked first) |
| `--sparse_reward` | {0, −1} only | overrides `--pbrs` silently; the control cost still applies unless `--nopen` |
| `--pen_factor` (default 0.02), `--nopen` | weight of the control cost | goal-*independent*, which is why it travels through `info` under HER (§4.2) |
| `--pen_metabolic` | pay the actuation model's own capacity-weighted `cost()` instead of the flat `sum(action²)` | §3.7; `--pen_factor` still applies, but on a ~2100× smaller quantity |
| `--goal_low` / `--goal_high` | sample the target per episode instead of a fixed 0.95 | both or neither, else `ValueError`; makes `rollout/success_rate` unreadable (§5) |
| `--side_lying` | success at ρ ≥ 0.5 instead of 0.95, implemented by `sample_goal` returning 0.5 | ignored when `--goal_low`/`--goal_high` are set — the sampled range wins (`:722` is checked before `:734`) |
| `--cos_goal_pool` (`mean\|min\|max\|none`, default `mean`) | how hip and chest are combined into the goal (§3.4) | `none` makes the goal 2-D and silently ignores `--goal_low`/`--goal_high`; it also rescales the PBRS potential |

`--pbrs_w` defaults to 100 because the raw potential difference between two consecutive steps is
tiny; without a large weight the shaping signal does not drive learning at all
(`roll_over.py:168`).

### 3.2 Success

Success is `achieved_goal >= desired_goal`, evaluated by `is_success` (`:508`). The threshold
lives **in the goal**, not in the check: `sample_goal` (`:687`) returns

- `0.95` by default,
- `0.5` with `success_at_side_lying` (`--side_lying`),
- `U(goal_low, effective_goal_high)` when a range is configured.

Under `--goal_tolerance` the threshold becomes a band, `‖achieved − desired‖ ≤ tolerance`
(§3.6), and under `--goal_achievement_function=gravity` a ball of radius `--gravity_goal_eps`
(§3.5). All three rules live in `_success_mask`, which reduces with `.all(axis=-1)` so that a 2-D
goal needs *both* bodies past the threshold; `is_success` is a thin wrapper that returns a plain
`bool` for a single goal and a `(N,)` array for a batch. The per-row reduction is not cosmetic —
a whole-array `.all()` collapses a relabelled HER batch of N transitions to one reward.

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
The environment validates its own arguments — unknown posture, unknown goal function, unknown
`cos_goal_pool`, an age outside `AGES`, a `goal_low`/`goal_high` mismatch, a non-positive
`goal_tolerance` or `gravity_goal_eps`, `--goal_tolerance` combined with `gravity` (which is a
point goal with a radius already), and the missing-limb combinations of §2.8 — but it knows
nothing about the combination above.

### 3.4 `cos` and `--cos_goal_pool`

`cos` is the default goal function and what every reported run uses. Its achieved goal is built
from the same two per-body values ρ is (§1), and `--cos_goal_pool` decides how they are combined:

| `--cos_goal_pool` | achieved goal | `goal_dim` | success (threshold rule) |
|---|---|---|---|
| `mean` (default) | `(ρ_hip + ρ_chest) / 2` | 1 | `ρ ≥ target` |
| `min` | `min(ρ_hip, ρ_chest)` | 1 | the *worse* body has to reach the target |
| `max` | `max(ρ_hip, ρ_chest)` | 1 | the *better* body suffices |
| `none` | `(ρ_hip, ρ_chest)` | 2 | both bodies, via `.all(axis=-1)` |

Three things follow from the table and are easy to get wrong:

- **`none` and `min` have the *same* success criterion.** `ρ_hip ≥ g ∧ ρ_chest ≥ g` is exactly
  `min(ρ_hip, ρ_chest) ≥ g`; measured over 2500 steps of a rolling policy, 0 disagreements. What
  separates them is the width of the goal, i.e. what HER relabels onto — which is the whole point
  of `none` (§3.5).
- **`--goal_low`/`--goal_high` do nothing under `none`** and no error is raised
  (`sample_goal:1362`). The 2-D goal is always the fixed reference `(g, g)`.
- **The PBRS potential is rescaled for `none`.** A 2-D goal in [0, 1]² sits at distance √2 from
  the reset, not 1, so `_potential_scale` returns `1/√2` there and 1.0 for the scalar pools. Both
  `--pbrs_w` and `reward_success` therefore mean the same thing across all four.

`mean` is the historical behaviour and the reporting baseline; `min` and `max` are scalar variants
that change *which* body has to finish the roll, and exist to probe the pelvis-twist failure mode
that made `mean` average two bodies in the first place (§1).

### 3.5 The `gravity` goal function -- a proof of concept, and its cheap replacement

`cos` is read off the **root free joint**, which proprioception does not report (§2.4). MIMo
therefore cannot sense the quantity he is optimising. `gravity`
(`--goal_achievement_function=gravity`) exists to show that he could: it reconstructs the same
per-body quantity from the vestibular sensors and the joint chain, and nothing else.

```
t = 0, MIMo demonstrably at rest:  g_site <- normalize(accelerometer)
every step, gyroscope only:        g_site <- rotate(g_site, -omega * dt)     # Rodrigues
goal:                              [ (R_body^T R_site @ g_site)[0]
                                     for body in GRAVITY_GOAL_BODIES ]       # hip, chest
```

+1 supine, -1 prone per body -- the same scale as `get_dot_local_x_to_global_z(body)`. The
accelerometer is read **once**, at reset, where MIMo has settled and specific force is gravity;
afterwards only the gyroscope enters, so linear acceleration cannot forge the signal and turning
the head cancels as an identity rather than being counterweighted. `R_body^T R_site` depends only
on the joints between that body and the head, so the root free joint cancels.

Success is a ball of radius `--gravity_goal_eps` (default 0.15) around a **reference posture**
recorded at construction: `--gravity_reference_samples` (default 20) ISR-free resets in the
*opposite* posture, averaged. Measured on the spring-damper model, that reference is
`(-0.99896, -0.99752)` prone and `(+0.999, +0.999)` supine, sd 0.001 -- essentially the ideal
`(∓1, ∓1)`.

#### It works

*(supine, age 9/9, PPO + PBRS(100), pen 0.02, 250-step episodes, matched to
`26-08-19_supine_age9_ep250` in everything but the goal function)*

| | rho_max | success | side | hip | chest | gap |
|---|---|---|---|---|---|---|
| **`gravity` hip+chest, normalised, 2M** | **0.938** | **0.997** | **1.00** | 165.9 | 141.3 | 24.6 |
| `cos` baseline, 1M | 0.952 | 0.984 | 0.99 | 161.0 | 151.0 | 10.0 |

Under the fixed protocol of §6 (`eval_rollover.py`, 50 deterministic episodes, ISR off):

| | full roll | rho_max mean/min | steps to roll |
|---|---|---|---|
| **`gravity`** | **100 %** | 0.995 / 0.985 | **41.2 +- 8.7** |
| `cos` baseline | 100 % | 0.995 / 0.988 | 51.3 +- 8.9 |

**MIMo learns to roll from a goal built only out of what he can sense**, as reliably as from the
hand-designed extrinsic one. That is the whole claim, and it is made.

Two findings from getting there are worth keeping:

- **Hip alone is not enough.** A hip-only version plateaued at rho 0.385 with the hip at 101.9°
  and the chest at 42.5° -- MIMo twists the pelvis and leaves the torso lying, a 59° gap. Adding
  the chest closed a third of it. Same failure, and same fix, as the `cos` average (§1).
- **Two dimensions, not their average.** An average cannot distinguish `(hip -1, chest +1)` from
  `(hip 0, chest 0)`; the vector distance penalises the gap directly.

Caveats: one seed per configuration (the fork's convention elsewhere is 6-18), PPO + PBRS only,
and `--obs_noise` perturbs the observation rather than `data.sensordata`, so the goal computation
inside the env never sees any noise.

#### The potential is normalised

`cos` measures progress in [0, 1]; `gravity` runs from +1 to -1 per body, so with two bodies a
reset sits at distance 2.83. With `--pbrs_w=100` and `reward_success=500` unchanged the shaping
term was 2.83x larger than in the baseline while the terminal bonus was not, and the policy farmed
shaping reward to rho 0.48 without paying for the last part of the roll. `_potential_scale`
divides by the reset distance, returning `1/(2*sqrt(n_bodies))` = 0.3536 for `gravity` against
`1/sqrt(2)` = 0.7071 for `--cos_goal_pool=none` and 1.0 for the scalar pools. It deliberately does
**not** scale `--gravity_goal_eps`, which stays in the readable +-1 units of the goal.

#### `--cos_goal_pool=none` is the same goal without the estimator

Once the claim is made, the estimator is a cost, not a feature: it carries per-episode integration
state, a reset hook and a scale of its own, and it is fragile (below). **`--cos_goal_pool=none`
gives the identical goal read off the global state instead.**

The two achieved goals are related by a fixed affine map. With `s = -1` supine, `+1` prone,

```
rho_b = (s * d_b + 1) / 2        so        ||Δrho|| = ||Δd|| / 2      exactly
```

so a `gravity` ball of radius `eps` is a `none` ball of radius `eps / 2`:
**`--gravity_goal_eps=0.15` ≡ `--cos_goal_pool=none --goal_tolerance=0.075`.** Verified over 2500
steps of a rolling policy (10 episodes, `26-08-19_supine_age9_ep250_run_0`, hip-chest spread mean
0.070 / max 0.426, so the two bodies genuinely differ):

| criterion | steps scoring success | disagreements with `gravity` |
|---|---|---|
| `gravity`, eps 0.15 | 1259 / 2500 | — |
| `none` + tol 0.075, goal = gravity's recorded reference | 1246 / 2500 | **0 / 2500** |
| `none` + tol 0.075, goal = `(1, 1)` | 1246 / 2500 | 13 / 2500 (0.5 %) |

The distance identity holds to machine precision (max residual 2.22e-16). The 13 steps are the
one real difference: `gravity`'s goal is the *recorded* reference, which in rho units is
`(0.99948, 0.99876)` rather than exactly `(1, 1)` -- an offset of 0.00135, so only states sitting
within that of the decision boundary flip. `_potential_scale` is consistent with the same factor
of two (0.3536 against 0.7071), so PBRS is on the same scale as well.

What `none` does **not** reproduce is the sensing claim: it reads the root free joint like every
other `cos` variant. Keep `gravity` when the point is what MIMo can sense; use `none` when the
point is the 2-D goal geometry that HER needs (§3.6).

#### `--use_muscle` does not work with `gravity`

The reference posture cannot be recorded properly under the muscle model. Measured, 20 ISR-free
resets each:

| actuation | prone reference | sd | supine reference | sd |
|---|---|---|---|---|
| spring-damper | `(-0.999, -0.998)` | 0.001 | `(+0.999, +0.999)` | 0.001 |
| muscle | `(-0.947, -0.975)` | **0.015** | `(+1.000, +0.999)` | 0.002 |

The prone reference is 0.059 away from the true prone posture and its spread across resets is 15x
larger, so the success ball is both off-centre and noisy. The cause is upstream of the goal:
`MuscleModel._compute_parametrization` zeroes `jnt_stiffness` and divides `dof_damping` by 20 for
every `robot:` joint, and `steps_after_reset=30` does not settle that body -- at the moment the
reference is read, `|qvel|` is ~15 against ~4.5 for spring-damper (8 resets each).

The estimate is then wrong *during* the episode too. Driving the env with purely random actions,
200 steps, supine:

| actuation | max per-body \|error\| | closest approach to the prone reference |
|---|---|---|
| spring-damper | 0.014 | 2.79 |
| muscle | **1.686** | **0.546** |

Under muscle the estimate reads `(-0.52, -0.59)` -- half rolled -- while the truth is
`(+1.00, +0.91)`, i.e. flat on his back, and it comes within 3.6x the success radius by accident
alone. A policy rewarded for closing that gap finds the rest: a live SAC+HER muscle run logged
`rollout/success_rate` 0.06-0.08 at `ep_rho_max_mean` 0.048 and `eval/roll_rate` 0.000. The error
is present within the first control step and stays flat, so it is not slow drift; the obvious
cause is ruled out, since the muscle model's angular rates are *lower* than the spring-damper's
(mean `|omega|` 0.78 against 1.28 rad/s). It has not been diagnosed further.

**This is the motivation for `--cos_goal_pool=none`**: the same goal geometry, no reference
recording, no integration, and it works with every actuation model.

#### What carries the sensing claim outside training

`results/intrinsic/intrinsic_rho_check.py` makes the claim without training at all, and is the
citation for it. It reimplements the estimator standalone -- nothing in it imports the
environment's goal code -- and measures it against the truth step by step along real rollouts.
`g_site` is seeded and integrated exactly as above; `R_b^T R_site` comes from **forward kinematics
on the joint angles alone** (`mj_kinematics` on a scratch `MjData` whose root free joint is pinned
to the identity pose, so `qpos[:7]` is overwritten and never read). That is stronger than the
environment's own implementation, which takes the same rotation out of the live simulation and
argues that the root cancels.

Measured 26.08.2026, 10 episodes per policy:

| policy | posture | steps | hip mean \|err\| | chest mean \|err\| | rho mean \|err\| | rho corr | rho_max mean \|err\| | same success call |
|---|---|---|---|---|---|---|---|---|
| `cos`, PPO+PBRS (`26-08-19_supine_age9_ep250_run_0`) | supine | 2510 | 0.0227 | 0.0154 | 0.0060 | 0.9998 | 0.0013 | 10/10 |
| `cos`, SAC+HER sparse (`26-08-24_prone_sac_her_ep200_run_0`, `model_5`) | prone | 2010 | 0.0094 | 0.0095 | 0.0046 | 0.9998 | 0.0007 | 10/10 |
| `gravity`, PPO+PBRS (`26-08-22_supine_gravity2n_ppo_pbrs`, `model_8`) | supine | 2510 | 0.0173 | 0.0095 | 0.0051 | 0.9997 | 0.0009 | 10/10 |

`d_b` spans 2.0 from prone to supine, so a mean error of 0.02 is 1 % of the range; on rho the
error is 0.005 and the worst single step is 0.040. **The two signals make the same success call
at the 0.95 roll threshold in every episode.** Error does not accumulate (last decile no worse
than the first), and the per-step root-invariance check reads **max 0.0000 deg**. The two `cos`
policies never saw the reconstructed signal in training. All of this is on the spring-damper model.

> **Trap, and it applies to any script here that reads `data.xmat` after a step.** `mj_step`
> evaluates the forward dynamics at the state it is given and integrates `qpos` afterwards, so on
> return `data.xmat`, the site frames and `data.sensordata` describe the state *before* the last
> integration step while `data.qpos` describes the state after it. Reading the truth from `xmat`
> and the estimate from `qpos` compares two instants one control step apart: it showed up as a
> 4.01 deg root-invariance residual that looked exactly like a leak of root orientation. One
> `mj_forward` after each step -- a pure recomputation, which `mj_step` does again at the start of
> the next step -- takes it to 4e-6 deg.

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

Note the band alone gives `cos` the ball criterion but not the second dimension.
`--cos_goal_pool=none --goal_tolerance=0.075` gives it both, and is exactly the `gravity` goal
without the estimator (§3.5) -- which makes it the controlled way to separate "ball" from
"2-D goal" as the mechanism.

Under `--side_lying` the goal stays 0.5 but then means "stop at side lying" rather than "reach
at least side lying". `_potential` is untouched -- it was a distance already.

The experiment this exists for:

```bash
# 6 seeds, cos, sparse + HER, NO goal range -- against 26-08-23_supine_sac_her_ep200_nolohi
python mimoEnv/illustrations.py --algorithm=SAC --her --sparse_reward --no_done_active \
    --goal_achievement_function=cos --goal_tolerance=0.05 --episode_steps=200 ...
```

If it trains, the mechanism above is confirmed. The counter-test is `gravity` with
`--gravity_goal_eps=0.02`, which should then collapse. Note the two groups compared above also
differ in horizon (100 vs 200 steps), so a `cos` run at 100 steps is still needed to rule that
out.

The reported numbers stay comparable either way: `eval_rollover.py` scores `rho_max >= 0.95`
measured off the simulation and never calls `is_success` (`:168`). It does pin `desired_goal` to
1.0 for a run trained with a tolerance, because the policy is conditioned on that input.

---

### 3.7 `--pen_metabolic` — the capacity-weighted control cost

The default penalty counts every actuator equally: `pen_factor * sum(x²)`, where `x` is the
clipped action — `data.ctrl` (46 motors in [−1, 1]) under the spring-damper model, or
`target_activity` (92 muscles in [0, 1]) under `--use_muscle`. An eye motor (`gear` 0.0033) then
costs exactly as much as `act:hip_bend` (`gear` 8.93), a factor 2715 in torque capacity, in a task
whose entire content is trunk torque. Measured over the age-9 actuator set: the 20 smallest
actuators hold 8 % of the torque capacity but 43 % of the flat penalty.

`--pen_metabolic` pays the actuation model's own `cost()` instead. Both models implement the same
form — a **capacity-weighted** mean of squared commands, divided by the number of channels:

| model | formula | signal | range |
|---|---|---|---|
| `SpringDamperModel` (`actuation.py:171`) | `Σ u_i² T_max,i / (n · Σ T_max,i)` | `control_input`, the clipped action | [0, 1/46] |
| `MuscleModel` (`muscle.py:176`) | `Σ a_i² f_max,i / (2n · Σ f_max,i)` | `activity`, **after** the τ = 0.01 s lag | [0, 1/92] |

`Σ a² · f_max` with `f_max ∝ PCSA ∝ muscle volume` is the standard metabolic proxy in
biomechanics (volume-weighted activation squared, as in OpenSim static optimization), which is
where the flag's name comes from. Under `--use_muscle` it is also the only one of the two that
sees co-contraction correctly, since the spring-damper model has one bidirectional motor per joint
and the sign vanishes in the square either way.

**`--pen_factor` still applies, but its calibration does not transfer.** `cost()` is bounded by
`1/n_channels`, so at `--pen_factor=0.02` the penalty is ~2100× smaller than the flat one and
effectively `--nopen`. Measured over 20 random actions, spring-damper: flat raw 13.4/step against
metabolic 0.0060/step. **Use `--pen_factor≈40`** to land on the same scale — which is exactly the
factor `mimoEnv/envs/catch.py:199` already uses for `40 * cost()`, and it puts the maximum penalty
at 0.87/step against the flat penalty's 0.92/step. Verified in a short PPO run: `pen_factor=40`
with `--pen_metabolic` gives 0.39/step, against ~0.42/step for the flat `--pen_factor=0.02`
baseline.

**`raw_ctrl_cost` now means what it says** — the penalty before `pen_factor`, so
`ctrl_cost == pen_factor * raw_ctrl_cost` holds under both schemes and under both actuation
models (checked in all four combinations). What it *contains* therefore changes with the flag,
and `rollout/raw_ctrl_cost` is not comparable across runs that differ in it. That is what
`rollout/metabolic_cost_mean` (§5) is for: it is logged unconditionally, so the effort of a
flat-penalty run and a metabolic-penalty run can be read on one scale.

**Off by default, and stored in `data.yml`.** Every run on disk trained against the flat penalty;
the default path is unchanged (bit-identical). A stored run without the key reloads as `False`.

Two caveats:

- Under `--use_muscle` the muscle model's `cost()` reads `activity`, the lagged activation, not
  the commanded `target_activity`. The penalty is then a function of the recent action *history*
  rather than of this step's action alone. HER is fine — the term is goal-independent and travels
  through `info` (§4.2) — but it cannot be reconstructed from an action vector, so the
  `compute_reward` fallback for a batch without `info['ctrl_cost']` uses the live value.
- Weighted, the eye and finger actuators are effectively free (the six eye motors carry ~0.005 %
  of the capacity each). Irrelevant for rolling and for `eval_emg.py`, which scores ES/AB/QUAD/HAM
  — all large actuators — but not a free swap for anything where gaze behaviour is the point, i.e.
  `--look_reward`.

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

This holds for **every** goal function and every pool. All three reshape their arguments to
`(N, goal_dim)` through `_as_goal_batch` (`roll_over.py:480`), so the scalar pools
(`goal_dim = 1`), `--cos_goal_pool=none` and `gravity` (`goal_dim = 2`) share one code path; the
only difference is the success test in `_success_mask`. That reduction must be per row —
`.all(axis=-1)`, not `.all()` — or a relabelled batch of N transitions collapses to a single
reward and HER silently does nothing.

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
  region**. No reward path uses it any more; it survives only as a diagnostic, read by
  `goalenv_check.py` and `results/collect_observation_util.py`.
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
| `raw_ctrl_cost` | the penalty before `pen_factor`: `sum(action²)`, or `cost()` under `--pen_metabolic` (§3.7) | `rollout/raw_ctrl_cost` |
| `metabolic_cost` | the actuation model's `cost()`, always, whether or not it is what the reward pays | `rollout/metabolic_cost_mean` |
| `ctrl_cost` | `pen_factor · raw_ctrl_cost` | `compute_reward` under HER |
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
  | `rollout/raw_ctrl_cost` | per-episode **sum** of the penalty before `pen_factor` — `sum(action²)`, or `cost()` under `--pen_metabolic` |
  | `rollout/metabolic_cost_mean` | per-episode **sum** of the actuation model's `cost()`, logged unconditionally — the tag that stays comparable across `--pen_metabolic` (§3.7) |
  | `rollout/ep_rho_max_mean` | mean over episodes of the episode **maximum** ρ — the quantity comparable to `eval_rollover.py` |

  Both cost tags are per-episode sums, not means: the mean rewarded long episodes, so
  a policy that rolled and then ran out the clock logged a *lower* control cost than one that
  rolled in a short episode. Do not compare either across runs with different `--episode_steps`.
  `rollout/raw_ctrl_cost` is additionally not comparable across `--pen_metabolic`, because the
  flag changes which quantity it holds; `rollout/metabolic_cost_mean` always holds the same one.
  Both are averaged over SB3's stats window (`_stats_window_size`, 100 episodes by default).

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

`env_kwargs` only reads the keys it knows about. These are silently **not** reproduced:

- **`--cos_goal_pool`** — stored in `data.yml` (§3.4) but never read back, so a run trained with
  `min`/`max` is scored against the `mean` goal it was not conditioned on, and a `none` run cannot
  be loaded at all (`goal_dim` 2 against 1: SB3 refuses on the observation space). Pass the pool
  by hand until `env_kwargs` carries it.
- **observation noise** — `--obs_noise` is stored but `env_kwargs` never applies
  `GaussianNoiseObsWrapper`, so noise-trained policies are evaluated noise-free. That may be what
  you want; it is not what the run saw.
- **`--proprio_config`** — see §7; the observation layout can differ from training.

Two that used to be on this list and are not any more: the **starting posture**, which is absent
from `data.yml` by design and is now recovered from the save path by `starting_position_from_path`
(before that, all 198 prone checkpoints were silently evaluated as supine), and the **actuation
model**, since `--use_muscle` moved into `data.yml` on 02.09.2026 and `env_kwargs` reads it back.

---

## 7. `data.yml` round-tripping

Every run writes `data.yml` next to its checkpoints (`illustrations.py:1007`). `--load_model` reads
it through `mimoEnv.utils.load_model_yaml` (`utils.py:955`) and pushes the values into argparse
defaults, so reloading a model reconstructs the environment it was trained in.

**Consequence: any new experiment-defining flag must be added to the `yaml_data` dict**, or
reloading will silently evaluate the model under different settings than it was trained with.

`load_model_yaml` also carries back-compat shims for renamed and retired flags — the old single
`age` is expanded into `morph_age` + `physio_age`, `proprio_only_qpos` / `no_proprio` are
translated into `proprio_config`, and `intrinsic_goal_eps` / `intrinsic_reference_samples` are
renamed to `gravity_goal_eps` / `gravity_reference_samples` (§3.5). Keys belonging to flags that
no longer exist are dropped with a printed notice, so they do not land on the argparse namespace
as settings nothing reads. Note the rename has to happen here and not only as an argparse alias:
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

All 59 flags of `mimoEnv/illustrations.py`, generated from its `argparse` block and grouped by
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
| `--use_muscle` | flag | ✓ | muscle instead of spring-damper actuation; **does not work with `gravity`** (§3.5) |
| `--missing_limb` | choice, `None` | ✓ | §2.8 |
| `--missing_limb_mode` | choice `cut\|ghost`, default `cut` | ✓ | `cut` to train, `ghost` to transfer |
| `--ghost_obs` | choice `rest\|zero`, default `rest` | ✓ | ghost mode only |
| `--ghost_reference_samples` | int, `20` | ✓ | ghost mode only |
| `--floor_softness`, `--floor_friction`, `--floor_solimp_width` | float, `None` | ✓ | §2.9; `None` = the floor the scenes compile |
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
| `--pen_factor` | float, `0.02` | ✓ | weight of `raw_ctrl_cost`. Use ~40 with `--pen_metabolic` |
| `--pen_metabolic` | flag, `False` | ✓ | §3.7; pay the actuation model's capacity-weighted `cost()` instead of `sum(action²)` |
| `--nopen` | flag | ✓ | control cost off |
| `--side_lying` | flag, `False` | ✓ | success at ρ ≥ 0.5; ignored under a goal range |
| `--goal_low`, `--goal_high` | float, `None` | ✓ | both or neither |
| `--goal_achievement_function` | choice `cos\|gravity`, default `cos` | ✓ | §3.4 / §3.5; fixes the width of the goal space, so a run cannot be reloaded under the other one |
| `--cos_goal_pool` | choice `mean\|min\|max\|none`, default `mean` | ✓ | §3.4; `none` gives the 2-D goal |
| `--gravity_goal_eps` | float, `0.15` | ✓ | success radius of the `gravity` ball; alias `--intrinsic_goal_eps` |
| `--gravity_reference_samples` | int, `20` | ✓ | resets averaged into the reference posture |
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

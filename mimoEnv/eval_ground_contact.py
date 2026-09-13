""" Ground contact forces during MIMo's roll, matching Kobayashi et al. (2021), Exp Brain Res 239:2887-2904.

    MUJOCO_GL=osmesa python mimoEnv/eval_ground_contact.py --model=<path/to/model_1.zip> [--episodes=50]
    python mimoEnv/eval_ground_contact.py --selfcheck        # the analysis on synthetic data, no env
    python mimoEnv/eval_ground_contact.py --physics_check    # the force extraction, one env, no model

Protocol is `eval_rollover.py`'s, imported rather than restated: ISR off, goal pinned,
``done_active=False``, deterministic actions, environment rebuilt from the run's own ``data.yml``.
The seeds default to eval_rollover's (1000 + episode), so these are the same episodes its roll
rate was measured on. Roll types and the stationary/moving limb labels are `eval_emg.py`'s.

What Kobayashi measured
-----------------------
17 infants (9-10 months), 129 supine-to-prone rolls, on a BIGMAT 2000 pressure mat: four tiled
mats covering 880 x 960 mm, one resistive sensor per cm^2, 60 Hz, resolution 0.2-2 N/cm^2. Each
frame of the 2-D pressure image was segmented **by hand**, against video and marker positions,
into six regions: ipsilateral arm (IA), ipsilateral leg (IL), contralateral arm (CA),
contralateral leg (CL), head (H) and proximal body (PB). "The distal arm does not include the
elbow, and the distal leg does not include the knee"; PB is the trunk, the pelvis and everything
not attributed to the other five. Pressure was summed per region and divided by body weight,
itself estimated as the mean total pressure over 1 s of quiet supine before the roll.

Time was aligned at T_TR, the moment the normalised lateral displacement of a xiphisternum marker
crosses 0.5 (i.e. peak trunk speed). Pressure was time-averaged over T_TR +- 0.25 s for IA and IL
and over T_TR - 1.0 s .. + 0.5 s for CA, CL, H and PB. Headline results: the arms carry almost
nothing (>70 % of trials show no pressure even under a *stationary* IA), H and PB carry most of
the load, IL supports body weight in pattern B, and CL shows a large pressure peak before T_TR in
patterns B, D and E, read as the contralateral leg pushing the floor to initiate the roll.

What is measured here instead
-----------------------------
The floor's contact forces, straight from the solver: every contact between the ``floor`` geom and
a MIMo geom, ``mj_contactForce`` in the contact frame, rotated to world. Region attribution is
``model.geom_bodyid`` -- exact, where Kobayashi's was manual.

**Not `mimoTouch`.** The touch module senses MIMo's skin, spreads each contact over sensor points
and is expensive enough that training switches it off. It answers "what does MIMo feel", not
"what does the floor carry", and it would only earn its cost for the 2-D pressure *maps*, which
are not reproduced (see below).

Forces are **averaged over the physics substeps** of each env step (frame_skip 2 at 5 ms), via
``_substep_callback``, rather than read once after ``env.step``. A stiff contact can alternate
between substeps, and the last substep alone is an arbitrary sample of that; the mean is the
impulse over the 10 ms step divided by 10 ms, which is also closer to what a mat's integrating
sensor reports. Everything read alongside (sites, body frames, rho) is post-``mj_step`` state,
which describes the configuration *before* the last integration -- the same instant the contacts
were solved at. Nothing here reads ``data.qpos``.

Forces are **not filtered**. Kobayashi low-passed the marker positions (6 Hz Butterworth), not the
pressure. On the rigid floor the series are spiky -- the landing after reset reaches twice body
weight for a step -- so compare the time averages, not single samples.

Regions
-------
Assigned by subtree, so fingers, eyes and any hand variant follow their parent:

    distal arm   subtree of <side>_lower_arm     -> IA / CA
    distal leg   subtree of <side>_lower_leg     -> IL / CL
    H            subtree of head
    PB           everything else in MIMo -- including <side>_upper_arm and <side>_upper_leg

The last line is Kobayashi's definition, and it is easy to get wrong: PB absorbs the upper arms
and the thighs, which is part of why "the arms carry nothing" comes out of the paper. PB is
therefore also reported split into trunk / upper arms / upper legs, as supplementary channels.

Left and right become ipsilateral and contralateral by the direction of the roll, read the way
`eval_rollover.py` reads it (the sign of the hip and chest local y axis against global z at the
first side-lying crossing). Ipsilateral is the side MIMo rolls onto.

Normalisation
-------------
By **body weight = MIMo's mass x |g|**, from the model. Kobayashi's estimate (mean total normal
force over the 1 s before onset) is computed per episode as well and reported as a ratio to it:
their estimate is a proxy for exactly this number, and here the ground truth is available. The
ratio is also a free check on the settle: MIMo is placed by rigid kinematics at reset and is still
moving when the episode starts (CLAUDE.md, "Roll-over specifics"), and a ratio far from 1 says so.
``--normalise=kobayashi`` divides by the estimate instead.

Normalised force can legitimately exceed 1 during the roll, exactly as the paper's can: the total
vertical ground reaction force is m (g + a_z) of the centre of mass, not m g.

Time base
---------
T_TR is Kobayashi's definition applied to the ``KOBAYASHI_Torso`` site over `eval_emg.py`'s rolling
window (last flat step up to the first side-lying step): the lateral (world y) displacement
normalised to 0..1 over the window, first crossing of 0.5, interpolated between steps. Kobayashi
fitted a sigmoid first and took its midpoint; at 100 Hz on a simulated marker there is no
measurement noise to fit through.

The windows are Kobayashi's, **in seconds**, and that has a consequence to read before any table:
MIMo rolls in tens of steps, an order of magnitude faster than an infant, and episodes begin with
a policy that may start moving at step 0. A window reaching 1 s before T_TR mostly lies before the
episode starts. Each window's **coverage** (the share of it inside the episode) is reported, and
the time average is over the covered part only. ``--limb_window`` and ``--proximal_window`` move
the windows; a shorter pre-window is the obvious sensitivity check, and has to be stated as a
departure from the paper.

Beyond the mat: the tangential component
----------------------------------------
A pressure mat measures the normal component only. Kobayashi's pushing claim is inferred from a
*normal* pressure peak, but a push that sets the body rolling is largely *tangential*. Both are
recorded here. "Toward roll" is the lateral ground reaction force on MIMo, signed positive in the
direction he rolls; a leg pushing the floor away from the roll receives a positive one. The push
table compares, per region and per episode, the time-averaged normal and toward-roll forces before
T_TR (their Spearman rank correlation across episodes, and the lag between their peaks). If the
two dissociate, the normal-pressure peak is not the proxy the paper uses it as.

Roll types
----------
`eval_emg.py`'s classifier, imported: a limb is *moving* when its peak lateral speed relative to the
torso reaches ``--moving_fraction`` of the torso's own, and a moving limb is leading, synchronous
or following by the timing of that peak against the torso's. Kobayashi's stationary criterion was
absolute (< 100 mm/s within T_TR +- 0.25 s); the relative one is used so that the roll types here
are the ones `eval_emg.py` reports. Check the 'other' fraction before reading a per-type table.

What does not transfer
----------------------
* **The 2-D pressure images** (Kobayashi's Fig. 1C and Fig. 8). A capsule on a plane yields one or
  two contact points, not a field of 1 cm^2 cells. Force per region is exact; pressure per cm^2 is
  not defined. Everything here is force as a fraction of body weight.
* **The statistics.** Kobayashi used Kruskal-Wallis with Tukey-Kramer and Mann-Whitney U. scipy is
  not installed in the ``mimo`` env, so medians, quartiles and counts are reported and the tests
  are left to whoever reads the ``--json``.
* **"No pressure".** The mat cannot see below its resolution; the simulation sees any force. A
  region counts as loaded when its time-averaged normal force exceeds ``--load_threshold`` of body
  weight (default 1 %, ~0.9 N at 9 months, about what a small contact patch reads at the mat's
  0.2 N/cm^2 floor). The share of loaded episodes depends on this choice; the means do not.
* **Infants.** Kobayashi only measured supine-to-prone. A prone run is evaluated but flagged.

Checks
------
``--selfcheck`` exercises the analysis on synthetic episodes (region labels, the force sign,
T_TR, laterality, windows and coverage, normalisation, the aggregation), no MuJoCo needed.
``--physics_check`` builds one env and checks the extraction against physics: a resting MIMo's
total vertical ground reaction force equals m g, and a known horizontal force applied at the hip
is balanced by an equal and opposite tangential one -- the sign test for the component the mat
never had.
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import gymnasium as gym
import mujoco

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
from mimoEnv.eval_rollover import (
    load_run_config, env_kwargs, load_policy, floor_label, starting_position_from_path,
    FULL_ROLL_GOAL, DEFAULT_EPISODE_STEPS, LATERALITY_BODIES,
)
from mimoEnv.eval_emg import rolling_window, limb_timings, classify

FLOOR_GEOM = "floor"
# Parent of every MIMo body; bodies sharing its root body are MIMo, anything else is scenery.
MIMO_BODY = "hip"

# Anatomical labels, left/right. Relabelled to ipsi/contra per episode.
LABELS = ("L_arm", "R_arm", "L_leg", "R_leg", "H", "PB_trunk", "PB_upper_arms", "PB_upper_legs")
# Nearest ancestor wins, so a lower arm (child of the upper arm) is distal, not proximal.
SUBTREE_LABELS = {
    "left_lower_arm": "L_arm", "right_lower_arm": "R_arm",
    "left_lower_leg": "L_leg", "right_lower_leg": "R_leg",
    "head": "H",
    "left_upper_arm": "PB_upper_arms", "right_upper_arm": "PB_upper_arms",
    "left_upper_leg": "PB_upper_legs", "right_upper_leg": "PB_upper_legs",
}
REGIONS = ("IA", "IL", "CA", "CL", "H", "PB")
LIMB_REGIONS = ("IA", "IL")
SUPPLEMENTARY = ("PB_trunk", "PB_upper_arms", "PB_upper_legs")

# Kobayashi's windows around T_TR, in seconds.
LIMB_WINDOW = (-0.25, 0.25)
PROXIMAL_WINDOW = (-1.0, 0.5)
# Kobayashi's body weight: mean total pressure over this long before onset.
REST_BEFORE_ONSET_S = 1.0
# Margin of the aligned time series beyond the widest window, in seconds.
ALIGN_MARGIN_S = 0.25

KOBAYASHI_SITES = ("KOBAYASHI_Torso", "KOBAYASHI_LWrist", "KOBAYASHI_RWrist",
                   "KOBAYASHI_LAnkle", "KOBAYASHI_RAnkle")


# ----------------------------------------------------------------------------------------------
# Force extraction
# ----------------------------------------------------------------------------------------------

def label_bodies(names, parents, bodies):
    """Kobayashi region label for each MIMo body, by nearest labelled ancestor.

    Args:
        names (list[str]): Body names, indexed by body id.
        parents (array-like): Parent body id per body id; 0 is the world body.
        bodies (iterable[int]): The body ids that belong to MIMo.

    Returns:
        tuple[dict, list]: (body id -> label in LABELS, subtree roots absent from the model --
            a cut limb, which then simply carries no force).
    """
    labels = {}
    for body in bodies:
        label = "PB_trunk"
        node = int(body)
        while node > 0:
            if names[node] in SUBTREE_LABELS:
                label = SUBTREE_LABELS[names[node]]
                break
            node = int(parents[node])
        labels[int(body)] = label
    present = set(names)
    missing = sorted(name for name in SUBTREE_LABELS if name not in present)
    return labels, missing


def contact_force_on_mimo(frame, force, mimo_is_geom1):
    """World-frame force the floor exerts on MIMo at one contact.

    ``mj_contactForce`` reports the force in the contact frame, whose rows are the frame's axes
    with the normal first, pointing from geom1 to geom2; the reported force acts on geom2. So it
    is ``frame.T @ force`` on geom2 and its negative on geom1. Verified against physics by
    ``--physics_check`` (normal: total equals m g; tangential: balances an applied force).
    """
    world = np.asarray(frame, dtype=float).reshape(3, 3).T @ np.asarray(force[:3], dtype=float)
    return -world if mimo_is_geom1 else world


class ContactRecorder:
    """Accumulates the floor's force on each region over the physics substeps of an env step.

    Installs itself as the env's ``_substep_callback`` (chaining the existing one), so the forces
    of every substep enter the mean and nothing in the environment changes.
    """

    def __init__(self, env):
        self.env = env
        self.model = env.model
        model = env.model
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        if self.floor_id < 0:
            raise RuntimeError(f"No geom named '{FLOOR_GEOM}' in this scene.")
        mimo_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MIMO_BODY)
        if mimo_body < 0:
            raise RuntimeError(f"No body named '{MIMO_BODY}' in this scene.")
        root = model.body_rootid[mimo_body]
        bodies = [b for b in range(model.nbody) if model.body_rootid[b] == root]
        names = [model.body(b).name for b in range(model.nbody)]
        labels, self.missing = label_bodies(names, model.body_parentid, bodies)
        self.body_names = {b: names[b] for b in bodies}
        self.body_labels = labels
        self.body_label = np.full(model.nbody, -1, dtype=int)
        for body, label in labels.items():
            self.body_label[body] = LABELS.index(label)
        self.mass = float(model.body_mass[bodies].sum())
        self.body_weight = self.mass * float(np.linalg.norm(model.opt.gravity))
        self._f6 = np.zeros(6)
        self._sum = np.zeros((len(LABELS), 3))
        self._count = 0
        # Vertical force on the floor from bodies that are not MIMo. Should stay zero.
        self.foreign = 0.0

        previous = env._substep_callback

        def substep_callback():
            previous()
            self._accumulate()

        env._substep_callback = substep_callback

    def instant(self):
        """The floor's force on each label at this instant, (len(LABELS), 3) in newtons."""
        model, data = self.model, self.env.data
        forces = np.zeros((len(LABELS), 3))
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == self.floor_id:
                other, mimo_is_geom1 = contact.geom2, False
            elif contact.geom2 == self.floor_id:
                other, mimo_is_geom1 = contact.geom1, True
            else:
                continue
            mujoco.mj_contactForce(model, data, i, self._f6)
            force = contact_force_on_mimo(contact.frame, self._f6, mimo_is_geom1)
            label = self.body_label[model.geom_bodyid[other]]
            if label < 0:
                self.foreign += abs(force[2])
                continue
            forces[label] += force
        return forces

    def _accumulate(self):
        self._sum += self.instant()
        self._count += 1

    def discard(self):
        self._sum[:] = 0.0
        self._count = 0

    def take(self):
        """Mean force per label over the substeps since the last call; and how many there were."""
        if self.env.model is not self.model:
            raise RuntimeError("The env's model was replaced (set_embodiment?); body and geom ids "
                               "are stale. Build a new ContactRecorder.")
        if self._count == 0:
            raise RuntimeError("No substep was recorded since the last take().")
        mean = self._sum / self._count
        count = self._count
        self.discard()
        return mean, count


# ----------------------------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------------------------

def left_up(env):
    """eval_rollover's laterality signal: hip/chest local y (his left) against global z."""
    return float(np.mean([env.data.body(body).xmat.reshape(3, 3)[2, 1]
                          for body in LATERALITY_BODIES]))


def collect_episode(env, policy, recorder, seed, episode_steps):
    """Roll out one episode, recording what the ground contact analysis needs.

    Returns:
        dict: 'forces' (T, len(LABELS), 3) newtons, 'rho' (T,), 'left_up' (T,),
            'sites' {label: (T, 3)}, 'left_positive_y' (bool).
    """
    obs, _ = env.reset(seed=seed)
    # The settle steps inside reset call mj_step directly, but do not rely on it.
    recorder.discard()
    forces, rho, lateral = [], [], []
    sites = {label: [] for label in ("torso", "LWrist", "RWrist", "LAnkle", "RAnkle")}

    def sample(force):
        forces.append(force)
        rho.append(float(np.asarray(env.get_achieved_goal_cos_mean()).reshape(-1)[0]))
        lateral.append(left_up(env))
        sites["torso"].append(env.data.site("KOBAYASHI_Torso").xpos.copy())
        for side in ("L", "R"):
            sites[f"{side}Wrist"].append(env.data.site(f"KOBAYASHI_{side}Wrist").xpos.copy())
            sites[f"{side}Ankle"].append(env.data.site(f"KOBAYASHI_{side}Ankle").xpos.copy())

    left_positive_y = bool(env.data.site("KOBAYASHI_LAnkle").xpos[1]
                           > env.data.site("KOBAYASHI_RAnkle").xpos[1])
    sample(recorder.instant())
    for _ in range(episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        force, _count = recorder.take()
        sample(force)
        if terminated or truncated:
            break
    return {
        "forces": np.asarray(forces),
        "rho": np.asarray(rho),
        "left_up": np.asarray(lateral),
        "sites": {label: np.asarray(values) for label, values in sites.items()},
        "left_positive_y": left_positive_y,
    }


# ----------------------------------------------------------------------------------------------
# Analysis (pure: arrays in, numbers out)
# ----------------------------------------------------------------------------------------------

def trunk_rotation_time(torso_y, window):
    """Kobayashi's T_TR, in (fractional) steps.

    The lateral trunk displacement over the window, normalised so that the window's first step is
    0 and its last is 1, whichever way MIMo rolls; T_TR is its first crossing of 0.5.

    Returns:
        float|None: T_TR as a step index, or None if the trunk did not move laterally.
    """
    start, end = window
    y = np.asarray(torso_y[start:end + 1], dtype=float)
    span = y[-1] - y[0]
    if abs(span) < 1e-9:
        return None
    normalised = (y - y[0]) / span
    k = int(np.flatnonzero(normalised >= 0.5)[0])
    if k == 0:
        return float(start)
    low, high = normalised[k - 1], normalised[k]
    fraction = (0.5 - low) / (high - low) if high > low else 0.0
    return start + (k - 1) + float(fraction)


def roll_direction(lateral, end):
    """eval_rollover's convention: his left side pointing down at side lying means 'left'."""
    return "left" if lateral[end] < 0 else "right"


def toward_roll_sign(direction, left_positive_y):
    """+1 if rolling toward world +y, else -1. He translates toward the side he rolls onto."""
    return 1.0 if (direction == "left") == bool(left_positive_y) else -1.0


def region_forces(forces, direction):
    """Relabel per-label forces to Kobayashi's regions, plus the supplementary PB split.

    Args:
        forces (np.ndarray): (T, len(LABELS), 3).
        direction (str): 'left' or 'right'.

    Returns:
        dict[str, np.ndarray]: region -> (T, 3).
    """
    index = {label: i for i, label in enumerate(LABELS)}
    ipsi, contra = ("L", "R") if direction == "left" else ("R", "L")
    regions = {
        "IA": forces[:, index[f"{ipsi}_arm"]],
        "IL": forces[:, index[f"{ipsi}_leg"]],
        "CA": forces[:, index[f"{contra}_arm"]],
        "CL": forces[:, index[f"{contra}_leg"]],
        "H": forces[:, index["H"]],
        "PB": sum(forces[:, index[label]] for label in SUPPLEMENTARY),
    }
    for label in SUPPLEMENTARY:
        regions[label] = forces[:, index[label]]
    return regions


def window_bounds(center, window, dt):
    return center + int(round(window[0] / dt)), center + int(round(window[1] / dt))


def window_mean(values, center, window, dt):
    """Mean of 'values' over [center + window[0], center + window[1]], clipped to the episode.

    Returns:
        tuple[float, float]: (mean over the covered steps or NaN, covered share of the window).
    """
    low, high = window_bounds(center, window, dt)
    wanted = high - low + 1
    first, last = max(low, 0), min(high, len(values) - 1)
    if last < first:
        return float("nan"), 0.0
    return float(np.mean(values[first:last + 1])), (last - first + 1) / wanted


def aligned(values, center, align, dt):
    """'values' on a grid of offsets around 'center', NaN where the episode does not reach."""
    low, high = int(round(align[0] / dt)), int(round(align[1] / dt))
    index = center + np.arange(low, high + 1)
    out = np.full(index.shape, np.nan)
    valid = (index >= 0) & (index < len(values))
    out[valid] = np.asarray(values)[index[valid]]
    return out


def align_range(limb_window, proximal_window):
    return (min(limb_window[0], proximal_window[0]) - ALIGN_MARGIN_S,
            max(limb_window[1], proximal_window[1]) + ALIGN_MARGIN_S)


def peak_time(series, align, dt, threshold, bounds):
    """Offset from T_TR (s) of a series' maximum within 'bounds' (s, inclusive).

    Bounded because the push table compares peaks *before* T_TR: over the whole aligned range a
    thigh that only loads after T_TR would report a peak the table's window never saw.

    Returns:
        float|None: The offset, or None if the series never exceeds 'threshold' within bounds.
    """
    series = np.asarray(series, dtype=float)
    time = align[0] + np.arange(series.size) * dt
    inside = (time >= bounds[0] - 1e-9) & (time <= bounds[1] + 1e-9)
    series = np.where(inside, series, np.nan)
    if not np.any(np.isfinite(series)) or np.nanmax(series) <= threshold:
        return None
    return float(time[int(np.nanargmax(series))])


def analyse_episode(episode, dt, body_weight, onset_rho=0.05, limb_window=LIMB_WINDOW,
                    proximal_window=PROXIMAL_WINDOW, load_threshold=0.01, moving_fraction=0.25,
                    timing_tolerance=3, patterns=True, normalise="mass"):
    """Everything the tables need from one episode.

    Returns:
        tuple[dict|None, str|None]: (result, None) or (None, why the episode was skipped).
    """
    window = rolling_window(episode["rho"], onset_rho)
    if window is None:
        return None, "never reached side lying"
    start, end = window
    t_tr = trunk_rotation_time(episode["sites"]["torso"][:, 1], window)
    if t_tr is None:
        return None, "no lateral trunk displacement"
    center = int(round(t_tr))
    direction = roll_direction(episode["left_up"], end)
    sign = toward_roll_sign(direction, episode["left_positive_y"])

    forces = np.asarray(episode["forces"], dtype=float)
    total_z = forces[:, :, 2].sum(axis=1)
    rest_steps = int(round(REST_BEFORE_ONSET_S / dt))
    rest = total_z[max(0, start - rest_steps):start] if start > 0 else total_z[:1]
    weight_estimate = float(rest.mean())
    if normalise == "kobayashi":
        if weight_estimate <= 0:
            return None, "no load before onset (--normalise=kobayashi)"
        denominator = weight_estimate
    else:
        denominator = body_weight

    regions = region_forces(forces, direction)
    align = align_range(limb_window, proximal_window)
    result = {
        "direction": direction,
        "onset_s": start * dt,
        "side_lying_s": end * dt,
        "t_tr_s": t_tr * dt,
        "onset_to_t_tr_s": (t_tr - start) * dt,
        "weight_estimate_ratio": weight_estimate / body_weight,
        "pattern": "all",
        "limb_timings": None,
        "limb_states": None,
        "mean_normal": {}, "mean_toward": {}, "coverage": {}, "loaded": {},
        "pre_normal": {}, "pre_toward": {}, "peak_normal_time_s": {}, "peak_toward_time_s": {},
        "series_normal": {}, "series_toward": {},
    }
    if patterns:
        timings = limb_timings(episode, window, direction, moving_fraction, timing_tolerance)
        result["pattern"] = classify(timings)
        result["limb_timings"] = timings
        if timings is not None:
            result["limb_states"] = {region: ("stationary" if timings[region] == "stationary"
                                              else "moving") for region in LIMB_REGIONS}

    pre_window = (proximal_window[0], 0.0)
    for region, force in regions.items():
        normal = force[:, 2] / denominator
        toward = sign * force[:, 1] / denominator
        window_s = limb_window if region in LIMB_REGIONS else proximal_window
        mean_normal, coverage = window_mean(normal, center, window_s, dt)
        result["mean_normal"][region] = mean_normal
        result["mean_toward"][region] = window_mean(toward, center, window_s, dt)[0]
        result["coverage"][region] = coverage
        result["loaded"][region] = (None if not np.isfinite(mean_normal)
                                    else bool(mean_normal > load_threshold))
        result["pre_normal"][region] = window_mean(normal, center, pre_window, dt)[0]
        result["pre_toward"][region] = window_mean(toward, center, pre_window, dt)[0]
        series_normal = aligned(normal, center, align, dt)
        series_toward = aligned(toward, center, align, dt)
        result["series_normal"][region] = series_normal
        result["series_toward"][region] = series_toward
        result["peak_normal_time_s"][region] = peak_time(series_normal, align, dt,
                                                         load_threshold, pre_window)
        result["peak_toward_time_s"][region] = peak_time(series_toward, align, dt,
                                                         load_threshold, pre_window)
    return result, None


def describe(values):
    """Median, quartiles, mean and n of the finite entries, or None if there are none."""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if values.size == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": int(values.size), "median": float(median), "q1": float(q1), "q3": float(q3),
            "mean": float(values.mean())}


def share(flags):
    flags = [flag for flag in flags if flag is not None]
    return float(np.mean(flags)) if flags else None


def ranks(values):
    """Ranks with ties averaged."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size)
    result[order] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            result[tied] = result[tied].mean()
    return result


def spearman(x, y):
    """Spearman rank correlation over the pairs where both are finite; None if undefined."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return float(np.corrcoef(ranks(x[keep]), ranks(y[keep]))[0, 1])


def series_stats(stack):
    """Per-column mean, SD and n of a (N, K) stack with NaN padding, without all-NaN warnings."""
    stack = np.asarray(stack, dtype=float)
    finite = np.isfinite(stack)
    n = finite.sum(axis=0)
    filled = np.where(finite, stack, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, filled.sum(axis=0) / np.maximum(n, 1), np.nan)
        square = np.where(finite, (stack - mean) ** 2, 0.0)
        sd = np.where(n > 1, np.sqrt(square.sum(axis=0) / np.maximum(n - 1, 1)), np.nan)
    return mean, sd, n


def summarise(results):
    """Aggregate per-episode results into the paper's tables."""
    counts = {}
    for result in results:
        counts[result["pattern"]] = counts.get(result["pattern"], 0) + 1
    all_regions = REGIONS + SUPPLEMENTARY

    per_pattern = {}
    for pattern in sorted(counts):
        group = [r for r in results if r["pattern"] == pattern]
        per_pattern[pattern] = {
            region: {"mean_normal": describe([r["mean_normal"][region] for r in group]),
                     "mean_toward": describe([r["mean_toward"][region] for r in group]),
                     "loaded_share": share([r["loaded"][region] for r in group])}
            for region in all_regions}

    limb_states = {}
    for region in LIMB_REGIONS:
        for state in ("stationary", "moving"):
            group = [r for r in results
                     if r["limb_states"] is not None and r["limb_states"][region] == state]
            limb_states[f"{region} {state}"] = {
                "n": len(group),
                "mean_normal": describe([r["mean_normal"][region] for r in group]),
                "loaded_share": share([r["loaded"][region] for r in group]),
            }

    push = {}
    for region in all_regions:
        normal = [r["pre_normal"][region] for r in results]
        toward = [r["pre_toward"][region] for r in results]
        lags = [r["peak_toward_time_s"][region] - r["peak_normal_time_s"][region]
                for r in results
                if r["peak_toward_time_s"][region] is not None
                and r["peak_normal_time_s"][region] is not None]
        push[region] = {"pre_normal": describe(normal), "pre_toward": describe(toward),
                        "spearman": spearman(normal, toward), "peak_lag_s": describe(lags)}

    series = {}
    for pattern in sorted(counts):
        group = [r for r in results if r["pattern"] == pattern]
        series[pattern] = {}
        for region in all_regions:
            entry = {}
            for kind in ("normal", "toward"):
                mean, sd, n = series_stats([r[f"series_{kind}"][region] for r in group])
                entry[kind] = {"mean": mean, "sd": sd, "n": n}
            series[pattern][region] = entry

    # What the unmatched episodes did instead, as (IL, IA, CA, CL) -- eval_emg's PATTERNS order.
    other_timings = {}
    for result in results:
        if result["pattern"] == "other" and result["limb_timings"] is not None:
            key = " / ".join(f"{limb} {result['limb_timings'][limb]}"
                             for limb in ("IL", "IA", "CA", "CL"))
            other_timings[key] = other_timings.get(key, 0) + 1

    return {
        "counts": counts,
        "other_timings": dict(sorted(other_timings.items(), key=lambda item: -item[1])),
        "per_pattern": per_pattern,
        "limb_states": limb_states,
        "push": push,
        "series": series,
        "t_tr_s": describe([r["t_tr_s"] for r in results]),
        "onset_to_t_tr_s": describe([r["onset_to_t_tr_s"] for r in results]),
        "coverage": {region: describe([r["coverage"][region] for r in results])
                     for region in REGIONS},
        "weight_estimate_ratio": describe([r["weight_estimate_ratio"] for r in results]),
    }


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------

def _synthetic_episode(steps=80, roll_start=20, roll_steps=40, lateral_sign=1.0,
                       left_positive_y=True, body_weight=100.0):
    """A pattern-B-like roll: IL/IA ride the torso, CA moves with it, CL moves late.

    rho reaches side lying at roll_start + roll_steps / 2, and the trunk's peak speed sits at 30 %
    of the roll, well inside eval_emg's rolling window: at the window's very end a late limb's
    speed peak would be cut off and read as synchronous.

    Forces: H and PB carry the weight throughout, CL presses and pushes toward the roll between
    onset and T_TR (starting after the onset step, so the pre-onset weight estimate stays clean),
    the arms carry nothing, IL carries a small constant load.
    """
    time = np.arange(steps)
    mid = roll_start + 0.3 * roll_steps
    progress = np.clip((time - roll_start) / roll_steps, 0.0, 1.0)
    rho = progress
    torso_y = lateral_sign * 0.10 * (np.tanh((time - mid) / 4.0) + 1.0)

    def limb(offset, amplitude):
        return torso_y + lateral_sign * amplitude * np.tanh((time - mid - offset) / 4.0)

    def xyz(y):
        return np.stack([np.zeros(steps), y, np.zeros(steps)], axis=1)

    rolled_toward_positive = lateral_sign > 0
    # Rolling toward +y with left at +y means rolling left, i.e. his left side goes down.
    direction_left = rolled_toward_positive == left_positive_y
    ipsi, contra = ("L", "R") if direction_left else ("R", "L")
    sites = {
        "torso": xyz(torso_y),
        f"{ipsi}Wrist": xyz(limb(0, 0.0)),
        f"{ipsi}Ankle": xyz(limb(0, 0.0)),
        f"{contra}Wrist": xyz(limb(0, 0.10)),
        f"{contra}Ankle": xyz(limb(6, 0.10)),
    }
    lateral = np.where(progress > 0.3, -1.0 if direction_left else 1.0, 0.0)

    forces = np.zeros((steps, len(LABELS), 3))
    index = {label: i for i, label in enumerate(LABELS)}
    forces[:, index["H"], 2] = 0.2 * body_weight
    forces[:, index["PB_trunk"], 2] = 0.6 * body_weight
    forces[:, index[f"{ipsi}_leg"], 2] = 0.05 * body_weight
    push = slice(roll_start + 3, int(mid))
    forces[push, index[f"{contra}_leg"], 2] = 0.3 * body_weight
    forces[push, index[f"{contra}_leg"], 1] = lateral_sign * 0.1 * body_weight
    return {"forces": forces, "rho": rho, "left_up": lateral, "sites": sites,
            "left_positive_y": left_positive_y}, ("left" if direction_left else "right")


def selfcheck():
    """Exercise the analysis on synthetic data, with no MuJoCo env.

    A sign error in the force, a swapped ipsi/contra or an off-by-one window would all still print
    plausible tables; these are the places it would hide.
    """
    checks, failures = 0, []

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    # -- label_bodies --------------------------------------------------------------------------
    names = ["world", "mimo_location", "hip", "lower_body", "upper_body", "chest", "head",
             "left_eye", "left_upper_arm", "left_lower_arm", "left_hand", "left_fingers",
             "right_upper_leg", "right_lower_leg", "right_foot", "right_toes"]
    parents = [0, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 2, 12, 13, 14]
    labels, missing = label_bodies(names, parents, range(1, len(names)))
    expected = {"hip": "PB_trunk", "chest": "PB_trunk", "head": "H", "left_eye": "H",
                "left_upper_arm": "PB_upper_arms", "left_lower_arm": "L_arm",
                "left_hand": "L_arm", "left_fingers": "L_arm",
                "right_upper_leg": "PB_upper_legs", "right_lower_leg": "R_leg",
                "right_foot": "R_leg", "right_toes": "R_leg"}
    for name, label in expected.items():
        check(labels[names.index(name)] == label,
              f"label_bodies: {name} -> {labels[names.index(name)]}, expected {label}")
    check("right_lower_arm" in missing and "left_lower_leg" in missing
          and "left_lower_arm" not in missing,
          f"label_bodies: missing subtree roots wrong: {missing}")

    # -- contact_force_on_mimo -----------------------------------------------------------------
    up = [0, 0, 1, 1, 0, 0, 0, 1, 0]      # normal +z, tangents +x, +y
    down = [0, 0, -1, 1, 0, 0, 0, -1, 0]  # normal -z (geom1 above geom2), tangents +x, -y
    on_geom2 = contact_force_on_mimo(up, [10.0, 2.0, 3.0], mimo_is_geom1=False)
    check(np.allclose(on_geom2, [2.0, 3.0, 10.0]),
          f"contact force: floor geom1, MIMo geom2 gave {on_geom2}")
    on_geom1 = contact_force_on_mimo(down, [10.0, 2.0, 3.0], mimo_is_geom1=True)
    check(np.allclose(on_geom1, [-2.0, 3.0, 10.0]),
          f"contact force: MIMo geom1 must be pushed up, gave {on_geom1}")

    # -- trunk_rotation_time -------------------------------------------------------------------
    for sign in (+1.0, -1.0):
        y = sign * np.concatenate([np.zeros(5), np.linspace(0.0, 1.0, 11), np.ones(5)])
        t = trunk_rotation_time(y, (5, 15))
        check(t is not None and abs(t - 10.0) < 1e-9,
              f"T_TR: linear ramp over 5..15 must cross 0.5 at 10.0 (sign {sign}), got {t}")
    y = np.concatenate([np.zeros(5), np.linspace(0.0, 1.0, 5) ** 2])
    t = trunk_rotation_time(y, (5, 9))
    check(t is not None and 7.0 < t < 8.0 and abs(t - (7 + (0.5 - 0.25) / (0.5625 - 0.25))) < 1e-9,
          f"T_TR: must interpolate between steps, got {t}")
    check(trunk_rotation_time(np.zeros(20), (0, 19)) is None,
          "T_TR: a trunk that does not move must give None")

    # -- laterality ----------------------------------------------------------------------------
    check(roll_direction(np.array([0.0, -0.9]), 1) == "left", "direction: left side down is left")
    check(roll_direction(np.array([0.0, 0.9]), 1) == "right", "direction: left side up is right")
    check(toward_roll_sign("left", True) == 1.0 and toward_roll_sign("left", False) == -1.0
          and toward_roll_sign("right", True) == -1.0 and toward_roll_sign("right", False) == 1.0,
          "toward_roll_sign: wrong sign table")

    # -- region_forces -------------------------------------------------------------------------
    forces = np.zeros((1, len(LABELS), 3))
    for i in range(len(LABELS)):
        forces[0, i, 2] = 10.0 ** i
    left = region_forces(forces, "left")
    right = region_forces(forces, "right")
    value = {label: 10.0 ** i for i, label in enumerate(LABELS)}
    check(left["IA"][0, 2] == value["L_arm"] and left["CL"][0, 2] == value["R_leg"],
          "region_forces: rolling left, ipsilateral must be the left side")
    check(right["IA"][0, 2] == value["R_arm"] and right["CL"][0, 2] == value["L_leg"],
          "region_forces: rolling right, ipsilateral must be the right side")
    check(left["PB"][0, 2] == value["PB_trunk"] + value["PB_upper_arms"] + value["PB_upper_legs"],
          "region_forces: PB must include the upper arms and upper legs")
    check(np.isclose(sum(left[region][0, 2] for region in REGIONS), forces[0, :, 2].sum()),
          "region_forces: the six regions must partition the total")

    # -- windows -------------------------------------------------------------------------------
    values = np.arange(100, dtype=float)
    mean, coverage = window_mean(values, 50, (-0.25, 0.25), 0.01)
    check(np.isclose(mean, 50.0) and coverage == 1.0, f"window_mean: centred, got {mean}, {coverage}")
    mean, coverage = window_mean(values, 10, (-1.0, 0.5), 0.01)
    check(np.isclose(mean, 30.0) and np.isclose(coverage, 61 / 151),
          f"window_mean: clipped at the episode start, got {mean}, {coverage}")
    mean, coverage = window_mean(values, 10, (-1.0, -0.5), 0.01)
    check(np.isnan(mean) and coverage == 0.0, "window_mean: entirely before the episode is NaN")
    series = aligned(values, 2, (-0.05, 0.05), 0.01)
    check(series.shape == (11,) and np.all(np.isnan(series[:3])) and series[3] == 0.0
          and series[-1] == 7.0, f"aligned: NaN padding wrong, got {series}")

    # -- peak_time -----------------------------------------------------------------------------
    series = np.array([0.0, 0.5, 0.2, 0.0, 0.9, 0.0])  # offsets -0.02 .. +0.03 at dt 0.01
    check(np.isclose(peak_time(series, (-0.02, 0.03), 0.01, 0.01, (-0.02, 0.0)), -0.01),
          "peak_time: a larger peak after the bounds must be ignored")
    check(peak_time(series, (-0.02, 0.03), 0.01, 0.6, (-0.02, 0.0)) is None,
          "peak_time: nothing above threshold within bounds must be None")

    # -- analyse_episode, end to end -----------------------------------------------------------
    body_weight = 100.0
    for lateral_sign in (+1.0, -1.0):
        for left_positive_y in (True, False):
            episode, direction = _synthetic_episode(lateral_sign=lateral_sign,
                                                    left_positive_y=left_positive_y,
                                                    body_weight=body_weight)
            result, reason = analyse_episode(episode, 0.01, body_weight,
                                             limb_window=(-0.05, 0.05),
                                             proximal_window=(-0.10, 0.05))
            tag = f"(lateral {lateral_sign:+}, left +y {left_positive_y})"
            check(result is not None, f"analyse: synthetic roll skipped: {reason} {tag}")
            if result is None:
                continue
            check(result["direction"] == direction,
                  f"analyse: direction {result['direction']}, expected {direction} {tag}")
            check(result["pattern"] == "B", f"analyse: pattern {result['pattern']}, expected B {tag}")
            check(result["limb_states"] == {"IA": "stationary", "IL": "stationary"},
                  f"analyse: limb states {result['limb_states']} {tag}")
            check(np.isclose(result["mean_normal"]["H"], 0.2)
                  and np.isclose(result["mean_normal"]["IL"], 0.05),
                  f"analyse: normalisation by body weight wrong {tag}")
            check(result["loaded"]["IA"] is False and result["loaded"]["IL"] is True,
                  f"analyse: loaded flags wrong {result['loaded']} {tag}")
            check(result["pre_normal"]["CL"] > 0.2 and result["pre_toward"]["CL"] > 0.05,
                  f"analyse: CL push before T_TR must be positive in both components, got "
                  f"{result['pre_normal']['CL']}, {result['pre_toward']['CL']} {tag}")
            check(result["mean_toward"]["IA"] == 0.0, f"analyse: IA carries no tangential force {tag}")
            cl_peak = result["peak_normal_time_s"]["CL"]
            check(cl_peak is not None and -0.10 - 1e-9 <= cl_peak <= 0.0,
                  f"analyse: CL normal peak must lie in the pre-T_TR window, got {cl_peak} {tag}")
            check(result["limb_timings"] is not None and result["limb_timings"]["CL"] == "following",
                  f"analyse: limb timings must be stored, got {result['limb_timings']} {tag}")
            check(np.isclose(result["weight_estimate_ratio"], 0.85),
                  f"analyse: weight estimate {result['weight_estimate_ratio']}, expected 0.85 "
                  f"(H 0.2 + PB 0.6 + IL 0.05 before onset) {tag}")
    episode, _ = _synthetic_episode(body_weight=body_weight)
    kobayashi, _ = analyse_episode(episode, 0.01, body_weight, normalise="kobayashi")
    check(kobayashi is not None and np.isclose(kobayashi["mean_normal"]["H"], 0.2 / 0.85),
          "analyse: --normalise=kobayashi must divide by the pre-onset total")
    still = dict(episode, rho=np.zeros(80))
    check(analyse_episode(still, 0.01, body_weight)[0] is None,
          "analyse: an episode that never rolls must be skipped")

    # -- aggregation ---------------------------------------------------------------------------
    check(spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0, "spearman: monotone must be 1")
    check(np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0), "spearman: reversed must be -1")
    check(spearman([1, 1, 1], [1, 2, 3]) is None, "spearman: constant input is undefined")
    check(np.allclose(ranks([3.0, 1.0, 3.0]), [1.5, 0.0, 1.5]), "ranks: ties must be averaged")
    mean, sd, n = series_stats([[1.0, np.nan], [3.0, np.nan]])
    check(mean[0] == 2.0 and np.isnan(mean[1]) and n.tolist() == [2, 0]
          and np.isclose(sd[0], np.sqrt(2.0)), "series_stats: NaN handling wrong")
    check(describe([np.nan, None]) is None, "describe: nothing finite must be None")
    results = []
    for lateral_sign in (+1.0, -1.0):
        episode, _ = _synthetic_episode(lateral_sign=lateral_sign, body_weight=body_weight)
        results.append(analyse_episode(episode, 0.01, body_weight)[0])
    summary = summarise(results)
    check(summary["counts"] == {"B": 2}, f"summarise: counts {summary['counts']}")
    check(summary["limb_states"]["IA stationary"]["n"] == 2
          and summary["limb_states"]["IA moving"]["n"] == 0,
          "summarise: stationary/moving split wrong")
    check(summary["per_pattern"]["B"]["IA"]["loaded_share"] == 0.0,
          "summarise: loaded share wrong")
    check(summary["series"]["B"]["H"]["normal"]["mean"].shape
          == results[0]["series_normal"]["H"].shape, "summarise: series shape wrong")

    if failures:
        print(f"selfcheck: {len(failures)} of {checks} checks FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"selfcheck: all {checks} checks passed")
    return 0


def physics_check(starting_position="supine", rest_steps=300, push_steps=300, window=100,
                  tolerance=0.02, push_fraction=0.05, push_tolerance=0.10):
    """Check the extraction against physics, in one env with no policy.

    1. Every MIMo body gets a label, and the named ones get the right label.
    2. MIMo lying limp: the total vertical ground reaction force equals m g, nothing else touches
       the floor, head and trunk carry load, and each env step averages frame_skip substeps.
    3. A horizontal force applied at the hip, in +x and then +y: the tangential ground reaction
       force balances it. This is the sign test for the component the paper's mat never had.
    """
    checks, failures = 0, []

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)
        print(f"  [{'ok' if condition else 'FAIL'}] {message}")

    kwargs = env_kwargs({}, starting_position, FULL_ROLL_GOAL)
    kwargs["strip_textures"] = True
    env = gym.make("MIMoRollOver-v0", **kwargs).unwrapped
    recorder = ContactRecorder(env)
    print(f"physics_check: {starting_position}, MIMo mass {recorder.mass:.3f} kg, "
          f"body weight {recorder.body_weight:.2f} N")

    expected = {"hip": "PB_trunk", "chest": "PB_trunk", "head": "H",
                "left_upper_arm": "PB_upper_arms", "left_lower_arm": "L_arm",
                "left_hand": "L_arm", "right_lower_arm": "R_arm",
                "right_upper_leg": "PB_upper_legs", "right_lower_leg": "R_leg",
                "right_foot": "R_leg", "left_toes": "L_leg"}
    by_name = {recorder.body_names[b]: label for b, label in recorder.body_labels.items()}
    wrong = {name: by_name.get(name) for name, label in expected.items() if by_name.get(name) != label}
    check(not wrong, f"region labels of named bodies (wrong: {wrong})")
    check(not recorder.missing, f"no subtree root missing in the intact body (missing: {recorder.missing})")
    check(np.isclose(recorder.mass, float(env.model.body_mass.sum())),
          f"MIMo is the only massive body ({recorder.mass:.3f} of {env.model.body_mass.sum():.3f} kg)")

    neutral = env.actuation_model.neutral_action()
    env.reset(seed=0)
    recorder.discard()
    totals, per_label, counts = [], [], set()
    for _ in range(rest_steps):
        env.step(neutral)
        force, count = recorder.take()
        counts.add(count)
        totals.append(force.sum(axis=0))
        per_label.append(force)
    totals = np.asarray(totals)[-window:]
    per_label = np.asarray(per_label)[-window:]
    vertical = totals[:, 2].mean() / recorder.body_weight
    check(abs(vertical - 1.0) <= tolerance,
          f"at rest, total vertical GRF / m g = {vertical:.4f} (tolerance {tolerance})")
    horizontal = np.abs(totals[:, :2].mean(axis=0)).max() / recorder.body_weight
    check(horizontal <= tolerance, f"at rest, net horizontal GRF / m g = {horizontal:.4f}")
    check(counts == {env.frame_skip}, f"each env step averages frame_skip={env.frame_skip} "
                                      f"substeps (saw {sorted(counts)})")
    check(recorder.foreign == 0.0, f"nothing but MIMo touches the floor ({recorder.foreign:.3g} N)")
    share_by_label = per_label[:, :, 2].mean(axis=0) / recorder.body_weight
    print("  resting load by label: " + ", ".join(f"{label} {value:.3f}"
                                                  for label, value in zip(LABELS, share_by_label)))
    check(share_by_label[LABELS.index("H")] > 0 and share_by_label[LABELS.index("PB_trunk")] > 0,
          "head and trunk carry load lying down")

    hip = env.model.body(MIMO_BODY).id
    applied = push_fraction * recorder.body_weight
    for axis, name in ((0, "x"), (1, "y")):
        env.data.xfrc_applied[hip, :] = 0.0
        env.data.xfrc_applied[hip, axis] = applied
        recorder.discard()
        reaction = []
        for _ in range(push_steps):
            env.step(neutral)
            reaction.append(recorder.take()[0].sum(axis=0)[axis])
        env.data.xfrc_applied[hip, :] = 0.0
        balance = -np.mean(reaction[-window:]) / applied
        check(abs(balance - 1.0) <= push_tolerance,
              f"{applied:.2f} N applied at the hip along +{name}: tangential GRF balances "
              f"{balance:.3f} of it, opposite in sign (tolerance {push_tolerance})")
    env.close()

    if failures:
        print(f"physics_check: {len(failures)} of {checks} checks FAILED")
        return 1
    print(f"physics_check: all {checks} checks passed")
    return 0


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------

def _fmt(entry, digits=3):
    if entry is None:
        return "--"
    return f"{entry['median']:.{digits}f} [{entry['q1']:.{digits}f}, {entry['q3']:.{digits}f}]"


def _pct(value):
    return "--" if value is None else f"{100.0 * value:.0f} %"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def print_report(summary, analysed, episodes, skipped, args, body_weight, mass):
    print(f"rolled to side lying: {analysed}/{episodes} episodes")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")
    print()

    print("time base")
    print(f"  T_TR after episode start  {_fmt(summary['t_tr_s'])} s")
    print(f"  onset -> T_TR             {_fmt(summary['onset_to_t_tr_s'])} s   "
          f"(Kobayashi's inclusion criterion: <= 3 s)")
    limb = summary["coverage"]["IA"]
    proximal = summary["coverage"]["PB"]
    print(f"  window coverage  IA/IL {args.limb_window}: {_fmt(limb, 2)}   "
          f"CA/CL/H/PB {args.proximal_window}: {_fmt(proximal, 2)}")
    if proximal is not None and proximal["median"] < 0.75:
        print("  NOTE: most of the proximal window lies outside the episode. Its time averages "
              "cover only the part that exists; see 'Time base' in the module docstring.")
    print()

    ratio = summary["weight_estimate_ratio"]
    print(f"body weight  m g = {body_weight:.2f} N ({mass:.3f} kg), normalising by "
          f"{'m g' if args.normalise == 'mass' else 'the Kobayashi estimate'}")
    print(f"  Kobayashi estimate (total normal force, {REST_BEFORE_ONSET_S:g} s before onset) / m g: "
          f"{_fmt(ratio)}")
    if ratio is not None and abs(ratio["median"] - 1.0) > 0.05:
        print("  NOTE: the estimate is more than 5 % off m g -- MIMo is not at rest before the "
              "roll, so the paper's normalisation would not measure his weight here, and the "
              "push table's pre-T_TR averages contain the same transient.")
    print()

    counts = summary["counts"]
    print("roll types (eval_emg classifier)")
    for pattern in sorted(counts):
        print(f"  {pattern:6} {counts[pattern]:4}  ({100.0 * counts[pattern] / analysed:5.1f} %)")
    for timing, count in list(summary["other_timings"].items())[:3]:
        print(f"    other {count:4}x  {timing}")
    if counts.get("other", 0) > analysed * 0.5:
        print("  NOTE: most episodes match no Kobayashi pattern; read the per-type tables with care.")
    print()

    print(f"Fig. 7 equivalent -- ipsilateral limbs within T_TR {args.limb_window} s, "
          f"normal force / body weight, median [IQR]")
    print(f"  {'':16}{'n':>4}   {'mean normal':28}{'loaded (> ' + str(args.load_threshold) + ')':>18}")
    for key, entry in summary["limb_states"].items():
        print(f"  {key:16}{entry['n']:4}   {_fmt(entry['mean_normal']):28}"
              f"{_pct(entry['loaded_share']):>18}")
    print("  Kobayashi: >70 % of stationary IA trials show no pressure; IL contributes more.\n")

    print("Fig. 10 equivalent -- time-averaged normal force / body weight, median [IQR], per roll type")
    header = f"  {'type':6}{'n':>4}  " + "".join(f"{region:>24}" for region in REGIONS)
    print(header)
    for pattern, regions in summary["per_pattern"].items():
        row = "".join(f"{_fmt(regions[region]['mean_normal'], 2):>24}" for region in REGIONS)
        print(f"  {pattern:6}{counts[pattern]:4}  {row}")
    print("  Kobayashi: IL larger in B; CL larger in B, D, E; PB larger in A; arms small.\n")

    pre = (args.proximal_window[0], 0.0)
    print(f"Push before T_TR -- time-averaged over {pre} s, normal vs toward-roll tangential force "
          f"(both / body weight)")
    print(f"  {'region':15}{'normal':>24}{'toward roll':>24}{'Spearman':>10}{'peak lag (s)':>26}")
    for region in REGIONS + SUPPLEMENTARY:
        entry = summary["push"][region]
        rho = "--" if entry["spearman"] is None else f"{entry['spearman']:+.2f}"
        print(f"  {region:15}{_fmt(entry['pre_normal']):>24}{_fmt(entry['pre_toward']):>24}"
              f"{rho:>10}{_fmt(entry['peak_lag_s'], 2):>26}")
    print("  peak lag = toward-roll peak minus normal peak, both within the window. The paper reads a")
    print("  CL normal peak before T_TR as a push; the tangential column is what a push would be.")
    print("  Spearman is across episodes; for one deterministic policy that variation is reset noise only.\n")

    print("Supplementary -- what PB is made of, time-averaged normal force / body weight")
    for pattern, regions in summary["per_pattern"].items():
        row = "".join(f"{label:>16} {_fmt(regions[label]['mean_normal'], 2):<20}"
                      for label in SUPPLEMENTARY)
        print(f"  {pattern:6}{row}")
    print()


def plot(summary, args, dt, path):
    """Kobayashi's Fig. 9: per roll type, the six regions' mean normal force aligned at T_TR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The first six slots of the dataviz reference categorical order, which is validated on
    # adjacent pairs; side is also carried by line style, so identity is never colour alone.
    colours = {"IA": "#2a78d6", "IL": "#eb6834", "CA": "#1baf7a", "CL": "#eda100",
               "H": "#e87ba4", "PB": "#008300"}
    styles = {"IA": "-", "IL": "-", "CA": "--", "CL": "--", "H": "-", "PB": "-"}
    align = align_range(args.limb_window, args.proximal_window)

    patterns = sorted(summary["series"], key=lambda p: (p == "other", p))
    columns = min(3, len(patterns))
    rows = int(np.ceil(len(patterns) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows),
                                sharex=True, sharey=True, squeeze=False)
    for index, pattern in enumerate(patterns):
        axis = axes[index // columns][index % columns]
        n_pattern = summary["counts"][pattern]
        # Shade where fewer than half of this type's episodes reach: with Kobayashi's windows most
        # of the pre-T_TR range lies before MIMo's episode begins, and that is a finding, not
        # whitespace to crop.
        reach = (summary["series"][pattern][REGIONS[0]]["normal"]["n"]
                 >= max(1, int(np.ceil(n_pattern / 2))))
        grid_time = align[0] + np.arange(reach.size) * dt
        if reach.any():
            first = grid_time[int(np.argmax(reach))]
            last = grid_time[reach.size - 1 - int(np.argmax(reach[::-1]))]
            for low, high, label in ((align[0], first - dt / 2, "before episode start"),
                                     (last + dt / 2, align[1], "after episode end")):
                if high > low:
                    axis.axvspan(low, high, color="0.95", linewidth=0, zorder=0)
                    if high - low > 0.3:
                        axis.text((low + high) / 2, 0.03, label, fontsize=7, color="0.45",
                                  ha="center", va="bottom", transform=axis.get_xaxis_transform())
        for region in REGIONS:
            entry = summary["series"][pattern][region]["normal"]
            mean, sd, n = entry["mean"], entry["sd"], entry["n"]
            # Drop the ends where fewer than half of this type's episodes reach.
            enough = n >= max(1, int(np.ceil(n_pattern / 2)))
            time = align[0] + np.arange(mean.size) * dt
            shown = np.where(enough, mean, np.nan)
            axis.plot(time, shown, color=colours[region], linestyle=styles[region],
                      linewidth=1.5, label=region)
            if n_pattern > 1:
                # Normal force cannot be negative; a symmetric SD band would claim it can.
                axis.fill_between(time, np.where(enough, np.maximum(mean - sd, 0.0), np.nan),
                                  np.where(enough, mean + sd, np.nan),
                                  color=colours[region], alpha=0.12, linewidth=0)
        axis.axvline(0.0, color="0.25", linewidth=0.9)
        for bound in args.limb_window:
            axis.axvline(bound, color="0.55", linewidth=0.8, linestyle=(0, (4, 3)))
        for bound in args.proximal_window:
            axis.axvline(bound, color="0.55", linewidth=0.8, linestyle=(0, (1, 2)))
        axis.axhline(0.0, color="0.8", linewidth=0.8)
        axis.set_title(f"roll type {pattern}  (n = {n_pattern})", fontsize=10)
        axis.grid(True, color="0.93", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("time from T_TR (s)")
    for row in axes:
        row[0].set_ylabel("normal force / body weight")
    axes[0][0].legend(fontsize=8, ncol=2, frameon=False)
    for index in range(len(patterns), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_window(spec):
    low, high = (float(part) for part in spec.split(","))
    if not low < high:
        raise argparse.ArgumentTypeError(f"window start must be before its end: {spec!r}")
    return (low, high)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Path to a saved model_*.zip.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run the analysis assertions on synthetic data and exit. Needs no "
                             "model and no MuJoCo env.")
    parser.add_argument("--physics_check", action="store_true",
                        help="Check the force extraction against physics in one env, no model.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000,
                        help="First episode seed. The default is eval_rollover's, so the episodes "
                             "are the ones its roll rate was measured on. Default: %(default)s")
    parser.add_argument("--starting_position", default=None,
                        help="Override the posture read from the model path.")
    parser.add_argument("--limb_window", type=parse_window, default=LIMB_WINDOW,
                        help="Window around T_TR for IA and IL, seconds, 'start,end'. "
                             "Default: Kobayashi's %(default)s")
    parser.add_argument("--proximal_window", type=parse_window, default=PROXIMAL_WINDOW,
                        help="Window around T_TR for CA, CL, H and PB, seconds, 'start,end'. "
                             "Its start also bounds the push table. Default: Kobayashi's %(default)s")
    parser.add_argument("--load_threshold", type=float, default=0.01,
                        help="A region counts as loaded above this fraction of body weight. "
                             "Default: %(default)s")
    parser.add_argument("--normalise", choices=("mass", "kobayashi"), default="mass",
                        help="Divide by m g (default) or by Kobayashi's estimate, the mean total "
                             "normal force over the second before onset.")
    parser.add_argument("--onset_rho", type=float, default=0.05,
                        help="rho below which MIMo counts as not yet rolling (eval_emg's window). "
                             "Default: %(default)s")
    parser.add_argument("--moving_fraction", type=float, default=0.25,
                        help="eval_emg's moving-limb criterion. Default: %(default)s")
    parser.add_argument("--timing_tolerance", type=int, default=3,
                        help="eval_emg's synchronous band, in steps. Default: %(default)s")
    parser.add_argument("--no_patterns", action="store_true",
                        help="Pool every episode into one group instead of classifying roll types.")
    parser.add_argument("--json", default=None, help="Write the full numbers here.")
    parser.add_argument("--plot", default=None, help="Write a Figure 9 style plot here (.png).")
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(selfcheck())
    if args.physics_check:
        raise SystemExit(physics_check())
    if not args.model:
        parser.error("--model is required (or use --selfcheck / --physics_check)")

    config = load_run_config(args.model)
    starting_position = args.starting_position or starting_position_from_path(args.model)
    episode_steps = config.get("episode_steps") or DEFAULT_EPISODE_STEPS
    kwargs = env_kwargs(config, starting_position, FULL_ROLL_GOAL)
    # Nothing here renders; the texture strip is physics-neutral and halves the env's RSS.
    kwargs["strip_textures"] = True
    env = gym.make("MIMoRollOver-v0", **kwargs).unwrapped
    policy = load_policy(args.model, config.get("algorithm", "PPO"), env)
    recorder = ContactRecorder(env)

    absent = [site for site in KOBAYASHI_SITES
              if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site) < 0]
    if absent:
        raise SystemExit(f"This embodiment lacks the marker sites {absent}, which T_TR and the "
                         "roll types are measured from.")

    print(f"model               : {args.model}")
    print(f"posture             : {starting_position}")
    print(f"embodiment          : morph {config.get('morph_age', 9)} / "
          f"physio {config.get('physio_age', 9)} months"
          + (f", missing {config['missing_limb']} ({config.get('missing_limb_mode', 'cut')})"
             if config.get("missing_limb") else ""))
    print(f"actuation           : {'muscle' if config.get('use_muscle') else 'spring-damper'}")
    print(f"floor               : {floor_label(config)}")
    print(f"episodes            : {args.episodes} (episode_steps {episode_steps}, "
          f"seeds {args.seed}..{args.seed + args.episodes - 1}, dt {env.dt:g} s)")
    if recorder.missing:
        print(f"absent subtrees     : {recorder.missing} (their regions carry no force)")
    if starting_position != "supine":
        print("NOTE: Kobayashi measured supine-to-prone rolls only.")
    print()

    results, skipped = [], {}
    for episode in range(args.episodes):
        data = collect_episode(env, policy, recorder, args.seed + episode, episode_steps)
        result, reason = analyse_episode(
            data, env.dt, recorder.body_weight, onset_rho=args.onset_rho,
            limb_window=args.limb_window, proximal_window=args.proximal_window,
            load_threshold=args.load_threshold, moving_fraction=args.moving_fraction,
            timing_tolerance=args.timing_tolerance, patterns=not args.no_patterns,
            normalise=args.normalise)
        if result is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        result["seed"] = args.seed + episode
        results.append(result)
    foreign = recorder.foreign
    dt = env.dt
    env.close()

    if foreign > 0:
        print(f"NOTE: bodies other than MIMo pressed on the floor ({foreign:.3g} N summed over "
              "substeps); they are excluded from every region.\n")
    if not results:
        raise SystemExit("No episode reached side lying; nothing to report.")

    summary = summarise(results)
    print_report(summary, len(results), args.episodes, skipped, args, recorder.body_weight,
                 recorder.mass)

    if args.json:
        per_episode = [{key: value for key, value in result.items()
                        if not key.startswith("series_")} for result in results]
        payload = {
            "model": args.model,
            "starting_position": starting_position,
            "morph_age": config.get("morph_age", 9),
            "physio_age": config.get("physio_age", 9),
            "floor": floor_label(config),
            "episodes": args.episodes,
            "seed": args.seed,
            "analysed": len(results),
            "skipped": skipped,
            "dt": dt,
            "mass_kg": recorder.mass,
            "body_weight_n": recorder.body_weight,
            "normalise": args.normalise,
            "limb_window_s": args.limb_window,
            "proximal_window_s": args.proximal_window,
            "load_threshold": args.load_threshold,
            "align_range_s": align_range(args.limb_window, args.proximal_window),
            "summary": summary,
            "per_episode": per_episode,
        }
        with open(args.json, "w") as handle:
            json.dump(_jsonable(payload), handle, indent=2)
        print(f"wrote {args.json}")

    if args.plot:
        plot(summary, args, dt, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()

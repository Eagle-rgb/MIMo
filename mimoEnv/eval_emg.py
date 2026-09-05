""" Simulated EMG for MIMo's roll, matching Siegel et al. (2024), J Biomech 162:111890.

    MUJOCO_GL=osmesa python mimoEnv/eval_emg.py --model=<path/to/model_5.zip> [--episodes=50]

Requires a run trained with ``--use_muscle``. See "Why the muscle model is mandatory" below --
this script refuses a spring-damper run rather than reporting a number that cannot mean what the
paper's numbers mean.

Protocol is `eval_rollover.py`'s, imported rather than restated: ISR off, goal pinned,
``done_active=False``, deterministic actions, environment rebuilt from the run's own ``data.yml``.

What Siegel measured
--------------------
Four bilateral surface EMG channels -- erector spinae (ES), abdominals (AB), quadriceps (QUAD),
hamstrings (HAM) -- on 24 infants (6.7 +- 0.7 months), 72 rolling movements, at 2000 Hz. Raw EMG
was band-passed 35-500 Hz, notch filtered at 60 Hz, rectified, and low-pass filtered at 50 Hz to
an envelope; divided by the mean of a five-second quiet-supine rest period (maximum voluntary
isometric contractions cannot be obtained from infants); time-normalised to 0-100 % of the rolling
movement; and finally, per muscle group, divided by the maximum mean value across all six roll
types. Activation was then binned low (0-49 %), moderate (50-74 %) or high (>75 %) over the
beginning (0-24 %), middle (25-74 %), end (75-100 %) and whole of the movement.

The "rolling movement" is the supine-to-lateral-rotation portion of the roll, following Kobayashi
et al. (2016), i.e. roll initiation and not the completed roll.

Why the muscle model is mandatory
---------------------------------
Under `SpringDamperModel` every actuator is one *bidirectional* torque motor. Erector spinae and
abdominals are then not two channels but the two signs of ``act:hip_bend``, and quadriceps and
hamstrings the two signs of ``act:*_knee``. A control input of zero is indistinguishable from
"both antagonists firing hard and cancelling", so Siegel's headline result -- that every roll type
begins with *all* measured muscle groups active, i.e. co-contraction -- is not representable at
all.

`MuscleModel` represents it: ``actuation_model.activity`` is a 2N vector, the first N being the
muscle that pulls the joint in its negative direction and the second N the positive one, each in
[0, 1] with its own FMAX. That is a unipolar per-muscle activation, which is what an electrode
measures. It also arrives correctly filtered: ``_update_activity`` is a first-order lag with
``tau = 0.01 s`` toward the commanded value, run every physics step, which stands in for the
calcium kinetics and electromechanical delay that Siegel's 50 Hz envelope filter recovers from raw
EMG. The spring-damper model has no such stage, and a cutoff invented for it would be arbitrary.

`muscle_forces` is deliberately *not* used: it includes the passive component ``fp``, so it is
non-zero at zero activation, and EMG does not see passive tissue.

The mapping, and the literature behind it
-----------------------------------------
Siegel's electrode sites identify the muscles, and the sites are the standard SENIAM ones:

* **AB** -- "approximately two finger widths apart just above the belly button" is the SENIAM
  *rectus abdominis* site (~2 cm lateral to the umbilicus). Rectus abdominis is the trunk flexor;
  it has essentially no axial-rotation moment arm, which belongs to the obliques. So AB maps onto
  trunk flexion **only**, and no twist term is folded into it.
* **ES** -- "mid back on either side of the spine" is the thoracolumbar erector spinae. Erector
  spinae is the primary extensor of the vertebral column and the direct antagonist of rectus
  abdominis. So ES maps onto trunk extension.
* **QUAD** -- "midway between the knee and the anterior superior iliac spine aligned with the
  femur bone" is the SENIAM *rectus femoris* site (half way from ASIS to the superior patella).
  Rectus femoris is **biarticular**: it extends the knee and flexes the hip, on one activation.
* **HAM** -- "directly behind the QUAD sensors on the posterior side of the leg" is the mid
  posterior thigh, i.e. biceps femoris long head / semitendinosus. Also **biarticular**: knee
  flexion and hip extension.

In MIMo that gives, with the joint sign of each direction verified against the XMLs:

    ES    act:hip_bend, negative direction
    AB    act:hip_bend, positive direction
    QUAD  act:<side>_knee positive  +  act:<side>_hip_flex negative   (rectus femoris, biarticular)
    HAM   act:<side>_knee negative  +  act:<side>_hip_flex positive   (hamstrings, biarticular)

Sign derivations, from `MIMo_model.xml`:

* ``robot:hip_bend1`` has axis (0, 1, 0) and range -17..30.5 deg. A positive rotation about +y
  tilts the trunk's up-axis toward +x, which is the direction MIMo faces, i.e. flexion. Negative
  is therefore extension. The stored FMAX agrees: extensor 32.93 against flexor 22.97, and the
  erector spinae is the stronger of the pair.
* ``robot:*_knee`` has axis (0, -1, 0) and range -145..4 deg, so the large excursion at negative
  angles is flexion and positive is extension. FMAX agrees: extensor 89.27 against flexor 63.16,
  quadriceps being the stronger.
* ``robot:*_hip1`` has range -133..20 deg, so negative is hip flexion and positive hip extension.
  FMAX agrees: extensor 92.49 against flexor 71.25.

In all three the direction the anatomy calls stronger is the direction the XML gives the larger
FMAX, which is independent evidence that the sign reading is right.

`--biarticular_weight` (default 0.5) is how much of the hip term enters QUAD and HAM. Rectus
femoris and the hamstrings are single muscles with a single activation acting at two joints, while
MIMo splits those joints across two actuators, so some pooling is unavoidable; 0.5 assumes neither
joint dominates, which is the least-assumption choice given that MIMo has no muscle-specific
moment arms to weight with. ``--biarticular_weight=0`` reduces every channel to its unambiguous
mono-articular part, and is the sensitivity check to report alongside any result that depends on
the choice.

Channels are the **weighted mean** of their muscles' activations, not the sum: activation is
already dimensionless in [0, 1], and summing would make multi-muscle channels systematically
larger than single-muscle ones for no physiological reason. It also makes the channel comparable
across ages without further work -- FMAX is age-scaled (see
`mimoEnv/assets/mimo/age/generate_age_actuators.py`) but activation is not, exactly as Siegel's
rest-normalised EMG is comparable across infants of different size.

What is deliberately *not* mapped
---------------------------------
MIMo's axial rotators (``act:hip_twist``, ``act:chest_twist``) and lateral benders
(``act:hip_lean``, ``act:chest_lean``) are reported as **supplementary** channels and are kept out
of ES.

The temptation is to fold twist into ES, because Siegel's central finding is that roll types
featuring axial rotation of the torso relative to the pelvis show high ES activation. But erector
spinae is not the prime axial rotator -- in the lumbar spine it produces ipsilateral bending, while
contralateral rotation comes from multifidus and the bulk of rotation torque from the obliques and
latissimus dorsi, none of which MIMo has as separate actuators. Folding twist into ES would build
Siegel's result into the instrument that is supposed to test it. Kept separate, "does MIMo's ES
channel rise in episodes that use the twist actuator?" stays an actual question.

ES and AB have **no ipsilateral/contralateral split**. MIMo's trunk is a single sagittal chain:
``hip_bend``, ``hip_twist``, ``hip_lean``, ``chest_twist`` and ``chest_lean`` are all midline
actuators with no left and right counterpart. Siegel's bilateral ES and AB comparison (significant
for movement B and movement C respectively) therefore has no MIMo counterpart and is reported as
unavailable rather than faked. QUAD and HAM are genuinely bilateral and are split, with left/right
relabelled to ipsilateral/contralateral by the direction of the roll.

Normalisation
-------------
Siegel's first stage -- divide by the mean of a five-second rest period -- **cannot be
transferred**. MIMo's resting activation is exactly zero: `_set_initial_muscle_state` sets
``activity = 0`` and the 30 null-action settle steps after reset leave it there, so the denominator
is zero rather than small.

Only Siegel's second stage is applied: per channel, divide by the maximum, over time and over roll
types, of the per-roll-type mean envelope. That is the stage Figure 5 and Table 2 are actually
built on, it needs no baseline, and the low/moderate/high bands transfer to it unchanged. The
consequence to state when reporting: **within-channel shape and between-roll-type differences are
comparable to Siegel, absolute levels and between-channel magnitudes are not.**

Time base
---------
The environment steps at 100 Hz (timestep 0.005 s, frame_skip 2). MIMo's roll takes roughly 45
steps, about 0.45 s, against the 3.6 +- 2.8 s Siegel measured, so absolute envelope timing is not
comparable and the 0-100 % normalisation is doing real work. It also means a window holds ~45
samples, which this script resamples up to 101 grid points; the envelopes are smooth because the
activation lag already filtered them, not because of the resampling.

Roll types
----------
Kobayashi's six coordinated movements are classified from the limbs' lateral displacement relative
to the torso over the rolling window, using the ``KOBAYASHI_*`` marker sites. A limb is *moving*
when its peak speed relative to the torso reaches `--moving_fraction` of the torso's own peak
speed, and *stationary* otherwise; a moving limb is *leading*, *synchronous* or *following*
according to whether its moment of peak speed falls before, within or after `--timing_tolerance`
of the torso's. The resulting (IL, IA, CA, CL) tuple is matched against the six patterns.

This is a deliberately simple, self-contained classifier and it is not
`results/kobayashi/kobayashi16.py` -- that file fits sigmoids to the displacement traces to recover
Kobayashi's T and V quantities, which is the more faithful reconstruction and the thing to use when
the roll type is itself the result. Here the roll type is only a grouping variable for the EMG, and
episodes that match no pattern are reported as 'other' rather than forced into one. Check the
'other' fraction before reading anything into a per-pattern table.
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import gymnasium as gym

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
from mimoEnv.eval_rollover import (
    load_run_config, env_kwargs, starting_position_from_path, SIDE_LYING_THRESHOLD,
    FULL_ROLL_GOAL, DEFAULT_EPISODE_STEPS,
)

# Siegel's four channels. Each entry maps to a list of (actuator suffix, direction, kind) where
# 'direction' is 'neg' or 'pos' -- the muscle pulling the joint toward its negative or positive
# limit -- and 'kind' is 'mono' for the unambiguous single-joint term or 'bi' for the biarticular
# term whose share is set by --biarticular_weight. See the module docstring for the derivation.
TRUNK_CHANNELS = {
    "ES": [("act:hip_bend", "neg", "mono")],
    "AB": [("act:hip_bend", "pos", "mono")],
}
LIMB_CHANNELS = {
    "QUAD": [("act:{side}_knee", "pos", "mono"), ("act:{side}_hip_flex", "neg", "bi")],
    "HAM": [("act:{side}_knee", "neg", "mono"), ("act:{side}_hip_flex", "pos", "bi")],
}
# Reported alongside, with no Siegel counterpart. Directions are relabelled per episode to
# 'toward'/'away' with respect to the direction of the roll, which is what makes them readable.
SUPPLEMENTARY_CHANNELS = {
    "AXIAL_ROT": ["act:hip_twist", "act:chest_twist"],
    "LAT_FLEX": ["act:hip_lean", "act:chest_lean"],
}

# Kobayashi's six coordinated movements as (IL, IA, CA, CL) timings. 'stationary' means the limb
# does not move relative to the torso; the others are the moving limb's timing against the torso.
PATTERNS = {
    "A": ("stationary", "stationary", "synchronous", "synchronous"),
    "B": ("stationary", "stationary", "synchronous", "following"),
    "C": ("synchronous", "stationary", "synchronous", "synchronous"),
    "D": ("synchronous", "stationary", "synchronous", "following"),
    "E": ("stationary", "synchronous", "synchronous", "following"),
    "F": ("synchronous", "synchronous", "synchronous", "synchronous"),
}

# Siegel's sections of the rolling movement, as fractions of it.
SECTIONS = {"beginning": (0.00, 0.24), "middle": (0.25, 0.74), "end": (0.75, 1.00),
            "whole": (0.00, 1.00)}
# Siegel's activation bands, as fractions of the per-channel maximum.
BAND_MODERATE, BAND_HIGH = 0.50, 0.75
# Points on the 0-100 % grid every episode is resampled onto.
GRID = 101

SITES = {"IA": "KOBAYASHI_{side}Wrist", "IL": "KOBAYASHI_{side}Ankle"}


def band(value):
    """Siegel's low / moderate / high label for a normalised activation."""
    if value >= BAND_HIGH:
        return "high"
    if value >= BAND_MODERATE:
        return "moderate"
    return "low"


def muscle_index(env):
    """Map each actuator name to its index in the muscle model's 2N activity vector.

    Returns:
        tuple[dict, dict]: (name -> index of the negative-direction muscle,
                            name -> index of the positive-direction muscle).
    """
    names = [env.model.actuator(actuator).name for actuator in env.mimo_actuators]
    n_actuators = len(names)
    negative = {name: i for i, name in enumerate(names)}
    positive = {name: i + n_actuators for i, name in enumerate(names)}
    return negative, positive


def left_is_positive_y(env):
    """Which world y sign is MIMo's left, measured rather than assumed.

    The ipsilateral/contralateral relabelling needs to know which side is which, and hard-coding
    a handedness convention is exactly the kind of thing that silently inverts a figure.
    """
    left = env.data.site("KOBAYASHI_LAnkle").xpos[1]
    right = env.data.site("KOBAYASHI_RAnkle").xpos[1]
    return left > right


def collect_episode(env, policy, seed, episode_steps):
    """Roll out one episode, recording everything the EMG and roll-type analysis need.

    Returns:
        dict: 'activity' (T, 2N), 'rho' (T,), 'sites' {label: (T, 3)}, 'left_positive_y' (bool).
    """
    obs, _ = env.reset(seed=seed)
    activity, rho, sites = [], [], {label: [] for label in
                                    ["torso", "LWrist", "RWrist", "LAnkle", "RAnkle"]}

    def sample():
        activity.append(env.actuation_model.activity.copy())
        rho.append(float(np.asarray(env.get_achieved_goal_cos_mean()).reshape(-1)[0]))
        sites["torso"].append(env.data.site("KOBAYASHI_Torso").xpos.copy())
        for side in ("L", "R"):
            sites[f"{side}Wrist"].append(env.data.site(f"KOBAYASHI_{side}Wrist").xpos.copy())
            sites[f"{side}Ankle"].append(env.data.site(f"KOBAYASHI_{side}Ankle").xpos.copy())

    left_positive_y = left_is_positive_y(env)
    sample()
    for _ in range(episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        sample()
        if terminated or truncated:
            break
    return {
        "activity": np.asarray(activity),
        "rho": np.asarray(rho),
        "sites": {label: np.asarray(values) for label, values in sites.items()},
        "left_positive_y": left_positive_y,
    }


def rolling_window(rho, onset_rho):
    """Siegel's rolling movement: supine up to lateral rotation.

    The end is the first step at which rho reaches side lying, which is Kobayashi's and Siegel's
    "lateral rotation". The start is the last step before that at which MIMo was still flat, so
    the window covers roll initiation and excludes the quiet period before it.

    Args:
        rho (np.ndarray): Per-step task progress, 0 at the starting posture and 1 at a full roll.
        onset_rho (float): rho below which MIMo counts as not yet rolling.

    Returns:
        tuple[int, int]|None: (start, end) indices inclusive, or None if he never rolled.
    """
    reached = np.flatnonzero(rho >= SIDE_LYING_THRESHOLD)
    if reached.size == 0:
        return None
    end = int(reached[0])
    flat = np.flatnonzero(rho[:end + 1] <= onset_rho)
    start = int(flat[-1]) if flat.size else 0
    if end - start < 2:
        return None
    return start, end


def resample(values, start, end):
    """Resample a per-step series over [start, end] onto the 0-100 % grid."""
    source = np.linspace(0.0, 1.0, end - start + 1)
    target = np.linspace(0.0, 1.0, GRID)
    return np.interp(target, source, values[start:end + 1])


def roll_direction(episode, window):
    """Which of MIMo's sides he rolled toward.

    Returns:
        str: 'left' or 'right'.
    """
    start, end = window
    torso = episode["sites"]["torso"]
    lateral = torso[end, 1] - torso[start, 1]
    rolled_positive_y = lateral >= 0
    if rolled_positive_y == episode["left_positive_y"]:
        return "left"
    return "right"


def limb_timings(episode, window, direction, moving_fraction, timing_tolerance):
    """Classify each limb as stationary, or as leading/synchronous/following the torso.

    Speeds are lateral (world y) and relative to the torso, which is how Kobayashi separates a
    limb that moves from a body that carries it along.

    Returns:
        dict: {'IL', 'IA', 'CL', 'CA'} -> timing label.
    """
    start, end = window
    torso = episode["sites"]["torso"][start:end + 1, 1]
    torso_speed = np.abs(np.diff(torso))
    if torso_speed.max() <= 0:
        return None
    torso_peak_time = int(np.argmax(torso_speed))

    ipsi = "L" if direction == "left" else "R"
    contra = "R" if direction == "left" else "L"
    limbs = {"IA": f"{ipsi}Wrist", "IL": f"{ipsi}Ankle",
             "CA": f"{contra}Wrist", "CL": f"{contra}Ankle"}

    timings = {}
    for label, site in limbs.items():
        lateral = episode["sites"][site][start:end + 1, 1] - torso
        speed = np.abs(np.diff(lateral))
        if speed.max() < moving_fraction * torso_speed.max():
            timings[label] = "stationary"
            continue
        # A limb whose peak speed comes *after* the torso's moves once the roll is already
        # under way, which is Kobayashi's "following" limb -- the one that pushes off the ground
        # in patterns B, D and E. Earlier than the torso is "leading", i.e. it initiates.
        # Note `results/kobayashi/kobayashi16.py:get_timing_moving_limb` has this the other way
        # round (it labels a late limb 'leading'); this sign is the one the selfcheck pins.
        delta = int(np.argmax(speed)) - torso_peak_time
        if delta > timing_tolerance:
            timings[label] = "following"
        elif delta < -timing_tolerance:
            timings[label] = "leading"
        else:
            timings[label] = "synchronous"
    return timings


def classify(timings):
    """Match a limb-timing tuple against Kobayashi's six coordinated movements."""
    if timings is None:
        return "other"
    key = (timings["IL"], timings["IA"], timings["CA"], timings["CL"])
    for name, pattern in PATTERNS.items():
        if key == pattern:
            return name
    return "other"


def channel_envelopes(episode, window, direction, negative, positive, biarticular_weight):
    """Per-channel activation envelopes on the 0-100 % grid for one episode.

    Returns:
        dict[str, np.ndarray]: Channel name -> (GRID,) envelope.
    """
    start, end = window
    activity = episode["activity"]
    envelopes = {}

    def series(actuator, sign):
        index = (negative if sign == "neg" else positive)[actuator]
        return activity[:, index]

    for channel, terms in TRUNK_CHANNELS.items():
        values, weights = [], []
        for actuator, sign, _kind in terms:
            values.append(series(actuator, sign))
            weights.append(1.0)
        envelopes[channel] = resample(
            np.average(values, axis=0, weights=weights), start, end)

    ipsi = "left" if direction == "left" else "right"
    contra = "right" if direction == "left" else "left"
    for channel, terms in LIMB_CHANNELS.items():
        for label, side in (("ipsi", ipsi), ("contra", contra)):
            values, weights = [], []
            for actuator, sign, kind in terms:
                weight = 1.0 if kind == "mono" else biarticular_weight
                if weight <= 0.0:
                    continue
                values.append(series(actuator.format(side=side), sign))
                weights.append(weight)
            envelopes[f"{channel}_{label}"] = resample(
                np.average(values, axis=0, weights=weights), start, end)

    # Supplementary: split by whether the muscle rotates/bends toward the roll or away from it.
    # 'neg' and 'pos' are joint directions, so which one is 'toward' depends on the roll side;
    # the sign convention is resolved once here from the direction MIMo actually rolled.
    toward_sign = "pos" if direction == "left" else "neg"
    away_sign = "neg" if direction == "left" else "pos"
    for channel, actuators in SUPPLEMENTARY_CHANNELS.items():
        for label, sign in (("toward", toward_sign), ("away", away_sign)):
            values = [series(actuator, sign) for actuator in actuators]
            envelopes[f"{channel}_{label}"] = resample(np.mean(values, axis=0), start, end)
    return envelopes


def siegel_normalise(per_pattern):
    """Siegel's second normalisation stage.

    Each channel is divided by the maximum, over time and over roll types, of the per-roll-type
    mean envelope. The first stage (divide by a resting baseline) is not transferable -- MIMo's
    resting activation is exactly zero -- and is not applied; see the module docstring.

    Args:
        per_pattern (dict): pattern -> channel -> list of (GRID,) envelopes.

    Returns:
        tuple[dict, dict]: (pattern -> channel -> normalised mean envelope,
                            channel -> the divisor used).
    """
    channels = sorted({channel for group in per_pattern.values() for channel in group})
    means = {pattern: {channel: np.mean(group[channel], axis=0)
                       for channel in group} for pattern, group in per_pattern.items()}
    divisors = {}
    for channel in channels:
        peak = max((means[pattern][channel].max() for pattern in means
                    if channel in means[pattern]), default=0.0)
        divisors[channel] = peak if peak > 0 else 1.0
    normalised = {pattern: {channel: values / divisors[channel]
                            for channel, values in group.items()}
                  for pattern, group in means.items()}
    return normalised, divisors


def section_table(normalised):
    """Siegel's Table 2: the mean level and its band per section of the rolling movement."""
    table = {}
    for pattern, group in normalised.items():
        table[pattern] = {}
        for channel, envelope in group.items():
            entry = {}
            for section, (low, high) in SECTIONS.items():
                lo = int(round(low * (GRID - 1)))
                hi = int(round(high * (GRID - 1)))
                value = float(envelope[lo:hi + 1].mean())
                entry[section] = {"value": value, "band": band(value)}
            table[pattern][channel] = entry
    return table


def selfcheck():
    """Exercise the analysis on synthetic episodes, with no MuJoCo env.

    The pipeline between the rollout and the printed table -- windowing, roll direction, limb
    timing, the channel mapping, Siegel's normalisation and the banding -- is where a silent sign
    error or an off-by-one would survive a real run unnoticed, because every output would still
    look like a plausible envelope. A trained policy is not needed to test any of it.
    """
    checks, failures = 0, []

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    # -- rolling_window ------------------------------------------------------------------------
    rho = np.concatenate([np.zeros(10), np.linspace(0.0, 1.0, 21)])
    window = rolling_window(rho, onset_rho=0.05)
    check(window is not None, "window: a rolling episode must produce a window")
    start, end = window
    check(rho[end] >= SIDE_LYING_THRESHOLD and rho[end - 1] < SIDE_LYING_THRESHOLD,
          f"window: end must be the first step at side lying, got {end}")
    check(rho[start] <= 0.05 and (start + 1 > end or rho[start + 1] > 0.05),
          f"window: start must be the last flat step, got {start}")
    check(rolling_window(np.zeros(30), 0.05) is None,
          "window: an episode that never rolls must be excluded")
    check(rolling_window(np.linspace(0.0, 1.0, 30), 0.05) is not None,
          "window: a roll starting at step 0 must still yield a window")

    # -- resample ------------------------------------------------------------------------------
    ramp = np.arange(50, dtype=float)
    grid = resample(ramp, 10, 30)
    check(grid.shape == (GRID,), f"resample: wrong shape {grid.shape}")
    check(np.isclose(grid[0], 10.0) and np.isclose(grid[-1], 30.0),
          "resample: endpoints must be the window bounds")

    # -- roll_direction ------------------------------------------------------------------------
    steps = 20
    for left_positive_y in (True, False):
        for lateral_sign in (+1.0, -1.0):
            torso = np.zeros((steps, 3))
            torso[:, 1] = lateral_sign * np.linspace(0.0, 0.1, steps)
            episode = {"sites": {"torso": torso}, "left_positive_y": left_positive_y}
            direction = roll_direction(episode, (0, steps - 1))
            rolled_positive = lateral_sign > 0
            expected = "left" if rolled_positive == left_positive_y else "right"
            check(direction == expected,
                  f"roll_direction: left_positive_y={left_positive_y} sign={lateral_sign} "
                  f"gave {direction}, expected {expected}")

    # -- limb_timings and classify -------------------------------------------------------------
    # Torso peaks in the middle. IL/IA held on the torso (stationary), CA moving with it
    # (synchronous), CL moving late (following) -- Kobayashi's pattern B.
    steps = 41
    time = np.arange(steps)
    torso_y = np.tanh((time - 20) / 4.0) * 0.10
    def limb(offset, amplitude):
        return torso_y + np.tanh((time - 20 - offset) / 4.0) * amplitude
    sites = {
        "torso": np.stack([np.zeros(steps), torso_y, np.zeros(steps)], axis=1),
        "LWrist": np.stack([np.zeros(steps), limb(0, 0.0), np.zeros(steps)], axis=1),
        "LAnkle": np.stack([np.zeros(steps), limb(0, 0.0), np.zeros(steps)], axis=1),
        "RWrist": np.stack([np.zeros(steps), limb(0, 0.10), np.zeros(steps)], axis=1),
        "RAnkle": np.stack([np.zeros(steps), limb(10, 0.10), np.zeros(steps)], axis=1),
    }
    episode = {"sites": sites, "left_positive_y": True}
    timings = limb_timings(episode, (0, steps - 1), "left", 0.25, 3)
    check(timings["IA"] == "stationary" and timings["IL"] == "stationary",
          f"timings: ipsilateral limbs should be stationary, got {timings}")
    check(timings["CA"] == "synchronous", f"timings: CA should be synchronous, got {timings}")
    check(timings["CL"] == "following", f"timings: CL should be following, got {timings}")
    check(classify(timings) == "B", f"classify: expected pattern B, got {classify(timings)}")
    # And the mirror case: a limb peaking before the torso initiates the roll.
    early = dict(sites)
    early["RAnkle"] = np.stack([np.zeros(steps), limb(-10, 0.10), np.zeros(steps)], axis=1)
    early_timings = limb_timings({"sites": early, "left_positive_y": True},
                                 (0, steps - 1), "left", 0.25, 3)
    check(early_timings["CL"] == "leading",
          f"timings: a limb peaking before the torso must lead, got {early_timings}")
    check(classify(None) == "other", "classify: a failed timing must fall through to 'other'")
    check(classify({"IL": "leading", "IA": "leading", "CA": "leading", "CL": "leading"})
          == "other", "classify: an unmatched tuple must be 'other'")
    check(len(set(PATTERNS.values())) == 6, "patterns: the six definitions must be distinct")

    # -- channel_envelopes ---------------------------------------------------------------------
    # A fake muscle vector where exactly one muscle is on, so each channel's value is forced.
    names = ["act:hip_bend", "act:left_knee", "act:right_knee",
             "act:left_hip_flex", "act:right_hip_flex",
             "act:hip_twist", "act:chest_twist", "act:hip_lean", "act:chest_lean"]
    n = len(names)
    negative = {name: i for i, name in enumerate(names)}
    positive = {name: i + n for i, name in enumerate(names)}
    steps = 21
    activity = np.zeros((steps, 2 * n))
    activity[:, negative["act:hip_bend"]] = 1.0          # pure erector spinae
    activity[:, positive["act:left_knee"]] = 1.0         # pure left quadriceps, knee term only
    episode = {"activity": activity,
               "sites": {"torso": np.zeros((steps, 3))}, "left_positive_y": True}

    envelopes = channel_envelopes(episode, (0, steps - 1), "left", negative, positive, 0.5)
    check(np.allclose(envelopes["ES"], 1.0), "channels: ES must read the hip_bend extensor")
    check(np.allclose(envelopes["AB"], 0.0), "channels: AB must not see the extensor")
    # QUAD_ipsi = mean of (knee extensor = 1, hip flexor = 0) weighted 1.0 and 0.5 -> 1/1.5.
    check(np.allclose(envelopes["QUAD_ipsi"], 1.0 / 1.5),
          f"channels: QUAD_ipsi weighting wrong, got {envelopes['QUAD_ipsi'][0]}")
    check(np.allclose(envelopes["QUAD_contra"], 0.0),
          "channels: the contralateral side must be the other leg")
    check(np.allclose(envelopes["HAM_ipsi"], 0.0),
          "channels: HAM must not see the knee extensor")

    mono = channel_envelopes(episode, (0, steps - 1), "left", negative, positive, 0.0)
    check(np.allclose(mono["QUAD_ipsi"], 1.0),
          "channels: --biarticular_weight=0 must leave only the knee term")

    swapped = channel_envelopes(episode, (0, steps - 1), "right", negative, positive, 0.5)
    check(np.allclose(swapped["QUAD_contra"], 1.0 / 1.5),
          "channels: rolling right must move the left leg to contralateral")

    # -- siegel_normalise and section_table ----------------------------------------------------
    low = np.full(GRID, 0.2)
    high = np.concatenate([np.full(GRID // 2, 0.2), np.full(GRID - GRID // 2, 0.8)])
    per_pattern = {"A": {"ES": [low.copy()]}, "B": {"ES": [high.copy()]}}
    normalised, divisors = siegel_normalise(per_pattern)
    check(np.isclose(divisors["ES"], 0.8),
          f"normalise: divisor must be the max over roll types, got {divisors['ES']}")
    check(np.isclose(normalised["B"]["ES"].max(), 1.0),
          "normalise: the strongest roll type must reach 1.0")
    check(np.isclose(normalised["A"]["ES"].max(), 0.25),
          "normalise: a weaker roll type must stay below 1.0")

    table = section_table(normalised)
    check(table["A"]["ES"]["whole"]["band"] == "low",
          "table: 0.25 of the maximum must band as low")
    check(table["B"]["ES"]["end"]["band"] == "high",
          "table: the late half of pattern B must band as high")
    check(table["B"]["ES"]["beginning"]["band"] == "low",
          "table: the early half of pattern B must band as low")
    check(band(0.49) == "low" and band(0.5) == "moderate" and band(0.74) == "moderate"
          and band(0.75) == "high", "band: Siegel's cut points are wrong")

    print(f"eval_emg selfcheck: {checks - len(failures)}/{checks} assertions passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Path to a saved model_*.zip.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run the analysis assertions on synthetic data and exit. Needs no "
                             "model and no MuJoCo env.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starting_position", default=None,
                        help="Override the posture read from the model path.")
    parser.add_argument("--biarticular_weight", type=float, default=0.5,
                        help="Share of the hip term in QUAD and HAM. 0 keeps only the "
                             "mono-articular knee term. Default: %(default)s")
    parser.add_argument("--onset_rho", type=float, default=0.05,
                        help="rho below which MIMo counts as not yet rolling, for the window "
                             "start. Default: %(default)s")
    parser.add_argument("--moving_fraction", type=float, default=0.25,
                        help="A limb counts as moving when its peak speed relative to the torso "
                             "reaches this fraction of the torso's own. Default: %(default)s")
    parser.add_argument("--timing_tolerance", type=int, default=3,
                        help="Steps within which a moving limb counts as synchronous with the "
                             "torso rather than leading or following. Default: %(default)s")
    parser.add_argument("--no_patterns", action="store_true",
                        help="Pool every episode into one group instead of classifying roll "
                             "types. Siegel's normalisation then has a single group to take its "
                             "maximum over, which is well defined but loses the comparison.")
    parser.add_argument("--json", default=None, help="Write the full numbers here.")
    parser.add_argument("--plot", default=None, help="Write a Figure 5 style plot here (.png).")
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(selfcheck())
    if not args.model:
        parser.error("--model is required (or use --selfcheck)")

    config = load_run_config(args.model)
    if not config.get("use_muscle", False):
        raise SystemExit(
            "This run was not trained with --use_muscle, so it has no per-muscle activations and\n"
            "no EMG analogue. Under the spring-damper model ES and AB are the two signs of one\n"
            "torque motor, which cannot represent the co-contraction Siegel reports. Retrain with\n"
            "--use_muscle, or read the signed control input yourself and state the limitation.\n"
            f"  data.yml: {os.path.join(os.path.dirname(args.model), 'data.yml')}")

    starting_position = args.starting_position or starting_position_from_path(args.model)
    episode_steps = config.get("episode_steps") or DEFAULT_EPISODE_STEPS
    kwargs = env_kwargs(config, starting_position, FULL_ROLL_GOAL)
    env = gym.make("MIMoRollOver-v0", **kwargs).unwrapped

    from stable_baselines3 import PPO, SAC, TD3, DDPG
    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DDPG": DDPG}
    algorithm = algorithms[config.get("algorithm", "PPO")]
    # HER-saved models need the env, and their buffer is shrunk so loading does not allocate it.
    policy = algorithm.load(args.model, env=env,
                            custom_objects={"buffer_size": 1, "learning_starts": 0})

    negative, positive = muscle_index(env)
    missing = [name for group in (TRUNK_CHANNELS, LIMB_CHANNELS) for terms in group.values()
               for name, _s, _k in terms
               if name.format(side="left") not in negative]
    if missing:
        raise SystemExit(f"Actuators missing from this embodiment: {sorted(set(missing))}")

    print(f"model               : {args.model}")
    print(f"posture             : {starting_position}")
    print(f"embodiment          : morph {config.get('morph_age', 9)} / "
          f"physio {config.get('physio_age', 9)} months")
    print(f"biarticular weight  : {args.biarticular_weight}")
    print(f"episodes            : {args.episodes} (episode_steps {episode_steps})\n")

    per_pattern, counts, skipped = {}, {}, 0
    for episode in range(args.episodes):
        data = collect_episode(env, policy, args.seed + episode, episode_steps)
        window = rolling_window(data["rho"], args.onset_rho)
        if window is None:
            skipped += 1
            continue
        direction = roll_direction(data, window)
        if args.no_patterns:
            pattern = "all"
        else:
            pattern = classify(limb_timings(data, window, direction,
                                            args.moving_fraction, args.timing_tolerance))
        counts[pattern] = counts.get(pattern, 0) + 1
        envelopes = channel_envelopes(data, window, direction, negative, positive,
                                      args.biarticular_weight)
        group = per_pattern.setdefault(pattern, {})
        for channel, values in envelopes.items():
            group.setdefault(channel, []).append(values)
    env.close()

    analysed = args.episodes - skipped
    print(f"rolled to lateral rotation: {analysed}/{args.episodes} episodes "
          f"({skipped} never reached side lying and are excluded, as in Siegel)\n")
    if analysed == 0:
        raise SystemExit("No episode reached lateral rotation; nothing to report.")

    print("roll types")
    for pattern in sorted(counts):
        print(f"  {pattern:6} {counts[pattern]:4}  ({100.0 * counts[pattern] / analysed:5.1f} %)")
    if counts.get("other", 0) > analysed * 0.5:
        print("  NOTE: most episodes match no Kobayashi pattern; read the per-pattern table with "
              "care.")
    print()

    normalised, divisors = siegel_normalise(per_pattern)
    table = section_table(normalised)

    channels = sorted({channel for group in normalised.values() for channel in group})
    print("Table 2 equivalent -- mean activation per section, as a fraction of the per-channel "
          "maximum")
    print("(low < 0.50, moderate 0.50-0.74, high >= 0.75)\n")
    for pattern in sorted(table):
        print(f"  roll type {pattern}  (n = {counts[pattern]})")
        header = f"    {'channel':16}" + "".join(f"{s:>22}" for s in SECTIONS)
        print(header)
        for channel in channels:
            if channel not in table[pattern]:
                continue
            row = "".join(f"{table[pattern][channel][s]['value']:11.3f} "
                          f"{table[pattern][channel][s]['band']:>10}" for s in SECTIONS)
            print(f"    {channel:16}{row}")
        print()

    print("ipsilateral vs contralateral: available for QUAD and HAM only. MIMo's trunk actuators")
    print("are midline, so Siegel's bilateral ES and AB comparison has no counterpart here.\n")

    if args.json:
        payload = {
            "model": args.model,
            "starting_position": starting_position,
            "morph_age": config.get("morph_age", 9),
            "physio_age": config.get("physio_age", 9),
            "biarticular_weight": args.biarticular_weight,
            "episodes": args.episodes,
            "analysed": analysed,
            "skipped": skipped,
            "counts": counts,
            "divisors": {k: float(v) for k, v in divisors.items()},
            "sections": table,
            "envelopes": {pattern: {channel: values.tolist()
                                    for channel, values in group.items()}
                          for pattern, group in normalised.items()},
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")

    if args.plot:
        plot(normalised, counts, args.plot)
        print(f"wrote {args.plot}")


def plot(normalised, counts, path):
    """Siegel Figure 5 equivalent: one panel per roll type, channels over 0-100 %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patterns = sorted(normalised)
    primary = ["ES", "AB", "QUAD_ipsi", "QUAD_contra", "HAM_ipsi", "HAM_contra"]
    grid = np.linspace(0, 100, GRID)
    columns = min(3, len(patterns))
    rows = int(np.ceil(len(patterns) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows),
                                squeeze=False, sharey=True)
    for index, pattern in enumerate(patterns):
        axis = axes[index // columns][index % columns]
        axis.axhspan(0, BAND_MODERATE, color="tab:green", alpha=0.07)
        axis.axhspan(BAND_MODERATE, BAND_HIGH, color="tab:orange", alpha=0.07)
        axis.axhspan(BAND_HIGH, 1.05, color="tab:red", alpha=0.07)
        for channel in primary:
            if channel in normalised[pattern]:
                axis.plot(grid, normalised[pattern][channel], label=channel, linewidth=1.6)
        axis.set_title(f"roll type {pattern}  (n = {counts.get(pattern, 0)})")
        axis.set_xlabel("% of rolling movement")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("activation / channel max")
    axes[0][0].legend(fontsize=7, ncol=2)
    for index in range(len(patterns), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()

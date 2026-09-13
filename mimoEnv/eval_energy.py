""" Mechanical energy of MIMo's roll, matching Agrawal (2017), MSc thesis, Wichita State University.

    MUJOCO_GL=osmesa python mimoEnv/eval_energy.py --model=<path/to/model_1.zip> [--episodes=50]
    python mimoEnv/eval_energy.py --selfcheck

"Rolling pattern and energy requirements when rolling from supine to side-lying on rigid and soft
surfaces", http://hdl.handle.net/10057/15296, chapter 3. The energy counterpart of `eval_emg.py`.
Like it, the protocol is `eval_rollover.py`'s, imported rather than restated -- `resolve_run` and
`build_env`, so ISR off, goal pinned, deterministic actions, and the floor and age overrides work
exactly as they do there.

What Agrawal measured
---------------------
18 young adults (76.3 +- 13.9 kg, 173.6 +- 7.6 cm) rolled from supine to *right* side lying, five
times each, with the arms crossed over the chest or free, on a hard floor, an exercise mat and a
mattress. Markers at 50 Hz, OpenSim inverse kinematics on a 20-segment model, kinematics low-passed
at 4 Hz (zero-phase Butterworth). Then per segment (eq. 3.1-3.3):

    Ep = m g h                        h = height of the segment's centre of mass
    Ek = 1/2 m v^2 + 1/2 I w^2        v = velocity of that centre of mass

and the energy of the roll as the sum of the *positive increments* of those curves over the rolling
window (Cavagna et al. 1976). Reported: Etot, Ep, Ek (Table 3.3); per segment for right leg, left
leg, right arm, left arm, pelvis, torso (Tables 3.5-3.7); whole-body COM height at supine and at
side lying (Table 3.8). Ep is ~90 % of Etot, rotational energy ~1 % of Ek.

How the increments are combined -- read off his tables, not his text
--------------------------------------------------------------------
"Adding the positive increment of the position in potential and kinetic energy" leaves open whether
increments are taken of Ep and Ek separately or of their sum, and per segment or for the whole body.
It matters: a gain in one curve cancels a loss in the other only if they are summed first. His
tables settle it. Ep + Ek equals Etot in all six cells to the rounding (70.8 + 7.7 = 78.5 against
78.4; 79.9 + 8.9 = 88.8 exactly), and the six segments of Table 3.5 sum to Etot within 0.3 J. So

    W = sum over segments [ W+(Ep_seg) + W+(Ek_seg) ],      W+(x) = sum_t max(0, x[t+1] - x[t])

i.e. no exchange between the two energy forms and none between segments. That is the headline
number here. Within a segment the bodies' energies are summed first, because the segment is the
unit he reports. Two lower bounds are printed beside it, since "no exchange" is a bookkeeping
choice and not physics: W+ of the whole-body Ep + Ek (every exchange allowed), and M g times the
net rise of the COM. `--selfcheck` asserts the additivity on the transcribed tables, which also
catches a typo in them.

MIMo
----
Everything is read from the compiled model at the evaluated age, so growth, fractional ages and
`--missing_limb` need nothing special (a ghost limb weighs ~0 and contributes ~0).

* h is `xipos[:, 2]`, each body's COM, against the floor plane at z = 0. On a compliant floor MIMo
  sinks into the plane, as Agrawal's subjects sank ~1 cm into the mattress (Table 3.8).
* v is each body's COM velocity: `cvel` is expressed at the subtree COM and is moved to `xipos`,
  which is what `mj_objectVelocity(mjOBJ_BODY)` returns (asserted in `--selfcheck`). The joint
  velocities alone are not v: a body's COM velocity sums every joint velocity up the chain, the
  root free joint included, and `cvel` is exactly that sum.
* I w^2 is `w' diag(body_inertia) w` with w in the principal frame `ximat`, not a scalar I times
  |w|^2 -- MIMo's capsules and boxes are far from isotropic.

On every step the sum over bodies of translational plus rotational energy is checked against
1/2 qvel' M qvel, minus the rotor term 1/2 armature qvel^2 that MuJoCo adds to the diagonal of M and
that no body segment carries (MIMo's joints have armature, e.g. 0.01 on `hip_bend1`). The largest
relative error is printed; it should be round-off.

The state is copied into a private `MjData` and `mj_forward` runs on the copy. After `mj_step` the
live `env.data` kinematics describe the state before the last integration (see CLAUDE.md), and
running `mj_forward` on the live data would also refresh the actuator lengths the muscle model
reads before its next step. The copy is consistent and cannot touch the rollout.

Segments are Agrawal's six, with sides relabelled. His subjects always rolled onto their right
side, so his right limbs are the ipsilateral (down) side and his left limbs the contralateral
(lifted) side. MIMo rolls either way; the side is read off R[2, 1] of hip and chest at the end of
the window, as `eval_rollover.py` reads laterality. pelvis = `hip`; torso = `lower_body`,
`upper_body`, `chest`, `head` and the eyes (OpenSim's full-body models join the torso to the pelvis
at one lumbar joint and fold the head into it); arm = upper arm to fingers; leg = thigh to toes.

The window is `eval_emg.rolling_window`: from the last step with rho <= `--onset_rho` to the first
step at side lying (`--window=side`, Agrawal's end point) or at the full roll (`--window=full`).
His own initiation/cessation rule (local maxima of shoulder and pelvis angular velocity, cessation
at zero velocity) is not used: MIMo does not stop at side lying, so it would put the end at prone.
rho is recomputed from the copied state, so window and energies share one time base.

What does not transfer, and what to report with it
--------------------------------------------------
* **Absolute joules.** MIMo is a fraction of an adult's mass. J/kg and the dimensionless
  E / (M g L), L = body length, are printed for MIMo and for the paper; the latter compares the
  *shape* of rolling rather than body size. L is measured on the model (qpos0, upright, lowest to
  highest point of MIMo's geoms), not taken from a growth chart.
* **Ek does not scale like Ep.** Ep scales with M g L, Ek with M (L/T)^2, so their ratio is the
  Froude number Fr = L / (g T^2) of the roll, T its duration. It is printed for both. Only at equal
  Fr are the Ek shares comparable; a faster roll has a larger Ek share for purely dynamical
  reasons. Ep and the COM rise are the size- and speed-robust comparison.
* **Filtering.** Positive increments add up every wiggle, so W+ grows with sampling rate and with
  contact chatter. Agrawal filtered at 4 Hz for a ~3 s roll; the same cutoff would smooth away a
  roll as fast as MIMo's. The default is unfiltered at the env's 100 Hz. `--lowpass_hz` applies a
  second-order Butterworth forward and backward -- the fourth-order zero-phase filter as
  biomechanics usually means it -- to h, v and w before energies are formed. Report the
  sensitivity.
* **Arms crossed** has no MIMo counterpart, so the default reference column is arms uncrossed;
  `--paper_condition=crossed` prints the other.
* **Surfaces.** `--floor_softness` is a contact time constant, not calibrated to an exercise mat or
  a mattress. `--paper_surface` only picks the reference column.
* **Elastic energy** in MIMo's passive joint springs is in neither number, and has no counterpart
  in Agrawal's rigid-segment model either.

Measured 11.09.2026 -- the supine spring-damper baseline
--------------------------------------------------------
`26-08-19_supine_ep100_run_0`, 20 episodes, 20/20 rolled (all to the right), rigid floor, age 9:
8.958 kg, L = 0.710 m. Window steps ~9-37, i.e. 0.28 s to side lying. Decomposition check 1.4e-14.

                          MIMo      Agrawal (uncrossed, hard)
    Etot     E/(MgL)      0.180     0.070
    Ep       E/(MgL)      0.085     0.063
    Ek       E/(MgL)      0.095     0.007
    Ep share              47 %      90 %
    rotational of Ek      21 %      1 %
    COM rise / L          0.073     0.041
    Froude L/(gT^2)       0.94      0.017

* **Ep is robust, Ek is not.** Low-passing at 20 / 10 Hz moves Ep by -0.7 / -2.2 % and Ek by
  -15 / -25 %; M g x COM rise moves by less than 0.3 %. Ep and the COM rise are the comparison to report.
* **MIMo is not at rest at either end of the window.** Ek is already ~2.4 J at its start and ~4.2 J
  at side lying, close to its peak; peak roll rates are 316 deg/s pelvis and 489 deg/s shoulder
  against 90 / 106. Agrawal's window ends at zero angular velocity in side lying, MIMo rolls
  straight through it. That, at 54 times the Froude number, is the Ek gap -- not a technique.
  It also means W+ over the window misses whatever Ek MIMo built up before rho left 0.05.
* **The segment split is a technique difference, and survives removing Ek.** Ep-only shares
  (from Tables 3.5 minus 3.7 for Agrawal): lifted leg 9.3 % against 28.4 %, torso 59.2 % against
  39.1 %, pelvis 15.0 % against 10.8 %; unchanged at 10 Hz. Mass alone does not explain it: MIMo's
  legs are 11.3 % of his mass each (an adult's ~20 %, de Leva 1996), so per unit mass the lifted
  leg does 0.8x its share in MIMo and ~1.4x in Agrawal's adults, while the torso is ~1x in both.
  MIMo's roll is trunk-led; the adults lift and swing the upper leg.
* The 21 % against 1 % rotational share is unexplained. It is not a Froude effect -- for a roll
  about a fixed axis the rotational-to-translational ratio depends on geometry only, not on speed
  or size -- so it is either how far the COM translates sideways, or Agrawal's rotational term.
  Do not interpret it without checking which.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
from mimoEnv.eval_rollover import (
    resolve_run, build_env, load_policy, floor_label, LATERALITY_BODIES,
    SIDE_LYING_THRESHOLD, ROLL_THRESHOLD,
)
from mimoEnv.eval_emg import rolling_window, resample, GRID
from mimoEnv.utils import get_minimal_z_coordinate

MIMO_MODEL_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "assets", "mimo", "MIMo_model.xml")
ROOT_BODY = "hip"

# Agrawal's six segments as MIMo bodies. Limb sides are filled in per episode from the roll
# direction; see the module docstring for the ipsilateral/contralateral relabelling.
TRUNK_SEGMENTS = {
    "pelvis": ["hip"],
    "torso": ["lower_body", "upper_body", "chest", "head", "left_eye", "right_eye"],
}
LIMB_SEGMENTS = {
    "arm": ["{side}_upper_arm", "{side}_lower_arm", "{side}_hand", "{side}_fingers"],
    "leg": ["{side}_upper_leg", "{side}_lower_leg", "{side}_foot", "{side}_toes"],
}
# His table order: right leg, left leg, right arm, left arm, pelvis, torso.
SEGMENT_ORDER = ["leg_ipsi", "leg_contra", "arm_ipsi", "arm_contra", "pelvis", "torso"]
PAPER_SEGMENT_LABELS = {"leg_ipsi": "right leg", "leg_contra": "left leg",
                        "arm_ipsi": "right arm", "arm_contra": "left arm",
                        "pelvis": "pelvis", "torso": "torso"}

AGRAWAL_SUBJECTS = {"n": 18, "mass_kg": 76.3, "height_m": 1.736, "g": 9.81}
# Transcribed from Tables 2.3, 3.3, 3.5, 3.6, 3.7 and 3.8, means over 18 subjects. Tuples follow
# SEGMENT_ORDER. 'segment_share' is Table 3.6 as printed -- a mean of per-subject ratios, which is
# why it does not exactly equal 'segments' / 'E_tot'. Kinematics are Table 2.3 (chapter 2, same
# subjects and trials): duration from initiation to cessation, peak pelvis and shoulder angular
# velocity.
AGRAWAL = {
    "crossed": {
        "hard": dict(E_tot=78.4, E_p=70.8, E_k=7.7,
                     segments=(5.5, 26.2, 0.8, 5.0, 8.9, 31.8),
                     segment_share=(7, 33, 1, 6, 12, 41),
                     segment_ek=(0.8, 3.1, 0.4, 0.8, 0.7, 1.7),
                     com_supine=0.095, com_side=0.149,
                     duration_s=3.1, peak_pelvis_dps=88.5, peak_shoulder_dps=98.3),
        "mat": dict(E_tot=77.1, E_p=69.8, E_k=7.3,
                    segments=(5.5, 25.3, 0.8, 5.4, 9.2, 30.8),
                    segment_share=(7, 33, 1, 7, 12, 40),
                    segment_ek=(0.7, 2.9, 0.3, 0.8, 0.6, 1.6),
                    com_supine=0.078, com_side=0.132,
                    duration_s=3.1, peak_pelvis_dps=88.2, peak_shoulder_dps=99.6),
        "mattress": dict(E_tot=77.7, E_p=70.43, E_k=7.3,
                         segments=(6.0, 24.9, 0.8, 4.9, 9.6, 31.3),
                         segment_share=(8, 32, 1, 6, 12, 41),
                         segment_ek=(0.9, 3.1, 0.3, 0.8, 0.5, 1.5),
                         com_supine=0.045, com_side=0.089,
                         duration_s=3.1, peak_pelvis_dps=85.8, peak_shoulder_dps=103.1),
    },
    "uncrossed": {
        "hard": dict(E_tot=91.3, E_p=82.3, E_k=9.1,
                     segments=(5.8, 26.7, 1.4, 13.8, 9.5, 33.8),
                     segment_share=(6, 29, 1, 15, 11, 38),
                     segment_ek=(0.9, 3.3, 0.3, 1.9, 0.6, 1.6),
                     com_supine=0.079, com_side=0.150,
                     duration_s=3.2, peak_pelvis_dps=89.5, peak_shoulder_dps=106.3),
        "mat": dict(E_tot=88.8, E_p=79.9, E_k=8.9,
                    segments=(5.3, 25.3, 1.5, 13.4, 9.8, 33.3),
                    segment_share=(6, 29, 1, 15, 11, 38),
                    segment_ek=(0.9, 3.1, 0.3, 2.1, 0.7, 1.8),
                    com_supine=0.063, com_side=0.134,
                    duration_s=3.2, peak_pelvis_dps=93.8, peak_shoulder_dps=110.1),
        "mattress": dict(E_tot=87.9, E_p=79.4, E_k=8.5,
                         segments=(5.5, 24.4, 1.7, 12.3, 10.2, 33.6),
                         segment_share=(6, 28, 2, 14, 12, 38),
                         segment_ek=(0.9, 2.9, 0.3, 1.9, 0.8, 1.7),
                         com_supine=0.030, com_side=0.091,
                         duration_s=3.1, peak_pelvis_dps=93.5, peak_shoulder_dps=112.3),
    },
}
# Section 3.4: Ep ~90 % of Etot, rotational ~1 % of Ek, for every condition and surface.
AGRAWAL_EP_SHARE, AGRAWAL_ROT_SHARE = 0.90, 0.01

# Below this reference kinetic energy (J) the decomposition check compares absolutely, not
# relatively: MIMo at rest has Ek ~ 0 and a relative error there means nothing.
KE_CHECK_FLOOR = 1e-6


def segment_bodies(direction):
    """Agrawal's six segments as MIMo body names, limb sides relabelled by the roll direction.

    Args:
        direction (str): 'left' or 'right', the side MIMo rolled onto.

    Returns:
        dict[str, list[str]]: Segment name (see SEGMENT_ORDER) -> body names.
    """
    contra = "right" if direction == "left" else "left"
    segments = {name: list(bodies) for name, bodies in TRUNK_SEGMENTS.items()}
    for name, bodies in LIMB_SEGMENTS.items():
        segments[f"{name}_ipsi"] = [body.format(side=direction) for body in bodies]
        segments[f"{name}_contra"] = [body.format(side=contra) for body in bodies]
    return segments


def mimo_body_names(path=MIMO_MODEL_XML):
    """Every body name in the MIMo kinematic tree, read from the XML without compiling it."""
    return [body.get("name") for body in ET.parse(path).getroot().iter("body")]


def rho_from_xmat(xmat_hip, xmat_chest, starting_position):
    """`MIMoRollOverEnv.get_achieved_goal_cos_mean`, from given rotation matrices.

    Duplicated rather than called so that rho comes from the same copied state as the energies;
    the env's own reads `env.data`, whose kinematics lag the state by one integration step.
    """
    sign = -1.0 if starting_position == "supine" else 1.0
    return float(np.mean([(sign * xmat[2, 0] + 1.0) / 2.0 for xmat in (xmat_hip, xmat_chest)]))


def body_length(model, root=ROOT_BODY):
    """MIMo's body length: lowest to highest point of his geoms, upright at qpos0.

    The root free joint is set to the identity pose, and the lowest point is measured once upright
    and once rotated 180 deg about x. `get_minimal_z_coordinate` is exact per geom type, so the
    two together bound the body without a bounding-radius overestimate.
    """
    data = mujoco.MjData(model)
    # The scenes wrap 'hip' in a body that carries the free joint, so go up to the tree's root.
    joint = model.body_jntadr[model.body_rootid[model.body(root).id]]
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"The tree of body '{root}' has no free joint; cannot pose it upright.")
    address = model.jnt_qposadr[joint]
    lowest = []
    for quat in ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)):
        data.qpos[:] = model.qpos0
        data.qpos[address:address + 3] = 0.0
        data.qpos[address + 3:address + 7] = quat
        mujoco.mj_kinematics(model, data)
        lowest.append(get_minimal_z_coordinate(model, data))
    return float(-lowest[1] - lowest[0])


class EnergyProbe:
    """Reads what the energies need from a private copy of the simulation state.

    Args:
        model (mujoco.MjModel): The compiled model. Rebuild the probe if the env swaps it
            (`set_embodiment`); `matches` tells.
        root (str): Root body of the tree to measure; every body with that root is included.
        laterality (tuple[str]): The two bodies whose frames give rho, roll direction and roll rate,
            in the order (pelvis, shoulder girdle).
    """

    def __init__(self, model, root=ROOT_BODY, laterality=LATERALITY_BODIES):
        self.model = model
        self.scratch = mujoco.MjData(model)
        # 'hip' is not the tree's root in the scenes (a wrapper body carries the free joint), so
        # every body sharing its root is measured, the wrapper included.
        root_id = model.body_rootid[model.body(root).id]
        self.body_ids = np.flatnonzero(model.body_rootid == root_id)
        self.names = [model.body(int(i)).name for i in self.body_ids]
        self.masses = model.body_mass[self.body_ids].copy()
        self.inertia = model.body_inertia[self.body_ids].copy()
        self.g = float(np.linalg.norm(model.opt.gravity))
        self.armature = model.dof_armature.copy()
        self.laterality_ids = np.array([model.body(name).id for name in laterality])
        self._mv = np.zeros(model.nv)

    def matches(self, model):
        return model is self.model

    def constants(self):
        return {"names": self.names, "masses": self.masses, "inertia": self.inertia,
                "g": self.g, "mass": float(self.masses.sum()),
                "length": body_length(self.model)}

    def sample(self, data):
        """One sample of the state in `data`, which is only read.

        Returns:
            dict: 'z' (nb,) COM heights, 'v' (nb, 3) COM velocities, 'wp' (nb, 3) angular velocity
            in each body's principal frame, 'xmat' (2, 3, 3) of the laterality bodies, 'roll_rate'
            (2,) their angular velocity about their own longitudinal (z) axis in rad/s, 'ke_error'
            the decomposition check, 'ke_reference' 1/2 qvel' (M - armature) qvel.
        """
        model, scratch, ids = self.model, self.scratch, self.body_ids
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        if model.nmocap:
            scratch.mocap_pos[:] = data.mocap_pos
            scratch.mocap_quat[:] = data.mocap_quat
        mujoco.mj_forward(model, scratch)

        omega = scratch.cvel[ids, :3]
        offset = scratch.xipos[ids] - scratch.subtree_com[model.body_rootid[ids]]
        velocity = scratch.cvel[ids, 3:] + np.cross(omega, offset)
        ximat = scratch.ximat[ids].reshape(-1, 3, 3)
        omega_principal = np.einsum("bji,bj->bi", ximat, omega)

        ekt = 0.5 * self.masses * np.sum(velocity ** 2, axis=1)
        ekr = 0.5 * np.sum(self.inertia * omega_principal ** 2, axis=1)
        mujoco.mj_mulM(model, scratch, self._mv, scratch.qvel)
        reference = 0.5 * scratch.qvel @ self._mv - 0.5 * np.sum(self.armature * scratch.qvel ** 2)
        difference = abs(float(ekt.sum() + ekr.sum()) - reference)
        error = difference / reference if reference > KE_CHECK_FLOOR else difference

        lat = self.laterality_ids
        xmat = scratch.xmat[lat].reshape(-1, 3, 3)
        roll_rate = np.einsum("bi,bi->b", scratch.cvel[lat, :3], xmat[:, :, 2])
        return {"z": scratch.xipos[ids, 2].copy(), "v": velocity.copy(),
                "wp": omega_principal.copy(), "xmat": xmat.copy(), "roll_rate": roll_rate.copy(),
                "ke_error": float(error), "ke_reference": float(reference)}


def collect_episode(env, policy, probe, seed, episode_steps, starting_position):
    """Roll out one episode and sample the probe after reset and after every step.

    Returns:
        dict: 'z' (T, nb), 'v' (T, nb, 3), 'wp' (T, nb, 3), 'rho' (T,), 'left_up' (T,),
        'roll_rate' (T, 2), 'ke_error' (T,).
    """
    obs, _ = env.reset(seed=seed)
    samples = [probe.sample(env.data)]
    for _ in range(episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        samples.append(probe.sample(env.data))
        if terminated or truncated:
            break
    return {
        "z": np.array([s["z"] for s in samples]),
        "v": np.array([s["v"] for s in samples]),
        "wp": np.array([s["wp"] for s in samples]),
        "rho": np.array([rho_from_xmat(s["xmat"][0], s["xmat"][1], starting_position)
                         for s in samples]),
        # R[2, 1] -- the left-pointing axis against global z -- averaged over hip and chest, as
        # eval_rollover.evaluate reads laterality.
        "left_up": np.array([float(np.mean(s["xmat"][:, 2, 1])) for s in samples]),
        "roll_rate": np.array([s["roll_rate"] for s in samples]),
        "ke_error": np.array([s["ke_error"] for s in samples]),
    }


def positive_work(series):
    """Cavagna's positive increments: sum over time (axis 0) of max(0, x[t+1] - x[t])."""
    return np.clip(np.diff(series, axis=0), 0.0, None).sum(axis=0)


def lowpass(values, cutoff_hz, sample_hz):
    """Zero-phase Butterworth: second order, run forward and backward, along axis 0."""
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff_hz / (0.5 * sample_hz))
    return filtfilt(b, a, values, axis=0)


def body_energies(z, v, wp, masses, inertia, g):
    """Per-body potential, translational and rotational kinetic energy over time, each (T, nb)."""
    potential = masses * g * z
    translational = 0.5 * masses * np.sum(v ** 2, axis=-1)
    rotational = 0.5 * np.sum(inertia * wp ** 2, axis=-1)
    return potential, translational, rotational


def analyse_episode(episode, window, constants, sample_hz, lowpass_hz=None):
    """Agrawal's quantities for one rolling window.

    Args:
        episode (dict): From `collect_episode`.
        window (tuple[int, int]): (start, end), inclusive.
        constants (dict): From `EnergyProbe.constants`.
        sample_hz (float): Sampling rate of the episode, for the filter and the durations.
        lowpass_hz (float|None): Filter h, v and w before forming energies. Applied to the whole
            episode before cropping, so the window edges carry no filter transient.

    Returns:
        dict: Scalars, 'segments' {segment: {'E', 'E_p', 'E_k', 'share'}}, 'direction', and the
        whole-body 'course_p'/'course_k' on the 0-100 % grid.
    """
    start, end = window
    z, v, wp = episode["z"], episode["v"], episode["wp"]
    if lowpass_hz:
        z, v, wp = (lowpass(values, lowpass_hz, sample_hz) for values in (z, v, wp))
    masses, g = constants["masses"], constants["g"]
    potential, translational, rotational = body_energies(z, v, wp, masses, constants["inertia"], g)
    crop = slice(start, end + 1)
    potential, translational, rotational = (x[crop] for x in (potential, translational, rotational))

    direction = "left" if episode["left_up"][end] < 0 else "right"
    index = {name: i for i, name in enumerate(constants["names"])}
    segments, work_translational, work_rotational = {}, 0.0, 0.0
    for segment, bodies in segment_bodies(direction).items():
        columns = [index[body] for body in bodies if body in index]
        seg_p = potential[:, columns].sum(axis=1)
        seg_t = translational[:, columns].sum(axis=1)
        seg_r = rotational[:, columns].sum(axis=1)
        work_p = float(positive_work(seg_p))
        work_k = float(positive_work(seg_t + seg_r))
        segments[segment] = {"E": work_p + work_k, "E_p": work_p, "E_k": work_k}
        work_translational += float(positive_work(seg_t))
        work_rotational += float(positive_work(seg_r))

    e_p = sum(entry["E_p"] for entry in segments.values())
    e_k = sum(entry["E_k"] for entry in segments.values())
    e_tot = e_p + e_k
    for entry in segments.values():
        entry["share"] = entry["E"] / e_tot if e_tot > 0 else 0.0

    total_mass = float(masses.sum())
    com = (potential.sum(axis=1)) / (total_mass * g)
    kinetic = translational.sum(axis=1) + rotational.sum(axis=1)
    whole = potential.sum(axis=1) + kinetic
    rate = np.degrees(np.abs(episode["roll_rate"][crop]))
    return {
        "direction": direction,
        "E_tot": e_tot, "E_p": e_p, "E_k": e_k,
        "E_p_share": e_p / e_tot if e_tot > 0 else 0.0,
        "E_k_rot_share": (work_rotational / (work_translational + work_rotational)
                          if work_translational + work_rotational > 0 else 0.0),
        "E_exchange": float(positive_work(whole)),
        "E_com": total_mass * g * float(com[-1] - com[0]),
        "com_start": float(com[0]), "com_end": float(com[-1]),
        "duration_s": (end - start) / sample_hz,
        "peak_pelvis_dps": float(rate[:, 0].max()),
        "peak_shoulder_dps": float(rate[:, 1].max()),
        "segments": segments,
        "course_p": resample(potential.sum(axis=1) - potential.sum(axis=1)[0], 0, end - start),
        "course_k": resample(kinetic, 0, end - start),
    }


SCALARS = ["E_tot", "E_p", "E_k", "E_p_share", "E_k_rot_share", "E_exchange", "E_com",
           "com_start", "com_end", "duration_s", "peak_pelvis_dps", "peak_shoulder_dps"]


def summarise(rows):
    """Mean and SD over rolls of every scalar and every segment quantity."""
    def stats(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}

    summary = {key: stats([row[key] for row in rows]) for key in SCALARS}
    summary["segments"] = {
        segment: {key: stats([row["segments"][segment][key] for row in rows])
                  for key in ("E", "E_p", "E_k", "share")}
        for segment in SEGMENT_ORDER}
    summary["course_p"] = np.mean([row["course_p"] for row in rows], axis=0)
    summary["course_k"] = np.mean([row["course_k"] for row in rows], axis=0)
    summary["course_p_sd"] = np.std([row["course_p"] for row in rows], axis=0)
    summary["course_k_sd"] = np.std([row["course_k"] for row in rows], axis=0)
    directions = [row["direction"] for row in rows]
    summary["left"], summary["right"] = directions.count("left"), directions.count("right")
    return summary


def froude(length, duration, g):
    """Fr = L / (g T^2): the ratio of kinetic to potential energy scales for a roll of duration T."""
    return length / (g * duration ** 2)


def print_report(summary, constants, paper, label):
    mass, length, g = constants["mass"], constants["length"], constants["g"]
    p_mass, p_length, p_g = (AGRAWAL_SUBJECTS["mass_kg"], AGRAWAL_SUBJECTS["height_m"],
                             AGRAWAL_SUBJECTS["g"])
    mimo_scale, paper_scale = mass * g * length, p_mass * p_g * p_length

    print(f"Table 3.3 -- energy of the roll, positive increments (mean +- SD over rolls)")
    print(f"{'':12}{'MIMo J':>22}{'J/kg':>10}{'E/(MgL)':>10}   |{'Agrawal J':>10}{'J/kg':>8}"
          f"{'E/(MgL)':>10}   ({label})")
    for key in ("E_tot", "E_p", "E_k"):
        entry = summary[key]
        print(f"  {key:10}{entry['mean']:12.4f} +- {entry['sd']:6.4f}{entry['mean'] / mass:10.4f}"
              f"{entry['mean'] / mimo_scale:10.4f}   |{paper[key]:10.2f}{paper[key] / p_mass:8.3f}"
              f"{paper[key] / paper_scale:10.4f}")
    print(f"  {'Ep share':10}{100 * summary['E_p_share']['mean']:12.1f} %{'':30}|"
          f"{100 * AGRAWAL_EP_SHARE:10.0f} %")
    print(f"  {'rot. of Ek':10}{100 * summary['E_k_rot_share']['mean']:12.1f} %{'':30}|"
          f"{100 * AGRAWAL_ROT_SHARE:10.0f} %")
    print("  lower bounds, MIMo only:")
    for key, text in (("E_exchange", "W+ of whole-body Ep+Ek (all exchange allowed)"),
                      ("E_com", "M g x net COM rise")):
        entry = summary[key]
        print(f"    {text:48}{entry['mean']:9.4f} J  {entry['mean'] / mimo_scale:8.4f} E/(MgL)")
    print()

    print("Tables 3.5-3.7 -- per segment. ipsi = the side rolled onto (his right), "
          "contra = the lifted side (his left)")
    print(f"{'':30}{'MIMo E J':>10}{'share':>8}{'Ek J':>9}{'E/(MgL)':>9}   |"
          f"{'Agrawal J':>10}{'share':>7}{'Ek J':>6}{'E/(MgL)':>9}")
    for i, segment in enumerate(SEGMENT_ORDER):
        entry = summary["segments"][segment]
        label_segment = f"{segment} ({PAPER_SEGMENT_LABELS[segment]})"
        print(f"  {label_segment:28}{entry['E']['mean']:10.4f}{100 * entry['share']['mean']:7.1f}%"
              f"{entry['E_k']['mean']:9.4f}{entry['E']['mean'] / mimo_scale:9.4f}   |"
              f"{paper['segments'][i]:10.1f}{paper['segment_share'][i]:6d}%"
              f"{paper['segment_ek'][i]:6.1f}{paper['segments'][i] / paper_scale:9.4f}")
    print()

    print("Table 3.8 -- whole-body COM height")
    rise = summary["com_end"]["mean"] - summary["com_start"]["mean"]
    p_rise = paper["com_side"] - paper["com_supine"]
    print(f"{'':22}{'MIMo m':>10}{'/ L':>8}   |{'Agrawal m':>10}{'/ L':>8}")
    for text, value, p_value in (("start (supine)", summary["com_start"]["mean"], paper["com_supine"]),
                                 ("end of window", summary["com_end"]["mean"], paper["com_side"]),
                                 ("rise", rise, p_rise)):
        print(f"  {text:20}{value:10.4f}{value / length:8.4f}   |{p_value:10.3f}{p_value / p_length:8.4f}")
    print()

    print("Table 2.3 -- kinematics of the roll")
    duration, p_duration = summary["duration_s"]["mean"], paper["duration_s"]
    print(f"{'':22}{'MIMo':>10}   |{'Agrawal':>10}")
    print(f"  {'duration s':20}{duration:10.3f}   |{p_duration:10.1f}")
    print(f"  {'peak pelvis deg/s':20}{summary['peak_pelvis_dps']['mean']:10.1f}   |"
          f"{paper['peak_pelvis_dps']:10.1f}")
    print(f"  {'peak shoulder deg/s':20}{summary['peak_shoulder_dps']['mean']:10.1f}   |"
          f"{paper['peak_shoulder_dps']:10.1f}")
    print(f"  {'Froude L/(gT^2)':20}{froude(length, duration, g):10.4f}   |"
          f"{froude(p_length, p_duration, p_g):10.4f}")
    print("  Agrawal's duration runs from initiation to cessation (angular velocity rule); MIMo's is "
          "the rolling window.\n")


def plot(summary, constants, paper, label, path):
    """Normalised energies against Agrawal, segment shares, and the mean time course."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mimo_scale = constants["mass"] * constants["g"] * constants["length"]
    paper_scale = AGRAWAL_SUBJECTS["mass_kg"] * AGRAWAL_SUBJECTS["g"] * AGRAWAL_SUBJECTS["height_m"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    keys = ["E_tot", "E_p", "E_k"]
    x = np.arange(len(keys))
    axes[0].bar(x - 0.2, [summary[k]["mean"] / mimo_scale for k in keys], 0.4,
                yerr=[summary[k]["sd"] / mimo_scale for k in keys], label="MIMo", capsize=3)
    axes[0].bar(x + 0.2, [paper[k] / paper_scale for k in keys], 0.4,
                label=f"Agrawal ({label})")
    axes[0].set_xticks(x, ["$E_{tot}$", "$E_p$", "$E_k$"])
    axes[0].set_ylabel("E / (M g L)")
    axes[0].legend(fontsize=8)

    x = np.arange(len(SEGMENT_ORDER))
    axes[1].bar(x - 0.2, [100 * summary["segments"][s]["share"]["mean"] for s in SEGMENT_ORDER],
                0.4, yerr=[100 * summary["segments"][s]["share"]["sd"] for s in SEGMENT_ORDER],
                label="MIMo", capsize=3)
    axes[1].bar(x + 0.2, paper["segment_share"], 0.4, label="Agrawal")
    axes[1].set_xticks(x, [s.replace("_", "\n") for s in SEGMENT_ORDER], fontsize=8)
    axes[1].set_ylabel("share of $E_{tot}$ (%)")

    grid = np.linspace(0, 100, GRID)
    for key, name in (("course_p", "$E_p - E_p(0)$"), ("course_k", "$E_k$")):
        mean, sd = summary[key] / mimo_scale, summary[f"{key}_sd"] / mimo_scale
        axes[2].plot(grid, mean, label=name)
        axes[2].fill_between(grid, mean - sd, mean + sd, alpha=0.2)
    axes[2].set_xlabel("% of rolling window")
    axes[2].set_ylabel("E / (M g L)")
    axes[2].set_xlim(0, 100)
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def selfcheck():
    """Exercise the probe on a toy MuJoCo model and the analysis on synthetic episodes.

    Needs no MIMo env. The toy model is where the physics is checked -- COM velocity transport,
    principal-frame inertia, armature -- because on MIMo an error there would still produce
    plausible-looking joules.
    """
    checks, failures = 0, []

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    # -- positive_work -------------------------------------------------------------------------
    check(np.isclose(positive_work(np.array([0.0, 1.0, 0.0, 1.0])), 2.0),
          "positive_work: 0,1,0,1 must count both rises and no fall")
    check(np.isclose(positive_work(np.linspace(0.0, 0.3, 17)), 0.3),
          "positive_work: a ramp must return its rise")
    check(np.isclose(positive_work(np.full(9, 5.0)), 0.0), "positive_work: a constant is 0")
    check(np.allclose(positive_work(np.stack([np.arange(4.0), -np.arange(4.0)], axis=1)),
                      [3.0, 0.0]), "positive_work: must work per column along axis 0")

    # -- lowpass -------------------------------------------------------------------------------
    t = np.arange(200) / 100.0
    check(np.allclose(lowpass(np.full(200, 2.0), 4.0, 100.0), 2.0),
          "lowpass: must preserve a constant")
    fast = lowpass(np.sin(2 * np.pi * 40.0 * t), 4.0, 100.0)
    check(np.abs(fast[50:150]).max() < 0.05, "lowpass: 40 Hz must be removed at a 4 Hz cutoff")

    # -- segments ------------------------------------------------------------------------------
    names = mimo_body_names()
    check(len(names) == len(set(names)) and ROOT_BODY in names,
          "segments: MIMo_model.xml body names must be unique and include the root")
    for direction in ("left", "right"):
        assigned = [body for bodies in segment_bodies(direction).values() for body in bodies]
        check(sorted(assigned) == sorted(names),
              f"segments ({direction}): every MIMo body must be in exactly one segment")
    check(segment_bodies("left")["leg_ipsi"][0] == "left_upper_leg"
          and segment_bodies("right")["leg_contra"][0] == "left_upper_leg",
          "segments: ipsilateral must be the side rolled onto")
    check(set(segment_bodies("left")) == set(SEGMENT_ORDER), "segments: order must name all six")

    # -- rho and the window --------------------------------------------------------------------
    flat, turned = np.eye(3), np.eye(3)
    flat[2, 0] = -1.0
    turned[2, 0] = 1.0
    check(np.isclose(rho_from_xmat(flat, flat, "supine"), 1.0)
          and np.isclose(rho_from_xmat(turned, turned, "supine"), 0.0)
          and np.isclose(rho_from_xmat(turned, turned, "prone"), 1.0),
          "rho: supine at R[2,0]=+1 must read 0, and prone the mirror")
    rho = np.concatenate([np.zeros(5), np.linspace(0.0, 1.0, 21)])
    side, full = rolling_window(rho, 0.05, SIDE_LYING_THRESHOLD), rolling_window(rho, 0.05, ROLL_THRESHOLD)
    check(side is not None and full is not None and full[1] > side[1] and full[0] == side[0],
          "window: --window=full must end later than side lying from the same start")

    # -- the probe, on a toy model -------------------------------------------------------------
    toy = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="a" pos="0 0 1">
          <freejoint/>
          <geom type="capsule" size=".05 .2" mass="2" euler="90 0 0"/>
          <body name="b" pos="0 0 .3">
            <joint type="hinge" axis="0 1 0" armature="0.01"/>
            <geom type="box" size=".05 .1 .15" mass="1" pos=".02 0 .1" euler="0 30 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>""")
    probe = EnergyProbe(toy, root="a", laterality=("a", "b"))
    data = mujoco.MjData(toy)
    rng = np.random.default_rng(0)
    data.qpos[3:7] = rng.normal(size=4)
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qpos[7] = 0.4
    data.qvel[:] = rng.normal(size=toy.nv)
    before = (data.qpos.copy(), data.qvel.copy(), data.xpos.copy())
    sample = probe.sample(data)
    check(np.array_equal(data.qpos, before[0]) and np.array_equal(data.qvel, before[1])
          and np.array_equal(data.xpos, before[2]), "probe: must not write to the data it reads")

    reference = mujoco.MjData(toy)
    reference.qpos[:], reference.qvel[:] = data.qpos, data.qvel
    mujoco.mj_forward(toy, reference)
    velocity = np.zeros(6)
    for i, body in enumerate(probe.body_ids):
        mujoco.mj_objectVelocity(toy, reference, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity, 0)
        check(np.allclose(sample["v"][i], velocity[3:], atol=1e-12),
              f"probe: COM velocity of body {i} must equal mj_objectVelocity(mjOBJ_BODY)")
        principal = reference.ximat[body].reshape(3, 3).T @ velocity[:3]
        check(np.allclose(sample["wp"][i], principal, atol=1e-12),
              f"probe: angular velocity of body {i} must be in the principal frame")
    check(not np.allclose(reference.xipos[probe.body_ids], reference.xpos[probe.body_ids]),
          "probe: the toy model must offset a COM from its body origin, or velocity transport "
          "goes untested")
    check(sample["ke_error"] < 1e-10,
          f"probe: body energies must sum to 1/2 qvel'(M - armature) qvel, error {sample['ke_error']:.2e}")
    potential, translational, rotational = body_energies(
        sample["z"][None], sample["v"][None], sample["wp"][None], probe.masses, probe.inertia, probe.g)
    check(np.isclose(translational.sum() + rotational.sum(), sample["ke_reference"], rtol=1e-10),
          "body_energies: must agree with the probe's own decomposition check")
    # Rotor energy is not segment energy: with armature ignored the check must fail.
    full_matrix = 0.5 * data.qvel @ probe._mv
    check(not np.isclose(full_matrix, sample["ke_reference"]),
          "probe: the toy's armature must be large enough to matter, or its subtraction is untested")

    lifted = mujoco.MjData(toy)
    lifted.qpos[:], lifted.qvel[:] = data.qpos, data.qvel
    lifted.qpos[2] += 0.1
    rise = (probe.masses * probe.g * (probe.sample(lifted)["z"] - sample["z"])).sum()
    check(np.isclose(rise, 3.0 * probe.g * 0.1), "probe: lifting the root 0.1 m must add M g 0.1")
    check(np.isclose(body_length(toy, root="a"), 0.05 + 0.3 + 0.1
                     + 0.15 * np.cos(np.radians(30)) + 0.05 * np.sin(np.radians(30)), atol=1e-9),
          f"body_length: toy must measure capsule bottom to tilted box top, got {body_length(toy, 'a')}")

    # -- analyse_episode, on synthetic episodes ------------------------------------------------
    nb, steps = len(names), 31
    constants = {"names": names, "masses": np.ones(nb), "inertia": np.ones((nb, 3)), "g": 10.0}
    index = {name: i for i, name in enumerate(names)}

    def episode(left_up=-1.0):
        return {"z": np.zeros((steps, nb)), "v": np.zeros((steps, nb, 3)),
                "wp": np.zeros((steps, nb, 3)), "left_up": np.full(steps, left_up),
                "roll_rate": np.zeros((steps, 2))}

    lift = episode()
    lift["z"][:, index["hip"]] = np.concatenate([np.linspace(0.0, 0.1, 21), np.linspace(0.1, 0.0, 10)])
    result = analyse_episode(lift, (0, 20), constants, 100.0)
    check(np.isclose(result["segments"]["pelvis"]["E_p"], 1.0) and np.isclose(result["E_tot"], 1.0),
          f"analyse: lifting 1 kg by 0.1 m at g=10 must cost 1 J, got {result['E_tot']}")
    check(np.isclose(result["E_com"], 1.0) and np.isclose(result["com_end"] - result["com_start"],
                                                          0.1 / nb),
          "analyse: the COM lower bound must equal M g x COM rise")
    check(np.isclose(analyse_episode(lift, (0, 30), constants, 100.0)["E_tot"], 1.0),
          "analyse: the fall after the peak must not count")
    check(np.isclose(sum(s["share"] for s in result["segments"].values()), 1.0)
          and np.isclose(result["segments"]["pelvis"]["share"], 1.0),
          "analyse: segment shares must sum to 1")

    # Ep rises exactly as Ek falls: no exchange counts the rise, full exchange counts nothing.
    swap = episode()
    s = np.linspace(0.0, 1.0, steps)
    swap["z"][:, index["hip"]] = s / 10.0
    swap["v"][:, index["hip"], 0] = np.sqrt(2.0 * (1.0 - s))
    result = analyse_episode(swap, (0, steps - 1), constants, 100.0)
    check(np.isclose(result["E_tot"], 1.0) and np.isclose(result["E_exchange"], 0.0, atol=1e-9),
          f"analyse: Ep<->Ek exchange must count under E_tot and not under E_exchange, got "
          f"{result['E_tot']}, {result['E_exchange']}")

    for left_up, segment in ((-1.0, "leg_ipsi"), (+1.0, "leg_contra")):
        leg = episode(left_up)
        leg["z"][:, index["left_upper_leg"]] = np.linspace(0.0, 0.1, steps)
        result = analyse_episode(leg, (0, steps - 1), constants, 100.0)
        check(np.isclose(result["segments"][segment]["E_p"], 1.0),
              f"analyse: left_up={left_up} must put the left leg in {segment}")

    spin = episode()
    spin["wp"][:, index["head"], 2] = np.linspace(0.0, 1.0, steps)
    spin["v"][:, index["head"], 0] = np.linspace(0.0, 1.0, steps)
    result = analyse_episode(spin, (0, steps - 1), constants, 100.0)
    check(np.isclose(result["E_k_rot_share"], 0.5) and np.isclose(result["segments"]["torso"]["E_k"], 1.0),
          "analyse: equal translational and rotational rise must give a 50 % rotational share")

    # -- the transcribed tables ----------------------------------------------------------------
    for condition, surfaces in AGRAWAL.items():
        for surface, cell in surfaces.items():
            name = f"{condition}/{surface}"
            check(abs(cell["E_p"] + cell["E_k"] - cell["E_tot"]) <= 0.15,
                  f"paper {name}: Ep + Ek must equal Etot (Table 3.3)")
            check(abs(sum(cell["segments"]) - cell["E_tot"]) <= 0.35,
                  f"paper {name}: segments must sum to Etot (Table 3.5)")
            # Up to 0.5 J: uncrossed/hard is 8.6 against 9.1 in the thesis itself, more than six
            # one-decimal roundings allow, so this bound is the thesis' consistency, not ours.
            check(abs(sum(cell["segment_ek"]) - cell["E_k"]) <= 0.55,
                  f"paper {name}: segment Ek must sum to Ek (Table 3.7)")
            check(abs(sum(cell["segment_share"]) - 100) <= 2,
                  f"paper {name}: Table 3.6 shares must sum to 100 %")
            check(cell["com_side"] > cell["com_supine"], f"paper {name}: COM must rise")

    print(f"eval_energy selfcheck: {checks - len(failures)}/{checks} assertions passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Path to a saved model_*.zip.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run the assertions on a toy model and synthetic data and exit.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000,
                        help="First episode seed. Default: %(default)s, eval_rollover's, so the "
                             "episodes are the ones its roll rate was measured on.")
    parser.add_argument("--starting_position", default=None,
                        help="Override the posture read from the model path.")
    parser.add_argument("--episode_steps", type=int, default=None)
    parser.add_argument("--onset_rho", type=float, default=0.05,
                        help="rho below which MIMo counts as not yet rolling. Default: %(default)s")
    parser.add_argument("--window", choices=("side", "full"), default="side",
                        help="End the window at side lying (Agrawal's end point) or at the full "
                             "roll. Default: %(default)s")
    parser.add_argument("--lowpass_hz", type=float, default=None,
                        help="Zero-phase low-pass on h, v and w before forming energies. Agrawal "
                             "used 4 Hz on a ~3 s roll. Default: unfiltered.")
    parser.add_argument("--paper_condition", choices=("uncrossed", "crossed"), default="uncrossed")
    parser.add_argument("--paper_surface", choices=("hard", "mat", "mattress"), default="hard")
    # The overrides eval_rollover.resolve_run understands, with its meaning.
    parser.add_argument("--floor_softness", type=float, default=None)
    parser.add_argument("--floor_friction", type=float, default=None)
    parser.add_argument("--floor_solimp_width", type=float, default=None)
    parser.add_argument("--rigid_floor", action="store_true")
    parser.add_argument("--physio_age", type=float, default=None)
    parser.add_argument("--morph_age", type=float, default=None)
    parser.add_argument("--json", default=None, help="Write the full numbers here.")
    parser.add_argument("--plot", default=None, help="Write a comparison figure here (.png).")
    parser.set_defaults(goal=None, missing_limb=None, ghost_obs=None, missing_limb_mode=None)
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(selfcheck())
    if not args.model:
        parser.error("--model is required (or use --selfcheck)")

    config, start, goal, episode_steps = resolve_run(args.model, args)
    env = build_env(config, start, goal)
    policy = load_policy(args.model, config.get("algorithm", "PPO"), env)
    probe = EnergyProbe(env.model)
    constants = probe.constants()
    sample_hz = 1.0 / env.dt
    paper = AGRAWAL[args.paper_condition][args.paper_surface]
    label = f"{args.paper_condition}, {args.paper_surface}"
    end_rho = SIDE_LYING_THRESHOLD if args.window == "side" else ROLL_THRESHOLD

    # Massless bodies (the 'mimo_location' wrapper) carry no energy and need no segment.
    uncovered = ({name for name, mass in zip(constants["names"], constants["masses"]) if mass > 0}
                 - {b for bodies in segment_bodies("left").values() for b in bodies})
    print(f"model               : {args.model}")
    print(f"posture             : {start}")
    print(f"embodiment          : act {config.get('physio_age', 9)} / body "
          f"{config.get('morph_age', 9)} months, {constants['mass']:.3f} kg, "
          f"L = {constants['length']:.3f} m")
    print(f"floor               : {floor_label(config)}")
    print(f"window              : rho <= {args.onset_rho} to rho >= {end_rho} ({args.window})")
    print(f"filter              : {f'{args.lowpass_hz:g} Hz zero-phase' if args.lowpass_hz else 'none'}"
          f" at {sample_hz:g} Hz")
    print(f"reference           : Agrawal 2017, {label}; {AGRAWAL_SUBJECTS['mass_kg']} kg, "
          f"L = {AGRAWAL_SUBJECTS['height_m']} m")
    print(f"episodes            : {args.episodes} (episode_steps {episode_steps})")
    if start != "supine":
        print("NOTE: Agrawal measured supine to side lying only; a prone roll has no reference.")
    if uncovered:
        print(f"NOTE: bodies in no segment, counted nowhere: {sorted(uncovered)}")
    print()

    rows, skipped, ke_error = [], 0, 0.0
    for episode_index in range(args.episodes):
        if not probe.matches(env.model):
            probe = EnergyProbe(env.model)
        data = collect_episode(env, policy, probe, args.seed + episode_index, episode_steps, start)
        ke_error = max(ke_error, float(data["ke_error"].max()))
        window = rolling_window(data["rho"], args.onset_rho, end_rho)
        if window is None:
            skipped += 1
            continue
        row = analyse_episode(data, window, constants, sample_hz, args.lowpass_hz)
        row.update(seed=args.seed + episode_index, window=list(window))
        rows.append(row)
    env.close()

    print(f"rolled into the window: {len(rows)}/{args.episodes} episodes "
          f"({skipped} never reached rho >= {end_rho} and are excluded)")
    print(f"Ek decomposition check: max relative error {ke_error:.1e} against "
          f"1/2 qvel'(M - armature) qvel\n")
    if not rows:
        raise SystemExit("No episode rolled; nothing to report.")
    if ke_error > 1e-6:
        print("WARNING: the body energies do not add up to the joint-space kinetic energy.\n")

    summary = summarise(rows)
    print(f"direction           : {summary['left']} left, {summary['right']} right\n")
    print_report(summary, constants, paper, label)

    if args.json:
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        payload = {
            "model": args.model, "starting_position": start,
            "morph_age": config.get("morph_age", 9), "physio_age": config.get("physio_age", 9),
            "floor": floor_label(config), "window": args.window, "onset_rho": args.onset_rho,
            "lowpass_hz": args.lowpass_hz, "sample_hz": sample_hz, "episodes": args.episodes,
            "analysed": len(rows), "skipped": skipped, "ke_error": ke_error,
            "mass_kg": constants["mass"], "length_m": constants["length"], "g": constants["g"],
            "paper": {"condition": args.paper_condition, "surface": args.paper_surface,
                      "subjects": AGRAWAL_SUBJECTS, **paper},
            "summary": plain(summary), "rows": plain(rows),
        }
        directory = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(directory, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")
    if args.plot:
        plot(summary, constants, paper, label, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()

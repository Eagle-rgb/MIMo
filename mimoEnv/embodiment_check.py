""" Acceptance checks that the environment builds the embodiment it claims to.

Run this after touching the age, the amputation or the curriculum path::

    for t in $(python mimoEnv/embodiment_check.py --list); do
        MUJOCO_GL=osmesa python mimoEnv/embodiment_check.py $t
    done

08.09.2026 The environment used to select one of 96 pre-generated scene files -- 16 age pairs and
5 amputations of each. It now grows :data:`~mimoEnv.envs.roll_over.BASE_SCENE` in an editable
``MjSpec`` (:mod:`mimoGrowth.spec`), which is what makes fractional ages and a 30-stage
curriculum possible and what stops 18 cluster machines racing on one temporary filename.

``mimoGrowth/spec_check.py`` proves the growth itself reproduces those files. This file proves
the *environment* does: that the model reaching MuJoCo after ``__init__`` and after every
``set_embodiment`` is the one the corresponding stored scene compiles to. The two are not the
same claim -- the environment also patches the compiled model (the floor, the ghost limb) and
re-runs ``initialize()``, and it is the environment that stored policies were trained against.

**Memory: one env is ~3.6 GB and this holds a reference model beside it.** Run the sections one
per process, as above.

Two fields need a statement rather than a zero tolerance, and both were found by writing the
zero-tolerance version first:

* ``actuator_user`` holds FMAX, which the pre-generated ``act_<n>_mo.xml`` files write with
  ``%.6g``. The reference itself carries six digits, so this is compared relatively against
  :data:`FMAX_TOLERANCE`. It shows up only when the physiological age differs from the base
  scene's, which is exactly when FMAX is rescaled.
* the ghost limb is a deliberate runtime patch that zeroes the gear of the limb's actuators, so
  a ghost model is *supposed* to differ from the intact scene. Checked against what the patch
  claims to do rather than against the file.
"""
import argparse
import gc
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

import gymnasium as gym
import mujoco

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY

PRONE = os.path.join(SCENE_DIRECTORY, "roll_over", "prone")

FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_type", "body_pos", "body_mass",
          "body_inertia", "body_ipos", "jnt_pos", "jnt_range", "actuator_gear",
          "dof_damping", "jnt_stiffness")

HAND_TYPED_SITES = {"KOBAYASHI_Torso", "KOBAYASHI_RWrist", "KOBAYASHI_LWrist",
                    "KOBAYASHI_RAnkle", "KOBAYASHI_LAnkle"}
""" Marker sites the stored scenes carry as hand-typed numbers and the schema now derives.

Sub-millimetre, analysis-only. See ``mimoGrowth/spec_check.py:test_kobayashi_sites``, which
measures the gap; here they are simply held out so it cannot mask a real drift.
"""

FMAX_TOLERANCE = 1e-5
""" Relative, and it is the stored files' own ``%.6g`` precision rather than slack. """

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_env(**kwargs):
    params = dict(starting_position='supine', touch_params=None, isr=False,
                  age_physio=9, age_morph=9, achieved_goal_in_observation=True)
    params.update(kwargs)
    return gym.make('MIMoRollOver-v0', **params).unwrapped


def compare_to_scene(model, scene, label):
    """ Assert a live env model equals what a stored scene compiles to. """
    reference = mujoco.MjModel.from_xml_path(os.path.join(PRONE, scene))

    worst, where = 0.0, "-"
    for field in FIELDS:
        a, b = getattr(reference, field), getattr(model, field)
        if a.shape != b.shape:
            worst, where = float("inf"), f"{field} shape {a.shape} vs {b.shape}"
            break
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        if difference > worst:
            worst, where = difference, field

    keep = [i for i in range(reference.nsite)
            if reference.site(i).name not in HAND_TYPED_SITES]
    sites = float(np.max(np.abs(reference.site_pos[keep] - model.site_pos[keep])))

    driven = np.abs(reference.actuator_user) > 0
    fmax = float(np.max(np.abs(reference.actuator_user[driven] - model.actuator_user[driven])
                        / np.abs(reference.actuator_user[driven]))) if driven.any() else 0.0

    sizes = (reference.nq, reference.nu, reference.nsensor)
    live = (model.nq, model.nu, model.nsensor)

    check(f"{label:36s}", worst == 0.0 and sites == 0.0 and sizes == live
          and fmax < FMAX_TOLERANCE,
          f"fields {worst:.3e} ({where}), sites {sites:.3e}, FMAX {fmax:.2e} rel, "
          f"nq/nu/nsensor {live}" + ("" if sizes == live else f" != {sizes}"))
    del reference
    gc.collect()


def test_construction_matches_stored_scenes():
    """ The env built at an age pair is the scene that age pair used to have on disk. """
    print("\ntest_construction_matches_stored_scenes")
    for morph, physio in ((9, 9), (1, 1), (1, 9)):
        env = make_env(age_morph=morph, age_physio=physio)
        compare_to_scene(env.model, f"scene_act_{physio}_body_{morph}.xml",
                         f"constructed ({morph}, {physio})")
        env.close()
        del env
        gc.collect()


def test_set_embodiment_matches_stored_scenes():
    """ Every hot-swap lands on the same model the stored scene compiles to.

    The off-diagonal pairs are the ones worth having: they are the experiment, and a
    single-age implementation would pass the diagonal and fail here.
    """
    print("\ntest_set_embodiment_matches_stored_scenes")
    env = make_env()
    for morph, physio in ((1, 1), (9, 1), (1, 9), (3, 6), (6, 3), (9, 9)):
        env.set_embodiment(morph, physio)
        compare_to_scene(env.model, f"scene_act_{physio}_body_{morph}.xml",
                         f"set_embodiment({morph}, {physio})")
    env.close()


def test_fractional_ages():
    """ The point of the change: ages the pre-generated files could not express. """
    print("\ntest_fractional_ages")
    env = make_env()
    masses, gears = [], []
    for morph, physio in ((2.25, 7.75), (4.5, 4.5), (7.0, 2.0)):
        env.set_embodiment(morph, physio)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        masses.append(float(env.model.body_mass.sum()))
        gears.append(float(env.model.actuator_gear[:, 0].mean()))
        check(f"({morph}, {physio}) compiles, resets and steps", True,
              f"mass {masses[-1]:.4f} kg, mean gear {gears[-1]:.4f}")

    check("mass follows the morphological age alone",
          masses == sorted(masses), " < ".join(f"{m:.3f}" for m in masses))
    check("gear follows the physiological age alone",
          gears == sorted(gears, reverse=True), " > ".join(f"{g:.3f}" for g in gears))

    for bad in (-0.5, 24.5):
        try:
            env.set_embodiment(bad, 9)
            check(f"age {bad} raises", False, "it did not")
        except ValueError:
            check(f"age {bad} raises", True)
    env.close()


def test_missing_limb_cut():
    """ A cut limb is gone before compilation, and stays gone across a swap.

    The amputated reference scenes are gitignored (a783ce6), so on a fresh clone they do not
    exist. The structural half of this section runs regardless; only the comparisons against
    those files are skipped, with the command that would produce them.
    """
    print("\ntest_missing_limb_cut")
    references = {(9, 9): "scene_act_9_body_9_left_arm.xml",
                  (1, 3): "scene_act_3_body_1_left_arm.xml"}
    absent = [name for name in references.values()
              if not os.path.exists(os.path.join(PRONE, name))]
    if absent:
        print(f"  [SKIP] {len(absent)} amputated reference scene(s) absent (gitignored). "
              "Generate with\n"
              "         python mimoEnv/assets/roll_over/generate_amputated_scenes.py")

    env = make_env(missing_limb='left_arm', missing_limb_mode='cut')
    if not absent:
        compare_to_scene(env.model, references[(9, 9)], "cut left_arm at (9, 9)")
    check("action space shrank", env.action_space.shape == (38,), str(env.action_space.shape))

    env.set_embodiment(1, 3)
    if not absent:
        compare_to_scene(env.model, references[(1, 3)], "cut left_arm, swapped to (1, 3)")
    check("the swap kept the limb missing", env.action_space.shape == (38,),
          "an embodiment swap must not silently restore an amputated limb")

    env.reset(seed=0)
    env.step(env.action_space.sample())
    check("cut env steps after the swap", True)
    env.close()


def test_missing_limb_ghost():
    """ Ghost keeps the intact body and the intact spaces, and only patches what it says. """
    print("\ntest_missing_limb_ghost")
    env = make_env(missing_limb='left_arm', missing_limb_mode='ghost')
    intact = mujoco.MjModel.from_xml_path(os.path.join(PRONE, "scene_act_9_body_9.xml"))

    live = (env.model.nq, env.model.nu, env.model.nsensor)
    check("spaces are unchanged -- this is what makes transfer possible",
          live == (intact.nq, intact.nu, intact.nsensor), str(live))
    check("geometry is the intact body",
          np.array_equal(env.model.geom_size, intact.geom_size)
          and np.array_equal(env.model.geom_pos, intact.geom_pos))

    ghost = np.asarray(env._ghost_actuators)
    others = np.setdiff1d(np.arange(env.model.nu), ghost)
    check("gear is zeroed on the ghost actuators and untouched elsewhere",
          np.all(env.model.actuator_gear[ghost, 0] == 0.0)
          and np.array_equal(env.model.actuator_gear[others], intact.actuator_gear[others]),
          f"{len(ghost)} zeroed, {len(others)} untouched")
    check("the ghost limb is massless",
          np.all(env.model.body_mass[env._ghost_bodies] < 1e-5),
          f"{len(env._ghost_bodies)} bodies")
    env.close()


def test_age_curriculum_ladder():
    """ The default ladder is unchanged, and a fine one divides the budget rather than growing it.

    No environment is built here -- it is pure schedule arithmetic -- so this section is cheap.
    """
    print("\ntest_age_curriculum_ladder")
    from mimoEnv.envs.morphological_curriculum import (AGES, age_ladder,
                                                       make_curriculum_callback)

    check("the default ladder is the historical one", age_ladder() == list(AGES), str(AGES))

    fine = age_ladder(30)
    check("30 stages span the same interval",
          len(fine) == 30 and fine[0] == float(AGES[0]) and fine[-1] == float(AGES[-1]),
          f"step {fine[1] - fine[0]:.4f} months")
    check("the ladder is strictly increasing", all(a < b for a, b in zip(fine, fine[1:])))

    try:
        age_ladder(1)
        check("a one-stage ladder raises", False, "it did not")
    except ValueError:
        check("a one-stage ladder raises", True)

    common = dict(mgc_stochastic_interval=20_000)
    default = make_curriculum_callback(argparse.Namespace(mgc='growth', mgc_stages=None, **common))
    check("the default curriculum is exactly the old behaviour",
          default.ages == list(AGES) and default.phase_steps == 250_000,
          f"{default.ages}, {default.phase_steps:,} steps per stage")

    staged = make_curriculum_callback(argparse.Namespace(mgc='growth', mgc_stages=30, **common))
    check("30 stages divide the same budget rather than multiplying it",
          len(staged.ages) == 30 and staged.phase_steps == 1_000_000 // 30,
          f"{staged.phase_steps:,} steps per stage")

    inverse = make_curriculum_callback(argparse.Namespace(mgc='inverse', mgc_stages=30, **common))
    check("the inverse curriculum walks the same ladder backwards",
          inverse._get_age_for_step(0) == staged._get_age_for_step(10 ** 9),
          f"{inverse._get_age_for_step(0):g} vs {staged._get_age_for_step(10 ** 9):g}")


def test_writes_no_files():
    """ The cluster regression, at the level the cluster actually runs.

    Building an env and swapping its embodiment must leave the asset tree untouched, because on
    the cluster it is one shared home and may be read-only. This is the failure the whole change
    exists to remove: the old growth wrote a fixed '<scene>_temp.xml' next to the original.
    """
    print("\ntest_writes_no_files")
    watched = os.path.join(SCENE_DIRECTORY)

    def fingerprint():
        entries = {}
        for root, _, files in os.walk(watched):
            for name in files:
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                entries[path] = (stat.st_mtime_ns, stat.st_size)
        return entries

    before = fingerprint()
    env = make_env(age_morph=4.5, age_physio=2.5, missing_limb='left_side',
                   missing_limb_mode='cut')
    env.set_embodiment(7.25, 3.75)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    del env
    gc.collect()
    after = fingerprint()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    check("no file created", not added, ", ".join(os.path.basename(p) for p in added[:5]))
    check("no file deleted", not removed, ", ".join(os.path.basename(p) for p in removed[:5]))
    check("no file modified", not changed, ", ".join(os.path.basename(p) for p in changed[:5]))


SECTIONS = [
    'test_construction_matches_stored_scenes', 'test_set_embodiment_matches_stored_scenes',
    'test_fractional_ages', 'test_missing_limb_cut', 'test_missing_limb_ghost',
    'test_age_curriculum_ladder', 'test_writes_no_files',
]


if __name__ == '__main__':
    import sys

    argv = sys.argv[1:]
    if argv == ['--list']:
        print("\n".join(SECTIONS))
        raise SystemExit(0)

    names = argv or SECTIONS
    unknown = [name for name in names if name not in SECTIONS]
    if unknown:
        raise SystemExit(f"Unknown section(s) {unknown}. Use --list to see the names.")

    print(f"Embodiment acceptance checks for MIMoRollOver-v0 -- {', '.join(names)}")
    for name in names:
        globals()[name]()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        raise SystemExit(1)
    print(f"All checks passed ({len(names)} section(s)).")

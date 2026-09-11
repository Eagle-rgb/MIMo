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

TEXTURE_BUDGET = 10_000
""" Bytes of ``tex_data`` a stripped model may still hold.

The strip replaces each texture with a 1x1 one rather than deleting it, so the ``<material>``
elements that name them still resolve. Thirteen textures come to 219 bytes against 1.02 GB
unstripped, so this bound is three orders of magnitude clear of both.
"""

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


def snapshot(model):
    """ Copy the compared arrays out of a model so the model itself can be freed.

    Sections that hold two envs at once would otherwise peak at ~7.4 GB.

    Arguments:
        model (mujoco.MjModel): The model to read.

    Returns:
        dict[str, numpy.ndarray]: One copied array per entry of :data:`FIELDS`.
    """
    return {field: np.array(getattr(model, field), copy=True) for field in FIELDS}


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


def test_age_curriculum_variance():
    """ '--mgc_interval_cv' / '--mgc_jump_cv' vary the ladder without breaking what it promises.

    At both CVs zero the schedule must be the periodic one bit for bit, whatever the budget. With
    either on it must still start at step 0 and the first age, end at the last age, stay strictly
    monotone, come back identical from its seed, and keep the two knobs independent of each
    other. And 'mgc.yml' must hold exactly what the callback did. No environment is built.
    """
    print("\ntest_age_curriculum_variance")
    import os
    import tempfile

    import yaml

    from mimoEnv.envs.morphological_curriculum import (AGES, RECORD_FILE, age_ladder,
                                                       growth_schedule, make_curriculum_callback)

    exact = True
    for stages in (None, 5, 20, 30):
        for total in (300_000, 1_000_000, 2_000_000):
            n = len(age_ladder(stages))
            phase = max(1, total // n)
            starts, ages = growth_schedule(stages, total)
            exact &= starts == [i * phase for i in range(n)] and ages == age_ladder(stages)
    check("both CVs at 0 reproduce the periodic ladder exactly", exact, "4 ladders x 3 budgets")

    starts, ages = growth_schedule(20, 1_000_000, interval_cv=0.55, jump_cv=0.32, seed=3)
    check("a varied ladder starts at step 0 and the first age and ends at the last age",
          starts[0] == 0 and ages[0] == float(AGES[0]) and ages[-1] == float(AGES[-1]),
          f"{ages[0]:g} -> {ages[-1]:g}")
    check("... stays strictly monotone in steps and in age",
          all(a < b for a, b in zip(starts, starts[1:])) and all(a < b for a, b in zip(ages, ages[1:])))
    durations = np.diff(starts + [1_000_000])
    check("... and is actually varied", np.ptp(durations) > 0 and np.ptp(np.diff(ages)) > 1e-6,
          f"rungs {durations.min():,}..{durations.max():,} steps, "
          f"jumps {np.diff(ages).min():.3f}..{np.diff(ages).max():.3f} months")
    check("the same seed draws the same ladder",
          growth_schedule(20, 1_000_000, 0.55, 0.32, 3) == (starts, ages))
    check("another seed draws another ladder",
          growth_schedule(20, 1_000_000, 0.55, 0.32, 4) != (starts, ages))
    check("the jumps do not depend on the interval CV",
          growth_schedule(20, 1_000_000, 0.0, 0.32, 3)[1] == ages)
    check("the intervals do not depend on the jump CV",
          growth_schedule(20, 1_000_000, 0.55, 0.0, 3)[0] == starts)

    many_starts, many_ages = growth_schedule(2000, 2_000_000, interval_cv=0.55, jump_cv=0.32, seed=0)
    d = np.diff(many_starts + [2_000_000])
    j = np.diff(many_ages)
    cv_d, cv_j = d.std() / d.mean(), j.std() / j.mean()
    check("the realised CVs are the requested ones", abs(cv_d - 0.55) < 0.03 and abs(cv_j - 0.32) < 0.03,
          f"interval {cv_d:.3f} (0.55), jump {cv_j:.3f} (0.32), 2000 rungs")

    try:
        growth_schedule(20, 1_000_000, interval_cv=-0.1)
        check("a negative CV raises", False, "it did not")
    except ValueError:
        check("a negative CV raises", True)
    for mode in ("stochastic", "none"):
        try:
            make_curriculum_callback(argparse.Namespace(mgc=mode, mgc_stages=None,
                                                        mgc_stochastic_interval=20_000,
                                                        mgc_jump_cv=0.3))
            check(f"a CV with --mgc={mode} raises instead of doing nothing", False, "it did not")
        except ValueError:
            check(f"a CV with --mgc={mode} raises instead of doing nothing", True)

    flags = dict(mgc_stages=20, train_for=1_000_000, mgc_stochastic_interval=20_000,
                 mgc_interval_cv=0.55, mgc_jump_cv=0.32, mgc_seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        cb = make_curriculum_callback(argparse.Namespace(mgc="growth", **flags), save_dir=tmp)
        check("the callback walks the ladder growth_schedule draws",
              cb.rung_starts == starts and cb.ages == ages)
        inverse = make_curriculum_callback(argparse.Namespace(mgc="inverse", **flags))
        check("the inverse curriculum walks the same varied ladder backwards",
              [inverse._get_age_for_step(s) for s in starts] == ages[::-1])

        # What '_on_step' does, at an episode boundary every 500 steps.
        for boundary in range(0, 1_000_000, 500):
            cb.num_timesteps = boundary
            age = cb._get_age_for_step(boundary)
            if age != cb.current_age:
                cb.current_age = age
                cb._note_swap(age)
        with open(os.path.join(tmp, RECORD_FILE)) as infile:
            record = yaml.safe_load(infile)
        check("mgc.yml holds the planned ladder",
              [r["step"] for r in record["planned"]] == starts
              and [r["age"] for r in record["planned"]] == ages)
        lags = [r["lag"] for r in record["realised"]]
        check("mgc.yml holds every realised swap, each at most one episode late",
              len(record["realised"]) == 20 and all(0 <= lag < 500 for lag in lags),
              f"{len(record['realised'])} swaps, max lag {max(lags)} steps")
        check("mgc.yml carries the flags that drew it, and no temp file is left behind",
              (record["mgc_interval_cv"], record["mgc_jump_cv"], record["mgc_seed"]) == (0.55, 0.32, 3)
              and record["skipped_rungs"] == [] and os.listdir(tmp) == [RECORD_FILE])

    with tempfile.TemporaryDirectory() as tmp:
        cb = make_curriculum_callback(argparse.Namespace(mgc="growth", **flags), save_dir=tmp)
        for s in (starts[0], starts[2]):
            cb.num_timesteps = s
            cb._note_swap(cb._get_age_for_step(s))
        with open(os.path.join(tmp, RECORD_FILE)) as infile:
            record = yaml.safe_load(infile)
        check("a rung passed without a swap is listed as skipped", record["skipped_rungs"] == [1],
              str(record["skipped_rungs"]))


def test_strip_textures():
    """ Dropping the textures must change the memory and nothing else.

    This is the whole claim behind ``--keep_textures`` being off by default: 1.02 GB of the
    1.024 GB model is ``tex_data`` -- seven emotion faces at 2500x15000 the roll-over experiment
    never displays -- and removing it leaves the compiled physics bit-identical. If that ever
    stops being true, every training run since 08.09.2026 is on a different body than it claims.

    Compared on an *amputated, off-diagonal, fractional* embodiment rather than the default, so
    the strip is exercised together with everything else that edits the spec.
    """
    print("\ntest_strip_textures")
    kwargs = dict(age_morph=4.5, age_physio=2.5, missing_limb='left_arm',
                  missing_limb_mode='cut')

    full = make_env(strip_textures=False, **kwargs)
    full_arrays = snapshot(full.model)
    full_arrays["_user"] = np.array(full.model.actuator_user, copy=True)
    full_sizes = (full.model.nq, full.model.nu, full.model.ngeom, full.model.nsensor,
                  full.model.nbody, full.model.njnt, full.model.nsite)
    full_tex = full.model.ntexdata
    full.close()
    del full
    gc.collect()

    lean = make_env(strip_textures=True, **kwargs)
    lean_arrays = snapshot(lean.model)
    lean_arrays["_user"] = np.array(lean.model.actuator_user, copy=True)
    lean_sizes = (lean.model.nq, lean.model.nu, lean.model.ngeom, lean.model.nsensor,
                  lean.model.nbody, lean.model.njnt, lean.model.nsite)

    worst, where = 0.0, "-"
    for field in FIELDS + ("_user",):
        a, b = full_arrays[field], lean_arrays[field]
        if a.shape != b.shape:
            worst, where = float("inf"), f"{field} shape"
            break
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        if difference > worst:
            worst, where = difference, field

    check("physics is bit-identical with and without textures", worst == 0.0,
          f"worst {worst:.3e} in {where} over {len(FIELDS) + 1} fields")
    check("the model sizes are unchanged", full_sizes == lean_sizes,
          f"nq/nu/ngeom/nsensor/nbody/njnt/nsite {lean_sizes}")
    # Not zero: the textures still exist, they are 1x1. Thirteen of them -- eleven cube maps at
    # 6*1*1*3 bytes, one 2D at 3, one more cube -- come to exactly 219 bytes, against 1.02 GB.
    # Asserted as a bound rather than an equality so that adding a texture to the scene does not
    # fail this, while anything full-resolution surviving the strip does.
    check("the texture data is actually gone", lean.model.ntexdata < TEXTURE_BUDGET,
          f"{full_tex / 1e6:.1f} MB -> {lean.model.ntexdata} bytes")

    # The strip has to survive an embodiment swap, or a curriculum would quietly pay the 937 ms
    # again from the second stage onwards.
    lean.set_embodiment(7.0, 9.0)
    check("it survives an embodiment swap", lean.model.ntexdata < TEXTURE_BUDGET,
          f"{lean.model.ntexdata} bytes after set_embodiment")

    lean.reset(seed=0)
    lean.step(lean.action_space.sample())
    check("a textureless env resets and steps", True)
    lean.close()


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
    'test_age_curriculum_ladder', 'test_age_curriculum_variance', 'test_strip_textures',
    'test_writes_no_files',
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

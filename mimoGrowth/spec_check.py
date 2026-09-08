""" Regression gate for :mod:`mimoGrowth.spec` -- the file-less growth path.

Run this after touching ``spec.py``, the growth schema, or anything that regenerates the scenes
under ``mimoEnv/assets/roll_over/prone/``::

    MUJOCO_GL=osmesa python mimoGrowth/spec_check.py

The claim under test is that growing **one** base scene in memory reproduces the 96 scene files
that are currently pre-generated on disk -- 16 age pairs, plus 5 amputations of each -- so that
switching the environment over to ``spec.py`` re-baselines no stored run. Everything here
compares against those files rather than against a golden value written down by hand, which is
what makes it a regression gate and not a restatement of the implementation.

Two places do **not** come out bit-identical, and both are measured rather than tolerated
silently: four ``KOBAYASHI_*`` marker sites, which the stored files carry as hand-typed numbers
and the schema now derives (0.43 mm, ``test_kobayashi_sites``), and FMAX on a *reused* spec,
whose scaling rule is multiplicative and so is not bit-exact (3.9e-16 relative, one double
epsilon, ``test_spec_reuse``).

**Memory and time: this holds two compiled models at once (~3.6 GB each) and takes ~20 minutes**
-- ``test_amputated_scenes`` alone compiles 164 models. It frees them per comparison and keeps
only the arrays, but do not run it beside a training job on a 16 GB machine. Sections can be run
one per process; ``--list`` prints the names::

    for t in $(python mimoGrowth/spec_check.py --list); do python mimoGrowth/spec_check.py $t; done

Not covered, deliberately, because the environment is not wired up yet: that ``MIMoEnv`` loads
the compiled model, that ``data.yml`` round-trips a fractional age, and that the morphological
curriculum can swap to one. Those belong with that change.
"""
import gc
import itertools
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

import mujoco

from mimoGrowth.spec import apply_growth, grow_spec, growth_params

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRONE = os.path.join(REPO, "mimoEnv", "assets", "roll_over", "prone")
BASE = os.path.join(PRONE, "scene_act_9_body_9.xml")
""" The base scene every comparison grows from.

Any of the sixteen works -- the growth parameters are absolute, not relative to the current
size -- and picking the one that is also the reference for ``(9, 9)`` makes that cell a genuine
identity check rather than a round trip through two different files.
"""

# The scene the *file* route can still handle, for the equivalence check in section one. The
# roll-over scenes cannot be grown by 'adjust_mimo_to_age' at all (KeyError: 'model'), which is
# itself asserted below.
STOCK_SCENE = os.path.join(REPO, "mimoEnv", "assets", "roll_over_prone_scene.xml")

AGES = [1, 3, 6, 9]

# Compared field by field rather than through a single model hash: when something does drift,
# the field name says immediately whether it is the schema, the mass model or the actuators.
FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_type",
          "body_pos", "body_mass", "body_inertia", "body_ipos",
          "jnt_pos", "jnt_range", "site_pos",
          "actuator_gear", "dof_damping", "jnt_stiffness")

# The only elements where the spec route does not reproduce the stored scenes bit for bit. These
# marker sites were *hand-typed* into 'body_<n>_mo.xml' and only later derived from the schema
# (e81cd38, "Added KOBAYASHI sites to the generation of the age scene file, so that we do not have
# to manually type them in"), so the stored files still carry the typed numbers and the schema now
# computes slightly different ones. The gap is sub-millimetre, non-systematic, and the schema
# value is the more correct of the two -- see 'test_kobayashi_sites', which measures it rather
# than waving it through. They are excluded from the age and amputation comparisons so that this
# known 0.4 mm does not mask a real drift somewhere else.
HAND_TYPED_SITES = ("KOBAYASHI_Torso", "KOBAYASHI_RWrist", "KOBAYASHI_LWrist",
                    "KOBAYASHI_RAnkle", "KOBAYASHI_LAnkle")

# How far the stored files may sit from the schema, in metres. Measured worst case is 0.43 mm on
# KOBAYASHI_RWrist at age 1; KOBAYASHI_Torso agrees exactly at every age.
HAND_TYPED_TOLERANCE = 5e-4

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def snapshot(model, fields=FIELDS):
    """ Copy the compared arrays out of a model so the model itself can be freed. """
    return {field: np.array(getattr(model, field), copy=True) for field in fields}


def compile_file(path):
    """ Compile a scene from disk and return only its arrays. """
    model = mujoco.MjModel.from_xml_path(path)
    arrays = snapshot(model)
    arrays["_sizes"] = (model.nbody, model.nq, model.nu, model.ngeom,
                        model.njnt, model.nsite, model.nsensor)
    arrays["_user"] = np.array(model.actuator_user, copy=True)
    arrays["_names"] = [model.site(i).name for i in range(model.nsite)]
    del model
    gc.collect()
    return arrays


def compile_spec(spec):
    """ Compile a spec and return only its arrays, in the same shape as :func:`compile_file`. """
    model = spec.compile()
    arrays = snapshot(model)
    arrays["_sizes"] = (model.nbody, model.nq, model.nu, model.ngeom,
                        model.njnt, model.nsite, model.nsensor)
    arrays["_user"] = np.array(model.actuator_user, copy=True)
    arrays["_names"] = [model.site(i).name for i in range(model.nsite)]
    del model
    gc.collect()
    return arrays


def worst_difference(reference, candidate, skip_sites=()):
    """ The largest absolute difference over :data:`FIELDS`, and which field carried it.

    Arguments:
        skip_sites (Iterable[str]): Site names excluded from ``site_pos``, used to hold the
            known schema gap out of the age comparisons that are about growth itself.

    Returns:
        tuple[float, str]: The difference and the field name, or ``(inf, field)`` on a shape
        mismatch, which means the two models are not even the same model.
    """
    worst, where = 0.0, ""
    for field in FIELDS:
        a, b = reference[field], candidate[field]
        if a.shape != b.shape:
            return float("inf"), f"{field} shape {a.shape} vs {b.shape}"
        if field == "site_pos" and skip_sites:
            keep = [i for i, name in enumerate(reference["_names"]) if name not in skip_sites]
            a, b = a[keep], b[keep]
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        if difference > worst:
            worst, where = difference, field
    return worst, where


def test_file_route_is_dead():
    """ The pre-condition for all of this: the old path cannot grow a roll-over scene. """
    print("\ntest_file_route_is_dead")
    from mimoGrowth.growth import adjust_mimo_to_age, get_version

    try:
        adjust_mimo_to_age(4.0, BASE, create_log=False)
        check("adjust_mimo_to_age raises on a roll-over scene", False,
              "it succeeded -- the include-name assumption must have been fixed")
    except KeyError as error:
        check("adjust_mimo_to_age raises on a roll-over scene", str(error) == "'model'",
              f"KeyError {error}")

    # The scenes were generated with the v1 schema, and 'get_version' happens to report that
    # correctly only because none of their include paths contains "v2". Asserted rather than
    # assumed: with 'v2' the masses are off by 3.4e-2 kg and every comparison below would fail
    # for a reason that has nothing to do with MjSpec.
    check("get_version reads the roll-over scenes as v1", get_version(BASE) == "v1",
          get_version(BASE))


def test_equivalence_with_file_route():
    """ On a scene both routes can handle, they must produce the same model. """
    print("\ntest_equivalence_with_file_route")
    from mimoGrowth.growth import adjust_mimo_to_age
    from mimoGrowth.scene import delete_growth_scene

    for age in (4.7, 9):
        path = adjust_mimo_to_age(age, STOCK_SCENE, create_log=False)
        reference = compile_file(path)
        delete_growth_scene(path)

        candidate = compile_spec(grow_spec(STOCK_SCENE, age))
        difference, where = worst_difference(reference, candidate)
        check(f"spec growth == file growth at age {age}", difference == 0.0,
              f"worst {difference:.3e}" + (f" in {where}" if where else ""))


def test_age_scenes():
    """ All sixteen pre-generated age pairs, from one base scene. """
    print("\ntest_age_scenes")
    worst_overall, worst_cell = 0.0, ""
    for morph, physio in itertools.product(AGES, AGES):
        reference = compile_file(os.path.join(PRONE, f"scene_act_{physio}_body_{morph}.xml"))
        candidate = compile_spec(grow_spec(BASE, morph, physio))
        difference, where = worst_difference(reference, candidate, skip_sites=HAND_TYPED_SITES)
        if difference > worst_overall:
            worst_overall, worst_cell = difference, f"act_{physio}_body_{morph} ({where})"
        del reference, candidate
        gc.collect()
    check("all 16 age pairs reproduce their pre-generated scene", worst_overall == 0.0,
          f"worst {worst_overall:.3e}" + (f" at {worst_cell}" if worst_cell else ""))


def test_age_pairs_are_independent():
    """ Morphological age must move only geometry, physiological age only gear.

    A single-age implementation passes ``test_age_scenes`` on the diagonal and fails here, which
    is the mistake worth guarding: the whole roll-over experiment is the off-diagonal.
    """
    print("\ntest_age_pairs_are_independent")
    diagonal = compile_spec(grow_spec(BASE, 9, 9))
    off_morph = compile_spec(grow_spec(BASE, 1, 9))
    off_physio = compile_spec(grow_spec(BASE, 9, 1))

    check("morph age alone leaves actuator_gear untouched",
          np.array_equal(diagonal["actuator_gear"], off_morph["actuator_gear"]),
          f"max {np.max(np.abs(diagonal['actuator_gear'] - off_morph['actuator_gear'])):.3e}")
    check("morph age alone moves the geometry",
          not np.array_equal(diagonal["geom_size"], off_morph["geom_size"]),
          f"max {np.max(np.abs(diagonal['geom_size'] - off_morph['geom_size'])):.3e}")
    check("physio age alone leaves the geometry untouched",
          np.array_equal(diagonal["geom_size"], off_physio["geom_size"])
          and np.array_equal(diagonal["body_mass"], off_physio["body_mass"]),
          f"max {np.max(np.abs(diagonal['geom_size'] - off_physio['geom_size'])):.3e}")
    check("physio age alone moves actuator_gear",
          not np.array_equal(diagonal["actuator_gear"], off_physio["actuator_gear"]),
          f"max {np.max(np.abs(diagonal['actuator_gear'] - off_physio['actuator_gear'])):.3e}")


def test_amputated_scenes():
    """ ``spec.delete(body)`` must reproduce ``generate_amputated_scenes.py``.

    That script prunes ``<sensor>``, ``<equality>``, ``<tendon>`` and ``<contact>`` by hand,
    because MuJoCo raises on a dangling reference. The point of this section is that MuJoCo does
    that pruning itself when the subtree is deleted from the spec, so the 80 files and the 338
    lines behind them can go.
    """
    print("\ntest_amputated_scenes")
    from mimoEnv.envs.roll_over import MISSING_LIMBS

    worst_overall, worst_cell, missing = 0.0, "", []
    for limb, bodies in MISSING_LIMBS.items():
        for morph, physio in itertools.product(AGES, AGES):
            path = os.path.join(PRONE, f"scene_act_{physio}_body_{morph}_{limb}.xml")
            if not os.path.exists(path):
                missing.append(os.path.basename(path))
                continue
            reference = compile_file(path)
            candidate = compile_spec(grow_spec(BASE, morph, physio, remove=bodies))
            difference, where = worst_difference(reference, candidate, skip_sites=HAND_TYPED_SITES)
            if difference > worst_overall:
                worst_overall, worst_cell = difference, f"{limb} act_{physio}_body_{morph} ({where})"
            if reference["_sizes"] != candidate["_sizes"]:
                worst_overall, worst_cell = float("inf"), f"{limb}: sizes differ"
            del reference, candidate
            gc.collect()

    check("all pre-generated amputated scenes reproduce", worst_overall == 0.0,
          f"worst {worst_overall:.3e}" + (f" at {worst_cell}" if worst_cell else ""))
    check("no amputated scene was missing", not missing,
          f"{len(missing)} absent, generate with "
          "'python mimoEnv/assets/roll_over/generate_amputated_scenes.py'"
          if missing else "")

    # The spaces really do shrink -- 'cut' is the mode you train in, so this is the property that
    # makes it different from 'ghost'.
    intact = compile_spec(grow_spec(BASE, 9, 9))
    cut = compile_spec(grow_spec(BASE, 9, 9, remove=MISSING_LIMBS["left_arm"]))
    check("amputation shrinks the action space",
          cut["_sizes"][2] < intact["_sizes"][2],
          f"nu {intact['_sizes'][2]} -> {cut['_sizes'][2]}")
    check("amputation shrinks the observation space",
          cut["_sizes"][1] < intact["_sizes"][1] and cut["_sizes"][6] < intact["_sizes"][6],
          f"nq {intact['_sizes'][1]} -> {cut['_sizes'][1]}, "
          f"nsensor {intact['_sizes'][6]} -> {cut['_sizes'][6]}")


def test_amputation_errors():
    """ A misspelled body must raise, not silently produce an intact MIMo. """
    print("\ntest_amputation_errors")
    try:
        grow_spec(BASE, 9, 9, remove=("left_upper_arn",))
        check("unknown body raises", False, "it did not")
    except KeyError as error:
        check("unknown body raises", "left_upper_arn" in str(error), str(error))

    for age in (-1, 25):
        try:
            grow_spec(BASE, age)
            check(f"age {age} raises", False, "it did not")
        except ValueError:
            check(f"age {age} raises", True)


def test_muscle_fmax():
    """ FMAX must follow gear, or ``--physio_age`` is a no-op under ``--use_muscle``.

    The pre-generated ``act_<n>_mo.xml`` files write ``user`` with ``%.6g``, so the reference is
    only good to six significant digits -- this compares relatively, and the tolerance is that
    rounding, not slack in the rule.
    """
    print("\ntest_muscle_fmax")
    worst = 0.0
    for physio in AGES:
        reference = compile_file(os.path.join(PRONE, f"scene_act_{physio}_body_9.xml"))
        candidate = compile_spec(grow_spec(BASE, 9, physio))
        a, b = reference["_user"], candidate["_user"]
        nonzero = np.abs(a) > 0
        worst = max(worst, float(np.max(np.abs(a[nonzero] - b[nonzero]) / np.abs(a[nonzero]))))
        del reference, candidate
        gc.collect()
    check("actuator user (VMAX, FMAX) matches the pre-generated act files", worst < 1e-5,
          f"worst relative {worst:.2e}, against the %.6g rounding in those files")

    base = compile_spec(grow_spec(BASE, 9, 9))
    for physio in (1, 4.5):
        grown = compile_spec(grow_spec(BASE, 9, physio))
        ratio_gear = grown["actuator_gear"][:, 0] / base["actuator_gear"][:, 0]
        driven = base["_user"][:, 1] > 0
        ratio_fmax = grown["_user"][driven, 1] / base["_user"][driven, 1]
        check(f"FMAX tracks gear at physio age {physio}",
              np.allclose(ratio_fmax, ratio_gear[driven], rtol=1e-12),
              f"gear x{ratio_gear[driven][0]:.4f}, FMAX x{ratio_fmax[0]:.4f}")
        check(f"VMAX is age-invariant at physio age {physio}",
              np.array_equal(grown["_user"][:, 0], base["_user"][:, 0]))
        del grown
        gc.collect()

    off = compile_spec(grow_spec(BASE, 9, 1, scale_muscle_fmax=False))
    check("scale_muscle_fmax=False leaves user alone",
          np.array_equal(off["_user"], base["_user"])
          and not np.array_equal(off["actuator_gear"], base["actuator_gear"]))


def test_continuous_ages():
    """ Fractional ages compile and interpolate -- the reason for doing any of this. """
    print("\ntest_continuous_ages")
    ages = [1, 2.5, 4.0, 5.5, 7.0, 9]
    volumes, gears = [], []
    for age in ages:
        arrays = compile_spec(grow_spec(BASE, age, age))
        volumes.append(float(np.sum(arrays["body_mass"])))
        gears.append(float(np.mean(arrays["actuator_gear"][:, 0])))
        del arrays
        gc.collect()

    check("every fractional age compiles", len(volumes) == len(ages))
    check("total mass increases monotonically with morphological age",
          all(a < b for a, b in zip(volumes, volumes[1:])),
          " -> ".join(f"{v:.3f}" for v in volumes) + " kg")
    check("mean gear increases monotonically with physiological age",
          all(a < b for a, b in zip(gears, gears[1:])),
          " -> ".join(f"{g:.3f}" for g in gears))

    # A 30-step curriculum is the stated use, so check that neighbouring steps really differ.
    fine = [compile_spec(grow_spec(BASE, 1 + i * 8 / 29))["body_mass"].sum() for i in (0, 1, 2)]
    check("neighbouring steps of a 30-stage curriculum are distinct",
          len(set(np.round(fine, 9))) == 3,
          " -> ".join(f"{v:.6f}" for v in fine) + " kg")


def test_kobayashi_sites():
    """ The one place the spec route does not bit-match the stored scenes, and by how much.

    The five ``KOBAYASHI_*`` marker sites feed the ``framelinvel`` sensors that
    ``results/kobayashi/`` and ``eval_emg.py`` read; they are not in the observation and not in
    the physics. They used to be typed into ``body_<n>_mo.xml`` by hand and are now derived from
    the growth schema, so the stored files and the schema disagree slightly. This section pins
    that disagreement to those five sites and to sub-millimetre, so that a real regression cannot
    hide behind it -- and asserts they are still in the schema at all, because losing them again
    is exactly what would break a fine-grained curriculum.
    """
    print("\ntest_kobayashi_sites")
    params = growth_params(BASE, 1)

    missing = [name for name in HAND_TYPED_SITES if name not in params["sites"]]
    check("all KOBAYASHI marker sites are in the growth schema", not missing,
          f"absent: {missing}" if missing else f"{len(HAND_TYPED_SITES)} sites, "
          f"schema names {len(params['sites'])} in total")

    worst, worst_where = 0.0, ""
    for age in AGES:
        reference = compile_file(os.path.join(PRONE, f"scene_act_9_body_{age}.xml"))
        candidate = compile_spec(grow_spec(BASE, age, 9))
        drifted = {name: float(np.max(np.abs(reference["site_pos"][index]
                                             - candidate["site_pos"][index])))
                   for index, name in enumerate(reference["_names"])
                   if np.max(np.abs(reference["site_pos"][index]
                                    - candidate["site_pos"][index])) > 1e-12}
        unexpected = sorted(set(drifted) - set(HAND_TYPED_SITES))
        check(f"age {age}: only the hand-typed sites differ", not unexpected,
              f"also {unexpected}" if unexpected else
              f"{len(drifted)} of {len(HAND_TYPED_SITES)}")
        for name, difference in drifted.items():
            if difference > worst:
                worst, worst_where = difference, f"{name} at age {age}"
        del reference, candidate
        gc.collect()

    check("the disagreement stays sub-millimetre", worst < HAND_TYPED_TOLERANCE,
          f"worst {worst * 1000:.4f} mm on {worst_where}")


def test_schema_coverage():
    """ Nothing else silently falls outside the growth schema.

    A geom, body, joint or site that the schema does not name keeps whatever the base scene had,
    which is invisible until someone plots a marker that never moved. Growing to an age far from
    the base scene's own makes any such element show up as a difference against the stored file
    for that age.
    """
    print("\ntest_schema_coverage")
    params = growth_params(BASE, 1)
    reference = compile_file(os.path.join(PRONE, "scene_act_9_body_1.xml"))
    candidate = compile_spec(grow_spec(BASE, 1, 9))

    difference, where = worst_difference(reference, candidate, skip_sites=HAND_TYPED_SITES)
    check("no field drifts between age 9 and age 1 beyond the hand-typed sites",
          difference == 0.0, f"{where} {difference:.3e}" if where else "")

    for element in ("geoms", "bodies", "joints", "sites", "motors"):
        print(f"         schema names {len(params[element]):3d} {element}")


def test_writes_no_files():
    """ The cluster regression: growing a scene must not touch the asset tree.

    This is the whole bug. The file route wrote three fixed names into the shared asset
    directory, so on 18 machines the first one owned them; here the tree may be read-only.
    """
    print("\ntest_writes_no_files")
    # Imported before the fingerprint is taken: an import can write a __pycache__ entry, and one
    # of those under the asset tree would read as "growth wrote a file".
    from mimoEnv.envs.roll_over import MISSING_LIMBS

    watched = os.path.join(REPO, "mimoEnv", "assets")

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
    for age in (1, 4.5, 9):
        compile_spec(grow_spec(BASE, age, age, remove=MISSING_LIMBS["left_side"]))
        gc.collect()
    after = fingerprint()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    check("no file created", not added, ", ".join(os.path.basename(p) for p in added[:5]))
    check("no file deleted", not removed, ", ".join(os.path.basename(p) for p in removed[:5]))
    check("no file modified", not changed, ", ".join(os.path.basename(p) for p in changed[:5]))
    check("mimoGrowth/log.txt untouched",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt") not in after,
          "the file route appends to it on every call, from every machine")


def test_spec_reuse():
    """ Re-applying growth to one parsed spec must equal a fresh parse.

    This is how a curriculum should use it: parsing costs 0.01 s and compiling 0.86 s, so a
    30-stage curriculum wants to parse once. Everything the growth schema supplies is an
    *absolute* size rather than a scale factor, so those fields come out bit-identical however
    many times the spec has been grown before.

    The exception is FMAX, and it is worth understanding rather than papering over: it is not in
    the schema, it is derived as ``FMAX * gear_new / gear_old`` from whatever the actuator
    currently holds (see :func:`mimoGrowth.spec.scale_fmax_with_gear`). That chain of ratios is
    mathematically but not bit-wise equal to a single ratio from the base scene, so a reused spec
    picks up rounding -- measured 3.9e-16 relative over four applications, i.e. a single double
    epsilon and not a compounding drift, against the 1e-6 that the pre-generated
    ``act_<n>_mo.xml`` files themselves are written to. Carrying hidden per-spec state to buy
    back the last digits of a six-digit number would be the worse trade, so this asserts the
    bound instead of pretending it is zero.
    """
    print("\ntest_spec_reuse")
    spec = mujoco.MjSpec.from_file(BASE)
    worst_fields, where_fields, worst_user = 0.0, "", 0.0
    for morph, physio in ((1, 1), (9, 3), (4.5, 4.5), (1, 1)):
        apply_growth(spec, morph, physio, path_scene=BASE)
        reused = compile_spec(spec)
        fresh = compile_spec(grow_spec(BASE, morph, physio))

        difference, where = worst_difference(reused, fresh)
        if difference > worst_fields:
            worst_fields, where_fields = difference, f"{where} at ({morph}, {physio})"

        nonzero = np.abs(fresh["_user"]) > 0
        worst_user = max(worst_user, float(np.max(
            np.abs(reused["_user"][nonzero] - fresh["_user"][nonzero])
            / np.abs(fresh["_user"][nonzero]))))
        del reused, fresh
        gc.collect()

    check("geometry and gear are bit-identical on a reused spec", worst_fields == 0.0,
          f"worst {worst_fields:.3e}" + (f" in {where_fields}" if where_fields else ""))
    check("FMAX drift on a reused spec stays far below the stored files' own precision",
          worst_user < 1e-9,
          f"worst relative {worst_user:.2e}, against 1e-6 in the act files")
    check("a round trip back to the first age reproduces it", worst_fields == 0.0)


SECTIONS = [
    'test_file_route_is_dead', 'test_equivalence_with_file_route', 'test_age_scenes',
    'test_age_pairs_are_independent', 'test_amputated_scenes', 'test_amputation_errors',
    'test_muscle_fmax', 'test_continuous_ages', 'test_kobayashi_sites',
    'test_schema_coverage', 'test_writes_no_files', 'test_spec_reuse',
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

    print(f"mimoGrowth.spec regression -- {', '.join(names)}")
    for name in names:
        globals()[name]()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        raise SystemExit(1)
    print(f"All checks passed ({len(names)} section(s)).")

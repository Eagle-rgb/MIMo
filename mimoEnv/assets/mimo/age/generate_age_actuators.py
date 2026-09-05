"""Regenerate ``mimoEnv/assets/mimo/age/act/act_<n>_mo.xml`` for every age in :data:`AGES`.

    python mimoEnv/assets/mimo/age/generate_age_actuators.py            # write
    python mimoEnv/assets/mimo/age/generate_age_actuators.py --report   # print only

Run this instead of hand-editing the age actuator files, including when an age is added.

Why this exists
---------------
``gear`` is the *spring-damper* model's strength. The muscle model reads its strength from the
``user`` attribute instead (``user="VMAX FMAX_neg FMAX_pos"``, see
:meth:`mimoActuation.muscle.MuscleModel._collect_muscle_parameters`) and **discards gear**:
``_apply_torque`` overwrites ``model.actuator_gear`` on every physics step with the muscle torque
and drives ``ctrl = 1``.

:func:`mimoGrowth.scene.create_growth_scene` only ever writes ``gear``, so ``user`` was identical
across all ages and ``--physio_age`` was a no-op under ``--use_muscle``. Measured 04.09.2026 with
``verify_gear_muscle.py``: one env step with only ``act:hip_bend`` active, XML gear 4.7928 at
1 month against 8.9256 at 9 months, ``qfrc_actuator`` -1.543637 both times.

The rule
--------
    FMAX(age) = FMAX_base * gear(age) / gear_base

Not a new biomechanical assumption -- it is the calibration ``MIMo_meta.xml`` already encodes,
restated so that it survives growth. A muscle's peak torque about its joint is

    tau_max = moment * FMAX * (fl * fv + fp)          [muscle.py:_update_torque]

with ``moment = (lce_max - lce_min) / (phi_max - phi_min)``. MIMo's joint ranges do not change with
age (verified: ``robot:right_knee`` is ``-145 4`` in all four body files), so ``moment`` is
age-invariant and ``tau_max ~ FMAX``. The spring-damper counterpart is ``gear * |forcerange|``,
documented in :class:`mimoActuation.actuation.SpringDamperModel` as the maximum voluntary isometric
torque along that axis. The ratio between the two,

    r = moment * FMAX / (gear * |forcerange|)

is constant across actuators (median 1.070), i.e. the FMAX values were tuned so the muscle model
reproduces the same MVIC torques with ~7 % headroom for the force-length/force-velocity curves.
Preserving that calibration at every age forces exactly the rule above.

``mimoGrowth.physics.calc_motor_gear`` scales gear with the volume of the associated geom, so this
is equivalently ``FMAX(age) = FMAX_base * vol(age) / vol_base``. That is dimensionally right:
muscle force scales with physiological cross-sectional area (~L^2) and torque with force times
moment arm (~L^3 = volume), while ``moment`` above carries no length scale at all.

VMAX is deliberately left age-invariant. It is a *normalised* fibre velocity (lce units per second)
and lce is normalised by the joint range, which does not change, so there is no length scale to
rescale. Maximal shortening velocity does change with development (fibre-type composition), but
there are no infant data for it, and inventing a scaling here would be undeclared modelling.

What this does not do
---------------------
``gear`` is never touched. It is what every stored spring-damper run was trained against, and the
generated files are byte-identical to the previous ones in their ``gear`` attributes.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

# Ages with a pre-generated scene. Must stay in sync with `mimoEnv.envs.roll_over.AGES`.
AGES = [1, 3, 6, 9]

# Muscle model constants, from `mimoActuation.muscle.MuscleModel.__init__`. Duplicated rather than
# imported because importing the model would need a live env.
LCE_MIN, LCE_MAX = 0.75, 1.05
EPS = 1e-8

# An actuator counts as mis-calibrated when its r sits this many robust standard deviations
# (1.4826 * MAD) from the median. This is a self-check now: the three trunk outliers
# (act:hip_lean, act:chest_twist, act:chest_lean) were repaired directly in MIMo_meta.xml on
# 04.09.2026, so nothing should trip here any more.
OUTLIER_Z = 10.0

DIRNAME = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(DIRNAME, "..", "..", "..", ".."))
PATH_META = os.path.join(REPO, "mimoEnv", "assets", "mimo", "MIMo_meta.xml")
# Joint ranges are age-invariant, so any compiled scene yields the same moments.
PATH_REFERENCE_SCENE = os.path.join(
    REPO, "mimoEnv", "assets", "roll_over", "prone", "scene_act_9_body_9.xml")
PATH_OUT = os.path.join(DIRNAME, "act", "act_{age}_mo.xml")


def collect_moments():
    """ Normalised fibre length per radian of joint travel, per actuator.

    Mirrors `MuscleModel._compute_parametrization`. Only the magnitude is needed here; `moment_2`
    is its negative.

    Returns:
        dict[str, float]: Actuator name -> moment, for every actuator prefixed `act:`.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(PATH_REFERENCE_SCENE)
    moments = {}
    for idx in range(model.nu):
        name = model.actuator(idx).name
        if not name.startswith("act:"):
            continue
        joint_id = model.actuator_trnid[idx, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        low, high = model.jnt_range[joint_id]
        spring = model.qpos_spring[qpos_adr]
        moments[name] = (LCE_MAX - LCE_MIN + EPS) / ((high - spring) - (low - spring) + EPS)
    return moments


def read_base_actuators():
    """ Read gear, forcerange and user from the stock (~18 month) meta file.

    Returns:
        dict[str, dict]: Actuator name -> {'gear', 'forcerange', 'vmax', 'fmax_neg', 'fmax_pos'}.
    """
    root = ET.parse(PATH_META).getroot()
    base = {}
    for motor in root.find("actuator").findall("motor"):
        name = motor.attrib["name"]
        if not name.startswith("act:"):
            continue
        vmax, fmax_neg, fmax_pos = (float(v) for v in motor.attrib["user"].split())
        base[name] = {
            "gear": float(motor.attrib["gear"]),
            "forcerange": [abs(float(v)) for v in motor.attrib["forcerange"].split()],
            "vmax": vmax,
            "fmax_neg": fmax_neg,
            "fmax_pos": fmax_pos,
        }
    return base


def calibration_ratios(base, moments):
    """ r = moment * FMAX / (gear * |forcerange|), for both muscles of every actuator.

    r is what the muscle model's peak torque is worth in units of the spring-damper model's
    maximum voluntary isometric torque. It should be the same constant for every muscle.

    Returns:
        dict[str, tuple[float, float]]: Actuator name -> (r_negative, r_positive).
    """
    ratios = {}
    for name, values in base.items():
        moment, gear = moments[name], values["gear"]
        fr_neg, fr_pos = values["forcerange"]
        ratios[name] = (moment * values["fmax_neg"] / (fr_neg * gear),
                        moment * values["fmax_pos"] / (fr_pos * gear))
    return ratios


def find_outliers(ratios, r_median):
    """ Actuators whose stored FMAX is inconsistent with their own gear (robust z score).

    Returns:
        dict[str, tuple[float, float]]: The offending subset of `ratios`.
    """
    values = np.array([value for pair in ratios.values() for value in pair])
    robust_sd = 1.4826 * np.median(np.abs(values - r_median))
    if robust_sd <= 0:
        return {}
    return {name: r for name, r in ratios.items()
            if max(abs(r[0] - r_median), abs(r[1] - r_median)) / robust_sd > OUTLIER_Z}


def scaled_user_fields(base, gears_by_age):
    """ The `user` attribute for every actuator at every age.

    FMAX scales with gear (equivalently, with geom volume); VMAX is left unchanged.

    Args:
        base (dict): Output of :func:`read_base_actuators`.
        gears_by_age (dict): Age -> {actuator name -> gear}.

    Returns:
        dict[int, dict[str, str]]: Age -> actuator name -> `user` attribute string.
    """
    fields = {}
    for age, gears in gears_by_age.items():
        per_age = {}
        for name, values in base.items():
            scale = gears[name] / values["gear"]
            per_age[name] = (f"{values['vmax']:.6g} "
                             f"{values['fmax_neg'] * scale:.6g} {values['fmax_pos'] * scale:.6g}")
        fields[age] = per_age
    return fields


def write_age_file(age, gears, user_fields):
    """ Write ``act/act_<age>_mo.xml`` from the stock meta with gear and user substituted.

    Args:
        age (int): Age in months.
        gears (dict): Actuator name -> gear at this age.
        user_fields (dict): Actuator name -> `user` attribute string at this age.

    Returns:
        str: The path that was written.
    """
    actuator = ET.parse(PATH_META).getroot().find("actuator")
    for motor in actuator.findall("motor"):
        name = motor.attrib["name"]
        if name not in gears:
            continue
        motor.attrib["gear"] = str(gears[name])
        motor.attrib["user"] = user_fields[name]

    root = ET.Element("mujocoinclude")
    root.append(actuator)
    ET.indent(root, space="  ")
    path = PATH_OUT.format(age=age)
    ET.ElementTree(root).write(path, encoding="unicode")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ages", default=",".join(str(a) for a in AGES),
                        help="Comma separated ages in months. Default: %(default)s")
    parser.add_argument("--report", action="store_true",
                        help="Print the table and write nothing.")
    args = parser.parse_args()
    ages = [int(value) for value in args.ages.split(",")]

    from mimoGrowth.growth import get_growth_params

    base = read_base_actuators()
    moments = collect_moments()
    gears_by_age = {age: {name: get_growth_params(age, "v1")["motors"][name]["gear"]
                          for name in base} for age in ages}
    user_fields = scaled_user_fields(base, gears_by_age)

    # Self-check: does every FMAX in the base meta match its own gear?
    ratios = calibration_ratios(base, moments)
    values = np.array([value for pair in ratios.values() for value in pair])
    r_median = float(np.median(values))
    outliers = find_outliers(ratios, r_median)
    print(f"Calibration r = moment * FMAX / (gear * |forcerange|) in {os.path.basename(PATH_META)}")
    print(f"  median {r_median:.4f}   mean {values.mean():.4f}   "
          f"sd {values.std():.4f}   n = {len(values)} muscles")
    if outliers:
        print(f"  WARNING -- FMAX does not match its own gear (robust z > {OUTLIER_Z}):")
        for name, (r_neg, r_pos) in sorted(outliers.items()):
            print(f"    {name:20} r = ({r_neg:.2f}, {r_pos:.2f})")
        print("  Fix those in MIMo_meta.xml, not here.")
    else:
        print(f"  no outliers (robust z <= {OUTLIER_Z}) -- base meta is consistent")

    print("\nFMAX_neg by age (VMAX unchanged):")
    print("  " + f"{'actuator':24}" + "".join(f"{a:>11}" for a in ages) + f"{'factor':>9}")
    for name in ["act:hip_bend", "act:hip_lean", "act:hip_twist", "act:left_knee",
                 "act:left_hip_flex"]:
        if name not in base:
            continue
        row = [float(user_fields[age][name].split()[1]) for age in ages]
        print(f"  {name:24}" + "".join(f"{v:11.4f}" for v in row)
              + f"{row[-1] / row[0]:8.2f}x")

    if args.report:
        print("\n--report: nothing written.")
        return

    print()
    for age in ages:
        path = write_age_file(age, gears_by_age[age], user_fields[age])
        print(f"wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()

""" Generate the amputated roll-over scenes used by ``--missing_limb_mode=cut``.

A ``cut`` amputation has to happen before the model is compiled: the body subtree is deleted
from the XML, so the joints, actuators and sensors of that limb never exist and the action and
observation spaces shrink accordingly. That is what makes ``cut`` the mode to *train* in --
a policy trained here has no channels for a limb it does not have. (``ghost`` is the other
mode; it patches the compiled model at runtime and keeps the spaces intact, which is what a
policy trained with all limbs needs in order to be evaluated without one. It uses the intact
scenes and never touches this script.)

Scenes are pre-generated rather than written on demand for the same reason the age scenes are:
parallel runs on the cluster raced on creating and deleting temporary scene files.

For every limb in ``MISSING_LIMBS`` this writes, into ``roll_over/prone/``:

* ``body_<morph>_mo_<limb>.xml``            -- the kinematic tree without the subtree
* ``act_<physio>_mo_<limb>.xml``            -- the actuators without that limb's motors
* ``scene_act_<p>_body_<m>_<limb>.xml``     -- the scene, includes rewritten and every
  ``<sensor>``/``<equality>``/``<tendon>``/``<contact>`` entry that references something inside
  the subtree removed. MuJoCo raises at compile time on a dangling reference, so this pruning is
  not cosmetic.

Usage::

    python mimoEnv/assets/roll_over/generate_amputated_scenes.py
    python mimoEnv/assets/roll_over/generate_amputated_scenes.py --limbs left_arm left_leg
    python mimoEnv/assets/roll_over/generate_amputated_scenes.py --check

``--check`` writes nothing: it compiles the existing scenes and prints their model sizes, and
exits non-zero if one is missing. That makes it usable as an existence probe -- which is how
``run_missing_limb.sh`` decides whether it has to generate.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# Python puts this script's own directory on sys.path, not the repository root, so an
# invocation by path ('python mimoEnv/assets/roll_over/generate_amputated_scenes.py') cannot
# see 'mimoEnv' without help.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Imported rather than restated: the limb definitions belong to the environment, and a second
# copy here would rot the first time a limb is added.
from mimoEnv.envs.roll_over import AGES, MISSING_LIMBS, PREGENERATED_LIMBS

SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
PRONE_DIR = os.path.join(SCENE_DIR, "prone")
AGE_DIR = os.path.abspath(os.path.join(SCENE_DIR, "..", "mimo", "age"))
BODY_DIR = os.path.join(AGE_DIR, "body")
ACT_DIR = os.path.join(AGE_DIR, "act")

# Attributes through which one XML element refers to another by name. An element is dropped when
# any of these -- on the element itself or on a descendant -- names something inside the removed
# subtree. The name spaces do not collide in MIMo's XMLs (joints carry the 'robot:' prefix, geoms
# 'geom:', sites are 'torque_*'/'KOBAYASHI_*'), so a single set of removed names is unambiguous.
REFERENCE_ATTRIBUTES = (
    "joint", "joint1", "joint2",
    "body", "body1", "body2",
    "geom", "geom1", "geom2",
    "site", "site1", "site2",
    "objname", "tendon", "tendon1", "tendon2",
)

# Sections of the scene that are pruned. The worldbody is not among them: it holds the floor,
# the lights and the cameras, none of which reference the limb.
PRUNED_SECTIONS = ("sensor", "equality", "tendon", "contact")


def find_body(root, name):
    """ Find the ``<body>`` element with this name, and its parent. """
    for parent in root.iter():
        for child in parent:
            if child.tag == "body" and child.get("name") == name:
                return parent, child
    raise KeyError(f"No body named {name!r} in the kinematic tree.")


def collect_names(element):
    """ Every name defined anywhere inside this element, including the element itself. """
    names = set()
    for node in element.iter():
        name = node.get("name")
        if name is not None:
            names.add(name)
    return names


def references(element, removed):
    """ Whether this element or any descendant refers to a removed name. """
    for node in element.iter():
        for attribute in REFERENCE_ATTRIBUTES:
            if node.get(attribute) in removed:
                return True
    return False


def prune(parent, removed):
    """ Drop every child of ``parent`` that references a removed name. Returns the count. """
    dropped = [child for child in parent if references(child, removed)]
    for child in dropped:
        parent.remove(child)
    return len(dropped)


def write(tree, path):
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    # ElementTree writes no trailing newline, which makes the files awkward to diff.
    with open(path, "a") as handle:
        handle.write("\n")


def generate_body(morph_age, limb, bodies):
    """ Write the kinematic tree without the limb. Returns the names it removed. """
    source = os.path.join(BODY_DIR, f"body_{morph_age}_mo.xml")
    tree = ET.parse(source)
    root = tree.getroot()

    removed = set()
    for body_name in bodies:
        parent, element = find_body(root, body_name)
        removed |= collect_names(element)
        parent.remove(element)

    target = os.path.join(PRONE_DIR, f"body_{morph_age}_mo_{limb}.xml")
    write(tree, target)
    return removed, target


def generate_actuators(physio_age, limb, removed):
    """ Write the actuator block without the motors driving removed joints. """
    source = os.path.join(ACT_DIR, f"act_{physio_age}_mo.xml")
    tree = ET.parse(source)
    root = tree.getroot()

    dropped = 0
    for actuator in root.iter("actuator"):
        dropped += prune(actuator, removed)

    target = os.path.join(PRONE_DIR, f"act_{physio_age}_mo_{limb}.xml")
    write(tree, target)
    return dropped, target


def generate_scene(physio_age, morph_age, limb, removed):
    """ Write the scene: includes rewritten, dangling references pruned. """
    source = os.path.join(PRONE_DIR, f"scene_act_{physio_age}_body_{morph_age}.xml")
    tree = ET.parse(source)
    root = tree.getroot()

    # The amputated body and actuator files sit in 'prone/' beside the scene, while the intact
    # ones live two directories up under 'mimo/age/' -- so the include paths change shape, not
    # just their file name.
    rewritten = 0
    for include in root.iter("include"):
        file = include.get("file", "")
        if file.endswith(f"act_{physio_age}_mo.xml"):
            include.set("file", f"act_{physio_age}_mo_{limb}.xml")
            rewritten += 1
        elif file.endswith(f"body_{morph_age}_mo.xml"):
            include.set("file", f"body_{morph_age}_mo_{limb}.xml")
            rewritten += 1
    if rewritten != 2:
        raise RuntimeError(f"Expected two includes to rewrite in {source!r}, rewrote {rewritten}.")

    dropped = 0
    for section in PRUNED_SECTIONS:
        for element in root.iter(section):
            dropped += prune(element, removed)

    target = os.path.join(PRONE_DIR, f"scene_act_{physio_age}_body_{morph_age}_{limb}.xml")
    write(tree, target)
    return dropped, target


def generate(limbs, verbose=True):
    written = []
    for limb in limbs:
        bodies = MISSING_LIMBS[limb]
        removed_by_morph = {}
        for morph_age in AGES:
            removed, target = generate_body(morph_age, limb, bodies)
            removed_by_morph[morph_age] = removed
            written.append(target)
        # The removed names are the same at every age -- the ages rescale MIMo, they do not
        # rename anything -- but assert it rather than assume it, because a divergence would
        # silently produce scenes that prune different sensors per age.
        reference = removed_by_morph[AGES[0]]
        for morph_age, removed in removed_by_morph.items():
            if removed != reference:
                raise RuntimeError(f"Body {morph_age} months removes different names than "
                                   f"{AGES[0]} months for {limb!r}: "
                                   f"{sorted(removed ^ reference)}")

        for physio_age in AGES:
            dropped, target = generate_actuators(physio_age, limb, reference)
            written.append(target)
        for physio_age in AGES:
            for morph_age in AGES:
                pruned, target = generate_scene(physio_age, morph_age, limb, reference)
                written.append(target)

        if verbose:
            print(f"{limb:<10} bodies {', '.join(bodies)}: removed {len(reference)} names, "
                  f"{dropped} actuators, {pruned} scene references")
    return written


def check(limbs):
    """ Compile every amputated scene and report the resulting model sizes. """
    import mujoco

    for limb in limbs:
        for physio_age in AGES:
            for morph_age in AGES:
                path = os.path.join(PRONE_DIR,
                                    f"scene_act_{physio_age}_body_{morph_age}_{limb}.xml")
                if not os.path.exists(path):
                    raise SystemExit(f"Missing {path!r}. Run this script without '--check'.")
                model = mujoco.MjModel.from_xml_path(path)
                if physio_age == morph_age == AGES[-1]:
                    print(f"{limb:<10} act {physio_age} body {morph_age}: "
                          f"nu {model.nu:3d}  nq {model.nq:3d}  nv {model.nv:3d}  "
                          f"nsensordata {model.nsensordata:4d}  mass {model.body_mass.sum():.4f}")
        print(f"{limb:<10} {len(AGES) ** 2} scenes compile")

    intact = mujoco.MjModel.from_xml_path(
        os.path.join(PRONE_DIR, f"scene_act_{AGES[-1]}_body_{AGES[-1]}.xml"))
    print(f"{'intact':<10} act {AGES[-1]} body {AGES[-1]}: "
          f"nu {intact.nu:3d}  nq {intact.nq:3d}  nv {intact.nv:3d}  "
          f"nsensordata {intact.nsensordata:4d}  mass {intact.body_mass.sum():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # PREGENERATED_LIMBS, not every name in MISSING_LIMBS: this script is the reference the
    # MjSpec route is checked against, and that reference is a fixed historical set. Limb
    # combinations added since are built in memory and deliberately have no file here.
    parser.add_argument("--limbs", nargs="+", default=list(PREGENERATED_LIMBS),
                        choices=list(PREGENERATED_LIMBS),
                        help="Which limbs to generate. Default: the pre-generated reference set.")
    parser.add_argument("--check", action="store_true",
                        help="Compile the generated scenes and print their model sizes.")
    args = parser.parse_args()

    # '--check' verifies and does not write, so callers can use it as an existence probe and
    # generate only on a non-zero exit -- which is what 'run_missing_limb.sh' does.
    if args.check:
        check(args.limbs)
        return

    written = generate(args.limbs)
    print(f"Wrote {len(written)} files to {PRONE_DIR}")


if __name__ == "__main__":
    main()

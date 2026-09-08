""" Growth without temporary files, via MuJoCo's ``MjSpec``.

This is the replacement for :func:`mimoGrowth.growth.adjust_mimo_to_age` +
:func:`mimoGrowth.scene.create_growth_scene`, which write ``<name>_temp.xml`` next to the
original scene and delete it after loading. Two things are wrong with that on the cluster:

* **The name is fixed and the asset directory is shared.** All 18 machines write the same three
  paths into the same NFS home, so the first one to start owns them and the rest die -- either on
  the write, or on a half-written file, or because a neighbour's ``delete_growth_scene`` removed
  the file between compile and read.
* **It does not work on the roll-over scenes at all.** ``create_growth_scene`` locates the model
  and meta file by ``"model" in include.attrib["file"]``; the roll-over scenes include
  ``act_<n>_mo.xml`` / ``body_<n>_mo.xml`` / ``scene_texture_incl.xml``, none of which match, so
  it raises ``KeyError: 'model'``. The pre-generated age scenes exist because of this, not only
  because of the race.

``MjSpec`` removes both problems: :meth:`mujoco.MjSpec.from_file` resolves every ``<include>``
and every relative texture path against the original file, hands back an editable model, and
:meth:`~mujoco.MjSpec.compile` produces the ``MjModel`` directly. Nothing is written, so the
asset tree can be read-only and any number of processes can grow the same scene at once.

What this buys beyond the race:

* **Continuous ages.** A single base scene grows to any float age, so ``AGES = [1, 3, 6, 9]`` is
  no longer a property of which files happen to exist. The four integer ages come out
  bit-identical to the stored scenes -- see ``mimoGrowth/spec_check.py`` -- so nothing on disk is
  re-baselined. The one exception is four ``KOBAYASHI_*`` marker sites, which the stored files
  still carry as hand-typed numbers and the schema now derives; they disagree by at most 0.43 mm
  and feed only the analysis scripts, never the physics or the observation.
* **Amputation in memory.** ``spec.delete(body)`` drops the subtree and MuJoCo prunes the
  dangling sensors, actuators and contact pairs itself, reproducing the pre-generated
  ``*_<limb>.xml`` scenes exactly, all 80 of them.

Example::

    from mimoGrowth.spec import grow_model

    model, data = grow_model(scene, morph_age=4.7, physio_age=9,
                             remove=('left_upper_arm',))

For a curriculum, parse once and re-apply -- parsing costs 0.01 s, compiling 0.86 s, and the
growth parameters are absolute sizes rather than scale factors, so re-applying to the same spec
is not a compounding operation::

    spec = mujoco.MjSpec.from_file(scene)
    for age in ages:
        apply_growth(spec, age, path_scene=scene)
        model = spec.compile()
"""

import numpy as np

import mujoco

from mimoGrowth.growth import get_growth_params, get_version


def growth_params(path_scene, age, custom=None, version=None):
    """ The growth parameters for one age, with the MIMo version resolved from the scene.

    Arguments:
        path_scene (str): Path to the MuJoCo scene. Only read to determine the version.
        age (float): Age of MIMo in months, 0 to 24. May be fractional.
        custom (dict): Custom geom sizes. See :func:`mimoGrowth.growth.adjust_mimo_to_age`.

            Note that :func:`~mimoGrowth.growth.get_growth_params` writes these into the
            module-level ``SCHEMA`` dict, so they persist into later calls that pass no
            ``custom`` at all. Pre-existing behaviour, not introduced here, but it matters now
            that a curriculum may call this many times in one process.

        version (str): ``'v1'`` or ``'v2'``. Read from the scene when omitted.

    Returns:
        dict: The growth parameters.

    Raises:
        ValueError: If the age is outside 0 to 24.
    """
    if not 0 <= age <= 24:
        raise ValueError(f"The age '{age}' is invalid. Must be between 0 and 24.")
    if version is None:
        version = get_version(path_scene)
    return get_growth_params(age, version, custom)


def remove_bodies(spec, names):
    """ Delete body subtrees from the spec, before it is compiled.

    MuJoCo drops everything that referenced the subtree along with it -- the joints, their
    actuators, the sensors, the contact pairs and exclusions -- which is why this reproduces
    ``mimoEnv/assets/roll_over/generate_amputated_scenes.py`` exactly. A dangling reference would
    raise at compile time, so there is nothing silent to get wrong here.

    Arguments:
        spec (mujoco.MjSpec): The spec to modify, in place.
        names (Iterable[str]): Names of the bodies to remove. The subtrees must be disjoint.

    Raises:
        KeyError: If a name is not a body in this spec.
    """
    for name in names:
        body = spec.body(name)
        if body is None:
            raise KeyError(f"No body named {name!r} in {spec.modelname!r}.")
        spec.delete(body)


def apply_growth(spec, morph_age, physio_age=None, path_scene=None,
                 custom=None, version=None, scale_muscle_fmax=True):
    """ Rescale MIMo in an already-parsed spec, in place.

    The two ages are independent, which is the whole point of the roll-over experiment:
    everything geometric (geom size, position and mass, body, joint and site positions) follows
    ``morph_age``, and only ``actuator_gear`` -- the spring-damper model's maximum voluntary
    isometric torque -- follows ``physio_age``.

    Elements the growth schema does not name are left untouched, which is what keeps the floor,
    the lights, the cameras and the playroom toys out of it. Anything of MIMo's own that falls
    out of the schema would silently keep the base scene's age instead, so
    ``spec_check.py:test_schema_coverage`` fences that off by growing across the widest age gap
    and requiring every field to match the stored scene for the target age.

    Arguments:
        spec (mujoco.MjSpec): The spec to modify, in place.
        morph_age (float): Morphological (body) age in months.
        physio_age (float): Physiological (actuation) age in months. Defaults to ``morph_age``.
        path_scene (str): Path the spec came from. Read only to resolve the MIMo version, so it
            can be omitted when ``version`` is given directly -- which is what a curriculum
            holding one parsed spec should do.
        custom (dict): Custom geom sizes, applied to the morphological age.
        version (str): ``'v1'`` or ``'v2'``. Required if ``path_scene`` is omitted.
        scale_muscle_fmax (bool): Also scale the actuators' ``user`` field, which is where
            :class:`~mimoActuation.muscle.MuscleModel` reads FMAX and VMAX. See
            :func:`scale_fmax_with_gear`.

    Returns:
        mujoco.MjSpec: The same spec, for chaining.

    Raises:
        ValueError: If neither ``path_scene`` nor ``version`` is given.
    """
    if version is None and path_scene is None:
        raise ValueError("apply_growth needs either 'path_scene' or 'version'.")

    if physio_age is None:
        physio_age = morph_age

    # Resolved once: 'get_version' re-parses the scene, and the two ages share a version.
    if version is None:
        version = get_version(path_scene)

    params_morph = growth_params(path_scene, morph_age, custom, version)
    params_physio = (params_morph if physio_age == morph_age
                     else growth_params(path_scene, physio_age, None, version))

    for geom in spec.geoms:
        values = params_morph["geoms"].get(geom.name)
        if values is None:
            continue
        size = np.asarray(values["size"], dtype=float)
        # Assigned by slice: MuJoCo keeps three size slots whatever the geom type uses, and the
        # schema supplies only the ones that mean something for this type.
        geom.size[:len(size)] = size
        geom.pos[:] = np.asarray(values["pos"], dtype=float)
        geom.mass = float(values["mass"])

    for body in spec.bodies:
        values = params_morph["bodies"].get(body.name)
        if values is not None:
            body.pos[:] = np.asarray(values["pos"], dtype=float)

    for joint in spec.joints:
        values = params_morph["joints"].get(joint.name)
        if values is not None:
            joint.pos[:] = np.asarray(values["pos"], dtype=float)

    for site in spec.sites:
        values = params_morph["sites"].get(site.name)
        if values is not None:
            site.pos[:] = np.asarray(values["pos"], dtype=float)

    for actuator in spec.actuators:
        values = params_physio["motors"].get(actuator.name)
        if values is None:
            continue
        gear = float(values["gear"])
        if scale_muscle_fmax:
            scale_fmax_with_gear(actuator, gear)
        actuator.gear[0] = gear

    return spec


def scale_fmax_with_gear(actuator, gear):
    """ Rescale one actuator's FMAX so that its muscle calibration survives the age change.

    :class:`~mimoActuation.muscle.MuscleModel` reads FMAX and VMAX out of the actuator's ``user``
    attribute (``vmax fmax_negative fmax_positive``) and never looks at ``gear`` -- it overwrites
    ``actuator_gear`` on every physics step. So growing ``gear`` alone leaves ``--physio_age`` a
    no-op under ``--use_muscle``, which is exactly the bug ``mimoEnv/assets/mimo/age/
    generate_age_actuators.py`` was written to repair for the four pre-generated ages.

    The rule is that file's: ``FMAX(age) = FMAX_base * gear(age) / gear_base``, i.e. FMAX tracks
    geom volume. It is not a new assumption -- the stored FMAX values satisfy
    ``moment * FMAX = 1.07 * gear * |forcerange|``, and ``moment`` is age-invariant because joint
    ranges are, so preserving that calibration at every age forces FMAX to track gear. VMAX is a
    normalised fibre velocity, carries no length scale, and is left alone.

    Expressed here against the actuator's *current* gear rather than against the stock meta file,
    which makes it self-referential and independent of what age the base scene was written for.
    The price is that re-growing one spec several times chains ratios instead of applying one, so
    the result is equal but not bit-equal to a fresh parse -- measured at 3.9e-16 relative, one
    double epsilon, against the 1e-6 the pre-generated act files are themselves written to. See
    ``spec_check.py:test_spec_reuse``; everything the growth schema supplies is an absolute size
    and stays bit-identical.

    Arguments:
        actuator (mujoco.MjsActuator): The actuator, modified in place.
        gear (float): The gear this actuator is about to be given.
    """
    if len(actuator.userdata) < 3:
        return
    old_gear = float(actuator.gear[0])
    if old_gear == 0.0:
        return
    scale = gear / old_gear
    userdata = list(actuator.userdata)
    userdata[1] *= scale
    userdata[2] *= scale
    actuator.userdata = userdata


def grow_spec(path_scene, morph_age, physio_age=None, remove=(),
              custom=None, scale_muscle_fmax=True):
    """ Parse a scene and return a spec with MIMo grown to the given ages.

    Arguments:
        path_scene (str): Path to the MuJoCo scene.
        morph_age (float): Morphological (body) age in months.
        physio_age (float): Physiological (actuation) age in months. Defaults to ``morph_age``.
        remove (Iterable[str]): Body subtrees to amputate. See :func:`remove_bodies`.
        custom (dict): Custom geom sizes.
        scale_muscle_fmax (bool): See :func:`apply_growth`.

    Returns:
        mujoco.MjSpec: The grown spec. Call ``.compile()`` for the model.
    """
    spec = mujoco.MjSpec.from_file(path_scene)
    if remove:
        remove_bodies(spec, remove)
    return apply_growth(spec, morph_age, physio_age, path_scene=path_scene,
                        custom=custom, scale_muscle_fmax=scale_muscle_fmax)


def grow_model(path_scene, morph_age, physio_age=None, remove=(),
               custom=None, scale_muscle_fmax=True):
    """ Compile a grown scene straight to ``(MjModel, MjData)``.

    Arguments are those of :func:`grow_spec`.

    Returns:
        tuple[mujoco.MjModel, mujoco.MjData]: The compiled model and fresh data.
    """
    model = grow_spec(path_scene, morph_age, physio_age, remove,
                      custom, scale_muscle_fmax).compile()
    return model, mujoco.MjData(model)

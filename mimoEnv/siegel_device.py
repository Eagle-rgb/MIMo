""" The inclined infant device of Siegel et al. (2024), as a scene MIMo can be laid into.

Siegel, Siddicky, Davis & Mannen (2024), "Mechanical environment influences muscle activity during
infant rolling", Human Movement Science 95, 103208. Their infants rolled supine-to-prone on a flat
playmat and in a custom device that stands in for bouncers, rockers, swings and car seats: two
padded panels meeting in a hinge under the pelvis (their Fig. 5),

* the **seatback** under trunk and head, raised towards the head by the *seatback angle*, and
* the **base** under the legs, raised towards the feet by the *base angle*,

each with a raised soft rim -- the "sidewalls" the paper holds responsible for the leg push-off in
its 0 deg condition. Four configurations were tested, seatback/base 0/0, 10/15, 18/15 and 28/15
(:data:`SIEGEL_CONFIGURATIONS`); here both angles are free.

06.10.2026 This is **not** what ``--slope`` does, and the two must not be confused. ``--slope``
tilts the floor about MIMo's long axis, so the incline runs *across* the roll (rolling uphill or
downhill). Siegel's incline runs *along* the body: head up, legs up, the trunk flexed in a V, and
the infant rolls about an axis that is itself inclined. ``--slope`` could only reproduce it if it
tilted about the y axis instead, and even then it would be one plane, not a hinge with sidewalls.

This module owns the geometry and the placement, and can render both on its own (below).
``MIMoRollOverEnv`` uses it since 06.10.2026: ``--seatback_angle``/``--base_angle`` make
``_build_model_spec`` call :func:`add_siegel_device` and ``put_in_starting_position`` call
:func:`pitch_and_flex`/:func:`lower_onto_device`. What the env changes beyond that -- rho becomes
a roll about the body's own long axis -- is documented there, in ``_roll_cosine``.

**The dimensions are estimates.** The paper gives the angles and nothing else -- no panel length,
width or rim height. The defaults are read off its photographs against the infant in them
(Fig. 4C, Fig. 5B-E) and sized to MIMo at 6.5 months, the sample's mean age: MIMo is 0.41 m from
hip to crown and 0.30 m from hip to sole, so a 0.55 m seatback and a 0.42 m base leave the same
kind of margin the photographs show. Every one of them is a keyword argument.

Usage::

    MUJOCO_GL=osmesa python mimoEnv/siegel_device.py --seatback_angle=18 --base_angle=15
    MUJOCO_GL=osmesa python mimoEnv/siegel_device.py --siegel      # the paper's four

Still open: the device is rigid. Siegel's had "firm support with thin soft-goods", and the floor
flags (``--floor_softness``) address the floor geom only.
"""
import numpy as np
import mujoco


SIEGEL_CONFIGURATIONS = ((0.0, 0.0), (10.0, 15.0), (18.0, 15.0), (28.0, 15.0))
""" (seatback angle, base angle) in degrees, the four device conditions of Siegel et al. (2024). """

SIEGEL_AGE_MONTHS = 6.5
""" Mean age of Siegel's sample (6.5 +- 0.7 months). """

DEVICE_PREFIX = "device_"
""" Name prefix of every device geom, so callers can find them without a list. """

# Colours only -- the soft goods are white in the photographs, the stand is stained wood.
_RGBA_PAD = [0.93, 0.93, 0.90, 1.0]
_RGBA_RIM = [0.98, 0.98, 0.96, 1.0]
_RGBA_WOOD = [0.42, 0.27, 0.15, 1.0]


def _quat_about_y(degrees):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), np.deg2rad(degrees))
    return quat


HINGE_HEIGHT = 0.30
""" Default height of the hinge line above the floor, metres. """


def add_siegel_device(spec, seatback_angle, base_angle, hinge_height=HINGE_HEIGHT,
                      seatback_length=0.55, base_length=0.42, width=0.50,
                      rim_radius=0.035, thickness=0.02, stand=True):
    """ Add the two-panel device to a spec. The hinge line is the world y axis at ``hinge_height``.

    MIMo lies with his head towards -x in the supine reset pose, so the seatback extends towards
    -x and the base towards +x. Both angles are measured from the horizontal and positive
    *upwards away from the hinge*, as in the paper's Fig. 5F; negative values are accepted and
    tilt the panel downwards.

    Args:
        spec (mujoco.MjSpec): The scene, edited in place.
        seatback_angle (float): Incline of the panel under trunk and head, degrees.
        base_angle (float): Incline of the panel under the legs, degrees.
        hinge_height (float): Height of the hinge line above the floor, metres.
        seatback_length (float): Hinge to head end, metres.
        base_length (float): Hinge to foot end, metres.
        width (float): Extent along the hinge, metres.
        rim_radius (float): Radius of the padded rim. It sits on the panel, so the sidewall is
            twice this high.
        thickness (float): Panel thickness, metres.
        stand (bool): Add the wooden stand. Visual only; it does not collide.

    Returns:
        The device's root body.
    """
    for name, value in (("seatback_angle", seatback_angle), ("base_angle", base_angle)):
        if not -90.0 < value < 90.0:
            raise ValueError(f"'{name}' must lie strictly between -90 and 90 degrees, got {value}.")

    device = spec.worldbody.add_body(name="siegel_device", pos=[0.0, 0.0, hinge_height])

    # A rotation about +y by 'a' sends +x to (cos a, 0, -sin a). The seatback runs along -x, so
    # +a raises it; the base runs along +x, so it needs -a.
    panels = (("seatback", -1.0, seatback_length, seatback_angle),
              ("base", +1.0, base_length, -base_angle))
    for name, direction, length, rotation in panels:
        panel = device.add_body(name=f"siegel_{name}", quat=_quat_about_y(rotation))
        # Top surface in the panel's z = 0 plane, so it passes exactly through the hinge line.
        panel.add_geom(name=f"{DEVICE_PREFIX}{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[length / 2.0, width / 2.0, thickness / 2.0],
                       pos=[direction * length / 2.0, 0.0, -thickness / 2.0], rgba=_RGBA_PAD)
        # The rim: both long edges and the far end, open towards the hinge like the two separate
        # cushions in the photographs. It starts a rim's width away from the hinge so the two
        # panels' rims do not interpenetrate when the device folds.
        near, far, side = direction * 2.0 * rim_radius, direction * length, width / 2.0
        edges = (("left", [near, side, rim_radius, far, side, rim_radius]),
                 ("right", [near, -side, rim_radius, far, -side, rim_radius]),
                 ("end", [far, -side, rim_radius, far, side, rim_radius]))
        for edge, fromto in edges:
            panel.add_geom(name=f"{DEVICE_PREFIX}{name}_rim_{edge}",
                           type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rim_radius, 0.0, 0.0],
                           fromto=fromto, rgba=_RGBA_RIM)

    if stand:
        # Visual only. The panels are welded to the world, so nothing has to hold them up.
        reach = max(seatback_length, base_length)
        pillar_height = hinge_height - thickness
        device.add_geom(name="stand_pillar", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.05, 0.12, pillar_height / 2.0],
                        pos=[0.0, 0.0, -thickness - pillar_height / 2.0],
                        rgba=_RGBA_WOOD, contype=0, conaffinity=0)
        device.add_geom(name="stand_board", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[reach, 0.18, 0.012],
                        pos=[(base_length - seatback_length) / 2.0, 0.0, 0.012 - hinge_height],
                        rgba=_RGBA_WOOD, contype=0, conaffinity=0)
    return device


def device_geom_ids(model):
    """ Ids of the colliding device geoms (panels and rims, not the stand). """
    return {i for i in range(model.ngeom) if model.geom(i).name.startswith(DEVICE_PREFIX)}


def _touches_device(model, data, device_ids):
    # Positions and contacts only; the dynamics 'mj_forward' would add are not needed here.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    for contact in data.contact[:data.ncon]:
        if (contact.geom1 in device_ids) != (contact.geom2 in device_ids):
            return True
    return False


def pitch_and_flex(model, qpos, seatback_angle, base_angle):
    """ Turn a supine pose into one that fits the device, in place. Does not set the position.

    Placing a jointed body by its root alone does not work in a V, so two things change: the root
    is pitched by the seatback angle, which makes the trunk parallel to the seatback, and both
    hips are flexed by a further seatback + base angle, which makes the legs parallel to the base
    instead of passing through it. The flexion is *added* to what ``qpos`` already holds, so the
    env's per-joint reset noise survives.

    Args:
        qpos (np.ndarray): A full qpos whose quaternion (3:7) is the env's supine starting pose.
    """
    # Pitched about the world y axis -- a world-frame rotation, hence the left multiplication.
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, _quat_about_y(seatback_angle), np.asarray(qpos[3:7], dtype=float))
    qpos[3:7] = quat

    # 'hip1' is flexion/extension, negative is flexion (range -133..20 deg). A body with a leg
    # cut off has no such joint, hence the lookup.
    flexion = -np.deg2rad(seatback_angle + base_angle)
    for joint in ("robot:left_hip1", "robot:right_hip1"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            continue
        address = model.jnt_qposadr[joint_id]
        low, high = model.jnt_range[joint_id]
        qpos[address] = np.clip(qpos[address] + flexion, low, high)
    return qpos


def lower_onto_device(model, data, hinge_height=HINGE_HEIGHT,
                      clearance=0.001, resolution=0.0005, start=0.40):
    """ Lower MIMo vertically until he just touches the device.

    The analogue of ``get_minimal_z_coordinate`` on the floor, where the surface is two planes
    and six capsules and there is no closed form. ``data.qpos`` must hold the pose already
    (:func:`pitch_and_flex`); only ``qpos[0:3]`` is written. The hip ends up over the hinge line.

    A bisection, not a walk: this runs on every reset, and it needs ~10 collision passes instead
    of 300. That assumes contact is monotone in the height, which holds for a body approaching a
    concave V from above.

    Vertically, not along the seatback's normal: the normal leans towards the feet, so backing
    off along it pushes the legs out over the base's end rim, and at steep angles (40/30) MIMo
    then "touches" the device from 0.30 m away.

    Returns:
        float: How far above the hinge line the root ended up.
    """
    device_ids = device_geom_ids(model)
    normal = np.array([0.0, 0.0, 1.0])
    hinge = np.array([0.0, 0.0, hinge_height])

    def touches(height):
        data.qpos[0:3] = hinge + height * normal
        return _touches_device(model, data, device_ids)

    if touches(start):
        raise RuntimeError(f"MIMo already touches the device {start} m above it; the device is "
                           "too small for this body or the angles fold it shut.")
    free, blocked = start, 0.0
    if not touches(blocked):
        # Nothing to rest on at the hinge line (possible only for a body with no geom there).
        free = blocked
    while free - blocked > resolution:
        middle = 0.5 * (free + blocked)
        if touches(middle):
            blocked = middle
        else:
            free = middle
    data.qpos[0:3] = hinge + (free + clearance) * normal
    mujoco.mj_forward(model, data)
    return free + clearance


def place_in_device(model, data, seatback_angle, base_angle, hinge_height=HINGE_HEIGHT):
    """ Lay MIMo supine into the device from the model's default pose, velocities zeroed.

    The standalone version of what ``MIMoRollOverEnv.put_in_starting_position`` does with its
    own noisy pose. MIMo is in contact afterwards but not yet settled -- step the model for that.
    """
    mujoco.mj_resetData(model, data)
    # The env's supine pose: euler y = -90 deg, head towards -x.
    supine = np.zeros(4)
    mujoco.mju_euler2Quat(supine, np.array([0.0, -np.pi / 2.0, 0.0]), "xyz")
    data.qpos[3:7] = supine
    pitch_and_flex(model, data.qpos, seatback_angle, base_angle)
    height = lower_onto_device(model, data, hinge_height)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return height


def build(seatback_angle, base_angle, age=SIEGEL_AGE_MONTHS, **device_kwargs):
    """ Compile the base scene with MIMo grown to ``age`` and the device added.

    Textures are kept: this exists to be rendered. One such model is ~1 GB.
    """
    # Imported here: 'roll_over' imports this module, so the top level may not import it back.
    from mimoGrowth.spec import grow_spec
    from mimoEnv.envs.roll_over import BASE_SCENE
    spec = grow_spec(BASE_SCENE, age, age)
    add_siegel_device(spec, seatback_angle, base_angle, **device_kwargs)
    model = spec.compile()
    return model, mujoco.MjData(model)


def settle(model, data, steps=600):
    """ Step with zero control, i.e. a limp MIMo. Returns the root's final speed in m/s. """
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data, nstep=steps)
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.qvel[0:3]))


VIEWS = {
    # name: (azimuth, elevation, distance). Azimuth 90 looks along +y, i.e. from MIMo's right,
    # with his head on the left -- the orientation of the paper's Fig. 5.
    "side": (90.0, -8.0, 1.75),
    "oblique": (55.0, -28.0, 1.85),
    "top": (90.0, -89.0, 1.6),
}


def render_views(model, data, views=("side", "oblique", "top"), width=800, height=600,
                 hinge_height=HINGE_HEIGHT):
    """ Render the current state from the named :data:`VIEWS`, left to right in one image. """
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.05, 0.0, hinge_height + 0.05]
    frames = []
    for view in views:
        camera.azimuth, camera.elevation, camera.distance = VIEWS[view]
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return np.concatenate(frames, axis=1)


def render_configuration(seatback_angle, base_angle, age=SIEGEL_AGE_MONTHS, settle_steps=600):
    """ Build, place, settle and render one configuration. Returns (image, report dict). """
    model, data = build(seatback_angle, base_angle, age=age)
    height = place_in_device(model, data, seatback_angle, base_angle)
    start = data.body("hip").xpos.copy()
    speed = settle(model, data, settle_steps)
    hip = data.body("hip").xpos.copy()
    report = {
        "seatback_angle": seatback_angle, "base_angle": base_angle, "age": age,
        "hip_height_over_hinge": height,
        "hip_drift_m": float(np.linalg.norm(hip - start)),
        "hip_drift_xyz": (hip - start).round(4).tolist(),
        "root_speed_after_settle": speed,
        "settle_seconds": settle_steps * model.opt.timestep,
        "mass_kg": float(model.body_subtreemass[model.body("mimo_location").id]),
    }
    return render_views(model, data), report


if __name__ == "__main__":
    import argparse
    import os
    import sys
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seatback_angle", type=float, default=18.0,
                        help="Incline of the panel under trunk and head, degrees (default 18).")
    parser.add_argument("--base_angle", type=float, default=15.0,
                        help="Incline of the panel under the legs, degrees (default 15).")
    parser.add_argument("--siegel", action="store_true",
                        help="Render the paper's four configurations (0/0, 10/15, 18/15, 28/15) "
                             "instead of --seatback_angle/--base_angle, one row each.")
    parser.add_argument("--age", type=float, default=SIEGEL_AGE_MONTHS,
                        help="MIMo's age in months, body and actuation (default 6.5, the mean "
                             "age of Siegel's sample).")
    parser.add_argument("--settle_steps", type=int, default=600,
                        help="Physics steps of zero control before the picture (default 600, "
                             "i.e. 3 s).")
    parser.add_argument("--out", default="siegel_device.png", help="Output PNG.")
    args = parser.parse_args()

    configurations = SIEGEL_CONFIGURATIONS if args.siegel else ((args.seatback_angle,
                                                                 args.base_angle),)
    rows = []
    for seatback, base in configurations:
        image, report = render_configuration(seatback, base, age=args.age,
                                             settle_steps=args.settle_steps)
        rows.append(image)
        print(f"seatback {seatback:g} deg / base {base:g} deg: hip {report['hip_height_over_hinge'] * 1000:.0f} mm "
              f"over the hinge, drifted {report['hip_drift_m'] * 1000:.1f} mm {report['hip_drift_xyz']} "
              f"in {report['settle_seconds']:.1f} s, root speed {report['root_speed_after_settle']:.4f} m/s")
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out}")

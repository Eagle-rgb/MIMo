""" Generates the playroom variants of the roll-over scenes.

01.09.2026 The roll-over scenes are a bare checkerboard plane under an empty gradient skybox.
Measured from MIMo's own eye camera at reset: supine he looks up at +87.9 degrees and sees a
single flat blue, prone he looks down at -86.1 degrees and sees black. There is nothing in either
image, which makes any vision-driven reward vacuous. This script adds the toys.

Why a generator and not sixteen hand-edited files: the scenes are pre-generated on purpose (see
"Age scenes are pre-generated" in CLAUDE.md -- parallel cluster runs raced on temporary scenes), so
the playroom cannot be built at runtime either. Sixteen more hand-maintained XMLs would rot; this
script rebuilds all of them from 'scene_act_<p>_body_<m>.xml' plus one include line.

The toy meshes have been sitting unused in 'mimoEnv/assets/meshes' since 4251dbf (Aubret, 2022).
They are Toys4K models normalised to roughly five units across, hence the 0.03 scale for a ~15 cm
toy.

Two decisions worth knowing:

- **Only the low-poly meshes are used.** Rendering, not physics, is the wall-clock cost of this
  experiment (see the comment in 'mimoVision.SimpleVision.get_vision_obs'), and it is software
  rasterisation, so it scales with triangles. The seven meshes used here total 55.8 k faces.
  'apple_000' and 'sheep_002' alone are 930 k, i.e. seventeen times the rest of the playroom put
  together, and 'banana/mushroom/cookie/penguin/cake' are 400 k more. They are left out rather
  than decimated because decimation needs a dependency ('fast_simplification') and
  'requirements.txt' is pinned deliberately.
- **The toys are static and non-colliding** ('contype=0 conaffinity=0', no freejoint). They sit
  0.15-0.2 m beyond MIMo's fingertips at 9 months, so he cannot reach them in any case, and this
  way they cannot drift between episodes, cost nothing in the solver, and cannot perturb the roll.
  Give them a freejoint here if the experiment ever grows a reaching component.

Run it from the repository root:

    python mimoEnv/assets/roll_over/generate_playroom_scenes.py
"""

import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PRONE = os.path.join(HERE, "prone")
INCLUDE_NAME = "playroom_incl.xml"

# MIMo lies along x and spans x in [-0.35, 0.35], y in [-0.29, 0.29], z <= 0.13 (measured at
# reset, 9 months, both postures). A ring at 0.5 m therefore clears his fingertips by 0.15 m.
# Measured, not guessed. 'mimoEnv/eval_look_landscape.py' sweeps MIMo through the roll and
# reports how much toy he can get into the fovea at each angle. At 0.50 m the toys sit 0.22 m from
# his face and a single cow fills a third of the eye; at 0.80 m the whole landscape flattens out.
# 0.65 m puts them 0.37 m from either head position -- about an arm's length -- and gives
# 0.027 foveal share supine against 0.231 prone, a factor of 8.5.
RING_RADIUS = 0.65
# The second tier. Supine the eye sits at z = 0.14 and looks straight up; prone it sits on the
# floor and looks straight down. In both postures a floor-level toy is ~105 degrees off the eye
# axis, i.e. far outside the 60 degree field of view, and only becomes visible once MIMo turns
# onto his side. The raised toys are what he can see while prone with the head lifted -- without
# them the look reward would peak at side-lying and pay nothing extra for completing the roll.
PEDESTAL_HEIGHT = 0.15
PEDESTAL_RADIUS = 0.70

# name, mesh file, scale, material, angle around MIMo in degrees (0 = +x)
TOYS = [
    ("toy_truck",    "truck_000.stl",    0.030, "toy_red",     0),
    ("toy_dinosaur", "dinosaur_004.stl", 0.030, "toy_green",   45),
    ("toy_ball",     "ball_004.stl",     0.032, "toy_orange",  90),
    ("toy_train",    "train_017.stl",    0.030, "toy_blue",    135),
    ("toy_cow",      "cow_001.stl",      0.030, "toy_white",   180),
    ("toy_airplane", "airplane_021.stl", 0.030, "toy_yellow",  225),
    ("toy_hammer",   "hammer_001.stl",   0.030, "toy_purple",  270),
]

# Bright saturated colours, because that is both what infant toys look like and what a 64x64
# image can still resolve. 'toy_white' keeps the cow readable against the coloured ones.
MATERIALS = [
    ("toy_red",    "0.85 0.13 0.13 1"),
    ("toy_green",  "0.13 0.70 0.20 1"),
    ("toy_orange", "0.95 0.50 0.05 1"),
    ("toy_blue",   "0.15 0.35 0.90 1"),
    ("toy_white",  "0.95 0.95 0.92 1"),
    ("toy_yellow", "0.95 0.85 0.10 1"),
    ("toy_purple", "0.60 0.20 0.80 1"),
    ("toy_pink",   "0.95 0.45 0.65 1"),
    ("toy_cyan",   "0.10 0.80 0.85 1"),
    ("toy_post",   "0.55 0.40 0.28 1"),
]

# The raised tier: a coloured sphere on a thin post. Primitives rather than meshes, so the second
# tier costs essentially nothing to rasterise.
# The rewarding geoms are exactly those whose name starts with 'toy_' -- see
# 'mimoEnv/envs/roll_over_look.py', which discovers them by that prefix rather than by a list it
# would have to be kept in sync with. The stands are deliberately NOT called that: they are
# furniture holding a toy up, not something to look at.
PEDESTALS = [
    ("toy_post_pink", "toy_pink", 60),
    ("toy_post_cyan", "toy_cyan", 180),
    ("toy_post_red",  "toy_red",  300),
]


def _ring(radius, angle_deg):
    import math
    a = math.radians(angle_deg)
    return radius * math.cos(a), radius * math.sin(a)


def build_include(z_offsets):
    """ Writes 'playroom_incl.xml'.

    Args:
        z_offsets (dict): Per-toy z correction, so that the bottom of the mesh sits on the floor.
            MuJoCo's compiler recentres mesh vertices, so where a mesh sits relative to its geom
            frame is not something to read off the STL -- it is measured from the compiled model
            by :func:`.calibrate` and passed back in here.
    """
    lines = ['<mujocoinclude>', '', '  <asset>']
    for name, rgba in MATERIALS:
        lines.append(f'    <material name="{name}" rgba="{rgba}" specular=".3" shininess=".4" />')
    lines.append('')
    for name, mesh, scale, _material, _angle in TOYS:
        lines.append(f'    <mesh name="{name}_mesh" file="../../meshes/{mesh}" '
                     f'scale="{scale} {scale} {scale}" />')
    lines += ['  </asset>', '', '  <worldbody>', '']
    # 01.09.2026 The roll-over scene is lit by one dim directional light and one aimed at the
    # upper body from (3, 0, 5). That is fine for a third-person video, but MIMo's eye sits 1-14 cm
    # off the floor and looks sideways, where the rendered toys came out nearly black -- and a
    # 64x64 vision observation of a black image carries no signal. This light is inside the
    # playroom include, so the scenes without toys are bit-identical to what they were and no
    # stored run changes meaning.
    lines.append('    <light directional="true" ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35" '
                 'specular="0 0 0" pos="0 0 3" dir="0 0 -1" castshadow="false" />')
    lines.append('')
    lines.append('    <!-- Floor ring. Static and non-colliding, see generate_playroom_scenes.py. -->')
    for name, _mesh, _scale, material, angle in TOYS:
        x, y = _ring(RING_RADIUS, angle)
        z = z_offsets.get(name, 0.0)
        # Turned to face MIMo, so the recognisable side of each toy points inwards.
        lines.append(f'    <body name="{name}" pos="{x:.4f} {y:.4f} 0" euler="0 0 {angle + 180}">')
        lines.append(f'      <geom name="{name}" type="mesh" mesh="{name}_mesh" '
                     f'material="{material}" pos="0 0 {z:.4f}" contype="0" conaffinity="0" />')
        lines.append('    </body>')
    lines += ['', '    <!-- Raised tier, visible from prone with the head lifted. -->']
    for name, material, angle in PEDESTALS:
        x, y = _ring(PEDESTAL_RADIUS, angle)
        lines.append(f'    <body name="{name}" pos="{x:.4f} {y:.4f} 0">')
        lines.append(f'      <geom name="stand_{name}" type="cylinder" material="toy_post" '
                     f'size="0.012 {PEDESTAL_HEIGHT / 2:.4f}" pos="0 0 {PEDESTAL_HEIGHT / 2:.4f}" '
                     f'contype="0" conaffinity="0" />')
        lines.append(f'      <geom name="{name}" type="sphere" material="{material}" '
                     f'size="0.045" pos="0 0 {PEDESTAL_HEIGHT + 0.045:.4f}" '
                     f'contype="0" conaffinity="0" />')
        lines.append('    </body>')
    lines += ['', '  </worldbody>', '', '</mujocoinclude>', '']
    path = os.path.join(PRONE, INCLUDE_NAME)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def calibrate():
    """ Measures how far each toy has to be lifted so that it rests on the floor.

    Compiles the playroom once with every toy at z = 0 and reads 'model.geom_aabb', which is the
    geom's axis-aligned bounding box in its own frame *after* the compiler has recentred the mesh.
    The correction is the distance from the frame origin down to the box's lower z face, rotated
    into the world -- the toys are only yawed, so that rotation leaves z alone.
    """
    import mujoco
    scene = os.path.join(PRONE, "scene_act_9_body_9_playroom.xml")
    offsets = {name: 0.0 for name, _m, _s, _mat, _a in TOYS}
    # Iterated rather than solved in one shot: 'geom_aabb' is a bound on the mesh, and it does not
    # come back bit-identical between two compilations of the same mesh at different geom offsets
    # -- a single pass left the cow floating 1.5 cm and the truck sunk 0.5 cm. Three passes bring
    # every toy under a tenth of a millimetre.
    for _ in range(3):
        build_include(offsets)
        build_scenes()
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        residuals = {}
        for name, _mesh, _scale, _material, _angle in TOYS:
            gid = model.geom(name).id
            # Lowest point of the geom in the world. The toys are only yawed, and yaw about z
            # leaves the z extent of an axis-aligned box alone, so this needs no rotation.
            lowest = data.geom_xpos[gid][2] + model.geom_aabb[gid][2] - model.geom_aabb[gid][5]
            residuals[name] = lowest
            offsets[name] -= lowest
        worst = max(abs(v) for v in residuals.values())
    print(f"  z calibration converged to {worst * 1000:.4f} mm")
    return offsets


def build_scenes():
    """ Writes a '<scene>_playroom.xml' next to every 'scene_act_<p>_body_<m>.xml'. """
    written = []
    for scene in sorted(glob.glob(os.path.join(PRONE, "scene_act_*_body_*.xml"))):
        if scene.endswith("_playroom.xml"):
            continue
        with open(scene) as fh:
            text = fh.read()
        marker = "    <worldbody>"
        if marker not in text:
            raise RuntimeError(f"{scene}: no '<worldbody>' to anchor the include on.")
        # Top level, next to the other includes: a <mujocoinclude> carrying its own <asset> and
        # <worldbody> is merged into the corresponding sections of the including file.
        include = f'    <include file="{INCLUDE_NAME}" />\n\n{marker}'
        out = text.replace(marker, include, 1)
        target = scene[: -len(".xml")] + "_playroom.xml"
        with open(target, "w") as fh:
            fh.write(out)
        written.append(target)
    return written


def main():
    offsets = calibrate()
    build_include(offsets)
    written = build_scenes()
    print(f"Wrote {os.path.join(PRONE, INCLUDE_NAME)}")
    for name, z in sorted(offsets.items()):
        print(f"  {name:14s} lifted by {z * 100:6.2f} cm")
    print(f"Wrote {len(written)} playroom scenes.")


if __name__ == "__main__":
    main()

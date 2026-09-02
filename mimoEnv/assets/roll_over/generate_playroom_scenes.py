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

# 01.09.2026 The layout is measured, not arranged -- but read the correction below before
# trusting the mechanism.
#
# What the toys have to do is make *coverage* -- how many different toys MIMo has looked at in one
# episode -- cheaper from prone than from supine. Measured with the same segmentation renderer the
# reward uses, counting a toy as seen at 0.01 foveal share:
#
#   roll     0  20  40  60  80 100 120 140 160 180
#   comfortable head range (70 %)
#   toys     3   4   4   5   5   5   6   8   9   8
#   full head range
#   toys    10  10  10  10  10  10  10  10  10  10
#
# So the honest statement is a **cost gradient, not a hard constraint**. At comfortable head poses
# coverage climbs 3 -> 9 across the roll, which is the behaviour the reward is meant to select.
# At full stretch MIMo reaches all ten from flat on his back, and no arrangement of toys on a
# floor defeats that: his neck sweeps nearly a hemisphere (swivel +-111 deg, tilt -70..81,
# tilt_side +-70), so posture barely limits which *directions* he can point the eye. The foveal
# weighting in 'roll_over_look.py' is what converts "cheaper" into "actually preferred", because
# the contorted pose only ever catches a toy at the edge of the field, where it is discounted.
#
# **Correction, same day.** An earlier version of this comment claimed full coverage was
# geometrically unreachable from supine, on the strength of an 'mj_ray' probe that reported 39
# prone-only positions and 0 supine-only. That probe was wrong: a ray from the supine eye at
# z = 0.14 down to a toy at floor level grazes the floor plane and was counted as occluded, while
# the same ray from the prone eye at z = 0.001 runs nearly parallel to it and hits nothing. The
# 68-against-107 split it produced is that artefact, not occlusion by MIMo's own body. The
# render-based numbers above supersede it.
#
# The layout is still built along the axis the probe suggested, because the rendered numbers agree
# with it under the comfort constraint: three toys towards MIMo's feet, where he can pick them up
# lying on his back and so has something to earn from the first step, and seven behind his head
# and low to the floor, which is where the comfortable-pose gradient lives.
#
# The previous layout was a 0.65 m ring with three toys raised on 0.15 m stands. It scored 5 -> 8
# on the same comfortable-range test, and the stands were actively counterproductive: height is
# what makes a position easy to see from supine.
#
# One fact constrains any future rearrangement: **MIMo's head does not translate during a roll.**
# Measured across 0..180 degrees it stays at x = -0.276; only the eye's optical axis rotates, from
# +z through -y to -z. (The env's two reset poses do differ, -0.276 supine against +0.274 prone,
# but that is 'get_starting_quat' using euler[1] = +-90 -- a rotation about y that swaps head and
# feet, not a roll. A MIMo who rolls out of supine ends up mirrored head-to-toe from the 'prone'
# reset pose.) So a layout can only exploit where the eye *points* and what MIMo's own body
# occludes, never where his head has moved to.

# name, mesh file, scale, material, x, y
TOYS = [
    # --- visible from supine as well: the bootstrap set -------------------------------
    ("toy_ball",     "ball_004.stl",     0.032, "toy_orange",  0.65,  0.60),
    ("toy_truck",    "truck_000.stl",    0.030, "toy_red",     0.65, -0.60),
    # Scaled up against the others: the mesh is flat (5.5 cm tall at 0.030) and a floor-
    # level eye sees it almost edge-on, which left it at 0.0097 best foveal share -- just
    # under the 0.01 seen threshold, i.e. a toy that could never be collected.
    ("toy_airplane", "airplane_021.stl", 0.042, "toy_yellow",  0.70,  0.35),
    # --- prone only: these are what force the roll ------------------------------------
    ("toy_cow",      "cow_001.stl",      0.030, "toy_white",  -0.70,  0.30),
    ("toy_dinosaur", "dinosaur_004.stl", 0.030, "toy_green",  -0.70, -0.30),
    ("toy_train",    "train_017.stl",    0.030, "toy_blue",   -0.90,  0.15),
    ("toy_hammer",   "hammer_001.stl",   0.030, "toy_purple", -0.45, -0.75),
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
]

# Plain spheres resting on the floor -- no stands, see above. Same two groups.
BALLS = [
    ("toy_ball_pink", "toy_pink",  0.80, -0.45),   # visible from supine too
    ("toy_ball_cyan", "toy_cyan", -0.90, -0.45),   # prone only
    ("toy_ball_red",  "toy_red",  -0.45,  0.60),   # prone only
]
# 6.5 cm rather than 5: at 5 cm the far ball peaked at 0.0127 foveal share, barely over
# the seen threshold, so a small pose error made it uncollectable.
BALL_RADIUS = 0.065

# 02.09.2026 The crib mobile -- the bootstrap that replaces '--isr'.
#
# Without it the looking reward is unreachable from a cold start. Measured: a toy only enters
# MIMo's field of view after he has HELD a near-maximal head action for ~100 consecutive steps
# (swivel 15.7 deg after 10 steps, 70.0 after 80, first toy over the 0.01 threshold at 100; a
# single-step impulse then release reaches 1.9 deg and sees nothing). Uncorrelated exploration
# never produces that, so a random policy saw a toy in 0 of 10 episodes and a 1M-step SAC run
# plateaued at 1.6 of 10 toys with rho flat on its ISR baseline.
#
# A toy hanging over his face needs no head movement at all. Measured on a random policy, 200
# steps: visible on 200/200 at heights 0.35, 0.45 and 0.60 m (foveal share 0.537 / 0.289 / 0.148).
# It is offset in y so that it sits ~23 degrees off the optical axis at rest -- inside the 60
# degree field but foveally discounted, so a small head turn is worth something and MIMo has a
# gradient linking "turn head" to "see more" from the first episode.
#
# It does not become a free lunch: habituation halves what it pays every 50 steps of viewing and
# '--look_recovery_steps=0' means it never recovers, so it is spent within a few hundred steps and
# the floor toys -- which need the roll -- are the only source left. It counts as one more toy
# towards coverage, so full coverage still requires turning over.
#
# The supine eye sits at (-0.338, 0.024, 0.139) and looks almost straight up. MIMo's head does not
# translate during a roll, so the mobile stays over him throughout.
MOBILE = ("toy_mobile", "toy_cyan", -0.338, 0.180, 0.50, 0.05)  # name, material, x, y, z, radius

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
    for name, mesh, scale, _material, _x, _y in TOYS:
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
    lines.append('    <!-- Static and non-colliding, see generate_playroom_scenes.py. -->')
    for name, _mesh, _scale, material, x, y in TOYS:
        z = z_offsets.get(name, 0.0)
        # Yawed to face MIMo, so the recognisable side of each toy points inwards.
        import math
        yaw = math.degrees(math.atan2(-y, -x))
        lines.append(f'    <body name="{name}" pos="{x:.4f} {y:.4f} 0" euler="0 0 {yaw:.1f}">')
        lines.append(f'      <geom name="{name}" type="mesh" mesh="{name}_mesh" '
                     f'material="{material}" pos="0 0 {z:.4f}" contype="0" conaffinity="0" />')
        lines.append('    </body>')
    lines.append('')
    name, material, mx, my, mz, mr = MOBILE
    lines.append(f'    <body name="{name}" pos="{mx:.4f} {my:.4f} {mz:.4f}">')
    lines.append(f'      <geom name="{name}" type="sphere" material="{material}" '
                 f'size="{mr}" contype="0" conaffinity="0" />')
    lines.append('    </body>')
    for name, material, x, y in BALLS:
        lines.append(f'    <body name="{name}" pos="{x:.4f} {y:.4f} 0">')
        lines.append(f'      <geom name="{name}" type="sphere" material="{material}" '
                     f'size="{BALL_RADIUS}" pos="0 0 {BALL_RADIUS}" '
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
    offsets = {name: 0.0 for name, _m, _s, _mat, _x, _y in TOYS}
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
        for name, _mesh, _scale, _material, _x, _y in TOYS:
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

"""Render one episode of a saved roll-over policy top-down, one transparent PNG per frame.

    conda run -n mimo python render/render_topdown_episode.py \
        --run models/arch/age9/supine/26-03-07_supine_age9_run_0 \
        --out /path/to/output_dir

Reproduces the environment the run was trained with (read from its data.yml), the same way
mimoEnv/illustrations.py does for --test, so the episode ends the natural way (success or the
500-step horizon) rather than under eval_rollover.py's fixed-length evaluation protocol.

Background transparency is built from MuJoCo's segmentation render: a pixel is opaque only if
it belongs to a geom of a body in MIMo's kinematic subtree (root 'mimo_location'); floor,
skybox etc. are made transparent (alpha=0).

Alongside the PNG sequence an .mp4 is written (--video / --no_video, framerate via --fps).

The two outputs use *different cameras*, on purpose. The PNG frames track MIMo -- the camera
re-aims at the centre of his bounding box every frame, so he is always centred and the sequence
is a clean cutout for a figure. The video instead uses a camera that is fixed for the whole
episode (aimed once, at MIMo's centre at reset), so the roll shows as motion across the frame
rather than as MIMo squirming in place while the world slides past him.

mp4 carries no alpha either, so the video's frames are composited onto --video_bg: by default
'rug', i.e. the nursery carpet texture that ships with MIMo (mimoEnv/assets/tex/base_rug.png),
otherwise any image file or a flat colour. Because the video camera is static, that background
is world-fixed too -- exactly what a real carpet would look like.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import cv2
import gymnasium as gym
import mujoco
from PIL import Image

import mimoEnv  # noqa: F401  registers MIMoRollOver-v0
from mimoEnv.envs.roll_over import TOUCH_PARAMS
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from mimoEnv.eval_rollover import load_run_config, pick_checkpoint, load_policy
import mimoEnv.utils as env_utils
from render.utils import create_top_down_camera, get_surrounding_rect_center


def mimo_geom_ids(model):
    root = env_utils.get_body_id(model, body_name='mimo_location')
    bodies = env_utils.get_child_bodies(model, root)
    geoms = set()
    for b in bodies:
        geoms.update(env_utils.get_geoms_for_body(model, b))
    return geoms


def render_rgba(data, renderer, cam, geom_ids):
    """RGB render plus an alpha channel that is opaque exactly on MIMo's own geoms."""
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    seg_geom_id = seg[:, :, 0]
    seg_obj_type = seg[:, :, 1]
    mask = (seg_obj_type == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg_geom_id, list(geom_ids))
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


NAMED_COLORS = {'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (128, 128, 128),
                'grey': (128, 128, 128), 'green': (0, 177, 64)}

# The nursery carpet. It ships with MIMo but is referenced by no scene XML (grep: no hit outside
# this file), so using it here costs nothing and adds nothing to the simulation -- it is painted
# on afterwards, never compiled into the model. 4962x4962 teal with scattered dots.
RUG_TEXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'mimoEnv', 'assets', 'tex', 'base_rug.png')


def parse_color(value):
    """'white' | 'black' | 'gray' | 'green' | 'R,G,B' -> an (R, G, B) uint8 tuple."""
    key = value.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    parts = key.replace(';', ',').split(',')
    if len(parts) != 3:
        raise ValueError(
            f"--video_bg takes 'rug', an image path, {'/'.join(sorted(NAMED_COLORS))} "
            f"or 'R,G,B', got '{value}'")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"--video_bg components must be integers, got '{value}'")
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError(f"--video_bg components must be in 0..255, got '{value}'")
    return rgb


def make_background(value, size):
    """Build the (size, size, 3) image the video's frames are composited onto.

    'rug' is the shipped carpet texture, any other existing path is used as given, everything
    else is read as a colour. The whole texture is mapped onto the frame rather than tiled: at
    the default camera the frame spans ~1.4 m, which puts the rug's dots ~15 cm apart, i.e. at a
    plausible size for a carpet. Tiling it would shrink them into noise.
    """
    key = value.strip()
    path = RUG_TEXTURE if key.lower() == 'rug' else (key if os.path.isfile(key) else None)
    if path is None:
        return np.full((size, size, 3), parse_color(key), dtype=np.uint8)
    img = Image.open(path).convert('RGB')
    # Centre-crop to a square first, so a non-square background is not distorted.
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def composite(rgba, background):
    """Alpha-composite an RGBA frame onto the background image. mp4 has no alpha channel, so
    without this the transparent pixels would be whatever the codec happens to make of them."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_env(config, start, size):
    actuation_model = MuscleModel if config.get('use_muscle', False) else SpringDamperModel
    proprio = config.get('proprio_params')
    kwargs = dict(
        starting_position=start,
        pbrs=config.get('pbrs', False),
        pbrs_w=config.get('pbrs_w', 100),
        pen_factor=config.get('pen_factor', 0.02),
        pen_metabolic=config.get('pen_metabolic', False),
        nopen=config.get('nopen', False),
        sparse_reward=config.get('sparse_reward', False),
        goal_function=config.get('goal_achievement_function', 'cos'),
        gravity_goal_eps=config.get('gravity_goal_eps', config.get('intrinsic_goal_eps', 0.15)),
        gravity_reference_samples=config.get('gravity_reference_samples',
                                             config.get('intrinsic_reference_samples', 20)),
        goal_tolerance=config.get('goal_tolerance'),
        achieved_goal_in_observation=config.get('achieved_goal_in_observation', False),
        age_physio=config.get('physio_age', 9),
        age_morph=config.get('morph_age', 9),
        floor_softness=config.get('floor_softness'),
        floor_friction=config.get('floor_friction'),
        floor_solimp_width=config.get('floor_solimp_width'),
        freeze_arm=config.get('freeze_arm', False),
        freeze_leg=config.get('freeze_leg', False),
        missing_limb=config.get('missing_limb'),
        missing_limb_mode=config.get('missing_limb_mode', 'cut'),
        ghost_obs=config.get('ghost_obs', 'rest'),
        cos_goal_pool=config.get('cos_goal_pool', 'mean'),
        muscle_action_space=config.get('muscle_action_space', 'unit'),
        ghost_reference_samples=config.get('ghost_reference_samples', 20),
        isr=config.get('isr', False),
        success_at_side_lying=config.get('side_lying', False),
        done_active=not config.get('no_done_active', False),
        touch_params=TOUCH_PARAMS if config.get('touch', False) else None,
        width=size, height=size,
        render_mode='rgb_array',
    )
    if isinstance(proprio, dict):
        kwargs['proprio_params'] = proprio
    episode_steps = config.get('episode_steps') or 500
    return gym.make('MIMoRollOver-v0', actuation_model=actuation_model,
                    max_episode_steps=episode_steps, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help="Run directory holding data.yml and model_*.zip")
    parser.add_argument('--checkpoint', default='last')
    parser.add_argument('--out', required=True, help="Output directory for frame_XXXX.png")
    parser.add_argument('--size', type=int, default=480, help="Render resolution (square)")
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--video', default='episode.mp4',
                        help="mp4 written alongside the frames. A bare name lands in --out, "
                             "a path with a separator is used as given. Default: episode.mp4")
    parser.add_argument('--no_video', action='store_true', help="Write only the PNG sequence.")
    parser.add_argument('--fps', type=float, default=0,
                        help="Video framerate. 0 (the default) means real time, i.e. 1/env.dt, "
                             "so one simulated second is one second of video.")
    parser.add_argument('--video_bg', default='rug',
                        help="Background the video's frames are composited onto: 'rug' (the "
                             "nursery carpet texture shipped with MIMo), a path to any image, "
                             "or a flat colour (white/black/gray/green or 'R,G,B'). "
                             "Default: rug")
    parser.add_argument('--video_distance', type=float, default=0,
                        help="Camera distance for the video's static camera. 0 (the default) "
                             "means 1.5x the tracking camera's, i.e. enough headroom that MIMo "
                             "does not leave the frame while he rolls away from his start.")
    parser.add_argument('--start', action='store_true', help="Only render the starting position.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model_path = pick_checkpoint(args.run, args.checkpoint)
    if model_path is None:
        raise SystemExit(f"No checkpoint found in {args.run}")
    config = load_run_config(model_path)
    algorithm = config.get('algorithm', 'PPO')
    start = 'supine' if 'supine' in os.path.abspath(args.run) else 'prone'

    env = build_env(config, start, args.size)
    model = load_policy(model_path, algorithm, env)

    unw = env.unwrapped
    renderer = mujoco.Renderer(unw.model, height=args.size, width=args.size)
    # The PNG sequence's camera. Re-aimed at MIMo every frame in save_frame(), so he stays centred.
    track_cam = create_top_down_camera(start)
    geom_ids = mimo_geom_ids(unw.model)

    obs, _ = env.reset(seed=args.seed)
    unw.isr = False

    # Real time by default: one simulated second is one second of video.
    fps = args.fps if args.fps > 0 else 1.0 / unw.dt
    video_path = None
    writer = None
    static_cam = None
    background = None
    if not args.no_video:
        try:
            background = make_background(args.video_bg, args.size)
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc))
        # The video's camera, aimed once and then left alone for the whole episode: a roll should
        # read as MIMo moving across a fixed carpet, not as the carpet sliding under a fixed MIMo.
        static_cam = create_top_down_camera(start)
        static_cam.distance = args.video_distance if args.video_distance > 0 else track_cam.distance * 1.5
        static_cam.lookat[0], static_cam.lookat[1] = get_surrounding_rect_center(unw.data)

        video_path = args.video if os.sep in args.video else os.path.join(args.out, args.video)
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (args.size, args.size))
        if not writer.isOpened():
            raise SystemExit(f"OpenCV could not open a writer for {video_path}.")

    frame_idx = 0

    def save_frame():
        nonlocal frame_idx
        track_cam.lookat[0], track_cam.lookat[1] = get_surrounding_rect_center(unw.data)
        rgba = render_rgba(unw.data, renderer, track_cam, geom_ids)
        Image.fromarray(rgba, mode='RGBA').save(os.path.join(args.out, f'frame_{frame_idx:04d}.png'))
        if writer is not None:
            # A second render pass, because the video's camera does not move with MIMo and the
            # tracking frame therefore cannot be reused. Frames are streamed into the writer
            # rather than collected: 500 frames of 480x480 RGB would be ~345 MB held for no reason.
            static_rgba = render_rgba(unw.data, renderer, static_cam, geom_ids)
            writer.write(cv2.cvtColor(composite(static_rgba, background), cv2.COLOR_RGB2BGR))
        frame_idx += 1

    try:
        save_frame()
        done = False
        step = 0

        if not args.start:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                step += 1
                save_frame()
                done = terminated or truncated
    finally:
        if writer is not None:
            writer.release()

    print(f"Wrote {frame_idx} frames ({step} env steps) to '{args.out}'")
    if video_path is not None:
        print(f"Wrote '{video_path}' at {fps:g} fps ({frame_idx / fps:.1f} s), "
              f"static camera on background '{args.video_bg}'")
    env.close()


if __name__ == '__main__':
    main()

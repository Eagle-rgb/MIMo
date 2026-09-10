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
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
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
    x, y = get_surrounding_rect_center(data)
    cam.lookat[0] = x
    cam.lookat[1] = y

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
    cam = create_top_down_camera(start)
    geom_ids = mimo_geom_ids(unw.model)

    obs, _ = env.reset(seed=args.seed)
    unw.isr = False

    frame_idx = 0

    def save_frame():
        nonlocal frame_idx
        rgba = render_rgba(unw.data, renderer, cam, geom_ids)
        Image.fromarray(rgba, mode='RGBA').save(os.path.join(args.out, f'frame_{frame_idx:04d}.png'))
        frame_idx += 1

    save_frame()

    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        save_frame()
        done = terminated or truncated

    print(f"Wrote {frame_idx} frames ({step} env steps) to '{args.out}'")
    env.close()


if __name__ == '__main__':
    main()

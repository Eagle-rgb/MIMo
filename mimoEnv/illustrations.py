""" Training script for the demonstration experiments.

This script allows simple training and testing of RL algorithms in the demo environments with a command line
interface. A selection of RL algorithms from the Stable Baselines3 library can be selected.
Interactive rendering is disabled during training to speed up computation, but enabled during testing, so the behaviour
of the model can be observed directly.

Trained models are saved into the "models/<scenario>" directory, i.e. if you train a reach model and name it
"my_model", it will be saved under "models/reach/my_model".

To train a given algorithm for some number of time steps::

    python illustrations.py --env=reach --train_for=200000 --test_for=1000 --algorithm=PPO --save_model=<model_suffix>

To review a trained model::

    python illustrations.py --env=reach --test_for=1000 --load_model=<your_model_suffix>

The available algorithms are ``PPO, SAC, TD3, DDPG, A2C``.
"""

import csv
import os
import gymnasium as gym
import time
import argparse
import cv2

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoEnv.envs.mimo_env import DEFAULT_PROPRIOCEPTION_PARAMS, PROPRIOCEPTION_PARAMS_ONLY_QPOS
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY

from render.utils import evaluation_img, evaluation_video, create_renderer, create_top_down_camera, render_top_down

from mimoEnv.envs.roll_over_wrapper import MIMoRollOverWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime
import yaml

import numpy as np

from mimoEnv.envs.roll_over import TOUCH_PARAMS as ROLL_OVER_TOUCH_PARAMS
from mimoEnv.envs.roll_over_callback import RollOverCallback, RollOverEvalCallback, \
    GlobalStepCallback
from mimoEnv.envs.morphological_curriculum import make_curriculum_callback
from mimoEnv.envs.isr_callback import ISRCallback
from stable_baselines3.common.callbacks import CallbackList

from mimoEnv.utils import load_model_yaml

from PIL import Image
import mujoco

# Algorithms that keep a replay buffer, and therefore accept --buffer_size / --train_freq /
# --gradient_steps and can carry a HerReplayBuffer. PPO and A2C are on-policy and take none.
OFF_POLICY_ALGORITHMS = ('SAC', 'TD3', 'DDPG')

# Episode horizon of MIMoRollOver-v0, from the TimeLimit set in mimoEnv/__init__.py.
ROLL_OVER_EPISODE_STEPS = 500

from mimoEnv.envs.gaussiannoiseobswrapper import GaussianNoiseObsWrapper

def test(wrapped_env, save_dir, model=None, render_video=False, render_frames=False, render_actuations=False, roll_over_starting_position='prone',
         action_noise='white', action_sigma=0.3, action_seq_len=None, action_noise_seed=0,
         action_beta=1.0, log_obs=None):
    """ Tests the model for one episode.

    Args:
        wrapped_env (MIMoEnv): The wrapped (!) environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        save_dir (str): The directory in which any rendered videos will be saved.
        model:  The stable baselines model object. If ``None`` we take random actions instead. Default ``None``.
        render_video (bool): If ``True``, all episodes during testing will be recorded and saved as videos in
            `save_dir`.
        render_frames (bool): If ``True``, records in total 4 frames during the rollover - including the final image.
        render_actuations (bool): If ``True``, renders on the top right corner a plot of the muscle actuations.
        log_obs (bool | None): If ``True``, writes 'episode_<n>_obs.csv' next to the video: one row
            per step with every 'robot:*' joint angle in degrees, the vestibular sensors, and the
            derived rotation measures. ``None`` (the default) means "follow 'render_video'", so a
            rendered video always comes with the numbers behind it -- reading a posture off a video
            is guesswork otherwise.
    """ 
    obs, _ = wrapped_env.reset()
    images = []
    done=False
    trunc=False
    im_counter = 0
    # Disable isr for testing.
    # 08.08.2026 Guarded against 'model is None'. main() explicitly allows a
    # None model ("we just take random actions"), but this loop dereferenced it
    # unconditionally, so that path raised AttributeError and had in fact never
    # run. Without a model there is no VecEnv, so reach the env directly.
    if model is not None:
        for env in model.get_env().envs:
            env.unwrapped.isr=False
    else:
        wrapped_env.unwrapped.isr=False

    # 08.08.2026 Action noise for the random-action rollout, so that the videos
    # show the same exploration distribution the H1 buffers are collected with
    # (mimoComposer/h1_latent_probe.py). White = the original
    # action_space.sample(); pink = 1/f colored noise (Eberhard et al. 2023).
    #
    # NB pink is clipped into the action range, and at large sigma that clip is
    # not cosmetic: at sigma=1.0 about a third of the samples saturate, which
    # distorts the 1/f spectrum. Sigma is therefore not the effective amplitude
    # -- measure the clipped signal, do not infer it from the flag.
    noise_sampler = None
    if action_noise in ('pink', 'colored'):
        from pink import ColoredActionNoise
        seq_len = action_seq_len or wrapped_env.spec.max_episode_steps or 500
        beta = 1.0 if action_noise == 'pink' else action_beta
        noise_sampler = ColoredActionNoise(beta=beta, sigma=action_sigma,
                                           action_dim=wrapped_env.action_space.shape[0],
                                           seq_len=seq_len,
                                           rng=np.random.default_rng(action_noise_seed))
        noise_sampler.reset()
        print(f"Using colored action noise (beta={beta}, sigma={action_sigma}, seq_len={seq_len})")

    print("Testing model...")

    # 20.08.2026 Per-step observation log written alongside the video. Replaces the commented-out
    # 'proprio_observations' blocks that used to sit here and only ever produced summary .npz
    # files; what is actually needed when watching a rollout is "what was this joint doing at that
    # frame", which needs the per-step values under their joint names.
    if log_obs is None:
        log_obs = render_video
    obs_rows = []
    unw = wrapped_env.unwrapped
    joint_names = [unw.model.joint(i).name for i in range(unw.model.njnt)
                   if unw.model.joint(i).name.startswith('robot:')]
    joint_adr = [unw.model.jnt_qposadr[unw.model.joint(n).id] for n in joint_names]
    joint_labels = [n.replace('robot:', '') for n in joint_names]
    is_roll_over = hasattr(unw, 'get_achieved_goal_cos')

    def observation_row(step, action):
        row = {'step': step}
        row.update({f'{lbl}_deg': float(np.degrees(unw.data.qpos[adr]))
                    for lbl, adr in zip(joint_labels, joint_adr)})
        vest = unw.get_vestibular_obs()
        for i, k in enumerate(('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
            row[f'vestibular_{k}'] = float(vest[i])
        if is_roll_over:
            row['rho'] = float(unw.get_achieved_goal_cos()[0])
            row['hip_deg'] = float(unw.get_achieved_rotation_degrees('hip'))
            row['chest_deg'] = float(unw.get_achieved_rotation_degrees('chest'))
            row['dot_hip'] = float(unw.get_dot_local_x_to_global_z('hip'))
            row['dot_chest'] = float(unw.get_dot_local_x_to_global_z('chest'))
            # Gravity direction expressed in the HIP frame. The head->hip rotation depends only on
            # the joints between them, so the root free joint cancels and this stays something
            # MIMo could in principle sense. At rest its x component equals 'dot_hip'; while he is
            # moving it does not, because an accelerometer reports gravity PLUS self-acceleration.
            site = unw.model.site('vestibular').id
            R_site = unw.data.site_xmat[site].reshape(3, 3)
            R_hip = unw.data.xmat[unw.model.body('hip').id].reshape(3, 3)
            acc_hip = (R_hip.T @ R_site) @ vest[:3]
            for i, k in enumerate('xyz'):
                row[f'hipframe_acc_{k}'] = float(acc_hip[i])
        row['action_sq_sum'] = float(np.square(action).sum()) if action is not None else 0.0
        return row

    def write_obs_csv(counter):
        if not (log_obs and obs_rows):
            return
        path = os.path.join(save_dir, f'episode_{counter}_obs.csv')
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(obs_rows[0].keys()))
            writer.writeheader()
            writer.writerows(obs_rows)
        print(f"Wrote {len(obs_rows)} observation rows to '{path}'")
        last = obs_rows[-1]
        print("Final joint angles of the intrinsic-goal joints (degrees):")
        for lbl in ('head_swivel', 'head_tilt_side', 'head_tilt',
                    'hip_lean1', 'hip_rot1', 'hip_bend1'):
            key = f'{lbl}_deg'
            if key in last:
                jid = unw.model.joint(f'robot:{lbl}').id
                rng = np.degrees(unw.model.jnt_range[jid])
                print(f"    {lbl:<16} {last[key]:+8.2f}   (range {rng[0]:+.0f} .. {rng[1]:+.0f})")

    n_steps = 0
    reached_45_deg=False
    reached_side_lying=False

    renderer = create_renderer(wrapped_env.unwrapped.model)
    cam = create_top_down_camera(roll_over_starting_position)

    def get_frame():
        if render_actuations:
            return evaluation_img(wrapped_env, up='actuations')
        else:
            # 08.08.2026 '.unwrapped': gym.make returns a TimeLimit wrapper, so
            # mujoco_renderer is not reachable on the top-level object.
            return wrapped_env.unwrapped.mujoco_renderer.render(render_mode="rgb_array")
        
    def save_image(name):
        save_name=os.path.join(save_dir, f'{name}.pdf')
        frame = render_top_down(wrapped_env.unwrapped.data, renderer, cam)
        Image.fromarray(frame).save(save_name)
        
    # Render initial frame.
    if render_frames:
        save_image('frame_1')

    if log_obs:
        obs_rows.append(observation_row(0, None))

    while not done and not trunc:
        if model is None:
            if noise_sampler is None:
                action = wrapped_env.action_space.sample()
            else:
                action = np.clip(noise_sampler(),
                                 wrapped_env.action_space.low,
                                 wrapped_env.action_space.high)
        else:
            # 08.08.2026 Was 'determinstic=True' (typo). SB3 2.5.0 rejects the
            # unknown kwarg with TypeError, so --test crashed for every loaded
            # model, not just here. Commit e599f62 ("Changed policy execution
            # to 'deterministic' when not training") shows the intent.
            action, _ = model.predict(obs, deterministic=True)

        obs, _, done, trunc, info = wrapped_env.step(action)
        n_steps += 1

        if log_obs:
            obs_rows.append(observation_row(n_steps, action))
        if render_video:
            images.append(get_frame())

        curr_45_deg_reached = info['45_deg'] == 1.0
        curr_side_lying_reached = info['side_lying'] == 1.0

        if not reached_45_deg and curr_45_deg_reached:
            reached_45_deg = True
            print("Reached 45 deg! Saving image...")
            save_image('frame_2')

        if not reached_side_lying and curr_side_lying_reached:
            reached_side_lying = True
            print("Reached side lying! Saving image...")
            save_image('frame_3')

        if done or trunc:
            time.sleep(1)

            print(f"Roll Over took {n_steps} steps.")

            if render_frames:
                save_image('frame_4')

            write_obs_csv(im_counter)
            obs_rows = []

            obs, _ = wrapped_env.reset()
            if render_video:
                save_name=os.path.join(save_dir, 'episode_{}.mp4'.format(im_counter))
                print("Rendering video as '"+save_name+"'")
                render_height = 720 if render_actuations else 480
                render_width = 480
                evaluation_video(images, save_name=save_name, resolution=((render_width, render_height)))

                images = []
                im_counter += 1

    wrapped_env.reset()

def train(model, train_for, save_every, save_dir, isr, argparse_args, save_intermediate=False,
          eval_callback=None, lr_state=None):
    """ Training function of a model.

    If 'isr' is active, then trains the model for 75% with 'isr' enabled and the remaining last 25%
    with it disabled to make a training that is comparable to method that have 'isr' disabled
    throughout.

    Args:
        model: The stable baselines model object. Must not be ``None``.
        train_for (int): The number of timesteps to train. This will be broken into multiple episodes.
        save_every (int): Number of timesteps where we save a model.
        save_dir (str): The path to save the model.
        isr (bool): Activate Initial State Randomization?
        save_intermediate (bool): Save intermediate model at reaching 50% side lying success rate?
        eval_callback: Optional RollOverEvalCallback. Runs the reported evaluation protocol
            periodically and keeps the best model, since a run's final policy is not reliably
            its best one.
    """ 
    counter = 0
    train_for_total = train_for
    callback_logger = RollOverCallback(save_intermediate=save_intermediate, save_dir=save_dir)
    callback_morph = make_curriculum_callback(argparse_args)

    callbacks = [callback_logger]

    if isr:
        callbacks.append(ISRCallback(train_for_total))

    if callback_morph:
        callbacks.append(callback_morph)

    if eval_callback is not None:
        callbacks.append(eval_callback)

    if lr_state is not None:
        callbacks.append(GlobalStepCallback(lr_state))

    while train_for > 0:
        counter += 1
        
        # How many steps we should take in this training iteration
        train_for_iter = min(train_for, save_every)

        # How many training steps will remain after this training iteration.
        train_for = train_for - train_for_iter

        model.learn(total_timesteps=train_for_iter,
                    reset_num_timesteps=False,
                    callback=callbacks)

        model.save(os.path.join(save_dir, "model_" + str(counter)))

def load_observation_normalization_dict(obs):
    """ Loads the dictionary containing values for obseration normalization.
    Returns 'None' on error. Requires 'obs' as parameter to check for matching
    dimensions. If dimensions missmatch, returns 'None' aswell.
    Else returns tuple mean_dict and std_dict.
    """
    # The location of the normalization data. The files are created on
    # 01.02.2026.
    path = os.path.join('.', 'mimoEnv', 'envs', 'normalization')
    mean_dict = {}
    std_dict = {}

    for key in ["observation", "touch", "vestibular"]:
        if key not in obs:
            continue

        try:
            data = np.load(os.path.join(path, f"obs_stats_{key}.npz"))
        except:
            print(f"Could not load normalization files. Could not find "\
                  f"normalization file for key {key}.")
            return None, None
        
        try:
            mean_dict[key] = data['mean']
            std_dict[key] = data['std']

            # Fix very small values for std to prevent divide by zeros.
            std_dict[key][abs(std_dict[key]) < 1e-6] = 1.0

            # Check dimensions.
            dim_mean = mean_dict[key].shape[0]
            dim_std = std_dict[key].shape[0]

            obs_shape = obs[key].shape[0]

            if obs_shape != dim_mean:
                print(f"Dimension mismatch of mean statistic for observation {key}. "\
                      f"Expected shape: {obs_shape}, Actual shape: {dim_mean}.")
                return None, None
            
            if obs_shape != dim_std:
                print(f"Dimension mismatch of std statistic for observation {key}. "\
                      f"Expected shape: {obs_shape}, Actual shape: {dim_std}.")
                return None, None
        except:
            print(f"Observation normalization: Something went wrong...")
            return None, None

    print("Successfully loaded observation normalization.")
    return mean_dict, std_dict

def parse_intrinsic_goal_joints(joints_string: str):
    """ Parses the --intrinsic_goal_joints list.

    Comma-separated joint names, with or without the 'robot:' prefix (the model always uses it,
    but the note this came from does not). An empty string means "use the default six", which is
    signalled to the environment as None rather than as an empty list -- an empty list would be a
    goal with no joint dimensions at all, which is a different, valid configuration.
    """
    names = [name.strip() for name in joints_string.split(',') if name.strip()]
    if not names:
        return None
    return [name if name.startswith('robot:') else f'robot:{name}' for name in names]


def parse_proprio(proprio_args_string: str):
    """ Parses proprio arguments. Returns None if invalid. Expects
    proprio arguments of the format <arg>|<arg>|...
    'arg' may only be 'position','velocity','torque','limits' and 'actuation'
    and may not contain duplicates! Returns a list of proprioception parameters. """
    separated = proprio_args_string.split('|')

    # Check if each argument is valid.
    # Check for duplicates.
    for i in range(len(separated)):
        item = separated[i]
        if (item in separated[i+1:]):
            raise ValueError("Proprio Argument " + item + " duplicate! Do not specify duplicates!")

    # Check that each entry is a valid proprioception argument.
    for item in separated:
        if item not in ["position", "velocity", "torque", "limits", "actuation"]:
            raise ValueError("Proprio List contains invalid argument " + item)

    return separated


def main():
    """ CLI for the demonstration environments.

    Command line interface that can train and load models for the standup scenario. Possible parameters are:

    - ``--env``: The demonstration environment to use. Must be one of ``reach, standup, selfbody, catch, roll_over``.
    - ``--train_for``: The number of time steps to train. No training takes place if this is 0. Default 0.
    - ``--test``: Whether to test the trained model. If set, tests the trained model for one episode. Use
      flag '--render_video' to render the testing.
    - ``--save_every``: The number of time steps between model saves. This can be larger than the total training time,
      in which case we save once when training completes. Default 100000.
    - ``--algorithm``: The algorithm to train. This argument must be provided if you train. Must be one of
      ``PPO, SAC, TD3, DDPG, A2C, HER``.
    - ``--load_model``: The path to the model to load.
    - ``--save_model``: The directory name where the trained model will be saved. An input of "my_model", will lead to
        the model being saved under "models/<env>/my_model".
    - ``--use_muscles``: This flag switches between actuation models. By default, the spring-damper model is used. If
        this flag is set, the muscle model is used instead.
    - ``--render_video``: If this flag is set, each testing episode is recorded and saved as a video in the same
        directory as the models.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='roll_over',
                        choices=['reach', 'standup', 'selfbody', 'catch', 'roll_over'],
                        help='The demonstration environment to use. Must be one of "reach", "standup", "selfbody", '
                             '"catch", "roll_over"')
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test', action='store_true',
                        help='Test trained policy for one episode.')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', type=str,
                        default='PPO',
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C'],
                        help='RL algorithm from Stable Baselines3. NB "HER" used to be listed '
                             'here but was never dispatched, so it raised RuntimeError. Since '
                             'SB3 1.1 HER is not an algorithm but a replay buffer -- use '
                             '--her together with an off-policy algorithm instead.')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='model', type=str,
                        help='Name of model to save')
    parser.add_argument('--render_video', action='store_true',
                        help='Renders a video for each episode during the test run.')
    parser.add_argument('--random_actions', action='store_true',
                        help='Test with random actions instead of a policy. Needed because '
                             '--algorithm defaults to PPO, so without --load_model the script '
                             'builds a fresh UNTRAINED PPO rather than passing model=None, and '
                             "test()'s random-action branch was unreachable from the CLI.")
    parser.add_argument('--action_beta', default=1.0, type=float,
                        help='Colored-noise exponent: 0 white, 1 pink, 2 red/brownian. Only '
                             'used with --action_noise=colored.')
    parser.add_argument('--action_noise', default='white', choices=['white', 'pink', 'colored'],
                        help='Action sampler for random-action test rollouts (--test without '
                             '--load_model). "white" = action_space.sample(); "pink" = 1/f '
                             'colored noise. Ignored when a model is loaded. Not stored in '
                             'data.yml -- it describes the invocation, not the model.')
    parser.add_argument('--action_sigma', default=0.3, type=float,
                        help='Noise scale for --action_noise=pink. NB uniform[-1,1] has std '
                             '0.577, so 0.3 is weaker than white and 1.0 saturates the clip.')
    parser.add_argument('--action_seq_len', default=None, type=int,
                        help='Pink noise correlation length. Defaults to the environment '
                             'horizon (500 for roll_over).')
    parser.add_argument('--action_noise_seed', default=0, type=int,
                        help='Seed for the pink noise generator. Needed because '
                             'PinkActionNoise without an explicit rng draws from a fresh '
                             'unseeded np.random.default_rng().')
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')
    parser.add_argument('--roll_over_starting_position', required=False,
                        choices=['supine', 'prone', 'alternating'],
                        default='prone',
                        help='Choose the starting position of MIMo in the roll_over environment. Put '
                             'either \'supine\', \'prone\' or \'alternating\'. Default: \'prone\'.')
    parser.add_argument('--goal_achievement_function', required=False, # Previously: --roll_over_goal_function
                        choices=['angle', 'cos', 'intrinsic', 'gravity'],
                        default='cos',
                        help='Choose the function of achieved goal for the roll_over environment. Put '
                             'either \'angle\', \'cos\' or \'intrinsic\'. \'angle\' and \'cos\' are '
                             'scalar rotations read off the root free joint, which MIMo cannot '
                             'sense. \'intrinsic\' is the non-scalar, non-extrinsic posture goal: a '
                             'vector of joint angles plus vestibular acc-z, all of it already in '
                             'the observation, but it does NOT work -- see docs/roll_over.md 3.4. '
                             '\'gravity\' is its successor: the gravity direction in the hip frame, '
                             'integrated from the gyroscope, +1 supine to -1 prone. '
                             'Default: \'cos\'.')
    # --- 'intrinsic' goal function ---------------------------------------------------------
    # 19.08.2026 The old '--intrinsic_goal' sub-mode selector ('all', 'vesti', 'vesti_acc',
    # 'sparse_proprio') was removed: those goals were dicts of raw sensor readings, which cannot
    # be a SB3 goal space and cannot be relabelled by HER, and only 3 of 539 stored runs used
    # them. 'intrinsic' now means the one posture goal, configured by the flags below.
    parser.add_argument('--intrinsic_goal_joints', default='', type=str, required=False,
                        help="Comma-separated joint names making up the intrinsic goal vector, "
                             "with or without the 'robot:' prefix. Empty means the default six "
                             "(head swivel/tilt/tilt_side, hip lean1/rot1/bend1). All must be "
                             "1-DoF hinges.")
    parser.add_argument('--intrinsic_acc_axes', default='x', type=str, required=False,
                        help="Vestibular accelerometer components in the intrinsic goal, as a "
                             "subset of 'xyz'. Default 'x': the accelerometer reports in the head "
                             "site's local frame, whose x axis is the one aligned with world z, so "
                             "x is the only component that separates prone (-9.7) from supine "
                             "(+9.6). Pass '' to drop the accelerometer entirely -- that leaves "
                             "only joint angles, which barely differ between the two postures.")
    parser.add_argument('--intrinsic_acc_w', default=1.0, type=float, required=False,
                        help="Weight of the accelerometer dimension of the intrinsic goal "
                             "relative to the range-normalised joint angles. Default: 1.0.")
    parser.add_argument('--intrinsic_goal_eps', default=0.15, type=float, required=False,
                        help="Success radius of the intrinsic goal: success is "
                             "||achieved - desired|| <= eps. Under --sparse_reward this value is "
                             "the task definition. Default: 0.15.")
    parser.add_argument('--intrinsic_reference_samples', default=20, type=int, required=False,
                        help="Resets averaged into the recorded reference posture. Default: 20.")
    parser.add_argument('--roll_over_model_path_auto', action='store_true',
                        help="""If set, the path of the model for the roll_over environment
is automatically set to the following:
- Folder structure models/roll_over/<date>/<starting_position>
- The name of the model is <date>_<starting_position>_<reward_function>_<--save_model suffix>
'--save_model' is used as a suffix in model names.
An example is '251206_prone_linear_1e6_test'
""")
    parser.add_argument('--render_actuations', action='store_true',
                        help="Render plot of muscle actuations additionally to scene video.")
    parser.add_argument('--log_actuations', action='store_true',
                        help="Create a .csv log file of actuations of all actuators per step of the environment.")
    parser.add_argument('--nopen', action='store_true',
                        help="Disable action penalty in reward function.")
    parser.add_argument('--lr', required=False, default=3e-4, type=float,
                        help="Learning rate. Default 1e-3 for PPO algorithm. Only used for PPO algorithm.")
    parser.add_argument('--buffer_size', default=300_000, type=int,
                        help="Replay buffer size for the off-policy algorithms (SAC/TD3/DDPG). "
                             "Must stay well below the SB3 default of 1e6: one roll_over observation "
                             "is 379 floats across 5 keys and the buffer holds obs AND next_obs, so "
                             "1e6 needs 6.06 GB at float64 on a machine with ~6 GB free. The default "
                             "300k needs 1.82 GB and holds 600 episodes of 500 steps. "
                             "Ignored by PPO/A2C.")
    parser.add_argument('--train_freq', default=1, type=int,
                        help="Off-policy only: env steps between gradient updates. The SB3 default "
                             "of 1 means one gradient step per env step, which is what makes SAC far "
                             "slower per env step than PPO. Raise it to trade sample efficiency for "
                             "wallclock. Ignored by PPO/A2C.")
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help="Off-policy only: gradient steps per update. -1 means 'as many as "
                             "--train_freq'. Ignored by PPO/A2C.")
    parser.add_argument('--learning_starts', default=100, type=int,
                        help="Off-policy only: env steps collected before training begins. SB3's "
                             "SAC default is 100, but HER cannot sample until a full episode is "
                             "in the buffer, so with --her this is raised to above the 500-step "
                             "episode horizon unless you set it higher yourself.")
    parser.add_argument('--her', action='store_true',
                        help="Use Hindsight Experience Replay. Requires an off-policy "
                             "--algorithm (SAC/TD3/DDPG). Works with every "
                             "--goal_achievement_function, including 'intrinsic': its goals are "
                             "flat vectors and its reward is pure, so relabelling is real. Forces "
                             "--achieved_goal_in_observation on, since HER needs that key.")
    parser.add_argument('--n_sampled_goal', default=4, type=int,
                        help="HER: virtual transitions created per real transition.")
    parser.add_argument('--goal_selection_strategy', default='future',
                        choices=['future', 'final', 'episode'],
                        help="HER: where the relabelled goal comes from.")
    parser.add_argument('--sparse_reward', action='store_true',
                        help="Reward 0 on reaching the goal and -1 otherwise, instead of PBRS or "
                             "distance shaping. This is the point of the HER experiments: it "
                             "removes the hand-designed rotation shaping. The action penalty "
                             "still applies unless --nopen.")
    parser.add_argument('--goal_low', default=None, type=float,
                        help="Sample the target rotation uniformly from [--goal_low, "
                             "--goal_high] each episode instead of using a fixed target. HER "
                             "needs goal variation, otherwise the policy never learns to "
                             "condition on the goal. Evaluate with a fixed 0.95 regardless.")
    parser.add_argument('--goal_high', default=None, type=float,
                        help="Upper end of the sampled goal range. Must be given with --goal_low.")
    parser.add_argument('--goal_curriculum', action='store_true',
                        help="Raise the upper end of the sampled goal range along with what has "
                             "recently been achieved, rather than sampling all of [--goal_low, "
                             "--goal_high] from the start. HER only ever relabels onto goals that "
                             "were actually reached, so goals above the current plateau are "
                             "trained almost only on the original -1 transitions; a run stuck at "
                             "rho ~ 0.6 then scores rho_max 0.09 when queried at 0.95, i.e. worse "
                             "than ignoring the goal input altogether. Requires --goal_low/high.")
    parser.add_argument('--goal_curriculum_window', default=50, type=int,
                        help="How many finished episodes the curriculum averages over.")
    parser.add_argument('--goal_curriculum_quantile', default=0.8, type=float,
                        help="Quantile of the recent episode maxima the curriculum tracks. 0.8 "
                             "follows the good episodes rather than the median one.")
    parser.add_argument('--goal_curriculum_margin', default=0.1, type=float,
                        help="How far the sampled goals may reach beyond that quantile. Also the "
                             "width of the initial range, before any episode has finished.")
    parser.add_argument('--lr_schedule', default='constant',
                        choices=['constant', 'linear', 'linear_tail'],
                        help="Decay the learning rate so the endpoint is a frozen policy rather "
                             "than a drifting one. 'linear' decays from step 0 to 0 at "
                             "--train_for. **Measured on ep100, 6 seeds: it does what it is meant "
                             "to do for seeds that learn early (retention 84 %%, best of three "
                             "variants, and zero catastrophic dropouts) but it cannot rescue a "
                             "late starter -- one seed was still at rho 0.076 at 500k, climbed "
                             "afterwards on an already-halved rate and ended at 0.907. Only 3 of "
                             "6 seeds ever reached 100 %%, against 5 and 6 without it.** "
                             "'linear_tail' therefore holds the rate constant until "
                             "--lr_decay_start of the run and only then decays to 0, leaving the "
                             "whole learning phase at full rate (the first 100 %% falls between "
                             "442k and 667k across variants) and freezing only the endpoint.")
    parser.add_argument('--lr_decay_start', default=0.6, type=float,
                        help="Fraction of --train_for at which 'linear_tail' starts decaying. "
                             "0.6 of 1e6 = 600k, just after the observed learning phase. Must be "
                             "in [0, 1).")
    parser.add_argument('--target_entropy', default=None, type=float,
                        help="SAC's entropy target, default -dim(action) = -46 here. The "
                             "automatic coefficient rises whenever the policy becomes more "
                             "deterministic than this, so on a solved task it re-injects noise: "
                             "across 18 seeds the gap between the best and the final checkpoint "
                             "correlates with the rise of 'train/ent_coef' at r = +0.68, and the "
                             "five collapsing seeds ended at 0.0153 against 0.0034 for the "
                             "stable ones. A lower value (e.g. -92) permits convergence while "
                             "keeping the automatic tuning. SAC only.")
    parser.add_argument('--eval_every', default=0, type=int,
                        help="Run a deterministic evaluation every N steps (0 = off) under the "
                             "protocol of eval_rollover.py: ISR off, goal pinned to 0.95, no "
                             "curriculum, episodes not cut short. Logs 'eval/*' and saves "
                             "'model_best.zip' at the peak. Costs a second environment "
                             "(~3.6 GB RSS) and about 4 %% wall clock at the defaults.")
    parser.add_argument('--eval_episodes', default=20, type=int,
                        help="Episodes per evaluation when --eval_every is set.")
    parser.add_argument('--episode_steps', default=None, type=int,
                        help=f"Episode horizon in steps, overriding the "
                             f"{ROLL_OVER_EPISODE_STEPS}-step TimeLimit that "
                             f"mimoEnv/__init__.py registers for MIMoRollOver-v0. Only the "
                             f"roll_over environment reads this. Longer episodes give MIMo more "
                             f"time per attempt but cost proportionally more simulation per "
                             f"episode, and with --her they also lengthen the window the 'future' "
                             f"strategy samples relabelled goals from. Stored in data.yml, so "
                             f"eval_rollover.py evaluates a run at the horizon it was trained on.")
    parser.add_argument('--no_done_active', action='store_true',
                        help="Do not terminate the episode on success; run the full episode. "
                             "Recommended with --her: a relabelled goal is reached mid-episode "
                             "in a trajectory that kept going, so the virtual transition is not "
                             "marked terminal and the critic bootstraps past it. The Fetch envs "
                             "HER was designed on never terminate either, and this module's own "
                             "docstring describes fixed-length episodes.")
    parser.add_argument('--pbrs', action='store_true',
                        help="Use PBRS in roll_over reward shaping.")
    parser.add_argument('--pbrs_w', default=100, type=float,
                        help="Potential difference weighting in PBRS.")
    parser.add_argument('--isr', action='store_true',
                        help="Use Initial State Randomization.")
    parser.add_argument('--obs_norm', action='store_true', default=False,
                        help="Use observation normalization.")
    parser.add_argument('--touch', action='store_true', default=False,
                        help="Use touch observation")
    parser.add_argument('--achieved_goal_in_observation', action='store_true', default=False)
    parser.add_argument('--pen_factor', default=0.02, type=float, required=False,  # Previously, '--pen_fac'
                        help="Penalization factor when action penalization is active.")
    parser.add_argument('--freeze_leg', default=False, action='store_true', required=False,
                        help="Freezes leg.")
    parser.add_argument('--freeze_arm', default=False, action='store_true', required=False,
                        help="Freezes arm.")
    parser.add_argument('--side_lying', default=False, action='store_true', required=False,
                        help="Yields success already at side lying instead of making MIMo " \
                        "do the full rollover.")
    # Frame 1: Start
    # Frame 2: 45° mean rollover
    # Frame 3: Side Lying
    # Frame 4: Roll Over - Final Image
    parser.add_argument('--log_obs', default=None, action=argparse.BooleanOptionalAction,
                        help="Write 'episode_<n>_obs.csv' next to the rendered video: one row per "
                             "step with every 'robot:*' joint angle in degrees, the six vestibular "
                             "sensors, rho / hip_deg / chest_deg, and the gravity vector rotated "
                             "into the hip frame. Defaults to following --render_video, so a video "
                             "always ships with the numbers behind it. Pass --no-log_obs to "
                             "suppress. Describes the invocation, so not stored in data.yml.")
    parser.add_argument('--render_frames', default=False, action='store_true', required=False,
                        help="Renders many frames - including the final image of the episode in testing - "
                        " and saves them as 'frame_{1-5}.png'.")
    parser.add_argument('--morph_age', default=9, required=False, type=int,
                        help="MIMo's morphological (body) age in months. Default: 9.")
    parser.add_argument('--physio_age', default=9, required=False, type=int,
                        help="MIMo's phyisological (actuation) age in months. Default: 9.")
    parser.add_argument('--save_intermediate', action='store_true', help="Save intermediate model at reaching " \
                        "90% side lying success rate.")
    parser.add_argument('--mgc', type=str,
                        choices=['growth', 'inverse', 'stochastic', 'none'],
                        default='none',
                        help="Morphological Growth Curriculum-Strategy: growth=1M->9M, " \
                            "inverse=9M->1M, stochastic=random, none=Baseline")
    parser.add_argument('--mgc_stochastic_interval', type=int,
                        default=20_000,
                        help="Steps between embodiment change in the " \
                        "stochastic mgc (default: 20000)")
    parser.add_argument('--obs_noise', type=float,
                        default=0.0,
                        help="Introduces observation noise. Adds a normal distribution with stddev " \
                        "'obs_noise' and mean 0 on top of the observation.")
    parser.add_argument('--proprio_config', type=str, default="position|velocity|torque|limits|actuation",
                        help="Proprioception config. Configure which, if any, infos are included in the proprioception observation."\
                        "The following arguments are available:\n" \
                        "- position\n" \
                        "- velocity\n" \
                        "- torque\n" \
                        "- limits\n" \
                        "- actuation\n\n" \
                        "Combine arguments using '|'.")
    
    # Parse yaml if we specified '--load_model'.
    args, remaining_argv = parser.parse_known_args()

    if args.load_model:
        yaml_data = load_model_yaml(args.load_model)
        if yaml_data:
            parser.set_defaults(**yaml_data)
    
    args = parser.parse_args()
    load_model = args.load_model
    env_name = args.env
    algorithm = args.algorithm
    save_model = args.save_model        # not in yaml
    save_every = args.save_every        # not in yaml
    train_for = args.train_for          # in yaml as 'num_train', but is not loaded when loading model.
    should_test = args.test             # not in yaml
    render = args.render_video          # not in yaml
    use_muscle = args.use_muscle        # not in yaml
    roll_over_starting_position = args.roll_over_starting_position  # not in yaml
    roll_over_goal_function = args.goal_achievement_function
    roll_over_model_path_auto = args.roll_over_model_path_auto      # not in yaml
    render_actuations = args.render_actuations
    log_actuations = args.log_actuations
    nopen = args.nopen
    learning_rate = args.lr
    pbrs = args.pbrs
    pbrs_w = args.pbrs_w
    isr = args.isr
    observation_normalization = args.obs_norm
    touch = args.touch
    achieved_goal_in_observation=args.achieved_goal_in_observation
    pen_factor = args.pen_factor
    intrinsic_goal_joints = parse_intrinsic_goal_joints(args.intrinsic_goal_joints)
    freeze_arm = args.freeze_arm
    freeze_leg = args.freeze_leg
    side_lying = args.side_lying
    render_frames = args.render_frames
    morph_age = args.morph_age
    physio_age = args.physio_age
    save_intermediate = args.save_intermediate
    use_her = args.her
    sparse_reward = args.sparse_reward
    goal_low = args.goal_low
    goal_high = args.goal_high
    done_active = not args.no_done_active

    # SB3 accepts either a float or a callable of the remaining progress (1.0 -> 0.0). That
    # argument cannot be used here: train() calls learn() once per --save_every segment, and SB3
    # recomputes the progress within each call, so the rate would saw-tooth back up at every
    # checkpoint (measured: segment 1 ran 3e-4 -> 0, segment 2 restarted at 1.3e-4). The schedule
    # therefore reads the model's global step count instead; 'lr_state' is filled in once the
    # model exists.
    lr_state = {'steps': 0, 'total': args.train_for}
    if args.lr_schedule != 'constant':
        if not 0.0 <= args.lr_decay_start < 1.0:
            raise ValueError(f"--lr_decay_start must be in [0, 1), got {args.lr_decay_start}.")
        base_lr = learning_rate
        # 'linear' is 'linear_tail' with the decay starting at step 0.
        decay_start = args.lr_decay_start if args.lr_schedule == 'linear_tail' else 0.0

        def learning_rate(progress_remaining, _base=base_lr, _state=lr_state,
                          _start=decay_start):
            total = _state['total']
            if not total:
                return progress_remaining * _base
            done = _state['steps'] / total
            if done <= _start:
                return _base
            # Linear from the full rate at '_start' to 0 at the end of the run.
            return max(0.0, (1.0 - done) / (1.0 - _start)) * _base

    if args.target_entropy is not None and algorithm != 'SAC':
        raise ValueError(
            f"--target_entropy is a SAC parameter, got --algorithm={algorithm}. TD3 and DDPG "
            f"have no entropy term, and PPO's is a fixed coefficient (--ent_coef upstream).")

    # The horizon actually in force. gym.make's 'max_episode_steps' overrides the TimeLimit from
    # the registration, so everything downstream has to read this rather than the constant.
    episode_steps = args.episode_steps if args.episode_steps is not None \
        else ROLL_OVER_EPISODE_STEPS

    if args.episode_steps is not None:
        if env_name != 'roll_over':
            raise ValueError(
                f"--episode_steps is only implemented for roll_over, got --env={env_name}. The "
                f"other environments keep the horizon from their registration.")
        if args.episode_steps < 1:
            raise ValueError(f"--episode_steps must be at least 1, got {args.episode_steps}.")

    if use_her:
        if algorithm not in OFF_POLICY_ALGORITHMS:
            raise ValueError(
                f"--her needs an off-policy algorithm to attach the replay buffer to, got "
                f"--algorithm={algorithm}. Use one of {', '.join(OFF_POLICY_ALGORITHMS)}.")
        # HerReplayBuffer reads next_obs['achieved_goal'], so the key has to be in the
        # observation. Forcing it here rather than erroring keeps the CLI usable.
        if not achieved_goal_in_observation:
            print("--her set: enabling --achieved_goal_in_observation (required by HER).")
            achieved_goal_in_observation = True
        if done_active:
            print("Warning! --her without --no_done_active: episodes terminate on success, so "
                  "relabelled transitions are not marked terminal and the critic bootstraps "
                  "past the virtual goal.")
        # HerReplayBuffer.sample() raises until at least one episode has finished, because it
        # needs episode boundaries to pick a 'future' goal. The horizon is 'episode_steps',
        # which --episode_steps may have moved away from the registered default.
        if args.learning_starts <= episode_steps:
            args.learning_starts = 2 * episode_steps
            print(f"--her set: raising --learning_starts to {args.learning_starts} "
                  f"(must exceed the {episode_steps}-step episode horizon).")

    if (goal_low is None) != (goal_high is None):
        raise ValueError("Provide both --goal_low and --goal_high, or neither.")

    if args.goal_curriculum and goal_low is None:
        raise ValueError(
            "--goal_curriculum needs a goal range: pass --goal_low and --goal_high. The "
            "curriculum moves the upper end of that range, so with a fixed goal it has nothing "
            "to do.")

    if pbrs and not sparse_reward and not done_active:
        # The PBRS potential jumps to +reward_success at the goal. That is only safe while the
        # goal is terminal: with --no_done_active MIMo can leave the goal region again, and the
        # shaping term then pays pbrs_w * (-reward_success), i.e. about -50000 at the defaults.
        # Measured: potential 500.0 inside the goal, -0.01 just outside, reward -50001.0.
        # It blows the critic up (critic_loss ~ 2.8e7 within 1000 updates).
        raise ValueError(
            "--pbrs with --no_done_active is unsound: the potential is discontinuous at the "
            "goal, so leaving the goal region pays about -pbrs_w * reward_success (~-50000) and "
            "the critic diverges. Use --pbrs with terminating episodes (drop --no_done_active), "
            "or use --sparse_reward, which has no potential to be discontinuous.")

    proprio_params = DEFAULT_PROPRIOCEPTION_PARAMS

    if len(args.proprio_config) > 0:
        proprio_params["components"] = parse_proprio(args.proprio_config)

    if freeze_arm or freeze_leg:
        print("Warning! Some limbs are frozen.")

    actuation_model = MuscleModel if use_muscle else SpringDamperModel

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL
    else:
        raise RuntimeError("Algorithm not defined. Please provide a valid algorithm name.")

    env_names = {"reach": "MIMoReach-v0",
                 "standup": "MIMoStandup-v0",
                 "selfbody": "MIMoSelfBody-v0",
                 "catch": "MIMoCatch-v0",
                 "roll_over": "MIMoRollOver-v0"}

    # We use the same model folder when loading a model. This is for example when we want to extend the
    # training from 1e6 steps to 2e6 steps and so on. In case we do not want that, we specify
    # 'roll_over_model_path_auto' parameter and then save it as a separate model.
    if roll_over_model_path_auto or not load_model:
        if env_name == 'roll_over' and roll_over_model_path_auto:
            date_str_yymmdd = datetime.today().strftime('%y-%m-%d')
            save_model_suffix = save_model
            save_model = date_str_yymmdd +\
                "_" + roll_over_starting_position +\
                "_" + save_model_suffix
            save_dir = os.path.join("models", env_name, date_str_yymmdd, roll_over_starting_position, save_model)
            print("Saving model under '" + save_dir + "'")
        else:
            save_dir = os.path.join("models", env_name, save_model)

        if not os.path.exists(save_dir):
            print("Creating folders for model save path '" + save_dir + "'")
            os.makedirs(save_dir)
    else:
        save_dir = os.path.abspath(os.path.dirname(load_model))
        print(f"Saving to save_dir {save_dir}.")

    # wrapped_env = None
    # Set render size to 480, because babybench used this render size
    # and we copy-pasted the utils from there.
    render_height = 720 if render_actuations else 480

    # 15.12.2025 Added 'done_active=True' to allow environment termination
    # when we reached a goal state.
    if env_name == 'roll_over':
        print(f"Using proprioception parameters: " + ','.join(proprio_params["components"]))
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            # Overrides the TimeLimit from the registration. Passing the registered default back
            # in is a no-op, so this is safe when --episode_steps was not given.
            max_episode_steps=episode_steps,
            starting_position=roll_over_starting_position,
            goal_function=roll_over_goal_function,
            width=480, # always 480 regardless whether we render actuations or not.
            height=render_height,
            nopen=nopen,
            isr=isr,
            pbrs=pbrs,
            render_mode='rgb_array',
            touch_params=ROLL_OVER_TOUCH_PARAMS if touch else None,
            achieved_goal_in_observation=achieved_goal_in_observation,
            proprio_params=proprio_params,
            pbrs_w=pbrs_w,
            pen_factor=pen_factor,
            intrinsic_goal_joints=intrinsic_goal_joints,
            intrinsic_acc_axes=args.intrinsic_acc_axes,
            intrinsic_acc_w=args.intrinsic_acc_w,
            intrinsic_goal_eps=args.intrinsic_goal_eps,
            intrinsic_reference_samples=args.intrinsic_reference_samples,
            freeze_leg=freeze_leg,
            freeze_arm=freeze_arm,
            success_at_side_lying=side_lying,
            sparse_reward=sparse_reward,
            goal_low=goal_low,
            goal_high=goal_high,
            goal_curriculum=args.goal_curriculum,
            goal_curriculum_window=args.goal_curriculum_window,
            goal_curriculum_quantile=args.goal_curriculum_quantile,
            goal_curriculum_margin=args.goal_curriculum_margin,
            done_active=done_active,
            age_physio=physio_age,
            age_morph=morph_age)
        # if log_actuations:
        #     wrapped_env = MIMoRollOverWrapper(env, log_file=os.path.join(save_dir,"actuation_log.csv"))
        # else:
        #     wrapped_env = env
    else:
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            width=480,
            height=render_height)
        
    obs, _ = env.reset()

    # Wrap in noisy observation wrapper (optional).
    # Target keys are all keys except for 'achieved_goal' and 'desired_goal', because we do not want to add noise to those.
    if args.obs_noise > 0.0:
        env = GaussianNoiseObsWrapper(env, noise_std=args.obs_noise,
                                      target_keys=[key for key in obs.keys() if key not in ['achieved_goal', 'desired_goal']])

    if observation_normalization:
        mean_dict, std_dict = load_observation_normalization_dict(obs)
        if mean_dict:
            env.observation_normalization_mean = mean_dict
            env.observation_normalization_std = std_dict

    # load pretrained model or create new one
    # Set learning rate for PPO algorithm.
    if algorithm=='PPO':
        if load_model:
            model = RL.load(load_model, env,
                            tensorboard_log=save_dir,
                            learning_rate=learning_rate,
                            verbose=1)
        else:
            model = RL("MultiInputPolicy", env,
                    tensorboard_log=save_dir,
                    learning_rate=learning_rate,
                    verbose=1)
    elif algorithm in OFF_POLICY_ALGORITHMS:
        # Off-policy algorithms keep a replay buffer, which PPO/A2C do not. Its size must be
        # passed explicitly: SB3 defaults to 1e6 transitions, and with this environment's
        # 379-float Dict observation stored twice (obs + next_obs) that is 6.06 GB at float64 --
        # more than the machine has free, so the default OOMs before training starts.
        replay_buffer_class = None
        replay_buffer_kwargs = None
        if use_her:
            from stable_baselines3 import HerReplayBuffer
            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=args.goal_selection_strategy,
                # Not optional. The roll_over reward splits into a goal-dependent success term
                # and two goal-INDEPENDENT terms (the action penalty, and the previous achieved
                # goal that the PBRS difference needs). Those two ride in the info dict, so if
                # HER does not copy it, the penalty silently drops to zero in every virtual
                # transition and PBRS cannot be recomputed at all.
                copy_info_dict=True,
            )

        if load_model:
            model = RL.load(load_model, env, buffer_size=args.buffer_size)
        else:
            # 18.08.2026 'learning_rate' was missing here: --lr was silently ignored for every
            # off-policy run and SB3's default of 3e-4 applied instead. It happens to equal the
            # default of --lr, so every run so far is unaffected -- but --lr=1e-4 would have
            # trained at 3e-4 while data.yml recorded 1e-4.
            off_policy_kwargs = dict(
                tensorboard_log=save_dir,
                learning_rate=learning_rate,
                buffer_size=args.buffer_size,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                learning_starts=args.learning_starts,
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                verbose=1)
            if args.target_entropy is not None:
                off_policy_kwargs['target_entropy'] = args.target_entropy
            model = RL("MultiInputPolicy", env, **off_policy_kwargs)
    else:
        if load_model:
            model = RL.load(load_model, env)
        else:
            model = RL("MultiInputPolicy", env,
                    tensorboard_log=save_dir,
                    verbose=1)

    # Save model metadata in model.
    yaml_data = {
        # args.lr, not 'learning_rate': with --lr_schedule the latter is a callable, and
        # yaml.dump would write a '!!python/name:' tag that yaml.safe_load then refuses.
        'lr': args.lr,
        'nopen': nopen,
        'pbrs': pbrs,
        'pbrs_w': pbrs_w,
        'goal_achievement_function': roll_over_goal_function,
        'isr': isr,
        'algorithm': algorithm,
        'num_train': train_for,
        'obs_norm': observation_normalization,
        'touch': touch,
        'pen_factor': pen_factor,
        # Intrinsic goal function. All experiment-defining, so all of them round-trip: reloading
        # a model trained with eps=0.15 under a different eps would evaluate a different task.
        'intrinsic_goal_joints': args.intrinsic_goal_joints,
        'intrinsic_acc_axes': args.intrinsic_acc_axes,
        'intrinsic_acc_w': args.intrinsic_acc_w,
        'intrinsic_goal_eps': args.intrinsic_goal_eps,
        'intrinsic_reference_samples': args.intrinsic_reference_samples,
        'freeze_leg': freeze_leg,
        'freeze_arm': freeze_arm,
        'side_lying': side_lying,
        'physio_age': physio_age,
        'morph_age': morph_age,
        'headfree': True,  # this is just a reminder for me that all models going forward can freely move their head.
        'obs_noise': args.obs_noise,
        'proprio_params': proprio_params,
        # Off-policy settings. Stored because they define the run: reloading a model trained with
        # a 300k buffer under a different --buffer_size would evaluate a different experiment.
        'buffer_size': args.buffer_size,
        'train_freq': args.train_freq,
        'gradient_steps': args.gradient_steps,
        'learning_starts': args.learning_starts,
        # HER / goal settings. These define the experiment, so they must round-trip: reloading a
        # sparse-reward model under the default shaped reward would evaluate a different thing.
        'her': use_her,
        'n_sampled_goal': args.n_sampled_goal,
        'goal_selection_strategy': args.goal_selection_strategy,
        'sparse_reward': sparse_reward,
        'goal_low': goal_low,
        'goal_high': goal_high,
        'goal_curriculum': args.goal_curriculum,
        'goal_curriculum_window': args.goal_curriculum_window,
        'goal_curriculum_quantile': args.goal_curriculum_quantile,
        'goal_curriculum_margin': args.goal_curriculum_margin,
        'no_done_active': args.no_done_active,
        # The horizon in force, not args.episode_steps: eval_rollover.py reads this to evaluate
        # a run at the length it was trained on, and 'None' would send it back to the default.
        'episode_steps': episode_steps,
        # Stability knobs. 'lr' above is the base rate; with a schedule it is the value at step 0.
        'lr_schedule': args.lr_schedule,
        'lr_decay_start': args.lr_decay_start if args.lr_schedule == 'linear_tail' else None,
        'target_entropy': args.target_entropy,
        'eval_every': args.eval_every,
        'eval_episodes': args.eval_episodes,
        'achieved_goal_in_observation': achieved_goal_in_observation,
    }
    with open(f'{save_dir}/data.yml', 'w') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False)

    # Second environment for the periodic evaluation, with the reported protocol baked in:
    # ISR off (it inflates rho_max), goal pinned to the milestone rather than sampled, no
    # curriculum (it would move the goal), and episodes never cut short so every episode gets
    # the same number of chances. Costs another ~3.6 GB RSS, hence opt-in.
    eval_callback = None
    if args.eval_every > 0:
        if env_name != 'roll_over':
            raise ValueError(f"--eval_every is only implemented for roll_over, got "
                             f"--env={env_name}.")
        eval_env = gym.make(env_names[env_name], actuation_model=actuation_model,
            max_episode_steps=episode_steps,
            starting_position=roll_over_starting_position,
            goal_function=roll_over_goal_function,
            width=480, height=render_height, nopen=nopen,
            isr=False,
            pbrs=pbrs, render_mode='rgb_array',
            touch_params=ROLL_OVER_TOUCH_PARAMS if touch else None,
            achieved_goal_in_observation=achieved_goal_in_observation,
            proprio_params=proprio_params, pbrs_w=pbrs_w, pen_factor=pen_factor,
            intrinsic_goal_joints=intrinsic_goal_joints,
            intrinsic_acc_axes=args.intrinsic_acc_axes,
            intrinsic_acc_w=args.intrinsic_acc_w,
            intrinsic_goal_eps=args.intrinsic_goal_eps,
            intrinsic_reference_samples=args.intrinsic_reference_samples,
            freeze_leg=freeze_leg, freeze_arm=freeze_arm,
            success_at_side_lying=False,
            sparse_reward=sparse_reward,
            goal_low=0.95, goal_high=0.95,
            goal_curriculum=False,
            done_active=False,
            age_physio=physio_age, age_morph=morph_age).unwrapped
        eval_callback = RollOverEvalCallback(eval_env=eval_env,
                                             eval_every=args.eval_every,
                                             n_episodes=args.eval_episodes,
                                             save_dir=save_dir,
                                             episode_steps=episode_steps)

    if train_for > 0:
        if model is None:
            raise RuntimeError("Model not defined. Please provide an algorithm name.")
        train(model=model,
              save_dir=save_dir,
              train_for=train_for,
              save_every=save_every,
              isr=isr,
              argparse_args=args,
              save_intermediate=save_intermediate,
              eval_callback=eval_callback,
              lr_state=lr_state if args.lr_schedule != 'constant' else None)
        if eval_callback is not None:
            eval_callback.eval_env.close()

    if should_test:
        # Note here we do not check for 'model is None', because we allow it. If in testing the model is
        # 'None', we just take random actions.
        test(env,
             save_dir,
             model=None if args.random_actions else model,
             render_video=render,
             render_frames=render_frames,
             render_actuations=render_actuations,
             roll_over_starting_position=roll_over_starting_position,
             action_noise=args.action_noise,
             action_sigma=args.action_sigma,
             action_seq_len=args.action_seq_len,
             action_noise_seed=args.action_noise_seed,
             action_beta=args.action_beta,
             log_obs=args.log_obs)

    env.close()

if __name__ == '__main__':
    main()

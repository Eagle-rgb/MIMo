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
from mimoEnv.envs.mimo_env import DEFAULT_PROPRIOCEPTION_PARAMS, PROPRIOCEPTION_PARAMS_ONLY_QPOS, DEFAULT_VISION_PARAMS
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
from mimoEnv.envs.roll_over_callback import RollOverCallback, RollOverEvalCallback
from mimoEnv.envs.morphological_curriculum import make_curriculum_callback
from mimoEnv.envs.isr_callback import ISRCallback
from stable_baselines3.common.callbacks import CallbackList

from mimoEnv.utils import load_model_yaml

from PIL import Image
import mujoco

# Algorithms that keep a replay buffer, and therefore accept --buffer_size / --train_freq and
# can carry a HerReplayBuffer. PPO and A2C are on-policy and take none.
OFF_POLICY_ALGORITHMS = ('SAC', 'TD3', 'DDPG')

# Episode horizon of MIMoRollOver-v0, from the TimeLimit set in mimoEnv/__init__.py.
ROLL_OVER_EPISODE_STEPS = 500

from mimoEnv.envs.gaussiannoiseobswrapper import GaussianNoiseObsWrapper

def test(wrapped_env, save_dir, model=None, render_video=False, render_frames=False,
         render_actuations=False, roll_over_starting_position='prone', log_obs=None):
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
        print("Final head and hip joint angles (degrees):")
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
            action = wrapped_env.action_space.sample()
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
          eval_callback=None):
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
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')
    parser.add_argument('--roll_over_starting_position', required=False,
                        choices=['supine', 'prone', 'alternating'],
                        default='prone',
                        help='Choose the starting position of MIMo in the roll_over environment. Put '
                             'either \'supine\', \'prone\' or \'alternating\'. Default: \'prone\'.')
    parser.add_argument('--goal_achievement_function', required=False,  # Previously: --roll_over_goal_function
                        choices=['cos', 'gravity'],
                        default='cos',
                        help="Which quantity the goal is defined on. 'cos' is the scalar "
                             "rotation rho in [0, 1], read off the root free joint, with success "
                             "'rho >= desired'. 'gravity' is the 2-vector of the gravity "
                             "direction's x component in the hip and chest frames (+1 supine, -1 "
                             "prone), reconstructed from the gyroscope and the joint chain "
                             "instead of the root joint, with success a ball of radius "
                             "--gravity_goal_eps around a recorded reference posture. The two "
                             "describe the same roll -- 'gravity' averaged over the two bodies "
                             "IS rho -- but HER sees the vector, not the average, and trains "
                             "from it without --goal_low/--goal_high (14/16 seeds against 3/16). "
                             "'angle' and 'intrinsic' were removed on 26.08.2026; see "
                             "docs/roll_over.md 3.4. Default: 'cos'.")
    # --- 'gravity' goal function -----------------------------------------------------------
    parser.add_argument('--gravity_goal_eps', '--intrinsic_goal_eps',  # Previously: --intrinsic_goal_eps
                        default=0.15, type=float, required=False,
                        help="Success radius of the gravity goal: success is "
                             "||achieved - desired|| <= eps. In the goal's own +-1 units, "
                             "eps = 2*(1 - rho_target) per body, so 0.15 over two bodies is "
                             "about rho 0.925. Ignored by --goal_achievement_function=cos.")
    parser.add_argument('--gravity_reference_samples', '--intrinsic_reference_samples',
                        default=20, type=int, required=False,
                        help="How many ISR-free resets the prone/supine reference goals are "
                             "averaged over. One reset would pin the goal to a single draw of "
                             "the initial joint noise, and MIMo would be scored on reproducing "
                             "that draw. Ignored by --goal_achievement_function=cos.")
    # 25.08.2026 See MIMoRollOverEnv.__init__ for the measurement this comes out of.
    parser.add_argument('--goal_tolerance', default=None, type=float, required=False,
                        help="Turn the scalar success test into a band: success is "
                             "|achieved - desired| <= goal_tolerance instead of "
                             "achieved >= desired, and the fixed full-roll goal becomes 1.0 "
                             "instead of 0.95. The real task is unchanged (rho is capped at 1.0, "
                             "so |rho-1|<=0.05 IS rho>=0.95); what changes is the reward of the "
                             "goals HER relabels onto, which is the point. 0.05 matches the "
                             "0.15 radius --goal_achievement_function=gravity uses over two "
                             "bodies. 'cos' only -- 'gravity' is a point goal already, and "
                             "combining the two raises. Default: unset (threshold).")
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
    parser.add_argument('--her', action='store_true',
                        help="Use Hindsight Experience Replay. Requires an off-policy "
                             "--algorithm (SAC/TD3/DDPG). Forces "
                             "--achieved_goal_in_observation on, since HER needs that key.")
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
    parser.add_argument('--eval_every', default=0, type=int,
                        help="Run a deterministic evaluation every N steps (0 = off) under the "
                             "protocol of eval_rollover.py: ISR off, goal pinned to 0.95, "
                             "episodes not cut short. Logs 'eval/*' and saves "
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
    parser.add_argument("--vision", action="store_true",
                        help="Enable vision observation")
    
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
    goal_function = args.goal_achievement_function
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
    # SB3's SAC default. Not a flag any more: every stored run used it, and under --her it is
    # overridden below anyway because HerReplayBuffer cannot sample before an episode is done.
    learning_starts = 100



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
        if learning_starts <= episode_steps:
            learning_starts = 2 * episode_steps
            print(f"--her set: raising learning_starts to {learning_starts} "
                  f"(must exceed the {episode_steps}-step episode horizon).")

    if (goal_low is None) != (goal_high is None):
        raise ValueError("Provide both --goal_low and --goal_high, or neither.")


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

    if args.proprio_config is not None:
        if len(args.proprio_config) == 0:
            proprio_params = None
        else:
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
        if proprio_params:
            print(f"Using proprioception parameters: " + ','.join(proprio_params["components"]))
        else:
            print(f"Not using proprioception")
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            # Overrides the TimeLimit from the registration. Passing the registered default back
            # in is a no-op, so this is safe when --episode_steps was not given.
            max_episode_steps=episode_steps,
            starting_position=roll_over_starting_position,
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
            goal_function=goal_function,
            gravity_goal_eps=args.gravity_goal_eps,
            gravity_reference_samples=args.gravity_reference_samples,
            goal_tolerance=args.goal_tolerance,
            freeze_leg=freeze_leg,
            freeze_arm=freeze_arm,
            success_at_side_lying=side_lying,
            sparse_reward=sparse_reward,
            goal_low=goal_low,
            goal_high=goal_high,
            done_active=done_active,
            age_physio=physio_age,
            age_morph=morph_age,
            vision_params=DEFAULT_VISION_PARAMS if args.vision else None)
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
                # The HER paper's own defaults, and what every stored run used.
                n_sampled_goal=4,
                goal_selection_strategy='future',
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
                gradient_steps=1,
                learning_starts=learning_starts,
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                verbose=1)
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
        'lr': args.lr,
        'nopen': nopen,
        'pbrs': pbrs,
        'pbrs_w': pbrs_w,
        'isr': isr,
        'algorithm': algorithm,
        'num_train': train_for,
        'obs_norm': observation_normalization,
        'touch': touch,
        'pen_factor': pen_factor,
        # Experiment-defining: the goal function fixes the width of the goal space, so a model
        # reloaded without it cannot even be loaded, let alone evaluated.
        'goal_achievement_function': goal_function,
        'gravity_goal_eps': args.gravity_goal_eps,
        'gravity_reference_samples': args.gravity_reference_samples,
        # Experiment-defining: it changes the success test and the value of the fixed goal, so a
        # model reloaded without it would be evaluated against a different task.
        'goal_tolerance': args.goal_tolerance,
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
        'learning_starts': learning_starts,
        # HER / goal settings. These define the experiment, so they must round-trip: reloading a
        # sparse-reward model under the default shaped reward would evaluate a different thing.
        'her': use_her,
        'sparse_reward': sparse_reward,
        'goal_low': goal_low,
        'goal_high': goal_high,
        'no_done_active': args.no_done_active,
        # The horizon in force, not args.episode_steps: eval_rollover.py reads this to evaluate
        # a run at the length it was trained on, and 'None' would send it back to the default.
        'episode_steps': episode_steps,
        # Stability knobs. 'lr' above is the base rate; with a schedule it is the value at step 0.
        'eval_every': args.eval_every,
        'eval_episodes': args.eval_episodes,
        'achieved_goal_in_observation': achieved_goal_in_observation,
        'vision': args.vision,
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
            width=480, height=render_height, nopen=nopen,
            isr=False,
            pbrs=pbrs, render_mode='rgb_array',
            touch_params=ROLL_OVER_TOUCH_PARAMS if touch else None,
            achieved_goal_in_observation=achieved_goal_in_observation,
            proprio_params=proprio_params, pbrs_w=pbrs_w, pen_factor=pen_factor,
            goal_function=goal_function,
            gravity_goal_eps=args.gravity_goal_eps,
            gravity_reference_samples=args.gravity_reference_samples,
            goal_tolerance=args.goal_tolerance,
            freeze_leg=freeze_leg, freeze_arm=freeze_arm,
            success_at_side_lying=False,
            sparse_reward=sparse_reward,
            # Pin the goal to the full roll. Under --goal_tolerance that value is 1.0, not 0.95:
            # the policy is conditioned on 'desired_goal', so feeding it the number it was never
            # trained on would evaluate an out-of-distribution query.
            goal_low=(1.0 if args.goal_tolerance is not None else 0.95),
            goal_high=(1.0 if args.goal_tolerance is not None else 0.95),
            done_active=False,
            vision_params=DEFAULT_VISION_PARAMS if args.vision else None,
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
              eval_callback=eval_callback)
        if eval_callback is not None:
            eval_callback.eval_env.close()

    if should_test:
        # Note here we do not check for 'model is None', because we allow it. If in testing the model is
        # 'None', we just take random actions.
        test(env,
             save_dir,
             model=model,
             render_video=render,
             render_frames=render_frames,
             render_actuations=render_actuations,
             roll_over_starting_position=roll_over_starting_position,
             log_obs=args.log_obs)

    env.close()

if __name__ == '__main__':
    main()

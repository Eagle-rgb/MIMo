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
from mimoEnv.envs.roll_over_callback import RollOverCallback
from mimoEnv.envs.morphological_curriculum import make_curriculum_callback
from mimoEnv.envs.isr_callback import ISRCallback
from stable_baselines3.common.callbacks import CallbackList

from mimoEnv.utils import load_model_yaml

from PIL import Image
import mujoco

from mimoEnv.envs.gaussiannoiseobswrapper import GaussianNoiseObsWrapper

def test(wrapped_env, save_dir, model=None, render_video=False, render_frames=False, render_actuations=False, roll_over_starting_position='prone',
         action_noise='white', action_sigma=0.3, action_seq_len=None, action_noise_seed=0,
         action_beta=1.0):
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

    #proprio_observations = []
    #vesti_observations = []
    #touch_observations = []

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

        #proprio_observations.append(obs['observation'])
        #vesti_observations.append(obs['vestibular'])
        #touch_observations.append(obs['touch'])
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

            obs, _ = wrapped_env.reset()
            if render_video:
                save_name=os.path.join(save_dir, 'episode_{}.avi'.format(im_counter))
                print("Rendering video as '"+save_name+"'")
                render_height = 720 if render_actuations else 480
                render_width = 480
                evaluation_video(images, save_name=save_name, resolution=((render_width, render_height)))

                images = []
                im_counter += 1

    # Calculate mean and variance of observations.
    #proprio_observations = np.array(proprio_observations)
    #vesti_observations = np.array(vesti_observations)
    #touch_observations = np.array(touch_observations)

    #proprio_mean = np.mean(proprio_observations, axis=0)
    #vesti_mean = np.mean(vesti_observations, axis=0)
    #proprio_std = np.std(proprio_observations, axis=0)
    #vesti_std = np.std(vesti_observations, axis=0)
    #touch_mean = np.mean(touch_observations, axis=0)
    #touch_std = np.std(touch_observations, axis=0)

    #np.savez("obs_stats_proprio.npz", mean=proprio_mean, std=proprio_std)
    #np.savez("obs_stats_vesti.npz", mean=vesti_mean, std=vesti_std)
    #np.savez("obs_stats_touch.npz", mean=touch_mean, std=touch_std)

    wrapped_env.reset()

def train(model, train_for, save_every, save_dir, isr, argparse_args, save_intermediate=False):
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
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
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
                        choices=['angle', 'cos', 'intrinsic'],
                        default='cos',
                        help='Choose the function of achieved goal for the roll_over environment. Put '
                             'either \'angle\', \'cos\' or \'intrinsic\'. Default: \'cos\'.')
    parser.add_argument('--intrinsic_goal', required=False,
                        choices=['all', 'vesti', 'vesti_acc', 'sparse_proprio'],
                        default='all',
                        help='Goal to use for intrinsic goal achievement function.')
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
    parser.add_argument('--proprio_w', default=0.01, type=float, required=False,  # Previously, 'intrinsic_goal_proprio_w'
                        help="Weighting of proprio goal in intrinsic goal for state potential. Default: 0.01.")
    parser.add_argument('--vesti_w', default=1.0, type=float, required=False,  # Previously, 'intrinsic_goal_vesti_w'
                        help="Weighting of vesti goal in intrinsic goal for state potential. Default: 1.0.")
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
    intrinsic_goal = args.intrinsic_goal
    freeze_arm = args.freeze_arm
    freeze_leg = args.freeze_leg
    side_lying = args.side_lying
    render_frames = args.render_frames
    morph_age = args.morph_age
    physio_age = args.physio_age
    save_intermediate = args.save_intermediate

    proprio_params = DEFAULT_PROPRIOCEPTION_PARAMS

    if len(args.proprio_config) > 0:
        proprio_params["components"] = parse_proprio(args.proprio_config)

    # Weightings of different sensors in intrinsic goals. We usually weight vestibular much
    # higher (1.0 compared to 0.01) than proprioception observation.
    intrinsic_goal_proprio_w = args.proprio_w
    intrinsic_goal_vesti_w = args.vesti_w

    # Create a dict of the weights to pass to the roll_over environment. Missing weights like
    # touch are automatically created and defaulted to 1.0.
    intrinsic_goal_w = {
        'observation': intrinsic_goal_proprio_w,
        'vestibular': intrinsic_goal_vesti_w
    }

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
            intrinsic_goal=intrinsic_goal,
            intrinsic_goal_w=intrinsic_goal_w,
            freeze_leg=freeze_leg,
            freeze_arm=freeze_arm,
            success_at_side_lying=side_lying,
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
    else:
        if load_model:
            model = RL.load(load_model, env)
        else:
            model = RL("MultiInputPolicy", env,
                    tensorboard_log=save_dir,
                    verbose=1)
            
    # Save model metadata in model.
    yaml_data = {
        'lr': learning_rate,
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
        'vesti_w': intrinsic_goal_vesti_w,
        'proprio_w': intrinsic_goal_proprio_w,
        'intrinsic_goal': intrinsic_goal,
        'freeze_leg': freeze_leg,
        'freeze_arm': freeze_arm,
        'side_lying': side_lying,
        'physio_age': physio_age,
        'morph_age': morph_age,
        'headfree': True,  # this is just a reminder for me that all models going forward can freely move their head.
        'obs_noise': args.obs_noise,
        'proprio_params': proprio_params
    }
    with open(f'{save_dir}/data.yml', 'w') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False)

    if train_for > 0:
        if model is None:
            raise RuntimeError("Model not defined. Please provide an algorithm name.")
        train(model=model,
              save_dir=save_dir,
              train_for=train_for,
              save_every=save_every,
              isr=isr,
              argparse_args=args,
              save_intermediate=save_intermediate)

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
             action_beta=args.action_beta)

    env.close()

if __name__ == '__main__':
    main()

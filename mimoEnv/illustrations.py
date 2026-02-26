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

from render.utils import evaluation_img, evaluation_video

from mimoEnv.envs.roll_over_wrapper import MIMoRollOverWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime
import yaml

import numpy as np

from mimoEnv.envs.roll_over import TOUCH_PARAMS as ROLL_OVER_TOUCH_PARAMS
from mimoEnv.envs.roll_over_logger import HipChestAngleLogger


def test(wrapped_env, save_dir, model=None, render_video=False, render_actuations=False):
    """ Tests the model for one episode.

    Args:
        wrapped_env (MIMoEnv): The wrapped (!) environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        save_dir (str): The directory in which any rendered videos will be saved.
        model:  The stable baselines model object. If ``None`` we take random actions instead. Default ``None``.
        render_video (bool): If ``True``, all episodes during testing will be recorded and saved as videos in
            `save_dir`.
        render_actuations (bool): If ``True``, renders on the top right corner a plot of the muscle actuations.
    """ 
    obs, _ = wrapped_env.reset()
    images = []
    done=False
    trunc=False
    im_counter = 0
    # Disable isr for testing.
    for env in model.get_env().envs:
        env.unwrapped.isr=False

    print("Testing model...")

    #proprio_observations = []
    #vesti_observations = []
    #touch_observations = []

    while not done and not trunc:
        if model is None:
            print("No model, taking random actions")
            action = wrapped_env.action_space.sample()
        else:
            action, _ = model.predict(obs)

        obs, _, done, trunc, _ = wrapped_env.step(action)
        #proprio_observations.append(obs['observation'])
        #vesti_observations.append(obs['vestibular'])
        #touch_observations.append(obs['touch'])
        if render_video:
            if render_actuations:
                img = evaluation_img(wrapped_env, up='actuations')
            else:
                img = wrapped_env.mujoco_renderer.render(render_mode="rgb_array")
            images.append(img)
        if done or trunc:
            time.sleep(1)
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

def train(model, train_for, save_every, save_dir, isr):
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
    """ 
    counter = 0
    train_for_total = train_for
    while train_for > 0:
        counter += 1

        # How much we need to train going into this training iteration.
        train_for_cpy = train_for

        # How many steps we should take in this training iteration
        train_for_iter = min(train_for_cpy, save_every)

        # How many training steps will remain after this training iteration.
        train_for = train_for - train_for_iter

        # After this episode, we would exceed 75% training. Split up this
        # training run into two: Train until reaching exactly 75% training,
        # turn off isr in the environment and then train the remaining
        # steps we specified with 'train_for_iter'.
        train_for_75_thresh = train_for_total // 4
        if isr and train_for_cpy > train_for_75_thresh and train_for <= train_for_75_thresh:
            train_for_until_reaching_75 = train_for_cpy - train_for_75_thresh
            print("I will reach 75% ISR training threshold after this training iteration. " \
            f"Splitting up this iteration in first {train_for_until_reaching_75} steps and then " \
            "disabling isr.")
            model.learn(total_timesteps=train_for_until_reaching_75,
                        reset_num_timesteps=False,
                        callback=HipChestAngleLogger())

            print("Disabling isr...")

            # model.get_env().isr=False does not seem to work. Probably an issue with all the many
            # wrappers in stable_baselines3. This way however seems to work.
            for env in model.get_env().envs:
                env.unwrapped.isr=False

            # And now train the remaining timesteps.
            if train_for_until_reaching_75 < train_for_iter:
                print(f"Training remaining {train_for_iter-train_for_until_reaching_75} timesteps.")
                model.learn(total_timesteps=train_for_iter-train_for_until_reaching_75,
                            reset_num_timesteps=False,
                            callback=HipChestAngleLogger())

        else:
            model.learn(total_timesteps=train_for_iter,
                        reset_num_timesteps=False,
                        callback=HipChestAngleLogger())

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
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')
    parser.add_argument('--roll_over_starting_position', required=False,
                        choices=['supine', 'prone', 'alternating'],
                        default='prone',
                        help='Choose the starting position of MIMo in the roll_over environment. Put '
                             'either \'supine\', \'prone\' or \'alternating\'. Default: \'prone\'.')
    parser.add_argument('--roll_over_goal_function', required=False,
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
    parser.add_argument('--proprio_only_qpos', action='store_true', default=False,
                        help="Only uses 'qpos' of each joint in proprio observation.")
    parser.add_argument('--pen_fac', default=0.02, type=float, required=False,
                        help="Penalization factor when action penalization is active.")
    parser.add_argument('--intrinsic_goal_proprio_w', default=0.01, type=float, required=False,
                        help="Weighting of proprio goal in intrinsic goal for state potential. Default: 0.01.")
    parser.add_argument('--intrinsic_goal_vesti_w', default=1.0, type=float, required=False,
                        help="Weighting of vesti goal in intrinsic goal for state potential. Default: 1.0.")
    parser.add_argument('--freeze_leg', default=False, action='store_true', required=False,
                        help="Freezes leg.")
    parser.add_argument('--freeze_arm', default=False, action='store_true', required=False,
                        help="Freezes arm.")
    parser.add_argument('--side_lying', default=False, action='store_true', required=False,
                        help="Yields success already at side lying instead of making MIMo " \
                        "do the full rollover.")
    
    args = parser.parse_args()
    env_name = args.env
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    should_test = args.test
    render = args.render_video
    use_muscle = args.use_muscle
    roll_over_starting_position = args.roll_over_starting_position
    roll_over_goal_function = args.roll_over_goal_function
    roll_over_model_path_auto = args.roll_over_model_path_auto
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
    proprio_only_qpos = args.proprio_only_qpos
    pen_factor = args.pen_fac
    intrinsic_goal = args.intrinsic_goal
    freeze_arm = args.freeze_arm
    freeze_leg = args.freeze_leg
    side_lying = args.side_lying

    if freeze_arm or freeze_leg:
        print("Warning! Some limbs are frozen.")

    # Weightings of different sensors in intrinsic goals. We usually weight vestibular much
    # higher (1.0 compared to 0.01) than proprioception observation.
    intrinsic_goal_proprio_w = args.intrinsic_goal_proprio_w
    intrinsic_goal_vesti_w = args.intrinsic_goal_vesti_w

    # Create a dict of the weights to pass to the roll_over environment. Missing weights like
    # touch are automatically created and defaulted to 1.0.
    intrinsic_goal_w = {
        'observation': intrinsic_goal_proprio_w,
        'vestibular': intrinsic_goal_vesti_w
    }

    if proprio_only_qpos:
        print("Warning! Only using qpos in proprioception obseration.")

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

    # wrapped_env = None
    render_height = 720 if render_actuations else 480

    # Set render size to 480, because babybench used this render size
    # and we copy-pasted the utils from there.

    # 15.12.2025 Added 'done_active=True' to allow environment termination
    # when we reached a goal state.
    if env_name == 'roll_over':
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
            proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS if not proprio_only_qpos else PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            pbrs_w=pbrs_w,
            pen_factor=pen_factor,
            intrinsic_goal=intrinsic_goal,
            intrinsic_goal_w=intrinsic_goal_w,
            freeze_leg=freeze_leg,
            freeze_arm=freeze_arm,
            success_at_side_lying=side_lying)
        # if log_actuations:
        #     wrapped_env = MIMoRollOverWrapper(env, log_file=os.path.join(save_dir,"actuation_log.csv"))
        # else:
        #     wrapped_env = env
    else:
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            width=480,
            height=render_height)
        
    obs, _ = env.reset()

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
        'proprio_only_qpos': proprio_only_qpos,
        'obs_norm': observation_normalization,
        'touch': touch,
        'pen_factor': pen_factor,
        'vesti_w': intrinsic_goal_vesti_w,
        'proprio_w': intrinsic_goal_proprio_w,
        'intrinsic_goal': intrinsic_goal,
        'freeze_leg': freeze_leg,
        'freeze_arm': freeze_arm,
        'side_lying': side_lying
    }
    with open(f'{save_dir}/data.yml', 'w') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False)

    if train_for > 0:
        if model is None:
            raise RuntimeError("Model not defined. Please provide an algorithm name.")
        train(model=model, save_dir=save_dir, train_for=train_for, save_every=save_every, isr=isr)

    if should_test:
        # Note here we do not check for 'model is None', because we allow it. If in testing the model is
        # 'None', we just take random actions.
        test(env, save_dir, model=model, render_video=render, render_actuations=render_actuations)

    env.close()

if __name__ == '__main__':
    main()

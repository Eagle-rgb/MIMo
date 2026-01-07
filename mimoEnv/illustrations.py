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
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel

from render.utils import evaluation_img, evaluation_video

from mimoEnv.envs.roll_over_wrapper import MIMoRollOverWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime


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
    im_counter = 0

    print("Testing model...")

    while not done:
        if model is None:
            print("No model, taking random actions")
            action = wrapped_env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, trunc, _ = wrapped_env.step(action)
        if render_video:
            if render_actuations:
                img = evaluation_img(wrapped_env.unwrapped, up='actuations')
            else:
                img = wrapped_env.unwrapped.mujoco_renderer.render(render_mode="rgb_array")
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

    wrapped_env.reset()

def train(model, train_for, save_every, save_dir):
    """ Training function of a model.

    Args:
        model: The stable baselines model object. Must not be ``None``.
        train_for (int): The number of timesteps to train. This will be broken into multiple episodes.
        save_every (int): Number of timesteps where we save a model.
        save_dir (str): The path to save the model.
    """ 
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter, reset_num_timesteps=False)
        model.save(os.path.join(save_dir, "model_" + str(counter)))

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
    parser.add_argument('--roll_over_reward_function', required=False,
                        choices=['winkel', 'linear', 'quad'],
                        default='winkel',
                        help='Choose the reward function for the roll_over environment. Put '
                             'either \'winkel\', \'linear\' or \'quad\'. Default: \'winkel\'.')
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
    roll_over_reward_function = args.roll_over_reward_function
    roll_over_model_path_auto = args.roll_over_model_path_auto
    render_actuations = args.render_actuations
    log_actuations = args.log_actuations

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

    wrapped_env = None
    render_height = 720 if render_actuations else 480

    # Set render size to 480, because babybench used this render size
    # and we copy-pasted the utils from there.

    # 15.12.2025 Added 'done_active=True' to allow environment termination
    # when we reached a goal state.
    if env_name == 'roll_over':
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            starting_position=roll_over_starting_position,
            reward_function=roll_over_reward_function,
            width=480, # always 480 regardless whether we render actuations or not.
            height=render_height)
        if log_actuations:
            wrapped_env = MIMoRollOverWrapper(env, log_file=os.path.join(save_dir,"actuation_log.csv"))
        else:
            wrapped_env = env
    else:
        env = gym.make(env_names[env_name], actuation_model=actuation_model,
            width=480,
            height=render_height)
        wrapped_env = env
    env.reset()

    # load pretrained model or create new one
    if load_model:
        model = RL.load(load_model, wrapped_env)
    else:
        model = RL("MultiInputPolicy", wrapped_env,
                   tensorboard_log=save_dir,
                   verbose=1)

    if train_for > 0:
        if model is None:
            raise RuntimeError("Model not defined. Please provide an algorithm name.")
        train(model=model, save_dir=save_dir, train_for=train_for, save_every=save_every)

    if should_test:
        # Note here we do not check for 'model is None', because we allow it. If in testing the model is
        # 'None', we just take random actions.
        test(wrapped_env, save_dir, model=model, render_video=render, render_actuations=render_actuations)

    wrapped_env.close()

if __name__ == '__main__':
    main()

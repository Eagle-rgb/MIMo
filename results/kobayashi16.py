"""
Docstring for results.kobayashi16

Plot velocities of joints used for measurement in Kobayashi 2016 during one episode.
"""
from mimoEnv.test.collect_observation_util import collect_kobayashi_framelinvel_sensor_data
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', required=True, type=str)
    args = parser.parse_args()
    env = gym.make("MIMoRollOver-v0", actuation_model=SpringDamperModel,
        starting_position='supine',
        width=480, # always 480 regardless whether we render actuations or not.
        height=480,
        render_mode='rgb_array',
        touch_params=None,
        nopen=False,
		pen_factor=0.02,
        goal_function='cos',
        achieved_goal_in_observation=False,
        pbrs=True,
		#proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
        isr=False)
    
    model = RL.load(args.load_model, env)
    obs_df = collect_kobayashi_framelinvel_sensor_data(env, model, n_episodes=1)
    obs_df.plot()
    plt.xlabel("Step")
    plt.ylabel("Velocity (mm/sec)")
    plt.show()


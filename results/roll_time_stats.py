""" This file is used to find out rolling time statistics over many models and many
individual successfull tries. We also """
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY

from collect_observation_util import collect_run_statistics_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data', action='store_true')
    args = parser.parse_args()

    if not args.load_data:
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
            model_path=os.path.join(SCENE_DIRECTORY, f"roll_over_prone_scene_{9}_mo.xml"),
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)
        data = collect_run_statistics_all(env, '26-03-07', 'supine', 'age9')
        data.to_csv('statistics.csv')

    else:
        data = pd.read_csv('statistics.csv', index_col=['Run', 'Episode'])

    successful_df = data[data['Success'] == True]
    time_series = successful_df['Time']
    min_time = np.min(time_series)
    max_time = np.max(time_series)
    avg_time = np.mean(time_series)

    print(f"Min: {min_time} ms")
    print(f"Max: {max_time} ms")
    print(f"Avg: {avg_time} ms")

    print(time_series.idxmax())
    # print(successful_df.xs(8, level='Run'))

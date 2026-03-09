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

    labels = np.logspace(np.log10(30), np.log10(5000), 20)
    counts, _ = np.histogram(time_series.values, bins=labels)
    print(counts)
    counts = counts.astype(np.float64)
    num_models = np.sum(counts)
    counts /= num_models
    counts *= 100.0
    print(counts)

    plt.figure(figsize=(2,2))
    plt.bar(labels[:-1], counts, width=np.diff(labels),
            edgecolor='black', align='edge',
            color='lightgray')
    plt.xscale('log')
    plt.xlim(100, 5000)
    plt.ylabel('Number of models (%)')
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tick_params(axis='x', direction='in', length=6, width=1)
    plt.tick_params(axis='y', direction='in', length=6, width=1)
    plt.tight_layout(pad=1.0)
    #plt.savefig('sp.pdf',
    #            dpi=300,
    #            bbox_inches='tight',
    #            format='pdf')
    plt.show()
    print(time_series.idxmax())
    # print(successful_df.xs(8, level='Run'))

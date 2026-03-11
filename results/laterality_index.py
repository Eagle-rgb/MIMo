""" This file is used to load ~40 individual training runs and lets each
trained model play for 10 episodes. We record the laterality and output
a graph like in kobayashi16 as the laterality index. """
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_observation_util import collect_run_statistics_all

def get_laterality(df):
    # Count number of left rolls and right rolls.
    num_left = 0.0
    num_right = 0.0

    for episode, groupby in df.groupby(['Episode']):
        success = groupby['Success'].values[0]
        if not success: continue
        roll_to_left = groupby['Left_Roll'].values[0]
        if roll_to_left:
            num_left += 1.0
        else:
            num_right += 1.0

    laterality_index = (num_left - num_right) / (num_left + num_right)
    return laterality_index

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
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)
        #data = collect_run_statistics_all(env, '26-03-07', 'supine', 'age9')
        data = collect_run_statistics_all(env, '26-03-11', 'supine', 'random_rot_45')
        data.to_csv('statistics.csv')
    else:
        data = pd.read_csv('statistics.csv', index_col=['Run', 'Episode'])

    data_laterality = []
    for run, groupby in data.groupby('Run'):
        li = get_laterality(groupby)
        data_laterality.append({
            'Run': run,
            'LI': li,
        })
    df_laterality = pd.DataFrame(data_laterality).set_index('Run')
    data = df_laterality

    labels = np.linspace(-1, 1, 11)
    counts, _ = np.histogram(data['LI'].values, bins=labels)
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
    ticks = [-1, -0.5, 0, 0.5, 1]
    tick_labels=[]
    for tick in ticks:
        if np.isclose(tick, -1):
            tick_labels.append("-1\nLeft")
        elif np.isclose(tick, 1):
            tick_labels.append("1\nRight")
        else:
            tick_labels.append(f"{tick:.1f}")
    plt.xticks(ticks, tick_labels)
    plt.ylabel('Number of models (%)')
    ax = plt.gca()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tick_params(axis='x', direction='in', length=6, width=1)
    plt.tick_params(axis='y', direction='in', length=6, width=1)
    plt.tight_layout(pad=1.0)
    plt.savefig('laterality_index_plot.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')

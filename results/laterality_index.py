""" This file is used to load ~40 individual training runs and lets each
trained model play for 10 episodes. We record the laterality and output
a graph like in kobayashi16 as the laterality index. """


""" This file is used to plot one episode of MIMo's actuations. We group
actuators into limbs (IA, CA, IL, CL, TR) just like in Kobayashi '16."""
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

from collect_observation_util import collect_laterality_and_success
from kobayashi16 import is_roll_to_left

def get_laterality(env, model, n_episodes=10):
    """ Lets the (trained) model 'model' play in the environment 'env'
    for 'n_episodes' episodes. In each episode calculates the direction
    of rolling and calculates the laterality index at the end. Returns
    only that exact number.
    """
    df = collect_laterality_and_success(env, model, n_success_episodes=10, n_abort=40)
    if df is None or df.empty:
        return None

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

def get_all_laterality(date, pos, suffix):
    """ Searches for all models matching 'date', starting position 'pos' and suffix 'suffix'.
    Loads all runs of these models and plays 'get_laterality' on them for 10 episodes. Returns
    a pd.DataFrame with index run_ID and value column 'laterality_index'. """
    data = []
    run_ids = []

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

    # Pattern of model folder: <date>_<startingposition>_<suffix>_run_<i>
    pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_([a-z]+)_([a-z0-9_-]+)_run_(\d+)')

    for root, dirs, files in os.walk('.'):
        # root: Current folder on walk
        # dirs: Directories in 'root'
        # files: Files in 'root'
        root_name = os.path.basename(root)
        match = pattern.search(root_name)
        if not match: continue
        _date, haltung, _suffix, run_num = match.groups()

        if _date != date: continue
        if haltung != pos: continue
        if _suffix != suffix: continue

        print(f"Found run {run_num}!")

        model_file = os.path.join(os.path.abspath(root), "model_1.zip")
        model = RL.load(model_file, env)
        laterality_index = get_laterality(env, model, n_episodes=10)
        if laterality_index is None:
            print(f"Run {run_num} did not reach 10 successful episodes. Skipping...")
            continue
        entry = {
            'Run': run_num,
            'laterality_index': laterality_index,
        }
        data.append(entry)
        run_ids.append(run_num)

    df = pd.DataFrame(data).set_index('Run')
    df.to_csv('laterality.csv')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data', action='store_true')
    args = parser.parse_args()

    if not args.load_data:
        data = get_all_laterality('26-03-07', 'supine', 'age9')
    else:
        data = pd.read_csv('laterality.csv', index_col='Run')

    labels = np.linspace(-1, 1, 11)
    counts, _ = np.histogram(data['laterality_index'].values)
    counts = counts.astype(np.float64)
    num_models = np.sum(counts)
    counts /= num_models
    counts *= 100.0

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

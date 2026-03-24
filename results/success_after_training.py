""" This file is used to get the success rate of individual models after training. It loads all
the (18) training runs and tests them for 40 episodes and notes in which the model succeeds and
in which it fails. Saves this data in a pd.DataFrame and saves it on disk as .csv with the name
<date>_<haltung>_<suffix>_test_success_rate.csv """
import argparse
from stable_baselines3 import PPO as RL
import numpy as np
import pandas as pd
import os
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY
from utils import make_env

from collect_observation_util import collect_run_statistics_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', choices=[1,3,6,9], type=int, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--pen_fac', type=float, required=False)
    args = parser.parse_args()
    age = args.age
    env = make_env(age, pen_fac=args.pen_fac)
    data = collect_run_statistics_all(env, date=args.date, pos='supine', suffix=args.suffix, n_episodes=40, n_success_episodes=-1)

    # Drop not-needed columns
    print(data.keys())
    data = data.drop(['Left_Roll', 'Time', 'Time_SideLying'], axis=1)

    entries = []

    for run, run_df in data.groupby(['Run']):
        num_true = run_df['Success'].sum()
        num_total = len(run_df)
        success_rate = num_true / num_total

        entries.append({
            'Run': run[0],
            'Success_Rate': success_rate
        })

    entries_df = pd.DataFrame(entries).set_index(['Run'])
    entries_df.to_csv(f'{args.date}_{'supine'}_{args.suffix}_test_success_rate.csv')

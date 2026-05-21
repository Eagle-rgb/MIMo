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

from mimoEnv.envs.roll_over import AGES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', choices=AGES, type=int, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--pen_fac', type=float, default=0.02, required=False)
    parser.add_argument('--haltung', required=True, choices=['supine', 'prone'])
    parser.add_argument('--age_act', type=int, choices=AGES, required=False,
                        help="The age of actuators used for evaluation.")
    parser.add_argument('--age_body', type=int, choices=AGES, required=False,
                    help="The age of the body used for evaluation.")
    
    args = parser.parse_args()
    age = args.age

    # If either of cross-embodiment actuator age or body age is supplied, both must be
    # supplied. Else it is unclear what age to use for body / actuators.
    if (args.age_act is not None) ^ (args.age_body is not None):
        raise ValueError("Cross-Embodiment Evaluation: Only one age parameter supplied. Please supply both.")
    
    cross_embodiment_evaluation = args.age_act is not None and args.age_act != args.age and args.age_body != args.age

    age_act = args.age_act
    age_body = args.age_body

    if age_act is None:
        age_act = age
    if age_body is None:
        age_body = age

    env = make_env(age_act=age_act, age_body=age_body, pen_fac=args.pen_fac, starting_position=args.haltung)

    data = collect_run_statistics_all(env, date=args.date, pos=args.haltung, suffix=args.suffix, n_episodes=40, n_success_episodes=-1)

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

    if cross_embodiment_evaluation:
        entries_df.to_csv(f'{args.date}_{args.haltung}_{args.suffix}_' +\
                          f'cee_act{args.age_act}_body{args.body_act}_test_success_rate.csv')
    else:
        entries_df.to_csv(f'{args.date}_{args.haltung}_{args.suffix}_test_success_rate.csv')

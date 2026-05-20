""" This file is used to find out rolling time statistics over many models and many
individual successfull tries. We also """
import argparse
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY
import seaborn as sns
from utils import make_env

from collect_observation_util import collect_run_statistics_all
import icdlplot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--haltung', type=str, choices=['supine', 'prone'], required=True)
    parser.add_argument('--age', type=int, choices=[1,3,6,9], required=True)
    parser.add_argument('--pen_fac', type=float, required=False, default=0.02)
    parser.add_argument('--to_npy_run_duration', action='store_true', required=False,
                        help="This option takes an existing **_statistics.csv file in the " \
                        "current folder ('.'), cuts it to only containing the run durations, " \
                        "and converts it into a .npy file. Saves the .npy file in the current folder.")
    parser.add_argument('--duration_until', choices=['lateral', 'full'], required=False,
                        default='lateral', help="This option selects up to which goal we want to measure " \
                        "duration. If 'lateral' is selected (the default), we nonetheless only consider " \
                        "episodes that achieved a full goal, i.e. this option does not affect which " \
                        "samples we choose.")
    parser.add_argument('--transfer_learning', action='store_true', required=True, default=False,
                        help="Test model using different embodiment in a zero-shot setting.")
    parser.add_argument('--transferlearning_age', choices=[1,3,6,9], type=int, required=False)
    args = parser.parse_args()

    if args.transfer_learning:
        output_path_csv = f'{args.date}_{args.haltung}_{args.suffix}_transferlearning_age{args.transferlearning_age}_statistics.csv'
    else:
        output_path_csv = f'{args.date}_{args.haltung}_{args.suffix}_statistics.csv'

    if not args.to_npy_run_duration:
        age = args.age if not args.transfer_learning else args.transferlearning_age
        env = make_env(age=age, starting_position=args.haltung, pen_fac=args.pen_fac)
        data = collect_run_statistics_all(env, args.date, args.haltung, args.suffix)
        data.to_csv(output_path_csv)

    else:
        # Load existing .csv data.
        data = pd.read_csv(output_path_csv, index_col=['Run', 'Episode'])
        # Reduce to only successful episodes.
        successful_df = data[data['Success'] == True]
        # Get durations until goal, i.e. lateral or full roll.
        if args.duration_until == 'lateral':
            durations = successful_df['Time_SideLying']
        else:
            durations = successful_df['Time']

        if args.transfer_learning:
            np.save(f'duration_{args.duration_until}_{args.haltung}_{args.suffix}_transferlearning_age{args.transferlearning_age}.npy')
        else:
            np.save(f'duration_{args.duration_until}_{args.haltung}_{args.suffix}.npy')
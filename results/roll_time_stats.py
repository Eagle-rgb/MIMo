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
import seaborn as sns

from collect_observation_util import collect_run_statistics_all
import icdlplot

def load_successful_age_statistic_df(age):
    data = pd.read_csv(f'statistics_age{age}.csv', index_col=['Run', 'Episode'])
    successful_df = data[data['Success'] == True]
    return successful_df

def print_min_max_avg_median(time_series):
    min_time = np.min(time_series)
    max_time = np.max(time_series)
    avg_time = np.mean(time_series)
    median_time = np.median(time_series)
    print(f"Min: {min_time} ms")
    print(f"Max: {max_time} ms")
    print(f"Avg: {avg_time} ms")
    print(f"Median: {median_time} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--age', choices=[1,3,6,9], type=int)
    parser.add_argument('--load_all_ages', action='store_true')
    parser.add_argument('--side_lying', action='store_true',
                        help="Measure time until sidelying instead of until reaching " \
                        "full roll over.")
    args = parser.parse_args()
    age = args.age
    if not age:
        age = 9
    ages = [1, 3, 6, 9]
    dates = ['26-03-09', '26-03-09', '26-03-10', '26-03-07']
    haltung = 'supine'
    suffixes = ['age1', 'age3', 'age6', 'age9']

    if age not in ages:
        raise ValueError
    
    age_idx = ages.index(age)

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
            age=age,
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)

        data = collect_run_statistics_all(env, dates[age_idx], haltung, suffixes[age_idx])
        data.to_csv(f'statistics_age{age}.csv')

    elif not args.load_all_ages:
        if args.side_lying:
            data = load_successful_age_statistic_df(age)['Time_SideLying']
        else:
            data = load_successful_age_statistic_df(age)['Time']
        print(f"Age {age} ---------------------")
        print_min_max_avg_median(data)

    else:
        df_all = []
        time_col = 'Time' if not args.side_lying else 'Time_SideLying'
        for age in ages:
            df = load_successful_age_statistic_df(age)
            print(f"Age {age}")
            if args.side_lying:
                print_min_max_avg_median(df[time_col])
            else:
                print_min_max_avg_median(df[time_col])
            df['Age'] = age
            df_all.append(df)

        df_all = pd.concat(df_all, ignore_index=True)
        plt.figure(figsize=(2,2))

        ax = sns.violinplot(
            data=df_all,
            x="Age",
            y=time_col,
            color="#d1d1d1",
            inner="quartile",  # shows 25%, 50%, 75% percentiles
            cut=0,              # restricts the violin on true min/max values.
            bw_adjust=0.5       # makes it less "verschwommen"
        )

        sns.stripplot(
            data=df_all,
            x="Age",
            y=time_col,
            color="black",
            size=2,
            alpha=0.3,
            jitter=True
        )

        # Show siegel data in the plot as reference
        siegel_mean = 3600
        siegel_std = 2800
        lower_bound = siegel_mean - siegel_std # 800ms
        upper_bound = siegel_mean

        plt.axhspan(lower_bound, upper_bound, color='red', alpha=0.1, label='Siegel et al.')
        plt.axhline(siegel_mean, color='red', linestyle='--', alpha=0.4, linewidth=1)

        plt.legend(loc='upper right')
        plt.ylabel("Roll Duration [ms]")
        plt.xlabel("Age [months]")

        plt.tight_layout()
        plt.savefig(f'mimo_speed_plot_{'lateral' if args.side_lying else 'full'}.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')

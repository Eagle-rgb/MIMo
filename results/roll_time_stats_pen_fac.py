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

from collect_observation_util import collect_run_statistics_all
import icdlplot

def load_successful_pen_fac_statistic_df(date, suffix):
    filename = f"{date}_supine_{suffix}_statistics.csv"
    data = pd.read_csv(filename, index_col=['Run', 'Episode'])
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
    dates = ['26-03-08', '26-03-07', '26-03-08', '26-03-08', '26-03-08']
    haltung = 'supine'
    suffixes = ['age9_pen1', 'age9', 'age9_pen4', 'age9_pen6', 'age9_pen8']
    pen_facs = [0.01, 0.02, 0.04, 0.06, 0.08]

    df_all = []
    time_col = 'Time_SideLying'

    for i in range(len(pen_facs)):
        df = load_successful_pen_fac_statistic_df(date=dates[i], suffix=suffixes[i])
        print(f"Pen Fac: {pen_facs[i]}")
        print_min_max_avg_median(df[time_col])
        df['Pen_Fac'] = pen_facs[i]
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)
    plt.figure(figsize=(2.5,2.5))

    ax = sns.violinplot(
        data=df_all,
        x="Pen_Fac",
        y=time_col,
        color="#d1d1d1",
        inner="quartile",  # shows 25%, 50%, 75% percentiles
        cut=0,              # restricts the violin on true min/max values.
        bw_adjust=0.5       # makes it less "verschwommen"
    )

    sns.stripplot(
        data=df_all,
        x="Pen_Fac",
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
    upper_bound = siegel_mean + siegel_std
    kobayashi = 1500

    ax.set_yscale('log')
    ax.set_yticks([400, 600, 1000, 1500, 2000, 3600])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.axhspan(lower_bound, upper_bound, color='red', alpha=0.1, label='Siegel et al.')
    plt.axhline(siegel_mean, color='red', linestyle='--', alpha=0.4, linewidth=1)
    plt.axhline(kobayashi, color='green', linestyle='--', alpha=0.4, linewidth=1, label='Kobayashi 2016')

    plt.legend(loc='upper right')
    plt.ylabel("Roll Duration [ms]")
    plt.xlabel("Age [months]")
    

    plt.tight_layout()
    plt.savefig(f'mimo_speed_plot_pen_comparison.pdf',
            dpi=300,
            bbox_inches='tight',
            format='pdf')

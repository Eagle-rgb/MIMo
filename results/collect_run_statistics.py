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
    args = parser.parse_args()

    env = make_env(age=args.age, starting_position=args.haltung, pen_fac=args.pen_fac)
    data = collect_run_statistics_all(env, args.date, args.haltung, args.suffix)
    data.to_csv(f'{args.date}_{args.haltung}_{args.suffix}_statistics.csv')
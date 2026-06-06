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

from mimoEnv.envs.roll_over import AGES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--haltung', type=str, choices=['supine', 'prone'], required=True)
    parser.add_argument('--pen_fac', type=float, required=False, default=0.02)
    parser.add_argument('--duration_until', choices=['lateral', 'full'], required=False,
                        default='lateral', help="This option selects up to which goal we want to measure " \
                        "duration. If 'lateral' is selected (the default), we nonetheless only consider " \
                        "episodes that achieved a full goal, i.e. this option does not affect which " \
                        "samples we choose.")
    parser.add_argument('--age_physio', type=int, choices=AGES, required=False,
                        help="The age of actuators used for evaluation.")
    parser.add_argument('--age_morph', type=int, choices=AGES, required=False,
                    help="The age of the body used for evaluation.")
    args = parser.parse_args()
    age_physio = args.age_physio
    age_morph = args.age_morph
    output_path_csv = f'{args.date}_{args.haltung}_{args.suffix}_cee_act{age_physio}_body{age_morph}_statistics.csv'
    env = make_env(age_physio=age_physio, age_morph=age_morph, starting_position=args.haltung, pen_fac=args.pen_fac)
    data = collect_run_statistics_all(env, args.date, args.haltung, args.suffix)
    data.to_csv(output_path_csv)
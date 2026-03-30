""" This file is used to plot one episode of MIMo's actuations. We group
actuators into limbs (IA, CA, IL, CL, TR) just like in Kobayashi '16."""
import argparse
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from signal_utils import resample_df_to_60hz, smooth_x_butterworth
import seaborn as sns
from utils import make_env

# Need to be prefixed with 'act:left_' or 'act:right_' - depending on
# the roll direction.
ARM_ACTUATORS = ["shoulder_horizontal",
                 "shoulder_abduction",
                 "shoulder_internal",
                 "elbow"]

# Need to be prefixed with 'act:left_' or 'act:right_' - depending on
# the roll direction.
LEG_ACTUATORS = ["hip_flex",
                 "hip_abduction",
                 "hip_rotation",
                 "knee"]

# Need to be prefixed with 'act:'
TORSO_ACTUATORS = ["chest_twist",
                   "chest_lean"]

# Need to be prefixed with 'act:'
HIP_ACTUATORS = ["hip_bend",
                 "hip_twist",
                 "hip_lean"]

QUAD_HAM = ["left_knee", "right_knee"]



def get_actuator_index(env, act):
    """ Returns the index of the actuator 'act' - specified
    using full qualified name in 'env.mimo_actuators' array.
    
    You may then proceed to read out the torque values of this
    actuator from the actuation model at that control_input index."""
    for i in range(len(env.mimo_actuators)):
        act_id = env.mimo_actuators[i]  # id in mjModel.actuator(...)
        act_name = env.model.actuator(act_id).name
        if act_name != act: continue
        return i
    return -1

def collect_actuations(env, model, n_episodes=1):
    """ Lets the (trained) model 'model' play in the environment 'env'
    for one episode. Records the actuations in each step and groups
    them into the limbs above. Writes the average actuation into a
    pandas DataFrame with keys 'IA', 'IL', 'CA', 'CL', 'TR' and 'HP'
    ('HP' is made up - it stands for 'hip'). 
    """
    # Prepare by collecting index of each actuator in 'env.mimo_actuators' so that
    # we have fast access to reading the actuation values of each actuators from the
    # actuation model.
    torso_act_idx = {}
    hip_act_idx = {}
    left_arm_act_idx = {}
    right_arm_act_idx = {}
    left_leg_act_idx = {}
    right_leg_act_idx = {}

    for act_shortname in ARM_ACTUATORS:
        act_fullname = "act:left_" + act_shortname
        left_arm_act_idx[act_fullname] = get_actuator_index(env, act_fullname)
        act_fullname = "act:right_" + act_shortname
        right_arm_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in LEG_ACTUATORS:
        act_fullname = "act:left_" + act_shortname
        left_leg_act_idx[act_fullname] = get_actuator_index(env, act_fullname)
        act_fullname = "act:right_" + act_shortname
        right_leg_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in TORSO_ACTUATORS:
        act_fullname = "act:" + act_shortname
        torso_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in HIP_ACTUATORS:
        act_fullname = "act:" + act_shortname
        hip_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    # Collection of all the indexes above into the groups we will use.
    act_idx = {
        'Torso': torso_act_idx,
        'Hip': hip_act_idx,
        'Right Leg': right_leg_act_idx,
        'Left Leg': left_leg_act_idx,
        'Right Arm': right_arm_act_idx,
        'Left Arm': left_arm_act_idx
    }

    # Time [ms] from onset until reaching this step.
    time_from_onset_ms = []
    # List of dictionaries - ordered by steps in the environment.
    data = []

    def get_actuation_of_group(group_key):
        vals = []
        for _, act_indx in act_idx[group_key].items():
            vals.append(abs(env.actuation_model.control_input[act_indx]))
        return np.mean(np.array(vals))
    
    def collect_actuations(zero=False):
        entry = {}
        for group_key in act_idx.keys():
            mean_act = get_actuation_of_group(group_key) if not zero else 0.0
            entry[group_key] = mean_act
        return entry

    for episode in range(n_episodes):
        print(f"Playing episode {episode+1} from {n_episodes}.")
        obs, _ = env.reset()
        done = False
        data.append(collect_actuations(zero=True))
        time_from_onset_ms.append(env.data.time * 1000.0)  # 'env.data.time' is in sec

        while not done:
            action, _ = model.predict(obs)
            obs, _, truncated, terminated, _ = env.step(action)
            data.append(collect_actuations())
            time_from_onset_ms.append(env.data.time * 1000.0)  # 'env.data.time' is in sec
            
            done = truncated or terminated

            if truncated:
                print("Success!")

            elif terminated:
                print("No Success!")

            if done:
                env.reset()

    # For some reason, we get the first entry in 'time_from_onset_ms' = 15ms > 0ms. This is weird,
    # but it means we do not start at 0. To start at 0ms, we simply subtract this offset from
    # all values.
    time_from_onset_ms = np.array(time_from_onset_ms)
    offset = time_from_onset_ms[0]
    time_from_onset_ms -= offset
    df = pd.DataFrame(data, index=time_from_onset_ms)
    df.index.name = 'Time from Onset [ms]'
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', required=False, type=str)
    args = parser.parse_args()

    raise ValueError("It is not clear what age this file uses.")

    env = make_env(age=9)
    
    model = RL.load(args.load_model, env)
    data = []
    x_max = 0
    n_runs = 10

    for i in range(n_runs):
        df = collect_actuations(env, model)
        df = resample_df_to_60hz(df, original_fs=100, target_fs=60)
        df = df.apply(smooth_x_butterworth)
        df['Run_ID'] = i+1
        print(df.head(3))
        data.append(df)
        x_max = max(x_max, df.index.max())

    # 2. Zusammenführen OHNE ignore_index, aber mit reset_index()
    # Wir wollen, dass der Index zur Spalte wird, damit Seaborn ihn als x-Achse nutzt
    df_final = pd.concat(data).reset_index()
    df_final = df_final.melt(id_vars=['Time from Onset [ms]', 'Run_ID'],
                             var_name='Actuator Group',
                             value_name='Value')

    # 3. Plotten
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df_final, 
        x='Time from Onset [ms]', 
        y='Value',
        hue='Actuator Group',
        errorbar='sd'
    )

    plt.title(f'{n_runs} Runs - Actuation Value Mean per actuator group. 60Hz resampled & butter-filtered.')
    plt.xlabel('Time from Onset [ms]')
    plt.show()

"""
Docstring for results.kobayashi16

Plot velocities of joints used for measurement in Kobayashi 2016 during one episode.
"""
from results.collect_observation_util import collect_kobayashi_displacements_all, find_runs
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from results.signal_utils import resample_df_to_60hz, smooth_x_butterworth
from scipy.interpolate import interp1d
from collections import Counter
from mimoEnv.eval_rollover import (build_env, load_run_config, ROLL_THRESHOLD,
                                   DEFAULT_EPISODE_STEPS)

SUPPORT_PATTERNS = [
    'Two Stationary',
    'Stationary IA',
    'Stationary IA - IL moving backward',
    'Stationary IL',
    'Stationary IL - IA moving backward',
    'No stationary'
]

MOVEMENT_PATTERNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'R'
]

PATTERN_COLORS = [
    "tab:blue", "tab:orange", "tab:yellow", "tab:red", "tab:blue", "tab:brown", "tab:purple"
]

def is_roll_to_left(data):
    """ Returns 'True' if the data shows a roll to the left side. Else,
    it returns 'False'. This is determined on the last displacement value
    of the torso. If it is negative, we have a roll over the right and else
    it is a roll over the left. """
    return data['TR'].values[-1] >= 0

def crop_until_milestone(data: pd.DataFrame, milestone="Side_Lying"):
    """ Crops given DataFrame 'data' so that it goes only up until reaching the
    given milestone. The milestone is presented as a string and is either
    'Side_Lying' or '45_Deg'. The DataFrame has columns 'Side_Lying' and
    '45_Deg' and we filter so that we only have entries with those being 'False'.

    Also removes milestone columns from data.
    """
    if milestone is None:
        return data.drop(columns=['Side_Lying', '45_Deg'])

    if milestone not in ["Side_Lying", "45_Deg"]:
        raise ValueError()
    
    df = data[data[milestone] == False]
    return df.drop(columns=['Side_Lying', '45_Deg'])

def relabel_right_left_limbs_in_rolling_direction(data):
    """ Relabel wrists and ankles to 'ipsilateral' or 'contralateral' arm/leg, i.e.
    IA, IL, CA and CL based on the direction of the rollover.
    We expect 'data' to contain them labeled as 'Right Wrist', 'Left Wrist',
    'Right Ankle' and 'Left Ankle'.
    This operation does not happen in-place. A new dataframe is returned. """
    if is_roll_to_left(data):
        return data.rename(columns={
            'Left Ankle': 'IL',
            'Right Ankle': 'CL',
            'Left Wrist': 'IA',
            'Right Wrist': 'CA'})
    
    # Roll over the right.
    return data.rename(columns={
        'Left Ankle': 'CL',
        'Right Ankle': 'IL',
        'Left Wrist': 'CA',
        'Right Wrist': 'IA'})

def reorient_rollover(data):
    """ Reorients the rollover so that it is always a
    rollover over the left side so that y displacements are positive.
    
    Guesses the direction of rollover by checking final torso relative
    displacement. If it is negative, this was a right rollover (bad!!).
    """
    if is_roll_to_left(data):
        # good!
        return
    data.iloc[:, :] *= -1

def fit_normalized_to_sigmoid(data):
    """ Normalizes 'data' and fits normalized data to a sigmoid curve
    using logistic regression. Returns the fitted curve evaluated at the same
    time steps as 'data'. For this, make sure 'data' is supplied as-is from the
    pandas dataframe, i.e. including time steps. Returns it as pd series.
    Also returns tuple (beta_0, beta_1) of the resulting beta distribution
    parameters.
    """
    # Get normalization statistics, i.e. max and min.
    max_value = data.max()
    min_value = data.min()

    # Normalize the data.
    data = data.apply(lambda x: (x - min_value) / (max_value - min_value))
    y = data.values
    x = data.index.values

    # Fit to sigmoid using logit (gemini says that we can only use logistic
    # regression for class labeled data, not for linear data)
    # 1. Clip data so that the values lie in (0,1) - we do not want log(0)
    # or log(inf).
    vals_clipped = np.clip(y, 1e-5, 1-1e-5)
    
    # 2. Logit transformation.
    vals_logit = np.log(vals_clipped  / (1-vals_clipped))

    # 3. Linear Regression. sklearn wants a 2D-Array [[x1], [x2], ..]
    model = LinearRegression()
    x_reshaped = x.reshape(-1, 1)
    model.fit(x_reshaped, vals_logit)

    # 4. Get parameters.
    beta_1 = model.coef_[0]
    beta_0 = model.intercept_

    # 5. Reconstruction
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * x)))
    vals_fitted = sigmoid(x)

    # 6. Unnormalize data
    vals_fitted = (max_value - min_value) * vals_fitted + min_value

    data[:] = vals_fitted
    return data, (beta_0, beta_1), (min_value, max_value)

def calculate_velocities(displacement_series):
    """ Calculates the velocity by differentiation of the
    given series. Returns a np.array of the same size
    as 'displacement_series'.
    Unit of speed is mm/sec
    """
    velocities = np.zeros(len(displacement_series))
    velocities[0] = 0  # we always start with 0 velocity.
    for i in range(1, len(displacement_series)):
        y_1 = displacement_series.values[i-1]
        y_2 = displacement_series.values[i]
        x_1 = displacement_series.index[i-1]
        x_2 = displacement_series.index[i]
        timestep = x_2 - x_1
        # timestep is in [ms], but we want mm/sec, so divide by 1000
        timestep /= 1000.0
        velocities[i] = (y_2 - y_1) / timestep

    return velocities

def calculate_velocities_df(df):
    df_velocities = df.copy()
    for key in df_velocities.keys():
        displacement_series = df_velocities[key]
        velocities = calculate_velocities(displacement_series)
        df_velocities[key] = velocities

    return df_velocities

def normalize_velocities_to_torso(df: pd.DataFrame):
    """ Manipulates 'df' in the following way: For each body part key
    except for 'TR' it calculates the relative speed to the torso speed in
    %. """
    torso_velocity_series: pd.Series = df['TR']
    for key in df.keys().difference(['TR']):
        df[key] = df[key].div(torso_velocity_series, fill_value=0) * 100.0

def calculate_average_velocity(velocities_df, a, b):
    """ Calculates the average velocity for each limb (key) in 'velocities_df'
    in the interval [a, b]. """
    velocity_mean_df = []

    for key in velocities_df:
        velocities = velocities_df[key].values
        steps = velocities_df[key].index.values
        f_interp = interp1d(steps, velocities, kind='linear')
        x = np.linspace(a, b, num=10)
        y = f_interp(x)

        mean = np.trapezoid(y, x) / (b - a)
        velocity_mean_df.append({
            'key': key,
            'mean': mean
        })

    return pd.DataFrame(velocity_mean_df)

def classify_stationary_limbs_kobayashi(velocity_mean_df: pd.DataFrame, thresh_mm_sec: float):
    return velocity_mean_df[abs(velocity_mean_df['mean']) <= thresh_mm_sec]

def classify_stationary_limbs_siegel(normalized_to_torso_velocity_ipsilateral_df: pd.DataFrame):
    """ Classifies ipsilateral limbs as in Siegel 2024 "How do babies roll? Identifying the
    coordinated movements of infant rolling through video compared to laboratory techniques."
    
    A moving limb is a limb with a normalized velocity >= 125% of the torso velocity. It is not
    fully clear what Siegel means with "velocity". Is it the mean velocity? In this function we
    assume it is.
    """
    means = normalized_to_torso_velocity_ipsilateral_df.mean()
    mask = means >= 125.0
    return normalized_to_torso_velocity_ipsilateral_df.loc[:, mask]

def calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms):
    """ Returns the (displacement) velocity of the sigmoid defined by
    parameters 'beta_0' and 'beta_1' at its maximum.
    Derivative of the sigmoid P(x)=.. is simply P'(x)=beta_1*P(x)*(1-P(x)).
    Since we consider maximum displacement speed, P(x)=0.5."""
    velocity_mm_msec = beta_1 * 0.5**2.0
    return velocity_mm_msec * 1000.0

def get_time_and_velocity_maximum_sigmoid_velocity(df_displacement, key):
    """ Returns T_{key} and V_{key} and reconstruction. """
    reconstructed_values, (beta_0, beta_1), _ = fit_normalized_to_sigmoid(df_displacement[key])
    T = -beta_0 / beta_1
    V = calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms=10.0)
    return T, V, reconstructed_values

def get_timing_moving_limb(T_TR, T_Limb, Th):
    """ Returns timing of limb (leading, synchronous, following) relative
    to torso maximum displacement timestamp 'T_TR'. Returns it as a string.
    Specify 'Th' as the same as the range to classify stationary limbs. """
    Delta = T_Limb - T_TR
    if Delta < -Th:
        return 'leading'
    elif Delta > Th:
        return 'following'
    else:
        return 'synchronous'

def analysis_kobayashi(df_60hz_butter: pd.DataFrame, thresh=100.0, T_H=250.0):
    """ Performs Kobayashi 2016 analysis. This function does:
    1. Normalizes torso displacement to [0,1] and fits it to a sigmoid.
    2. Verifies "good" fit by checking r2 score being at least 0.6. If not, this function
        returns 'None'.
    3. Calculates T_TR, T_CA and T_CL and their speeds V_TR, V_CA and V_CL for torso and
        contralateral limbs as times of maximum displacement velocity of the fitted sigmoid.
    4. Kobayashi uses a 250ms interval left / right of T_TR in which he classifies moving / stationary
        limbs. MIMo is much faster in rolling than normal infants. In the plots in Kobayashi's study,
        he has an example with ~1500ms supine -> lateral time. We take that as a reference. Instead of
        using 250ms, we calculate a fraction proportional to MIMo's supine -> lateral time in this run.
        We compute the mean velocity of ipsilateral limbs in that timeframe and compare to a threshold -
        in Kobayashi this is 100mm/sec, but we use 400mm/sec. Ipsilateral limbs with less mean velocity
        in the range are classified as stationary and else as moving.
    
    """
    # Get the torso speeds, normalize them to [0, 1] and fit to a sigmoid using log. regression.
    torso = df_60hz_butter['TR']

    T_TR, V_TR, torso_sigmoid = get_time_and_velocity_maximum_sigmoid_velocity(df_60hz_butter, 'TR')

    # Verify that R-squared value is > 0.6
    r2 = r2_score(torso, torso_sigmoid)
    if r2 < 0.6:
        print(f"Too low R-squared value: {r2}!")
        return None
    
    # Calculate T_CA and T_CL.
    T_CA, V_CA, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_60hz_butter, 'CA')
    T_CL, V_CL, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_60hz_butter, 'CL')

    # Pattern Classification.
    # Copy so we can use it later to fit nonstationary ipsilateral limb displacement to sigmoid.
    df_velocities = calculate_velocities_df(df_60hz_butter)
    #normalize_velocities_to_torso(df_episode)

    #df_episode.plot()
    #plt.ylim(-2000, 2000)
    #plt.ylabel('% of Torso')
    #plt.show()
    #raise ValueError

    # DataFrame containing only the ipsilateral limbs.
    df_ipsilateral = df_velocities[['IA', 'IL']]

    #df_episode.plot()
    #plt.show()

    T_TR_Left = T_TR - T_H
    T_TR_Right = T_TR + T_H
    duration_mimo = df_60hz_butter.index.max() # ms
    if T_TR_Left < 0: T_TR_Left = 0
    if T_TR_Right > duration_mimo: T_TR_Right = duration_mimo
    df_mean = calculate_average_velocity(df_ipsilateral, T_TR_Left, T_TR_Right)
    df_stationary = classify_stationary_limbs_kobayashi(df_mean, thresh_mm_sec=thresh)

    stationary_ia = not df_stationary[df_stationary['key'] == 'IA'].empty
    stationary_il = not df_stationary[df_stationary['key'] == 'IL'].empty

    # Calculate T_IA, T_IL for nonstationary ipsilateral limbs.
    T_IA = 0
    T_IL = 0
    V_IA = 0
    V_IL = 0
    direction_il = 'stationary'
    direction_ia = 'stationary'
    if not stationary_ia:
        T_IA, V_IA, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_60hz_butter, 'IA')
        direction_ia = 'forward' if V_IA > 0 else 'backward'
    if not stationary_il:
        T_IL, V_IL, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_60hz_butter, 'IL')
        direction_il = 'forward' if V_IL > 0 else 'backward'

    # Calculate timing of moving limbs (leading, synchronous, following)
    Timing_CA = get_timing_moving_limb(T_TR, T_CA, Th=T_H)
    Timing_CL = get_timing_moving_limb(T_TR, T_CL, Th=T_H)
    Timing_IL = 'stationary'
    Timing_IA = 'stationary'
    if not stationary_il:
        Timing_IL = get_timing_moving_limb(T_TR, T_IL, Th=T_H)
    if not stationary_ia:
        Timing_IA = get_timing_moving_limb(T_TR, T_IA, Th=T_H)

    entry_stats = {
        'Stationary_IA': stationary_ia,
        'Stationary_IL': stationary_il,
        'T_TR': T_TR,
        'T_CA': T_CA,
        'T_CL': T_CL,
        'T_IA': T_IA,
        'T_IL': T_IL,
        'V_TR': V_TR,
        'V_CA': V_CA,
        'V_CL': V_CL,
        'V_IA': V_IA,
        'V_IL': V_IL,
        'Direction_IA': direction_ia,
        'Direction_IL': direction_il,
        'Timing_IA': Timing_IA,
        'Timing_IL': Timing_IL,
        'Timing_CA': Timing_CA,
        'Timing_CL': Timing_CL,
        'T_H': T_H,
    }

    return entry_stats

def classify_support(df):
    ia_stat = df['Stationary_IA']
    il_stat = df['Stationary_IL']
    ia_dir = df['Direction_IA']
    il_dir = df['Direction_IL']
    
    if ia_stat and il_stat:
        return SUPPORT_PATTERNS[0]
    elif ia_stat and not il_stat:
        return SUPPORT_PATTERNS[2] if il_dir == 'backward' else SUPPORT_PATTERNS[1]
    elif il_stat and not ia_stat:
        return SUPPORT_PATTERNS[4] if ia_dir == 'backward' else SUPPORT_PATTERNS[3]
    else:
        return SUPPORT_PATTERNS[5]
    
def classify_movement(df):
    ia_s, il_s = df['Stationary_IA'], df['Stationary_IL']
    t_ia, t_il, t_ca, t_cl = df['Timing_IA'], df['Timing_IL'], df['Timing_CA'], df['Timing_CL']
    # Logik wie oben besprochen...
    if ia_s and il_s and t_ca == 'synchronous' and t_cl == 'synchronous': return MOVEMENT_PATTERNS[0]
    if ia_s and il_s and t_ca == 'synchronous' and t_cl == 'following': return MOVEMENT_PATTERNS[1]
    if ia_s and not il_s and t_il == 'synchronous' and t_ca == 'synchronous' and t_cl == 'synchronous': return MOVEMENT_PATTERNS[2]
    if ia_s and not il_s and t_il == 'synchronous' and t_ca == 'synchronous' and t_cl == 'following': return MOVEMENT_PATTERNS[3]
    if il_s and not ia_s and t_ia == 'synchronous' and t_ca == 'synchronous' and t_cl == 'following': return MOVEMENT_PATTERNS[4]
    # Movement pattern F: IA and IL not stationary and all four limbs move synchronously.
    if not ia_s and not il_s \
        and t_ca == 'synchronous' and t_cl == 'synchronous' \
        and t_ia == 'synchronous' and t_il == 'synchronous': \
            return MOVEMENT_PATTERNS[5]
    return MOVEMENT_PATTERNS[6]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_data', required=True, type=str,
                        help="Load the specified .csv file and perform Kobayashi classification on it.")
    parser.add_argument('--thresh', type=int, default=300,
                        help="Speed threshold to use in Kobayashi Analysis to classify stationary limbs. Default: 300mm/sec.")
    parser.add_argument('--range', type=int, default=83,
                        help="Range around T_TR to use for classifying stationary limbs. Default 83ms.")
    parser.add_argument('--save_full_report', type=str, required=False,
                        help="Saves the full pattern report under this .csv name.")
    parser.add_argument('--checkpoint', type=str, required=False, default='model_best.zip',
                        help="Checkpoint file to load from each run directory. Default "
                             "'model_best.zip', the peak under the reported evaluation protocol "
                             "(RollOverEvalCallback) -- the last numbered checkpoint is not "
                             "reliably the best one for off-policy runs.")
    parser.add_argument('--goal', type=float, required=False, default=ROLL_THRESHOLD,
                        help="desired_goal fed to the policy. Goal-conditioned runs (HER) are "
                             "conditioned on it, so it must be the full roll. Default 0.95.")
    parser.add_argument('--episode_steps', type=int, required=False, default=None,
                        help="Episode horizon. Defaults to the one recorded in the run's "
                             "data.yml, else 500.")
    parser.add_argument('--seed0', type=int, required=False, default=1000,
                        help="Episode i of every run is seeded with 'seed0 + i', so all runs "
                             "face identical start states. Pass -1 for unseeded resets.")
    parser.add_argument('--save_velocities_npy', type=str, required=False,
                        help="Saves normalized mean and std velocities of TR, CA and CL to .npy.")
    
    args = parser.parse_args()
    df = pd.read_csv(args.load_data, index_col=['Run', 'Episode', 'Time'])
    
    groupby_run = df.groupby('Run')

    ax_displacement = None

    # Statistics of the runs. Includes:
    # V_TR, V_CA, V_CL: Velocities of maximum displacement after normalized & fitted to sigmoid.
    # Stationary_IA: True / False if IA stationary
    # Stationary_IL: True / False if IL stationary
    stats_list = []

    for run, df_run in groupby_run:
        for episode, df_episode in df_run.groupby(['Episode']):
            episode = episode[0]
            # Crop DataFrame to the section relevant for our analysis. This until side lying.
            milestone = None
            df_episode = crop_until_milestone(df_episode, milestone='Side_Lying')

            if len(df_episode) <= 26:
                print(f"Skipping run {run}, episode {episode}, because it is too short!")
                continue

            df_episode = relabel_right_left_limbs_in_rolling_direction(df_episode)
            reorient_rollover(df_episode)

            df_episode = resample_df_to_60hz(df_episode, target_fs=100.0)
            df_episode = df_episode.apply(smooth_x_butterworth)
            entry_stats = analysis_kobayashi(df_episode, thresh=args.thresh, T_H=args.range)

            if entry_stats is None:
                continue

            entry_stats['Run'] = run
            entry_stats['Episode'] = episode

            stats_list.append(entry_stats)

    df_stats = pd.DataFrame(stats_list)
    df_stats['Support_Pattern'] = df_stats.apply(classify_support, axis=1)
    df_stats['Movement_Pattern'] = df_stats.apply(classify_movement, axis=1)

    V_TR_mean = df_stats['V_TR'].mean()
    V_CA_mean = df_stats['V_CA'].mean()
    V_CL_mean = df_stats['V_CL'].mean()
    V_TR_std = df_stats['V_TR'].std()
    V_CA_std = df_stats['V_CA'].std()
    V_CL_std = df_stats['V_CL'].std()

    print(f"V_TR_mean: {V_TR_mean}")
    print(f"V_CA_mean: {V_CA_mean}")
    print(f"V_CL_mean: {V_CL_mean}")
    print(f"V_TR_std: {V_TR_std}")
    print(f"V_CA_std: {V_CA_std}")
    print(f"V_CL_std: {V_CL_std}")

    print(f"T_H mean: {df_stats['T_H'].mean()}")

    if args.save_velocities_npy is not None:
        means_stds = np.array([V_TR_mean, V_CA_mean, V_CL_mean, V_TR_std, V_CA_std, V_CL_std])
        np.save(args.save_velocities_npy, means_stds)

    df_patterns = df_stats[['Run', 'Episode', 'Support_Pattern', 'Movement_Pattern']]

    if args.save_full_report is not None:
        df_patterns.to_csv(args.save_full_report)
"""
Docstring for results.kobayashi16

Plot velocities of joints used for measurement in Kobayashi 2016 during one episode.
"""
from collect_observation_util import collect_kobayashi_displacements_all
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from signal_utils import resample_df_to_60hz, smooth_x_butterworth
from scipy.interpolate import interp1d
from collections import Counter
import datetime
from utils import make_env

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

def get_kobayashi_left_right_T_TR_interval(T_TR, df_displacement):
    """ Reads the total duration of the MIMo supine -> prone roll from the maximum index
    in 'df_displacement'. Returns a tuple T_TR_Left, T_TR_Right representing the 250ms
    left/right interval around T_TR that Kobayashi 2016 used. Our MIMo rolls much faster than
    real infants. 250ms is way too long. We get a much smaller interval by comparing our
    duration to the duration Siegel has (3600ms) and calculate a proportionately smaller '250ms'
    time range. """
    kobayashi_time_range=250.0 # ms
    siegel_duration=3600.0 # ms
    duration_mimo = df_displacement.index.max() # ms
    time_range_mimo = int(kobayashi_time_range * duration_mimo / siegel_duration)
    time_range_mimo = kobayashi_time_range
    time_range_mimo = kobayashi_time_range / 3.0
    T_TR_Left = T_TR - time_range_mimo
    T_TR_Right = T_TR + time_range_mimo

    if T_TR_Left < 0: T_TR_Left = 0
    if T_TR_Right >= duration_mimo: T_TR_Right = duration_mimo

    return T_TR_Left, T_TR_Right, time_range_mimo

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
    Delta = T_TR - T_Limb
    if Delta < -Th:
        return 'leading'
    elif Delta > Th:
        return 'following'
    else:
        return 'synchronous'
    
def analysis_siegel(df_100hz_butter: pd.DataFrame):
    # Step 1: Calculate speeds of each limb.
    df_velocities = calculate_velocities_df(df_100hz_butter)
    # Step 2: Normalize velocities of limbs to torso.
    normalize_velocities_to_torso(df_velocities)
    #df_velocities.plot()
    #plt.show()
    # Step 3: Classify stationary / moving ipsilateral limbs.
    ipsilateral_velocities_df = df_velocities[['IL', 'IA']]
    stationary_limbs_df = classify_stationary_limbs_siegel(ipsilateral_velocities_df)
    stationary_ia = 'IA' in stationary_limbs_df.keys()
    stationary_il = 'IL' in stationary_limbs_df.keys()

    # Step 4: Identify direction of moving ipsilateral limbs (forwards / backwards)
    ipsilateral_velocities_mean = ipsilateral_velocities_df.mean()
    direction_il = 'stationary'
    direction_ia = 'stationary'

    if not stationary_ia:
        direction_ia = 'forward' if ipsilateral_velocities_mean['IA'] >= 0 else 'backward'
    if not stationary_il:
        direction_il = 'forward' if ipsilateral_velocities_mean['IL'] >= 0 else 'backward'

    entry_stats = {
        'Stationary_IA': stationary_ia,
        'Stationary_IL': stationary_il,
        #'T_TR': None,
        #'T_CA': None,
        #'T_CL': None,
        #'T_IA': None,
        #'T_IL': None,
        #'V_TR': None,
        #'V_CA': None,
        #'V_CL': None,
        #'V_IA': None,
        #'V_IL': None,
        'Direction_IA': direction_ia,
        'Direction_IL': direction_il,
        #'Timing_IA': None,
        #'Timing_IL': None,
        #'Timing_CA': None,
        #'Timing_CL': None
    }

    return entry_stats

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
    torso = df_episode['TR']

    T_TR, V_TR, torso_sigmoid = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'TR')

    # Verify that R-squared value is > 0.6
    r2 = r2_score(torso, torso_sigmoid)
    if r2 < 0.6:
        print(f"Too low R-squared value: {r2}!")
        return None
    
    # Calculate T_CA and T_CL.
    T_CA, V_CA, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'CA')
    T_CL, V_CL, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'CL')

    # Pattern Classification.
    # Copy so we can use it later to fit nonstationary ipsilateral limb displacement to sigmoid.
    df_velocities = calculate_velocities_df(df_episode)
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

    # T_TR_Left, T_TR_Right, T_H = get_kobayashi_left_right_T_TR_interval(T_TR, df_episode)
    T_TR_Left = T_TR - T_H
    T_TR_Right = T_TR + T_H
    duration_mimo = df_episode.index.max() # ms
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
        T_IA, V_IA, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'IA')
        direction_ia = 'forward' if V_IA > 0 else 'backward'
    if not stationary_il:
        T_IL, V_IL, _ = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'IL')
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

def classify_abcdef(entry_stats):
    stationary_ia = entry_stats['Stationary_IA']
    stationary_il = entry_stats['Stationary_IL']
    timing_ia = entry_stats['Timing_IA']
    timing_il = entry_stats['Timing_IL']
    timing_ca = entry_stats['Timing_CA']
    timing_cl = entry_stats['Timing_CL']

    if stationary_ia and stationary_il:
        # A or B or else?
        if timing_ca == 'synchronous' and timing_cl == 'synchronous':
            return 'A'
        elif timing_ca =='synchronous' and timing_cl == 'following':
            return 'B'
        else:
            return 'O'
    
    if stationary_ia and not stationary_il:
        # C or D or else?
        if timing_ca == 'synchronous' and \
            timing_cl == 'synchronous' and \
            timing_il == 'synchronous':
            return 'C'
        elif timing_ca == 'synchronous' and \
            timing_cl == 'following' and \
            timing_il == 'synchronous':
            return 'D'
        else:
            return 'O'
        
    if stationary_il and not stationary_ia:
        # E?
        if timing_ca == 'synchronous' and \
            timing_ia == 'synchronous' and \
            timing_cl == 'following':
            return 'E'
        else:
            return 'O'
        
    # F
    return 'F'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_data', required=False, action='store_true')
    parser.add_argument('--save_data', required=False, action='store_true', default=False,
                        help="Saves collected dataframe of '--load_model' as 'data.csv'.")
    parser.add_argument('--load_data', required=False, action='store_true', default=False,
                        help="Loads data 'data.csv'.")
    parser.add_argument('--plot_displacement', action='store_true')
    parser.add_argument('--age', choices=[6, 9], type=int, help="Infant Age. Choices: 6, 9", required=True)
    parser.add_argument('--siegel', action='store_true')
    parser.add_argument('--until', type=str, default="side_lying",
                        choices=["side_lying", "45", "full"], help="Roll milestone to analyze until.")
    parser.add_argument('--thresh', type=int, default=100,
                        help="Speed threshold to use in Kobayashi Analysis to classify stationary limbs. Default: 100mm/sec.")
    parser.add_argument('--range', type=int, default=250,
                        help="Range around T_TR to use for classifying stationary limbs. Default 250ms.")
    
    args = parser.parse_args()

    if args.collect_data:
        if args.age == 6:
            model_date = '26-03-10'
            model_suffix = 'age6'
        elif args.age == 9:
            model_date = '26-03-07'
            model_suffix = 'age9'

        env = make_env(args.age)
        
        df = collect_kobayashi_displacements_all(env, model_date, 'supine', model_suffix)
        if args.save_data:
            df.to_csv(f'kobayashidata_age{args.age}.csv')

    elif args.load_data:
        # df = pd.read_csv('kobayashidata.csv', index_col=['Run', 'Time'])
        df = pd.read_csv(f'kobayashidata_age{args.age}.csv', index_col=['Run', 'Episode', 'Time'])

    else:
        raise ValueError
    
    groupby_run = df.groupby('Run')

    ax_displacement = None

    # Statistics of the runs. Includes:
    # V_TR, V_CA, V_CL: Velocities of maximum displacement after normalized & fitted to sigmoid.
    # Stationary_IA: True / False if IA stationary
    # Stationary_IL: True / False if IL stationary
    stats_list = []

    for run, df_run in groupby_run:
        for episode, df_episode in df_run.groupby(['Episode']):
            # Crop DataFrame to the section relevant for our analysis. This is either until
            # side lying or until reaching 45°.
            milestone = None
            if args.until == 'side_lying':
                milestone = 'Side_Lying'
            elif args.until == '45':
                milestone = '45_Deg'
            df_episode = crop_until_milestone(df_episode, milestone=milestone)

            if len(df_episode) <= 26:
                print(f"Skipping run {run}, episode {episode}, because it is too short!")
                continue

            df_episode = relabel_right_left_limbs_in_rolling_direction(df_episode)
            reorient_rollover(df_episode)

            # Kobayashi uses 60Hz, Siegel uses 100Hz. Our data is already at 100Hz, so for Siegel,
            # we do not need to resample
            if not args.siegel:
                df_episode = resample_df_to_60hz(df_episode)
            else:
                df_episode = resample_df_to_60hz(df_episode, target_fs=100.0)

            df_episode = df_episode.apply(smooth_x_butterworth)

            if args.siegel:
                entry_stats = analysis_siegel(df_episode)
            else:
                entry_stats = analysis_kobayashi(df_episode, thresh=args.thresh, T_H=args.range)

            if entry_stats is None:
                continue

            entry_stats['Run'] = run
            entry_stats['Episode'] = episode

            stats_list.append(entry_stats)

    df_stats = pd.DataFrame(stats_list)

    if not args.siegel:
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

    n_pattern = {
        'Two Stationary': 0,
        'Stationary IA': 0,
        'Stationary IA - IL moving backward': 0,
        'Stationary IL': 0,
        'Stationary IL - IA moving backward': 0,
        'No Stationary': 0,
    }

    for entry in stats_list:
        stationary_ia = entry['Stationary_IA']
        stationary_il = entry['Stationary_IL']
        direction_ia = entry['Direction_IA']
        direction_il = entry['Direction_IL']

        if stationary_ia and stationary_il:
            n_pattern['Two Stationary'] += 1
        elif stationary_ia and not stationary_il:
            if direction_il == 'forward':
                n_pattern['Stationary IA'] += 1
            else:
                n_pattern['Stationary IA - IL moving backward'] += 1
        elif stationary_il and not stationary_ia:
            if direction_ia == 'forward':
                n_pattern['Stationary IL'] += 1
            else:
                n_pattern['Stationary IL - IA moving backward'] += 1
        else:
            n_pattern['No Stationary'] += 1

    print(n_pattern)

    # print(df_stats)
    date_today = datetime.datetime.today().strftime('%y-%m-%d')
    analysis_type = 'siegel' if args.siegel else 'kobayashi'
    if args.until == 'side_lying':
        analysis_until = 'lateral'
    elif args.until == '45':
        analysis_until = '45'
    elif args.until == 'full':
        analyiss_until = 'full'
    df_stats.to_csv(
        f'kobayashiresults/{date_today}_{analysis_type}_thresh_{args.thresh}_range_{args.range}_until_{analysis_until}_age{args.age}.csv')
    
    # A, B, C, D, E, F classification
    if not args.siegel:
        keys = ['Timing_IL', 'Timing_IA','Timing_CA', 'Timing_CL']
        filtered_list = [{k: v for k, v in d.items() if k in keys} for d in stats_list]


        patterns = [tuple(d.values()) for d in filtered_list]

        df_patterns = []
    
        cnt = Counter(patterns)
        total_cnt = cnt.total()
        for pattern, count in cnt.items():
            timing_il = pattern[0]
            timing_ia = pattern[1]
            timing_ca = pattern[2]
            timing_cl = pattern[3]
            entry = {
                'IL': timing_il,
                'IA': timing_ia,
                'CA': timing_ca,
                'CL': timing_cl,
                'count': count,
                'fraction': count / total_cnt
            }
            df_patterns.append(entry)

        df_patterns = pd.DataFrame(df_patterns)
        df_patterns.to_csv(
            f'kobayashiresults/{date_today}_patterns_thresh_{args.thresh}_range_{args.range}_until_{analysis_until}_age{args.age}.csv')

        pattern_A = ('stationary', 'stationary', 'synchronous', 'synchronous')
        pattern_B = ('stationary', 'stationary', 'synchronous', 'following')
        pattern_C = ('synchronous', 'stationary', 'synchronous', 'synchronous')
        pattern_D = ('synchronous', 'stationary', 'synchronous', 'following')
        pattern_E = ('stationary', 'synchronous', 'synchronous', 'following')
        pattern_F = ('synchronous', 'synchronous', 'synchronous', 'synchronous')

        print(f"Cnt A: {cnt[pattern_A]/total_cnt}")
        print(f"Cnt B: {cnt[pattern_B]/total_cnt}")
        print(f"Cnt C: {cnt[pattern_C]/total_cnt}")
        print(f"Cnt D: {cnt[pattern_D]/total_cnt}")
        print(f"Cnt E: {cnt[pattern_E]/total_cnt}")
        print(f"Cnt F: {cnt[pattern_F]/total_cnt}")
        print(f"Total: {total_cnt}")




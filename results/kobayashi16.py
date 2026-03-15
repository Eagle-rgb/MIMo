"""
Docstring for results.kobayashi16

Plot velocities of joints used for measurement in Kobayashi 2016 during one episode.
"""
from collect_observation_util import collect_kobayashi_displacements_all
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from signal_utils import resample_df_to_60hz, smooth_x_butterworth
from scipy.interpolate import interp1d

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
    print(f"Reorienting rollover...")
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
    vals_fitted = 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * x)))

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
    for key in df.keys():
        displacement_series = df[key]
        velocities = calculate_velocities(displacement_series)
        df[key] = velocities

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

def classify_stationary_limbs(velocity_mean_df: pd.DataFrame, thresh_mm_sec: float):
    return velocity_mean_df[abs(velocity_mean_df['mean']) <= thresh_mm_sec]

def calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms):
    """ Returns the (displacement) velocity of the sigmoid defined by
    parameters 'beta_0' and 'beta_1' at its maximum.
    Derivative of the sigmoid P(x)=.. is simply P'(x)=beta_1*P(x)*(1-P(x)).
    Since we consider maximum displacement speed, P(x)=0.5."""
    velocity_mm_step = beta_1 * 0.5**2.0
    velocity_mm_ms = velocity_mm_step / timestep_ms
    return velocity_mm_ms * 1000.0  # to mm/sec

def get_time_and_velocity_maximum_sigmoid_velocity(df_displacement, key):
    """ Returns T_{key} and V_{key}. """
    _, (beta_0, beta_1), _ = fit_normalized_to_sigmoid(df_displacement[key])
    T = -beta_0 / beta_1
    V = calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms=10.0)
    return T, V

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_data', required=False, action='store_true')
    parser.add_argument('--save_data', required=False, action='store_true', default=False,
                        help="Saves collected dataframe of '--load_model' as 'data.csv'.")
    parser.add_argument('--load_data', required=False, action='store_true', default=False,
                        help="Loads data 'data.csv'.")
    parser.add_argument('--plot_displacement', action='store_true')
    parser.add_argument('--age', choices=[6, 9], type=int, help="Infant Age. Choices: 6, 9", required=True)
    parser.add_argument('--normalized_to_torso', action='store_true',
                        help="Normalize limb displacement speeds to torso and classify moving / stationary limbs " \
                        "based on 125% torso displacement velocity threshold.")
    parser.add_argument('--until', type=str, default="side_lying",
                        choices=["side_lying", "45"], help="Roll milestone to analyze until.")
    args = parser.parse_args()

    if args.collect_data:
        if args.age == 6:
            model_date = '26-03-10'
            model_suffix = 'age6'
        elif args.age == 9:
            model_date = '26-03-07'
            model_suffix = 'age9'

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
            success_at_side_lying=False,
            age=args.age,
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)
        
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
            df_episode = crop_until_milestone(df_episode, milestone='Side_Lying' if args.until=='side_lying' else '45_Deg')

            if len(df_episode) <= 26:
                print(f"Skipping run {run}, episode {episode}, because it is too short!")
                continue

            df_episode = relabel_right_left_limbs_in_rolling_direction(df_episode)
            reorient_rollover(df_episode)

            df_episode = resample_df_to_60hz(df_episode)
            df_episode = df_episode.apply(smooth_x_butterworth)

            # Get the torso speeds, normalize them to [0, 1] and fit to a sigmoid using log. regression.
            torso = df_episode['TR']
            ax_displacement = torso.plot()

            torso_sigmoid, (beta_0, beta_1), (min_tr, max_tr) = fit_normalized_to_sigmoid(torso)

            if args.plot_displacement:
                if ax_displacement is None:
                    ax_displacement = torso_sigmoid.plot(label=f"Run {run}, Episode {episode}")
                else:
                    torso_sigmoid.plot(ax=ax_displacement, label=f"Run {run}, Episode {episode}")

            # Verify that R-squared value is > 0.6
            r2 = r2_score(torso, torso_sigmoid)
            if r2 < 0.6:
                print(f"Too low R-squared value: {r2}!")
                continue
            
            # Get time of maximum torso velocity T_TR. For log. sigmoid, we can very elegantly calculate
            # time at which the curve hits 0.5 - it all boils down to this simple quotient of the beta
            # parameters.
            T_TR = -beta_0 / beta_1

            # Calculate left and right interval bounds for the range to classify stationary limbs.
            # Kobayashi used 0.25s left/right to 'T_TR', but our MIMo is too fast, so we must use
            # smaller values. Based on Siegel 2024, infants on average roll in 3.6+-2.8 sec to
            # lateral position from a supine. So we say that 0.25s is based on the 3.6 and vary
            # that proportional to the speed our MIMo has.
            duration_ms = df_episode.index.max()
            duration_siegel_mean_ms = 3600.0
            time_range_left_right_kob_ms = 250.0
            time_range_left_right_our_ms = int(time_range_left_right_kob_ms * duration_ms / duration_siegel_mean_ms)

            T_TR_Left = T_TR - time_range_left_right_our_ms
            T_TR_Right = T_TR + time_range_left_right_our_ms

            if T_TR_Right > duration_ms:
                print(f"T_TR is very late: {T_TR} ms for total duration {duration_ms} ms...")
                T_TR_Right = duration_ms

            V_TR = calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms=10.0)

            # Calculate T_CA and T_CL.
            T_CA, V_CA = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'CA')
            T_CL, V_CL = get_time_and_velocity_maximum_sigmoid_velocity(df_episode, 'CL')

            # Moving direction of contralatera limbs ('forward' or 'backward' relative to rolling direction)
            direction_ca = 'forward' if V_CA > 0 else 'backward'
            direction_cl = 'forward' if V_CL > 0 else 'backward'

            # Pattern Classification.
            # Copy so we can use it later to fit nonstationary ipsilateral limb displacement to sigmoid.
            df_raw_displacement = df_episode.copy()
            calculate_velocities_df(df_episode)
            #normalize_velocities_to_torso(df_episode)

            #df_episode.plot()
            #plt.ylim(-2000, 2000)
            #plt.ylabel('% of Torso')
            #plt.show()
            #raise ValueError
        
            # DataFrame containing only the ipsilateral limbs.
            df_ipsilateral = df_episode[['IA', 'IL']]

            #df_episode.plot()
            #plt.show()

            df_mean = calculate_average_velocity(df_ipsilateral, T_TR_Left, T_TR_Right)
            df_stationary = classify_stationary_limbs(df_mean, thresh_mm_sec=400)

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
                T_IA, V_IA = get_time_and_velocity_maximum_sigmoid_velocity(df_raw_displacement, 'IA')
                direction_ia = 'forward' if V_IA > 0 else 'backward'
            if not stationary_il:
                T_IL, V_IL = get_time_and_velocity_maximum_sigmoid_velocity(df_raw_displacement, 'IL')
                direction_il = 'forward' if V_IL > 0 else 'backward'

            # Calculate timing of moving limbs (leading, synchronous, following)
            T_H = time_range_left_right_our_ms
            Timing_CA = get_timing_moving_limb(T_TR, T_CA, Th=T_H)
            Timing_CL = get_timing_moving_limb(T_TR, T_CL, Th=T_H)
            Timing_IL = 'stationary'
            Timing_IA = 'stationary'
            if not stationary_il:
                Timing_IL = get_timing_moving_limb(T_TR, T_IL, Th=T_H)
            if not stationary_ia:
                Timing_IA = get_timing_moving_limb(T_TR, T_IA, Th=T_H)

            entry_stats = {
                'Episode': episode,
                'Run': run,
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
                'Timing_CL': Timing_CL
            }

            stats_list.append(entry_stats)

    df_stats = pd.DataFrame(stats_list)

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

    if args.plot_displacement:
        plt.show()

    print(df_stats)
    df_stats.to_csv(
        f'kobayashiresults_260315/kobayashi_results_until_{'sidelying' if args.until == 'side_lying' else '45'}age{args.age}.csv')


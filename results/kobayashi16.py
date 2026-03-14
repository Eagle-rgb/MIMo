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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_data', required=False, action='store_true')
    parser.add_argument('--save_data', required=False, action='store_true', default=False,
                        help="Saves collected dataframe of '--load_model' as 'data.csv'.")
    parser.add_argument('--load_data', required=False, action='store_true', default=False,
                        help="Loads data 'data.csv'.")
    parser.add_argument('--plot_displacement', action='store_true')
    parser.add_argument('--age', choices=[6, 9], type=int, help="Infant Age. Choices: 6, 9", required=True)
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
            success_at_side_lying=True,
            age=args.age,
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)
        
        df = collect_kobayashi_displacements_all(env, model_date, 'supine', model_suffix)
        if args.save_data:
            df.to_csv(f'kobayashidata_age{args.age}.csv')

    elif args.load_data:
        # df = pd.read_csv('kobayashidata.csv', index_col=['Run', 'Time'])
        df = pd.read_csv(f'kobayashidata_age{args.age}.csv', index_col=['Run', 'Time'])

    else:
        raise ValueError
    
    groupby_run = df.groupby('Run')

    ax_displacement = None

    stationary_limbs_df = []

    n_pattern = {
        'Two Stationary': 0,
        'Stationary IA': 0,
        'Stationary IL': 0,
        'No Stationary': 0
    }

    # Statistics of the runs. Includes:
    # V_TR, V_CA, V_CL: Velocities of maximum displacement after normalized & fitted to sigmoid.
    # Stationary_IA: True / False if IA stationary
    # Stationary_IL: True / False if IL stationary
    df_stats = []

    for run, df_run in groupby_run:
        df_run = relabel_right_left_limbs_in_rolling_direction(df_run)
        reorient_rollover(df_run)

        df_run = resample_df_to_60hz(df_run)
        df_run = df_run.apply(smooth_x_butterworth)

        # Get the torso speeds, normalize them to [0, 1] and fit to a sigmoid using log. regression.
        torso = df_run['TR']
        ax_displacement = torso.plot()

        torso_sigmoid, (beta_0, beta_1), (min_tr, max_tr) = fit_normalized_to_sigmoid(torso)

        if args.plot_displacement:
            if ax_displacement is None:
                ax_displacement = torso_sigmoid.plot(label=f"Run {run}")
            else:
                torso_sigmoid.plot(ax=ax_displacement, label=f"Run {run}")

        # Verify that R-squared value is > 0.6
        r2 = r2_score(torso, torso_sigmoid)
        if r2 < 0.6:
            print(f"Too low R-squared value: {r2}!")
            continue
            raise ValueError
        
        # Get time of maximum torso velocity T_TR. For log. sigmoid, we can very elegantly calculate
        # time at which the curve hits 0.5 - it all boils down to this simple quotient of the beta
        # parameters.
        T_TR = -beta_0 / beta_1

        # Calculate left and right interval bounds for the range to classify stationary limbs.
        # Kobayashi used 0.25s left/right to 'T_TR', but our MIMo is too fast, so we must use
        # smaller values. Based on Siegel 2024, infants on average roll in 3.6+-2.8 sec to
        # lateral position from a supine. So we say that 0.25s is based on the 3.6 and vary
        # that proportional to the speed our MIMo has.
        duration_ms = df_run.index.max()
        duration_siegel_mean_ms = 3600.0
        time_range_left_right_kob_ms = 250.0
        time_range_left_right_our_ms = int(time_range_left_right_kob_ms * duration_ms / duration_siegel_mean_ms)

        T_TR_Left = T_TR - time_range_left_right_our_ms
        T_TR_Right = T_TR + time_range_left_right_our_ms

        if T_TR_Right > duration_ms:
            print(f"T_TR is very late: {T_TR} ms for total duration {duration_ms} ms...")
            T_TR_Right = duration_ms

        # Calculate T_CA and T_CL.
        _, (beta_ca_0, beta_ca_1), (min_ca, max_ca) = fit_normalized_to_sigmoid(df_run['CA'])
        _, (beta_cl_0, beta_cl_1), (min_cl, max_cl) = fit_normalized_to_sigmoid(df_run['CL'])
        T_CA = -beta_ca_0 / beta_ca_1
        T_CL = -beta_cl_0 / beta_cl_1

        # Get speeds.
        V_TR = calculate_max_sigmoid_velocity(beta_0, beta_1, timestep_ms=10.0)
        V_CA = calculate_max_sigmoid_velocity(beta_ca_0, beta_ca_1, timestep_ms=10.0)
        V_CL = calculate_max_sigmoid_velocity(beta_cl_0, beta_cl_1, timestep_ms=10.0)

        # Velocities.
        calculate_velocities_df(df_run)
        #normalize_velocities_to_torso(df_run)

        #df_run.plot()
        #plt.ylim(-2000, 2000)
        #plt.ylabel('% of Torso')
        #plt.show()
        #raise ValueError

        #df_run.plot()
        #plt.show()
        df_ipsilateral = df_run[['IA', 'IL']]

        df_mean = calculate_average_velocity(df_ipsilateral, T_TR_Left, T_TR_Right)
        df_stationary = classify_stationary_limbs(df_mean, thresh_mm_sec=400)
        stationary_limbs_df.append(df_stationary)

        stationary_ia = not df_stationary[df_stationary['key'] == 'IA'].empty
        stationary_il = not df_stationary[df_stationary['key'] == 'IL'].empty

        entry_stats = {
            'Run': run,
            'Stationary_IA': stationary_ia,
            'Stationary_IL': stationary_il,
            'V_TR': V_TR,
            'V_CA': V_CA,
            'V_CL': V_CL
        }

        df_stats.append(entry_stats)

        if stationary_ia and stationary_il:
            n_pattern['Two Stationary'] += 1
        elif stationary_ia and not stationary_il:
            n_pattern['Stationary IA'] += 1
        elif not stationary_ia and stationary_il:
            n_pattern['Stationary IL'] += 1
        else:
            n_pattern['No Stationary'] += 1

    df_stats = pd.DataFrame(df_stats)

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

    stationary_limbs_df = pd.concat(stationary_limbs_df, ignore_index=True)
    print(n_pattern)
    print(stationary_limbs_df)

    if args.plot_displacement:
        plt.show()

    print(df_stats)


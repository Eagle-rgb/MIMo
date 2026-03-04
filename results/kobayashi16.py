"""
Docstring for results.kobayashi16

Plot velocities of joints used for measurement in Kobayashi 2016 during one episode.
"""
from mimoEnv.test.collect_observation_util import collect_kobayashi_site_y_displacement_series
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from signal_utils import resample_df_to_60hz, smooth_x_butterworth

def reorient_rollover(data):
    """ Reorients the rollover so that it is always a
    rollover over the left side so that y displacements are positive.
    
    Guesses the direction of rollover by checking final torso relative
    displacement. If it is negative, this was a right rollover (bad!!).
    """
    if data['Torso'].values[-1] >= 0:
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
    return data, (beta_0, beta_1)

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

def calculate_average_velocity_before_and_after(velocities_df, T, R):
    """ Calculates the average velocity for each limb (key) in 'velocities_df'
    in the time range [T-R, T] ('before') and [T, T+R] ('after'). Velocities in
    'velocities_df' are defined only pointwise. This function uses interpolation
    to estimate velocities for points T-R, T and T+R. ... """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', required=False, type=str)
    parser.add_argument('--save_data', required=False, action='store_true', default=False,
                        help="Saves collected dataframe of '--load_model' as 'data.csv'.")
    parser.add_argument('--load_data', required=False, action='store_true', default=False,
                        help="Loads data 'data.csv'.")
    parser.add_argument('--plot_displacement', action='store_true')
    args = parser.parse_args()

    if args.load_model:
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
            #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
            isr=False)
        
        model = RL.load(args.load_model, env)
        df = collect_kobayashi_site_y_displacement_series(env, model, n_episodes=1)
        if args.save_data:
            df.to_csv('data.csv')

    elif args.load_data:
        df = pd.read_csv('data.csv', index_col='Time from Onset [ms]')

    else:
        raise ValueError

    reorient_rollover(df)

    if args.plot_displacement:
        ax = df['Torso'].plot(color='red', label='Raw 100Hz unsmoothed')

    df = resample_df_to_60hz(df)
    df = df.apply(smooth_x_butterworth)

    # Get the torso speeds, normalize them to [0, 1] and fit to a sigmoid using log. regression.
    torso = df['Torso']
    if args.plot_displacement:
        torso.plot(ax=ax, color='blue', label="60Hz Smoothed")

    torso_sigmoid, (beta_0, beta_1) = fit_normalized_to_sigmoid(torso)

    if args.plot_displacement:
        torso_sigmoid.plot(ax=ax, color='green', label='Sigmoid Fitted')
        plt.legend()
        # plt.hlines(100.0, df.index[0], df.index[-1], linestyles=['dashed'], colors=['yellow'])
        plt.xlabel("Milliseconds from Onset")
        plt.ylabel("Relative Displacement (mm)")
        plt.show()

    # Verify that R-squared value is > 0.6
    r2 = r2_score(torso, torso_sigmoid)
    if r2 < 0.6:
        print(f"Too low R-squared value!")
        raise ValueError
    
    # Get time of maximum torso velocity T_TR. For log. sigmoid, we can very elegantly calculate
    # time at which the curve hits 0.5 - it all boils down to this simple quotient of the beta
    # parameters.
    T_TR = -beta_0 / beta_1
    print(f"T_TR: {T_TR} ms")

    # Velocities.
    calculate_velocities_df(df)
    df.plot()
    plt.title("Velocities")
    plt.xlabel("Milliseconds from Onset")
    plt.ylabel("Velocity [mm/sec]")


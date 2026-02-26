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
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.linear_model import LinearRegression

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

def resample_to_60hz(data_original, original_fs=100, target_fs=60):
    """ Kobayashi uses a 60Hz camera to record infant movement, while we sample
    each two timesteps, i.e. with a frequency of 100Hz. This is not a perfect
    multiple of kobayashi's 60Hz, which is why we must resample to get values
    as they would be in a 60Hz recording.
    
    Returns both the new time scale and the resampled values.
    """
    # Create time scales
    duration = len(data_original) / original_fs
    time_old = np.linspace(0, duration, len(data_original))
    
    # New time scale for 60Hz frequency.
    num_samples_new = int(duration * target_fs)
    time_new = np.linspace(0, duration, num_samples_new)
    
    # Interpolate to calculate values.
    f = interp1d(time_old, data_original, kind='linear') # oder 'cubic' für mehr Glätte
    
    return time_new, f(time_new)

def resample_df_to_60hz(df, original_fs=100, target_fs=60):
    """ Resamples an entire pandas Dataframe. """
    entries = {}
    time_scale_resampled = None
    for key in df.keys():
        time_scale_resampled, val_resampled = resample_to_60hz(data_original=df[key], original_fs=original_fs, target_fs=target_fs)
        entries[key] = val_resampled

    time_scale_resampled *= 1000.0  # convert to ms.
    df = pd.DataFrame(entries, index=time_scale_resampled)
    df.index.name = 'Time from Onset [ms]'
    return df

def smooth_x_butterworth(data, cutoff_hz=6, fs=60):
    """ Butterworth Lowpass Filter zero-phase.

    Parameters:
    * data: List of x values.
    * cutoff_hz: 6Hz to align with Kobayashi.
    * fs: Frequency of data, i.e. 60Hz
    """
    # Nyquist criteria
    nyquist = 0.5 * fs
    low = cutoff_hz / nyquist
    
    # order: 2
    b, a = butter(2, low, btype='low')
    
    # filtfilt wendet den Filter vorwärts und rückwärts an -> kein Delay
    smoothed_data = filtfilt(b, a, data)
    
    return smoothed_data

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

def calculate_velocities(displacement_series, timestep):
    """ Calculates the velocity by differentiation of the
    given series. Returns a np.array of the same size
    as 'displacement_series'.
    Be careful with chosing the correct 'timestep'. To check,
    print out env.data.time before and after a step in the
    environment. It will probably be 2 times the mujoco
    timestep.
    """
    velocities = np.zeros(len(displacement_series))
    velocities[0] = 0  # we always start with 0 velocity.
    for i in range(1, len(displacement_series)):
        y_1 = displacement_series[i-1]
        y_2 = displacement_series[i]
        velocities[i] = (y_2 - y_1) / timestep

    return velocities

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', required=True, type=str)
    args = parser.parse_args()
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
    reorient_rollover(df)
    print(df)
    
    df = resample_df_to_60hz(df)
    df = df.apply(smooth_x_butterworth)

    # Get the torso speeds, normalize them to [0, 1] and fit to a sigmoid using log. regression.
    torso = df['Torso'].copy()
    torso.plot()
    torso, (beta_0, beta_1) = fit_normalized_to_sigmoid(torso)
    torso.plot()

    # plt.hlines(100.0, df.index[0], df.index[-1], linestyles=['dashed'], colors=['yellow'])
    plt.xlabel("Milliseconds from Onset")
    plt.ylabel("Velocity (mm/sec)")
    plt.show()


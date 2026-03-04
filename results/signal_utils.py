
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import pandas as pd

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
    # do not use np.linspace!!! this only gives us an approximate 60Hz, but we want EXACTLY
    # 60Hz. This is especially important for something like seaborn linplot where we want to
    # plot the std and that is only calculated if the x-axis values match EXACTLY.
    time_new = [x * 1.0 / target_fs for x in range(num_samples_new)]
    time_new = np.array(time_new)
    
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
""" This file i"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict
from sklearn.linear_model import LinearRegression
import pandas as pd

# --- Konfiguration ---
BASE_DIR = "."
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate", "rollout/side_lying_success_rate"]
N_POINTS = 500  # Auflösung der X-Achse
DATE_FORMAT = r'%y-%m-%d'

def normalize(values):
    """ Normalizes 'data' and returns it together with the normalization
    statistics. """
    # Get normalization statistics, i.e. max and min.
    max_value = values.max()
    min_value = values.min()

    values = (values - min_value) / (max_value - min_value)

    return values, (min_value, max_value)

def fit_sigmoid(values, steps):
    """ Fits normalized (!!) data to a sigmoid curve
    using logistic regression. Returns beta parameters of sigmoid.
    """
    y = values
    x = steps

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

    return beta_0, beta_1

def get_model_training_data_sigmoid_fits(dates, suffixes, haltungen, tags):
    """ Expects 'dates', 'suffixes', 'haltungen' and 'tags' of
    equal size. Loads data for each model training. Normalizes and
    fits to a sigmoid. Returns a list matching the size of the
    parameter lists containing entries with normlization statistics
    and sigmoid parameters. """
    all_run_data = load_tensorboard_runs(base_dir=os.path.abspath(BASE_DIR), tags=tags, date_filter=dates, suffix_filter=suffixes)
    # list of individual model training data.
    model_data = []
    n_runs = len(dates)

    if all_run_data.empty:
        raise ValueError

    for i in range(n_runs):
        date = dates[i]
        suffix = suffixes[i]
        haltung = haltungen[i]
        tag = tags[i]

        # 'df_model' still has many runs. We need to average over runs.
        df_model = all_run_data[(all_run_data['Date']==date) &
                              (all_run_data['Suffix']==suffix) &
                              (all_run_data['Haltung']==haltung) &
                              (all_run_data['Tag']==tag)]
        
        if df_model.empty:
            model_data.append(None)
            print(f"No data found for date {date}, suffix {suffix}, haltung {haltung}, tag {tag}")
            continue
        
        df_stats = interpolate_runs_to_dict(df_model, n_points=500)
        values = df_stats['mean']
        steps = df_stats['steps']

        values, (min_value, max_value) = normalize(values)
        beta_0, beta_1 = fit_sigmoid(values, steps)

        entry = {
            'model_idx': i,
            'min': min_value,
            'max': max_value,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'steps': df_model['Step'].max()
        }

        model_data.append(entry)
 
    # 5. Reconstruction
    #vals_fitted = 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * x)))

    # 6. Unnormalize data
    #vals_fitted = (max_value - min_value) * vals_fitted + min_value

    #data[:] = vals_fitted

    return pd.DataFrame(model_data)

def plot_suffix_run_data(axIndi, axAggre, groupby, label):
    """ Plots run data for one suffix into individual plot and into
    aggregated plot.
    
    Parameters:
        axIndi: The individual axes object from pyplot
        axAggre: The aggregated axes object from pyplot
        groupby: rundata, i.e. dataframe grouped by suffix
        label: The label for this suffix of the model
    """
    # Die einzelnen runs dieses suffixes für den tag und die Haltung.
    run_data = interpolate_runs_to_dict(groupby, N_POINTS)

    # Plot des Mittelwerts
    axAggre.plot(run_data['steps'], run_data['mean'], label=label, linewidth=2)

    # Plot der Standardabweichung als Fehlerband
    axAggre.fill_between(
        run_data['steps'], 
        run_data['mean'] - run_data['std'], 
        run_data['mean'] + run_data['std'], 
        alpha=0.15 
    )

    # 16.01.26 Also plot each individual run lightly in the background.
    for key in run_data['runs'].keys():
        values = run_data['runs'][key]
        axIndi.plot(run_data['steps'], values, label=str(key), linewidth=2)

def plot_data(data: pd.DataFrame, labels: list[str]):
    max_step = data['steps'].max()
    x_axis = np.linspace(0, max_step, 500)

    for model_idx, model_df in data.groupby(['model_idx']):
        min_value = model_df['min'].values[0]
        max_value = model_df['max'].values[0]
        beta_0 = model_df['beta_0'].values[0]
        beta_1 = model_df['beta_1'].values[0]
        steps = model_df['steps'].values[0]

        capped_x = np.clip(x_axis, 0, steps)

        y_vals_normalized = 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * capped_x)))
        y_vals = (max_value - min_value) * y_vals_normalized + min_value
        label = labels[model_idx[0]]

        plt.plot(x_axis, y_vals, label=label)

    plt.legend()
    plt.xlabel('Steps')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def valid_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {s!r}")

# --- Start ---
if __name__ == "__main__":
        # 0. Argumente laden. Zeit und Modelsuffix.
    # '--date' and '--suffix' are optional parameters used to filter which training
    # data to include in the plot. '--date' may be specified and is specified followed
    # by a date in 'yy-mm-dd' format. If a date is specified, this date is used in
    # the output .png file name.
    # '--suffix' is specified by following it up with
    # a list of suffixes that should be allowed. If exactly one suffix is specified, then
    # this suffix is included in the output .png file name.
    parser = argparse.ArgumentParser()
    max_models = 8

    for i in range(1,max_models+1):
        parser.add_argument(f'--date{i}', required=i==1, type=valid_date, help=f"Date of the runs {i}")
        parser.add_argument(f'--suffix{i}', required=i==1, type=str, help=f"Model name suffix {i}")
        parser.add_argument(f'--label{i}', required=i==1, type=str, help=f"Label for model {i}")
        parser.add_argument(f'--haltung{i}', required=i==1, type=str, choices=['prone', 'supine'], help=f"Haltung of model {i}")
        parser.add_argument(f'--tag{i}', required=i==1, choices=['success_rate'], help=f"Tag to load for model {i}")
    args = parser.parse_args()
    dates = []
    suffixes = []
    labels = []
    haltungen = []
    tags = []

    for i in range(1,max_models+1):
        date = getattr(args, f'date{i}')
        if date is None:
            break

        suffix = getattr(args, f'suffix{i}')
        if suffix is None:
            raise ValueError(f"No suffix provided for model {i}")
        
        label = getattr(args, f'label{i}')
        if suffix is None:
            raise ValueError(f"No label provided for model {i}")
        
        haltung = getattr(args, f'haltung{i}')
        if suffix is None:
            raise ValueError(f"No haltung provided for model {i}")
        
        tag = getattr(args, f'tag{i}')
        if suffix is None:
            raise ValueError(f"No tag provided for model {i}")
        
        dates.append(date.strftime(DATE_FORMAT))
        suffixes.append(suffix)
        labels.append(label)
        haltungen.append(haltung)
        tags.append("rollout/" + tag)

    if len(dates) == 0:
        raise ValueError("No models specified...")
    
    data = get_model_training_data_sigmoid_fits(dates, suffixes, haltungen, tags)
    plot_data(data, labels)
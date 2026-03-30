""" This file i"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tb_plot_utils import load_tensorboard_runs, load_model_hyperparams, interpolate_runs_to_dict
from sklearn.linear_model import LinearRegression
import pandas as pd
import icdlplot

# --- Konfiguration ---
BASE_DIR = "."
TAGS_TO_LOAD = ["rollout/ep_rew_mean", "rollout/success_rate", "rollout/side_lying_success_rate"]
N_POINTS = 500  # Auflösung der X-Achse
DATE_FORMAT = r'%y-%m-%d'

def get_model_training_data_aggregated(dates, suffixes, haltungen, tags, xmax):
    """ Expects 'dates', 'suffixes', 'haltungen' and 'tags' of
    equal size. Loads data for each model training. Normalizes and
    fits to a sigmoid. Returns a list matching the size of the
    parameter lists containing entries with normlization statistics
    and sigmoid parameters. """
    all_model_data = load_tensorboard_runs(base_dir=os.path.abspath(BASE_DIR), tags=tags, date_filter=dates, suffix_filter=suffixes)
    # list of individual model training data.
    model_data = []
    n_models = len(dates)
    x_axis = np.linspace(0, xmax, N_POINTS)

    if all_model_data.empty:
        raise ValueError

    for i in range(n_models):
        date = dates[i]
        suffix = suffixes[i]
        haltung = haltungen[i]
        tag = tags[i]

        # 'df_model' still has many runs. We need to average over runs.
        df_model = all_model_data[(all_model_data['Date']==date) &
                              (all_model_data['Suffix']==suffix) &
                              (all_model_data['Haltung']==haltung) &
                              (all_model_data['Tag']==tag)]
        
        if df_model.empty:
            raise ValueError(f"No data found for date {date}, suffix {suffix}, haltung {haltung}, tag {tag}")
        
        df_stats = interpolate_runs_to_dict(df_model, n_points=N_POINTS, min_step=0, max_step=xmax)
        values = df_stats['mean']
        steps = df_stats['steps']
        std = df_stats['std']

        num_runs = len(df_model['Run'].unique())

        entry = {
            'model_idx': i,
            'value': values,
            'step': steps,
            'std': std,
            'num_runs': num_runs,
        }

        model_data.append(entry)

    return model_data

def plot_data(data, labels: list[str], max_x: int, save_file: str, append_num_runs_to_label: bool=False,
              legend_title: str=None):
    x_axis = np.linspace(0, max_x, N_POINTS)

    plt.figure(figsize=(3,3))
    color_idx = 0

    for model_data in data:
        model_idx = model_data['model_idx']
        values = model_data['value']
        label = labels[model_idx]

        if append_num_runs_to_label:
            label += f", {model_data['num_runs']} runs"

        plt.plot(x_axis, values, label=label, color=PLT_COLORS[color_idx])
        # Prevent drawing over 1.0 or under 0.0 for success rate.
        std_min = np.clip(values - model_data['std'], 0.0, 1.0)
        std_max = np.clip(values + model_data['std'], 0.0, 1.0)
        plt.fill_between(x_axis, std_min, std_max, alpha=0.15, color=PLT_COLORS[color_idx])
        color_idx += 1

    if legend_title:
        plt.legend(title=legend_title)
    else:
        plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Mean Success Rate')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_file}.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')
        
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

    parser.add_argument('--name', required=True, type=str, help=f"Output filename")
    parser.add_argument('--num_runs_in_label', action='store_true', help=f"Suffix label " \
                        "with number of runs.")
    parser.add_argument('--legend_title', type=str, required=False, help="Title for legend.")

    for i in range(1,max_models+1):
        parser.add_argument(f'--date{i}', required=i==1, type=valid_date, help=f"Date of the runs {i}")
        parser.add_argument(f'--suffix{i}', required=i==1, type=str, help=f"Model name suffix {i}")
        parser.add_argument(f'--label{i}', required=i==1, type=str, help=f"Label for model {i}")
        parser.add_argument(f'--haltung{i}', required=i==1, type=str, choices=['prone', 'supine'], help=f"Haltung of model {i}")
        #parser.add_argument(f'--tag{i}', required=i==1, choices=['success_rate'], help=f"Tag to load for model {i}")
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
        if label is None:
            raise ValueError(f"No label provided for model {i}")
        
        haltung = getattr(args, f'haltung{i}')
        if haltung is None:
            raise ValueError(f"No haltung provided for model {i}")
        
        #tag = getattr(args, f'tag{i}')
        #if suffix is None:
        #    raise ValueError(f"No tag provided for model {i}")
        
        dates.append(date.strftime(DATE_FORMAT))
        suffixes.append(suffix)
        labels.append(label)
        haltungen.append(haltung)
        #tags.append("rollout/" + tag)
        tags.append("rollout/success_rate")

    if len(dates) == 0:
        raise ValueError("No models specified...")
    
    data: pd.DataFrame = get_model_training_data_aggregated(dates, suffixes, haltungen, tags, 1e6)
    plot_data(data, labels, 1e6, args.name, args.num_runs_in_label, args.legend_title)
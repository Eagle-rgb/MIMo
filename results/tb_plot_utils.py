import os
import re
import pandas as pd
from tbparse import SummaryReader
import yaml
import numpy as np

TB_ALGORITHM_FOLDER_NAMES = ['PPO_0', 'DDPG_0', 'SAC_0', 'A2C_0']

def load_model_hyperparams(model_folder):
    """ Loads model hyperparameters from the .yml file in the model folder.
    Returns hyperparameters as '|' separated string.
    Returns 'None' if the file could not be found.
    """
    out_str = ""
    try:
        with open(os.path.join(model_folder, 'data.yml'), 'r') as file:
            data = yaml.safe_load(file)
            for key in data.keys():
                out_str += f'{key}: {data[key]} | '
    except Exception:
        return None

    # Remove trailing ' | '
    return out_str[:-3]

def load_tensorboard_runs(base_dir, tags, date_filter, suffix_filter):
    """ Loads tensorboard runs and creates and returns a pandas DataFrame with the following information:
    * Haltung: prone / supine
    * Suffix: Model suffix
    * Date: Model date
    * Run: Number of this run
    * Step: timestep of value
    * Value: Value at step 'Step'
    * Tag: Tensorboard tag (for example: rollout/success_rate)
    
    Arguments:
        - base_dir (str): The directory to start looking for tensorboard data recursively.
        - tags (list[str]): List of tags to load.
        - date_filter (str): Filter for model dates to look for. 'None' if should be ignored.
            Can be list[str] or a single [str]. Internally, it is then converted to a single-element list.
        - suffix_filter (list[str]): Filter for model suffixed to look for. 'None' if should be ignored.
    """
    all_data_list = []
    
    # Convert string date_filter to list
    if type(date_filter) == str:
        date_filter = [date_filter]

    # yy-mm-dd_<prone/supine>_<suffix>_run_xx
    pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_([a-z]+)_([a-z0-9_-]+)_run_(\d+)')

    print(f"Suche nach TensorBoard Logs in {base_dir} ...")
    
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) not in TB_ALGORITHM_FOLDER_NAMES: continue

        algorithm_name = os.path.basename(root)[:-2]
        run_folder_name = os.path.basename(os.path.dirname(root))
        match = pattern.search(run_folder_name)

        if not match: continue
        date, haltung, suffix, run_num = match.groups()

        if date_filter and date not in date_filter: continue
        if suffix_filter and suffix not in suffix_filter: continue

        print(f"Lade Run: Date={date}, Haltung={haltung}, Suffix={suffix}, Run={run_num} aus {root}.")

        try:
            reader = SummaryReader(root)
            df_scalars = reader.scalars
            tag_data = df_scalars[df_scalars['tag'].isin(tags)].copy()
            if not tag_data.empty:
                tag_data['Haltung'] = haltung
                tag_data['Run'] = int(run_num)
                tag_data['Suffix'] = suffix
                tag_data['Date'] = date
                tag_data['Folder'] = os.path.dirname(root)
                tag_data.rename(columns={'step': 'Step', 'value': 'Value', 'tag': 'Tag'}, inplace=True)
                all_data_list.append(tag_data[['Date', 'Folder', 'Haltung', 'Run', 'Suffix', 'Tag', 'Step', 'Value']])

        except Exception as e:
            print("Error loading tensorboard data...")
    return pd.concat(all_data_list, ignore_index=True) if all_data_list else pd.DataFrame()

# --- 2. Interpolations-Logik ---
def interpolate_runs_to_dict(group_df, n_points, min_step=None, max_step=None):
    """ 07.01.2026: DDPG and SAC have steps in their tensorboard logs that do not follow a "pattern", i.e.
    we can not just do a group_by on the step and then get a list of the values for each run. We must
    manipulate the pandas dataframe samples such that the steps match how we would have them in PPO or
    A2C, i.e. each 500 steps in A2C or 2048 in PPO there is one entry in the tensorboard log.
    
    Args:
      group_df (panda.DataFrame): DataFrame consisting of arbitrary number of runs.
      n_points: Number of sample points on the x-axis.
      min_step: Minimum step.
      max_step: Maximum step. If either 'min_step' or 'max_step' is not specified,
        they are automatically derived from the supplied data by taking the maximum
        and minimum step from the data. It is useful to explicitly specify them
        to compare models with different number of training steps.
    
    Returns:
      run_dict (dict): Dictionary containing the following key-value pairs:
      * steps: x-axis, list of step points
      * mean: Mean at x-step over all runs
      * std: Std at x-step over all runs
      * runs: Individual run data
    """
    if group_df.empty: return None

    # Automatically infer maximum and minimum step from supplied data if not
    # explicitly specified.
    if min_step is None or max_step is None:
        min_step, max_step = group_df['Step'].min(), group_df['Step'].max()
    common_steps = np.linspace(min_step, max_step, n_points)
    
    run_dict = {}
    for run_id in sorted(group_df['Run'].unique()):
        run_data = group_df[group_df['Run'] == run_id].sort_values('Step')
        if len(run_data) < 2: continue
        run_dict[int(run_id)] = np.interp(common_steps, run_data['Step'], run_data['Value'])
    
    if not run_dict: return None
    all_vals = np.array(list(run_dict.values()))
    return {
        'steps': common_steps,
        'mean': np.mean(all_vals, axis=0),
        'std': np.std(all_vals, axis=0),
        'runs': run_dict
    }
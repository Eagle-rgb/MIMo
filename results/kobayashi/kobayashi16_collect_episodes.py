import argparse
from results.utils import DATE_FORMAT, valid_date
from mimoEnv.eval_rollover import (build_env, load_run_config, ROLL_THRESHOLD,
                                   DEFAULT_EPISODE_STEPS)
import os
import mujoco
import numpy as np
import pandas as pd
import re

# Need to be prefixed with 'act:left_' or 'act:right_' - depending on
# the roll direction.
ARM_ACTUATORS = ["shoulder_horizontal",
                 "shoulder_abduction",
                 "shoulder_internal",
                 "elbow"]

# Need to be prefixed with 'act:left_' or 'act:right_' - depending on
# the roll direction.
LEG_ACTUATORS = ["hip_flex",
                 "hip_abduction",
                 "hip_rotation",
                 "knee"]

# Need to be prefixed with 'act:'
TORSO_ACTUATORS = ["chest_twist",
                   "chest_lean"]

# Need to be prefixed with 'act:'
HIP_ACTUATORS = ["hip_bend",
                 "hip_twist",
                 "hip_lean"]

def load_policy(model_file, env, algorithm='PPO'):
    """ Loads a saved policy for evaluation only.

    20.08.2026 The Kobayashi collection used to be hardcoded to ``PPO.load(file, env)``. The
    SAC+HER runs need three things that call cannot give them:

    * the right algorithm class, read from the run's ``data.yml``;
    * an environment -- SB3 asserts on it for anything saved with a ``HerReplayBuffer``
      ("You must pass an environment when using `HerReplayBuffer`"), because the buffer needs
      ``env.compute_reward`` to relabel;
    * a shrunk replay buffer, or loading reallocates the 300k-transition training buffer for
      every one of the 18 runs. We never train here, so a single transition is enough.

    Same construction as ``mimoEnv.eval_rollover.load_policy``.
    """
    import stable_baselines3 as sb3
    cls = getattr(sb3, algorithm)
    return cls.load(model_file, env=env,
                    custom_objects={'buffer_size': 1, 'learning_starts': 0})

def get_actuator_index(env, act):
    """ Returns the index of the actuator 'act' - specified
    using full qualified name in 'env.mimo_actuators' array.
    
    You may then proceed to read out the torque values of this
    actuator from the actuation model at that control_input index."""
    for i in range(len(env.mimo_actuators)):
        act_id = env.mimo_actuators[i]  # id in mjModel.actuator(...)
        act_name = env.model.actuator(act_id).name
        if act_name != act: continue
        return i
    return -1

# Pattern of a model folder written by --roll_over_model_path_auto:
# <date>_<startingposition>_<suffix>_run_<i>
RUN_DIR_PATTERN = re.compile(r'(\d{2}-\d{2}-\d{2})_([a-z]+)_([a-z0-9_-]+)_run_(\d+)')

def find_runs(model_dir, date, pos, suffix, run_num=None):
    """ Finds every run directory under 'model_dir' matching 'date', starting position 'pos' and
    'suffix'. Returns a list of (run_num, absolute path) tuples sorted numerically by run number.

    Split out of 'collect_kobayashi_displacements_all' so that callers can read a run's 'data.yml'
    *before* building the environment -- the SAC+HER runs need an env configured from their own
    config (goal in the observation, no termination on success, their episode horizon), which is
    not knowable until a run has been found.
    """
    if model_dir is None:
        model_dir = '.'

    runs = []
    for root, dirs, files in os.walk(model_dir):
        match = RUN_DIR_PATTERN.search(os.path.basename(root))
        if not match: continue
        _date, haltung, _suffix, _run_num = match.groups()

        if _date != date: continue
        if haltung != pos: continue
        if _suffix != suffix: continue
        if run_num is not None and _run_num != str(run_num): continue

        runs.append((_run_num, os.path.abspath(root)))

    return sorted(runs, key=lambda entry: int(entry[0]))

def collect_kobayashi_site_y_displacement_series(env, model, n_tries=10, with_actuations=True,
                                                 diss=None, episode_steps=None, seed0=None):
    """ Lets the (trained) model 'model' play in the environment 'env'
    for a total of 'n_tries' episodes. Records the y displacement of each site defined
    as kobayashi site. Collects in total a number of (n_steps + 1)
    values if the model takes 'n_steps' in the environment, because
    we record one set of value before each step, but also after the final
    step. Only records successful episodes. Tries in total 'n_tries' and if
    after that, no successful episode was done, returns 'None'.

    Specify diss as np.array of shape (n_tries, qpos[7:].shape[0]) to enable deterministic
    initial state sampling.

    20.08.2026 Success is measured as rho_max >= 0.95 -- the highest rotation reached anywhere in
    the episode -- rather than read off the termination flag, and the episode is cut at that
    point. The HER runs are trained with '--no_done_active', so nothing terminates on success and
    the old reading of 'terminated' would have reported zero successful episodes for every one of
    them. For a run trained with done_active=True the two agree, because that env terminates
    exactly when rho crosses 0.95, so the recorded trajectories are unchanged.

    'episode_steps' is the horizon. Required for envs without a TimeLimit (build_env in
    eval_rollover.py returns the unwrapped env), which would otherwise never end an episode.
    Pass 'seed0' to seed episode i with 'seed0 + i', so every model sees the same start states.

    Very important distinction: This function records the RELATIVE displacement
    of each site, i.e. the displacement relative to the coordinate the site
    had after the environment reset.

    Returns the data as a pandas DataFrame with modified more-beautiful
    names for the sites as keys. Has a standard count index.
    """
    KOBAYASHI_SITES = ["KOBAYASHI_RWrist",
                    "KOBAYASHI_RAnkle",
                    "KOBAYASHI_LWrist",
                    "KOBAYASHI_LAnkle",
                    "KOBAYASHI_Torso"]
    
    SENSOR_NAMES = ["Right Wrist", "Right Ankle", "Left Wrist", "Left Ankle", "TR"]

    # Prepare by collecting id for each site. This list at index i then contains the id
    # of site in KOBAYASHI_SITES at index i.
    kobayashi_site_ids = []

    for site in KOBAYASHI_SITES:
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site)
        kobayashi_site_ids.append(site_id)

    # Dictionary of reference coordinates to calculate relative displacement
    # of site. Set after the first 'env.reset()'.
    reference_coords = None

    # Prepare muscles
    # Prepare by collecting index of each actuator in 'env.mimo_actuators' so that
    # we have fast access to reading the actuation values of each actuators from the
    # actuation model.
    torso_act_idx = {}
    hip_act_idx = {}
    left_arm_act_idx = {}
    right_arm_act_idx = {}
    left_leg_act_idx = {}
    right_leg_act_idx = {}

    for act_shortname in ARM_ACTUATORS:
        act_fullname = "act:left_" + act_shortname
        left_arm_act_idx[act_fullname] = get_actuator_index(env, act_fullname)
        act_fullname = "act:right_" + act_shortname
        right_arm_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in LEG_ACTUATORS:
        act_fullname = "act:left_" + act_shortname
        left_leg_act_idx[act_fullname] = get_actuator_index(env, act_fullname)
        act_fullname = "act:right_" + act_shortname
        right_leg_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in TORSO_ACTUATORS:
        act_fullname = "act:" + act_shortname
        torso_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    for act_shortname in HIP_ACTUATORS:
        act_fullname = "act:" + act_shortname
        hip_act_idx[act_fullname] = get_actuator_index(env, act_fullname)

    # Collection of all the indexes above into the groups we will use.
    act_idx = {
        'Act_Torso': torso_act_idx,
        'Act_Hip': hip_act_idx,
        'Act_Right_Leg': right_leg_act_idx,
        'Act_Left_Leg': left_leg_act_idx,
        'Act_Right_Arm': right_arm_act_idx,
        'Act_Left_Arm': left_arm_act_idx
    }

    def get_actuation_of_group(group_key):
        vals = []
        for _, act_indx in act_idx[group_key].items():
            vals.append(abs(env.actuation_model.control_input[act_indx]))
        return np.mean(np.array(vals))
    
    def collect_actuations(zero=False):
        entry = {}
        for group_key in act_idx.keys():
            mean_act = get_actuation_of_group(group_key) if not zero else 0.0
            entry[group_key] = mean_act
        return entry

    def get_site_absolute_displacements():
        """ Returns the absolute displacements, i.e.
        the displacement from the global origin. """
        vals = {}
        for i in range(len(KOBAYASHI_SITES)):
            site_id = kobayashi_site_ids[i]
            site_y = env.data.site_xpos[site_id][1] * 1000.0  # from m to mm
            vals[SENSOR_NAMES[i]] = site_y
        return vals
    
    def get_site_relative_displacements():
        """ Returns the relative displacements, i.e.
        the displacements from the respective site origin.
        Uses 'reference_coords' as site origins. """
        absolute_displacements = get_site_absolute_displacements()
        for key in absolute_displacements.keys():
            absolute_displacements[key] -= reference_coords[key]
        return absolute_displacements

    is_success = False

    # keeps track of number of already recorded successful episodes.
    n_successful_episodes = 0
    all_successful_data = []

    for n_try in range(n_tries):
        print(f"Doing try {n_try+1}")
        # List of dictionaries - ordered by steps in the environment.
        data = []

        if diss is not None:
            env.deterministic_initial_state_sampling = diss[n_try]

        obs, _ = env.reset(seed=None if seed0 is None else seed0 + n_try)
        reference_coords = get_site_absolute_displacements()
        done = False
        deg_45_reached = False
        side_lying_reached = False
        rho_max = float(env.unwrapped.get_achieved_goal_cos_mean()[0])
        step = 0

        # For some reason, at the start of the episode, we do not start with
        # 'env.data.time=0'. So we simply subtract this from each following
        # time.
        time_offset = env.data.time * 1000.0
        entry = get_site_relative_displacements()
        entry['Episode'] = n_successful_episodes + 1
        entry['Time'] = env.data.time * 1000.0 - time_offset # env.data.time is in sec.
        entry['Side_Lying'] = False
        entry['45_Deg'] = False
        entry = entry | collect_actuations(zero=True)
        data.append(entry)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            side_lying_reached = side_lying_reached or (info['side_lying'] == 1.0)
            deg_45_reached = deg_45_reached or (info['45_deg'] == 1.0)
            rho_max = max(rho_max, float(env.unwrapped.get_achieved_goal_cos_mean()[0]))

            entry = get_site_relative_displacements()
            entry['Episode'] = n_successful_episodes + 1
            entry['Time'] = env.data.time * 1000.0 - time_offset # env.data.time is in sec.
            entry['Side_Lying'] = side_lying_reached
            entry['45_Deg'] = deg_45_reached
            entry = entry | collect_actuations()
            data.append(entry)

            rolled = rho_max >= ROLL_THRESHOLD
            out_of_time = episode_steps is not None and step >= episode_steps
            done = rolled or terminated or truncated or out_of_time

            if rolled:
                is_success = True
                print(f"Success! (rho_max {rho_max:.3f} after {step} steps)")

                for entry in data:
                    all_successful_data.append(entry)

                n_successful_episodes += 1

            elif done:
                print(f"No Success! (rho_max {rho_max:.3f})")

    if is_success:
        return pd.DataFrame(all_successful_data)
    else:
        return None

def collect_kobayashi_displacements_all(env, date, pos, suffix, n_tries=10,
                                        with_actuations=False, diss=None, run_num=None,
                                        model_dir=None, algorithm='PPO', checkpoint='model_1.zip',
                                        episode_steps=None, seed0=None):
    """ Searches for all models matching 'date', starting position 'pos' and suffix 'suffix'.
    Loads all runs of these models and plays 'collect_kobayashi_site_y_displacement_series' on them.
    
    Parameters:
        - env: The MIMoRollOverEnv instance
        - date: The date of the model to load
        - pos: The starting position ('supine' or 'prone')
        - suffix: The suffix of the model to load
        - n_tries: Number of tries for each loaded model to perform rolling.
        - with_actuations: Include muscle actuations per step in the output statistic csv file
        - diss: Enabled deterministic initial state sampling. Provide np.array of shape
            (n_tries, qpos[7:].shape)
        - run_num: Only collect data for a specific run. If 'run_num=None', all runs are considered.
        - model_dir: The root directory to start searching for files.
        - algorithm: SB3 algorithm the checkpoints were saved with ('PPO', 'SAC', ...).
        - checkpoint: Which checkpoint file to load from each run directory. 'model_best.zip' is
            the peak under the reported evaluation protocol (written by RollOverEvalCallback);
            the last numbered checkpoint is not reliably the best one.
        - episode_steps: Episode horizon, needed for envs without a TimeLimit.
        - seed0: Seed episode i of every run with 'seed0 + i', so all runs face identical start
            states. 'None' keeps the previous unseeded behaviour.
    """
    data = []
    runs = find_runs(model_dir, date, pos, suffix, run_num=run_num)

    if not runs:
        raise ValueError(f"No run directories matching {date}_{pos}_{suffix}_run_* "
                         f"under {model_dir}")

    for _run_num, root in runs:
        print(f"Found run {_run_num}!")

        model_file = os.path.join(root, checkpoint)
        model = load_policy(model_file, env, algorithm=algorithm)
        df = collect_kobayashi_site_y_displacement_series(env, model, n_tries=n_tries,
                                                          with_actuations=with_actuations,
                                                          diss=diss,
                                                          episode_steps=episode_steps,
                                                          seed0=seed0)
        if df is None:
            print(f"Run {_run_num} had no successfull episodes in {n_tries} tries. Skipping...")
            continue
        df['Run'] = _run_num
        df = df.set_index(['Run', 'Episode', 'Time'])
        data.append(df)

    if len(data) > 1:
        df = pd.concat(data)
    elif len(data) == 1:
        df = data[0]
    else:
        raise ValueError(f"No data found in directory {model_dir}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', required=False, default=10, type=int,
                    help="Number of episodes to let each model play in the environment. This " \
                    "is not the number of episodes taken into account for kobayashi analysis. " \
                    "For kobayashi analysis, only the successful episodes out of the 'num_episodes' " \
                    "performed episodes is taken. Default: 10")
    parser.add_argument('--episode_steps', type=int, default=500, required=False, help="Number of steps per episode.")
    parser.add_argument('--suffix', type=str, required=True, help="Model suffix.")
    parser.add_argument('--date', type=valid_date, required=True, help="Date of the model.")
    parser.add_argument('--age', type=int, required=True, choices=[6,9], help="Age of the model.")
    parser.add_argument('--model_dir', type=str, default='.', help="Directories to look for models recursively.")
    parser.add_argument('--out', type=str, help="Name of the .csv file being saved. Include the '.csv'.")
    args = parser.parse_args()

    model_date = args.date.strftime(DATE_FORMAT)
    model_suffix = args.suffix
    age = args.age

    # 20.08.2026 The environment used to come from 'results.utils.make_env', which hardcodes
    # PPO-era settings (pbrs, no goal in the observation, done_active). A SAC+HER policy
    # cannot even be fed by that env: its observation space has no 'desired_goal'. Build the
    # env from the run's own 'data.yml' instead, through the same 'build_env' the reported
    # evaluation protocol uses -- ISR off, goal pinned, no termination on success.
    runs = find_runs(args.model_dir, model_date, 'supine', model_suffix)
    if not runs:
        raise ValueError(f"No run directories matching "
                            f"{model_date}_supine_{model_suffix}_run_* "
                            f"under {args.model_dir}")
    print(f"Found {len(runs)} runs. Reading configuration from {runs[0][1]}.")

    config = load_run_config(os.path.join(runs[0][1], "model_1.zip"))
    # The age is an explicit argument because of transfer learning, where the env is
    # deliberately not the one the policy was trained in.
    config = dict(config, physio_age=age, morph_age=age)
    algorithm = config.get('algorithm', 'PPO')
    episode_steps = (args.episode_steps or config.get('episode_steps')
                        or DEFAULT_EPISODE_STEPS)

    env = build_env(config, "supine", None)

    df = collect_kobayashi_displacements_all(
        env, model_date, 'supine', model_suffix,
        model_dir=args.model_dir, n_tries=args.num_episodes,
        algorithm=algorithm, checkpoint="model_1.zip",
        episode_steps=episode_steps,
        seed0=0)  # always seed the experiments.

    out = args.out
    if out is None:
        out = f'kobayashidata_age{args.age}.csv'
    df.to_csv(out)
    print(f"Wrote {out}")
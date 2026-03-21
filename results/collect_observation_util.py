import pandas as pd
import numpy as np
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
import argparse
from stable_baselines3 import PPO as RL
from mimoEnv.envs.mimo_env import PROPRIOCEPTION_PARAMS_ONLY_QPOS
from render.utils import evaluation_video
from mimoActuation.actuation_pc1 import SpringDamperModel_PC1
import mujoco
from PIL import Image
import os
import re

OBS_VESTI_KEYS = ["accelerometer_x", "accelerometer_y", "accelerometer_z", 
                  "gyro_x", "gyro_y", "gyro_z"]

def flatten_obs(obs):
    return np.concatenate([obs[key] for key in sorted(obs.keys())])

def proprio_obs_to_dict(env, obs):
    """ Gets a proprioception observation and returns it as a dictionary. The
    keys have the following structure:
    For 'qpos' values: '{joint_name}_qpos'
    For 'velocity' values: '{joint_name}_vel'
    For 'torque' values: '{sensor_name}_torque'. These values are in Proprioception::sensors
    For 'limits' values: '{joint_name}_lim'
    For 'actuation' control values: '{actuator_name}_actctrl'
    For 'actuation' torque values: '{actuator_name}_acttrq'

    Actuation is a special one, because we have two concatenated list of size <n_actuators>
    of the form [ctrl_values, torque_values] in the SpringDamperModel.

    To get 'actuator_name', the actuation array is sorted according to env.mimo_actuators
    parameter from the mimo_env class. It contains indexes in the self.model.actuator(i)
    list with i being the index stored in 'env.mimo_actuators'. You can get actuator name
    by calling 'self.model.actuator(i).name'.

    The only spe

    The proprioception observation is sorted in parts
    qpos -- veloctiy -- torque -- limits -- actuation

    We know the size of each part and they all only contain 1d - values.
    """
    use_velocity = "velocity" in env.proprioception.output_components
    use_torque = "torque" in env.proprioception.output_components
    use_limits = "limits" in env.proprioception.output_components
    use_actuations = "actuation" in env.proprioception.output_components

    joint_names = env.proprioception.joint_names
    n_joints = len(joint_names)
    sensors = env.proprioception.sensors
    n_sensors = len(sensors)

    # observation is structured like this:
    # <n_joints> - <n_joints> - <n_sensors> - <n_joints> - <n_actuators>
    qpos = obs[:n_joints]

    # accumulating last index
    idx = n_joints

    if use_velocity:
        velocity = obs[idx:idx+n_joints]
        idx = idx + n_joints
    if use_torque:
        torque = obs[idx:idx+n_sensors]
        idx = idx + n_sensors
    if use_limits:
        limits = obs[idx:idx+n_joints]
        idx = idx + n_joints
    if use_actuations:
        actuations = obs[idx:]

    out = { }

    for i in range(len(qpos)):
        jnt_name = joint_names[i]
        out[f"{jnt_name}_qpos"] = qpos[i]

    if use_velocity:
        for i in range(len(velocity)):
            jnt_name = joint_names[i]
            out[f"{jnt_name}_vel"] = velocity[i]

    if use_torque:
        for i in range(len(torque)):
            sensor_name = sensors[i]
            out[f"{sensor_name}_torque"] = torque[i]

    if use_limits:
        for i in range(len(limits)):
            jnt_name = joint_names[i]
            out[f"{jnt_name}_lim"] = limits[i]

    if use_actuations:
        for i in range(len(env.mimo_actuators)):
            act_idx = env.mimo_actuators[i]
            act_name = env.model.actuator(act_idx).name
            out[f"{act_name}_actctrl"] = actuations[i]
            out[f"{act_name}_acttrq"] = actuations[i+len(env.mimo_actuators)]

    return out

def vesti_obs_to_dict(env, obs):
    """ Returns the vestibular observation 'obs' as a dictionary in the following 
    format:
    accelerometer_{x/y/z}
    gyroscope_{x/y/z}
    """
    obs = {
        "accelerometer_x": obs[0],
        "accelerometer_y": obs[1],
        "accelerometer_z": obs[2],
        "gyroscope_x": obs[3],
        "gyroscope_y": obs[4],
        "gyroscope_z": obs[5]
    }

    return obs

def action_to_dict(env, action):
    out = { }
    for i in range(len(env.mimo_actuators)):
        act_idx = env.mimo_actuators[i]
        act_name = env.model.actuator(act_idx).name
        out[f"{act_name}"] = action[i]

    return out

def collect_observations(env, model, n_episodes, save_file=None, pca=None, render=False):
    """ Lets the trained model 'model' play in the environment 'env' for a
    total of 'n_episodes'. Specify 'save_file' to a file location of your
    choice to save the output pandas DataFrame to that file.
    
    Returns:
        pd.DataFrame with keys
        all_obs: numpy ndarray containing all observations. For each
            step in the environment, this matrix has one row. It has as
            many columns as the shape of the observation.
    """
    all_obs_dicts = []
    all_action_dicts = []

    #proprio_valid_components = env.proprio.VALID_COMPONENTS
    #proprio_valid_components.prepend("qpos")

    for episode in range(n_episodes):
        print(f"Playing episode {episode+1} from {n_episodes}.")
        obs, _ = env.reset()
        done = False

        imgs = []
        
        proprio_obs = obs['observation']
        vesti_obs = obs['vestibular']
        proprio_dict = proprio_obs_to_dict(env, proprio_obs)
        vesti_dict = vesti_obs_to_dict(env, vesti_obs)
        potential_dict = { "potential": env.get_potential() }

        while not done:
            action, _ = model.predict(obs)

            if render:
                imgs.append(env.mujoco_renderer.render(render_mode='rgb_array'))

            if pca:
                # 2. Projiziere die Aktion in den PCA-Raum (Synergie-Raum)
                # Wir müssen raw_action kurz in 2D bringen (1, n_motors) für sklearn
                action_reshaped = action.reshape(1, -1)
                pca_space = pca.transform(action_reshaped) 

                # 3. "Ablation": Setze alle Komponenten außer der ersten auf Null
                # pca_space hat die Form (1, n_components)
                pca_space_filtered = np.zeros_like(pca_space)
                pca_space_filtered[0, 0] = pca_space[0, 0] # Nur PC1 behalten

                # 4. Zurück-Transformation in den Motor-Raum
                # Das ergibt eine Aktion, die NUR aus der ersten Synergie besteht
                action = pca.inverse_transform(pca_space_filtered)

            all_obs_dicts.append(proprio_dict | vesti_dict | potential_dict)
            all_action_dicts.append(action_to_dict(env, action))
            obs, reward, truncated, terminated, info = env.step(action)
            proprio_obs = obs['observation']
            vesti_obs = obs['vestibular']
            proprio_dict = proprio_obs_to_dict(env, proprio_obs)
            vesti_dict = vesti_obs_to_dict(env, vesti_obs)
            potential_dict = { "potential": env.get_potential() }
            
            done = truncated or terminated

            if truncated:
                print("Success!")

            elif terminated:
                print("No Success!")

            if done:
                env.reset()

                if render:
                    evaluation_video(imgs, save_name="test_render.mp4", resolution=((480, 480)))

    df = pd.DataFrame(all_obs_dicts)
    df_action = pd.DataFrame(all_action_dicts)
    return df, df_action

def collect_kobayashi_site_y_displacement_series(env, model, n_tries=10):
    """ Lets the (trained) model 'model' play in the environment 'env'
    for a total of 'n_tries' episodes. Records the y displacement of each site defined
    as kobayashi site. Collects in total a number of (n_steps + 1)
    values if the model takes 'n_steps' in the environment, because
    we record one set of value before each step, but also after the final
    step. Only records successful episodes. Tries in total 'n_tries' and if
    after that, no successful episode was done, returns 'None'.

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
        obs, _ = env.reset()
        reference_coords = get_site_absolute_displacements()
        done = False
        deg_45_reached = False
        side_lying_reached = False

        # For some reason, at the start of the episode, we do not start with
        # 'env.data.time=0'. So we simply subtract this from each following
        # time.
        time_offset = env.data.time * 1000.0
        entry = get_site_relative_displacements()
        entry['Episode'] = n_successful_episodes + 1
        entry['Time'] = env.data.time * 1000.0 - time_offset # env.data.time is in sec.
        entry['Side_Lying'] = False
        entry['45_Deg'] = False
        data.append(entry)
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, truncated, terminated, info = env.step(action)

            side_lying_reached = side_lying_reached or (info['side_lying'] == 1.0)
            deg_45_reached = deg_45_reached or (info['45_deg'] == 1.0)

            entry = get_site_relative_displacements()
            entry['Episode'] = n_successful_episodes + 1
            entry['Time'] = env.data.time * 1000.0 - time_offset # env.data.time is in sec.
            entry['Side_Lying'] = side_lying_reached
            entry['45_Deg'] = deg_45_reached
            data.append(entry)

            done = truncated or terminated

            if truncated:
                is_success = True
                print("Success!")

                for entry in data:
                    all_successful_data.append(entry)

                n_successful_episodes += 1

            elif terminated:
                print("No Success!")

    if is_success:
        return pd.DataFrame(all_successful_data)
    else:
        return None

def collect_kobayashi_displacements_all(env, date, pos, suffix):
    """ Searches for all models matching 'date', starting position 'pos' and suffix 'suffix'.
    Loads all runs of these models and plays 'collect_kobayashi_site_y_displacement_series' on them for 1 episode. """
    data = []

    # Pattern of model folder: <date>_<startingposition>_<suffix>_run_<i>
    pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_([a-z]+)_([a-z0-9_-]+)_run_(\d+)')

    for root, dirs, files in os.walk('.'):
        # root: Current folder on walk
        # dirs: Directories in 'root'
        # files: Files in 'root'
        root_name = os.path.basename(root)
        match = pattern.search(root_name)
        if not match: continue
        _date, haltung, _suffix, run_num = match.groups()

        if _date != date: continue
        if haltung != pos: continue
        if _suffix != suffix: continue

        print(f"Found run {run_num}!")

        model_file = os.path.join(os.path.abspath(root), "model_1.zip")
        model = RL.load(model_file, env)
        n_tries = 10
        df = collect_kobayashi_site_y_displacement_series(env, model, n_tries=n_tries)
        if df is None:
            print(f"Run {run_num} had no successfull episodes in {n_tries} tries. Skipping...")
            continue
        df['Run'] = run_num
        df = df.set_index(['Run', 'Episode', 'Time'])
        data.append(df)

    df = pd.concat(data)
    return df

def collect_run_statistics(env, model, n_success_episodes, n_abort):
    """ Lets the (trained) model 'model' play in the environment 'env' for as many episodes
    until it reaches 'n_success_episodes' number of successful episodes. We let him play for
    a maximum of 'n_abort' episodes, after which if we did not reach 'n_success_episodes', we
    return 'None'. Otherwise we return a pandas DataFrame containing the laterality of the
    run and if it was successful or not.
    """
    cnt_success = 0
    torso_site_id  = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "KOBAYASHI_Torso")
    data = []

    def get_frame():
        return env.mujoco_renderer.render(render_mode="rgb_array")

    for episode in range(n_abort):
        print("Playing episode...")
        #imgs = []
        done = False
        obs, _ = env.reset()
        #imgs.append(get_frame())
        time_zero_sec = env.data.time

        ref_displacement = env.data.site_xpos[torso_site_id][1]

        deg_45_reached = False
        side_lying_reached = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, success, failure, info = env.step(action)
            #imgs.append(get_frame())

            done = success or failure

            if not side_lying_reached and info['side_lying'] == 1:
                side_lying_reached = True
                time_sidelying_sec = env.data.time

            if success:
                cnt_success += 1

            if done:
                entry = {}
                entry['Episode'] = episode+1
                entry['Success'] = success

                torso_displacement = env.data.site_xpos[torso_site_id][1]
                relative_displacement = torso_displacement - ref_displacement
                entry['Left_Roll'] = relative_displacement >= 0

                time_finish_sec = env.data.time
                entry['Time'] = (time_finish_sec - time_zero_sec) * 1000.0  # ms

                if side_lying_reached:
                    entry['Time_SideLying'] = (time_sidelying_sec - time_zero_sec) * 1000.0 # ms
                else:
                    entry['Time_SideLying'] = 0

                data.append(entry)

                #if success and entry['Time'] > 4000.0:
                #    print("We got one! Saving as video...")
                #    save_name=os.path.join('.', 'long_episode.avi')
                #    render_height = 480
                #    render_width = 480
                #    evaluation_video(imgs, save_name=save_name, resolution=((render_width, render_height)))
                #    raise ValueError

                if cnt_success >= n_success_episodes:
                    return pd.DataFrame(data)
                
    return None

def collect_run_statistics_all(env, date, pos, suffix):
    """ Searches for all models matching 'date', starting position 'pos' and suffix 'suffix'.
    Loads all runs of these models and plays 'collect_run_statistics' on them for 10 episodes. """
    data = []

    # Pattern of model folder: <date>_<startingposition>_<suffix>_run_<i>
    pattern = re.compile(r'(\d{2}-\d{2}-\d{2})_([a-z]+)_([a-z0-9_-]+)_run_(\d+)')

    for root, dirs, files in os.walk('.'):
        # root: Current folder on walk
        # dirs: Directories in 'root'
        # files: Files in 'root'
        root_name = os.path.basename(root)
        match = pattern.search(root_name)
        if not match: continue
        _date, haltung, _suffix, run_num = match.groups()

        if _date != date: continue
        if haltung != pos: continue
        if _suffix != suffix: continue

        print(f"Found run {run_num}!")

        model_file = os.path.join(os.path.abspath(root), "model_1.zip")
        model = RL.load(model_file, env)
        n_success_episodes=10
        n_abort=40
        run_stats = collect_run_statistics(env, model, n_success_episodes=n_success_episodes, n_abort=n_abort)
        if run_stats is None:
            print(f"Run {run_num} did not reach {n_success_episodes} successful episodes in {n_abort} tries. Skipping...")
            continue
        run_stats['Run'] = run_num
        run_stats = run_stats.set_index(['Run', 'Episode'])
        data.append(run_stats)

    df = pd.concat(data)
    return df

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
		pen_factor=0.04,
        goal_function='cos',
        achieved_goal_in_observation=False,
        pbrs=True,
		#proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
        isr=False)
    
    model = RL.load(args.load_model, env)
    
    print("Collecting observations for 1 episode...")
    df, df_action = collect_observations(env, model, 1)
    # Korrelation berechnen
    correlations = df.corr()['potential'].drop('potential').sort_values(ascending=False)

    print(correlations)

    A = df_action.values

    from sklearn.decomposition import PCA

    pca_act = PCA().fit(A)

    print("Synergien:")
    print(pca_act.explained_variance_ratio_)

    # Plotte die erklärte Varianz
    import matplotlib.pyplot as plt
    plt.plot(np.cumsum(pca_act.explained_variance_ratio_))
    plt.xlabel('Anzahl der Synergien (PCs)')
    plt.ylabel('Kumulative erklärte Varianz')
    plt.axhline(y=0.9, color='r', linestyle='--') # 90% Marke
    plt.show()

    # 3. Die Gewichte (Loadings) der ersten Komponente extrahieren
    pc1_loadings = pca_act.components_[0]

    # 4. In ein schönes Pandas-Objekt mit Namen umwandeln
    loadings_series = pd.Series(pc1_loadings, index=df_action.columns)

    # 5. Nach Absolutwert sortieren (Wichtigkeit ist egal ob positiv oder negativ)
    top_motors_pc1 = loadings_series.abs().sort_values(ascending=False)

    print("Top 10 Motoren in Synergie 1 (PC1):")
    print(loadings_series[top_motors_pc1.index[:10]])

    # Projiziere die originalen Aktionen auf die Hauptkomponenten
    # 'scores' hat die Form (8000, 5)
    scores = pca_act.transform(df_action)

    # Plotte den Verlauf der ersten Synergie für die erste Episode (z.B. erste 200 Schritte)
    plt.figure(figsize=(10, 4))
    plt.plot(scores[:200, 0], label='Aktivität Synergie 1 (PC1)')
#plt.plot(scores[:200, 1], label='Aktivität Synergie 2 (PC2)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("Wann ist die Haupt-Synergie aktiv?")
    plt.xlabel("Zeitschritte")
    plt.ylabel("Intensität")
    plt.legend()
    plt.show()
    collect_observations(env, model, 1, pca_act, render=True)
    exit
    print(f"Training model with found PC1")
    env_pca = gym.make("MIMoRollOver-v0",
        starting_position='supine',
        actuation_model=SpringDamperModel_PC1,
        pca=pca_act,
        width=480, # always 480 regardless whether we render actuations or not.
        height=480,
        render_mode='rgb_array',
        touch_params=None,
        nopen=False,
		pen_factor=0.04,
        goal_function='cos',
        achieved_goal_in_observation=False,
        pbrs=True,
		#proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
        isr=False)
    
    model = RL("MultiInputPolicy", env_pca,
            tensorboard_log='.',
            learning_rate=3e-3,
            verbose=1)
    
    model.learn(total_timesteps=1000000, reset_num_timesteps=False)
    


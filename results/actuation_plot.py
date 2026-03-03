""" This file is used to plot one episode of MIMo's actuations. We group
actuators into limbs (IA, CA, IL, CL, TR) just like in Kobayashi '16."""
import argparse
import gymnasium as gym
import mimoEnv
from mimoActuation.actuation import SpringDamperModel
from stable_baselines3 import PPO as RL
import matplotlib.pyplot as plt
import numpy as np

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

def collect_actuations(env, model, n_episodes=1):
    """ Lets the (trained) model 'model' play in the environment 'env'
    for one episode. Records the actuations in each step and groups
    them into the limbs above. Writes the average actuation into a
    pandas DataFrame with keys 'IA', 'IL', 'CA', 'CL', 'TR' and 'HP'
    ('HP' is made up - it stands for 'hip'). 
    """

    # Prepare by collecting id for each site. This list at index i then contains the id
    # of site in KOBAYASHI_SITES at index i.
    kobayashi_site_ids = []

    for site in KOBAYASHI_SITES:
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site)
        kobayashi_site_ids.append(site_id)

    # Time [ms] from onset until reaching this step.
    time_from_onset_ms = []
    # List of dictionaries - ordered by steps in the environment.
    data = []

    # Dictionary of reference coordinates to calculate relative displacement
    # of site. Set after the first 'env.reset()'.
    reference_coords = None

    def get_site_absolute_displacements():
        """ Returns the absolute displacements, i.e.
        the displacement from the global origin. """
        vals = {}
        for i in range(len(KOBAYASHI_SITES)):
            site_id = kobayashi_site_ids[i]
            site_y = env.data.site_xpos[site_id][1]
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

    for episode in range(n_episodes):
        print(f"Playing episode {episode+1} from {n_episodes}.")
        obs, _ = env.reset()
        reference_coords = get_site_absolute_displacements()
        done = False

        while not done:
            action, _ = model.predict(obs)
            data.append(get_site_relative_displacements())
            time_from_onset_ms.append(env.data.time * 1000.0)  # 'env.data.time' is in sec
            obs, reward, truncated, terminated, info = env.step(action)
            
            done = truncated or terminated

            if truncated:
                print("Success!")

            elif terminated:
                print("No Success!")

            if done:
                frame = env.mujoco_renderer.render(render_mode='rgb_array', camera_name='top')
                img = Image.fromarray(frame)
                img.save(os.path.join('.', f'test.png'))
                data.append(get_site_relative_displacements())
                time_from_onset_ms.append(env.data.time * 1000.0)  # 'env.data.time' is in sec.

    # For some reason, we get the first entry in 'time_from_onset_ms' = 15ms > 0ms. This is weird,
    # but it means we do not start at 0. To start at 0ms, we simply subtract this offset from
    # all values.
    time_from_onset_ms = np.array(time_from_onset_ms)
    offset = time_from_onset_ms[0]
    time_from_onset_ms -= offset
    df = pd.DataFrame(data, index=time_from_onset_ms)
    df.index.name = 'Time from Onset [ms]'
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', required=False, type=str)
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

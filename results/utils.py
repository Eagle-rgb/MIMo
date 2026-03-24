import mimoEnv
from mimoActuation.actuation import SpringDamperModel
import gymnasium as gym

def make_env(age, starting_position='supine', pen_fac=0.02):
    """ Creates and returns the roll over env. """
    env = gym.make("MIMoRollOver-v0", actuation_model=SpringDamperModel,
        starting_position=starting_position,
        width=480, # always 480 regardless whether we render actuations or not.
        height=480,
        render_mode='rgb_array',
        touch_params=None,
        nopen=False,
        pen_factor=pen_fac,
        goal_function='cos',
        achieved_goal_in_observation=False,
        pbrs=True,
        age=age,
        #proprio_params=PROPRIOCEPTION_PARAMS_ONLY_QPOS,
        isr=False)
    
    return env
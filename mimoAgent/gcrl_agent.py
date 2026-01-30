"""
Docstring for mimoAgent.gcrl_agent
"""
import numpy as np

class GCRL_Agent:
    def __init__(self, env):
        self.env = env
        self.goals_inited=False

    def init_goals(self):
        action = np.zeros(self.env.action_space.shape)

        # Once for the current starting position and once for the opposite starting
        # position.
        for _ in range(2):
            self.env.reset()
            for _ in range(20):
                obs, _, _, _, _ = self.env.step(action)
            
            if self.env.starting_position == 'prone':
                self.prone_intrinsic_goal = obs.copy()
                self.starting_position = 'supine'

            else: # supine
                self.supine_intrinsic_goal = obs.copy()
                self.env.starting_position = 'prone'

        self.goals_inited = True

    def 
        
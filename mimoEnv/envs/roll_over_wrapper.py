import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import os
import mimoEnv.utils as env_utils

class MIMoRollOverWrapper(gym.Wrapper):
    def __init__(self, env, log_file="actuation_log.csv"):
        super().__init__(env)
        self.log_file = log_file
        self.episode_counter = 0
        self.step_counter = 0

        # header row
        acts = env_utils.get_actuator_joint_map(env.model)
        act_names = []
        for entry in acts:
            act_names.append(entry['act_name'])

        self.header = "episode,step,time," + ",".join(act_names) + "\n"

        # Open log file on initialization.
        if not os.path.exists(self.log_file) or os.stat(self.log_file).st_size == 0:
            with open(self.log_file, "w") as f:
                f.write(self.header)

        self.log_handle = open(self.log_file, 'a') # a: append

    def __del__(self):
        if self.log_handle:
            self.log_handle.close()

    def step(self, action):
        # 1. Exectue the actual environment step.
        obs, reward, terminated, truncated, info =\
            self.env.step(action)

        if self.log_handle is None:
            return obs, reward, terminated, truncated, info
  
        # 2. Log muscle actuations.
        act_values = env_utils.get_actuation_values(self.env.model, self.env.data)
        act_values_str = ",".join([f"{entry['actuation']:.6f}" for entry in act_values])
        self.step_counter += 1

        log_entry = (
            f"{self.episode_counter},"
            f"{self.step_counter},"
            f"{time.time()},"
            f"{act_values_str}\n"
        )

        self.log_handle.write(log_entry)

        # flush (optional, but good to get instant logs).
        self.log_handle.flush()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_counter += 1
        self.step_counter = 0
        return self.env.reset(**kwargs)

    def close(self):
        self.log_handle.write("finished\n")
        self.log_handle.close()
        self.log_handle = None
        self.env.close()
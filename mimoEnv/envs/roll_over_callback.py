""" This file is used to log achieved hip & chest angle as a mean over many
episodes and the achieved side lying success rate.
Alongside this, you have the option to save an intermediate model at reaching
50% side_lying success rate. """

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np
import os

class RollOverCallback(BaseCallback):
    def __init__(self, save_intermediate=False, save_dir=None):
        super().__init__()
        self.end_hip_deg = None
        self.end_chest_deg = None
        self.side_lying_success = None
        self.save_intermediate = save_intermediate
        self.save_dir = save_dir

    def _on_training_start(self):
        window_size = self.model._stats_window_size
        print(f"Using RollOverCallback with window_size {window_size}.")
        self.end_hip_deg = deque(maxlen=window_size)
        self.end_chest_deg = deque(maxlen=window_size)
        self.side_lying_success = deque(maxlen=window_size)
        self.intermediate_saved = False

    def _on_step(self) -> bool:
        # We are in a DummyVecEnv
        for i, info in enumerate(self.locals["infos"]):
            # Is the episode finished?
            if "episode" in info:
                # Call environment function to get hip and chest angle.
                hip = info['hip_deg']
                chest = info['chest_deg']
                side_lying_success = info['side_lying']

                self.end_hip_deg.append(hip)
                self.end_chest_deg.append(chest)
                self.side_lying_success.append(side_lying_success)

                self.logger.record("rollout/ep_end_hip_deg_mean", np.mean(self.end_hip_deg))
                self.logger.record("rollout/ep_end_chest_deg_mean", np.mean(self.end_chest_deg))
                self.logger.record("rollout/side_lying_success_rate", np.mean(self.side_lying_success))

        # Save intermediate model - if specified.
        # We need the 'len(self.side....) > 0' to prevent in the first step callbacks accessing
        # the empty object.
        if len(self.side_lying_success) > 0 and self.save_intermediate and not self.intermediate_saved and self.save_dir is not None:
            if np.mean(self.side_lying_success) > 0.5:
                self.intermediate_saved = True
                # Save the model ...
                self.model.save(os.path.join(self.save_dir, "model_intermediate"))
        return True
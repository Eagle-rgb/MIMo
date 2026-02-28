from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class HipChestAngleLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.end_hip_deg = None
        self.end_chest_deg = None
        self.side_lying_success = None

    def _on_training_start(self):
        window_size = self.model._stats_window_size
        print(f"Using HipChestAngleLogger with window_size {window_size}.")
        self.end_hip_deg = deque(maxlen=window_size)
        self.end_chest_deg = deque(maxlen=window_size)
        self.side_lying_success = deque(maxlen=window_size)

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
        return True
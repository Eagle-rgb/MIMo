""" This file is used to log achieved hip & chest angle as a mean over many
episodes and the achieved side lying success rate.
Alongside this, you have the option to save an intermediate model at reaching
90% side_lying success rate. """

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
        self.raw_ctrl_cost = None

    def _on_training_start(self):
        window_size = self.model._stats_window_size
        print(f"Using RollOverCallback with window_size {window_size}.")
        self.end_hip_deg = deque(maxlen=window_size)
        self.end_chest_deg = deque(maxlen=window_size)
        self.side_lying_success = deque(maxlen=window_size)
        self.intermediate_90_saved = False
        self.raw_ctrl_cost = deque(maxlen=window_size)
        # Highest rotation reached anywhere in the episode, as opposed to 'side_lying' below,
        # which only looks at the final step. This is the quantity 'eval_rollover.py' reports, so
        # logging it makes the training curve comparable to the evaluation numbers.
        self.episode_rho_max = deque(maxlen=window_size)

        # Aggregated raw control cost in an episode. Is reset to 0 after each episode. Mean
        # is calculated by dividing the sum by 'self.episode_stp_cnt' - a counter of how many
        # entries we have added to 'aggr_raw_ctrl_cost_in_episode'.
        self.aggr_raw_ctrl_cost_in_episode=0
        self.episode_stp_cnt=0

    def _on_step(self) -> bool:
        # We are in a DummyVecEnv
        for i, info in enumerate(self.locals["infos"]):
            self.aggr_raw_ctrl_cost_in_episode += info['raw_ctrl_cost']
            self.episode_stp_cnt += 1

            # Is the episode finished?
            if "episode" in info:
                # Call environment function to get hip and chest angle.
                hip = info['hip_deg']
                chest = info['chest_deg']
                side_lying_success = info['side_lying']

                # Calculate mean control cost in this episode and reset aggregation and step count.
                if self.episode_stp_cnt > 0:
                    self.raw_ctrl_cost.append(self.aggr_raw_ctrl_cost_in_episode / self.episode_stp_cnt)
                else:
                    self.raw_ctrl_cost.append(0)

                self.aggr_raw_ctrl_cost_in_episode = 0
                self.episode_stp_cnt = 0

                self.end_hip_deg.append(hip)
                self.end_chest_deg.append(chest)
                self.side_lying_success.append(side_lying_success)

                self.logger.record("rollout/ep_end_hip_deg_mean", np.mean(self.end_hip_deg))
                self.logger.record("rollout/ep_end_chest_deg_mean", np.mean(self.end_chest_deg))
                self.logger.record("rollout/side_lying_success_rate", np.mean(self.side_lying_success))
                self.logger.record("rollout/raw_ctrl_cost", np.mean(self.raw_ctrl_cost))

                if 'episode_rho_max' in info:
                    self.episode_rho_max.append(info['episode_rho_max'])
                    self.logger.record("rollout/ep_rho_max_mean", np.mean(self.episode_rho_max))
                # Upper end of the goal range actually in use. Constant without
                # '--goal_curriculum'; with it, this curve is the curriculum itself and shows
                # whether it advanced or stalled.
                if not np.isnan(info.get('goal_high_effective', np.nan)):
                    self.logger.record("rollout/goal_high_effective", info['goal_high_effective'])

        # Save intermediate model - if specified.
        # We need the 'len(self.side....) > 0' to prevent in the first step callbacks accessing
        # the empty object.
        if len(self.side_lying_success) > 0 and self.save_intermediate and not self.intermediate_90_saved and self.save_dir is not None:
            if np.mean(self.side_lying_success) > 0.9:
                self.intermediate_90_saved = True
                # Save the model ...
                self.model.save(os.path.join(self.save_dir, "model_intermediate_90"))
        return True
from stable_baselines3.common.callbacks import BaseCallback

class HipChestAngleLogger(BaseCallback):
    def _on_step(self) -> bool:
        # We are in a DummyVecEnv
        for i, info in enumerate(self.locals["infos"]):
            # Is the episode finished?
            if "episode" in info:
                # Call environment function to get hip and chest angle.
                hip = self.training_env.env_method("get_achieved_rotation_degrees", "hip")[i]
                chest = self.training_env.env_method("get_achieved_rotation_degrees", "chest")[i]

                self.logger.record("rollout/ep_end_hip_deg_mean", hip)
                self.logger.record("rollout/ep_end_chest_deg_mean", chest)
        return True
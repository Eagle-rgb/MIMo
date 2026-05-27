from stable_baselines3.common.callbacks import BaseCallback

class ISRCallback(BaseCallback):
    """ Callback that disables ISR (Initial State Randomization)
    if we reached 75% of training. """

    def __init__(self, total_timesteps, isr_relative_turnoff=0.75):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.isr_absolute_turnoff = total_timesteps * isr_relative_turnoff

        # flag to remember one-time trigger.
        self.isr_turnoff_triggered = False

    def _on_step(self) -> bool:
        if self.isr_turnoff_triggered:
            return True
        
        if any(self.locals["dones"]): 
            if self.num_timesteps >= self.isr_absolute_turnoff:
                print(f"Disabling ISR...")
                self.training_env.env_method("disable_isr")

        return True

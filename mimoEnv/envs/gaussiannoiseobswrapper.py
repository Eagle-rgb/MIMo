import gymnasium as gym
import numpy as np

class GaussianNoiseObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.01, target_keys=None):
        super().__init__(env)
        self.noise_std = noise_std
        # If 'None', apply to all target keys.
        self.target_keys = target_keys

    def observation(self, observation):
        noisy_observation = observation.copy()
        
        # If dict space
        if isinstance(self.observation_space, gym.spaces.Dict):
            keys = self.target_keys if self.target_keys else observation.keys()
            for key in keys:
                box = self.observation_space[key]
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=observation[key].shape)
                noisy_val = observation[key] + noise
                noisy_observation[key] = np.clip(noisy_val, box.low, box.high)
        else:
            # If box space.
            box = self.observation_space
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=observation.shape)
            noisy_observation = np.clip(observation + noise, box.low, box.high)
            
        return noisy_observation
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

        # Buffer of aggregated (sum) control cost in episodes. Before, we stored the mean control cost over an episode,
        # but this favors long episode and so it skews the control cost result so that policies rolling over longer
        # episodes have less control cost in the log.
        self.raw_ctrl_cost = deque(maxlen=window_size)
        # Highest rotation reached anywhere in the episode, as opposed to 'side_lying' below,
        # which only looks at the final step. This is the quantity 'eval_rollover.py' reports, so
        # logging it makes the training curve comparable to the evaluation numbers.
        self.episode_rho_max = deque(maxlen=window_size)

        # Aggregated raw control cost in an episode. Is reset to 0 after each episode.
        self.aggr_raw_ctrl_cost_in_episode=0

    def _on_step(self) -> bool:
        # We are in a DummyVecEnv
        for i, info in enumerate(self.locals["infos"]):
            self.aggr_raw_ctrl_cost_in_episode += info['raw_ctrl_cost']

            # Is the episode finished?
            if "episode" in info:
                # Call environment function to get hip and chest angle.
                hip = info['hip_deg']
                chest = info['chest_deg']
                side_lying_success = info['side_lying']

                # Aggregated control cost of this episode, then reset the aggregation. No step
                # count is involved any more: this is deliberately the sum, not the mean, because
                # the mean rewarded long episodes (see the note in '_on_training_start'). The
                # 'episode_stp_cnt > 0' guard that used to sit here was left behind by that
                # change and read an attribute that no longer exists, crashing at the first
                # episode end.
                self.raw_ctrl_cost.append(self.aggr_raw_ctrl_cost_in_episode)

                self.aggr_raw_ctrl_cost_in_episode = 0

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

        # Save intermediate model - if specified.
        # We need the 'len(self.side....) > 0' to prevent in the first step callbacks accessing
        # the empty object.
        if len(self.side_lying_success) > 0 and self.save_intermediate and not self.intermediate_90_saved and self.save_dir is not None:
            if np.mean(self.side_lying_success) > 0.9:
                self.intermediate_90_saved = True
                # Save the model ...
                self.model.save(os.path.join(self.save_dir, "model_intermediate_90"))
        return True

class RollOverEvalCallback(BaseCallback):
    """ Periodic deterministic evaluation under the protocol from 'eval_rollover.py'.

    Why this exists rather than SB3's own EvalCallback: SB3 scores the mean episode *reward*,
    which under a sparse {0, -1} reward is dominated by how long MIMo took, and under HER is
    computed against sampled goals. Neither is the milestone. This measures rho_max -- the
    highest rotation reached anywhere in the episode -- against the pinned goal of 0.95, exactly
    as the reported evaluation does, so the training curve and the reported number are the same
    quantity.

    It also fixes a measurement gap: checkpoints every 200k sample the run at five points, and
    runs were observed peaking between them (one seed's optimum sat at 400k). The best model is
    saved whenever rho_max improves, so the peak is kept rather than reconstructed afterwards.

    The evaluation environment must be built with the protocol already applied -- ISR off, goal
    pinned, episodes not terminating on success -- and must be the unwrapped environment, since
    the horizon is enforced here.

    Args:
        eval_env: Unwrapped MIMoRollOverEnv configured per the evaluation protocol.
        eval_every (int): Evaluate every this many training steps.
        n_episodes (int): Episodes per evaluation.
        save_dir (str): Where to write 'model_best.zip'. No best-model saving if None.
        episode_steps (int): Episode horizon, matching the training environment.
        seed0 (int): Episodes use seeds seed0..seed0+n_episodes-1, so every evaluation sees the
            same start states and the curve is not noise from varying initial poses.
    """

    ROLL_THRESHOLD = 0.95
    SIDE_LYING_THRESHOLD = 0.5

    def __init__(self, eval_env, eval_every, n_episodes=20, save_dir=None, episode_steps=500,
                 seed0=1000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.n_episodes = n_episodes
        self.save_dir = save_dir
        self.episode_steps = episode_steps
        self.seed0 = seed0
        self.best_rho = -np.inf
        self.best_step = None
        self._next_eval = None

    def _on_training_start(self):
        # Anchor on the current step count: 'train()' calls learn() once per checkpoint segment
        # with reset_num_timesteps=False, so num_timesteps keeps rising across segments.
        if self._next_eval is None:
            self._next_eval = self.model.num_timesteps + self.eval_every
        print(f"Using RollOverEvalCallback: {self.n_episodes} episodes every "
              f"{self.eval_every} steps, horizon {self.episode_steps}.")

    def _run_episodes(self):
        rho_max, rolled, side, first = [], [], [], []
        for episode in range(self.n_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed0 + episode)
            best = float(self.eval_env.get_achieved_goal_cos()[0])
            step, hit, done = 0, None, False
            while not done and step < self.episode_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                step += 1
                best = max(best, float(self.eval_env.get_achieved_goal_cos()[0]))
                if hit is None and best >= self.ROLL_THRESHOLD:
                    hit = step
                done = terminated or truncated
            rho_max.append(best)
            rolled.append(1.0 if best >= self.ROLL_THRESHOLD else 0.0)
            side.append(1.0 if best >= self.SIDE_LYING_THRESHOLD else 0.0)
            first.append(hit if hit is not None else np.nan)
        return np.array(rho_max), np.array(rolled), np.array(side), np.array(first, dtype=float)

    def _on_step(self) -> bool:
        if self.eval_every <= 0 or self.model.num_timesteps < self._next_eval:
            return True
        self._next_eval = self.model.num_timesteps + self.eval_every

        rho_max, rolled, side, first = self._run_episodes()
        self.logger.record("eval/rho_max_mean", float(rho_max.mean()))
        self.logger.record("eval/rho_max_min", float(rho_max.min()))
        self.logger.record("eval/roll_rate", float(rolled.mean()))
        self.logger.record("eval/side_lying_rate", float(side.mean()))
        finished = first[~np.isnan(first)]
        if finished.size:
            self.logger.record("eval/steps_to_roll", float(finished.mean()))

        # Keep the peak. Runs collapse late without warning -- one seed went from 100 % roll at
        # 800k to 0 % at 1e6 -- so the final model is not reliably the run's result.
        if rho_max.mean() > self.best_rho:
            self.best_rho = float(rho_max.mean())
            self.best_step = self.model.num_timesteps
            if self.save_dir is not None:
                self.model.save(os.path.join(self.save_dir, "model_best"))
        self.logger.record("eval/rho_max_best", self.best_rho)
        return True

    def _on_training_end(self):
        if self.best_step is not None:
            print(f"RollOverEvalCallback: best eval rho_max {self.best_rho:.3f} at step "
                  f"{self.best_step}" + (f", saved as model_best.zip" if self.save_dir else ""))



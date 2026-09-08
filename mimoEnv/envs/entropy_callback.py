""" Scheduled entropy penalty for on-policy algorithms. """

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class EntropyPenaltyCallback(BaseCallback):
    """ Ramps a NEGATIVE ``ent_coef`` in over the second part of training.

    08.09.2026 Why this exists. SB3's PPO builds its loss as

        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss,
        entropy_loss = -mean(entropy)

    with ``ent_coef = 0.0`` by default, so **PPO gets no entropy bonus at all** and there is
    nothing to "turn down": the only way to push the policy's exploration noise down is a
    negative coefficient, i.e. an entropy *penalty*. Measured on the muscle roll-over runs of
    06.09.2026, ``train/std`` sits at 0.91-0.96 after 1M steps (against 0.65 for the
    spring-damper) and the stochastic rollout scores far better than the deterministic policy --
    ``rollout/ep_rho_max_mean`` 0.805 against ``eval_rollover.py`` 0.298 at ``--pen_factor=20``.
    The noise is doing the rolling, not the policy.

    Two properties make a *scheduled* penalty the right shape, rather than a constant one:

    * The muscle task needs the exploration early. Side-lying first appears at 300-400k steps in
      every group; a policy that has already committed by then has nothing to commit to.
    * ``ent_coef`` is typed ``float`` in SB3 -- unlike ``learning_rate`` and ``clip_range`` it
      accepts no schedule. But ``PPO.train`` reads ``self.ent_coef`` on every update, so setting
      it from a callback between rollouts is a schedule in all but name.

    **What sets the usable magnitude is the gradient on ``log_std``, not the loss value.** For a
    diagonal Gaussian :math:`H = \\sum_i [\\frac{1}{2}\\ln(2\\pi e) + \\log\\sigma_i]`, so
    :math:`\\partial H / \\partial \\log\\sigma_i = 1` exactly, independent of sigma -- the penalty
    puts a *constant* gradient of ``|ent_coef|`` on every ``log_std``. It therefore competes with
    the policy gradient once ``|ent_coef|`` reaches ``|d policy_loss / d log_std|``, and that
    quantity has to be measured, not inferred from the loss terms.

    Measured 08.09.2026 over one 4096-step PPO rollout on the muscle roll-over env (92 muscles,
    PBRS, ``--pen_metabolic --pen_factor=50``):

    * ``|d policy_loss / d log_std|`` = **0.0144** (median over minibatches, p90 0.024)
    * on Pendulum-v1, for comparison, the same quantity is 0.0015 -- a factor 100 apart, so a
      coefficient tuned on a toy task transfers to nothing.

    So the band that does anything here is around **-0.005 to -0.03**. An earlier estimate in
    this file put it at -1e-5..-1e-4, derived from the ratio of the loss *terms*
    (``ent_coef * entropy_loss`` against ``policy_gradient_loss``); that reasoning was wrong and
    the values are ~100x too small to move the policy at all.

    The effect is also gentler than a naive Adam argument suggests. A constant coefficient swept
    on Pendulum over 40k steps moved sigma by at most 4 %:

    ======================  =====================
    constant ``ent_coef``   sigma after 40k steps
    ======================  =====================
    0                       0.8925
    -1e-4                   0.8895
    -1e-3                   0.8913
    -3e-3                   0.8853
    -1e-2                   0.8589
    ======================  =====================

    Monotone in the right direction, but no runaway even at ~7x the policy gradient: the PPO
    objective itself pushes sigma back up as it shrinks, because a narrower policy lowers the
    probability of the high-advantage actions exploration found. ':attr:`.std_floor`' is
    therefore a cheap safety net rather than a routinely needed brake -- but it stays, because
    PPO does **not** clamp ``log_std`` (SAC clamps to [-20, 2];
    ``DiagGaussianDistribution`` holds a bare ``nn.Parameter``) and ``max_grad_norm=0.5`` gives
    no protection either: 92 muscles contribute only ``sqrt(92) * |ent_coef|`` to the global norm.

    Args:
        total_timesteps (int): The run's full budget, used to turn ``num_timesteps`` into a
            progress fraction. Must be the total across all ``model.learn`` calls -- 'train()'
            loops over 'save_every' chunks with ``reset_num_timesteps=False``, so
            ``num_timesteps`` accumulates across them.
        ent_coef (float): The coefficient to reach at the end of training. Must be negative;
            a positive value would be an entropy *bonus*, which is not what this class is for.
        start_fraction (float): Fraction of training before the penalty starts ramping in. The
            coefficient is 0 below it and interpolates linearly from 0 to 'ent_coef' above it.
        std_floor (float): Safety brake. Once the policy's mean standard deviation falls below
            this, the penalty is switched off permanently -- there is no lower clamp on
            ``log_std`` and a runaway ends in a degenerate deterministic policy.
    """

    def __init__(self, total_timesteps, ent_coef, start_fraction=0.5, std_floor=0.1):
        super().__init__()
        if ent_coef >= 0.0:
            raise ValueError(
                f"'ent_coef' must be negative -- this callback exists to *reduce* entropy, and "
                f"SB3's default of 0.0 already means 'no entropy term'. Got {ent_coef}.")
        if not 0.0 <= start_fraction < 1.0:
            raise ValueError(f"'start_fraction' must be in [0, 1), got {start_fraction}.")
        if std_floor <= 0.0:
            raise ValueError(f"'std_floor' must be positive, got {std_floor}.")
        self.total_timesteps = total_timesteps
        self.target_ent_coef = ent_coef
        self.start_fraction = start_fraction
        self.std_floor = std_floor
        self.floor_hit = False

    def _on_training_start(self) -> None:
        # Guard rather than warn: on SAC/TD3/DDPG 'model.ent_coef' is not what the loss reads
        # (SAC keeps 'ent_coef_tensor'/'log_ent_coef' and tunes them itself), so attaching this
        # to an off-policy algorithm would be a silent no-op.
        log_std = getattr(self.model.policy, "log_std", None)
        if not isinstance(log_std, th.nn.Parameter):
            raise TypeError(
                f"{type(self).__name__} needs an on-policy algorithm whose policy carries a "
                f"'log_std' parameter (PPO or A2C with a Box action space). "
                f"{type(self.model).__name__} does not; its entropy coefficient is tuned "
                f"internally and setting 'model.ent_coef' would do nothing.")
        print(f"Using EntropyPenaltyCallback: ent_coef 0 -> {self.target_ent_coef:g} between "
              f"{self.start_fraction:.0%} and 100% of {self.total_timesteps} steps, "
              f"std floor {self.std_floor}.")

    def _current_std(self):
        with th.no_grad():
            return float(self.model.policy.log_std.exp().mean())

    def _on_rollout_end(self) -> None:
        std = self._current_std()

        if not self.floor_hit and std < self.std_floor:
            self.floor_hit = True
            print(f"EntropyPenaltyCallback: train/std fell to {std:.4f} < {self.std_floor} at "
                  f"{self.num_timesteps} steps. Disabling the entropy penalty for the rest of "
                  f"the run.")

        if self.floor_hit:
            coefficient = 0.0
        else:
            progress = np.clip(self.num_timesteps / max(self.total_timesteps, 1), 0.0, 1.0)
            ramp = (progress - self.start_fraction) / max(1.0 - self.start_fraction, 1e-8)
            coefficient = self.target_ent_coef * float(np.clip(ramp, 0.0, 1.0))

        self.model.ent_coef = coefficient
        # The coefficient actually in force this update, so the schedule is visible next to
        # 'train/entropy_loss' and 'train/std' rather than having to be inferred from the flags.
        self.logger.record("train/ent_coef", coefficient)

    def _on_step(self) -> bool:
        return True

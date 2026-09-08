from stable_baselines3.common.callbacks import BaseCallback
import random

import numpy as np

AGES = [1, 3, 6, 9]
""" The default age ladder, and the one every stored MGC run used.

08.09.2026 The comment here used to read "Fest durch verfuegbare XML-Dateien", and it was true:
'set_embodiment' loaded one of 16 pre-generated scenes. It now grows MIMo in memory
(:mod:`mimoGrowth.spec`), so the ladder is free -- see :func:`age_ladder`. This list stays the
default so that an MGC run started today is comparable with the ones on disk.
"""

DEFAULT_TOTAL_STEPS = 1_000_000
""" Budget the four default phases of 250k steps add up to.

Used to derive 'phase_steps' when a custom number of stages is asked for, so that
'--mgc_stages=30' spreads the same budget over 30 phases instead of running 30 x 250k.
"""


def age_ladder(stages=None):
    """ The ages an embodiment curriculum steps through, youngest first.

    Arguments:
        stages (int|None): How many stages. ``None`` (the default) gives :data:`AGES` verbatim,
            which keeps a run comparable with everything on disk. Any other value spreads that
            many ages evenly over the same interval, so ``stages=30`` approximates continuous
            growth from 1 to 9 months.

    Returns:
        list[float]: The ages, ascending.

    Raises:
        ValueError: If fewer than two stages are asked for.
    """
    if stages is None:
        return list(AGES)
    if stages < 2:
        raise ValueError(f"An age curriculum needs at least two stages, got {stages}.")
    return [float(age) for age in np.linspace(AGES[0], AGES[-1], stages)]


class _MorphologicalCurriculumBase(BaseCallback):
    """ Abstract base for all morphological-growth-curriculum MGC callbacks.

    Arguments:
        phase_steps (int): Steps per stage.
        stages (int|None): Number of stages, see :func:`age_ladder`. ``None`` uses :data:`AGES`.
        verbose (int): 1 prints every swap.
    """

    def __init__(self, phase_steps: int = 250_000, stages=None, verbose: int = 1):
        super().__init__(verbose)
        self.phase_steps = phase_steps
        self.ages = age_ladder(stages)
        self.current_age: float | None = None

    def _get_age_for_step(self, step: int) -> float:
        raise NotImplementedError

    def _on_training_start(self) -> None:
        self._apply_embodiment(self._get_age_for_step(0))

    def _on_step(self) -> bool:
        # Change embodiment only after an episode is
        # completed.
        if any(self.locals["dones"]):
            new_age = self._get_age_for_step(self.num_timesteps)
            if new_age != self.current_age:
                self._apply_embodiment(new_age)
        return True

    def _apply_embodiment(self, age: float) -> None:
        self.current_age = age
        if self.verbose >= 1:
            print(f"[MGC] Step {self.num_timesteps:,} -> set_embodiment({age:g}M)")
        self.training_env.env_method("set_embodiment", age, age)


class MorphologicalGrowthCurriculum(_MorphologicalCurriculumBase):
    """ Youngest to oldest -- 1M -> 3M -> 6M -> 9M by default, each `phase_steps` steps. """

    def _get_age_for_step(self, step: int) -> float:
        idx = min(step // self.phase_steps, len(self.ages) - 1)
        return self.ages[idx]


class InverseMorphologicalCurriculum(_MorphologicalCurriculumBase):
    """ Oldest to youngest -- 9M -> 6M -> 3M -> 1M by default, each `phase_steps` steps. """

    def _get_age_for_step(self, step: int) -> float:
        idx = min(step // self.phase_steps, len(self.ages) - 1)
        return self.ages[-(idx + 1)]


class StochasticAgeCurriculum(_MorphologicalCurriculumBase):
    """ Sets a uniform random age out of the ladder each `interval` steps. """

    def __init__(self, interval: int = 20_000, stages=None, verbose: int = 1):
        super().__init__(phase_steps=interval, stages=stages, verbose=verbose)
        self._next_age: float = self.ages[0]

    def _get_age_for_step(self, step: int) -> float:
        if step % self.phase_steps == 0:
            self._next_age = random.choice(self.ages)
        return self._next_age


def make_curriculum_callback(args) -> _MorphologicalCurriculumBase | None:
    stages = getattr(args, "mgc_stages", None)
    # With a custom stage count the same budget is divided over more phases, so '--mgc_stages=30'
    # is a finer curriculum rather than a 7.5M-step one. The default (None -> four phases of
    # 250k) is left exactly as it was, so stored MGC runs stay comparable.
    phase_steps = 250_000 if stages is None else max(1, DEFAULT_TOTAL_STEPS // stages)

    if args.mgc == "growth":
        return MorphologicalGrowthCurriculum(phase_steps=phase_steps, stages=stages)
    elif args.mgc == "inverse":
        return InverseMorphologicalCurriculum(phase_steps=phase_steps, stages=stages)
    elif args.mgc == "stochastic":
        return StochasticAgeCurriculum(interval=args.mgc_stochastic_interval, stages=stages)
    return None  # Baseline: no Callback
from stable_baselines3.common.callbacks import BaseCallback
import random

AGES = [1, 3, 6, 9]  # Fest durch verfuegbare XML-Dateien


class _MorphologicalCurriculumBase(BaseCallback):
    """ Abstract base for all morphological-growth-curriculum MGC callbacks. """

    def __init__(self, phase_steps: int = 250_000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_steps = phase_steps
        self.current_age: int | None = None

    def _get_age_for_step(self, step: int) -> int:
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

    def _apply_embodiment(self, age: int) -> None:
        self.current_age = age
        if self.verbose >= 1:
            print(f"[MGC] Step {self.num_timesteps:,} -> set_embodiment({age}M)")
        self.training_env.env_method("set_embodiment", age, age)


class MorphologicalGrowthCurriculum(_MorphologicalCurriculumBase):
    """1M -> 3M -> 6M -> 9M, each `phase_steps` steps."""

    def _get_age_for_step(self, step: int) -> int:
        idx = min(step // self.phase_steps, len(AGES) - 1)
        return AGES[idx]


class InverseMorphologicalCurriculum(_MorphologicalCurriculumBase):
    """9M -> 6M -> 3M -> 1M, each `phase_steps` steps."""

    def _get_age_for_step(self, step: int) -> int:
        idx = min(step // self.phase_steps, len(AGES) - 1)
        return AGES[-(idx + 1)]


class StochasticAgeCurriculum(_MorphologicalCurriculumBase):
    """ Sets a uniform random age out of AGES each `interval` steps. """

    def __init__(self, interval: int = 20_000, verbose: int = 1):
        super().__init__(phase_steps=interval, verbose=verbose)
        self._next_age: int = AGES[0]

    def _get_age_for_step(self, step: int) -> int:
        if step % self.phase_steps == 0:
            self._next_age = random.choice(AGES)
        return self._next_age


def make_curriculum_callback(args) -> _MorphologicalCurriculumBase | None:
    if args.curriculum == "growth":
        return MorphologicalGrowthCurriculum(phase_steps=250_000)
    elif args.curriculum == "inverse":
        return InverseMorphologicalCurriculum(phase_steps=250_000)
    elif args.curriculum == "stochastic":
        return StochasticAgeCurriculum(interval=args.curriculum_interval)
    return None  # Baseline: no Callback
import os
import random

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback

AGES = [1, 3, 6, 9]
""" The default age ladder, and the one every stored MGC run used.

08.09.2026 The comment here used to read "Fest durch verfuegbare XML-Dateien", and it was true:
'set_embodiment' loaded one of 16 pre-generated scenes. It now grows MIMo in memory
(:mod:`mimoGrowth.spec`), so the ladder is free -- see :func:`age_ladder`. This list stays the
default so that an MGC run started today is comparable with the ones on disk.
"""

DEFAULT_TOTAL_STEPS = 1_000_000
""" Fallback budget, used only when the caller has no '--train_for' to offer.

09.09.2026 This used to be the budget *unconditionally*, and that was a bug: the ladder was
always spread over 1M steps whatever the run actually trained for. '--train_for=2000000
--mgc_stages=20' finished growing at the halfway point and spent the entire second half at 9
months, and '--train_for=500000' ended the run with MIMo at 5.21 months while 'data.yml' claimed
9 -- so 'eval_rollover.py' would have evaluated that checkpoint on a body it never trained in.
The budget now comes from '--train_for'; this constant is what is left when that is unavailable.

Nothing on disk is re-baselined by the change: '--mgc' appears in no '*.sh' and exactly one
stored 'data.yml' carries the key at all (as 'none'), so the MGC has never produced a stored run.
At the 1M budget every sweep would have used, the two agree exactly anyway.
"""

RECORD_FILE = "mgc.yml"
""" The curriculum's own record, written into the run directory next to 'data.yml'.

11.09.2026 Once the ladder is drawn per run ('--mgc_interval_cv', '--mgc_jump_cv') the flags no
longer say which body MIMo had when, so the curve itself has to be kept. 'data.yml' holds the
flags that *define* the run -- that is what '--load_model' and mimolab read -- and this file holds
what the curriculum *did*:

* ``planned``: every rung as ``{rung, step, age}``, i.e. the schedule drawn before training.
* ``realised``: every ``set_embodiment`` that actually fired, as ``{step, age}`` plus ``rung`` and
  ``lag``. A swap only happens on an episode boundary, so ``lag`` is how many steps it trailed its
  planned start -- up to one episode, and more if several environments are out of phase.
* ``skipped_rungs``: rungs passed without ever being applied. A rung shorter than an episode can
  be jumped over entirely, which an aperiodic ladder makes possible; the record says so rather
  than leaving it to be inferred.

Rewritten atomically after every swap, so a run that dies mid-training still says how far its
body had grown. Unscheduled curricula ('stochastic') write ``planned: null`` and no rungs.
"""

LAMPL_INTERVAL_CV = 6.5 / 11.9
LAMPL_JUMP_CV = 0.30 / 0.95
""" Reference values for the two CVs -- *not* defaults.

Lampl, Veldhuis & Johnson (1992), Science 258:801, Table 1, daily protocol (the only one that
resolves single saltations): stasis between growth events 11.9 +- 6.5 days, saltation amplitude
0.95 +- 0.30 cm. Growth there is aperiodic and happens in <= 24 h bursts; the ladder's
instantaneous swaps already are the bursts.

The jump CV is measured on centimetres and applied here to the age increment. Locally that is the
same variance, since a jump in cm is a jump in months times the slope of the growth curve; what it
does *not* reproduce is Lampl's constant *mean* in cm, because the ladder's mean increment is
constant in months and 'mimoGrowth' grows logarithmically. That is the mean of the ladder, a
separate choice from its variance.
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


def _unit_gamma(rng, cv, n):
    """ ``n`` positive factors with mean 1 and coefficient of variation ``cv``.

    Gamma because both quantities are strictly positive and Lampl's stasis intervals are
    right-skewed (median 10 < mean 11.9 days); shape ``1/cv**2`` and scale ``cv**2`` give mean 1
    and CV ``cv``. Exactly ones at ``cv == 0``, without touching the generator.
    """
    if cv == 0:
        return np.ones(n)
    return rng.gamma(shape=1.0 / cv ** 2, scale=cv ** 2, size=n)


def growth_schedule(stages=None, total_steps=DEFAULT_TOTAL_STEPS, interval_cv=0.0, jump_cv=0.0,
                    seed=0):
    """ When each rung of the ladder starts, and at which age.

    11.09.2026 The ladder used to be periodic with an even age step. Two independent departures
    are now allowed, both as a coefficient of variation around that ladder:

    * ``interval_cv`` -- aperiodicity. Each rung's duration is the mean ``total_steps / n`` times
      a mean-1 gamma factor, renormalised so the rungs still add up to the budget.
    * ``jump_cv`` -- variance of the growth itself. Each age increment is the ladder's own
      increment times a mean-1 gamma factor, renormalised so MIMo still ends at the last age.

    With both at 0 this is the periodic ladder bit for bit, which is what keeps '--mgc=growth'
    unchanged. The two factors come from two generators spawned from one seed, so changing one
    CV leaves the other's realisation exactly where it was -- the variance and the aperiodicity
    of a run can be switched on separately against the *same* draw.

    Arguments:
        stages (int|None): See :func:`age_ladder`.
        total_steps (int): The training budget the ladder is spread over.
        interval_cv (float): CV of the rung durations. 0 is periodic.
        jump_cv (float): CV of the age increments. 0 is the even ladder.
        seed (int): Seeds both draws.

    Returns:
        tuple[list[int], list]: The start step of every rung (the first is 0) and its age, both
        in rung order. Ages are :func:`age_ladder`'s own values while ``jump_cv == 0``.

    Raises:
        ValueError: On a negative CV.
    """
    if interval_cv < 0 or jump_cv < 0:
        raise ValueError(f"The curriculum CVs must be >= 0, got interval {interval_cv}, "
                         f"jump {jump_cv}.")
    base = age_ladder(stages)
    n = len(base)
    phase_steps = max(1, total_steps // n)
    rng_jump, rng_interval = (np.random.default_rng(child)
                              for child in np.random.SeedSequence(seed).spawn(2))

    if jump_cv == 0:
        ages = list(base)
    else:
        jumps = np.diff(np.asarray(base, dtype=float)) * _unit_gamma(rng_jump, jump_cv, n - 1)
        jumps *= (base[-1] - base[0]) / jumps.sum()
        ages = [float(age) for age in base[0] + np.concatenate(([0.0], np.cumsum(jumps)))]
        # Pinned rather than left to the cumulative sum, which lands within an epsilon of them.
        ages[0], ages[-1] = float(base[0]), float(base[-1])

    if interval_cv == 0:
        starts = [i * phase_steps for i in range(n)]
    else:
        durations = _unit_gamma(rng_interval, interval_cv, n)
        durations *= n * phase_steps / durations.sum()
        starts = [0] + [int(start) for start in np.floor(np.cumsum(durations)[:-1])]

    return starts, ages


class _MorphologicalCurriculumBase(BaseCallback):
    """ Abstract base for all morphological-growth-curriculum MGC callbacks.

    Arguments:
        phase_steps (int): Steps per stage.
        stages (int|None): Number of stages, see :func:`age_ladder`. ``None`` uses :data:`AGES`.
        verbose (int): 1 prints every swap.
        record_path (str|None): Where to keep the :data:`RECORD_FILE`. ``None`` keeps none.
        record_meta (dict|None): The flags that produced this curriculum, written at the top of
            the record so it can be read without the run's 'data.yml'.
    """

    def __init__(self, phase_steps: int = 250_000, stages=None, verbose: int = 1,
                 record_path=None, record_meta=None):
        super().__init__(verbose)
        self.phase_steps = phase_steps
        self.ages = age_ladder(stages)
        self.current_age: float | None = None
        self.record_path = record_path
        self.record_meta = dict(record_meta or {})
        self.realised: list[dict] = []

    def _get_age_for_step(self, step: int) -> float:
        raise NotImplementedError

    def _rung_for_step(self, step: int):
        """ Index of the planned rung that owns ``step``, or ``None`` without a fixed plan. """
        return None

    def planned(self):
        """ The planned rungs as ``{rung, step, age}``, or ``None`` without a fixed plan. """
        return None

    def _on_training_start(self) -> None:
        # 09.09.2026 This read '_get_age_for_step(0)'. 'illustrations.train' calls 'model.learn'
        # once per '--save_every' chunk, so the hook fires once per chunk rather than once per
        # run -- and each of those re-applied the *first* rung. At '--save_every=200000' over a
        # 1M-step run that is four spurious swaps back to the youngest body, each holding until
        # the next episode boundary corrected it. 'num_timesteps' is global here because
        # 'train()' passes 'reset_num_timesteps=False'. The guard is the same one '_on_step'
        # carries: the env object survives across chunks, so on every call but the first the
        # embodiment is already the right one and re-applying it would only spend a compile and
        # a 'reset()' to arrive where it already is.
        age = self._get_age_for_step(self.num_timesteps)
        if age != self.current_age:
            self._apply_embodiment(age)

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
        # 09.09.2026 Logged as well as printed. A four-rung ladder is legible in stdout; a
        # '--mgc_stages=30' one is not, and the age is the independent variable of the run --
        # it belongs in the event file next to the metrics it is supposed to explain. Recorded
        # under both names because 'set_embodiment' sets them together but need not forever.
        self.logger.record("curriculum/morph_age", age)
        self.logger.record("curriculum/physio_age", age)
        self._note_swap(age)

    def _note_swap(self, age) -> None:
        """ Append the swap that just happened to the record and rewrite it.

        Kept apart from '_apply_embodiment' so the record can be exercised without an env.
        """
        step = int(self.num_timesteps)
        entry = {"step": step, "age": float(age)}
        rung = self._rung_for_step(step)
        if rung is not None:
            entry["rung"] = rung
            entry["lag"] = step - self.rung_starts[rung]
        self.realised.append(entry)
        self.write_record()

    def write_record(self) -> None:
        if self.record_path is None:
            return
        planned = self.planned()
        record = dict(self.record_meta)
        record["planned"] = planned
        record["realised"] = self.realised
        if planned is not None:
            applied = {entry["rung"] for entry in self.realised}
            reached = max(applied, default=-1)
            record["skipped_rungs"] = [r for r in range(reached) if r not in applied]
        # Via a temp file and 'os.replace', so a kill mid-write leaves the previous record intact
        # instead of a truncated one.
        tmp = self.record_path + ".tmp"
        with open(tmp, "w") as outfile:
            yaml.safe_dump(record, outfile, default_flow_style=None, sort_keys=False)
        os.replace(tmp, self.record_path)


class _ScheduledCurriculum(_MorphologicalCurriculumBase):
    """ A curriculum with a ladder fixed before training -- see :func:`growth_schedule`. """

    def __init__(self, phase_steps: int = 250_000, stages=None, interval_cv: float = 0.0,
                 jump_cv: float = 0.0, seed: int = 0, verbose: int = 1, record_path=None,
                 record_meta=None):
        super().__init__(phase_steps=phase_steps, stages=stages, verbose=verbose,
                         record_path=record_path, record_meta=record_meta)
        # 'phase_steps * n' rather than the raw budget, so the mean rung is 'phase_steps' exactly
        # and the periodic case is the old integer arithmetic.
        self.rung_starts, self.ages = growth_schedule(
            stages, phase_steps * len(self.ages), interval_cv, jump_cv, seed)
        self.sequence = self._order(self.ages)

    def _order(self, ages):
        raise NotImplementedError

    def _rung_for_step(self, step: int) -> int:
        return int(np.searchsorted(self.rung_starts, step, side="right")) - 1

    def _get_age_for_step(self, step: int) -> float:
        return self.sequence[self._rung_for_step(step)]

    def planned(self):
        return [{"rung": i, "step": int(start), "age": float(age)}
                for i, (start, age) in enumerate(zip(self.rung_starts, self.sequence))]


class MorphologicalGrowthCurriculum(_ScheduledCurriculum):
    """ Youngest to oldest -- 1M -> 3M -> 6M -> 9M by default, each `phase_steps` steps. """

    def _order(self, ages):
        return list(ages)


class InverseMorphologicalCurriculum(_ScheduledCurriculum):
    """ Oldest to youngest -- 9M -> 6M -> 3M -> 1M by default, each `phase_steps` steps. """

    def _order(self, ages):
        return list(ages)[::-1]


class StochasticAgeCurriculum(_MorphologicalCurriculumBase):
    """ Sets a uniform random age out of the ladder each `interval` steps. """

    def __init__(self, interval: int = 20_000, stages=None, verbose: int = 1, record_path=None,
                 record_meta=None):
        super().__init__(phase_steps=interval, stages=stages, verbose=verbose,
                         record_path=record_path, record_meta=record_meta)
        self._next_age: float = self.ages[0]

    def _get_age_for_step(self, step: int) -> float:
        if step % self.phase_steps == 0:
            self._next_age = random.choice(self.ages)
        return self._next_age


def make_curriculum_callback(args, save_dir=None) -> _MorphologicalCurriculumBase | None:
    """ The curriculum callback the flags ask for, or ``None`` for '--mgc=none'.

    Arguments:
        args: The parsed flags. Missing 'mgc_*' attributes fall back to their defaults.
        save_dir (str|None): The run directory. The :data:`RECORD_FILE` is kept there; ``None``
            keeps no record.

    Raises:
        ValueError: If a CV is set for a curriculum that has no ladder to vary.
    """
    stages = getattr(args, "mgc_stages", None)
    interval_cv = float(getattr(args, "mgc_interval_cv", 0.0) or 0.0)
    jump_cv = float(getattr(args, "mgc_jump_cv", 0.0) or 0.0)
    seed = int(getattr(args, "mgc_seed", 0) or 0)

    # Raised rather than ignored: a sweep asking for variance on a curriculum without a ladder
    # would otherwise train the unvaried thing under a name that says it is varied.
    if (interval_cv or jump_cv) and args.mgc not in ("growth", "inverse"):
        raise ValueError(f"'--mgc_interval_cv'/'--mgc_jump_cv' vary a growth ladder and need "
                         f"'--mgc=growth' or '--mgc=inverse', got '--mgc={args.mgc}'.")

    # The ladder spans the run: every rung gets an equal share of '--train_for', so the last one
    # is reached just as training ends whatever the budget. '--mgc_stages=30' is therefore a
    # finer curriculum over the same run, never a longer one. At the 1M budget the old hardcoded
    # arithmetic assumed, this is identical (4 rungs -> 250k, 20 rungs -> 50k).
    total_steps = getattr(args, "train_for", 0) or DEFAULT_TOTAL_STEPS
    n_rungs = len(age_ladder(stages))
    phase_steps = max(1, total_steps // n_rungs)

    record_path = None if save_dir is None else os.path.join(save_dir, RECORD_FILE)
    meta = {"mgc": args.mgc, "mgc_stages": stages, "train_for": int(total_steps),
            "phase_steps": int(phase_steps), "mgc_interval_cv": interval_cv,
            "mgc_jump_cv": jump_cv, "mgc_seed": seed}

    if args.mgc == "growth":
        return MorphologicalGrowthCurriculum(phase_steps=phase_steps, stages=stages,
                                             interval_cv=interval_cv, jump_cv=jump_cv, seed=seed,
                                             record_path=record_path, record_meta=meta)
    elif args.mgc == "inverse":
        return InverseMorphologicalCurriculum(phase_steps=phase_steps, stages=stages,
                                              interval_cv=interval_cv, jump_cv=jump_cv, seed=seed,
                                              record_path=record_path, record_meta=meta)
    elif args.mgc == "stochastic":
        interval = args.mgc_stochastic_interval
        meta.update(phase_steps=int(interval), mgc_stochastic_interval=int(interval))
        return StochasticAgeCurriculum(interval=interval, stages=stages,
                                       record_path=record_path, record_meta=meta)
    return None  # Baseline: no Callback

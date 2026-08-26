""" Acceptance checks that :class:`~mimoEnv.envs.roll_over.MIMoRollOverEnv` is a real GoalEnv.

Run this after touching anything in the reward path::

    MUJOCO_GL=osmesa python mimoEnv/goalenv_check.py

Why this exists rather than just calling ``stable_baselines3.common.env_checker.check_env``:
that checker verifies ``reward == compute_reward(achieved_goal, desired_goal, info)`` on the
*current* transition, which passes trivially for a reward function that ignores its arguments and
reads the live simulation instead. That is exactly the bug this file is here to prevent, and the
only way to see it is to call ``compute_reward`` with a goal that differs from the live state.

Before the Phase 1 rewrite every one of these calls returned the identical value -0.224019::

    compute_reward(ag=0.0017, dg=0.00)    = -0.224019
    compute_reward(ag=0.0017, dg=0.95)    = -0.224019
    compute_reward(FAKE ag=0.99, dg=0.95) = -0.224019

which would have made HER a silent no-op: every virtual transition would have carried the reward
of the real one.

**Memory: run the sections in separate processes on a 16 GB machine.** One MIMo env is ~3.8 GB
RSS, and closing it does not return all of it to the OS -- a section that builds three envs peaks
at 5.8 GB, and running every section in a single process OOM-kills the machine. Pass section names to run
them one at a time::

    for t in $(MUJOCO_GL=osmesa python mimoEnv/goalenv_check.py --list); do
        MUJOCO_GL=osmesa python mimoEnv/goalenv_check.py $t
    done
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import gymnasium as gym

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)


FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_env(**kwargs):
    params = dict(starting_position='supine', goal_function='cos', touch_params=None,
                  pbrs=True, pbrs_w=100, pen_factor=0.02, isr=False,
                  age_physio=9, age_morph=9, achieved_goal_in_observation=True,
                  render_mode='rgb_array')
    params.update(kwargs)
    return gym.make('MIMoRollOver-v0', **params).unwrapped


def test_sb3_env_checker():
    """The standard checker: catches missing vectorization and shape errors."""
    print("\n1. SB3 env_checker")
    from stable_baselines3.common.env_checker import check_env
    env = make_env()
    try:
        check_env(env, warn=False)
        check("check_env passes", True)
    except Exception as exc:
        check("check_env passes", False, f"{type(exc).__name__}: {exc}")
    env.close()


def test_purity():
    """The checks env_checker cannot do: the reward must actually depend on its arguments."""
    print("\n2. Purity -- reward depends on its arguments")
    env = make_env()
    env.reset(seed=0)
    obs, _, _, _, info = env.step(env.action_space.sample())
    ag = np.asarray(obs['achieved_goal'], dtype=np.float64)

    by_desired = [env.compute_reward(ag, np.array([d]), info) for d in (0.0, 0.25, 0.5, 0.95)]
    check("reward varies with desired_goal", len(set(by_desired)) > 1,
          f"values {[round(v, 4) for v in by_desired]}")

    by_achieved = [env.compute_reward(np.array([a]), np.array([0.95]), info)
                   for a in (0.0, 0.3, 0.6, 0.99)]
    check("reward varies with achieved_goal", len(set(by_achieved)) > 1,
          f"values {[round(v, 4) for v in by_achieved]}")

    # The real proof of purity: stepping the simulation must not change the answer for a fixed
    # (achieved, desired) pair. A reward reading self.data would drift here.
    before = env.compute_reward(np.array([0.42]), np.array([0.95]), info)
    for _ in range(10):
        env.step(env.action_space.sample())
    after = env.compute_reward(np.array([0.42]), np.array([0.95]), info)
    check("reward is invariant under stepping the sim", np.isclose(before, after),
          f"{before:.6f} vs {after:.6f}")
    env.close()


def test_vectorization():
    """HER calls compute_reward on whole batches; it must match the elementwise answers."""
    print("\n3. Vectorization")
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())

    ags = np.array([[0.0], [0.3], [0.6], [0.96]])
    dgs = np.array([[0.95], [0.95], [0.5], [0.95]])
    infos = np.array([info] * 4, dtype=object)

    batch = env.compute_reward(ags, dgs, infos)
    check("batch returns shape (N,)", np.shape(batch) == (4,), f"got {np.shape(batch)}")

    singles = np.array([env.compute_reward(a, d, info) for a, d in zip(ags, dgs)])
    check("batch matches elementwise", np.allclose(batch, singles),
          f"batch {np.round(batch, 4)} vs singles {np.round(singles, 4)}")
    env.close()


def test_pbrs_regression():
    """The rewrite must not change the reward the PPO/SAC pipeline actually sees.

    E0 (SAC + PBRS) scored 94%; if this drifts, E1 is no longer comparable to it.
    """
    print("\n4. PBRS regression -- step() reward matches the original formula")
    env = make_env()
    env.reset(seed=3)
    worst = 0.0
    for _ in range(40):
        potential_before = env.get_potential()
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())

        penalty = env.compute_penalization()
        succeeded = env.get_achieved_goal_cos()[0] >= 0.95
        if succeeded:
            expected = env.reward_success - penalty
        else:
            expected = env.pbrs_w * (env.get_potential() - potential_before) - penalty
        worst = max(worst, abs(reward - expected))
        if terminated or truncated:
            env.reset()
    check("step reward equals the original PBRS formula", worst < 1e-9,
          f"max abs deviation {worst:.3e}")
    env.close()


def test_sparse_reward():
    print("\n5. Sparse reward mode")
    env = make_env(sparse_reward=True, nopen=True)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    values = {env.compute_reward(np.array([a]), np.array([0.5]), info) for a in (0.0, 0.2, 0.49)}
    check("below goal gives exactly -1", values == {-1.0}, f"got {values}")
    values = {env.compute_reward(np.array([a]), np.array([0.5]), info) for a in (0.5, 0.7, 1.0)}
    check("at or above goal gives exactly 0", values == {0.0}, f"got {values}")
    env.close()


def test_goal_sampling():
    print("\n6. Goal sampling")
    env = make_env()
    goals = {float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(5)}
    check("default goal is fixed at 0.95", goals == {0.95}, f"got {goals}")
    env.close()

    env = make_env(success_at_side_lying=True)
    goals = {float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(5)}
    check("side-lying goal is 0.5", goals == {0.5}, f"got {goals}")
    env.close()

    env = make_env(goal_low=0.25, goal_high=0.95)
    goals = [float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(20)]
    check("sampled goals vary", len(set(goals)) > 1, f"{len(set(goals))} distinct")
    check("sampled goals stay in range", all(0.25 <= g <= 0.95 for g in goals),
          f"range [{min(goals):.3f}, {max(goals):.3f}]")
    env.close()


def test_info_contract():
    """HER carries the goal-independent terms through info; they have to be there."""
    print("\n7. Info dict carries the goal-independent reward terms")
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ('ctrl_cost', 'prev_achieved_goal', 'rolled_over'):
        check(f"info['{key}'] present", key in info)

    # With the stored ctrl_cost, a relabelled reward must still subtract the same penalty.
    penalty = float(info['ctrl_cost'])
    reward = env.compute_reward(np.array([0.99]), np.array([0.95]), info)
    check("relabelled success reward uses stored ctrl_cost",
          np.isclose(reward, env.reward_success - penalty),
          f"{reward:.4f} vs {env.reward_success - penalty:.4f}")
    env.close()


def test_her_relabel_end_to_end():
    """The bug this whole file exists for: relabelling must change the rewards."""
    print("\n8. End-to-end -- HER-style relabelling changes rewards")
    env = make_env()
    env.reset(seed=1)
    achieved, infos = [], []
    for _ in range(30):
        obs, _, _, _, info = env.step(env.action_space.sample())
        achieved.append(np.asarray(obs['achieved_goal'], dtype=np.float64))
        infos.append(info)

    ags = np.stack(achieved)
    real = env.compute_reward(ags, np.full_like(ags, 0.95), np.array(infos, dtype=object))
    # 'future' strategy: relabel each goal to a rotation actually reached later on.
    relabelled = env.compute_reward(ags, np.full_like(ags, float(ags.max())),
                                    np.array(infos, dtype=object))
    check("relabelled rewards differ from the real ones", not np.allclose(real, relabelled),
          f"max abs difference {np.max(np.abs(real - relabelled)):.4f}")
    env.close()


def test_pbrs_bounded_under_relabelling():
    """PBRS rewards must stay bounded when HER relabels the goal.

    The potential used to jump to +reward_success at the goal. Terminating episodes hid that,
    because they only ever terminate on the *real* goal (0.95) -- but HER relabels to rotations
    MIMo reached mid-trajectory and then drifted back out of, which pays
    pbrs_w * (-reward_success). Measured before the fix: -50002.0, and SAC's critic loss reached
    3.85e7 within 4000 steps.
    """
    print("\n9. PBRS stays bounded under goal relabelling")
    env = make_env(done_active=True)
    env.reset(seed=0)
    info = {'ctrl_cost': 0.0, 'prev_achieved_goal': np.array([0.45])}
    # A relabelled goal that was reached and then lost again.
    reward = env.compute_reward(np.array([0.38]), np.array([0.40]), info)
    bound = 10.0 * env.pbrs_w
    check("leaving a relabelled goal is not catastrophic", abs(reward) < bound,
          f"reward {reward:.2f}, bound {bound:.0f}")
    env.close()


def test_goal_tolerance():
    """--goal_tolerance: the scalar success test as a band instead of a threshold.

    The point of the flag is that it changes the reward of *relabelled* goals while leaving the
    real task alone. Both halves of that claim are checked here, because either one silently
    breaking would invalidate the experiment it exists for: if the real task moved, the numbers
    would not be comparable with every run trained before it; if the relabelled goals did not
    change, the flag would do nothing at all.

    One environment for the whole section, with the rule flipped by assignment where the two
    have to be compared. Building a second one costs another 3.6 GB of RSS -- four live envs
    OOM-killed a 16 GB machine while this check was being written.
    """
    print("\n10. Goal tolerance (--goal_tolerance)")

    # Rejected combinations first: all three raise in the constructor, before the MuJoCo model is
    # built, so they cost nothing.
    for name, kwargs in (("intrinsic", dict(goal_function='intrinsic', goal_tolerance=0.05)),
                         ("a non-positive radius", dict(goal_tolerance=0.0))):
        try:
            make_env(**kwargs).close()
            refused = False
        except ValueError:
            refused = True
        check(f"refuses {name}", refused)

    env = make_env(goal_tolerance=0.05, sparse_reward=True, pbrs=False, done_active=False)

    check("the fixed goal moves to 1.0", float(env.sample_goal()[0]) == 1.0,
          f"got {float(env.sample_goal()[0])}")

    # 1. The real task is unchanged: rho is capped at 1.0, so |rho - 1.0| <= 0.05 is rho >= 0.95.
    rho = np.array([0.0, 0.5, 0.9, 0.94, 0.96, 0.99, 1.0]).reshape(-1, 1)
    band_hits = env.is_success(rho, np.full_like(rho, 1.0))
    env.goal_tolerance = None
    check("threshold is the default rule", float(env.sample_goal()[0]) == 0.95)
    threshold_hits = env.is_success(rho, np.full_like(rho, 0.95))
    overshoot_threshold = bool(env.is_success(np.array([0.60]), np.array([0.40])))
    undershoot_threshold = bool(env.is_success(np.array([0.30]), np.array([0.40])))
    env.goal_tolerance = 0.05

    check("identical success at the real goal", np.array_equal(band_hits, threshold_hits),
          f"band={band_hits} threshold={threshold_hits}")

    # 2. Relabelled goals -- the half that is supposed to change. Under the threshold, any
    #    achieved goal at or above the relabelled one scores 0 no matter how far past it; under
    #    the band only the ones close to it do. That difference is the whole mechanism.
    check("overshooting a relabelled goal succeeds under the threshold", overshoot_threshold)
    check("overshooting a relabelled goal fails under the band",
          not bool(env.is_success(np.array([0.60]), np.array([0.40]))))
    check("landing on a relabelled goal succeeds under the band",
          bool(env.is_success(np.array([0.42]), np.array([0.40]))))
    check("undershooting fails under both",
          not bool(env.is_success(np.array([0.30]), np.array([0.40])))
          and not undershoot_threshold)

    # 3. The sparse reward follows the mask, purely from the arguments, batched.
    ag = np.array([[0.42], [0.60], [0.30]])
    dg = np.full_like(ag, 0.40)
    rewards = env.compute_reward(ag, dg, [{'ctrl_cost': 0.0}] * 3)
    check("sparse reward follows the band", np.array_equal(rewards, np.array([0.0, -1.0, -1.0])),
          f"got {rewards}")
    check("compute_reward stays vectorized", np.asarray(rewards).shape == (3,))

    # 4. Same goal, different achieved goals -> different rewards. This is the purity property
    #    the whole file exists for, re-checked on the new branch.
    lo = env.compute_reward(np.array([0.30]), np.array([0.40]), {'ctrl_cost': 0.0})
    hi = env.compute_reward(np.array([0.42]), np.array([0.40]), {'ctrl_cost': 0.0})
    check("relabelling actually changes the reward", lo != hi, f"{lo} vs {hi}")

    # 5. Distance shaping is unaffected: the potential was already a distance, so the band only
    #    moves where the terminal bonus is paid.
    env.sparse_reward = False
    p_far = env.compute_reward(np.array([0.30]), np.array([0.40]), {'ctrl_cost': 0.0})
    check("distance shaping still reads as a distance", np.isclose(p_far, -0.1, atol=1e-9),
          f"got {p_far}")
    env.sparse_reward = True

    # 6. Side lying keeps its value -- but note it then means "stop there", not "reach at least".
    env.success_at_side_lying = True
    check("side lying keeps 0.5", float(env.sample_goal()[0]) == 0.5)
    env.success_at_side_lying = False
    env.close()
    del env

    # 7. Round-trip. A flag that does not reach data.yml would evaluate reloaded models against
    #    the other success rule, which is the failure mode the yaml_data dict exists to prevent.
    import re
    source = open(os.path.join(os.path.dirname(__file__), 'illustrations.py')).read()
    check("goal_tolerance is in the yaml_data dict",
          re.search(r"'goal_tolerance':\s*args\.goal_tolerance", source) is not None)

    from mimoEnv.eval_rollover import env_kwargs, FULL_ROLL_GOAL
    kwargs = env_kwargs({'goal_tolerance': 0.05}, 'supine', FULL_ROLL_GOAL)
    check("eval_rollover forwards the tolerance", kwargs['goal_tolerance'] == 0.05)
    check("eval_rollover pins the goal the policy was trained on",
          kwargs['goal_low'] == 1.0 and kwargs['goal_high'] == 1.0)
    check("a run without the flag is unaffected",
          env_kwargs({}, 'supine', 0.95)['goal_tolerance'] is None)


def test_early_stop():
    """--stop_at_roll_rate must need consecutive hits, and must not fire when it is off.

    25.08.2026 Stub-driven: the logic under test is the counter, and the only thing a live env
    would add is 20 deterministic rollouts per case at 3.6 GB. '_run_episodes' is replaced with a
    scripted sequence of roll rates, which is also the only way to test the "hit, miss, hit"
    case -- a real policy will not produce it on demand.
    """
    print("\n11. Early stopping on the evaluation roll rate")

    from mimoEnv.envs.roll_over_callback import RollOverEvalCallback

    class FakeLogger:
        def __init__(self):
            self.values = {}

        def record(self, key, value):
            self.values[key] = value

    class FakeModel:
        # 'BaseCallback.logger' is a read-only property returning 'self.model.logger', so the
        # stub logger has to hang off the model rather than off the callback.
        def __init__(self):
            self.num_timesteps = 0
            self.logger = FakeLogger()

    def drive(rates, stop_at, patience):
        """Feed a sequence of roll rates through the callback; return (returns, cb)."""
        cb = RollOverEvalCallback(eval_env=None, eval_every=1, n_episodes=20,
                                  save_dir=None, episode_steps=200,
                                  stop_at_roll_rate=stop_at, stop_patience=patience)
        cb.model = FakeModel()
        cb._next_eval = 0
        returns = []
        for i, rate in enumerate(rates):
            cb.model.num_timesteps = (i + 1) * 25000
            rolled = np.array([1.0] * int(round(rate * 20)) + [0.0] * (20 - int(round(rate * 20))))
            cb._run_episodes = lambda r=rolled: (np.where(r > 0, 0.99, 0.30), r, r, np.full(20, 50.0))
            returns.append(cb._on_step())
        return returns, cb

    returns, cb = drive([0.8, 0.9], stop_at=1.0, patience=2)
    check("below threshold does not stop", all(returns) and not cb.stop_requested)

    returns, cb = drive([1.0], stop_at=1.0, patience=2)
    check("one hit is not enough", all(returns) and not cb.stop_requested,
          f"hits {cb._consecutive_hits}")

    returns, cb = drive([1.0, 1.0], stop_at=1.0, patience=2)
    check("two consecutive hits stop the run", cb.stop_requested and returns[-1] is False)
    check("the stop step is recorded", cb.stop_step == 50000, f"got {cb.stop_step}")

    # The reason patience exists: a policy hovering under the threshold crosses it by chance.
    returns, cb = drive([1.0, 0.9, 1.0], stop_at=1.0, patience=2)
    check("a miss resets the counter", not cb.stop_requested and all(returns),
          f"hits {cb._consecutive_hits}")

    returns, cb = drive([0.85, 0.85], stop_at=0.75, patience=2)
    check("a threshold below 1.0 works", cb.stop_requested, "0.85 >= 0.75 twice")

    returns, cb = drive([1.0, 1.0, 1.0], stop_at=None, patience=2)
    check("off by default -- never stops", all(returns) and not cb.stop_requested)
    check("no stop counter is logged when off", 'eval/stop_hits' not in cb.model.logger.values)

    returns, cb = drive([1.0], stop_at=1.0, patience=1)
    check("patience 1 stops on the first hit", cb.stop_requested and returns[-1] is False)


def test_episode_horizon():
    """--episode_steps must move the TimeLimit, and must not change anything when unset.

    The horizon is enforced by the TimeLimit wrapper from the registration, so this is the one
    section that must keep the wrapper instead of unwrapping it. A silently ignored override
    would be invisible in training: episodes would just keep their old length while data.yml
    claims otherwise.
    """
    print("\n12. Episode horizon (--episode_steps)")

    def run_to_truncation(env, limit):
        """Step with zero actions until the wrapper ends the episode. Returns the step count."""
        env.reset(seed=0)
        for step in range(1, limit + 2):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
            if terminated or truncated:
                return step, truncated
        return None, False

    params = dict(starting_position='supine', goal_function='cos', touch_params=None,
                  pbrs=False, sparse_reward=True, pen_factor=0.02, isr=False,
                  age_physio=9, age_morph=9, achieved_goal_in_observation=True,
                  # Nothing may terminate early, or the horizon is not what ends the episode.
                  goal_low=0.95, goal_high=0.95, done_active=False, render_mode='rgb_array')

    env = gym.make('MIMoRollOver-v0', max_episode_steps=40, **params)
    steps, truncated = run_to_truncation(env, 40)
    check("override takes effect", steps == 40 and truncated,
          f"episode ended after {steps} steps, truncated={truncated}, expected 40")
    env.close()

    # The registered default, which every existing run was trained under.
    env = gym.make('MIMoRollOver-v0', **params)
    default = env.spec.max_episode_steps
    check("registered default is 500", default == 500, f"registration says {default}")
    env.close()

    # Passing the default back in is what illustrations.py does when --episode_steps is unset.
    env = gym.make('MIMoRollOver-v0', max_episode_steps=500, **params)
    check("passing the default back in is a no-op", env.spec.max_episode_steps == 500,
          f"got {env.spec.max_episode_steps}")
    env.close()


SECTIONS = [
    'test_sb3_env_checker', 'test_purity', 'test_vectorization', 'test_pbrs_regression',
    'test_sparse_reward', 'test_goal_sampling', 'test_info_contract',
    'test_her_relabel_end_to_end', 'test_pbrs_bounded_under_relabelling',
    'test_goal_tolerance', 'test_early_stop', 'test_episode_horizon',
]


if __name__ == '__main__':
    import sys

    argv = sys.argv[1:]
    if argv == ['--list']:
        print("\n".join(SECTIONS))
        raise SystemExit(0)
    if argv:
        unknown = [name for name in argv if name not in SECTIONS]
        if unknown:
            raise SystemExit(f"Unknown section(s) {unknown}. Use --list to see the names.")
        print(f"GoalEnv acceptance checks for MIMoRollOver-v0 -- {', '.join(argv)}")
        for name in argv:
            globals()[name]()
        print()
        if FAILURES:
            print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
            raise SystemExit(1)
        print("Section(s) passed.")
        raise SystemExit(0)

    print("GoalEnv acceptance checks for MIMoRollOver-v0")
    test_sb3_env_checker()
    test_purity()
    test_vectorization()
    test_pbrs_regression()
    test_sparse_reward()
    test_goal_sampling()
    test_info_contract()
    test_her_relabel_end_to_end()
    test_pbrs_bounded_under_relabelling()
    test_goal_tolerance()
    test_early_stop()
    test_episode_horizon()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        raise SystemExit(1)
    print("All checks passed.")

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


def test_goal_curriculum():
    """The curriculum must start narrow, follow what is achieved, and never exceed goal_high."""
    print("\n10. Goal curriculum")

    env = make_env(goal_low=0.25, goal_high=0.95, goal_curriculum=True,
                   goal_curriculum_window=10, goal_curriculum_margin=0.1)
    unwrapped = env.unwrapped

    # Before any episode has finished the range is just [goal_low, goal_low + margin].
    goals = [float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(10)]
    check("starts narrow, not at goal_high", max(goals) <= 0.35 + 1e-9,
          f"max sampled {max(goals):.3f}, cap 0.350")

    # Feed the statistics directly: the point under test is the mapping from achieved rotations
    # to the sampled range, not whether MIMo can produce those rotations.
    unwrapped._recent_episode_max.extend([0.60] * 10)
    goals = [float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(10)]
    check("follows what was achieved", 0.60 < max(goals) <= 0.70 + 1e-9,
          f"max sampled {max(goals):.3f}, expected in (0.60, 0.70]")

    unwrapped._recent_episode_max.clear()
    unwrapped._recent_episode_max.extend([0.99] * 10)
    goals = [float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(10)]
    check("never exceeds goal_high", max(goals) <= 0.95 + 1e-9,
          f"max sampled {max(goals):.3f}, cap 0.950")
    env.close()

    # Off by default: the flag must not change any existing run.
    env = make_env(goal_low=0.25, goal_high=0.95)
    goals = [float(env.reset(seed=s)[0]['desired_goal'][0]) for s in range(20)]
    check("off by default -- full range still sampled", max(goals) > 0.5,
          f"max sampled {max(goals):.3f}")
    env.close()

    # A run without a goal range must be refused rather than silently ignoring the flag.
    try:
        make_env(goal_curriculum=True).close()
        refused = False
    except ValueError:
        refused = True
    check("refused without a goal range", refused)


if __name__ == '__main__':
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
    test_goal_curriculum()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        raise SystemExit(1)
    print("All checks passed.")

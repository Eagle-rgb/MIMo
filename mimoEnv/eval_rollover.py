""" Evaluate a saved roll-over policy under the protocol agreed for the HER experiments.

    MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py --model=<path/to/model_1.zip> [--episodes=50]

The protocol, and why each rule is there:

* **ISR off.** Initial state randomisation draws the starting roll from Beta(1, 3), so some
  episodes begin nearly rolled. Training logs of rho_max around 0.94 were pure ISR artefacts;
  without ISR the same policies read 0.26-0.36. Always evaluate with it off.
* **Goal pinned to 0.95.** Runs trained with --goal_low/--goal_high sample easy targets during
  training. Scoring against those would report a healthy success rate at zero real rolls.
* **Success measured as rho >= 0.95, aggregated per episode**, not read off 'terminated'. With
  --no_done_active nothing terminates at all, so 'terminated' would read a constant 0%.
* **Deterministic actions**, since the stochastic policy is a training device.

Reads the run's own data.yml so the evaluation environment matches how the model was trained.
"""
import argparse
import json
import os
import re

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import gymnasium as gym
import yaml

import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)

SIDE_LYING_THRESHOLD = 0.5
ROLL_THRESHOLD = 0.95
# Horizon registered for MIMoRollOver-v0 in mimoEnv/__init__.py. Runs trained with
# --episode_steps record their own in data.yml and are evaluated at that length instead:
# a policy that needs 300 steps to roll fails every episode if scored over 200.
DEFAULT_EPISODE_STEPS = 500


def load_run_config(model_path):
    """Read the data.yml that sits next to a saved model, if there is one."""
    config_path = os.path.join(os.path.dirname(model_path), 'data.yml')
    if not os.path.exists(config_path):
        print(f"Warning! No data.yml next to {model_path}; falling back to defaults.")
        return {}
    with open(config_path) as handle:
        return yaml.safe_load(handle) or {}


def build_env(config, starting_position, goal):
    proprio = config.get('proprio_params')
    kwargs = dict(
        starting_position=starting_position,
        goal_function=config.get('goal_achievement_function', 'cos'),
        pbrs=config.get('pbrs', False),
        pbrs_w=config.get('pbrs_w', 100),
        pen_factor=config.get('pen_factor', 0.02),
        nopen=config.get('nopen', False),
        sparse_reward=config.get('sparse_reward', False),
        achieved_goal_in_observation=config.get('achieved_goal_in_observation', False),
        age_physio=config.get('physio_age', 9),
        age_morph=config.get('morph_age', 9),
        freeze_arm=config.get('freeze_arm', False),
        freeze_leg=config.get('freeze_leg', False),
        # Protocol: ISR off, goal pinned to the full roll, episodes never cut short by success
        # so that every episode gets the same number of chances.
        isr=False,
        # Pin the goal: runs trained with --goal_low/--goal_high saw easy targets during
        # training, and scoring against those would report success at zero real rolls.
        goal_low=goal,
        goal_high=goal,
        success_at_side_lying=False,
        done_active=False,
        render_mode='rgb_array',
    )
    if config.get('touch', False):
        from mimoEnv.envs.roll_over import TOUCH_PARAMS
        kwargs['touch_params'] = TOUCH_PARAMS
    else:
        kwargs['touch_params'] = None
    if isinstance(proprio, dict):
        kwargs['proprio_params'] = proprio
    return gym.make('MIMoRollOver-v0', **kwargs).unwrapped


def parse_sweep(spec):
    """Parse '0.25:0.95:0.05' or '0.05,0.5,2.0' into a list of goal values to feed the policy."""
    if ':' in spec:
        low, high, step = (float(part) for part in spec.split(':'))
        if step <= 0:
            raise ValueError("Sweep step must be positive.")
        # +step/2 so the upper end is included despite floating point.
        return [round(v, 6) for v in np.arange(low, high + step / 2, step)]
    return [float(part) for part in spec.split(',')]


def load_policy(model_path, algorithm, env):
    """Load a saved policy for evaluation only.

    The env has to be passed: SB3 asserts on it for anything saved with a HerReplayBuffer
    ("You must pass an environment when using `HerReplayBuffer`"), because the buffer needs
    env.compute_reward to relabel. We never train here, so the buffer is shrunk to a single
    transition rather than reallocating the training-sized one.
    """
    import stable_baselines3 as sb3
    cls = getattr(sb3, algorithm)
    return cls.load(model_path, env=env, custom_objects={'buffer_size': 1, 'learning_starts': 0})


def evaluate(model, env, episodes, seed0=1000, policy_goal=None,
             episode_steps=DEFAULT_EPISODE_STEPS):
    """Roll out the policy and measure rho_max, the highest rotation reached in the episode.

    'policy_goal' lies to the policy: the environment keeps its own desired_goal for reward and
    success, but the observation handed to 'model.predict' carries this value instead. Since the
    input is then constant, the resulting behaviour cannot depend on what was actually asked for
    -- it is a goal-agnostic policy built from the goal-conditioned one, without retraining.

    Only 'desired_goal' is replaced. 'achieved_goal' and every sensor reading stay real, and
    rho_max is measured from the simulation, so the score is honest even though the input is not.

    'episode_steps' is the horizon. build_env returns the unwrapped environment, so the TimeLimit
    from the registration is gone and this loop is what ends an episode.
    """
    rolled, side, rho_max, steps = [], [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed0 + episode)
        best = float(env.get_achieved_goal_cos()[0])
        step = 0
        first_success = None
        done = False
        while not done and step < episode_steps:
            if policy_goal is not None:
                obs = dict(obs)
                obs['desired_goal'] = np.full_like(np.asarray(obs['desired_goal'], dtype=np.float64),
                                                  policy_goal)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            best = max(best, float(env.get_achieved_goal_cos()[0]))
            if first_success is None and best >= ROLL_THRESHOLD:
                first_success = step
            done = terminated or truncated
        rolled.append(1.0 if best >= ROLL_THRESHOLD else 0.0)
        side.append(1.0 if best >= SIDE_LYING_THRESHOLD else 0.0)
        rho_max.append(best)
        steps.append(first_success if first_success is not None else np.nan)
    return dict(rolled=np.array(rolled), side=np.array(side),
                rho_max=np.array(rho_max), steps=np.array(steps, dtype=float))


def _row(results, policy_goal=None):
    """One result row, in the same numbers the table prints."""
    finished = results['steps'][~np.isnan(results['steps'])]
    return {
        'policy_goal': policy_goal,
        'rolled': float(results['rolled'].mean()),
        'side': float(results['side'].mean()),
        'rho_mean': float(results['rho_max'].mean()),
        'rho_min': float(results['rho_max'].min()),
        'rho_max': float(results['rho_max'].max()),
        'steps_mean': float(finished.mean()) if finished.size else None,
        'steps_std': float(finished.std()) if finished.size else None,
        'steps_n': int(finished.size),
    }


def write_json(path, payload):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)


def starting_position_from_path(model_path):
    """Recover the starting posture from the save path.

    --roll_over_model_path_auto writes models/roll_over/<date>/<posture>/<date>_<posture>_<name>/,
    so the posture appears both as a directory component and in the run directory's name. Returns
    None when neither is present rather than guessing.
    """
    parts = os.path.abspath(model_path).split(os.sep)
    for part in reversed(parts):
        for posture in ('prone', 'supine'):
            if part == posture or re.search(rf'(^|[_-]){posture}([_-]|$)', part):
                return posture
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Path to model_*.zip")
    parser.add_argument('--episodes', default=50, type=int)
    parser.add_argument('--starting_position', default=None,
                        choices=['prone', 'supine'],
                        help="Defaults to the position recorded in data.yml, else supine.")
    parser.add_argument('--label', default=None, help="Name to print for this run.")
    parser.add_argument('--episode_steps', default=None, type=int,
                        help="Episode horizon. Defaults to the one recorded in the run's "
                             "data.yml, else 500. Override only to compare runs trained at "
                             "different horizons on a common one -- otherwise a policy is scored "
                             "over a length it never saw.")
    parser.add_argument('--goal', default=None, type=float,
                        help="Target rotation fed to the policy as desired_goal. Defaults to "
                             "0.95, or to 0.5 for runs trained with --side_lying. The policy is "
                             "conditioned on this input, so a run trained at a fixed 0.5 scores "
                             "far worse if queried at 0.95 -- that is out of its training "
                             "distribution, not a failure to roll.")
    parser.add_argument('--policy_goal', default=None, type=float,
                        help="Feed the policy this constant desired_goal instead of the real one. "
                             "The environment keeps --goal, so the score stays honest; only the "
                             "policy's input is a lie. With a constant input the behaviour cannot "
                             "depend on the requested goal, which is how you build a goal-agnostic "
                             "policy out of a goal-conditioned one without retraining.")
    parser.add_argument('--policy_goal_sweep', default=None,
                        help="Sweep the fed desired_goal instead of a single value. Either a "
                             "comma-separated list ('0.05,0.5,2.0') or 'low:high:step' "
                             "('0.25:0.95:0.05'). Prints one row per value. Locates the point "
                             "where goal conditioning stops working.")
    parser.add_argument('--json', default=None, type=str,
                        help="Also write the results to this path as JSON. The printed table is "
                             "for reading; this is for tooling (mimolab stores it in its index). "
                             "Both carry the same numbers.")
    args = parser.parse_args()

    config = load_run_config(args.model)
    algorithm = config.get('algorithm', 'SAC')
    # 19.08.2026 'roll_over_starting_position' is deliberately not stored in data.yml (it
    # describes the invocation, not the model), so config.get() here always missed and every
    # prone run was silently evaluated as supine -- 198 checkpoints in models/ are affected.
    # The save path does record the posture, so read it from there before falling back.
    start = (args.starting_position
             or config.get('roll_over_starting_position')
             or starting_position_from_path(args.model)
             or 'supine')
    if args.goal is not None:
        goal = args.goal
    elif config.get('side_lying', False):
        goal = SIDE_LYING_THRESHOLD
    else:
        goal = ROLL_THRESHOLD

    # A run trained with --episode_steps is scored over its own horizon, not the default.
    episode_steps = args.episode_steps or config.get('episode_steps') or DEFAULT_EPISODE_STEPS

    env = build_env(config, start, goal)
    model = load_policy(args.model, algorithm, env)
    label = args.label or os.path.basename(os.path.dirname(args.model))

    payload = {'run': label, 'model': os.path.abspath(args.model), 'algorithm': algorithm,
               'her': bool(config.get('her', False)),
               'reward': 'sparse' if config.get('sparse_reward') else (
                   'pbrs' if config.get('pbrs') else 'distance'),
               'starting_position': start, 'goal': goal, 'episodes': args.episodes,
               'episode_steps': episode_steps, 'rows': []}

    if args.policy_goal_sweep is not None:
        values = parse_sweep(args.policy_goal_sweep)
        print()
        print(f"run                 : {label}")
        print(f"episodes            : {args.episodes} per row (deterministic, ISR off, {start}, "
              f"{episode_steps} steps)")
        print(f"env goal            : {goal:.2f} (fixed; rho_max is measured from the simulation, "
              f"so it does not depend on this)")
        print()
        print(f"{'fed goal':>9}  {'roll':>6}  {'side':>6}  {'rho_max mean':>13}  {'min':>6}  "
              f"{'max':>6}")
        for value in values:
            r = evaluate(model, env, args.episodes, policy_goal=value,
                         episode_steps=episode_steps)
            print(f"{value:>9.2f}  {r['rolled'].mean() * 100:>5.0f}%  {r['side'].mean() * 100:>5.0f}%"
                  f"  {r['rho_max'].mean():>13.3f}  {r['rho_max'].min():>6.3f}  "
                  f"{r['rho_max'].max():>6.3f}")
            payload['rows'].append(_row(r, policy_goal=value))
        env.close()
        write_json(args.json, payload)
        return

    results = evaluate(model, env, args.episodes, policy_goal=args.policy_goal,
                       episode_steps=episode_steps)
    env.close()

    finished = results['steps'][~np.isnan(results['steps'])]
    print()
    print(f"run                 : {label}")
    print(f"algorithm / her     : {algorithm} / {config.get('her', False)}")
    print(f"reward              : {'sparse' if config.get('sparse_reward') else ('pbrs' if config.get('pbrs') else 'distance')}")
    print(f"episodes            : {args.episodes} (deterministic, ISR off, goal={goal:.2f}, "
          f"{start}, {episode_steps} steps)")
    if args.policy_goal is not None:
        print(f"policy was fed      : desired_goal={args.policy_goal:.2f} (constant -- the score "
              f"below is still measured against {goal:.2f})")
    print(f"full roll (rho>=.95): {results['rolled'].mean() * 100:.0f} %")
    print(f"side lying(rho>=.50): {results['side'].mean() * 100:.0f} %")
    print(f"rho_max mean/min    : {results['rho_max'].mean():.3f} / {results['rho_max'].min():.3f}")
    if finished.size:
        print(f"steps to roll       : {finished.mean():.1f} +- {finished.std():.1f} (n={finished.size})")
    else:
        print(f"steps to roll       : n/a (no episode reached the goal)")

    payload['rows'].append(_row(results, policy_goal=args.policy_goal))
    write_json(args.json, payload)


if __name__ == '__main__':
    main()

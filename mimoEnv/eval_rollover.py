""" Evaluate a saved roll-over policy under the protocol agreed for the HER experiments.

    MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py --model=<path/to/model_1.zip> [--episodes=50]

or, for a whole training batch at once -- every '<save_path>_run_<i>' directory, one row each:

    MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
        --group=models/roll_over/<yy-mm-dd>/<posture>/<yy-mm-dd>_<posture>_<name> \
        [--episodes=40] [--csv=<name>_test_success_rate.csv]

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

Group mode adds two rules on top:

* **The last checkpoint, not the best one.** 'model_best.zip' is the EvalCallback's pick under
  its own protocol, so a table of best checkpoints measures the checkpoint selection as much as
  the runs. --checkpoint=best is available when that is what you want.
* **A run counts as successful when it rolls in more than 75 % of its episodes** (40 by default),
  the convention used throughout the thesis. --success_threshold moves the line; the printed
  summary also carries the >90 % / <10 % banding that results/success_after_training_plot.py
  draws, and --csv writes the file that script reads.
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
# The desired_goal a run trained with --goal_tolerance was conditioned on. Not a threshold:
# scoring stays 'rho_max >= ROLL_THRESHOLD' for every run, which is what keeps the numbers
# comparable across the two success rules.
FULL_ROLL_GOAL = 1.0
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


def env_kwargs(config, starting_position, goal):
    """The gym.make kwargs for one run's evaluation environment.

    Split out of build_env so that group mode can compare two runs' environments without
    constructing them: one MIMo env is ~3.6 GB RSS, so the group loop keeps a single env alive
    and rebuilds it only when this dict actually changes.
    """
    proprio = config.get('proprio_params')
    kwargs = dict(
        starting_position=starting_position,
        pbrs=config.get('pbrs', False),
        pbrs_w=config.get('pbrs_w', 100),
        pen_factor=config.get('pen_factor', 0.02),
        nopen=config.get('nopen', False),
        sparse_reward=config.get('sparse_reward', False),
        # 26.08.2026 The goal function fixes the width of the goal space, so a run trained with
        # 'gravity' cannot even be loaded against a 'cos' env. Absent from every data.yml written
        # between 26.08.2026 and the readd, where it correctly reads as 'cos'.
        goal_function=config.get('goal_achievement_function', 'cos'),
        gravity_goal_eps=config.get('gravity_goal_eps',
                                    config.get('intrinsic_goal_eps', 0.15)),
        gravity_reference_samples=config.get('gravity_reference_samples',
                                             config.get('intrinsic_reference_samples', 20)),
        # 25.08.2026 Band instead of threshold, if the run was trained that way. Absent from
        # every data.yml written before that date, where it correctly reads as None.
        goal_tolerance=config.get('goal_tolerance'),
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
    return kwargs


def goal_label(config, goal):
    """How to describe the environment's desired_goal in the printed header.

    'goal_low'/'goal_high' pin a scalar for 'cos', but the gravity goal ignores them: its target
    is the reference vector recorded in the opposite posture. Printing "goal=0.95" for such a run
    would name a number the environment never used.
    """
    if config.get('goal_achievement_function', 'cos') == 'gravity':
        return "opposite-posture reference (gravity)"
    return f"{goal:.2f}"


def build_env(config, starting_position, goal):
    return gym.make('MIMoRollOver-v0', **env_kwargs(config, starting_position, goal)).unwrapped


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


RUN_DIR_RE = re.compile(r'_run_(\d+)$')
# 'model_best.zip' is the EvalCallback's pick and 'model_intermediate_90.zip' the 90%-side-lying
# snapshot. Neither is a checkpoint of the training schedule, so neither counts as "the last one".
CHECKPOINT_RE = re.compile(r'^model_(\d+)\.zip$')
# A model is called successful when it rolls in more than this fraction of the evaluated episodes.
# The standard used throughout the thesis (results/success_after_training_plot.py bands the same
# per-run success rates for its stacked bars).
SUCCESS_THRESHOLD = 0.75


def discover_runs(spec):
    """Expand a group specification into the run directories it names, sorted by run index.

    'spec' may be

    * a run directory itself (it contains model_*.zip)  -- a group of one;
    * a directory holding '<...>_run_<i>' subdirectories -- all of them;
    * a path prefix, i.e. the run directory name without the '_run_<i>' tail, which is what
      --roll_over_model_path_auto builds from --save_model;
    * a shell glob.

    Directories without a usable checkpoint are not filtered here; pick_checkpoint reports them.
    """
    import glob as globlib

    candidates = []
    if os.path.isdir(spec):
        if globlib.glob(os.path.join(spec, 'model_*.zip')):
            return [os.path.abspath(spec)]
        candidates = [d for d in globlib.glob(os.path.join(spec, '*')) if os.path.isdir(d)]
        candidates = [d for d in candidates if RUN_DIR_RE.search(os.path.basename(d))]
    if not candidates:
        candidates = [d for d in globlib.glob(spec) if os.path.isdir(d)]
    if not candidates:
        # Prefix form: 'models/.../26-08-23_supine_td3_her_ep200' -> '..._run_0', '..._run_1', ...
        candidates = [d for d in globlib.glob(spec + '_run_*') if os.path.isdir(d)]

    def sort_key(path):
        # Group first, index second: a directory holding several sweeps stays readable instead of
        # interleaving run_0 of every suffix.
        name = os.path.basename(path)
        match = RUN_DIR_RE.search(name)
        return (name[:match.start()], 0, int(match.group(1))) if match else (name, 1, 0)

    return sorted((os.path.abspath(d) for d in candidates), key=sort_key)


def pick_checkpoint(run_dir, which='last'):
    """The checkpoint of 'run_dir' to evaluate. Returns None when there is none.

    'last' is the highest-numbered model_<n>.zip -- the end of training, deliberately *not*
    model_best.zip: 'best' is scored by the EvalCallback under its own protocol, and comparing
    runs by their best checkpoint measures the checkpoint selection as much as the run.
    """
    if which not in ('last', 'best'):
        path = os.path.join(run_dir, which)
        return path if os.path.exists(path) else None
    if which == 'best':
        path = os.path.join(run_dir, 'model_best.zip')
        return path if os.path.exists(path) else None
    numbered = []
    for name in os.listdir(run_dir):
        match = CHECKPOINT_RE.match(name)
        if match:
            numbered.append((int(match.group(1)), os.path.join(run_dir, name)))
    if not numbered:
        return None
    return max(numbered)[1]


def resolve_run(model_path, args):
    """Everything the protocol needs for one model, read from its own data.yml."""
    config = load_run_config(model_path)
    # 19.08.2026 'roll_over_starting_position' is deliberately not stored in data.yml (it
    # describes the invocation, not the model), so config.get() here always missed and every
    # prone run was silently evaluated as supine -- 198 checkpoints in models/ are affected.
    # The save path does record the posture, so read it from there before falling back.
    start = (args.starting_position
             or config.get('roll_over_starting_position')
             or starting_position_from_path(model_path)
             or 'supine')
    if args.goal is not None:
        goal = args.goal
    elif config.get('side_lying', False):
        goal = SIDE_LYING_THRESHOLD
    elif config.get('goal_tolerance') is not None:
        # Trained with the band criterion, so its full-roll goal is 1.0 rather than 0.95. The
        # policy is conditioned on 'desired_goal'; pinning 0.95 here would feed it a number it
        # never saw. The reported score is unaffected either way -- 'evaluate' measures rho_max
        # off the simulation and never calls 'is_success'.
        goal = FULL_ROLL_GOAL
    else:
        goal = ROLL_THRESHOLD
    # A run trained with --episode_steps is scored over its own horizon, not the default.
    episode_steps = args.episode_steps or config.get('episode_steps') or DEFAULT_EPISODE_STEPS
    return config, start, goal, episode_steps


class _EnvCache:
    """Holds one evaluation env alive across runs, rebuilding it only when the config changes.

    A MIMo env costs ~3.6 GB RSS, so group mode may never hold two at once. Runs of the same
    sweep share their data.yml apart from the seed, so in practice this builds the env once.
    """

    def __init__(self):
        self._signature = None
        self._env = None

    def get(self, config, start, goal):
        signature = json.dumps(env_kwargs(config, start, goal), sort_keys=True, default=str)
        if signature != self._signature:
            self.close()
            self._env = build_env(config, start, goal)
            self._signature = signature
        return self._env

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env, self._signature = None, None


def evaluate_group(run_dirs, args, episodes):
    """Evaluate the chosen checkpoint of every run and classify each one.

    Every run sees the same episode seeds, so the comparison between runs is paired.
    """
    cache = _EnvCache()
    rows, skipped = [], []
    try:
        for index, run_dir in enumerate(run_dirs, start=1):
            name = os.path.basename(run_dir)
            model_path = pick_checkpoint(run_dir, args.checkpoint)
            if model_path is None:
                skipped.append((name, f"no checkpoint matching '{args.checkpoint}'"))
                continue
            config, start, goal, episode_steps = resolve_run(model_path, args)
            algorithm = config.get('algorithm', 'SAC')
            print(f"[{index}/{len(run_dirs)}] {name}  ({os.path.basename(model_path)}, "
                  f"{algorithm}, {start}, {episode_steps} steps)", flush=True)
            env = cache.get(config, start, goal)
            model = load_policy(model_path, algorithm, env)
            results = evaluate(model, env, episodes, policy_goal=args.policy_goal,
                               episode_steps=episode_steps)
            row = _row(results, policy_goal=args.policy_goal)
            row.update(run=name, model=model_path, checkpoint=os.path.basename(model_path),
                       algorithm=algorithm, starting_position=start, goal=goal,
                       episode_steps=episode_steps, episodes=episodes,
                       successful=bool(row['rolled'] > args.success_threshold))
            rows.append(row)
    finally:
        cache.close()
    return rows, skipped


def _summarise(rows, threshold):
    rolled = np.array([row['rolled'] for row in rows], dtype=float)
    rho = np.array([row['rho_mean'] for row in rows], dtype=float)
    steps = np.array([row['steps_mean'] for row in rows if row['steps_mean'] is not None],
                     dtype=float)
    return {
        'runs': len(rows),
        'success_threshold': threshold,
        'successful': int((rolled > threshold).sum()),
        'success_fraction': float((rolled > threshold).mean()),
        'roll_rate_mean': float(rolled.mean()),
        'roll_rate_std': float(rolled.std()),
        'rho_mean': float(rho.mean()),
        'steps_mean': float(steps.mean()) if steps.size else None,
        # The banding of results/success_after_training_plot.py, kept so that the stacked bar
        # chart of the thesis can be read straight off this summary.
        'band_successful_90': int((rolled > 0.9).sum()),
        'band_not_successful_10': int((rolled < 0.1).sum()),
        'band_ambiguous': int(len(rows) - (rolled > 0.9).sum() - (rolled < 0.1).sum()),
    }


def _print_group(rows, skipped, summary, args, episodes):
    prefix = os.path.commonprefix([row['run'] for row in rows]) if len(rows) > 1 else ''
    # Cut the common prefix back to a '_' boundary: for the pair {run_1, run_10} the raw common
    # prefix ends inside the index and the table would list them as '' and '0'.
    prefix = prefix[:prefix.rfind('_') + 1]
    width = max([len(row['run'][len(prefix):]) for row in rows] + [4])
    print()
    print(f"group               : {args.group}")
    if prefix:
        print(f"common prefix       : {prefix}")
    print(f"checkpoint          : {args.checkpoint} "
          f"({'highest-numbered model_<n>.zip' if args.checkpoint == 'last' else args.checkpoint})")
    print(f"episodes            : {episodes} per run (deterministic, ISR off)")
    if args.policy_goal is not None:
        print(f"policy was fed      : desired_goal={args.policy_goal:.2f} (constant)")
    print()
    print(f"{'run':<{width}}  {'roll':>6}  {'side':>6}  {'rho mean':>8}  {'rho min':>7}  "
          f"{'steps':>7}  status")
    for row in rows:
        steps = f"{row['steps_mean']:.1f}" if row['steps_mean'] is not None else "-"
        print(f"{row['run'][len(prefix):]:<{width}}  {row['rolled'] * 100:>5.0f}%  "
              f"{row['side'] * 100:>5.0f}%  {row['rho_mean']:>8.3f}  {row['rho_min']:>7.3f}  "
              f"{steps:>7}  {'successful' if row['successful'] else '-'}")
    for name, reason in skipped:
        print(f"{name[len(prefix):]:<{width}}  skipped: {reason}")
    print()
    print(f"runs evaluated      : {summary['runs']}")
    print(f"successful (>{args.success_threshold * 100:.0f}%)   : {summary['successful']} / "
          f"{summary['runs']} ({summary['success_fraction'] * 100:.1f} %)")
    print(f"roll rate           : {summary['roll_rate_mean'] * 100:.1f} % +- "
          f"{summary['roll_rate_std'] * 100:.1f} (mean over runs)")
    print(f"rho_max mean        : {summary['rho_mean']:.3f}")
    if summary['steps_mean'] is not None:
        print(f"steps to roll       : {summary['steps_mean']:.1f} (mean over runs that rolled)")
    print(f"thesis banding      : successful >90%: {summary['band_successful_90']} | "
          f"ambiguous: {summary['band_ambiguous']} | "
          f"not successful <10%: {summary['band_not_successful_10']}")


def write_csv(path, rows):
    """Write the per-run success rates in the layout results/success_after_training_plot.py reads.

    That script reads 'Run' and 'Success_Rate' out of
    '<date>_<haltung>_<suffix>_test_success_rate.csv', so the group mode can feed the existing
    stacked-bar figure without going through results/success_after_training.py.
    """
    if not path:
        return
    import csv
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Run', 'Success_Rate'])
        for row in rows:
            match = RUN_DIR_RE.search(row['run'])
            writer.writerow([match.group(1) if match else row['run'], row['rolled']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help="Path to a single model_*.zip.")
    parser.add_argument('--group', default=None,
                        help="Evaluate every run of a training batch instead of a single model. "
                             "Takes the run directories' shared path prefix (the save path "
                             "without the '_run_<i>' tail), a directory holding them, or a glob. "
                             "One checkpoint per run -- see --checkpoint.")
    parser.add_argument('--checkpoint', default='last',
                        help="Which checkpoint of each run to evaluate in --group mode: 'last' "
                             "(default, the highest-numbered model_<n>.zip, i.e. the end of "
                             "training), 'best' (model_best.zip), or an explicit file name such "
                             "as 'model_3.zip'. 'last' is the default on purpose: comparing runs "
                             "by their best checkpoint measures the checkpoint selection as much "
                             "as the run.")
    parser.add_argument('--success_threshold', default=SUCCESS_THRESHOLD, type=float,
                        help="A run counts as successful when it rolls in more than this "
                             "fraction of its episodes (default 0.75, the thesis standard).")
    parser.add_argument('--episodes', default=None, type=int,
                        help="Episodes per model. Defaults to 40 with --group (the thesis "
                             "standard) and 50 for a single --model.")
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
    parser.add_argument('--csv', default=None, type=str,
                        help="--group only. Write per-run success rates as Run,Success_Rate -- "
                             "the layout results/success_after_training_plot.py reads, so the "
                             "stacked-bar figure can be built straight from this run.")
    parser.add_argument('--json', default=None, type=str,
                        help="Also write the results to this path as JSON. The printed table is "
                             "for reading; this is for tooling (mimolab stores it in its index). "
                             "Both carry the same numbers.")
    args = parser.parse_args()

    if bool(args.model) == bool(args.group):
        parser.error("Pass exactly one of --model (a single checkpoint) or --group (every run "
                     "of a training batch).")

    if args.group:
        if args.policy_goal_sweep is not None:
            parser.error("--policy_goal_sweep evaluates one model across many fed goals; it does "
                         "not combine with --group. Sweep a single --model instead.")
        episodes = args.episodes or 40
        run_dirs = discover_runs(args.group)
        if not run_dirs:
            parser.error(f"No run directories matched --group={args.group!r}.")
        rows, skipped = evaluate_group(run_dirs, args, episodes)
        if not rows:
            parser.error("No run had an evaluable checkpoint.")
        summary = _summarise(rows, args.success_threshold)
        _print_group(rows, skipped, summary, args, episodes)
        write_csv(args.csv, rows)
        write_json(args.json, {'group': args.group, 'checkpoint': args.checkpoint,
                               'episodes': episodes, 'summary': summary, 'rows': rows,
                               'skipped': [{'run': n, 'reason': r} for n, r in skipped]})
        return

    episodes = args.episodes or 50
    config, start, goal, episode_steps = resolve_run(args.model, args)
    algorithm = config.get('algorithm', 'SAC')

    env = build_env(config, start, goal)
    model = load_policy(args.model, algorithm, env)
    label = args.label or os.path.basename(os.path.dirname(args.model))

    payload = {'run': label, 'model': os.path.abspath(args.model), 'algorithm': algorithm,
               'her': bool(config.get('her', False)),
               'reward': 'sparse' if config.get('sparse_reward') else (
                   'pbrs' if config.get('pbrs') else 'distance'),
               'starting_position': start, 'goal': goal, 'episodes': episodes,
               'episode_steps': episode_steps, 'rows': []}

    if args.policy_goal_sweep is not None:
        values = parse_sweep(args.policy_goal_sweep)
        print()
        print(f"run                 : {label}")
        print(f"episodes            : {episodes} per row (deterministic, ISR off, {start}, "
              f"{episode_steps} steps)")
        print(f"env goal            : {goal_label(config, goal)} (fixed; rho_max is measured "
              f"from the simulation, so it does not depend on this)")
        print()
        print(f"{'fed goal':>9}  {'roll':>6}  {'side':>6}  {'rho_max mean':>13}  {'min':>6}  "
              f"{'max':>6}")
        for value in values:
            r = evaluate(model, env, episodes, policy_goal=value,
                         episode_steps=episode_steps)
            print(f"{value:>9.2f}  {r['rolled'].mean() * 100:>5.0f}%  {r['side'].mean() * 100:>5.0f}%"
                  f"  {r['rho_max'].mean():>13.3f}  {r['rho_max'].min():>6.3f}  "
                  f"{r['rho_max'].max():>6.3f}")
            payload['rows'].append(_row(r, policy_goal=value))
        env.close()
        write_json(args.json, payload)
        return

    results = evaluate(model, env, episodes, policy_goal=args.policy_goal,
                       episode_steps=episode_steps)
    env.close()

    finished = results['steps'][~np.isnan(results['steps'])]
    print()
    print(f"run                 : {label}")
    print(f"algorithm / her     : {algorithm} / {config.get('her', False)}")
    print(f"reward              : {'sparse' if config.get('sparse_reward') else ('pbrs' if config.get('pbrs') else 'distance')}")
    print(f"episodes            : {episodes} (deterministic, ISR off, goal={goal_label(config, goal)}, "
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

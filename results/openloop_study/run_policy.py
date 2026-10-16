"""
13.10.2026 Can a learned roll be played back as a fixed sequence of actions, or does MIMo need
to see what is happening while he rolls?

This script tests ONE trained policy and saves everything into ONE file.

Run it from the MIMo folder:

    MUJOCO_GL=osmesa python results/openloop_study/run_policy.py \
        --model models/roll_over/.../model_1.zip \
        --out   results/openloop_study/data/supine/run_0.npz

It does three things, one after the other:

  Part 1  RECORD
          The policy rolls from 10 different start states. At every step we store the action
          and the joint angles.

  Part 2  PLAY BACK ON OTHER START STATES
          The actions recorded on start state i are simply played back on start state j.
          The network is not used at all. We do this for all 10 x 10 combinations.

  Part 3  DISTURBANCES
          We disturb MIMo (for example with noise on the actions) and compare two cases:
            "policy"   the network sees the state at every step and can react
            "replay"   the recorded actions are played back no matter what happens
          Both get exactly the same disturbance.

An episode counts as "rolled" if rho reaches 0.95 at some point.
rho is 0 in the start posture and 1 after a complete roll.

The plots are made in openloop_study.ipynb.
"""

import argparse
import os
import sys

import numpy as np
import gymnasium as gym

MIMO_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, MIMO_FOLDER)
import mimoEnv  # noqa: F401,E402  (registers the MIMo environments with gymnasium)
from mimoEnv.eval_rollover import (env_kwargs, load_policy, load_run_config,  # noqa: E402
                                   starting_position_from_path, LATERALITY_BODIES)

N_STARTS = 10        # number of start states
N_STEPS = 300        # length of one episode in steps (1 step = 10 ms)
FIRST_SEED = 1000    # start state j comes from seed 1000 + j (same as eval_rollover.py)
ROLLED = 0.95        # an episode counts as rolled from this rho on

# How strong the disturbances in part 3 are.
START_NOISE = [0.01, 0.03, 0.1, 0.2, 0.3]   # random offset on every joint angle at the start, in rad
MOTOR_NOISE = [0.1, 0.2, 0.4, 0.8]          # random offset on every action at every step
PUSH = [-10, -5, -2, 2, 5, 10]              # short push in N m:  + in roll direction, - against it
SLOPE = [-15, -10, -5, 5, 10, 15]           # tilted floor in degrees: + downhill, - uphill


def run_episode(env, seed, choose_action, stop_at_roll=True,
                start_noise=None, motor_noise=None, push=0.0, slope=0.0):
    """Runs MIMo for one episode and returns what happened.

    choose_action(t, obs)  function that gives the action for step t.
                           This is either the network or a recorded sequence.
    stop_at_roll           True: the episode ends as soon as MIMo has rolled (saves time).

    The four disturbances (all of them are normally off):
    start_noise   one value per joint, added to the joint angles at the start
    motor_noise   table (step x actuator), added to the actions
    push          torque around MIMo's long axis, only active in steps 5 to 14
    slope         tilt of the floor in degrees
    """
    # --- Tilt the floor. We rotate gravity instead of the floor, which is the same thing. ---
    # A positive angle means the floor rises towards MIMo's left side.
    gravity_before = env.model.opt.gravity.copy()
    g = np.linalg.norm(gravity_before)
    angle = np.deg2rad(slope)
    env.model.opt.gravity[:] = [0.0, -g * np.sin(angle), -g * np.cos(angle)]

    # --- Create the start state. The same seed always gives the same start state. ---
    # If start_noise is given, the environment uses these values instead of its own small
    # random offsets (+-0.01 rad).
    env.deterministic_initial_state_sampling = start_noise
    obs, _ = env.reset(seed=seed)

    hip_and_chest = [env.data.body(name).id for name in LATERALITY_BODIES]
    actions, joint_angles = [], []
    rho_max = 0.0
    side = "none"          # the side MIMo rolls to: "left", "right" or "none"
    rolled_at_step = -1    # the step in which he rolled, -1 = not at all

    for t in range(N_STEPS):
        # The first 7 values of qpos are the position and rotation of the whole body.
        joint_angles.append(env.data.qpos[7:].copy())

        action = np.array(choose_action(t, obs), dtype=float)
        if motor_noise is not None:
            action = action + motor_noise[t]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        actions.append(action)

        # Push: entry 3 of xfrc_applied is the torque around the x axis of the world.
        # MIMo lies along x, so this turns him around his long axis.
        # Half of the torque goes to the hip and half to the chest.
        push_now = push if 5 <= t < 15 else 0.0
        env.data.xfrc_applied[hip_and_chest, 3] = push_now / 2

        obs, _, _, _, _ = env.step(action)

        rho = float(env.get_achieved_goal_cos_mean()[0])
        rho_max = max(rho_max, rho)

        # Find out the side as soon as MIMo lies on his side (rho >= 0.5).
        if side == "none" and rho >= 0.5:
            left_up = np.mean([env.data.body(b).xmat.reshape(3, 3)[2, 1] for b in hip_and_chest])
            side = "left" if left_up < 0 else "right"

        if rolled_at_step < 0 and rho >= ROLLED:
            rolled_at_step = t + 1
            if stop_at_roll:
                break

    # --- Clean up, so that the next episode starts undisturbed. ---
    env.data.xfrc_applied[:] = 0.0
    env.model.opt.gravity[:] = gravity_before
    env.deterministic_initial_state_sampling = None

    return dict(rho_max=rho_max, rolled_at_step=rolled_at_step, side=side,
                actions=np.array(actions), joint_angles=np.array(joint_angles))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model_1.zip")
    parser.add_argument("--out", required=True, help="Result file (.npz)")
    parser.add_argument("--starts", type=int, default=N_STARTS, help="Make this smaller only for a quick test")
    args = parser.parse_args()
    n = args.starts
    seeds = [FIRST_SEED + j for j in range(n)]

    # --- Load the environment and the policy exactly like eval_rollover.py does. ---
    config = load_run_config(args.model)                      # reads data.yml of the training run
    posture = starting_position_from_path(args.model)         # "supine" or "prone", taken from the path
    kwargs = env_kwargs(config, posture, ROLLED)
    kwargs["strip_textures"] = True                           # saves 2 GB of RAM, does not change the physics
    env = gym.make("MIMoRollOver-v0", **kwargs).unwrapped
    model = load_policy(args.model, config.get("algorithm", "PPO"), env)
    n_joints = env.data.qpos.size - 7
    n_actuators = env.action_space.shape[0]

    # Two ways to choose an action:
    def policy(t, obs):
        """The network looks at the observation and decides."""
        action, _ = model.predict(obs, deterministic=True)
        return action

    def replay(sequence):
        """The observation is ignored, we just return the recorded action number t."""
        return lambda t, obs: sequence[t]

    # =====================================================================================
    # Part 1: record
    # =====================================================================================
    print(f"{args.model}\nPart 1: record the policy on {n} start states", flush=True)
    recorded = [run_episode(env, seed, policy, stop_at_roll=False) for seed in seeds]
    actions = np.array([e["actions"] for e in recorded])             # (start, step, actuator)
    joint_angles = np.array([e["joint_angles"] for e in recorded])   # (start, step, joint)
    rho_max = np.array([e["rho_max"] for e in recorded])
    rolled_at_step = np.array([e["rolled_at_step"] for e in recorded])
    sides = [e["side"] for e in recorded]
    rolls_left = sides.count("left") >= sides.count("right")
    print(f"  rolled: {100 * np.mean(rho_max >= ROLLED):.1f} %, side: {'left' if rolls_left else 'right'}")

    # =====================================================================================
    # Part 2: play the sequence of start state i back on start state j
    # =====================================================================================
    print("Part 2: play back on other start states", flush=True)
    replay_matrix = np.zeros((n, n))        # replay_matrix[i, j] = rho_max
    for i in range(n):
        for j in range(n):
            replay_matrix[i, j] = run_episode(env, seeds[j], replay(actions[i]))["rho_max"]
    other = ~np.eye(n, dtype=bool)          # all cells with i != j
    print(f"  own start state (i = j):    {100 * np.mean(np.diag(replay_matrix) >= ROLLED):.1f} % rolled")
    print(f"  other start state (i != j): {100 * np.mean(replay_matrix[other] >= ROLLED):.1f} % rolled")

    # =====================================================================================
    # Part 3: disturbances, policy against replay
    # =====================================================================================
    # Push and slope should mean "in roll direction" (+) or "against it" (-), no matter which
    # side the policy rolls to. A positive torque around x and a floor that rises to the left
    # both help a roll to the RIGHT. For policies that roll left we therefore flip the sign.
    helps = -1.0 if rolls_left else 1.0

    def disturbance(kind, level, j):
        """Builds the disturbance for start state j. The random numbers depend only on j,
        so the policy and the replay get exactly the same disturbance."""
        if kind == "start_noise":
            return dict(start_noise=np.random.default_rng(5000 + j).uniform(-level, level, n_joints))
        if kind == "motor_noise":
            return dict(motor_noise=np.random.default_rng(7000 + j).normal(0, level, (N_STEPS, n_actuators)))
        if kind == "push":
            return dict(push=helps * level)
        if kind == "slope":
            return dict(slope=helps * level)

    results = {}
    for kind, levels in (("start_noise", START_NOISE), ("motor_noise", MOTOR_NOISE),
                         ("push", PUSH), ("slope", SLOPE)):
        with_policy = np.zeros((len(levels), n))    # rho_max, row = strength, column = start state
        with_replay = np.zeros((len(levels), n))
        for k, level in enumerate(levels):
            for j in range(n):
                d = disturbance(kind, level, j)
                with_policy[k, j] = run_episode(env, seeds[j], policy, **d)["rho_max"]
                # We play back the sequence recorded on the same start state WITHOUT disturbance.
                with_replay[k, j] = run_episode(env, seeds[j], replay(actions[j]), **d)["rho_max"]
            print(f"  {kind:12s} {level:6g}:  policy {100 * np.mean(with_policy[k] >= ROLLED):5.1f} %   "
                  f"replay {100 * np.mean(with_replay[k] >= ROLLED):5.1f} %", flush=True)
        results[kind + "_levels"] = np.array(levels, dtype=float)
        results[kind + "_policy"] = with_policy
        results[kind + "_replay"] = with_replay

    np.savez_compressed(args.out, model=args.model, posture=posture, rolls_left=rolls_left,
                        actions=actions.astype(np.float32), joint_angles=joint_angles.astype(np.float32),
                        rho_max=rho_max, rolled_at_step=rolled_at_step,
                        replay_matrix=replay_matrix, **results)
    print(f"saved: {args.out}")
    env.close()


if __name__ == "__main__":
    main()

""" How much of a muscle policy's effort is co-contraction?

    MUJOCO_GL=osmesa python mimoEnv/eval_cocontraction.py --model=<run>/model_4.zip
    MUJOCO_GL=osmesa python mimoEnv/eval_cocontraction.py --group=<save_path>   # all seeds
    python mimoEnv/eval_cocontraction.py --selfcheck                           # no MuJoCo

Protocol and seeds are ``eval_rollover.py``'s (``resolve_run``/``build_env``, deterministic
actions, seeds 1000 + episode, 40 episodes by default), so the episodes are the same ones the roll
rate is scored on. Needs a ``--use_muscle`` run: under the spring-damper model one bidirectional
motor per joint cannot co-contract, so there is nothing to measure.

What is measured
----------------
Every actuator of ``MuscleModel`` drives one joint through two muscles pulling in opposite
directions (``moment_1 > 0 > moment_2`` by construction), so each pair is an exact
agonist/antagonist pair. Per env step and per pair, with ``a`` the lagged ``activity`` -- the
quantity ``MuscleModel.cost`` squares, i.e. what ``--pen_metabolic`` pays and
``rollout/metabolic_cost_mean`` logs:

* **Metabolic cost** of one muscle is ``a^2 * fmax``, normalised exactly as ``MuscleModel.cost``
  (``/ (2n * sum fmax)``), so the episode sums printed here are on the scale of
  ``rollout/metabolic_cost_mean``. The script asserts it reproduces ``info['metabolic_cost']``.

* **Wasted cost** is the actual cost minus the cheapest activation that produces the *same net
  active joint torque in the same state*. A muscle's active torque is
  ``-moment * fl(lce) * fv(lce_dot) * fmax * a`` (``MuscleModel._update_torque``; the passive
  ``fp`` term does not depend on activation and is left out of the accounting). Because cost is
  convex and the two muscles pull against each other, the cheapest way to deliver a net torque is
  the agonist alone, at ``a* = |net torque| / |moment * fl * fv * fmax|_agonist`` -- anything the
  antagonist does has to be paid twice, once by itself and once by the extra agonist activation
  that cancels it. So ``waste = a_ant^2 fmax_ant + (a_ag^2 - a*^2) fmax_ag``. An unopposed
  muscle has zero waste by construction.

  This is an *instantaneous* counterfactual: it keeps the current joint angles and velocities, and
  does not ask what a different activation would have done to the trajectory afterwards. It is
  also a *mechanical* waste measure. Co-contraction is not useless -- it stiffens joints and is
  what infants do early in learning a movement (Siegel et al. 2024 report every muscle group
  firing at roll onset) -- "wasted" means only that it produced no net torque.

* **Wasted activation** is the same counterfactual on activation instead of cost:
  ``(a_ant + a_ag - a*) / (a_ant + a_ag)``, pooled over muscles and steps.

* **Falconer & Winter co-contraction index** (1985, Electromyogr. Clin. Neurophysiol. 25:135) is
  printed alongside because it is the index EMG studies report: ``2 * min(a1, a2) / (a1 + a2)``.
  It ignores that the two muscles differ in strength (``fmax``) and in operating point (``fl``,
  ``fv``), so it can call a pair fully co-contracted while it still delivers net torque. On two
  muscles at 0.5 activation, one of them twice as strong, the index reads 100 % and the
  torque-preserving waste 83 % of cost. Use the index for comparison with the literature, the
  counterfactual for "how much was paid for nothing".

All shares are computed per episode and then averaged over episodes (mean +- sd). Four windows
are reported:

* the whole episode;
* **until the roll** -- every step before the first ``rho >= 0.95``, or the whole episode if MIMo
  never gets there. This is the part to quote for a trained policy. Training runs with
  ``done_active`` and ends the episode on success, so this is all a policy ever practised; the
  evaluation protocol runs ``done_active=False`` for the full horizon, and what follows the roll
  is behaviour the reward never shaped. It is also what ``rollout/metabolic_cost_mean`` logs, so
  the two are on the same footing;
* the rolling movement (supine to lateral rotation, ``rho <= 0.05`` to ``rho >= 0.5``,
  ``eval_emg.rolling_window`` -- Siegel's window);
* everything after the first step at ``rho >= 0.95`` -- the untrained remainder, reported so it
  can be kept apart rather than silently diluting the episode numbers.

A degenerate case is counted and printed rather than hidden: a pair whose agonist has no force
capacity at all (``fl * fv * fmax ~ 0``, i.e. shortening faster than ``vmax``) has no defined
cheapest activation. There only the antagonist is removed, which undercounts waste.
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

EPS = 1e-12
# Agonist force capacity below which the counterfactual is undefined (see module docstring).
CAPACITY_EPS = 1e-9
WINDOWS = ("episode", "until_roll", "roll", "after_roll")


# --------------------------------------------------------------------------------------------
# The accounting. Pure numpy, shared by the evaluation and the selfcheck.
# --------------------------------------------------------------------------------------------

def cost_normaliser(fmax):
    """ The divisor of ``MuscleModel.cost``, ``2n * sum(fmax)``, for the full (2n,) fmax vector.

    One definition for every caller: a mutation test showed a wrong factor in one of two copies
    passing the selfcheck, because the shares cancel it and only the absolute costs move.
    """
    return fmax.size * fmax.sum()


def pair_accounting(a_neg, a_pos, cap_neg, cap_pos, fmax_neg, fmax_pos):
    """ Actual and minimum-cost activation of every agonist/antagonist pair.

    All arguments broadcast; the last axis is the actuator. ``cap_*`` is the *signed* active
    torque per unit activation, ``-moment * fl * fv * fmax``, so the net active torque is
    ``cap_neg * a_neg + cap_pos * a_pos``.

    Returns:
        dict: 'cost' and 'cost_min' (per muscle pair, un-normalised ``sum a^2 fmax``),
        'act' and 'act_min' (summed activation of the pair), 'cci_shared' (``2 min(a1, a2)``),
        'torque' and 'torque_min' (net active torque before and after), 'undefined' (bool).
    """
    a_neg, a_pos = np.asarray(a_neg, float), np.asarray(a_pos, float)
    t_neg, t_pos = cap_neg * a_neg, cap_pos * a_pos
    torque = t_neg + t_pos

    # The agonist is the muscle delivering more active torque; ties go to the larger activation.
    neg_is_agonist = (np.abs(t_neg) > np.abs(t_pos)) | (
        (np.abs(t_neg) == np.abs(t_pos)) & (a_neg >= a_pos))
    a_ag = np.where(neg_is_agonist, a_neg, a_pos)
    cap_ag = np.abs(np.where(neg_is_agonist, cap_neg, cap_pos))
    fmax_ag = np.where(neg_is_agonist, fmax_neg, fmax_pos)

    undefined = cap_ag <= CAPACITY_EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        a_star = np.where(undefined, a_ag, np.abs(torque) / np.where(undefined, 1.0, cap_ag))
    # a* <= a_ag holds analytically (the antagonist only ever subtracts); clip float noise.
    a_star = np.clip(a_star, 0.0, a_ag)
    undefined &= (np.minimum(a_neg, a_pos) > 0)

    cost = a_neg ** 2 * fmax_neg + a_pos ** 2 * fmax_pos
    cost_min = a_star ** 2 * fmax_ag
    torque_min = np.sign(torque) * a_star * cap_ag
    return {
        "cost": cost, "cost_min": cost_min,
        "act": a_neg + a_pos, "act_min": a_star,
        "cci_shared": 2.0 * np.minimum(a_neg, a_pos),
        "torque": torque, "torque_min": np.where(undefined, torque, torque_min),
        "undefined": undefined,
    }


def window_summary(acc, mask, normaliser):
    """ Pool one episode's per-step, per-pair accounting over the steps in 'mask'.

    Args:
        acc (dict): Output of :func:`pair_accounting` with arrays shaped (T, n).
        mask (np.ndarray): (T,) bool, the steps to include.
        normaliser (float): ``2n * sum(fmax)``, the divisor of ``MuscleModel.cost``.

    Returns:
        dict|None: Sums and shares for this window, or None when it is empty.
    """
    if not np.any(mask):
        return None
    cost = acc["cost"][mask].sum()
    wasted = cost - acc["cost_min"][mask].sum()
    act = acc["act"][mask].sum()
    act_wasted = act - acc["act_min"][mask].sum()
    return {
        "steps": int(mask.sum()),
        "cost": float(cost / normaliser),
        "cost_wasted": float(wasted / normaliser),
        "cost_share": float(wasted / cost) if cost > EPS else float("nan"),
        "act_mean": float(act / (mask.sum() * 2 * acc["act"].shape[1])),
        "act_share": float(act_wasted / act) if act > EPS else float("nan"),
        "cci": float(acc["cci_shared"][mask].sum() / act) if act > EPS else float("nan"),
        "per_actuator_wasted": ((acc["cost"][mask] - acc["cost_min"][mask]).sum(axis=0)
                                / normaliser),
    }


def episode_windows(rho, onset_rho=0.05, side_rho=0.5, roll_rho=0.95):
    """ Step masks for the three reported windows, over T steps of 'rho'.

    'roll' is ``eval_emg.rolling_window``'s definition, restated so the selfcheck needs no env.
    """
    rho = np.asarray(rho)
    T = rho.size
    masks = {"episode": np.ones(T, bool), "until_roll": np.ones(T, bool),
             "roll": np.zeros(T, bool), "after_roll": np.zeros(T, bool)}
    reached = np.flatnonzero(rho >= side_rho)
    if reached.size:
        end = int(reached[0])
        flat = np.flatnonzero(rho[:end + 1] <= onset_rho)
        start = int(flat[-1]) if flat.size else 0
        if end - start >= 2:
            masks["roll"][start:end + 1] = True
    rolled = np.flatnonzero(rho >= roll_rho)
    if rolled.size:
        masks["after_roll"][int(rolled[0]):] = True
        masks["until_roll"][int(rolled[0]):] = False
    return masks


# --------------------------------------------------------------------------------------------
# Recording.
# --------------------------------------------------------------------------------------------

def collect_episode(env, policy, seed, episode_steps):
    """ Roll out one episode and record what the accounting needs after every step.

    The muscle quantities are read straight off the actuation model after ``env.step``. They are
    one consistent snapshot: ``_update_muscle_state`` sets activity, lengths, velocities and
    forces in the same call during the last physics substep, and that is the torque MuJoCo
    applied. Nothing is recomputed from ``qpos``, which has moved on since.

    Returns:
        dict: 'a' (T, 2n), 'cap' (T, 2n) signed active torque per unit activation, 'rho' (T,),
        'env_cost' (T,), plus the largest consistency errors seen.
    """
    am = env.actuation_model
    n = am.n_actuators
    obs, _ = env.reset(seed=seed)
    a, cap, rho, env_cost = [], [], [], []
    err_force = err_torque = err_cost = 0.0
    normaliser = cost_normaliser(am.fmax)
    for _ in range(episode_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        act = am.activity.copy()
        gain_neg = am.fl(am.lce_1) * am.fv(am.lce_dot_1)
        gain_pos = am.fl(am.lce_2) * am.fv(am.lce_dot_2)
        # The model's own force, rebuilt: pins that 'gain' is what _update_torque multiplied by.
        err_force = max(err_force,
                        np.abs(gain_neg * act[:n] + am.fp(am.lce_1) - am.force_muscles_1).max(),
                        np.abs(gain_pos * act[n:] + am.fp(am.lce_2) - am.force_muscles_2).max())
        cap_neg = -am.moment_1 * gain_neg * am.fmax[:n]
        cap_pos = -am.moment_2 * gain_pos * am.fmax[n:]
        passive = (-am.moment_1 * am.fp(am.lce_1) * am.fmax[:n]
                   - am.moment_2 * am.fp(am.lce_2) * am.fmax[n:])
        torque = cap_neg * act[:n] + cap_pos * act[n:] + passive
        err_torque = max(err_torque, np.abs(torque - am.joint_torque).max()
                         / max(np.abs(am.joint_torque).max(), 1.0))
        own_cost = float((act ** 2 * am.fmax).sum() / normaliser)
        err_cost = max(err_cost, abs(own_cost - info["metabolic_cost"])
                       / max(abs(info["metabolic_cost"]), EPS))

        a.append(act)
        cap.append(np.concatenate([cap_neg, cap_pos]))
        rho.append(float(np.asarray(env.get_achieved_goal_cos_mean()).reshape(-1)[0]))
        env_cost.append(info["metabolic_cost"])
        if terminated or truncated:
            break
    return {"a": np.asarray(a), "cap": np.asarray(cap), "rho": np.asarray(rho),
            "env_cost": np.asarray(env_cost), "err_force": float(err_force),
            "err_torque": float(err_torque), "err_cost": float(err_cost)}


def analyse_episode(data, fmax, n):
    acc = pair_accounting(data["a"][:, :n], data["a"][:, n:], data["cap"][:, :n],
                          data["cap"][:, n:], fmax[:n], fmax[n:])
    normaliser = cost_normaliser(fmax)
    masks = episode_windows(data["rho"])
    row = {name: window_summary(acc, masks[name], normaliser) for name in WINDOWS}
    row["undefined_share"] = float(acc["undefined"].mean())
    row["err_net_torque"] = float(np.abs(acc["torque_min"] - acc["torque"]).max())
    row["rolled"] = bool(np.any(data["rho"] >= 0.95))
    return row


# --------------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------------

def aggregate(rows, names):
    """ Mean and sd over episodes, per window. Empty windows are left out of their own mean. """
    out = {}
    for window in WINDOWS:
        present = [row[window] for row in rows if row[window] is not None]
        if not present:
            out[window] = None
            continue
        stats = {"n": len(present)}
        for key in ("steps", "cost", "cost_wasted", "cost_share", "act_mean", "act_share", "cci"):
            values = np.array([p[key] for p in present], float)
            stats[key] = (float(np.nanmean(values)), float(np.nanstd(values)))
        per_actuator = np.mean([p["per_actuator_wasted"] for p in present], axis=0)
        total = per_actuator.sum()
        order = np.argsort(per_actuator)[::-1]
        stats["top_actuators"] = [(names[i], float(per_actuator[i]),
                                   float(per_actuator[i] / total) if total > EPS else 0.0)
                                  for i in order[:8]]
        out[window] = stats
    return out


def print_report(summary, pen_factor, pen_metabolic, episodes):
    labels = {"episode": "whole episode (includes the untrained post-roll phase)",
              "until_roll": "until the roll (before rho >= 0.95; what training practised)",
              "roll": "rolling movement (supine -> side lying)",
              "after_roll": "after the roll (rho >= 0.95 reached; never seen in training)"}
    for window in WINDOWS:
        stats = summary[window]
        print(f"--- {labels[window]}")
        if stats is None:
            print("    no episode has this window\n")
            continue
        pm = lambda key, scale=1.0, fmt=".1f": (f"{stats[key][0] * scale:{fmt}} "
                                                 f"+- {stats[key][1] * scale:{fmt}}")
        print(f"    episodes with this window        : {stats['n']}/{episodes}   "
              f"steps {pm('steps', fmt='.0f')}")
        print(f"    metabolic cost (metabolic_cost)  : {pm('cost', fmt='.3f')}   per episode")
        print(f"    ... of which co-contraction       : {pm('cost_wasted', fmt='.3f')}   "
              f"= {pm('cost_share', 100)} % of the metabolic cost")
        if pen_metabolic:
            print(f"    reward paid for co-contraction    : "
                  f"{stats['cost_wasted'][0] * pen_factor:.1f} per episode "
                  f"(pen_factor {pen_factor:g})")
        print(f"    mean activation, all 92 muscles   : {pm('act_mean', fmt='.3f')}")
        print(f"    activation spent on co-contraction: {pm('act_share', 100)} %")
        print(f"    Falconer-Winter CCI               : {pm('cci', 100)} %")
        print("    most wasted cost (share of this window's co-contraction cost):")
        for name, value, share in stats["top_actuators"]:
            print(f"      {name:<24} {value:8.4f}  {share * 100:5.1f} %")
        print()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------------------------
# Selfcheck: the accounting on hand-built pairs. No MuJoCo.
# --------------------------------------------------------------------------------------------

def selfcheck():
    failures, count = [], 0

    def check(condition, message):
        nonlocal count
        count += 1
        if not condition:
            failures.append(message)

    def one(a_neg, a_pos, f_neg=1.0, f_pos=1.0, g_neg=1.0, g_pos=1.0, m=1.0):
        # moment_1 = +m, moment_2 = -m, as MuscleModel builds them.
        return pair_accounting(np.array([a_neg]), np.array([a_pos]),
                               np.array([-m * g_neg * f_neg]), np.array([m * g_pos * f_pos]),
                               np.array([f_neg]), np.array([f_pos]))

    def share(acc, key="cost"):
        return float((acc[key] - acc[key + "_min"]).sum() / acc[key].sum())

    # 1. An unopposed muscle wastes nothing, whatever its strength or operating point.
    for a, f, g in [(0.3, 1.0, 1.0), (1.0, 7.3, 0.4), (0.01, 0.002, 1.2)]:
        acc = one(0.0, a, f_pos=f, g_pos=g)
        check(abs(share(acc)) < 1e-12, f"unopposed muscle a={a} wastes {share(acc)}")
        check(abs(share(acc, "act")) < 1e-12, "unopposed muscle wastes activation")
        check(acc["cci_shared"][0] == 0.0, "unopposed muscle has a non-zero CCI")
    acc = one(0.4, 0.0)
    check(abs(share(acc)) < 1e-12, "unopposed negative muscle wastes cost")

    # 2. Symmetric full cancellation: everything is waste, and all three measures agree.
    acc = one(0.5, 0.5)
    check(abs(share(acc) - 1.0) < 1e-12, f"balanced pair: cost share {share(acc)} != 1")
    check(abs(share(acc, "act") - 1.0) < 1e-12, "balanced pair: activation share != 1")
    check(abs(acc["cci_shared"][0] / acc["act"][0] - 1.0) < 1e-12, "balanced pair: CCI != 1")

    # 3. The worked example of the docstring: equal activation, one muscle twice as strong.
    #    Forces 0.5 and 1.0 -> net 0.5 -> a* = 0.25 on the strong muscle.
    #    cost 0.25 + 0.5 = 0.75, minimum 0.0625 * 2 = 0.125 -> waste 0.625 / 0.75 = 83.3 %.
    acc = one(0.5, 0.5, f_neg=1.0, f_pos=2.0)
    check(abs(acc["act_min"][0] - 0.25) < 1e-12, f"docstring example: a* {acc['act_min'][0]}")
    check(abs(share(acc) - 0.625 / 0.75) < 1e-12, f"docstring example: share {share(acc)}")
    check(abs(acc["cci_shared"][0] / acc["act"][0] - 1.0) < 1e-12,
          "docstring example: CCI should read 100 %")
    # ... and the same with the asymmetry in the operating point instead of the strength.
    acc_g = one(0.5, 0.5, g_neg=1.0, g_pos=2.0)
    check(abs(share(acc_g, "act") - share(acc, "act")) < 1e-12,
          "strength and operating point should enter the activation counterfactual alike")

    # 4. On random pairs: the counterfactual keeps the net torque, never costs more, never asks
    #    the agonist for more than it gave, and its sign follows the torque.
    rng = np.random.default_rng(0)
    N = 20000
    a_neg, a_pos = rng.uniform(0, 1, N), rng.uniform(0, 1, N)
    a_neg[rng.uniform(size=N) < 0.2] = 0.0
    f_neg, f_pos = rng.uniform(0.01, 10, N), rng.uniform(0.01, 10, N)
    g_neg, g_pos = rng.uniform(0, 1.2, N), rng.uniform(0, 1.2, N)
    m = rng.uniform(0.05, 2, N)
    acc = pair_accounting(a_neg, a_pos, -m * g_neg * f_neg, m * g_pos * f_pos, f_neg, f_pos)
    defined = ~acc["undefined"]
    check(np.abs(acc["torque_min"] - acc["torque"])[defined].max() < 1e-9,
          "counterfactual does not reproduce the net torque")
    check(np.all(acc["cost_min"] <= acc["cost"] + 1e-12), "counterfactual costs more")
    check(np.all(acc["act_min"] <= acc["act"] + 1e-12), "counterfactual activates more")
    check(np.all(acc["act_min"] >= 0), "negative counterfactual activation")

    # 5. It is the *cheapest* such activation: no feasible pair with the same net torque on a grid
    #    beats it.
    worst = 0.0
    for i in range(200):
        grid = np.linspace(0, 1, 401)
        c_neg, c_pos = -m[i] * g_neg[i] * f_neg[i], m[i] * g_pos[i] * f_pos[i]
        if abs(c_pos) < 1e-6 or abs(c_neg) < 1e-6:
            continue
        target = acc["torque"][i]
        # For every a_neg on the grid, the a_pos that hits the torque exactly.
        partner = (target - c_neg * grid) / c_pos
        ok = (partner >= 0) & (partner <= 1)
        if not np.any(ok):
            continue
        best = np.min(grid[ok] ** 2 * f_neg[i] + partner[ok] ** 2 * f_pos[i])
        worst = max(worst, acc["cost_min"][i] - best)
    check(worst < 1e-9, f"a cheaper activation with the same torque exists (by {worst:.2e})")

    # 6. Invariances. Shares do not depend on the unit of fmax or of the moment arm.
    base = share(pair_accounting(a_neg, a_pos, -m * g_neg * f_neg, m * g_pos * f_pos, f_neg, f_pos))
    scaled = share(pair_accounting(a_neg, a_pos, -3 * m * g_neg * 7 * f_neg,
                                   3 * m * g_pos * 7 * f_pos, 7 * f_neg, 7 * f_pos))
    check(abs(base - scaled) < 1e-12, "cost share changes with the unit of fmax / moment")
    swapped = share(pair_accounting(a_pos, a_neg, -m * g_pos * f_pos, m * g_neg * f_neg,
                                    f_pos, f_neg))
    check(abs(base - swapped) < 1e-12, "cost share depends on which muscle is called negative")

    # 7. An agonist with no force capacity has no defined minimum: only the antagonist is
    #    removed, and the case is flagged.
    acc = one(0.6, 0.4, g_neg=0.0, g_pos=0.0)
    check(bool(acc["undefined"][0]), "zero-capacity pair not flagged")
    check(abs(acc["act_min"][0] - 0.6) < 1e-12, "zero-capacity pair: agonist should be kept")
    acc = one(0.6, 0.0, g_neg=0.0)
    check(not bool(acc["undefined"][0]), "an unopposed zero-capacity muscle is not co-contraction")

    # 8. Window summary: the normaliser is MuscleModel.cost's, and shares pool over steps.
    n = 3
    fmax = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 0.5])
    a = rng.uniform(0, 1, (10, 2 * n))
    cap = np.concatenate([-np.ones((10, n)), np.ones((10, n))], axis=1) * fmax
    acc = pair_accounting(a[:, :n], a[:, n:], cap[:, :n], cap[:, n:], fmax[:n], fmax[n:])
    summary = window_summary(acc, np.ones(10, bool), 2 * n * fmax.sum())
    model_cost = sum((a[t] ** 2 * fmax).sum() / (2 * n * fmax.sum()) for t in range(10))
    check(abs(summary["cost"] - model_cost) < 1e-12, "episode cost does not match MuscleModel.cost")
    check(abs(summary["per_actuator_wasted"].sum() - summary["cost_wasted"]) < 1e-12,
          "per-actuator waste does not add up to the total")
    check(window_summary(acc, np.zeros(10, bool), 1.0) is None, "empty window not None")
    # The path the evaluation takes, against MuscleModel.cost written out independently.
    row = analyse_episode({"a": a, "cap": cap, "rho": np.zeros(10)}, fmax, n)
    check(abs(row["episode"]["cost"] - model_cost) < 1e-12,
          f"analyse_episode cost {row['episode']['cost']} != MuscleModel.cost {model_cost}")

    # 9. Windows.
    rho = np.array([0.0, 0.01, 0.03, 0.02, 0.2, 0.4, 0.6, 0.9, 0.97, 1.0, 0.99])
    masks = episode_windows(rho)
    check(np.flatnonzero(masks["roll"]).tolist() == [3, 4, 5, 6],
          f"roll window {np.flatnonzero(masks['roll']).tolist()}")
    check(np.flatnonzero(masks["after_roll"]).tolist() == [8, 9, 10], "after-roll window")
    check(masks["episode"].all(), "episode window")
    check(np.flatnonzero(masks["until_roll"]).tolist() == list(range(8)),
          f"until-roll window {np.flatnonzero(masks['until_roll']).tolist()}")
    masks = episode_windows(np.linspace(0, 0.4, 20))
    check(not masks["roll"].any() and not masks["after_roll"].any(), "no roll, no windows")
    check(masks["until_roll"].all(), "no roll: until-roll should be the whole episode")

    for message in failures:
        print(f"FAIL: {message}")
    print(f"eval_cocontraction selfcheck: {count - len(failures)}/{count} assertions passed")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Path to a saved model_*.zip.")
    parser.add_argument("--group", default=None,
                        help="Evaluate a whole sweep instead, pooled over seeds: a save-path "
                             "prefix, a directory of '_run_<i>' dirs or a glob, as in "
                             "eval_rollover.py. Uses the last checkpoint of each run.")
    parser.add_argument("--checkpoint", default="last",
                        help="With --group: 'last' (default), 'best', or a file name.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run the assertions on synthetic pairs and exit. No MuJoCo.")
    parser.add_argument("--episodes", type=int, default=40,
                        help="Episodes per model. Default 40, the thesis standard.")
    parser.add_argument("--seed", type=int, default=1000,
                        help="First seed; episode i uses seed + i, as eval_rollover.py does.")
    parser.add_argument("--starting_position", default=None, choices=["prone", "supine"])
    parser.add_argument("--episode_steps", type=int, default=None)
    parser.add_argument("--json", default=None, help="Write the full numbers here.")
    parser.set_defaults(goal=None, missing_limb=None, ghost_obs=None, missing_limb_mode=None,
                        floor_softness=None, floor_friction=None, floor_solimp_width=None,
                        rigid_floor=False, physio_age=None, morph_age=None)
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(selfcheck())
    if bool(args.model) == bool(args.group):
        parser.error("give exactly one of --model or --group (or use --selfcheck)")

    import mimoEnv  # noqa: F401  (registers MIMoRollOver-v0)
    from mimoEnv.eval_rollover import (resolve_run, load_policy, discover_runs,
                                       pick_checkpoint, _EnvCache)

    if args.group:
        models = [pick_checkpoint(run, args.checkpoint) for run in discover_runs(args.group)]
        models = [m for m in models if m]
        if not models:
            raise SystemExit(f"No checkpoints found for --group={args.group}")
    else:
        models = [args.model]

    cache = _EnvCache()
    rows, per_model = [], []
    errors = {"force": 0.0, "torque": 0.0, "cost": 0.0, "episode_cost": 0.0, "net": 0.0}
    pen_factor = pen_metabolic = None
    names = None
    for model_path in models:
        config, start, goal, episode_steps = resolve_run(model_path, args)
        if not config.get("use_muscle", False):
            raise SystemExit(f"{model_path} was not trained with --use_muscle. Under the "
                             f"spring-damper model a joint has one bidirectional motor and "
                             f"cannot co-contract.")
        env = cache.get(config, start, goal)
        policy = load_policy(model_path, config.get("algorithm", "PPO"), env)
        am = env.actuation_model
        n = am.n_actuators
        if not (np.all(am.moment_1 > 0) and np.all(am.moment_2 < 0)):
            raise SystemExit("MuscleModel's muscle pairs are not antagonists (moment signs).")
        names = [env.model.actuator(i).name for i in env.mimo_actuators]
        pen_factor = config.get("pen_factor")
        pen_metabolic = bool(config.get("pen_metabolic", False)) and not config.get("nopen", False)
        print(f"model   : {model_path}")
        print(f"posture : {start}   episodes {args.episodes} (steps {episode_steps}, seeds "
              f"{args.seed}..{args.seed + args.episodes - 1})")

        model_rows = []
        for episode in range(args.episodes):
            data = collect_episode(env, policy, args.seed + episode, episode_steps)
            errors["force"] = max(errors["force"], data["err_force"])
            errors["torque"] = max(errors["torque"], data["err_torque"])
            errors["cost"] = max(errors["cost"], data["err_cost"])
            row = analyse_episode(data, am.fmax.copy(), n)
            errors["net"] = max(errors["net"], row["err_net_torque"])
            # The printed totals, not just the per-step helper, must be the env's own cost.
            env_total = float(data["env_cost"].sum())
            errors["episode_cost"] = max(errors["episode_cost"],
                                         abs(row["episode"]["cost"] - env_total)
                                         / max(env_total, EPS))
            row.update(model=model_path, seed=args.seed + episode)
            model_rows.append(row)
        rolled = sum(r["rolled"] for r in model_rows)
        share = np.mean([r["episode"]["cost_share"] for r in model_rows])
        print(f"          rolled {rolled}/{args.episodes}, co-contraction "
              f"{share * 100:.1f} % of metabolic cost\n")
        per_model.append({"model": model_path, "rolled": rolled,
                          "summary": aggregate(model_rows, names)})
        rows.extend(model_rows)
    cache.close()

    print("consistency, max over all steps:")
    print(f"  muscle force rebuilt from fl*fv*a + fp   : {errors['force']:.1e}")
    print(f"  joint torque rebuilt (relative)          : {errors['torque']:.1e}")
    print(f"  metabolic cost vs info['metabolic_cost'] : {errors['cost']:.1e} (relative)")
    print(f"  episode total vs sum of info             : {errors['episode_cost']:.1e} (relative)")
    print(f"  net torque kept by the counterfactual    : {errors['net']:.1e}")
    undefined = np.mean([r["undefined_share"] for r in rows])
    print(f"  pair-steps with no agonist capacity      : {undefined * 100:.3f} % "
          f"(waste undercounted there)\n")
    if max(errors.values()) > 1e-6:
        raise SystemExit("Consistency check failed: the accounting does not reproduce the "
                         "muscle model. Do not trust the numbers below.")

    summary = aggregate(rows, names)
    scope = f"{len(models)} model(s) x {args.episodes} episodes" if args.group else \
        f"{args.episodes} episodes"
    print(f"===== co-contraction, mean +- sd over {scope}\n")
    print_report(summary, pen_factor, pen_metabolic, len(rows))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(plain({"models": models, "episodes": args.episodes, "seed": args.seed,
                             "actuators": names, "errors": errors,
                             "undefined_share": undefined, "summary": summary,
                             "per_model": per_model,
                             "rows": [{k: v for k, v in r.items()} for r in rows]}),
                      handle, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

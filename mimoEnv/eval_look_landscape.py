""" The looking reward as a function of posture: does the reward landscape point at prone?

01.09.2026 This is the measurement that has to be made *before* training anything on the looking
reward, because the reward can be perfectly well implemented and still select for the wrong thing.
The failure mode it exists to catch: from a side-lying posture MIMo's eye looks along the floor and
takes in the whole ring of toys, whereas prone his face is 1 cm from the ground. A looking reward
can therefore have its optimum at side-lying and pay *nothing* for finishing the roll, in which
case a run would plateau at rho ~ 0.5 and the plateau would be the reward's fault, not the
policy's.

It sweeps MIMo through the roll from supine (0 degrees, rho 0) to prone (180 degrees, rho 1) by
writing the root free joint's quaternion directly, drops him onto the floor at each angle, and
reports the foveal toy share in two ways:

- **neutral**: head joints at zero. This is what the reward looks like early in training, when the
  policy has not learned to move the head yet.
- **cover**: how many *different* toys MIMo could collect from this posture by sweeping his head
  over the comfortable range -- the quantity the coverage reward optimises. Measured for the
  shipped layout: 3/10 supine, 5/10 side-lying, and 10/10 only from roll 160 degrees onward.
- **best**: the maximum over a grid of comfortable 'head_tilt' and 'head_swivel' values (70 % of
  each joint's range by default -- the limits themselves are a contortion, and a landscape
  measured there says more about the neck than about the posture). This is the ceiling a competent
  policy could reach from that posture.

Measured for the shipped layout (ring at 0.65 m, foveal weighting on): 0.027 supine against 0.231
prone at comfortable head poses, a factor of 8.5, rising monotonically past side-lying. With the
foveal weighting switched off ('--fovea=0') the same layout gives 0.18 against 0.31, because MIMo
can then earn the reward by cranking his neck to its limits from any posture at all -- which is
the measurement that put the weighting in.

Note the neutral column peaks near 120 degrees and falls back to zero at 180. That is not a bug:
prone with a slack neck is face-down. Getting the reward from prone requires lifting the head,
which is exactly the motor milestone that precedes rolling in infants.

Uses the roll-over env's own conventions throughout -- in particular 'mju_euler2Quat(..., "xyz")',
which is *intrinsic*. scipy's lowercase "xyz" is extrinsic and composes the roll the other way
round, which leaves the body upright at every angle and reports a flat zero landscape.

    MUJOCO_GL=osmesa python mimoEnv/eval_look_landscape.py
    MUJOCO_GL=osmesa python mimoEnv/eval_look_landscape.py --fovea=0 --step=5 --csv=out.csv
"""

import argparse
import csv
import os

import numpy as np
import mujoco
import gymnasium as gym

import mimoEnv  # noqa: F401  -- registers MIMoRollOver-v0
from mimoEnv.envs.roll_over_look import LookReward
from mimoEnv.utils import get_minimal_z_coordinate


def landscape(env, look, rolls, head_fraction=0.7, head_grid=(5, 7)):
    """ Foveal toy share at each roll angle, with a neutral head and with the best head pose.

    Args:
        env: An unwrapped :class:`~mimoEnv.envs.roll_over.MIMoRollOverEnv` on a playroom scene.
        look (LookReward): The measurement, normally built with ``weight=1.0`` so the numbers come
            back as raw foveal shares.
        rolls (Sequence[float]): Roll angles in degrees. 0 is supine, 180 prone.
        head_fraction (float): Fraction of each head joint's range to search over.
        head_grid (tuple): Number of 'head_tilt' and 'head_swivel' samples.

    Returns:
        list: One dict per roll angle.
    """
    model = env.model
    base_qpos = env.data.qpos.copy()
    tilt = model.joint("robot:head_tilt")
    swivel = model.joint("robot:head_swivel")
    tilt_adr = model.jnt_qposadr[tilt.id]
    swivel_adr = model.jnt_qposadr[swivel.id]
    tilts = np.linspace(*(model.jnt_range[tilt.id] * head_fraction), head_grid[0])
    swivels = np.linspace(*(model.jnt_range[swivel.id] * head_fraction), head_grid[1])

    def place(roll_deg, tilt_rad, swivel_rad):
        env.data.qpos[:] = base_qpos
        quat = np.zeros(4)
        # Intrinsic 'xyz', the same convention as 'MIMoRollOverEnv.get_starting_quat': the roll
        # about x, then the -90 degrees about y that lays MIMo on his back.
        mujoco.mju_euler2Quat(quat, np.array([np.radians(roll_deg), -np.pi / 2, 0.0]), 'xyz')
        env.data.qpos[3:7] = quat
        env.data.qpos[tilt_adr] = tilt_rad
        env.data.qpos[swivel_adr] = swivel_rad
        env.data.qpos[2] = 0.5
        mujoco.mj_forward(model, env.data)
        # Same drop-onto-the-floor rule the env uses at reset.
        env.data.qpos[2] += get_minimal_z_coordinate(model, env.data) + 0.001
        mujoco.mj_forward(model, env.data)

    rows = []
    for roll in rolls:
        place(roll, 0.0, 0.0)
        rho = float(env.get_achieved_goal_cos()[0])
        neutral = float(look.visible_fractions().sum())
        best, best_pose, best_visible = -1.0, (0.0, 0.0), None
        # Union over the head poses: how many DIFFERENT toys MIMo could collect from this posture
        # if he swept his head over the comfortable range. This is the quantity the coverage
        # reward optimises, and the one the playroom is arranged around.
        reachable = np.zeros(len(look.toy_names), dtype=bool)
        for t in tilts:
            for s in swivels:
                place(roll, t, s)
                visible = look.visible_fractions()
                reachable |= visible >= look.seen_threshold
                total = float(visible.sum())
                if total > best:
                    best, best_pose, best_visible = total, (t, s), visible
        rows.append({
            "roll_deg": float(roll),
            "rho": rho,
            "neutral": neutral,
            "best": best,
            "best_tilt_deg": float(np.degrees(best_pose[0])),
            "best_swivel_deg": float(np.degrees(best_pose[1])),
            "n_toys": int(np.count_nonzero(best_visible > 0.001)),
            "coverage": int(reachable.sum()),
            "n_total": len(look.toy_names),
            "toys": ",".join(n for n, v in zip(look.toy_names, best_visible) if v > 0.001),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=10, help="Roll angle step in degrees.")
    parser.add_argument("--fovea", type=float, default=0.35,
                        help="Foveal weighting width; 0 weights every pixel equally.")
    parser.add_argument("--eyes", default="left", help="'left', 'right' or 'both'.")
    parser.add_argument("--resolution", type=int, default=32,
                        help="Segmentation render resolution.")
    parser.add_argument("--head_fraction", type=float, default=0.7,
                        help="Fraction of each head joint's range to search over.")
    parser.add_argument("--morph_age", type=int, default=9)
    parser.add_argument("--physio_age", type=int, default=9)
    parser.add_argument("--csv", default=None, help="Write the table here as well.")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from mimoEnv.envs.roll_over import EYES

    env = gym.make("MIMoRollOver-v0", starting_position="supine", playroom=True,
                   age_morph=args.morph_age, age_physio=args.physio_age, isr=False).unwrapped
    env.reset(seed=0)
    look = LookReward(env, cameras=EYES[args.eyes], weight=1.0,
                      fovea=(None if args.fovea == 0 else args.fovea),
                      resolution=args.resolution)

    rolls = list(range(0, 181, args.step))
    rows = landscape(env, look, rolls, head_fraction=args.head_fraction)

    print(f"\nPlayroom look landscape -- fovea={args.fovea}, eyes={args.eyes}, "
          f"head range {args.head_fraction:.0%}, ages {args.morph_age}/{args.physio_age}")
    print(f"{'roll':>5s} {'rho':>6s} {'neutral':>8s} {'best':>8s}  "
          f"{'tilt':>5s} {'swiv':>5s} {'cover':>6s}  best view")
    for r in rows:
        print(f"{r['roll_deg']:5.0f} {r['rho']:6.3f} {r['neutral']:8.4f} {r['best']:8.4f}  "
              f"{r['best_tilt_deg']:+5.0f} {r['best_swivel_deg']:+5.0f} "
              f"{r['coverage']:3d}/{r['n_total']:<2d}  {r['toys'].replace('toy_', '')}")

    supine, prone = rows[0]["best"], rows[-1]["best"]
    peak = max(rows, key=lambda r: r["best"])
    print(f"\nsupine {supine:.4f} -> prone {prone:.4f}  "
          f"(factor {prone / supine:.1f})" if supine > 0 else
          f"\nsupine {supine:.4f} -> prone {prone:.4f}")
    print(f"best-head optimum at roll {peak['roll_deg']:.0f} deg (rho {peak['rho']:.3f})")
    if peak["roll_deg"] < 150:
        print("NOTE: the per-step foveal share peaks before prone -- side-lying puts the eye along "
              "the floor and takes in the most at once. That term alone would park a run there; it "
              "is the coverage bonus ('--look_novelty_w', 200 per toy against a continuous term of "
              "order 0.2 per step) that pays for finishing the roll. Worth checking that the two "
              "are still balanced if you have changed either weight.")

    full = [r for r in rows if r["coverage"] == r["n_total"]]
    if not full:
        print(f"WARNING: no roll angle reaches all {rows[0]['n_total']} toys at comfortable head "
              f"poses (best {max(r['coverage'] for r in rows)}). Some toy can never be collected, "
              f"so the coverage reward has an unreachable ceiling -- check the per-toy foveal "
              f"shares before training.")
    else:
        first = min(r["roll_deg"] for r in full)
        print(f"full coverage ({rows[0]['n_total']} toys) first reachable at roll {first:.0f} deg "
              f"(rho {min(r['rho'] for r in full if r['roll_deg'] == first):.3f}); "
              f"supine reaches {rows[0]['coverage']}/{rows[0]['n_total']}")
        if first < 90:
            print("WARNING: full coverage is already reachable from near-supine. The coverage "
                  "reward then does not require the roll.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")

    look.close()
    env.close()


if __name__ == "__main__":
    main()

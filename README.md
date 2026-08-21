# MIMo — roll-over fork

Upstream [MIMo](https://mimo.readthedocs.io) is a Gymnasium + MuJoCo platform for a multimodal
infant model (vision, touch, proprioception, vestibular). **This checkout is a research fork.**
The default branch `v2/main` is upstream MIMo-v2; the working branch **`roll_over`** adds a
roll-over learning experiment: MIMo starts prone or supine and has to roll onto his other side,
with MIMo's morphological (body) age and physiological (actuation) age as independent variables,
and — in the later part of the work — a sparse-reward + HER variant of the same task.

Almost all fork-specific work lives in `mimoEnv/envs/roll_over.py`, `mimoEnv/illustrations.py`
and `mimoEnv/eval_rollover.py`. **The manual for that stack is
[`docs/roll_over.md`](docs/roll_over.md)** — read it before changing anything in the reward,
goal or evaluation path. This file only gets you running.

## Install

```bash
conda activate mimo          # the reference environment, Python 3.12; the cluster scripts assume this name
pip install -r requirements.txt
pip install -e .
```

Versions in `requirements.txt` are pinned deliberately, `gymnasium==1.0.0` in particular:
`MIMoEnv._initialize_simulation` must assign `self.model`/`self.data` itself *and* return them,
which is version-dependent behaviour. Changing that pin has broken the fork repeatedly.

Verify the install with the regression gate, not with a training run:

```bash
MUJOCO_GL=osmesa python mimoEnv/goalenv_check.py     # a couple of minutes, PASS/FAIL per check
```

`mimoEnv/showroom.py` opens an interactive viewer (registered with `render_mode="human"`, so it
needs a display) and is a quick visual sanity check of the model, nothing more.

**There is no test suite.** Files named `*test*` are experiments, not tests
(`mimoActuation/muscle_testing.py`, `mimoEnv/isr_test.py`, `mimoEnv/envs/muscle_test.py`).
`goalenv_check.py` is the one real check — run it after touching `compute_reward`, `is_success`,
`_potential`, `sample_goal` or anything on the HER path.

## Train

`mimoEnv/illustrations.py` is the single train/eval CLI for all demo environments and defaults to
`--env=roll_over`. There are no per-environment training scripts in this fork.

The shaped-reward baseline (PPO + potential-based reward shaping):

```bash
MUJOCO_GL=osmesa python mimoEnv/illustrations.py \
    --train_for=1000000 --save_every=200000 \
    --algorithm=PPO --pbrs --pbrs_w=100 \
    --roll_over_starting_position=prone \
    --goal_achievement_function=cos --pen_factor=0.02 \
    --morph_age=9 --physio_age=9 \
    --roll_over_model_path_auto --save_model=my_run
```

The sparse-reward + HER configuration (SAC) is wrapped in a script, because it needs several
flags that only make sense together:

```bash
./run_her_sparse.sh                  # or: ./run_her_sparse.sh <name> <n_seeds>
```

The **intrinsic-goal** variant replaces the hand-designed rotation target with one MIMo can
actually sense — six joint angles plus the vestibular accelerometer, instead of the root free
joint that proprioception does not report:

```bash
MUJOCO_GL=osmesa python mimoEnv/illustrations.py \
    --algorithm=SAC --goal_achievement_function=intrinsic \
    --her --sparse_reward --no_done_active \
    --roll_over_starting_position=prone --roll_over_model_path_auto \
    --train_for=1000000 --save_every=200000 --save_model=intrinsic_her
```

`--intrinsic_goal_eps` (the success radius) has **not** been calibrated yet; read
[`docs/roll_over.md` §3.4](docs/roll_over.md#34-the-intrinsic-goal-function--a-non-scalar-non-extrinsic-goal)
before trusting a number from it.

Models land in
`models/roll_over/<yy-mm-dd>/<prone|supine>/<yy-mm-dd>_<prone|supine>_<save_model>/`, next to a
TensorBoard directory (`PPO_0/`, `SAC_0/`, …) and `data.yml`, which records the run's
hyperparameters and is read back by `--load_model`.

## Evaluate

Use `eval_rollover.py` for any number that goes into a write-up. `illustrations.py --test` runs a
single episode and is a rendering tool, not a measurement.

```bash
MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
    --model=models/roll_over/<yy-mm-dd>/supine/<run_dir>/model_5.zip \
    --starting_position=supine --episodes=50
```

It reads the run's `data.yml`, then forces the fixed protocol (ISR off, goal pinned to 0.95,
no early termination, deterministic actions, success = per-episode maximum rotation ≥ 0.95).
Each rule exists because of a specific measurement — see
[`docs/roll_over.md`](docs/roll_over.md#6-evaluation-protocol).

To render instead of measure:

```bash
MUJOCO_GL=osmesa python mimoEnv/illustrations.py --test --render_video --render_frames \
    --load_model=models/roll_over/<yy-mm-dd>/<prone|supine>/<run_dir>/model_1.zip
```

## Things that will catch you in the first hour

- **`MUJOCO_GL=osmesa`, not `egl`.** `egl` fails in the `mimo` conda env. `eval_rollover.py` and
  `goalenv_check.py` set it themselves via `os.environ.setdefault`; `illustrations.py` does not.
  (`mimoComposer/README.md` claims the opposite; it is wrong.)
- **One MIMo env costs ~3.6 GB RSS** — that is the MuJoCo model, not the renderer and not the
  replay buffer (measured identical with and without `render_mode`, and unchanged between a 100k
  and a 200k buffer). On a 16 GB machine runs must be **sequential**; four in parallel were
  OOM-killed. This is why `run_her_sparse.sh` loops instead of backgrounding.
- **The last checkpoint is not reliably the best one.** One HER run collapsed after 600k steps
  (critic loss 1 → 901) and its final model scored 2 % while its 600k checkpoint was far better.
  Do not set `--save_every` equal to `--train_for` for off-policy runs; use `--save_every=200000`
  and evaluate every checkpoint.
- **Always pass `--starting_position` to `eval_rollover.py`.** The starting posture is
  deliberately *not* stored in `data.yml`, so the script's fallback silently evaluates every run
  as `supine`. A prone run scored against the supine reset is meaningless.
- **`rollout/success_rate` in TensorBoard is not the roll rate.** It scores against the *sampled*
  goal, which is as low as 0.25 under `--goal_low`. Read `rollout/ep_rho_max_mean` instead.

## Repository layout

| Path | What it is |
|---|---|
| `mimoEnv/envs/roll_over.py` | The roll-over environment. The centre of this fork. |
| `mimoEnv/envs/roll_over_callback.py`, `isr_callback.py`, `morphological_curriculum.py` | Training callbacks: logging, ISR shutdown, embodiment swapping. |
| `mimoEnv/illustrations.py` | The train/eval CLI (58 flags; reference table in the manual). |
| `mimoEnv/eval_rollover.py` | The fixed evaluation protocol. |
| `mimoEnv/goalenv_check.py` | Regression gate for the goal/reward/HER path. |
| `mimoEnv/assets/roll_over/prone/` | 16 pre-generated age scenes, `scene_act_<physio>_body_<morph>.xml`. |
| `mimoEnv/envs/mimo_env.py` | Upstream base class: Dict observation space, sensor modules, action space. |
| `mimoProprioception/`, `mimoTouch/`, `mimoVision/`, `mimoVestibular/` | Upstream sensor modules. |
| `mimoActuation/` | Upstream actuation models (spring-damper, muscle, frozen-limb variant). |
| `mimoGrowth/` | Upstream age rescaling. **Bypassed** by roll-over — see the manual. |
| `results/` | Standalone analysis scripts (`python results/<script>.py --date=… --suffix=…`). |
| `rbi_*.sh` | Cluster launch scripts; the record of which sweeps were actually run. |
| `docs/source/` | The upstream Sphinx API docs (`cd docs && make html`). |
| `docs/roll_over.md` | This fork's manual. Standalone Markdown, not part of the Sphinx build. |

`.gitignore` excludes `models*`, `*.csv`, `*.png`, `*.pdf`, `*.npy`, `png/`, `csv/`, `pdf/` —
trained models and plots are local artifacts. It also excludes `CLAUDE.md`, `mimoComposer/` and
`results/composer/`, so those exist only in a working copy and not in the repository.

## License and citation

MIT — see [LICENSE](LICENSE). If you use MIMo, cite the upstream papers:

```
@article{mattern2024mimo,
  title={MIMo: A Multimodal Infant Model for Studying Cognitive Development},
  author={Mattern, Dominik and Schumacher, Pierre and L{\'o}pez, Francisco M and Raabe, Marcel C
          and Ernst, Markus R and Aubret, Arthur and Triesch, Jochen},
  journal={IEEE Transactions on Cognitive and Developmental Systems},
  volume={16}, number={4}, pages={1291--1301}, year={2024}, publisher={IEEE}
}
```

For body growth, developing visual acuity or sensorimotor delays (MIMo-v2), also cite:

```
@inproceedings{lopez2025mimo,
  title={MIMo Grows! Simulating Body and Sensory Development in a Multimodal Infant Model},
  author={L{\'o}pez, Francisco M and Lenz, Miles and Fedozzi, Marco G and Aubret, Arthur
          and Triesch, Jochen},
  booktitle={2025 IEEE International Conference on Development and Learning (ICDL)},
  pages={1--6}, year={2025}, organization={IEEE}
}
```

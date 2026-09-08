#!/bin/bash
# Train roll-over with the MUSCLE actuation model, PPO + PBRS.
#
# The point of this configuration is that it is the documented spring-damper roll-over recipe
# (PPO, --pbrs --pbrs_w=100 --pen_factor=0.02, supine, age 9) with `--use_muscle` added and
# nothing else changed, so the muscle runs can be compared against the PPO+PBRS supine baselines
# already on disk (e.g. models/roll_over/26-08-12/supine/26-08-12_supine_tst_ctr_cost_logger_run_0).
#
# Supine, not prone: Siegel et al. (2024) placed every infant supine and analysed only the
# supine-to-lateral-rotation portion of a supine-to-prone roll, so a prone run has no counterpart
# in the data mimoEnv/eval_emg.py compares against.
#
#   ./run_muscle_pbrs.sh                  # one run, default name
#   ./run_muscle_pbrs.sh myname           # one run, custom name suffix
#   ./run_muscle_pbrs.sh myname 5         # five sequential runs (seeds), suffixed _s0.._s4
#   ./run_muscle_pbrs.sh pilot 1 500000   # a feasibility pilot -- see the budget note below
#
# Do NOT judge feasibility before ~500k steps. The spring-damper ep100 baseline
# (26-08-19_supine_ep100_run_0) sat at rho_max 0.273 with zero side-lying successes at 200k, and
# only reached 0.910 / 99 % at 500k and 0.957 / 100 % at 1M. A muscle run stopped at 200k tells
# you almost nothing.
#
# Needs `conda activate mimo` first.
#
# 02.09.2026 NOTE: no stored run has ever used the muscle model, so whether MIMo can learn to roll
# at all with 92 muscles and the softened unactuated spine is genuinely unknown. Run the short
# pilot first and look at rollout/ep_rho_max_mean before committing to a multi-seed batch.
set -u

NAME=${1:-muscle_pbrs}
RUNS=${2:-1}
STEPS=${3:-1000000}
# 04.09.2026 Match the ep100 spring-damper baseline, not the registered 500-step default. Two
# reasons, both learned from the first pilot, which ran at 500:
#   * Comparability. raw_ctrl_cost is a per-episode SUM, so a 500-step run reports 5x the number
#     for identical behaviour.
#   * The reward budget. The effort term costs ~0.48/step whatever the horizon, so 500 steps spend
#     ~238 against a total PBRS payoff for a complete roll of pbrs_w * 0.95 = 95. Before MIMo finds
#     the roll the only reliable gradient is then "activate less", and the pilot did exactly that:
#     raw_ctrl_cost fell 11895 -> 7178 while rho_max crawled 0.010 -> 0.148. At 100 steps the
#     effort is ~48 against the same 95, which is the ratio the baseline learned under.
EPISODE_STEPS=${4:-100}

# osmesa, not egl: egl fails in this conda env.
export MUJOCO_GL=osmesa

# One MIMo env costs ~3.6 GB RSS. On a 16 GB machine only ONE run fits comfortably, hence the
# sequential loop rather than backgrounding them all.
for i in $(seq 0 $((RUNS - 1))); do
  suffix=$NAME
  [ "$RUNS" -gt 1 ] && suffix="${NAME}_s${i}"

  echo "=== $(date +%F\ %H:%M) START $suffix ==="
  python mimoEnv/illustrations.py \
    --train_for="$STEPS" \
    --save_every=200000 \
    --episode_steps="$EPISODE_STEPS" \
    --algorithm=PPO \
    --use_muscle \
    --pbrs --pbrs_w=100 \
    --pen_factor=0.02 \
    --roll_over_starting_position=supine \
    --morph_age=9 --physio_age=9 \
    --roll_over_model_path_auto \
    --save_model="$suffix"
  echo "=== $(date +%F\ %H:%M) DONE $suffix exit=$? ==="
done

cat <<'MSG'

Done. First check that it rolls at all, under the reportable protocol:

  MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
      --model=models/roll_over/<yy-mm-dd>/supine/<dir>/model_5.zip --episodes=50

Evaluate every checkpoint, not just the last one. Then the EMG:

  MUJOCO_GL=osmesa python mimoEnv/eval_emg.py \
      --model=models/roll_over/<yy-mm-dd>/supine/<dir>/model_5.zip --episodes=50 \
      --json emg.json --plot emg_fig5.png

and the sensitivity check on the one modelling choice that is not forced by the anatomy:

  ... --biarticular_weight=0    # QUAD and HAM reduced to their knee terms alone

In the first PPO log block, check that ep_rew_mean is consistent with its parts: raw_ctrl_cost
times --pen_factor is the effort term the reward actually pays. Under the muscle model that
identity was broken until 02.09.2026 -- data.ctrl is a constant 1, so the penalty was a constant
and --pen_factor did nothing. See MIMoRollOverEnv.penalized_control.
MSG

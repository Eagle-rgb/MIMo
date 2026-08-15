#!/bin/bash
# Train roll-over with a SPARSE reward + HER (the "E3b" configuration).
#
# This is the run that reaches 100% full roll without any hand-designed rotation shaping,
# beating the PBRS baseline (94%). See ~/MyVault/Projects/MIMo-intrinsic/ for the write-up.
#
#   ./run_her_sparse.sh              # one run, default name
#   ./run_her_sparse.sh myname       # one run, custom name suffix
#   ./run_her_sparse.sh myname 3     # three sequential runs (seeds), suffixed _s0.._s2
#
# Needs `conda activate mimo` first. Takes ~4.5 h per run.
set -u

NAME=${1:-her_sparse}
RUNS=${2:-1}

# osmesa, not egl: egl fails in this conda env despite what CLAUDE.md says.
export MUJOCO_GL=osmesa

# One MIMo env costs ~3.6 GB RSS (the MuJoCo model itself -- not the renderer, not the replay
# buffer). On a 16 GB machine only ONE run fits comfortably; four in parallel were OOM-killed.
# Hence the sequential loop rather than backgrounding them all.
for i in $(seq 0 $((RUNS - 1))); do
  suffix=$NAME
  [ "$RUNS" -gt 1 ] && suffix="${NAME}_s${i}"

  echo "=== $(date +%F\ %H:%M) START $suffix ==="
  python mimoEnv/illustrations.py \
    --train_for=1000000 \
    --save_every=200000 \
    --algorithm=SAC \
    --her \
    --sparse_reward \
    --no_done_active \
    --goal_low=0.25 --goal_high=0.95 \
    --learning_starts=1000 \
    --buffer_size=200000 \
    --roll_over_starting_position=supine \
    --goal_achievement_function=cos \
    --pen_factor=0.02 \
    --morph_age=9 --physio_age=9 \
    --lr=0.0003 \
    --roll_over_model_path_auto \
    --save_model="$suffix"
  echo "=== $(date +%F\ %H:%M) DONE $suffix exit=$? ==="
done

cat <<'MSG'

Done. Evaluate a checkpoint with:

  MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
      --model=models/roll_over/<yy-mm-dd>/supine/<dir>/model_5.zip --episodes=50

model_5.zip is the final one (5 x 200k). Evaluate the earlier checkpoints too: one run
collapsed after 600k (critic_loss 1 -> 901) and its final model scored 2% while its 600k
checkpoint was far better, so the last model is not automatically the best.
MSG

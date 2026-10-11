#!/bin/bash
# 11.10.2026 Does start noise improve training and make the evaluation more stable?
#
# The 26-08-19_supine_ep100 PPO+PBRS baseline (default noise, +-0.573 deg) retrained with
# --init_noise, nothing else changed. 10 deg is the largest value at which a limp MIMo still
# stays put in both postures (rho_max <= 0.04 over 30 episodes; at 15 deg one supine episode
# reaches 0.26 with zero action), so anything above it hands out rotation for free.
#
# Sequential on purpose: runs in parallel were OOM-killed on a 16 GB machine.
#
#   ./run_init_noise.sh                 # 5 and 10 deg, 3 seeds each
#   ./run_init_noise.sh "10" 6          # one level, 6 seeds
levels=${1:-"5 10"}
n_seeds=${2:-3}

for noise in $levels; do
  for seed in $(seq 0 $((n_seeds - 1))); do
    suffix="ep100_initnoise${noise}_run_${seed}"
    echo "=== $(date +%F\ %H:%M) START $suffix ==="
    python mimoEnv/illustrations.py --train_for=1000000 --save_every=1000000 \
      --algorithm=PPO --pbrs --pbrs_w=100 \
      --pen_factor=0.02 \
      --episode_steps=100 \
      --init_noise="$noise" \
      --roll_over_starting_position=supine \
      --morph_age=9 --physio_age=9 \
      --roll_over_model_path_auto \
      --save_model="$suffix"
    echo "=== $(date +%F\ %H:%M) DONE $suffix exit=$? ==="
  done
done

cat <<'MSG'

Done. Score each group at BOTH noise levels -- the one it trained with and the baseline's:

  MUJOCO_GL=osmesa python mimoEnv/eval_rollover.py \
      --group=models/roll_over/<yy-mm-dd>/supine/<yy-mm-dd>_supine_ep100_initnoise10
  ... --init_noise=0.573      # the default start noise, for the comparison with 26-08-19

and the baseline the other way round:

  ... --group=models/roll_over/26-08-19/supine/26-08-19_supine_ep100 --init_noise=10
MSG

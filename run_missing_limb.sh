#!/bin/bash
# Roll-over with a missing limb.
#
# Two modes, and which one you want depends on whether a policy has to survive the change:
#
#   cut     The limb is really gone. Joints, actuators and sensors go with it, so the spaces
#           shrink (46 -> 38 actuators, 305 -> 253 proprioception values). This is the mode
#           for TRAINING a policy that never had the limb. It cannot load an intact policy.
#
#   ghost   Only the physics is removed -- mass, inertia, collisions -- while joints,
#           actuators and sensors stay, so the spaces are unchanged and an intact-trained
#           policy loads. This is the mode for TRANSFER: continue training from an intact
#           policy, or evaluate one zero-shot. Proprioception reports the limb's INTACT rest
#           pose (--ghost_obs=rest), so the policy does not also face a distribution jump.
#
# Both are verified to be the same body: remaining mass 7.9469 kg either way for left_leg,
# and the centre of mass of the remaining bodies agrees to |diff| = 0.0e+00.
#
#   ./run_missing_limb.sh                        # left_leg, 3 seeds, cut, from scratch
#   ./run_missing_limb.sh left_arm 3 cut         # train without the arm
#   ./run_missing_limb.sh left_leg 3 ghost       # fine-tune the intact policy onto a ghost leg
#
# Needs `conda activate mimo` first. ~28 min per cut run (measured 593 fps), ~9 min per
# ghost fine-tuning run.
set -u

LIMB=${1:-left_leg}
RUNS=${2:-3}
MODE=${3:-cut}

# The intact reference policy: PPO+PBRS supine, 100 % full roll over 10 episodes,
# 52.9 +- 3.2 steps to roll (eval_rollover.py, ISR off, goal pinned).
BASELINE=models/roll_over/26-08-12/supine/26-08-12_supine_tst_ctr_cost_logger_run_0/model_10.zip

# osmesa, not egl -- egl fails in this conda env.
export MUJOCO_GL=osmesa

# The cut scenes have to exist before anything starts.
python mimoEnv/assets/roll_over/generate_amputated_scenes.py --check || {
  echo "Generating the amputated scenes first."
  python mimoEnv/assets/roll_over/generate_amputated_scenes.py
}

# One MIMo env costs ~3.6 GB RSS. Only one fits on 16 GB, hence the sequential loop rather
# than backgrounding -- same reasoning as run_her_sparse.sh.
for i in $(seq 0 $((RUNS - 1))); do
  NAME="missing_${LIMB}_${MODE}_run_${i}"
  echo "=== $(date +%F\ %H:%M) START $NAME ==="

  COMMON=(
    --roll_over_starting_position=supine
    --algorithm=PPO --pbrs --pbrs_w=100 --pen_factor=0.02
    --morph_age=9 --physio_age=9
    --missing_limb="$LIMB" --missing_limb_mode="$MODE"
    # Not optional when loading a model: without it illustrations.py sets
    # 'save_dir = dirname(--load_model)' and writes the new checkpoints into the baseline
    # policy's own directory.
    --roll_over_model_path_auto
    --save_model="$NAME"
  )

  if [ "$MODE" = "ghost" ]; then
    # Ghost keeps the spaces, so start from the intact policy. 300k rather than 1M: the
    # question is adaptation to the missing limb, not learning to roll from nothing.
    python mimoEnv/illustrations.py "${COMMON[@]}" --ghost_obs=rest \
      --load_model="$BASELINE" --train_for=300000 --save_every=100000
  else
    # Cut cannot load the intact policy, so this is from scratch. Intermediate checkpoints
    # because the last one is not reliably the best.
    python mimoEnv/illustrations.py "${COMMON[@]}" \
      --train_for=1000000 --save_every=200000
  fi

  echo "=== $(date +%F\ %H:%M) DONE  $NAME ==="
done

echo
echo "Evaluate (never --test, always the protocol):"
echo "  python mimoEnv/eval_rollover.py --group='models/roll_over/*/supine/*missing_${LIMB}_${MODE}_run_*' --episodes=50 --json=missing_${LIMB}_${MODE}.json"

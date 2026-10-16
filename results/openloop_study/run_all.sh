#!/bin/bash
# 13.10.2026 Runs run_policy.py for all policies, one after the other (never two at once,
# one MIMo env needs 1.8 GB). Policies that are already done are skipped.
# Afterwards open openloop_study.ipynb for the plots.
#
#   ./results/openloop_study/run_all.sh

cd "$(dirname "$0")/../.."      # the MIMo folder
OUT=results/openloop_study/data
export MUJOCO_GL=osmesa
mkdir -p $OUT/supine $OUT/prone

# Four policies per posture. All of them roll in 100 % of the episodes
# in the stored group evaluation (.mimolab/evals).
SUPINE_RUNS="0 1 2 3"     # all four roll left
PRONE_RUNS="8 11 0 12"     # 8 and 11 roll left, 0 and 12 roll right

for r in $SUPINE_RUNS; do
  [ -f $OUT/supine/run_$r.npz ] || python results/openloop_study/run_policy.py \
      --model models/roll_over/26-10-11/supine/26-10-11_supine_age9_metabolicpen34_run_$r/model_1.zip \
      --out $OUT/supine/run_$r.npz > $OUT/supine/run_$r.log 2>&1
done
for r in $PRONE_RUNS; do
  [ -f $OUT/prone/run_$r.npz ] || python results/openloop_study/run_policy.py \
      --model models/roll_over/26-10-07/prone/26-10-07_prone_age9_metabolicpen34_run_$r/model_1.zip \
      --out $OUT/prone/run_$r.npz > $OUT/prone/run_$r.log 2>&1
done
echo done

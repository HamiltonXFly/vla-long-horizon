#!/usr/bin/env bash
# Mid-episode perturbation sweep on LIBERO-10.
# After the first sub-goal is completed (+DELAY steps), the object of the next
# unfinished sub-goal is shifted in a random horizontal direction ("auto" mode
# falls back to robot joint noise when that sub-goal has no movable object).
#
# Usage: bash scripts/run_perturb_sweep.sh [SEED] [N_EPISODES]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1000}"
N_EP="${2:-50}"
DELAY="${DELAY:-5}"
MODE="${MODE:-auto}"
MAGS="${MAGS:-0.03 0.06 0.10}"   # metres for object shift (radians std if MODE=robot)

export MUJOCO_GL=egl
mkdir -p outputs
for MAG in $MAGS; do
  OUT="outputs/perturb_${MODE}_mag${MAG}_delay${DELAY}_seed${SEED}"
  echo "=== $OUT ==="
  VLA_LOG_DIR="$OUT" VLA_PERTURB_MODE="$MODE" VLA_PERTURB_MAG="$MAG" \
  VLA_PERTURB_DELAY="$DELAY" VLA_PERTURB_SEED="$SEED" \
  python scripts/run_eval.py \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero --env.task=libero_10 \
    --eval.batch_size=1 --eval.n_episodes="$N_EP" \
    --policy.n_action_steps=10 --policy.compile_model=false \
    --seed="$SEED" \
    --output_dir="$OUT/lerobot" 2>&1 | tee "$OUT.log"
done

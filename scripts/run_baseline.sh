#!/usr/bin/env bash
# Baseline: pi0.5 on LIBERO-10, no perturbation, stage-level logging.
# Usage: bash scripts/run_baseline.sh [SEED] [N_EPISODES]
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${1:-1000}"
N_EP="${2:-50}"
OUT="outputs/baseline_seed${SEED}"

export MUJOCO_GL=egl
mkdir -p outputs
export VLA_LOG_DIR="$OUT"
export VLA_PERTURB_MODE=none

python scripts/run_eval.py \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero --env.task=libero_10 \
  --eval.batch_size=1 --eval.n_episodes="$N_EP" \
  --policy.n_action_steps=10 --policy.compile_model=false \
  --seed="$SEED" \
  --output_dir="$OUT/lerobot" 2>&1 | tee "$OUT.log"

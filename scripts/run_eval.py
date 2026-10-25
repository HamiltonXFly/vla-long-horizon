#!/usr/bin/env python
"""Run `lerobot-eval` with stage-level logging (and optional perturbation).

Usage (all normal lerobot-eval arguments are passed through):

    VLA_LOG_DIR=outputs/baseline_seed1000 \
    python scripts/run_eval.py \
        --policy.path=lerobot/pi05_libero_finetuned \
        --env.type=libero --env.task=libero_10 \
        --eval.batch_size=1 --eval.n_episodes=50 \
        --policy.n_action_steps=10 --policy.compile_model=false \
        --seed=1000 --output_dir=outputs/baseline_seed1000/lerobot

Perturbation is configured with VLA_PERTURB_MODE / VLA_PERTURB_MAG /
VLA_PERTURB_DELAY / VLA_PERTURB_SEED (see vlalh/instrument.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlalh.instrument import patch_libero  # noqa: E402


def _check_args(argv: list[str]) -> None:
    bs = [a for a in argv if a.startswith("--eval.batch_size")]
    if not bs or bs[-1].split("=", 1)[-1].strip() != "1":
        sys.exit(
            "[vlalh] Please pass --eval.batch_size=1. The instrumentation patches the env "
            "in the main process, which LeRobot only uses for a single env per task."
        )
    if not os.environ.get("VLA_LOG_DIR"):
        sys.exit("[vlalh] Please set VLA_LOG_DIR, e.g. VLA_LOG_DIR=outputs/baseline_seed1000")


def main() -> None:
    _check_args(sys.argv[1:])
    os.environ.setdefault("MUJOCO_GL", "egl")
    patch_libero()
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()


if __name__ == "__main__":
    main()

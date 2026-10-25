#!/usr/bin/env python
"""Print the goal predicates of every LIBERO task and their truth value at reset.

This is a cheap sanity check (no policy, no GPU inference) that the
instrumentation can read sub-goal progress correctly:

    python scripts/inspect_tasks.py --suite libero_10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--init-state", type=int, default=0)
    args = ap.parse_args()

    from lerobot.envs.libero import LiberoEnv, _get_suite

    from vlalh.instrument import _eval_predicates, _format_pred, _goal_states, _inner, _movable_goal_objects

    suite = _get_suite(args.suite)
    for tid in range(len(suite.tasks)):
        env = LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=args.suite, episode_index=args.init_state)
        env.reset(seed=0)
        inner = _inner(env)
        goal = _goal_states(inner)
        vals = _eval_predicates(inner, goal)
        print(f"\n[{tid}] {env.task_description}")
        for s, v in zip(goal, vals):
            print(f"     {'TRUE ' if v else 'false'}  {_format_pred(s)}")
        print(f"     movable goal objects: {_movable_goal_objects(inner, goal)}")
        env.close()


if __name__ == "__main__":
    main()

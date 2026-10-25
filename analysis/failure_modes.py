#!/usr/bin/env python
"""Heuristic failure-mode labels for failed episodes of an instrumented run.

    python analysis/failure_modes.py outputs/baseline_seed1000 --out reports/baseline

Labels are computed from logged trajectories and are NOT mutually exclusive.
All criteria are deliberately simple so they can be checked against videos:

* stall          end-effector path length over the last 150 steps < 0.15 m
* regression     a sub-goal that became true is false again at the end
* carried_not_placed
                 an object of an unfinished sub-goal was lifted > 5 cm at some
                 point after the first sub-goal, but the sub-goal is still false
* repeated_grasp >= 5 gripper close commands after the first sub-goal
* no_subgoal     the episode never completed any sub-goal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STALL_WINDOW, STALL_PATH_M = 150, 0.15
LIFT_M = 0.05
REPEATED_CLOSES = 5


def label(run: Path, r: dict) -> dict:
    e = np.load(run / r["episode_file"])
    P, eef, act, O = e["predicates"], e["eef_pos"], e["actions"], e["obj_pose"]
    objs = list(e["tracked_objects"])
    init = np.array(r["initially_true"] or [False] * P.shape[1])
    reached = [j for j in range(P.shape[1]) if P[:, j].any() and not init[j]]
    t1 = min(int(np.flatnonzero(P[:, j])[0]) for j in reached) if reached else None
    seg = slice(t1 if t1 is not None else 0, None)

    last = eef[-STALL_WINDOW:]
    path = float(np.sum(np.linalg.norm(np.diff(last, axis=0), axis=1)))
    g = act[seg, -1]
    closes = int(np.sum((g[1:] > 0) & (g[:-1] <= 0)))
    regression = any(P[:, j].any() and not P[-1, j] and not init[j] for j in range(P.shape[1]))
    carried = False
    if t1 is not None and O.size:
        for j, name in enumerate(r["predicates"]):
            if P[-1, j]:
                continue
            for k, o in enumerate(objs):
                if f" {o} " in f" {name.strip('()')} ":
                    z = O[seg, k, 2]
                    if z.size and z.max() - z[0] > LIFT_M:
                        carried = True
    return {
        "task_id": r["task_id"], "init_state_id": r["init_state_id"],
        "first_subgoal_step": t1, "eef_path_last150_m": round(path, 3), "closes_after_first": closes,
        "stall": path < STALL_PATH_M, "regression": regression, "carried_not_placed": carried,
        "repeated_grasp": closes >= REPEATED_CLOSES, "no_subgoal": t1 is None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/baseline"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (args.run / "summary.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame([label(args.run, r) for r in rows if not r["success"]])
    df.to_csv(args.out / "failure_modes.csv", index=False)
    flags = ["stall", "regression", "carried_not_placed", "repeated_grasp", "no_subgoal"]
    print(f"{len(df)} failed episodes")
    for f in flags:
        print(f"  {f:20s} {int(df[f].sum()):3d}   tasks: {sorted(df.loc[df[f], 'task_id'].tolist())}")
    print(f"  {'none of the above':20s} {int((~df[flags].any(axis=1)).sum()):3d}")


if __name__ == "__main__":
    main()

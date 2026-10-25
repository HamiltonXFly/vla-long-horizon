#!/usr/bin/env python
"""Recovery from mid-episode perturbation (dose-response).

    python analysis/analyze_perturbation.py --baseline outputs/baseline_seed1000 \
        --runs outputs/perturb_auto_mag0.03_delay5_seed1000 \
               outputs/perturb_auto_mag0.06_delay5_seed1000 \
               outputs/perturb_auto_mag0.10_delay5_seed1000 \
        --out reports/perturbation

Definitions
-----------
* A perturbation is applied DELAY steps after the first sub-goal is completed
  (see vlalh/instrument.py). "auto" mode shifts the object of the next unfinished
  sub-goal in a random horizontal direction; if that sub-goal has no movable
  object (closing a drawer / microwave door: T3, T9) it adds Gaussian noise to
  the arm joints instead. The two kinds are reported separately because their
  magnitudes are in different units.
* Recovery rate = P(task success | perturbed).
* Matched baseline = P(task success | first sub-goal reached) in the
  unperturbed run, restricted to the tasks that receive that perturbation kind.
* Unstable episodes are excluded: after the perturbation, a goal object moves
  more than 0.15 m in a single control step (> 3 m/s) or becomes non-finite.
  This happens when an object is teleported into another body and the physics
  solver explodes; such outcomes say nothing about the policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3df"
JUMP_THRESHOLD_M = 0.15
KIND_LABEL = {"object_shift_xy": "Object shift (m)", "joint_noise": "Arm joint noise, std (rad)"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "summary.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["_run"] = run
    return rows


def reached_first(r: dict) -> bool:
    init = r["initially_true"] or [False] * len(r["predicates"])
    return any(f is not None and not i for f, i in zip(r["first_true_step"], init))


def is_unstable(r: dict) -> bool:
    ps = r.get("perturb_step")
    if ps is None:
        return False
    obj = np.load(r["_run"] / r["episode_file"])["obj_pose"]
    if obj.size == 0:
        return False
    if not np.isfinite(obj).all():
        return True
    step = np.linalg.norm(np.diff(obj[:, :, :3], axis=0), axis=2)
    step[max(ps - 1, 0) : ps + 1] = 0.0  # the teleport itself is not a jump
    after = step[max(ps - 1, 0) :]
    return bool(after.size and after.max() > JUMP_THRESHOLD_M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/perturbation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = load(args.baseline)
    runs = []
    for run in args.runs:
        cfg = json.loads((run / "instrument_config.json").read_text())
        runs.append((float(cfg["perturb_mag"]), run, load(run)))
    runs.sort(key=lambda x: x[0])

    table, excluded = [], []
    for kind in ["object_shift_xy", "joint_noise"]:
        tasks = sorted({r["task_id"] for _, _, rows in runs for r in rows if r["perturb_kind"] == kind})
        # ignore tasks where this kind only happened by accident (< 10% of episodes)
        counts = {t: sum(1 for _, _, rows in runs for r in rows if r["task_id"] == t and r["perturb_kind"] == kind) for t in tasks}
        tasks = [t for t in tasks if counts[t] >= 0.1 * len(runs) * 50]
        b = [r for r in base if r["task_id"] in tasks and reached_first(r)]
        k = sum(r["success"] for r in b)
        lo, hi = wilson(k, len(b))
        table.append({"kind": kind, "magnitude": 0.0, "tasks": tasks, "n": len(b), "k": k,
                      "rate": k / len(b), "ci_lo": lo, "ci_hi": hi, "n_unstable_excluded": 0})
        for mag, run, rows in runs:
            p = [r for r in rows if r["perturb_kind"] == kind and r["task_id"] in tasks]
            unstable = [r for r in p if is_unstable(r)]
            excluded += [{"run": run.name, "task_id": r["task_id"], "init_state_id": r["init_state_id"], "kind": kind} for r in unstable]
            v = [r for r in p if r not in unstable]
            k = sum(r["success"] for r in v)
            lo, hi = wilson(k, len(v))
            per_task = {f"T{t}": f"{sum(r['success'] for r in v if r['task_id'] == t)}/{sum(1 for r in v if r['task_id'] == t)}" for t in tasks}
            table.append({"kind": kind, "magnitude": mag, "tasks": tasks, "n": len(v), "k": k,
                          "rate": k / len(v) if v else float("nan"), "ci_lo": lo, "ci_hi": hi,
                          "n_unstable_excluded": len(unstable), **per_task})

    df = pd.DataFrame(table)
    df.to_csv(args.out / "recovery.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.out / "excluded_unstable.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.drop(columns=["tasks"]).round(3).to_string(index=False))
    print(f"\nexcluded unstable episodes: {len(excluded)}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160, sharey=True)
    for ax, kind in zip(axes, ["object_shift_xy", "joint_noise"]):
        d = df[df["kind"] == kind].sort_values("magnitude")
        x, y = d["magnitude"].to_numpy(), d["rate"].to_numpy() * 100
        ax.fill_between(x, d["ci_lo"] * 100, d["ci_hi"] * 100, color=SERIES_1, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=SERIES_1, linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
        for xi, yi, n in zip(x, y, d["n"]):
            ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8, color=TEXT_PRIMARY)
        tasks = d["tasks"].iloc[0]
        ax.set_title(f"{KIND_LABEL[kind].split(',')[0].split(' (')[0]}  (tasks {', '.join('T'+str(t) for t in tasks)})",
                     fontsize=9, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel(KIND_LABEL[kind] + "  (0 = unperturbed)", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Task success after first sub-goal (%)", fontsize=8, color=TEXT_PRIMARY)
    fig.suptitle("π0.5 on LIBERO-10: recovery from a mid-episode perturbation (95% Wilson CI)",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "fig_recovery.png")
    print(f"wrote {args.out}/recovery.csv, excluded_unstable.csv, fig_recovery.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage-level analysis of instrumented LIBERO runs.

    python analysis/analyze_stages.py outputs/baseline_seed1000 \
        [outputs/perturb_auto_mag0.03_delay5_seed1000 ...] --out reports/run1

Writes CSV tables and PNG figures into --out:

    episodes.csv            one row per episode, all runs
    per_task.csv            success / stage outcome per (run, task)
    per_run.csv             overall numbers per run (with Wilson 95% CIs)
    fig_task_success.png    per-task success of the first run, with CIs
    fig_stage_outcome.png   how far each episode got (none / partial / all sub-goals)
    fig_recovery.png        success after mid-episode perturbation vs magnitude
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Ordinal ramp (one hue, light -> dark) for "how far did the episode get".
STAGE_COLORS = {"none": "#9ec5f4", "partial": "#3987e5", "complete": "#184f95"}
SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3df"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - was VLA_LOG_DIR set to {run_dir}?")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["run"] = run_dir.name
    cfg_path = run_dir / "instrument_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    df["run_mode"] = cfg.get("perturb_mode", "none")
    df["run_mag"] = float(cfg.get("perturb_mag", 0.0))
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    def reached(row) -> int:
        init = row["initially_true"] or [False] * len(row["predicates"])
        return int(sum(1 for f, i in zip(row["first_true_step"], init) if f is not None and not i))

    def needed(row) -> int:
        init = row["initially_true"] or [False] * len(row["predicates"])
        return int(sum(1 for i in init if not i))

    df = df.copy()
    df["n_subgoals"] = df["predicates"].apply(len)
    df["n_needed"] = df.apply(needed, axis=1)
    df["n_reached"] = df.apply(reached, axis=1)
    df["multi_stage"] = df["n_needed"] >= 2

    def stage(row) -> str:
        if row["success"]:
            return "complete"
        return "partial" if row["n_reached"] > 0 else "none"

    df["stage"] = df.apply(stage, axis=1)
    df["reached_first"] = df["n_reached"] > 0
    df["task_label"] = df["task_id"].apply(lambda t: f"T{int(t)}")
    return df


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def per_task(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, tid), g in df.groupby(["run", "task_id"]):
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson(k, n)
        first = g[g["reached_first"]]
        rows.append(
            {
                "run": run,
                "task_id": int(tid),
                "task": g["task_description"].iloc[0],
                "n_needed_subgoals": int(g["n_needed"].iloc[0]),
                "n": n,
                "success_rate": k / n,
                "success_ci_lo": lo,
                "success_ci_hi": hi,
                "p_none": float((g["stage"] == "none").mean()),
                "p_partial": float((g["stage"] == "partial").mean()),
                "p_complete": float((g["stage"] == "complete").mean()),
                "p_success_given_first": float(first["success"].mean()) if len(first) else float("nan"),
                "n_reached_first": len(first),
                "median_success_step": float(g.loc[g["success"], "success_step"].median()) if k else float("nan"),
                "n_perturbed": int(g["perturbed"].sum()),
            }
        )
    return pd.DataFrame(rows)


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run", sort=False):
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson(k, n)
        multi = g[g["multi_stage"]]
        first = multi[multi["reached_first"]]
        kf = int(first["success"].sum())
        flo, fhi = wilson(kf, len(first))
        pert = g[g["perturb_kind"].isin(["object_shift_xy", "joint_noise"])]
        kp = int(pert["success"].sum())
        plo, phi = wilson(kp, len(pert))
        rows.append(
            {
                "run": run,
                "mode": g["run_mode"].iloc[0],
                "mag": g["run_mag"].iloc[0],
                "n": n,
                "success_rate": k / n,
                "success_ci_lo": lo,
                "success_ci_hi": hi,
                "multi_stage_n": len(multi),
                "p_success_given_first_subgoal": kf / len(first) if len(first) else float("nan"),
                "p_sgf_ci_lo": flo,
                "p_sgf_ci_hi": fhi,
                "n_perturbed": len(pert),
                "success_after_perturbation": kp / len(pert) if len(pert) else float("nan"),
                "sap_ci_lo": plo,
                "sap_ci_hi": phi,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_task_success(pt: pd.DataFrame, run: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    d = pt[pt["run"] == run].sort_values("task_id")
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    x = np.arange(len(d))
    y = d["success_rate"].to_numpy() * 100
    err = np.vstack([y - d["success_ci_lo"] * 100, d["success_ci_hi"] * 100 - y])
    ax.bar(x, y, width=0.6, color=SERIES_1, edgecolor="white", linewidth=2)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3)
    ax.set_xticks(x, [f"T{t}" for t in d["task_id"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_title(f"Per-task success, {run} (n={int(d['n'].iloc[0])}/task, 95% Wilson CI)",
                 color=TEXT_PRIMARY, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_task_success.png")
    plt.close(fig)


def fig_stage_outcome(pt: pd.DataFrame, run: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    d = pt[pt["run"] == run].sort_values("task_id")
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    labels = {"none": "No sub-goal reached", "partial": "Some sub-goals reached", "complete": "Task completed"}
    for key in ["none", "partial", "complete"]:
        vals = d[f"p_{key}"].to_numpy() * 100
        ax.bar(x, vals, bottom=bottom, width=0.6, color=STAGE_COLORS[key], edgecolor="white",
               linewidth=2, label=labels[key])
        bottom += vals
    ax.set_xticks(x, [f"T{t}" for t in d["task_id"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of episodes (%)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_title(f"How far each episode got, {run}", color=TEXT_PRIMARY, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12),
              labelcolor=TEXT_SECONDARY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_stage_outcome.png")
    plt.close(fig)


def fig_recovery(pr: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    base = pr[pr["mode"] == "none"]
    pert = pr[(pr["mode"] != "none") & (pr["n_perturbed"] > 0)].sort_values("mag")
    if pert.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    xs = [0.0] + pert["mag"].tolist() if not base.empty else pert["mag"].tolist()
    ys, lo, hi = [], [], []
    if not base.empty:
        b = base.iloc[0]
        ys.append(b["p_success_given_first_subgoal"]); lo.append(b["p_sgf_ci_lo"]); hi.append(b["p_sgf_ci_hi"])
    ys += pert["success_after_perturbation"].tolist()
    lo += pert["sap_ci_lo"].tolist()
    hi += pert["sap_ci_hi"].tolist()
    ys, lo, hi = (np.array(v) * 100 for v in (ys, lo, hi))
    ax.fill_between(xs, lo, hi, color=SERIES_1, alpha=0.15, linewidth=0)
    ax.plot(xs, ys, color=SERIES_1, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    ax.set_xlabel("Perturbation magnitude (m object shift / rad joint std)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel("Task success after first sub-goal (%)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Recovery from mid-episode perturbation (0 = unperturbed baseline)",
                 color=TEXT_PRIMARY, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_recovery.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/latest"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = enrich(pd.concat([load_run(r) for r in args.runs], ignore_index=True))
    dup = df.duplicated(["run", "task_id", "init_state_id"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate (run, task, init_state) rows - did a run append twice to the same log dir?")
    pt, pr = per_task(df), per_run(df)
    df.drop(columns=["predicates"]).to_csv(args.out / "episodes.csv", index=False)
    pt.to_csv(args.out / "per_task.csv", index=False)
    pr.to_csv(args.out / "per_run.csv", index=False)

    first_run = args.runs[0].name
    fig_task_success(pt, first_run, args.out)
    fig_stage_outcome(pt, first_run, args.out)
    fig_recovery(pr, args.out)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n=== per run ===")
        print(pr.round(3).to_string(index=False))
        print("\n=== per task (first run) ===")
        cols = ["task_id", "n_needed_subgoals", "n", "success_rate", "p_none", "p_partial",
                "p_complete", "p_success_given_first", "median_success_step"]
        print(pt[pt["run"] == first_run][cols].round(3).to_string(index=False))
    print(f"\nWrote tables and figures to {args.out}")


if __name__ == "__main__":
    main()

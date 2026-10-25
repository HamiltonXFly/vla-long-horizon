#!/usr/bin/env python
"""Plain-language headline figures for the README (docs/fig*.png).

Run after analyze_stages.py, analyze_perturbation.py and part2_failure_monitor.py:

    python analysis/make_readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BLUE, ORANGE, ORANGE_LIGHT, GRAY = "#2a78d6", "#eb6834", "#f6b393", "#8a8a85"
TEXT, TEXT2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
TASK_NAMES = {
    0: "soup + sauce → basket",
    1: "cheese + butter → basket",
    2: "stove on, moka pot on stove",
    3: "bowl → drawer, close drawer",
    4: "two mugs → two plates",
    5: "book → caddy (1 step)",
    6: "mug → plate, pudding → right",
    7: "soup + cheese → basket",
    8: "two moka pots → stove",
    9: "mug → microwave, close door",
}


def style(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_where_failures():
    rows = [json.loads(l) for l in (ROOT / "outputs/baseline_seed1000/summary.jsonl").read_text().splitlines()]
    df = pd.DataFrame(rows)

    def reached(r):
        init = r["initially_true"] or [False] * len(r["predicates"])
        return any(f is not None and not i for f, i in zip(r["first_true_step"], init))

    df["reached"] = df.apply(reached, axis=1)
    g = df.groupby("task_id").agg(n=("success", "size"), ok=("success", "sum"),
                                  fail_after=("reached", lambda s: 0)).reset_index()
    g["fail_after"] = [int(((df.task_id == t) & ~df.success & df.reached).sum()) for t in g.task_id]
    g["fail_before"] = g["n"] - g["ok"] - g["fail_after"]
    g = g.iloc[::-1]  # T0 at the top
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=170)
    left = np.zeros(len(g))
    for col, color, lab in [("ok", BLUE, "completed the whole task"),
                            ("fail_after", ORANGE, "failed AFTER finishing step 1"),
                            ("fail_before", ORANGE_LIGHT, "failed before finishing step 1")]:
        v = g[col].to_numpy() / g["n"].to_numpy() * 100
        ax.barh(y, v, left=left, color=color, edgecolor="white", linewidth=1.5, height=0.62, label=lab)
        left += v
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.text(101, yi, f"{r.ok}/{r.n}", va="center", fontsize=8, color=TEXT2)
    ax.set_yticks(y, [TASK_NAMES[t] for t in g.task_id], fontsize=9, color=TEXT)
    ax.set_xlim(0, 109)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("27 of 30 failures happen after the robot has already finished step 1",
                 fontsize=10.5, color=TEXT, loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 0.995),
              labelcolor=TEXT2, handlelength=1.2)
    style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "fig1_where_failures.png")
    plt.close(fig)


def fig_disturbance():
    d = pd.read_csv(ROOT / "reports/perturbation/recovery.csv")
    d = d[d.kind == "object_shift_xy"].sort_values("magnitude")
    x = np.arange(len(d))
    y = d.rate.to_numpy() * 100
    err = np.vstack([y - d.ci_lo * 100, d.ci_hi * 100 - y])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=170)
    colors = [GRAY] + [BLUE] * (len(d) - 1)
    ax.bar(x, y, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=TEXT2, elinewidth=1, capsize=3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 6, f"{yi:.0f}%", ha="center", fontsize=10, color=TEXT, fontweight="bold")
    labels = ["not moved"] + [f"moved {m*100:.0f} cm" for m in d.magnitude[1:]]
    ax.set_xticks(x, labels, fontsize=9, color=TEXT)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Task still completed", fontsize=9, color=TEXT)
    ax.set_title("After step 1, we move the object needed for step 2", fontsize=10.5, color=TEXT, loc="left")
    style(ax, "y")
    fig.tight_layout()
    fig.savefig(DOCS / "fig2_disturbance.png")
    plt.close(fig)


def fig_early_warning():
    d = pd.read_csv(ROOT / "reports/part2/auroc.csv")
    d = d[d.decision_steps_after == 40]
    groups = [("object_shift_xy", "Object moved\n(7 tasks)"), ("joint_noise", "Robot arm bumped\n(2 tasks)")]
    preds = [("WM residual (MLP)", "World model (ours)", BLUE),
             ("Stall heuristic", "Rule: \"robot stopped moving\"", ORANGE),
             ("Magnitude (sim oracle)", "Knowing how big the disturbance was*", GRAY)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=170)
    w = 0.26
    for j, (key, lab, color) in enumerate(preds):
        for i, (kind, _) in enumerate(groups):
            r = d[(d.kind == kind) & (d.predictor == key)].iloc[0]
            xi = i + (j - 1) * w
            ax.errorbar(xi, r.auroc, yerr=[[r.auroc - r.ci_lo], [r.ci_hi - r.auroc]], fmt="none",
                        ecolor=color, elinewidth=2, capsize=0)
            ax.plot(xi, r.auroc, "o", color=color, markersize=11, markeredgecolor="white", markeredgewidth=2,
                    label=lab if i == 0 else None)
            ax.text(xi + 0.045, r.auroc, f"{r.auroc:.2f}", ha="left", va="center", fontsize=9, color=TEXT)
    ax.axhline(0.5, color=TEXT2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(-0.45, 0.507, "coin flip", fontsize=8, color=TEXT2, ha="left", va="bottom")
    ax.set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=9, color=TEXT)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("How well it predicts failure\n(AUROC: 0.5 = guessing, 1 = perfect)", fontsize=8.5, color=TEXT)
    ax.set_title("2 seconds after a disturbance: who can tell the robot will fail?", fontsize=10.5,
                 color=TEXT, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 0.99), labelcolor=TEXT2,
              handlelength=1.2)
    style(ax, "y")
    fig.text(0.01, 0.01, "* only known in simulation", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "fig3_early_warning.png")
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    fig_where_failures()
    fig_disturbance()
    fig_early_warning()
    print("wrote", sorted(p.name for p in DOCS.glob("fig*.png")))

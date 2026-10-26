#!/usr/bin/env python
"""README figures (docs/fig*.png).

Run after analyze_stages.py, analyze_perturbation.py and part2_failure_monitor.py:

    python analysis/make_readme_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("VLA_ROOT", Path(__file__).resolve().parents[1]))
DOCS = Path(os.environ.get("VLA_DOCS", ROOT / "docs"))
BLUE, ORANGE, ORANGE_LIGHT, GRAY = "#2a78d6", "#eb6834", "#f6b393", "#8a8a85"
TEXT, TEXT2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
TASK_NAMES = {
    0: "soup + sauce → basket",
    1: "cheese + butter → basket",
    2: "stove on, moka pot on stove",
    3: "bowl → drawer, close drawer",
    4: "two mugs → two plates",
    5: "book → caddy",
    6: "mug → plate, pudding → right",
    7: "soup + cheese → basket",
    8: "two moka pots → stove",
    9: "mug → microwave, close door",
}


def style(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_where_failures():
    """Failed episodes per task, split by whether step 1 had been completed."""
    rows = [json.loads(l) for l in (ROOT / "outputs/baseline_seed1000/summary.jsonl").read_text().splitlines()]
    df = pd.DataFrame(rows)

    def reached(r):
        init = r["initially_true"] or [False] * len(r["predicates"])
        return any(f is not None and not i for f, i in zip(r["first_true_step"], init))

    df["reached"] = df.apply(reached, axis=1)
    f = df[~df.success]
    g = pd.DataFrame({"after": f[f.reached].groupby("task_id").size(),
                      "before": f[~f.reached].groupby("task_id").size()}).fillna(0).astype(int)
    g["total"] = g.after + g.before
    g = g.sort_values("total")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.6, 2.9), dpi=170)
    ax.barh(y, g.after, color=ORANGE, height=0.6, edgecolor="white", linewidth=1.5, label="after step 1 was done")
    ax.barh(y, g.before, left=g.after, color=ORANGE_LIGHT, height=0.6, edgecolor="white", linewidth=1.5,
            label="before step 1")
    for yi, t in zip(y, g.total):
        ax.text(t + 0.25, yi, str(t), va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, [TASK_NAMES[t] for t in g.index], fontsize=9, color=TEXT)
    ax.set_xlabel("failed episodes (of 50 per task)", fontsize=9, color=TEXT2)
    ax.set_xlim(0, g.total.max() + 1.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=TEXT2, handlelength=1.2)
    style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "fig1_where_failures.png")
    plt.close(fig)


def fig_disturbance():
    """Success rate vs disturbance size, object shift and arm push side by side."""
    d = pd.read_csv(ROOT / "reports/perturbation/recovery.csv")
    panels = [("object_shift_xy", "Next object moved (7 tasks)", lambda m: "none" if m == 0 else f"{m * 100:.0f} cm"),
              ("joint_noise", "Arm pushed (2 tasks)", lambda m: "none" if m == 0 else f"{m:.2f} rad")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=170, sharey=True)
    for ax, (kind, title, fmt) in zip(axes, panels):
        s = d[d.kind == kind].sort_values("magnitude")
        x = np.arange(len(s))
        yv = s.rate.to_numpy() * 100
        ax.fill_between(x, s.ci_lo * 100, s.ci_hi * 100, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(x, yv, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="white",
                markeredgewidth=1.5)
        for xi, yi in zip(x, yv):
            ax.text(xi, yi + 5, f"{yi:.0f}%", ha="center", fontsize=9, color=TEXT)
        ax.set_xticks(x, [fmt(m) for m in s.magnitude], fontsize=9, color=TEXT)
        ax.set_xlim(-0.4, len(s) - 0.6)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        style(ax, "y")
    axes[0].set_ylim(0, 108)
    axes[0].set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("tasks completed", fontsize=9, color=TEXT2)
    fig.tight_layout()
    fig.savefig(DOCS / "fig2_disturbance.png")
    plt.close(fig)


def fig_early_warning():
    """AUROC 2 s after the disturbance: world model vs stall rule."""
    d = pd.read_csv(ROOT / "reports/part2/auroc.csv")
    d = d[d.decision_steps_after == 40]
    groups = [("object_shift_xy", "next object moved"), ("joint_noise", "arm pushed")]
    preds = [("WM residual (MLP)", "world model", BLUE), ("Stall heuristic", "\"robot stopped moving\" rule", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=170)
    y = np.arange(len(groups))[::-1]
    for j, (key, lab, color) in enumerate(preds):
        for yi, (kind, _) in zip(y, groups):
            r = d[(d.kind == kind) & (d.predictor == key)].iloc[0]
            yy = yi + (0.14 if j == 0 else -0.14)
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=color, linewidth=2.5, solid_capstyle="round", alpha=0.45)
            ax.plot(r.auroc, yy, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=1.5,
                    label=lab if yi == y[0] else None)
            ax.text(r.ci_hi + 0.012, yy, f"{r.auroc:.2f}", va="center", fontsize=9, color=TEXT)
    ax.axvline(0.5, color=TEXT2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, y[-1] - 0.45, "guessing", fontsize=8, color=TEXT2, va="bottom")
    ax.set_yticks(y, [g[1] for g in groups], fontsize=9.5, color=TEXT)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(y[-1] - 0.5, y[0] + 0.5)
    ax.set_xlabel("AUROC for predicting failure, 2 s after the disturbance (1 = perfect)", fontsize=9, color=TEXT2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=TEXT2,
              handlelength=1.0)
    style(ax)
    fig.tight_layout()
    fig.savefig(DOCS / "fig3_early_warning.png")
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    fig_where_failures()
    fig_disturbance()
    fig_early_warning()
    print("wrote", sorted(p.name for p in DOCS.glob("fig*.png")))

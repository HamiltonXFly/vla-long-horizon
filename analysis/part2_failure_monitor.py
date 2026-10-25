#!/usr/bin/env python
"""Part 2: can a small world model tell, shortly after a disturbance, whether
π0.5 will recover?

    python analysis/part2_failure_monitor.py \
        --baseline outputs/baseline_seed1000 \
        --perturbed outputs/perturb_auto_mag0.03_delay5_seed1000 \
                    outputs/perturb_auto_mag0.06_delay5_seed1000 \
                    outputs/perturb_auto_mag0.10_delay5_seed1000 \
        --out reports/part2

Protocol
--------
1. Train an action-conditioned dynamics model (vlalh/world_model.py) on the
   successful *unperturbed* baseline episodes with init_state_id < 35;
   calibrate residual scales on successful baseline episodes with id >= 35.
   No perturbed episode is ever used for training.
2. For every perturbed episode (perturbation at step p) pick a decision time
   tau = p + W. Only information up to tau may be used. Episodes that already
   ended before tau, and physics-unstable episodes, are excluded.
3. Predict the final outcome (failure = 1) with
     - WM residual    : mean residual score of predictions that are complete by tau
     - WM residual (linear): same with a ridge-regression dynamics model
     - Stall heuristic: negative end-effector path length in (p, tau]
     - Magnitude      : the perturbation size (an oracle only available in simulation)
   and report AUROC with 95% bootstrap CIs. "Within-magnitude" / "within-task" /
   "within-task x magnitude" AUROCs average the AUROC computed separately inside
   each stratum (weighted by size). They ask whether a score adds information
   beyond knowing how hard the perturbation was and which task is running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalh.world_model import DynamicsModel, Episode, load_run, training_set  # noqa: E402

K = 10
WINDOWS = [20, 40, 80]
JUMP_THRESHOLD_M = 0.15
SERIES = {"WM residual (MLP)": "#2a78d6", "WM residual (linear)": "#eb6834",
          "Stall heuristic": "#1baf7a", "Magnitude (sim oracle)": "#8a8a85"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"


# --------------------------------------------------------------------------- #
def unstable(ep: Episode) -> bool:
    p = ep.perturb_step
    if p is None:
        return False
    obj = ep.state[:, 12:]
    if not np.isfinite(obj).all():
        return True
    pos = obj.reshape(ep.n_steps, -1, 3)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    step[max(p - 1, 0) : p + 1] = 0.0
    after = step[max(p - 1, 0) :]
    return bool(after.size and after.max() > JUMP_THRESHOLD_M)


def first_subgoal_step(ep: Episode) -> int | None:
    s = ep.summary
    init = s["initially_true"] or [False] * len(s["predicates"])
    steps = [f for f, i in zip(s["first_true_step"], init) if f is not None and not i]
    return min(steps) if steps else None


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = auroc(y[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def within_strata_auroc(y, s, strata):
    tot, acc = 0, 0.0
    for g in np.unique(strata):
        m = strata == g
        a = auroc(y[m], s[m])
        if not np.isnan(a):
            acc += a * m.sum()
            tot += m.sum()
    return acc / tot if tot else float("nan")


# --------------------------------------------------------------------------- #
def features_at(ep: Episode, p: int, W: int, res_mlp: np.ndarray, res_lin: np.ndarray) -> dict | None:
    tau = p + W
    if ep.n_steps <= tau:  # episode already over (success) before the decision time
        return None
    lo, hi = p + 1, tau - K  # prediction t -> t+K must be complete by tau
    if hi < lo:
        return None
    path = float(np.sum(np.linalg.norm(np.diff(ep.eef_pos[p + 1 : tau + 1], axis=0), axis=1)))
    return {
        "WM residual (MLP)": float(np.nanmean(res_mlp[lo : hi + 1])),
        "WM residual (linear)": float(np.nanmean(res_lin[lo : hi + 1])),
        "Stall heuristic": -path,
        "Magnitude (sim oracle)": ep.perturb_mag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--perturbed", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/part2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng_seed = 0

    base = load_run(args.baseline)
    train = [e for e in base if e.success and e.init_state_id < 35]
    calib = [e for e in base if e.success and e.init_state_id >= 35]
    print(f"train episodes {len(train)}, calibration episodes {len(calib)}")

    X, Y, M = training_set(train, K, stride=2)
    print(f"training samples {len(X)}, input dim {X.shape[1]}")
    mlp = DynamicsModel("mlp", K, seed=rng_seed).fit(X, Y, M).calibrate(calib)
    lin = DynamicsModel("ridge", K).fit(X, Y, M).calibrate(calib)

    # held-out prediction quality (R^2 on calibration episodes)
    Xc, Yc, Mc = training_set(calib, K, stride=1)
    quality = {}
    for name, mdl in [("mlp", mlp), ("linear", lin)]:
        P = mdl.predict(Xc)
        for grp, sl in [("robot", slice(0, 12)), ("objects", slice(12, 24))]:
            m = Mc[:, sl]
            yt, yp = Yc[:, sl][m], P[:, sl][m]
            quality[f"{name}_{grp}_R2"] = float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))
    print("held-out R^2:", {k: round(v, 3) for k, v in quality.items()})

    # ---------------- perturbed episodes ----------------
    pert = [e for run in args.perturbed for e in load_run(run)
            if e.perturb_kind in ("object_shift_xy", "joint_noise")]
    n_unstable = sum(unstable(e) for e in pert)
    pert = [e for e in pert if not unstable(e)]
    res_cache = {id(e): (mlp.residual_series(e), lin.residual_series(e)) for e in pert}

    # unperturbed reference: same decision rule, anchored at first sub-goal + 5 steps
    base_eval = []
    for e in base:
        t1 = first_subgoal_step(e)
        if t1 is not None and e.summary["n_steps"] > t1 + 5:
            base_eval.append((e, t1 + 5))
    res_base = {id(e): (mlp.residual_series(e), lin.residual_series(e)) for e, _ in base_eval}

    rows, per_ep = [], []
    for kind in ["object_shift_xy", "joint_noise", "unperturbed"]:
        for W in WINDOWS:
            feats, ys, mags, tasks = [], [], [], []
            items = [(e, e.perturb_step) for e in pert if e.perturb_kind == kind] if kind != "unperturbed" else base_eval
            cache = res_cache if kind != "unperturbed" else res_base
            n_ended = 0
            for e, p in items:
                rm, rl = cache[id(e)]
                f = features_at(e, p, W, rm, rl)
                if f is None:
                    n_ended += 1
                    continue
                feats.append(f)
                ys.append(0 if e.success else 1)
                mags.append(e.perturb_mag)
                tasks.append(e.task_id)
                if W == 40:
                    per_ep.append({"kind": kind, "run": e.run, "task_id": e.task_id, "init_state_id": e.init_state_id,
                                   "magnitude": e.perturb_mag, "failed": int(not e.success), **f})
            y = np.array(ys)
            mags = np.array(mags)
            tasks = np.array(tasks)
            task_mag = np.array([f"{t}_{m}" for t, m in zip(tasks, mags)])
            for name in SERIES:
                if kind == "unperturbed" and name == "Magnitude (sim oracle)":
                    continue
                s = np.array([f[name] for f in feats])
                a = auroc(y, s)
                lo, hi = boot_ci(y, s, seed=rng_seed)
                rows.append({"kind": kind, "decision_steps_after": W, "predictor": name, "n": len(y),
                             "n_fail": int(y.sum()), "n_ended_before_decision": n_ended,
                             "auroc": a, "ci_lo": lo, "ci_hi": hi,
                             "auroc_within_magnitude": within_strata_auroc(y, s, mags) if kind != "unperturbed" else np.nan,
                             "auroc_within_task": within_strata_auroc(y, s, tasks),
                             "auroc_within_task_x_magnitude": within_strata_auroc(y, s, task_mag) if kind != "unperturbed" else np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "auroc.csv", index=False)
    pd.DataFrame(per_ep).to_csv(args.out / "episode_scores_W40.csv", index=False)
    json.dump({"K": K, "windows": WINDOWS, "n_train_episodes": len(train), "n_calib_episodes": len(calib),
               "n_train_samples": int(len(X)), "n_unstable_excluded": int(n_unstable), **quality},
              open(args.out / "summary.json", "w"), indent=2)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))

    # ---------------- figures ----------------
    import matplotlib.pyplot as plt

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=160, sharey=True)
    titles = {"object_shift_xy": "Object shift (T0 T1 T2 T4 T6 T7 T8)", "joint_noise": "Arm joint noise (T3 T9)",
              "unperturbed": "Unperturbed, after first sub-goal"}
    for ax, kind in zip(axes, ["object_shift_xy", "joint_noise", "unperturbed"]):
        d = df[df["kind"] == kind]
        for name, color in SERIES.items():
            q = d[d["predictor"] == name].sort_values("decision_steps_after")
            if q.empty:
                continue
            x = q["decision_steps_after"].to_numpy() / 20.0  # seconds at 20 Hz
            ax.plot(x, q["auroc"], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=name)
            ax.fill_between(x, q["ci_lo"], q["ci_hi"], color=color, alpha=0.10, linewidth=0)
        ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
        n, nf = int(d["n"].iloc[0]), int(d["n_fail"].iloc[0])
        ax.set_title(f"{titles[kind]}\nn={n}, failures={nf} (at 2 s)", fontsize=9, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Decision time after perturbation (s)", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylim(0.3, 1.0)
        style(ax)
    axes[0].set_ylabel("AUROC for predicting failure", fontsize=8, color=TEXT_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="lower center", labelcolor=TEXT_SECONDARY)
    fig.suptitle("Can we tell early whether π0.5 will recover?  (95% bootstrap CI; dashed = chance)",
                 fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out / "fig_auroc_vs_time.png")
    plt.close(fig)

    # score distributions at 2 s after the perturbation: recovered vs failed (empirical CDF)
    sc = pd.DataFrame(per_ep)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=160, sharey=True)
    for ax, kind in zip(axes, ["object_shift_xy", "joint_noise"]):
        q = sc[sc["kind"] == kind]
        for failed, color, lab in [(0, "#2a78d6", "recovered"), (1, "#eb6834", "failed")]:
            v = np.sort(q.loc[q["failed"] == failed, "WM residual (MLP)"].to_numpy())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, linewidth=2,
                    label=f"{lab} (n={len(v)})")
        ax.set_xscale("log")
        ax.set_title(titles[kind], fontsize=9, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Mean WM residual in the 2 s after the perturbation (log)", fontsize=8, color=TEXT_SECONDARY)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_SECONDARY)
        style(ax)
    axes[0].set_ylabel("Share of episodes (cumulative)", fontsize=8, color=TEXT_PRIMARY)
    fig.suptitle("Failed episodes deviate more from the world model's prediction (curve further right = larger residual)",
                 fontsize=9, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "fig_residual_distribution.png")
    plt.close(fig)
    print(f"\nexcluded unstable: {n_unstable}; wrote {args.out}")


if __name__ == "__main__":
    main()

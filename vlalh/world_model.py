"""A small state-based world model (action-conditioned dynamics) for failure monitoring.

Idea
----
A world model answers "if I execute these actions from this state, what will the
state be?". We train one only on *successful, unperturbed* executions of π0.5,
so it learns what normal progress looks like:

    s_{t+k}  ≈  s_t + f( s_t, a_t ... a_{t+k-1}, task )

During a new episode we compare its prediction with what actually happened.
A large, persistent prediction residual means "the episode is not evolving the
way successful episodes do", which may signal an upcoming failure.

This is the same principle as the residual test in Kalman-filter based anomaly
detection: predict, observe, and look at the innovation (prediction error).

State s_t (privileged simulator state, see Limitations in the README):
    end-effector position (3), gripper joint positions (2), arm joint positions (7),
    positions of up to 4 goal objects (4 x 3, zero-padded, with a mask).
Inputs: s_t, the next k executed actions (k x 7), task one-hot (10).
Output: Δs = s_{t+k} - s_t (24 numbers; padded object slots are ignored).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_TASKS = 10
MAX_OBJ = 4
ROBOT_DIM = 3 + 2 + 7
STATE_DIM = ROBOT_DIM + MAX_OBJ * 3


# --------------------------------------------------------------------------- #
# Loading logged episodes
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    run: str
    task_id: int
    init_state_id: int
    success: bool
    n_steps: int
    state: np.ndarray        # (T, STATE_DIM)
    mask: np.ndarray         # (STATE_DIM,) which dims are real
    actions: np.ndarray      # (T, 7)
    eef_pos: np.ndarray      # (T, 3)
    perturb_step: int | None
    perturb_kind: str | None
    perturb_mag: float
    first_partial_step: int | None
    summary: dict


def _state_from_npz(e) -> tuple[np.ndarray, np.ndarray]:
    T = e["actions"].shape[0]
    robot = np.concatenate([e["eef_pos"], e["gripper_qpos"], e["joint_pos"]], axis=1)
    obj = np.zeros((T, MAX_OBJ * 3), np.float32)
    mask = np.zeros(STATE_DIM, bool)
    mask[:ROBOT_DIM] = True
    n_obj = min(e["obj_pose"].shape[1], MAX_OBJ) if e["obj_pose"].ndim == 3 else 0
    for k in range(n_obj):
        obj[:, 3 * k : 3 * k + 3] = e["obj_pose"][:, k, :3]
        mask[ROBOT_DIM + 3 * k : ROBOT_DIM + 3 * k + 3] = True
    return np.concatenate([robot, obj], axis=1).astype(np.float32), mask


def load_run(run_dir: Path) -> list[Episode]:
    cfg = json.loads((run_dir / "instrument_config.json").read_text())
    out = []
    for line in (run_dir / "summary.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        e = np.load(run_dir / r["episode_file"])
        state, mask = _state_from_npz(e)
        out.append(
            Episode(
                run=run_dir.name,
                task_id=int(r["task_id"]),
                init_state_id=int(r["init_state_id"]),
                success=bool(r["success"]),
                n_steps=int(r["n_steps"]),
                state=state,
                mask=mask,
                actions=e["actions"].astype(np.float32),
                eef_pos=e["eef_pos"].astype(np.float32),
                perturb_step=r.get("perturb_step"),
                perturb_kind=r.get("perturb_kind"),
                perturb_mag=float(cfg.get("perturb_mag", 0.0)),
                first_partial_step=r.get("first_partial_step"),
                summary=r,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
def make_inputs(ep: Episode, ts: np.ndarray, k: int) -> np.ndarray:
    """Model inputs for start times ts (each t needs t + k < n_steps)."""
    s = ep.state[ts]
    a = np.stack([ep.actions[t : t + k].reshape(-1) for t in ts])
    onehot = np.zeros((len(ts), N_TASKS), np.float32)
    onehot[:, ep.task_id] = 1.0
    return np.concatenate([s, a, onehot], axis=1)


def make_targets(ep: Episode, ts: np.ndarray, k: int) -> np.ndarray:
    return ep.state[ts + k] - ep.state[ts]


def training_set(episodes: list[Episode], k: int, stride: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, M = [], [], []
    for ep in episodes:
        ts = np.arange(0, ep.n_steps - k, stride)
        if len(ts) == 0:
            continue
        X.append(make_inputs(ep, ts, k))
        Y.append(make_targets(ep, ts, k))
        M.append(np.repeat(ep.mask[None], len(ts), axis=0))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DynamicsModel:
    """Standardised MLP (or ridge) regressor for Δs, plus residual scoring."""

    def __init__(self, kind: str = "mlp", k: int = 10, seed: int = 0):
        self.kind, self.k, self.seed = kind, k, seed

    def fit(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray) -> "DynamicsModel":
        from sklearn.preprocessing import StandardScaler

        self.x_scaler = StandardScaler().fit(X)
        # padded object dims are always 0 -> std 0; StandardScaler handles that (scale=1)
        self.y_mean = np.array([Y[M[:, j], j].mean() if M[:, j].any() else 0.0 for j in range(Y.shape[1])])
        self.y_std = np.array([Y[M[:, j], j].std() + 1e-6 if M[:, j].any() else 1.0 for j in range(Y.shape[1])])
        Xs = self.x_scaler.transform(X)
        Ys = (Y - self.y_mean) / self.y_std
        if self.kind == "mlp":
            from sklearn.neural_network import MLPRegressor

            self.model = MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu", alpha=1e-4,
                                      batch_size=256, learning_rate_init=1e-3, max_iter=60,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                                      random_state=self.seed)
        elif self.kind == "ridge":
            from sklearn.linear_model import Ridge

            self.model = Ridge(alpha=1.0)
        else:
            raise ValueError(self.kind)
        self.model.fit(Xs, Ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.x_scaler.transform(X)) * self.y_std + self.y_mean

    def calibrate(self, episodes: list[Episode]) -> "DynamicsModel":
        """Per-dimension residual scale from held-out successful episodes."""
        X, Y, M = training_set(episodes, self.k, stride=1)
        R = Y - self.predict(X)
        self.res_std = np.array([R[M[:, j], j].std() + 1e-6 if M[:, j].any() else 1.0 for j in range(R.shape[1])])
        return self

    def residual_series(self, ep: Episode) -> np.ndarray:
        """score[t] = mean over real dims of (residual / res_std)^2 for the prediction t -> t+k.

        NaN where t + k >= n_steps. The score for start time t becomes available at time t + k.
        """
        out = np.full(ep.n_steps, np.nan)
        ts = np.arange(0, ep.n_steps - self.k)
        if len(ts) == 0:
            return out
        R = (make_targets(ep, ts, self.k) - self.predict(make_inputs(ep, ts, self.k))) / self.res_std
        out[ts] = np.mean(R[:, ep.mask] ** 2, axis=1)
        return out

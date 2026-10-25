"""Instrumentation for LeRobot's LIBERO environment.

Patches ``lerobot.envs.libero.LiberoEnv`` so that every rollout run through
``lerobot-eval`` additionally records, per simulation step:

* the truth value of every BDDL goal predicate (= sub-goal progress),
* robot state (end-effector pose, gripper, joint positions),
* poses of all movable objects that appear in the goal,
* the executed action,

and can optionally inject a *mid-episode perturbation* right after the first
sub-goal is completed.

Configuration is read from environment variables so that the normal
``lerobot-eval`` command line can be used unchanged:

    VLA_LOG_DIR          output directory for logs (required to enable logging)
    VLA_PERTURB_MODE     none | object | robot | auto   (default: none)
    VLA_PERTURB_MAG      magnitude: metres for object shift, radians (std) for
                         robot joint noise                (default: 0.0)
    VLA_PERTURB_DELAY    steps to wait after the first sub-goal completes
                         before perturbing                 (default: 5)
    VLA_PERTURB_SEED     base seed for perturbation randomness (default: 0)

The patch only works when environments live in the main process, i.e. with
``--eval.batch_size=1`` (LeRobot then uses a SyncVectorEnv).
"""

from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PATCHED = False
_LIVE_ENVS: list[Any] = []


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class InstrumentConfig:
    log_dir: Path | None
    perturb_mode: str = "none"
    perturb_mag: float = 0.0
    perturb_delay: int = 5
    perturb_seed: int = 0

    @classmethod
    def from_env(cls) -> "InstrumentConfig":
        log_dir = os.environ.get("VLA_LOG_DIR")
        mode = os.environ.get("VLA_PERTURB_MODE", "none").strip().lower()
        if mode not in {"none", "object", "robot", "auto"}:
            raise ValueError(f"VLA_PERTURB_MODE must be none|object|robot|auto, got {mode!r}")
        return cls(
            log_dir=Path(log_dir) if log_dir else None,
            perturb_mode=mode,
            perturb_mag=float(os.environ.get("VLA_PERTURB_MAG", "0.0")),
            perturb_delay=int(os.environ.get("VLA_PERTURB_DELAY", "5")),
            perturb_seed=int(os.environ.get("VLA_PERTURB_SEED", "0")),
        )

    @property
    def perturb_enabled(self) -> bool:
        return self.perturb_mode != "none" and self.perturb_mag > 0


# --------------------------------------------------------------------------- #
# Episode recorder
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRecord:
    task_id: int
    task_name: str
    task_description: str
    init_state_id: int
    seed: int | None
    predicates: list[str]
    tracked_objects: list[str]
    initial_preds: list[bool] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    pred_rows: list[list[bool]] = field(default_factory=list)
    eef_pos: list[np.ndarray] = field(default_factory=list)
    eef_quat: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    obj_pose: list[np.ndarray] = field(default_factory=list)  # (n_obj, 7) per step
    actions: list[np.ndarray] = field(default_factory=list)
    success_flags: list[bool] = field(default_factory=list)
    # perturbation bookkeeping
    perturbed: bool = False
    perturb_step: int | None = None
    perturb_target: str | None = None
    perturb_kind: str | None = None
    perturb_vector: list[float] | None = None
    first_partial_step: int | None = None

    def n_steps(self) -> int:
        return len(self.actions)


def _first_true(col: np.ndarray) -> int | None:
    idx = np.flatnonzero(col)
    return int(idx[0]) if idx.size else None


def _format_pred(state: Any) -> str:
    return "(" + " ".join(str(s) for s in state) + ")"


# --------------------------------------------------------------------------- #
# Helpers operating on the inner LIBERO problem env
# --------------------------------------------------------------------------- #
def _inner(libero_env: Any) -> Any:
    """Return the BDDL problem env (has sim, objects_dict, parsed_problem)."""
    return libero_env._env.env


def _goal_states(inner: Any) -> list[Any]:
    return list(inner.parsed_problem["goal_state"])


def _eval_predicates(inner: Any, goal: list[Any]) -> list[bool]:
    return [bool(inner._eval_predicate(s)) for s in goal]


def _movable_goal_objects(inner: Any, goal: list[Any]) -> list[str]:
    names: list[str] = []
    for s in goal:
        for arg in s[1:]:
            if arg in inner.objects_dict and arg not in names:
                names.append(arg)
    return names


def _object_pose(inner: Any, name: str) -> np.ndarray:
    bid = inner.obj_body_id[name]
    pos = np.array(inner.sim.data.body_xpos[bid], dtype=np.float32)
    quat = np.array(inner.sim.data.body_xquat[bid], dtype=np.float32)  # wxyz
    return np.concatenate([pos, quat])


def _free_joint_name(inner: Any, name: str) -> str | None:
    obj = inner.objects_dict.get(name)
    joints = getattr(obj, "joints", None) if obj is not None else None
    return joints[-1] if joints else None


def _apply_object_shift(inner: Any, name: str, rng: np.random.Generator, mag: float) -> list[float] | None:
    joint = _free_joint_name(inner, name)
    if joint is None:
        return None
    qpos = np.array(inner.sim.data.get_joint_qpos(joint), dtype=np.float64)
    theta = rng.uniform(0, 2 * np.pi)
    delta = np.array([mag * np.cos(theta), mag * np.sin(theta), 0.0])
    qpos[:3] = qpos[:3] + delta
    inner.sim.data.set_joint_qpos(joint, qpos)
    try:
        inner.sim.data.set_joint_qvel(joint, np.zeros(6))
    except Exception:  # noqa: BLE001 - qvel layout differs on some objects
        pass
    inner.sim.forward()
    return delta.tolist()


def _apply_robot_noise(inner: Any, rng: np.random.Generator, std: float) -> list[float]:
    robot = inner.robots[0]
    idx = list(robot._ref_joint_pos_indexes)
    vidx = list(robot._ref_joint_vel_indexes)
    delta = rng.normal(0.0, std, size=len(idx))
    inner.sim.data.qpos[idx] = inner.sim.data.qpos[idx] + delta
    inner.sim.data.qvel[vidx] = 0.0
    inner.sim.forward()
    return delta.tolist()


def _refresh_obs(libero_env: Any) -> Any:
    inner = _inner(libero_env)
    try:
        raw = inner._get_observations(force_update=True)
    except TypeError:
        raw = inner._get_observations()
    return libero_env._format_raw_obs(raw)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def _flush(env: Any) -> None:
    rec: EpisodeRecord | None = getattr(env, "_vla_record", None)
    cfg: InstrumentConfig | None = getattr(env, "_vla_cfg", None)
    if rec is None or cfg is None or cfg.log_dir is None or rec.n_steps() == 0:
        env._vla_record = None
        return
    out = cfg.log_dir
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    stem = f"task{rec.task_id:02d}_init{rec.init_state_id:03d}"
    preds = np.array(rec.pred_rows, dtype=bool).reshape(-1, len(rec.predicates))
    obj = (
        np.stack(rec.obj_pose).astype(np.float32)
        if rec.obj_pose and rec.tracked_objects
        else np.zeros((rec.n_steps(), 0, 7), np.float32)
    )
    np.savez_compressed(
        out / "episodes" / f"{stem}.npz",
        predicates=preds,
        eef_pos=np.stack(rec.eef_pos).astype(np.float32),
        eef_quat=np.stack(rec.eef_quat).astype(np.float32),
        gripper_qpos=np.stack(rec.gripper_qpos).astype(np.float32),
        joint_pos=np.stack(rec.joint_pos).astype(np.float32),
        obj_pose=obj,
        actions=np.stack(rec.actions).astype(np.float32),
        success=np.array(rec.success_flags, dtype=bool),
        predicate_names=np.array(rec.predicates),
        tracked_objects=np.array(rec.tracked_objects),
    )
    success = bool(any(rec.success_flags))
    first_true = [_first_true(preds[:, j]) for j in range(preds.shape[1])]
    summary = {
        "task_id": rec.task_id,
        "task_name": rec.task_name,
        "task_description": rec.task_description,
        "init_state_id": rec.init_state_id,
        "seed": rec.seed,
        "n_steps": rec.n_steps(),
        "success": success,
        "success_step": _first_true(np.array(rec.success_flags)) if success else None,
        "predicates": rec.predicates,
        "initially_true": list(rec.initial_preds),
        "first_true_step": first_true,
        "final_true": preds[-1].tolist() if len(preds) else [],
        "max_n_satisfied": int(preds.sum(1).max()) if len(preds) else 0,
        "first_partial_step": rec.first_partial_step,
        "perturb_mode": cfg.perturb_mode,
        "perturb_mag": cfg.perturb_mag,
        "perturb_delay": cfg.perturb_delay,
        "perturbed": rec.perturbed,
        "perturb_step": rec.perturb_step,
        "perturb_target": rec.perturb_target,
        "perturb_kind": rec.perturb_kind,
        "perturb_vector": rec.perturb_vector,
        "wall_time_s": round(time.time() - rec.start_time, 3),
        "episode_file": f"episodes/{stem}.npz",
    }
    with open(out / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    env._vla_record = None


def _flush_all() -> None:
    for env in list(_LIVE_ENVS):
        try:
            _flush(env)
        except Exception as e:  # noqa: BLE001
            print(f"[vlalh] flush failed: {e}")


# --------------------------------------------------------------------------- #
# Recording a step
# --------------------------------------------------------------------------- #
def _record_step(env: Any, action: np.ndarray, success: bool) -> None:
    rec: EpisodeRecord = env._vla_record
    inner = _inner(env)
    goal = env._vla_goal
    rec.pred_rows.append(_eval_predicates(inner, goal))
    raw_robot = inner.robots[0]
    sim = inner.sim
    eef_site = raw_robot.eef_site_id if not isinstance(raw_robot.eef_site_id, dict) else raw_robot.eef_site_id["right"]
    rec.eef_pos.append(np.array(sim.data.site_xpos[eef_site], dtype=np.float32))
    try:
        from robosuite.utils.transform_utils import mat2quat

        rec.eef_quat.append(np.array(mat2quat(sim.data.site_xmat[eef_site].reshape(3, 3)), dtype=np.float32))
    except Exception:  # noqa: BLE001
        rec.eef_quat.append(np.zeros(4, np.float32))
    g_idx = getattr(raw_robot, "_ref_gripper_joint_pos_indexes", None)
    if isinstance(g_idx, dict):
        g_idx = g_idx.get("right")
    rec.gripper_qpos.append(
        np.array(sim.data.qpos[list(g_idx)], dtype=np.float32) if g_idx is not None else np.zeros(2, np.float32)
    )
    rec.joint_pos.append(np.array(sim.data.qpos[list(raw_robot._ref_joint_pos_indexes)], dtype=np.float32))
    if rec.tracked_objects:
        rec.obj_pose.append(np.stack([_object_pose(inner, n) for n in rec.tracked_objects]))
    rec.actions.append(np.asarray(action, dtype=np.float32).copy())
    rec.success_flags.append(bool(success))


def _maybe_perturb(env: Any, obs: Any) -> Any:
    cfg: InstrumentConfig = env._vla_cfg
    rec: EpisodeRecord = env._vla_record
    if not cfg.perturb_enabled or rec.perturbed:
        return obs
    row = rec.pred_rows[-1]
    n_sat, n_tot = sum(row), len(row)
    if rec.first_partial_step is None:
        initially = rec.initial_preds or [False] * n_tot
        # a sub-goal counts as "completed" only if it was not already true at the start
        newly = [r and not i for r, i in zip(row, initially)]
        if any(newly) and n_sat < n_tot:
            rec.first_partial_step = rec.n_steps() - 1
    if rec.first_partial_step is None:
        return obs
    if rec.n_steps() - 1 < rec.first_partial_step + cfg.perturb_delay:
        return obs
    if n_sat == n_tot:  # task already finished, nothing left to perturb
        return obs

    inner = _inner(env)
    rng = np.random.default_rng(cfg.perturb_seed * 100003 + rec.task_id * 1009 + rec.init_state_id)
    remaining = [s for s, ok in zip(env._vla_goal, row) if not ok]
    target_obj = next((a for s in remaining for a in s[1:] if a in inner.objects_dict), None)

    mode = cfg.perturb_mode
    if mode == "auto":
        mode = "object" if target_obj is not None else "robot"
    vec = None
    if mode == "object" and target_obj is not None:
        vec = _apply_object_shift(inner, target_obj, rng, cfg.perturb_mag)
        if vec is not None:
            rec.perturb_target, rec.perturb_kind = target_obj, "object_shift_xy"
    if vec is None and mode in {"robot", "object"}:
        if mode == "object":
            # requested object perturbation but the remaining sub-goal has no movable object
            rec.perturbed = True
            rec.perturb_kind = "skipped_no_movable_target"
            rec.perturb_step = rec.n_steps() - 1
            return obs
        vec = _apply_robot_noise(inner, rng, cfg.perturb_mag)
        rec.perturb_target, rec.perturb_kind = "robot_arm", "joint_noise"
    rec.perturbed = True
    rec.perturb_step = rec.n_steps() - 1
    rec.perturb_vector = vec
    return _refresh_obs(env)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def patch_libero(cfg: InstrumentConfig | None = None) -> InstrumentConfig:
    """Monkey-patch LeRobot's LiberoEnv. Safe to call multiple times."""
    global _PATCHED
    cfg = cfg or InstrumentConfig.from_env()
    if _PATCHED:
        return cfg

    from lerobot.envs import libero as L

    orig_reset = L.LiberoEnv.reset
    orig_step = L.LiberoEnv.step
    orig_close = L.LiberoEnv.close

    def reset(self, seed=None, **kwargs):
        _flush(self)
        init_id = int(self.init_state_id)
        observation, info = orig_reset(self, seed=seed, **kwargs)
        self._vla_cfg = cfg
        if self not in _LIVE_ENVS:
            _LIVE_ENVS.append(self)
        inner = _inner(self)
        goal = _goal_states(inner)
        self._vla_goal = goal
        self._vla_record = EpisodeRecord(
            task_id=int(self.task_id),
            task_name=str(self.task),
            task_description=str(self.task_description),
            init_state_id=init_id,
            seed=seed,
            predicates=[_format_pred(s) for s in goal],
            tracked_objects=_movable_goal_objects(inner, goal),
            initial_preds=_eval_predicates(inner, goal),
        )
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = orig_step(self, action)
        if getattr(self, "_vla_record", None) is not None:
            _record_step(self, action, bool(info.get("is_success", False)))
            observation = _maybe_perturb(self, observation)
            rec = self._vla_record
            info["vla_n_satisfied"] = int(sum(rec.pred_rows[-1]))
            info["vla_perturbed"] = bool(rec.perturbed)
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            _flush(self)
        finally:
            orig_close(self)

    L.LiberoEnv.reset = reset
    L.LiberoEnv.step = step
    L.LiberoEnv.close = close
    atexit.register(_flush_all)
    _PATCHED = True

    if cfg.log_dir is not None:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_dir / "instrument_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "log_dir": str(cfg.log_dir),
                    "perturb_mode": cfg.perturb_mode,
                    "perturb_mag": cfg.perturb_mag,
                    "perturb_delay": cfg.perturb_delay,
                    "perturb_seed": cfg.perturb_seed,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )
    print(
        f"[vlalh] LiberoEnv instrumented | log_dir={cfg.log_dir} | perturb={cfg.perturb_mode} "
        f"mag={cfg.perturb_mag} delay={cfg.perturb_delay}"
    )
    return cfg

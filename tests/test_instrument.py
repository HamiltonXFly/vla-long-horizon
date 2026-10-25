"""Offline test of vlalh.instrument with a fake LeRobot/LIBERO env.

Runs without LeRobot, LIBERO or a GPU:  python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# Fake simulator pieces
# --------------------------------------------------------------------------- #
class FakeData:
    def __init__(self):
        self.qpos = np.zeros(7 + 2 + 7 + 7)  # arm, gripper, obj A free joint, obj B free joint
        self.qvel = np.zeros(7 + 2 + 6 + 6)
        self.site_xpos = np.zeros((1, 3))
        self.site_xmat = np.tile(np.eye(3).reshape(1, 9), (1, 1))
        self._joint_q = {"a_joint0": slice(9, 16), "b_joint0": slice(16, 23)}
        self._joint_v = {"a_joint0": slice(9, 15), "b_joint0": slice(15, 21)}
        self.qpos[9:12] = [0.0, 0.0, 0.0]
        self.qpos[12] = 1.0
        self.qpos[16:19] = [0.5, 0.0, 0.0]
        self.qpos[19] = 1.0

    @property
    def body_xpos(self):
        return np.stack([self.qpos[9:12], self.qpos[16:19]])

    @property
    def body_xquat(self):
        return np.stack([self.qpos[12:16], self.qpos[19:23]])

    def get_joint_qpos(self, name):
        return self.qpos[self._joint_q[name]].copy()

    def set_joint_qpos(self, name, v):
        self.qpos[self._joint_q[name]] = v

    def set_joint_qvel(self, name, v):
        self.qvel[self._joint_v[name]] = v


class FakeSim:
    def __init__(self):
        self.data = FakeData()

    def forward(self):
        pass


class FakeRobot:
    eef_site_id = 0
    _ref_joint_pos_indexes = list(range(7))
    _ref_joint_vel_indexes = list(range(7))
    _ref_gripper_joint_pos_indexes = [7, 8]


class FakeObj:
    def __init__(self, j):
        self.joints = [j]


class FakeInner:
    """Two sub-goals: object A reaches x>=1 ('in basket'), then object B reaches x>=1."""

    def __init__(self):
        self.sim = FakeSim()
        self.robots = [FakeRobot()]
        self.objects_dict = {"obj_a": FakeObj("a_joint0"), "obj_b": FakeObj("b_joint0")}
        self.obj_body_id = {"obj_a": 0, "obj_b": 1}
        self.parsed_problem = {"goal_state": [["in", "obj_a", "basket"], ["in", "obj_b", "basket"]]}

    def _eval_predicate(self, state):
        bid = self.obj_body_id[state[1]]
        return bool(self.sim.data.body_xpos[bid][0] >= 1.0)

    def _check_success(self):
        return all(self._eval_predicate(s) for s in self.parsed_problem["goal_state"])

    def _get_observations(self, force_update=False):
        return {"x": float(self.sim.data.qpos[9])}

    def step(self, action):
        # policy: push A in x-direction first, then B
        d = self.sim.data
        if d.qpos[9] < 1.0:
            d.qpos[9] += 0.125
        else:
            d.qpos[16] += 0.125
        return self._get_observations(), 0.0, False, {}


class FakeControlEnv:
    def __init__(self):
        self.env = FakeInner()

    def check_success(self):
        return self.env._check_success()


class FakeLiberoEnv:
    def __init__(self, task_id=0, episode_index=0):
        self.task_id = task_id
        self.task = "fake_task"
        self.task_description = "push both objects into the basket"
        self.init_state_id = episode_index
        self._env = None

    def _format_raw_obs(self, raw):
        return {"pixels": raw}

    def reset(self, seed=None, **kwargs):
        self._env = FakeControlEnv()
        self.init_state_id += 1
        return {"pixels": {}}, {"is_success": False}

    def step(self, action):
        raw, r, done, info = self._env.env.step(action)
        ok = self._env.check_success()
        info.update({"is_success": ok})
        return self._format_raw_obs(raw), r, done or ok, False, info

    def close(self):
        self._env = None


def _install_fake_lerobot():
    lerobot = types.ModuleType("lerobot")
    envs = types.ModuleType("lerobot.envs")
    libero = types.ModuleType("lerobot.envs.libero")
    libero.LiberoEnv = FakeLiberoEnv
    envs.libero = libero
    lerobot.envs = envs
    sys.modules.update({"lerobot": lerobot, "lerobot.envs": envs, "lerobot.envs.libero": libero})


def _run_episode(env, max_steps=40):
    env.reset(seed=0)
    for _ in range(max_steps):
        _, _, term, _, _ = env.step(np.zeros(7, np.float32))
        if term:
            break


# --------------------------------------------------------------------------- #
def test_logging_and_perturbation(tmp_path):
    _install_fake_lerobot()
    import importlib

    import vlalh.instrument as ins

    importlib.reload(ins)
    cfg = ins.InstrumentConfig(log_dir=tmp_path, perturb_mode="auto", perturb_mag=0.3,
                               perturb_delay=2, perturb_seed=1)
    ins.patch_libero(cfg)

    env = FakeLiberoEnv()
    _run_episode(env)  # episode 0
    _run_episode(env)  # episode 1 (flushes 0)
    env.close()        # flushes 1

    rows = [json.loads(l) for l in (tmp_path / "summary.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    r = rows[0]
    assert r["predicates"] == ["(in obj_a basket)", "(in obj_b basket)"]
    assert r["initially_true"] == [False, False]
    assert r["first_true_step"][0] == 7            # A crosses x>=1 after 8 pushes
    assert r["perturbed"] and r["perturb_kind"] == "object_shift_xy"
    assert r["perturb_target"] == "obj_b"
    assert r["perturb_step"] == 7 + 2
    assert abs(np.linalg.norm(r["perturb_vector"][:2]) - 0.3) < 1e-6
    ep = np.load(tmp_path / r["episode_file"])
    assert ep["predicates"].shape == (r["n_steps"], 2)
    assert ep["obj_pose"].shape == (r["n_steps"], 2, 7)
    assert ep["joint_pos"].shape == (r["n_steps"], 7)
    assert rows[1]["init_state_id"] == 1

    # the analysis script runs on the produced logs
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import analyze_stages as an

    df = an.enrich(an.load_run(tmp_path))
    assert set(df["stage"]) <= {"none", "partial", "complete"}
    pt, pr = an.per_task(df), an.per_run(df)
    assert pr["n_perturbed"].iloc[0] == 2
    out = tmp_path / "report"
    out.mkdir()
    an.fig_task_success(pt, tmp_path.name, out)
    an.fig_stage_outcome(pt, tmp_path.name, out)
    assert (out / "fig_stage_outcome.png").exists()


def test_wilson():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import analyze_stages as an

    lo, hi = an.wilson(3, 3)
    assert abs(lo - 0.4385) < 1e-3 and hi == 1.0  # matches LeRobot's reported CI for 3/3

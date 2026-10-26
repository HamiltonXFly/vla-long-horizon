#!/usr/bin/env python
"""Re-render logged episodes as GIFs by replaying their recorded actions.

MuJoCo is deterministic, so the same initial state + the same action sequence (+ the same
perturbation at the same step) reproduces the episode without running the policy again.
The script prints the max end-effector deviation from the log as a check.

    MUJOCO_GL=egl python scripts/render_episodes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

OUT = Path("docs/behaviors")
STEP_STRIDE = 4          # keep every 4th control step (20 Hz -> 5 frames per simulated second)
FPS = 10                 # played at 2x real time
SIZE = 256

MAX_DEV_MM = 10.0        # accept a replay only if the gripper stays within 1 cm of the log
                         # and the outcome (success / failure) is the same

B, P = "outputs/baseline_seed1000", "outputs/perturb_auto_mag0.10_delay5_seed1000"
# (file name, run dir, task id, candidate init states in order of preference, caption)
EPISODES = [
    ("success", B, 8, [0], "Success"),
    ("stall", B, 8, [3], "Stops moving after step 1"),
    ("carried_not_placed", B, 8, [6, 39], "Lifts the 2nd pot, never places it"),
    ("repeated_grasp", B, 8, [47], "Grasps again and again"),
    ("regression", B, 8, [27, 23], "Undoes step 1"),
    ("no_subgoal", B, 0, [49], "Never completes step 1"),
    ("object_shift_recovered", P, 0, [0], "Object moved 10 cm: recovers"),
    ("object_shift_failed", P, 0, [3], "Object moved 10 cm: fails"),
    ("arm_push_recovered", P, 9, [0], "Arm pushed: recovers"),
    ("arm_push_failed", P, 9, [2, 4, 5, 10, 11, 12], "Arm pushed: fails"),
]


def summary_row(run: str, task: int, init: int) -> dict:
    for line in Path(run, "summary.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["task_id"] == task and r["init_state_id"] == init:
            return r
    raise KeyError((run, task, init))


def apply_perturbation(inner, r: dict) -> None:
    vec = np.asarray(r["perturb_vector"], dtype=np.float64)
    if r["perturb_kind"] == "object_shift_xy":
        joint = inner.objects_dict[r["perturb_target"]].joints[-1]
        qpos = np.array(inner.sim.data.get_joint_qpos(joint), dtype=np.float64)
        qpos[:3] += vec
        inner.sim.data.set_joint_qpos(joint, qpos)
        try:
            inner.sim.data.set_joint_qvel(joint, np.zeros(6))
        except Exception:  # noqa: BLE001
            pass
    elif r["perturb_kind"] == "joint_noise":
        robot = inner.robots[0]
        idx, vidx = list(robot._ref_joint_pos_indexes), list(robot._ref_joint_vel_indexes)
        inner.sim.data.qpos[idx] = inner.sim.data.qpos[idx] + vec
        inner.sim.data.qvel[vidx] = 0.0
    inner.sim.forward()


def eef(inner) -> np.ndarray:
    robot = inner.robots[0]
    site = robot.eef_site_id if not isinstance(robot.eef_site_id, dict) else robot.eef_site_id["right"]
    return np.array(inner.sim.data.site_xpos[site])


def replay(run: str, task: int, init: int):
    r = summary_row(run, task, init)
    ep = np.load(Path(run) / r["episode_file"])
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env = LiberoEnv(task_suite=suite, task_id=task, task_suite_name="libero_10",
                    camera_name="agentview_image", observation_width=SIZE, observation_height=SIZE,
                    init_states=True, episode_index=init)
    obs, _ = env.reset(seed=r["seed"])
    inner = env._env.env
    frames, dev, first_div, ok = [], 0.0, None, False
    for i, a in enumerate(ep["actions"]):
        obs, _, _, _, info = env.step(np.asarray(a, dtype=np.float64))
        ok = ok or bool(info.get("is_success", False))
        d = float(np.linalg.norm(eef(inner) - ep["eef_pos"][i]))
        if d * 1000 > 1.0 and first_div is None:
            first_div = i
        dev = max(dev, d)
        if r.get("perturbed") and r.get("perturb_step") == i and r.get("perturb_vector") is not None:
            apply_perturbation(inner, r)
            obs = env._format_raw_obs(inner._get_observations())
        if i % STEP_STRIDE == 0 or i == len(ep["actions"]) - 1:
            img = next(iter(obs["pixels"].values()))
            frames.append(np.ascontiguousarray(img[::-1, ::-1]))
    env.close()
    return frames, r, dev, r.get("perturb_step") if r.get("perturbed") else None, ok, first_div


def to_gif(frames, r, caption: str, perturb_step: int | None, path: Path) -> None:
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    out = []
    for k, f in enumerate(frames):
        step = k * STEP_STRIDE
        im = Image.fromarray(f).convert("RGB")
        canvas = Image.new("RGB", (SIZE, SIZE + 44), "white")
        canvas.paste(im, (0, 44))
        d = ImageDraw.Draw(canvas)
        d.text((6, 4), caption, fill="#0b0b0b", font=font)
        d.text((6, 23), f"t = {step / 20:4.1f} s", fill="#52514e", font=font)
        if perturb_step is not None and 0 <= step - perturb_step < 30:
            d.rectangle([0, 44, SIZE - 1, SIZE + 43], outline="#e34948", width=4)
            d.text((SIZE - 92, 23), "disturbance", fill="#e34948", font=font)
        out.append(canvas)
    out += [out[-1]] * FPS                       # hold the last frame for 1 s
    out[0].save(path, save_all=True, append_images=out[1:], duration=1000 // FPS, loop=0, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for name, run, task, inits, caption in EPISODES:
        for init in inits:
            frames, r, dev, pstep, ok, first_div = replay(run, task, init)
            exact = dev * 1000 <= MAX_DEV_MM and ok == r["success"]
            print(f"{name:24s} task {task} init {init:2d}  steps {r['n_steps']:4d}  logged success {r['success']!s:5s}  "
                  f"replay success {ok!s:5s}  max dev {dev * 1000:7.2f} mm  first >1mm at step {first_div}")
            if exact:
                to_gif(frames, r, caption, pstep, OUT / f"{name}.gif")
                chosen[name] = {"run": run, "task_id": task, "init_state_id": init, "max_dev_mm": round(dev * 1000, 3)}
                break
        else:
            print(f"  -> no exact replay for {name}, skipped")
    (OUT / "episodes.json").write_text(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()

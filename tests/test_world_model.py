"""Offline test of vlalh.world_model on synthetic episodes (no simulator needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlalh.world_model import STATE_DIM, DynamicsModel, Episode, training_set  # noqa: E402


def _episode(rng, T=120, broken=False):
    """State integrates actions (s_{t+1} = s_t + 0.01 a_t on the eef dims); 'broken' ignores them."""
    actions = rng.uniform(-1, 1, size=(T, 7)).astype(np.float32)
    state = np.zeros((T, STATE_DIM), np.float32)
    for t in range(1, T):
        state[t] = state[t - 1]
        if not broken:
            state[t, :3] += 0.01 * actions[t - 1, :3]
        else:
            state[t, :3] += 0.01 * rng.uniform(-1, 1, 3)
    mask = np.zeros(STATE_DIM, bool)
    mask[:12] = True
    return Episode(run="synthetic", task_id=0, init_state_id=0, success=not broken, n_steps=T, state=state,
                   mask=mask, actions=actions, eef_pos=state[:, :3], perturb_step=None, perturb_kind=None,
                   perturb_mag=0.0, first_partial_step=None, summary={})


def test_residual_separates_normal_from_broken():
    rng = np.random.default_rng(0)
    train = [_episode(rng) for _ in range(20)]
    calib = [_episode(rng) for _ in range(5)]
    X, Y, M = training_set(train, k=5, stride=1)
    assert X.shape[1] == STATE_DIM + 5 * 7 + 10
    model = DynamicsModel("ridge", k=5).fit(X, Y, M).calibrate(calib)
    normal = np.nanmean(model.residual_series(_episode(rng)))
    broken = np.nanmean(model.residual_series(_episode(rng, broken=True)))
    assert broken > 5 * normal
    r = model.residual_series(_episode(rng, T=30))
    assert np.isnan(r[-5:]).all() and np.isfinite(r[:-5]).all()

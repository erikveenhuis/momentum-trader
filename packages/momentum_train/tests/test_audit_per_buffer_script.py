"""Smoke tests for ``scripts/audit_per_buffer.py`` (Tier 5b).

Exercises the pure helpers:

* ``_compute_buffer_arrays`` — extracts rewards/actions/priorities/age from a
  fake buffer state mirroring the on-disk layout.
* ``_build_report`` — collapses arrays into the JSON-serializable report.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_per_buffer.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def audit_mod():
    spec = importlib.util.spec_from_file_location("audit_per_buffer", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_per_buffer"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_buffer_state(rewards, actions, priorities):
    """Mimic the buffer state_dict produced by PrioritizedReplayBuffer.state_dict."""
    n = len(rewards)
    capacity = max(n, 4)
    tree = np.zeros(2 * capacity - 1, dtype=np.float64)
    for i, p in enumerate(priorities):
        tree[i + capacity - 1] = float(p)
    experiences = [SimpleNamespace(reward=float(r), action=int(a)) for r, a in zip(rewards, actions)]
    return {
        "buffer": experiences,
        "tree_state": {"tree": tree, "data_indices": np.arange(capacity), "write": n % capacity, "size": n},
        "buffer_write_idx": n % capacity,
        "max_priority": float(max(priorities)) if priorities else 0.0,
        "capacity": capacity,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_start": 0.4,
        "beta_frames": 100000,
        "epsilon": 1e-5,
    }


@pytest.mark.unit
def test_compute_buffer_arrays_extracts_rewards_actions_priorities(audit_mod):
    """Tier 5b: arrays line up by buffer index with the lockstep SumTree leaves."""
    rewards = [0.1, -0.2, 0.3, -0.4, 0.5]
    actions = [0, 1, 2, 0, 5]
    priorities = [0.5, 0.6, 0.7, 0.8, 0.9]
    state = _make_buffer_state(rewards, actions, priorities)
    out = audit_mod._compute_buffer_arrays(state)
    np.testing.assert_array_almost_equal(out["rewards"], rewards)
    np.testing.assert_array_equal(out["actions"], actions)
    np.testing.assert_array_almost_equal(out["priorities"], priorities)
    assert out["size"] == 5
    assert out["capacity"] == 5
    assert out["ages"].min() >= 1


@pytest.mark.unit
def test_build_report_per_action_share_and_top_n(audit_mod):
    """Tier 5b: by-action shares + top-N records reflect the input distribution."""
    rewards = [0.0, 1.0, 2.0, 3.0, 100.0]
    actions = [0, 0, 5, 5, 5]
    priorities = [0.1, 0.1, 0.5, 0.5, 50.0]  # action 5 dominates priority.
    state = _make_buffer_state(rewards, actions, priorities)
    arrays = audit_mod._compute_buffer_arrays(state)
    report = audit_mod._build_report(arrays, reward_bins=10, top_n=3)

    assert report["buffer_meta"]["size"] == 5
    assert report["buffer_meta"]["fill_ratio"] == pytest.approx(1.0)

    by_action = report["by_action"]
    assert set(by_action) == {0, 5}
    assert by_action[0]["count"] == 2
    assert by_action[5]["count"] == 3
    # Priority share for action 5: (0.5+0.5+50.0) / total. Action 5 must dominate.
    assert by_action[5]["priority_share"] > by_action[0]["priority_share"] * 5
    # Top-3 should include the 50.0-priority sample as rank 1.
    top = report["top_n"]
    assert len(top) == 3
    assert top[0]["priority"] == pytest.approx(50.0)
    assert top[0]["action"] == 5
    assert top[0]["reward"] == pytest.approx(100.0)


@pytest.mark.unit
def test_build_report_handles_empty_buffer(audit_mod):
    """Tier 5b: empty buffer produces a structurally valid (NaN-laden) report."""
    state = _make_buffer_state([], [], [])
    arrays = audit_mod._compute_buffer_arrays(state)
    report = audit_mod._build_report(arrays, reward_bins=5, top_n=10)
    assert report["buffer_meta"]["size"] == 0
    assert report["buffer_meta"]["fifo_half_life_steps"] == 0.0
    assert report["reward"]["count"] == 0
    assert report["top_n"] == []


@pytest.mark.unit
def test_extract_buffer_state_raises_on_missing_payload(audit_mod):
    """Tier 5b: missing buffer_state in checkpoint surfaces a clear ValueError."""
    with pytest.raises(ValueError, match="does not contain a buffer_state"):
        audit_mod._extract_buffer_state({})


@pytest.mark.unit
def test_extract_buffer_state_raises_on_missing_keys(audit_mod):
    """Tier 5b: partial buffer_state is rejected (missing tree_state)."""
    bad = {"buffer_state": {"buffer": [], "buffer_write_idx": 0, "max_priority": 1.0, "capacity": 10}}
    with pytest.raises(ValueError, match="missing required key"):
        audit_mod._extract_buffer_state(bad)

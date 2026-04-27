"""Tests for the BTR Stage 0 migration script (C51 -> IQN warm-start).

The migration script is the *only* path the user has to preserve any of
the encoder training that happened under C51, so its invariants are
load-bearing:

* Encoder, account-processor, head-norm, and aux-head tensors transfer
  bit-exactly from source to target.
* IQN-specific tensors (``tau_embedding.*``, ``value_stream.*``,
  ``advantage_stream.*``) come from the fresh agent, *not* from the
  source dict, even when they coincidentally share names.
* Optimizer / scheduler / scaler state and validation-resume metadata are
  reset so the post-warmstart trainer starts from a clean head.
* Replay-buffer side-cars are copied + their PER priorities flattened
  to ``priority_max`` so initial sampling is uniform.
* ``--dry-run`` writes nothing.
* Already-IQN sources are rejected up-front (idempotency guard).
* The migrated checkpoint loads cleanly through the live agent
  (no ``RuntimeError`` from the Stage 1 pre-IQN guard).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

# Load the migration script as a top-level module ("scripts" is not a
# Python package and we don't want to add it to sys.path globally).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "migrate_c51_to_iqn.py"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location("migrate_c51_to_iqn", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["migrate_c51_to_iqn"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


migration = _load_migration_module()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _minimal_iqn_yaml(tmp_path: Path) -> Path:
    """Write a minimal IQN-shaped training YAML that ``RainbowDQNAgent`` can
    consume in ``inference_only=True`` mode."""
    config = {
        "agent": {
            "seed": 0,
            "gamma": 0.99,
            "lr": 1e-4,
            "batch_size": 8,
            "replay_buffer_size": 64,
            "target_update_freq": 100,
            "window_size": 6,
            "n_features": 12,
            "hidden_dim": 16,
            "num_actions": 4,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "transformer_dropout": 0.0,
            "n_steps": 3,
            "n_quantiles_online": 8,
            "n_quantiles_target": 8,
            "n_quantiles_policy": 8,
            "quantile_embedding_dim": 8,
            "huber_kappa": 1.0,
            "quantile_logging_interval": 100,
            "quantile_logging_percentiles": [5, 50, 95],
            "munchausen_alpha": 0.9,
            "munchausen_entropy_tau": 0.03,
            "munchausen_log_pi_clip": -1.0,
            "spectral_norm_enabled": False,
            "noisy_sigma_logging_interval": 100,
            "q_value_logging_interval": 100,
            "q_value_histogram_interval": 100,
            "grad_logging_interval": 100,
            "target_net_logging_interval": 100,
            "td_error_logging_interval": 100,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 100,
            "lr_scheduler_enabled": False,
            "lr_scheduler_type": "CosineAnnealingLR",
            "lr_scheduler_params": {"T_max": 1000, "eta_min": 1e-5},
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "entropy_coeff": 0.0,
            "grad_clip_norm": 1.0,
            "polyak_tau": 0.001,
            "store_partial_n_step": True,
            "debug": False,
            "aux_loss_weight": 0.1,
            "aux_target_feature_index": 6,
        },
        "environment": {
            "window_size": 6,
            "n_features": 12,
            "num_actions": 4,
        },
    }
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _make_synthetic_pre_iqn_state_dict(
    *,
    window_size: int = 6,
    n_features: int = 12,
    hidden_dim: int = 16,
    num_actions: int = 4,
    num_atoms: int = 51,
) -> dict[str, torch.Tensor]:
    """Fabricate a state dict shaped like the *old* C51 RainbowNetwork.

    The shapes only need to be plausible for the *transferred* tensors
    (encoder + aux + account_processor + head_norm). For tensors we don't
    transfer (value_stream / advantage_stream / support / delta_z), the
    shapes are irrelevant — they get dropped.
    """
    shared_dim = hidden_dim + hidden_dim // 4
    head_hidden = hidden_dim // 2
    sd: dict[str, torch.Tensor] = {
        "feature_embedding.weight": torch.randn(hidden_dim, n_features),
        "feature_embedding.bias": torch.randn(hidden_dim),
        "cls_token": torch.randn(1, 1, hidden_dim),
        # PositionalEncoding stores ``pe`` as ``[max_len, 1, d_model]``.
        "pos_encoder.pe": torch.randn(window_size + 1, 1, hidden_dim),
        # One transformer-encoder layer: include the canonical sub-tensors.
        "transformer_encoder.layers.0.self_attn.in_proj_weight": torch.randn(3 * hidden_dim, hidden_dim),
        "transformer_encoder.layers.0.self_attn.in_proj_bias": torch.randn(3 * hidden_dim),
        "transformer_encoder.layers.0.self_attn.out_proj.weight": torch.randn(hidden_dim, hidden_dim),
        "transformer_encoder.layers.0.self_attn.out_proj.bias": torch.randn(hidden_dim),
        "transformer_encoder.layers.0.linear1.weight": torch.randn(32, hidden_dim),
        "transformer_encoder.layers.0.linear1.bias": torch.randn(32),
        "transformer_encoder.layers.0.linear2.weight": torch.randn(hidden_dim, 32),
        "transformer_encoder.layers.0.linear2.bias": torch.randn(hidden_dim),
        "transformer_encoder.layers.0.norm1.weight": torch.randn(hidden_dim),
        "transformer_encoder.layers.0.norm1.bias": torch.randn(hidden_dim),
        "transformer_encoder.layers.0.norm2.weight": torch.randn(hidden_dim),
        "transformer_encoder.layers.0.norm2.bias": torch.randn(hidden_dim),
        # Account processor (Sequential([Linear, GELU, Dropout])).
        "account_processor.0.weight": torch.randn(hidden_dim // 4, 5),
        "account_processor.0.bias": torch.randn(hidden_dim // 4),
        "head_norm.weight": torch.randn(shared_dim),
        "head_norm.bias": torch.randn(shared_dim),
        # Aux head (Sequential([Linear, GELU, Linear])).
        "aux_return_head.0.weight": torch.randn(head_hidden, hidden_dim),
        "aux_return_head.0.bias": torch.randn(head_hidden),
        "aux_return_head.2.weight": torch.randn(1, head_hidden),
        "aux_return_head.2.bias": torch.randn(1),
        # C51-specific: dueling heads with last dim = num_atoms. These will
        # be DROPPED, not transferred.
        "value_stream.0.weight_mu": torch.randn(head_hidden, shared_dim),
        "value_stream.0.bias_mu": torch.randn(head_hidden),
        "value_stream.0.weight_sigma": torch.randn(head_hidden, shared_dim),
        "value_stream.0.bias_sigma": torch.randn(head_hidden),
        "value_stream.0.weight_epsilon": torch.randn(head_hidden, shared_dim),
        "value_stream.0.bias_epsilon": torch.randn(head_hidden),
        "value_stream.2.weight_mu": torch.randn(num_atoms, head_hidden),
        "value_stream.2.bias_mu": torch.randn(num_atoms),
        "advantage_stream.2.weight_mu": torch.randn(num_actions * num_atoms, head_hidden),
        "advantage_stream.2.bias_mu": torch.randn(num_actions * num_atoms),
        # C51 buffers we should drop entirely.
        "support": torch.linspace(-1.0, 1.0, num_atoms),
        "delta_z": torch.tensor(2.0 / (num_atoms - 1)),
    }
    return sd


def _make_pre_iqn_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "episode": 6803,
        "total_train_steps": 5_700_000,
        "best_validation_metric": 0.4479,
        "early_stopping_counter": 6,
        "buffer_state": None,
        "buffer_sidecar_relpath": None,
        "agent_config": {"this_is_legacy_c51": True},
        "agent_total_steps": 5_700_000,
        "agent_env_steps": 11_400_000,
        "total_steps": 5_700_000,
        "network_state_dict": state_dict,
        "target_network_state_dict": state_dict,
        # The old checkpoint has these — we must verify the migrator drops them.
        "optimizer_state_dict": {"this_is_a_stale_optimizer_state": True},
        "scheduler_state_dict": {"stale": True},
        "scaler_state_dict": {"stale": True},
        "validation_score": 0.4479,
    }


def _write_synthetic_buffer_sidecar(
    buffer_dir: Path,
    *,
    capacity: int,
    size: int,
    max_priority: float,
) -> None:
    """Write a minimal but valid PER side-car layout (matching
    ``PrioritizedReplayBuffer.save_to_path``)."""
    buffer_dir.mkdir(parents=True, exist_ok=False)
    tree = np.zeros(2 * capacity - 1)
    # Set all leaves to a *different* priority than max_priority so that
    # the migrator's reset is observable. Internal sums are arbitrary
    # before the migrator runs.
    leaf_start = capacity - 1
    if size > 0:
        tree[leaf_start : leaf_start + size] = 0.123
    for i in range(leaf_start - 1, -1, -1):
        tree[i] = tree[2 * i + 1] + tree[2 * i + 2]
    np.savez(
        buffer_dir / "sumtree.npz",
        tree=tree,
        data_indices=np.zeros(capacity, dtype=np.int64),
        write=np.int64(size % capacity),
        size=np.int64(size),
    )
    meta = {
        "format_version": 1,
        "size": size,
        "buffer_write_idx": size % capacity,
        "max_priority": max_priority,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_start": 0.4,
        "beta_frames": 100,
        "epsilon": 1e-5,
        "capacity": capacity,
    }
    (buffer_dir / "meta.json").write_text(json.dumps(meta))
    (buffer_dir / "_COMPLETE").write_bytes(b"")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_migrate_round_trip_loads_into_iqn_agent(tmp_path):
    """End-to-end: synthetic pre-IQN .pt -> migration -> the resulting
    network state dict loads cleanly into a freshly-built IQN agent."""
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    summary = migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=None,
        dry_run=False,
    )

    out_pt = Path(summary["output_pt"])
    assert out_pt.exists()
    blob = torch.load(out_pt, map_location="cpu", weights_only=False)
    assert "network_state_dict" in blob
    assert "tau_embedding.linear.weight" in blob["network_state_dict"], (
        "Migrated state dict should have the IQN tau_embedding param."
    )

    # Loading into a real IQN agent shouldn't trip the Stage 1 pre-IQN guard.
    from momentum_agent.agent import RainbowDQNAgent

    yaml_blob = yaml.safe_load(yaml_path.read_text())
    agent = RainbowDQNAgent(
        config={**yaml_blob["agent"], **yaml_blob["environment"]},
        device="cpu",
        inference_only=True,
    )
    agent.network.load_state_dict(blob["network_state_dict"])


@pytest.mark.unit
def test_migrate_preserves_encoder_tensor_identity(tmp_path):
    """Every transferred encoder/aux tensor must equal the source bit-for-bit."""
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src_state = _make_synthetic_pre_iqn_state_dict()
    src = _make_pre_iqn_checkpoint(src_state)
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=None,
        dry_run=False,
    )

    out_pt = output_stem.with_suffix(".pt")
    blob = torch.load(out_pt, map_location="cpu", weights_only=False)
    new_state = blob["network_state_dict"]

    transferred_keys = [
        "feature_embedding.weight",
        "feature_embedding.bias",
        "cls_token",
        "pos_encoder.pe",
        "account_processor.0.weight",
        "account_processor.0.bias",
        "head_norm.weight",
        "head_norm.bias",
        "aux_return_head.0.weight",
        "aux_return_head.0.bias",
        "aux_return_head.2.weight",
        "aux_return_head.2.bias",
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
        "transformer_encoder.layers.0.self_attn.out_proj.weight",
        "transformer_encoder.layers.0.linear1.weight",
        "transformer_encoder.layers.0.linear2.weight",
        "transformer_encoder.layers.0.norm1.weight",
    ]
    for key in transferred_keys:
        assert key in new_state, f"Migrated state missing key: {key}"
        assert torch.allclose(new_state[key], src_state[key], atol=0.0), (
            f"Transferred tensor {key} does not equal source bit-for-bit."
        )


@pytest.mark.unit
def test_migrate_resets_dueling_heads(tmp_path):
    """The IQN dueling-head tensors come from the *fresh* agent, not the
    source. We assert this by inserting a sentinel pattern in the source's
    value_stream and verifying it does *not* appear in the migrated state."""
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src_state = _make_synthetic_pre_iqn_state_dict()
    # Sentinel: fill C51's first value_stream weight with a constant 7.0; it
    # must not survive the migration (shapes differ anyway, but the script's
    # whitelist is what we care about here).
    src_state["value_stream.0.weight_mu"] = torch.full_like(src_state["value_stream.0.weight_mu"], 7.0)
    src = _make_pre_iqn_checkpoint(src_state)
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=None,
        dry_run=False,
    )
    blob = torch.load(output_stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
    new_state = blob["network_state_dict"]

    assert "value_stream.0.weight_mu" in new_state
    new_value = new_state["value_stream.0.weight_mu"]
    assert not torch.allclose(new_value, torch.full_like(new_value, 7.0)), (
        "value_stream.0.weight_mu picked up the C51 sentinel — fresh init was bypassed."
    )


@pytest.mark.unit
def test_migrate_strips_optimizer_state(tmp_path):
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=None,
        dry_run=False,
    )
    blob = torch.load(output_stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
    # Keys are omitted entirely (not written as None) so that the agent's
    # ``"key in dict"`` guards in ``load_state`` short-circuit cleanly without
    # ever calling ``optimizer.load_state_dict(None)``.
    assert "optimizer_state_dict" not in blob
    assert "scheduler_state_dict" not in blob
    assert "scaler_state_dict" not in blob


@pytest.mark.unit
def test_migrate_resets_validation_metadata(tmp_path):
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=None,
        dry_run=False,
    )
    blob = torch.load(output_stem.with_suffix(".pt"), map_location="cpu", weights_only=False)
    assert blob["best_validation_metric"] == float("-inf")
    assert blob["early_stopping_counter"] == 0
    assert blob.get("_iqn_warmstart_from") == "src.pt"


@pytest.mark.unit
def test_migrate_buffer_priority_reset(tmp_path):
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    capacity = 64
    size = 40
    max_priority = 0.85
    buffer_src = tmp_path / "src.buffer"
    _write_synthetic_buffer_sidecar(buffer_src, capacity=capacity, size=size, max_priority=max_priority)

    output_stem = tmp_path / "warmstart"
    summary = migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=buffer_src,
        dry_run=False,
    )

    buf_out = Path(summary["output_buffer"])
    assert buf_out.exists()
    npz = np.load(buf_out / "sumtree.npz")
    tree = npz["tree"]
    leaves = tree[capacity - 1 : capacity - 1 + size]
    assert np.allclose(leaves, max_priority), f"All populated leaves should be {max_priority}, got {leaves[:5]}..."
    unused = tree[capacity - 1 + size : 2 * capacity - 1]
    assert np.allclose(unused, 0.0)
    # Root sum should equal size * max_priority by linearity of the segment tree.
    assert tree[0] == pytest.approx(size * max_priority, rel=1e-6, abs=1e-9)
    assert summary["buffer_priority_reset_count"] == size
    assert summary["buffer_priority_value"] == pytest.approx(max_priority)


@pytest.mark.unit
def test_migrate_dry_run_writes_no_files(tmp_path):
    yaml_path = _minimal_iqn_yaml(tmp_path)
    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src_path = tmp_path / "src.pt"
    torch.save(src, src_path)

    capacity = 16
    size = 8
    buffer_src = tmp_path / "src.buffer"
    _write_synthetic_buffer_sidecar(buffer_src, capacity=capacity, size=size, max_priority=0.5)

    output_stem = tmp_path / "warmstart"
    summary = migration.migrate_checkpoint(
        source_path=src_path,
        output_stem=output_stem,
        yaml_config_path=yaml_path,
        buffer_source=buffer_src,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert not output_stem.with_suffix(".pt").exists()
    assert not output_stem.with_suffix(".buffer").exists()


@pytest.mark.unit
def test_migrate_rejects_iqn_source(tmp_path):
    """Running the migration on an already-IQN checkpoint must fail loudly
    so we never accidentally re-migrate a freshly-warmstarted file."""
    yaml_path = _minimal_iqn_yaml(tmp_path)
    # Load the YAML and instantiate a real IQN agent to harvest its
    # ``state_dict`` — that has the ``tau_embedding.*`` markers.
    yaml_blob = yaml.safe_load(yaml_path.read_text())
    from momentum_agent.agent import RainbowDQNAgent

    agent = RainbowDQNAgent(
        config={**yaml_blob["agent"], **yaml_blob["environment"]},
        device="cpu",
        inference_only=True,
    )
    iqn_state = {k: v.clone() for k, v in agent.network.state_dict().items()}

    src = _make_pre_iqn_checkpoint(_make_synthetic_pre_iqn_state_dict())
    src["network_state_dict"] = iqn_state  # already IQN
    src["target_network_state_dict"] = iqn_state
    src_path = tmp_path / "src_iqn.pt"
    torch.save(src, src_path)

    output_stem = tmp_path / "warmstart"
    with pytest.raises(RuntimeError, match="already looks like an IQN"):
        migration.migrate_checkpoint(
            source_path=src_path,
            output_stem=output_stem,
            yaml_config_path=yaml_path,
            buffer_source=None,
            dry_run=False,
        )

"""Tests for ``momentum_train.config_schema`` — strict trainer/run/logging schemas.

Contract we care about:
1. A complete config from ``config/training_config.yaml`` parses successfully.
2. A partial dict raises ``KeyError`` naming the missing field(s).
3. Range validation on ``TrainerConfig`` rejects obviously-bad values.

See ``.cursor/rules/no-defaults.mdc``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from momentum_agent.config_schema import AgentConfig
from momentum_train.config_schema import (
    LoggingConfig,
    RunConfig,
    TrainerConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = PROJECT_ROOT / "config" / "training_config.yaml"


@pytest.fixture(scope="module")
def yaml_config() -> dict:
    return yaml.safe_load(YAML_PATH.read_text())


def test_live_yaml_parses_trainer_run_logging_agent(yaml_config):
    """The checked-in ``training_config.yaml`` must satisfy every schema."""

    trainer_cfg = TrainerConfig.from_dict(yaml_config["trainer"])
    run_cfg = RunConfig.from_dict(yaml_config["run"])
    logging_cfg = LoggingConfig.from_dict(yaml_config["logging"])

    # Inject seed like run_training.py does before building the agent.
    agent_raw = dict(yaml_config["agent"])
    agent_raw["seed"] = trainer_cfg.seed
    agent_cfg = AgentConfig.from_dict(agent_raw)

    assert trainer_cfg.seed == yaml_config["trainer"]["seed"]
    assert run_cfg.episodes == yaml_config["run"]["episodes"]
    assert logging_cfg.log_filename == yaml_config["logging"]["log_filename"]
    assert agent_cfg.gamma == yaml_config["agent"]["gamma"]
    assert run_cfg.archive_model_dir_before_fresh_start is yaml_config["run"]["archive_model_dir_before_fresh_start"]
    assert run_cfg.archive_tensorboard_runs is yaml_config["run"]["archive_tensorboard_runs"]


def test_run_config_archive_defaults_when_keys_omitted():
    """Optional archive toggles default sanely when absent from YAML-shaped dict."""
    raw = {
        "mode": "train",
        "episodes": 1,
        "model_dir": "models",
        "resume": False,
        "skip_evaluation": True,
    }
    cfg = RunConfig.from_dict(raw)
    assert cfg.archive_model_dir_before_fresh_start is True
    assert cfg.archive_tensorboard_runs is False


def test_trainer_config_missing_key_raises_keyerror():
    """Drop one field and make sure ``from_dict`` names it explicitly."""
    raw = {
        "seed": 1,
        "num_vector_envs": 1,
        "warmup_steps": 10,
        "update_freq": 1,
        "gradient_updates_per_step": 1,
        "log_freq": 10,
        "per_stats_log_freq": 0,
        "validation_freq": 10,
        "checkpoint_save_freq": 100,
        "latest_checkpoint_keep_last_n": 0,
        "reward_window": 5,
        "early_stopping_patience": 10,
        "min_episodes_before_early_stopping": 0,
        "min_episodes_before_checkpoint_pinning": 0,
        "min_validation_threshold": 0.0,
        "benchmark_allocation_frac_start": 0.5,
        "benchmark_allocation_frac_end": 0.1,
        "benchmark_allocation_frac_anneal_episodes": 100,
        "final_phase_lr_start_frac": 0.8,
        "final_phase_lr_multiplier": 0.3,
        "curriculum_mode": "flat",
        "curriculum_start_frac": 0.3,
        "curriculum_end_frac": 1.0,
        "curriculum_recent_frac": 0.25,
    }
    # Complete dict: succeeds.
    TrainerConfig.from_dict(raw)

    # Remove one required field; expect KeyError mentioning it.
    del raw["warmup_steps"]
    with pytest.raises(KeyError, match="warmup_steps"):
        TrainerConfig.from_dict(raw)


def test_agent_config_range_checks_polyak_tau():
    raw = {
        "seed": 1,
        "gamma": 0.99,
        "lr": 1e-4,
        "batch_size": 4,
        "target_update_freq": 5,
        "polyak_tau": 1.5,  # invalid: must be in (0, 1)
        "n_steps": 1,
        # IQN distributional head (replaces C51 num_atoms/v_min/v_max).
        "n_quantiles_online": 16,
        "n_quantiles_target": 16,
        "n_quantiles_policy": 8,
        "quantile_embedding_dim": 16,
        "huber_kappa": 1.0,
        "munchausen_alpha": 0.9,
        "munchausen_entropy_tau": 0.03,
        "munchausen_log_pi_clip": -1.0,
        "iqn_bootstrap_mode": "soft",
        "spectral_norm_enabled": False,
        "num_actions": 6,
        "window_size": 10,
        "n_features": 12,
        "hidden_dim": 64,
        "replay_buffer_size": 1000,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,
        "per_new_transition_priority": 1.0,
        "per_priority_cap": 50.0,
        "grad_clip_norm": 1.0,
        "epsilon_start": 0.3,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 1000,
        "entropy_coeff": 0.0,
        "store_partial_n_step": False,
        "debug": False,
        "quantile_logging_interval": 100,
        "quantile_logging_percentiles": [5, 50, 95],
        "noisy_sigma_logging_interval": 0,
        "q_value_logging_interval": 0,
        "q_value_histogram_interval": 0,
        "grad_logging_interval": 0,
        "target_net_logging_interval": 0,
        "td_error_logging_interval": 0,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 128,
        "transformer_dropout": 0.1,
        "lr_scheduler_enabled": False,
        "lr_scheduler_type": "StepLR",
        "lr_scheduler_params": {},
        "aux_loss_weight": 0.1,
        "aux_target_feature_index": 6,
    }
    with pytest.raises(ValueError, match="polyak_tau"):
        AgentConfig.from_dict(raw)

    raw["polyak_tau"] = 0.005
    AgentConfig.from_dict(raw)  # succeeds


def test_agent_config_rejects_negative_new_transition_priority():
    """``per_new_transition_priority`` must be > 0 (it seeds the SumTree)."""
    raw = _minimal_agent_dict()
    raw["per_new_transition_priority"] = 0.0
    with pytest.raises(ValueError, match="per_new_transition_priority"):
        AgentConfig.from_dict(raw)


def test_agent_config_rejects_cap_below_seed():
    """``per_priority_cap`` must be >= ``per_new_transition_priority``."""
    raw = _minimal_agent_dict()
    raw["per_new_transition_priority"] = 10.0
    raw["per_priority_cap"] = 5.0
    with pytest.raises(ValueError, match="per_priority_cap"):
        AgentConfig.from_dict(raw)


def test_agent_config_rejects_invalid_bootstrap_mode():
    """``iqn_bootstrap_mode`` must be ``soft``, ``greedy``, or ``double``."""
    raw = _minimal_agent_dict()
    raw["iqn_bootstrap_mode"] = "double_dqn"
    with pytest.raises(ValueError, match="iqn_bootstrap_mode"):
        AgentConfig.from_dict(raw)


def test_agent_config_accepts_double_bootstrap_mode():
    """``double`` is a valid IQN bootstrap mode."""
    raw = _minimal_agent_dict()
    raw["iqn_bootstrap_mode"] = "double"
    cfg = AgentConfig.from_dict(raw)
    assert cfg.iqn_bootstrap_mode == "double"


def _minimal_agent_dict() -> dict:
    """Minimal-but-complete AgentConfig dict, used by validation-failure tests."""
    return {
        "seed": 1,
        "gamma": 0.99,
        "lr": 1e-4,
        "batch_size": 4,
        "target_update_freq": 5,
        "polyak_tau": 0.005,
        "n_steps": 1,
        "n_quantiles_online": 16,
        "n_quantiles_target": 16,
        "n_quantiles_policy": 8,
        "quantile_embedding_dim": 16,
        "huber_kappa": 1.0,
        "munchausen_alpha": 0.9,
        "munchausen_entropy_tau": 0.03,
        "munchausen_log_pi_clip": -1.0,
        "iqn_bootstrap_mode": "soft",
        "spectral_norm_enabled": False,
        "num_actions": 6,
        "window_size": 10,
        "n_features": 12,
        "hidden_dim": 64,
        "replay_buffer_size": 1000,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,
        "per_new_transition_priority": 1.0,
        "per_priority_cap": 50.0,
        "grad_clip_norm": 1.0,
        "epsilon_start": 0.3,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 1000,
        "entropy_coeff": 0.0,
        "store_partial_n_step": False,
        "debug": False,
        "quantile_logging_interval": 100,
        "quantile_logging_percentiles": [5, 50, 95],
        "noisy_sigma_logging_interval": 0,
        "q_value_logging_interval": 0,
        "q_value_histogram_interval": 0,
        "grad_logging_interval": 0,
        "target_net_logging_interval": 0,
        "td_error_logging_interval": 0,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 128,
        "transformer_dropout": 0.1,
        "lr_scheduler_enabled": False,
        "lr_scheduler_type": "StepLR",
        "lr_scheduler_params": {},
        "aux_loss_weight": 0.1,
        "aux_target_feature_index": 6,
    }


def test_trainer_config_rejects_non_dict():
    with pytest.raises(TypeError):
        TrainerConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_agent_config_missing_multiple_keys_reports_all():
    with pytest.raises(KeyError) as excinfo:
        AgentConfig.from_dict({"seed": 1})
    message = str(excinfo.value)
    # A few random required fields should all be mentioned.
    for key in ("gamma", "lr", "batch_size", "n_quantiles_online", "huber_kappa"):
        assert key in message

"""Integration test for the trainer training loop with real GPU agent."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from momentum_agent import RainbowDQNAgent
from momentum_train.data import DataManager
from momentum_train.trainer import RainbowTrainerModule

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _create_test_npz(directory: Path, name: str, rows: int = 100) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    np.random.seed(42)
    close = np.cumsum(np.random.randn(rows) * 0.5) + 100.0
    close = np.maximum(close, 1.0).astype(np.float32)
    features = np.random.randn(rows, 12).astype(np.float32)
    features[:, 3] = close
    np.savez_compressed(path, close_prices=close, features=features)
    return path


@requires_cuda
def test_trainer_runs_episodes_and_logs_progress(tmp_path):
    processed = tmp_path / "processed"
    _create_test_npz(processed / "train", "2024-01-01_TEST-USD.npz")
    _create_test_npz(processed / "validation", "2024-06-01_TEST-USD.npz")
    _create_test_npz(processed / "test", "2025-01-01_TEST-USD.npz")

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    agent_config = {
        "seed": 42,
        "gamma": 0.99,
        "lr": 1e-4,
        "batch_size": 16,
        "replay_buffer_size": 500,
        "target_update_freq": 10,
        "n_quantiles_online": 16,
        "n_quantiles_target": 16,
        "n_quantiles_policy": 8,
        "quantile_embedding_dim": 16,
        "huber_kappa": 1.0,
        "munchausen_alpha": 0.9,
        "munchausen_entropy_tau": 0.03,
        "munchausen_log_pi_clip": -1.0,
        "spectral_norm_enabled": False,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,
        "n_steps": 3,
        "window_size": 10,
        "n_features": 12,
        "hidden_dim": 64,
        "num_actions": 6,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 128,
        "transformer_dropout": 0.1,
        "grad_clip_norm": 1.0,
        "polyak_tau": 0.005,
        "store_partial_n_step": True,
        "epsilon_start": 0.3,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": 1000,
        "entropy_coeff": 0.03,
        "debug": False,
        "quantile_logging_interval": 2000,
        "quantile_logging_percentiles": [5, 25, 50, 75, 95],
        "noisy_sigma_logging_interval": 0,
        "q_value_logging_interval": 0,
        "q_value_histogram_interval": 0,
        "grad_logging_interval": 0,
        "target_net_logging_interval": 0,
        "td_error_logging_interval": 0,
        "lr_scheduler_enabled": False,
        "lr_scheduler_type": "StepLR",
        "lr_scheduler_params": {},
        "aux_loss_weight": 0.1,
        "aux_target_feature_index": 6,
    }

    config = {
        "agent": agent_config,
        "environment": {
            "window_size": 10,
            "initial_balance": 1000.0,
            "transaction_fee": 0.001,
            "reward_scale": 1.0,
            "invalid_action_penalty": -0.1,
            "drawdown_penalty_lambda": 0.5,
            "slippage_bps": 5.0,
            "opportunity_cost_lambda": 0.1,
            "benchmark_allocation_frac": 0.5,
            "min_rebalance_pct": 0.02,
            "min_trade_value": 1.0,
        },
        "trainer": {
            "seed": 42,
            "num_vector_envs": 1,
            "warmup_steps": 10,
            "update_freq": 10,
            "gradient_updates_per_step": 1,
            "log_freq": 10,
            "per_stats_log_freq": 0,
            "validation_freq": 3,
            "checkpoint_save_freq": 100,
            "latest_checkpoint_keep_last_n": 0,
            "reward_window": 5,
            "reward_clip": 5.0,
            "early_stopping_patience": 100,
            "min_episodes_before_early_stopping": 0,
            "min_validation_threshold": 0.0,
            "benchmark_allocation_frac_start": 0.5,
            "benchmark_allocation_frac_end": 0.1,
            "benchmark_allocation_frac_anneal_episodes": 0,
            "final_phase_lr_start_frac": 0.85,
            "final_phase_lr_multiplier": 0.3,
        },
        "run": {
            "mode": "train",
            "episodes": 3,
            "model_dir": str(model_dir),
            "resume": False,
            "specific_file": None,
            "skip_evaluation": True,
        },
    }

    dm = DataManager(base_dir=tmp_path, processed_dir_name="processed")
    dm.organize_data()

    agent = RainbowDQNAgent(config=agent_config, device="cuda")
    trainer = RainbowTrainerModule(
        agent=agent,
        device=torch.device("cuda"),
        data_manager=dm,
        config=config,
        writer=None,
    )

    trainer.train(
        num_episodes=3,
        start_episode=0,
        start_total_steps=0,
        initial_best_score=-float("inf"),
        initial_early_stopping_counter=0,
    )

    progress_file = model_dir / "progress.jsonl"
    assert progress_file.exists()
    lines = progress_file.read_text().strip().split("\n")
    assert len(lines) >= 3

    for line in lines:
        record = json.loads(line)
        assert "event" in record
        # Episode-end records have ``episode``; validation records (now firing
        # because validation_freq is no longer larger than num_episodes thanks
        # to the new cadence guard) do not. Either is acceptable.
        if record["event"] != "validation":
            assert "episode" in record

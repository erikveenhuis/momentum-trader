from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from momentum_train.trainer import RainbowTrainerModule


def _create_minimal_trainer(tmp_path) -> RainbowTrainerModule:
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.validation_freq = 1
    trainer.checkpoint_save_freq = 5
    trainer.best_validation_metric = -np.inf
    trainer.early_stopping_counter = 0
    trainer.min_validation_threshold = 0.0
    trainer.model_dir = tmp_path
    trainer.latest_trainer_checkpoint_path = str(tmp_path / "checkpoint_latest.pt")
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "best_checkpoint")
    trainer.writer = None
    trainer.reward_window = 10
    trainer.agent = SimpleNamespace(lr_scheduler_enabled=False)
    return trainer


@pytest.mark.unittest
def test_validate_aggregates_metrics_and_writes_results(tmp_path, monkeypatch):
    trainer = _create_minimal_trainer(tmp_path)

    metrics_sequence = [
        {
            "avg_reward": 1.0,
            "portfolio_value": 1005.0,
            "total_return": 5.0,
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.02,
            "transaction_costs": 1.5,
        },
        {
            "avg_reward": 2.0,
            "portfolio_value": 1010.0,
            "total_return": 6.0,
            "sharpe_ratio": 0.2,
            "max_drawdown": 0.01,
            "transaction_costs": 2.0,
        },
    ]

    detailed_results = [
        {"file": "file1.csv", "reward": 1.0},
        {"file": "file2.csv", "reward": 2.0},
    ]

    episode_scores = [0.4, 0.6]
    iterator = iter(
        {
            "file_metrics": metrics_sequence[i],
            "detailed_result": detailed_results[i],
            "episode_score": episode_scores[i],
        }
        for i in range(len(metrics_sequence))
    )

    trainer._validate_single_file = lambda path: next(iterator)

    saved_records = []
    original_save = RainbowTrainerModule._save_validation_results

    def capture_save(self, validation_score, avg_metrics, results):
        saved_records.append((validation_score, avg_metrics, results))
        original_save(self, validation_score, avg_metrics, results)

    trainer._save_validation_results = capture_save.__get__(trainer, RainbowTrainerModule)

    def record_early_stopping(score):
        trainer.best_validation_metric = score
        trainer.early_stopping_counter = 0
        return False

    trainer._check_early_stopping = record_early_stopping

    should_stop, validation_score, avg_metrics = trainer.validate(
        [Path("file1.csv"), Path("file2.csv")]
    )

    assert not should_stop
    assert validation_score == pytest.approx(np.mean(episode_scores))
    assert trainer.best_validation_metric == pytest.approx(validation_score)
    assert saved_records, "Validation results were not persisted"

    saved_score, saved_avg, saved_details = saved_records[0]
    assert saved_score == validation_score
    assert saved_avg == avg_metrics
    assert len(saved_details) == len(detailed_results)

    output_files = list(tmp_path.glob("validation_results_*.json"))
    assert output_files, "Validation summary JSON was not generated"


@pytest.mark.unittest
def test_handle_validation_and_checkpointing_triggers_best_checkpoint(tmp_path):
    trainer = _create_minimal_trainer(tmp_path)
    trainer.best_validation_metric = 0.2

    save_calls = []

    def capture_save(episode, total_steps, is_best, validation_score):
        save_calls.append(
            {
                "episode": episode,
                "total_steps": total_steps,
                "is_best": is_best,
                "validation_score": validation_score,
            }
        )

    trainer._save_checkpoint = capture_save.__get__(trainer, RainbowTrainerModule)

    def fake_validate(self, val_files):
        return False, 0.5, {"avg_reward": 1.0}

    def record_early_stopping(score):
        trainer.best_validation_metric = score
        trainer.early_stopping_counter = 0
        return False

    trainer.validate = fake_validate.__get__(trainer, RainbowTrainerModule)
    trainer._check_early_stopping = record_early_stopping

    tracker = SimpleNamespace(get_recent_metrics=lambda: {})

    should_stop = trainer._handle_validation_and_checkpointing(
        episode=0,
        total_train_steps=42,
        val_files=[Path("file1.csv")],
        tracker=tracker,
    )

    assert should_stop is False
    assert trainer.best_validation_metric == pytest.approx(0.5)
    assert len(save_calls) == 1

    call = save_calls[0]
    assert call["episode"] == 1
    assert call["total_steps"] == 42
    assert call["is_best"] is True
    assert call["validation_score"] == pytest.approx(0.5)

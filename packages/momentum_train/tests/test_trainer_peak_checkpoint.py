"""Peak checkpoint stream (strict validation improvement + buffer)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from momentum_train.trainer import RainbowTrainerModule


class _Net:
    def state_dict(self) -> dict:
        return {"w": 0}


class _Optim:
    def state_dict(self) -> dict:
        return {"o": 0}


def _make_trainer(tmp_path: Path, *, peak_keep: int) -> RainbowTrainerModule:
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.best_validation_metric = 0.5
    trainer.early_stopping_counter = 0
    trainer.min_episodes_before_early_stopping = 0
    trainer.min_episodes_before_checkpoint_pinning = 0
    trainer.peak_checkpoint_keep_last_n = peak_keep
    trainer.top_k_skip_flat_benchmark = True
    trainer.top_k_flat_score_threshold = 0.585
    trainer.top_k_flat_max_abs_return_pct = 2.0
    trainer._checkpoints_saved_this_run = 0
    trainer.writer = None
    trainer.latest_trainer_checkpoint_path = str(tmp_path / "checkpoint_trainer_latest.pt")
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "checkpoint_trainer_best")
    trainer.run_config = {"model_dir": str(tmp_path)}
    trainer.latest_checkpoint_keep_last_n = 0
    trainer.agent = SimpleNamespace(
        config={"lr": 1e-4},
        total_steps=100,
        buffer=None,
        network=_Net(),
        target_network=_Net(),
        optimizer=_Optim(),
        scheduler=None,
        lr_scheduler_enabled=False,
    )
    return trainer


@pytest.mark.unit
def test_peak_save_disabled_when_keep_zero(tmp_path):
    trainer = _make_trainer(tmp_path, peak_keep=0)
    ckpt = {"episode": 10, "total_train_steps": 1000}
    assert (
        trainer._maybe_save_peak_checkpoint(
            episode=10,
            total_steps=1000,
            validation_score=0.55,
            validation_metrics={"total_return": -5.0},
            checkpoint=ckpt,
        )
        is None
    )


@pytest.mark.unit
def test_peak_save_skips_flat_benchmark(tmp_path):
    trainer = _make_trainer(tmp_path, peak_keep=3)
    ckpt = {"episode": 10, "total_train_steps": 1000}
    assert (
        trainer._maybe_save_peak_checkpoint(
            episode=10,
            total_steps=1000,
            validation_score=0.5961,
            validation_metrics={"total_return": -0.4},
            checkpoint=ckpt,
        )
        is None
    )
    assert list(tmp_path.glob("checkpoint_trainer_peak_*")) == []


@pytest.mark.unit
def test_peak_save_writes_pt_without_buffer_when_no_buffer(tmp_path):
    """Agent has no buffer → peak .pt still written (resume needs paired buffer)."""
    trainer = _make_trainer(tmp_path, peak_keep=2)
    ckpt = {"episode": 10, "total_train_steps": 1000, "best_validation_metric": 0.5}
    path = trainer._maybe_save_peak_checkpoint(
        episode=10,
        total_steps=1000,
        validation_score=0.5403,
        validation_metrics={"total_return": -3.3},
        checkpoint=ckpt,
    )
    assert path is not None
    assert path.exists()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["episode"] == 10

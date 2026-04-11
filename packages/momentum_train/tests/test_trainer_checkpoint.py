from types import SimpleNamespace

import pytest
import torch
from momentum_train.trainer import RainbowTrainerModule


class DummyNetwork:
    def state_dict(self):
        return {"weights": 1}


class DummyOptimizer:
    def state_dict(self):
        return {"optimizer": 1}


class DummyBuffer:
    def state_dict(self):
        return {"buffer": 1}


@pytest.mark.unit
def test_save_checkpoint_persists_expected_keys(tmp_path):
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.best_validation_metric = 0.75
    trainer.early_stopping_counter = 1
    trainer.agent = SimpleNamespace(
        config={"lr": 0.001},
        total_steps=123,
        network=DummyNetwork(),
        target_network=DummyNetwork(),
        optimizer=DummyOptimizer(),
        scheduler=None,
        lr_scheduler_enabled=False,
        buffer=DummyBuffer(),
    )
    trainer.writer = None
    trainer.latest_trainer_checkpoint_path = str(tmp_path / "checkpoint_latest.pt")
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "best_checkpoint")
    trainer.best_model_base_prefix = str(tmp_path / "best_model")

    trainer._save_checkpoint(
        episode=10,
        total_steps=456,
        is_best=True,
        validation_score=0.8,
    )

    latest_files = list(tmp_path.glob("checkpoint_latest_*_ep10_reward0.7500.pt"))
    assert latest_files, "Latest checkpoint file was not created"
    checkpoint = torch.load(latest_files[0], map_location="cpu", weights_only=False)

    expected_keys = {
        "episode",
        "total_train_steps",
        "best_validation_metric",
        "early_stopping_counter",
        "buffer_state",
        "agent_config",
        "agent_total_steps",
        "total_steps",
        "network_state_dict",
        "target_network_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "validation_score",
    }

    for key in expected_keys:
        assert key in checkpoint, f"Missing key in checkpoint: {key}"

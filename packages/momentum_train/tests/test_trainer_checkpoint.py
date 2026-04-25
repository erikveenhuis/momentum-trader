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
    # Bookkeeping counter introduced for the "don't clobber _final.pt if no
    # real saves happened this run" guard; _save_checkpoint increments it.
    trainer._checkpoints_saved_this_run = 0
    trainer.agent = SimpleNamespace(
        config={"lr": 0.001},
        total_steps=123,
        # ``env_steps`` is persisted in the checkpoint dict for
        # backward-compat with the env-step/learn-step split.
        env_steps=456,
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
    # _save_checkpoint now calls _rotate_latest_checkpoints, which reads
    # these two attributes; set them to no-op values for this test.
    trainer.run_config = {"model_dir": str(tmp_path)}
    trainer.latest_checkpoint_keep_last_n = 0

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
        # ``buffer_state`` stays present but is now always None -- the
        # replay buffer itself is persisted to a side-car directory next
        # to the .pt. ``buffer_sidecar_relpath`` is the hint the resume
        # path uses to locate that side-car.
        "buffer_state",
        "buffer_sidecar_relpath",
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
    # The migration invariant: the pickled buffer blob must NOT be
    # present in the ``.pt`` file any more (it was the direct cause of
    # the OOM kills during torch.save). The buffer now lives in the
    # ``<checkpoint>.buffer/`` side-car directory written via
    # ``_save_buffer_sidecar`` / ``buffer.save_to_path``, so the .pt is
    # small (network + optimizer state) and torch.save no longer
    # produces a multi-GB pickle stream.
    assert checkpoint["buffer_state"] is None, (
        "buffer_state must be None after the side-car migration; "
        "embedding the ~6 GiB PER buffer in torch.save was the root cause of OOM kills."
    )

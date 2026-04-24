from types import SimpleNamespace

import pytest
import torch
from momentum_train import trainer_checkpoint as trainer_module
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
    # the OOM kills during torch.save).
    assert checkpoint["buffer_state"] is None, (
        "buffer_state must be None after the side-car migration; "
        "embedding the ~6 GiB PER buffer in torch.save was the root cause of OOM kills."
    )


def _make_trainer_with_minimal_save_state(tmp_path):
    """Build a bare-bones trainer whose ``_save_checkpoint`` call will
    succeed. Used by the memory-release regression tests below -- those don't
    care about checkpoint content, only about the side-effect of
    ``release_memory_to_os`` being invoked after every torch.save.
    """
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.best_validation_metric = 0.0
    trainer.early_stopping_counter = 0
    trainer._checkpoints_saved_this_run = 0
    trainer.agent = SimpleNamespace(
        config={"lr": 0.001},
        total_steps=123,
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
    trainer.run_config = {"model_dir": str(tmp_path)}
    trainer.latest_checkpoint_keep_last_n = 0
    return trainer


@pytest.mark.unit
def test_save_checkpoint_invokes_release_memory_to_os(tmp_path, monkeypatch):
    """Regression guard for the Apr 2026 OOM mitigation.

    After each ``torch.save`` in ``_save_checkpoint`` we must call
    ``release_memory_to_os`` so glibc can return the pickle-buffer pages
    (~8 GB per save) back to the kernel. Without this the RSS HWM
    ratchets up by several GiB every checkpoint -- the direct cause of
    the observed 8-13 h "kill by OOM killer at ~54 GiB anon-rss" pattern.
    """
    calls = {"release": 0, "rss": 0}

    def _fake_release():
        calls["release"] += 1
        return True

    def _fake_rss():
        calls["rss"] += 1
        return 1.0  # GiB -- plausible but clearly synthetic

    monkeypatch.setattr(trainer_module, "release_memory_to_os", _fake_release)
    monkeypatch.setattr(trainer_module, "current_rss_gb", _fake_rss)

    trainer = _make_trainer_with_minimal_save_state(tmp_path)
    # Non-best save (one torch.save invocation).
    trainer._save_checkpoint(episode=1, total_steps=100, is_best=False)
    assert calls["release"] == 1, (
        f"Expected exactly one release_memory_to_os call on a non-best save, got {calls['release']}"
    )
    # current_rss_gb is called twice: once before save, once after trim.
    assert calls["rss"] >= 2


@pytest.mark.unit
def test_save_checkpoint_still_releases_on_is_best(tmp_path, monkeypatch):
    """The best-checkpoint path writes an additional file. One release call at
    the tail of ``_save_checkpoint`` covers both saves -- the pickle-buffer
    arena pages from both ``torch.save`` invocations are reclaimed in one trim.
    """
    calls = {"release": 0}

    def _fake_release():
        calls["release"] += 1
        return True

    monkeypatch.setattr(trainer_module, "release_memory_to_os", _fake_release)
    monkeypatch.setattr(trainer_module, "current_rss_gb", lambda: 1.0)

    trainer = _make_trainer_with_minimal_save_state(tmp_path)
    trainer._save_checkpoint(episode=1, total_steps=100, is_best=True, validation_score=0.5)
    assert calls["release"] == 1, (
        f"Expected one release_memory_to_os call even when both latest + best saves fire, got {calls['release']}"
    )

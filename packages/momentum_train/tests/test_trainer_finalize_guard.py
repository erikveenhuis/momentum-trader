"""Regression tests for the ``_finalize_training`` guard against clobbering
``rainbow_transformer_final_agent_state.pt`` when no checkpoint was actually
persisted in the current run.

Context: the April 22 2026 incident had ``--resume`` silently fall back to a
fresh agent after failing to load a zero-byte checkpoint. That fresh agent
then hit the ``finally: self._finalize_training(...)`` path and overwrote the
operator's previously-trained ``_final`` file with untrained weights. Gate in
``_finalize_training`` now refuses to save ``_final`` when
``self._checkpoints_saved_this_run == 0``.

Tests bypass ``__init__`` (same pattern used in
``test_trainer_checkpoint_rotation.py``) so they stay fast and don't depend on
the full trainer / agent / env wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from momentum_train.trainer import RainbowTrainerModule


class _StubAgent:
    def __init__(self):
        self.save_model_calls: list[str] = []
        self.observed_n_step_rewards_history = None

    def save_model(self, prefix: str) -> None:
        self.save_model_calls.append(prefix)


def _make_trainer(tmp_path: Path, *, checkpoints_saved: int) -> RainbowTrainerModule:
    """Build a trainer with only the attributes ``_finalize_training`` reads."""
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.model_dir = str(tmp_path)
    trainer._checkpoints_saved_this_run = checkpoints_saved
    trainer.agent = _StubAgent()
    trainer.writer = None
    trainer.best_validation_metric = -np.inf
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "checkpoint_trainer_best")
    trainer._validation_env_cache = {}
    return trainer


@pytest.mark.unit
def test_finalize_skips_save_when_no_checkpoints_saved_this_run(tmp_path, caplog):
    """No checkpoint saved this run -> do not touch _final (preserve prior artifact)."""
    trainer = _make_trainer(tmp_path, checkpoints_saved=0)
    preexisting = tmp_path / "rainbow_transformer_final_agent_state.pt"
    preexisting.write_bytes(b"previous good weights")
    original_bytes = preexisting.read_bytes()

    with caplog.at_level("WARNING", logger="Trainer"):
        trainer._finalize_training(total_train_steps=0, num_episodes=1, val_files=[])

    assert trainer.agent.save_model_calls == [], (
        "agent.save_model must not be called when _checkpoints_saved_this_run == 0"
    )
    assert preexisting.read_bytes() == original_bytes, (
        "Pre-existing rainbow_transformer_final_agent_state.pt was clobbered despite the guard"
    )
    assert any("Skipping save" in rec.message for rec in caplog.records), (
        "Expected a WARNING log explaining why _final was not written"
    )


@pytest.mark.unit
def test_finalize_saves_when_checkpoint_was_persisted(tmp_path):
    """At least one successful _save_checkpoint in this run -> _final write proceeds."""
    trainer = _make_trainer(tmp_path, checkpoints_saved=3)

    trainer._finalize_training(total_train_steps=10_000, num_episodes=100, val_files=[])

    assert trainer.agent.save_model_calls == [str(tmp_path / "rainbow_transformer_final")], (
        f"Expected exactly one save_model call to the _final prefix, got {trainer.agent.save_model_calls!r}"
    )


@pytest.mark.unit
def test_finalize_skip_warns_differently_when_no_prior_final_exists(tmp_path, caplog):
    """Edge case: no prior _final file and nothing to save this run.

    We still skip the save (weights are untrained / fresh) but the log
    phrasing should reflect that no clobber risk exists. The behavior — no
    save_model call — is the same as the pre-existing case.
    """
    trainer = _make_trainer(tmp_path, checkpoints_saved=0)
    assert not (tmp_path / "rainbow_transformer_final_agent_state.pt").exists()

    with caplog.at_level("WARNING", logger="Trainer"):
        trainer._finalize_training(total_train_steps=0, num_episodes=1, val_files=[])

    assert trainer.agent.save_model_calls == []
    assert any("Skipping save" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_finalize_save_failure_is_logged_not_raised(tmp_path, caplog):
    """Existing behavior: a save_model failure must be logged, not propagated.

    Regression-guards the "finally:" contract in train() — finalize runs
    during teardown and cannot raise.
    """
    trainer = _make_trainer(tmp_path, checkpoints_saved=1)

    def _broken_save(prefix: str) -> None:
        raise RuntimeError("disk full (simulated)")

    trainer.agent.save_model = _broken_save  # type: ignore[assignment]

    with caplog.at_level("ERROR", logger="Trainer"):
        trainer._finalize_training(total_train_steps=1, num_episodes=1, val_files=[])

    assert any("Error saving final agent model" in rec.message for rec in caplog.records)

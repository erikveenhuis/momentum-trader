from unittest.mock import MagicMock

import pytest
from momentum_train.trainer import RainbowTrainerModule


@pytest.mark.unit
def test_maybe_save_emergency_checkpoint_noop_when_not_requested():
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer._emergency_checkpoint_requested = False
    trainer._save_checkpoint = MagicMock()
    trainer._flush_writer = MagicMock()

    trainer._maybe_save_emergency_checkpoint(episode=10, total_steps=1000)

    trainer._save_checkpoint.assert_not_called()
    trainer._flush_writer.assert_not_called()


@pytest.mark.unit
def test_maybe_save_emergency_checkpoint_triggers_save():
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer._emergency_checkpoint_requested = True
    trainer._save_checkpoint = MagicMock()
    trainer._flush_writer = MagicMock()

    trainer._maybe_save_emergency_checkpoint(episode=10, total_steps=1000)

    assert trainer._emergency_checkpoint_requested is False
    trainer._save_checkpoint.assert_called_once_with(
        episode=10,
        total_steps=1000,
        is_best=False,
        validation_score=None,
    )
    trainer._flush_writer.assert_called_once()


@pytest.mark.unit
def test_maybe_save_emergency_checkpoint_requeues_on_failure():
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer._emergency_checkpoint_requested = True
    trainer._save_checkpoint = MagicMock(side_effect=RuntimeError("disk full"))
    trainer._flush_writer = MagicMock()

    trainer._maybe_save_emergency_checkpoint(episode=3, total_steps=500)

    assert trainer._emergency_checkpoint_requested is True
    trainer._flush_writer.assert_not_called()


@pytest.mark.unit
def test_install_emergency_checkpoint_signal_handler_sets_flag():
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer._emergency_checkpoint_requested = False
    trainer._install_emergency_checkpoint_signal_handler()

    import signal

    handler = signal.getsignal(signal.SIGUSR1)
    assert callable(handler)
    handler(signal.SIGUSR1, None)
    assert trainer._emergency_checkpoint_requested is True

    # Restore default so pytest is not affected by later tests in the same process.
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)

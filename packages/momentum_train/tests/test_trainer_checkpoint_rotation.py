"""Unit tests for ``RainbowTrainerModule._rotate_latest_checkpoints``.

The rotation logic is a small disk-hygiene helper that runs after each
``latest_*`` checkpoint write to keep at most ``keep_last_n`` files.

Tests intentionally bypass ``__init__`` and only set the two attributes
the helper actually reads (``latest_checkpoint_keep_last_n`` and
``run_config["model_dir"]``) so they stay fast and decoupled from the
rest of the trainer wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from momentum_train.trainer import RainbowTrainerModule


def _make_trainer(tmp_path: Path, keep_last_n: int) -> RainbowTrainerModule:
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.latest_checkpoint_keep_last_n = keep_last_n
    trainer.run_config = {"model_dir": str(tmp_path)}
    return trainer


def _write_latest(tmp_path: Path, episode: int, *, date: str = "20260420") -> Path:
    path = tmp_path / f"checkpoint_trainer_latest_{date}_ep{episode}_reward0.5000.pt"
    path.write_bytes(b"x")
    return path


@pytest.mark.unit
def test_rotation_disabled_when_keep_last_n_zero(tmp_path):
    trainer = _make_trainer(tmp_path, keep_last_n=0)
    files = [_write_latest(tmp_path, ep) for ep in (100, 200, 300, 400, 500)]

    deleted = trainer._rotate_latest_checkpoints()

    assert deleted == []
    for f in files:
        assert f.exists(), f"Rotation should be a no-op when keep_last_n=0; missing {f.name}"


@pytest.mark.unit
def test_rotation_keeps_only_latest_n_by_episode(tmp_path):
    trainer = _make_trainer(tmp_path, keep_last_n=3)
    # Write deliberately out of episode order; rotation must sort by episode.
    files_by_ep = {ep: _write_latest(tmp_path, ep) for ep in (300, 100, 500, 400, 200)}

    deleted = trainer._rotate_latest_checkpoints()

    deleted_eps = sorted(int(p.name.split("_ep")[1].split("_")[0]) for p in deleted)
    assert deleted_eps == [100, 200], f"Expected eps 100 and 200 to be pruned, got {deleted_eps}"
    for ep in (300, 400, 500):
        assert files_by_ep[ep].exists(), f"Top-3 episode {ep} should have survived"
    for ep in (100, 200):
        assert not files_by_ep[ep].exists(), f"Episode {ep} should have been deleted"


@pytest.mark.unit
def test_rotation_no_op_when_count_at_or_below_keep_n(tmp_path):
    trainer = _make_trainer(tmp_path, keep_last_n=5)
    files = [_write_latest(tmp_path, ep) for ep in (100, 200, 300)]

    deleted = trainer._rotate_latest_checkpoints()

    assert deleted == []
    for f in files:
        assert f.exists()


@pytest.mark.unit
def test_rotation_ignores_best_and_other_files(tmp_path):
    """``best_*`` and unrelated files must never be touched."""
    trainer = _make_trainer(tmp_path, keep_last_n=1)

    latest_files = [_write_latest(tmp_path, ep) for ep in (100, 200, 300)]
    best_files = [
        tmp_path / "checkpoint_trainer_best_20260420_ep250_score_0.6000.pt",
        tmp_path / "checkpoint_trainer_best_20260418_ep180_score_0.5500.pt",
    ]
    for f in best_files:
        f.write_bytes(b"x")
    unrelated = [
        tmp_path / "rainbow_transformer_final_agent_state.pt",
        tmp_path / "progress.jsonl",
        tmp_path / "validation_results_20260420.json",
    ]
    for f in unrelated:
        f.write_bytes(b"x")

    trainer._rotate_latest_checkpoints()

    assert latest_files[-1].exists(), "Most recent latest_* must survive"
    for f in latest_files[:-1]:
        assert not f.exists(), f"Older latest_* should have been pruned: {f.name}"
    for f in best_files + unrelated:
        assert f.exists(), f"Non-latest file must not be touched: {f.name}"


@pytest.mark.unit
def test_rotation_includes_recover_checkpoints_in_latest_stream(tmp_path):
    """Recover-script outputs use the same ``latest_*_reward<token>.pt`` pattern.

    Their reward token is ``rewardrecover`` (no float). The rotation regex
    must match them so they participate in the keep-last-N window — the
    recover script sets a higher episode number so they will naturally be
    the most-recent file and survive a single-keep window.
    """
    trainer = _make_trainer(tmp_path, keep_last_n=1)

    older = _write_latest(tmp_path, episode=300)
    recover = tmp_path / "checkpoint_trainer_latest_20260420_ep301_rewardrecover.pt"
    recover.write_bytes(b"x")

    trainer._rotate_latest_checkpoints()

    assert recover.exists(), "Recover checkpoint at higher ep must survive"
    assert not older.exists(), "Older latest_* should have been pruned"


@pytest.mark.unit
def test_rotation_skips_malformed_filenames(tmp_path):
    """Files matching the glob but not the canonical regex must be left alone."""
    trainer = _make_trainer(tmp_path, keep_last_n=1)

    canonical = [_write_latest(tmp_path, ep) for ep in (100, 200, 300)]
    malformed = [
        # No `_ep<int>_` segment.
        tmp_path / "checkpoint_trainer_latest_20260420_epXYZ_rewardABC.pt",
        # Missing reward token entirely (won't match the glob — sanity guard).
        tmp_path / "checkpoint_trainer_latest_legacy.pt",
    ]
    for f in malformed:
        f.write_bytes(b"x")

    trainer._rotate_latest_checkpoints()

    assert canonical[-1].exists()
    assert not canonical[0].exists()
    for f in malformed:
        assert f.exists(), f"Malformed file must not be deleted: {f.name}"


@pytest.mark.unit
def test_rotation_silently_handles_unlink_errors(tmp_path, monkeypatch, caplog):
    """If ``Path.unlink`` raises OSError the helper logs and continues."""
    trainer = _make_trainer(tmp_path, keep_last_n=1)

    files = [_write_latest(tmp_path, ep) for ep in (100, 200)]

    real_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if "ep100" in self.name:
            raise OSError("simulated permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    deleted = trainer._rotate_latest_checkpoints()

    assert deleted == []
    assert files[0].exists(), "Failed-unlink file should still exist"
    assert files[1].exists(), "Top-keep file must never be considered for deletion"

"""Tests for ``model_dir_archive.archive_model_dir_for_fresh_training``."""

from __future__ import annotations

import pytest
from momentum_train.utils.model_dir_archive import (
    archive_model_dir_for_fresh_training,
    archive_model_dir_for_fresh_training_if_needed,
)


def _stage_stale_run(root):
    """Reproduce the high-episode stale state that broke the May 2026 fresh run.

    A previous run left ``checkpoint_trainer_latest_*_ep14001_*.pt`` and its
    sibling ``.buffer`` directory at the top of model_dir. Without archiving,
    ``latest_checkpoint_keep_last_n`` rotation prunes the new run's
    lower-episode saves and keeps these stale ones, which then have no
    matching network weights for the new architecture (or are simply from a
    different run).
    """
    (root / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.pt").write_text("stale-pt")
    buf = root / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.buffer"
    buf.mkdir()
    (buf / "memmap.npy").write_bytes(b"\0" * 16)
    (root / "progress.jsonl").write_text('{"event":"episode"}\n')


@pytest.mark.unit
def test_archive_no_op_when_model_dir_missing(tmp_path):
    missing = tmp_path / "nope"
    assert archive_model_dir_for_fresh_training(missing) is None


@pytest.mark.unit
def test_archive_no_op_when_model_dir_empty(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    assert archive_model_dir_for_fresh_training(root) is None


@pytest.mark.unit
def test_archive_moves_checkpoint_progress_validation_topk_best_final(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    prev_archive = root / "_archive" / "old"
    prev_archive.mkdir(parents=True)
    (root / "checkpoint_trainer_latest_20260101_ep99_reward0.1000.pt").write_text("pt")
    buf = root / "checkpoint_trainer_latest_20260101_ep99_reward0.1000.buffer"
    buf.mkdir()
    (buf / "meta.json").write_text("{}")
    (root / "checkpoint_trainer_best_20260101_ep100_score_0.2000.pt").write_text("best")
    (root / "checkpoint_trainer_topk_20260101_ep101_score_0.2100.pt").write_text("topk")
    (root / "progress.jsonl").write_text('{"event":"episode"}\n')
    (root / "validation_results_20260101_120000.json").write_text("{}")
    (root / "test_results_20260101_120000.json").write_text("{}")
    (root / "trades_validation.jsonl").write_text("{}\n")
    (root / "rainbow_transformer_final_agent_state.pt").write_text("final")

    dest = archive_model_dir_for_fresh_training(root)
    assert dest is not None
    assert dest.is_dir()
    assert dest.parent.name == "_archive"
    assert "pre_fresh_" in dest.name

    assert not (root / "progress.jsonl").exists()
    assert (dest / "progress.jsonl").exists()
    assert (dest / "checkpoint_trainer_latest_20260101_ep99_reward0.1000.pt").exists()
    assert (dest / "checkpoint_trainer_latest_20260101_ep99_reward0.1000.buffer" / "meta.json").exists()

    # Prior nested _archive untouched
    assert prev_archive.is_dir()
    assert not (dest / "old").exists()


@pytest.mark.unit
def test_archive_skips_runs_by_default(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    runs = root / "runs"
    runs.mkdir()
    (runs / "events.out").write_text("tb")
    (root / "checkpoint_trainer_latest_20260101_ep1_reward0.pt").write_text("x")

    dest = archive_model_dir_for_fresh_training(root, include_tensorboard_runs=False)
    assert dest is not None
    assert (runs / "events.out").exists()
    assert not (dest / "runs").exists()


@pytest.mark.unit
def test_archive_moves_runs_when_requested(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    runs = root / "runs"
    runs.mkdir()
    (runs / "events.out").write_text("tb")
    (root / "progress.jsonl").write_text("x")

    dest = archive_model_dir_for_fresh_training(root, include_tensorboard_runs=True)
    assert dest is not None
    assert (dest / "runs" / "events.out").read_text() == "tb"
    assert not runs.exists()


@pytest.mark.unit
def test_archive_does_not_walk_into_nested_ignored_dirs(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    nested = root / "runs" / "nested"
    nested.mkdir(parents=True)
    (nested / "fake_latest.pt").write_text("nope")

    dest = archive_model_dir_for_fresh_training(root, include_tensorboard_runs=False)
    assert dest is None
    assert (nested / "fake_latest.pt").exists()


@pytest.mark.unit
def test_archive_ignores_unrelated_top_level_files(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "README.txt").write_text("keep")
    (root / "scratch.npy").write_bytes(b"\0")

    assert archive_model_dir_for_fresh_training(root) is None
    assert (root / "README.txt").exists()


# ---------------------------------------------------------------------------
# Gating regression tests for ``archive_model_dir_for_fresh_training_if_needed``.
#
# These tests reproduce the May 2026 silent-data-loss scenario: a fresh run
# was launched while stale ``checkpoint_trainer_latest_*_ep14001_*.pt`` files
# from a prior run sat at the top of ``model_dir``. Because the gating logic
# was inlined in ``run_training`` and never directly tested, it was possible
# for the archive call to be skipped (or never wired at all) and rotation to
# silently delete every new save in favour of the high-episode stale ones.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_if_needed_archives_on_clean_fresh_start(tmp_path):
    """The happy path: fresh start, archive enabled, no opt-out flags."""
    root = tmp_path / "models"
    root.mkdir()
    _stage_stale_run(root)

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=False,
        skip_archive_on_fresh_start=False,
        archive_enabled_in_config=True,
    )

    assert dest is not None
    assert dest.parent.name == "_archive"
    assert "pre_fresh_" in dest.name
    assert (dest / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.pt").exists()
    assert (dest / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.buffer" / "memmap.npy").exists()
    assert not (root / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.pt").exists()


@pytest.mark.unit
def test_if_needed_no_op_when_resuming(tmp_path):
    """``--resume`` must never archive: the resume path needs the latest_* file."""
    root = tmp_path / "models"
    root.mkdir()
    _stage_stale_run(root)

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=True,
        skip_archive_on_fresh_start=False,
        archive_enabled_in_config=True,
    )

    assert dest is None
    assert (root / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.pt").exists()
    assert not (root / "_archive").exists()


@pytest.mark.unit
def test_if_needed_no_op_when_skip_archive_flag_set(tmp_path):
    """``--skip-archive-on-fresh-start`` is the user opt-out for fresh starts."""
    root = tmp_path / "models"
    root.mkdir()
    _stage_stale_run(root)

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=False,
        skip_archive_on_fresh_start=True,
        archive_enabled_in_config=True,
    )

    assert dest is None
    assert (root / "progress.jsonl").exists()
    assert not (root / "_archive").exists()


@pytest.mark.unit
def test_if_needed_no_op_when_disabled_in_config(tmp_path):
    """``run.archive_model_dir_before_fresh_start: false`` is the config opt-out."""
    root = tmp_path / "models"
    root.mkdir()
    _stage_stale_run(root)

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=False,
        skip_archive_on_fresh_start=False,
        archive_enabled_in_config=False,
    )

    assert dest is None
    assert (root / "progress.jsonl").exists()
    assert not (root / "_archive").exists()


@pytest.mark.unit
def test_if_needed_returns_none_when_nothing_to_archive(tmp_path):
    """Empty model_dir on a fresh start: no archive directory created."""
    root = tmp_path / "models"
    root.mkdir()

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=False,
        skip_archive_on_fresh_start=False,
        archive_enabled_in_config=True,
    )

    assert dest is None
    assert not (root / "_archive").exists()


@pytest.mark.unit
def test_if_needed_resume_takes_precedence_over_skip_flag(tmp_path):
    """``resume`` short-circuits before any other flag is consulted, even when
    callers accidentally set both. Documents the precedence order."""
    root = tmp_path / "models"
    root.mkdir()
    _stage_stale_run(root)

    dest = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=True,
        skip_archive_on_fresh_start=True,
        archive_enabled_in_config=False,
    )

    assert dest is None
    assert (root / "checkpoint_trainer_latest_20260501_ep14001_reward0.3911.pt").exists()


@pytest.mark.unit
def test_if_needed_runs_dir_passes_through_include_flag(tmp_path):
    """The ``include_tensorboard_runs`` knob must reach the underlying helper."""
    root = tmp_path / "models"
    root.mkdir()
    runs = root / "runs"
    runs.mkdir()
    (runs / "events.out").write_text("tb")
    _stage_stale_run(root)

    dest_kept = archive_model_dir_for_fresh_training_if_needed(
        root,
        resume_training_flag=False,
        skip_archive_on_fresh_start=False,
        archive_enabled_in_config=True,
        include_tensorboard_runs=False,
    )
    assert dest_kept is not None
    assert (runs / "events.out").exists()
    assert not (dest_kept / "runs").exists()

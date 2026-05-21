"""Archive prior ``model_dir`` artifacts before a fresh training run.

Without this, old ``checkpoint_trainer_latest_*`` files from a previous run
(higher episode numbers) make ``latest_checkpoint_keep_last_n`` rotation drop
new-run checkpoints immediately. Moving checkpoints, progress logs, and related
sidecars into ``<model_dir>/_archive/pre_fresh_<tag>/`` clears the deck while
preserving a recoverable snapshot.
"""

from __future__ import annotations

import secrets
import shutil
from datetime import datetime
from pathlib import Path

from momentum_core.logging import get_logger

logger = get_logger(__name__)


def _is_checkpoint_buffer_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("checkpoint_trainer_") and path.name.endswith(".buffer")


def _should_archive_top_level(path: Path, *, include_tensorboard_runs: bool) -> bool:
    if path.name == "_archive":
        return False
    if path.name == "runs" and path.is_dir():
        return include_tensorboard_runs
    if path.name == "progress.jsonl" and path.is_file():
        return True
    if not path.is_file() and not path.is_dir():
        return False
    if path.is_file():
        name = path.name
        return (
            (name.startswith("validation_results_") and name.endswith(".json"))
            or (name.startswith("test_results_") and name.endswith(".json"))
            or (name.startswith("trades_") and name.endswith(".jsonl"))
            or (name.startswith("checkpoint_trainer_") and name.endswith(".pt"))
            or (name.startswith("rainbow_transformer_") and name.endswith(".pt"))
        )
    if path.is_dir():
        return _is_checkpoint_buffer_dir(path)
    return False


def archive_model_dir_for_fresh_training(
    model_dir: str | Path,
    *,
    include_tensorboard_runs: bool = False,
) -> Path | None:
    """Move training artifacts into ``model_dir/_archive/pre_fresh_<tag>/``.

    Only **top-level** children of ``model_dir`` are considered (nothing under
    ``runs/`` or ``_archive/`` is walked). Returns the archive directory path
    when at least one item was moved, otherwise ``None``.

    ``include_tensorboard_runs=True`` moves the entire ``runs/`` directory
    (TensorBoard event files); it can be large.
    """
    root = Path(model_dir).resolve()
    if not root.is_dir():
        return None

    candidates = sorted(
        p for p in root.iterdir() if _should_archive_top_level(p, include_tensorboard_runs=include_tensorboard_runs)
    )
    if not candidates:
        logger.info("Fresh-run archive: nothing to move under %s (skipping).", root)
        return None

    tag = f"pre_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    dest_root = root / "_archive" / tag
    dest_root.mkdir(parents=True, exist_ok=False)

    logger.info(
        "Fresh-run archive: moving %d top-level artifact(s) from %s to %s",
        len(candidates),
        root,
        dest_root,
    )
    for src in candidates:
        dest = dest_root / src.name
        try:
            shutil.move(str(src), str(dest))
        except OSError as exc:
            logger.error("Failed to move %s -> %s: %s", src, dest, exc, exc_info=True)
            raise
        logger.info("  archived %s", src.name)

    return dest_root


def archive_model_dir_for_fresh_training_if_needed(
    model_dir: str | Path,
    *,
    resume_training_flag: bool,
    skip_archive_on_fresh_start: bool,
    archive_enabled_in_config: bool,
    include_tensorboard_runs: bool = False,
) -> Path | None:
    """Apply the gating policy and conditionally archive ``model_dir``.

    This is the single source of truth for the four-way decision:

    * ``resume_training_flag=True`` -> never archive (the resume path needs the
      existing ``checkpoint_trainer_latest_*`` and its buffer side-car).
    * ``skip_archive_on_fresh_start=True`` -> opt-out; user took responsibility
      for ensuring the model_dir is clean.
    * ``archive_enabled_in_config=False`` -> config opt-out
      (``run.archive_model_dir_before_fresh_start: false``).
    * Otherwise -> delegate to :func:`archive_model_dir_for_fresh_training`.

    Returning the same ``None`` / ``Path`` contract as the underlying helper
    keeps callers simple.

    Why this exists as a separate helper from the bare archive function:
    callers (``run_training``) used to inline this gating, which made it
    impossible to unit-test without spinning up the whole training entrypoint.
    A previous fresh run silently lost ~7400 episodes of checkpoints because
    the gating was bypassed (the archive call lived behind a config flag that
    pre-existed but the *invocation* was added later); regression coverage at
    the gating layer is the intended fix.
    """
    if resume_training_flag:
        logger.debug("Skipping fresh-run archive: --resume in progress.")
        return None
    if skip_archive_on_fresh_start:
        logger.info("Skipping fresh-run archive: --skip-archive-on-fresh-start was set.")
        return None
    if not archive_enabled_in_config:
        logger.info("Skipping fresh-run archive: run.archive_model_dir_before_fresh_start=false in config.")
        return None
    return archive_model_dir_for_fresh_training(
        model_dir,
        include_tensorboard_runs=include_tensorboard_runs,
    )

"""Auto-split from trainer.py (Tier 3.1).

Do not add new methods here directly; extend the mixin class in-place or
refactor through the facade. See .cursor/plans/prioritized-codebase-cleanup_*.plan.md.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from momentum_core.logging import get_logger

from .utils.memory_utils import current_rss_gb, release_memory_to_os

logger = get_logger(__name__)


class CheckpointMixin:
    """Checkpoint save/rotate/finalize methods split from the monolithic trainer."""

    @staticmethod
    def _buffer_sidecar_dir_for(checkpoint_path: str | Path) -> Path:
        """Return the side-car directory path for a given ``.pt`` checkpoint path.

        Convention: ``<name>.pt`` → ``<name>.buffer/``. Keeping them
        co-located makes rotation straightforward (one predictable sibling
        to delete per checkpoint) and makes the pairing obvious on disk.
        """
        return Path(checkpoint_path).with_suffix(".buffer")

    def _save_buffer_sidecar(self, checkpoint_path: str | Path) -> Path | None:
        """Persist the replay buffer next to the given checkpoint ``.pt`` file.

        Returns the side-car directory Path on success, or None if the
        attached buffer does not support side-car persistence (e.g. a
        mock in tests). Raises on real I/O failure so the caller can skip
        the ``.pt`` save and avoid committing a checkpoint with no buffer.
        """
        buffer = getattr(self.agent, "buffer", None)
        if buffer is None or not hasattr(buffer, "save_to_path"):
            return None
        sidecar_dir = self._buffer_sidecar_dir_for(checkpoint_path)
        buffer.save_to_path(sidecar_dir)
        return sidecar_dir

    def _rotate_latest_checkpoints(self) -> list[Path]:
        """Delete `checkpoint_trainer_latest_*` files older than the keep-N window.

        Episode number (from the canonical
        ``checkpoint_trainer_latest_<DATE>_ep<N>_reward<...>.pt`` filename) is
        the sort key, **not** mtime — episode numbers are monotonically
        non-decreasing with training progress and survive timezone /
        clock-skew weirdness, while mtime can be reset by file moves.

        ``best_*.pt`` files are never touched: those are the curated "model
        I want to keep" snapshots and the user / recovery script may rely on
        them. The recover-script outputs (``rewardrecover`` marker) are
        intentionally included in the latest-stream rotation because they
        are themselves `latest_*` files written with bumped episode numbers.

        Returns the list of paths that were deleted (for logging / tests).
        Silently returns ``[]`` if rotation is disabled (``keep_last_n=0``).
        """
        keep_n = self.latest_checkpoint_keep_last_n
        if keep_n <= 0:
            return []

        model_dir = Path(self.run_config.get("model_dir", "models"))
        try:
            candidates = list(model_dir.glob("checkpoint_trainer_latest_*_ep*_reward*.pt"))
        except OSError as exc:
            logger.warning("Checkpoint rotation: failed to list %s: %s", model_dir, exc)
            return []

        # Pull the episode number out of the filename. Files that don't match
        # the canonical pattern are skipped (left untouched) rather than
        # deleted — safer to leave an unrecognised file in place.
        ep_re = re.compile(r"^checkpoint_trainer_latest_\d{8}_ep(\d+)_reward.+\.pt$")
        episode_files: list[tuple[int, Path]] = []
        skipped: list[Path] = []
        for path in candidates:
            m = ep_re.match(path.name)
            if not m:
                skipped.append(path)
                continue
            try:
                episode_files.append((int(m.group(1)), path))
            except ValueError:
                skipped.append(path)
        if skipped:
            logger.debug(
                "Checkpoint rotation: skipping %d non-canonical filename(s): %s",
                len(skipped),
                [p.name for p in skipped],
            )

        if len(episode_files) <= keep_n:
            return []

        # Highest episode first; everything after the keep-N window gets pruned.
        episode_files.sort(key=lambda item: item[0], reverse=True)
        to_delete = [path for _, path in episode_files[keep_n:]]
        deleted: list[Path] = []
        deleted_sidecars: list[Path] = []
        for path in to_delete:
            try:
                path.unlink()
                deleted.append(path)
            except OSError as exc:
                logger.warning(
                    "Checkpoint rotation: failed to delete %s: %s",
                    path,
                    exc,
                )
                # Don't try to delete the side-car if the .pt itself failed
                # -- keep the pair intact so the file is still resumable.
                continue
            # Delete the matching buffer side-car (if any). A ~9 GiB
            # side-car per checkpoint dwarfs the .pt file, so leaving
            # stale ones behind would quickly fill the model directory.
            sidecar_dir = self._buffer_sidecar_dir_for(path)
            if sidecar_dir.is_dir():
                try:
                    shutil.rmtree(sidecar_dir)
                    deleted_sidecars.append(sidecar_dir)
                except OSError as exc:
                    logger.warning(
                        "Checkpoint rotation: failed to delete buffer side-car %s: %s",
                        sidecar_dir,
                        exc,
                    )
        if deleted:
            logger.info(
                "Checkpoint rotation: pruned %d old `latest_*` checkpoint(s) (kept the most recent %d). Deleted: %s",
                len(deleted),
                keep_n,
                [p.name for p in deleted],
            )
        if deleted_sidecars:
            logger.info(
                "Checkpoint rotation: pruned %d old buffer side-car(s): %s",
                len(deleted_sidecars),
                [p.name for p in deleted_sidecars],
            )
        return deleted

    def _flush_writer(self) -> None:
        """Force-flush TensorBoard events to disk.

        The default ``SummaryWriter`` buffers scalars in-memory for up to
        ``flush_secs`` (120s) or until ``max_queue`` events accumulate. If the
        machine hard-freezes (GPU driver hang, power loss, kernel deadlock) the
        buffered events are lost even though the writer never crashed, which
        produces a gap in TensorBoard between the last flush and the freeze.
        Calling this after every checkpoint / episode guarantees the on-disk
        event file is in lockstep with the on-disk checkpoint state, so the
        largest gap we can ever lose is the few events added since the last
        flush call (rather than up to 2 minutes of training).

        Safe to call when ``self.writer`` is ``None``; errors are swallowed.
        """
        writer = getattr(self, "writer", None)
        if writer is None:
            return
        try:
            writer.flush()
        except (OSError, RuntimeError, ValueError, AttributeError):  # pragma: no cover - defensive
            logger.debug("Failed to flush TensorBoard writer", exc_info=True)

    def _save_checkpoint(
        self,
        episode: int,
        total_steps: int,
        is_best: bool,
        validation_score: float | None = None,  # Add optional validation score
    ):
        """Save trainer-specific checkpoint (episode, validation score, etc.)."""
        assert isinstance(episode, int) and episode >= 0, "Invalid episode number for checkpoint"
        assert isinstance(total_steps, int) and total_steps >= 0, "Invalid total_steps for checkpoint"
        assert isinstance(is_best, bool), "is_best flag must be boolean"

        # Get current date in YYYYMMDD format
        current_date = datetime.now().strftime("%Y%m%d")

        # Agent state (network, optimizer, agent_steps) is saved via agent.save_model()
        # This checkpoint primarily stores trainer state.
        #
        # IMPORTANT: ``buffer_state`` is deliberately NOT embedded in this
        # dict. For a 1M-entry PER buffer the deque holds ~6 GiB of live
        # numpy arrays; pickling it inside torch.save produced a 10+ GiB
        # transient in-memory pickle stream on top of the live buffer,
        # which was the direct cause of repeated OOM kills (~49 GiB
        # anon-rss) during scheduled saves. The buffer is now persisted
        # as a side-car directory (``<checkpoint>.buffer/`` containing
        # per-field ``.npy`` memmaps), see ``_save_buffer_sidecar``. The
        # ``buffer_sidecar_relpath`` field below is the resume path's hint
        # for where to find that side-car; ``buffer_state`` remains as a
        # ``None`` sentinel so older resume code paths fail loudly rather
        # than silently loading a stale buffer.
        checkpoint = {
            "episode": episode,
            "total_train_steps": total_steps,  # Store steps from trainer perspective
            "best_validation_metric": self.best_validation_metric,
            "early_stopping_counter": self.early_stopping_counter,
            "buffer_state": None,
            "buffer_sidecar_relpath": None,  # filled in after side-car is written
            # --- ADDED Agent State ---
            "agent_config": self.agent.config,
            "agent_total_steps": self.agent.total_steps,
            "agent_env_steps": self.agent.env_steps,
            "total_steps": self.agent.total_steps,
            "network_state_dict": self.agent.network.state_dict() if self.agent.network is not None else None,
            "target_network_state_dict": (self.agent.target_network.state_dict() if self.agent.target_network is not None else None),
            "optimizer_state_dict": (self.agent.optimizer.state_dict() if self.agent.optimizer else None),
            "scaler_state_dict": None,
            # --- ADDED Scheduler State ---
            "scheduler_state_dict": (
                self.agent.scheduler.state_dict() if self.agent.scheduler and self.agent.lr_scheduler_enabled else None
            ),
            # --- END ADDED Scheduler State ---
            # --- END ADDED Agent State ---
        }

        if self.writer:
            checkpoint["tensorboard_log_dir"] = getattr(self.writer, "log_dir", None)

        # Optionally add current validation score to the checkpoint data if available
        if validation_score is not None:
            assert isinstance(validation_score, float), "Validation score must be float if provided"
            checkpoint["validation_score"] = validation_score

        # Basic check on checkpoint contents
        # assert isinstance(checkpoint['network_state_dict'], dict), "Invalid network state dict in checkpoint"
        # assert isinstance(checkpoint['optimizer_state_dict'], dict), "Invalid optimizer state dict in checkpoint"
        assert isinstance(checkpoint["best_validation_metric"], float), "Invalid best validation metric type in checkpoint"

        # Reverted: Removed checks for mock objects here
        if not self.agent.optimizer:
            logger.warning("Agent optimizer not initialized, cannot save checkpoint.")
            return
        if self.agent.network is None or self.agent.target_network is None:
            logger.warning("Agent networks not initialized, cannot save checkpoint.")
            return

        # Snapshot RSS before save so the post-trim log shows whether
        # release_memory_to_os actually reclaimed anything from the transient
        # torch.save pickle spike.
        rss_before_save = current_rss_gb()

        # Save the latest checkpoint with date, episode and reward in filename.
        #
        # Order matters here: we write the buffer side-car FIRST, then the
        # ``.pt`` file. If the side-car write fails, we bail out before
        # committing the ``.pt`` -- this preserves the invariant that every
        # ``.pt`` file on disk has a matching intact side-car. The reverse
        # order could leave a ``.pt`` whose resume path would have no
        # buffer to attach to.
        try:
            # Construct filename with date, episode and reward
            latest_checkpoint_path = f"{self.latest_trainer_checkpoint_path.rsplit('.', 1)[0]}_{current_date}_ep{episode}_reward{self.best_validation_metric:.4f}.pt"
            try:
                sidecar_dir = self._save_buffer_sidecar(latest_checkpoint_path)
            except Exception as sidecar_exc:  # noqa: BLE001 — logged and re-raised as a skip
                logger.error(
                    "Buffer side-car write to %s failed; skipping this checkpoint so .pt and buffer stay paired.",
                    self._buffer_sidecar_dir_for(latest_checkpoint_path),
                    exc_info=sidecar_exc,
                )
                raise
            if sidecar_dir is not None:
                # Store as a relative path (filename only) so that moving
                # the models/ directory doesn't invalidate old checkpoints.
                checkpoint["buffer_sidecar_relpath"] = sidecar_dir.name
            torch.save(checkpoint, latest_checkpoint_path)
            self._checkpoints_saved_this_run += 1
            logger.info(f"Latest checkpoint saved to {latest_checkpoint_path}")
            if sidecar_dir is not None:
                logger.info(f"  Buffer side-car: {sidecar_dir}")
            logger.info(f"  Episode: {episode}")
            logger.info(f"  Total Steps: {total_steps}")
            logger.info(f"  Best Validation Score: {self.best_validation_metric:.4f}")
            logger.info(f"  Early Stopping Counter: {self.early_stopping_counter}")
            # Disk hygiene: prune older latest_* checkpoints once the new one
            # is safely on disk. Only runs if `latest_checkpoint_keep_last_n > 0`.
            try:
                self._rotate_latest_checkpoints()
            except Exception:  # noqa: BLE001 — never let rotation break a save
                logger.warning("Checkpoint rotation raised; continuing.", exc_info=True)
        except Exception as e:
            logger.error(f"Error saving latest checkpoint: {e}", exc_info=True)  # Kept traceback

        # Save the best checkpoint if this is the best model so far
        if is_best:
            # Construct the filename with date, episode and score
            if validation_score is not None:
                best_checkpoint_path = (
                    f"{self.best_trainer_checkpoint_base_path}_{current_date}_ep{episode}_score_{validation_score:.4f}.pt"
                )
            else:
                # Fallback if score isn't passed (shouldn't happen for best)
                best_checkpoint_path = f"{self.best_trainer_checkpoint_base_path}_{current_date}_ep{episode}.pt"
            try:
                # Best checkpoints are never rotated and accumulate one file
                # per validation improvement, so we intentionally do NOT
                # drop a ~9 GiB buffer side-car next to each one -- disk
                # usage would explode during early training. The ``.pt``
                # stores a ``None`` sidecar pointer so a resume from best
                # fails fast with a clear "no buffer" error rather than
                # silently training on an empty buffer. Latest-stream
                # checkpoints are the supported resume path; to resume
                # from a best.pt, copy the nearest latest_*.buffer/ by
                # hand.
                checkpoint["buffer_sidecar_relpath"] = None
                # Use the dynamically constructed path
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Best checkpoint saved to {best_checkpoint_path}")
                logger.info("  (no buffer side-car written for best; resume from latest_* for full buffer state)")
                # Log the actual score achieved that triggered this save
                if validation_score is not None:
                    self.best_validation_metric = validation_score
                    # --- MODIFIED: Construct filename with score ---
                    # best_model_save_prefix = f"{self.best_model_base_prefix}_{current_date}_ep{episode}_score_{validation_score:.4f}"
                    # self.agent.save_model(best_model_save_prefix) # REMOVED - Agent state is in checkpoint
                    # --- END MODIFICATION ---
                    logger.info(f"  Score: {validation_score:.4f}")
                    logger.info(f"  Best checkpoint with agent state saved to: {best_checkpoint_path}")  # Modified log message
                else:
                    logger.info("  No improvement over previous best model")
            except Exception as e:
                logger.error(f"Error saving best checkpoint: {e}", exc_info=True)  # Kept traceback

        # Keep TB event file in sync with the checkpoint we just flushed to
        # disk. Without this, a hard freeze right after a checkpoint save can
        # leave TB scalars stuck ~2 minutes behind the checkpoint's
        # ``total_train_steps``, producing the visible step-axis gap on resume.
        self._flush_writer()

        # Reclaim the pickle-buffer pages that torch.save just freed back
        # inside glibc's arenas. Without this, each save ratchets steady-state
        # RSS up by several GiB even though the bytes are no longer in use,
        # which was the direct cause of the Apr 22 / Apr 23 OOM kills at
        # ~54 GiB anon-rss. Drop the local reference first so gc.collect
        # has nothing holding the large state_dict values alive.
        del checkpoint
        rss_before_trim = current_rss_gb()
        released = release_memory_to_os()
        rss_after_trim = current_rss_gb()
        if rss_before_save is not None and rss_after_trim is not None:
            logger.info(
                "RSS after checkpoint save: %.2f GiB (pre-save %.2f GiB, pre-trim %.2f GiB); malloc_trim released pages: %s.",
                rss_after_trim,
                rss_before_save,
                rss_before_trim if rss_before_trim is not None else float("nan"),
                released,
            )

    def _finalize_training(self, total_train_steps: int, num_episodes: int, val_files: list[Path]):
        """Saves final model and logs overall training summary."""
        # Guard: only overwrite ``rainbow_transformer_final_agent_state.pt`` if
        # *this* invocation persisted at least one trainer checkpoint. Without
        # this, any early crash (failed resume, init exception, OOM before the
        # first checkpoint-save cadence, etc.) reaches this ``finally`` block
        # and clobbers a previously-trained ``_final`` with untrained weights.
        # A prior ``_final`` from a successful run stays intact; a legit long
        # training run will have saved many checkpoints and still writes
        # ``_final`` as before.
        final_model_prefix = str(Path(self.model_dir) / "rainbow_transformer_final")
        final_state_path = f"{final_model_prefix}_agent_state.pt"
        if self._checkpoints_saved_this_run == 0:
            if os.path.exists(final_state_path):
                logger.warning(
                    "Skipping save of %s: no trainer checkpoints were persisted in this run, "
                    "so the agent state is likely untrained / fresh and would clobber the "
                    "existing file. Leaving the previous final-agent state untouched.",
                    final_state_path,
                )
            else:
                logger.warning(
                    "Skipping save of %s: no trainer checkpoints were persisted in this run. "
                    "If this was a deliberately short run and you want the final agent state "
                    "written anyway, save a checkpoint before finalize.",
                    final_state_path,
                )
        else:
            try:
                self.agent.save_model(final_model_prefix)
                logger.info(f"Final agent model saved to {final_model_prefix}*")
            except Exception as e:
                logger.error(f"Error saving final agent model: {e}")

        # Final checkpoint save (optional, could rely on last periodic/best save)
        # self._save_checkpoint(...)

        logger.info("====== RAINBOW DQN TRAINING COMPLETED ======")
        logger.info(f"Total steps: {total_train_steps}")

        # --- ADDED: Log N-step reward statistics ---
        if hasattr(self.agent, "observed_n_step_rewards_history") and self.agent.observed_n_step_rewards_history:
            rewards_history = np.array(self.agent.observed_n_step_rewards_history)
            logger.info("--- N-Step Reward Statistics (Full Training Run) ---")
            logger.info(f"  Count: {len(rewards_history)}")
            logger.info(f"  Min: {np.min(rewards_history):.4f}")
            logger.info(f"  Max: {np.max(rewards_history):.4f}")
            logger.info(f"  Mean: {np.mean(rewards_history):.4f}")
            logger.info(f"  Median: {np.median(rewards_history):.4f}")
            logger.info(f"  5th Percentile: {np.percentile(rewards_history, 5):.4f}")
            logger.info(f"  95th Percentile: {np.percentile(rewards_history, 95):.4f}")
            logger.info(f"  Std Dev: {np.std(rewards_history):.4f}")
        else:
            logger.info("--- N-Step Reward Statistics: No history collected or attribute not found. ---")
        # --- END ADDED ---

        # Log best validation score achieved during training
        if val_files and self.best_validation_metric > -np.inf:
            logger.info(f"Best validation score during training: {self.best_validation_metric:.4f}")
            # The specific file for the best checkpoint is saved in _save_checkpoint
            logger.info(f"Best checkpoint base path: {self.best_trainer_checkpoint_base_path}*.pt")
        elif not val_files:
            logger.warning("Training completed without validation.")
            # Log best training reward if available (not currently tracked across episodes)
            # logger.info(f"Best average reward during training: {best_train_reward:.2f}")

        self.close_cached_environments()

    # --- END Refactored Helper Methods ---

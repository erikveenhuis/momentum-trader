import glob
import logging
import os
import zipfile
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("CheckpointUtils")


def _probe_checkpoint_usable(file_path: str) -> tuple[bool, str]:
    """Cheaply determine whether ``file_path`` is plausibly a loadable torch checkpoint.

    Returns ``(usable, reason)``. ``usable=False`` means ``torch.load`` will
    almost certainly fail on the file; we should skip it and try the
    next-oldest candidate instead of surfacing a ``ResumeFailedError``.

    The probe covers the two interrupted-save failure modes we've hit in
    production:

    * **Zero-byte stub** — the rotation code wrote the new filename before
      ``torch.save`` flushed any bytes to it (seen April 22 2026, OOM-kill).
    * **Truncated mid-write** — e.g. a ~479 MB fragment of an expected
      ~8.3 GB file where ``torch.save`` died part-way through flushing the
      pickled tensors (seen April 23 2026, same root cause). The file has
      bytes but the ZIP central directory record at the tail is missing,
      which ``torch.load`` reports as
      ``PytorchStreamReader failed reading zip archive: failed finding
      central directory``.

    The probe intentionally does NOT call ``torch.load`` — that would cost
    multiple GB of RAM for an 8 GB checkpoint and defeat the purpose of a
    cheap triage step. ``zipfile.is_zipfile`` just seeks to the end of the
    file looking for the EOCD signature, so it's O(1) in checkpoint size.

    PyTorch's ``torch.save`` (since 1.6) writes a standard ZIP archive that
    ``zipfile.is_zipfile`` recognises; if this codebase ever regresses to
    legacy tar-pickle checkpoints, tighten the probe rather than loosen the
    caller.
    """
    try:
        size = os.path.getsize(file_path)
    except OSError as exc:
        return False, f"stat failed: {exc}"
    if size == 0:
        return False, "zero-byte file (likely partial write from an interrupted torch.save)"
    try:
        if not zipfile.is_zipfile(file_path):
            return False, (
                "not a valid ZIP archive (central directory missing); torch.save was likely "
                "truncated mid-write -- torch.load would raise PytorchStreamReader error"
            )
    except OSError as exc:
        return False, f"read failed while probing ZIP structure: {exc}"
    return True, ""


def find_latest_checkpoint(model_dir: str = "models", model_prefix: str = "checkpoint_trainer") -> str | None:
    """Finds the latest checkpoint file based on episode number in filename.

    Uses episode number rather than modification time because:
    - Episode number directly indicates training progress
    - Modification time can be unreliable (files touched without being latest)
    - In this training setup, episodes have consistent step counts (~14K steps each)
    - Faster than loading checkpoints to check total_steps

    Alternative approaches considered:
    - Total training steps: More precise but requires loading each checkpoint
    - Validation score: Would resume from best model, not latest progress
    - Modification time: Unreliable as demonstrated by the ep10 file issue
    """
    import re

    # First try to find the latest checkpoint with the new naming pattern
    # Look for files matching the pattern: checkpoint_trainer_latest_YYYYMMDD_epXXX_rewardX.XXXX.pt
    pattern = os.path.join(model_dir, f"{model_prefix}_latest_*_ep*_reward*.pt")
    matching_files = glob.glob(pattern)

    if matching_files:
        # Extract episode numbers from filenames and find the one with highest episode
        episode_files = []
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            # Match pattern: checkpoint_trainer_latest_YYYYMMDD_ep{episode}_reward{reward}.pt
            match = re.search(r"_ep(\d+)_", filename)
            if match:
                episode_num = int(match.group(1))
                episode_files.append((episode_num, file_path))

        if episode_files:
            # Sort by episode number (highest first) and walk down until we find
            # a candidate that passes the usability probe. Interrupted saves
            # (zero-byte stubs OR truncated ZIPs missing the central directory)
            # would otherwise get returned here, then blow up in torch.load and
            # -- via the new ResumeFailedError gate in run_training.py -- abort
            # the run. Falling back to the next-oldest complete file keeps
            # ``--resume`` working unattended across OOM-during-save events.
            episode_files.sort(key=lambda x: x[0], reverse=True)
            for episode_num, file_path in episode_files:
                usable, reason = _probe_checkpoint_usable(file_path)
                if not usable:
                    logger.warning(
                        "Skipping checkpoint %s (ep%d): %s. Trying next-oldest.",
                        file_path,
                        episode_num,
                        reason,
                    )
                    continue
                logger.info(f"Found latest checkpoint (episode {episode_num}): {file_path}")
                return file_path
            logger.warning(
                "All %d candidate checkpoints under %s/%s_latest_*_ep*_reward*.pt failed the "
                "usability probe (empty, truncated, or unreadable).",
                len(episode_files),
                model_dir,
                model_prefix,
            )

    # Fallback to old naming pattern
    latest_path = os.path.join(model_dir, f"{model_prefix}_latest.pt")
    if os.path.exists(latest_path):
        usable, reason = _probe_checkpoint_usable(latest_path)
        if usable:
            logger.info(f"Found latest checkpoint with old naming pattern: {latest_path}")
            return latest_path
        logger.warning("Skipping legacy checkpoint %s: %s.", latest_path, reason)

    best_path = os.path.join(model_dir, f"{model_prefix}_best.pt")
    if os.path.exists(best_path):
        usable, reason = _probe_checkpoint_usable(best_path)
        if usable:
            logger.warning(f"Latest checkpoint not found, using best checkpoint: {best_path}")
            return best_path
        logger.warning("Skipping best-checkpoint fallback %s: %s.", best_path, reason)

    logger.warning("No suitable checkpoint file found.")
    return None


def load_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
    """Loads a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The loaded checkpoint dictionary, or None if loading fails.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
        return None

    logger.info(f"Attempting to load checkpoint: {checkpoint_path}")
    try:
        # Load to CPU first to avoid GPU memory issues if loading on different device.
        # weights_only=False: PyTorch>=2.6 defaults True; trainer checkpoints pickle buffer objects (e.g. Experience).
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Basic validation of checkpoint structure.
        # optimizer_state_dict / scheduler_state_dict / scaler_state_dict are
        # intentionally NOT required: --reset-lr-on-resume pops them after load,
        # and scripts/recover_from_collapse.py --strip-optimizer bakes that same
        # removal into the file. Both flows expect the trainer downstream to
        # handle the absence (it does — the optimizer is freshly constructed
        # from config when no state is present).
        required_keys = [
            "episode",
            "total_train_steps",
            "network_state_dict",
            "best_validation_metric",
            "target_network_state_dict",
            "agent_total_steps",
            "early_stopping_counter",
            "agent_config",
        ]
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            logger.error(f"Checkpoint {checkpoint_path} is missing required keys: {missing}. Cannot resume.")
            return None

        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Resuming from Episode: {checkpoint.get('episode', 'N/A')}")
        logger.info(f"  Resuming from Total Steps: {checkpoint.get('total_train_steps', 'N/A')}")
        logger.info(f"  Previous Best Validation Score: {checkpoint.get('best_validation_metric', -np.inf):.4f}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}.", exc_info=True)
        return None

import glob
import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("CheckpointUtils")


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
            # Sort by episode number (highest first)
            episode_files.sort(key=lambda x: x[0], reverse=True)
            latest_episode, latest_path = episode_files[0]
            logger.info(f"Found latest checkpoint (episode {latest_episode}): {latest_path}")
            return latest_path

    # Fallback to old naming pattern
    latest_path = os.path.join(model_dir, f"{model_prefix}_latest.pt")
    if os.path.exists(latest_path):
        logger.info(f"Found latest checkpoint with old naming pattern: {latest_path}")
        return latest_path

    best_path = os.path.join(model_dir, f"{model_prefix}_best.pt")
    if os.path.exists(best_path):
        logger.warning(f"Latest checkpoint not found, using best checkpoint: {best_path}")
        return best_path

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

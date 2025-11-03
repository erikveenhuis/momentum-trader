"""Utilities for loading trained agents for live inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger

LOGGER = get_logger("momentum_live.agent_loader")

_SCORE_PATTERN = re.compile(r"_score_(?P<score>-?\d+(?:\.\d+)?)")


def _extract_score(path: Path) -> float:
    match = _SCORE_PATTERN.search(path.name)
    if not match:
        return float("-inf")
    try:
        return float(match.group("score"))
    except ValueError:
        return float("-inf")


def find_best_checkpoint(
    models_dir: str | Path,
    pattern: str = "checkpoint_trainer_best_*.pt",
) -> Path:
    """Locate the checkpoint with the highest validation score."""

    models_path = Path(models_dir)
    candidates = sorted(models_path.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching '{pattern}' were found in '{models_path}'.")

    best = max(candidates, key=lambda path: (_extract_score(path), path.stat().st_mtime))
    LOGGER.info("Selected checkpoint %s for live inference", best)
    return best


def _resolve_device(device: Optional[str | torch.device]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def load_agent_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: Optional[str | torch.device] = None,
) -> RainbowDQNAgent:
    """Instantiate and load a trained ``RainbowDQNAgent`` from a checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Try different config key names
    agent_config = None
    for config_key in ["agent_config", "config"]:
        if config_key in checkpoint:
            agent_config = checkpoint[config_key]
            break

    if agent_config is None:
        raise KeyError("Checkpoint is missing agent config ('agent_config' or 'config'); cannot reconstruct agent")

    agent_config = agent_config.copy()
    agent_config.setdefault("seed", 42)

    resolved_device = _resolve_device(device)
    LOGGER.info("Instantiating agent on %s", resolved_device)
    agent = RainbowDQNAgent(config=agent_config, device=resolved_device, scaler=None)

    loaded = agent.load_state(checkpoint)
    if not loaded:
        raise RuntimeError(f"Failed to load agent state from checkpoint {checkpoint_path}")

    agent.set_training_mode(False)
    return agent


__all__ = ["find_best_checkpoint", "load_agent_from_checkpoint"]

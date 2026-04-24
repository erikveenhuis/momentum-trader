"""Utilities for loading trained agents for live inference."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger

LOGGER = get_logger(__name__)

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


def resolve_live_device(device: str | torch.device | None) -> str:
    """Pick device for live inference. Defaults to CPU; set ``MOMENTUM_LIVE_DEVICE=cuda`` for GPU."""

    if device is not None:
        if isinstance(device, torch.device):
            return device.type
        return str(device)

    env = os.getenv("MOMENTUM_LIVE_DEVICE", "").strip().lower()
    if env in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            LOGGER.warning("MOMENTUM_LIVE_DEVICE requests CUDA but torch.cuda.is_available() is false; using CPU")
            return "cpu"
        return "cuda"
    return "cpu"


def load_agent_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    inference_only: bool = True,
) -> RainbowDQNAgent:
    """Instantiate and load a trained ``RainbowDQNAgent`` from a checkpoint.

    Live inference uses ``inference_only=True`` (eager forward, CPU allowed). Training code must
    instantiate the agent with ``inference_only=False`` on CUDA.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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

    resolved_device = resolve_live_device(device)
    LOGGER.info("Instantiating agent on %s (inference_only=%s)", resolved_device, inference_only)
    agent = RainbowDQNAgent(
        config=agent_config,
        device=resolved_device,
        inference_only=inference_only,
    )

    loaded = agent.load_state(checkpoint)
    if not loaded:
        raise RuntimeError(f"Failed to load agent state from checkpoint {checkpoint_path}")

    agent.set_training_mode(False)
    return agent


__all__ = ["find_best_checkpoint", "load_agent_from_checkpoint", "resolve_live_device"]

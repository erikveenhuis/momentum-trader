"""Factory and helpers for vectorized training environments.

Uses SyncVectorEnv with DISABLED autoreset so the trainer controls
episode boundaries and per-env data file assignment.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from momentum_env import TradingEnv, TradingEnvConfig

from .data import DataManager

logger = logging.getLogger("Trainer")


def _make_train_env(env_config: dict[str, Any], data_path: str) -> TradingEnv:
    cfg = dict(env_config)
    cfg["data_path"] = data_path
    return TradingEnv(config=TradingEnvConfig(**cfg))


def create_vector_env(
    num_envs: int,
    env_config: dict[str, Any],
    data_manager: DataManager,
) -> gym.vector.SyncVectorEnv:
    """Create a SyncVectorEnv with DISABLED autoreset.

    Each sub-env is constructed with a real training file.  The trainer
    manages episode boundaries by detecting done flags and calling
    ``reset_done_envs`` to inject new data files per sub-env.
    """
    training_files = data_manager.get_training_files()
    if not training_files:
        raise RuntimeError("No training files available for vector env construction")

    env_fns = []
    for i in range(num_envs):
        init_file = str(training_files[i % len(training_files)])
        env_fns.append(lambda p=init_file: _make_train_env(env_config, p))

    return gym.vector.SyncVectorEnv(
        env_fns,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def reset_done_envs(
    vec_env: gym.vector.SyncVectorEnv,
    done_mask: np.ndarray,
    data_manager: DataManager,
    curriculum_frac: float,
) -> dict[str, np.ndarray]:
    """Reset sub-envs indicated by *done_mask*, injecting fresh data files.

    First loads new market data into each done sub-env via direct reset,
    then issues the vector-level ``reset(options={"reset_mask": ...})``
    to clear the internal ``_autoreset_envs`` flags.  The vector-level
    reset triggers a second portfolio-init on the already-loaded data,
    which is cheap and keeps bookkeeping consistent.

    Returns the stacked observation dict from the vector reset.
    """
    for i in range(vec_env.num_envs):
        if not done_mask[i]:
            continue
        new_path = str(data_manager.get_random_training_file(curriculum_frac=curriculum_frac))
        vec_env.envs[i].reset(options={"data_path": new_path})

    obs, _info = vec_env.reset(options={"reset_mask": done_mask})
    return obs

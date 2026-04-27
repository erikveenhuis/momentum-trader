"""Checkpoint save/load helpers for :class:`RainbowDQNAgent`.

Extracted from ``agent.py`` to separate the (sizeable) I/O and compile-wrapper
state-dict plumbing from the hot training path. The module-level
``_maybe_unwrap_orig_mod_state_dict`` /
``_maybe_wrap_orig_mod_state_dict`` /
``_load_state_dict_with_orig_mod_fallback`` helpers live here as well; they
are re-exported from ``agent.py`` for backward compatibility with existing
tests and external callers.
"""

from __future__ import annotations

import os

import torch
from momentum_core.logging import get_logger

logger = get_logger(__name__)


def _maybe_unwrap_orig_mod_state_dict(state_dict: dict) -> dict:
    """If ``state_dict`` was saved from ``torch.compile`` (``OptimizedModule``), keys are ``_orig_mod.*``; map to plain module keys."""

    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(isinstance(k, str) and k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def _maybe_wrap_orig_mod_state_dict(state_dict: dict) -> dict:
    """Reverse of :func:`_maybe_unwrap_orig_mod_state_dict`.

    If ``state_dict`` has plain keys but we're loading into an
    ``OptimizedModule``, add the ``_orig_mod.`` prefix back. Returns the
    original dict unchanged if any key already has the prefix (so we don't
    double-prefix).
    """

    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if any(isinstance(k, str) and k.startswith("_orig_mod.") for k in keys):
        return state_dict
    return {f"_orig_mod.{k}": v for k, v in state_dict.items()}


_PRE_IQN_CHECKPOINT_MESSAGE = (
    "Pre-IQN (C51) checkpoint detected: state_dict has no `tau_embedding.linear.*` keys, "
    "which means it was saved before the Beyond-the-Rainbow IQN/Munchausen/Spectral-Norm upgrade. "
    "There is no in-place fallback to C51. To preserve training progress, run "
    "`python -m momentum_train.scripts.migrate_c51_to_iqn --src <c51.pt> --dst <iqn.pt>` to warm-start "
    "the new agent from the encoder/aux-head weights of this checkpoint."
)


def _is_pre_iqn_state_dict(state_dict: dict) -> bool:
    """Return True iff ``state_dict`` looks like a pre-IQN (C51) network snapshot.

    The IQN head adds ``tau_embedding.linear.weight``/``bias`` keys that did
    not exist in C51. We check for the absence of *any* tau-embedding key
    (after stripping a possible ``_orig_mod.`` prefix) and the presence of
    *some* network keys (so an empty dict is treated as a generic load
    failure rather than a stale C51 checkpoint).
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return False

    has_tau_embedding = False
    has_net_keys = False
    for key in state_dict:
        if not isinstance(key, str):
            continue
        bare = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        if bare.startswith("tau_embedding."):
            has_tau_embedding = True
            break
        if (
            bare.startswith("encoder.")
            or bare.startswith("value_stream.")
            or bare.startswith("advantage_stream.")
            or bare.startswith("aux_return_head.")
            or bare.startswith("input_projection.")
        ):
            has_net_keys = True
    return has_net_keys and not has_tau_embedding


def _assert_iqn_state_dict(state_dict: dict, *, context: str) -> None:
    """Raise a clear RuntimeError if ``state_dict`` is a pre-IQN snapshot."""
    if _is_pre_iqn_state_dict(state_dict):
        raise RuntimeError(f"[{context}] {_PRE_IQN_CHECKPOINT_MESSAGE}")


def _load_state_dict_with_orig_mod_fallback(module: torch.nn.Module, state_dict: dict) -> None:
    _assert_iqn_state_dict(state_dict, context="network load")
    try:
        module.load_state_dict(state_dict)
    except Exception:
        unwrapped = _maybe_unwrap_orig_mod_state_dict(state_dict)
        if unwrapped is not state_dict:
            try:
                module.load_state_dict(unwrapped)
                logger.info("Loaded weights from torch.compile checkpoint (_orig_mod keys) into eager module.")
                return
            except Exception:
                pass
        wrapped = _maybe_wrap_orig_mod_state_dict(state_dict)
        if wrapped is not state_dict:
            module.load_state_dict(wrapped)
            logger.info("Loaded weights from eager checkpoint (plain keys) into torch.compile module.")
            return
        raise


class AgentCheckpointMixin:
    """Save/load helpers attached to :class:`RainbowDQNAgent`."""

    def save_model(self, path_prefix):
        """Saves the agent's model and optimizer state."""
        if self.network is None or self.optimizer is None:
            logger.error("Network or optimizer not initialized. Cannot save model.")
            return

        # ``scaler_state_dict`` is no longer written. bfloat16 autocast
        # doesn't need a GradScaler. Old checkpoints that still carry the key
        # continue to load (see ``load_state`` which pops it).
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "config": self.config,
        }
        if self.scheduler and self.lr_scheduler_enabled:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        try:
            if path_prefix.endswith(".pt"):
                base_name = path_prefix[:-3]
                final_save_path = f"{base_name}_agent_state.pt"
            else:
                final_save_path = f"{path_prefix}_agent_state.pt"

            torch.save(checkpoint, final_save_path)
            logger.info(f"Unified agent checkpoint saved to {final_save_path}")
            logger.info(
                "  Includes: Network, Target Network, Optimizer, Scaler (if applicable), Scheduler (if applicable), Total Steps, Config"
            )

        except Exception as e:
            logger.error(f"Error saving unified agent checkpoint to {final_save_path}: {e}", exc_info=True)

    def load_model(self, path_prefix):
        """Loads the agent's model and optimizer state from a unified checkpoint."""
        if path_prefix.endswith(".pt"):
            base_name = path_prefix[:-3]
            checkpoint_path = f"{base_name}_agent_state.pt"
        else:
            checkpoint_path = f"{path_prefix}_agent_state.pt"

        if not os.path.exists(checkpoint_path):
            logger.error(f"Unified agent checkpoint file not found at {checkpoint_path}. Cannot load model.")
            return False

        try:
            logger.info(f"Attempting to load unified agent checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if "network_state_dict" in checkpoint and self.network is not None:
                _assert_iqn_state_dict(checkpoint["network_state_dict"], context="network load")
                self.network.load_state_dict(checkpoint["network_state_dict"])
                logger.info("Network state loaded.")
            else:
                logger.warning("Network state_dict not found in checkpoint or network not initialized.")
                return False

            if "target_network_state_dict" in checkpoint and self.target_network is not None:
                _assert_iqn_state_dict(checkpoint["target_network_state_dict"], context="target network load")
                self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
                logger.info("Target network state loaded.")
            else:
                logger.warning("Target network state_dict not found in checkpoint or target_network not initialized.")

            if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded.")
            else:
                logger.warning("Optimizer state_dict not found in checkpoint or optimizer not initialized.")

            if "total_steps" in checkpoint:
                self.total_steps = checkpoint["total_steps"]
                logger.info(f"Total steps loaded: {self.total_steps}")
            else:
                logger.warning("Total steps not found in checkpoint. Resetting to 0.")
                self.total_steps = 0

            if "scheduler_state_dict" in checkpoint and self.scheduler and self.lr_scheduler_enabled:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info("LR Scheduler state loaded.")
                except Exception as e:
                    logger.error(
                        f"Error loading LR scheduler state: {e}. Scheduler may not resume correctly.",
                        exc_info=True,
                    )
            elif self.scheduler and self.lr_scheduler_enabled and "scheduler_state_dict" not in checkpoint:
                logger.warning(
                    "LR Scheduler is enabled but its state was not found in the checkpoint. Scheduler will start fresh."
                )

            if "config" in checkpoint:
                logger.info("Agent config found in checkpoint. Consider validating compatibility.")
            else:
                logger.warning("Agent config not found in checkpoint.")

            logger.info(f"Agent model and associated states loaded successfully from {checkpoint_path}")
            self.network.to(self.device)
            self.target_network.to(self.device)
            self._update_target_network()
            return True

        except FileNotFoundError:
            logger.error(f"Checkpoint file not found at {checkpoint_path}")
            return False
        except RuntimeError as e:
            if _PRE_IQN_CHECKPOINT_MESSAGE in str(e):
                logger.error("Refusing to load pre-IQN checkpoint: %s", e)
                raise
            logger.error(f"Error loading agent checkpoint from {checkpoint_path}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error loading agent checkpoint from {checkpoint_path}: {e}", exc_info=True)
            return False

    def load_state(self, agent_state_dict: dict):
        """Loads the agent's state from a dictionary (typically part of a larger checkpoint).

        Used by the trainer when resuming. Legacy ``'scaler_state_dict'``
        entries from the GradScaler era are silently ignored.
        """
        logger.info("Attempting to load agent state from provided dictionary...")

        if not isinstance(agent_state_dict, dict):
            logger.error("Provided agent_state_dict is not a dictionary.")
            return False

        # Silently drop any legacy ``scaler_state_dict`` — we no longer use
        # GradScaler (bfloat16 autocast doesn't need it). Keeps old ``.pt``
        # files resumable without forcing a re-train.
        agent_state_dict.pop("scaler_state_dict", None)

        successful_load = True

        if "network_state_dict" in agent_state_dict and self.network is not None:
            try:
                _load_state_dict_with_orig_mod_fallback(self.network, agent_state_dict["network_state_dict"])
                self.network.to(self.device)
                logger.info("Network state loaded from dictionary.")
            except RuntimeError as e:
                if _PRE_IQN_CHECKPOINT_MESSAGE in str(e):
                    raise
                logger.error(f"Error loading network state_dict from dictionary: {e}", exc_info=True)
                successful_load = False
            except Exception as e:
                logger.error(f"Error loading network state_dict from dictionary: {e}", exc_info=True)
                successful_load = False
        else:
            logger.warning("Network state_dict not found in provided dictionary or agent.network is None.")
            successful_load = False

        if "target_network_state_dict" in agent_state_dict and self.target_network is not None:
            try:
                _load_state_dict_with_orig_mod_fallback(
                    self.target_network,
                    agent_state_dict["target_network_state_dict"],
                )
                self.target_network.to(self.device)
                logger.info("Target network state loaded from dictionary.")
            except RuntimeError as e:
                if _PRE_IQN_CHECKPOINT_MESSAGE in str(e):
                    raise
                logger.error(f"Error loading target_network state_dict from dictionary: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error loading target_network state_dict from dictionary: {e}", exc_info=True)
        else:
            logger.warning(
                "Target network state_dict not found in provided dictionary or agent.target_network is None."
            )

        if "optimizer_state_dict" in agent_state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(agent_state_dict["optimizer_state_dict"])
                logger.info("Optimizer state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading optimizer state_dict from dictionary: {e}", exc_info=True)
        else:
            logger.warning("Optimizer state_dict not found in provided dictionary or agent.optimizer is None.")

        if "total_steps" in agent_state_dict:
            self.total_steps = agent_state_dict["total_steps"]
            logger.info(f"Total steps loaded from dictionary: {self.total_steps}")

        if "agent_env_steps" in agent_state_dict:
            self.env_steps = agent_state_dict["agent_env_steps"]
            logger.info(f"Env steps loaded from dictionary: {self.env_steps} (epsilon={self.current_epsilon:.4f})")
        else:
            logger.warning("Total steps not found in provided dictionary. Agent's total_steps not updated.")

        if (
            "scheduler_state_dict" in agent_state_dict
            and agent_state_dict["scheduler_state_dict"] is not None
            and self.scheduler
            and self.lr_scheduler_enabled
        ):
            try:
                self.scheduler.load_state_dict(agent_state_dict["scheduler_state_dict"])
                logger.info("LR Scheduler state loaded from dictionary.")
            except Exception as e:
                logger.error(
                    f"Error loading LR scheduler state_dict from dictionary: {e}. Scheduler may not resume correctly.",
                    exc_info=True,
                )
        elif (
            self.scheduler
            and self.lr_scheduler_enabled
            and ("scheduler_state_dict" not in agent_state_dict or agent_state_dict.get("scheduler_state_dict") is None)
        ):
            logger.warning(
                "LR Scheduler is enabled but its state was not found in the dictionary. Scheduler will start fresh."
            )

        if successful_load:
            logger.info("Agent state loaded successfully from dictionary.")
        else:
            logger.error("One or more critical components failed to load from the agent state dictionary.")

        return successful_load

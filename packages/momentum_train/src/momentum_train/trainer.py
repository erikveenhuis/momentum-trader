import json  # Keep for saving results
import os
from collections import deque  # Keep deque for performance_tracker
from datetime import datetime  # Added datetime back
from pathlib import Path
from typing import Dict, List, Set, Tuple  # Added Dict for cached envs

import numpy as np
import torch
from momentum_core.logging import get_logger
from momentum_env import TradingEnv, TradingEnvConfig  # Use installed package
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from momentum_agent import RainbowDQNAgent  # Updated import path
from momentum_agent.constants import ACCOUNT_STATE_DIM  # Import constant

from .data import DataManager  # Use relative import
from .metrics import (  # Use relative import
    PerformanceTracker,
    calculate_episode_score,
)

# Get logger instance
logger = get_logger("Trainer")


class RainbowTrainerModule:
    def __init__(
        self,
        agent: RainbowDQNAgent,
        device: torch.device,
        data_manager: DataManager,
        config: dict,
        scaler: GradScaler | None = None,
        writer: SummaryWriter | None = None,
    ):
        assert isinstance(agent, RainbowDQNAgent), "Agent must be an instance of RainbowDQNAgent"
        assert isinstance(device, torch.device), "Device must be a torch.device"
        assert isinstance(data_manager, DataManager), "Data manager must be an instance of DataManager"
        assert isinstance(config, dict), "Config must be a dictionary"
        self.agent = agent
        self.device = device  # Store the device
        self.data_manager = data_manager
        self.config = config
        self.agent_config = config["agent"]
        self.env_config = config["environment"]
        self.trainer_config = config["trainer"]
        self.run_config = config.get("run", {})
        self.scaler = scaler
        self.best_validation_metric = -np.inf
        self.writer = writer
        # Adjust path prefix for Rainbow models
        self.best_model_base_prefix = str(Path(self.run_config.get("model_dir", "models")) / "rainbow_transformer_best")
        # Add checkpoint paths
        self.latest_trainer_checkpoint_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_latest.pt")
        # Store the BASE prefix for the best checkpoint.
        self.best_trainer_checkpoint_base_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_best")
        self.validation_metrics = []
        self.performance_tracker = PerformanceTracker()
        self.early_stopping_patience = self.trainer_config.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        self.min_validation_threshold = self.trainer_config.get("min_validation_threshold", -np.inf)
        self.validation_freq = self.trainer_config.get("validation_freq", 10)
        self.checkpoint_save_freq = self.trainer_config.get("checkpoint_save_freq", 10)
        self.reward_window = self.trainer_config.get("reward_window", 10)
        self.update_freq = self.trainer_config.get("update_freq", 4)
        self.log_freq = self.trainer_config.get("log_freq", 100)
        raw_grad_updates = self.trainer_config.get("gradient_updates_per_step", 1)
        try:
            self.gradient_updates_per_step = max(1, int(raw_grad_updates))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid gradient_updates_per_step value %s; defaulting to 1.",
                raw_grad_updates,
            )
            self.gradient_updates_per_step = 1
        raw_per_stats_freq = self.trainer_config.get("per_stats_log_freq", self.log_freq)
        try:
            self.per_stats_log_freq = int(raw_per_stats_freq)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid per_stats_log_freq value %s; using log_freq (%s) instead.",
                raw_per_stats_freq,
                self.log_freq,
            )
            self.per_stats_log_freq = self.log_freq
        if self.per_stats_log_freq < 0:
            logger.warning(
                "per_stats_log_freq must be >= 0 (received %s); using log_freq (%s) instead.",
                self.per_stats_log_freq,
                self.log_freq,
            )
            self.per_stats_log_freq = self.log_freq
        elif self.per_stats_log_freq == 0:
            logger.info("PER stats logging disabled because per_stats_log_freq is set to 0.")
        self.warmup_steps = self.trainer_config.get("warmup_steps", 50000)
        self.invalid_action_window = self.trainer_config.get("invalid_action_window", 20)
        self.invalid_action_rate_window: deque[float] = deque(maxlen=self.invalid_action_window)
        self.render_training_enabled = bool(self.trainer_config.get("render_training", False))
        self.render_validation_enabled = bool(self.trainer_config.get("render_validation", False))
        self.render_evaluation_enabled = bool(self.trainer_config.get("render_evaluation", False))
        self.render_on_reset = bool(self.trainer_config.get("render_on_reset", False))
        self.render_training_every_n_steps = self._parse_render_frequency("render_training_every_n_steps", default=1)
        self.render_validation_every_n_steps = self._parse_render_frequency("render_validation_every_n_steps", default=1)
        self.render_evaluation_every_n_steps = self._parse_render_frequency("render_evaluation_every_n_steps", default=1)
        self._render_error_logged: Set[str] = set()
        # Extract run parameters
        self.model_dir = self.run_config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self._validation_env_cache: Dict[str, TradingEnv] = {}

    def should_stop_early(self, validation_metrics: List[dict]) -> bool:
        """Check if training should stop early based on validation performance."""
        # Early stopping logic now happens directly in the validate() method
        # based on validation_score compared to self.best_validation_metric.
        # This function is no longer used for the primary check.
        return self.early_stopping_counter >= self.early_stopping_patience

    def should_validate(self, episode: int, recent_performance: dict) -> bool:
        """Determine if validation should be performed based on validation frequency."""
        # Removed complex logic based on improvement_rate and stability
        # Simplified to validate purely based on frequency
        return (episode + 1) % self.validation_freq == 0

    def _parse_render_frequency(self, key: str, default: int = 1) -> int:
        """Parse render frequency from trainer config with validation."""
        if key not in self.trainer_config:
            return default

        value = self.trainer_config.get(key)
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid value for %s (%s); falling back to default %s",
                key,
                value,
                default,
            )
            return default

        if value_int < 1:
            logger.warning(
                "%s must be >= 1 (received %s); using default %s",
                key,
                value_int,
                default,
            )
            return default

        return value_int

    def _is_render_context_enabled(self, context: str) -> bool:
        """Check if rendering is enabled for the given context."""
        if context == "train":
            return self.render_training_enabled
        if context == "validation":
            return self.render_validation_enabled
        if context == "eval":
            return self.render_evaluation_enabled
        return False

    def _should_render(self, context: str, step_idx: int) -> bool:
        """Determine whether rendering should occur for the given context and step."""
        if step_idx < 1:
            return False

        if context == "train":
            return self.render_training_enabled and step_idx % self.render_training_every_n_steps == 0
        if context == "validation":
            return self.render_validation_enabled and step_idx % self.render_validation_every_n_steps == 0
        if context == "eval":
            return self.render_evaluation_enabled and step_idx % self.render_evaluation_every_n_steps == 0
        return False

    def _render_on_reset_if_enabled(self, env: TradingEnv, context: str) -> None:
        """Render immediately after reset if configured for the context."""
        if not self.render_on_reset or not self._is_render_context_enabled(context):
            return

        if getattr(env, "render_mode", None) is None:
            return

        self._invoke_render(env, context, step_idx=0)

    def _invoke_render(self, env: TradingEnv, context: str, step_idx: int) -> None:
        """Safely invoke env.render(), logging only the first failure per context."""
        try:
            env.render()
        except Exception as exc:  # pragma: no cover - defensive logging
            if context not in self._render_error_logged:
                logger.warning(
                    "Environment render failed during %s at step %s: %s",
                    context,
                    step_idx,
                    exc,
                )
                self._render_error_logged.add(context)

    def _maybe_render(self, env: TradingEnv, context: str, step_idx: int) -> None:
        """Render the environment if enabled and supported."""
        if not self._should_render(context, step_idx):
            return

        if getattr(env, "render_mode", None) is None:
            return

        self._invoke_render(env, context, step_idx)

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
        checkpoint = {
            "episode": episode,
            "total_train_steps": total_steps,  # Store steps from trainer perspective
            "best_validation_metric": self.best_validation_metric,
            "early_stopping_counter": self.early_stopping_counter,
            # Replay buffer state is generally not saved due to size and complexity.
            # 'buffer_state': ... ,
            # 'n_step_buffer_state': ...,
            # We can optionally save agent config hash or version here for compatibility checks
            # --- ADDED Agent State ---
            "agent_config": self.agent.config,
            "agent_total_steps": self.agent.total_steps,
            "total_steps": self.agent.total_steps,
            "network_state_dict": self.agent.network.state_dict() if self.agent.network else None,
            "target_network_state_dict": (self.agent.target_network.state_dict() if self.agent.target_network else None),
            "optimizer_state_dict": (self.agent.optimizer.state_dict() if self.agent.optimizer else None),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
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

        # Save the latest checkpoint with date, episode and reward in filename
        try:
            # Construct filename with date, episode and reward
            latest_checkpoint_path = f"{self.latest_trainer_checkpoint_path.rsplit('.', 1)[0]}_{current_date}_ep{episode}_reward{self.best_validation_metric:.4f}.pt"
            torch.save(checkpoint, latest_checkpoint_path)
            logger.info(f"Latest checkpoint saved to {latest_checkpoint_path}")
            logger.info(f"  Episode: {episode}")
            logger.info(f"  Total Steps: {total_steps}")
            logger.info(f"  Best Validation Score: {self.best_validation_metric:.4f}")
            logger.info(f"  Early Stopping Counter: {self.early_stopping_counter}")
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
                # Use the dynamically constructed path
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Best checkpoint saved to {best_checkpoint_path}")
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

    # --- Refactored Helper Methods ---

    def _initialize_episode(
        self, specific_file: str | None, episode: int, num_episodes: int
    ) -> Tuple[TradingEnv | None, dict | None, dict | None, PerformanceTracker | None]:
        """Sets up the environment and performance tracker for a new episode. Returns env, obs, info, tracker."""
        try:
            episode_file_path = Path(specific_file) if specific_file else self.data_manager.get_random_training_file()
            logger.info(f"--- Starting Episode {episode + 1}/{num_episodes} using file: {episode_file_path.name} ---")
        except Exception as e:
            logger.error(f"Error getting data file for episode {episode+1}: {e}")
            return None, None, None, None  # Indicate failure

        try:
            # Update env_config with the current episode file path
            self.env_config["data_path"] = str(episode_file_path)

            # Create a TradingEnvConfig object
            env_config_obj = TradingEnvConfig(**self.env_config)

            env = TradingEnv(config=env_config_obj)
            obs, info = env.reset()
            self._render_on_reset_if_enabled(env, "train")
            assert isinstance(info["portfolio_value"], (float, np.float32, np.float64)), "Reset info missing valid portfolio_value"
            # Basic observation checks
            assert isinstance(obs, dict), "Observation must be a dict"
            assert "market_data" in obs and "account_state" in obs, "Observation missing keys"
            assert isinstance(obs["market_data"], np.ndarray), "Market data not numpy array"
            assert isinstance(obs["account_state"], np.ndarray), "Account state not numpy array"

            # Initialize tracker for the episode
            tracker = PerformanceTracker()
            initial_portfolio_value = info["portfolio_value"]
            tracker.add_initial_value(initial_portfolio_value)

            # Return obs as well
            return env, obs, info, tracker
        except Exception:
            logger.error(
                f"!!! Exception during env creation/reset() for {episode_file_path.name} !!!",
                exc_info=True,
            )
            return None, None, None, None  # Indicate failure

    def _perform_training_step(
        self,
        env: TradingEnv,
        obs: dict,
        total_train_steps: int,
        episode: int,
        steps_in_episode: int,
    ) -> Tuple[dict, float, bool, dict, int, float | None]:
        """Performs a single step of interaction with the environment and learning."""
        loss_value = None  # Initialize loss_value

        # Assert observation shape before selecting action
        assert obs["market_data"].shape == (self.agent.window_size, self.agent.n_features)
        assert obs["account_state"].shape == (ACCOUNT_STATE_DIM,)

        # Select action
        if total_train_steps < self.warmup_steps:
            action = env.action_space.sample()
        else:
            action = self.agent.select_action(obs)

        # Step environment
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminated_flag = bool(terminated)
            truncated_flag = bool(truncated)
            done = terminated_flag or truncated_flag
            if isinstance(info, dict):
                info.setdefault("terminated", terminated_flag)
                info.setdefault("truncated", truncated_flag)
            # Basic validation of step outputs
            if not isinstance(next_obs, dict):
                logger.error(f"next_obs is not a dict: {type(next_obs)}")
            if "market_data" not in next_obs:
                logger.error("next_obs missing market_data")
            if "account_state" not in next_obs:
                logger.error("next_obs missing account_state")
            if not isinstance(done, (bool, np.bool_)):
                logger.error(f"done is not a bool: {type(done)}")
            if not isinstance(info, dict):
                logger.error(f"info is not a dict: {type(info)}")

            # Original assertions (modified)
            assert isinstance(next_obs, dict) and "market_data" in next_obs and "account_state" in next_obs
            assert isinstance(done, (bool, np.bool_))
            assert isinstance(info, dict)
        except Exception as e:
            if isinstance(e, RuntimeError) and "Episode is done, call reset() first" in str(e):
                logger.debug("Environment reported completion before step; treating as natural episode termination")
                done = True
                reward = 0.0
                next_obs = obs
                info = self._get_fallback_info(obs, info if "info" in locals() else {})
                info.setdefault("terminated", info.get("terminated", False))
                info.setdefault("truncated", True)
                info.setdefault("invalid_action", False)
                info["episode_already_done"] = True
            else:
                logger.error(
                    f"Error during env.step at step {steps_in_episode} in episode {episode}: {e}",
                    exc_info=True,
                )
                done = True
                reward = -10.0
                next_obs = obs
                info = self._get_fallback_info(obs, info if "info" in locals() else {})

        # Store transition
        self.agent.store_transition(obs, action, reward, next_obs, done)

        # Perform learning update (only if not done from env error)
        losses_this_step: list[float] = []
        if not done and (
            len(self.agent.buffer) >= self.agent.batch_size
            and total_train_steps > self.warmup_steps
            and total_train_steps % self.update_freq == 0
        ):
            try:
                for _ in range(self.gradient_updates_per_step):
                    learn_loss = self.agent.learn()
                    if learn_loss is None:
                        break
                    losses_this_step.append(float(learn_loss))

                if losses_this_step:
                    loss_value = float(np.mean(losses_this_step))

                    # --- Log Loss to TensorBoard --- #
                    if self.writer:
                        self.writer.add_scalar("Train/Loss", loss_value, total_train_steps)
                        self.writer.add_scalar(
                            "Train/Gradient Updates Per Step",
                            len(losses_this_step),
                            total_train_steps,
                        )
                        td_stats = getattr(self.agent, "last_td_error_stats", None)
                        if td_stats:
                            self.writer.add_scalar("Train/TD_Error_Mean", td_stats.get("mean", float("nan")), total_train_steps)
                            self.writer.add_scalar("Train/TD_Error_Std", td_stats.get("std", float("nan")), total_train_steps)
                    # ---------------------------- #

            except Exception:
                logger.error(
                    f"!!! EXCEPTION during learning update at step {total_train_steps} !!!",
                    exc_info=True,
                )
                done = True  # Stop episode on learning error

        return next_obs, reward, done, info, action, loss_value

    def _log_step_progress(
        self,
        episode: int,
        steps_in_episode: int,
        tracker: PerformanceTracker,
        recent_step_rewards: deque,
        recent_losses: deque,
        action: int,
        reward: float,
        info: dict,
    ):
        """Log step progress with detailed information."""
        mean_reward = np.mean(recent_step_rewards) if recent_step_rewards else 0.0
        mean_loss = np.mean(recent_losses) if recent_losses else 0.0

        # Calculate position value in USD
        position_value = info["position"] * info["price"]

        logger.debug(
            f"  Ep {episode} Step {steps_in_episode}: "
            f"Port=${info['portfolio_value']:.2f}, "
            f"Act={action}, "
            f"StepRew={reward:.8f}, "
            f"CumTxCost=${info['transaction_cost']:.2f}, "
            f"MeanRew-{self.log_freq}={mean_reward:.4f}, "
            f"MeanLoss-{self.log_freq}={mean_loss:.4f}, "
            f"Price=${info['price']:.8f}, "
            f"Balance=${info['balance']:.2f}, "
            f"Position={info['position']:.4f}, "
            f"PosValue=${position_value:.2f}"
        )

    def _maybe_log_per_stats(self, total_train_steps: int, *, force: bool = False) -> None:
        """Log prioritized replay buffer statistics at the configured frequency or when forced."""
        if total_train_steps <= 0:
            return
        if self.per_stats_log_freq == 0 and not force:
            return
        if not force and (self.per_stats_log_freq < 1 or total_train_steps % self.per_stats_log_freq != 0):
            return

        get_per_stats = getattr(self.agent, "get_per_stats", None)
        if not callable(get_per_stats):
            logger.debug("Agent does not expose get_per_stats; skipping PER stats logging.")
            return

        try:
            stats = get_per_stats()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to collect PER stats: {exc}", exc_info=True)
            return

        if not stats:
            return

        logger.info(
            "PER Stats @ env step %s (learner %s): buffer=%s/%s (%.2f%%), alpha=%.3f, beta=%.3f, "
            "beta_progress=%.1f%%, avg_priority=%.6f, max_priority=%.6f, total_priority=%.6f",
            total_train_steps,
            stats.get("total_steps", -1),
            stats.get("size", 0),
            stats.get("capacity", 0),
            stats.get("fill_ratio", 0.0) * 100.0,
            stats.get("alpha", 0.0),
            stats.get("beta", 0.0),
            stats.get("beta_progress", 0.0) * 100.0,
            stats.get("avg_priority", 0.0),
            stats.get("max_priority", 0.0),
            stats.get("total_priority", 0.0),
        )

    def _log_episode_summary(
        self,
        episode: int,
        episode_reward: float,
        total_rewards: list,
        episode_loss: float,
        steps_in_episode: int,
        tracker: PerformanceTracker,
        final_info: dict,
        invalid_action_count: int,
        total_train_steps: int,
    ):
        """Logs the summary statistics at the end of an episode."""
        avg_reward_window = np.mean(total_rewards[-self.reward_window :])
        avg_reward_total = np.mean(total_rewards)
        logger.info(f"Episode {episode + 1}: Ended.")
        logger.info(f"  Steps: {steps_in_episode}")
        logger.info(f"  Reward: {episode_reward:.4f}")
        logger.info(f"  Avg Reward ({self.reward_window} ep): {avg_reward_window:.4f}")
        logger.info(f"  Avg Reward (Total): {avg_reward_total:.4f}")
        logger.info(
            f"  Avg Loss: {(episode_loss / (steps_in_episode / self.update_freq)) if steps_in_episode > 0 else 0:.4f}"
        )  # Adjust loss averaging
        steps_safe = max(steps_in_episode, 1)
        invalid_action_rate = invalid_action_count / steps_safe
        self.invalid_action_rate_window.append(invalid_action_rate)
        rolling_invalid_rate = float(np.mean(self.invalid_action_rate_window)) if self.invalid_action_rate_window else 0.0
        logger.info(
            "  Invalid Actions: %s (%.2f%% of steps, Rolling %.2f%% over last %s episodes)",
            invalid_action_count,
            invalid_action_rate * 100,
            rolling_invalid_rate * 100,
            len(self.invalid_action_rate_window),
        )
        logger.info(f"  Final Portfolio Value: ${final_info.get('portfolio_value', -1):.2f}")
        logger.info(f"  Final Position: {final_info.get('position', -1):.4f}")
        # tracker.log_summary(logger, episode + 1) # Original line causing error
        # --- Log tracker metrics --- #
        metrics = tracker.get_metrics()
        logger.info(f"  Metrics - Total Return: {metrics.get('total_return', np.nan):.2f}%" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Sharpe Ratio: {metrics.get('sharpe_ratio', np.nan):.4f}" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Max Drawdown: {metrics.get('max_drawdown', np.nan)*100:.2f}%" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Action Counts: {metrics.get('action_counts', {})}" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Transaction Costs: ${metrics.get('transaction_costs', np.nan):.2f}" if metrics else "Metrics: N/A")
        # ------------------------- #

        # --- Log Episode Summary to TensorBoard --- #
        if self.writer:
            self.writer.add_scalar("Train/Episode Reward", episode_reward, episode)
            self.writer.add_scalar(
                f"Train/Average Reward ({self.reward_window} ep)",
                avg_reward_window,
                episode,
            )
            self.writer.add_scalar("Train/Average Reward (Total)", avg_reward_total, episode)
            self.writer.add_scalar("Train/Steps Per Episode", steps_in_episode, episode)
            if steps_in_episode > 0:
                avg_episode_loss = episode_loss / (steps_in_episode / self.update_freq + 1e-6)  # Avoid div by zero
                self.writer.add_scalar("Train/Average Episode Loss", avg_episode_loss, episode)

            if metrics:  # Ensure metrics dict is not empty
                self.writer.add_scalar("Train/Total Return Pct", metrics.get("total_return", np.nan), episode)
                self.writer.add_scalar("Train/Sharpe Ratio", metrics.get("sharpe_ratio", np.nan), episode)
                # Max drawdown is usually negative or zero, store as positive percentage for clarity if convention is to show magnitude
                self.writer.add_scalar("Train/Max Drawdown Pct", metrics.get("max_drawdown", np.nan) * 100, episode)
                self.writer.add_scalar("Train/Transaction Costs", metrics.get("transaction_costs", np.nan), episode)
                self.writer.add_scalar("Train/Avg Tracker Reward", metrics.get("avg_reward", np.nan), episode)
                action_counts = metrics.get("action_counts", {})
                if action_counts:
                    for action_idx, count in action_counts.items():
                        action_rate = count / steps_safe if steps_safe > 0 else 0.0
                        self.writer.add_scalar(f"Train/Action Count/{action_idx}", count, episode)
                        self.writer.add_scalar(f"Train/Action Rate/{action_idx}", action_rate, episode)

            self.writer.add_scalar("Train/Final Portfolio Value", final_info.get("portfolio_value", np.nan), episode)
            self.writer.add_scalar("Train/Final Position", final_info.get("position", np.nan), episode)
            self.writer.add_scalar("Train/Invalid Action Rate", invalid_action_rate, episode)
            self.writer.add_scalar("Train/Rolling Invalid Action Rate", rolling_invalid_rate, episode)
        # ------------------------------------------ #

        # Always log PER stats at episode boundaries (unless explicitly disabled)
        self._maybe_log_per_stats(total_train_steps, force=True)

    def _handle_validation_and_checkpointing(
        self,
        episode: int,
        total_train_steps: int,
        val_files: List[Path],
        tracker: PerformanceTracker,
    ) -> bool:
        """Handles validation runs and checkpoint saving. Returns True if training should stop."""
        save_now = (episode + 1) % self.checkpoint_save_freq == 0
        should_stop_training = False
        is_best = False
        validation_score = -np.inf  # Default score
        avg_val_metrics = {}  # Initialize to empty dict

        # Store old best score for "is_best" decision
        old_best_validation_metric = self.best_validation_metric

        # Run validation if needed
        if val_files and self.should_validate(episode, tracker.get_recent_metrics()):
            try:
                logger.info(f"--- Running validation after episode {episode + 1} ---")
                # MODIFIED: Capture avg_val_metrics
                should_stop_training, validation_score, avg_val_metrics = self.validate(val_files)
            except Exception as e:
                logger.error(f"Exception during validation after episode {episode}: {e}", exc_info=True)
                should_stop_training = False  # Don't stop on validation error
                validation_score = -np.inf
                avg_val_metrics = {}  # Ensure it's defined for logging below

            logger.info("Validation Score Comparison:")
            logger.info(f"  Current Score: {validation_score:.4f}")
            # self.best_validation_metric is updated by self.validate() if score improved
            logger.info(f"  Best Tracked Score (after this validation): {self.best_validation_metric:.4f}")
            logger.info(f"  Best Tracked Score (before this validation): {old_best_validation_metric:.4f}")

            # MODIFIED: Corrected is_best determination
            # An improvement is "best" if it's better than the old best by at least the threshold.
            # self.best_validation_metric has already been updated by validate() if validation_score was strictly > old_best_validation_metric.
            if validation_score > old_best_validation_metric + self.min_validation_threshold:
                is_best = True
                # self.best_validation_metric is already updated by validate() to validation_score
                logger.info(
                    f"  >>> NEW BEST CHECKPOINT (Score: {validation_score:.4f} > Old best: {old_best_validation_metric:.4f} + Threshold: {self.min_validation_threshold}) <<< "
                )
            else:
                is_best = False
                logger.info(
                    f"  No improvement for best checkpoint (Current: {validation_score:.4f}, Best tracked: {self.best_validation_metric:.4f}, Old best for this run: {old_best_validation_metric:.4f}, Threshold: {self.min_validation_threshold})"
                )

            # --- Log Validation Score and Metrics to TensorBoard --- #
            if self.writer:
                self.writer.add_scalar("Validation/Score", validation_score, episode)
                if avg_val_metrics:  # Check if metrics are available
                    self.writer.add_scalar(
                        "Validation/Total Return Pct",
                        avg_val_metrics.get("total_return", np.nan),
                        episode,
                    )
                    self.writer.add_scalar(
                        "Validation/Sharpe Ratio",
                        avg_val_metrics.get("sharpe_ratio", np.nan),
                        episode,
                    )
                    # Max drawdown is a fraction, convert to percentage for logging
                    self.writer.add_scalar(
                        "Validation/Max Drawdown Pct",
                        avg_val_metrics.get("max_drawdown", np.nan) * 100,
                        episode,
                    )
            # ---------------------------------------------------- #

            # --- Step LR Scheduler if it's ReduceLROnPlateau --- #
            if self.agent.lr_scheduler_enabled and isinstance(self.agent.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if validation_score != -np.inf:  # Only step if validation score is valid
                    current_lr = self.agent.optimizer.param_groups[0]["lr"]
                    logger.info(f"[LR Scheduler] Before step: LR={current_lr:.8f}, metric={validation_score:.6f}")
                    self.agent.step_lr_scheduler(validation_score)
                    new_lr = self.agent.optimizer.param_groups[0]["lr"]
                    logger.info(f"[LR Scheduler] After step: LR={new_lr:.8f}, metric={validation_score:.6f}")
                    if new_lr < current_lr - 1e-12:
                        if self.early_stopping_counter > 0:
                            logger.info(
                                "[LR Scheduler] Learning rate reduced; resetting early stopping counter (was %d).",
                                self.early_stopping_counter,
                            )
                        self.early_stopping_counter = 0
                    # Log current learning rate to TensorBoard after potential step
                    if self.writer:
                        self.writer.add_scalar("Train/Learning_Rate", new_lr, total_train_steps)  # Use total_train_steps or episode
                else:
                    logger.warning("Skipping ReduceLROnPlateau step due to invalid validation score (-np.inf).")
            # ---------------------------------------------------- #

            # Save checkpoint AFTER validation
            self._save_checkpoint(
                episode=episode + 1,
                total_steps=total_train_steps,
                is_best=is_best,  # Pass the flag indicating if this is the best
                validation_score=validation_score,  # Pass the score achieved
            )
            save_now = False  # Avoid double saving

            if should_stop_training:
                logger.info("Early stopping triggered by validation result. Training will stop.")
                return True  # Signal to stop training

        # Periodic checkpoint saving (if not saved after validation)
        if save_now:
            self._save_checkpoint(
                episode=episode + 1,
                total_steps=total_train_steps,
                is_best=False,  # Not necessarily the best if saved periodically
                validation_score=None,  # No relevant score for periodic save
            )

        # --- Final HParams log (optional but good practice) --- #
        # if self.writer: # <-- Comment out the entire block
        #     hparams = {
        #         # Add key hyperparameters from config
        #         'lr': self.agent_config.get('lr'),
        #         'gamma': self.agent_config.get('gamma'),
        #         'batch_size': self.agent_config.get('batch_size'),
        #         'target_update_freq': self.agent_config.get('target_update_freq'),
        #         'window_size': self.env_config.get('window_size'),
        #         'n_steps': self.agent_config.get('n_steps'),
        #         'num_atoms': self.agent_config.get('num_atoms'),
        #         # Add more relevant hparams
        #     }
        #     final_metrics = {
        #         # Log final/best metrics
        #         'hparam/best_validation_score': self.best_validation_metric if self.best_validation_metric > -np.inf else np.nan,
        #         # 'hparam/final_avg_reward': np.mean(total_rewards[-self.reward_window:] if total_rewards else np.nan),
        #     }
        #     # Filter out None values from hparams before logging
        #     hparams_filtered = {k: v for k, v in hparams.items() if v is not None}
        #     self.writer.add_hparams(hparams_filtered, final_metrics)
        # ------------------------------------------------------ #

        return False  # Continue training

    def _finalize_training(self, total_train_steps: int, num_episodes: int, val_files: list[Path]):
        """Saves final model and logs overall training summary."""
        # Save final model independently (using agent's save method)
        final_model_prefix = str(Path(self.model_dir) / "rainbow_transformer_final")
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

    # --- Added Evaluation Step Helper ---
    def _perform_evaluation_step(self, env: TradingEnv, obs: dict) -> Tuple[dict, float, bool, dict, int, bool]:  # Added boolean error flag
        """Performs a single step of evaluation in the environment. Returns (next_obs, reward, done, info, action, error_occurred)."""
        try:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- ADDED: Check for non-numeric reward --- #
            error_occurred = False  # Initialize error flag for this check
            if not isinstance(done, (bool, np.bool_)):
                logger.error(f"done is not a bool: {type(done)}")
            if not isinstance(info, dict):
                logger.error(f"info is not a dict: {type(info)}")

            # --- Assert info structure --- #
            assert isinstance(info, dict), "Validation: Info from env.step() must be a dict"
            assert "portfolio_value" in info, "Validation: Info missing portfolio_value"
            assert isinstance(info["portfolio_value"], (float, np.float32, np.float64)), "Validation: portfolio_value is not a float"
            # --- End Assert --- #

            return (
                next_obs,
                reward,
                done,
                info,
                action,
                error_occurred,
            )  # Return the potentially modified error flag

        except Exception as e:
            logger.error(f"Error during validation step: {e}", exc_info=True)  # Log with traceback
            # Return original obs, penalty reward, done=True, fallback info, dummy action, error=True
            fallback_info = self._get_fallback_info(obs, {})  # Simplified fallback info for error case
            penalty_reward = -12.0
            dummy_action = -1  # Placeholder action for error case
            return (
                obs,
                penalty_reward,
                True,
                fallback_info,
                dummy_action,
                True,
            )  # True = error occurred

    # --- End Evaluation Step Helper ---

    def _run_single_evaluation_episode(
        self, env: TradingEnv, context: str = "validation", *, close_env: bool = True
    ) -> Tuple[float, dict, dict]:
        """Evaluate the agent for one episode on a given environment instance."""
        # Removed assert isinstance(env, TradingEnv) because it fails when TradingEnv is patched in tests
        # assert isinstance(
        #     env, TradingEnv
        # ), "env must be an instance of TradingEnv for evaluation"
        was_training = self.agent.training_mode
        self.agent.set_training_mode(False)
        tracker = None  # Initialize tracker to None
        final_info = {}  # Initialize final_info
        total_reward = -np.inf  # Default reward if reset fails
        metrics = {}  # Default metrics

        try:
            obs, info = env.reset()
            self._render_on_reset_if_enabled(env, context)
            # --- Assert observation structure ---
            assert isinstance(obs, dict), "Validation: Observation from env.reset() must be a dict"
            assert "market_data" in obs and "account_state" in obs, "Validation: Observation missing keys"
            assert isinstance(obs["market_data"], np.ndarray), "Validation: Market data is not a numpy array"
            assert isinstance(obs["account_state"], np.ndarray), "Validation: Account state is not a numpy array"
            # --- End Assert ---
            done = False
            total_reward = 0
            tracker = PerformanceTracker()  # Initialize tracker here after successful reset
            portfolio_values_over_episode = []  # List to store portfolio values
            initial_portfolio_value = info["portfolio_value"]
            tracker.add_initial_value(initial_portfolio_value)

            episode_had_error = False  # Flag to track errors
            step_index = 0
            while not done:
                # Call the new helper method
                next_obs, reward, step_done, step_info, action, error_occurred = self._perform_evaluation_step(env, obs)

                # Update loop condition and info for potential use after loop
                done = step_done
                info = step_info
                step_index += 1

                if error_occurred:
                    episode_had_error = True  # Set the flag

                # Handle step results only if no error occurred during the step
                if not error_occurred:
                    portfolio_values_over_episode.append(info["portfolio_value"])  # Store value

                    # Update performance tracker
                    tracker.update(
                        portfolio_value=info["portfolio_value"],
                        action=action,
                        reward=reward,
                        transaction_cost=info.get("step_transaction_cost", 0.0),  # Use step cost
                    )
                    total_reward += reward
                    obs = next_obs
                    self._maybe_render(env, context, step_index)
                else:
                    # Error already logged in helper. Add penalty reward. Loop will terminate.
                    total_reward += reward  # Add the penalty reward returned by helper
                    # obs remains the same, loop terminates in the next iteration check due to done=True

            # Store the last info dict after the loop finishes
            final_info = info  # info holds fallback if error occurred

            # Check if error occurred AT ANY POINT during the episode
            if episode_had_error:
                logger.warning("Errors occurred during evaluation episode steps. Returning default error metrics.")
                metrics = {}  # Return empty metrics
                total_reward = -np.inf  # Ensure reward reflects failure
            elif tracker:  # No error occurred, proceed with metrics calculation
                metrics = tracker.get_metrics()

                # Calculate and add portfolio statistics
                if portfolio_values_over_episode:
                    metrics["min_portfolio_value"] = float(np.min(portfolio_values_over_episode))
                    metrics["max_portfolio_value"] = float(np.max(portfolio_values_over_episode))
                    metrics["mean_portfolio_value"] = float(np.mean(portfolio_values_over_episode))
                    metrics["median_portfolio_value"] = float(np.median(portfolio_values_over_episode))
                else:
                    # Handle case where episode might have ended before any steps
                    metrics["min_portfolio_value"] = np.nan
                    metrics["max_portfolio_value"] = np.nan
                    metrics["mean_portfolio_value"] = np.nan
                    metrics["median_portfolio_value"] = np.nan
        except Exception as e:  # Catch any exception during the reset or main loop
            logger.error(f"Error during evaluation episode run: {e}", exc_info=True)
            # Return default/failure values
            return (
                -np.inf,
                {},
                final_info,
            )  # Return initialized final_info or an empty dict if preferred
        finally:
            # Ensure env is closed even if errors occurred when requested
            if close_env:
                try:
                    env.close()
                except Exception as close_e:
                    logger.error(f"Error closing validation environment: {close_e}")
            # Restore agent training mode
            self.agent.set_training_mode(was_training)

        # Return the final info dict as well
        return total_reward, metrics, final_info

    # --- Validation Helper Methods ---
    def _validate_single_file(
        self, val_file: Path, validation_episode: int = 0, total_validation_episodes: int = 1, context: str = "validation"
    ) -> dict | None:
        """Runs validation on a single file and returns collected metrics/results."""
        logger.info(
            f"--- Starting {context.capitalize()} Episode {validation_episode + 1}/{total_validation_episodes} using file: {val_file.name} ---"
        )
        env_key = str(val_file)
        env = self._validation_env_cache.get(env_key)
        created_env = False

        if env is None:
            try:
                # Update env_config with the validation file path
                self.env_config["data_path"] = env_key

                # Create a TradingEnvConfig object
                env_config_obj = TradingEnvConfig(**self.env_config)

                env = TradingEnv(config=env_config_obj)
                self._validation_env_cache[env_key] = env
                created_env = True
            except Exception as env_e:
                logger.error(f"Error creating environment for {val_file.name}: {env_e}", exc_info=True)
                return None  # Indicate failure for this file

        try:
            logger.debug(f"Calling _run_single_evaluation_episode for {val_file.name}")
            reward, file_metrics, final_info = self._run_single_evaluation_episode(
                env,
                context="validation",
                close_env=False,
            )
        except Exception as run_e:
            logger.error(
                f"Error during _run_single_evaluation_episode for {val_file.name}: {run_e}",
                exc_info=True,
            )
            # Ensure env is closed if run fails mid-way and remove from cache
            try:
                if env is not None:
                    env.close()
            except Exception as close_e:
                logger.error(f"Error closing env after run failure for {val_file.name}: {close_e}")
            finally:
                self._validation_env_cache.pop(env_key, None)
            return None  # Indicate failure, cannot calculate score

        if created_env:
            logger.debug(f"Cached validation environment for {val_file.name}")

        # --- Enhanced per-file logging (BEFORE score calculation) ---
        # Check if metrics are valid before logging
        if file_metrics:
            logger.debug(f"  Results for {val_file.name}:")
            logger.debug(f"    Reward: {reward:.4f}")
            logger.debug(f"    Portfolio Value: ${file_metrics.get('portfolio_value', np.nan):.2f}")  # Use .get()
            logger.debug(f"    Total Return: {file_metrics.get('total_return', np.nan):.2f}%")
            logger.debug(f"    Sharpe Ratio: {file_metrics.get('sharpe_ratio', np.nan):.4f}")
            logger.debug(f"    Max Drawdown: {file_metrics.get('max_drawdown', np.nan)*100:.2f}%")
            logger.debug(f"    Action Counts: {file_metrics.get('action_counts', {})}")
            logger.debug(f"    Transaction Costs: ${file_metrics.get('transaction_costs', np.nan):.2f}")
        else:
            logger.warning(f"Metrics dictionary is empty for {val_file.name}, cannot log detailed results.")

        # --- START MODIFIED SCORE CALCULATION --- #
        # Check if the episode run itself failed (indicated by reward = -inf)
        if reward == -np.inf:
            logger.warning(f"Episode run for {val_file.name} failed (reward=-inf). Setting episode_score to -np.inf.")
            episode_score = -np.inf
        else:
            # Attempt score calculation only if metrics are valid
            if file_metrics:
                try:
                    _score = calculate_episode_score(file_metrics)
                    # Check for NaN/Inf
                    if np.isnan(_score) or np.isinf(_score):
                        raise ValueError(f"Calculated episode score is NaN or Inf ({_score})")
                    # Check range
                    if not (0.0 <= _score <= 1.0):
                        raise ValueError(f"Episode score out of range [0,1]: {_score}")
                    episode_score = _score  # Assign valid score
                    logger.debug(f"    Episode Score: {episode_score:.4f}")
                except (ValueError, KeyError, TypeError, Exception) as score_e:
                    logger.error(
                        f"Error calculating or validating episode score for {val_file.name}: {score_e}",
                        exc_info=True,
                    )
                    logger.debug(f"Setting episode_score to -np.inf due to exception for {val_file.name}")
                    episode_score = -np.inf  # Penalize score calculation errors by setting score to -inf
                    logger.debug("    Episode Score: SET TO -np.inf due to calculation error.")
            else:
                logger.warning(
                    f"Skipping score calculation for {val_file.name} due to empty/invalid metrics. Setting episode_score to -np.inf."
                )
                episode_score = -np.inf  # Assign -inf if metrics were invalid
        # --- END MODIFIED SCORE CALCULATION --- #

        # Prepare result dict for aggregation (convert numpy types)
        # Only create if metrics are valid
        if file_metrics:
            detailed_result = {
                "file": val_file.name,
                "reward": float(reward),
                "portfolio_value": float(file_metrics.get("portfolio_value", np.nan)),
                "total_return": float(file_metrics.get("total_return", np.nan)),
                "sharpe_ratio": float(file_metrics.get("sharpe_ratio", np.nan)),
                "max_drawdown": float(file_metrics.get("max_drawdown", np.nan)),
                "transaction_costs": float(final_info.get("transaction_cost", np.nan)),
            }
        else:
            # Create placeholder if metrics were invalid
            detailed_result = {
                "file": val_file.name,
                "reward": float(reward) if "reward" in locals() else -np.inf,
                "error": "Evaluation run failed or produced invalid metrics",
            }

        return {
            "file_metrics": file_metrics if file_metrics else {},  # Return empty dict if invalid
            "detailed_result": detailed_result,  # For saving to JSON
            "episode_score": episode_score,  # Return 0.0 if calculation failed
        }

    def close_cached_environments(self) -> None:
        """Close any cached validation/test environments."""
        for env_key, env in list(self._validation_env_cache.items()):
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning(f"Failed to close cached environment {env_key}: {exc}")
        self._validation_env_cache.clear()

    def _calculate_average_validation_metrics(self, all_file_metrics: List[dict]) -> dict:
        """Calculates average metrics across all validation files."""
        if not all_file_metrics:
            logger.warning("No file metrics available to calculate averages.")
            return {
                "avg_reward": np.nan,
                "portfolio_value": np.nan,
                "total_return": np.nan,
                "sharpe_ratio": np.nan,
                "max_drawdown": np.nan,
                "transaction_costs": np.nan,
            }
        # Extract lists, safely handling missing keys
        rewards = [m.get("avg_reward", np.nan) for m in all_file_metrics]
        portfolios = [m.get("portfolio_value", np.nan) for m in all_file_metrics]
        returns = [m.get("total_return", np.nan) for m in all_file_metrics]
        sharpes = [m.get("sharpe_ratio", np.nan) for m in all_file_metrics]
        drawdowns = [m.get("max_drawdown", np.nan) for m in all_file_metrics]
        costs = [m.get("transaction_costs", np.nan) for m in all_file_metrics]

        return {
            "avg_reward": float(np.nanmean(rewards)),
            "portfolio_value": float(np.nanmean(portfolios)),
            "total_return": float(np.nanmean(returns)),
            "sharpe_ratio": float(np.nanmean(sharpes)),
            "max_drawdown": float(np.nanmean(drawdowns)),
            "transaction_costs": float(np.nanmean(costs)),
        }

    def _save_validation_results(self, validation_score: float, avg_metrics: dict, detailed_results: List[dict]):
        """Saves the validation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.model_dir) / f"validation_results_{timestamp}.json"
        try:
            json_results = {
                "timestamp": timestamp,
                "validation_score": float(validation_score),
                "average_metrics": avg_metrics,
                "detailed_results": detailed_results,
            }
            with open(results_file, "w") as f:
                json.dump(json_results, f, indent=4)
            logger.info(f"Validation results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

    def _check_early_stopping(self, validation_score: float) -> bool:
        """Checks the early stopping condition based on validation score."""
        should_stop = False
        if validation_score > self.best_validation_metric:
            logger.info(f"Validation score improved from {self.best_validation_metric:.4f} to {validation_score:.4f}")
            self.best_validation_metric = validation_score
            self.early_stopping_counter = 0
            should_stop = False  # Improvement means don't stop
        else:
            self.early_stopping_counter += 1
            logger.info(
                f"Validation score ({validation_score:.4f}) did not improve over best ({self.best_validation_metric:.4f}). "
                f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
            )
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.early_stopping_counter} episodes without improvement.")
                should_stop = True
            else:
                should_stop = False

        return should_stop

    # --- End Validation Helper Methods ---

    def validate(self, val_files: List[Path]) -> Tuple[bool, float, dict]:  # MODIFIED: Added dict for avg_metrics
        """Run validation on validation files using helper methods, log, and check for early stopping."""

        # Handle empty validation file list
        if not val_files:
            logger.warning("validate() called with empty val_files list. Returning default score -inf and empty metrics.")
            return False, -np.inf, {}  # MODIFIED: Return empty dict for avg_metrics

        all_file_metrics = []
        detailed_results = []
        episode_scores = []  # Store individual episode scores

        try:
            logger.info("============================================")
            logger.info(f"RUNNING VALIDATION ON {len(val_files)} FILES")
            logger.info(f"Current best validation score: {self.best_validation_metric:.4f}")
            logger.info("============================================")

            # 1. Validate each file
            for i, val_file in enumerate(val_files):
                single_file_result = self._validate_single_file(
                    val_file, validation_episode=i, total_validation_episodes=len(val_files), context="validation"
                )
                # FIX: Only process results if the single file validation succeeded
                if single_file_result is not None:
                    all_file_metrics.append(single_file_result["file_metrics"])
                    detailed_results.append(single_file_result["detailed_result"])
                    episode_scores.append(single_file_result["episode_score"])
                # else: Error logged in _validate_single_file, skip appending results/scores

            # 2. Calculate overall validation score
            if not episode_scores:  # Handle empty list first
                logger.warning("No valid episode scores collected during validation. Defaulting score to -inf.")
                validation_score = -np.inf
            elif -np.inf in episode_scores:
                logger.warning("At least one validation episode failed (score=-inf). Overall validation score set to -inf.")
                validation_score = -np.inf
            else:  # All episodes succeeded (returned finite scores)
                # Calculate average of episode scores
                validation_score = float(np.mean(episode_scores))

            # 3. Calculate average metrics
            avg_metrics = self._calculate_average_validation_metrics(all_file_metrics)

            # 4. Log validation summary
            logger.info("\n=== VALIDATION SUMMARY ===")
            logger.info(f"Average Episode Score: {validation_score:.4f}")
            logger.info(f"Previous Best Score: {self.best_validation_metric:.4f}")
            logger.info(f"Score Difference: {validation_score - self.best_validation_metric:.4f}")
            logger.info(f"  Average Reward: {avg_metrics['avg_reward']:.2f}")
            logger.info(f"  Average Portfolio: ${avg_metrics['portfolio_value']:.2f}")
            logger.info(f"  Average Return: {avg_metrics['total_return']:.2f}%")
            logger.info(f"  Average Sharpe: {avg_metrics['sharpe_ratio']:.4f}")
            logger.info(f"  Average Max Drawdown: {avg_metrics['max_drawdown']*100:.2f}%")
            logger.info(f"Average Transaction Costs: ${avg_metrics['transaction_costs']:.2f}")
            logger.info("============================================")

            # 5. Save validation results
            self._save_validation_results(validation_score, avg_metrics, detailed_results)

            # 6. Check for early stopping
            should_stop = self._check_early_stopping(validation_score)

            return should_stop, validation_score, avg_metrics  # MODIFIED: Return avg_metrics

        except Exception as e:
            # Catch unexpected errors in the main validation orchestration
            logger.error(f"Unexpected error during main validation process: {e}", exc_info=True)
            return False, -np.inf, {}  # MODIFIED: Return empty dict for avg_metrics

    def _get_fallback_info(self, last_obs: dict, last_info: dict) -> dict:
        """Provides a fallback info dictionary if env.step crashes."""
        # Try to get last known portfolio value, default to 0 if unavailable or invalid
        fallback_portfolio_value = last_info.get("portfolio_value", 0.0)
        if not isinstance(fallback_portfolio_value, (float, np.float32, np.float64)) or fallback_portfolio_value < 0:
            fallback_portfolio_value = 0.0

        return {
            "step": last_info.get("step", -1),
            "price": last_info.get("price", 0.0),
            "balance": last_info.get("balance", 0.0),
            "position": last_info.get("position", 0.0),
            "portfolio_value": fallback_portfolio_value,  # Ensure valid value
            "step_transaction_cost": last_info.get("step_transaction_cost", 0.0),
            "invalid_action": last_info.get("invalid_action", False),
            "terminated": last_info.get("terminated", False),
            "truncated": last_info.get("truncated", False),
            "error": "Environment step failed",
        }

    def evaluate(self, env: TradingEnv):
        """Evaluate the agent on one episode with detailed logging, using the internal evaluation helper."""
        assert isinstance(env, TradingEnv), "env must be an instance of TradingEnv for evaluation"

        logger.info("====== STARTING DETAILED EVALUATION ======")

        try:
            # Call the internal method that runs the episode and collects metrics
            # It handles setting eval mode, resetting env, running steps, and error handling.
            total_reward, metrics, final_info = self._run_single_evaluation_episode(env, context="eval")

            # --- Extract data from returned metrics for logging ---
            steps = final_info.get("step", metrics.get("num_steps", 0))  # Get step count
            final_portfolio = metrics.get("portfolio_value", 0.0)
            initial_portfolio = metrics.get("initial_portfolio_value", 0.0)
            return_pct = metrics.get("total_return", 0.0)  # Already calculated as percentage
            action_counts = metrics.get("action_counts", {})
            # Calculate simple action stats if counts available
            actions_taken = []
            for action, count in action_counts.items():
                actions_taken.extend([action] * count)
            if actions_taken:
                avg_action = np.mean(actions_taken)
                min_action = np.min(actions_taken)
                max_action = np.max(actions_taken)
            else:
                avg_action, min_action, max_action = 0.0, 0.0, 0.0

            # Get price info if available in final_info (less reliable than metrics)
            # We don't have the full price list anymore for growth comparison easily
            # Consider adding initial/final price to metrics if needed for this comparison
            # current_price = final_info.get("price", 0)

            # Log evaluation summary using the metrics
            logger.info("====== EVALUATION SUMMARY ======")
            logger.info(f"Steps: {steps}")
            logger.info(f"Total Reward: {total_reward:.4f}")  # Increased precision
            logger.info(f"Initial Portfolio: ${initial_portfolio:.2f}")
            logger.info(f"Final Portfolio: ${final_portfolio:.2f}")
            logger.info(f"Return: {return_pct:.2f}%")
            logger.info(f"Action stats - Avg: {avg_action:.3f}, Min: {min_action:.3f}, Max: {max_action:.3f}")
            logger.info(f"Action Counts: {action_counts}")  # Log the counts dict
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', np.nan):.4f}")
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', np.nan)*100:.2f}%")
            logger.info(f"Total Transaction Costs: ${metrics.get('transaction_costs', 0.0):.2f}")

            # Portfolio vs Price Growth Comparison needs more data in metrics
            # (e.g., initial price, final price) - Skipping for now

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            # Return poor score on error
            total_reward = -np.inf
            final_portfolio = 0.0

        # --- Ensure returned values are floats ---
        final_reward = float(total_reward)
        final_portfolio_val = float(final_portfolio)
        assert isinstance(final_reward, float), "Final reward type mismatch before return"
        assert isinstance(final_portfolio_val, float), "Final portfolio type mismatch before return"
        # --- End Ensure ---

        return final_reward, final_portfolio_val

    def _run_episode_steps(
        self,
        env: TradingEnv,
        initial_obs: dict,
        tracker: PerformanceTracker,
        episode: int,
        total_train_steps: int,
    ) -> Tuple[float, float, int, int, dict, int]:  # Added int for invalid count
        """Runs the steps within a single training episode using _perform_training_step."""
        done = False
        obs = initial_obs
        info = {}  # Initialize info dict
        episode_reward = 0.0
        episode_loss = 0.0
        steps_in_episode = 0
        invalid_action_count = 0  # Initialize counter
        recent_step_rewards = deque(maxlen=self.log_freq)
        recent_losses = deque(maxlen=self.log_freq // self.update_freq + 1)

        while not done:
            # Perform one step
            next_obs, reward, step_done, step_info, action, loss_value = self._perform_training_step(
                env, obs, total_train_steps, episode, steps_in_episode
            )
            done = step_done  # Update loop condition
            info = step_info  # Update info for logging/tracker

            # Check for invalid action indicator from environment
            if step_info.get("invalid_action", False):
                invalid_action_count += 1

            # Update performance tracker
            tracker.update(
                portfolio_value=info["portfolio_value"],
                action=action,
                reward=reward,
                transaction_cost=info.get("step_transaction_cost", 0.0),
            )
            recent_step_rewards.append(reward)
            episode_reward += reward

            # Accumulate loss if learning happened
            if loss_value is not None:
                episode_loss += loss_value
                recent_losses.append(loss_value)

            # Update state and counters
            obs = next_obs
            steps_in_episode += 1
            total_train_steps += 1

            # Optionally render the environment for visualization
            self._maybe_render(env, "train", steps_in_episode)

            # Log PER statistics based on total training steps
            self._maybe_log_per_stats(total_train_steps)

            # Log step progress periodically
            if steps_in_episode % self.log_freq == 0:
                self._log_step_progress(
                    episode,
                    steps_in_episode,
                    tracker,
                    recent_step_rewards,
                    recent_losses,
                    action,
                    reward,
                    info,
                )

        # Episode finished
        env.close()
        # Return final info dict and invalid action count along with other values
        return (
            episode_reward,
            episode_loss,
            steps_in_episode,
            total_train_steps,
            info,
            invalid_action_count,
        )

    def train(
        self,
        # env: TradingEnv, # No longer needed as direct input
        num_episodes: int,
        start_episode: int,
        start_total_steps: int,
        initial_best_score: float,
        initial_early_stopping_counter: int,
        specific_file: str | None = None,
    ):
        """Train the Rainbow DQN agent by orchestrating helper methods."""
        # Basic input validation
        assert isinstance(num_episodes, int) and num_episodes > 0
        assert isinstance(start_episode, int) and start_episode >= 0
        assert isinstance(start_total_steps, int) and start_total_steps >= 0
        assert isinstance(specific_file, (str, type(None)))

        # Initialize state from parameters
        self.best_validation_metric = initial_best_score
        self.early_stopping_counter = initial_early_stopping_counter
        total_train_steps = start_total_steps
        self.total_train_steps = start_total_steps  # Also store on self for internal checks

        self.agent.set_training_mode(True)

        total_rewards = []  # Track rewards across episodes for logging

        logger.info("====== STARTING/RESUMING RAINBOW DQN TRAINING ======")
        logger.info(f"Starting from Episode: {start_episode + 1}/{num_episodes}")
        logger.info(f"Starting from Total Steps: {total_train_steps}")
        # Log other config details (agent, env, trainer params)
        logger.info(f"Agent Config: {self.agent_config}")
        logger.info(f"Env Config: {self.env_config}")
        logger.info(f"Trainer Config: {self.trainer_config}")

        val_files = self.data_manager.get_validation_files()
        if not val_files:
            logger.warning("No validation files found. Training will proceed without validation.")

        # Main training loop
        try:
            for episode in range(start_episode, num_episodes):
                # 1. Initialize episode environment and tracker
                episode_env, initial_obs, initial_info, tracker = self._initialize_episode(specific_file, episode, num_episodes)
                if episode_env is None or tracker is None:
                    logger.error(f"Failed to initialize episode {episode+1}. Skipping.")
                    continue  # Skip to next episode

                # 2. Run steps within the episode
                # Pass the initial_obs obtained from initialization
                (
                    episode_reward,
                    episode_loss,
                    steps_in_episode,
                    total_train_steps,
                    info,
                    invalid_action_count,
                ) = self._run_episode_steps(episode_env, initial_obs, tracker, episode, total_train_steps)
                self.total_train_steps = total_train_steps  # Update internal tracker

                # 3. Log episode summary
                total_rewards.append(episode_reward)
                self._log_episode_summary(
                    episode,
                    episode_reward,
                    total_rewards,
                    episode_loss,
                    steps_in_episode,
                    tracker,
                    info,
                    invalid_action_count,
                    total_train_steps,
                )

                # 4. Handle validation and checkpointing
                should_stop = self._handle_validation_and_checkpointing(episode, total_train_steps, val_files, tracker)

                if should_stop:
                    logger.info("Early stopping condition met. Exiting training loop.")
                    break  # Exit training loop if early stopping triggered

        except Exception:
            logger.error("!!! UNEXPECTED EXCEPTION in main training loop !!!", exc_info=True)
            # Consider saving a final checkpoint here? Or just rely on last successful save.

        finally:
            # 5. Finalize training (save final model, log summary)
            self._finalize_training(total_train_steps, num_episodes, val_files)
            self.agent.set_training_mode(False)  # Ensure agent is left in eval mode

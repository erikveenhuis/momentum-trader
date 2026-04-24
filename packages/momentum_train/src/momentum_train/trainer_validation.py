"""Auto-split from trainer.py (Tier 3.1).

Do not add new methods here directly; extend the mixin class in-place or
refactor through the facade. See .cursor/plans/prioritized-codebase-cleanup_*.plan.md.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from momentum_core.logging import get_logger
from momentum_env import TradingEnv, TradingEnvConfig

from .metrics import PerformanceTracker, calculate_episode_score
from .trade_metrics import aggregate_trade_metrics, segment_trades

logger = get_logger(__name__)


class ValidationMixin:
    """Validation / evaluation methods split from the monolithic trainer."""

    def _handle_validation_and_checkpointing(
        self,
        episode: int,
        total_train_steps: int,
        val_files: list[Path],
        tracker: PerformanceTracker,
        *,
        force_validation: bool = False,
        force_save: bool = False,
    ) -> bool:
        """Handles validation runs and checkpoint saving. Returns True if training should stop.

        ``force_validation`` / ``force_save`` bypass the internal
        ``(episode + 1) % freq == 0`` gates and are set by the vectorized
        training loop when its ``crossed_validation_boundary`` /
        ``crossed_checkpoint_boundary`` gate has already established that the
        relevant boundary was crossed on this step — even if multi-env jumps
        meant we never landed on the exact multiple. Without these overrides
        both validation and the paired best-checkpoint save silently no-op on
        every ``validation_freq`` boundary where ``num_vector_envs > 1`` caused
        a jump.
        """
        save_now = force_save or ((episode + 1) % self.checkpoint_save_freq == 0)
        should_stop_training = False
        is_best = False
        validation_score = -np.inf  # Default score
        avg_val_metrics = {}  # Initialize to empty dict

        # Store old best score for "is_best" decision
        old_best_validation_metric = self.best_validation_metric

        # Run validation if needed
        if val_files and self.should_validate(episode, tracker.get_recent_metrics(), force=force_validation):
            try:
                logger.info(f"--- Running validation after episode {episode + 1} ---")
                # MODIFIED: Capture avg_val_metrics
                should_stop_training, validation_score, avg_val_metrics = self.validate(val_files, episode)
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
            # During min_episodes_before_early_stopping, validate() does not update best; force is_best False so we do not save misleading "best" checkpoints.
            completed_episodes = episode + 1
            eligible_for_best = (
                self.min_episodes_before_early_stopping <= 0
                or completed_episodes >= self.min_episodes_before_early_stopping
            )
            if eligible_for_best and validation_score > old_best_validation_metric + self.min_validation_threshold:
                is_best = True
                # self.best_validation_metric is already updated by validate() to validation_score
                logger.info(
                    f"  >>> NEW BEST CHECKPOINT (Score: {validation_score:.4f} > Old best: {old_best_validation_metric:.4f} + Threshold: {self.min_validation_threshold}) <<< "
                )
            else:
                is_best = False
                if not eligible_for_best:
                    logger.info(
                        "  Skipping best checkpoint (completed %d/%d episodes before min_episodes_before_early_stopping).",
                        completed_episodes,
                        self.min_episodes_before_early_stopping,
                    )
                else:
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
                    if "avg_exposure_pct" in avg_val_metrics:
                        self.writer.add_scalar(
                            "Validation/Avg Exposure Pct",
                            avg_val_metrics.get("avg_exposure_pct", np.nan),
                            episode,
                        )
                        self.writer.add_scalar(
                            "Validation/Max Exposure Pct",
                            avg_val_metrics.get("max_exposure_pct", np.nan),
                            episode,
                        )
                    if "avg_position" in avg_val_metrics:
                        self.writer.add_scalar(
                            "Validation/Avg Position",
                            avg_val_metrics.get("avg_position", np.nan),
                            episode,
                        )
                    if "avg_abs_position" in avg_val_metrics:
                        self.writer.add_scalar(
                            "Validation/Avg Abs Position",
                            avg_val_metrics.get("avg_abs_position", np.nan),
                            episode,
                        )
                    if "avg_balance" in avg_val_metrics:
                        self.writer.add_scalar(
                            "Validation/Avg Balance",
                            avg_val_metrics.get("avg_balance", np.nan),
                            episode,
                        )
                    # Tier 1b: parity with Train/Final Portfolio Value and Train/Transaction Costs.
                    final_pv = avg_val_metrics.get("final_portfolio_value")
                    if final_pv is None:
                        final_pv = avg_val_metrics.get("portfolio_value")
                    if final_pv is not None and not (
                        isinstance(final_pv, float) and (math.isnan(final_pv) or math.isinf(final_pv))
                    ):
                        self.writer.add_scalar("Validation/Final Portfolio Value", float(final_pv), episode)
                    txn_costs = avg_val_metrics.get("transaction_costs")
                    if txn_costs is not None and not (
                        isinstance(txn_costs, float) and (math.isnan(txn_costs) or math.isinf(txn_costs))
                    ):
                        self.writer.add_scalar("Validation/Transaction Costs", float(txn_costs), episode)
                    # Tier 1b: per-action rate parity with Train/Action Rate/{0..5}.
                    action_rates = avg_val_metrics.get("action_rates", {}) or {}
                    for action_idx, rate in action_rates.items():
                        if rate is None:
                            continue
                        rate_f = float(rate)
                        if math.isnan(rate_f) or math.isinf(rate_f):
                            continue
                        self.writer.add_scalar(f"Validation/Action Rate/{int(action_idx)}", rate_f, episode)
                    # Tier 2b: per-trade KPIs (Validation/Trade/* scalars + PnL histogram).
                    trade_metrics = avg_val_metrics.get("trade_metrics", {}) or {}
                    for key, value in trade_metrics.items():
                        if value is None:
                            continue
                        try:
                            value_f = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isnan(value_f) or math.isinf(value_f):
                            continue
                        self.writer.add_scalar(f"Validation/Trade/{key}", value_f, episode)
                    trade_pnls = avg_val_metrics.get("trade_pnls", []) or []
                    if trade_pnls:
                        try:
                            self.writer.add_histogram(
                                "Validation/Trade/PnLDistribution",
                                np.asarray(trade_pnls, dtype=np.float32),
                                episode,
                            )
                        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                            logger.debug("Failed to emit Validation/Trade/PnLDistribution: %s", exc)

                # Tier 2d: Train/EvalGap/* — diff between greedy validation and
                # the most recent stochastic-training rolling window. Positive =
                # greedy outperforms training. NaN-safe via _safe_eval_gap_scalar.
                try:
                    train_recent = tracker.get_recent_metrics() if tracker is not None else {}
                except (ValueError, KeyError, AttributeError, TypeError):  # pragma: no cover - defensive
                    train_recent = {}
                if train_recent:
                    self._emit_eval_gap_scalars(avg_val_metrics, train_recent, episode)
            # ---------------------------------------------------- #

            # --- Step LR Scheduler if it's ReduceLROnPlateau --- #
            if self.agent.lr_scheduler_enabled and isinstance(
                self.agent.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
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
                        self.writer.add_scalar(
                            "Train/Learning_Rate", new_lr, total_train_steps
                        )  # Use total_train_steps or episode
                else:
                    logger.warning("Skipping ReduceLROnPlateau step due to invalid validation score (-np.inf).")
            # ---------------------------------------------------- #

            if should_stop_training and self.early_stopping_counter < self.early_stopping_patience:
                logger.info(
                    "[EarlyStopping] Counter reset after scheduler step; deferring stop to observe reduced learning rate."
                )
                should_stop_training = False

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

    def _perform_evaluation_step(self, env: TradingEnv, obs: dict) -> tuple[dict, float, bool, dict, int, bool]:
        """Performs a single step of evaluation in the environment. Returns (next_obs, reward, done, info, action, error_occurred)."""
        try:
            # Tier 2c: ask the agent for action provenance. In eval mode the agent
            # never explores, so was_greedy is True; capture it anyway so the
            # downstream per-trade aggregator (Tier 2b) can attribute trades.
            select_with_provenance = getattr(self.agent, "select_action_with_provenance", None)
            if callable(select_with_provenance):
                action, was_greedy = select_with_provenance(obs)
            else:  # backward-compat with stub agents in tests
                action = self.agent.select_action(obs)
                was_greedy = True
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if isinstance(info, dict):
                info.setdefault("was_greedy", bool(was_greedy))

            # --- ADDED: Check for non-numeric reward --- #
            error_occurred = False  # Initialize error flag for this check
            if not isinstance(done, (bool, np.bool_)):
                logger.error(f"done is not a bool: {type(done)}")
            if not isinstance(info, dict):
                logger.error(f"info is not a dict: {type(info)}")

            # --- Assert info structure --- #
            assert isinstance(info, dict), "Validation: Info from env.step() must be a dict"
            assert "portfolio_value" in info, "Validation: Info missing portfolio_value"
            assert isinstance(info["portfolio_value"], (float, np.float32, np.float64)), (
                "Validation: portfolio_value is not a float"
            )
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
    ) -> tuple[float, dict, dict]:
        """Evaluate the agent for one episode on a given environment instance."""
        # Removed assert isinstance(env, TradingEnv) because it fails when TradingEnv is patched in tests
        # assert isinstance(
        #     env, TradingEnv
        # ), "env must be an instance of TradingEnv for evaluation"
        # Tier 2d: by default put the agent in deterministic ``greedy()`` mode
        # for evaluation. ``--eval-stochastic`` (config: ``run.eval_stochastic``)
        # leaves the agent in its current (training) mode so we can measure the
        # gap between the policy being trained and its greedy projection.
        was_training = self.agent.training_mode
        if not getattr(self, "eval_stochastic", False):
            self.agent.set_training_mode(False)
        tracker = None  # Initialize tracker to None
        final_info = {}  # Initialize final_info
        total_reward = -np.inf  # Default reward if reset fails
        metrics = {}  # Default metrics

        try:
            obs, info = env.reset()
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
            # Tier 2b: per-step trace used by trade segmentation. Kept lightweight
            # (pure floats / ints) so memory stays bounded even on long episodes.
            step_records: list[dict] = []
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
                        position=info.get("position"),
                        balance=info.get("balance"),
                        price=info.get("price"),
                    )
                    # Tier 2b: capture the minimal fields the trade segmenter needs.
                    step_records.append(
                        {
                            "step_index": step_index,
                            "portfolio_value": float(info["portfolio_value"]),
                            "position": float(info.get("position", 0.0) or 0.0),
                            "price": float(info.get("price", 0.0) or 0.0),
                            "action": int(action),
                            "transaction_cost": float(info.get("step_transaction_cost", 0.0) or 0.0),
                            # Provenance is filled in by Tier 2c; default to None so the
                            # segmenter reports pct_greedy_actions=NaN until then.
                            "was_greedy": info.get("was_greedy"),
                        }
                    )
                    total_reward += reward
                    obs = next_obs
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

                # Tier 2b: trade segmentation + per-trade economics aggregation.
                # ``trades`` is a list of dicts (JSON-serializable for sidecar files);
                # ``trade_metrics`` is the flat dict consumed by the validation/test
                # TensorBoard emitters (see _handle_validation_and_checkpointing and
                # run_training._emit_trade_metrics).
                try:
                    trades = segment_trades(step_records)
                except (ValueError, KeyError, TypeError, IndexError) as exc:  # pragma: no cover - defensive
                    logger.warning(f"Trade segmentation failed in {context} episode: {exc}")
                    trades = []
                metrics["trades"] = [t.to_dict() for t in trades]
                metrics["trade_metrics"] = aggregate_trade_metrics(trades)
                metrics["total_steps"] = step_index
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
        self,
        val_file: Path,
        validation_episode: int = 0,
        total_validation_episodes: int = 1,
        context: str = "validation",
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
            logger.debug(f"    Max Drawdown: {file_metrics.get('max_drawdown', np.nan) * 100:.2f}%")
            logger.debug(f"    Action Counts: {file_metrics.get('action_counts', {})}")
            logger.debug(f"    Transaction Costs: ${file_metrics.get('transaction_costs', np.nan):.2f}")
            logger.debug(f"    Avg Exposure: {file_metrics.get('avg_exposure_pct', np.nan):.2f}%")
            logger.debug(f"    Max Exposure: {file_metrics.get('max_exposure_pct', np.nan):.2f}%")
            logger.debug(f"    Avg Position: {file_metrics.get('avg_position', np.nan):.4f}")
            logger.debug(f"    Avg Balance: ${file_metrics.get('avg_balance', np.nan):.2f}")
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
                "avg_exposure_pct": float(file_metrics.get("avg_exposure_pct", np.nan)),
                "max_exposure_pct": float(file_metrics.get("max_exposure_pct", np.nan)),
                "avg_position": float(file_metrics.get("avg_position", np.nan)),
                "avg_abs_position": float(file_metrics.get("avg_abs_position", np.nan)),
                "avg_balance": float(file_metrics.get("avg_balance", np.nan)),
                "avg_position_value": float(file_metrics.get("avg_position_value", np.nan)),
                # Tier 2b: surface per-trade economics so run_training._emit_trade_metrics
                # and the validation TB block can mirror them. ``trades`` is also kept
                # so the JSONL sidecar / offline analyzer (Tier 5c) can consume them.
                "trade_metrics": dict(file_metrics.get("trade_metrics", {}))
                if file_metrics.get("trade_metrics")
                else {},
                "trades": list(file_metrics.get("trades", [])),
                "total_steps": int(file_metrics.get("total_steps", 0) or 0),
                "action_counts": dict(file_metrics.get("action_counts", {}) or {}),
            }
            # Persist per-trade JSONL sidecar so downstream tools (Tier 5c analyzer,
            # KPI dashboards) have an append-only stream of trades per validation file.
            if detailed_result["trades"]:
                self._write_trades_jsonl(val_file, detailed_result["trades"], context=context)
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

    def _write_trades_jsonl(self, val_file: Path, trades: list[dict], *, context: str) -> None:
        """Append per-trade records to ``<model_dir>/trades_<context>.jsonl``.

        The sidecar carries enough information (file name, entry/exit indices,
        PnL %, MAE, MFE, transaction cost, action provenance) for the offline
        analyzer (Tier 5c) and any external dashboard to reconstruct the trade
        timeline without rerunning the agent.
        """
        if not trades:
            return
        try:
            sidecar_path = Path(self.model_dir) / f"trades_{context}.jsonl"
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(sidecar_path, "a", encoding="utf-8") as f:
                for trade in trades:
                    record = dict(trade)
                    record["file"] = val_file.name
                    record["context"] = context
                    record["written_at"] = timestamp
                    f.write(json.dumps(record, default=float) + "\n")
        except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - persistence is best-effort
            logger.debug("Failed to write trades JSONL sidecar: %s", exc)

    def close_cached_environments(self) -> None:
        """Close any cached validation/test environments."""
        for env_key, env in list(self._validation_env_cache.items()):
            try:
                env.close()
            except (OSError, RuntimeError, AttributeError) as exc:  # pragma: no cover - best effort cleanup
                logger.warning(f"Failed to close cached environment {env_key}: {exc}")
        self._validation_env_cache.clear()

    def _calculate_average_validation_metrics(self, all_file_metrics: list[dict]) -> dict:
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
        avg_positions = [m.get("avg_position", np.nan) for m in all_file_metrics]
        avg_abs_positions = [m.get("avg_abs_position", np.nan) for m in all_file_metrics]
        avg_balances = [m.get("avg_balance", np.nan) for m in all_file_metrics]
        avg_exposures = [m.get("avg_exposure_pct", np.nan) for m in all_file_metrics]
        max_exposures = [m.get("max_exposure_pct", np.nan) for m in all_file_metrics]
        avg_position_values = [m.get("avg_position_value", np.nan) for m in all_file_metrics]

        # Per-action rate aggregation (rate = count/total_steps per file, then mean across files).
        # See logging plan Tier 1b: parity with Train/Action Rate/{0..5} and Test/Action Rate/{0..5}.
        num_actions = int(getattr(self.agent, "num_actions", 6))
        per_file_rates: list[dict[int, float]] = []
        for m in all_file_metrics:
            action_counts = m.get("action_counts", {}) or {}
            steps = float(m.get("total_steps", 0) or 0)
            if steps <= 0:
                continue
            rates: dict[int, float] = {}
            for a in range(num_actions):
                count = float(action_counts.get(a, 0) or 0)
                rates[a] = count / steps
            per_file_rates.append(rates)
        action_rates: dict[int, float] = {}
        if per_file_rates:
            for a in range(num_actions):
                action_rates[a] = float(np.mean([r.get(a, 0.0) for r in per_file_rates]))

        # Tier 2b: average each per-trade KPI across files (skipping NaN values
        # so files with zero trades don't poison the aggregate). Also flatten the
        # per-trade PnL list for histogram emission.
        trade_payloads = [
            m.get("trade_metrics", {})
            for m in all_file_metrics
            if isinstance(m.get("trade_metrics"), dict) and m.get("trade_metrics")
        ]
        avg_trade_metrics: dict[str, float] = {}
        if trade_payloads:
            keys: set[str] = set()
            for payload in trade_payloads:
                keys.update(payload.keys())
            for key in keys:
                values = [
                    float(payload[key])
                    for payload in trade_payloads
                    if key in payload
                    and payload[key] is not None
                    and isinstance(payload[key], (int, float, np.integer, np.floating))
                    and np.isfinite(payload[key])
                ]
                if values:
                    avg_trade_metrics[key] = float(np.mean(values))
        all_trade_pnls: list[float] = []
        for m in all_file_metrics:
            for trade in m.get("trades", []) or []:
                pnl = trade.get("pnl_pct") if isinstance(trade, dict) else None
                if pnl is not None and np.isfinite(pnl):
                    all_trade_pnls.append(float(pnl))

        return {
            "avg_reward": float(np.nanmean(rewards)),
            "portfolio_value": float(np.nanmean(portfolios)),
            "final_portfolio_value": float(np.nanmean(portfolios)),
            "total_return": float(np.nanmean(returns)),
            "sharpe_ratio": float(np.nanmean(sharpes)),
            "max_drawdown": float(np.nanmean(drawdowns)),
            "transaction_costs": float(np.nanmean(costs)),
            "avg_position": float(np.nanmean(avg_positions)),
            "avg_abs_position": float(np.nanmean(avg_abs_positions)),
            "avg_balance": float(np.nanmean(avg_balances)),
            "avg_exposure_pct": float(np.nanmean(avg_exposures)),
            "max_exposure_pct": float(np.nanmean(max_exposures)),
            "avg_position_value": float(np.nanmean(avg_position_values)),
            "action_rates": action_rates,
            "trade_metrics": avg_trade_metrics,
            "trade_pnls": all_trade_pnls,
        }

    def _save_validation_results(self, validation_score: float, avg_metrics: dict, detailed_results: list[dict]):
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
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=4)
            logger.info(f"Validation results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

    def _check_early_stopping(self, validation_score: float, episode: int) -> bool:
        """Checks the early stopping condition based on validation score."""
        completed_episodes = episode + 1
        if self.min_episodes_before_early_stopping > 0 and completed_episodes < self.min_episodes_before_early_stopping:
            logger.info(
                "Completed %d/%d episodes: logging validation score but deferring best-validation tracking and early stopping.",
                completed_episodes,
                self.min_episodes_before_early_stopping,
            )
            return False
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
                logger.info(
                    f"Early stopping triggered after {self.early_stopping_counter} episodes without improvement."
                )
                should_stop = True
            else:
                should_stop = False

        return should_stop

    # --- End Validation Helper Methods ---

    def validate(self, val_files: list[Path], episode: int = 0) -> tuple[bool, float, dict]:
        """Run validation on validation files using helper methods, log, and check for early stopping."""

        # Handle empty validation file list
        if not val_files:
            logger.warning(
                "validate() called with empty val_files list. Returning default score -inf and empty metrics."
            )
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
                logger.warning(
                    "At least one validation episode failed (score=-inf). Overall validation score set to -inf."
                )
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
            logger.info(f"  Average Max Drawdown: {avg_metrics['max_drawdown'] * 100:.2f}%")
            logger.info(f"Average Transaction Costs: ${avg_metrics['transaction_costs']:.2f}")
            logger.info(f"Average Exposure: {avg_metrics['avg_exposure_pct']:.2f}%")
            logger.info(f"Average Max Exposure: {avg_metrics['max_exposure_pct']:.2f}%")
            logger.info(f"Average Position: {avg_metrics['avg_position']:.4f}")
            logger.info(f"Average Abs Position: {avg_metrics['avg_abs_position']:.4f}")
            logger.info(f"Average Balance: ${avg_metrics['avg_balance']:.2f}")
            logger.info("============================================")

            self._log_progress(
                "validation",
                score=round(validation_score, 4),
                best=round(self.best_validation_metric, 4),
                avg_return=round(avg_metrics["total_return"], 2),
                sharpe=round(avg_metrics["sharpe_ratio"], 4),
                max_dd=round(avg_metrics["max_drawdown"] * 100, 2),
                pv=round(avg_metrics["portfolio_value"], 2),
                early_stop_counter=self.early_stopping_counter,
            )

            # 5. Save validation results
            self._save_validation_results(validation_score, avg_metrics, detailed_results)

            # 6. Check for early stopping
            should_stop = self._check_early_stopping(validation_score, episode)

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
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', np.nan) * 100:.2f}%")
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

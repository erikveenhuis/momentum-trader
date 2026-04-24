"""Auto-split from trainer.py (Tier 3.1).

Do not add new methods here directly; extend the mixin class in-place or
refactor through the facade. See .cursor/plans/prioritized-codebase-cleanup_*.plan.md.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
from momentum_agent.constants import ACCOUNT_STATE_DIM
from momentum_core.logging import get_logger
from momentum_env import TradingEnv, TradingEnvConfig

from .metrics import PerformanceTracker

logger = get_logger(__name__)


class LoopMixin:
    """Episode + vectorized training loop methods split from the monolithic trainer."""

    def _initialize_episode(
        self, specific_file: str | None, episode: int, num_episodes: int
    ) -> tuple[TradingEnv | None, dict | None, dict | None, PerformanceTracker | None]:
        """Sets up the environment and performance tracker for a new episode. Returns env, obs, info, tracker."""
        try:
            curriculum_frac = min(1.0, 0.3 + 0.7 * (episode / max(num_episodes, 1)))
            episode_file_path = (
                Path(specific_file)
                if specific_file
                else self.data_manager.get_random_training_file(curriculum_frac=curriculum_frac)
            )
            logger.info(
                f"--- Starting Episode {episode + 1}/{num_episodes} using file: {episode_file_path.name} (curriculum={curriculum_frac:.2f}) ---"
            )
        except Exception as e:
            logger.error(f"Error getting data file for episode {episode + 1}: {e}")
            return None, None, None, None  # Indicate failure

        try:
            # Update env_config with the current episode file path
            self.env_config["data_path"] = str(episode_file_path)

            # Create a TradingEnvConfig object
            env_config_obj = TradingEnvConfig(**self.env_config)

            env = TradingEnv(config=env_config_obj)
            obs, info = env.reset()
            applied_frac = self._apply_benchmark_frac_to_env(env, episode)
            self._maybe_emit_benchmark_frac(episode, applied_frac)
            assert isinstance(info["portfolio_value"], (float, np.float32, np.float64)), (
                "Reset info missing valid portfolio_value"
            )
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
    ) -> tuple[dict, float, bool, dict, int, float | None]:
        """Performs a single step of interaction with the environment and learning."""
        loss_value = None  # Initialize loss_value

        # Assert observation shape before selecting action
        assert obs["market_data"].shape == (self.agent.window_size, self.agent.n_features)
        assert obs["account_state"].shape == (ACCOUNT_STATE_DIM,)

        # Select action
        # Tier 2c: track action provenance (greedy vs epsilon-forced) so the
        # per-action greedy/eps split is visible in TB and so trade segmentation
        # downstream can attribute trades. Warmup actions are sampled uniformly
        # from the env action space → treat them as non-greedy ("eps") for the
        # purposes of action-rate split.
        if total_train_steps < self.warmup_steps:
            action = env.action_space.sample()
            was_greedy = False
        else:
            self.agent.env_steps = total_train_steps - self.warmup_steps
            select_with_provenance = getattr(self.agent, "select_action_with_provenance", None)
            if callable(select_with_provenance):
                action, was_greedy = select_with_provenance(obs)
            else:  # backward-compat with stub agents
                action = self.agent.select_action(obs)
                was_greedy = True

        # Step environment
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminated_flag = bool(terminated)
            truncated_flag = bool(truncated)
            done = terminated_flag or truncated_flag
            if isinstance(info, dict):
                info.setdefault("terminated", terminated_flag)
                info.setdefault("truncated", truncated_flag)
                info.setdefault("was_greedy", bool(was_greedy))
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

            reward = float(reward)
            original_reward = reward
            if self.reward_clip_value is not None:
                clipped_reward = float(np.clip(reward, -self.reward_clip_value, self.reward_clip_value))
                if clipped_reward != reward:
                    logger.debug(
                        "Reward clipped from %.6f to %.6f at episode %s step %s.",
                        reward,
                        clipped_reward,
                        episode,
                        steps_in_episode,
                    )
                reward = clipped_reward
                if isinstance(info, dict):
                    info["unclipped_reward"] = original_reward
                    info["clipped_reward"] = reward

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
                reward = -1.0
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
                            self.writer.add_scalar(
                                "Train/TD_Error_Mean", td_stats.get("mean", float("nan")), total_train_steps
                            )
                            self.writer.add_scalar(
                                "Train/TD_Error_Std", td_stats.get("std", float("nan")), total_train_steps
                            )
                        last_entropy = getattr(self.agent, "last_entropy", None)
                        if last_entropy is not None:
                            self.writer.add_scalar("Train/Action_Entropy", last_entropy, total_train_steps)
                    # ---------------------------- #

            except FloatingPointError as exc:
                logger.error(
                    f"!!! EXCEPTION during learning update at step {total_train_steps} !!!",
                    exc_info=True,
                )
                if not self._abort_training:
                    self._abort_training = True
                    self._abort_reason = f"FloatingPointError during learning update: {exc}"
                    self._abort_step = total_train_steps
                done = True  # Stop episode on learning error
            except Exception:
                logger.error(
                    f"!!! EXCEPTION during learning update at step {total_train_steps} !!!",
                    exc_info=True,
                )
                done = True  # Stop episode on learning error

        return next_obs, reward, done, info, action, loss_value

    def _run_episode_steps(
        self,
        env: TradingEnv,
        initial_obs: dict,
        tracker: PerformanceTracker,
        episode: int,
        total_train_steps: int,
    ) -> tuple[float, float, int, int, dict, int]:
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
                position=info.get("position"),
                balance=info.get("balance"),
                price=info.get("price"),
                was_greedy=info.get("was_greedy"),
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

    # ------------------------------------------------------------------
    # Vectorized training loop
    # ------------------------------------------------------------------

    def _train_vectorized(
        self,
        num_episodes: int,
        start_episode: int,
        total_train_steps: int,
        val_files: list,
    ) -> int:
        """Step-based training loop with *num_vector_envs* parallel environments.

        Returns the final ``total_train_steps``.
        """
        from .vector_env import create_vector_env, reset_done_envs

        num_envs = self.num_vector_envs
        self.agent.set_num_envs(num_envs)
        logger.info(f"Vectorized training with {num_envs} parallel envs (SyncVectorEnv, DISABLED autoreset)")

        vec_env = create_vector_env(num_envs, self.env_config, self.data_manager)

        # Initial reset with curriculum files
        curriculum_frac = min(1.0, 0.3 + 0.7 * (start_episode / max(num_episodes, 1)))
        for i in range(num_envs):
            path = str(self.data_manager.get_random_training_file(curriculum_frac=curriculum_frac))
            vec_env.envs[i].reset(options={"data_path": path})
        obs, _info = vec_env.reset()

        # Push scheduled benchmark frac into every env at startup so the very
        # first step uses the resumed/scheduled value, not the constructor
        # default coming from env_config.
        initial_benchmark_frac = self.current_benchmark_frac(start_episode)
        for i in range(num_envs):
            try:
                vec_env.envs[i].trading_logic.set_benchmark_allocation_frac(initial_benchmark_frac)
            except AttributeError:
                logger.debug(
                    "Vector env %d exposes no trading_logic.set_benchmark_allocation_frac; skipping.",
                    i,
                )
        self._maybe_emit_benchmark_frac(start_episode, initial_benchmark_frac)

        # Per-env tracking
        per_env_episode_reward = np.zeros(num_envs)
        per_env_steps = np.zeros(num_envs, dtype=int)
        per_env_trackers = [PerformanceTracker() for _ in range(num_envs)]
        for i in range(num_envs):
            initial_info = vec_env.envs[i]._get_info()
            per_env_trackers[i].add_initial_value(initial_info["portfolio_value"])

        completed_episodes = start_episode
        # Track previous count so the validation/checkpoint gates fire even when
        # multiple envs finish on the same vec_env.step (which can make
        # ``completed_episodes`` jump by N in one iteration and silently skip a
        # multiple of validation_freq with the naive ``% == 0`` check).
        prev_completed_episodes = start_episode
        total_rewards: list[float] = []
        steps_since_last_learn = 0

        # Aggregate action distribution across all envs (rolling window)
        vec_action_counts = np.zeros(self.agent.num_actions, dtype=int)
        vec_action_window = deque(maxlen=10_000)

        while completed_episodes < num_episodes:
            if self._abort_training:
                break

            # Select actions (Tier 2c: capture greedy/eps provenance per env)
            if total_train_steps < self.warmup_steps:
                actions = np.array([vec_env.envs[i].action_space.sample() for i in range(num_envs)])
                was_greedy_batch = np.zeros(num_envs, dtype=bool)
            else:
                self.agent.env_steps = total_train_steps - self.warmup_steps
                select_batch_with_provenance = getattr(self.agent, "select_actions_batch_with_provenance", None)
                if callable(select_batch_with_provenance):
                    actions, was_greedy_batch = select_batch_with_provenance(obs)
                else:  # backward-compat with stub agents
                    actions = self.agent.select_actions_batch(obs)
                    was_greedy_batch = np.ones(num_envs, dtype=bool)

            # Step all envs
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            # Process each env's transition
            for i in range(num_envs):
                reward_i = float(rewards[i])
                if self.reward_clip_value is not None:
                    reward_i = float(np.clip(reward_i, -self.reward_clip_value, self.reward_clip_value))

                obs_i = {"market_data": obs["market_data"][i], "account_state": obs["account_state"][i]}
                next_obs_i = {"market_data": next_obs["market_data"][i], "account_state": next_obs["account_state"][i]}

                self.agent.store_transition(obs_i, int(actions[i]), reward_i, next_obs_i, bool(dones[i]), env_id=i)

                per_env_episode_reward[i] += reward_i
                per_env_steps[i] += 1

                info_i = (
                    {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in infos.items()}
                    if isinstance(infos, dict)
                    else {}
                )
                per_env_trackers[i].update(
                    portfolio_value=info_i.get("portfolio_value", 0.0),
                    action=int(actions[i]),
                    reward=reward_i,
                    transaction_cost=info_i.get("step_transaction_cost", 0.0),
                    position=info_i.get("position"),
                    balance=info_i.get("balance"),
                    price=info_i.get("price"),
                    was_greedy=bool(was_greedy_batch[i]),
                )

            for a in actions:
                vec_action_counts[int(a)] += 1
                vec_action_window.append(int(a))

            total_train_steps += num_envs
            self.total_train_steps = total_train_steps
            steps_since_last_learn += num_envs

            # Threshold-based learning schedule
            if len(self.agent.buffer) >= self.agent.batch_size and total_train_steps > self.warmup_steps:
                while steps_since_last_learn >= self.update_freq:
                    try:
                        for _ in range(self.gradient_updates_per_step):
                            loss = self.agent.learn()
                            if loss is None:
                                break
                            if self.writer:
                                self.writer.add_scalar("Train/Loss", float(loss), total_train_steps)
                    except Exception:
                        logger.error("Exception during learning update", exc_info=True)
                        self._abort_training = True
                        self._abort_reason = "Learning error in vectorized loop"
                        self._abort_step = total_train_steps
                        break
                    steps_since_last_learn -= self.update_freq

            # Handle done envs
            if dones.any():
                done_indices = np.where(dones)[0]
                for i in done_indices:
                    completed_episodes += 1
                    total_rewards.append(per_env_episode_reward[i])

                    ep_metrics = per_env_trackers[i].get_metrics()
                    ep_action_counts = ep_metrics.get("action_counts", {})
                    ep_steps = int(per_env_steps[i])
                    ep_reward = per_env_episode_reward[i]

                    should_log_full = completed_episodes <= 5 or completed_episodes % max(1, self.log_freq) == 0

                    if should_log_full:
                        avg_rw = np.mean(total_rewards[-self.reward_window :])
                        curriculum_frac_now = min(1.0, 0.3 + 0.7 * (completed_episodes / max(num_episodes, 1)))

                        act_str = " ".join(f"{k}:{v}" for k, v in sorted(ep_action_counts.items()))
                        # Rolling action distribution from recent window
                        if vec_action_window:
                            window_counts = np.bincount(
                                list(vec_action_window),
                                minlength=self.agent.num_actions,
                            )
                            window_total = window_counts.sum()
                            act_pct = " ".join(
                                f"{ai}:{window_counts[ai] * 100 / window_total:.0f}%"
                                for ai in range(self.agent.num_actions)
                            )
                        else:
                            act_pct = "n/a"

                        logger.info(
                            f"[Vec] Ep {completed_episodes}/{num_episodes} (env {i}) | "
                            f"steps={ep_steps} reward={ep_reward:.4f} "
                            f"avg_{self.reward_window}={avg_rw:.4f} | "
                            f"PV=${ep_metrics.get('portfolio_value', 0):.2f} "
                            f"ret={ep_metrics.get('total_return', 0):.2f}% "
                            f"sharpe={ep_metrics.get('sharpe_ratio', 0):.4f} "
                            f"dd={ep_metrics.get('max_drawdown', 0) * 100:.2f}% | "
                            f"actions=[{act_str}] dist=[{act_pct}] | "
                            f"curriculum={curriculum_frac_now:.2f} "
                            f"eps={self.agent.current_epsilon:.4f} "
                            f"lr={self.agent.optimizer.param_groups[0]['lr']:.2e} "
                            f"total_steps={total_train_steps}"
                        )

                    self._log_progress(
                        "episode",
                        episode=completed_episodes,
                        steps=ep_steps,
                        total_steps=total_train_steps,
                        reward=round(ep_reward, 4),
                        avg_reward=round(np.mean(total_rewards[-self.reward_window :]), 4),
                        total_return=round(ep_metrics.get("total_return", 0.0), 2) if ep_metrics else 0.0,
                        sharpe=round(ep_metrics.get("sharpe_ratio", 0.0), 4) if ep_metrics else 0.0,
                        max_dd=round(ep_metrics.get("max_drawdown", 0.0) * 100, 2) if ep_metrics else 0.0,
                        action_counts=ep_action_counts,
                        lr=self.agent.optimizer.param_groups[0]["lr"],
                        pv=round(ep_metrics.get("portfolio_value", 0.0), 2) if ep_metrics else 0.0,
                        env_id=int(i),
                    )

                    if self.writer:
                        self.writer.add_scalar("Train/Episode Reward", ep_reward, completed_episodes)
                        self.writer.add_scalar(
                            f"Train/Average Reward ({self.reward_window} ep)",
                            np.mean(total_rewards[-self.reward_window :]),
                            completed_episodes,
                        )
                        self.writer.add_scalar("Train/Steps Per Episode", ep_steps, completed_episodes)
                        if ep_metrics:
                            self.writer.add_scalar(
                                "Train/Total Return Pct", ep_metrics.get("total_return", 0), completed_episodes
                            )
                            self.writer.add_scalar(
                                "Train/Sharpe Ratio", ep_metrics.get("sharpe_ratio", 0), completed_episodes
                            )
                            self.writer.add_scalar(
                                "Train/Max Drawdown Pct", ep_metrics.get("max_drawdown", 0) * 100, completed_episodes
                            )
                            self.writer.add_scalar(
                                "Train/Transaction Costs", ep_metrics.get("transaction_costs", 0), completed_episodes
                            )
                            for action_idx, count in ep_action_counts.items():
                                rate = count / max(ep_steps, 1)
                                self.writer.add_scalar(f"Train/Action Rate/{action_idx}", rate, completed_episodes)
                            # Tier 2c: greedy/eps split + epsilon-forced trade fraction.
                            ep_provenance = ep_metrics.get("action_provenance_counts", {}) or {}
                            ep_greedy = ep_provenance.get("greedy", {}) or {}
                            ep_eps = ep_provenance.get("eps", {}) or {}
                            if ep_greedy or ep_eps:
                                steps_safe = max(ep_steps, 1)
                                for action_idx in range(int(getattr(self.agent, "num_actions", 6))):
                                    gr = float(ep_greedy.get(action_idx, 0) or 0) / steps_safe
                                    er = float(ep_eps.get(action_idx, 0) or 0) / steps_safe
                                    self.writer.add_scalar(
                                        f"Train/Action Rate/Greedy/{action_idx}", gr, completed_episodes
                                    )
                                    self.writer.add_scalar(
                                        f"Train/Action Rate/Eps/{action_idx}", er, completed_episodes
                                    )
                            self.writer.add_scalar(
                                "Train/EpsilonForcedTradeFraction",
                                float(ep_metrics.get("epsilon_forced_trade_fraction", 0.0) or 0.0),
                                completed_episodes,
                            )
                            # Tier 2c: per-episode Train/Trade/* (HitRate/Expectancy/PctGreedy/...)
                            try:
                                vec_trade_metrics = self._trade_metrics_from_tracker(per_env_trackers[i])
                            except (
                                ValueError,
                                KeyError,
                                AttributeError,
                                TypeError,
                                IndexError,
                            ):  # pragma: no cover - defensive
                                logger.debug("Failed to compute Train/Trade metrics for env %d", i, exc_info=True)
                                vec_trade_metrics = {}
                            for key, value in vec_trade_metrics.items():
                                if not isinstance(value, (int, float, np.floating, np.integer)):
                                    continue
                                fv = float(value)
                                if math.isnan(fv) or math.isinf(fv):
                                    continue
                                tag = "".join(part.capitalize() for part in str(key).split("_"))
                                self.writer.add_scalar(f"Train/Trade/{tag}", fv, completed_episodes)
                            # Tier 4b: per-episode reward outlier guard (vectorized loop).
                            try:
                                vec_outlier_stats = per_env_trackers[i].get_reward_outlier_stats(self.reward_clip_value)
                            except (ValueError, KeyError, AttributeError, TypeError):  # pragma: no cover - defensive
                                logger.debug(
                                    "Failed to compute Tier 4b reward outlier stats for env %d", i, exc_info=True
                                )
                                vec_outlier_stats = {}
                            if vec_outlier_stats:
                                self.writer.add_scalar(
                                    "Train/Episode/RewardMin", vec_outlier_stats["reward_min"], completed_episodes
                                )
                                self.writer.add_scalar(
                                    "Train/Episode/RewardMax", vec_outlier_stats["reward_max"], completed_episodes
                                )
                                self.writer.add_scalar(
                                    "Train/Episode/RewardP99Abs",
                                    vec_outlier_stats["reward_p99_abs"],
                                    completed_episodes,
                                )
                                self.writer.add_scalar(
                                    "Train/Episode/RewardOutlierFlag",
                                    vec_outlier_stats["reward_outlier_flag"],
                                    completed_episodes,
                                )
                            # Tier 4c: per-action reward mean/std (vectorized).
                            try:
                                vec_by_action = per_env_trackers[i].get_reward_by_action_stats()
                            except (ValueError, KeyError, AttributeError, TypeError):  # pragma: no cover - defensive
                                logger.debug(
                                    "Failed to compute Tier 4c reward-by-action stats for env %d",
                                    i,
                                    exc_info=True,
                                )
                                vec_by_action = {}
                            for k in range(int(getattr(self.agent, "num_actions", 6))):
                                bucket = vec_by_action.get(k, {"mean": 0.0, "std": 0.0})
                                self.writer.add_scalar(
                                    f"Train/Reward/MeanByAction/{k}", float(bucket["mean"]), completed_episodes
                                )
                                self.writer.add_scalar(
                                    f"Train/Reward/StdByAction/{k}", float(bucket["std"]), completed_episodes
                                )
                        self.writer.add_scalar("Train/Epsilon", self.agent.current_epsilon, completed_episodes)
                        self.writer.add_scalar(
                            "Train/Final Portfolio Value", ep_metrics.get("portfolio_value", 0), completed_episodes
                        )

                    # Vectorized episodes are long; flush after each one so the
                    # gap-on-freeze window is bounded by a single env-episode.
                    self._flush_writer()

                    per_env_episode_reward[i] = 0.0
                    per_env_steps[i] = 0
                    per_env_trackers[i] = PerformanceTracker()

                # Validation and checkpointing at episode boundaries.
                # Use a "crossed a multiple of N" gate (rather than ``% == 0``)
                # so that when several envs finish on the same step and
                # ``completed_episodes`` jumps over a multiple of
                # validation_freq / checkpoint_save_freq, we still trigger
                # exactly one validation/checkpoint event for that boundary.
                validation_freq_safe = max(1, int(self.validation_freq))
                checkpoint_freq_safe = max(1, int(self.checkpoint_save_freq))
                crossed_validation_boundary = (completed_episodes // validation_freq_safe) > (
                    prev_completed_episodes // validation_freq_safe
                )
                crossed_checkpoint_boundary = (completed_episodes // checkpoint_freq_safe) > (
                    prev_completed_episodes // checkpoint_freq_safe
                )
                # One-time visibility scalar: would the legacy ``% == 0`` check
                # have skipped this boundary? Useful while operators get used to
                # the new behaviour.
                if (
                    crossed_validation_boundary
                    and (completed_episodes % validation_freq_safe != 0)
                    and self.writer is not None
                ):
                    try:
                        self.writer.add_scalar(
                            "Train/Diagnostics/ValidationSkippedDueToVectorJump",
                            1.0,
                            completed_episodes,
                        )
                    except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive
                        logger.debug("Failed to emit Train/Diagnostics/ValidationSkippedDueToVectorJump", exc_info=True)

                if crossed_validation_boundary:
                    last_tracker = per_env_trackers[done_indices[-1]]
                    # Force both validation and the paired checkpoint save:
                    # the outer gate has already proven the boundary was
                    # crossed, and the internal ``% == 0`` checks would
                    # otherwise silently drop the event whenever a vec-env
                    # step jumps past the exact multiple (e.g. completed
                    # goes 1998 -> 2002 with 4 parallel envs, which is how
                    # ep2000/2200/2400 validations and checkpoints went
                    # missing on the April 21-22 run).
                    should_stop = self._handle_validation_and_checkpointing(
                        completed_episodes - 1,
                        total_train_steps,
                        val_files,
                        last_tracker,
                        force_validation=True,
                        force_save=True,
                    )
                    if should_stop:
                        logger.info("Early stopping condition met. Exiting vectorized training loop.")
                        break
                elif crossed_checkpoint_boundary:
                    self._save_checkpoint(
                        completed_episodes - 1,
                        total_train_steps,
                        is_best=False,
                        validation_score=None,
                    )

                prev_completed_episodes = completed_episodes

                # Reset done envs with new data
                curriculum_frac = min(1.0, 0.3 + 0.7 * (completed_episodes / max(num_episodes, 1)))
                next_obs = reset_done_envs(vec_env, dones, self.data_manager, curriculum_frac)

                # Push scheduled benchmark frac into the freshly-reset envs.
                # Anchored on ``completed_episodes`` so the schedule advances
                # smoothly across the run regardless of how many envs finish
                # together on a single vec_env.step.
                fresh_benchmark_frac = self.current_benchmark_frac(completed_episodes)
                for i in done_indices:
                    try:
                        vec_env.envs[i].trading_logic.set_benchmark_allocation_frac(fresh_benchmark_frac)
                    except AttributeError:
                        logger.debug(
                            "Vector env %d exposes no trading_logic.set_benchmark_allocation_frac; skipping.",
                            i,
                        )
                self._maybe_emit_benchmark_frac(completed_episodes, fresh_benchmark_frac)

                for i in done_indices:
                    init_info = vec_env.envs[i]._get_info()
                    per_env_trackers[i].add_initial_value(init_info["portfolio_value"])

            obs = next_obs

            # Periodic PER stats logging
            self._maybe_log_per_stats(total_train_steps)

        vec_env.close()
        return total_train_steps

    # ------------------------------------------------------------------

    def train(
        self,
        num_episodes: int,
        start_episode: int,
        start_total_steps: int,
        initial_best_score: float,
        initial_early_stopping_counter: int,
        specific_file: str | None = None,
    ):
        """Train the Rainbow DQN agent by orchestrating helper methods."""
        assert isinstance(num_episodes, int) and num_episodes > 0
        assert isinstance(start_episode, int) and start_episode >= 0
        assert isinstance(start_total_steps, int) and start_total_steps >= 0
        assert isinstance(specific_file, (str, type(None)))

        self.best_validation_metric = initial_best_score
        self.early_stopping_counter = initial_early_stopping_counter
        total_train_steps = start_total_steps
        self.total_train_steps = start_total_steps

        self.agent.set_training_mode(True)
        self._abort_training = False
        self._abort_reason = None
        self._abort_step = None

        logger.info("====== STARTING/RESUMING RAINBOW DQN TRAINING ======")
        logger.info(f"Starting from Episode: {start_episode + 1}/{num_episodes}")
        logger.info(f"Starting from Total Steps: {total_train_steps}")
        logger.info(f"Agent Config: {self.agent_config}")
        logger.info(f"Env Config: {self.env_config}")
        logger.info(f"Trainer Config: {self.trainer_config}")

        val_files = self.data_manager.get_validation_files()
        if not val_files:
            logger.warning("No validation files found. Training will proceed without validation.")

        self._validate_validation_cadence_config(num_episodes=num_episodes, has_val_files=bool(val_files))

        # Dispatch to vectorized loop when num_vector_envs > 1
        if self.num_vector_envs > 1 and specific_file is None:
            logger.info(f"Using vectorized training with {self.num_vector_envs} parallel envs.")
            try:
                total_train_steps = self._train_vectorized(
                    num_episodes,
                    start_episode,
                    total_train_steps,
                    val_files,
                )
            except Exception:
                logger.error("!!! UNEXPECTED EXCEPTION in vectorized training loop !!!", exc_info=True)
            finally:
                self._finalize_training(total_train_steps, num_episodes, val_files)
                self.agent.set_training_mode(False)
            return

        # Legacy single-env episodic loop
        total_rewards = []

        try:
            for episode in range(start_episode, num_episodes):
                if self._maybe_apply_final_phase_lr_decay(
                    current_episode=episode,
                    total_episodes=num_episodes,
                    total_train_steps=total_train_steps,
                ):
                    if self.writer:
                        current_lr = self.agent.optimizer.param_groups[0]["lr"]
                        self.writer.add_scalar("Train/Learning_Rate", current_lr, total_train_steps)

                # 1. Initialize episode environment and tracker
                episode_env, initial_obs, initial_info, tracker = self._initialize_episode(
                    specific_file, episode, num_episodes
                )
                if episode_env is None or tracker is None:
                    logger.error(f"Failed to initialize episode {episode + 1}. Skipping.")
                    continue

                # 2. Run steps within the episode
                (
                    episode_reward,
                    episode_loss,
                    steps_in_episode,
                    total_train_steps,
                    info,
                    invalid_action_count,
                ) = self._run_episode_steps(episode_env, initial_obs, tracker, episode, total_train_steps)
                self.total_train_steps = total_train_steps

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

                if self._abort_training:
                    abort_message = self._abort_reason or "Unrecoverable learning error encountered."
                    if self._abort_step is not None:
                        logger.error(
                            "Terminating training loop at step %s due to: %s",
                            self._abort_step,
                            abort_message,
                        )
                    else:
                        logger.error("Terminating training loop due to: %s", abort_message)
                    break

                # 4. Handle validation and checkpointing
                should_stop = self._handle_validation_and_checkpointing(episode, total_train_steps, val_files, tracker)

                if should_stop:
                    logger.info("Early stopping condition met. Exiting training loop.")
                    break

        except Exception:
            logger.error("!!! UNEXPECTED EXCEPTION in main training loop !!!", exc_info=True)

        finally:
            self._finalize_training(total_train_steps, num_episodes, val_files)
            self.agent.set_training_mode(False)

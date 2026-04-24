"""Auto-split from trainer.py (Tier 3.1).

Do not add new methods here directly; extend the mixin class in-place or
refactor through the facade. See .cursor/plans/prioritized-codebase-cleanup_*.plan.md.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from momentum_core.logging import get_logger

from .metrics import PerformanceTracker
from .trade_metrics import StepRecord, aggregate_trade_metrics, segment_trades

logger = get_logger(__name__)


class DiagnosticsMixin:
    """TB diagnostic / PER-audit / LR-decay methods split from the monolithic trainer."""

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
        except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - defensive logging
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

        # Tier 1d: mirror PER stats to TensorBoard so beta/alpha/fill/avg/max priority are
        # visible alongside Train/* curves. Stepped on env-side training step so the time
        # axis is consistent with reward/action-rate panels.
        if self.writer is not None:
            try:
                step = int(total_train_steps)
                self.writer.add_scalar("Train/PER/AvgPriority", float(stats.get("avg_priority", 0.0)), step)
                self.writer.add_scalar("Train/PER/MaxPriority", float(stats.get("max_priority", 0.0)), step)
                self.writer.add_scalar("Train/PER/TotalPriority", float(stats.get("total_priority", 0.0)), step)
                self.writer.add_scalar("Train/PER/Beta", float(stats.get("beta", 0.0)), step)
                self.writer.add_scalar("Train/PER/Alpha", float(stats.get("alpha", 0.0)), step)
                self.writer.add_scalar("Train/PER/Fill", float(stats.get("fill_ratio", 0.0)), step)
                self.writer.add_scalar("Train/PER/Size", float(stats.get("size", 0)), step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror PER stats to TensorBoard: %s", exc)

        # Tier 1d: light invariant audit on stored n-step rewards.
        # If clipping is enabled the stored n-step reward magnitude must satisfy
        #   |R_n| <= reward_clip * sum_{i=0}^{n-1} gamma^i
        # Any sample exceeding that bound indicates a code bug (clipping bypass) or
        # a stale checkpointed buffer carrying pre-clip outliers. Counting the events
        # is a cheap regression guard.
        self._maybe_audit_per_buffer_clip_bypass(total_train_steps)

        # Tier 4a: full reward + priority distribution audit, runs at a slower
        # cadence than the cheap PER scalars above.
        self._maybe_audit_per_buffer_distribution(total_train_steps)

    def _maybe_audit_per_buffer_clip_bypass(self, total_train_steps: int) -> None:
        """Sample up to 4096 stored transitions and count |reward| > clip bound.

        Invariant: rewards are clipped by ``self.reward_clip_value`` *before* being
        accumulated into n-step returns by the agent. The maximum legal magnitude
        of a stored n-step reward is therefore::

            bound = reward_clip * sum_{i=0}^{n-1} gamma^i

        Any sample exceeding ``bound`` after a small floating-point tolerance is a
        bug (or a stale buffer reload) — surface it as a TB scalar so it is caught
        immediately rather than silently re-injected into the learner.
        """
        if self.writer is None or self.reward_clip_value is None:
            return
        buffer = getattr(self.agent, "buffer", None)
        if buffer is None:
            return
        stored_buffer = getattr(buffer, "buffer", None)
        if stored_buffer is None or len(stored_buffer) == 0:
            return

        try:
            gamma = float(self.agent_config.get("gamma", 0.99))
            n_steps = int(self.agent_config.get("n_steps", 1))
        except (TypeError, ValueError):
            gamma, n_steps = 0.99, 1
        if n_steps <= 0:
            return

        # Geometric series sum_{i=0}^{n-1} gamma^i
        if abs(gamma - 1.0) < 1e-9:
            discount_sum = float(n_steps)
        else:
            discount_sum = (1.0 - gamma**n_steps) / (1.0 - gamma)
        bound = float(self.reward_clip_value) * discount_sum * (1.0 + 1e-6)

        size = len(stored_buffer)
        sample_size = min(4096, size)
        try:
            indices = np.random.randint(0, size, size=sample_size) if sample_size < size else np.arange(size)
            rewards = np.empty(sample_size, dtype=np.float64)
            for i, idx in enumerate(indices):
                experience = stored_buffer[int(idx)]
                rewards[i] = float(getattr(experience, "reward", 0.0))
        except (RuntimeError, ValueError, AttributeError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            logger.debug("PER buffer audit failed during sampling: %s", exc)
            return

        bypass_mask = np.abs(rewards) > bound
        bypass_count = int(bypass_mask.sum())
        bypass_fraction = float(bypass_mask.mean()) if sample_size > 0 else 0.0

        try:
            step = int(total_train_steps)
            self.writer.add_scalar("Train/PER/ClipBypassEventCount", float(bypass_count), step)
            self.writer.add_scalar("Train/PER/ClipBypassFraction", bypass_fraction, step)
            self.writer.add_scalar("Train/PER/AuditSampleSize", float(sample_size), step)
            self.writer.add_scalar("Train/PER/StoredRewardClipBound", bound, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror PER buffer audit to TensorBoard: %s", exc)

        if bypass_count > 0:
            top = float(np.max(np.abs(rewards)))
            logger.warning(
                "PER buffer clip-bypass audit: %d/%d sampled transitions exceed bound=%.6f "
                "(max |reward|=%.6f). Possible code bug or stale checkpointed buffer.",
                bypass_count,
                sample_size,
                bound,
                top,
            )

    def _maybe_audit_per_buffer_distribution(self, total_train_steps: int) -> None:
        """Sample up to 4096 stored transitions and emit reward + priority distribution audit.

        Tier 4a. Emits:

        * ``Train/PER/Reward/Histogram`` — full reward distribution shape;
          catches multi-modal collapse the scalar avg/max can't see.
        * ``Train/PER/Reward/OutlierFrac`` — fraction of |reward| > 5*reward_clip
          (relative to the stored n-step bound). Looser than the clip-bypass
          audit on purpose; flags "concerning tail" rather than "is a bug".
        * ``Train/PER/PriorityByAction/{k}`` — mean SumTree priority per action
          k. The "always action 5" failure mode shows up here as a single
          action dominating priority well before it dominates Action Rate.
        * ``Train/PER/Top1PctActionShare/{k}`` — share of the top-1% highest
          priority transitions belonging to action k. Even sharper signal than
          PriorityByAction since it isolates *which* action the learner is
          chasing the largest TD errors on.
        """
        if self.writer is None:
            return
        if self.per_buffer_audit_interval <= 0:
            return
        if total_train_steps <= 0 or total_train_steps % self.per_buffer_audit_interval != 0:
            return
        buffer = getattr(self.agent, "buffer", None)
        if buffer is None:
            return
        stored_buffer = getattr(buffer, "buffer", None)
        sum_tree = getattr(buffer, "tree", None)
        if stored_buffer is None or len(stored_buffer) == 0 or sum_tree is None:
            return
        tree_arr = getattr(sum_tree, "tree", None)
        capacity = int(getattr(sum_tree, "capacity", 0))
        if tree_arr is None or capacity <= 0:
            return

        size = len(stored_buffer)
        sample_size = min(4096, size)
        try:
            indices = np.random.randint(0, size, size=sample_size) if sample_size < size else np.arange(size)
            rewards = np.empty(sample_size, dtype=np.float64)
            actions = np.empty(sample_size, dtype=np.int64)
            priorities = np.empty(sample_size, dtype=np.float64)
            for i, idx in enumerate(indices):
                experience = stored_buffer[int(idx)]
                rewards[i] = float(getattr(experience, "reward", 0.0))
                actions[i] = int(getattr(experience, "action", 0))
                # SumTree leaves live at indices [capacity-1, 2*capacity-2]; the
                # buffer index equals the SumTree's data pointer (lockstep
                # writes — see PrioritizedReplayBuffer.store), so this lookup
                # is direct and identity-safe.
                priorities[i] = float(tree_arr[int(idx) + capacity - 1])
        except (RuntimeError, ValueError, AttributeError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            logger.debug("PER distribution audit failed during sampling: %s", exc)
            return

        step = int(total_train_steps)
        try:
            self.writer.add_histogram("Train/PER/Reward/Histogram", rewards, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror PER reward histogram: %s", exc)

        # Outlier fraction is *informational* (not a bug guard like
        # ClipBypass); use a generous 5x reward_clip threshold when clipping
        # is enabled, otherwise fall back to 5x the empirical std.
        try:
            if self.reward_clip_value is not None:
                threshold = 5.0 * float(self.reward_clip_value)
            else:
                std = float(np.std(rewards)) if rewards.size > 1 else 0.0
                threshold = 5.0 * std if std > 0 else float("inf")
            outlier_frac = float(np.mean(np.abs(rewards) > threshold)) if math.isfinite(threshold) else 0.0
            self.writer.add_scalar("Train/PER/Reward/OutlierFrac", outlier_frac, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror PER reward outlier fraction: %s", exc)

        num_actions = int(getattr(self.agent, "num_actions", int(actions.max() + 1) if actions.size else 1))
        try:
            for k in range(num_actions):
                mask = actions == k
                if mask.any():
                    mean_p = float(priorities[mask].mean())
                else:
                    mean_p = 0.0
                self.writer.add_scalar(f"Train/PER/PriorityByAction/{k}", mean_p, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror PER per-action priorities: %s", exc)

        # Top 1% by priority — the slice of the buffer the learner is
        # actually being shaped by right now.
        try:
            top_n = max(1, sample_size // 100)
            top_idx = np.argpartition(-priorities, kth=min(top_n - 1, sample_size - 1))[:top_n]
            top_actions = actions[top_idx]
            for k in range(num_actions):
                share = float(np.mean(top_actions == k)) if top_actions.size else 0.0
                self.writer.add_scalar(f"Train/PER/Top1PctActionShare/{k}", share, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror PER top-1pct action shares: %s", exc)

    def _maybe_apply_final_phase_lr_decay(
        self,
        *,
        current_episode: int,
        total_episodes: int,
        total_train_steps: int,
    ) -> bool:
        """Apply a one-time LR decay near the end of training if configured."""
        if self.final_phase_lr_start_frac is None or self.final_phase_lr_multiplier is None or self._final_phase_lr_applied:
            return False

        try:
            threshold_episode = int(total_episodes * self.final_phase_lr_start_frac)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid final-phase LR start fraction %s; skipping final-phase decay.",
                self.final_phase_lr_start_frac,
            )
            self._final_phase_lr_applied = True
            return False

        if current_episode < threshold_episode:
            return False

        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None:
            logger.warning("Optimizer unavailable when attempting final-phase LR decay; skipping.")
            self._final_phase_lr_applied = True
            return False

        scheduler_params = {}
        if isinstance(self.agent_config, dict):
            scheduler_params = self.agent_config.get("lr_scheduler_params", {}) or {}
        min_lr = None
        if isinstance(scheduler_params, dict) and "min_lr" in scheduler_params:
            try:
                min_lr = float(scheduler_params["min_lr"])
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid min_lr value %s; ignoring min_lr constraint for final-phase decay.",
                    scheduler_params["min_lr"],
                )
                min_lr = None

        changed = False
        for param_group in optimizer.param_groups:
            old_lr = param_group.get("lr")
            if old_lr is None:
                continue
            old_lr_float = float(old_lr)
            new_lr = old_lr_float * self.final_phase_lr_multiplier
            if min_lr is not None:
                new_lr = max(new_lr, min_lr)
            if abs(new_lr - old_lr_float) > 1e-12:
                param_group["lr"] = new_lr
                changed = True

        if changed:
            self._final_phase_lr_applied = True
            logger.info(
                "[FinalPhaseLR] Applied LR multiplier %.3f at episode %s/%s (total steps %s).",
                self.final_phase_lr_multiplier,
                current_episode + 1,
                total_episodes,
                total_train_steps,
            )
        return changed

    # Eval-gap scalar set: keys must exist in *both* Validation and Train to
    # produce a meaningful diff. Tags are deliberately short so they group
    # nicely in the TensorBoard left-rail tree under ``Train/EvalGap/``.
    _EVAL_GAP_METRIC_TAGS: dict[str, str] = {
        "total_return": "TotalReturnPct",
        "sharpe_ratio": "SharpeRatio",
        "max_drawdown": "MaxDrawdown",
        "transaction_costs": "TransactionCosts",
        "avg_reward": "AvgReward",
        "portfolio_value": "PortfolioValue",
    }

    def _emit_eval_gap_scalars(self, avg_val_metrics: dict, train_recent: dict, episode: int) -> None:
        """Emit ``Train/EvalGap/*`` = greedy_validation - stochastic_training.

        Tier 2d. Each tag is the *signed* difference; positive values mean the
        greedy projection outperformed the stochastic training window on that
        metric. Skips a tag if either side is missing or non-finite, so eval
        gaps remain readable even on partial validation runs.
        """
        if not self.writer:
            return
        for key, tag in self._EVAL_GAP_METRIC_TAGS.items():
            val = avg_val_metrics.get(key)
            ref = train_recent.get(key)
            if val is None or ref is None:
                continue
            try:
                fv = float(val)
                fr = float(ref)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv) or math.isnan(fr) or math.isinf(fr):
                continue
            self.writer.add_scalar(f"Train/EvalGap/{tag}", fv - fr, episode)

    def _trade_metrics_from_tracker(self, tracker: PerformanceTracker) -> dict[str, float]:
        """Build per-trade aggregate metrics from a training :class:`PerformanceTracker`.

        Tier 2c: reuses Tier 2a's :func:`segment_trades` so train-time and
        eval-time per-trade KPIs share one definition. Returns an empty dict if
        the tracker doesn't have enough state (positions/prices) to segment.
        """
        positions = list(getattr(tracker, "positions", []) or [])
        actions = list(getattr(tracker, "actions", []) or [])
        prices_full = list(getattr(tracker, "position_values", []) or [])
        portfolio_values = list(getattr(tracker, "portfolio_values", []) or [])
        transaction_costs = list(getattr(tracker, "transaction_costs", []) or [])
        was_greedy = list(getattr(tracker, "was_greedy", []) or [])
        if not positions or not actions or not portfolio_values:
            return {}
        # PerformanceTracker stores price implicitly via position_values/positions; we
        # require an explicit price stream to segment trades. Recover price as
        # ``position_value/position`` where possible; fall back to a flat 1.0 series
        # otherwise. The resulting MAE/MFE will then be measured in PnL units, which
        # is still useful for the headline KPIs (HitRate/Expectancy/PctGreedy).
        prices: list[float] = []
        for pos, pv in zip(positions, prices_full, strict=False):
            if abs(pos) > 1e-12 and pv is not None:
                prices.append(float(pv) / float(pos))
            else:
                prices.append(prices[-1] if prices else 1.0)
        if not prices:
            prices = [1.0] * len(positions)
        n = min(
            len(positions),
            len(actions),
            len(prices),
            len(portfolio_values) - 1 if len(portfolio_values) > 1 else 0,
        )
        if n <= 0:
            return {}
        steps: list[StepRecord] = []
        for i in range(n):
            wg = was_greedy[i] if i < len(was_greedy) else None
            tc = transaction_costs[i] if i < len(transaction_costs) else 0.0
            steps.append(
                StepRecord(
                    step_index=i,
                    portfolio_value=float(portfolio_values[i + 1]),
                    position=float(positions[i]),
                    price=float(prices[i]),
                    action=int(actions[i]),
                    transaction_cost=float(tc or 0.0),
                    was_greedy=None if wg is None else bool(wg),
                )
            )
        trades = segment_trades(steps)
        return aggregate_trade_metrics(trades)

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
        logger.info(f"  Metrics - Max Drawdown: {metrics.get('max_drawdown', np.nan) * 100:.2f}%" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Action Counts: {metrics.get('action_counts', {})}" if metrics else "Metrics: N/A")
        logger.info(f"  Metrics - Transaction Costs: ${metrics.get('transaction_costs', np.nan):.2f}" if metrics else "Metrics: N/A")
        if metrics:
            logger.info(f"  Metrics - Avg Exposure: {metrics.get('avg_exposure_pct', np.nan):.2f}%")
            logger.info(f"  Metrics - Max Exposure: {metrics.get('max_exposure_pct', np.nan):.2f}%")
            logger.info(f"  Metrics - Avg Position: {metrics.get('avg_position', np.nan):.4f}")
            logger.info(f"  Metrics - Avg Balance: ${metrics.get('avg_balance', np.nan):.2f}")

        self._log_progress(
            "episode",
            episode=episode + 1,
            steps=steps_in_episode,
            total_steps=total_train_steps,
            reward=round(episode_reward, 4),
            avg_reward=round(avg_reward_window, 4),
            total_return=round(metrics.get("total_return", 0.0), 2) if metrics else 0.0,
            sharpe=round(metrics.get("sharpe_ratio", 0.0), 4) if metrics else 0.0,
            max_dd=round(metrics.get("max_drawdown", 0.0) * 100, 2) if metrics else 0.0,
            invalid_pct=round(invalid_action_rate * 100, 1),
            lr=self.agent.optimizer.param_groups[0]["lr"],
            pv=round(final_info.get("portfolio_value", 0.0), 2),
        )

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
                if "avg_exposure_pct" in metrics:
                    self.writer.add_scalar("Train/Avg Exposure Pct", metrics.get("avg_exposure_pct", np.nan), episode)
                    self.writer.add_scalar("Train/Max Exposure Pct", metrics.get("max_exposure_pct", np.nan), episode)
                if "avg_position" in metrics:
                    self.writer.add_scalar("Train/Avg Position", metrics.get("avg_position", np.nan), episode)
                if "avg_abs_position" in metrics:
                    self.writer.add_scalar("Train/Avg Abs Position", metrics.get("avg_abs_position", np.nan), episode)
                if "avg_balance" in metrics:
                    self.writer.add_scalar("Train/Avg Balance", metrics.get("avg_balance", np.nan), episode)
                if "avg_position_value" in metrics:
                    self.writer.add_scalar("Train/Avg Position Value", metrics.get("avg_position_value", np.nan), episode)
                action_counts = metrics.get("action_counts", {})
                if action_counts:
                    for action_idx, count in action_counts.items():
                        action_rate = count / steps_safe if steps_safe > 0 else 0.0
                        self.writer.add_scalar(f"Train/Action Count/{action_idx}", count, episode)
                        self.writer.add_scalar(f"Train/Action Rate/{action_idx}", action_rate, episode)
                # Tier 2c: per-action greedy/eps split + epsilon-forced trade fraction.
                provenance = metrics.get("action_provenance_counts", {}) or {}
                greedy_counts = provenance.get("greedy", {}) or {}
                eps_counts = provenance.get("eps", {}) or {}
                if greedy_counts or eps_counts:
                    for action_idx in range(int(getattr(self.agent, "num_actions", 6))):
                        gc = float(greedy_counts.get(action_idx, 0) or 0)
                        ec = float(eps_counts.get(action_idx, 0) or 0)
                        gr = gc / steps_safe if steps_safe > 0 else 0.0
                        er = ec / steps_safe if steps_safe > 0 else 0.0
                        self.writer.add_scalar(f"Train/Action Rate/Greedy/{action_idx}", gr, episode)
                        self.writer.add_scalar(f"Train/Action Rate/Eps/{action_idx}", er, episode)
                self.writer.add_scalar(
                    "Train/EpsilonForcedTradeFraction",
                    float(metrics.get("epsilon_forced_trade_fraction", 0.0) or 0.0),
                    episode,
                )
                # Tier 2c: per-episode Train/Trade/* (HitRate/Expectancy/PctGreedy/...)
                try:
                    train_trade_metrics = self._trade_metrics_from_tracker(tracker)
                except (ValueError, KeyError, AttributeError, TypeError, IndexError):  # pragma: no cover - defensive, never block training
                    logger.debug("Failed to compute Train/Trade metrics from tracker", exc_info=True)
                    train_trade_metrics = {}
                for key, value in train_trade_metrics.items():
                    if not isinstance(value, (int, float, np.floating, np.integer)):
                        continue
                    fv = float(value)
                    if math.isnan(fv) or math.isinf(fv):
                        continue
                    tag = "".join(part.capitalize() for part in str(key).split("_"))
                    self.writer.add_scalar(f"Train/Trade/{tag}", fv, episode)
                # Tier 4b: per-episode reward outlier guard. Cheap (one pass over
                # the tracker's rewards) and gives an immediate "did we just blow
                # up the loss?" signal independent of PER buffer audits, which
                # are throttled to ~5x slower.
                try:
                    outlier_stats = tracker.get_reward_outlier_stats(self.reward_clip_value)
                except (ValueError, KeyError, AttributeError, TypeError):  # pragma: no cover - defensive
                    logger.debug("Failed to compute Tier 4b reward outlier stats", exc_info=True)
                    outlier_stats = {}
                if outlier_stats:
                    self.writer.add_scalar("Train/Episode/RewardMin", outlier_stats["reward_min"], episode)
                    self.writer.add_scalar("Train/Episode/RewardMax", outlier_stats["reward_max"], episode)
                    self.writer.add_scalar("Train/Episode/RewardP99Abs", outlier_stats["reward_p99_abs"], episode)
                    self.writer.add_scalar("Train/Episode/RewardOutlierFlag", outlier_stats["reward_outlier_flag"], episode)
                # Tier 4c: per-action reward mean/std. Action 0 (hold)
                # dominates avg_reward by count, so this is the only place we
                # can read off "is action 5 actually pulling its weight?".
                try:
                    by_action = tracker.get_reward_by_action_stats()
                except (ValueError, KeyError, AttributeError, TypeError):  # pragma: no cover - defensive
                    logger.debug("Failed to compute Tier 4c reward-by-action stats", exc_info=True)
                    by_action = {}
                for k in range(int(getattr(self.agent, "num_actions", 6))):
                    bucket = by_action.get(k, {"mean": 0.0, "std": 0.0})
                    self.writer.add_scalar(f"Train/Reward/MeanByAction/{k}", float(bucket["mean"]), episode)
                    self.writer.add_scalar(f"Train/Reward/StdByAction/{k}", float(bucket["std"]), episode)

            self.writer.add_scalar("Train/Final Portfolio Value", final_info.get("portfolio_value", np.nan), episode)
            self.writer.add_scalar("Train/Final Position", final_info.get("position", np.nan), episode)
            self.writer.add_scalar("Train/Invalid Action Rate", invalid_action_rate, episode)
            self.writer.add_scalar("Train/Rolling Invalid Action Rate", rolling_invalid_rate, episode)
            self.writer.add_scalar("Train/Epsilon", self.agent.current_epsilon, episode)
        # ------------------------------------------ #

        # Always log PER stats at episode boundaries (unless explicitly disabled)
        self._maybe_log_per_stats(total_train_steps, force=True)

        # End-of-episode flush: ensure every episode's scalars are on disk
        # before we start the next one. Combined with the per-checkpoint flush
        # in ``_save_checkpoint``, this bounds the "lost on freeze" window to a
        # single episode's worth of step-axis scalars.
        self._flush_writer()

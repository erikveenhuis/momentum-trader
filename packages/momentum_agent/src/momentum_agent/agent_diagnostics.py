"""Diagnostic TensorBoard emitters for :class:`RainbowDQNAgent`.

Extracted from ``agent.py`` as a mixin to keep the hot-path training code and
the (verbose) diagnostic code visually separated. All methods here are pure
reads of agent state plus optional ``self.tb_writer`` emission; none of them
mutate learning state.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from momentum_core.logging import get_logger

logger = get_logger(__name__)


class AgentDiagnosticsMixin:
    """TB-emitter methods for categorical targets, grad norms, Q-stats, noisy sigma."""

    def _accumulate_categorical_target_stats(self, target_distribution: torch.Tensor) -> None:
        """Accumulates categorical target distributions for periodic logging."""
        if self.categorical_logging_interval <= 0 or target_distribution is None:
            return

        if target_distribution.numel() == 0:
            return

        try:
            batch_mass = target_distribution.detach().sum(dim=0).to(device="cpu", dtype=torch.float64).numpy()
        except (RuntimeError, ValueError) as error:
            logger.warning(f"Failed to accumulate categorical target stats: {error}")
            return

        if not np.isfinite(batch_mass).all():
            logger.warning(
                "Non-finite values encountered while accumulating categorical target stats; skipping update."
            )
            return

        self._categorical_target_accumulator["mass"] += batch_mass
        self._categorical_target_accumulator["samples"] += target_distribution.shape[0]

    def _log_categorical_target_stats(self) -> None:
        """Logs histogram and percentiles for accumulated categorical target distributions."""
        accumulator = self._categorical_target_accumulator

        total_samples = accumulator["samples"]
        if total_samples == 0:
            return

        mass = accumulator["mass"]
        total_mass = mass.sum()
        if not np.isfinite(total_mass) or total_mass <= 0:
            logger.warning(
                "Invalid total mass encountered while logging categorical target stats; resetting accumulator."
            )
            accumulator["mass"].fill(0.0)
            accumulator["samples"] = 0
            return

        probs = mass / total_mass
        if not np.isfinite(probs).all():
            logger.warning("Non-finite probabilities encountered in categorical target stats; resetting accumulator.")
            accumulator["mass"].fill(0.0)
            accumulator["samples"] = 0
            return

        cdf = np.cumsum(probs)
        percentile_strings = []
        for percentile in self.categorical_logging_percentiles:
            target = percentile / 100.0
            idx = int(np.searchsorted(cdf, target, side="left"))
            idx = min(max(idx, 0), self.num_atoms - 1)
            percentile_strings.append(f"{percentile:.1f}%={self.support_cpu[idx]:.4f}")

        mean_value = float(np.dot(probs, self.support_cpu))
        edge_min = float(probs[0])
        edge_max = float(probs[-1])

        percentile_values: dict[float, float] = {}
        for percentile in self.categorical_logging_percentiles:
            target = percentile / 100.0
            idx = int(np.searchsorted(cdf, target, side="left"))
            idx = min(max(idx, 0), self.num_atoms - 1)
            percentile_values[float(percentile)] = float(self.support_cpu[idx])

        logger.info(
            "Categorical target stats at learn step %s (accumulated over %s samples): mean=%.4f, edge_mass=(min=%.4f, max=%.4f), percentiles=[%s]",
            self.total_steps,
            total_samples,
            mean_value,
            edge_min,
            edge_max,
            ", ".join(percentile_strings),
        )
        logger.info(
            "Categorical target histogram (avg prob per atom, sum=1.0): %s",
            np.array2string(probs, precision=4, suppress_small=True),
        )

        if self.tb_writer is not None:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Mean", mean_value, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Edge_Mass_Min", edge_min, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Edge_Mass_Max", edge_max, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Samples", float(total_samples), step)
                for percentile, support_value in percentile_values.items():
                    tag = f"Train/CategoricalTarget/P{percentile:g}"
                    self.tb_writer.add_scalar(tag, support_value, step)
                self.tb_writer.add_histogram(
                    "Train/CategoricalTarget/Distribution",
                    probs.astype(np.float32),
                    step,
                )
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror categorical target stats to TensorBoard: %s", exc)

        accumulator["mass"].fill(0.0)
        accumulator["samples"] = 0

    def _log_n_step_reward_window_stats(self) -> None:
        """Logs statistics of the rolling n-step reward window to logger and TensorBoard."""
        if len(self.n_step_reward_window) == 0:
            return
        try:
            rewards_array = np.fromiter(self.n_step_reward_window, dtype=float)
        except ValueError:  # pragma: no cover - safeguard
            logger.warning("Could not materialize n-step reward window for stats.")
            return

        min_r = float(rewards_array.min())
        max_r = float(rewards_array.max())
        mean_r = float(rewards_array.mean())
        std_r = float(rewards_array.std(ddof=0))
        window_len = len(self.n_step_reward_window)

        logger.info(
            "N-Step Reward Window (last %s learns): Min=%.4f, Max=%.4f, Mean=%.4f, Std=%.4f",
            window_len,
            min_r,
            max_r,
            mean_r,
            std_r,
        )

        if self.tb_writer is not None:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Train/NStepReward/Mean", mean_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Std", std_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Min", min_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Max", max_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/WindowSize", float(window_len), step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to mirror n-step reward window stats to TensorBoard: %s",
                    exc,
                )

    def _log_grad_stats(self, pre_clip_norm: torch.Tensor | float | None) -> None:
        """Mirror gradient norms + param-update-ratio per module group to TB."""
        if self.tb_writer is None or self.network is None:
            return
        step = int(self.total_steps + 1)
        net = getattr(self.network, "_orig_mod", self.network)
        try:
            if pre_clip_norm is not None:
                norm_val = float(pre_clip_norm) if not torch.is_tensor(pre_clip_norm) else float(pre_clip_norm.item())
                if math.isfinite(norm_val):
                    self.tb_writer.add_scalar("Train/Grad/Norm", norm_val, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Train/Grad/Norm: %s", exc)

        group_grad_sq: dict[str, float] = {}
        group_param_sq: dict[str, float] = {}
        try:
            for name, param in net.named_parameters():
                if param.grad is None:
                    continue
                group = name.split(".", 1)[0] or "root"
                g_sq = float(param.grad.detach().pow(2).sum().item())
                p_sq = float(param.detach().pow(2).sum().item())
                if not (math.isfinite(g_sq) and math.isfinite(p_sq)):
                    continue
                group_grad_sq[group] = group_grad_sq.get(group, 0.0) + g_sq
                group_param_sq[group] = group_param_sq.get(group, 0.0) + p_sq
        except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to compute per-group grad stats: %s", exc)
            return

        try:
            for group in sorted(group_grad_sq):
                self.tb_writer.add_scalar(
                    f"Train/Grad/PerGroup/{group}/Norm",
                    math.sqrt(group_grad_sq[group]),
                    step,
                )
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror per-group grad norms: %s", exc)

        try:
            total_grad = math.sqrt(sum(group_grad_sq.values()))
            total_param = math.sqrt(sum(group_param_sq.values()))
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0)) if self.optimizer is not None else 0.0
            if total_param > 0 and math.isfinite(total_grad) and math.isfinite(lr):
                ratio = (lr * total_grad) / total_param
                if math.isfinite(ratio):
                    self.tb_writer.add_scalar("Train/ParamUpdateRatio", ratio, step)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Train/ParamUpdateRatio: %s", exc)

    def _log_q_value_stats(self, *, emit_histogram: bool = False) -> None:
        """Mirror Q-value distribution stats from the most recent training batch."""
        if self.tb_writer is None:
            return
        q = getattr(self, "_last_batch_q", None)
        if q is None:
            return
        try:
            q_np = q.detach().to(torch.float32).cpu().numpy()
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to materialize cached Q-values: %s", exc)
            return
        if q_np.ndim != 2 or q_np.size == 0:
            return
        step = int(self.total_steps)
        try:
            self.tb_writer.add_scalar("Train/Q/Mean", float(q_np.mean()), step)
            self.tb_writer.add_scalar("Train/Q/Std", float(q_np.std(ddof=0)), step)
            self.tb_writer.add_scalar("Train/Q/MaxAcrossActions", float(q_np.max()), step)
            self.tb_writer.add_scalar("Train/Q/MinAcrossActions", float(q_np.min()), step)
            sorted_q = np.sort(q_np, axis=1)
            if sorted_q.shape[1] >= 2:
                margin = float((sorted_q[:, -1] - sorted_q[:, -2]).mean())
                self.tb_writer.add_scalar("Train/Q/ActionMargin", margin, step)
            for action_idx in range(q_np.shape[1]):
                self.tb_writer.add_scalar(
                    f"Train/Q/PerAction/Mean/{action_idx}",
                    float(q_np[:, action_idx].mean()),
                    step,
                )
            if emit_histogram:
                try:
                    self.tb_writer.add_histogram("Train/Q/Distribution", q_np.astype(np.float32), step)
                except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to emit Train/Q/Distribution histogram: %s", exc)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Q-value stats to TB: %s", exc)

    def _log_noisy_sigma_stats(self) -> None:
        """Mirror NoisyLinear sigma stats per module to TensorBoard."""
        if self.tb_writer is None or self.network is None:
            return
        net = getattr(self.network, "_orig_mod", self.network)
        from .model import NoisyLinear

        step = int(self.total_steps)
        all_means: list[float] = []
        for name, module in net.named_modules():
            if not isinstance(module, NoisyLinear):
                continue
            tag = name.replace(".", "/") if name else "root"
            try:
                w_sigma = module.weight_sigma.detach().abs()
                b_sigma = module.bias_sigma.detach().abs()
                sigma_mean = float((w_sigma.sum() + b_sigma.sum()) / (w_sigma.numel() + b_sigma.numel()))
                sigma_max = float(max(w_sigma.max().item(), b_sigma.max().item()))
                sigma_min = float(min(w_sigma.min().item(), b_sigma.min().item()))
            except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read NoisyLinear sigma for %s: %s", name, exc)
                continue
            try:
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMean", sigma_mean, step)
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMax", sigma_max, step)
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMin", sigma_min, step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror NoisyLinear sigma to TB: %s", exc)
                continue
            all_means.append(sigma_mean)
        if all_means:
            try:
                self.tb_writer.add_scalar("Train/Noisy/AggregateSigmaMean", float(np.mean(all_means)), step)
                self.tb_writer.add_scalar("Train/Noisy/ModuleCount", float(len(all_means)), step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror aggregate NoisyLinear sigma to TB: %s", exc)

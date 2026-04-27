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
    """TB-emitter methods for quantile targets, grad norms, Q-stats, noisy sigma."""

    def _accumulate_quantile_target_stats(self, target_q_z: torch.Tensor) -> None:
        """Stash a batch of IQN target quantile values for periodic logging.

        ``target_q_z`` has shape ``[B, K']`` where ``K'`` = ``n_quantiles_target``.
        We move each batch to CPU/float32 so the GPU buffer can be released, then
        materialise the full distribution at log time. This is bounded by the
        product of ``batch_size * n_quantiles_target * quantile_logging_interval``,
        which sits well within available host RAM at default settings.
        """
        if self.quantile_logging_interval <= 0 or target_q_z is None:
            return

        if target_q_z.numel() == 0:
            return

        try:
            batch_np = target_q_z.detach().to(device="cpu", dtype=torch.float32).numpy()
        except (RuntimeError, ValueError) as error:
            logger.warning(f"Failed to accumulate IQN target quantile stats: {error}")
            return

        if not np.isfinite(batch_np).all():
            logger.warning(
                "Non-finite values encountered while accumulating IQN target quantile stats; skipping batch."
            )
            return

        accumulator = self._quantile_target_accumulator
        accumulator["values"].append(batch_np)
        accumulator["samples"] += int(batch_np.shape[0])

    def _log_quantile_target_stats(self) -> None:
        """Emit ``Train/Quantile/*`` summaries from accumulated IQN target batches."""
        accumulator = self._quantile_target_accumulator

        total_samples = accumulator["samples"]
        batches = accumulator["values"]
        if total_samples == 0 or not batches:
            return

        try:
            stacked = np.concatenate(batches, axis=0)  # [N, K']
        except ValueError as exc:  # pragma: no cover - defensive
            logger.warning("Failed to stack IQN target quantile batches: %s", exc)
            accumulator["values"] = []
            accumulator["samples"] = 0
            return

        flat = stacked.reshape(-1)
        if not np.isfinite(flat).all() or flat.size == 0:
            logger.warning("Non-finite or empty IQN target quantile distribution; resetting accumulator.")
            accumulator["values"] = []
            accumulator["samples"] = 0
            return

        mean_value = float(flat.mean())
        std_value = float(flat.std(ddof=0))
        min_value = float(flat.min())
        max_value = float(flat.max())

        percentile_values: dict[float, float] = {}
        percentile_strings: list[str] = []
        for percentile in self.quantile_logging_percentiles:
            value = float(np.percentile(flat, percentile))
            percentile_values[float(percentile)] = value
            percentile_strings.append(f"{percentile:.1f}%={value:.4f}")

        logger.info(
            "IQN target quantile stats at learn step %s (accumulated over %s samples, %s atoms): "
            "mean=%.4f, std=%.4f, range=[%.4f, %.4f], percentiles=[%s]",
            self.total_steps,
            total_samples,
            stacked.shape[1],
            mean_value,
            std_value,
            min_value,
            max_value,
            ", ".join(percentile_strings),
        )

        if self.tb_writer is not None:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Train/Quantile/Mean", mean_value, step)
                self.tb_writer.add_scalar("Train/Quantile/Std", std_value, step)
                self.tb_writer.add_scalar("Train/Quantile/Min", min_value, step)
                self.tb_writer.add_scalar("Train/Quantile/Max", max_value, step)
                self.tb_writer.add_scalar("Train/Quantile/Samples", float(total_samples), step)
                for percentile, value in percentile_values.items():
                    tag = f"Train/Quantile/P{percentile:g}"
                    self.tb_writer.add_scalar(tag, value, step)
                self.tb_writer.add_histogram(
                    "Train/Quantile/Distribution",
                    flat.astype(np.float32),
                    step,
                )
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror IQN target quantile stats to TensorBoard: %s", exc)

        accumulator["values"] = []
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

    def _log_spectral_norm_stats(self) -> None:
        """Mirror per-layer spectral-norm operator-norm estimates to TensorBoard.

        ``torch.nn.utils.parametrizations.spectral_norm`` keeps the running
        right-singular-vector estimate ``_v`` on the parametrization module;
        the operator norm of the un-normalized weight is then ``u^T W v``
        (or, simply, the largest singular value of ``weight_mu`` evaluated
        with the latest power-iteration estimate). We surface the *bound*
        being enforced — ``sigma_max(weight_mu_normalized)`` should hover
        near ``1`` when the constraint is binding, so a single per-layer
        ``Train/SpectralNorm/<layer>/SigmaMax`` tag is enough to notice
        the rare "constraint never binds" pathology that signals a bug
        in the wrapping itself.
        """
        if self.tb_writer is None or self.network is None:
            return
        net = getattr(self.network, "_orig_mod", self.network)
        if not getattr(net, "spectral_norm_enabled", False):
            return

        step = int(self.total_steps)
        any_logged = False
        for name, module in net.named_modules():
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is None:
                continue
            if "weight_mu" not in parametrizations:
                continue
            try:
                weight_mu = module.weight_mu  # parametrized -> normalized weight
                if weight_mu.ndim < 2:
                    continue
                w2d = weight_mu.detach().reshape(weight_mu.shape[0], -1).float()
                sigma_max = float(torch.linalg.matrix_norm(w2d, ord=2).item())
            except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read spectral-norm sigma for %s: %s", name, exc)
                continue
            tag = name.replace(".", "/") if name else "root"
            try:
                self.tb_writer.add_scalar(f"Train/SpectralNorm/{tag}/SigmaMax", sigma_max, step)
                any_logged = True
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror spectral-norm sigma for %s: %s", name, exc)
        if any_logged:
            try:
                self.tb_writer.add_scalar("Train/SpectralNorm/Enabled", 1.0, step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to emit Train/SpectralNorm/Enabled: %s", exc)

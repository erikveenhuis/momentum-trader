import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from momentum_agent import RainbowDQNAgent  # Updated import path
from momentum_core.logging import get_logger
from momentum_env import TradingEnv  # Use installed package
from torch.utils.tensorboard import SummaryWriter

from .config_schema import RunConfig, TrainerConfig
from .data import DataManager  # Use relative import
from .metrics import (  # Use relative import
    PerformanceTracker,
)
from .schedules import compute_benchmark_frac
from .trainer_checkpoint import CheckpointMixin
from .trainer_diagnostics import DiagnosticsMixin
from .trainer_loop import LoopMixin
from .trainer_validation import ValidationMixin

logger = get_logger(__name__)


class RainbowTrainerModule(CheckpointMixin, ValidationMixin, DiagnosticsMixin, LoopMixin):
    """Orchestrates Rainbow DQN training: episode management, validation, checkpointing, and early stopping.

    Tier 3.1 of the cleanup plan split the ~3145-line implementation into four
    sibling mixin modules:
      * :mod:`trainer_checkpoint` — save / rotate / finalize.
      * :mod:`trainer_validation` — validate / evaluate.
      * :mod:`trainer_diagnostics` — TB + PER audits + per-episode summary.
      * :mod:`trainer_loop` — single-env + vectorized training loop.
    This class stays as the facade so ``run_training.py`` and every test
    that constructs ``RainbowTrainerModule`` directly keep working unchanged.
    """

    def __init__(
        self,
        agent: RainbowDQNAgent,
        device: torch.device,
        data_manager: DataManager,
        config: dict,
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
        # Tier 1.1: strict validation of every tunable in ``trainer`` / ``run``
        # up front. Missing keys raise ``KeyError`` here rather than silently
        # falling back to a stale default two hours into training.
        self._cfg = TrainerConfig.from_dict(self.trainer_config)
        self._run_cfg = RunConfig.from_dict(config["run"]) if "run" in config else None
        self.run_config = config.get("run", {})
        self.best_validation_metric = -np.inf
        self.writer = writer
        # Inject the trainer's writer into the agent so internal diagnostics
        # (categorical target stats, NoisyNet sigma, Q-stats, grad norms, target-net
        # deviation, etc.) can mirror to TensorBoard without crossing package boundaries.
        # See logging plan Tier 1c/3a-3d.
        self.agent.tb_writer = writer
        # Adjust path prefix for Rainbow models
        self.best_model_base_prefix = str(Path(self.run_config.get("model_dir", "models")) / "rainbow_transformer_best")
        # Add checkpoint paths
        self.latest_trainer_checkpoint_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_latest.pt")
        # Store the BASE prefix for the best checkpoint.
        self.best_trainer_checkpoint_base_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_best")
        self.validation_metrics = []
        self.performance_tracker = PerformanceTracker()
        # Guard for ``_finalize_training``: we only overwrite
        # ``rainbow_transformer_final_agent_state.pt`` if at least one trainer
        # checkpoint was persisted during this invocation. Without this guard,
        # an early crash (including the scenario where --resume silently fell
        # back to a fresh agent before this gate was added) would clobber a
        # previously-trained ``_final`` file with untrained weights at finalize.
        self._checkpoints_saved_this_run: int = 0

        progress_path = Path(self.run_config.get("model_dir", "models")) / "progress.jsonl"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_path = str(progress_path)
        logger.info(f"Training progress log: {self._progress_path}")
        self.early_stopping_patience = int(self._cfg.early_stopping_patience)
        self.early_stopping_counter = 0
        self.min_episodes_before_early_stopping = max(0, int(self._cfg.min_episodes_before_early_stopping))
        if self.min_episodes_before_early_stopping > 0:
            logger.info(
                "Early stopping and best-validation tracking deferred until at least %d training episodes complete.",
                self.min_episodes_before_early_stopping,
            )
        self.min_validation_threshold = float(self._cfg.min_validation_threshold)
        self.validation_freq = int(self._cfg.validation_freq)
        self.checkpoint_save_freq = int(self._cfg.checkpoint_save_freq)
        # Disk hygiene: cap the number of `checkpoint_trainer_latest_*` files
        # that are kept on disk after each save. Each checkpoint is ~9 GB
        # because it bundles the full PER buffer state, so an unbounded
        # 50k-episode run would need ~9 TB. Set to 0 to disable rotation
        # (every save kept). `best_*.pt` and `recover` checkpoints are
        # **never** auto-rotated; only the periodic `latest_*` stream is.
        self.latest_checkpoint_keep_last_n = max(0, int(self._cfg.latest_checkpoint_keep_last_n))
        if self.latest_checkpoint_keep_last_n > 0:
            logger.info(
                "Latest-checkpoint rotation enabled: keeping the most recent %d "
                "`checkpoint_trainer_latest_*_ep*_reward*.pt` file(s) after each save.",
                self.latest_checkpoint_keep_last_n,
            )

        # ------------------------------------------------------------------
        # Benchmark allocation fraction schedule (anneal start -> end over N
        # episodes, anchored on the absolute episode index so resume continues
        # the schedule). The CLI flag ``--benchmark-frac-override`` (surfaced
        # via ``run.benchmark_frac_override``) pins a constant, ignoring the
        # schedule entirely.
        #
        # Default behaviour when schedule keys are absent: use
        # ``environment.benchmark_allocation_frac`` (required by
        # :class:`TradingEnvConfig`) as a constant for the whole run. This
        # also keeps live-trader and random-agent flows — which never had a
        # schedule — working unchanged.
        # ------------------------------------------------------------------
        # Benchmark allocation frac schedule — required trainer fields, no silent fallback.
        self.benchmark_frac_start: float = float(self._cfg.benchmark_allocation_frac_start)
        self.benchmark_frac_end: float = float(self._cfg.benchmark_allocation_frac_end)
        self.benchmark_frac_anneal_episodes: int = max(0, int(self._cfg.benchmark_allocation_frac_anneal_episodes))
        # The CLI override is genuinely optional (set via ``--benchmark-frac-override``).
        raw_override = self.run_config.get("benchmark_frac_override")
        try:
            self.benchmark_frac_override: float | None = float(raw_override) if raw_override is not None else None
        except (TypeError, ValueError):
            logger.warning(
                "Invalid run.benchmark_frac_override=%r; ignoring override.",
                raw_override,
            )
            self.benchmark_frac_override = None
        if self.benchmark_frac_override is not None:
            logger.info(
                "Benchmark allocation frac PINNED to %.4f via override (schedule ignored).",
                self.benchmark_frac_override,
            )
        elif self.benchmark_frac_anneal_episodes > 0 and self.benchmark_frac_start != self.benchmark_frac_end:
            logger.info(
                "Benchmark allocation frac scheduled: %.4f -> %.4f over %d episodes.",
                self.benchmark_frac_start,
                self.benchmark_frac_end,
                self.benchmark_frac_anneal_episodes,
            )
        else:
            logger.info(
                "Benchmark allocation frac constant: %.4f (no schedule configured).",
                self.benchmark_frac_start,
            )
        self.reward_window = int(self._cfg.reward_window)
        self.update_freq = int(self._cfg.update_freq)
        self.log_freq = int(self._cfg.log_freq)
        # Reward clipping: optional toggle — ``reward_clip=None`` disables it.
        raw_reward_clip = self._cfg.reward_clip
        self.reward_clip_value: float | None = None
        if raw_reward_clip is not None:
            reward_clip_float = float(raw_reward_clip)
            if reward_clip_float <= 0:
                raise ValueError(f"reward_clip must be positive when set (received {reward_clip_float}); use null in YAML to disable.")
            self.reward_clip_value = reward_clip_float
            logger.info("Reward clipping enabled with ±%.4f bound.", self.reward_clip_value)
        self.gradient_updates_per_step = max(1, int(self._cfg.gradient_updates_per_step))
        self.per_stats_log_freq = max(0, int(self._cfg.per_stats_log_freq))
        if self.per_stats_log_freq == 0:
            logger.info("PER stats logging disabled because per_stats_log_freq is set to 0.")

        # Tier 4a: full-distribution PER audit — optional. ``None`` means
        # "derive from per_stats_log_freq"; 0 disables; positive = explicit cadence.
        if self._cfg.per_buffer_audit_interval is None:
            self.per_buffer_audit_interval = self.per_stats_log_freq * 5 if self.per_stats_log_freq > 0 else 0
        else:
            self.per_buffer_audit_interval = max(0, int(self._cfg.per_buffer_audit_interval))
        if self.per_buffer_audit_interval == 0:
            logger.info("Tier 4a per-buffer distribution audit disabled (interval=0).")

        start_frac = float(self._cfg.final_phase_lr_start_frac)
        multiplier = float(self._cfg.final_phase_lr_multiplier)
        if not (0.0 < start_frac < 1.0):
            raise ValueError(f"final_phase_lr_start_frac must be in (0, 1), got {start_frac}")
        if not (0.0 < multiplier < 1.0):
            raise ValueError(f"final_phase_lr_multiplier must be in (0, 1), got {multiplier}")
        self.final_phase_lr_start_frac: float = start_frac
        self.final_phase_lr_multiplier: float = multiplier
        logger.info(
            "Final-phase LR decay enabled (start_frac=%.3f, multiplier=%.3f).",
            start_frac,
            multiplier,
        )
        self._final_phase_lr_applied = False
        self.warmup_steps = int(self._cfg.warmup_steps)
        # Tier 2d: validation/test runs greedy by default (epsilon=0 + frozen
        # NoisyNet sigma). ``--eval-stochastic`` propagates to ``run.eval_stochastic``
        # to flip back to the legacy stochastic eval path.
        self.eval_stochastic = bool(self.run_config.get("eval_stochastic", False))
        self.num_vector_envs = max(1, int(self._cfg.num_vector_envs))
        self.invalid_action_window = int(self._cfg.invalid_action_window)
        self.invalid_action_rate_window: deque[float] = deque(maxlen=self.invalid_action_window)
        # Extract run parameters
        self.model_dir = self.run_config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self._validation_env_cache: dict[str, TradingEnv] = {}
        self._abort_training = False
        self._abort_reason: str | None = None
        self._abort_step: int | None = None

    def should_stop_early(self, validation_metrics: list[dict]) -> bool:
        """Check if training should stop early based on validation performance."""
        # Early stopping logic now happens directly in the validate() method
        # based on validation_score compared to self.best_validation_metric.
        # This function is no longer used for the primary check.
        return self.early_stopping_counter >= self.early_stopping_patience

    def should_validate(self, episode: int, recent_performance: dict, force: bool = False) -> bool:
        """Determine if validation should be performed based on validation frequency.

        ``force=True`` is used by the vectorized training loop when it has
        already determined (via the ``crossed_validation_boundary`` gate) that a
        validation-frequency boundary was crossed between the previous and
        current ``completed_episodes`` — even if neither value is an exact
        multiple of ``validation_freq``. Without this override the legacy
        ``% == 0`` check silently drops validation whenever multiple envs
        finish in the same ``vec_env.step`` and push the episode counter past
        the exact boundary (the root cause of the April 2026 "no validations"
        regression).
        """
        if self.validation_freq <= 0:
            return False
        if force:
            return True
        return (episode + 1) % self.validation_freq == 0

    def current_benchmark_frac(self, episode: int) -> float:
        """Return the benchmark allocation frac for the given absolute episode.

        Honours the CLI override first; otherwise computes the linear anneal
        from ``benchmark_frac_start`` to ``benchmark_frac_end`` over
        ``benchmark_frac_anneal_episodes``.
        """
        if self.benchmark_frac_override is not None:
            return float(self.benchmark_frac_override)
        return compute_benchmark_frac(
            episode=int(episode),
            start=self.benchmark_frac_start,
            end=self.benchmark_frac_end,
            anneal_episodes=self.benchmark_frac_anneal_episodes,
        )

    def _apply_benchmark_frac_to_env(self, env, episode: int) -> float:
        """Push the current scheduled benchmark frac into a single env's
        :class:`TradingLogic`. Returns the value applied (for logging)."""
        value = self.current_benchmark_frac(episode)
        try:
            env.trading_logic.set_benchmark_allocation_frac(value)
        except AttributeError:
            logger.debug(
                "Env %r exposes no trading_logic.set_benchmark_allocation_frac; skipping.",
                type(env).__name__,
            )
        return value

    def _maybe_emit_benchmark_frac(self, episode: int, value: float) -> None:
        """Emit ``Train/Hyper/BenchmarkFrac`` for the given episode."""
        if self.writer is None:
            return
        try:
            self.writer.add_scalar("Train/Hyper/BenchmarkFrac", float(value), int(episode))
        except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Failed to emit Train/Hyper/BenchmarkFrac", exc_info=True)

    def _validate_validation_cadence_config(self, num_episodes: int, has_val_files: bool) -> None:
        """Hard-fail / loudly warn if the validation cadence won't meaningfully fire.

        Catches three foot-guns we've actually hit in production:

        1. ``validation_freq`` larger than the entire run length, with
           validation files present — guarantees zero ``Validation/*`` scalars
           and no ``best`` checkpoints. Hard-fail with a clear message.
        2. ``validation_freq`` so large that fewer than 5 validations would
           happen across the whole run — the curriculum / best-tracking signal
           gets too sparse to act on. Loud warning.
        3. ``min_episodes_before_early_stopping`` larger than ``num_episodes``
           — guarantees no ``best`` checkpoint ever lands on disk, which
           silently breaks the recovery flow. Loud warning.
        """
        try:
            num_episodes_int = int(num_episodes)
            validation_freq_int = int(self.validation_freq)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping validation-cadence guard: non-integer num_episodes=%r or validation_freq=%r.",
                num_episodes,
                self.validation_freq,
            )
            return

        if has_val_files and validation_freq_int > 0 and validation_freq_int > num_episodes_int:
            raise RuntimeError(
                f"validation_freq ({validation_freq_int}) is larger than num_episodes "
                f"({num_episodes_int}); validation would never fire and no `best` "
                f"checkpoint would ever be saved. Lower validation_freq or raise "
                f"run.episodes (or remove all validation files to opt out explicitly)."
            )

        if has_val_files and validation_freq_int > 0 and num_episodes_int >= 5 and validation_freq_int > max(1, num_episodes_int // 5):
            expected_runs = num_episodes_int // validation_freq_int
            logger.warning(
                "validation_freq=%d will only run validation ~%d time(s) across %d episodes "
                "(less than 5). Curriculum / best-checkpoint signal will be very sparse; "
                "consider lowering validation_freq.",
                validation_freq_int,
                expected_runs,
                num_episodes_int,
            )

        if self.min_episodes_before_early_stopping > 0 and self.min_episodes_before_early_stopping > num_episodes_int:
            logger.warning(
                "min_episodes_before_early_stopping=%d > num_episodes=%d: no `best` "
                "checkpoint will ever be saved during this run. Lower the threshold or "
                "raise run.episodes if you want best-tracking enabled.",
                self.min_episodes_before_early_stopping,
                num_episodes_int,
            )

    def _log_progress(self, event: str, **kwargs) -> None:
        """Append a JSON line to the progress log file."""
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
        }
        record.update(kwargs)
        try:
            with open(self._progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            logger.debug("Failed to append progress record", exc_info=True)

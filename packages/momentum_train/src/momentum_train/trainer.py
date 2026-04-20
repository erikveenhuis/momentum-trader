import json
import math
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from momentum_agent import RainbowDQNAgent  # Updated import path
from momentum_agent.constants import ACCOUNT_STATE_DIM  # Import constant
from momentum_core.logging import get_logger
from momentum_env import TradingEnv, TradingEnvConfig  # Use installed package
from torch.utils.tensorboard import SummaryWriter

from .data import DataManager  # Use relative import
from .metrics import (  # Use relative import
    PerformanceTracker,
    calculate_episode_score,
)
from .schedules import compute_benchmark_frac
from .trade_metrics import (
    StepRecord,
    aggregate_trade_metrics,
    segment_trades,
)

# Get logger instance
logger = get_logger("Trainer")


class RainbowTrainerModule:
    """Orchestrates Rainbow DQN training: episode management, validation, checkpointing, and early stopping."""

    def __init__(
        self,
        agent: RainbowDQNAgent,
        device: torch.device,
        data_manager: DataManager,
        config: dict,
        scaler=None,
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
        self.best_validation_metric = -np.inf
        self.writer = writer
        # Inject the trainer's writer into the agent so internal diagnostics
        # (categorical target stats, NoisyNet sigma, Q-stats, grad norms, target-net
        # deviation, etc.) can mirror to TensorBoard without crossing package boundaries.
        # See logging plan Tier 1c/3a-3d.
        try:
            self.agent.tb_writer = writer
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to attach SummaryWriter to agent.tb_writer", exc_info=True)
        # Adjust path prefix for Rainbow models
        self.best_model_base_prefix = str(Path(self.run_config.get("model_dir", "models")) / "rainbow_transformer_best")
        # Add checkpoint paths
        self.latest_trainer_checkpoint_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_latest.pt")
        # Store the BASE prefix for the best checkpoint.
        self.best_trainer_checkpoint_base_path = str(Path(self.run_config.get("model_dir", "models")) / "checkpoint_trainer_best")
        self.validation_metrics = []
        self.performance_tracker = PerformanceTracker()

        progress_path = Path(self.run_config.get("model_dir", "models")) / "progress.jsonl"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_path = str(progress_path)
        logger.info(f"Training progress log: {self._progress_path}")
        self.early_stopping_patience = self.trainer_config.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        raw_min_eps_before_stop = self.trainer_config.get("min_episodes_before_early_stopping", 0)
        try:
            self.min_episodes_before_early_stopping = max(0, int(raw_min_eps_before_stop))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid min_episodes_before_early_stopping value %s; using 0 (early stopping may begin immediately).",
                raw_min_eps_before_stop,
            )
            self.min_episodes_before_early_stopping = 0
        if self.min_episodes_before_early_stopping > 0:
            logger.info(
                "Early stopping and best-validation tracking deferred until at least %d training episodes complete.",
                self.min_episodes_before_early_stopping,
            )
        self.min_validation_threshold = self.trainer_config.get("min_validation_threshold", -np.inf)
        self.validation_freq = self.trainer_config.get("validation_freq", 10)
        self.checkpoint_save_freq = self.trainer_config.get("checkpoint_save_freq", 10)
        # Disk hygiene: cap the number of `checkpoint_trainer_latest_*` files
        # that are kept on disk after each save. Each checkpoint is ~9 GB
        # because it bundles the full PER buffer state, so an unbounded
        # 50k-episode run would need ~9 TB. Set to 0 to disable rotation
        # (every save kept). `best_*.pt` and `recover` checkpoints are
        # **never** auto-rotated; only the periodic `latest_*` stream is.
        raw_keep_last_n = self.trainer_config.get("latest_checkpoint_keep_last_n", 0)
        try:
            self.latest_checkpoint_keep_last_n = max(0, int(raw_keep_last_n))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid latest_checkpoint_keep_last_n=%r; disabling rotation.",
                raw_keep_last_n,
            )
            self.latest_checkpoint_keep_last_n = 0
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
        env_constant_raw = self.env_config.get("benchmark_allocation_frac")
        env_constant: float = float(env_constant_raw) if env_constant_raw is not None else 0.5
        raw_start = self.trainer_config.get("benchmark_allocation_frac_start")
        raw_end = self.trainer_config.get("benchmark_allocation_frac_end")
        raw_anneal = self.trainer_config.get("benchmark_allocation_frac_anneal_episodes")
        try:
            self.benchmark_frac_start: float = float(raw_start) if raw_start is not None else env_constant
        except (TypeError, ValueError):
            logger.warning(
                "Invalid benchmark_allocation_frac_start=%r; using env-config constant %.4f.",
                raw_start,
                env_constant,
            )
            self.benchmark_frac_start = env_constant
        try:
            self.benchmark_frac_end: float = float(raw_end) if raw_end is not None else env_constant
        except (TypeError, ValueError):
            logger.warning(
                "Invalid benchmark_allocation_frac_end=%r; using env-config constant %.4f.",
                raw_end,
                env_constant,
            )
            self.benchmark_frac_end = env_constant
        try:
            self.benchmark_frac_anneal_episodes: int = int(raw_anneal) if raw_anneal is not None else 0
        except (TypeError, ValueError):
            logger.warning(
                "Invalid benchmark_allocation_frac_anneal_episodes=%r; disabling schedule.",
                raw_anneal,
            )
            self.benchmark_frac_anneal_episodes = 0
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
        self.reward_window = self.trainer_config.get("reward_window", 10)
        self.update_freq = self.trainer_config.get("update_freq", 4)
        self.log_freq = self.trainer_config.get("log_freq", 100)
        self.reward_clip_value: float | None = None
        raw_reward_clip = self.trainer_config.get("reward_clip")
        if raw_reward_clip is not None:
            try:
                reward_clip_float = float(raw_reward_clip)
            except (TypeError, ValueError):
                logger.warning("Invalid reward_clip value %s; disabling reward clipping.", raw_reward_clip)
            else:
                if reward_clip_float <= 0:
                    logger.warning("reward_clip must be positive (received %s); disabling reward clipping.", reward_clip_float)
                else:
                    self.reward_clip_value = reward_clip_float
                    logger.info("Reward clipping enabled with ±%.4f bound.", self.reward_clip_value)
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

        # Tier 4a: full-distribution PER audit — runs less often than the cheap
        # scalar PER stats since it walks 4096 transitions and computes per-action
        # priority + reward histograms. Default = 5x per_stats_log_freq so we get
        # ~5 audits per "interesting" priority drift on TB by default.
        raw_per_audit = self.trainer_config.get(
            "per_buffer_audit_interval",
            (self.per_stats_log_freq or 0) * 5 if self.per_stats_log_freq > 0 else 0,
        )
        try:
            self.per_buffer_audit_interval = int(raw_per_audit)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid per_buffer_audit_interval value %s; disabling Tier 4a audit.",
                raw_per_audit,
            )
            self.per_buffer_audit_interval = 0
        if self.per_buffer_audit_interval < 0:
            logger.warning(
                "per_buffer_audit_interval must be >= 0 (received %s); disabling Tier 4a audit.",
                self.per_buffer_audit_interval,
            )
            self.per_buffer_audit_interval = 0
        elif self.per_buffer_audit_interval == 0:
            logger.info("Tier 4a per-buffer distribution audit disabled (interval=0).")
        raw_final_phase_start = self.trainer_config.get("final_phase_lr_start_frac")
        raw_final_phase_multiplier = self.trainer_config.get("final_phase_lr_multiplier")
        self.final_phase_lr_start_frac: float | None = None
        self.final_phase_lr_multiplier: float | None = None
        if raw_final_phase_start is not None and raw_final_phase_multiplier is not None:
            try:
                start_frac = float(raw_final_phase_start)
                multiplier = float(raw_final_phase_multiplier)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid final phase LR configuration (start=%s, multiplier=%s); disabling final-phase decay.",
                    raw_final_phase_start,
                    raw_final_phase_multiplier,
                )
            else:
                if not (0.0 < start_frac < 1.0):
                    logger.warning(
                        "final_phase_lr_start_frac must be between 0 and 1 (received %.4f); disabling final-phase decay.",
                        start_frac,
                    )
                elif not (0.0 < multiplier < 1.0):
                    logger.warning(
                        "final_phase_lr_multiplier must be between 0 and 1 (received %.4f); disabling final-phase decay.",
                        multiplier,
                    )
                else:
                    self.final_phase_lr_start_frac = start_frac
                    self.final_phase_lr_multiplier = multiplier
                    logger.info(
                        "Final-phase LR decay enabled (start_frac=%.3f, multiplier=%.3f).",
                        start_frac,
                        multiplier,
                    )
        elif raw_final_phase_start is not None or raw_final_phase_multiplier is not None:
            logger.warning(
                "Incomplete final phase LR configuration (start_frac=%s, multiplier=%s); both must be provided. Disabling final-phase decay.",
                raw_final_phase_start,
                raw_final_phase_multiplier,
            )
        self._final_phase_lr_applied = False
        self.warmup_steps = self.trainer_config.get("warmup_steps", 50000)
        # Tier 2d: validation/test runs greedy by default (epsilon=0 + frozen
        # NoisyNet sigma). ``--eval-stochastic`` propagates to ``run.eval_stochastic``
        # to flip back to the legacy stochastic eval path.
        self.eval_stochastic = bool(self.run_config.get("eval_stochastic", False))
        self.num_vector_envs = max(1, int(self.trainer_config.get("num_vector_envs", 1)))
        self.invalid_action_window = self.trainer_config.get("invalid_action_window", 20)
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

    def should_validate(self, episode: int, recent_performance: dict) -> bool:
        """Determine if validation should be performed based on validation frequency."""
        # Removed complex logic based on improvement_rate and stability
        # Simplified to validate purely based on frequency
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
        except Exception:  # pragma: no cover - defensive
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

    @staticmethod
    def _render_on_reset_if_enabled(env, context):
        pass

    @staticmethod
    def _invoke_render(env, context, step_idx):
        pass

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
        except Exception:
            logger.debug("Failed to append progress record", exc_info=True)

    def _rotate_latest_checkpoints(self) -> list[Path]:
        """Delete `checkpoint_trainer_latest_*` files older than the keep-N window.

        Episode number (from the canonical
        ``checkpoint_trainer_latest_<DATE>_ep<N>_reward<...>.pt`` filename) is
        the sort key, **not** mtime — episode numbers are monotonically
        non-decreasing with training progress and survive timezone /
        clock-skew weirdness, while mtime can be reset by file moves.

        ``best_*.pt`` files are never touched: those are the curated "model
        I want to keep" snapshots and the user / recovery script may rely on
        them. The recover-script outputs (``rewardrecover`` marker) are
        intentionally included in the latest-stream rotation because they
        are themselves `latest_*` files written with bumped episode numbers.

        Returns the list of paths that were deleted (for logging / tests).
        Silently returns ``[]`` if rotation is disabled (``keep_last_n=0``).
        """
        keep_n = self.latest_checkpoint_keep_last_n
        if keep_n <= 0:
            return []

        model_dir = Path(self.run_config.get("model_dir", "models"))
        try:
            candidates = list(model_dir.glob("checkpoint_trainer_latest_*_ep*_reward*.pt"))
        except OSError as exc:
            logger.warning("Checkpoint rotation: failed to list %s: %s", model_dir, exc)
            return []

        # Pull the episode number out of the filename. Files that don't match
        # the canonical pattern are skipped (left untouched) rather than
        # deleted — safer to leave an unrecognised file in place.
        ep_re = re.compile(r"^checkpoint_trainer_latest_\d{8}_ep(\d+)_reward.+\.pt$")
        episode_files: list[tuple[int, Path]] = []
        skipped: list[Path] = []
        for path in candidates:
            m = ep_re.match(path.name)
            if not m:
                skipped.append(path)
                continue
            try:
                episode_files.append((int(m.group(1)), path))
            except ValueError:
                skipped.append(path)
        if skipped:
            logger.debug(
                "Checkpoint rotation: skipping %d non-canonical filename(s): %s",
                len(skipped),
                [p.name for p in skipped],
            )

        if len(episode_files) <= keep_n:
            return []

        # Highest episode first; everything after the keep-N window gets pruned.
        episode_files.sort(key=lambda item: item[0], reverse=True)
        to_delete = [path for _, path in episode_files[keep_n:]]
        deleted: list[Path] = []
        for path in to_delete:
            try:
                path.unlink()
                deleted.append(path)
            except OSError as exc:
                logger.warning(
                    "Checkpoint rotation: failed to delete %s: %s",
                    path,
                    exc,
                )
        if deleted:
            logger.info(
                "Checkpoint rotation: pruned %d old `latest_*` checkpoint(s) (kept the most recent %d). Deleted: %s",
                len(deleted),
                keep_n,
                [p.name for p in deleted],
            )
        return deleted

    def _flush_writer(self) -> None:
        """Force-flush TensorBoard events to disk.

        The default ``SummaryWriter`` buffers scalars in-memory for up to
        ``flush_secs`` (120s) or until ``max_queue`` events accumulate. If the
        machine hard-freezes (GPU driver hang, power loss, kernel deadlock) the
        buffered events are lost even though the writer never crashed, which
        produces a gap in TensorBoard between the last flush and the freeze.
        Calling this after every checkpoint / episode guarantees the on-disk
        event file is in lockstep with the on-disk checkpoint state, so the
        largest gap we can ever lose is the few events added since the last
        flush call (rather than up to 2 minutes of training).

        Safe to call when ``self.writer`` is ``None``; errors are swallowed.
        """
        writer = getattr(self, "writer", None)
        if writer is None:
            return
        try:
            writer.flush()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to flush TensorBoard writer", exc_info=True)

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
            "buffer_state": self.agent.buffer.state_dict() if hasattr(self.agent.buffer, "state_dict") else None,
            # --- ADDED Agent State ---
            "agent_config": self.agent.config,
            "agent_total_steps": self.agent.total_steps,
            "agent_env_steps": self.agent.env_steps,
            "total_steps": self.agent.total_steps,
            "network_state_dict": self.agent.network.state_dict() if self.agent.network is not None else None,
            "target_network_state_dict": (self.agent.target_network.state_dict() if self.agent.target_network is not None else None),
            "optimizer_state_dict": (self.agent.optimizer.state_dict() if self.agent.optimizer else None),
            "scaler_state_dict": None,
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
            # Disk hygiene: prune older latest_* checkpoints once the new one
            # is safely on disk. Only runs if `latest_checkpoint_keep_last_n > 0`.
            try:
                self._rotate_latest_checkpoints()
            except Exception:  # noqa: BLE001 — never let rotation break a save
                logger.warning("Checkpoint rotation raised; continuing.", exc_info=True)
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

        # Keep TB event file in sync with the checkpoint we just flushed to
        # disk. Without this, a hard freeze right after a checkpoint save can
        # leave TB scalars stuck ~2 minutes behind the checkpoint's
        # ``total_train_steps``, producing the visible step-axis gap on resume.
        self._flush_writer()

    # --- Refactored Helper Methods ---

    def _initialize_episode(
        self, specific_file: str | None, episode: int, num_episodes: int
    ) -> tuple[TradingEnv | None, dict | None, dict | None, PerformanceTracker | None]:
        """Sets up the environment and performance tracker for a new episode. Returns env, obs, info, tracker."""
        try:
            curriculum_frac = min(1.0, 0.3 + 0.7 * (episode / max(num_episodes, 1)))
            episode_file_path = (
                Path(specific_file) if specific_file else self.data_manager.get_random_training_file(curriculum_frac=curriculum_frac)
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
            self._render_on_reset_if_enabled(env, "train")
            applied_frac = self._apply_benchmark_frac_to_env(env, episode)
            self._maybe_emit_benchmark_frac(episode, applied_frac)
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
                            self.writer.add_scalar("Train/TD_Error_Mean", td_stats.get("mean", float("nan")), total_train_steps)
                            self.writer.add_scalar("Train/TD_Error_Std", td_stats.get("std", float("nan")), total_train_steps)
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
            except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PER distribution audit failed during sampling: %s", exc)
            return

        step = int(total_train_steps)
        try:
            self.writer.add_histogram("Train/PER/Reward/Histogram", rewards, step)
        except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
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
        except Exception as exc:  # pragma: no cover - defensive
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
        for pos, pv in zip(positions, prices_full):
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
                except Exception:  # pragma: no cover - defensive, never block training
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
                except Exception:  # pragma: no cover - defensive
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
                except Exception:  # pragma: no cover - defensive
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

    def _handle_validation_and_checkpointing(
        self,
        episode: int,
        total_train_steps: int,
        val_files: list[Path],
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
                self.min_episodes_before_early_stopping <= 0 or completed_episodes >= self.min_episodes_before_early_stopping
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
                    if final_pv is not None and not (isinstance(final_pv, float) and (math.isnan(final_pv) or math.isinf(final_pv))):
                        self.writer.add_scalar("Validation/Final Portfolio Value", float(final_pv), episode)
                    txn_costs = avg_val_metrics.get("transaction_costs")
                    if txn_costs is not None and not (isinstance(txn_costs, float) and (math.isnan(txn_costs) or math.isinf(txn_costs))):
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
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("Failed to emit Validation/Trade/PnLDistribution: %s", exc)

                # Tier 2d: Train/EvalGap/* — diff between greedy validation and
                # the most recent stochastic-training rolling window. Positive =
                # greedy outperforms training. NaN-safe via _safe_eval_gap_scalar.
                try:
                    train_recent = tracker.get_recent_metrics() if tracker is not None else {}
                except Exception:  # pragma: no cover - defensive
                    train_recent = {}
                if train_recent:
                    self._emit_eval_gap_scalars(avg_val_metrics, train_recent, episode)
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

            if should_stop_training and self.early_stopping_counter < self.early_stopping_patience:
                logger.info("[EarlyStopping] Counter reset after scheduler step; deferring stop to observe reduced learning rate.")
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
                except Exception as exc:  # pragma: no cover - defensive
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
                "trade_metrics": dict(file_metrics.get("trade_metrics", {})) if file_metrics.get("trade_metrics") else {},
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
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            logger.debug("Failed to write trades JSONL sidecar: %s", exc)

    def close_cached_environments(self) -> None:
        """Close any cached validation/test environments."""
        for env_key, env in list(self._validation_env_cache.items()):
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
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
            m.get("trade_metrics", {}) for m in all_file_metrics if isinstance(m.get("trade_metrics"), dict) and m.get("trade_metrics")
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
                logger.info(f"Early stopping triggered after {self.early_stopping_counter} episodes without improvement.")
                should_stop = True
            else:
                should_stop = False

        return should_stop

    # --- End Validation Helper Methods ---

    def validate(self, val_files: list[Path], episode: int = 0) -> tuple[bool, float, dict]:
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

                info_i = {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in infos.items()} if isinstance(infos, dict) else {}
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
                                f"{ai}:{window_counts[ai] * 100 / window_total:.0f}%" for ai in range(self.agent.num_actions)
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
                            self.writer.add_scalar("Train/Total Return Pct", ep_metrics.get("total_return", 0), completed_episodes)
                            self.writer.add_scalar("Train/Sharpe Ratio", ep_metrics.get("sharpe_ratio", 0), completed_episodes)
                            self.writer.add_scalar("Train/Max Drawdown Pct", ep_metrics.get("max_drawdown", 0) * 100, completed_episodes)
                            self.writer.add_scalar("Train/Transaction Costs", ep_metrics.get("transaction_costs", 0), completed_episodes)
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
                                    self.writer.add_scalar(f"Train/Action Rate/Greedy/{action_idx}", gr, completed_episodes)
                                    self.writer.add_scalar(f"Train/Action Rate/Eps/{action_idx}", er, completed_episodes)
                            self.writer.add_scalar(
                                "Train/EpsilonForcedTradeFraction",
                                float(ep_metrics.get("epsilon_forced_trade_fraction", 0.0) or 0.0),
                                completed_episodes,
                            )
                            # Tier 2c: per-episode Train/Trade/* (HitRate/Expectancy/PctGreedy/...)
                            try:
                                vec_trade_metrics = self._trade_metrics_from_tracker(per_env_trackers[i])
                            except Exception:  # pragma: no cover - defensive
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
                            except Exception:  # pragma: no cover - defensive
                                logger.debug("Failed to compute Tier 4b reward outlier stats for env %d", i, exc_info=True)
                                vec_outlier_stats = {}
                            if vec_outlier_stats:
                                self.writer.add_scalar("Train/Episode/RewardMin", vec_outlier_stats["reward_min"], completed_episodes)
                                self.writer.add_scalar("Train/Episode/RewardMax", vec_outlier_stats["reward_max"], completed_episodes)
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
                            except Exception:  # pragma: no cover - defensive
                                logger.debug(
                                    "Failed to compute Tier 4c reward-by-action stats for env %d",
                                    i,
                                    exc_info=True,
                                )
                                vec_by_action = {}
                            for k in range(int(getattr(self.agent, "num_actions", 6))):
                                bucket = vec_by_action.get(k, {"mean": 0.0, "std": 0.0})
                                self.writer.add_scalar(f"Train/Reward/MeanByAction/{k}", float(bucket["mean"]), completed_episodes)
                                self.writer.add_scalar(f"Train/Reward/StdByAction/{k}", float(bucket["std"]), completed_episodes)
                        self.writer.add_scalar("Train/Epsilon", self.agent.current_epsilon, completed_episodes)
                        self.writer.add_scalar("Train/Final Portfolio Value", ep_metrics.get("portfolio_value", 0), completed_episodes)

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
                if crossed_validation_boundary and (completed_episodes % validation_freq_safe != 0) and self.writer is not None:
                    try:
                        self.writer.add_scalar(
                            "Train/Diagnostics/ValidationSkippedDueToVectorJump",
                            1.0,
                            completed_episodes,
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("Failed to emit Train/Diagnostics/ValidationSkippedDueToVectorJump", exc_info=True)

                if crossed_validation_boundary:
                    last_tracker = per_env_trackers[done_indices[-1]]
                    should_stop = self._handle_validation_and_checkpointing(
                        completed_episodes - 1, total_train_steps, val_files, last_tracker
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
                episode_env, initial_obs, initial_info, tracker = self._initialize_episode(specific_file, episode, num_episodes)
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

"""Strict dataclass schema for trainer / run / logging config.

Replaces the ~300 lines of ``self.trainer_config.get("key", default)`` noise in
:class:`~momentum_train.trainer.RainbowTrainerModule.__init__`. Every tunable is
a required field: if the key is missing from ``training_config.yaml`` the run
fails at ``TrainerConfig.from_dict`` time with the offending key name, not
silently two hours into training with a stale 2024 default.

See ``.cursor/rules/no-defaults.mdc`` for the project convention.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Strict schema for the ``trainer`` section of ``training_config.yaml``."""

    seed: int
    num_vector_envs: int

    warmup_steps: int
    update_freq: int
    gradient_updates_per_step: int
    log_freq: int
    per_stats_log_freq: int
    validation_freq: int
    checkpoint_save_freq: int
    latest_checkpoint_keep_last_n: int

    reward_window: int
    early_stopping_patience: int
    min_episodes_before_early_stopping: int
    min_validation_threshold: float

    # Benchmark frac schedule (linear anneal).
    benchmark_allocation_frac_start: float
    benchmark_allocation_frac_end: float
    benchmark_allocation_frac_anneal_episodes: int

    final_phase_lr_start_frac: float
    final_phase_lr_multiplier: float

    # Optional toggles: ``None`` disables the feature entirely.
    reward_clip: float | None = None
    per_buffer_audit_interval: int | None = None
    invalid_action_window: int = 20
    # Top-K best-validation ring. 0 (default) keeps only the threshold-gated
    # ``best_*`` save; set >0 to also pin the K highest-scoring validation
    # checkpoints on disk (via separate ``checkpoint_trainer_topk_*`` files)
    # without honouring ``min_validation_threshold``. Survives restarts because
    # the ring is reconstructed from filenames in ``model_dir`` on each save.
    top_k_best_checkpoints: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainerConfig:
        return _build(cls, raw)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Strict schema for the ``run`` section of ``training_config.yaml``."""

    mode: str
    episodes: int
    model_dir: str
    resume: bool
    skip_evaluation: bool
    # Genuine optional toggles.
    specific_file: str | None = None
    eval_stochastic: bool = False
    eval_model_prefix: str | None = None
    data_base_dir: str = "data"
    benchmark_frac_override: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        return _build(cls, raw)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Strict schema for the ``logging`` block.

    ``level_overrides`` defaults to the empty dict because the trainer merges
    it with built-in package-level defaults; any real override must be spelled
    out explicitly in YAML.
    """

    log_filename: str
    root_level: str
    console_level: str
    file_level: str
    level_overrides: dict[str, str] = field(default_factory=dict)
    logs_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoggingConfig:
        return _build(cls, raw)


def _build(cls, raw: dict[str, Any]):
    """Shared builder: enforce required keys, drop unknowns, raise loudly."""

    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__}.from_dict expects a dict, got {type(raw).__name__}")

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for f in fields(cls):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is MISSING and f.default_factory is MISSING:
            missing.append(f.name)
    if missing:
        raise KeyError(f"{cls.__name__} is missing required keys: " + ", ".join(sorted(missing)))
    return cls(**kwargs)

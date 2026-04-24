"""Strict dataclass schema for the agent configuration block.

Purpose: every tunable that previously came through ``config.get(key, default)``
in :class:`RainbowDQNAgent.__init__` is now a named dataclass field with no
silent fallback. Calling :meth:`AgentConfig.from_dict` with a dict that is
missing any required key raises ``KeyError`` with the offending name so the
training run fails loudly at startup instead of after hours of silently using a
2024-era default for a 2026 run.

See ``.cursor/rules/no-defaults.mdc``: "Missing config must fail loudly with
TypeError/AttributeError/KeyError at startup."
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Strict schema for the ``agent`` section of ``training_config.yaml``.

    All fields are required. ``Optional[...] | None`` is reserved for genuine
    toggles that disable a feature when unset (none currently, but kept as an
    explicit escape hatch for future additions).
    """

    seed: int

    gamma: float
    lr: float
    batch_size: int
    target_update_freq: int
    polyak_tau: float

    window_size: int
    n_features: int
    hidden_dim: int
    num_actions: int

    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    transformer_dropout: float

    n_steps: int
    num_atoms: int
    v_min: float
    v_max: float
    alpha: float
    beta_start: float
    beta_frames: int

    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    entropy_coeff: float

    grad_clip_norm: float
    replay_buffer_size: int
    store_partial_n_step: bool
    debug: bool

    # Diagnostic logging cadences. Set to 0 to disable the corresponding stream.
    categorical_logging_interval: int
    noisy_sigma_logging_interval: int
    q_value_logging_interval: int
    q_value_histogram_interval: int
    grad_logging_interval: int
    target_net_logging_interval: int
    # Tier 2.2: cadence for TD-error mean/std D->H sync. Set to 0 to disable.
    td_error_logging_interval: int

    categorical_logging_percentiles: tuple[float, ...]

    # LR scheduler. The YAML specifies these explicitly; when the scheduler is
    # disabled only ``lr_scheduler_enabled=False`` is read.
    lr_scheduler_enabled: bool
    lr_scheduler_type: str

    # Tier 2.1: single source of truth for the return-prediction auxiliary head
    # (previously hardcoded as 0.1 * MSE on feature index 6 inside
    # ``_compute_loss``). Required — no silent fallback to the 2024 default.
    aux_loss_weight: float
    aux_target_feature_index: int

    lr_scheduler_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentConfig:
        """Build an ``AgentConfig`` from a plain dict, raising on missing keys."""

        if not isinstance(raw, dict):
            raise TypeError(f"AgentConfig.from_dict expects a dict, got {type(raw).__name__}")

        kwargs: dict[str, Any] = {}
        missing: list[str] = []
        for f in fields(cls):
            if f.name in raw:
                kwargs[f.name] = raw[f.name]
            elif _has_default(f):
                continue
            else:
                missing.append(f.name)
        if missing:
            raise KeyError("AgentConfig is missing required keys: " + ", ".join(sorted(missing)))

        # Normalize categorical_logging_percentiles to a tuple so the dataclass
        # stays hashable / frozen-friendly.
        if "categorical_logging_percentiles" in kwargs:
            value = kwargs["categorical_logging_percentiles"]
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"categorical_logging_percentiles must be a list or tuple, got {type(value).__name__}")
            kwargs["categorical_logging_percentiles"] = tuple(float(v) for v in value)

        instance = cls(**kwargs)
        _validate_agent_config(instance)
        return instance


def _has_default(f) -> bool:
    """Return True if a dataclass ``Field`` has a default or default_factory."""
    from dataclasses import MISSING

    return f.default is not MISSING or f.default_factory is not MISSING


def _validate_agent_config(cfg: AgentConfig) -> None:
    """Range-check values that would silently corrupt training if wrong."""

    if not (0.0 < cfg.polyak_tau < 1.0):
        raise ValueError(f"polyak_tau must be in (0, 1), got {cfg.polyak_tau}")
    if not (0.0 < cfg.gamma < 1.0):
        raise ValueError(f"gamma must be in (0, 1), got {cfg.gamma}")
    if cfg.num_atoms < 2:
        raise ValueError(f"num_atoms must be >= 2, got {cfg.num_atoms}")
    if cfg.v_min >= cfg.v_max:
        raise ValueError(f"v_min ({cfg.v_min}) must be < v_max ({cfg.v_max})")
    if cfg.n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {cfg.n_steps}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.replay_buffer_size < cfg.batch_size:
        raise ValueError(f"replay_buffer_size ({cfg.replay_buffer_size}) must be >= batch_size ({cfg.batch_size})")
    if not (0.0 <= cfg.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {cfg.alpha}")
    if not (0.0 <= cfg.beta_start <= 1.0):
        raise ValueError(f"beta_start must be in [0, 1], got {cfg.beta_start}")
    if cfg.lr <= 0:
        raise ValueError(f"lr must be positive, got {cfg.lr}")
    if not (0.0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1.0):
        raise ValueError(f"epsilon schedule invalid: start={cfg.epsilon_start}, end={cfg.epsilon_end} (require 0 <= end <= start <= 1)")
    if cfg.epsilon_decay_steps < 0:
        raise ValueError(f"epsilon_decay_steps must be >= 0, got {cfg.epsilon_decay_steps}")
    if cfg.aux_loss_weight < 0:
        raise ValueError(f"aux_loss_weight must be >= 0, got {cfg.aux_loss_weight}")
    if cfg.aux_target_feature_index < 0:
        raise ValueError(f"aux_target_feature_index must be >= 0, got {cfg.aux_target_feature_index}")

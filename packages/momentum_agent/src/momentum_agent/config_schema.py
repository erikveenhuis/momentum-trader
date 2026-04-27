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

    # IQN distributional head (replaces C51's num_atoms/v_min/v_max).
    n_quantiles_online: int
    n_quantiles_target: int
    n_quantiles_policy: int
    quantile_embedding_dim: int
    huber_kappa: float

    # Munchausen DQN (Vieillard et al. 2020) layered on top of the IQN target.
    # ``alpha=0`` collapses the target back to vanilla IQN/Double-DQN, which is
    # useful for ablations and to confirm the Munchausen path is the only
    # difference. ``entropy_tau`` controls the entropy-regularised softmax; the
    # log-pi clip keeps the bonus bounded when the target net is overconfident.
    munchausen_alpha: float
    munchausen_entropy_tau: float
    munchausen_log_pi_clip: float

    # Spectral normalization on the dueling head NoisyLinears (BTR Stage 3).
    spectral_norm_enabled: bool

    # Diagnostic logging cadences. Set to 0 to disable the corresponding stream.
    quantile_logging_interval: int
    noisy_sigma_logging_interval: int
    q_value_logging_interval: int
    q_value_histogram_interval: int
    grad_logging_interval: int
    target_net_logging_interval: int
    # Cadence for TD-error mean/std D->H sync. Set to 0 to disable.
    td_error_logging_interval: int

    quantile_logging_percentiles: tuple[float, ...]

    # LR scheduler. The YAML specifies these explicitly; when the scheduler is
    # disabled only ``lr_scheduler_enabled=False`` is read.
    lr_scheduler_enabled: bool
    lr_scheduler_type: str

    # Auxiliary return-prediction head.
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

        # Normalize quantile_logging_percentiles to a tuple so the dataclass
        # stays hashable / frozen-friendly.
        if "quantile_logging_percentiles" in kwargs:
            value = kwargs["quantile_logging_percentiles"]
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"quantile_logging_percentiles must be a list or tuple, got {type(value).__name__}")
            kwargs["quantile_logging_percentiles"] = tuple(float(v) for v in value)

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
    if cfg.n_quantiles_online < 8:
        raise ValueError(f"n_quantiles_online must be >= 8, got {cfg.n_quantiles_online}")
    if cfg.n_quantiles_target < 8:
        raise ValueError(f"n_quantiles_target must be >= 8, got {cfg.n_quantiles_target}")
    if cfg.n_quantiles_policy < 8:
        raise ValueError(f"n_quantiles_policy must be >= 8, got {cfg.n_quantiles_policy}")
    if cfg.quantile_embedding_dim < 8:
        raise ValueError(f"quantile_embedding_dim must be >= 8, got {cfg.quantile_embedding_dim}")
    if cfg.huber_kappa <= 0:
        raise ValueError(f"huber_kappa must be > 0, got {cfg.huber_kappa}")
    if not (0.0 <= cfg.munchausen_alpha <= 1.0):
        raise ValueError(f"munchausen_alpha must be in [0, 1], got {cfg.munchausen_alpha}")
    if cfg.munchausen_entropy_tau <= 0:
        raise ValueError(f"munchausen_entropy_tau must be > 0, got {cfg.munchausen_entropy_tau}")
    if cfg.munchausen_log_pi_clip > 0:
        raise ValueError(
            f"munchausen_log_pi_clip must be <= 0 (clip floor for log pi), got {cfg.munchausen_log_pi_clip}"
        )
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
        raise ValueError(
            f"epsilon schedule invalid: start={cfg.epsilon_start}, end={cfg.epsilon_end} (require 0 <= end <= start <= 1)"
        )
    if cfg.epsilon_decay_steps < 0:
        raise ValueError(f"epsilon_decay_steps must be >= 0, got {cfg.epsilon_decay_steps}")
    if cfg.aux_loss_weight < 0:
        raise ValueError(f"aux_loss_weight must be >= 0, got {cfg.aux_loss_weight}")
    if cfg.aux_target_feature_index < 0:
        raise ValueError(f"aux_target_feature_index must be >= 0, got {cfg.aux_target_feature_index}")

"""Network definitions for the IQN + Munchausen + spectral-norm agent.

The network exposes an IQN-style distributional head: a cosine quantile
embedding ``phi(tau) = ReLU(W cos(pi n tau) + b)`` is multiplied element-wise
into the shared encoder feature, after which the dueling value/advantage
heads emit one scalar quantile sample per tau. ``forward`` therefore takes
an extra ``taus`` argument (shape ``[B, K]``, values in ``[0, 1]``) and
returns quantiles ``[B, num_actions, K]``.

C51 specific state (``support``, ``v_min``, ``v_max``, ``num_atoms``) has
been removed — pre-IQN checkpoints are explicitly rejected by
``agent_checkpoint``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from momentum_core.logging import get_logger

from .constants import ACCOUNT_STATE_DIM

logger = get_logger(__name__)


# --- Start: Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """Noisy Linear Layer for Factorised Gaussian Noise.

    Code adapted from:
    https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5, debug: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.debug = debug

        # Learnable parameters for the noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug and x.ndim < 2:
            raise ValueError(f"Input to NoisyLinear must have at least 2 dims (Batch, Features), got shape {x.shape}")
        if self.debug and x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature dim ({x.shape[-1]}) does not match NoisyLinear in_features ({self.in_features})"
            )

        if self.training:
            if self.debug:
                if self.weight_epsilon.shape != (self.out_features, self.in_features):
                    raise ValueError("weight_epsilon shape mismatch")
                if self.bias_epsilon.shape != (self.out_features,):
                    raise ValueError("bias_epsilon shape mismatch")
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        output = F.linear(x, weight, bias)
        expected_shape = list(x.shape[:-1]) + [self.out_features]
        if self.debug and output.shape != torch.Size(expected_shape):
            raise ValueError(
                f"Output shape mismatch from NoisyLinear. Expected {torch.Size(expected_shape)}, got {output.shape}"
            )
        return output


# --- End: Noisy Linear Layer ---


def _apply_spectral_norm_to_noisy(noisy_linear: NoisyLinear) -> NoisyLinear:
    """Wrap ``noisy_linear.weight_mu`` with ``spectral_norm`` parametrization.

    Following BTR (Jauhri et al. 2024) we normalize *only* the deterministic
    ``weight_mu`` of each :class:`NoisyLinear` in the dueling heads. The
    learned exploration scale (``weight_sigma`` / ``bias_sigma``) is left
    unconstrained so NoisyNet's effective noise budget is preserved.
    """
    from torch.nn.utils.parametrizations import spectral_norm

    spectral_norm(noisy_linear, name="weight_mu")
    return noisy_linear


def _maybe_wrap_stream_with_spectral_norm(stream: nn.Sequential, enabled: bool) -> None:
    """Apply spectral norm to every NoisyLinear inside *stream* when enabled."""
    if not enabled:
        return
    for module in stream.modules():
        if isinstance(module, NoisyLinear):
            _apply_spectral_norm_to_noisy(module)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, debug: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.debug = debug

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.debug and x.ndim != 3:
            raise ValueError(f"Input to PositionalEncoding must be 3D (Batch, Seq, Emb), got shape {x.shape}")
        if self.debug and x.shape[2] != self.d_model:
            raise ValueError(
                f"Input embedding dim ({x.shape[2]}) does not match PositionalEncoding d_model ({self.d_model})"
            )
        if self.debug and x.shape[1] > self.pe.shape[0]:
            raise ValueError(
                f"Input sequence length ({x.shape[1]}) exceeds max_len ({self.pe.shape[0]}) of PositionalEncoding"
            )

        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)

        if self.debug and x.shape[2] != self.d_model:
            raise ValueError("Output shape mismatch from PositionalEncoding")
        return x


# --- Start: IQN cosine quantile embedding ---
class CosineQuantileEmbedding(nn.Module):
    """IQN cosine basis embedding (Dabney et al. 2018, Eq. 4).

    For each quantile sample ``tau`` in ``[0, 1]`` we build the cosine
    feature ``cos(pi * n * tau)`` for ``n = 1..embedding_dim`` and then
    project linearly to ``output_dim`` with a ReLU on top:

        phi(tau) = ReLU(W cos(pi n tau) + b)        # [B, K, output_dim]

    The output is element-wise multiplied into the shared encoder feature,
    so ``output_dim`` must equal the encoder's shared-feature dimension.
    """

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        # Buffer holding pi * [1, 2, ..., embedding_dim] for the cosine basis.
        self.register_buffer(
            "indices",
            torch.arange(1, embedding_dim + 1, dtype=torch.float32) * math.pi,
            persistent=False,
        )
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        if taus.ndim != 2:
            raise ValueError(f"taus must be 2D [B, K], got shape {tuple(taus.shape)}")
        # taus: [B, K] -> [B, K, 1] * [embedding_dim] -> [B, K, embedding_dim]
        cos_basis = torch.cos(taus.unsqueeze(-1) * self.indices)
        # Linear maps to output_dim; ReLU is the final non-linearity per the IQN paper.
        return F.relu(self.linear(cos_basis))


# --- End: IQN cosine quantile embedding ---


# --- Start: Rainbow Network Definition ---
class RainbowNetwork(nn.Module):
    """Transformer encoder + IQN dueling heads + auxiliary return predictor.

    The C51 categorical head has been replaced with an IQN quantile head.
    See ``forward`` for the new ``(market_data, account_state, taus)``
    signature.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        *,
        spectral_norm_enabled: bool = False,
    ):
        super().__init__()

        # All required keys; KeyError on missing — mirrors AgentConfig.
        self.window_size = config["window_size"]
        self.n_features = config["n_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_actions = config["num_actions"]
        self.quantile_embedding_dim = config["quantile_embedding_dim"]
        self.nhead = config["nhead"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.dim_feedforward = config["dim_feedforward"]
        self.transformer_dropout = config["transformer_dropout"]
        self.debug_mode = bool(config.get("debug", False))
        self.spectral_norm_enabled = bool(spectral_norm_enabled)

        if self.hidden_dim % self.nhead != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by nhead ({self.nhead})")
        if ACCOUNT_STATE_DIM < 2:
            raise ValueError(f"ACCOUNT_STATE_DIM must be >= 2, got {ACCOUNT_STATE_DIM}")
        if self.quantile_embedding_dim < 1:
            raise ValueError(f"quantile_embedding_dim must be >= 1, got {self.quantile_embedding_dim}")

        logger.info(
            "Initializing RainbowNetwork (IQN) with hidden_dim=%d, quantile_embedding_dim=%d, spectral_norm=%s",
            self.hidden_dim,
            self.quantile_embedding_dim,
            self.spectral_norm_enabled,
        )

        self.device = device

        # --- Shared Feature Extractor ---
        self.feature_embedding = nn.Linear(self.n_features, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_encoder = PositionalEncoding(
            self.hidden_dim,
            dropout=self.transformer_dropout,
            max_len=self.window_size + 1,
            debug=self.debug_mode,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        self.account_processor = nn.Sequential(
            nn.Linear(ACCOUNT_STATE_DIM, self.hidden_dim // 4), nn.GELU(), nn.Dropout(self.transformer_dropout)
        )
        # --- End Shared Feature Extractor ---

        self.shared_feature_dim = self.hidden_dim + self.hidden_dim // 4
        self.head_norm = nn.LayerNorm(self.shared_feature_dim)
        head_hidden_dim = self.hidden_dim // 2

        # IQN cosine quantile embedding maps tau into the shared-feature
        # space so it can be element-wise multiplied with shared_features.
        self.tau_embedding = CosineQuantileEmbedding(
            embedding_dim=self.quantile_embedding_dim,
            output_dim=self.shared_feature_dim,
        )

        # Dueling heads emit one scalar (value) / num_actions scalars
        # (advantage) per tau sample — *not* per atom.
        self.value_stream = nn.Sequential(
            NoisyLinear(self.shared_feature_dim, head_hidden_dim, debug=self.debug_mode),
            nn.ReLU(),
            NoisyLinear(head_hidden_dim, 1, debug=self.debug_mode),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.shared_feature_dim, head_hidden_dim, debug=self.debug_mode),
            nn.ReLU(),
            NoisyLinear(head_hidden_dim, self.num_actions, debug=self.debug_mode),
        )

        self.aux_return_head = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

        self._initialize_weights()

        # Spectral-norm wrapping has to happen *after* reset_parameters so
        # the parametrization sees a properly-initialized weight_mu.
        if self.spectral_norm_enabled:
            _maybe_wrap_stream_with_spectral_norm(self.value_stream, True)
            _maybe_wrap_stream_with_spectral_norm(self.advantage_stream, True)

    def _initialize_weights(self):
        """Initializes weights for Linear layers and resets NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, NoisyLinear):
                m.reset_parameters()
        nn.init.zeros_(self.cls_token)

    def _encode_market(self, market_data: torch.Tensor) -> torch.Tensor:
        """Run the (expensive) Transformer encoder once and return the CLS token."""
        batch_size = market_data.shape[0]
        market_emb = self.feature_embedding(market_data)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        market_emb = torch.cat([cls_tokens, market_emb], dim=1)
        market_emb = self.pos_encoder(market_emb)
        market_features = self.transformer_encoder(market_emb)
        return market_features[:, 0, :]  # CLS output: [B, hidden_dim]

    def _shared_features(self, cls_out: torch.Tensor, account_state: torch.Tensor) -> torch.Tensor:
        """Concat encoder + account features and apply the shared LayerNorm."""
        account_features = self.account_processor(account_state)
        shared = torch.cat([cls_out, account_features], dim=1)
        return self.head_norm(shared)

    def _quantiles_from_features(
        self,
        shared_features: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantile values from cached shared features + taus.

        Args:
            shared_features: ``[B, shared_feature_dim]``.
            taus: ``[B, K]`` quantile fractions in ``[0, 1]``.

        Returns:
            ``[B, num_actions, K]`` quantile estimates.
        """
        if shared_features.ndim != 2:
            raise ValueError(f"shared_features must be 2D, got shape {tuple(shared_features.shape)}")
        if taus.ndim != 2:
            raise ValueError(f"taus must be 2D [B, K], got shape {tuple(taus.shape)}")
        if taus.shape[0] != shared_features.shape[0]:
            raise ValueError(
                f"taus batch dim ({taus.shape[0]}) must match shared_features batch dim ({shared_features.shape[0]})"
            )
        batch_size = shared_features.shape[0]
        num_taus = taus.shape[1]

        phi = self.tau_embedding(taus)  # [B, K, sf_dim]
        # Element-wise gating: shared_features broadcast across K, then * phi.
        sf_tau = shared_features.unsqueeze(1) * phi  # [B, K, sf_dim]
        sf_flat = sf_tau.reshape(batch_size * num_taus, self.shared_feature_dim)

        # Heads emit per-tau scalars; reshape back to [B, K, ...].
        value = self.value_stream(sf_flat).view(batch_size, num_taus, 1)
        advantage = self.advantage_stream(sf_flat).view(batch_size, num_taus, self.num_actions)

        # Dueling combine over the action axis (axis 2), per IQN/dueling.
        quantiles = value + advantage - advantage.mean(dim=2, keepdim=True)  # [B, K, A]
        # Standardize layout: caller expects [B, num_actions, K].
        return quantiles.transpose(1, 2).contiguous()

    def forward(
        self,
        market_data: torch.Tensor,
        account_state: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Returns quantile estimates ``[B, num_actions, K]``."""
        if self.debug_mode:
            if market_data.ndim != 3:
                raise ValueError(f"Input market_data must be 3D (Batch, Seq, Feat), got shape {market_data.shape}")
            if market_data.shape[1] != self.window_size:
                raise ValueError(
                    f"Input market_data seq len ({market_data.shape[1]}) != window_size ({self.window_size})"
                )
            if market_data.shape[2] != self.n_features:
                raise ValueError(
                    f"Input market_data feat dim ({market_data.shape[2]}) != n_features ({self.n_features})"
                )
            if account_state.ndim != 2:
                raise ValueError(
                    f"Input account_state must be 2D (Batch, Feat={ACCOUNT_STATE_DIM}), got shape {account_state.shape}"
                )
            if account_state.shape[1] != ACCOUNT_STATE_DIM:
                raise ValueError(
                    f"Input account_state must have {ACCOUNT_STATE_DIM} features, got {account_state.shape[1]}"
                )
            if market_data.shape[0] != account_state.shape[0]:
                raise ValueError("Batch size mismatch between market_data and account_state")
            if taus.ndim != 2:
                raise ValueError(f"taus must be 2D [B, K], got shape {tuple(taus.shape)}")
            if taus.shape[0] != market_data.shape[0]:
                raise ValueError("Batch size mismatch between taus and market_data")

        cls_out = self._encode_market(market_data)
        shared_features = self._shared_features(cls_out, account_state)
        quantiles = self._quantiles_from_features(shared_features, taus)

        if self.debug_mode:
            if torch.isnan(quantiles).any():
                raise ValueError("NaN detected in IQN quantiles")
            if torch.isinf(quantiles).any():
                raise ValueError("Inf detected in IQN quantiles")

        return quantiles

    def forward_with_cls(
        self,
        market_data: torch.Tensor,
        account_state: torch.Tensor,
        taus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(quantiles, cls_out)`` with a single encoder pass.

        The training loss reuses ``cls_out`` for the auxiliary return-prediction
        head so we don't run the Transformer twice per learn step.
        """
        cls_out = self._encode_market(market_data)
        shared_features = self._shared_features(cls_out, account_state)
        quantiles = self._quantiles_from_features(shared_features, taus)
        return quantiles, cls_out

    def predict_return(self, market_data: torch.Tensor) -> torch.Tensor:
        """Auxiliary head: predict next-step log return from CLS features."""
        return self.aux_from_cls(self._encode_market(market_data))

    def aux_from_cls(self, cls_out: torch.Tensor) -> torch.Tensor:
        """Apply the auxiliary return-prediction head to a cached CLS encoding."""
        return self.aux_return_head(cls_out).squeeze(-1)

    def get_quantiles(
        self,
        market_data: torch.Tensor,
        account_state: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`. Returns ``[B, num_actions, K]`` quantiles."""
        return self.forward(market_data, account_state, taus)

    def get_q_values(
        self,
        market_data: torch.Tensor,
        account_state: torch.Tensor,
        taus: torch.Tensor,
    ) -> torch.Tensor:
        """Risk-neutral expected Q-values: mean of quantiles over the K axis."""
        quantiles = self.forward(market_data, account_state, taus)  # [B, A, K]
        q_values = quantiles.mean(dim=2)  # [B, A]
        if self.debug_mode and q_values.shape != (market_data.shape[0], self.num_actions):
            raise ValueError("Output q_values shape mismatch")
        return q_values

    def reset_noise(self) -> None:
        """Resets noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- End: Rainbow Network Definition ---

import math  # Import math for positional encoding calculation
from typing import Any, Dict  # Added for type hinting

import torch
import torch.nn as nn
import torch.nn.functional as F
from momentum_core.logging import get_logger

from .constants import ACCOUNT_STATE_DIM

# Get logger instance
logger = get_logger("TransformerModel")


# --- Start: Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """Noisy Linear Layer for Factorised Gaussian Noise.

    Code adapted from:
    https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

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
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))  # Corrected bias sigma init

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2, f"Input to NoisyLinear must have at least 2 dims (Batch, Features), got shape {x.shape}"
        assert (
            x.shape[-1] == self.in_features
        ), f"Input feature dim ({x.shape[-1]}) does not match NoisyLinear in_features ({self.in_features})"

        if self.training:
            assert self.weight_epsilon.shape == (
                self.out_features,
                self.in_features,
            ), "weight_epsilon shape mismatch"
            assert self.bias_epsilon.shape == (self.out_features,), "bias_epsilon shape mismatch"
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        output = F.linear(x, weight, bias)
        expected_shape = list(x.shape[:-1]) + [self.out_features]
        assert output.shape == torch.Size(
            expected_shape
        ), f"Output shape mismatch from NoisyLinear. Expected {torch.Size(expected_shape)}, got {output.shape}"
        return output


# --- End: Noisy Linear Layer ---


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model  # Store d_model

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
        assert x.ndim == 3, f"Input to PositionalEncoding must be 3D (Batch, Seq, Emb), got shape {x.shape}"
        assert x.shape[2] == self.d_model, f"Input embedding dim ({x.shape[2]}) does not match PositionalEncoding d_model ({self.d_model})"
        assert (
            x.shape[1] <= self.pe.shape[0]
        ), f"Input sequence length ({x.shape[1]}) exceeds max_len ({self.pe.shape[0]}) of PositionalEncoding"

        # Positional encoding expects shape [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # Permute back to [batch_size, seq_len, embedding_dim]

        assert x.shape == (
            x.shape[0],
            x.shape[1],
            self.d_model,
        ), "Output shape mismatch from PositionalEncoding"
        return x


# --- Start: Rainbow Network Definition ---
class RainbowNetwork(nn.Module):
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super(RainbowNetwork, self).__init__()

        # Extract parameters from config - NO DEFAULTS, will raise KeyError if missing
        self.window_size = config["window_size"]
        self.n_features = config["n_features"]
        self.hidden_dim = config["hidden_dim"]  # Also embedding dim
        self.num_actions = config["num_actions"]
        self.num_atoms = config["num_atoms"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        # Transformer specific params
        self.nhead = config["nhead"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.dim_feedforward = config["dim_feedforward"]
        self.transformer_dropout = config["transformer_dropout"]

        # Check validity
        assert self.hidden_dim % self.nhead == 0, f"hidden_dim ({self.hidden_dim}) must be divisible by nhead ({self.nhead})"
        assert ACCOUNT_STATE_DIM == 2, "ACCOUNT_STATE_DIM mismatch"

        logger.info(f"Initializing RainbowNetwork with hidden_dim={self.hidden_dim}")

        self.device = device

        # Save configuration
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.num_atoms))
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # --- Shared Feature Extractor (Similar to Actor/Critic start) ---
        self.feature_embedding = nn.Linear(self.n_features, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_encoder = PositionalEncoding(self.hidden_dim, dropout=self.transformer_dropout, max_len=self.window_size + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,  # Use config value
            dim_feedforward=self.dim_feedforward,  # Use config value
            dropout=self.transformer_dropout,  # Use config value
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)  # Use config value
        self.account_processor = nn.Sequential(nn.Linear(2, self.hidden_dim // 4), nn.ReLU(), nn.Dropout(self.transformer_dropout))
        # --- End Shared Feature Extractor ---

        # --- Dueling Network Heads (Using Noisy Layers) ---
        # Calculate input dimension for heads after feature extraction and aggregation
        shared_feature_dim = self.hidden_dim + self.hidden_dim // 4
        head_hidden_dim = self.hidden_dim // 2  # Hidden dimension for value/advantage streams

        # Value Stream
        self.value_stream = nn.Sequential(
            NoisyLinear(shared_feature_dim, head_hidden_dim),
            nn.ReLU(),
            NoisyLinear(head_hidden_dim, self.num_atoms),
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(shared_feature_dim, head_hidden_dim),
            nn.ReLU(),
            NoisyLinear(head_hidden_dim, self.num_actions * self.num_atoms),
        )
        # --- End Dueling Network Heads ---

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights for Linear layers and resets NoisyLinear layers."""
        # Basic initialization - could be refined
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, NoisyLinear):
                m.reset_parameters()
        nn.init.zeros_(self.cls_token)

    def forward(self, market_data: torch.Tensor, account_state: torch.Tensor) -> torch.Tensor:
        # --- Input Asserts ---
        assert market_data.ndim == 3, f"Input market_data must be 3D (Batch, Seq, Feat), got shape {market_data.shape}"
        assert (
            market_data.shape[1] == self.window_size
        ), f"Input market_data seq len ({market_data.shape[1]}) != window_size ({self.window_size})"
        assert (
            market_data.shape[2] == self.n_features
        ), f"Input market_data feat dim ({market_data.shape[2]}) != n_features ({self.n_features})"
        assert account_state.ndim == 2, f"Input account_state must be 2D (Batch, Feat=2), got shape {account_state.shape}"
        assert account_state.shape[1] == 2, f"Input account_state must have 2 features, got {account_state.shape[1]}"
        assert market_data.shape[0] == account_state.shape[0], "Batch size mismatch between market_data and account_state"
        batch_size = market_data.shape[0]
        # --- End Input Asserts ---

        # --- Run Shared Feature Extractor ---
        market_emb = self.feature_embedding(market_data)
        assert market_emb.shape == (
            batch_size,
            self.window_size,
            self.hidden_dim,
        ), "market_emb shape mismatch"
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        market_emb = torch.cat([cls_tokens, market_emb], dim=1)
        assert market_emb.shape == (
            batch_size,
            self.window_size + 1,
            self.hidden_dim,
        ), "market_emb shape with CLS token mismatch"
        market_emb = self.pos_encoder(market_emb)
        assert market_emb.shape == (
            batch_size,
            self.window_size + 1,
            self.hidden_dim,
        ), "market_emb shape after pos_encoder mismatch"
        market_features = self.transformer_encoder(market_emb)
        assert market_features.shape == (
            batch_size,
            self.window_size + 1,
            self.hidden_dim,
        ), "market_features shape mismatch"
        market_agg = market_features[:, 0, :]
        assert market_agg.shape == (
            batch_size,
            self.hidden_dim,
        ), "market_agg shape mismatch"
        account_features = self.account_processor(account_state)
        assert account_features.shape == (
            batch_size,
            self.hidden_dim // 4,
        ), "account_features shape mismatch"
        shared_features = torch.cat([market_agg, account_features], dim=1)
        expected_shared_dim = self.hidden_dim + self.hidden_dim // 4
        assert shared_features.shape == (
            batch_size,
            expected_shared_dim,
        ), "shared_features shape mismatch"
        # -----------------------------------

        # --- Dueling Streams ---
        value_logits = self.value_stream(shared_features)
        assert value_logits.shape == (
            batch_size,
            self.num_atoms,
        ), "value_logits shape mismatch"
        advantage_logits = self.advantage_stream(shared_features)
        assert advantage_logits.shape == (
            batch_size,
            self.num_actions * self.num_atoms,
        ), "advantage_logits shape mismatch"

        # Reshape for distributional calculations
        value_logits = value_logits.view(batch_size, 1, self.num_atoms)
        advantage_logits = advantage_logits.view(batch_size, self.num_actions, self.num_atoms)
        # ---------------------

        # --- Combine Streams for Q-Distribution ---
        q_logits = value_logits + advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
        # Add numerical stability
        q_logits = torch.clamp(q_logits, min=-1e4, max=1e4)
        assert q_logits.shape == (
            batch_size,
            self.num_actions,
            self.num_atoms,
        ), "q_logits shape mismatch"
        # --- ADDED: Check q_logits before log_softmax ---
        assert not torch.isnan(q_logits).any(), "NaN detected in q_logits before log_softmax"
        assert not torch.isinf(q_logits).any(), "Inf detected in q_logits before log_softmax"
        # --- END ADDED ---
        # -----------------------------------------

        # --- Get Log Probabilities (for training loss) ---
        log_probs = F.log_softmax(q_logits, dim=2)
        assert log_probs.shape == (
            batch_size,
            self.num_actions,
            self.num_atoms,
        ), "log_probs shape mismatch"
        # Check for NaN/Inf
        assert not torch.isnan(log_probs).any(), "NaN detected in final log_probs"
        assert not torch.isinf(log_probs).any(), "Inf detected in final log_probs"
        # ---------------------------------------------

        return log_probs

    def get_q_values(self, market_data: torch.Tensor, account_state: torch.Tensor) -> torch.Tensor:
        """Helper function to get expected Q-values for action selection."""
        log_probs = self.forward(market_data, account_state)
        probs = torch.exp(log_probs)
        # Calculate Q-values from probabilities and support
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(1)).sum(dim=2)  # Expand support dims for broadcast

        assert q_values.shape == (
            market_data.shape[0],
            self.num_actions,
        ), "Output q_values shape mismatch"
        return q_values

    def reset_noise(self) -> None:
        """Resets noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- End: Rainbow Network Definition ---

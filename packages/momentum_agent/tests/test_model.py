"""Unit tests for ``momentum_agent.model`` (IQN edition).

The Beyond-the-Rainbow upgrade replaced the C51 categorical head with an
IQN quantile head, so these tests now cover:

* :class:`NoisyLinear` / :class:`PositionalEncoding` (unchanged primitives).
* :class:`CosineQuantileEmbedding` shape + range invariants.
* :class:`RainbowNetwork` IQN forward signatures
  (``forward``, ``forward_with_cls``, ``get_quantiles``, ``get_q_values``).
* The auxiliary return-prediction head, which still consumes only the CLS
  encoding and is therefore independent of the quantile samples.

Spectral-norm wrapping has its own dedicated tests in ``test_spectral_norm.py``
once Stage 3 lands; here we restrict ourselves to the default
``spectral_norm_enabled=False`` path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from momentum_agent.constants import ACCOUNT_STATE_DIM
from momentum_agent.model import (
    CosineQuantileEmbedding,
    NoisyLinear,
    PositionalEncoding,
    RainbowNetwork,
)


@pytest.fixture(scope="module")
def default_config():
    """Minimal IQN-shaped config dict for :class:`RainbowNetwork`."""
    return {
        "seed": 42,
        "window_size": 10,
        "n_features": 12,
        "hidden_dim": 64,
        "num_actions": 6,
        "n_quantiles_online": 16,
        "n_quantiles_target": 16,
        "n_quantiles_policy": 8,
        "quantile_embedding_dim": 32,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "transformer_dropout": 0.1,
        "batch_size": 4,
    }


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# NoisyLinear
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noisy_linear_init():
    layer = NoisyLinear(in_features=10, out_features=5)
    assert layer.in_features == 10
    assert layer.out_features == 5
    assert layer.weight_mu.shape == (5, 10)
    assert layer.weight_sigma.shape == (5, 10)
    assert layer.bias_mu.shape == (5,)
    assert layer.bias_sigma.shape == (5,)
    assert hasattr(layer, "weight_epsilon")
    assert hasattr(layer, "bias_epsilon")


@pytest.mark.unit
def test_noisy_linear_forward_train(device):
    layer = NoisyLinear(in_features=10, out_features=5).to(device)
    layer.train()
    dummy_input = torch.randn(4, 10, device=device)

    output = layer(dummy_input)

    assert output.shape == (4, 5)
    assert output.device.type == device.type
    with torch.no_grad():
        mu_output = F.linear(dummy_input, layer.weight_mu, layer.bias_mu)
    assert not torch.allclose(output, mu_output)


@pytest.mark.unit
def test_noisy_linear_forward_eval(device):
    layer = NoisyLinear(in_features=10, out_features=5).to(device)
    layer.eval()
    dummy_input = torch.randn(4, 10, device=device)

    output = layer(dummy_input)
    with torch.no_grad():
        mu_output = F.linear(dummy_input, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(output, mu_output)


@pytest.mark.unit
def test_noisy_linear_reset_noise(device):
    layer = NoisyLinear(in_features=10, out_features=5).to(device)
    layer.train()
    initial_weight_eps = layer.weight_epsilon.clone().detach()
    initial_bias_eps = layer.bias_epsilon.clone().detach()
    layer.reset_noise()
    assert not torch.equal(initial_weight_eps, layer.weight_epsilon)
    assert not torch.equal(initial_bias_eps, layer.bias_epsilon)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_positional_encoding_init():
    pe = PositionalEncoding(d_model=64, max_len=50)
    assert pe.d_model == 64
    assert pe.pe.shape == (50, 1, 64)


@pytest.mark.unit
def test_positional_encoding_forward(device):
    pe = PositionalEncoding(d_model=64, max_len=50).to(device)
    dummy_input = torch.randn(4, 20, 64, device=device)
    output = pe(dummy_input)
    assert output.shape == (4, 20, 64)
    assert not torch.allclose(output, dummy_input)


# ---------------------------------------------------------------------------
# CosineQuantileEmbedding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cosine_quantile_embedding_shape(device):
    emb = CosineQuantileEmbedding(embedding_dim=32, output_dim=16).to(device)
    taus = torch.rand(4, 8, device=device)
    out = emb(taus)
    assert out.shape == (4, 8, 16)
    # ReLU output -> all non-negative.
    assert (out >= 0).all()


@pytest.mark.unit
def test_cosine_quantile_embedding_rejects_wrong_dims(device):
    emb = CosineQuantileEmbedding(embedding_dim=8, output_dim=4).to(device)
    with pytest.raises(ValueError, match="taus must be 2D"):
        emb(torch.rand(4, device=device))
    with pytest.raises(ValueError, match="taus must be 2D"):
        emb(torch.rand(2, 3, 4, device=device))


@pytest.mark.unit
def test_cosine_quantile_embedding_rejects_zero_dims():
    with pytest.raises(ValueError, match="embedding_dim"):
        CosineQuantileEmbedding(embedding_dim=0, output_dim=4)
    with pytest.raises(ValueError, match="output_dim"):
        CosineQuantileEmbedding(embedding_dim=4, output_dim=0)


@pytest.mark.unit
def test_cosine_quantile_embedding_distinct_taus_distinct_features(device):
    """Distinct tau samples should produce distinct embeddings."""
    torch.manual_seed(0)
    emb = CosineQuantileEmbedding(embedding_dim=16, output_dim=12).to(device)
    taus = torch.tensor([[0.1, 0.5, 0.9]], device=device)
    out = emb(taus)
    # Pairwise embeddings shouldn't collapse.
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 0], out[0, 2])
    assert not torch.allclose(out[0, 1], out[0, 2])


# ---------------------------------------------------------------------------
# RainbowNetwork
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def network(default_config, device):
    net = RainbowNetwork(config=default_config, device=device).to(device)
    net.eval()
    return net


def _make_inputs(default_config, device, *, batch_size: int, k: int):
    market = torch.randn(batch_size, default_config["window_size"], default_config["n_features"], device=device)
    account = torch.randn(batch_size, ACCOUNT_STATE_DIM, device=device)
    taus = torch.rand(batch_size, k, device=device)
    return market, account, taus


@pytest.mark.unit
def test_rainbow_network_init(network, default_config, device):
    assert network is not None
    assert network.window_size == default_config["window_size"]
    assert network.n_features == default_config["n_features"]
    assert network.hidden_dim == default_config["hidden_dim"]
    assert network.num_actions == default_config["num_actions"]
    assert network.quantile_embedding_dim == default_config["quantile_embedding_dim"]
    # IQN replaces C51 -- there must be no num_atoms / support state.
    assert not hasattr(network, "num_atoms")
    assert not hasattr(network, "support")
    assert not hasattr(network, "v_min")
    assert not hasattr(network, "v_max")
    # Tau embedding lives on the network.
    assert isinstance(network.tau_embedding, CosineQuantileEmbedding)
    assert network.tau_embedding.output_dim == network.shared_feature_dim
    for param in network.parameters():
        assert str(param.device).startswith(str(device))


@pytest.mark.unit
def test_rainbow_network_forward_returns_quantiles(network, default_config, device):
    batch_size = default_config["batch_size"]
    k = 12
    market, account, taus = _make_inputs(default_config, device, batch_size=batch_size, k=k)

    quantiles = network(market, account, taus)

    assert quantiles.shape == (batch_size, default_config["num_actions"], k)
    assert quantiles.device.type == device.type
    assert torch.isfinite(quantiles).all()


@pytest.mark.unit
def test_rainbow_network_forward_with_cls(network, default_config, device):
    batch_size = default_config["batch_size"]
    market, account, taus = _make_inputs(default_config, device, batch_size=batch_size, k=8)

    quantiles, cls_out = network.forward_with_cls(market, account, taus)

    assert quantiles.shape == (batch_size, default_config["num_actions"], 8)
    assert cls_out.shape == (batch_size, default_config["hidden_dim"])
    # Aux head consumes the same CLS token without re-encoding.
    aux = network.aux_from_cls(cls_out)
    assert aux.shape == (batch_size,)


@pytest.mark.unit
def test_rainbow_network_get_q_values_is_quantile_mean(network, default_config, device):
    batch_size = default_config["batch_size"]
    k = 16
    market, account, taus = _make_inputs(default_config, device, batch_size=batch_size, k=k)

    q_values = network.get_q_values(market, account, taus)
    quantiles = network.get_quantiles(market, account, taus)

    assert q_values.shape == (batch_size, default_config["num_actions"])
    assert torch.isfinite(q_values).all()
    # Risk-neutral expected Q == quantile mean.
    assert torch.allclose(q_values, quantiles.mean(dim=2), atol=1e-6)


@pytest.mark.unit
def test_rainbow_network_quantiles_change_with_taus(network, default_config, device):
    """Different tau samples should yield different quantile outputs."""
    network.eval()
    torch.manual_seed(0)
    market, account, _ = _make_inputs(default_config, device, batch_size=2, k=4)
    taus_a = torch.full((2, 4), 0.1, device=device)
    taus_b = torch.full((2, 4), 0.9, device=device)

    q_a = network(market, account, taus_a)
    q_b = network(market, account, taus_b)
    assert not torch.allclose(q_a, q_b)


@pytest.mark.unit
def test_rainbow_network_reset_noise(network, device):
    network.train()
    noisy_layer = next(m for m in network.modules() if isinstance(m, NoisyLinear))
    initial_weight_eps = noisy_layer.weight_epsilon.clone().detach()
    initial_bias_eps = noisy_layer.bias_epsilon.clone().detach()
    network.reset_noise()
    assert not torch.equal(initial_weight_eps, noisy_layer.weight_epsilon)
    assert not torch.equal(initial_bias_eps, noisy_layer.bias_epsilon)
    network.eval()


@pytest.mark.unit
def test_rainbow_network_train_eval_modes(network, default_config, device):
    market, account, taus = _make_inputs(default_config, device, batch_size=default_config["batch_size"], k=8)

    network.eval()
    q_eval = network.get_q_values(market, account, taus)
    assert not any(m.training for m in network.modules() if isinstance(m, (nn.Dropout, NoisyLinear)))

    network.train()
    q_train = network.get_q_values(market, account, taus)
    assert any(m.training for m in network.modules() if isinstance(m, (nn.Dropout, NoisyLinear)))
    assert not torch.allclose(q_eval, q_train)
    network.eval()


@pytest.mark.unit
def test_predict_return_output_shape(default_config, device):
    net = RainbowNetwork(config=default_config, device=device).to(device)
    market = torch.randn(4, default_config["window_size"], default_config["n_features"], device=device)
    pred = net.predict_return(market)
    assert pred.shape == (4,)
    assert torch.isfinite(pred).all()


@pytest.mark.unit
def test_debug_mode_catches_wrong_account_dim(default_config, device):
    config = {**default_config, "debug": True}
    net = RainbowNetwork(config=config, device=device).to(device)
    batch_size = 2
    market = torch.randn(batch_size, config["window_size"], config["n_features"], device=device)
    wrong_account = torch.randn(batch_size, ACCOUNT_STATE_DIM + 1, device=device)
    taus = torch.rand(batch_size, 4, device=device)
    with pytest.raises(ValueError, match="account_state"):
        net.forward(market, wrong_account, taus)


@pytest.mark.unit
def test_debug_mode_catches_wrong_market_dim(default_config, device):
    config = {**default_config, "debug": True}
    net = RainbowNetwork(config=config, device=device).to(device)
    batch_size = 2
    wrong_market = torch.randn(batch_size, config["window_size"], 3, device=device)
    account = torch.randn(batch_size, ACCOUNT_STATE_DIM, device=device)
    taus = torch.rand(batch_size, 4, device=device)
    with pytest.raises(ValueError, match="market_data"):
        net.forward(wrong_market, account, taus)


@pytest.mark.unit
def test_forward_rejects_wrong_taus_shape(network, default_config, device):
    market, account, _ = _make_inputs(default_config, device, batch_size=2, k=4)
    with pytest.raises(ValueError, match="taus must be 2D"):
        network(market, account, torch.rand(2, device=device))

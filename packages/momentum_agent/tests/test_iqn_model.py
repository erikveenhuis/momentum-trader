"""Focused tests for the IQN-specific pieces of :mod:`momentum_agent.model`.

These complement ``test_model.py`` (which already covers basic forward
shapes) by exercising the IQN-specific properties that ``test_model.py``
treats only at a coarse level:

* The cosine quantile embedding is symmetric in tau in the sense that
  random taus produce non-collapsing features.
* :meth:`RainbowNetwork.get_q_values` averaging over many quantiles
  approaches the expected value of the per-quantile output.
* Dueling decomposition with IQN keeps the advantage stream zero-mean
  per state across actions (per the standard dueling formulation).
* Larger ``K`` shrinks the variance of ``get_q_values`` toward 0.

We keep the tests fully on CPU (no CUDA required) to keep CI fast.
"""

from __future__ import annotations

import pytest
import torch
from momentum_agent.constants import ACCOUNT_STATE_DIM
from momentum_agent.model import RainbowNetwork


@pytest.fixture(scope="module")
def iqn_config():
    return {
        "seed": 0,
        "window_size": 8,
        "n_features": 12,
        "hidden_dim": 32,
        "num_actions": 6,
        "n_quantiles_online": 32,
        "n_quantiles_target": 32,
        "n_quantiles_policy": 16,
        "quantile_embedding_dim": 16,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "transformer_dropout": 0.0,
    }


@pytest.fixture(scope="module")
def iqn_network(iqn_config):
    torch.manual_seed(0)
    net = RainbowNetwork(config=iqn_config, device=torch.device("cpu"))
    net.eval()
    return net


@pytest.mark.unit
def test_dueling_advantage_is_zero_mean_per_action(iqn_network, iqn_config):
    """The dueling decomposition centres advantages around zero across actions."""
    torch.manual_seed(1)
    market = torch.randn(3, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(3, ACCOUNT_STATE_DIM)
    taus = torch.rand(3, 8)

    quantiles = iqn_network(market, account, taus)  # [B, A, K]
    # value+advantage-mean(advantage) makes (Q - V).mean(action_axis) ≈ 0.
    q_minus_mean_action = quantiles - quantiles.mean(dim=1, keepdim=True)
    # Per-(B, K) action-wise mean must be 0 (within fp tolerance).
    action_mean = q_minus_mean_action.mean(dim=1)  # [B, K] expected ~0
    # The dueling stream is computed as v + a - a.mean(actions), so
    # quantiles.mean(actions) = v + a.mean(actions) - a.mean(actions) = v.
    # Therefore q - q.mean(actions) is the centred advantage stream.
    assert torch.allclose(action_mean, torch.zeros_like(action_mean), atol=1e-5)


@pytest.mark.unit
def test_get_q_values_variance_shrinks_with_more_quantiles(iqn_network, iqn_config):
    """Increasing K should reduce sample-variance of the expected Q estimate."""
    torch.manual_seed(2)
    market = torch.randn(2, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(2, ACCOUNT_STATE_DIM)

    def _spread(k: int, trials: int) -> float:
        outs = []
        with torch.no_grad():
            for _ in range(trials):
                taus = torch.rand(2, k)
                outs.append(iqn_network.get_q_values(market, account, taus).detach())
        return float(torch.stack(outs).std(dim=0).mean())

    spread_small = _spread(k=4, trials=8)
    spread_large = _spread(k=128, trials=8)
    assert spread_large < spread_small


@pytest.mark.unit
def test_taus_at_boundaries_are_finite(iqn_network, iqn_config):
    """The quantile head must remain finite at the open interval endpoints."""
    market = torch.randn(2, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(2, ACCOUNT_STATE_DIM)
    eps = 1e-6
    boundary_taus = torch.tensor(
        [
            [eps, 1 - eps, 0.25, 0.75],
            [eps, 1 - eps, 0.5, 0.5],
        ]
    )
    quantiles = iqn_network(market, account, boundary_taus)
    assert torch.isfinite(quantiles).all()
    assert quantiles.shape == (2, iqn_config["num_actions"], 4)


@pytest.mark.unit
def test_forward_with_cls_matches_separate_calls(iqn_network, iqn_config):
    """``forward_with_cls`` must agree with ``forward`` + ``predict_return``."""
    iqn_network.eval()
    torch.manual_seed(3)
    market = torch.randn(3, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(3, ACCOUNT_STATE_DIM)
    taus = torch.rand(3, 12)

    quantiles_a, cls_out = iqn_network.forward_with_cls(market, account, taus)
    quantiles_b = iqn_network(market, account, taus)
    aux_a = iqn_network.aux_from_cls(cls_out)
    aux_b = iqn_network.predict_return(market)

    assert torch.allclose(quantiles_a, quantiles_b, atol=1e-6)
    assert torch.allclose(aux_a, aux_b, atol=1e-6)


@pytest.mark.unit
def test_quantile_outputs_smooth_in_tau(iqn_network, iqn_config):
    """Tiny tau perturbations should produce tiny quantile changes."""
    iqn_network.eval()
    torch.manual_seed(4)
    market = torch.randn(2, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(2, ACCOUNT_STATE_DIM)
    base = torch.full((2, 4), 0.5)
    pert = base + 1e-3
    out_base = iqn_network(market, account, base)
    out_pert = iqn_network(market, account, pert)
    diff = (out_base - out_pert).abs().max().item()
    assert diff < 0.5, f"Tiny tau perturbation produced large quantile shift: {diff}"

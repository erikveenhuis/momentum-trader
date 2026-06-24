"""Tests for the greedy (target-net argmax) IQN bootstrap path."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from momentum_agent.agent import RainbowDQNAgent
from momentum_agent.model import RainbowNetwork


@pytest.fixture
def iqn_config():
    return {
        "seed": 0,
        "window_size": 6,
        "n_features": 12,
        "hidden_dim": 16,
        "num_actions": 4,
        "n_quantiles_online": 8,
        "n_quantiles_target": 8,
        "n_quantiles_policy": 4,
        "quantile_embedding_dim": 8,
        "huber_kappa": 1.0,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "transformer_dropout": 0.0,
    }


def _make_dummy_agent(
    cfg: dict,
    *,
    munchausen_alpha: float = 0.0,
    munchausen_entropy_tau: float = 0.03,
    munchausen_log_pi_clip: float = -1.0,
    iqn_bootstrap_mode: str = "greedy",
):
    device = torch.device("cpu")
    online = RainbowNetwork(config=cfg, device=device)
    target = RainbowNetwork(config=cfg, device=device)
    target.load_state_dict(online.state_dict())
    online.eval()
    target.eval()

    class _DummyAgent:
        pass

    agent = _DummyAgent()
    agent.device = device
    agent.gamma = 0.99
    agent.n_steps = 3
    agent.num_actions = cfg["num_actions"]
    agent.n_quantiles_target = cfg["n_quantiles_target"]
    agent.debug_mode = False
    agent.network = online
    agent.target_network = target
    agent.munchausen_alpha = munchausen_alpha
    agent.munchausen_entropy_tau = munchausen_entropy_tau
    agent.munchausen_log_pi_clip = munchausen_log_pi_clip
    agent.iqn_bootstrap_mode = iqn_bootstrap_mode
    return agent


@pytest.mark.unit
def test_greedy_bootstrap_selects_argmax_quantiles(iqn_config):
    """Greedy mode bootstraps from Z(s', a*, tau) where a* = argmax mean Q."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="greedy")
    batch = 2
    k = cfg["n_quantiles_target"]
    actions_count = cfg["num_actions"]

    # Action 2 has the highest mean quantile (3.0); action 0 is lowest (0.0).
    quantiles = torch.zeros(batch, actions_count, k)
    for a in range(actions_count):
        quantiles[:, a, :] = float(a)
    # Make action 2 clearly best.
    quantiles[:, 2, :] = 3.0

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent.target_network, "get_quantiles", return_value=quantiles):
        target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    expected_bootstrap = (agent.gamma**agent.n_steps) * 3.0
    assert target_q_z.shape == (batch, k)
    assert torch.allclose(target_q_z, torch.full_like(target_q_z, expected_bootstrap), atol=1e-6)


@pytest.mark.unit
def test_greedy_bootstrap_terminal_masks_bootstrap(iqn_config):
    """Terminal transitions yield target_q_z == rewards only (no bootstrap)."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="greedy")
    batch = 2
    k = cfg["n_quantiles_target"]

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.full((batch, 1), 0.25)
    dones = torch.ones(batch, 1)

    quantiles = torch.ones(batch, cfg["num_actions"], k) * 100.0
    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent.target_network, "get_quantiles", return_value=quantiles):
        target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.allclose(target_q_z, torch.full_like(target_q_z, 0.25), atol=1e-6)


@pytest.mark.unit
def test_greedy_differs_from_soft_when_actions_compete(iqn_config):
    """Greedy and soft targets diverge when argmax and softmax mix disagree."""
    cfg = dict(iqn_config)
    agent_greedy = _make_dummy_agent(cfg, iqn_bootstrap_mode="greedy")
    agent_soft = _make_dummy_agent(cfg, iqn_bootstrap_mode="soft", munchausen_entropy_tau=0.5)
    agent_soft.target_network.load_state_dict(agent_greedy.target_network.state_dict())

    batch = 2
    k = cfg["n_quantiles_target"]
    quantiles = torch.zeros(batch, cfg["num_actions"], k)
    quantiles[:, 0, :] = 1.0
    quantiles[:, 1, :] = 2.0
    quantiles[:, 2, :] = 10.0  # argmax
    quantiles[:, 3, :] = 9.5  # close second -> softmax still mixes

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent_greedy.target_network, "get_quantiles", return_value=quantiles):
        with patch.object(agent_soft.target_network, "get_quantiles", return_value=quantiles):
            torch.manual_seed(42)
            t_greedy = bound(agent_greedy, market, account, next_market, next_account, actions, rewards, dones)
            torch.manual_seed(42)
            t_soft = bound(agent_soft, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.isfinite(t_greedy).all()
    assert torch.isfinite(t_soft).all()
    assert not torch.allclose(t_greedy, t_soft)


@pytest.mark.unit
def test_greedy_bootstrap_uses_only_target_network(iqn_config):
    """Greedy mode must not invoke the online network (double mode does)."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="greedy")
    batch = 2

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("Online network invoked during target computation")

    with patch.object(agent.network, "get_quantiles", side_effect=_raise_if_called):
        with patch.object(agent.network, "get_q_values", side_effect=_raise_if_called):
            bound = RainbowDQNAgent._compute_iqn_target_quantiles
            target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.isfinite(target_q_z).all()

"""Tests for the Double-DQN (online argmax, target quantiles) IQN bootstrap path."""

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
    iqn_bootstrap_mode: str = "double",
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
    agent.n_quantiles_policy = cfg["n_quantiles_policy"]
    agent.debug_mode = False
    agent.network = online
    agent.target_network = target
    agent.munchausen_alpha = munchausen_alpha
    agent.munchausen_entropy_tau = munchausen_entropy_tau
    agent.munchausen_log_pi_clip = munchausen_log_pi_clip
    agent.iqn_bootstrap_mode = iqn_bootstrap_mode
    return agent


@pytest.mark.unit
def test_double_selects_online_argmax_evaluates_target_quantiles(iqn_config):
    """Double mode uses online argmax but gathers quantiles from the target net."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="double")
    batch = 2
    k = cfg["n_quantiles_target"]
    actions_count = cfg["num_actions"]

    # Target net: action 2 has highest mean quantile (3.0).
    target_quantiles = torch.zeros(batch, actions_count, k)
    for a in range(actions_count):
        target_quantiles[:, a, :] = float(a)
    target_quantiles[:, 2, :] = 3.0

    # Online net: action 0 is best (greedy on target would pick action 2).
    online_q = torch.zeros(batch, actions_count)
    online_q[:, 0] = 10.0
    online_q[:, 2] = 1.0

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent.target_network, "get_quantiles", return_value=target_quantiles):
        with patch.object(agent.network, "get_q_values", return_value=online_q):
            target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    # Action 0 quantiles on target net are all 0.0.
    expected_bootstrap = (agent.gamma**agent.n_steps) * 0.0
    assert target_q_z.shape == (batch, k)
    assert torch.allclose(target_q_z, torch.full_like(target_q_z, expected_bootstrap), atol=1e-6)


@pytest.mark.unit
def test_double_differs_from_greedy_when_nets_disagree(iqn_config):
    """Double and greedy targets diverge when online and target argmax disagree."""
    cfg = dict(iqn_config)
    agent_double = _make_dummy_agent(cfg, iqn_bootstrap_mode="double")
    agent_greedy = _make_dummy_agent(cfg, iqn_bootstrap_mode="greedy")
    agent_greedy.target_network.load_state_dict(agent_double.target_network.state_dict())

    batch = 2
    k = cfg["n_quantiles_target"]
    quantiles = torch.zeros(batch, cfg["num_actions"], k)
    quantiles[:, 0, :] = 1.0
    quantiles[:, 1, :] = 2.0
    quantiles[:, 2, :] = 10.0  # target argmax
    quantiles[:, 3, :] = 9.5

    online_q = torch.zeros(batch, cfg["num_actions"])
    online_q[:, 0] = 100.0  # online argmax

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent_double.target_network, "get_quantiles", return_value=quantiles):
        with patch.object(agent_greedy.target_network, "get_quantiles", return_value=quantiles):
            with patch.object(agent_double.network, "get_q_values", return_value=online_q):
                torch.manual_seed(42)
                t_double = bound(agent_double, market, account, next_market, next_account, actions, rewards, dones)
                torch.manual_seed(42)
                t_greedy = bound(agent_greedy, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.isfinite(t_double).all()
    assert torch.isfinite(t_greedy).all()
    assert not torch.allclose(t_double, t_greedy)


@pytest.mark.unit
def test_double_terminal_masks_bootstrap(iqn_config):
    """Terminal transitions yield target_q_z == rewards only (no bootstrap)."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="double")
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
    online_q = torch.ones(batch, cfg["num_actions"]) * 50.0
    bound = RainbowDQNAgent._compute_iqn_target_quantiles
    with patch.object(agent.target_network, "get_quantiles", return_value=quantiles):
        with patch.object(agent.network, "get_q_values", return_value=online_q):
            target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.allclose(target_q_z, torch.full_like(target_q_z, 0.25), atol=1e-6)


@pytest.mark.unit
def test_double_invokes_online_get_q_values_only(iqn_config):
    """Double mode calls online ``get_q_values`` but not online ``get_quantiles``."""
    cfg = dict(iqn_config)
    agent = _make_dummy_agent(cfg, iqn_bootstrap_mode="double")
    batch = 2
    k = cfg["n_quantiles_target"]

    market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    account = torch.zeros(batch, 5)
    next_market = torch.zeros(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.zeros(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.zeros(batch, 1)

    quantiles = torch.zeros(batch, cfg["num_actions"], k)
    online_q = torch.zeros(batch, cfg["num_actions"])
    online_q[:, 1] = 1.0

    def _raise_if_online_quantiles(*_args, **_kwargs):
        raise AssertionError("Online get_quantiles must not be called in double bootstrap")

    with patch.object(agent.target_network, "get_quantiles", return_value=quantiles):
        with patch.object(agent.network, "get_q_values", return_value=online_q):
            with patch.object(agent.network, "get_quantiles", side_effect=_raise_if_online_quantiles):
                bound = RainbowDQNAgent._compute_iqn_target_quantiles
                target_q_z = bound(agent, market, account, next_market, next_account, actions, rewards, dones)

    assert torch.isfinite(target_q_z).all()

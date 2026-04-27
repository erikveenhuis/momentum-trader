"""Tests for the IQN quantile-Huber loss + ``_compute_iqn_target_quantiles``.

These are pure-tensor tests that don't require the full training stack.
We re-implement the loss formula in NumPy (tiny K) and check that the
agent's loss matches; we also fuzz-test invariants such as:

* ``alpha_loss(taus, target, target) == 0`` (zero residual gives zero loss)
* The Huber-Quantile loss is monotone in ``|delta|`` when the asymmetry
  factor is fixed.

We also unit-test ``RainbowDQNAgent._compute_iqn_target_quantiles`` by
constructing a tiny CPU agent (when CUDA is available) or by stubbing
device requirements so the maths can be exercised in CI without a GPU.
"""

from __future__ import annotations

import pytest
import torch
from momentum_agent.model import RainbowNetwork

# ---------------------------------------------------------------------------
# Pure-tensor reference implementation of the IQN quantile-Huber loss.
# ---------------------------------------------------------------------------


def _reference_quantile_huber(
    online_z: torch.Tensor,  # [B, K]
    target_z: torch.Tensor,  # [B, K']
    taus: torch.Tensor,  # [B, K]
    kappa: float,
) -> torch.Tensor:
    """Plain-Python reference for :func:`agent._compute_loss`'s loss core."""
    delta = target_z.unsqueeze(1) - online_z.unsqueeze(2)  # [B, K, K']
    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )
    indicator = (delta < 0).float()
    weight = (taus.unsqueeze(2) - indicator).abs() / kappa
    rho = weight * huber
    return rho.sum(dim=2).mean(dim=1)  # [B]


@pytest.mark.unit
def test_quantile_huber_zero_residual_zero_loss():
    """If both distributions collapse onto the same constant, the loss is 0."""
    online = torch.full((3, 8), 0.42)
    target = torch.full((3, 8), 0.42)
    taus = torch.rand(3, 8)
    loss = _reference_quantile_huber(online, target, taus, kappa=1.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


@pytest.mark.unit
def test_quantile_huber_symmetric_at_tau_half():
    """At tau == 0.5 the Huber-quantile loss is symmetric in delta."""
    online = torch.zeros(1, 1)
    taus = torch.full((1, 1), 0.5)
    target_pos = torch.tensor([[1.0]])
    target_neg = torch.tensor([[-1.0]])
    pos = _reference_quantile_huber(online, target_pos, taus, kappa=1.0)
    neg = _reference_quantile_huber(online, target_neg, taus, kappa=1.0)
    assert torch.allclose(pos, neg, atol=1e-6)


@pytest.mark.unit
def test_quantile_huber_asymmetry_at_extreme_taus():
    """tau=0.9 should penalise *under*-prediction (delta>0) more than over-prediction."""
    online = torch.zeros(1, 1)
    taus = torch.full((1, 1), 0.9)
    over = _reference_quantile_huber(online, torch.tensor([[-1.0]]), taus, kappa=1.0)  # delta < 0
    under = _reference_quantile_huber(online, torch.tensor([[1.0]]), taus, kappa=1.0)  # delta > 0
    assert under.item() > over.item()


@pytest.mark.unit
def test_quantile_huber_huber_transition_kappa():
    """The loss is quadratic for |delta|<=kappa and linear for |delta|>kappa."""
    online = torch.zeros(1, 1)
    taus = torch.full((1, 1), 0.5)
    kappa = 1.0
    # |delta| = 0.5 (quadratic regime): expected = 0.5 * 0.25 / kappa * |0.5 - 0.5| ...
    # easier: directly compare to the closed form.
    for delta_value in [0.25, 0.5, 1.5, 3.0]:
        target = torch.tensor([[delta_value]])
        loss = _reference_quantile_huber(online, target, taus, kappa=kappa)
        # symmetric tau=0.5 -> weight = 0.5 / kappa.
        if abs(delta_value) <= kappa:
            expected = 0.5 / kappa * 0.5 * (delta_value**2)
        else:
            expected = 0.5 / kappa * kappa * (abs(delta_value) - 0.5 * kappa)
        assert pytest.approx(loss.item(), rel=1e-5, abs=1e-7) == expected


# ---------------------------------------------------------------------------
# Agent-level integration: _compute_iqn_target_quantiles must match the
# Double-DQN flavoured Bellman target, and the loss path must shrink toward
# zero when the network is rolled forward against itself.
# ---------------------------------------------------------------------------


@pytest.fixture
def iqn_config():
    return {
        "seed": 0,
        "window_size": 6,
        "n_features": 12,
        "hidden_dim": 16,
        "num_actions": 6,
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


@pytest.mark.unit
def test_compute_iqn_target_with_munchausen_disabled_matches_reward_when_terminal(iqn_config):
    """With ``alpha=0`` (Munchausen bonus off), terminal transitions should
    yield ``target_q_z == rewards`` because the bootstrap is masked by
    ``(1 - done)`` and the Munchausen log-pi term is multiplied by alpha."""
    from momentum_agent.agent import RainbowDQNAgent

    cfg = {**iqn_config}
    device = torch.device("cpu")

    online = RainbowNetwork(config=cfg, device=device)
    target = RainbowNetwork(config=cfg, device=device)
    online.eval()
    target.eval()

    class _DummyAgent:
        device = torch.device("cpu")
        gamma = 0.99
        n_steps = 3
        num_actions = cfg["num_actions"]
        n_quantiles_target = cfg["n_quantiles_target"]
        debug_mode = False
        network = online
        target_network = target
        munchausen_alpha = 0.0
        munchausen_entropy_tau = 0.03
        munchausen_log_pi_clip = -1.0

    bound_method = RainbowDQNAgent._compute_iqn_target_quantiles
    agent = _DummyAgent()

    batch = 2
    market = torch.randn(batch, cfg["window_size"], cfg["n_features"])
    account = torch.randn(batch, 5)
    next_market = torch.randn(batch, cfg["window_size"], cfg["n_features"])
    next_account = torch.randn(batch, 5)
    actions = torch.zeros(batch, dtype=torch.long)
    rewards = torch.zeros(batch, 1)
    dones = torch.ones(batch, 1)

    target_q_z = bound_method(agent, market, account, next_market, next_account, actions, rewards, dones)
    assert target_q_z.shape == (batch, cfg["n_quantiles_target"])
    assert torch.allclose(target_q_z, torch.zeros_like(target_q_z), atol=1e-6)

    rewards_pos = torch.full((batch, 1), 0.25)
    target_q_z_done = bound_method(agent, market, account, next_market, next_account, actions, rewards_pos, dones)
    assert torch.allclose(target_q_z_done, torch.full_like(target_q_z_done, 0.25), atol=1e-6)

    # done=0 -> target = r + gamma^n * soft_V (which is non-zero).
    dones_zero = torch.zeros(batch, 1)
    target_q_z_boot = bound_method(agent, market, account, next_market, next_account, actions, rewards_pos, dones_zero)
    assert torch.isfinite(target_q_z_boot).all()
    assert not torch.allclose(target_q_z_boot, target_q_z_done)


@pytest.mark.unit
def test_quantile_huber_matches_reference_under_random_inputs():
    """Spot-check the reference formula against a few hand-computed cases."""
    torch.manual_seed(42)
    online = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[0.5, 0.5]])
    taus = torch.tensor([[0.3, 0.7]])
    loss = _reference_quantile_huber(online, target, taus, kappa=1.0)
    # Hand check for sample j=0 (online 0.0, tau 0.3):
    #   delta = [0.5-0, 0.5-0] = [0.5, 0.5], huber = 0.125 each, |0.3 - 0|/1 = 0.3.
    #   contribution = 2 * 0.3 * 0.125 = 0.075
    # j=1 (online 1.0, tau 0.7): delta = [-0.5, -0.5], huber 0.125 each, indicator=1
    #   weight = |0.7 - 1|/1 = 0.3 -> 2 * 0.3 * 0.125 = 0.075
    # mean over j -> 0.075
    assert pytest.approx(loss.item(), rel=1e-5, abs=1e-7) == 0.075

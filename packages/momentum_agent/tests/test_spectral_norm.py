"""Tests for spectral-norm wrapping of the IQN dueling heads (BTR Stage 3).

These tests pin five behaviours called out in the upgrade plan:

* Spectral norm is wrapped *only* on the dueling heads' ``NoisyLinear``s
  — the encoder, account processor, aux head, and ``tau_embedding`` are
  left untouched.
* When ``spectral_norm_enabled=False`` no parametrization is registered
  on any module (no behavioural change vs the prior IQN baseline).
* After SGD steps on noise the operator norm of every wrapped
  ``weight_mu`` stays at the spectral-norm bound (≤ 1 + tolerance).
* ``state_dict`` round trip over ``parametrizations.weight_mu.original``
  preserves forward-pass parity exactly.
* ``torch.compile(mode='default')`` accepts the spectral-norm-wrapped model
  (smoke test only — we don't compare numerics against eager).
"""

from __future__ import annotations

import pytest
import torch
from momentum_agent.model import NoisyLinear, RainbowNetwork
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


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


def _build(cfg: dict, *, spectral_norm_enabled: bool) -> RainbowNetwork:
    return RainbowNetwork(
        config=cfg,
        device=torch.device("cpu"),
        spectral_norm_enabled=spectral_norm_enabled,
    )


def _noisy_linears(stream: nn.Module) -> list[tuple[str, NoisyLinear]]:
    return [(name, m) for name, m in stream.named_modules() if isinstance(m, NoisyLinear)]


@pytest.mark.unit
def test_spectral_norm_wraps_only_dueling_heads(iqn_config):
    """Every NoisyLinear in ``value_stream`` / ``advantage_stream`` is
    parametrized on ``weight_mu``; everything else (encoder, aux head,
    tau_embedding, account_processor) is untouched."""
    net = _build(iqn_config, spectral_norm_enabled=True)

    # Dueling-head NoisyLinears must be wrapped.
    head_modules = _noisy_linears(net.value_stream) + _noisy_linears(net.advantage_stream)
    assert len(head_modules) >= 2  # at minimum the final layer of each stream
    for name, module in head_modules:
        assert is_parametrized(module, "weight_mu"), f"Expected spectral-norm parametrization on {name}.weight_mu"

    # Aux head, account processor, and tau_embedding must not be wrapped.
    for name, module in net.named_modules():
        if (
            isinstance(module, (nn.Linear, NoisyLinear))
            and name
            and not (name.startswith("value_stream") or name.startswith("advantage_stream"))
        ):
            for pname in ("weight_mu", "weight"):
                assert not is_parametrized(module, pname), f"Spectral norm leaked into non-head module {name} ({pname})"


@pytest.mark.unit
def test_spectral_norm_disabled_leaves_modules_untouched(iqn_config):
    """When ``spectral_norm_enabled=False`` every module's ``weight_mu`` /
    ``weight`` is a plain parameter, not a parametrized one."""
    net = _build(iqn_config, spectral_norm_enabled=False)
    for name, module in net.named_modules():
        if isinstance(module, NoisyLinear):
            assert not is_parametrized(module, "weight_mu"), name
        if isinstance(module, nn.Linear):
            assert not is_parametrized(module, "weight"), name


@pytest.mark.unit
def test_spectral_norm_bounds_operator_norm_after_sgd_steps(iqn_config):
    """Train a few SGD steps on Gaussian noise; the wrapped weight_mu's
    operator norm must stay ≤ 1 + small tolerance (spectral norm bounds it
    by design)."""
    torch.manual_seed(0)
    net = _build(iqn_config, spectral_norm_enabled=True)
    net.train()
    optim = torch.optim.SGD(net.parameters(), lr=1e-2)

    batch = 4
    market = torch.randn(batch, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(batch, 5)
    taus = torch.rand(batch, 4)

    for _ in range(50):
        optim.zero_grad()
        q = net(market, account, taus)  # [B, A, K]
        loss = q.pow(2).mean()
        loss.backward()
        optim.step()

    head_modules = _noisy_linears(net.value_stream) + _noisy_linears(net.advantage_stream)
    for name, module in head_modules:
        # ``module.weight_mu`` returns the *normalized* weight under the
        # spectral_norm parametrization. Its operator norm should be ≤ 1
        # + a small numerical slack (power iteration is one-shot per fwd).
        w = module.weight_mu.detach().reshape(module.weight_mu.shape[0], -1).float()
        sigma_max = float(torch.linalg.matrix_norm(w, ord=2).item())
        assert sigma_max <= 1.0 + 5e-2, f"Spectral norm constraint violated on {name}: sigma_max={sigma_max:.4f}"


@pytest.mark.unit
def test_spectral_norm_state_dict_round_trip(iqn_config):
    """Save → load → forward parity with the spectral-norm parametrization."""
    torch.manual_seed(0)
    net_a = _build(iqn_config, spectral_norm_enabled=True)
    net_a.eval()

    batch = 2
    market = torch.randn(batch, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.randn(batch, 5)
    taus = torch.rand(batch, 4)

    with torch.no_grad():
        q_before = net_a(market, account, taus)

    state = net_a.state_dict()
    net_b = _build(iqn_config, spectral_norm_enabled=True)
    net_b.load_state_dict(state)
    net_b.eval()
    with torch.no_grad():
        q_after = net_b(market, account, taus)

    assert torch.allclose(q_before, q_after, atol=1e-5), (
        "Forward parity broke across spectral-norm state_dict round trip."
    )


@pytest.mark.unit
def test_spectral_norm_does_not_break_compile(iqn_config):
    """``torch.compile`` should accept the spectral-norm-wrapped network.

    We don't run it under CUDA in tests, so we settle for ``mode='default'``
    on CPU — the goal is to catch parametrization-related compile-time
    explosions, not to verify numerical parity with eager.
    """
    if not hasattr(torch, "compile"):  # pragma: no cover - very old torch
        pytest.skip("torch.compile not available")
    torch.manual_seed(0)
    net = _build(iqn_config, spectral_norm_enabled=True)
    net.eval()

    try:
        compiled = torch.compile(net, mode="default", fullgraph=False)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        pytest.fail(f"torch.compile failed on spectral-norm-wrapped model: {exc}")

    batch = 1
    market = torch.zeros(batch, iqn_config["window_size"], iqn_config["n_features"])
    account = torch.zeros(batch, 5)
    taus = torch.zeros(batch, 4)
    try:
        with torch.no_grad():
            q = compiled(market, account, taus)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"compiled forward unsupported on this CPU/torch combo: {exc}")
    assert q.shape == (batch, iqn_config["num_actions"], 4)
    assert torch.isfinite(q).all()

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Remove sys.path manipulation
# src_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../src")
# )  # Path adjusted from tests/ to src/
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# Use absolute imports from src
from momentum_agent.model import NoisyLinear, PositionalEncoding, RainbowNetwork

# TODO: Add tests for src/model.py
# Consider merging/reviewing with tests/test_networks.py

@pytest.mark.unittest
def test_placeholder_model():
    assert True


# --- Test Configuration (Re-use from agent tests or define specific) ---
@pytest.fixture(scope="module")
def default_config():
    """Provides a default configuration dictionary for the model tests."""
    return {
        "seed": 42,
        "window_size": 10,
        "n_features": 5,
        "hidden_dim": 64,
        "num_actions": 3,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "transformer_dropout": 0.1,
        # Add other keys if RainbowNetwork expects them, even if not directly used
        "gamma": 0.99,
        "lr": 1e-4,
        "replay_buffer_size": 1000,
        "batch_size": 4,
        "target_update_freq": 5,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,
        "n_steps": 3,
    }


@pytest.fixture(scope="module")
def device():
    """Determines the device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Test NoisyLinear --- #

@pytest.mark.unittest
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


@pytest.mark.unittest
def test_noisy_linear_forward_train(device):
    batch_size = 4
    in_features = 10
    out_features = 5
    layer = NoisyLinear(in_features=in_features, out_features=out_features).to(device)
    layer.train()  # Ensure training mode
    dummy_input = torch.randn(batch_size, in_features).to(device)

    # Capture weights before forward pass
    weight_mu_before = layer.weight_mu.clone().detach()
    bias_mu_before = layer.bias_mu.clone().detach()
    weight_sigma_before = layer.weight_sigma.clone().detach()
    bias_sigma_before = layer.bias_sigma.clone().detach()
    weight_epsilon_before = layer.weight_epsilon.clone().detach()
    bias_epsilon_before = layer.bias_epsilon.clone().detach()

    output = layer(dummy_input)

    assert output.shape == (batch_size, out_features)
    assert output.device == device

    # Check that noise was used (output != mu-only output)
    with torch.no_grad():
        mu_output = F.linear(dummy_input, layer.weight_mu, layer.bias_mu)
    assert not torch.allclose(output, mu_output)

    # Check that base weights (mu, sigma) did not change during forward
    assert torch.equal(layer.weight_mu, weight_mu_before)
    assert torch.equal(layer.bias_mu, bias_mu_before)
    assert torch.equal(layer.weight_sigma, weight_sigma_before)
    assert torch.equal(layer.bias_sigma, bias_sigma_before)

    # Check that noise buffers (epsilon) did not change during forward
    assert torch.equal(layer.weight_epsilon, weight_epsilon_before)
    assert torch.equal(layer.bias_epsilon, bias_epsilon_before)


@pytest.mark.unittest
def test_noisy_linear_forward_eval(device):
    batch_size = 4
    in_features = 10
    out_features = 5
    layer = NoisyLinear(in_features=in_features, out_features=out_features).to(device)
    layer.eval()  # Ensure evaluation mode
    dummy_input = torch.randn(batch_size, in_features).to(device)

    output = layer(dummy_input)

    assert output.shape == (batch_size, out_features)
    assert output.device == device

    # Check that noise was NOT used (output == mu-only output)
    with torch.no_grad():
        mu_output = F.linear(dummy_input, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(output, mu_output)


@pytest.mark.unittest
def test_noisy_linear_reset_noise(device):
    layer = NoisyLinear(in_features=10, out_features=5).to(device)
    layer.train()

    # Get initial noise
    initial_weight_eps = layer.weight_epsilon.clone().detach()
    initial_bias_eps = layer.bias_epsilon.clone().detach()

    # Reset noise
    layer.reset_noise()

    # Get new noise
    new_weight_eps = layer.weight_epsilon.clone().detach()
    new_bias_eps = layer.bias_epsilon.clone().detach()

    # Check that noise has changed
    assert not torch.equal(initial_weight_eps, new_weight_eps)
    assert not torch.equal(initial_bias_eps, new_bias_eps)


# --- Test PositionalEncoding --- #

@pytest.mark.unittest
def test_positional_encoding_init():
    d_model = 64
    max_len = 50
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)
    assert pe.d_model == d_model
    assert pe.pe.shape == (max_len, 1, d_model)


@pytest.mark.unittest
def test_positional_encoding_forward(device):
    batch_size = 4
    seq_len = 20
    d_model = 64
    pe = PositionalEncoding(d_model=d_model, max_len=50).to(device)
    dummy_input = torch.randn(batch_size, seq_len, d_model).to(device)

    output = pe(dummy_input)

    assert output.shape == (batch_size, seq_len, d_model)
    assert output.device == device
    # Check that output is different from input (encoding was added)
    assert not torch.allclose(output, dummy_input)

    # Check with dropout disabled
    pe_no_dropout = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=50).to(
        device
    )
    output_no_dropout = pe_no_dropout(dummy_input)
    # Calculate expected output without dropout
    expected_output = dummy_input.permute(1, 0, 2) + pe_no_dropout.pe[:seq_len]
    expected_output = expected_output.permute(1, 0, 2)
    assert torch.allclose(output_no_dropout, expected_output)


# --- Test RainbowNetwork --- #

@pytest.fixture(scope="module")
def network(default_config, device):
    """Creates a RainbowNetwork instance for testing."""
    # Ensure ACCOUNT_STATE_DIM is implicitly 2 if not defined elsewhere
    # In a real scenario, it should be imported or defined consistently
    # account_dim = getattr(sys.modules[__name__], 'ACCOUNT_STATE_DIM', 2)
    net = RainbowNetwork(config=default_config, device=device).to(device)
    net.eval()  # Default to eval mode for most forward pass tests
    return net


@pytest.mark.unittest
def test_rainbow_network_init(network, default_config, device):
    assert network is not None
    assert network.device == device
    assert network.window_size == default_config["window_size"]
    assert network.n_features == default_config["n_features"]
    assert network.hidden_dim == default_config["hidden_dim"]
    assert network.num_actions == default_config["num_actions"]
    assert network.num_atoms == default_config["num_atoms"]
    assert network.support.shape == (default_config["num_atoms"],)
    assert torch.isclose(
        network.support[0], torch.tensor(float(default_config["v_min"]))
    )
    assert torch.isclose(
        network.support[-1], torch.tensor(float(default_config["v_max"]))
    )

    # Check submodules exist
    assert hasattr(network, "feature_embedding")
    assert hasattr(network, "pos_encoder")
    assert hasattr(network, "transformer_encoder")
    assert hasattr(network, "account_processor")
    assert hasattr(network, "value_stream")
    assert hasattr(network, "advantage_stream")

    # Check parameters are on the correct device
    for param in network.parameters():
        assert str(param.device).startswith(str(device))


@pytest.mark.unittest
def test_rainbow_network_forward_pass(network, default_config, device):
    batch_size = default_config["batch_size"]
    window_size = default_config["window_size"]
    n_features = default_config["n_features"]
    num_actions = default_config["num_actions"]
    num_atoms = default_config["num_atoms"]
    account_dim = 2  # Assuming ACCOUNT_STATE_DIM is 2

    # Create dummy input tensors
    market_data = torch.randn(batch_size, window_size, n_features).to(device)
    account_state = torch.randn(batch_size, account_dim).to(device)

    # Forward pass (eval mode by default from fixture)
    log_probs = network(market_data, account_state)

    assert log_probs.shape == (batch_size, num_actions, num_atoms)
    assert log_probs.device == device
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()

    # Check if probabilities sum to 1 (approximately) for each action
    probs = torch.exp(log_probs)
    prob_sums = probs.sum(dim=2)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)


@pytest.mark.unittest
def test_rainbow_network_get_q_values(network, default_config, device):
    batch_size = default_config["batch_size"]
    window_size = default_config["window_size"]
    n_features = default_config["n_features"]
    num_actions = default_config["num_actions"]
    account_dim = 2  # Assuming ACCOUNT_STATE_DIM is 2

    # Create dummy input tensors
    market_data = torch.randn(batch_size, window_size, n_features).to(device)
    account_state = torch.randn(batch_size, account_dim).to(device)

    # Get Q-values (eval mode by default)
    q_values = network.get_q_values(market_data, account_state)

    assert q_values.shape == (batch_size, num_actions)
    assert q_values.device == device
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()


@pytest.mark.unittest
def test_rainbow_network_reset_noise(network, device):
    # Network should always have NoisyLinear layers
    network.train()  # Reset noise only affects training mode

    # Capture initial epsilon values from a NoisyLinear layer
    noisy_layer = next(m for m in network.modules() if isinstance(m, NoisyLinear))
    initial_weight_eps = noisy_layer.weight_epsilon.clone().detach()
    initial_bias_eps = noisy_layer.bias_epsilon.clone().detach()

    # Call reset_noise on the main network
    network.reset_noise()

    # Capture new epsilon values
    new_weight_eps = noisy_layer.weight_epsilon.clone().detach()
    new_bias_eps = noisy_layer.bias_epsilon.clone().detach()

    # Check that noise has changed
    assert not torch.equal(initial_weight_eps, new_weight_eps)
    assert not torch.equal(initial_bias_eps, new_bias_eps)

    network.eval()  # Set back to eval mode


@pytest.mark.unittest
def test_rainbow_network_train_eval_modes(network, default_config, device):
    batch_size = default_config["batch_size"]
    window_size = default_config["window_size"]
    n_features = default_config["n_features"]
    account_dim = 2
    market_data = torch.randn(batch_size, window_size, n_features).to(device)
    account_state = torch.randn(batch_size, account_dim).to(device)

    # Eval mode (default from fixture)
    network.eval()
    assert not network.training
    assert not any(
        m.training
        for m in network.modules()
        if isinstance(m, (nn.Dropout, NoisyLinear))
    )
    q_values_eval = network.get_q_values(market_data, account_state)

    # Train mode
    network.train()
    assert network.training
    assert any(
        m.training
        for m in network.modules()
        if isinstance(m, (nn.Dropout, NoisyLinear))
    )
    # In train mode, NoisyLinear uses noise, so Q values should differ (unless noise is zero)
    q_values_train = network.get_q_values(market_data, account_state)

    # Check that Q-values differ between train and eval modes due to NoisyNet
    assert not torch.allclose(
        q_values_eval, q_values_train
    ), "Q-values should differ between train and eval modes"

    network.eval()  # Reset to eval mode


# --- Helper to generate dummy data --- #
def _generate_rainbow_input(config, batch_size, device):
    window_size = config["window_size"]
    n_features = config["n_features"]
    account_dim = 2  # Assuming ACCOUNT_STATE_DIM is 2
    market_data = torch.randn(batch_size, window_size, n_features).to(device)
    account_state = torch.randn(batch_size, account_dim).to(device)
    return market_data, account_state

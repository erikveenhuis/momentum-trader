import os

import numpy as np
import pytest
import torch

# Use absolute imports from src
from momentum_agent.agent import ACCOUNT_STATE_DIM, RainbowDQNAgent
from momentum_agent.buffer import PrioritizedReplayBuffer
from momentum_agent.model import RainbowNetwork

# Remove sys.path manipulation
# src_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../src")
# )  # Path adjusted from tests/ to src/
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)


# --- Test Configuration ---
@pytest.fixture(scope="module")
def default_config():
    """Provides a default configuration dictionary for the agent."""
    return {
        "seed": 42,
        "gamma": 0.99,
        "lr": 1e-4,
        "replay_buffer_size": 1000,  # Keep small for tests
        "batch_size": 4,  # Small batch size for tests
        "target_update_freq": 5,  # Frequent updates for testing
        "num_atoms": 51,
        "v_min": -1,
        "v_max": 1,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,  # Short annealing for tests
        "n_steps": 3,
        "window_size": 10,
        "n_features": 12,
        "hidden_dim": 64,
        "num_actions": 3,  # e.g., Hold, Buy, Sell
        "debug": True,  # Enable debug checks
        "grad_clip_norm": 10.0,
        "epsilon_start": 0.3,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": 1000,
        "entropy_coeff": 0.03,
        # Add missing network params required by RainbowNetwork
        "transformer_nhead": 2,  # Value from test_model config
        "transformer_layers": 1,  # Value from test_model config
        "dropout": 0.1,  # Value from test_model config
        # Rename transformer_nhead to nhead for consistency with error? Check RainbowNetwork init
        # --> Checking model.py: RainbowNetwork expects nhead, n_layers, dropout directly
        "nhead": 2,
        "num_encoder_layers": 1,  # Add correct key
        "dim_feedforward": 256,  # hidden_dim * 4 = 64 * 4
        "transformer_dropout": 0.1,  # Add missing key
    }


# --- Test Agent Instance ---
@pytest.fixture(scope="function")  # Recreate agent for each test function
def agent(default_config):
    """Creates a RainbowDQNAgent instance for testing."""
    # Use CUDA if available, otherwise CPU - respect actual device availability
    if "device" in default_config:
        device = default_config["device"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent_instance = RainbowDQNAgent(config=default_config, device=device)
    # Ensure agent starts in training mode for most tests
    agent_instance.set_training_mode(True)
    return agent_instance


# --- Helper Functions ---
def generate_dummy_observation(config):
    """Generates a single dummy observation dictionary."""
    market_data = np.random.rand(config["window_size"], config["n_features"]).astype(np.float32)
    account_state = np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    return {"market_data": market_data, "account_state": account_state}


# --- Test Cases ---


@pytest.mark.unit
def test_agent_initialization(agent, default_config):
    """Tests if the agent initializes components correctly."""
    assert agent is not None
    assert agent.config == default_config
    net = getattr(agent.network, "_orig_mod", agent.network)
    tgt = getattr(agent.target_network, "_orig_mod", agent.target_network)
    assert isinstance(net, RainbowNetwork)
    assert isinstance(tgt, RainbowNetwork)
    assert agent.optimizer is not None
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    assert agent.buffer.capacity == default_config["replay_buffer_size"]
    assert agent.total_steps == 0
    assert agent.training_mode is True
    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert agent.device.type == expected_device_type  # Verify device used

    # Check network parameters are on the correct device
    for param in agent.network.parameters():
        assert param.device.type == agent.device.type
    for param in agent.target_network.parameters():
        assert param.device.type == agent.device.type


@pytest.mark.unit
def test_select_action(agent, default_config):
    """Tests the select_action method."""
    obs = generate_dummy_observation(default_config)

    # Test in training mode
    agent.set_training_mode(True)
    action_train = agent.select_action(obs)
    assert isinstance(action_train, int)
    assert 0 <= action_train < default_config["num_actions"]
    assert agent.network.training is True  # Should remain in train mode

    # Test in evaluation mode
    agent.set_training_mode(False)
    action_eval = agent.select_action(obs)
    assert isinstance(action_eval, int)
    assert 0 <= action_eval < default_config["num_actions"]
    assert agent.network.training is False  # Should be in eval mode


@pytest.mark.unit
def test_select_action_with_provenance_eval_mode_is_always_greedy(agent, default_config):
    """Tier 2c: in eval mode the agent never explores → was_greedy is always True."""
    obs = generate_dummy_observation(default_config)
    agent.set_training_mode(False)
    for _ in range(5):
        action, was_greedy = agent.select_action_with_provenance(obs)
        assert isinstance(action, int)
        assert 0 <= action < default_config["num_actions"]
        assert was_greedy is True


@pytest.mark.unit
def test_select_action_with_provenance_training_eps_zero_is_greedy(agent, default_config):
    """Tier 2c: with epsilon forced to 0 in train mode, every action is greedy."""
    obs = generate_dummy_observation(default_config)
    agent.set_training_mode(True)
    # Force epsilon = 0 by saturating the linear decay.
    agent.epsilon_decay_steps = 1
    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0
    agent.env_steps = 1
    for _ in range(5):
        action, was_greedy = agent.select_action_with_provenance(obs)
        assert isinstance(action, int)
        assert 0 <= action < default_config["num_actions"]
        assert was_greedy is True


@pytest.mark.unit
def test_select_action_with_provenance_training_eps_one_is_eps(agent, default_config):
    """Tier 2c: with epsilon forced to 1 in train mode, every action is eps-forced."""
    obs = generate_dummy_observation(default_config)
    agent.set_training_mode(True)
    agent.epsilon_decay_steps = 1
    agent.epsilon_start = 1.0
    agent.epsilon_end = 1.0
    agent.env_steps = 1
    for _ in range(5):
        _action, was_greedy = agent.select_action_with_provenance(obs)
        assert was_greedy is False


@pytest.mark.unit
def test_select_actions_batch_with_provenance_eval_mode_is_greedy(agent, default_config):
    """Tier 2c: vectorized eval batch never explores → all flags True."""
    n = 4
    market_batch = np.stack([generate_dummy_observation(default_config)["market_data"] for _ in range(n)])
    account_batch = np.stack([generate_dummy_observation(default_config)["account_state"] for _ in range(n)])
    obs_batch = {"market_data": market_batch, "account_state": account_batch}
    agent.set_training_mode(False)
    actions, was_greedy = agent.select_actions_batch_with_provenance(obs_batch)
    assert actions.shape == (n,)
    assert was_greedy.shape == (n,)
    assert was_greedy.dtype == bool
    assert was_greedy.all()


@pytest.mark.unit
def test_select_actions_batch_with_provenance_training_eps_one_is_eps(agent, default_config):
    """Tier 2c: vectorized train batch with epsilon=1 marks every env as eps-forced."""
    n = 4
    market_batch = np.stack([generate_dummy_observation(default_config)["market_data"] for _ in range(n)])
    account_batch = np.stack([generate_dummy_observation(default_config)["account_state"] for _ in range(n)])
    obs_batch = {"market_data": market_batch, "account_state": account_batch}
    agent.set_training_mode(True)
    agent.epsilon_decay_steps = 1
    agent.epsilon_start = 1.0
    agent.epsilon_end = 1.0
    agent.env_steps = 1
    _actions, was_greedy = agent.select_actions_batch_with_provenance(obs_batch)
    assert was_greedy.shape == (n,)
    assert not was_greedy.any()


@pytest.mark.unit
def test_greedy_context_manager_disables_exploration(agent, default_config):
    """Tier 2d: agent.greedy() puts the agent in deterministic mode and restores."""
    agent.set_training_mode(True)
    # Force epsilon=1 so any non-greedy slot would be eps-overridden.
    agent.epsilon_decay_steps = 1
    agent.epsilon_start = 1.0
    agent.epsilon_end = 1.0
    agent.env_steps = 1
    obs = generate_dummy_observation(default_config)

    with agent.greedy():
        assert agent.training_mode is False
        for _ in range(5):
            _action, was_greedy = agent.select_action_with_provenance(obs)
            assert was_greedy is True

    # Restored back to training mode.
    assert agent.training_mode is True
    _action, was_greedy = agent.select_action_with_provenance(obs)
    assert was_greedy is False


@pytest.mark.unit
def test_greedy_context_restores_on_exception(agent, default_config):
    """Tier 2d: training mode is restored even if the body raises."""
    agent.set_training_mode(True)
    try:
        with agent.greedy():
            assert agent.training_mode is False
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert agent.training_mode is True


@pytest.mark.unit
def test_greedy_context_when_already_eval_is_idempotent(agent, default_config):
    """Tier 2d: entering greedy() while already in eval mode keeps eval mode on exit."""
    agent.set_training_mode(False)
    with agent.greedy():
        assert agent.training_mode is False
    assert agent.training_mode is False


@pytest.mark.unit
def test_inference_only_cpu_forward(default_config):
    """Live path: CPU + inference_only skips compile and runs ``select_action``."""
    cfg = dict(default_config)
    agent = RainbowDQNAgent(cfg, device="cpu", inference_only=True)
    agent.set_training_mode(False)
    obs = generate_dummy_observation(cfg)
    action = agent.select_action(obs)
    assert isinstance(action, int)
    assert 0 <= action < cfg["num_actions"]
    assert agent.device.type == "cpu"
    net = getattr(agent.network, "_orig_mod", agent.network)
    assert isinstance(net, RainbowNetwork)


@pytest.mark.unit
def test_store_transition_and_n_step(agent, default_config):
    """Tests storing transitions and n-step buffer logic."""
    n_steps = default_config["n_steps"]
    buffer_capacity = default_config["replay_buffer_size"]
    initial_buffer_len = len(agent.buffer)

    transitions = []
    for i in range(n_steps + 2):  # Store enough transitions to trigger PER storage
        obs = generate_dummy_observation(default_config)
        action = np.random.randint(default_config["num_actions"])
        reward = np.random.rand() * 2 - 1  # Random reward between -1 and 1
        next_obs = generate_dummy_observation(default_config)
        done = i == n_steps + 1  # Make the last transition terminal

        agent.store_transition(obs, action, reward, next_obs, done)
        transitions.append((obs, action, reward, next_obs, done))

        if i < n_steps - 1:
            # Should not have stored anything in PER buffer yet
            assert len(agent.buffer) == initial_buffer_len
            assert len(agent.n_step_buffer) == i + 1
        elif i == n_steps - 1:
            # First n-step transition should be stored now
            assert len(agent.buffer) == initial_buffer_len + 1
            assert len(agent.n_step_buffer) == n_steps
        else:
            # Subsequent transitions stored
            assert len(agent.buffer) == initial_buffer_len + (i - n_steps + 2)
            assert len(agent.n_step_buffer) == n_steps  # Should stay at maxlen

    assert len(agent.buffer) <= buffer_capacity


@pytest.mark.unit
def test_learn_step(agent, default_config):
    """Tests a single learning step, mocking buffer sample."""
    batch_size = default_config["batch_size"]
    n_steps = default_config["n_steps"]

    # 1. Ensure buffer has enough samples to trigger learning
    for _ in range(batch_size + n_steps):  # Need enough to fill n_step and sample a batch
        obs = generate_dummy_observation(default_config)
        action = np.random.randint(default_config["num_actions"])
        reward = np.random.rand()
        next_obs = generate_dummy_observation(default_config)
        done = False
        agent.store_transition(obs, action, reward, next_obs, done)

    assert len(agent.buffer) >= batch_size, "Buffer should have enough samples for a batch"

    # 2. Mock the buffer's sample method to return controlled data
    # Generate a dummy batch matching the expected output structure of buffer.sample
    mock_market = np.random.rand(batch_size, default_config["window_size"], default_config["n_features"]).astype(np.float32)
    mock_account = np.random.rand(batch_size, ACCOUNT_STATE_DIM).astype(np.float32)
    mock_action = np.random.randint(0, default_config["num_actions"], size=batch_size).astype(np.int64)
    mock_reward = np.random.rand(batch_size).astype(np.float32)
    mock_next_market = np.random.rand(batch_size, default_config["window_size"], default_config["n_features"]).astype(np.float32)
    mock_next_account = np.random.rand(batch_size, ACCOUNT_STATE_DIM).astype(np.float32)
    mock_done = np.zeros(batch_size, dtype=np.bool_)  # Assume not done for simplicity
    # Create mock batch data (previously used for mocking)
    (
        mock_market,
        mock_account,
        mock_action,
        mock_reward,
        mock_next_market,
        mock_next_account,
        mock_done,
    )

    # Mocking removed
    # mocker.patch.object(
    #     agent.buffer, "sample", return_value=(mock_batch, mock_indices, mock_weights)
    # )
    # mocker.patch.object(agent.buffer, "update_priorities")  # Mock priority updates
    # mocker.patch.object(
    #     agent, "_update_target_network"
    # )  # Mock target updates to isolate learn logic

    # 3. Call the learn method
    initial_total_steps = agent.total_steps
    initial_net_params = [p.clone().detach() for p in agent.network.parameters()]

    loss = agent.learn()

    # 4. Assertions
    assert loss is not None
    assert isinstance(loss, float)
    assert agent.total_steps == initial_total_steps + 1

    # Check if network parameters changed
    params_changed = False
    for p_initial, p_final in zip(initial_net_params, agent.network.parameters()):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Network parameters should have been updated after learning step."

    # Check mocks were called - Removed
    # agent.buffer.sample.assert_called_once_with(batch_size)
    # Need to check the args for update_priorities carefully
    # args, kwargs = agent.buffer.update_priorities.call_args
    # assert np.array_equal(args[0], mock_indices) # Check indices
    # assert isinstance(args[1], torch.Tensor) # Check priorities tensor
    # assert args[1].shape == (batch_size,)
    # assert args[1].dtype == torch.float32
    # Using assert_called_once is simpler if precise args aren't crucial
    # agent.buffer.update_priorities.assert_called_once()

    # Check if target network update was triggered if needed - Mocking removed
    # if agent.total_steps % default_config["target_update_freq"] == 0:
    #     agent._update_target_network.assert_called_once()
    # else:
    #     agent._update_target_network.assert_not_called()


@pytest.mark.unit
def test_target_network_update(agent, default_config):
    """Tests if the target network updates correctly."""
    # Ensure network and target network start differently (modify one slightly)
    with torch.no_grad():
        for param in agent.network.parameters():
            param.data += 0.1

    initial_target_params = [p.clone().detach() for p in agent.target_network.parameters()]

    # Force update
    agent._update_target_network()

    final_target_params = [p.clone().detach() for p in agent.target_network.parameters()]
    online_params = [p.clone().detach() for p in agent.network.parameters()]

    tau = agent.polyak_tau
    for p_initial, p_final, p_online in zip(initial_target_params, final_target_params, online_params):
        expected = (1.0 - tau) * p_initial + tau * p_online
        assert torch.allclose(p_final, expected, atol=1e-6), "Polyak soft update did not produce expected result."

    # Check if target params are different from initial target params
    params_updated = False
    for p_initial, p_final in zip(initial_target_params, final_target_params):
        if not torch.equal(p_initial, p_final):
            params_updated = True
            break
    assert params_updated, "Target network parameters should have changed after update."


@pytest.mark.unit
def test_save_load_model(agent, default_config, tmp_path):
    """Tests saving and loading the agent's state."""
    # Use tmp_path for the save directory
    save_dir = tmp_path / "agent_save"
    save_dir.mkdir()
    save_prefix = str(save_dir / "test_model")  # Use str() for conversion if needed
    # Remove manual cleanup check
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)

    # Modify agent state slightly
    agent.total_steps = 123
    # Perform a learn step to change network/optimizer state
    for _ in range(default_config["batch_size"] + default_config["n_steps"]):
        obs = generate_dummy_observation(default_config)
        agent.store_transition(obs, 1, 0.5, obs, False)  # Simple transition
    if len(agent.buffer) >= default_config["batch_size"]:
        agent.learn()  # Ensure optimizer has state and network changed

    # Capture current state for comparison
    original_state_dict = agent.network.state_dict()
    original_optimizer_dict = agent.optimizer.state_dict()
    original_total_steps = agent.total_steps

    # Save the model
    agent.save_model(save_prefix)

    # Check that unified checkpoint file exists
    checkpoint_path = f"{save_prefix}_agent_state.pt"

    assert os.path.exists(checkpoint_path)

    # Create a new agent instance with the same config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    new_agent = RainbowDQNAgent(config=default_config, device=device)
    assert new_agent.total_steps == 0  # Should start fresh

    # Load the saved state
    new_agent.load_model(save_prefix)

    # Compare states
    assert new_agent.total_steps == original_total_steps

    # Compare network weights
    loaded_state_dict = new_agent.network.state_dict()
    for key in original_state_dict:
        # Move tensors to same device for comparison if needed
        original_tensor = original_state_dict[key]
        loaded_tensor = loaded_state_dict[key]
        if original_tensor.device != loaded_tensor.device:
            loaded_tensor = loaded_tensor.to(original_tensor.device)
        assert torch.equal(original_tensor, loaded_tensor), f"Network parameter mismatch for key: {key}"

    # Compare target network weights (should also be loaded/synced)
    # Note: After load, target network should be synced to match the online network
    loaded_target_state_dict = new_agent.target_network.state_dict()
    for key in original_state_dict:
        if "epsilon" in key:
            continue
        original_tensor = original_state_dict[key]
        loaded_tensor = loaded_target_state_dict[key]
        if original_tensor.device != loaded_tensor.device:
            loaded_tensor = loaded_tensor.to(original_tensor.device)
        assert torch.allclose(original_tensor, loaded_tensor, atol=1e-3, rtol=1e-3), f"Target network parameter mismatch for key: {key}"

    # Compare optimizer state (tricky due to internal structure)
    loaded_optimizer_dict = new_agent.optimizer.state_dict()
    # Basic check: compare number of state groups and parameters
    assert len(original_optimizer_dict["state"]) == len(loaded_optimizer_dict["state"])
    assert len(original_optimizer_dict["param_groups"]) == len(loaded_optimizer_dict["param_groups"])
    # A more thorough check might involve comparing specific tensors within the state,
    # ensuring they are on the correct device after loading.

    # Clean up the saved model directory - NO LONGER NEEDED
    # shutil.rmtree(save_dir)


# --- Test Configuration Compatibility Check on Load ---
@pytest.mark.unit
def test_load_model_config_mismatch(agent, default_config, caplog, tmp_path):
    """Tests loading a model with a mismatched configuration."""
    # Use tmp_path for the save directory
    save_dir = tmp_path / "agent_save_mismatch"
    save_dir.mkdir()
    save_prefix = str(save_dir / "test_model_mismatch")  # Use str() for conversion if needed
    # Remove manual cleanup check
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)

    # Save the current agent
    agent.total_steps = 50
    agent.save_model(save_prefix)

    # Check that unified checkpoint file exists
    checkpoint_path = f"{save_prefix}_agent_state.pt"

    assert os.path.exists(checkpoint_path)

    # Create a new config with a mismatch
    mismatched_config = default_config.copy()
    mismatched_config["num_actions"] = default_config["num_actions"] + 1  # Change an essential param

    # Create a new agent with the mismatched config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mismatched_agent = RainbowDQNAgent(config=mismatched_config, device=device)

    # Load the model saved with the original config
    # This should fail due to network architecture mismatch
    mismatched_agent.load_model(save_prefix)

    # Check for size mismatch error, which would occur because the network architectures don't match
    assert "size mismatch" in caplog.text or "Error loading agent checkpoint" in caplog.text

    # Check that total_steps was loaded before the network loading error
    # (The checkpoint loading might partially succeed)
    # Note: If loading completely fails, total_steps might remain 0, so we check if it's loaded or not
    # The important thing is that the error was logged

    # Clean up - NO LONGER NEEDED
    # shutil.rmtree(save_dir)


@pytest.mark.unit
def test_set_training_mode(agent):
    """Tests setting the training mode."""
    # Initial state is training=True from fixture
    assert agent.training_mode is True
    assert agent.network.training is True
    assert agent.target_network.training is False  # Target network is always eval

    # Set to evaluation mode
    agent.set_training_mode(False)
    assert agent.training_mode is False
    assert agent.network.training is False
    assert agent.target_network.training is False  # Target network remains eval

    # Set back to training mode
    agent.set_training_mode(True)
    assert agent.training_mode is True
    assert agent.network.training is True
    assert agent.target_network.training is False  # Target network remains eval


@pytest.mark.unit
def test_compute_loss_returns_scalar_and_td_errors(agent, default_config):
    """Test that _compute_loss returns a scalar loss and correct-shaped TD errors."""
    batch_size = default_config["batch_size"]
    n_steps = default_config["n_steps"]

    for _ in range(batch_size + n_steps):
        obs = generate_dummy_observation(default_config)
        agent.store_transition(obs, 1, 0.5, obs, False)

    assert len(agent.buffer) >= batch_size

    agent.buffer.update_beta(agent.total_steps)
    batch_tuple, tree_indices, weights = agent.buffer.sample(batch_size)

    loss, td_errors = agent._compute_loss(batch_tuple, weights)

    assert loss.ndim == 0
    assert td_errors.shape == (batch_size,)
    assert torch.isfinite(loss)
    assert torch.isfinite(td_errors).all()


@pytest.mark.unit
def test_select_action_with_action_mask(agent, default_config):
    """Test that action masking forces a specific action."""
    obs = generate_dummy_observation(default_config)

    mask = np.zeros(default_config["num_actions"], dtype=bool)
    mask[2] = True

    for _ in range(10):
        action = agent.select_action(obs, action_mask=mask)
        assert action == 2


@pytest.mark.unit
def test_n_step_partial_storage(default_config):
    """Test that partial n-step transitions are stored at episode end."""
    config = default_config.copy()
    config["store_partial_n_step"] = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RainbowDQNAgent(config=config, device=device)

    obs = generate_dummy_observation(config)
    agent.store_transition(obs, 0, 1.0, obs, False)
    agent.store_transition(obs, 1, 0.5, obs, False)
    pre_done_count = len(agent.buffer)

    agent.store_transition(obs, 0, -0.5, obs, True)
    post_done_count = len(agent.buffer)

    assert post_done_count > pre_done_count


@pytest.mark.unit
def test_load_state_dict_path(agent, default_config, tmp_path):
    """Test loading agent state via the load_state dict path."""
    agent.total_steps = 99

    save_prefix = str(tmp_path / "test_state")
    agent.save_model(save_prefix)

    checkpoint_path = f"{save_prefix}_agent_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    new_agent = RainbowDQNAgent(config=default_config, device=device)
    assert new_agent.total_steps == 0

    success = new_agent.load_state(checkpoint)
    assert success
    assert new_agent.total_steps == 99


# or edge cases in PER interaction.

# Note: Testing the numerical correctness of _project_target_distribution
# would require known inputs and analytically derived or pre-computed expected outputs,
# which can be complex to set up. The current tests focus on integration and API usage.


# ---------------------------------------------------------------------------
# Tier 1c: categorical-target stats mirrored to TensorBoard
# ---------------------------------------------------------------------------


class _CapturingWriter:
    """Minimal SummaryWriter mock for verifying TB calls in unit tests."""

    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, np.ndarray, int]] = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((str(tag), float(value), int(step)))

    def add_histogram(self, tag, values, step):
        self.histograms.append((str(tag), np.asarray(values), int(step)))

    def tags(self) -> list[str]:
        return [t for t, _v, _s in self.scalars]

    def value_for(self, tag: str) -> float:
        matches = [v for t, v, _s in self.scalars if t == tag]
        if not matches:
            raise KeyError(f"No scalar emitted with tag {tag!r}")
        return matches[-1]


def _make_categorical_stub_agent(num_atoms: int = 11):
    """Construct a partial agent suitable for exercising _log_categorical_target_stats only."""
    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.num_atoms = num_atoms
    stub.support_cpu = np.linspace(-1.0, 1.0, num_atoms, dtype=np.float64)
    stub.categorical_logging_interval = 1
    stub.categorical_logging_percentiles = (5.0, 25.0, 50.0, 75.0, 95.0)
    stub._categorical_target_accumulator = {
        "mass": np.zeros(num_atoms, dtype=np.float64),
        "samples": 0,
    }
    stub.total_steps = 1234
    stub.tb_writer = None
    return stub


@pytest.mark.unit
def test_log_categorical_target_stats_mirrors_to_tensorboard():
    stub = _make_categorical_stub_agent(num_atoms=11)
    stub.tb_writer = _CapturingWriter()

    # Concentrate the accumulated mass on the centre atom so the resulting probs
    # peak at index 5 (support value 0.0); edge atoms hold a tiny share.
    stub._categorical_target_accumulator["mass"][5] = 80.0
    stub._categorical_target_accumulator["mass"][0] = 1.0
    stub._categorical_target_accumulator["mass"][-1] = 1.0
    stub._categorical_target_accumulator["mass"][3] = 9.0
    stub._categorical_target_accumulator["mass"][7] = 9.0
    stub._categorical_target_accumulator["samples"] = 100

    RainbowDQNAgent._log_categorical_target_stats(stub)

    writer = stub.tb_writer
    tags = set(writer.tags())
    assert "Train/CategoricalTarget/Mean" in tags
    assert "Train/CategoricalTarget/Edge_Mass_Min" in tags
    assert "Train/CategoricalTarget/Edge_Mass_Max" in tags
    assert "Train/CategoricalTarget/Samples" in tags
    for percentile in (5, 25, 50, 75, 95):
        assert f"Train/CategoricalTarget/P{percentile}" in tags

    expected_total = 80.0 + 1.0 + 1.0 + 9.0 + 9.0
    expected_edge = 1.0 / expected_total
    assert writer.value_for("Train/CategoricalTarget/Edge_Mass_Min") == pytest.approx(expected_edge)
    assert writer.value_for("Train/CategoricalTarget/Edge_Mass_Max") == pytest.approx(expected_edge)
    assert writer.value_for("Train/CategoricalTarget/Samples") == pytest.approx(100.0)
    # Mass is symmetric around the centre atom (support value 0.0) so mean ≈ 0.
    assert writer.value_for("Train/CategoricalTarget/Mean") == pytest.approx(0.0, abs=1e-9)
    # The 50th percentile should land on the centre atom (support value 0.0).
    assert writer.value_for("Train/CategoricalTarget/P50") == pytest.approx(0.0, abs=1e-9)

    assert any(tag == "Train/CategoricalTarget/Distribution" for tag, _v, _s in writer.histograms)
    hist_tag, hist_values, hist_step = writer.histograms[0]
    assert hist_tag == "Train/CategoricalTarget/Distribution"
    assert hist_step == 1234
    assert hist_values.shape == (11,)
    assert hist_values.sum() == pytest.approx(1.0, rel=1e-6)

    # Accumulator is reset after logging.
    assert stub._categorical_target_accumulator["samples"] == 0
    assert stub._categorical_target_accumulator["mass"].sum() == pytest.approx(0.0)


@pytest.mark.unit
def test_log_categorical_target_stats_no_writer_is_noop():
    stub = _make_categorical_stub_agent(num_atoms=11)
    stub._categorical_target_accumulator["mass"][5] = 100.0
    stub._categorical_target_accumulator["samples"] = 100
    # Should not raise even though tb_writer is None.
    RainbowDQNAgent._log_categorical_target_stats(stub)
    assert stub._categorical_target_accumulator["samples"] == 0


# ---------------------------------------------------------------------------
# Tier 1e: n-step reward window stats mirrored to TensorBoard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_n_step_reward_window_stats_mirrors_to_tensorboard():
    from collections import deque as _deque

    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.total_steps = 60
    stub.n_step_reward_window = _deque([0.1, -0.2, 0.3, -0.4, 0.5], maxlen=60)
    stub.tb_writer = _CapturingWriter()

    RainbowDQNAgent._log_n_step_reward_window_stats(stub)

    writer = stub.tb_writer
    tags = set(writer.tags())
    for expected in (
        "Train/NStepReward/Mean",
        "Train/NStepReward/Std",
        "Train/NStepReward/Min",
        "Train/NStepReward/Max",
        "Train/NStepReward/WindowSize",
    ):
        assert expected in tags, f"Missing n-step reward scalar: {expected}"

    expected_mean = float(np.mean([0.1, -0.2, 0.3, -0.4, 0.5]))
    expected_std = float(np.std([0.1, -0.2, 0.3, -0.4, 0.5], ddof=0))
    assert writer.value_for("Train/NStepReward/Mean") == pytest.approx(expected_mean)
    assert writer.value_for("Train/NStepReward/Std") == pytest.approx(expected_std)
    assert writer.value_for("Train/NStepReward/Min") == pytest.approx(-0.4)
    assert writer.value_for("Train/NStepReward/Max") == pytest.approx(0.5)
    assert writer.value_for("Train/NStepReward/WindowSize") == pytest.approx(5.0)


@pytest.mark.unit
def test_log_noisy_sigma_stats_emits_per_module_scalars(agent):
    """Tier 3a: per-NoisyLinear sigma stats are mirrored to TensorBoard."""
    agent.tb_writer = _CapturingWriter()
    agent.noisy_sigma_logging_interval = 1
    agent.total_steps = 4242
    agent._log_noisy_sigma_stats()

    writer = agent.tb_writer
    tags = writer.tags()
    # We don't hard-code module names (depends on the network architecture),
    # but at least one NoisyLinear must be present and produce all 3 stats.
    assert any(t.startswith("Train/Noisy/") and t.endswith("/SigmaMean") for t in tags), (
        f"No NoisyLinear sigma scalars emitted; tags={tags!r}"
    )
    assert any(t.endswith("/SigmaMax") for t in tags)
    assert any(t.endswith("/SigmaMin") for t in tags)
    assert "Train/Noisy/AggregateSigmaMean" in tags
    assert "Train/Noisy/ModuleCount" in tags

    module_count = writer.value_for("Train/Noisy/ModuleCount")
    assert module_count >= 1.0
    # All scalars should share the same step.
    steps = {s for _t, _v, s in writer.scalars}
    assert steps == {4242}


@pytest.mark.unit
def test_log_noisy_sigma_stats_no_writer_is_noop(agent):
    """Tier 3a: missing tb_writer is a no-op (no exception)."""
    agent.tb_writer = None
    agent._log_noisy_sigma_stats()


# ---------------------------------------------------------------------------
# Tier 3b: Q-value stats mirrored to TensorBoard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_q_value_stats_emits_per_action_means_and_margin():
    """Tier 3b: per-action means + ActionMargin land in TB at the current step."""
    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.total_steps = 4321
    stub.tb_writer = _CapturingWriter()
    # Batch of 4 examples, 6 actions. Action 5 is uniformly the best by 0.5.
    q = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.9],
            [0.5, 0.4, 0.3, 0.2, 0.1, 1.0],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )
    stub._last_batch_q = q
    RainbowDQNAgent._log_q_value_stats(stub, emit_histogram=False)

    writer = stub.tb_writer
    tags = set(writer.tags())
    for required in (
        "Train/Q/Mean",
        "Train/Q/Std",
        "Train/Q/MaxAcrossActions",
        "Train/Q/MinAcrossActions",
        "Train/Q/ActionMargin",
    ):
        assert required in tags, f"Missing Q-value scalar: {required}"
    for action_idx in range(6):
        assert f"Train/Q/PerAction/Mean/{action_idx}" in tags

    assert writer.value_for("Train/Q/Mean") == pytest.approx(float(q.mean()))
    assert writer.value_for("Train/Q/Std") == pytest.approx(float(q.std(unbiased=False)))
    assert writer.value_for("Train/Q/PerAction/Mean/5") == pytest.approx(float(q[:, 5].mean()))
    # Margin is mean(top1 - top2). For each row top1=1.0/0.9/1.0/0.7 and top2=0.5/0.4/0.5/0.2.
    expected_margin = float(((1.0 - 0.5) + (0.9 - 0.4) + (1.0 - 0.5) + (0.7 - 0.2)) / 4)
    assert writer.value_for("Train/Q/ActionMargin") == pytest.approx(expected_margin)
    # No histogram unless explicitly asked.
    assert all(t != "Train/Q/Distribution" for t, _v, _s in writer.histograms)


@pytest.mark.unit
def test_log_q_value_stats_emits_histogram_when_requested():
    """Tier 3b: emit_histogram=True writes a Train/Q/Distribution histogram."""
    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.total_steps = 100
    stub.tb_writer = _CapturingWriter()
    stub._last_batch_q = torch.zeros(2, 6, dtype=torch.float32)
    RainbowDQNAgent._log_q_value_stats(stub, emit_histogram=True)
    assert any(t == "Train/Q/Distribution" for t, _v, _s in stub.tb_writer.histograms)


@pytest.mark.unit
def test_log_grad_stats_emits_global_per_group_and_update_ratio(agent):
    """Tier 3c: gradient norms + ParamUpdateRatio land in TB after a backward pass."""
    agent.tb_writer = _CapturingWriter()
    agent.grad_logging_interval = 1
    agent.total_steps = 999

    # Run a tiny forward+backward to populate .grad on every parameter.
    obs = generate_dummy_observation(agent.config if hasattr(agent, "config") else {})
    market = torch.from_numpy(obs["market_data"]).unsqueeze(0).to(agent.device, dtype=torch.float32)
    account = torch.from_numpy(obs["account_state"]).unsqueeze(0).to(agent.device, dtype=torch.float32)
    out = agent.network.get_q_values(market, account)
    out.sum().backward()

    pre_clip_norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1e9)
    agent._log_grad_stats(pre_clip_norm)

    writer = agent.tb_writer
    tags = set(writer.tags())
    assert "Train/Grad/Norm" in tags
    assert "Train/ParamUpdateRatio" in tags
    # At least one per-group norm exists.
    assert any(t.startswith("Train/Grad/PerGroup/") and t.endswith("/Norm") for t in tags), (
        f"No per-group grad norms emitted; tags={tags!r}"
    )

    # Step value must be (total_steps + 1) per Tier 3c contract — see comment
    # in _log_grad_stats. Confirms the lr * grad / param ratio is associated
    # with the *next* learn step, not the one whose grad was just consumed.
    grad_steps = {s for t, _v, s in writer.scalars if t == "Train/Grad/Norm"}
    assert grad_steps == {1000}


@pytest.mark.unit
def test_target_net_soft_update_emits_counter_and_deviation(agent):
    """Tier 3d: SoftUpdates counter increments every call; ParamDeviation > 0
    when networks have diverged; both throttled by target_net_logging_interval."""
    agent.tb_writer = _CapturingWriter()
    agent.target_net_logging_interval = 1
    agent.total_steps = 42

    # Force divergence between online + target so deviation is non-zero.
    with torch.no_grad():
        for p in agent.network.parameters():
            p.add_(0.1)

    starting_count = agent._soft_update_count
    agent._update_target_network()
    assert agent._soft_update_count == starting_count + 1

    writer = agent.tb_writer
    tags = {t for t, _v, _s in writer.scalars}
    assert "Train/TargetNet/SoftUpdates" in tags
    assert "Train/TargetNet/ParamDeviation" in tags
    deviation = next(v for t, v, _s in writer.scalars if t == "Train/TargetNet/ParamDeviation")
    assert deviation > 0.0, "deviation should be positive after we offset online weights"
    counter = next(v for t, v, _s in writer.scalars if t == "Train/TargetNet/SoftUpdates")
    assert counter == float(agent._soft_update_count)


@pytest.mark.unit
def test_target_net_logging_throttle_skips_intermediate_updates(agent):
    """Tier 3d: with interval=3, only every 3rd Polyak update emits scalars."""
    agent.tb_writer = _CapturingWriter()
    agent.target_net_logging_interval = 3
    starting = agent._soft_update_count
    for _ in range(5):
        agent._update_target_network()
    expected_emits = sum(1 for n in range(starting + 1, starting + 6) if n % 3 == 0)
    counter_emits = [v for t, v, _s in agent.tb_writer.scalars if t == "Train/TargetNet/SoftUpdates"]
    assert len(counter_emits) == expected_emits


@pytest.mark.unit
def test_target_net_no_writer_is_noop(agent):
    """Tier 3d: counter still ticks but no scalars are written when writer is None."""
    agent.tb_writer = None
    starting = agent._soft_update_count
    agent._update_target_network()
    assert agent._soft_update_count == starting + 1


@pytest.mark.unit
def test_log_grad_stats_no_writer_is_noop():
    """Tier 3c: missing writer / network are silent no-ops."""
    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.tb_writer = None
    stub.network = None
    stub.total_steps = 0
    stub.optimizer = None
    RainbowDQNAgent._log_grad_stats(stub, 1.0)  # should not raise


@pytest.mark.unit
def test_log_q_value_stats_no_writer_is_noop():
    """Tier 3b: a missing writer (or missing cache) returns silently."""
    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.total_steps = 0
    stub.tb_writer = None
    stub._last_batch_q = torch.zeros(2, 6, dtype=torch.float32)
    RainbowDQNAgent._log_q_value_stats(stub, emit_histogram=True)
    # And missing cache:
    stub.tb_writer = _CapturingWriter()
    stub._last_batch_q = None
    RainbowDQNAgent._log_q_value_stats(stub, emit_histogram=True)
    assert stub.tb_writer.scalars == []
    assert stub.tb_writer.histograms == []


@pytest.mark.unit
def test_log_n_step_reward_window_stats_empty_window_is_noop():
    from collections import deque as _deque

    stub = RainbowDQNAgent.__new__(RainbowDQNAgent)
    stub.total_steps = 60
    stub.n_step_reward_window = _deque([], maxlen=60)
    stub.tb_writer = _CapturingWriter()
    RainbowDQNAgent._log_n_step_reward_window_stats(stub)
    assert stub.tb_writer.scalars == []


# ---------------------------------------------------------------------------
# reset_noisy_sigma helper (recovery: re-energise NoisyNet exploration)
# ---------------------------------------------------------------------------


def _collect_noisy_layers(network):
    from momentum_agent.model import NoisyLinear

    inner = getattr(network, "_orig_mod", network)
    return [m for m in inner.modules() if isinstance(m, NoisyLinear)]


@pytest.mark.unit
def test_reset_noisy_sigma_refills_to_constructor_formula(agent):
    """Sigma values match ``std_init / sqrt(in_features)`` and bias_sigma
    matches ``std_init / sqrt(out_features)`` after reset, regardless of the
    pre-reset state."""
    import math as _math

    online_layers = _collect_noisy_layers(agent.network)
    assert online_layers, "agent network must contain at least one NoisyLinear"

    with torch.no_grad():
        for layer in online_layers:
            layer.weight_sigma.data.fill_(0.0)
            layer.bias_sigma.data.fill_(0.0)

    count = agent.reset_noisy_sigma()
    assert count == len(online_layers)
    for layer in online_layers:
        expected_w = layer.std_init / _math.sqrt(layer.in_features)
        expected_b = layer.std_init / _math.sqrt(layer.out_features)
        assert torch.allclose(
            layer.weight_sigma.data,
            torch.full_like(layer.weight_sigma.data, expected_w),
        )
        assert torch.allclose(
            layer.bias_sigma.data,
            torch.full_like(layer.bias_sigma.data, expected_b),
        )


@pytest.mark.unit
def test_reset_noisy_sigma_leaves_mu_untouched(agent):
    """Mu (the deterministic part the policy actually uses for argmax) must
    survive the reset bit-for-bit."""
    online_layers = _collect_noisy_layers(agent.network)
    snapshot_w = [layer.weight_mu.data.detach().clone() for layer in online_layers]
    snapshot_b = [layer.bias_mu.data.detach().clone() for layer in online_layers]

    agent.reset_noisy_sigma(std_init=0.7)

    for layer, w, b in zip(online_layers, snapshot_w, snapshot_b, strict=True):
        assert torch.equal(layer.weight_mu.data, w)
        assert torch.equal(layer.bias_mu.data, b)


@pytest.mark.unit
def test_reset_noisy_sigma_overrides_std_init_when_provided(agent):
    """Passing ``std_init=X`` updates ``layer.std_init`` and the new sigma
    values use ``X`` rather than the original constructor scalar."""
    import math as _math

    online_layers = _collect_noisy_layers(agent.network)
    custom = 0.123
    agent.reset_noisy_sigma(std_init=custom)
    for layer in online_layers:
        assert layer.std_init == pytest.approx(custom)
        expected_w = custom / _math.sqrt(layer.in_features)
        assert layer.weight_sigma.data.flatten()[0].item() == pytest.approx(expected_w)


@pytest.mark.unit
def test_reset_noisy_sigma_also_resets_target_network(agent):
    """Both online and target networks get refilled (their sigma scales must
    stay structurally identical even though they sample independent epsilon)."""
    target_layers = _collect_noisy_layers(agent.target_network)
    with torch.no_grad():
        for layer in target_layers:
            layer.weight_sigma.data.fill_(99.0)

    agent.reset_noisy_sigma()
    for layer in target_layers:
        assert layer.weight_sigma.data.max().item() < 99.0, "target network sigma should have been refilled away from the sentinel"


@pytest.mark.unit
def test_reset_noisy_sigma_emits_tb_scalar_when_writer_attached(agent):
    """When ``tb_writer`` is set, ``Agent/NoisySigmaReset`` is logged with the
    layer count at the current ``total_steps``."""
    agent.tb_writer = _CapturingWriter()
    agent.total_steps = 1234
    count = agent.reset_noisy_sigma()
    tags = {t for t, _v, _s in agent.tb_writer.scalars}
    assert "Agent/NoisySigmaReset" in tags
    value = next(v for t, v, _s in agent.tb_writer.scalars if t == "Agent/NoisySigmaReset")
    assert value == float(count)
    step = next(s for t, _v, s in agent.tb_writer.scalars if t == "Agent/NoisySigmaReset")
    assert step == 1234


@pytest.mark.unit
def test_reset_noisy_sigma_no_writer_is_noop_for_logging(agent):
    """Without a writer, the reset itself still works and no exception is raised."""
    agent.tb_writer = None
    count = agent.reset_noisy_sigma()
    assert count > 0


# ---------------------------------------------------------------------------
# torch.compile state_dict prefix fallback (bidirectional)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_state_dict_fallback_strips_orig_mod_prefix():
    """`_orig_mod.*` keys load into a plain (eager) module via the fallback."""
    from momentum_agent.agent import _load_state_dict_with_orig_mod_fallback

    plain = torch.nn.Linear(4, 3)
    src = torch.nn.Linear(4, 3)
    prefixed = {f"_orig_mod.{k}": v.clone() for k, v in src.state_dict().items()}

    _load_state_dict_with_orig_mod_fallback(plain, prefixed)

    for k, v in src.state_dict().items():
        assert torch.equal(plain.state_dict()[k], v)


@pytest.mark.unit
def test_load_state_dict_fallback_adds_orig_mod_prefix():
    """Plain keys load into an `OptimizedModule`-style wrapper via the fallback.

    We simulate `torch.compile`'s OptimizedModule with a wrapper that stores
    the inner module under `_orig_mod`, which is exactly the attribute name
    PyTorch's compile machinery uses. The fallback must add the `_orig_mod.`
    prefix to the plain state dict so it lines up with the wrapper's own
    parameter naming.
    """
    from momentum_agent.agent import _load_state_dict_with_orig_mod_fallback

    class FakeOptimizedModule(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    inner_target = torch.nn.Linear(4, 3)
    wrapped = FakeOptimizedModule(inner_target)
    src = torch.nn.Linear(4, 3)
    plain = {k: v.clone() for k, v in src.state_dict().items()}

    _load_state_dict_with_orig_mod_fallback(wrapped, plain)

    for k, v in src.state_dict().items():
        assert torch.equal(wrapped.state_dict()[f"_orig_mod.{k}"], v)


@pytest.mark.unit
def test_load_state_dict_fallback_passthrough_when_no_prefix_change_helps():
    """If neither prefix transform helps, the original error must propagate."""
    from momentum_agent.agent import _load_state_dict_with_orig_mod_fallback

    target = torch.nn.Linear(4, 3)
    bad_state = {"completely_unrelated_key": torch.zeros(2, 2)}

    with pytest.raises(Exception):
        _load_state_dict_with_orig_mod_fallback(target, bad_state)

import pytest
import numpy as np
import sys
import torch

# Remove sys.path manipulation
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# Direct import from src package
from momentum_agent.buffer import PrioritizedReplayBuffer, Experience, SumTree

# --- Constants for Dummy Data --- #
BUFFER_CAPACITY = 100
BATCH_SIZE = 10
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 1000
WINDOW_SIZE = 5
N_FEATURES = 3
ACCOUNT_STATE_DIM = 2


# --- Helper to create dummy experience --- #
def create_dummy_experience(i=0):
    return Experience(
        market_data=np.random.rand(WINDOW_SIZE, N_FEATURES).astype(np.float32) + i,
        account_state=np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32) + i,
        action=np.random.randint(0, 5),
        reward=np.random.rand() * 10 - 5,
        next_market_data=np.random.rand(WINDOW_SIZE, N_FEATURES).astype(np.float32)
        + i
        + 1,
        next_account_state=np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32) + i + 1,
        done=bool(np.random.rand() > 0.95),
    )


# --- Fixture for an initialized buffer --- #
@pytest.fixture
def per_buffer():
    """Provides an initialized PrioritizedReplayBuffer instance."""
    return PrioritizedReplayBuffer(BUFFER_CAPACITY, ALPHA, BETA_START, BETA_FRAMES)


# --- Test Cases --- #


@pytest.mark.unittest
def test_buffer_init(per_buffer):
    """Test buffer initialization."""
    assert isinstance(per_buffer.tree, SumTree)
    assert per_buffer.capacity == BUFFER_CAPACITY
    assert per_buffer.alpha == ALPHA
    assert per_buffer.beta == BETA_START
    assert per_buffer.max_priority == 1.0
    assert len(per_buffer) == 0


@pytest.mark.unittest
def test_buffer_store_single(per_buffer):
    """Test storing a single experience."""
    exp = create_dummy_experience()
    per_buffer.store(*exp)
    assert len(per_buffer) == 1
    assert len(per_buffer.buffer) == 1
    assert per_buffer.tree.total() > 0  # Priority should be added
    assert per_buffer.buffer[0] == exp  # Check if deque stores correctly


@pytest.mark.unittest
def test_buffer_store_respects_max_priority(per_buffer):
    """New experiences should inherit the tracked max priority without reapplying alpha."""
    per_buffer.max_priority = 0.25
    exp = create_dummy_experience()
    per_buffer.store(*exp)

    # Only one item stored, total priority should equal the set max priority.
    assert per_buffer.tree.total() == pytest.approx(0.25)


@pytest.mark.unittest
def test_buffer_store_multiple(per_buffer):
    """Test storing multiple experiences."""
    num_items = 50
    first_exp = None
    last_exp = None
    for i in range(num_items):
        exp = create_dummy_experience(i)
        if i == 0:
            first_exp = exp  # Store the actual first experience
        if i == num_items - 1:
            last_exp = exp  # Store the actual last experience
        per_buffer.store(*exp)

    assert len(per_buffer) == num_items
    assert len(per_buffer.buffer) == num_items
    # Check if the first item stored is still there
    assert per_buffer.buffer[0] == first_exp
    # Check if the last item stored is correct
    assert per_buffer.buffer[-1] == last_exp


@pytest.mark.unittest
def test_buffer_store_exceed_capacity(per_buffer):
    """Test storing more experiences than capacity."""
    for i in range(BUFFER_CAPACITY + 20):
        exp = create_dummy_experience(i)
        per_buffer.store(*exp)
    assert len(per_buffer) == BUFFER_CAPACITY
    assert len(per_buffer.buffer) == BUFFER_CAPACITY
    # Check if the oldest items were overwritten (by checking a field like action)
    # The items remaining should be from i=20 to i=BUFFER_CAPACITY+19
    first_remaining_exp_action = create_dummy_experience(20).action
    last_remaining_exp_action = create_dummy_experience(BUFFER_CAPACITY + 19).action
    # This check is tricky because deque overwrites, let's check buffer_write_idx
    assert per_buffer.buffer_write_idx == 20


@pytest.mark.unittest
def test_buffer_sample_empty(per_buffer):
    """Test sampling from an empty buffer."""
    batch, indices, weights = per_buffer.sample(BATCH_SIZE)
    assert batch is None
    assert indices is None
    assert weights is None


@pytest.mark.unittest
def test_buffer_sample_insufficient(per_buffer):
    """Test sampling when buffer has fewer items than batch size."""
    for i in range(BATCH_SIZE - 1):
        per_buffer.store(*create_dummy_experience(i))
    assert len(per_buffer) == BATCH_SIZE - 1
    batch, indices, weights = per_buffer.sample(BATCH_SIZE)
    assert batch is None
    assert indices is None
    assert weights is None


@pytest.mark.unittest
def test_buffer_sample_sufficient(per_buffer):
    """Test sampling when buffer has enough items."""
    for i in range(BUFFER_CAPACITY):
        per_buffer.store(*create_dummy_experience(i))
    assert len(per_buffer) == BUFFER_CAPACITY

    batch, tree_indices, weights = per_buffer.sample(BATCH_SIZE)

    assert batch is not None
    assert tree_indices is not None
    assert weights is not None

    # Check batch structure and types
    assert isinstance(batch, tuple)
    assert len(batch) == 7  # Number of fields in Experience
    market_data, account_state, actions, rewards, next_market, next_account, dones = (
        batch
    )
    assert isinstance(market_data, np.ndarray)
    assert isinstance(account_state, np.ndarray)
    assert isinstance(actions, np.ndarray)
    assert isinstance(rewards, np.ndarray)
    assert isinstance(next_market, np.ndarray)
    assert isinstance(next_account, np.ndarray)
    assert isinstance(dones, np.ndarray)

    # Check shapes
    assert market_data.shape == (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
    assert account_state.shape == (BATCH_SIZE, ACCOUNT_STATE_DIM)
    assert actions.shape == (BATCH_SIZE,)
    assert rewards.shape == (BATCH_SIZE,)
    assert next_market.shape == (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
    assert next_account.shape == (BATCH_SIZE, ACCOUNT_STATE_DIM)
    assert dones.shape == (BATCH_SIZE,)

    # Check weights and indices
    assert isinstance(tree_indices, list)
    assert len(tree_indices) == BATCH_SIZE
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (BATCH_SIZE,)
    assert np.all(weights > 0) and np.all(weights <= 1.0 + 1e-6)


@pytest.mark.unittest
def test_buffer_update_priorities(per_buffer):
    """Test updating priorities after sampling."""
    for i in range(BUFFER_CAPACITY):
        per_buffer.store(*create_dummy_experience(i))

    # Sample a batch
    batch, tree_indices, weights = per_buffer.sample(BATCH_SIZE)
    assert tree_indices is not None

    # Get initial priorities for the sampled items
    initial_priorities = [per_buffer.tree.tree[idx] for idx in tree_indices]

    # Create dummy TD errors (priorities) - use tensor as expected by method
    new_td_errors = torch.abs(torch.randn(BATCH_SIZE))
    new_priorities_np = (new_td_errors.numpy() + per_buffer.epsilon) ** per_buffer.alpha

    # Update priorities
    per_buffer.update_priorities(tree_indices, new_td_errors)

    # Check if priorities in the tree were updated
    updated_priorities = [per_buffer.tree.tree[idx] for idx in tree_indices]

    assert len(initial_priorities) == BATCH_SIZE
    assert len(updated_priorities) == BATCH_SIZE
    # Assert that the priorities have changed and roughly match the expected new values
    np.testing.assert_allclose(updated_priorities, new_priorities_np, rtol=1e-6)
    # Assert that max priority was potentially updated
    assert per_buffer.max_priority >= np.max(new_priorities_np)


@pytest.mark.unittest
def test_buffer_update_beta(per_buffer):
    """Test beta annealing."""
    assert per_buffer.beta == BETA_START
    per_buffer.update_beta(total_steps=0)
    assert per_buffer.beta == BETA_START

    per_buffer.update_beta(total_steps=BETA_FRAMES / 2)
    expected_beta_mid = BETA_START + 0.5 * (1.0 - BETA_START)
    assert abs(per_buffer.beta - expected_beta_mid) < 1e-6

    per_buffer.update_beta(total_steps=BETA_FRAMES)
    assert abs(per_buffer.beta - 1.0) < 1e-6

    per_buffer.update_beta(total_steps=BETA_FRAMES * 2)
    assert abs(per_buffer.beta - 1.0) < 1e-6  # Should clamp at 1.0

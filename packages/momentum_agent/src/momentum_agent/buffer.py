import random
from collections import deque, namedtuple

import numpy as np
import torch

from .utils.logging_config import get_logger

# Get logger instance
logger = get_logger("Buffer")

# Define Experience namedtuple at module level for pickling (Copied from agent.py)
Experience = namedtuple(
    "Experience",
    field_names=[
        "market_data",
        "account_state",
        "action",
        "reward",
        "next_market_data",
        "next_account_state",
        "done",
    ],
)


# --- Start: SumTree Implementation ---
class SumTree:
    """Binary Sum Tree for efficient priority sampling."""

    write = 0  # Current position in the data array (leaves)

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores priorities (internal nodes are sums)
        self.data_indices = np.zeros(capacity, dtype=int)  # Maps tree leaf index to data index in buffer
        self.size = 0  # Current number of items stored

    def _propagate(self, idx: int, change: float):
        """Propagates priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Finds the leaf index corresponding to a cumulative priority s."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx  # Reached leaf node
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Returns the total sum of priorities (root node)."""
        return self.tree[0]

    def add(self, p: float, data_idx: int):
        """Stores priority p and associated data index."""
        tree_idx = self.write + self.capacity - 1  # Map write pointer to tree leaf index
        self.data_indices[self.write] = data_idx  # Store mapping
        self.update(tree_idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, p: float):
        """Updates the priority at a specific tree index."""
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, int]:
        """Samples a leaf node based on cumulative priority s."""
        assert s >= 0.0 and s <= self.total() + 1e-6, f"Sample value {s} out of range [0, {self.total()}]"
        idx = self._retrieve(0, s)
        data_idx_ptr = idx - self.capacity + 1  # Map tree leaf index back to data index pointer (0 to capacity-1)
        return (idx, self.tree[idx], self.data_indices[data_idx_ptr])

    def __len__(self) -> int:
        return self.size


# --- End: SumTree Implementation ---


# --- Start: Prioritized Replay Buffer (PER) ---
# Simplified PER implementation (SumTree can be more efficient for large buffers)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.epsilon = 1e-5  # Small constant to ensure non-zero priority
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Initial IS exponent
        self.beta_frames = beta_frames
        self.buffer = deque(maxlen=capacity)  # Stores Experience objects
        self.tree = SumTree(capacity)  # Manages priorities
        self.beta = beta_start  # Current beta value, updated externally
        self.max_priority = 1.0  # Track max priority efficiently
        self.buffer_write_idx = 0  # Tracks current write position in self.buffer

    def update_beta(self, total_steps: int):
        """Updates the beta value based on the total training steps."""
        self.beta = min(
            1.0,
            self.beta_start + total_steps * (1.0 - self.beta_start) / self.beta_frames,
        )

    def store(self, *args):
        """Stores experience and assigns max priority."""
        experience = Experience(*args)
        priority = self.max_priority  # Already tracked in alpha-space
        if priority <= 0:
            priority = 1.0

        # Add experience to buffer deque
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.buffer_write_idx] = experience
        # Add priority to SumTree, associating with the current buffer write index
        self.tree.add(priority, self.buffer_write_idx)
        # Increment buffer write index
        self.buffer_write_idx = (self.buffer_write_idx + 1) % self.capacity

    def sample(self, batch_size):
        """Samples batch, calculates IS weights."""
        if len(self.tree) < batch_size:
            return None, None, None  # Not enough samples

        indices = []
        tree_indices = []
        samples = []
        priorities = []
        segment = self.tree.total() / batch_size
        # Ensure beta is up-to-date (though agent should call update_beta)
        assert 0.0 <= self.beta <= 1.0, f"Invalid beta value: {self.beta}"
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (tree_idx, p, data_idx) = self.tree.get(s)
            priorities.append(p)
            samples.append(self.buffer[data_idx])
            indices.append(data_idx)
            tree_indices.append(tree_idx)
        sampling_probabilities = np.array(priorities) / self.tree.total()

        # Compute Importance Sampling weights
        # N = len(self.buffer) here is the *current* number of elements
        N = len(self)
        weights = (N * sampling_probabilities) ** (-self.beta)
        # Normalize by max weight for stability
        max_weight = weights.max() if weights.size > 0 else 1.0
        assert max_weight > 1e-9, f"Max IS weight is zero or negative ({max_weight})"
        weights /= max_weight  # Normalize for stability
        weights = np.array(weights, dtype=np.float32)
        assert weights.shape == (batch_size,), f"IS weights shape mismatch. Expected ({batch_size},), got {weights.shape}"
        assert np.all(weights >= 0) and np.all(weights <= 1.0 + 1e-6), "IS weights are outside [0, 1] range"  # Allow small tolerance

        # Unzip samples
        (
            market_data,
            account_state,
            actions,
            rewards,
            next_market_data,
            next_account_state,
            dones,
        ) = zip(*samples)

        # --- Start: Add assertions for sampled data types and basic structure ---
        assert len(market_data) == batch_size, "Incorrect number of market_data samples"
        assert len(account_state) == batch_size, "Incorrect number of account_state samples"
        assert len(actions) == batch_size, "Incorrect number of action samples"
        assert len(rewards) == batch_size, "Incorrect number of reward samples"
        assert len(next_market_data) == batch_size, "Incorrect number of next_market_data samples"
        assert len(next_account_state) == batch_size, "Incorrect number of next_account_state samples"
        assert len(dones) == batch_size, "Incorrect number of done samples"
        # Check types (assuming they are stored as numpy arrays originally)
        assert all(isinstance(x, np.ndarray) for x in market_data), "Market data samples are not all numpy arrays"
        assert all(isinstance(x, np.ndarray) for x in account_state), "Account state samples are not all numpy arrays"
        assert all(isinstance(x, np.ndarray) for x in next_market_data), "Next market data samples are not all numpy arrays"
        assert all(isinstance(x, np.ndarray) for x in next_account_state), "Next account state samples are not all numpy arrays"
        # --- End: Add assertions for sampled data types and basic structure ---

        return (
            (
                np.array(market_data, dtype=np.float32),
                np.array(account_state, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_market_data, dtype=np.float32),
                np.array(next_account_state, dtype=np.float32),
                np.array(dones, dtype=np.uint8),
            ),
            tree_indices,
            weights,
        )

    def update_priorities(self, tree_indices, batch_priorities_tensor):
        """Updates priorities of sampled transitions using a tensor of priorities."""
        assert isinstance(batch_priorities_tensor, torch.Tensor), "batch_priorities must be a tensor"
        assert len(tree_indices) == len(batch_priorities_tensor), "Indices and priorities length mismatch in update_priorities"

        td_errors = batch_priorities_tensor.detach().cpu().numpy()
        # Calculate new priorities: |TD_error|**alpha + epsilon
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        assert np.all(new_priorities > 0), f"New priority calculated is non-positive: min={new_priorities.min()}"

        # Update priorities in the deque
        for tree_idx, priority in zip(tree_indices, new_priorities):
            assert priority > 0
            self.tree.update(tree_idx, priority)

        # Calculate max priority in the current batch and update overall max
        if len(new_priorities) > 0:
            batch_max_prio = np.max(new_priorities)
            self.max_priority = max(self.max_priority, batch_max_prio)

    def state_dict(self):
        """Returns the state of the buffer for saving."""
        return {
            "buffer": list(self.buffer),  # Convert deque to list for saving
            "tree_state": {
                "tree": self.tree.tree,
                "data_indices": self.tree.data_indices,
                "write": self.tree.write,
                "size": self.tree.size,
            },
            "buffer_write_idx": self.buffer_write_idx,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "epsilon": self.epsilon,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state_dict):
        """Loads the buffer state from a state dictionary."""
        # Basic validation
        required_keys = [
            "buffer",
            "tree_state",
            "buffer_write_idx",
            "max_priority",
            "alpha",
            "beta",
            "beta_start",
            "beta_frames",
            "epsilon",
            "capacity",
        ]
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f"Missing key in buffer state_dict: {key}")
        if state_dict["capacity"] != self.capacity:
            # Maybe allow resizing later, but for now require capacity match
            raise ValueError(f"Capacity mismatch: state_dict has {state_dict['capacity']}, buffer has {self.capacity}")
        tree_state = state_dict["tree_state"]
        required_tree_keys = ["tree", "data_indices", "write", "size"]
        for key in required_tree_keys:
            if key not in tree_state:
                raise ValueError(f"Missing key in buffer tree_state: {key}")

        # Restore buffer deque from list
        self.buffer = deque(state_dict["buffer"], maxlen=self.capacity)

        # Restore SumTree state
        self.tree.tree = tree_state["tree"]
        self.tree.data_indices = tree_state["data_indices"]
        self.tree.write = tree_state["write"]
        self.tree.size = tree_state["size"]

        # Restore other attributes
        self.buffer_write_idx = state_dict["buffer_write_idx"]
        self.max_priority = state_dict["max_priority"]
        self.alpha = state_dict["alpha"]
        self.beta = state_dict["beta"]
        self.beta_start = state_dict["beta_start"]
        self.beta_frames = state_dict["beta_frames"]
        self.epsilon = state_dict["epsilon"]
        # self.capacity is checked above

        # Sanity check after loading
        assert len(self.buffer) == self.tree.size, "Buffer deque length doesn't match SumTree size after load"

    def __len__(self):
        # Return the current fill size of the buffer/tree
        return self.tree.size


# --- End: Prioritized Replay Buffer ---

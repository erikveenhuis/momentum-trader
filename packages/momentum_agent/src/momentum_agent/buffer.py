import json
import random
import shutil
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
from momentum_core.logging import get_logger

# Get logger instance
logger = get_logger(__name__)

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

    def __init__(self, capacity: int, *, debug: bool = False):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores priorities (internal nodes are sums)
        self.data_indices = np.zeros(capacity, dtype=int)  # Maps tree leaf index to data index in buffer
        self.size = 0  # Current number of items stored
        self.debug = debug

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
        if self.debug:
            total = self.total()
            if not (0.0 <= s <= total + 1e-6):
                raise ValueError(f"Sample value {s} out of range [0, {total}]")
        idx = self._retrieve(0, s)
        data_idx_ptr = idx - self.capacity + 1  # Map tree leaf index back to data index pointer (0 to capacity-1)
        return (idx, self.tree[idx], self.data_indices[data_idx_ptr])

    def __len__(self) -> int:
        return self.size


# --- End: SumTree Implementation ---


# --- Start: Prioritized Replay Buffer (PER) ---
# Simplified PER implementation (SumTree can be more efficient for large buffers)
class PrioritizedReplayBuffer:
    """SumTree-backed prioritized experience replay buffer with importance-sampling weight annealing."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, *, debug: bool = False):
        self.epsilon = 1e-5  # Small constant to ensure non-zero priority
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Initial IS exponent
        self.beta_frames = beta_frames
        self.buffer = deque(maxlen=capacity)  # Stores Experience objects
        self.tree = SumTree(capacity, debug=debug)  # Manages priorities
        self.beta = beta_start  # Current beta value, updated externally
        self.max_priority = 1.0  # Track max priority efficiently
        self.buffer_write_idx = 0  # Tracks current write position in self.buffer
        self.debug = debug

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
        if self.debug and not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"Invalid beta value: {self.beta}")
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
        if self.debug and max_weight <= 1e-9:
            raise ValueError(f"Max IS weight is zero or negative ({max_weight})")
        weights /= max_weight  # Normalize for stability
        weights = np.array(weights, dtype=np.float32)
        if self.debug and weights.shape != (batch_size,):
            raise ValueError(f"IS weights shape mismatch. Expected ({batch_size},), got {weights.shape}")
        if self.debug and (np.any(weights < 0) or np.any(weights > 1.0 + 1e-6)):
            raise ValueError("IS weights are outside [0, 1] range")

        # Unzip samples
        (
            market_data,
            account_state,
            actions,
            rewards,
            next_market_data,
            next_account_state,
            dones,
        ) = zip(*samples, strict=False)

        # --- Start: Add assertions for sampled data types and basic structure ---
        if self.debug:
            if len(market_data) != batch_size:
                raise ValueError("Incorrect number of market_data samples")
            if len(account_state) != batch_size:
                raise ValueError("Incorrect number of account_state samples")
            if len(actions) != batch_size:
                raise ValueError("Incorrect number of action samples")
            if len(rewards) != batch_size:
                raise ValueError("Incorrect number of reward samples")
            if len(next_market_data) != batch_size:
                raise ValueError("Incorrect number of next_market_data samples")
            if len(next_account_state) != batch_size:
                raise ValueError("Incorrect number of next_account_state samples")
            if len(dones) != batch_size:
                raise ValueError("Incorrect number of done samples")
            if not all(isinstance(x, np.ndarray) for x in market_data):
                raise TypeError("Market data samples are not all numpy arrays")
            if not all(isinstance(x, np.ndarray) for x in account_state):
                raise TypeError("Account state samples are not all numpy arrays")
            if not all(isinstance(x, np.ndarray) for x in next_market_data):
                raise TypeError("Next market data samples are not all numpy arrays")
            if not all(isinstance(x, np.ndarray) for x in next_account_state):
                raise TypeError("Next account state samples are not all numpy arrays")
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
        if self.debug and not isinstance(batch_priorities_tensor, torch.Tensor):
            raise TypeError("batch_priorities must be a tensor")
        if self.debug and len(tree_indices) != len(batch_priorities_tensor):
            raise ValueError("Indices and priorities length mismatch in update_priorities")

        td_errors = batch_priorities_tensor.detach().cpu().numpy()
        # Calculate new priorities: |TD_error|**alpha + epsilon
        new_priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        if self.debug and np.any(new_priorities <= 0):
            raise ValueError(f"New priority calculated is non-positive: min={new_priorities.min()}")

        # Update priorities in the deque
        for tree_idx, priority in zip(tree_indices, new_priorities, strict=False):
            if priority <= 0:
                if self.debug:
                    raise ValueError("Encountered non-positive priority during update")
                priority = self.epsilon
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
        if self.debug and len(self.buffer) != self.tree.size:
            raise ValueError("Buffer deque length doesn't match SumTree size after load")

    def __len__(self):
        # Return the current fill size of the buffer/tree
        return self.tree.size

    # --- Start: Side-car persistence (memmap-based, O(1) peak RAM) ---
    #
    # Motivation: embedding this buffer inside a ``torch.save(...)`` dict
    # pickles the deque, which (for capacity=1M, ~6 GiB of live arrays)
    # produces a 10+ GiB transient in-memory pickle stream on top of the
    # live buffer. Three separate OOM kills were directly traced to that
    # save-time peak exceeding the system memory ceiling. Writing each
    # field to its own ``.npy`` file via ``np.lib.format.open_memmap``
    # makes the per-field peak bounded by a single Experience row
    # (a few KiB), not by the cumulative buffer size.
    #
    # Layout under ``buffer_dir``:
    #   market_data.npy, next_market_data.npy        (N, W, F) float32
    #   account_state.npy, next_account_state.npy    (N, A)    float32
    #   actions.npy                                  (N,)      int64
    #   rewards.npy                                  (N,)      float32
    #   dones.npy                                    (N,)      uint8
    #   sumtree.npz                                  tree + data_indices + write + size
    #   meta.json                                    small scalar metadata
    #   _COMPLETE                                    zero-byte marker, written last
    #
    # The ``_COMPLETE`` marker is the side-car equivalent of a ZIP central
    # directory -- a resume that finds the directory without this marker
    # treats the side-car as truncated (mirror of
    # ``_probe_checkpoint_usable`` for torch.save files).
    _SIDECAR_FORMAT_VERSION = 1

    def save_to_path(self, buffer_dir: Path | str) -> Path:
        """Persist the replay buffer to ``buffer_dir`` as per-field ``.npy`` memmaps.

        Peak additional RSS during this call is ~O(one Experience row), not
        O(buffer size), because each field is written row-by-row into an
        on-disk memmap. Returns the ``Path`` written to.
        """
        buffer_dir = Path(buffer_dir)
        if buffer_dir.exists():
            # Atomic-ish replace: remove any stale side-car first so a
            # partially-written one can never shadow a previous good copy.
            shutil.rmtree(buffer_dir)
        buffer_dir.mkdir(parents=True, exist_ok=False)

        n = len(self.buffer)
        if n > 0:
            first = self.buffer[0]
            md_shape = np.asarray(first.market_data).shape
            as_shape = np.asarray(first.account_state).shape
            nmd_shape = np.asarray(first.next_market_data).shape
            nas_shape = np.asarray(first.next_account_state).shape

            md_mm = np.lib.format.open_memmap(
                buffer_dir / "market_data.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, *md_shape),
            )
            as_mm = np.lib.format.open_memmap(
                buffer_dir / "account_state.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, *as_shape),
            )
            nmd_mm = np.lib.format.open_memmap(
                buffer_dir / "next_market_data.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, *nmd_shape),
            )
            nas_mm = np.lib.format.open_memmap(
                buffer_dir / "next_account_state.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, *nas_shape),
            )
            actions_mm = np.lib.format.open_memmap(
                buffer_dir / "actions.npy",
                mode="w+",
                dtype=np.int64,
                shape=(n,),
            )
            rewards_mm = np.lib.format.open_memmap(
                buffer_dir / "rewards.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n,),
            )
            dones_mm = np.lib.format.open_memmap(
                buffer_dir / "dones.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(n,),
            )

            try:
                # deque supports indexed access; iterating is O(N) either way
                # but avoids materialising an intermediate list.
                for i, exp in enumerate(self.buffer):
                    md_mm[i] = exp.market_data
                    as_mm[i] = exp.account_state
                    nmd_mm[i] = exp.next_market_data
                    nas_mm[i] = exp.next_account_state
                    actions_mm[i] = int(exp.action)
                    rewards_mm[i] = float(exp.reward)
                    dones_mm[i] = int(bool(exp.done))
            finally:
                for mm in (md_mm, as_mm, nmd_mm, nas_mm, actions_mm, rewards_mm, dones_mm):
                    try:
                        mm.flush()
                    except Exception:  # noqa: BLE001 -- defensive; bubble only via marker absence
                        pass
                # Drop memmap refs so the backing pages can be reclaimed.
                del md_mm, as_mm, nmd_mm, nas_mm, actions_mm, rewards_mm, dones_mm

        np.savez(
            buffer_dir / "sumtree.npz",
            tree=self.tree.tree,
            data_indices=self.tree.data_indices,
            write=np.int64(self.tree.write),
            size=np.int64(self.tree.size),
        )

        meta = {
            "format_version": self._SIDECAR_FORMAT_VERSION,
            "size": int(n),
            "buffer_write_idx": int(self.buffer_write_idx),
            "max_priority": float(self.max_priority),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "beta_start": float(self.beta_start),
            "beta_frames": int(self.beta_frames),
            "epsilon": float(self.epsilon),
            "capacity": int(self.capacity),
        }
        (buffer_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Completion marker MUST be written last. Its presence is the
        # atomic signal to the resume path that this side-car is intact.
        (buffer_dir / "_COMPLETE").write_bytes(b"")
        return buffer_dir

    @classmethod
    def sidecar_is_complete(cls, buffer_dir: Path | str) -> bool:
        """Return True if ``buffer_dir`` looks like a fully-written side-car."""
        buffer_dir = Path(buffer_dir)
        return buffer_dir.is_dir() and (buffer_dir / "_COMPLETE").is_file()

    def load_from_path(self, buffer_dir: Path | str) -> None:
        """Restore buffer contents from a side-car directory previously written by :meth:`save_to_path`.

        Raises ``FileNotFoundError`` if the directory is missing and
        ``ValueError`` if the side-car is incomplete or the capacity doesn't
        match the live buffer's capacity.
        """
        buffer_dir = Path(buffer_dir)
        if not buffer_dir.is_dir():
            raise FileNotFoundError(f"Buffer side-car directory not found: {buffer_dir}")
        if not (buffer_dir / "_COMPLETE").is_file():
            raise ValueError(
                f"Buffer side-car at {buffer_dir} is missing the _COMPLETE marker -- "
                "the previous save was interrupted and this buffer cannot be safely loaded."
            )

        meta = json.loads((buffer_dir / "meta.json").read_text())
        if int(meta["capacity"]) != self.capacity:
            raise ValueError(
                f"Capacity mismatch loading buffer side-car: file has {meta['capacity']}, buffer was constructed with {self.capacity}"
            )

        n = int(meta["size"])
        new_deque: deque = deque(maxlen=self.capacity)
        if n > 0:
            md = np.load(buffer_dir / "market_data.npy", mmap_mode="r")
            ac = np.load(buffer_dir / "account_state.npy", mmap_mode="r")
            nmd = np.load(buffer_dir / "next_market_data.npy", mmap_mode="r")
            nas = np.load(buffer_dir / "next_account_state.npy", mmap_mode="r")
            actions = np.load(buffer_dir / "actions.npy", mmap_mode="r")
            rewards = np.load(buffer_dir / "rewards.npy", mmap_mode="r")
            dones = np.load(buffer_dir / "dones.npy", mmap_mode="r")

            # Copy each row out of the memmap into private (anonymous)
            # memory so subsequent sample() calls don't trigger per-row
            # major page faults during training. ``np.array(x)`` is the
            # standard idiom for materialising a memmap slice.
            for i in range(n):
                new_deque.append(
                    Experience(
                        market_data=np.array(md[i], dtype=np.float32),
                        account_state=np.array(ac[i], dtype=np.float32),
                        action=int(actions[i]),
                        reward=float(rewards[i]),
                        next_market_data=np.array(nmd[i], dtype=np.float32),
                        next_account_state=np.array(nas[i], dtype=np.float32),
                        done=int(dones[i]),
                    )
                )
            # Drop memmap references so the page cache can be reclaimed.
            del md, ac, nmd, nas, actions, rewards, dones
        self.buffer = new_deque

        with np.load(buffer_dir / "sumtree.npz") as tree_state:
            # ``.copy()`` detaches from the npz's lazy loader so it can be
            # closed; SumTree expects owning arrays.
            self.tree.tree = np.array(tree_state["tree"])
            self.tree.data_indices = np.array(tree_state["data_indices"])
            self.tree.write = int(tree_state["write"])
            self.tree.size = int(tree_state["size"])

        self.buffer_write_idx = int(meta["buffer_write_idx"])
        self.max_priority = float(meta["max_priority"])
        self.alpha = float(meta["alpha"])
        self.beta = float(meta["beta"])
        self.beta_start = float(meta["beta_start"])
        self.beta_frames = int(meta["beta_frames"])
        self.epsilon = float(meta["epsilon"])

        if self.debug and len(self.buffer) != self.tree.size:
            raise ValueError("Buffer deque length doesn't match SumTree size after side-car load")

    # --- End: Side-car persistence ---


# --- End: Prioritized Replay Buffer ---

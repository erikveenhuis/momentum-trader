import math
import random
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from momentum_core.logging import get_logger
from torch.amp import autocast

from .agent_checkpoint import (
    AgentCheckpointMixin,
    _load_state_dict_with_orig_mod_fallback,
    _maybe_unwrap_orig_mod_state_dict,
    _maybe_wrap_orig_mod_state_dict,
)
from .agent_compile import compile_networks_or_raise
from .agent_diagnostics import AgentDiagnosticsMixin
from .buffer import PrioritizedReplayBuffer
from .config_schema import AgentConfig
from .constants import ACCOUNT_STATE_DIM  # Import constant
from .model import RainbowNetwork

# Get logger instance
logger = get_logger(__name__)

# Re-exported for backward-compat with external callers / tests.
__all__ = [
    "RainbowDQNAgent",
    "_load_state_dict_with_orig_mod_fallback",
    "_maybe_unwrap_orig_mod_state_dict",
    "_maybe_wrap_orig_mod_state_dict",
]


def _network_forward_with_cls(
    network: torch.nn.Module,
    market_data: torch.Tensor,
    account_state: torch.Tensor,
):
    """Call RainbowNetwork.forward_with_cls, unwrapping torch.compile if needed."""
    target = getattr(network, "_orig_mod", network)
    return target.forward_with_cls(market_data, account_state)


def _network_aux_from_cls(network: torch.nn.Module, cls_out: torch.Tensor) -> torch.Tensor:
    """Call RainbowNetwork.aux_from_cls, unwrapping torch.compile if needed."""
    target = getattr(network, "_orig_mod", network)
    return target.aux_from_cls(cls_out)


# --- Start: Rainbow DQN Agent ---
class RainbowDQNAgent(AgentDiagnosticsMixin, AgentCheckpointMixin):
    """
    Rainbow DQN Agent incorporating:
    - Distributional RL (C51)
    - Prioritized Experience Replay (PER)
    - Dueling Networks (Implicit in RainbowNetwork)
    - Multi-step Returns
    - Double Q-Learning
    - Noisy Nets for exploration
    """

    def __init__(self, config: dict, device: str = "cuda", inference_only: bool = False):
        """
        Initializes the Rainbow DQN Agent.

        Args:
            config (dict): A dictionary containing all hyperparameters and network settings.
            device (str): The device to run the agent on ('cuda' or 'cpu').
            inference_only: If True, build for forward-only use (live trading, CPU allowed, no
                ``torch.compile`` requirement, small replay buffer). Training must use
                ``inference_only=False`` with CUDA.

        Notes:
            The old ``scaler`` parameter (``torch.cuda.amp.GradScaler``) was removed;
            we now use bfloat16 autocast which doesn't need loss scaling. Old
            checkpoints with a ``scaler_state_dict`` entry still load — see
            :meth:`load_state`, which pops the key if present.
        """
        self.inference_only = inference_only
        cfg = dict(config)
        if self.inference_only:
            cfg["batch_size"] = 1
            orig_rb = int(cfg.get("replay_buffer_size", 1_000_000))
            cfg["replay_buffer_size"] = min(orig_rb, 4096)

        # Validate all required keys up front (loud failure, no silent fallbacks).
        # See ``.cursor/rules/no-defaults.mdc`` and ``config_schema.AgentConfig``.
        self._cfg = AgentConfig.from_dict(cfg)

        self.config = cfg
        self.device = self._resolve_device(device)
        config = self.config

        if not self.inference_only:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. This training requires CUDA for optimal performance with torch.compile. "
                    "Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support installed."
                )

            if self.device.type != "cuda":
                raise RuntimeError(
                    f"CUDA device required for training, but got device: {self.device}. "
                    "Please specify device='cuda' or ensure CUDA is available."
                )
        elif self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested for inference_only agent but CUDA is not available (got device={self.device})."
            )

        # All tunables now sourced from the validated dataclass — no more
        # silent-default fallbacks. Keys that fail range-validation in
        # ``AgentConfig.__post_init__``/``_validate_agent_config`` have already
        # raised.
        c = self._cfg
        self.seed = c.seed
        self.gamma = c.gamma
        self.lr = c.lr
        self.batch_size = c.batch_size
        self.target_update_freq = c.target_update_freq
        self.polyak_tau = c.polyak_tau
        self.n_steps = c.n_steps
        self.num_atoms = c.num_atoms
        self.v_min = c.v_min
        self.v_max = c.v_max
        self.num_actions = c.num_actions
        self.window_size = c.window_size
        self.n_features = c.n_features
        self.hidden_dim = c.hidden_dim
        self.replay_buffer_size = c.replay_buffer_size
        self.alpha = c.alpha
        self.beta_start = c.beta_start
        self.beta_frames = c.beta_frames
        self.grad_clip_norm = c.grad_clip_norm
        self.epsilon_start = float(c.epsilon_start)
        self.epsilon_end = float(c.epsilon_end)
        self.epsilon_decay_steps = int(c.epsilon_decay_steps)
        self.entropy_coeff = float(c.entropy_coeff)
        self.store_partial_n_step = c.store_partial_n_step
        self.categorical_logging_interval = int(c.categorical_logging_interval)
        if self.categorical_logging_interval <= 0:
            raise ValueError(
                f"categorical_logging_interval must be a positive integer, got {self.categorical_logging_interval}"
            )
        # Tier 3a/3b/3c/3d diagnostic cadences. Set to 0 in YAML to disable.
        self.noisy_sigma_logging_interval = max(0, int(c.noisy_sigma_logging_interval))
        self.q_value_logging_interval = max(0, int(c.q_value_logging_interval))
        self.q_value_histogram_interval = max(0, int(c.q_value_histogram_interval))
        self.grad_logging_interval = max(0, int(c.grad_logging_interval))
        self.target_net_logging_interval = max(0, int(c.target_net_logging_interval))
        self.td_error_logging_interval = max(0, int(c.td_error_logging_interval))
        # Cumulative counter, surfaced as ``Train/TargetNet/SoftUpdates``. Lets
        # the caller verify Polyak is actually running at the expected cadence.
        self._soft_update_count: int = 0
        filtered_percentiles: list[float] = []
        for value in c.categorical_logging_percentiles:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if 0 < numeric < 100:
                filtered_percentiles.append(numeric)
        if not filtered_percentiles:
            raise ValueError(
                "categorical_logging_percentiles must contain at least one value in (0, 100), "
                f"got {list(c.categorical_logging_percentiles)}"
            )
        self.categorical_logging_percentiles = sorted(filtered_percentiles)
        # Tier 2.1: auxiliary return-prediction head config (required, no default).
        self.aux_loss_weight = float(c.aux_loss_weight)
        self.aux_target_feature_index = int(c.aux_target_feature_index)
        self.debug_mode = c.debug
        self.scaler = None  # GradScaler removed; bfloat16 autocast doesn't need it
        # Setup seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.device.type == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            elif hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
            else:
                logger.warning(
                    "torch.set_float32_matmul_precision is unavailable; TF32 settings fall back to torch defaults."
                )
            if (
                hasattr(torch.backends, "cudnn")
                and hasattr(torch.backends.cudnn, "conv")
                and hasattr(torch.backends.cudnn.conv, "fp32_precision")
            ):
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            logger.info("CUDA seed set. AMP enabled with bfloat16 autocast.")
        else:
            logger.info("Agent on CPU. AMP Disabled.")

        logger.info(f"Initializing RainbowDQNAgent on {self.device}")
        logger.info(f"Device type: {self.device.type}, CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Config: {config}")  # Log the entire config

        # Distributional RL setup
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support_cpu = np.linspace(self.v_min, self.v_max, self.num_atoms, dtype=np.float64)
        self._categorical_target_accumulator = {
            "mass": np.zeros(self.num_atoms, dtype=np.float64),
            "samples": 0,
        }
        # Optional TensorBoard writer injected by the trainer (see logging plan Tier 1c/3a-d).
        # Kept untyped (Any) to avoid forcing a torch.utils.tensorboard import here.
        self.tb_writer: Any = None
        # Default provenance state for batch action selection (Tier 2c).
        self._last_batch_was_greedy: np.ndarray = np.zeros(0, dtype=bool)
        self._last_select_q_values: np.ndarray | None = None

        # Initialize Networks
        # Pass the agent's config dictionary and device directly
        self.network = RainbowNetwork(config=self.config, device=self.device).to(self.device)
        self.target_network = RainbowNetwork(config=self.config, device=self.device).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        if not self.inference_only:
            self.network, self.target_network = compile_networks_or_raise(
                self.network,
                self.target_network,
                device=self.device,
                window_size=self.window_size,
                n_features=self.n_features,
            )
        else:
            logger.info("inference_only=True: skipping torch.compile (eager inference for live / CPU).")

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Learning Rate Scheduler Initialization (moved after optimizer init).
        # All three keys (enabled/type/params) are now required in YAML and
        # surfaced via ``AgentConfig`` — no silent fallback to the old
        # ``StepLR`` default if a config drops them.
        self.lr_scheduler_enabled = self._cfg.lr_scheduler_enabled
        self.scheduler = None
        self._scheduler_requires_metric = False
        if self.lr_scheduler_enabled:
            scheduler_type = self._cfg.lr_scheduler_type
            scheduler_params = dict(self._cfg.lr_scheduler_params)

            # Ensure optimizer is defined before scheduler initialization
            if hasattr(self, "optimizer") and self.optimizer is not None:
                if scheduler_type == "StepLR":
                    # Ensure all required params for StepLR are present or have defaults
                    step_size = scheduler_params.get("step_size")
                    gamma = scheduler_params.get("gamma", 0.1)  # Default gamma if not provided
                    if step_size is None:
                        logger.error("StepLR 'step_size' not provided in scheduler_params. Disabling scheduler.")
                        self.lr_scheduler_enabled = False
                    else:
                        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
                        self._scheduler_requires_metric = False
                elif scheduler_type == "CosineAnnealingLR":
                    t_max = scheduler_params.get("T_max")
                    eta_min = scheduler_params.get("min_lr", 0)  # min_lr maps to eta_min
                    if t_max is None:
                        logger.error("CosineAnnealingLR 'T_max' not provided in scheduler_params. Disabling scheduler.")
                        self.lr_scheduler_enabled = False
                    else:
                        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
                        self._scheduler_requires_metric = False
                # Add other schedulers like ReduceLROnPlateau if needed, with similar param checks
                elif scheduler_type == "ReduceLROnPlateau":
                    # Parameters for ReduceLROnPlateau
                    mode = scheduler_params.get("mode", "min")  # Default to min if not specified
                    factor = scheduler_params.get("factor", 0.1)
                    patience = scheduler_params.get("patience", 10)
                    threshold = scheduler_params.get("threshold", 1e-4)
                    min_lr = scheduler_params.get("min_lr", 0)

                    self.scheduler = lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode=mode,
                        factor=factor,
                        patience=patience,
                        threshold=threshold,
                        min_lr=min_lr,
                    )
                    logger.info(
                        f"Initialized ReduceLROnPlateau with mode='{mode}', factor={factor}, patience={patience}"
                    )
                    self._scheduler_requires_metric = True
                else:
                    logger.warning(f"Unsupported scheduler type: {scheduler_type}. No scheduler will be used.")
                    self.lr_scheduler_enabled = False
            else:
                logger.error("Optimizer not initialized before attempting to create LR scheduler. Disabling scheduler.")
                self.lr_scheduler_enabled = False

            if self.scheduler:
                logger.info(f"Initialized LR scheduler: {scheduler_type} with effective params for {scheduler_type}.")
        else:
            logger.info("LR scheduler is disabled by config.")

        logger.info("Rainbow networks and optimizer created.")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            self.replay_buffer_size,
            self.alpha,
            self.beta_start,
            self.beta_frames,
            debug=self.debug_mode,
        )
        # Per-env N-step return buffers (vectorized training support)
        self._num_envs = 1
        self._n_step_buffers: list[deque] = [deque(maxlen=self.n_steps)]
        self._n_step_needs_reset_flags: list[bool] = [False]
        # Legacy aliases for single-env access and checkpoint compat
        self.n_step_buffer = self._n_step_buffers[0]
        self.n_step_reward_window = deque(maxlen=60)
        self.observed_n_step_rewards_history = [] if self.debug_mode else None

        self.training_mode = True  # Start in training mode by default
        self._network_mode_training: bool | None = None
        self._apply_network_mode(self.training_mode)
        self._non_blocking_copy = self.device.type == "cuda" and torch.cuda.is_available()
        self._market_tensor = torch.zeros(
            (1, self.window_size, self.n_features), device=self.device, dtype=torch.float32
        )
        self._account_tensor = torch.zeros((1, ACCOUNT_STATE_DIM), device=self.device, dtype=torch.float32)
        self._initialize_batch_tensors()
        self.total_steps = 0  # Track total steps for target network updates and beta annealing
        self.env_steps = 0  # Track env steps for epsilon annealing (set by trainer)
        self.last_td_error_stats: dict[str, float] | None = None
        self.last_entropy: float | None = None

    def get_per_stats(self) -> dict[str, float | int]:
        """
        Returns summary statistics for the prioritized replay buffer.

        Keys:
            size (int): Number of experiences currently in the buffer.
            capacity (int): Maximum buffer capacity.
            fill_ratio (float): size / capacity (0 if capacity is 0).
            alpha (float): Current PER alpha exponent.
            beta (float): Current PER beta value.
            beta_progress (float): Fraction of beta annealing completed (0-1).
            total_priority (float): Sum of all priorities in the tree.
            avg_priority (float): Average priority for stored experiences.
            max_priority (float): Maximum priority observed so far.
            total_steps (int): Total learner steps recorded by the agent.
        """
        buffer = self.buffer
        size = len(buffer)
        capacity = getattr(buffer, "capacity", 0)
        fill_ratio = (size / capacity) if capacity else 0.0
        total_priority = float(buffer.tree.total()) if size > 0 else 0.0
        avg_priority = (total_priority / size) if size > 0 else 0.0
        max_priority = float(getattr(buffer, "max_priority", 0.0))
        beta = float(getattr(buffer, "beta", 0.0))
        alpha = float(getattr(buffer, "alpha", 0.0))
        beta_frames = getattr(buffer, "beta_frames", 0)
        beta_progress = min(1.0, self.total_steps / beta_frames) if beta_frames else 1.0

        return {
            "size": size,
            "capacity": capacity,
            "fill_ratio": fill_ratio,
            "alpha": alpha,
            "beta": beta,
            "beta_progress": beta_progress,
            "total_priority": total_priority,
            "avg_priority": avg_priority,
            "max_priority": max_priority,
            "total_steps": self.total_steps,
        }

    @property
    def current_epsilon(self) -> float:
        """Current epsilon for epsilon-greedy exploration, linearly annealed by env_steps."""
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        frac = min(1.0, self.env_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs, action_mask=None):
        """Selects action based on Q-values, with optional action masking.

        Thin wrapper around :meth:`select_action_with_provenance` that drops the
        provenance flag for callers that don't need it. Prefer the provenance
        variant in new code (it is required for Tier 2c TB metrics that split
        action rates by greedy vs epsilon-greedy origin).
        """
        action, _ = self.select_action_with_provenance(obs, action_mask=action_mask)
        return action

    def select_action_with_provenance(self, obs, action_mask=None) -> tuple[int, bool]:
        """Like :meth:`select_action` but also returns ``was_greedy``.

        ``was_greedy`` is True when the returned action was the argmax of Q,
        False when it was overridden by epsilon-greedy random exploration. In
        eval mode (``training_mode=False``) the agent never explores, so the
        flag is always True there.
        """
        if self.debug_mode:
            assert isinstance(obs, dict), "Observation must be a dictionary"
            assert "market_data" in obs and "account_state" in obs, "Observation missing required keys"
            assert isinstance(obs["market_data"], np.ndarray), "obs['market_data'] must be a numpy array"
            assert isinstance(obs["account_state"], np.ndarray), "obs['account_state'] must be a numpy array"
            # Check shapes (before adding batch dimension)
            assert obs["market_data"].shape == (
                self.window_size,
                self.n_features,
            ), (
                f"Input market_data shape mismatch. Expected {(self.window_size, self.n_features)}, got {obs['market_data'].shape}"
            )
            assert obs["account_state"].shape == (ACCOUNT_STATE_DIM,), (
                f"Input account_state shape mismatch. Expected ({ACCOUNT_STATE_DIM},), got {obs['account_state'].shape}"
            )

        # Convert observation to tensors using preallocated buffers
        market_source = torch.from_numpy(obs["market_data"]).to(dtype=self._market_tensor.dtype)
        account_source = torch.from_numpy(obs["account_state"]).to(dtype=self._account_tensor.dtype)

        self._market_tensor[0].copy_(market_source, non_blocking=self._non_blocking_copy)
        self._account_tensor[0].copy_(account_source, non_blocking=self._non_blocking_copy)

        market_data = self._market_tensor
        account_state = self._account_tensor
        if self.debug_mode:
            assert market_data.shape == (
                1,
                self.window_size,
                self.n_features,
            ), "Tensor market_data shape mismatch"
            assert account_state.shape == (
                1,
                ACCOUNT_STATE_DIM,
            ), "Tensor account_state shape mismatch"

        # Select action using the online network (with Noisy Layers for exploration)
        if self.training_mode and hasattr(self.network, "reset_noise"):
            self.network.reset_noise()

        autocast_enabled = self.device.type == "cuda" and torch.cuda.is_available()

        with torch.inference_mode():
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                q_values = self.network.get_q_values(market_data, account_state)
            if self.debug_mode:
                assert q_values.shape == (
                    1,
                    self.num_actions,
                ), f"Q-values shape mismatch. Expected (1, {self.num_actions}), got {q_values.shape}"
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=q_values.device)
                q_values = q_values.clone()
                q_values[0, ~mask_tensor] = float("-inf")
            action = q_values.argmax().item()
            if self.debug_mode:
                assert isinstance(action, int), f"Selected action is not an integer: {action}"
                assert 0 <= action < self.num_actions, (
                    f"Selected action ({action}) is out of bounds [0, {self.num_actions})"
                )
            # Tier 2.2: only the live trader consumes ``_last_select_q_values``.
            # In training_mode, no consumer reads it and the D->H sync every
            # action is pure overhead. Skip the copy unless we're evaluating
            # or live-trading.
            if not self.training_mode:
                self._last_select_q_values = q_values.detach().to(torch.float32).cpu().numpy().reshape(-1)

        was_greedy = True
        if self.training_mode and random.random() < self.current_epsilon:
            was_greedy = False
            if action_mask is not None:
                valid_actions = [i for i, m in enumerate(action_mask) if m]
                action = random.choice(valid_actions) if valid_actions else action
            else:
                action = random.randrange(self.num_actions)

        return action, was_greedy

    # ------------------------------------------------------------------
    # Vectorized environment support
    # ------------------------------------------------------------------

    def set_num_envs(self, num_envs: int) -> None:
        """Configure the agent for *num_envs* parallel environments.

        Replaces per-env n-step buffers.  Safe to call on a resumed agent
        (all deques start empty, matching the lazy-reset behaviour on resume).
        """
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._num_envs = num_envs
        self._n_step_buffers = [deque(maxlen=self.n_steps) for _ in range(num_envs)]
        self._n_step_needs_reset_flags = [False] * num_envs
        self.n_step_buffer = self._n_step_buffers[0]

    def select_actions_batch(self, obs_batch: dict[str, np.ndarray]) -> np.ndarray:
        """Select actions for a batch of N observations (one per env).

        Args:
            obs_batch: dict with ``market_data`` shape ``[N, W, F]`` and
                ``account_state`` shape ``[N, ACCOUNT_STATE_DIM]``.

        Returns:
            ``np.ndarray`` of ints with shape ``(N,)``.
        """
        n = obs_batch["market_data"].shape[0]

        if self.training_mode and hasattr(self.network, "reset_noise"):
            self.network.reset_noise()

        market_t = torch.from_numpy(obs_batch["market_data"]).to(
            device=self.device, dtype=torch.float32, non_blocking=self._non_blocking_copy
        )
        account_t = torch.from_numpy(obs_batch["account_state"]).to(
            device=self.device, dtype=torch.float32, non_blocking=self._non_blocking_copy
        )

        autocast_enabled = self.device.type == "cuda" and torch.cuda.is_available()

        with torch.inference_mode():
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                q_values = self.network.get_q_values(market_t, account_t)  # [N, num_actions]

        actions = q_values.argmax(dim=1).cpu().numpy()  # [N]

        was_greedy = np.ones(n, dtype=bool)
        if self.training_mode:
            eps = self.current_epsilon
            rand_mask = np.random.random(n) < eps
            if rand_mask.any():
                actions[rand_mask] = np.random.randint(0, self.num_actions, size=int(rand_mask.sum()))
                was_greedy[rand_mask] = False

        # Stash provenance on the agent so callers (vectorized trainer) can
        # retrieve it without changing the public return type. Tier 2c.
        self._last_batch_was_greedy = was_greedy
        return actions

    def select_actions_batch_with_provenance(self, obs_batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Like :meth:`select_actions_batch` but also returns the per-env greedy mask."""
        actions = self.select_actions_batch(obs_batch)
        return actions, np.asarray(self._last_batch_was_greedy, dtype=bool)

    # ------------------------------------------------------------------

    def _resolve_device(self, requested_device) -> torch.device:
        """Normalizes the requested device into a torch.device, falling back to CPU if needed."""
        if isinstance(requested_device, torch.device):
            resolved = requested_device
        elif isinstance(requested_device, str):
            try:
                resolved = torch.device(requested_device)
            except (TypeError, RuntimeError) as exc:
                logger.warning(
                    "Invalid device string '%s' (%s). Falling back to CPU.",
                    requested_device,
                    exc,
                )
                resolved = torch.device("cpu")
        elif requested_device is None:
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            logger.warning(f"Unsupported device specification '{requested_device}'. Falling back to CPU.")
            resolved = torch.device("cpu")

        if resolved.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            resolved = torch.device("cpu")

        return resolved

    def _initialize_batch_tensors(self) -> None:
        """Pre-allocate reusable tensors for PER batches to limit GPU allocations."""
        self._batch_market_tensor = torch.empty(
            (self.batch_size, self.window_size, self.n_features),
            device=self.device,
            dtype=torch.float32,
        )
        self._batch_account_tensor = torch.empty(
            (self.batch_size, ACCOUNT_STATE_DIM),
            device=self.device,
            dtype=torch.float32,
        )
        self._batch_next_market_tensor = torch.empty_like(self._batch_market_tensor)
        self._batch_next_account_tensor = torch.empty_like(self._batch_account_tensor)
        self._batch_actions_tensor = torch.empty(
            (self.batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        self._batch_rewards_tensor = torch.empty(
            (self.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self._batch_dones_tensor = torch.empty_like(self._batch_rewards_tensor)
        self._batch_weights_tensor = torch.empty_like(self._batch_rewards_tensor)

    def _prepare_batch_tensors(
        self,
        market_data: np.ndarray,
        account_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_market_data: np.ndarray,
        next_account_state: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Copy numpy arrays sampled from PER into persistent GPU tensors."""
        self._batch_market_tensor.copy_(
            torch.from_numpy(market_data),
            non_blocking=self._non_blocking_copy,
        )
        self._batch_account_tensor.copy_(
            torch.from_numpy(account_state),
            non_blocking=self._non_blocking_copy,
        )
        self._batch_next_market_tensor.copy_(
            torch.from_numpy(next_market_data),
            non_blocking=self._non_blocking_copy,
        )
        self._batch_next_account_tensor.copy_(
            torch.from_numpy(next_account_state),
            non_blocking=self._non_blocking_copy,
        )

        actions_tensor = torch.from_numpy(actions.astype(np.int64, copy=False))
        self._batch_actions_tensor.copy_(actions_tensor, non_blocking=self._non_blocking_copy)

        rewards_tensor = torch.from_numpy(rewards.astype(np.float32, copy=False)).view(-1, 1)
        self._batch_rewards_tensor.copy_(rewards_tensor, non_blocking=self._non_blocking_copy)

        dones_tensor = torch.from_numpy(dones.astype(np.float32, copy=False)).view(-1, 1)
        self._batch_dones_tensor.copy_(dones_tensor, non_blocking=self._non_blocking_copy)

        weights_tensor = torch.from_numpy(weights.astype(np.float32, copy=False)).view(-1, 1)
        self._batch_weights_tensor.copy_(weights_tensor, non_blocking=self._non_blocking_copy)

        return (
            self._batch_market_tensor,
            self._batch_account_tensor,
            self._batch_actions_tensor,
            self._batch_rewards_tensor,
            self._batch_next_market_tensor,
            self._batch_next_account_tensor,
            self._batch_dones_tensor,
            self._batch_weights_tensor,
        )

    def _get_n_step_info(self, buf: deque, steps: int | None = None):
        """Compute (truncated) n-step return from *buf*.

        Args:
            buf: The per-env n-step deque to read from.
            steps: Number of transitions to consider. Defaults to self.n_steps.

        Returns:
            tuple: (state_t, action_t, n_step_reward, next_state_tn, done_tn)
        """
        if steps is None:
            steps = self.n_steps

        if steps < 1:
            raise ValueError("Steps for n-step calculation must be >= 1")
        if len(buf) < steps:
            raise ValueError("Insufficient transitions in n-step buffer")

        transitions = list(buf)[:steps]

        market_data_t, account_state_t, action_t, _, _, _, _ = transitions[0]
        state_t = (market_data_t, account_state_t)

        _, _, _, _, next_market_tn, next_account_tn, done_tn = transitions[-1]
        next_state_tn = (next_market_tn, next_account_tn)

        n_step_reward = 0.0
        discount = 1.0
        for _, _, _, reward_i, _, _, done_i in transitions:
            n_step_reward += discount * reward_i
            if done_i:
                break
            discount *= self.gamma

        if self.debug_mode:
            assert isinstance(state_t, tuple) and len(state_t) == 2, "state_t is not a 2-tuple"
            assert isinstance(state_t[0], np.ndarray) and state_t[0].shape == (
                self.window_size,
                self.n_features,
            ), "state_t[0] (market_data) has wrong type/shape"
            assert isinstance(state_t[1], np.ndarray) and state_t[1].shape == (ACCOUNT_STATE_DIM,), (
                "state_t[1] (account_state) has wrong type/shape"
            )
            assert isinstance(action_t, (int, np.integer)), "action_t is not an integer"
            assert isinstance(n_step_reward, (float, np.float32, np.float64)), "n_step_reward is not a float"
            assert isinstance(next_state_tn, tuple) and len(next_state_tn) == 2, "next_state_tn is not a 2-tuple"
            assert isinstance(next_state_tn[0], np.ndarray) and next_state_tn[0].shape == (
                self.window_size,
                self.n_features,
            ), "next_state_tn[0] (market_data) has wrong type/shape"
            assert isinstance(next_state_tn[1], np.ndarray) and next_state_tn[1].shape == (ACCOUNT_STATE_DIM,), (
                "next_state_tn[1] (account_state) has wrong type/shape"
            )
            assert isinstance(done_tn, (bool, np.bool_)), "done_tn is not a boolean"

        return state_t, action_t, n_step_reward, next_state_tn, done_tn

    def _store_n_step_transition(self, buf: deque, steps: int, *, pop_after: bool) -> None:
        """Compute and store an n-step (or truncated) transition from *buf*."""
        state_t, action_t, n_step_reward, next_state_tn, done_tn = self._get_n_step_info(buf, steps)
        market_data_t, account_state_t = state_t
        next_market_tn, next_account_tn = next_state_tn

        self.n_step_reward_window.append(n_step_reward)
        if self.observed_n_step_rewards_history is not None:
            self.observed_n_step_rewards_history.append(n_step_reward)

        self.buffer.store(
            market_data_t,
            account_state_t,
            action_t,
            n_step_reward,
            next_market_tn,
            next_account_tn,
            done_tn,
        )

        if pop_after and len(buf) > 0:
            buf.popleft()

    def store_transition(self, obs, action, reward, next_obs, done, *, env_id: int = 0):
        """Store experience in the per-env N-step buffer and potentially transfer to PER.

        Args:
            obs, action, reward, next_obs, done: standard transition tuple.
            env_id: index of the parallel environment (default 0 for single-env compat).
        """
        buf = self._n_step_buffers[env_id]

        if self._n_step_needs_reset_flags[env_id]:
            buf.clear()
            self._n_step_needs_reset_flags[env_id] = False

        if self.debug_mode:
            assert isinstance(obs, dict) and "market_data" in obs and "account_state" in obs, (
                "Invalid current observation format"
            )
            assert isinstance(next_obs, dict) and "market_data" in next_obs and "account_state" in next_obs, (
                "Invalid next observation format"
            )
            assert isinstance(obs["market_data"], np.ndarray) and obs["market_data"].shape == (
                self.window_size,
                self.n_features,
            ), f"Invalid obs market data shape {obs['market_data'].shape}"
            assert isinstance(obs["account_state"], np.ndarray) and obs["account_state"].shape == (
                ACCOUNT_STATE_DIM,
            ), f"Invalid obs account state shape {obs['account_state'].shape}"
            assert isinstance(next_obs["market_data"], np.ndarray) and next_obs["market_data"].shape == (
                self.window_size,
                self.n_features,
            ), f"Invalid next_obs market data shape {next_obs['market_data'].shape}"
            assert isinstance(next_obs["account_state"], np.ndarray) and next_obs["account_state"].shape == (
                ACCOUNT_STATE_DIM,
            ), f"Invalid next_obs account state shape {next_obs['account_state'].shape}"
            assert isinstance(action, (int, np.integer)), "Action must be an integer"
            assert isinstance(reward, (float, np.float32, np.float64)), "Reward must be a float"
            assert isinstance(done, (bool, np.bool_)), "Done flag must be boolean"

        transition = (
            obs["market_data"],
            obs["account_state"],
            action,
            reward,
            next_obs["market_data"],
            next_obs["account_state"],
            done,
        )
        buf.append(transition)

        buffer_full = len(buf) == self.n_steps

        if buffer_full and not done:
            self._store_n_step_transition(buf, self.n_steps, pop_after=False)

        if done:
            if buffer_full:
                self._store_n_step_transition(
                    buf,
                    self.n_steps,
                    pop_after=self.store_partial_n_step,
                )

            if self.store_partial_n_step:
                while len(buf) > 0:
                    steps = len(buf)
                    self._store_n_step_transition(buf, steps, pop_after=True)

            self._n_step_needs_reset_flags[env_id] = True

    def _project_target_distribution(self, next_market_data_batch, next_account_state_batch, rewards, dones):
        """
        Computes the projected target distribution for the C51 algorithm.
        Applies the Bellman update for n-steps and projects the resulting
        distribution onto the fixed support atoms.

        Args:
            next_market_data_batch (torch.Tensor): Batch of next market data states (s_{t+n}).
            next_account_state_batch (torch.Tensor): Batch of next account states (s_{t+n}).
            rewards (torch.Tensor): Batch of n-step rewards (G_t^(n)), shape [B, 1].
            dones (torch.Tensor): Batch of done flags (d_{t+n}), shape [B, 1].

        Returns:
            torch.Tensor: The projected target distribution (m) with shape [batch_size, num_atoms].
        """
        with torch.no_grad():
            # Double DQN: Use online network to select best next action's index at state s_{t+n}
            next_q_values = self.network.get_q_values(next_market_data_batch, next_account_state_batch)
            if self.debug_mode:
                assert next_q_values.shape == (
                    self.batch_size,
                    self.num_actions,
                ), "Next Q-values shape mismatch"
            next_actions = next_q_values.argmax(dim=1)  # [batch_size]
            if self.debug_mode:
                assert next_actions.shape == (self.batch_size,), "Next actions shape mismatch"

            # Get next state's distribution Z(s_{t+n}, a*) from target network for selected actions a*
            next_log_dist = self.target_network(
                next_market_data_batch, next_account_state_batch
            )  # [B, num_actions, num_atoms]
            if self.debug_mode:
                assert next_log_dist.shape == (
                    self.batch_size,
                    self.num_actions,
                    self.num_atoms,
                ), "Next log distribution shape mismatch"
                assert next_actions.max() < self.num_actions and next_actions.min() >= 0, "Invalid next_action indices"

            # Get the probability distribution for the chosen actions: p(s_{t+n}, a*)
            next_dist = torch.exp(next_log_dist[range(self.batch_size), next_actions])  # [B, num_atoms]
            if self.debug_mode:
                assert next_dist.shape == (
                    self.batch_size,
                    self.num_atoms,
                ), "Next distribution shape mismatch"

            # Compute the projected Bellman target T_z = G_t^(n) + gamma^n * Z(s_{t+n}, a*)
            # Rewards are [B, 1], dones are [B, 1], support is [num_atoms]
            # Broadcasting applies correctly.
            Tz = rewards + (1 - dones) * (self.gamma**self.n_steps) * self.support  # [B, num_atoms]
            if self.debug_mode:
                assert Tz.shape == (
                    self.batch_size,
                    self.num_atoms,
                ), f"Projected Tz shape mismatch: {Tz.shape}"
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # Compute projection indices and weights
            b = (Tz - self.v_min) / self.delta_z  # Normalized position on support axis [B, num_atoms]
            if self.debug_mode:
                assert b.shape == (
                    self.batch_size,
                    self.num_atoms,
                ), f"Projection 'b' shape mismatch: {b.shape}"
            lower_atom_idx = b.floor().long()
            u = b.ceil().long()  # Upper atom index
            # Fix disappearing probability mass when l = b = u (b is int)
            lower_atom_idx[(u > 0) * (lower_atom_idx == u)] -= 1
            u[(lower_atom_idx < (self.num_atoms - 1)) * (lower_atom_idx == u)] += 1

            # Distribute probability
            m = torch.zeros_like(next_dist)
            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size)
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.num_atoms)
                .to(self.device)
            )

            # Ensure indices are within bounds [0, num_atoms - 1]
            lower_atom_idx = lower_atom_idx.clamp(0, self.num_atoms - 1)
            u = u.clamp(0, self.num_atoms - 1)

            # Debugging shapes right before indexing
            # logger.debug(f"Shapes before indexing: m={m.shape}, offset={offset.shape}, l={l.shape}, u={u.shape}, next_dist={next_dist.shape}, b={b.shape}")
            # logger.debug(f"Indices: l max={l.max()}, min={l.min()}; u max={u.max()}, min={u.min()}")

            m.view(-1).index_add_(
                0,
                (lower_atom_idx + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1),
            )
            m.view(-1).index_add_(
                0,
                (u + offset).view(-1),
                (next_dist * (b - lower_atom_idx.float())).view(-1),
            )

            # Optional: Check if target distribution sums to 1 (approximately)
            # This check can be expensive, use cautiously or only in debug mode
            if self.debug_mode:
                sums = m.sum(dim=1)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
                    logger.warning(
                        f"Target distribution M does not sum to 1. Sums: {sums}. Min sum: {sums.min()}, Max sum: {sums.max()}"
                    )
                    # It might not sum *exactly* to 1 due to floating point, clamping, and edge cases.
                    # A small tolerance is usually acceptable.

            return m

    def reset_noisy_sigma(self, std_init: float | None = None) -> int:
        """Re-initialise sigma parameters of every NoisyLinear layer in-place.

        Mu (the deterministic part learned by gradient descent) is left
        untouched, so the policy keeps everything it has learned, but
        exploration is re-energised. This is the canonical recovery move when
        :meth:`_log_noisy_sigma_stats` shows that sigma has collapsed and the
        agent has effectively stopped exploring.

        The same reset is applied to both the online and target networks so
        their NoisyLinear layers stay structurally identical (the target
        network samples its own epsilon, but the sigma scale must match).
        After the sigma refill we call ``reset_noise()`` to immediately
        refresh epsilon buffers so the next forward pass uses the new noise
        scale.

        Args:
            std_init: Sigma scale to use. ``None`` means "use whatever value
                each layer was constructed with" (so heterogeneous configs
                are preserved). A float overrides every layer uniformly.

        Returns:
            The number of NoisyLinear layers that had their sigma reset
            (counted on the online network only — the target network mirrors
            the same count).
        """
        from .model import NoisyLinear

        def _reset_one(net: torch.nn.Module) -> int:
            count = 0
            inner = getattr(net, "_orig_mod", net)
            for module in inner.modules():
                if not isinstance(module, NoisyLinear):
                    continue
                effective_std = float(std_init) if std_init is not None else float(module.std_init)
                in_f = int(module.in_features)
                out_f = int(module.out_features)
                with torch.no_grad():
                    module.weight_sigma.data.fill_(effective_std / math.sqrt(in_f))
                    module.bias_sigma.data.fill_(effective_std / math.sqrt(out_f))
                if std_init is not None:
                    module.std_init = effective_std
                module.reset_noise()
                count += 1
            return count

        online_count = _reset_one(self.network) if self.network is not None else 0
        target_count = _reset_one(self.target_network) if self.target_network is not None else 0
        if online_count != target_count:
            logger.warning(
                "reset_noisy_sigma: online/target NoisyLinear counts differ (online=%d target=%d).",
                online_count,
                target_count,
            )
        logger.info(
            "Reset NoisyLinear sigma on %d online + %d target layer(s) (std_init=%s).",
            online_count,
            target_count,
            "per-layer" if std_init is None else f"{float(std_init):.4f}",
        )
        if self.tb_writer is not None and online_count > 0:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Agent/NoisySigmaReset", float(online_count), step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to log Agent/NoisySigmaReset: %s", exc)
        return online_count

    def _compute_loss(self, batch, weights):
        """Computes the C51 loss using PER weights."""
        (
            market_data,
            account_state,
            actions,
            rewards,
            next_market_data,
            next_account_state,
            dones,
        ) = batch
        if self.debug_mode:
            # --- Start: Assert batch shapes and types ---
            assert market_data.shape == (
                self.batch_size,
                self.window_size,
                self.n_features,
            ), "Batch market_data shape mismatch"
            assert account_state.shape == (
                self.batch_size,
                ACCOUNT_STATE_DIM,
            ), "Batch account_state shape mismatch"
            assert actions.shape == (self.batch_size,), "Batch actions shape mismatch"
            assert rewards.shape == (self.batch_size,), "Batch rewards shape mismatch"
            assert next_market_data.shape == (
                self.batch_size,
                self.window_size,
                self.n_features,
            ), "Batch next_market_data shape mismatch"
            assert next_account_state.shape == (
                self.batch_size,
                ACCOUNT_STATE_DIM,
            ), "Batch next_account_state shape mismatch"
            assert dones.shape == (self.batch_size,), "Batch dones shape mismatch"
            assert weights.shape == (self.batch_size,), "Batch weights shape mismatch"
            # --- End: Assert batch shapes and types ---

        (
            market_data_batch,
            account_state_batch,
            actions_batch,
            rewards_batch,
            next_market_data_batch,
            next_account_state_batch,
            dones_batch,
            weights_batch,
        ) = self._prepare_batch_tensors(
            market_data,
            account_state,
            actions,
            rewards,
            next_market_data,
            next_account_state,
            dones,
            weights,
        )

        if self.debug_mode:
            # --- Start: Assert tensor shapes after conversion ---
            assert market_data_batch.shape == (
                self.batch_size,
                self.window_size,
                self.n_features,
            ), "Tensor market_data_batch shape mismatch"
            assert account_state_batch.shape == (
                self.batch_size,
                ACCOUNT_STATE_DIM,
            ), "Tensor account_state_batch shape mismatch"
            assert next_market_data_batch.shape == (
                self.batch_size,
                self.window_size,
                self.n_features,
            ), "Tensor next_market_data_batch shape mismatch"
            assert next_account_state_batch.shape == (
                self.batch_size,
                ACCOUNT_STATE_DIM,
            ), "Tensor next_account_state_batch shape mismatch"
            assert actions_batch.shape == (self.batch_size,), "Tensor actions_batch shape mismatch"
            assert rewards_batch.shape == (
                self.batch_size,
                1,
            ), "Tensor rewards_batch shape mismatch"
            assert dones_batch.shape == (
                self.batch_size,
                1,
            ), "Tensor dones_batch shape mismatch"
            assert weights_batch.shape == (
                self.batch_size,
                1,
            ), "Tensor weights_batch shape mismatch"
            # --- End: Assert tensor shapes ---

        amp_enabled = self.device.type == "cuda" and torch.cuda.is_available()

        with autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            target_distribution = self._project_target_distribution(
                next_market_data_batch, next_account_state_batch, rewards_batch, dones_batch
            )
            if self.debug_mode:
                assert target_distribution.shape == (
                    self.batch_size,
                    self.num_atoms,
                ), "Target distribution shape mismatch"
            self._accumulate_categorical_target_stats(target_distribution)
            # ---------------------------------------- #

            # --- Calculate Online Distribution and Loss --- #
            # Tier 2.1: one encoder pass feeds both the distributional head and
            # the auxiliary return-prediction head. The old code called
            # ``self.network.predict_return(...)`` afterwards, forcing a second
            # full Transformer forward pass at ~128 ms/step at batch 8192.
            log_ps, cls_out = _network_forward_with_cls(self.network, market_data_batch, account_state_batch)
            if self.debug_mode:
                assert log_ps.shape == (
                    self.batch_size,
                    self.num_actions,
                    self.num_atoms,
                ), "Online log_ps shape mismatch"

            # Tier 3b: stash the per-action expected Q-values from the *online* net
            # so _log_q_value_stats can mirror them to TB without recomputing.
            # Detached + cpu-bound to avoid keeping the autograd graph alive.
            with torch.no_grad():
                self._last_batch_q = (torch.exp(log_ps) * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2).detach()

            # Gather the log-probabilities for the actions actually taken: log Z(s_t, a_t)
            # We need to select the log probabilities corresponding to actions_batch
            actions_indices = actions_batch.view(self.batch_size, 1, 1).expand(self.batch_size, 1, self.num_atoms)
            log_ps_a = log_ps.gather(1, actions_indices).squeeze(1)  # [B, num_atoms]
            if self.debug_mode:
                assert log_ps_a.shape == (
                    self.batch_size,
                    self.num_atoms,
                ), "Online log_ps_a shape mismatch"

            # Calculate cross-entropy loss between target and online distributions
            # Loss = -sum_i [ target_distribution_i * log(online_distribution_i) ]
            # Target distribution is detached as it acts as the label.
            loss_elementwise = -(target_distribution.detach() * log_ps_a).sum(dim=1)  # [B]
            if self.debug_mode:
                assert loss_elementwise.shape == (self.batch_size,), "Per-sample loss shape mismatch"

            loss = (loss_elementwise * weights_batch.squeeze(1).detach()).mean()
            if loss.ndim != 0:
                raise RuntimeError("Final loss is not a scalar")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss calculation resulted in NaN or Inf: {loss.item()}")

            # Tier 2.1: auxiliary return-prediction head reuses the CLS
            # encoding computed above. Previous implementation called
            # ``predict_return`` which ran the full Transformer encoder a
            # second time on the same batch.
            aux_pred = _network_aux_from_cls(self.network, cls_out)
            log_ret_col = self.aux_target_feature_index
            if market_data_batch.shape[2] > log_ret_col:
                target_return = market_data_batch[:, -1, log_ret_col].detach()
                aux_loss = torch.nn.functional.mse_loss(aux_pred, target_return)
                loss = loss + self.aux_loss_weight * aux_loss

            # --- Action entropy regularization --- #
            if self.entropy_coeff > 0:
                all_q = (torch.exp(log_ps) * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [B, num_actions]
                q_range = (all_q.max(dim=1, keepdim=True).values - all_q.min(dim=1, keepdim=True).values).clamp(
                    min=1e-6
                )
                q_normalized = (all_q - all_q.mean(dim=1, keepdim=True)) / q_range
                policy = torch.softmax(q_normalized, dim=1)
                log_policy = torch.log(policy + 1e-8)
                entropy = -(policy * log_policy).sum(dim=1).mean()
                loss = loss - self.entropy_coeff * entropy
                self.last_entropy = float(entropy.detach())
            # ----------------------------------------- #

            # --- Calculate TD errors for PER update --- #
            # TD error is | E[Target Distribution] - E[Online Distribution for a_t] |
            # Calculate expected Q-values E[Z(s_t, a_t)] from the online distribution for the action taken
            q_values_online = (torch.exp(log_ps_a) * self.support.unsqueeze(0)).sum(dim=1)  # [B]
            # Calculate expected target Q-values E[Projected Target Distribution]
            q_values_target = (target_distribution * self.support.unsqueeze(0)).sum(dim=1)  # [B]
            # TD error = |Target Q - Online Q|
            td_errors_tensor = (q_values_target.detach() - q_values_online.detach()).abs()
            if self.debug_mode:
                assert td_errors_tensor.shape == (self.batch_size,), "TD errors tensor shape mismatch"
                assert torch.isfinite(td_errors_tensor).all(), "NaN or Inf found in TD errors tensor"
            # ----------------------------------------- #

        loss = loss.float()
        td_errors_tensor = td_errors_tensor.float()

        # Tier 2.2: only sync TD-error mean/std to host when the logging
        # interval fires. Previously this copied the whole batch every step
        # even though ``last_td_error_stats`` was only read every 100 learn
        # steps. Keep the reductions on device and use ``.item()`` so only two
        # scalars cross the bus.
        if self.td_error_logging_interval > 0 and (self.total_steps % self.td_error_logging_interval == 0):
            with torch.no_grad():
                td_mean_t = td_errors_tensor.mean()
                td_std_t = td_errors_tensor.std(unbiased=False)
            self.last_td_error_stats = {
                "mean": float(td_mean_t.item()),
                "std": float(td_std_t.item()),
            }

        return loss, td_errors_tensor

    def learn(self):
        """Samples from PER, computes loss, updates network, priorities, and target network."""
        if len(self.buffer) < self.batch_size:
            return None  # Not enough samples to learn yet

        # Sample batch from PER
        # Update beta (annealing) based on total steps *before* sampling
        self.buffer.update_beta(self.total_steps)
        beta = self.buffer.beta  # Get current beta for logging
        batch_tuple, tree_indices, weights = self.buffer.sample(self.batch_size)  # Sample returns tree_indices now

        if batch_tuple is None:
            logger.warning("PER sample returned None.")
            return None  # Should not happen if buffer size check passed, but safeguard

        # Compute loss and TD errors (TD errors are tensors)
        loss, td_errors_tensor = self._compute_loss(batch_tuple, weights)
        if self.debug_mode:
            assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Loss from _compute_loss is not a scalar tensor"
            assert isinstance(td_errors_tensor, torch.Tensor) and td_errors_tensor.shape == (self.batch_size,), (
                "TD errors tensor from _compute_loss has wrong shape/type"
            )

        # Log GPU memory usage if using CUDA (less frequently to reduce log spam)
        if self.device.type == "cuda" and torch.cuda.is_available() and self.total_steps % 100 == 0:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            logger.debug(f"GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(f"NaN or Inf detected in gradients BEFORE clipping for parameter: {p.shape}")

        # Capture the *pre-clip* global grad norm; clip_grad_norm_ returns it.
        # Tier 3c: this is the canonical "is the loss healthy?" telemetry.
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(
                        f"NaN or Inf detected in gradients AFTER clipping (Max Norm: {self.grad_clip_norm}) for parameter: {p.shape}"
                    )

        # Tier 3c: gradient + param-update-ratio diagnostics, throttled. Must
        # happen *before* optimizer.step() so the param tensors haven't moved
        # yet and the grad/param ratio is meaningful.
        if self.grad_logging_interval > 0 and (self.total_steps + 1) % self.grad_logging_interval == 0:
            self._log_grad_stats(pre_clip_norm)

        self.optimizer.step()

        # Step LR scheduler when appropriate (no-op for schedulers that require validation metrics)
        self._step_scheduler()

        # Tier 2.2: the buffer accepts the raw TD-error tensor; the prior
        # ``priorities = td_errors_tensor.cpu().numpy() + 1e-6`` block was a
        # dead D->H sync whose result was never read. Deleted.
        self.buffer.update_priorities(tree_indices, td_errors_tensor)

        # Reset noise in Noisy Linear layers (important!)
        self.network.reset_noise()
        self.target_network.reset_noise()

        # Increment step counter and update target network periodically
        self.total_steps += 1
        if self.categorical_logging_interval > 0 and self.total_steps % self.categorical_logging_interval == 0:
            self._log_categorical_target_stats()
        if self.noisy_sigma_logging_interval > 0 and self.total_steps % self.noisy_sigma_logging_interval == 0:
            self._log_noisy_sigma_stats()
        if self.q_value_logging_interval > 0 and self.total_steps % self.q_value_logging_interval == 0:
            emit_hist = self.q_value_histogram_interval > 0 and self.total_steps % self.q_value_histogram_interval == 0
            self._log_q_value_stats(emit_histogram=emit_hist)
        if self.last_td_error_stats is not None and self.total_steps % 100 == 0:
            logger.info(
                "TD error stats at learn step %s - mean: %.6f, std: %.6f",
                self.total_steps,
                self.last_td_error_stats.get("mean", float("nan")),
                self.last_td_error_stats.get("std", float("nan")),
            )
        self._update_target_network()
        if self.total_steps % self.target_update_freq == 0:
            logger.info(f"Step {self.total_steps}: Target network soft-updated (tau={self.polyak_tau}).")

        loss_item = loss.item()
        if not isinstance(loss_item, float) or np.isnan(loss_item) or np.isinf(loss_item):
            raise FloatingPointError("Final loss item is not a valid float")

        # Log loss and PER beta
        logger.debug(f"Step: {self.total_steps}, Loss: {loss_item:.4f}, PER Beta: {beta:.4f}")

        # Periodic n-step reward window diagnostics (logger + TensorBoard, see Tier 1e).
        if self.total_steps % 60 == 0:
            self._log_n_step_reward_window_stats()

        return loss_item  # Return loss for external logging/monitoring

    def _step_scheduler(self, metric: float | None = None) -> bool:
        """Internal helper that steps the scheduler when conditions are satisfied."""
        if not self.scheduler or not self.lr_scheduler_enabled:
            return False

        try:
            if self._scheduler_requires_metric:
                if metric is None:
                    return False
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
            return True
        except Exception as error:
            logger.error("Failed to step LR scheduler: %s", error, exc_info=True)
            return False

    def step_lr_scheduler(self, metric: float):
        """Public hook to step schedulers that require an evaluation metric."""
        stepped = self._step_scheduler(metric)
        if not stepped and self.scheduler and self.lr_scheduler_enabled and not self._scheduler_requires_metric:
            logger.debug(
                "step_lr_scheduler called with metric for scheduler type %s that does not require one.",
                type(self.scheduler),
            )

    def _update_target_network(self):
        """Polyak-averaged (soft) update: target <- tau * online + (1 - tau) * target.

        Tier 3d: emits ``Train/TargetNet/SoftUpdates`` (cumulative counter) and
        ``Train/TargetNet/ParamDeviation`` (pre-update L2 norm of
        ``online - target``). The deviation is captured *before* the Polyak
        blend so it actually reflects "how far apart were we?" — after the
        blend it's monotonically smaller by construction.
        """
        tau = self.polyak_tau
        self._soft_update_count += 1

        should_log = (
            self.tb_writer is not None
            and self.target_net_logging_interval > 0
            and self._soft_update_count % self.target_net_logging_interval == 0
        )
        deviation_sq: float | None = None
        if should_log:
            try:
                acc = 0.0
                for tp, op in zip(self.target_network.parameters(), self.network.parameters(), strict=False):
                    acc += float((op.data - tp.data).pow(2).sum().item())
                deviation_sq = acc
            except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to compute target-net param deviation: %s", exc)
                deviation_sq = None

        for tp, op in zip(self.target_network.parameters(), self.network.parameters(), strict=False):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
        logger.debug(f"Target network soft-updated (tau={tau}).")

        if should_log and self.tb_writer is not None:
            step = int(self.total_steps)
            try:
                self.tb_writer.add_scalar("Train/TargetNet/SoftUpdates", float(self._soft_update_count), step)
                if deviation_sq is not None and math.isfinite(deviation_sq):
                    self.tb_writer.add_scalar("Train/TargetNet/ParamDeviation", math.sqrt(deviation_sq), step)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror target-net diagnostics: %s", exc)

    def set_training_mode(self, training=True):
        """Sets the agent and network to training or evaluation mode."""
        if training == self.training_mode:
            return

        self.training_mode = training
        self._apply_network_mode(training)
        mode = "TRAINING" if training else "EVALUATION"
        logger.info(f"Set agent to {mode} mode")

    @contextmanager
    def greedy(self):
        """Context manager that puts the agent in fully deterministic eval mode.

        Tier 2d: makes the "greedy" rollout an explicit, named operation so
        validation/test/eval-script paths read identically. Inside the block:

        * ``training_mode`` is False → epsilon-greedy override is bypassed in
          :meth:`select_action_with_provenance`.
        * ``self.network`` is in eval mode → :class:`NoisyLinear` uses ``mu``
          weights only (sigma path is frozen out of the forward pass).
        * Any pre-existing ``weight_epsilon`` / ``bias_epsilon`` buffers are
          left in place; they are unused while ``training=False``, but we
          re-seed them on exit so a subsequent training step still starts from
          a fresh noise sample (matches the existing per-step ``reset_noise``
          discipline in :meth:`select_action_with_provenance`).

        Restoration is exception-safe: training mode is restored even on error.
        """
        was_training = self.training_mode
        try:
            self.set_training_mode(False)
            yield self
        finally:
            self.set_training_mode(was_training)
            if was_training and self.network is not None and hasattr(self.network, "reset_noise"):
                # Re-seed NoisyNet samples so the very next training step does
                # not reuse the snapshot that was active before we entered the
                # greedy block.
                try:
                    self.network.reset_noise()
                except (AttributeError, RuntimeError):  # pragma: no cover - defensive
                    logger.debug("Failed to reset NoisyNet noise after greedy() exit", exc_info=True)

    def _apply_network_mode(self, training: bool) -> None:
        if self.network is None:
            return

        if self._network_mode_training == training:
            return

        if training:
            self.network.train()
        else:
            self.network.eval()
            if self.target_network is not None:
                self.target_network.eval()

        self._network_mode_training = training

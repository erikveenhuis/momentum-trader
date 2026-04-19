import math
import os  # Added for save/load path handling
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

from .buffer import PrioritizedReplayBuffer
from .constants import ACCOUNT_STATE_DIM  # Import constant
from .model import RainbowNetwork

# Get logger instance
logger = get_logger("Agent")


def _maybe_unwrap_orig_mod_state_dict(state_dict: dict) -> dict:
    """If ``state_dict`` was saved from ``torch.compile`` (``OptimizedModule``), keys are ``_orig_mod.*``; map to plain module keys."""

    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(isinstance(k, str) and k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def _maybe_wrap_orig_mod_state_dict(state_dict: dict) -> dict:
    """Reverse of :func:`_maybe_unwrap_orig_mod_state_dict`.

    If ``state_dict`` has plain keys but we're loading into an
    ``OptimizedModule``, add the ``_orig_mod.`` prefix back. Returns the
    original dict unchanged if any key already has the prefix (so we don't
    double-prefix).
    """

    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if any(isinstance(k, str) and k.startswith("_orig_mod.") for k in keys):
        return state_dict
    return {f"_orig_mod.{k}": v for k, v in state_dict.items()}


def _load_state_dict_with_orig_mod_fallback(module: torch.nn.Module, state_dict: dict) -> None:
    try:
        module.load_state_dict(state_dict)
    except Exception:
        unwrapped = _maybe_unwrap_orig_mod_state_dict(state_dict)
        if unwrapped is not state_dict:
            try:
                module.load_state_dict(unwrapped)
                logger.info("Loaded weights from torch.compile checkpoint (_orig_mod keys) into eager module.")
                return
            except Exception:
                pass
        wrapped = _maybe_wrap_orig_mod_state_dict(state_dict)
        if wrapped is not state_dict:
            module.load_state_dict(wrapped)
            logger.info("Loaded weights from eager checkpoint (plain keys) into torch.compile module.")
            return
        raise


# --- Start: Rainbow DQN Agent ---
class RainbowDQNAgent:
    """
    Rainbow DQN Agent incorporating:
    - Distributional RL (C51)
    - Prioritized Experience Replay (PER)
    - Dueling Networks (Implicit in RainbowNetwork)
    - Multi-step Returns
    - Double Q-Learning
    - Noisy Nets for exploration
    """

    def __init__(self, config: dict, device: str = "cuda", scaler=None, inference_only: bool = False):
        """
        Initializes the Rainbow DQN Agent.

        Args:
            config (dict): A dictionary containing all hyperparameters and network settings.
            device (str): The device to run the agent on ('cuda' or 'cpu').
            scaler: Deprecated, kept for checkpoint compatibility. Ignored.
            inference_only: If True, build for forward-only use (live trading, CPU allowed, no
                ``torch.compile`` requirement, small replay buffer). Training must use
                ``inference_only=False`` with CUDA.
        """
        self.inference_only = inference_only
        cfg = dict(config)
        if self.inference_only:
            cfg["batch_size"] = 1
            orig_rb = int(cfg.get("replay_buffer_size", 1_000_000))
            cfg["replay_buffer_size"] = min(orig_rb, 4096)

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
            raise RuntimeError(f"CUDA device requested for inference_only agent but CUDA is not available (got device={self.device}).")

        # Use direct access for mandatory parameters
        self.seed = config["seed"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.polyak_tau = config.get("polyak_tau", 0.005)
        self.n_steps = config["n_steps"]
        self.num_atoms = config["num_atoms"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        self.num_actions = config["num_actions"]
        self.window_size = config["window_size"]
        self.n_features = config["n_features"]
        self.hidden_dim = config["hidden_dim"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.alpha = config["alpha"]
        self.beta_start = config["beta_start"]
        self.beta_frames = config["beta_frames"]
        self.grad_clip_norm = config["grad_clip_norm"]
        self.epsilon_start = float(config["epsilon_start"])
        self.epsilon_end = float(config["epsilon_end"])
        self.epsilon_decay_steps = int(config["epsilon_decay_steps"])
        self.entropy_coeff = float(config["entropy_coeff"])
        self.store_partial_n_step = self.config.get("store_partial_n_step", False)
        self.categorical_logging_interval = int(config.get("categorical_logging_interval", 2000))
        if self.categorical_logging_interval <= 0:
            logger.warning("categorical_logging_interval must be a positive integer; falling back to default of 2000 learner steps.")
            self.categorical_logging_interval = 2000
        # Tier 3a: NoisyNet sigma stats. Default cadence matches categorical
        # logging so the diagnostics line up on the same time axis. Set to 0 to
        # disable.
        self.noisy_sigma_logging_interval = int(config.get("noisy_sigma_logging_interval", self.categorical_logging_interval))
        if self.noisy_sigma_logging_interval < 0:
            logger.warning("noisy_sigma_logging_interval must be >= 0; disabling NoisyNet sigma logging.")
            self.noisy_sigma_logging_interval = 0
        # Tier 3b: Q-value scalar stats are cheap; default to log every 250
        # learn steps. The histogram is far heavier so it gets its own (longer)
        # cadence (~10x slower).
        self.q_value_logging_interval = int(config.get("q_value_logging_interval", 250))
        if self.q_value_logging_interval < 0:
            logger.warning("q_value_logging_interval must be >= 0; disabling Q-value logging.")
            self.q_value_logging_interval = 0
        self.q_value_histogram_interval = int(config.get("q_value_histogram_interval", max(self.q_value_logging_interval * 10, 2500)))
        if self.q_value_histogram_interval < 0:
            self.q_value_histogram_interval = 0
        # Tier 3c: gradient norms + param-update-ratio. Same-cadence as Q-value
        # scalars by default — both are cheap and benefit from being plotted on
        # a shared time axis.
        self.grad_logging_interval = int(config.get("grad_logging_interval", self.q_value_logging_interval or 250))
        if self.grad_logging_interval < 0:
            logger.warning("grad_logging_interval must be >= 0; disabling gradient diagnostics.")
            self.grad_logging_interval = 0
        # Tier 3d: target-network deviation. Polyak runs every learn step so we
        # MUST throttle — the L2 sweep over every param tensor is otherwise the
        # most expensive thing in the train loop. Default = same cadence as
        # gradient stats, since these are diagnostic siblings.
        self.target_net_logging_interval = int(config.get("target_net_logging_interval", self.grad_logging_interval or 250))
        if self.target_net_logging_interval < 0:
            logger.warning("target_net_logging_interval must be >= 0; disabling target-net diagnostics.")
            self.target_net_logging_interval = 0
        # Cumulative counter, surfaced as ``Train/TargetNet/SoftUpdates``. Lets
        # the caller verify Polyak is actually running at the expected cadence.
        self._soft_update_count: int = 0
        raw_percentiles = config.get("categorical_logging_percentiles", [5, 25, 50, 75, 95])
        filtered_percentiles: list[float] = []
        for value in raw_percentiles if isinstance(raw_percentiles, (list, tuple)) else [raw_percentiles]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if 0 < numeric < 100:
                filtered_percentiles.append(numeric)
        if filtered_percentiles:
            self.categorical_logging_percentiles = sorted(filtered_percentiles)
        else:
            self.categorical_logging_percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]
        # Optional flags can still use .get()
        self.debug_mode = config.get("debug", False)
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
                logger.warning("torch.set_float32_matmul_precision is unavailable; TF32 settings fall back to torch defaults.")
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

        # # Check if PyTorch version >= 2.0 to use torch.compile
        # if int(torch.__version__.split('.')[0]) >= 2:
        #     logger.info("Applying torch.compile to network and target_network.")
        #     # Add error handling in case compile fails on specific setups
        #     try:
        #         # Try the default compilation mode first
        #         self.network = torch.compile(self.network)
        #         self.target_network = torch.compile(self.target_network)
        #         logger.info("torch.compile applied successfully with default mode.")
        #     except ImportError as imp_err: # Catch potential import errors if compile isn't fully set up
        #          logger.warning(f"torch.compile skipped due to potential import issue: {imp_err}. Proceeding without compilation.")
        #     except Exception as e:
        #          # Check if it's the TritonMissing error specifically
        #          # Check class name string as direct import might fail if torch._inductor isn't available
        #          if "TritonMissing" in str(e.__class__):
        #              logger.warning(f"torch.compile failed as Triton backend is not available (common on non-Linux/CUDA setups): {e}. Proceeding without compilation.")
        #          else:
        #              logger.warning(f"torch.compile failed with an unexpected error: {e}. Proceeding without compilation.")
        # else:
        #     logger.warning("torch version < 2.0, torch.compile not available.")

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        if not self.inference_only:
            # REQUIRE torch.compile for optimal performance (training path)
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "torch.compile is not available in this PyTorch version. Please upgrade to PyTorch 2.0+ for optimal performance."
                )

            logger.info("Applying torch.compile to network and target_network (REQUIRED for training).")
            compile_success = False

            # Try different compilation modes in order of preference
            compile_modes = ["default", "reduce-overhead", "max-autotune"]

            for mode in compile_modes:
                try:
                    logger.info(f"Trying torch.compile with mode='{mode}'...")
                    compiled_network = torch.compile(self.network, mode=mode)
                    compiled_target_network = torch.compile(self.target_network, mode=mode)

                    # Test compilation by running a small forward pass
                    with torch.no_grad():
                        test_market = torch.zeros((1, self.window_size, self.n_features), device=self.device)
                        test_account = torch.zeros((1, ACCOUNT_STATE_DIM), device=self.device)
                        _ = compiled_network(test_market, test_account)

                    # Only assign if compilation and test succeeded
                    self.network = compiled_network
                    self.target_network = compiled_target_network
                    logger.info(f"torch.compile applied successfully with mode='{mode}'.")
                    compile_success = True
                    break

                except ImportError as imp_err:
                    logger.error(f"torch.compile mode '{mode}' failed due to import issue: {imp_err}.")
                except RuntimeError as runtime_err:
                    # Handle cases where compilation fails due to backend issues
                    if "Triton" in str(runtime_err) or "triton" in str(runtime_err).lower():
                        logger.error(f"torch.compile mode '{mode}' failed due to Triton backend issues: {runtime_err}.")
                    else:
                        logger.error(f"torch.compile mode '{mode}' failed with runtime error: {runtime_err}.")
                except Exception as e:
                    logger.error(f"torch.compile mode '{mode}' failed with unexpected error: {e}.")

            if not compile_success:
                raise RuntimeError(
                    "All torch.compile modes failed. torch.compile is REQUIRED for training. "
                    "Please ensure you have:\n"
                    "1. CUDA-compatible GPU\n"
                    "2. GCC compiler installed (sudo apt install build-essential)\n"
                    "3. Python development headers (sudo apt install python3-dev)\n"
                    "4. Working Triton installation\n"
                    "Training cannot proceed without compilation optimization."
                )
        else:
            logger.info("inference_only=True: skipping torch.compile (eager inference for live / CPU).")

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Learning Rate Scheduler Initialization (moved after optimizer init)
        self.lr_scheduler_enabled = self.config.get("lr_scheduler_enabled", False)  # Get from self.config
        self.scheduler = None
        self._scheduler_requires_metric = False
        if self.lr_scheduler_enabled:
            scheduler_type = self.config.get("lr_scheduler_type", "StepLR")
            scheduler_params = self.config.get("lr_scheduler_params", {})

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
                    logger.info(f"Initialized ReduceLROnPlateau with mode='{mode}', factor={factor}, patience={patience}")
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
        self._market_tensor = torch.zeros((1, self.window_size, self.n_features), device=self.device, dtype=torch.float32)
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
            ), f"Input market_data shape mismatch. Expected {(self.window_size, self.n_features)}, got {obs['market_data'].shape}"
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
                assert 0 <= action < self.num_actions, f"Selected action ({action}) is out of bounds [0, {self.num_actions})"
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
            assert isinstance(obs, dict) and "market_data" in obs and "account_state" in obs, "Invalid current observation format"
            assert isinstance(next_obs, dict) and "market_data" in next_obs and "account_state" in next_obs, (
                "Invalid next observation format"
            )
            assert isinstance(obs["market_data"], np.ndarray) and obs["market_data"].shape == (
                self.window_size,
                self.n_features,
            ), f"Invalid obs market data shape {obs['market_data'].shape}"
            assert isinstance(obs["account_state"], np.ndarray) and obs["account_state"].shape == (ACCOUNT_STATE_DIM,), (
                f"Invalid obs account state shape {obs['account_state'].shape}"
            )
            assert isinstance(next_obs["market_data"], np.ndarray) and next_obs["market_data"].shape == (
                self.window_size,
                self.n_features,
            ), f"Invalid next_obs market data shape {next_obs['market_data'].shape}"
            assert isinstance(next_obs["account_state"], np.ndarray) and next_obs["account_state"].shape == (ACCOUNT_STATE_DIM,), (
                f"Invalid next_obs account state shape {next_obs['account_state'].shape}"
            )
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
            next_log_dist = self.target_network(next_market_data_batch, next_account_state_batch)  # [B, num_actions, num_atoms]
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
                    logger.warning(f"Target distribution M does not sum to 1. Sums: {sums}. Min sum: {sums.min()}, Max sum: {sums.max()}")
                    # It might not sum *exactly* to 1 due to floating point, clamping, and edge cases.
                    # A small tolerance is usually acceptable.

            return m

    def _accumulate_categorical_target_stats(self, target_distribution: torch.Tensor) -> None:
        """Accumulates categorical target distributions for periodic logging."""
        if self.categorical_logging_interval <= 0 or target_distribution is None:
            return

        if target_distribution.numel() == 0:
            return

        try:
            batch_mass = target_distribution.detach().sum(dim=0).to(device="cpu", dtype=torch.float64).numpy()
        except (RuntimeError, ValueError) as error:
            logger.warning(f"Failed to accumulate categorical target stats: {error}")
            return

        if not np.isfinite(batch_mass).all():
            logger.warning("Non-finite values encountered while accumulating categorical target stats; skipping update.")
            return

        self._categorical_target_accumulator["mass"] += batch_mass
        self._categorical_target_accumulator["samples"] += target_distribution.shape[0]

    def _log_categorical_target_stats(self) -> None:
        """Logs histogram and percentiles for accumulated categorical target distributions."""
        accumulator = self._categorical_target_accumulator

        total_samples = accumulator["samples"]
        if total_samples == 0:
            return

        mass = accumulator["mass"]
        total_mass = mass.sum()
        if not np.isfinite(total_mass) or total_mass <= 0:
            logger.warning("Invalid total mass encountered while logging categorical target stats; resetting accumulator.")
            accumulator["mass"].fill(0.0)
            accumulator["samples"] = 0
            return

        probs = mass / total_mass
        if not np.isfinite(probs).all():
            logger.warning("Non-finite probabilities encountered in categorical target stats; resetting accumulator.")
            accumulator["mass"].fill(0.0)
            accumulator["samples"] = 0
            return

        cdf = np.cumsum(probs)
        percentile_strings = []
        for percentile in self.categorical_logging_percentiles:
            target = percentile / 100.0
            idx = int(np.searchsorted(cdf, target, side="left"))
            idx = min(max(idx, 0), self.num_atoms - 1)
            percentile_strings.append(f"{percentile:.1f}%={self.support_cpu[idx]:.4f}")

        mean_value = float(np.dot(probs, self.support_cpu))
        edge_min = float(probs[0])
        edge_max = float(probs[-1])

        # Build {percentile -> support_value} mapping for both logs and TB scalars.
        percentile_values: dict[float, float] = {}
        for percentile in self.categorical_logging_percentiles:
            target = percentile / 100.0
            idx = int(np.searchsorted(cdf, target, side="left"))
            idx = min(max(idx, 0), self.num_atoms - 1)
            percentile_values[float(percentile)] = float(self.support_cpu[idx])

        logger.info(
            "Categorical target stats at learn step %s (accumulated over %s samples): mean=%.4f, edge_mass=(min=%.4f, max=%.4f), percentiles=[%s]",
            self.total_steps,
            total_samples,
            mean_value,
            edge_min,
            edge_max,
            ", ".join(percentile_strings),
        )
        logger.info(
            "Categorical target histogram (avg prob per atom, sum=1.0): %s",
            np.array2string(probs, precision=4, suppress_small=True),
        )

        # Tier 1c: mirror categorical-target diagnostics to TensorBoard.
        if self.tb_writer is not None:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Mean", mean_value, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Edge_Mass_Min", edge_min, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Edge_Mass_Max", edge_max, step)
                self.tb_writer.add_scalar("Train/CategoricalTarget/Samples", float(total_samples), step)
                for percentile, support_value in percentile_values.items():
                    tag = f"Train/CategoricalTarget/P{percentile:g}"
                    self.tb_writer.add_scalar(tag, support_value, step)
                # Histogram: weighted distribution over the 101 atoms.
                # Use the per-atom probability directly so the TB histogram view shows
                # the full target distribution shape.
                self.tb_writer.add_histogram(
                    "Train/CategoricalTarget/Distribution",
                    probs.astype(np.float32),
                    step,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror categorical target stats to TensorBoard: %s", exc)

        accumulator["mass"].fill(0.0)
        accumulator["samples"] = 0

    def _log_n_step_reward_window_stats(self) -> None:
        """Logs statistics of the rolling n-step reward window to logger and TensorBoard.

        Tier 1e: realized n-step returns are mirrored to TB so they can be compared
        directly against the categorical target distribution (Tier 1c) and PER stats
        (Tier 1d) on the same time axis.
        """
        if len(self.n_step_reward_window) == 0:
            return
        try:
            rewards_array = np.fromiter(self.n_step_reward_window, dtype=float)
        except ValueError:  # pragma: no cover - safeguard
            logger.warning("Could not materialize n-step reward window for stats.")
            return

        min_r = float(rewards_array.min())
        max_r = float(rewards_array.max())
        mean_r = float(rewards_array.mean())
        std_r = float(rewards_array.std(ddof=0))
        window_len = len(self.n_step_reward_window)

        logger.info(
            "N-Step Reward Window (last %s learns): Min=%.4f, Max=%.4f, Mean=%.4f, Std=%.4f",
            window_len,
            min_r,
            max_r,
            mean_r,
            std_r,
        )

        if self.tb_writer is not None:
            try:
                step = int(self.total_steps)
                self.tb_writer.add_scalar("Train/NStepReward/Mean", mean_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Std", std_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Min", min_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/Max", max_r, step)
                self.tb_writer.add_scalar("Train/NStepReward/WindowSize", float(window_len), step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to mirror n-step reward window stats to TensorBoard: %s",
                    exc,
                )

    def _log_grad_stats(self, pre_clip_norm: torch.Tensor | float | None) -> None:
        """Mirror gradient norms + param-update-ratio per module group to TB.

        Tier 3c. Called immediately after :func:`torch.nn.utils.clip_grad_norm_`
        and before :meth:`optimizer.step`, so:

        * ``Train/Grad/Norm`` reports the *pre-clip* global L2 norm — this is
          what would have been applied without clipping. Watching it spike past
          ``grad_clip_norm`` quickly explains otherwise-invisible loss plateaus.
        * ``Train/Grad/PerGroup/{group}/Norm`` mirrors per-top-level-module
          gradients (encoder, advantage stream, etc.). A single dominating
          group is the canonical signature of a wonky aux head or transformer
          collapse.
        * ``Train/ParamUpdateRatio`` ≈ ``lr * ||grad|| / ||param||`` — the
          textbook "is the optimizer actually moving the weights?" metric.
        """
        if self.tb_writer is None or self.network is None:
            return
        step = int(self.total_steps + 1)
        # Use the underlying compiled module so group names are clean
        # (no ``_orig_mod`` prefix injected by torch.compile).
        net = getattr(self.network, "_orig_mod", self.network)
        try:
            if pre_clip_norm is not None:
                norm_val = float(pre_clip_norm) if not torch.is_tensor(pre_clip_norm) else float(pre_clip_norm.item())
                if math.isfinite(norm_val):
                    self.tb_writer.add_scalar("Train/Grad/Norm", norm_val, step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Train/Grad/Norm: %s", exc)

        # Per-top-level-module group L2 norms.
        group_grad_sq: dict[str, float] = {}
        group_param_sq: dict[str, float] = {}
        try:
            for name, param in net.named_parameters():
                if param.grad is None:
                    continue
                group = name.split(".", 1)[0] or "root"
                g_sq = float(param.grad.detach().pow(2).sum().item())
                p_sq = float(param.detach().pow(2).sum().item())
                if not (math.isfinite(g_sq) and math.isfinite(p_sq)):
                    continue
                group_grad_sq[group] = group_grad_sq.get(group, 0.0) + g_sq
                group_param_sq[group] = group_param_sq.get(group, 0.0) + p_sq
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to compute per-group grad stats: %s", exc)
            return

        try:
            for group in sorted(group_grad_sq):
                self.tb_writer.add_scalar(
                    f"Train/Grad/PerGroup/{group}/Norm",
                    math.sqrt(group_grad_sq[group]),
                    step,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror per-group grad norms: %s", exc)

        # Param-update ratio: lr * ||grad|| / ||param||. Computed across the
        # whole network so we get a single, comparable scalar per step.
        try:
            total_grad = math.sqrt(sum(group_grad_sq.values()))
            total_param = math.sqrt(sum(group_param_sq.values()))
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0)) if self.optimizer is not None else 0.0
            if total_param > 0 and math.isfinite(total_grad) and math.isfinite(lr):
                ratio = (lr * total_grad) / total_param
                if math.isfinite(ratio):
                    self.tb_writer.add_scalar("Train/ParamUpdateRatio", ratio, step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Train/ParamUpdateRatio: %s", exc)

    def _log_q_value_stats(self, *, emit_histogram: bool = False) -> None:
        """Mirror Q-value distribution stats from the most recent training batch.

        Tier 3b. Reads the cached online ``[B, num_actions]`` Q tensor stashed
        by :meth:`_compute_loss` and emits:

        * ``Train/Q/Mean`` / ``Train/Q/Std``
        * ``Train/Q/MaxAcrossActions`` / ``Train/Q/MinAcrossActions``
        * ``Train/Q/ActionMargin`` — mean over the batch of ``q_max - q_2nd``,
          a direct readout of how *decisive* the policy is. A collapsing margin
          flags the canonical "policy goes flat" failure mode.
        * ``Train/Q/PerAction/Mean/{0..num_actions-1}``
        * ``Train/Q/Distribution`` (histogram, sub-sampled cadence)

        All emissions are silent no-ops if the writer or the cached tensor are
        missing — keeps the call idempotent for tests / CPU smoke runs.
        """
        if self.tb_writer is None:
            return
        q = getattr(self, "_last_batch_q", None)
        if q is None:
            return
        try:
            q_np = q.detach().to(torch.float32).cpu().numpy()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to materialize cached Q-values: %s", exc)
            return
        if q_np.ndim != 2 or q_np.size == 0:
            return
        step = int(self.total_steps)
        try:
            self.tb_writer.add_scalar("Train/Q/Mean", float(q_np.mean()), step)
            self.tb_writer.add_scalar("Train/Q/Std", float(q_np.std(ddof=0)), step)
            self.tb_writer.add_scalar("Train/Q/MaxAcrossActions", float(q_np.max()), step)
            self.tb_writer.add_scalar("Train/Q/MinAcrossActions", float(q_np.min()), step)
            # Per-batch action margin: top-1 minus top-2 across the action axis,
            # averaged over the batch. Robust to ties / single-action collapse.
            sorted_q = np.sort(q_np, axis=1)
            if sorted_q.shape[1] >= 2:
                margin = float((sorted_q[:, -1] - sorted_q[:, -2]).mean())
                self.tb_writer.add_scalar("Train/Q/ActionMargin", margin, step)
            for action_idx in range(q_np.shape[1]):
                self.tb_writer.add_scalar(
                    f"Train/Q/PerAction/Mean/{action_idx}",
                    float(q_np[:, action_idx].mean()),
                    step,
                )
            if emit_histogram:
                try:
                    self.tb_writer.add_histogram("Train/Q/Distribution", q_np.astype(np.float32), step)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to emit Train/Q/Distribution histogram: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to mirror Q-value stats to TB: %s", exc)

    def _log_noisy_sigma_stats(self) -> None:
        """Mirror NoisyLinear sigma stats per module to TensorBoard.

        Tier 3a. Tracks ``Train/Noisy/{module}/SigmaMean/Max/Min`` for every
        :class:`NoisyLinear` layer in the online network. Sigma collapse is the
        canonical NoisyNet failure mode (the agent stops exploring); this lets
        us catch it early without parsing distributions by eye. The target
        network is intentionally excluded — its noise is independent and has no
        bearing on the policy that's currently being optimized.
        """
        if self.tb_writer is None or self.network is None:
            return
        # Walk the *underlying* network so torch.compile wrappers don't pollute
        # the module path with ``_orig_mod`` segments.
        net = getattr(self.network, "_orig_mod", self.network)
        try:
            from .model import NoisyLinear  # local import to avoid cycles
        except Exception:  # pragma: no cover - defensive
            return
        step = int(self.total_steps)
        all_means: list[float] = []
        for name, module in net.named_modules():
            if not isinstance(module, NoisyLinear):
                continue
            tag = name.replace(".", "/") if name else "root"
            try:
                w_sigma = module.weight_sigma.detach().abs()
                b_sigma = module.bias_sigma.detach().abs()
                sigma_mean = float((w_sigma.sum() + b_sigma.sum()) / (w_sigma.numel() + b_sigma.numel()))
                sigma_max = float(max(w_sigma.max().item(), b_sigma.max().item()))
                sigma_min = float(min(w_sigma.min().item(), b_sigma.min().item()))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to read NoisyLinear sigma for %s: %s", name, exc)
                continue
            try:
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMean", sigma_mean, step)
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMax", sigma_max, step)
                self.tb_writer.add_scalar(f"Train/Noisy/{tag}/SigmaMin", sigma_min, step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror NoisyLinear sigma to TB: %s", exc)
                continue
            all_means.append(sigma_mean)
        if all_means:
            try:
                self.tb_writer.add_scalar("Train/Noisy/AggregateSigmaMean", float(np.mean(all_means)), step)
                self.tb_writer.add_scalar("Train/Noisy/ModuleCount", float(len(all_means)), step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror aggregate NoisyLinear sigma to TB: %s", exc)

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
            except Exception as exc:  # pragma: no cover - defensive
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
            # Get log probabilities Z(s_t, a) from the online network
            log_ps = self.network(market_data_batch, account_state_batch)  # [B, num_actions, num_atoms]
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

            aux_pred = self.network.predict_return(market_data_batch)
            log_ret_col = 6
            if market_data_batch.shape[2] > log_ret_col:
                target_return = market_data_batch[:, -1, log_ret_col].detach()
                aux_loss = torch.nn.functional.mse_loss(aux_pred, target_return)
                loss = loss + 0.1 * aux_loss

            # --- Action entropy regularization --- #
            if self.entropy_coeff > 0:
                all_q = (torch.exp(log_ps) * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [B, num_actions]
                q_range = (all_q.max(dim=1, keepdim=True).values - all_q.min(dim=1, keepdim=True).values).clamp(min=1e-6)
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

        td_errors_np = td_errors_tensor.detach().cpu().numpy()
        td_mean = float(td_errors_np.mean())
        td_std = float(td_errors_np.std())
        self.last_td_error_stats = {
            "mean": td_mean,
            "std": td_std,
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

        # Update priorities in PER using the TD errors (ensure they are positive)
        # Add a small epsilon to prevent priorities of 0
        priorities = td_errors_tensor.cpu().numpy() + 1e-6
        if not np.isfinite(priorities).all():
            raise FloatingPointError("Non-finite priorities calculated for PER update")
        self.buffer.update_priorities(tree_indices, td_errors_tensor)  # Pass tree_indices and the original tensor

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
                for tp, op in zip(self.target_network.parameters(), self.network.parameters()):
                    acc += float((op.data - tp.data).pow(2).sum().item())
                deviation_sq = acc
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to compute target-net param deviation: %s", exc)
                deviation_sq = None

        for tp, op in zip(self.target_network.parameters(), self.network.parameters()):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
        logger.debug(f"Target network soft-updated (tau={tau}).")

        if should_log and self.tb_writer is not None:
            step = int(self.total_steps)
            try:
                self.tb_writer.add_scalar("Train/TargetNet/SoftUpdates", float(self._soft_update_count), step)
                if deviation_sq is not None and math.isfinite(deviation_sq):
                    self.tb_writer.add_scalar("Train/TargetNet/ParamDeviation", math.sqrt(deviation_sq), step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to mirror target-net diagnostics: %s", exc)

    def save_model(self, path_prefix):
        """Saves the agent's model and optimizer state."""
        if self.network is None or self.optimizer is None:
            logger.error("Network or optimizer not initialized. Cannot save model.")
            return

        # Ensure path_prefix ends with something to distinguish components if needed
        # For example, if path_prefix is "model_checkpoint", files will be "model_checkpoint_network.pth", etc.

        # --- Create a unified checkpoint dictionary ---
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,  # Save total steps for resuming
            "config": self.config,  # Save the agent's config
            "scaler_state_dict": None,
        }
        if self.scheduler and self.lr_scheduler_enabled:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save the unified checkpoint
        # Path prefix here should ideally be the full path including filename, e.g., "models/my_agent_checkpoint.pt"
        # If path_prefix is just a directory + base name like "models/rainbow_agent",
        # we might append "_checkpoint.pt" or similar.
        # For now, assuming path_prefix is a full path like "models/rainbow_transformer_best_XYZ.pt"

        try:
            # The path_prefix now includes date, episode, and score, making it unique.
            # So, we directly save to this path.
            final_save_path = f"{path_prefix}_agent_checkpoint.pt"  # Distinguish agent chkpt

            # If path_prefix already contains ".pt", we might want to adjust
            if path_prefix.endswith(".pt"):
                base_name = path_prefix[:-3]  # Remove .pt
                final_save_path = f"{base_name}_agent_state.pt"  # Use a more descriptive suffix
            else:
                # If it's just a prefix like "models/rainbow_transformer_best"
                final_save_path = f"{path_prefix}_agent_state.pt"

            torch.save(checkpoint, final_save_path)
            logger.info(f"Unified agent checkpoint saved to {final_save_path}")
            logger.info(
                "  Includes: Network, Target Network, Optimizer, Scaler (if applicable), Scheduler (if applicable), Total Steps, Config"
            )

        except Exception as e:
            logger.error(f"Error saving unified agent checkpoint to {final_save_path}: {e}", exc_info=True)

    def load_model(self, path_prefix):
        """Loads the agent's model and optimizer state from a unified checkpoint."""
        # --- Path for the unified checkpoint ---
        # Consistent with how save_model constructs it.
        # If path_prefix is "models/rainbow_transformer_best_XYZ.pt", then:
        if path_prefix.endswith(".pt"):
            base_name = path_prefix[:-3]
            checkpoint_path = f"{base_name}_agent_state.pt"
        else:
            checkpoint_path = f"{path_prefix}_agent_state.pt"  # Fallback if not ending with .pt

        if not os.path.exists(checkpoint_path):
            logger.error(f"Unified agent checkpoint file not found at {checkpoint_path}. Cannot load model.")
            return False  # Indicate failure

        try:
            logger.info(f"Attempting to load unified agent checkpoint from: {checkpoint_path}")
            # Ensure map_location is correctly set, especially if loading a CUDA-trained model on CPU
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Load network and target network
            if "network_state_dict" in checkpoint and self.network is not None:
                self.network.load_state_dict(checkpoint["network_state_dict"])
                logger.info("Network state loaded.")
            else:
                logger.warning("Network state_dict not found in checkpoint or network not initialized.")
                return False

            if "target_network_state_dict" in checkpoint and self.target_network is not None:
                self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
                logger.info("Target network state loaded.")
            else:
                logger.warning("Target network state_dict not found in checkpoint or target_network not initialized.")
                # This might be acceptable if target network is re-initialized from network after load
                # but for full resume, it's better to have it.

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded.")
            else:
                logger.warning("Optimizer state_dict not found in checkpoint or optimizer not initialized.")
                # Not returning False here, as sometimes one might want to load just weights with a new optimizer

            # Load total steps
            if "total_steps" in checkpoint:
                self.total_steps = checkpoint["total_steps"]
                logger.info(f"Total steps loaded: {self.total_steps}")
            else:
                logger.warning("Total steps not found in checkpoint. Resetting to 0.")
                self.total_steps = 0  # Or handle as error depending on requirements

            # Load scheduler state
            if "scheduler_state_dict" in checkpoint and self.scheduler and self.lr_scheduler_enabled:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info("LR Scheduler state loaded.")
                except Exception as e:
                    logger.error(
                        f"Error loading LR scheduler state: {e}. Scheduler may not resume correctly.",
                        exc_info=True,
                    )
            elif self.scheduler and self.lr_scheduler_enabled and "scheduler_state_dict" not in checkpoint:
                logger.warning("LR Scheduler is enabled but its state was not found in the checkpoint. Scheduler will start fresh.")

            # Optionally, compare or restore config (self.config vs checkpoint['config'])
            if "config" in checkpoint:
                # Basic check: e.g., if checkpoint['config']['lr'] != self.config['lr'], log a warning or error.
                # For now, just log that config was present.
                logger.info("Agent config found in checkpoint. Consider validating compatibility.")
            else:
                logger.warning("Agent config not found in checkpoint.")

            logger.info(f"Agent model and associated states loaded successfully from {checkpoint_path}")
            self.network.to(self.device)
            self.target_network.to(self.device)
            # Ensure target network is synchronised with the online network after loading
            self._update_target_network()
            # Ensure optimizer state is also on the correct device after loading
            # This is generally handled by PyTorch, but good to be mindful of.
            return True  # Indicate success

        except FileNotFoundError:
            logger.error(f"Checkpoint file not found at {checkpoint_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading agent checkpoint from {checkpoint_path}: {e}", exc_info=True)
            return False

    def load_state(self, agent_state_dict: dict):
        """
        Loads the agent's state from a dictionary (typically part of a larger checkpoint).
        This is used by the trainer when resuming.

        Args:
            agent_state_dict (dict): A dictionary containing the agent's state.
                                     Expected keys: 'network_state_dict',
                                                    'target_network_state_dict',
                                                    'optimizer_state_dict',
                                                    'total_steps',
                                                    'scaler_state_dict' (optional),
                                                    'scheduler_state_dict' (optional).
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        logger.info("Attempting to load agent state from provided dictionary...")

        if not isinstance(agent_state_dict, dict):
            logger.error("Provided agent_state_dict is not a dictionary.")
            return False

        successful_load = True

        # Load network state
        if "network_state_dict" in agent_state_dict and self.network is not None:
            try:
                _load_state_dict_with_orig_mod_fallback(self.network, agent_state_dict["network_state_dict"])
                self.network.to(self.device)  # Ensure model is on the correct device
                logger.info("Network state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading network state_dict from dictionary: {e}", exc_info=True)
                successful_load = False
        else:
            logger.warning("Network state_dict not found in provided dictionary or agent.network is None.")
            successful_load = False  # Critical component

        # Load target network state
        if "target_network_state_dict" in agent_state_dict and self.target_network is not None:
            try:
                _load_state_dict_with_orig_mod_fallback(
                    self.target_network,
                    agent_state_dict["target_network_state_dict"],
                )
                self.target_network.to(self.device)  # Ensure model is on the correct device
                logger.info("Target network state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading target_network state_dict from dictionary: {e}", exc_info=True)
                # successful_load = False # Could be considered non-critical if re-synced
        else:
            logger.warning("Target network state_dict not found in provided dictionary or agent.target_network is None.")
            # successful_load = False

        # Load optimizer state
        if "optimizer_state_dict" in agent_state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(agent_state_dict["optimizer_state_dict"])
                # Ensure optimizer's state is on the correct device if parameters were moved
                # This is usually handled by PyTorch loading mechanism if map_location was used or model params are already on device.
                logger.info("Optimizer state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading optimizer state_dict from dictionary: {e}", exc_info=True)
                # successful_load = False # Can be critical for proper resume
        else:
            logger.warning("Optimizer state_dict not found in provided dictionary or agent.optimizer is None.")
            # successful_load = False

        # Load total steps
        if "total_steps" in agent_state_dict:
            self.total_steps = agent_state_dict["total_steps"]
            logger.info(f"Total steps loaded from dictionary: {self.total_steps}")

        if "agent_env_steps" in agent_state_dict:
            self.env_steps = agent_state_dict["agent_env_steps"]
            logger.info(f"Env steps loaded from dictionary: {self.env_steps} (epsilon={self.current_epsilon:.4f})")
        else:
            logger.warning("Total steps not found in provided dictionary. Agent's total_steps not updated.")
            # Consider if this should be an error or if agent's current total_steps is acceptable.

        # Load scheduler state
        if (
            "scheduler_state_dict" in agent_state_dict
            and agent_state_dict["scheduler_state_dict"] is not None
            and self.scheduler
            and self.lr_scheduler_enabled
        ):
            try:
                self.scheduler.load_state_dict(agent_state_dict["scheduler_state_dict"])
                logger.info("LR Scheduler state loaded from dictionary.")
            except Exception as e:
                logger.error(
                    f"Error loading LR scheduler state_dict from dictionary: {e}. Scheduler may not resume correctly.",
                    exc_info=True,
                )
        elif (
            self.scheduler
            and self.lr_scheduler_enabled
            and ("scheduler_state_dict" not in agent_state_dict or agent_state_dict.get("scheduler_state_dict") is None)
        ):
            logger.warning("LR Scheduler is enabled but its state was not found in the dictionary. Scheduler will start fresh.")

        # Agent config compatibility check could also be done here if agent_config is part of agent_state_dict

        if successful_load:
            logger.info("Agent state loaded successfully from dictionary.")
        else:
            logger.error("One or more critical components failed to load from the agent state dictionary.")

        return successful_load

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
                except Exception:  # pragma: no cover - defensive
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

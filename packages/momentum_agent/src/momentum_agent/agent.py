import os  # Added for save/load path handling
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from momentum_core.logging import get_logger
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from .buffer import PrioritizedReplayBuffer
from .constants import ACCOUNT_STATE_DIM  # Import constant
from .model import RainbowNetwork

# Get logger instance
logger = get_logger("Agent")


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

    def __init__(self, config: dict, device: str = "cuda", scaler: GradScaler | None = None):
        """
        Initializes the Rainbow DQN Agent.

        Args:
            config (dict): A dictionary containing all hyperparameters and network settings.
                           Expected keys: seed, gamma, lr, replay_buffer_size, batch_size,
                           target_update_freq, num_atoms, v_min, v_max, alpha, beta_start,
                           beta_frames, n_steps, window_size, n_features, hidden_dim,
                           num_actions, grad_clip_norm, debug (optional).
            device (str): The device to run the agent on ('cuda' or 'cpu').
            scaler (GradScaler | None): Optional GradScaler for AMP.
        """
        self.config = config
        self.device = self._resolve_device(device)
        # Use direct access for mandatory parameters
        self.seed = config["seed"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
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
        self.store_partial_n_step = self.config.get("store_partial_n_step", False)
        # Optional flags can still use .get()
        self.debug_mode = config.get("debug", False)
        self.scaler = scaler  # Store the scaler instance
        self._n_step_needs_reset = False

        # Setup seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            logger.info(f"CUDA seed set. AMP Enabled: {self.scaler is not None}")
        else:
            if self.scaler is not None:
                logger.info("GradScaler provided for non-CUDA device; disabling AMP.")
                self.scaler = None
            logger.info("Agent on CPU. AMP Disabled.")

        logger.info(f"Initializing RainbowDQNAgent on {self.device}")
        logger.info(f"Device type: {self.device.type}, CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Config: {config}")  # Log the entire config

        # Distributional RL setup
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

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

        # Enable torch.compile for PyTorch 2.8+ with improved error handling
        # if hasattr(torch, 'compile'):
        #     logger.info("Attempting to apply torch.compile to network and target_network.")
        #     compile_success = False

        #     # Try different compilation modes in order of preference
        #     compile_modes = ["default", "reduce-overhead", "max-autotune"]

        #     for mode in compile_modes:
        #         try:
        #             logger.info(f"Trying torch.compile with mode='{mode}'...")
        #             self.network = torch.compile(self.network, mode=mode)
        #             self.target_network = torch.compile(self.target_network, mode=mode)

        #             # Test compilation by running a small forward pass
        #             with torch.no_grad():
        #                 test_market = torch.zeros((1, self.window_size, self.n_features), device=self.device)
        #                 test_account = torch.zeros((1, 2), device=self.device)
        #                 _ = self.network(test_market, test_account)

        #             logger.info(f"torch.compile applied successfully with mode='{mode}'.")
        #             compile_success = True
        #             break

        #         except ImportError as imp_err:
        #             logger.warning(f"torch.compile mode '{mode}' failed due to import issue: {imp_err}.")
        #         except RuntimeError as runtime_err:
        #             # Handle cases where compilation fails due to backend issues
        #             if "Triton" in str(runtime_err) or "triton" in str(runtime_err).lower():
        #                 logger.warning(f"torch.compile mode '{mode}' failed due to Triton backend issues: {runtime_err}. This is common on Windows.")
        #             else:
        #                 logger.warning(f"torch.compile mode '{mode}' failed with runtime error: {runtime_err}.")
        #         except Exception as e:
        #             logger.warning(f"torch.compile mode '{mode}' failed with unexpected error: {e}.")

        #     if not compile_success:
        #         logger.warning("All torch.compile modes failed. Proceeding without compilation. This is normal on some platforms (e.g., Windows without Triton).")
        # else:
        #     logger.warning("torch.compile not available in this PyTorch version.")

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
        # For N-step returns
        self.n_step_buffer = deque(maxlen=self.n_steps)
        # --- ADDED: Deque for n-step reward logging window ---
        self.n_step_reward_window = deque(maxlen=60)
        # --- END ADDED ---
        # --- ADDED: List for comprehensive N-step reward history ---
        self.observed_n_step_rewards_history = [] if self.debug_mode else None
        # --- END ADDED ---

        self.training_mode = True  # Start in training mode by default
        self._network_mode_training: bool | None = None
        self._apply_network_mode(self.training_mode)
        self._non_blocking_copy = self.device.type == "cuda" and torch.cuda.is_available()
        self._market_tensor = torch.zeros((1, self.window_size, self.n_features), device=self.device, dtype=torch.float32)
        self._account_tensor = torch.zeros((1, ACCOUNT_STATE_DIM), device=self.device, dtype=torch.float32)
        self.total_steps = 0  # Track total steps for target network updates and beta annealing
        self.last_td_error_stats: dict[str, float] | None = None

    def select_action(self, obs):
        """Selects action based on the current Q-value estimates using Noisy Nets."""
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
            assert obs["account_state"].shape == (
                ACCOUNT_STATE_DIM,
            ), f"Input account_state shape mismatch. Expected ({ACCOUNT_STATE_DIM},), got {obs['account_state'].shape}"

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
            with autocast("cuda", enabled=autocast_enabled):
                q_values = self.network.get_q_values(market_data, account_state)
            if self.debug_mode:
                assert q_values.shape == (
                    1,
                    self.num_actions,
                ), f"Q-values shape mismatch. Expected (1, {self.num_actions}), got {q_values.shape}"
            action = q_values.argmax().item()  # Choose action with highest expected Q-value
            if self.debug_mode:
                assert isinstance(action, int), f"Selected action is not an integer: {action}"
                assert 0 <= action < self.num_actions, f"Selected action ({action}) is out of bounds [0, {self.num_actions})"
        return action

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

    def _get_n_step_info(self, steps: int | None = None):
        """
        Calculates the (truncated) n-step return using the oldest `steps` transitions
        currently stored in the n-step buffer.

        Args:
            steps (int | None): Number of transitions to consider. Defaults to self.n_steps.

        Returns:
            tuple: Contains state_t, action_t, discounted reward, next_state_tn, done_tn.
        """
        if steps is None:
            steps = self.n_steps

        if steps < 1:
            raise ValueError("Steps for n-step calculation must be >= 1")
        if len(self.n_step_buffer) < steps:
            raise ValueError("Insufficient transitions in n-step buffer")

        transitions = list(self.n_step_buffer)[:steps]

        # State and action from the *first* transition in the slice (t)
        market_data_t, account_state_t, action_t, _, _, _, _ = transitions[0]
        state_t = (market_data_t, account_state_t)

        # Next state and done flag from the *last* transition in the slice (t + steps - 1)
        _, _, _, _, next_market_tn, next_account_tn, done_tn = transitions[-1]
        next_state_tn = (next_market_tn, next_account_tn)

        # Calculate cumulative discounted reward G_t^(steps)
        n_step_reward = 0.0
        discount = 1.0
        for _, _, _, reward_i, _, _, done_i in transitions:
            n_step_reward += discount * reward_i
            if done_i:
                break
            discount *= self.gamma

        if self.debug_mode:
            # --- Start: Assert return types and shapes ---
            assert isinstance(state_t, tuple) and len(state_t) == 2, "state_t is not a 2-tuple"
            assert isinstance(state_t[0], np.ndarray) and state_t[0].shape == (
                self.window_size,
                self.n_features,
            ), "state_t[0] (market_data) has wrong type/shape"
            assert isinstance(state_t[1], np.ndarray) and state_t[1].shape == (
                ACCOUNT_STATE_DIM,
            ), "state_t[1] (account_state) has wrong type/shape"
            assert isinstance(action_t, (int, np.integer)), "action_t is not an integer"
            assert isinstance(n_step_reward, (float, np.float32, np.float64)), "n_step_reward is not a float"
            assert isinstance(next_state_tn, tuple) and len(next_state_tn) == 2, "next_state_tn is not a 2-tuple"
            assert isinstance(next_state_tn[0], np.ndarray) and next_state_tn[0].shape == (
                self.window_size,
                self.n_features,
            ), "next_state_tn[0] (market_data) has wrong type/shape"
            assert isinstance(next_state_tn[1], np.ndarray) and next_state_tn[1].shape == (
                ACCOUNT_STATE_DIM,
            ), "next_state_tn[1] (account_state) has wrong type/shape"
            assert isinstance(done_tn, (bool, np.bool_)), "done_tn is not a boolean"
            # --- End: Assert return types and shapes ---

        return state_t, action_t, n_step_reward, next_state_tn, done_tn

    def _store_n_step_transition(self, steps: int, *, pop_after: bool) -> None:
        """Helper to compute and store an n-step (or truncated) transition."""
        state_t, action_t, n_step_reward, next_state_tn, done_tn = self._get_n_step_info(steps)
        market_data_t, account_state_t = state_t
        next_market_tn, next_account_tn = next_state_tn

        # Track reward statistics
        self.n_step_reward_window.append(n_step_reward)
        if self.observed_n_step_rewards_history is not None:
            self.observed_n_step_rewards_history.append(n_step_reward)

        # Store transition in PER
        self.buffer.store(
            market_data_t,
            account_state_t,
            action_t,
            n_step_reward,
            next_market_tn,
            next_account_tn,
            done_tn,
        )

        # Advance buffer window when requested (e.g., during terminal flush)
        if pop_after and len(self.n_step_buffer) > 0:
            self.n_step_buffer.popleft()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Stores experience in N-step buffer and potentially transfers to PER."""
        if self._n_step_needs_reset:
            self.n_step_buffer.clear()
            self._n_step_needs_reset = False

        if self.debug_mode:
            # --- Start: Assert input types and shapes for store_transition ---
            assert isinstance(obs, dict) and "market_data" in obs and "account_state" in obs, "Invalid current observation format"
            assert (
                isinstance(next_obs, dict) and "market_data" in next_obs and "account_state" in next_obs
            ), "Invalid next observation format"
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
            # --- End: Assert input types and shapes ---

        # Store raw single-step transition data needed for n-step calculation
        # (s_k, a_k, r_{k+1}, s_{k+1}_market, s_{k+1}_account, d_{k+1})
        transition = (
            obs["market_data"],
            obs["account_state"],
            action,
            reward,
            next_obs["market_data"],
            next_obs["account_state"],
            done,
        )
        self.n_step_buffer.append(transition)

        buffer_full = len(self.n_step_buffer) == self.n_steps

        # When buffer is full during ongoing episodes, store the rolling n-step transition
        if buffer_full and not done:
            self._store_n_step_transition(self.n_steps, pop_after=False)

        if done:
            # Ensure the final n-step transition is captured once
            if buffer_full:
                self._store_n_step_transition(
                    self.n_steps,
                    pop_after=self.store_partial_n_step,
                )

            if self.store_partial_n_step:
                # Optionally store truncated returns at episode end
                while len(self.n_step_buffer) > 0:
                    steps = len(self.n_step_buffer)
                    self._store_n_step_transition(steps, pop_after=True)

            # Defer clearing until the next transition to keep tests' expectations
            self._n_step_needs_reset = True

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

        # Convert numpy arrays from buffer to tensors
        market_data_batch = torch.FloatTensor(market_data).to(self.device)
        account_state_batch = torch.FloatTensor(account_state).to(self.device)
        next_market_data_batch = torch.FloatTensor(next_market_data).to(self.device)
        next_account_state_batch = torch.FloatTensor(next_account_state).to(self.device)
        actions_batch = torch.LongTensor(actions).to(self.device)  # Action indices [B]
        rewards_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # [B, 1]
        dones_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [B, 1]
        weights_batch = torch.FloatTensor(weights).unsqueeze(1).to(self.device)  # [B, 1]

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

        # --- Calculate Target Distribution (m) --- #
        # This is the projected distribution for the Bellman target Z(s_t, a_t)
        target_distribution = self._project_target_distribution(
            next_market_data_batch, next_account_state_batch, rewards_batch, dones_batch
        )
        if self.debug_mode:
            assert target_distribution.shape == (
                self.batch_size,
                self.num_atoms,
            ), "Target distribution shape mismatch"
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

        # Apply Importance Sampling weights and calculate mean loss
        loss = (loss_elementwise * weights_batch.squeeze(1).detach()).mean()  # Scalar
        if loss.ndim != 0:
            raise RuntimeError("Final loss is not a scalar")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss calculation resulted in NaN or Inf: {loss.item()}")
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
            assert isinstance(td_errors_tensor, torch.Tensor) and td_errors_tensor.shape == (
                self.batch_size,
            ), "TD errors tensor from _compute_loss has wrong shape/type"

        # Log GPU memory usage if using CUDA (less frequently to reduce log spam)
        if self.device.type == "cuda" and torch.cuda.is_available() and self.total_steps % 100 == 0:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            logger.debug(f"GPU Memory - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

        # Check if AMP is enabled (scaler exists)
        amp_enabled = self.scaler is not None and self.device.type == "cuda"

        # Optimize the model
        self.optimizer.zero_grad()

        if amp_enabled:
            # Scale loss and backpropagate
            self.scaler.scale(loss).backward()
        else:
            # Standard backward pass
            loss.backward()

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(f"NaN or Inf detected in gradients BEFORE clipping for parameter: {p.shape}")

        # Clip gradients
        if amp_enabled:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)

        if self.debug_mode:
            for p in self.network.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.error(
                        f"NaN or Inf detected in gradients AFTER clipping (Max Norm: {self.grad_clip_norm}) for parameter: {p.shape}"
                    )

        if amp_enabled:
            # Scaler steps the optimizer
            self.scaler.step(self.optimizer)
            # Update scaler for next iteration
            self.scaler.update()
        else:
            # Standard optimizer step
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
        if self.last_td_error_stats is not None and self.total_steps % 100 == 0:
            logger.info(
                "TD error stats at learn step %s - mean: %.6f, std: %.6f",
                self.total_steps,
                self.last_td_error_stats.get("mean", float("nan")),
                self.last_td_error_stats.get("std", float("nan")),
            )
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
            logger.info(f"Step {self.total_steps}: Target network updated.")

        loss_item = loss.item()
        if not isinstance(loss_item, float) or np.isnan(loss_item) or np.isinf(loss_item):
            raise FloatingPointError("Final loss item is not a valid float")

        # Log loss and PER beta
        logger.debug(f"Step: {self.total_steps}, Loss: {loss_item:.4f}, PER Beta: {beta:.4f}")

        # --- ADDED: Log min/max of n-step reward window periodically ---
        # Log every 60 agent learning steps if the window has data
        if self.total_steps % 60 == 0 and len(self.n_step_reward_window) > 0:
            try:
                rewards_array = np.fromiter(self.n_step_reward_window, dtype=float)
                min_r = float(rewards_array.min())
                max_r = float(rewards_array.max())
                mean_r = float(rewards_array.mean())
                std_r = float(rewards_array.std(ddof=0))
                logger.info(
                    "N-Step Reward Window (last %s learns): Min=%.4f, Max=%.4f, Mean=%.4f, Std=%.4f",
                    len(self.n_step_reward_window),
                    min_r,
                    max_r,
                    mean_r,
                    std_r,
                )
            except ValueError:
                # Should not happen if len > 0, but safeguard
                logger.warning("Could not calculate statistics for n-step reward window.")
        # --- END ADDED ---

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
        """Copies weights from online network to target network."""
        self.target_network.load_state_dict(self.network.state_dict())
        logger.debug("Target network weights updated.")

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
            "scaler_state_dict": (self.scaler.state_dict() if self.scaler else None),  # Save scaler state
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load network and target network
            if "network_state_dict" in checkpoint and self.network:
                self.network.load_state_dict(checkpoint["network_state_dict"])
                logger.info("Network state loaded.")
            else:
                logger.warning("Network state_dict not found in checkpoint or network not initialized.")
                return False

            if "target_network_state_dict" in checkpoint and self.target_network:
                self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
                logger.info("Target network state loaded.")
            else:
                logger.warning("Target network state_dict not found in checkpoint or target_network not initialized.")
                # This might be acceptable if target network is re-initialized from network after load
                # but for full resume, it's better to have it.

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint and self.optimizer:
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

            # Load scaler state
            if "scaler_state_dict" in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info("GradScaler state loaded.")
            elif self.scaler is None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
                logger.warning("Scaler state found in checkpoint, but agent's scaler is None. Scaler state not loaded.")
            elif self.scaler and "scaler_state_dict" not in checkpoint:
                logger.warning("Agent has a scaler, but no scaler state found in checkpoint.")

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
        if "network_state_dict" in agent_state_dict and self.network:
            try:
                self.network.load_state_dict(agent_state_dict["network_state_dict"])
                self.network.to(self.device)  # Ensure model is on the correct device
                logger.info("Network state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading network state_dict from dictionary: {e}", exc_info=True)
                successful_load = False
        else:
            logger.warning("Network state_dict not found in provided dictionary or agent.network is None.")
            successful_load = False  # Critical component

        # Load target network state
        if "target_network_state_dict" in agent_state_dict and self.target_network:
            try:
                self.target_network.load_state_dict(agent_state_dict["target_network_state_dict"])
                self.target_network.to(self.device)  # Ensure model is on the correct device
                logger.info("Target network state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading target_network state_dict from dictionary: {e}", exc_info=True)
                # successful_load = False # Could be considered non-critical if re-synced
        else:
            logger.warning("Target network state_dict not found in provided dictionary or agent.target_network is None.")
            # successful_load = False

        # Load optimizer state
        if "optimizer_state_dict" in agent_state_dict and self.optimizer:
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
        else:
            logger.warning("Total steps not found in provided dictionary. Agent's total_steps not updated.")
            # Consider if this should be an error or if agent's current total_steps is acceptable.

        # Load scaler state
        if "scaler_state_dict" in agent_state_dict and agent_state_dict["scaler_state_dict"] is not None and self.scaler:
            try:
                self.scaler.load_state_dict(agent_state_dict["scaler_state_dict"])
                logger.info("GradScaler state loaded from dictionary.")
            except Exception as e:
                logger.error(f"Error loading GradScaler state_dict from dictionary: {e}", exc_info=True)
        elif self.scaler is None and "scaler_state_dict" in agent_state_dict and agent_state_dict["scaler_state_dict"] is not None:
            logger.warning("Scaler state found in dictionary, but agent's scaler is None. Scaler state not loaded.")
        elif self.scaler and ("scaler_state_dict" not in agent_state_dict or agent_state_dict.get("scaler_state_dict") is None):
            logger.warning("Agent has a scaler, but no scaler state found in dictionary. Scaler state not loaded.")

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

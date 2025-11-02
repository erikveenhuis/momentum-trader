import os  # Added for save/load path handling
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler

from .buffer import PrioritizedReplayBuffer
from .constants import ACCOUNT_STATE_DIM  # Import constant
from .model import RainbowNetwork
from .utils.logging_config import get_logger

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
        self.device = device
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
        # Optional flags can still use .get()
        self.debug_mode = config.get("debug", False)
        self.scaler = scaler  # Store the scaler instance

        # Setup seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
            logger.info(f"CUDA seed set. AMP Enabled: {self.scaler is not None}")
        elif self.device == "cuda":
            logger.warning("CUDA device requested but not available. Using CPU.")
            self.device = "cpu"
            self.scaler = None  # Ensure scaler is None if not on CUDA
            logger.info("Agent on CPU. AMP Disabled.")
        else:
            logger.info("Agent on CPU. AMP Disabled.")

        logger.info(f"Initializing RainbowDQNAgent on {self.device}")
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

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Learning Rate Scheduler Initialization (moved after optimizer init)
        self.lr_scheduler_enabled = self.config.get("lr_scheduler_enabled", False)  # Get from self.config
        self.scheduler = None
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
                elif scheduler_type == "CosineAnnealingLR":
                    t_max = scheduler_params.get("T_max")
                    eta_min = scheduler_params.get("min_lr", 0)  # min_lr maps to eta_min
                    if t_max is None:
                        logger.error("CosineAnnealingLR 'T_max' not provided in scheduler_params. Disabling scheduler.")
                        self.lr_scheduler_enabled = False
                    else:
                        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
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
        )
        # For N-step returns
        self.n_step_buffer = deque(maxlen=self.n_steps)
        # --- ADDED: Deque for n-step reward logging window ---
        self.n_step_reward_window = deque(maxlen=60)
        # --- END ADDED ---
        # --- ADDED: List for comprehensive N-step reward history ---
        self.observed_n_step_rewards_history = []
        # --- END ADDED ---

        self.training_mode = True  # Start in training mode by default
        self.total_steps = 0  # Track total steps for target network updates and beta annealing

    def select_action(self, obs):
        """Selects action based on the current Q-value estimates using Noisy Nets."""
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

        # Convert observation to tensors
        market_data = torch.FloatTensor(obs["market_data"]).unsqueeze(0).to(self.device)
        account_state = torch.FloatTensor(obs["account_state"]).unsqueeze(0).to(self.device)
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
        self.network.eval()  # Ensure eval mode for action selection if using dropout/batchnorm (though NoisyNets handle exploration)
        with torch.no_grad():
            q_values = self.network.get_q_values(market_data, account_state)
            assert q_values.shape == (
                1,
                self.num_actions,
            ), f"Q-values shape mismatch. Expected (1, {self.num_actions}), got {q_values.shape}"
            action = q_values.argmax().item()  # Choose action with highest expected Q-value
            assert isinstance(action, int), f"Selected action is not an integer: {action}"
            assert 0 <= action < self.num_actions, f"Selected action ({action}) is out of bounds [0, {self.num_actions})"
        # Switch back to train mode if necessary (depends if eval() affects Noisy Layers - typically it doesn't disable noise generation)
        if self.training_mode:
            self.network.train()

        return action

    def _get_n_step_info(self):
        """
        Calculates the n-step return G_t^(n) and identifies the state s_{t+n}
        and done flag d_{t+n} from the transition n steps later.
        Uses the n-step buffer which contains (s_k, a_k, r_{k+1}, s_{k+1}, d_{k+1}).

        Returns:
            tuple: Contains:
                - state_t (tuple): (market_data_t, account_state_t) from the start of the n steps.
                - action_t (int): Action taken at state_t.
                - n_step_reward (float): The accumulated discounted n-step return.
                - next_state_tn (tuple): (market_data_{t+n}, account_state_{t+n}) observed n steps later.
                - done_tn (bool): Done flag from n steps later.
        """
        assert len(self.n_step_buffer) == self.n_steps, "N-step buffer size mismatch for calculation"

        # State and action from the *first* transition in the buffer (t)
        market_data_t, account_state_t, action_t, _, _, _, _ = self.n_step_buffer[0]
        state_t = (market_data_t, account_state_t)

        # Next state and done flag from the *last* transition in the buffer (t+n)
        _, _, _, _, next_market_tn, next_account_tn, done_tn = self.n_step_buffer[-1]
        next_state_tn = (next_market_tn, next_account_tn)

        # Calculate cumulative discounted reward G_t^(n)
        n_step_reward = 0.0
        discount = 1.0
        for i in range(self.n_steps):
            # r_{t+i+1}, d_{t+i+1}
            _, _, _, reward_iplus1, _, _, done_iplus1 = self.n_step_buffer[i]
            n_step_reward += discount * reward_iplus1
            discount *= self.gamma
            if done_iplus1:
                # If termination occurs at step t+i+1 (where i < n-1),
                # the n-step return calculation stops here.
                # The next_state and done used for the Bellman target are correctly
                # s_{t+n} and d_{t+n} from the last transition.
                break

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

    def store_transition(self, obs, action, reward, next_obs, done):
        """Stores experience in N-step buffer and potentially transfers to PER."""
        # --- Start: Assert input types and shapes for store_transition ---
        assert isinstance(obs, dict) and "market_data" in obs and "account_state" in obs, "Invalid current observation format"
        assert isinstance(next_obs, dict) and "market_data" in next_obs and "account_state" in next_obs, "Invalid next observation format"
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

        # If buffer has enough steps, calculate N-step return and store in PER
        if len(self.n_step_buffer) >= self.n_steps:
            state_t, action_t, n_step_reward, next_state_tn, done_tn = self._get_n_step_info()
            # --- REMOVED: Single reward log ---
            # logger.info(f"Calculated n_step_reward: {n_step_reward}")
            # --- END REMOVED ---
            # --- ADDED: Append reward to window deque ---
            self.n_step_reward_window.append(n_step_reward)
            # --- END ADDED ---
            # --- ADDED: Append reward to comprehensive history list ---
            self.observed_n_step_rewards_history.append(n_step_reward)
            # --- END ADDED ---
            market_data_t, account_state_t = state_t
            next_market_tn, next_account_tn = next_state_tn

            # --- Start: Assert types before storing in PER buffer ---
            assert isinstance(market_data_t, np.ndarray) and market_data_t.shape == (
                self.window_size,
                self.n_features,
            ), "Invalid market_data_t for PER store"
            assert isinstance(account_state_t, np.ndarray) and account_state_t.shape == (
                ACCOUNT_STATE_DIM,
            ), "Invalid account_state_t for PER store"
            assert isinstance(action_t, (int, np.integer)), "Invalid action_t for PER store"
            assert isinstance(n_step_reward, (float, np.float32, np.float64)), "Invalid n_step_reward for PER store"
            assert isinstance(next_market_tn, np.ndarray) and next_market_tn.shape == (
                self.window_size,
                self.n_features,
            ), "Invalid next_market_tn for PER store"
            assert isinstance(next_account_tn, np.ndarray) and next_account_tn.shape == (
                ACCOUNT_STATE_DIM,
            ), "Invalid next_account_tn for PER store"
            assert isinstance(done_tn, (bool, np.bool_)), "Invalid done_tn flag for PER store"
            # --- End: Assert types before storing ---

            # Store the calculated N-step transition in the main prioritized buffer
            # Format: (s_t, a_t, G_t^(n), s_{t+n}, d_{t+n})
            self.buffer.store(
                market_data_t,
                account_state_t,
                action_t,
                n_step_reward,
                next_market_tn,
                next_account_tn,
                done_tn,
            )

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
            assert next_q_values.shape == (
                self.batch_size,
                self.num_actions,
            ), "Next Q-values shape mismatch"
            next_actions = next_q_values.argmax(dim=1)  # [batch_size]
            assert next_actions.shape == (self.batch_size,), "Next actions shape mismatch"

            # Get next state's distribution Z(s_{t+n}, a*) from target network for selected actions a*
            next_log_dist = self.target_network(next_market_data_batch, next_account_state_batch)  # [B, num_actions, num_atoms]
            assert next_log_dist.shape == (
                self.batch_size,
                self.num_actions,
                self.num_atoms,
            ), "Next log distribution shape mismatch"
            assert next_actions.max() < self.num_actions and next_actions.min() >= 0, "Invalid next_action indices"

            # Get the probability distribution for the chosen actions: p(s_{t+n}, a*)
            next_dist = torch.exp(next_log_dist[range(self.batch_size), next_actions])  # [B, num_atoms]
            assert next_dist.shape == (
                self.batch_size,
                self.num_atoms,
            ), "Next distribution shape mismatch"

            # Compute the projected Bellman target T_z = G_t^(n) + gamma^n * Z(s_{t+n}, a*)
            # Rewards are [B, 1], dones are [B, 1], support is [num_atoms]
            # Broadcasting applies correctly.
            Tz = rewards + (1 - dones) * (self.gamma**self.n_steps) * self.support  # [B, num_atoms]
            assert Tz.shape == (
                self.batch_size,
                self.num_atoms,
            ), f"Projected Tz shape mismatch: {Tz.shape}"
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            # Compute projection indices and weights
            b = (Tz - self.v_min) / self.delta_z  # Normalized position on support axis [B, num_atoms]
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
        assert target_distribution.shape == (
            self.batch_size,
            self.num_atoms,
        ), "Target distribution shape mismatch"
        # ---------------------------------------- #

        # --- Calculate Online Distribution and Loss --- #
        # Get log probabilities Z(s_t, a) from the online network
        log_ps = self.network(market_data_batch, account_state_batch)  # [B, num_actions, num_atoms]
        assert log_ps.shape == (
            self.batch_size,
            self.num_actions,
            self.num_atoms,
        ), "Online log_ps shape mismatch"

        # Gather the log-probabilities for the actions actually taken: log Z(s_t, a_t)
        # We need to select the log probabilities corresponding to actions_batch
        actions_indices = actions_batch.view(self.batch_size, 1, 1).expand(self.batch_size, 1, self.num_atoms)
        log_ps_a = log_ps.gather(1, actions_indices).squeeze(1)  # [B, num_atoms]
        assert log_ps_a.shape == (
            self.batch_size,
            self.num_atoms,
        ), "Online log_ps_a shape mismatch"

        # Calculate cross-entropy loss between target and online distributions
        # Loss = -sum_i [ target_distribution_i * log(online_distribution_i) ]
        # Target distribution is detached as it acts as the label.
        loss_elementwise = -(target_distribution.detach() * log_ps_a).sum(dim=1)  # [B]
        assert loss_elementwise.shape == (self.batch_size,), "Per-sample loss shape mismatch"

        # Apply Importance Sampling weights and calculate mean loss
        loss = (loss_elementwise * weights_batch.squeeze(1).detach()).mean()  # Scalar
        assert loss.ndim == 0, "Final loss is not a scalar"
        assert torch.isfinite(loss), f"Loss calculation resulted in NaN or Inf: {loss.item()}"
        # ----------------------------------------- #

        # --- Calculate TD errors for PER update --- #
        # TD error is | E[Target Distribution] - E[Online Distribution for a_t] |
        # Calculate expected Q-values E[Z(s_t, a_t)] from the online distribution for the action taken
        q_values_online = (torch.exp(log_ps_a) * self.support.unsqueeze(0)).sum(dim=1)  # [B]
        # Calculate expected target Q-values E[Projected Target Distribution]
        q_values_target = (target_distribution * self.support.unsqueeze(0)).sum(dim=1)  # [B]
        # TD error = |Target Q - Online Q|
        td_errors_tensor = (q_values_target.detach() - q_values_online.detach()).abs()
        assert td_errors_tensor.shape == (self.batch_size,), "TD errors tensor shape mismatch"
        assert torch.isfinite(td_errors_tensor).all(), "NaN or Inf found in TD errors tensor"
        # ----------------------------------------- #

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
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Loss from _compute_loss is not a scalar tensor"
        assert isinstance(td_errors_tensor, torch.Tensor) and td_errors_tensor.shape == (
            self.batch_size,
        ), "TD errors tensor from _compute_loss has wrong shape/type"

        # Check if AMP is enabled (scaler exists)
        amp_enabled = self.scaler is not None and self.device == "cuda"

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

        # Step the scheduler if it's enabled and not ReduceLROnPlateau (which needs a metric)
        if self.scheduler and self.lr_scheduler_enabled:
            # ReduceLROnPlateau is stepped with a metric, e.g., validation loss.
            # Other schedulers like StepLR, CosineAnnealingLR are typically stepped per optimizer step or epoch.
            # Assuming per-optimizer-step for now for StepLR and CosineAnnealingLR.
            # If your chosen scheduler (e.g., ReduceLROnPlateau) needs a metric,
            # this call will need to be moved or conditionally executed based on the scheduler type
            # and the metric passed to scheduler.step(metric).
            if not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            # For logging current LR
            # current_lr = self.optimizer.param_groups[0]['lr']
            # logger.debug(f"Current LR: {current_lr}")

        # Update priorities in PER using the TD errors (ensure they are positive)
        # Add a small epsilon to prevent priorities of 0
        priorities = td_errors_tensor.cpu().numpy() + 1e-6
        assert np.isfinite(priorities).all(), "Non-finite priorities calculated for PER update"
        self.buffer.update_priorities(tree_indices, td_errors_tensor)  # Pass tree_indices and the original tensor

        # Reset noise in Noisy Linear layers (important!)
        self.network.reset_noise()
        self.target_network.reset_noise()

        # Increment step counter and update target network periodically
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self._update_target_network()
            logger.info(f"Step {self.total_steps}: Target network updated.")

        loss_item = loss.item()
        assert isinstance(loss_item, float) and not np.isnan(loss_item) and not np.isinf(loss_item), "Final loss item is not a valid float"

        # Log loss and PER beta
        logger.debug(f"Step: {self.total_steps}, Loss: {loss_item:.4f}, PER Beta: {beta:.4f}")

        # --- ADDED: Log min/max of n-step reward window periodically ---
        # Log every 60 agent learning steps if the window has data
        if self.total_steps % 60 == 0 and len(self.n_step_reward_window) > 0:
            try:
                min_r = min(self.n_step_reward_window)
                max_r = max(self.n_step_reward_window)
                logger.info(f"N-Step Reward Window (last {len(self.n_step_reward_window)} learns): Min={min_r:.4f}, Max={max_r:.4f}")
            except ValueError:
                # Should not happen if len > 0, but safeguard
                logger.warning("Could not calculate min/max for n-step reward window.")
        # --- END ADDED ---

        return loss_item  # Return loss for external logging/monitoring

    def step_lr_scheduler(self, metric: float):
        """Steps the learning rate scheduler if it's ReduceLROnPlateau and a metric is provided."""
        if self.scheduler and self.lr_scheduler_enabled and isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            try:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(metric)
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr < current_lr:
                    logger.info(
                        f"LR scheduler 'ReduceLROnPlateau' stepped. LR reduced from {current_lr} to {new_lr} based on metric: {metric:.4f}"
                    )
                else:
                    logger.debug(f"LR scheduler 'ReduceLROnPlateau' stepped with metric: {metric:.4f}. LR unchanged: {current_lr}")
            except Exception as e:
                logger.error(f"Error stepping ReduceLROnPlateau scheduler: {e}", exc_info=True)
        elif self.scheduler and self.lr_scheduler_enabled:
            logger.debug(f"LR scheduler is enabled but is not ReduceLROnPlateau. Not stepping with metric. Type: {type(self.scheduler)}")

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
        self.training_mode = training
        mode = "TRAINING" if training else "EVALUATION"
        logger.info(f"Set agent to {mode} mode")
        if self.network:
            if training:
                self.network.train()
            else:
                self.network.eval()
                # Ensure target is also in eval mode (should be already, but safe)
                self.target_network.eval()

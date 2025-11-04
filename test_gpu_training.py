#!/usr/bin/env python3
"""Test script to check GPU memory usage during agent training."""

import torch
from momentum_agent import RainbowDQNAgent


def test_gpu_memory():
    """Test GPU memory usage with a small agent instantiation."""
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory before agent creation: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        # Create agent config similar to training
        config = {
            "window_size": 60,
            "n_features": 5,
            "hidden_dim": 256,
            "num_actions": 7,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "nhead": 4,
            "num_encoder_layers": 3,
            "dim_feedforward": 512,
            "transformer_dropout": 0.2,
            "batch_size": 512,
            "replay_buffer_size": 500000,
            "gamma": 0.99,
            "lr": 0.0003,
            "target_update_freq": 600,
            "alpha": 0.6,
            "beta_start": 0.3,
            "beta_frames": 400000,
            "n_steps": 3,
            "grad_clip_norm": 1.0,
            "seed": 42,
        }

        device = torch.device("cuda")
        agent = RainbowDQNAgent(config=config, device=device)

        print(f"CUDA memory after agent creation: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Model parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

        # Try a forward pass with dummy data
        batch_size = 512
        market_data = torch.randn(batch_size, 60, 5, device=device)
        account_state = torch.randn(batch_size, 2, device=device)

        print(f"CUDA memory before forward pass: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        with torch.no_grad():
            output = agent.network(market_data, account_state)

        print(f"CUDA memory after forward pass: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Output shape: {output.shape}")

        # Try a training step simulation
        print(f"CUDA memory before training simulation: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        # Create fake batch data
        batch_size = 512
        market_data_batch = torch.randn(batch_size, 60, 5, device=device)
        account_state_batch = torch.randn(batch_size, 2, device=device)
        actions_batch = torch.randint(0, 7, (batch_size,), device=device)
        rewards_batch = torch.randn(batch_size, 1, device=device)
        next_market_data_batch = torch.randn(batch_size, 60, 5, device=device)
        next_account_state_batch = torch.randn(batch_size, 2, device=device)
        dones_batch = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float)
        weights_batch = torch.ones(batch_size, 1, device=device)

        batch_tuple = (
            market_data_batch.cpu().numpy(),
            account_state_batch.cpu().numpy(),
            actions_batch.cpu().numpy(),
            rewards_batch.cpu().numpy().squeeze(),
            next_market_data_batch.cpu().numpy(),
            next_account_state_batch.cpu().numpy(),
            dones_batch.cpu().numpy().squeeze(),
        )

        # Simulate learning (this will call _compute_loss)
        agent.debug_mode = True  # Enable debug mode temporarily
        loss, td_errors = agent._compute_loss(batch_tuple, weights_batch.cpu().numpy().squeeze())

        print(f"CUDA memory after loss computation: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"Loss value: {loss.item():.4f}")

        # Clean up
        del (
            agent,
            market_data,
            account_state,
            output,
            market_data_batch,
            account_state_batch,
            actions_batch,
            rewards_batch,
            next_market_data_batch,
            next_account_state_batch,
            dones_batch,
            weights_batch,
            batch_tuple,
            loss,
            td_errors,
        )
        torch.cuda.empty_cache()
        print(f"CUDA memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f} MB")


if __name__ == "__main__":
    test_gpu_memory()

#!/usr/bin/env python3
"""Test script to check GPU memory usage during agent training."""

import torch
from momentum_agent import RainbowDQNAgent
from momentum_agent.constants import ACCOUNT_STATE_DIM


def test_gpu_memory():
    """Test GPU memory usage with a small agent instantiation."""
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory before agent creation: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        config = {
            "window_size": 60,
            "n_features": 12,
            "hidden_dim": 256,
            "num_actions": 6,
            "num_atoms": 101,
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

        print(f"CUDA memory after agent creation: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Model parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

        batch_size = 512
        market_data = torch.randn(batch_size, 60, 12, device=device)
        account_state = torch.randn(batch_size, ACCOUNT_STATE_DIM, device=device)

        print(f"CUDA memory before forward pass: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        with torch.no_grad():
            output = agent.network(market_data, account_state)

        print(f"CUDA memory after forward pass: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Output shape: {output.shape}")

        del agent, market_data, account_state, output
        torch.cuda.empty_cache()
        print(f"CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


if __name__ == "__main__":
    test_gpu_memory()

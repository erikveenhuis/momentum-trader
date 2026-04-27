"""Helper for applying ``torch.compile`` to the Rainbow networks.

Extracted from ``agent.py`` so the large fall-back loop that tries multiple
compile modes lives outside the already-dense agent constructor.
"""

from __future__ import annotations

import torch
from momentum_core.logging import get_logger

from momentum_agent.constants import ACCOUNT_STATE_DIM

logger = get_logger(__name__)


def compile_networks_or_raise(
    network: torch.nn.Module,
    target_network: torch.nn.Module,
    *,
    device: torch.device,
    window_size: int,
    n_features: int,
    smoke_test_quantiles: int = 8,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Apply ``torch.compile`` to ``network`` / ``target_network``.

    Tries several compile modes in priority order and validates each by
    running a tiny forward pass. If every mode fails, raises
    ``RuntimeError`` with an actionable hint (torch.compile is required on
    the training path for acceptable throughput).

    The IQN forward signature is ``(market_data, account_state, taus)``; we
    pass a tiny ``[1, smoke_test_quantiles]`` tau tensor so the smoke test
    actually runs the quantile embedding + dueling head.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile is not available in this PyTorch version. Please upgrade to PyTorch 2.0+ for optimal performance."
        )

    logger.info("Applying torch.compile to network and target_network (REQUIRED for training).")
    compile_modes = ["default", "reduce-overhead", "max-autotune"]

    for mode in compile_modes:
        try:
            logger.info(f"Trying torch.compile with mode='{mode}'...")
            compiled_network = torch.compile(network, mode=mode)
            compiled_target_network = torch.compile(target_network, mode=mode)

            with torch.no_grad():
                test_market = torch.zeros((1, window_size, n_features), device=device)
                test_account = torch.zeros((1, ACCOUNT_STATE_DIM), device=device)
                test_taus = torch.rand((1, smoke_test_quantiles), device=device)
                _ = compiled_network(test_market, test_account, test_taus)

            logger.info(f"torch.compile applied successfully with mode='{mode}'.")
            return compiled_network, compiled_target_network

        except ImportError as imp_err:
            logger.error(f"torch.compile mode '{mode}' failed due to import issue: {imp_err}.")
        except RuntimeError as runtime_err:
            if "Triton" in str(runtime_err) or "triton" in str(runtime_err).lower():
                logger.error(f"torch.compile mode '{mode}' failed due to Triton backend issues: {runtime_err}.")
            else:
                logger.error(f"torch.compile mode '{mode}' failed with runtime error: {runtime_err}.")

    raise RuntimeError(
        "All torch.compile modes failed. torch.compile is REQUIRED for training. "
        "Please ensure you have:\n"
        "1. CUDA-compatible GPU\n"
        "2. GCC compiler installed (sudo apt install build-essential)\n"
        "3. Python development headers (sudo apt install python3-dev)\n"
        "4. Working Triton installation\n"
        "Training cannot proceed without compilation optimization."
    )

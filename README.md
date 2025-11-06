# Momentum Trader Monorepo

Reinforcement learning workflow for momentum-based cryptocurrency trading. The repository bundles reusable packages for data processing, training a Rainbow DQN agent, and running the policy live against Alpaca's crypto feed.

## Requirements

- Python 3.13+
- Windows 10/11 (automation scripts are written for Windows; Linux/macOS are supported via the Python CLIs)
- CUDA-capable GPU (optional but recommended for training and live inference)
- Alpaca Markets account with crypto paper trading enabled (for live trading mode)

## Project Structure

```
momentum-trader/
├── packages/
│   ├── momentum_core/     # Shared logging and utilities
│   ├── momentum_env/      # Gymnasium-based trading environment
│   ├── momentum_agent/    # Rainbow DQN (Transformer encoder + PER)
│   ├── momentum_train/    # Training loop, metrics, experiment tooling
│   └── momentum_live/     # Live trading CLI for Alpaca streams
├── config/                # YAML configs for training & data prep
├── data/                  # Raw, extracted, and processed datasets
├── logs/                  # Rotating logs for training & live trading
├── models/                # Saved checkpoints from `momentum_train`
├── scripts/               # Data ingestion and maintenance scripts
├── install_packages.bat   # Windows helper to install editable packages
├── refresh_packages.bat   # Reinstall editable packages in-place
├── start_training*.bat    # Training shortcuts (resume, tensorboard, etc.)
├── start_live_trading*.bat# Live trading launcher & PowerShell variant
├── LIVE_TRADING_README.md # Detailed live trading guide
└── README.md
```

## Packages

### momentum_core
Shared infrastructure (logging helpers, environment-variable overrides, rotating file handlers) used by every other package.

**Install**
```bash
cd packages/momentum_core
pip install -e .
```

**Use**
```python
from momentum_core.logging import setup_logging, get_logger
```

### momentum_env
Gymnasium-compatible trading environment that surfaces normalized OHLCV windows, shared cash state, configurable fees, and reward shaping hooks.

**Install**
```bash
cd packages/momentum_env
pip install -e .
```

**Use**
```python
from momentum_env import TradingEnv, TradingEnvConfig
```

### momentum_agent
Rainbow DQN agent with Transformer encoders, distributional value heads (C51), prioritized replay, noisy nets, dueling heads, and multi-step returns.

**Install**
```bash
cd packages/momentum_agent
pip install -e .
```

**Use**
```python
from momentum_agent import RainbowDQNAgent
```

### momentum_train
Training orchestration: checkpointing, evaluation, metrics (Sharpe ratio, max drawdown), curriculum hooks, WandB/TensorBoard logging, and CLI entry points.

**Install**
```bash
cd packages/momentum_train
pip install -e .
```

**Use**
```python
python -m momentum_train.run_training --config_path config/training_config.yaml
```

### momentum_live
End-to-end live trading loop that loads trained checkpoints, streams Alpaca crypto bars, and executes paper (or live) orders with shared cash management and safety rails.

**Install**
```bash
cd packages/momentum_live
pip install -e .
```

**Use**
```python
python -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO
```

## Setup

1. Create and activate a Python 3.13 virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate            # Linux/macOS
   .\venv\Scripts\activate           # Windows PowerShell / CMD
   python -m pip install --upgrade pip
   ```
2. Install the packages in editable mode (top-level script handles ordering on Windows).
   ```bash
   # Windows shortcut
   .\install_packages.bat

   # Cross-platform manual install
   pip install -e packages/momentum_core \
               -e packages/momentum_env \
               -e packages/momentum_agent \
               -e packages/momentum_train \
               -e packages/momentum_live
   ```
3. (Optional) Enable Git hooks to enforce `pytest` before each commit.
   ```bash
   git config core.hooksPath githooks
   ```

## Data Pipeline

Historical data lives in `data/` and is organized by stage:

```
data/
├── raw/        # Original vendor files (e.g. Polygon aggregates)
├── extracted/  # One file per symbol/day after cleaning
└── processed/  # Train/validation/test parquet/CSV windows
```

Helpers in `scripts/data_processing/` handle ingestion and cleaning:

- `extract_raw.py` – decompresses vendor archives, normalizes ticker names, filters incomplete days, and logs anomalies. Configure via `config/extract_raw_config.yaml`.
- `split_data.py` – builds rolling windows and splits into train/validation/test according to `config/split_config.yaml`.
- `collect_raw_data.sh` – optional shell script to fetch new raw dumps.

Run them with the virtual environment active, for example:

```bash
python scripts/data_processing/extract_raw.py --config config/extract_raw_config.yaml
python scripts/data_processing/split_data.py --config config/split_config.yaml
```

## Training Workflow

1. Review `config/training_config.yaml` for hyperparameters, logging, resume behaviour, and rendering toggles.
2. Launch training:
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml
   ```
   - `--resume` reuses the latest checkpoint in `models/`.
   - `--mode eval` runs evaluation against the validation/test splits.
3. Windows shortcuts
   - `start_training.bat` – launches training in a separate console, prevents sleep, and restores power settings on exit.
   - `start_training_resume.bat` – resumes from the latest checkpoint.
   - `start_tensorboard.bat` – opens TensorBoard pointed at `logs/tensorboard/`.

Metrics and debug output are written to `logs/training.log` using the shared `momentum_core` logging stack.

## Live Trading

All live-trading specifics live in `packages/momentum_live` plus the supporting scripts:

- `start_live_trading.bat` / `.ps1` – wraps environment activation, credential checks, lock-file handling, account reset for paper trading, and launches `momentum_live.cli`.
- `set_alpaca_credentials.(bat|ps1)` – convenience helper to export Alpaca keys (or load them via `.env`).
- `LIVE_TRADING_README.md` – detailed walkthrough, requirements, and troubleshooting tips.

Manual CLI usage:

```bash
python -m momentum_live.cli \
    --symbols BTC/USD,ETH/USD \
    --models-dir models \
    --log-level INFO
```

By default the CLI searches `models/` for the best checkpoint (based on validation score) and falls back to a freshly initialized agent if loading fails. Logs stream to `logs/live_trading.log`.

## Features Snapshot

- **Environment** – configurable window sizes, normalized OHLCV features, discrete allocation actions (buy/sell in 10–100% tiers), account state tracking, configurable fees and invalid-action penalties.
- **Agent** – Transformer encoder over price history, dueling C51 heads, prioritized replay with beta annealing, multi-step returns, noisy exploration layers, gradient clipping, and optional LR scheduling.
- **Training** – resumable checkpoints, gradient accumulation, validation hooks, automated metrics (Sharpe, max drawdown, volatility), TensorBoard + Weights & Biases support, configurable rendering for human/terminal modes.
- **Live Trading** – Alpaca data stream consumer with shared balance across symbols, safety checks (cash availability, position limits, min order sizes), optional automatic paper account reset, and graceful shutdown handling.
- **Logging** – consistent formatting and rotating file handlers via `momentum_core.logging` for Python CLIs and standalone scripts.

## Tests & Quality Gates

```bash
pytest                                   # run full test suite
pytest packages/momentum_env/tests       # per-package tests
pytest --cov=momentum_env --cov=momentum_agent --cov=momentum_train
```

Continuous integration is backed by the optional Git pre-commit hook (`githooks/pre-commit`) that runs `pytest` before allowing commits.

## Development Notes

- Editable installs mean changes under `packages/<name>/src/<name>/` are immediately importable without reinstalling.
- When adding dependencies, update the relevant `pyproject.toml` and rerun `pip install -e .` for that package (or `refresh_packages.bat`).
- Shared utilities belong in `momentum_core` to avoid circular dependencies.

## Additional Documentation

- `LIVE_TRADING_README.md` – deep dive into the live trading stack.
- `htmlcov/` – HTML coverage report produced by `pytest --cov`.
- `logs/` – inspect training and live trading runs.

## License

MIT License

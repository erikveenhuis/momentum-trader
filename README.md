# Momentum Trader Monorepo

Reinforcement learning workflow for intra-hour momentum-based cryptocurrency trading. The repository bundles reusable packages for data processing, training a Rainbow DQN agent with a Transformer encoder, and running the policy live against Alpaca's crypto feed.

## Requirements

- Python 3.13+
- Ubuntu Linux (tested on 6.17 kernel) with CUDA-capable GPU (RTX 5090 recommended)
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
├── models/                # Saved checkpoints from momentum_train
└── scripts/               # Data ingestion, preprocessing, and utilities
```

## Architecture Overview

The agent is a full **Rainbow DQN** (C51 distributional + PER + dueling + noisy nets + n-step + double DQN) with a **Transformer encoder** as the feature backbone.

**Key design decisions:**
- **12 features**: 6 raw (OHLCV + transactions) with window-level z-score normalization, plus 6 precomputed derived features (log returns at lag 1/5/10, realized volatility, volume ratio, high-low range ratio)
- **5-D account state**: position fraction, cash fraction, unrealized PnL, bars-in-position, cumulative fee fraction
- **Benchmark-relative reward**: excess return vs a fixed allocation benchmark, minus drawdown penalty (`lambda * max_drawdown_increment`)
- **Slippage model**: configurable basis-point slippage on all trades
- **bfloat16 AMP**: native Blackwell tensor core support, no GradScaler needed
- **Polyak soft target updates** (tau=0.001) instead of hard copy
- **Auxiliary return-prediction head** on the Transformer CLS output
- **Target allocation actions**: 6 discrete exposure levels (0%--100% in 20% steps) where every action is always valid
- **Curriculum learning**: progressively expands training data pool over the training run
- **Pre-norm Transformer + GELU** activation for better gradient flow

## Setup

1. Create and activate a Python 3.13 virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. Install packages in editable mode:
   ```bash
   python install_packages.py
   # or manually:
   pip install -e packages/momentum_core \
               -e packages/momentum_env \
               -e packages/momentum_agent \
               -e packages/momentum_train \
               -e packages/momentum_live
   ```

3. (Optional) Enable Git hooks:
   ```bash
   git config core.hooksPath githooks
   ```

## Data Pipeline

Historical data lives in `data/` and is organized by stage:

```
data/
├── raw/        # Original vendor files (Massive / formerly Polygon aggregates)
├── extracted/  # One file per symbol/day after cleaning
└── processed/  # Train/validation/test splits (CSV and/or .npz)
```

### Step 1: Extract raw data
```bash
python scripts/data_processing/extract_raw.py
```
Reads compressed CSVs, filters anomalies, and outputs clean per-day CSVs. Configure via `config/extract_raw_config.yaml`.

### Step 2: Split into train/validation/test
```bash
python scripts/data_processing/split_data.py
```
Chronological split by calendar months. Configure via `config/split_config.yaml`.

### Step 3: Preprocess to .npz (required before training)
```bash
python scripts/data_processing/preprocess_npz.py
```
Converts all CSVs in `data/processed/` to compressed `.npz` files containing precomputed features. This step is required before training -- it provides ~10-20x faster episode loading compared to raw CSV.

Each `.npz` file contains:
- `close_prices`: float32 array for trade execution
- `features`: float32 array with 12 columns (6 raw OHLCV+transactions + 6 derived features)

## Training

1. Review `config/training_config.yaml` for hyperparameters.
2. Launch training:
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml
   ```
   - `--resume` reuses the latest checkpoint in `models/`.
   - `--mode eval` runs evaluation against test splits.

## Live Trading

```bash
python -m momentum_live.cli \
    --symbols BTC/USD,ETH/USD \
    --models-dir models \
    --log-level INFO
```

See `LIVE_TRADING_README.md` for detailed setup, credentials, and troubleshooting.

## Tests

```bash
pytest                                   # run full test suite
pytest packages/momentum_env/tests       # per-package tests
pytest -m unit                           # unit tests only
pytest -m integration                    # integration tests (requires GPU + data)
```

## License

MIT License

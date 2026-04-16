# Momentum Trader

Reinforcement learning workflow for intra-hour momentum-based cryptocurrency trading. The repo bundles reusable packages for data processing, training a Rainbow DQN agent with a Transformer encoder, and running the policy live against Alpaca's crypto feed.

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

Key design decisions:
- **12 features**: 6 raw (OHLCV + transactions) with window-level z-score normalization, plus 6 precomputed derived features (log returns at lag 1/5/10, realized volatility, volume ratio, high-low range ratio)
- **5-D account state**: position fraction, cash fraction, unrealized PnL, bars-in-position, cumulative fee fraction
- **Benchmark-relative reward**: excess return vs a fixed allocation benchmark, minus drawdown penalty (`lambda * max_drawdown_increment`)
- **Slippage model**: configurable basis-point slippage on all trades
- **bfloat16 AMP**: native Blackwell tensor core support, no GradScaler needed
- **Polyak soft target updates** (tau=0.001) instead of hard copy
- **Auxiliary return-prediction head** on the Transformer CLS output
- **Target allocation actions**: 6 discrete exposure levels (0%–100% in 20% steps); every action is always valid
- **Curriculum learning**: progressively expands the training file pool (30% → 100%) over the run
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

Historical data lives under `data/` and is processed in three stages:

```
data/
├── raw/        # Original vendor files (Massive / formerly Polygon aggregates)
├── extracted/  # One CSV per symbol/day after cleaning
└── processed/  # Train/validation/test splits (CSV and .npz)
```

### Step 1: Extract raw data
```bash
python scripts/data_processing/extract_raw.py
```
Reads compressed Polygon/Massive aggregates, filters for USD pairs and complete trading days, drops excluded tickers, and writes per-day CSVs. Configure via `config/extract_raw_config.yaml`.

### Step 2: Split into train/validation/test
```bash
python scripts/data_processing/split_data.py
```
Chronological split by calendar months. Configure via `config/split_config.yaml`.

### Step 3: Preprocess to `.npz` (required before training)
```bash
python scripts/data_processing/preprocess_npz.py
```
Converts every CSV in `data/processed/` to a compressed `.npz` with precomputed features. This step is required before training and provides ~10–20× faster episode loading than reading CSVs.

Each `.npz` contains:
- `close_prices`: `float32` array for trade execution
- `features`: `float32` array with 12 columns (6 raw OHLCV+transactions + 6 derived features)

The preprocessor also **rejects dual-feed-contaminated files** (minute bars where historical Polygon/Massive aggregations interleaved two venue streams, producing close prices that alternate between two levels). A file is rejected if it has ≥100 alternating `>5%` bar pairs, or oracle log-growth ≥25 within a narrow (`<5×`) daily range — see `check_price_contamination()` in `scripts/data_processing/preprocess_npz.py`. Rejected symbols are logged with the reason and their `.npz` is not written.

## GPU Stability (RTX 5090)

The RTX 5090 requires the **open** NVIDIA kernel module (`nvidia-driver-590-open`) for display output. The `nvidia-powerd` daemon does not yet support this GPU (PCI ID `0x2b85`) and exits on startup; this is harmless on desktop since hardware power management is handled by the driver directly.

To prevent system freezes during sustained training, cap the power, and enable persistence mode:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 500
```

To persist across reboots, enable the provided systemd service:

```bash
sudo cp config/nvidia-powerlimit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-powerlimit.service
```

Monitor GPU thermals during training in a separate terminal:

```bash
nvidia-smi dmon -s pucvmet -d 5 | tee gpu_monitor.log
```

## Training

1. Review `config/training_config.yaml` for hyperparameters.
2. Launch training:
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml
   ```
3. Resume from the latest checkpoint after a crash or interruption:
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml --resume
   ```
4. Resume and reset the learning rate to the config value (discards optimizer/scheduler state):
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml --resume --reset-lr-on-resume
   ```
5. Run evaluation against the test splits:
   ```bash
   python -m momentum_train.run_training --config_path config/training_config.yaml --mode eval
   ```

TensorBoard (isolated venv):
```bash
./scripts/tensorboard.sh --port 6006 --logdir models/runs
```

## Live Trading (Alpaca)

### Quick start

```bash
source venv/bin/activate
source scripts/env-paper.sh   # loads ALPACA_API_KEY / ALPACA_API_SECRET from .env

python -m momentum_live.cli \
    --symbols BTC/USD,ETH/USD \
    --models-dir models \
    --log-level INFO
```

The CLI picks the best checkpoint in `--models-dir` (by validation score) and falls back to a fresh agent if loading fails. Live inference uses `inference_only=True` with eager forward (no `torch.compile`); default device is CPU. Set `MOMENTUM_LIVE_DEVICE=cuda` for GPU inference.

### Credentials

Put these in a gitignored `.env` at the repo root and use `scripts/env-paper.sh` to load them:

```bash
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
ALPACA_PAPER_TRADING=true
```

Verify with `echo $ALPACA_API_KEY` before running the CLI.

### What it does

- Connects to Alpaca's live crypto data stream.
- Processes real-time minute bars into the same 12-feature observation + 5-D account state used in training (identical window-level z-score + derived features).
- Queries the Rainbow agent for a target-allocation action (0%, 20%, 40%, 60%, 80%, 100%).
- Executes buy/sell deltas on your Alpaca paper account, with shared cash balance across symbols.

Log line format:
```
BTC/USD BUY 25% at 45123.45 | Shared PV 1050.67 (Balance: 750.23) | valid=True | Order: abc123
ETH/USD HOLD 0% at 2456.78 | Shared PV 1050.67 (Balance: 750.23) | valid=True
```

Stop with `Ctrl+C`; connections close gracefully and final state is logged.

### Safety features

- Paper trading by default (never uses real money unless `ALPACA_PAPER_TRADING=false` is set explicitly).
- Position limits (cannot sell more than owned).
- Cash checks (cannot buy with insufficient balance).
- Target-allocation actions (all 6 exposure levels are always valid, no action masking needed).
- `min_trade_notional` = $10 (matches Alpaca live requirement) to avoid tiny, costly orders.
- Basis-point slippage modeled during training for realism.
- Graceful error recovery on transient stream errors.

## Tests

```bash
pytest                                   # run full test suite
pytest packages/momentum_env/tests       # per-package tests
pytest -m unit                           # unit tests only
pytest -m integration                    # integration tests (requires GPU + data)
```

## Logging

All packages use the shared logger in `packages/momentum_core/logging.py`. Default behavior is console logging plus rotating file logs under `logs/` (10 files × 1 MB each). Environment overrides:

```bash
MOMENTUM_LOG_DIR=/var/log/momentum
MOMENTUM_LOG_LEVEL=WARNING           # global level
MOMENTUM_LOG_LEVEL_MOMENTUM_TRAIN=DEBUG   # per-package override
```

## Disclaimer

Experimental software for research purposes. Always verify trades and use at your own risk.

## License

MIT License

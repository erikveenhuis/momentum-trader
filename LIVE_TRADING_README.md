# Momentum Live Trading System

Real-time cryptocurrency trading using reinforcement learning on Alpaca's paper trading platform.

## Quick Start

```bash
source venv/bin/activate

# Set credentials
export ALPACA_API_KEY=your_key_here
export ALPACA_API_SECRET=your_secret_here
export ALPACA_PAPER_TRADING=true

# Run live trading
python -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO
```

## What It Does

- Connects to Alpaca's live crypto data stream
- Processes real-time bars and computes 12-feature observations (OHLCV + transactions + derived momentum/volatility features)
- Makes decisions using a Rainbow DQN agent with action masking
- Executes orders on your Alpaca paper trading account
- Tracks portfolio with shared balance across symbols

## System Requirements

- Ubuntu Linux with Python 3.13+
- Alpaca paper trading account
- CUDA-compatible GPU (recommended for inference speed)

## Configuration

### Symbols
```bash
python -m momentum_live.cli --symbols BTC/USD,ETH/USD,SOL/USD
```

### Credentials
Set these environment variables before running:
```bash
export ALPACA_API_KEY=your_key_here
export ALPACA_API_SECRET=your_secret_here
export ALPACA_PAPER_TRADING=true
```

Or add them to a `.env` file (not tracked by git).

### Model Directory
```bash
python -m momentum_live.cli --models-dir models --symbols BTC/USD
```

The CLI searches `models/` for the best checkpoint (by validation score) and falls back to a fresh agent if loading fails.

## Live Output

When running, you'll see logs like:
```
BTC/USD BUY 25% at 45123.45 | Shared PV 1050.67 (Balance: 750.23) | valid=True | Order: abc123
ETH/USD HOLD 0% at 2456.78 | Shared PV 1050.67 (Balance: 750.23) | valid=True
```

## Stopping the System

Press `Ctrl+C` to gracefully stop. The system will close connections and save final state.

## Troubleshooting

### Common Issues

1. **"Missing Alpaca credentials"**
   - Verify environment variables are set: `echo $ALPACA_API_KEY`

2. **"Checkpoint loading failed"**
   - System automatically falls back to fresh agent
   - Normal for new installations or after architecture changes

3. **"CUDA not available"**
   - System falls back to CPU automatically
   - Inference will be slower (~500-1000ms vs ~100-200ms per decision)

## Safety Features

- Paper trading only (never uses real money unless explicitly configured)
- Position limits (cannot sell more than owned)
- Cash checks (cannot buy with insufficient balance)
- Action masking (agent never proposes invalid actions)
- Minimum order sizes (prevents tiny, costly orders)
- Slippage-aware execution (basis-point slippage modeled during training)
- Graceful error recovery

## Architecture

```
Alpaca Stream -> Feature Window (60-bar) -> Z-Score Norm + Derived Features -> Rainbow Agent -> Order Execution
                                                                                    |
                                                            Action Mask (valid actions only)
```

Features: 12 channels (OHLCV + transactions + log returns at 1/5/10 lag + realized vol + volume ratio + HL range ratio).
Account state: 5-D vector (position fraction, cash fraction, unrealized PnL, bars-in-position, cumulative fees).

## Disclaimer

This is experimental software for educational purposes. Always verify trades and use at your own risk.

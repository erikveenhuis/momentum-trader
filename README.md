# Momentum Trader

Reinforcement learning workflow for intra-hour momentum-based cryptocurrency trading. The repo bundles reusable packages for data processing, training a Rainbow DQN agent with a Transformer encoder, and running the policy live against Alpaca's crypto feed.

## Requirements

- Python 3.13+
- Linux with a CUDA-capable GPU for training (CPU works for live inference)
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
- **bfloat16 AMP**: no GradScaler needed
- **Polyak soft target updates** (tau=0.001) instead of hard copy
- **Auxiliary return-prediction head** on the Transformer CLS output
- **Target allocation actions**: 6 discrete exposure levels (0%–100% in 20% steps); every action is always valid
- **Curriculum learning**: progressively expands the training file pool (30% → 100%) over the run
- **Pre-norm Transformer + GELU** activation for better gradient flow

## Setup

The repo is a single [uv](https://docs.astral.sh/uv/) workspace; the five
packages under `packages/` are the workspace members. Install them all with:

1. Install [uv](https://docs.astral.sh/uv/) (one-time):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. From the repo root, create the virtualenv and sync every workspace
   member (plus dev extras) in one shot:
   ```bash
   uv sync --all-packages --all-extras
   source .venv/bin/activate
   ```

   `uv sync` is idempotent: re-run it any time a `pyproject.toml` changes
   to pick up new dependencies.

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

## GPU notes

Long training runs benefit from enabling persistence mode and capping the GPU's
power draw to keep thermals predictable. The repo ships a sample systemd unit
at `config/nvidia-powerlimit.service` — the wattage in that file is tuned for
the developer's hardware, so adjust `nvidia-smi -pl <watts>` to a value
appropriate for your card before installing it:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl <watts>          # adjust to your card
```

Optionally monitor thermals in a separate terminal during training:

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

## TensorBoard KPI Hierarchy (Sniper-Focused)

The agent is a momentum *sniper* — it stays flat (action 0 = 0% allocation)
most of the time and only opens a position when an edge appears. Portfolio-
level Sharpe / Max-Drawdown are misleading for that pattern (they are diluted
by long flat stretches), so the per-trade economics under
`Validation/Trade/*` and `Test/Trade/*` are the **primary KPIs**. Portfolio
metrics remain available as secondary references and as inputs to anomaly
detection.

### Primary KPIs — per-trade economics

A *trade* opens when the position transitions from zero to non-zero and
closes when it returns to zero. Each closed trade contributes one
observation regardless of how long the agent stayed flat around it.

| Tag | Meaning |
| --- | --- |
| `Validation/Trade/PerTradeSharpe` | `mean(pnl%) / std(pnl%)` across closed trades |
| `Validation/Trade/HitRate` | Fraction of trades with positive PnL |
| `Validation/Trade/Expectancy` | Mean PnL % per trade |
| `Validation/Trade/ProfitFactor` | `sum(wins) / sum(|losses|)` |
| `Validation/Trade/AvgMAE` / `Validation/Trade/AvgMFE` | Avg max-adverse / max-favorable excursion (%) |
| `Validation/Trade/AvgDuration` | Avg in-trade step count |
| `Validation/Trade/PctGreedy` | Fraction of in-trade steps chosen by argmax (vs ε-greedy) |
| `Validation/Trade/PnLDistribution` | Histogram of trade PnL % |
| `Test/Trade/*` | Same surface, computed once per test sweep |

A JSONL sidecar with the full `TradeRecord` for each closed validation /
test trade is written next to the run logs for offline replay
(`*_trades.jsonl`).

### Secondary KPIs — portfolio & action mix

| Tag | Meaning |
| --- | --- |
| `Validation/Final Portfolio Value` | End-of-episode equity |
| `Validation/Transaction Costs` | Cumulative fees + slippage |
| `Validation/Action Rate/0..5` | Fraction of steps taken at each exposure level |
| `Train/Action Rate/Greedy/k` & `Train/Action Rate/Eps/k` | Train-time greedy vs ε-forced split per action |
| `Train/EpsilonForcedTradeFraction` | Share of non-zero training actions that came from ε-greedy |
| `Train/EvalGap/*` | Signed difference between greedy validation metric and the recent stochastic training metric (positive = eval better than train) |

### Diagnostics — model-health signals

These tags are emitted on a configurable interval to keep training cheap.
Use them to detect divergence, dead exploration, or PER pathology before
they show up as a degraded KPI.

| Tag prefix | Purpose |
| --- | --- |
| `Train/Q/*` (`Mean`, `Std`, `MaxAcrossActions`, `MinAcrossActions`, `ActionMargin`, `PerAction/Mean/k`, `Distribution`) | Q-value sanity (collapse, divergence, action confidence) |
| `Train/Noisy/*` (`weight_sigma_mean/max/min` per `NoisyLinear` block) | NoisyNet exploration breadth |
| `Train/Grad/Norm/*` (global + per module group) and `Train/ParamUpdateRatio` | Optimizer stability (vanishing / exploding / dead modules) |
| `Train/TargetNet/SoftUpdates` and `Train/TargetNet/ParamDeviation` | Polyak target-net divergence |
| `Train/Categorical/*` and `Train/Categorical/Distribution` | C51 distributional target health |
| `Train/PER/*` (priority stats, `Train/PER/Reward/*`, `Train/PER/PriorityByAction/k`, `Train/PER/Top1PctActionShare/k`) | Replay-buffer balance, action-class skew, FIFO half-life |
| `Train/NStep/*` | n-step reward window — confirms the n-step discounting integrates as expected |
| `Train/Episode/Reward{Min,Max,P99Abs,OutlierFlag}` | Per-episode reward outlier guard (flag fires when any reward exceeds 5× `reward_clip`) |
| `Train/Reward/MeanByAction/k` & `Train/Reward/StdByAction/k` | Per-episode reward attribution to each action |
| `Train/Hyper/BenchmarkFrac` | Current `benchmark_allocation_frac` (after schedule + override). See *Reward design — benchmark allocation* below. |
| `Train/Diagnostics/ValidationSkippedDueToVectorJump` | Fires once when the vectorized loop's "crossed-a-multiple" gate would have skipped a `validation_freq` boundary that the legacy `% == 0` check would have missed. |
| `Agent/NoisySigmaReset` | One-shot scalar logged whenever `RainbowDQNAgent.reset_noisy_sigma()` runs (recovery / `--reset-noisy-on-resume`). Value = number of `NoisyLinear` layers refilled. |

### Tag taxonomy at a glance

```
Train/Action Rate/{Greedy|Eps}/{0..5}
Train/Trade/{HitRate, Expectancy, PerTradeSharpe, ...}
Train/Q/{Mean, Std, MaxAcrossActions, MinAcrossActions, ActionMargin, PerAction/Mean/k, Distribution}
Train/Noisy/{layer}_{weight|bias}_sigma_{mean|max|min}
Train/Grad/{Norm, Group/{encoder|advantage|value|return_pred}}
Train/ParamUpdateRatio
Train/TargetNet/{SoftUpdates, ParamDeviation}
Train/Categorical/{Mean, Std, Entropy, Distribution}
Train/PER/{Sample/*, Priority/*, Reward/*, PriorityByAction/k, Top1PctActionShare/k}
Train/NStep/{Reward/Mean, Reward/Std, Reward/Skew, Reward/Kurtosis, ClipSaturationRate}
Train/Episode/{Reward, RewardMin, RewardMax, RewardP99Abs, RewardOutlierFlag}
Train/Reward/{MeanByAction, StdByAction}/k
Train/EvalGap/{ActionRate/k, ...}

Validation/{Action Rate/k, Final Portfolio Value, Transaction Costs}
Validation/Trade/{Count, HitRate, Expectancy, PerTradeSharpe, ProfitFactor,
                  AvgDuration, AvgMAE, AvgMFE, PctGreedy, PnLDistribution}

Test/{Portfolio/*, Action Rate/k}
Test/Trade/{Count, HitRate, Expectancy, PerTradeSharpe, ProfitFactor,
            AvgDuration, AvgMAE, AvgMFE, PctGreedy, PnLDistribution}

Live/{Action Rate/k, Action/<symbol>, PortfolioValue/<symbol>, Position/<symbol>}
Live/Q/{Mean, Std, MaxAcrossActions, MinAcrossActions, ActionMargin, Selected}
Live/Trade/{Count, HitRate, Expectancy, PerTradeSharpe, AvgDuration,
            AvgMAE, AvgMFE, PctGreedy, TotalPnLAbs, TotalTxnCost}/<symbol>
```

### Evaluation determinism (`agent.greedy()` / `--eval-stochastic`)

Validation and test rollouts run inside `agent.greedy()` by default, which
freezes NoisyNet noise and forces ε=0 so the recorded scalars reflect the
same policy that ships to the live trader. Pass `--eval-stochastic` to
`run_training` to keep the agent in training mode during evaluation
(useful for explicitly measuring exploration overhead).

The `Train/EvalGap/*` scalars are computed at validation time as
`(validation greedy metric) - (recent training stochastic metric)`. A
persistent gap above noise typically indicates either over-exploration in
training or a stale / over-fit greedy policy.

## Offline analysis utilities

Three scripts under `scripts/` consume the artifacts produced during
training to give a deeper, deterministic view than TensorBoard alone.

```bash
# 1) Deterministic rollouts on the latest checkpoint over train/val/test.
#    Writes per-step parquet, trade JSONL, Q-distribution snapshots, and
#    a non-action-0 trigger log under <output_dir>/<split>/.
python scripts/eval_greedy.py \
    --config_path config/training_config.yaml \
    --output-dir reports/eval_greedy

# 2) Full-buffer scan of the PER replay buffer from a trainer checkpoint.
#    Emits JSON with reward histogram, per-action priority stats, FIFO
#    half-life, and the top-N highest-priority transitions.
python scripts/audit_per_buffer.py \
    --checkpoint models/checkpoint_trainer_resume.pt \
    --output reports/per_buffer_audit.json

# 3) Per-trade analytics consuming the eval_greedy outputs. Splits
#    metrics by greedy/eps provenance and rolling windows, and emits
#    equity curves, PnL CDF, and bootstrap CIs for Sharpe / Expectancy.
python scripts/analyze_trades.py \
    --input-dir reports/eval_greedy \
    --output reports/trade_analysis.json
```

## Reward design — benchmark allocation

The trading reward subtracts a fraction of each bar's price return so the
agent has to **beat** a fixed-allocation baseline rather than just ride the
market (see `packages/momentum_env/src/momentum_env/trading.py`):

```
excess = (market_return + trade_return) - benchmark_allocation_frac * price_return
reward = reward_scale * (excess - drawdown_penalty - opportunity_cost) - invalid_action_penalty
```

A high `benchmark_allocation_frac` is useful **early** as anti-collapse
pressure — sitting flat in any uptrending bar earns negative reward, so
the agent can't just learn "always action 0". Once the agent has
demonstrated non-flat action diversity, the same pressure becomes a
structural long bias that fights the sniper objective ("only trade when
there is momentum for a few minutes"). To get both behaviours from a
single training run we anneal the value linearly from `start` to `end`
over the first `anneal_episodes`. The schedule is anchored on the
absolute episode index, so `--resume` continues an in-flight schedule
instead of restarting it.

### Config keys

```yaml
trainer:
  benchmark_allocation_frac_start: 0.5       # value at episode 0
  benchmark_allocation_frac_end: 0.10        # value at and after the anneal
  benchmark_allocation_frac_anneal_episodes: 5000
```

If the schedule keys are omitted (e.g. live-trading config, random-agent
baseline), the trainer treats `environment.benchmark_allocation_frac` as
a constant for the whole run.

### CLI override

```bash
python -m momentum_train.run_training --resume --benchmark-frac-override 0.15
```

Pins the value to `0.15` for the entire run, ignoring the schedule.
Useful during recovery when you want to immediately try a relaxed value
without waiting for the anneal to catch up.

The current value is mirrored to TensorBoard as `Train/Hyper/BenchmarkFrac`
once per episode.

## Recovering from a collapsed run

Use this runbook when training has clearly regressed — Sharpe / hit rate /
returns trending down for several validation cycles, action distribution
collapsed onto one or two actions, or `Train/Q/ActionMargin` near zero.

### Diagnostics checklist

Before recovering, confirm the regression is actually a sustained collapse
rather than a transient dip. Open TensorBoard and look at:

- `Train/Action Rate/0..5` — has the distribution collapsed onto a single
  action? Compare against the curve from earlier in the same run.
- `Train/Q/ActionMargin` and `Train/Q/Std` — both at or near zero means the
  network can no longer separate good from bad actions.
- `Train/Noisy/AggregateSigmaMean` and `Train/Noisy/{layer}/SigmaMean` —
  values near zero mean NoisyNet has stopped exploring and the policy has
  effectively gone deterministic on the wrong action.
- `Train/Trade/HitRate` / `Train/Trade/Expectancy` — sustained drop is the
  symptom; the four checks above are the cause.
- `Validation/*` — present at all? If `validation_freq > num_episodes` the
  trainer now hard-fails at startup; if validation is too sparse the new
  guard logs a warning. The vectorized loop also emits
  `Train/Diagnostics/ValidationSkippedDueToVectorJump` when it catches a
  multi-env step that the legacy gate would have skipped.
- `Train/Hyper/BenchmarkFrac` — confirm the schedule (or override) is
  doing what you expect.

### Recovery one-liner

```bash
python scripts/recover_from_collapse.py --strip-optimizer --reset-noisy
python -m momentum_train.run_training --resume
```

`scripts/recover_from_collapse.py` picks a healthy pre-collapse checkpoint
(highest-score `best`, or auto-picked from
`models/validation_results_*.json` paired with the corresponding
`checkpoint_trainer_latest_*_ep*.pt` when no `best` exists), bakes in the
requested resets, and writes the result as a new
`checkpoint_trainer_latest_<today>_ep<N>_rewardrecover.pt` so the bare
`--resume` picks it up. Useful flags:

- `--strip-optimizer` — drop optimizer / scheduler / scaler state so the
  fresh run starts from the config'd LR. Equivalent to baking
  `--reset-lr-on-resume` into the checkpoint.
- `--reset-noisy [--noisy-sigma-init 0.5]` — refill NoisyLinear sigma so
  exploration is re-energised. Mu (the deterministic part the policy
  actually uses for argmax) is left untouched.
- `--reset-best-validation` — zero `best_validation_metric` (-> `-inf`) and
  `early_stopping_counter` (-> `0`) in the recovery checkpoint. Use when
  the validation distribution is about to change (e.g. you widened the
  validation window — see *Re-splitting train/val/test* below) so old
  scores no longer compare to new ones, and the early-stop counter
  shouldn't carry over.
- `--from-episode N` — pick a specific `latest_*_epN_*.pt` instead of
  letting the auto-picker decide.
- `--benchmark-frac-override 0.15` — forwarded into the printed resume
  command so the run starts with a relaxed benchmark immediately.
- `--dry-run` — print the chosen checkpoint and intended mutations
  without writing anything.

### When NOT to recover

- Single noisy validation cycle. Wait one or two more `validation_freq`
  intervals to confirm the trend.
- KPI drop coincides with a curriculum jump (new training file
  distribution) — let the agent adapt for ~`validation_freq` episodes
  before deciding.
- `Train/Noisy/AggregateSigmaMean` is high and `Train/Q/ActionMargin > 0` —
  the agent is still exploring and Q-values are still informative; this
  is normal training noise, not collapse.

### Re-splitting train/val/test

When you change `validation_months` / `test_months` in
`config/split_config.yaml`, files physically move between
`data/processed/{train,validation,test}/` subdirs and the validation
distribution changes. Old `validation_results_*.json` scores and the
stored `best_validation_metric` are no longer comparable to new ones.

Two valid procedures depending on how much you want to reuse:

**Option A — Fresh start (recommended after data cleanup or non-trivial
split changes):** the cleanest substrate for evaluating new infrastructure
(reward schedule, validation cadence, trade-economics KPIs) since every
component anneals together from step 0.

```bash
# 1. SIGINT the running training; verify a final
#    checkpoint_trainer_latest_*_ep<N>_reward*.pt was written.
nvidia-smi   # confirm no rogue python processes hold GPU memory.

# 2. Archive the old run (keep for diagnostic value, isolate from new TB):
mkdir -p models/_archive_pre_split runs/_archive logs/_archive
mv models/checkpoint_trainer_*.pt   models/_archive_pre_split/
mv models/validation_results_*.json models/_archive_pre_split/
mv models/rainbow_transformer_final_agent_state.pt \
   models/_archive_pre_split/ 2>/dev/null || true
mv models/progress.jsonl            models/_archive_pre_split/
mv runs/<old_run_name>              runs/_archive/<old_run_name>_pre_split/
mv logs/training.log                logs/_archive/training_pre_split.log

# 3. Edit config/split_config.yaml (test_months, validation_months).
# 4. Re-run the data pipeline so files land in the new layout:
python scripts/data_processing/split_data.py
python scripts/data_processing/preprocess_npz.py

# 5. Sanity-check counts in data/processed/{train,validation,test}/.
# 6. Launch fresh training (no --resume, no --reset flags):
python -m momentum_train.run_training --config_path config/training_config.yaml
```

**Option B — Resume with new splits (preserves weights + PER buffer):**
only safe when boundaries move *forward* (file types-2/3 only — see the
leakage rule below). Moving the train/val boundary backward leaks
training data into validation and silently inflates scores; never do
that without retraining from scratch.

```bash
# 1-5. Same as Option A through the data pipeline re-run.

# 6. Wipe the no-longer-comparable validation_results without touching
#    checkpoints (they hold the still-useful weights + buffer):
mv models/validation_results_*.json models/_archive_pre_split/

# 7. Bake a checkpoint that resets best_validation_metric and the
#    early-stop counter (so the new validation distribution gets a
#    clean baseline) without dropping weights or NoisyLinear state:
python scripts/recover_from_collapse.py --reset-best-validation

# 8. Resume:
python -m momentum_train.run_training --resume
```

**Leakage rule** for split changes (strictest first):

1. Files the model has *trained on* (gradient updates) — never validate
   on these. Promoting train files into val is hard data leakage.
2. Files the model was *selected against* (used as validation) — safe to
   keep as validation; never promote to test.
3. Files the model has *never touched* — safe to use anywhere.

When in doubt, Option A is always correct.

### Disk hygiene — checkpoint rotation

Each periodic checkpoint is a pair on disk:

- `checkpoint_trainer_latest_<DATE>_ep<N>_reward<...>.pt` — small
  (~tens of MB; network + optimizer + scheduler state).
- `checkpoint_trainer_latest_<DATE>_ep<N>_reward<...>.buffer/` —
  side-car directory with the full PER buffer as per-field `.npy`
  memmaps (~9 GB at the default 1M capacity, dominates the pair).

The buffer was previously bundled into the `.pt` itself, but pickling
~6 GB of arrays inside `torch.save` produced 10+ GB transient pickle
streams that OOM-killed the trainer; the side-car layout writes each
field row-by-row into an on-disk memmap so peak extra RSS at save
time is O(one `Experience` row) rather than O(buffer size). See
`buffer.save_to_path` and `_save_buffer_sidecar`.

An unbounded run at the default `checkpoint_save_freq: 50` over 50k
episodes would still need ~9 TB if every pair were kept. The trainer
rotates the periodic-checkpoint stream after each save:

```yaml
# config/training_config.yaml → trainer:
checkpoint_save_freq: 50
latest_checkpoint_keep_last_n: 10    # keep ~90 GB of latest_* on disk
```

Safe to change between `--resume` runs — the new value takes effect on
the next periodic save (the trainer reads it at init, not from the
checkpoint).

Rules of the rotation:

- **Most recent N pairs survive**, sorted by the `_ep<N>_` number
  embedded in the filename (not mtime — episode ordering is the ground
  truth). Each `.pt` deletion also removes its sibling `.buffer/`
  directory so the pair stays in lockstep on disk.
- **`best_*.pt` checkpoints are never auto-rotated.** They're the
  curated "model I want to keep" snapshots and the recovery script and
  paper-trader may read them. `best_*.pt` files are written **without**
  a buffer side-car (it would explode disk usage during early
  training); to resume from one, copy the nearest `latest_*.buffer/`
  by hand.
- **Recover-script outputs** (`*_rewardrecover.pt`) are part of the
  `latest_*` stream and participate in rotation; they're written with
  bumped episode numbers so a single-keep window naturally preserves
  them as "the most recent".
- Set `latest_checkpoint_keep_last_n: 0` to disable rotation entirely
  (only do this if disk is unlimited or you're running a short
  experiment).
- Lower the value (e.g. `5` → ~45 GB) if disk is tight; raise it (e.g.
  `20` → ~180 GB) if you want a wider rollback window for
  `recover_from_collapse.py --from-episode N`.

If disk fills mid-run despite rotation, the most likely culprits are
`models/validation_results_*.json` (small but accumulates), `runs/`
TensorBoard event files (can be GBs over a long run), or stray
`models/_archive_*` directories — `du -sh models/* runs/* logs/* | sort
-h` is the right starting probe.

## Live Trading (Alpaca Broker API)

### Architecture

The agent is trained as a **single-asset allocator**: each training episode
samples one `<date>_<SYMBOL>-USD.npz` file with one cash bucket. To preserve
that distribution in production, the live runner uses the **Alpaca Broker API**
(sandbox) and provisions **one sub-account per trading pair**. Each pair gets
its own balance and its own positions, so a 100% allocation in BTC cannot
starve ETH — the multi-asset cash-pool race that the agent never saw during
training simply cannot occur.

```
                        ┌──────────────────────────────────┐
   1× CryptoDataStream ─┤  MultiPairRunner (single agent)  │
                        └─────┬──────────────────────┬─────┘
                              │ BTC bar              │ ETH bar
                              ▼                      ▼
                ┌─────────────────────┐  ┌─────────────────────┐
                │ MomentumLiveTrader  │  │ MomentumLiveTrader  │
                │  symbol=BTC/USD     │  │  symbol=ETH/USD     │
                │  → BrokerSubAcct A  │  │  → BrokerSubAcct B  │
                └─────────────────────┘  └─────────────────────┘
                          ▲                          ▲
                          └─────── BrokerClient ─────┘
                            (firm account funds JNLC)
```

### Credentials

`scripts/env-paper.sh` loads two API surfaces from a gitignored `.env`:

```bash
# Data API (CryptoDataStream / CryptoHistoricalDataClient)
ALPACA_API_KEY=...
ALPACA_API_SECRET=...

# Broker API sandbox (BrokerClient — sub-account creation, JNLC funding)
ALPACA_BROKER_API_KEY=...
ALPACA_BROKER_API_SECRET=...
ALPACA_BROKER_BASE_URL=https://broker-api.sandbox.alpaca.markets
ALPACA_BROKER_ACCOUNT_ID=<your-firm/funding-account-uuid>
```

`scripts/env-paper.sh` aborts loudly if any of these are missing. Source it
**once per shell**:

```bash
source .venv/bin/activate
source scripts/env-paper.sh
```

### Required Broker sandbox limits

The default sandbox JNLC limits are too low for typical funding/reset flows.
In the Alpaca Broker sandbox UI, raise:

- **JNLC Transaction Limit** to at least your `--initial-balance`
  (recommended: $10,000 for headroom).
- **JNLC Daily Transfer Limit** to $50,000 (matches firm cap; covers many
  resets per day across multiple pairs).

### Quick start

```bash
momentum-live \
    --symbols BTC/USD,ETH/USD \
    --models-dir models \
    --window-size 60 \
    --initial-balance 1000.0 \
    --transaction-fee 0.001 \
    --reward-scale 1.0 \
    --invalid-penalty -0.1 \
    --drawdown-penalty-lambda 0.3 \
    --slippage-bps 5.0 \
    --opportunity-cost-lambda 0.0 \
    --benchmark-allocation-frac 0.10 \
    --min-rebalance-pct 0.02 \
    --min-trade-value 1.0 \
    --reset-mode soft \
    --tb-log-dir models/runs/live \
    --log-level INFO
```

The first run **creates** one Broker sub-account per `--symbols` entry and
records them in `models/broker_subaccounts.json` (the registry is gitignored).
Subsequent runs **reuse** the same sub-accounts — IDs are stable across
restarts and across new checkpoints.

#### `--reset-mode {none, soft, hard}`

Controls the pre-run reset, the standard way to start a clean evaluation
of a new checkpoint:

- `none` — leave sub-accounts as is.
- `soft` (default) — for each sub-account: cancel open orders, close all
  positions, wait for fills, then JNLC the cash delta back to
  `--initial-balance`. Idempotent (delta < $0.01 → no journal).
- `hard` — reserved for a future revision (recreate sub-accounts).

A marker line is appended to `live_trades.jsonl` at every reset so downstream
tooling can split the log by checkpoint.

#### Manual reset between checkpoints

For ad-hoc resets without starting the live runner:

```bash
momentum-live-reset --initial-balance 1000.0
# or restrict to a subset:
momentum-live-reset --initial-balance 1000.0 --symbols BTC/USD,ETH/USD
```

This script holds an advisory file lock on `models/broker_subaccounts.lock`
so you cannot accidentally race the live runner or another reset.

### TensorBoard parity

`--tb-log-dir` mirrors the same `Live/Trade/*`, `Live/Action Rate/*`, and
`Live/Q/*` scalars that the training pipeline emits — direct parity between
the training-time TB view and the deployed agent's live behavior. A
`live_trades.jsonl` sidecar with per-trade `TradeRecord`s is written
alongside the run logs.

### Checkpoint selection

By default the CLI picks the best checkpoint in `--models-dir` matching
`checkpoint_trainer_best_*.pt` (by validation score parsed from the
filename). Override with `--checkpoint <path>` or `--checkpoint-pattern
<glob>`. If checkpoint loading fails the CLI **fails loudly** rather than
trading with a fresh agent. Inference uses `inference_only=True` with eager
forward (no `torch.compile`); default device is CPU. Set
`MOMENTUM_LIVE_DEVICE=cuda` for GPU inference.

### Safety features

- All trading is sandbox via the Broker API — no production endpoint reachable
  unless you swap the URL.
- Position limits (cannot sell more than owned per sub-account).
- Cash checks against the live sub-account balance before every order.
- Target-allocation actions (all 6 exposure levels are always valid, no
  action masking needed).
- `min_trade_value` enforced at order submission to avoid tiny, costly
  rounding errors.
- Basis-point slippage modeled during training for realism.
- Graceful error recovery on transient stream errors and Alpaca connection
  limits.

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

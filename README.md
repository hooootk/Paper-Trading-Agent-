# Paper Trading Agent

LLM + rule-based dual-engine quantitative paper trading system for US equities.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Paper_Trading_Agent.py                    │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐             │
│  │DataAgent│→ │MacroAgent│→ │StockAnalystAgent│──┐         │
│  └─────────┘  └──────────┘  └────────────────┘  │         │
│                                                  ↓          │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │FrequencyDecider│← │RiskManagerAgent │← │  Backtest    │  │
│  └──────────────┘  └──────────────────┘  │  Results     │  │
│                         ↓                 └──────────────┘  │
│                  ┌──────────────┐                           │
│                  │AlpacaExecutor│  ← Alpaca Paper API       │
│                  └──────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│                    BacktestEngine.py                         │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ FactorLibrary│→ │BacktestRunner  │→ │EvaluationMetrics│ │
│  │  (24 factors)│  │(deterministic) │  │  (12 metrics)  │  │
│  └──────────────┘  └────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## I. LLM Multi-Agent Trading Engine

### Agent pipeline

| Layer | Component | Role |
|-------|-----------|------|
| Data | **DataAgent** | Fetches US equity prices + SPY/VIX via yfinance; computes 15 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, volatility, momentum, drawdown, volume-price trend) |
| Macro | **MacroAgent** | Analyzes SPY returns + VIX to classify market phase (Bull/Bear/Ranging/Panic) and assign a 0-10 risk appetite score via GPT-4o |
| Stock | **StockAnalystAgent** | Generates BUY/SELL/HOLD signals per ticker using technical data, macro context, and backtest results |
| Risk | **RiskManagerAgent** | Evaluates portfolio-level VaR, max drawdown, volatility and backtest metrics; outputs Low/Medium/High risk level and capital action recommendation |
| Frequency | **FrequencyDecider** | LLM dynamically recommends rebalance cadence (daily/weekly/biweekly/monthly) based on VIX, market phase, realized volatility, and trend strength |
| Execution | **AlpacaExecutor** | Interfaces with Alpaca Paper Trading API — market orders, bracket orders (TP/SL), account queries, and position management |

### Rebalance execution order

Sells are submitted **before** buys to free up buying power. After a 3-second settlement wait, buying power is re-queried and each buy is individually checked against remaining budget. Buys that would exceed available funds are scaled down or skipped.

---

## II. 24-Factor Rule-Based Backtest Engine

### Factor library (5 categories, 24 factors)

| Category | Factors | Count |
|----------|---------|-------|
| **Trend** | SMA 20/50 crossover, SMA 50/200 crossover, EMA 12/26 crossover, ADX 14 | 4 |
| **Momentum** | RSI 14, MACD signal, MACD histogram, momentum 20d/60d/120d, Stochastic 14, CCI 20, Williams %R 14 | 7 |
| **Volatility** | Bollinger position, Bollinger squeeze, volatility 20d, volatility regime, ATR 14 | 5 |
| **Volume** | Volume ratio, volume price trend, OBV trend | 3 |
| **Risk/Other** | Beta (vs SPY), 5-day reversal, 60-day drawdown | 3 |

Each factor outputs a normalized signal in [-1, 1]; positive = bullish, negative = bearish.

### Strategy configuration

Each strategy is defined in `strategy_config.json`:

- **Factor selection**: subset from the 24-factor pool
- **Factor params**: weight, direction (long/inverse), long/short thresholds
- **Signal logic**: weighted sum / majority vote / top-N
- **Position sizing**: equal weight / factor-score weighted / risk parity
- **Rebalance frequency**: daily / weekly / biweekly / monthly
- **Max positions**: cap on concurrent holdings

### Backtest flow

```
Load strategy config → fetch historical data → compute all factors across full timeline
→ loop by rebalance frequency → compute target weights → log trades
→ compute daily portfolio returns → evaluate 12 performance metrics
→ single-strategy report + multi-strategy comparison table
```

### Performance metrics (12 metrics)

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return over backtest period |
| Annualized Return | CAGR equivalent |
| Sharpe Ratio | Risk-adjusted return (vs risk-free rate) |
| Calmar Ratio | Return / max drawdown |
| Sortino Ratio | Downside-only risk-adjusted return |
| Max Drawdown | Peak-to-trough decline |
| Volatility (ann.) | Annualized standard deviation |
| Beta | Sensitivity to SPY benchmark |
| Win Rate | Proportion of positive days |
| Profit/Loss Ratio | Average win / average loss |
| Profit Factor | Gross profit / gross loss |
| VaR / CVaR 95% | Value at Risk / Conditional VaR |

---

## III. Take-Profit / Stop-Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TAKE_PROFIT_PCT` | 15% | Profit reaches 15% above cost basis → take-profit trigger |
| `STOP_LOSS_PCT` | 8% | Loss reaches 8% below cost basis → stop-loss trigger |

Two usage modes:

- **Bracket order**: buy orders automatically attach TP limit + SL stop as an OCO pair
- **Post-hoc attach**: bulk-attach TP/SL orders to existing positions with custom percentages

`check_tpsl_status()` reports each position's unrealized P&L against TP/SL thresholds.

---

## IV. Multi-Strategy Parallel Trading

`run_multi_strategies()` runs 2+ strategies simultaneously with capital split:

```
Strategy #0 (momentum,   8 factors) → weights [AAPL 50%, NVDA 50%]
Strategy #2 (value,      6 factors) → weights [JNJ 60%, PG   40%]
Strategy #3 (reversal,   5 factors) → weights [XOM 40%, BAC  60%]

Capital 1:1:1 → merged weights:
  AAPL 16.7%, NVDA 16.7%, JNJ 20%, PG 13.3%, XOM 13.3%, BAC 20%
```

---

## V. Auto-Select Best Strategy

`auto_select_and_trade()` runs a two-phase pipeline:

1. **Backtest all** strategies in `strategy_config.json`
2. **Rank** by composite score across 6 metrics (Sharpe, Calmar, Sortino, Profit Factor, Max Drawdown, Win Rate)
3. **Execute** paper trading using the top-ranked strategy

Available via menu `[13]` or CLI `--auto-select`.

---

## VI. Natural Language Intent Recognition

Menu option `[14]` accepts free-form English or Chinese input and uses GPT-4o to match user intent to the closest existing function(s):

- **Single match with high confidence (≥0.7)**: auto-executes after confirmation
- **Single match with low confidence**: prompts for explicit confirmation
- **Multiple matches**: lists candidates, user picks one
- **No match**: reports failure and lists all available functions

---

## VII. Dual-Engine Comparison

| Dimension | LLM Multi-Agent | 24-Factor Rule Engine |
|-----------|----------------|----------------------|
| Decision method | GPT-4o reasoning | Deterministic algorithm |
| Factor source | 15 technical indicators | 24-factor library |
| Macro analysis | MacroAgent (SPY/VIX) | None |
| Risk management | RiskManagerAgent (VaR, DD) | Factor threshold filtering |
| Frequency | LLM dynamic recommendation | Fixed in strategy config |
| TP/SL | Bracket order support | Optional |
| Multi-strategy | Not supported | Supported |
| Cost | Azure OpenAI API calls | Zero API cost |
| Speed | Slower (LLM + rate limits) | Fast (pure computation) |

---

## VIII. Interactive Menu

```
╔══════════════════════════════════════════════════╗
║              PAPER TRADING AGENT                 ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  [1]  LLM Multi-Agent Trading (Dry Run)          ║
║  [2]  LLM Multi-Agent Trading (Real Orders)      ║
║  [3]  Strategy-Based Trading (Dry Run)           ║
║  [4]  Strategy-Based Trading (Real Orders)       ║
║  [5]  Backtest All Strategies                    ║
║  [6]  Backtest Single Strategy                   ║
║  [7]  Live Trading + Auto Backtest               ║
║  [8]  Regenerate Strategy Config                 ║
║  [9]  View Alpaca Account Summary                ║
║  [10] LLM Rebalance Frequency Analysis           ║
║  [11] Multi-Strategy Parallel Trading            ║
║  [12] Manage TP/SL (Attach + Status Check)       ║
║  [13] Auto-Select Best Strategy & Trade          ║
║  [14] Natural Language Input (AI Intent Recog.)  ║
║  [0]  Exit                                       ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

---

## IX. CLI Reference

### Mode selection (mutually exclusive)

| Flag | Description |
|------|-------------|
| (no flag) | Interactive menu |
| `--live` | Real orders to Alpaca paper account |
| `--dry-run` | Full pipeline without submitting orders |
| `--backtest` | Run backtests only, no trading |
| `--auto-backtest` | Trade + LLM decides whether to backtest |
| `--generate-config` | Regenerate `strategy_config.json` |
| `--summary` | View Alpaca account summary and exit |

### Options

| Flag | Description |
|------|-------------|
| `--use-strategy N` | Trade with strategy #N (0-based, bypasses LLM) |
| `--auto-select` | Backtest all, pick best, paper trade |
| `--strategy N` | With `--backtest`: backtest only strategy #N |
| `--strategies N` | With `--generate-config`: generate N strategies (default 5) |
| `--tickers A,B,C` | Custom ticker list (comma-separated) |
| `--date YYYY-MM-DD` | Target trading date (default: today) |
| `--quiet` | Suppress verbose output |

### Examples

```bash
# Interactive menu
python Paper_Trading_Agent.py

# LLM live trading
python Paper_Trading_Agent.py --live
python Paper_Trading_Agent.py --live --date 2026-05-20
python Paper_Trading_Agent.py --live --tickers AAPL,TSLA,NVDA

# Rule-based strategy trading
python Paper_Trading_Agent.py --use-strategy 2               # dry run
python Paper_Trading_Agent.py --live --use-strategy 0        # real orders

# Backtesting
python Paper_Trading_Agent.py --backtest                      # all strategies
python Paper_Trading_Agent.py --backtest --strategy 0         # single strategy

# Auto-select best strategy
python Paper_Trading_Agent.py --auto-select                   # dry run
python Paper_Trading_Agent.py --auto-select --live            # real orders

# Auto-backtest (trade first, then LLM decides backtest)
python Paper_Trading_Agent.py --auto-backtest
python Paper_Trading_Agent.py --auto-backtest --use-strategy 1

# Config management
python Paper_Trading_Agent.py --generate-config
python Paper_Trading_Agent.py --generate-config --strategies 10

# Account summary
python Paper_Trading_Agent.py --summary
```

---

## X. Typical Workflows

### Validate a strategy idea

```bash
python Paper_Trading_Agent.py --generate-config --strategies 10
python Paper_Trading_Agent.py --backtest                      # compare all
python Paper_Trading_Agent.py --use-strategy 3                # dry-run best
```

### Daily trading session

1. `[9]` Check account status
2. `[10]` LLM rebalance frequency analysis
3. `[1]` or `[3]` Dry run to review signals
4. `[2]` or `[4]` Execute real orders
5. `[12]` Attach TP/SL to positions

### Auto-pilot

```bash
# One command: backtest all → rank → trade with best strategy
python Paper_Trading_Agent.py --auto-select
```

### Multi-strategy deployment

Interactive menu → `[11]` → enter strategy indices (e.g., `0,1,3`) → choose capital weights → real orders → bracket TP/SL.

---

## XI. Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | Azure OpenAI GPT-4o |
| Market data | yfinance (Yahoo Finance) |
| Trading API | Alpaca Markets Paper Trading API |
| Computation | NumPy / Pandas |
| Language | Python 3 |

---

## XII. Default Portfolio

**LLM trading (Paper_Trading_Agent)**: AAPL, MSFT, JPM, JNJ, XOM, AMZN, NVDA, UNH, BAC, PG

**Backtest strategy pool** (32 tickers): above + GOOGL, META, TSLA, V, MA, HD, DIS, NFLX, ADBE, CRM, AMD, INTC, PFE, WMT, KO, PEP, CSCO, QCOM, TXN, AVGO, COST, ABBV

"""
Paper Trading Agent — Azure OpenAI + Alpaca
============================================
Multi-agent system: MacroAgent → StockAnalystAgent → RiskManagerAgent
orchestrated by a TradingAgent that executes Alpaca paper trades.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
import time
import re
from datetime import datetime
from openai import AzureOpenAI

from BacktestEngine import (
    BacktestEngine as BTEngine,
    StrategyConfig,
    FactorLibrary,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ── Azure OpenAI ──────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = "2025-02-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"

# ── Alpaca Paper Trading ──────────────────────────────────────────────
ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# ── Portfolio ──────────────────────────────────────────────────────────
PORTFOLIO_TICKERS = [
    "AAPL", "MSFT", "JPM", "JNJ", "XOM",
    "AMZN", "NVDA", "UNH", "BAC", "PG",
]

# ── Strategy params ───────────────────────────────────────────────────
LOOKBACK_DAYS = 20
BACKTEST_START = "2022-01-01"
INITIAL_CAPITAL = 100_000

# ── Take-Profit / Stop-Loss ────────────────────────────────────────────
TAKE_PROFIT_PCT = 0.15   # 15% profit → trigger take-profit
STOP_LOSS_PCT = 0.08     # 8% loss → trigger stop-loss
TRAILING_STOP_PCT = 0.05  # 5% trailing stop from peak (0 = disabled)

# ── Shared Azure OpenAI client ────────────────────────────────────────
_azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


# ═══════════════════════════════════════════════════════════════════════
# BASE AGENT
# ═══════════════════════════════════════════════════════════════════════

class BaseAgent:
    """Common LLM interaction layer for all agents."""

    def __init__(self, name: str, system_prompt: str, temperature: float = 0.1):
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def call_llm(self, user_prompt: str, retries: int = 5) -> dict | None:
        """Call Azure OpenAI with system + user prompt, return parsed JSON."""
        for attempt in range(retries):
            try:
                response = _azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)

            except json.JSONDecodeError:
                print(f"    [{self.name}] JSON parse error, attempt {attempt+1}/{retries}")
            except Exception as e:
                err_str = str(e)
                match = re.search(r"retry after (\d+)", err_str, re.IGNORECASE)
                wait = 2**attempt * 5
                if match:
                    wait = float(match.group(1)) + 3
                    print(f"    [{self.name}] Rate limited, waiting {wait:.0f}s ...")
                elif "429" in err_str:
                    wait = 60 * (attempt + 1)
                    print(f"    [{self.name}] 429, waiting {wait}s ...")
                else:
                    print(f"    [{self.name}] API error: {e}, retry in {wait}s ...")
                time.sleep(wait)
        return None


# ═══════════════════════════════════════════════════════════════════════
# DATA AGENT  — fetches market data & computes technical indicators
# ═══════════════════════════════════════════════════════════════════════

class DataAgent(BaseAgent):
    """Fetches price data and produces technical features for any ticker."""

    def __init__(self):
        super().__init__(
            name="DataAgent",
            system_prompt="You are a data assistant. Return JSON only.",
        )

    def fetch_market_data(self, tickers: list, start: str, end: str, extra_days: int = 730) -> pd.DataFrame:
        all_symbols = list(set(tickers + ["SPY", "^VIX"]))
        fetch_start = (pd.to_datetime(start) - pd.Timedelta(days=extra_days)).strftime("%Y-%m-%d")
        print(f"  [DataAgent] Downloading {len(all_symbols)} symbols {fetch_start} → {end} ...")
        raw = yf.download(all_symbols, start=fetch_start, end=end, progress=False)["Close"]
        raw = raw.ffill()
        missing = [t for t in all_symbols if t not in raw.columns]
        if missing:
            raise ValueError(f"Missing data for: {missing}")
        print(f"  [DataAgent] {len(raw)} trading days x {len(raw.columns)} symbols.")
        return raw

    def technical_features(self, prices: pd.Series) -> dict:
        """Compute 15 technical indicators (past data only).

        These mirror the factor categories used by the BacktestEngine (24 factors)
        but kept lean (15 indicators) for LLM token efficiency.
        """
        recent = prices.tail(max(LOOKBACK_DAYS, 120))
        current = recent.iloc[-1]
        n = len(recent)

        # ── Moving averages ──────────────────────────────────────────
        sma20 = recent.tail(20).mean()
        sma50 = recent.tail(50).mean() if n >= 50 else np.nan
        ema12 = recent.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = recent.ewm(span=26, adjust=False).mean().iloc[-1]

        # ── RSI ──────────────────────────────────────────────────────
        delta = recent.diff().dropna()
        gains = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        losses = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 100.0

        # ── MACD ─────────────────────────────────────────────────────
        macd_line = ema12 - ema26
        macd_signal = recent.ewm(span=9, adjust=False).mean().iloc[-1]
        # approximate: ema12[-9:] vs ema26[-9:] then ema9
        ema12_series = recent.ewm(span=12, adjust=False).mean()
        ema26_series = recent.ewm(span=26, adjust=False).mean()
        macd_series = ema12_series - ema26_series
        macd_signal_val = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
        macd_hist = macd_series.iloc[-1] - macd_signal_val

        # ── Bollinger Bands ──────────────────────────────────────────
        bb_std = recent.tail(20).std()
        bb_upper = sma20 + 2 * bb_std
        bb_lower = sma20 - 2 * bb_std
        bb_position = (current - sma20) / (2 * bb_std) if bb_std > 0 else 0  # 0=mid, ±1=band

        # ── ATR (14) ─────────────────────────────────────────────────
        tr = pd.DataFrame({
            "hl": recent.diff().abs(),  # simplified
            "hc": (recent - recent.shift(1)).abs(),
            "lc": (recent - recent.shift(1)).abs(),
        }).max(axis=1)
        atr = tr.tail(14).mean()

        # ── Stochastic %K (14,3) ─────────────────────────────────────
        low14 = recent.tail(14).min()
        high14 = recent.tail(14).max()
        stoch_k = 100 * (current - low14) / (high14 - low14) if high14 != low14 else 50

        # ── Returns & momentum ───────────────────────────────────────
        ret_5d = (current / recent.iloc[-5] - 1) * 100 if n >= 5 else np.nan
        ret_20d = (current / recent.iloc[-20] - 1) * 100 if n >= 20 else np.nan
        ret_60d = (current / recent.iloc[-60] - 1) * 100 if n >= 60 else np.nan

        # ── Volatility ───────────────────────────────────────────────
        vol_20d = recent.pct_change().tail(20).std() * np.sqrt(252) * 100
        vol_60d = recent.pct_change().tail(60).std() * np.sqrt(252) * 100 if n >= 60 else np.nan

        # ── 52-week range ────────────────────────────────────────────
        high_52w = recent.tail(252).max() if n >= 252 else recent.max()
        low_52w = recent.tail(252).min() if n >= 252 else recent.min()
        drawdown_60d = (current / recent.tail(60).max() - 1) * 100 if n >= 60 else np.nan

        # ── Volume proxy (price * return trend) ──────────────────────
        vpt = (recent.pct_change().fillna(0) * (1 + recent.pct_change().fillna(0))).tail(20).sum() * 100

        def _r(v):
            if isinstance(v, str):
                return v
            return round(v, 2) if not (isinstance(v, float) and np.isnan(v)) else "N/A"

        return {
            # Price & MAs
            "current_price": round(current, 2),
            "sma20": _r(sma20),
            "sma50": _r(sma50),
            "ema12": _r(ema12),
            "ema26": _r(ema26),
            # Oscillators
            "rsi14": _r(rsi),
            "macd_line": _r(macd_series.iloc[-1]),
            "macd_signal": _r(macd_signal_val),
            "macd_histogram": _r(macd_hist),
            "stoch_k_14": _r(stoch_k),
            # Bands & volatility
            "bb_position": _r(bb_position),  # -1 to +1
            "bb_upper": _r(bb_upper),
            "bb_lower": _r(bb_lower),
            "atr_14": _r(atr),
            "vol_20d_ann_pct": _r(vol_20d),
            "vol_60d_ann_pct": _r(vol_60d),
            # Momentum
            "return_5d_pct": _r(ret_5d),
            "return_20d_pct": _r(ret_20d),
            "return_60d_pct": _r(ret_60d),
            "drawdown_60d_pct": _r(drawdown_60d),
            # Volume-price
            "volume_price_trend": _r(vpt),
            # Range
            "pct_from_52w_high": _r((current / high_52w - 1) * 100),
            "pct_from_52w_low": _r((current / low_52w - 1) * 100),
        }


# ═══════════════════════════════════════════════════════════════════════
# MACRO AGENT  — Layer 1: market phase & risk appetite
# ═══════════════════════════════════════════════════════════════════════

MACRO_SYSTEM_PROMPT = (
    "You are a macro market analyst. Always return valid JSON only."
)

MACRO_USER_TEMPLATE = """
Today is {target_date} before market open.
Based on the following data, judge the current market phase and provide a risk appetite score.

[Macro Data]
- S&P 500 (SPY) last 5 days return : {spy_5d}%
- S&P 500 (SPY) last 20 days return: {spy_20d}%
- Market Volatility (VIX) recent average (5d): {vix_val}
- VIX trend (5d change): {vix_trend}

Classification guide:
- Bull   : sustained uptrend, low VIX (<20), positive momentum
- Bear   : sustained downtrend, elevated VIX (>25), negative momentum
- Ranging: sideways movement, moderate VIX
- Panic  : sharp drawdown, VIX spike (>35), extreme fear

Output strictly as JSON:
{{
    "market_phase": "Bull" | "Bear" | "Ranging" | "Panic",
    "risk_appetite": <integer 0-10, where 10 is extremely greedy>,
    "reasoning": "<one sentence citing specific data points>"
}}
"""


class MacroAgent(BaseAgent):
    """Assess macro environment from SPY and VIX data."""

    def __init__(self):
        super().__init__(name="MacroAgent", system_prompt=MACRO_SYSTEM_PROMPT)

    def assess(self, data: pd.DataFrame, target_date: str) -> dict:
        spy = data["SPY"]
        vix = data["^VIX"]

        spy_5d = (spy.iloc[-1] / spy.iloc[-5] - 1) * 100 if len(spy) >= 5 else 0
        spy_20d = (spy.iloc[-1] / spy.iloc[-20] - 1) * 100 if len(spy) >= 20 else 0
        vix_avg = vix.tail(5).mean()
        vix_trend = vix.iloc[-1] - vix.iloc[-5] if len(vix) >= 5 else 0

        prompt = MACRO_USER_TEMPLATE.format(
            target_date=target_date,
            spy_5d=round(spy_5d, 2),
            spy_20d=round(spy_20d, 2),
            vix_val=round(vix_avg, 2),
            vix_trend=round(vix_trend, 2),
        )
        result = self.call_llm(prompt)
        return result or {
            "market_phase": "Ranging",
            "risk_appetite": 5,
            "reasoning": "LLM unavailable",
        }


# ═══════════════════════════════════════════════════════════════════════
# STOCK ANALYST AGENT  — Layer 2: per-ticker signal
# ═══════════════════════════════════════════════════════════════════════

STOCK_SYSTEM_PROMPT = (
    "You are a quantitative analyst. Always return valid JSON only."
)

STOCK_USER_TEMPLATE = """
Today is {target_date} before market open.
Current macro environment: {market_phase}, risk appetite {risk_appetite}/10.

Evaluate {ticker} based ONLY on the historical data below – no future information.

[Technical & Price/Volume Data for {ticker}]
{tech_data}

[Backtest Context for {ticker}]
{backtest_context}

Indicator interpretation guide:
- ema12 vs ema26       : EMA12 > EMA26 = bullish momentum (like MACD direction)
- rsi14                 : >70 overbought, <30 oversold, 30-70 neutral
- macd_line vs signal   : line > signal = bullish crossover; histogram turning positive = accelerating
- stoch_k_14            : >80 overbought, <20 oversold
- bb_position           : -1 = at lower band (oversold/mean-reversion buy), +1 = at upper band (overbought)
- atr_14                : higher = more volatile, position size accordingly
- drawdown_60d_pct      : large negative = potential bounce candidate
- volume_price_trend    : positive = accumulation, negative = distribution

Our BacktestEngine supports 24 factors across categories:
  Trend: sma_crossover_20_50, sma_crossover_50_200, ema_crossover_12_26, adx_14
  Momentum: rsi_14, macd_signal, macd_histogram, momentum_20d/60d/120d, stochastic_14, cci_20, williams_r_14
  Volatility: bollinger_position, bollinger_squeeze, volatility_20d, volatility_regime, atr_14
  Volume: volume_ratio, volume_price_trend, obv_trend
  Risk/Other: beta_spy, return_5d_reversal, drawdown_60d

Rules:
- In Panic regimes, default to HOLD or SELL unless technical indicators are strongly positive across multiple factor categories.
- In Bull regimes with risk_appetite >= 7, favour BUY when ema12 > ema26, macd_histogram is positive, price > sma20, and stoch_k is not overbought (>80).
- In Bear regimes, only BUY if rsi14 < 30 (deeply oversold) AND drawdown_60d < -15% (capitulation).
- In Ranging, use mean-reversion: BUY at bb_position < -0.8, SELL at bb_position > +0.8.
- Confidence must reflect the consistency of signals across all three categories (trend + momentum + volatility).
- When backtest_context is available, use it to calibrate confidence (e.g. if backtest Sharpe < 0, lower confidence; if profit_factor > 1.5, raise confidence).

Output strictly as JSON:
{{
    "signal"    : "BUY" | "SELL" | "HOLD",
    "confidence": <float 0.0-1.0>,
    "reasoning" : "<one sentence citing specific indicators, macro context, and backtest results if available>"
}}
"""


class StockAnalystAgent(BaseAgent):
    """Generate a trading signal for a single stock."""

    def __init__(self):
        super().__init__(name="StockAnalystAgent", system_prompt=STOCK_SYSTEM_PROMPT)

    def analyze(self, ticker: str, tech_data: dict, market_phase: str,
                risk_appetite: int, target_date: str, backtest_context: str = "No backtest data available.") -> dict:
        tech_str = "\n".join(f"  {k}: {v}" for k, v in tech_data.items())
        prompt = STOCK_USER_TEMPLATE.format(
            target_date=target_date,
            market_phase=market_phase,
            risk_appetite=risk_appetite,
            ticker=ticker,
            tech_data=tech_str,
            backtest_context=backtest_context,
        )
        result = self.call_llm(prompt)
        return result or {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": "LLM unavailable",
        }


# ═══════════════════════════════════════════════════════════════════════
# RISK MANAGER AGENT  — Layer 3: portfolio-level risk
# ═══════════════════════════════════════════════════════════════════════

RISK_SYSTEM_PROMPT = (
    "You are a Chief Risk Officer. Always return valid JSON only."
)

RISK_USER_TEMPLATE = """
Today is {target_date}.
Portfolio holdings: {portfolio_holdings}.

[Quantitative Risk Metrics]
- S&P 500 rolling 252-day Max Drawdown: {max_dd}%
- Estimated Portfolio VaR (95%, 1-day): {var_95}%
- Current VIX: {vix_val}
- Portfolio 20-day rolling volatility (annualised): {port_vol}%

[Backtest Performance Summary]
{backtest_summary}

Decision thresholds:
- VaR < -2% OR Max Drawdown < -15% OR VIX > 35 -> consider Reduce/Liquidate
- VIX > 40 OR Max Drawdown < -25% -> Liquidate All Long Positions
- If backtest shows Sharpe < 0 and MaxDD > 20%, current strategy config is fragile → escalate risk level
- If backtest ProfitFactor < 1.0, the strategy is losing in historical simulation → recommend Reduce
- If no backtest data exists, assume strategies are unproven → be more conservative with exposure

Output strictly as JSON:
{{
    "risk_level": "Low" | "Medium" | "High",
    "action"    : "Hold Normally" | "Reduce Overall Exposure" | "Liquidate All Long Positions",
    "reasoning" : "<explain how VaR, VIX, Drawdown, Volatility and backtest results influenced the decision>"
}}
"""


class RiskManagerAgent(BaseAgent):
    """Assess portfolio-level risk and recommend action."""

    def __init__(self):
        super().__init__(name="RiskManagerAgent", system_prompt=RISK_SYSTEM_PROMPT)

    def assess(self, data: pd.DataFrame, tickers: list, target_date: str,
               backtest_summary: str = "No backtest data available.") -> dict:
        spy_1y = data["SPY"].tail(252)
        max_dd = ((spy_1y / spy_1y.cummax() - 1) * 100).min()

        port_returns = data[tickers].pct_change().dropna()
        eq_weighted = port_returns.mean(axis=1)
        var_95 = np.percentile(eq_weighted, 5) * 100
        port_vol = eq_weighted.tail(20).std() * np.sqrt(252) * 100
        vix_val = data["^VIX"].iloc[-1]

        prompt = RISK_USER_TEMPLATE.format(
            target_date=target_date,
            portfolio_holdings=", ".join(tickers),
            max_dd=round(max_dd, 2),
            var_95=round(var_95, 2),
            vix_val=round(vix_val, 2),
            port_vol=round(port_vol, 2),
            backtest_summary=backtest_summary,
        )
        result = self.call_llm(prompt)
        return result or {
            "risk_level": "Medium",
            "action": "Hold Normally",
            "reasoning": "LLM unavailable",
        }


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST TRIGGER  — agent decides whether to run backtests
# ═══════════════════════════════════════════════════════════════════════

BACKTEST_DECISION_PROMPT = (
    "You are a quantitative strategy auditor. Always return valid JSON only."
)

BACKTEST_DECISION_TEMPLATE = """
Today is {target_date}.
Current market conditions:
- Market phase: {market_phase}
- Risk appetite: {risk_appetite}/10
- Risk level: {risk_level}
- Portfolio risk action: {risk_action}
- VIX current: {vix_val}

Last backtest run: {last_backtest}

Our BacktestEngine evaluates strategies built from 24 technical factors across 5 categories:
  Trend      — sma_crossover_20_50, sma_crossover_50_200, ema_crossover_12_26, adx_14
  Momentum   — rsi_14, macd_signal, macd_histogram, momentum_20d/60d/120d, stochastic_14, cci_20, williams_r_14
  Volatility — bollinger_position, bollinger_squeeze, volatility_20d, volatility_regime, atr_14
  Volume     — volume_ratio, volume_price_trend, obv_trend
  Risk/Other — beta_spy, return_5d_reversal, drawdown_60d

Each strategy selects a subset with custom weights, thresholds, and direction.
Backtest results include: Sharpe, Calmar, Sortino, MaxDD, Volatility, Beta, WinRate, P/L Ratio, ProfitFactor.

Criteria for triggering a backtest:
- Market phase is Panic or Bear (regime stress) → STRONGLY trigger — strategies calibrated in Bull may fail
- Risk level is "High" and last backtest >30 days ago → trigger (stale data in new risk regime)
- VIX > 30 (elevated volatility) → trigger stress-test backtest across all strategies
- VIX > 40 → strongly trigger with urgency=high, recommend all strategies
- Normal conditions (Bull/Ranging, low VIX) but no backtest in >60 days → optionally trigger a refresh
- Bull market, low VIX, recent backtest (<30 days) with good metrics → skip

Output strictly as JSON:
{{
    "should_backtest": true | false,
    "urgency": "high" | "medium" | "low",
    "reasoning": "<one sentence citing the specific market condition + factor categories of concern>",
    "recommended_strategies": "<indices e.g. '0,1,2' or 'all'>"
}}
"""

FREQUENCY_DECISION_TEMPLATE = """
Today is {target_date}.
Current market conditions:
- Market phase: {market_phase}
- Risk appetite: {risk_appetite}/10
- Risk level: {risk_level}
- VIX current: {vix_val}
- SPY 20-day realized volatility (ann.): {spy_vol}%
- SPY 60-day trend strength (|ret|/vol): {trend_strength}

Rebalance frequency options:
- daily   : rebalance every trading day — highest turnover, fastest reaction
- weekly  : rebalance every 5 trading days
- biweekly: rebalance every 10 trading days
- monthly : rebalance every 21 trading days — lowest turnover, slowest reaction

Decision framework:
- VIX > 35 OR market_phase == "Panic" → daily (extreme conditions, need daily adjustments)
- VIX > 25 OR market_phase == "Bear" → weekly or daily (elevated risk, more frequent checks)
- VIX 15-25, market_phase == "Ranging" → weekly or biweekly (mean-reversion opportunities, moderate turnover)
- VIX < 15, market_phase == "Bull", trend_strength > 1.0 → monthly (stable uptrend, let winners run)
- High realized volatility (spy_vol > 30%) → favour higher frequency regardless of other signals
- Risk level "High" → favour higher frequency to respond quickly
- Low volatility + strong trend → favour lower frequency (reduce trading costs)

Output strictly as JSON:
{{
    "recommended_frequency": "daily" | "weekly" | "biweekly" | "monthly",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<one sentence citing VIX, market phase, volatility, and trend strength>"
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# ALPACA TRADER  — paper trade execution
# ═══════════════════════════════════════════════════════════════════════

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
    try:
        from alpaca.trading.enums import OrderClass
        from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
        BRACKET_SUPPORT = True
    except ImportError:
        OrderClass = None  # type: ignore
        TakeProfitRequest = None  # type: ignore
        StopLossRequest = None  # type: ignore
        BRACKET_SUPPORT = False
except ImportError:
    ALPACA_AVAILABLE = False
    BRACKET_SUPPORT = False
    MarketOrderRequest = None  # type: ignore
    LimitOrderRequest = None   # type: ignore
    StopOrderRequest = None    # type: ignore
    print("alpaca-py not installed. Run: pip install alpaca-py")


class AlpacaExecutor:
    """Wrapper around the Alpaca paper-trading API with TP/SL support."""

    def __init__(self):
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py not installed.")
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account = self.client.get_account()
        print(f"  [AlpacaExecutor] Connected — Equity: ${float(account.equity):,.2f}")

    def summary(self) -> dict:
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "positions": {p.symbol: {"qty": float(p.qty), "market_value": float(p.market_value),
                                     "unrealized_pl": float(p.unrealized_pl)} for p in positions},
        }

    def print_summary(self):
        info = self.summary()
        print("\n── Alpaca Account ──────────────────────────────────")
        print(f"  Cash           : ${info['cash']:>12,.2f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:>12,.2f}")
        print(f"  Equity         : ${info['equity']:>12,.2f}")
        print(f"  Buying Power   : ${info['buying_power']:>12,.2f}")
        print(f"  Positions ({len(info['positions'])}):")
        for sym, p in info['positions'].items():
            print(f"    {sym:<6} qty={p['qty']:>6.0f}  value=${p['market_value']:>10,.2f}"
                  f"  P&L=${p['unrealized_pl']:>+8,.2f}")
        return info

    def rebalance(self, target_weights: dict, dry_run: bool = True,
                  use_bracket: bool = False,
                  tp_pct: float = TAKE_PROFIT_PCT,
                  sl_pct: float = STOP_LOSS_PCT) -> list:
        """
        Align the paper portfolio to target_weights.
        Buying power is respected: buy quantities are scaled down if needed.
        If use_bracket=True, BUY orders use bracket orders with TP/SL attached.
        Returns list of tickers submitted.
        """
        account = self.client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        positions = {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

        tickers = list(target_weights.keys())
        prices = yf.download(tickers, period="2d", progress=False)["Close"].iloc[-1].to_dict()

        order_label = "bracket" if use_bracket else "market"
        print(f"\n── Rebalancing (equity=${equity:,.2f}, buying_power=${buying_power:,.2f}, "
              f"dry_run={dry_run}, {order_label}) ──")

        # ── Pass 1: calculate buy/sell amounts ──
        sells = []   # list of (ticker, qty, price, cost)
        buys = []    # list of (ticker, qty, price, cost)
        total_sell_proceeds = 0.0

        for ticker, weight in target_weights.items():
            if ticker not in prices or np.isnan(prices[ticker]):
                print(f"  {ticker}  no price → skip")
                continue
            price = prices[ticker]
            target_qty = int(equity * weight / price)
            current_qty = int(positions.get(ticker, 0))
            delta = target_qty - current_qty
            if delta == 0:
                print(f"  {ticker:<6}  no change  (qty={current_qty})")
                continue
            if delta > 0:
                buys.append((ticker, delta, price, delta * price))
            else:
                qty = abs(delta)
                sells.append((ticker, qty, price, qty * price))
                total_sell_proceeds += qty * price

        submitted = []

        if dry_run:
            for ticker, qty, price, cost in sells:
                print(f"  {ticker:<6}  SELL {qty:>4} @ ~${price:.2f}  ≈${cost:,.2f}")
            for ticker, qty, price, cost in buys:
                tp_str = f" TP={tp_pct*100:.0f}%" if use_bracket else ""
                sl_str = f" SL={sl_pct*100:.0f}%" if use_bracket else ""
                print(f"  {ticker:<6}  BUY {qty:>4} @ ~${price:.2f}  ≈${cost:,.2f}{tp_str}{sl_str}")
            print(f"  → Dry run (no orders).")
            return submitted

        # ── Pass 2: submit SELLS first ─────────────────────────────────
        if sells:
            print(f"  ── Submitting {len(sells)} sell(s) ──")
        for ticker, qty, price, cost in sells:
            print(f"  {ticker:<6}  SELL {qty:>4} @ ~${price:.2f}  ≈${cost:,.2f}")
            try:
                existing = self.client.get_orders(
                    GetOrdersRequest(status="open", symbols=[ticker], limit=50)
                )
                for o in existing:
                    self.client.cancel_order_by_id(str(o.id))
                    print(f"    Cancelled open order {o.id} for {ticker}")
                if existing:
                    time.sleep(0.3)
                order = MarketOrderRequest(
                    symbol=ticker, qty=qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                self.client.submit_order(order)
                submitted.append(ticker)
            except Exception as e:
                print(f"    Sell order failed: {e}")

        # ── Pass 3: wait for sells, then submit BUYS with fresh buying power ──
        if buys:
            if sells:
                print(f"  Waiting 3s for sells to settle ...")
                time.sleep(3)
                account = self.client.get_account()
                buying_power = float(account.buying_power)
                print(f"  ── Submitting {len(buys)} buy(s) (buying_power=${buying_power:,.2f}) ──")
            else:
                print(f"  ── Submitting {len(buys)} buy(s) (buying_power=${buying_power:,.2f}) ──")

            # Sort buys largest cost first so small buys don't drain the budget early
            buys.sort(key=lambda x: -x[3])

            # Track remaining buying power as we submit
            remaining_bp = buying_power

            for ticker, qty, price, cost in buys:
                # Scale this individual buy to fit remaining buying power
                scaled = False
                if cost > remaining_bp:
                    new_qty = int(remaining_bp / price) if price > 0 else 0
                    if new_qty <= 0:
                        print(f"  {ticker:<6}  BUY   skip — ${cost:,.2f} > remaining ${remaining_bp:,.2f}")
                        continue
                    qty = new_qty
                    cost = qty * price
                    scaled = True

                tp_str = f" TP={tp_pct*100:.0f}%" if use_bracket else ""
                sl_str = f" SL={sl_pct*100:.0f}%" if use_bracket else ""
                note = " (scaled to fit BP)" if scaled else ""
                print(f"  {ticker:<6}  BUY {qty:>4} @ ~${price:.2f}  ≈${cost:,.2f}{tp_str}{sl_str}{note}")
                try:
                    if use_bracket and qty > 0:
                        self._submit_bracket_order(ticker, qty, price, tp_pct, sl_pct)
                    else:
                        order = MarketOrderRequest(
                            symbol=ticker, qty=qty, side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                        self.client.submit_order(order)
                    submitted.append(ticker)
                    remaining_bp -= cost
                except Exception as e:
                    print(f"    Buy order failed: {e}")

        print(f"  → {len(submitted)} order(s) submitted.")
        return submitted

    def _submit_bracket_order(self, ticker: str, qty: int, price: float,
                              tp_pct: float, sl_pct: float):
        """Submit a bracket order: market buy + take-profit limit + stop-loss stop.
        Falls back to simple market order + separate GTC TP/SL orders if BRACKET unsupported."""
        tp_price = round(price * (1 + tp_pct), 2)
        sl_price = round(price * (1 - sl_pct), 2)

        if BRACKET_SUPPORT:
            take_profit = TakeProfitRequest(
                limit_price=tp_price,
            )
            stop_loss = StopLossRequest(
                stop_price=sl_price,
            )
            bracket_order = MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            self.client.submit_order(bracket_order)
            print(f"    Bracket: buy {qty} {ticker} | TP @ ${tp_price} | SL @ ${sl_price}")
        else:
            # Fallback: buy first, then submit separate TP/SL orders
            buy_order = MarketOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            self.client.submit_order(buy_order)
            tp_order = LimitOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL,
                limit_price=tp_price, time_in_force=TimeInForce.GTC,
            )
            sl_order = StopOrderRequest(
                symbol=ticker, qty=qty, side=OrderSide.SELL,
                stop_price=sl_price, time_in_force=TimeInForce.GTC,
            )
            self.client.submit_order(tp_order)
            self.client.submit_order(sl_order)
            print(f"    Buy + TP/SL (fallback): {qty} {ticker} | TP @ ${tp_price} | SL @ ${sl_price}")

    def attach_tpsl_all(self, tp_pct: float = TAKE_PROFIT_PCT,
                        sl_pct: float = STOP_LOSS_PCT,
                        dry_run: bool = True) -> list:
        """Attach TP/SL (OCO) orders to all positions that lack them.
        Cancels any existing TP/SL orders for the symbol first."""
        positions = self.client.get_all_positions()
        if not positions:
            print("  No open positions.")
            return []
        print(f"\n── Attaching TP/SL to {len(positions)} position(s) "
              f"(TP={tp_pct*100:.0f}%, SL={sl_pct*100:.0f}%, dry_run={dry_run}) ──")
        tickers = [p.symbol for p in positions]
        prices = yf.download(tickers, period="2d", progress=False)["Close"].iloc[-1].to_dict()
        attached = []
        for pos in positions:
            sym = pos.symbol
            qty = int(float(pos.qty))
            if qty <= 0 or sym not in prices:
                continue
            price = prices[sym]
            cost_basis = float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') and float(pos.avg_entry_price) > 0 else price

            tp_price = round(cost_basis * (1 + tp_pct), 2)
            sl_price = round(cost_basis * (1 - sl_pct), 2)
            print(f"  {sym:<6} qty={qty}  cost=${cost_basis:.2f}  "
                  f"TP=${tp_price:.2f} ({tp_pct*100:.0f}%)  SL=${sl_price:.2f} ({sl_pct*100:.0f}%)")
            if not dry_run:
                try:
                    # Cancel existing open orders for this symbol first
                    existing = self.client.get_orders(
                        GetOrdersRequest(status="open", symbols=[sym], limit=50)
                    )
                    for o in existing:
                        self.client.cancel_order_by_id(str(o.id))
                        print(f"    Cancelled existing order: {o.id}")
                    if existing:
                        time.sleep(0.3)  # let cancellations settle
                    tp_order = LimitOrderRequest(
                        symbol=sym, qty=qty, side=OrderSide.SELL,
                        limit_price=tp_price, time_in_force=TimeInForce.GTC,
                    )
                    sl_order = StopOrderRequest(
                        symbol=sym, qty=qty, side=OrderSide.SELL,
                        stop_price=sl_price, time_in_force=TimeInForce.GTC,
                    )
                    self.client.submit_order(tp_order)
                    self.client.submit_order(sl_order)
                    print(f"    TP+SL submitted")
                    attached.append(sym)
                except Exception as e:
                    print(f"    Failed: {e}")
        print(f"  → {'Dry run.' if dry_run else f'{len(attached)} position(s) protected.'}")
        return attached

    def check_tpsl_status(self):
        """Report P&L vs TP/SL thresholds for current positions."""
        info = self.summary()
        positions = info["positions"]
        if not positions:
            print("  No open positions.")
            return
        print(f"\n── TP/SL Status Check (TP={TAKE_PROFIT_PCT*100:.0f}%, SL={STOP_LOSS_PCT*100:.0f}%) ──")
        for sym, p in positions.items():
            pl_pct = (p["unrealized_pl"] / (p["market_value"] - p["unrealized_pl"])) * 100 if p["market_value"] != p["unrealized_pl"] else 0
            status = "● OK"
            if pl_pct >= TAKE_PROFIT_PCT * 100:
                status = "★ TP HIT"
            elif pl_pct <= -STOP_LOSS_PCT * 100:
                status = "▼ SL HIT"
            elif pl_pct >= TAKE_PROFIT_PCT * 80:
                status = "↑ Near TP"
            elif pl_pct <= -STOP_LOSS_PCT * 80:
                status = "↓ Near SL"
            print(f"  {sym:<6} P&L={pl_pct:+.2f}%  {status}  (value=${p['market_value']:,.2f})")


# ═══════════════════════════════════════════════════════════════════════
# TRADING AGENT  — orchestrator
# ═══════════════════════════════════════════════════════════════════════

class TradingAgent:
    """
    Orchestrator agent that runs the full paper-trading pipeline:
      1. DataAgent  ── fetch prices & compute technicals
      2. MacroAgent ── market phase + risk appetite
      3. StockAnalystAgent ── per-ticker signal
      4. RiskManagerAgent ── portfolio risk check
      5. Weight computation
      6. AlpacaExecutor ── submit orders
    """

    def __init__(self, tickers: list = None):
        self.tickers = tickers or PORTFOLIO_TICKERS
        self.data_agent = DataAgent()
        self.macro_agent = MacroAgent()
        self.stock_agent = StockAnalystAgent()
        self.risk_agent = RiskManagerAgent()
        self.executor: AlpacaExecutor | None = None

        # Pipeline state (populated by run())
        self.raw_data: pd.DataFrame | None = None
        self.market_state: dict = {}
        self.stock_signals: dict = {}
        self.risk_assessment: dict = {}
        self.target_weights: dict = {}

        # Backtest state
        self.backtest_results: list = []
        self.last_backtest_date: str = ""

        # Frequency decision state
        self.frequency_decision: dict = {}

        # Strategy mode state
        self.strategy_mode: bool = False
        self.active_strategy: dict | None = None

    # ── public API ─────────────────────────────────────────────────────

    def run(self, target_date: str = "", dry_run: bool = True):
        """Run the full pipeline and optionally execute trades."""
        target_date = target_date or datetime.today().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"  TradingAgent | {target_date}")
        print(f"{'='*60}")

        # Step 1 — Data
        self.raw_data = self.data_agent.fetch_market_data(
            self.tickers, BACKTEST_START, target_date
        )
        end_dt = pd.to_datetime(target_date)
        hist = self.raw_data[self.raw_data.index < end_dt].copy()

        # Step 2 — Macro
        print(f"  ── Macro ──")
        self.market_state = self.macro_agent.assess(hist, target_date)
        print(f"  Phase: {self.market_state['market_phase']}  "
              f"Appetite: {self.market_state['risk_appetite']}/10")

        # Step 3 — Stock signals
        print(f"  ── Signals ──")
        for ticker in self.tickers:
            if ticker not in hist.columns:
                self.stock_signals[ticker] = {"signal": "HOLD", "confidence": 0.0,
                                              "reasoning": "No data"}
                continue
            tech = self.data_agent.technical_features(hist[ticker])
            # Build backtest context for this ticker if results exist
            bt_ctx = self._build_backtest_context(ticker)
            sig = self.stock_agent.analyze(
                ticker, tech,
                self.market_state["market_phase"],
                self.market_state["risk_appetite"],
                target_date,
                backtest_context=bt_ctx,
            )
            self.stock_signals[ticker] = sig
            print(f"    {ticker:<6} {sig.get('signal','?'):<5}  "
                  f"confidence={sig.get('confidence',0):.2f}")
            time.sleep(1.5)

        # Step 4 — Risk
        print(f"  ── Risk ──")
        bt_summary = self._build_backtest_summary()
        self.risk_assessment = self.risk_agent.assess(hist, self.tickers, target_date, bt_summary)
        print(f"  Level: {self.risk_assessment['risk_level']}  "
              f"Action: {self.risk_assessment['action']}")

        # Step 4.5 — Frequency (LLM decides optimal rebalance frequency)
        print(f"  ── Rebalance Frequency ──")
        self.frequency_decision = self.decide_rebalance_frequency(target_date)

        # Step 5 — Weights
        self.target_weights = self._compute_weights()

        # Step 6 — Execute
        if ALPACA_AVAILABLE:
            self.executor = AlpacaExecutor()
            self.executor.print_summary()
            self.executor.rebalance(self.target_weights, dry_run=dry_run)
        else:
            print("  [TradingAgent] Alpaca unavailable — skipping execution.")

        return self.target_weights

    def _compute_weights(self) -> dict:
        action = self.risk_assessment.get("action", "Hold Normally")
        if action == "Liquidate All Long Positions":
            print(f"    Risk override: LIQUIDATE ALL")
            return {t: 0.0 for t in self.tickers}

        raw = {}
        for ticker, sig in self.stock_signals.items():
            if sig.get("signal") == "BUY":
                raw[ticker] = float(sig.get("confidence", 0.5))
            else:
                raw[ticker] = 0.0

        if action == "Reduce Overall Exposure":
            raw = {t: w * 0.5 for t, w in raw.items()}

        total = sum(raw.values())
        if total == 0:
            return {t: 1.0 / len(self.tickers) for t in self.tickers}
        return {t: w / total for t, w in raw.items()}

    # ── strategy-based trading (rule-based, no LLM) ──────────────────────

    def run_with_strategy(self, strategy_index: int, target_date: str = "",
                          dry_run: bool = True) -> dict:
        """Run paper trading using a strategy config from strategy_config.json.

        This bypasses LLM agents entirely — signals are generated by the same
        24-factor rule engine used by BacktestEngine (deterministic, fast, free).

        Args:
            strategy_index: 0-based index into strategy_config.json
            target_date: trading date YYYY-MM-DD (default: today)
            dry_run: if True, skip order submission

        Returns:
            target_weights dict
        """
        strategies = StrategyConfig.load()
        if strategy_index < 0 or strategy_index >= len(strategies):
            raise IndexError(f"Strategy index {strategy_index} out of range (0-{len(strategies)-1})")

        strategy = strategies[strategy_index]
        StrategyConfig.validate(strategy)  # raises if invalid

        self.strategy_mode = True
        self.active_strategy = strategy
        self.tickers = strategy["tickers"]

        target_date = target_date or datetime.today().strftime("%Y-%m-%d")
        strategy_name = strategy["name"]

        print(f"\n{'='*60}")
        print(f"  STRATEGY TRADING MODE — {strategy_name}")
        print(f"  Factors: {len(strategy['factors'])} | Logic: {strategy.get('signal_logic','weighted_sum')}")
        print(f"  Rebalance: {strategy.get('rebalance_frequency','monthly')} | "
              f"Sizing: {strategy.get('position_sizing','equal_weight')}")
        print(f"  Max positions: {strategy.get('max_positions',10)}")
        print(f"  '{strategy_name}' | {target_date}")
        print(f"{'='*60}")

        # Step 1 — Data (same as LLM path)
        self.raw_data = self.data_agent.fetch_market_data(
            self.tickers, BACKTEST_START, target_date
        )
        end_dt = pd.to_datetime(target_date)
        hist = self.raw_data[self.raw_data.index < end_dt].copy()

        # Step 2 — Factor-based signals (replaces Macro + StockAnalyst)
        print(f"  ── Factor Signals (rule-based) ──")
        benchmark = hist["SPY"] if "SPY" in hist.columns else hist.iloc[:, 0]
        ticker_prices = hist[[t for t in self.tickers if t in hist.columns]]
        self.tickers = [t for t in self.tickers if t in ticker_prices.columns]

        signals = self._compute_strategy_signals(strategy, ticker_prices, benchmark)

        # Populate stock_signals for report compatibility + store raw scores for weighting
        self.stock_signals = {}
        self._raw_strategy_scores = signals  # used by _strategy_weights
        for t in self.tickers:
            score = signals.get(t, 0.0)
            if score > 0.2:
                s, c = "BUY", min(score * 2, 1.0)
            elif score < -0.2:
                s, c = "SELL", min(abs(score) * 2, 1.0)
            else:
                s, c = "HOLD", 0.3
            self.stock_signals[t] = {
                "signal": s,
                "confidence": round(c, 2),
                "reasoning": f"Composite factor score: {score:.3f}",
            }
            print(f"    {t:<6} {s:<5}  score={score:+.3f}  confidence={c:.2f}")

        # Step 3 — Risk (still use LLM RiskManager for portfolio-level check)
        print(f"  ── Risk (LLM) ──")
        bt_summary = self._build_backtest_summary()
        self.risk_assessment = self.risk_agent.assess(hist, self.tickers, target_date, bt_summary)
        print(f"  Level: {self.risk_assessment['risk_level']}  "
              f"Action: {self.risk_assessment['action']}")

        # Step 3.5 — Frequency (LLM recommends; compare with strategy's fixed frequency)
        print(f"  ── Rebalance Frequency ──")
        self.frequency_decision = self.decide_rebalance_frequency(target_date)
        strategy_freq = strategy.get("rebalance_frequency", "monthly")
        llm_freq = self.frequency_decision.get("recommended_frequency", "monthly")
        if strategy_freq != llm_freq:
            print(f"    Note: strategy uses '{strategy_freq}', LLM recommends '{llm_freq}'")

        # Step 4 — Weights from strategy (uses raw composite scores, not mapped signals)
        self.target_weights = self._strategy_weights(strategy)

        # Step 5 — Execute
        if ALPACA_AVAILABLE:
            self.executor = AlpacaExecutor()
            self.executor.print_summary()
            self.executor.rebalance(self.target_weights, dry_run=dry_run)
        else:
            print("  [TradingAgent] Alpaca unavailable — skipping execution.")

        return self.target_weights

    def _compute_strategy_signals(self, strategy: dict, prices: pd.DataFrame,
                                   benchmark: pd.Series) -> dict[str, float]:
        """Compute today's composite signal for each ticker using the strategy's factors.

        Mirrors BacktestRunner._compute_signals but for live (single-row) use.
        """
        factor_names = list(strategy["factors"].keys())
        factors_config = strategy["factors"]

        lib = FactorLibrary(prices, benchmark)
        all_factors = lib.compute_all_factors(factor_names)

        # Get the last row (today) for each factor
        last_idx = prices.index[-1]
        composite = {t: 0.0 for t in prices.columns}
        total_weight = 0.0

        for fname, factor_df in all_factors.items():
            cfg = factors_config[fname]
            w = cfg.get("weight", 1.0)
            direction = cfg.get("direction", 1)
            thresh_long = cfg.get("threshold_long", 0.2)
            thresh_short = cfg.get("threshold_short", -0.2)

            if last_idx not in factor_df.index:
                continue
            row = factor_df.loc[last_idx]

            for t in prices.columns:
                if t not in row.index:
                    continue
                raw = row[t]
                # Apply thresholds: only strong signals pass
                if thresh_short < raw < thresh_long:
                    raw = 0.0
                composite[t] += raw * direction * w

            total_weight += abs(w)

        # Normalize
        if total_weight > 0:
            composite = {t: v / total_weight for t, v in composite.items()}

        return composite

    def _strategy_weights(self, strategy: dict) -> dict[str, float]:
        """Convert raw composite factor scores to target weights using the strategy's position sizing.

        Uses self._raw_strategy_scores (set by run_with_strategy) for ranking.
        Positive scores → long candidates; negative/zero scores → excluded.
        """
        position_sizing = strategy.get("position_sizing", "equal_weight")
        max_positions = strategy.get("max_positions", 10)

        action = self.risk_assessment.get("action", "Hold Normally")
        if action == "Liquidate All Long Positions":
            print(f"    Risk override: LIQUIDATE ALL")
            return {t: 0.0 for t in self.tickers}

        # Use raw composite scores for ranking (positive = bullish)
        scores = getattr(self, "_raw_strategy_scores", {})
        if not scores:
            return {t: 1.0 / len(self.tickers) for t in self.tickers}

        # Select top N by composite score (positive only)
        candidates = [(t, s) for t, s in scores.items() if s > 0]
        candidates.sort(key=lambda x: -x[1])

        if not candidates:
            # No positive signals — equal-weight defensive allocation
            print(f"    No positive signals — equal-weight all tickers")
            return {t: 1.0 / len(self.tickers) for t in self.tickers}

        top_n = min(max_positions, len(candidates))
        selected = {t: s for t, s in candidates[:top_n]}

        print(f"    Selected {len(selected)}/{len(self.tickers)} tickers "
              f"(max={max_positions}, sizing={position_sizing})")

        if position_sizing == "equal_weight":
            weights = {t: 1.0 / top_n for t in selected}
        elif position_sizing == "factor_score":
            total = sum(selected.values())
            weights = {t: s / total for t, s in selected.items()} if total > 0 else {}
        else:  # risk_parity (simplified)
            vols = {}
            if self.raw_data is not None:
                for t in selected:
                    if t in self.raw_data.columns:
                        rets = self.raw_data[t].pct_change().dropna()
                        vols[t] = rets.tail(60).std() if len(rets) > 20 else 0.02
                    else:
                        vols[t] = 0.02
            else:
                vols = {t: 0.02 for t in selected}
            inv_vol = {t: 1.0 / v for t, v in vols.items()}
            total_inv = sum(inv_vol.values())
            weights = {t: iv / total_inv for t, iv in inv_vol.items()} if total_inv > 0 else {}

        if action == "Reduce Overall Exposure":
            weights = {t: w * 0.5 for t, w in weights.items()}

        # Fill zeros for non-selected
        full_weights = {t: weights.get(t, 0.0) for t in self.tickers}
        return full_weights

    # ── backtest trigger ────────────────────────────────────────────────

    def decide_backtest(self, target_date: str = "") -> dict:
        """Use LLM + current market state to decide whether to trigger backtesting.

        Returns a decision dict:
            {should_backtest, urgency, reasoning, recommended_strategies}
        """
        target_date = target_date or datetime.today().strftime("%Y-%m-%d")

        # Gather current market state
        market_phase = self.market_state.get("market_phase", "Unknown")
        risk_appetite = self.market_state.get("risk_appetite", 5)
        risk_level = self.risk_assessment.get("risk_level", "Unknown")
        risk_action = self.risk_assessment.get("action", "Unknown")

        # Get current VIX
        vix_val = "N/A"
        if self.raw_data is not None and "^VIX" in self.raw_data.columns:
            vix_val = str(round(self.raw_data["^VIX"].iloc[-1], 2))

        last_bt = self.last_backtest_date if self.last_backtest_date else "Never"
        prompt = BACKTEST_DECISION_TEMPLATE.format(
            target_date=target_date,
            market_phase=market_phase,
            risk_appetite=risk_appetite,
            risk_level=risk_level,
            risk_action=risk_action,
            vix_val=vix_val,
            last_backtest=last_bt,
        )

        # Use a lightweight BaseAgent call (no subclass needed)
        decider = BaseAgent(name="BacktestDecider", system_prompt=BACKTEST_DECISION_PROMPT)
        result = decider.call_llm(prompt)

        if result is None:
            # Fallback: heuristic decision
            vix_num = float(vix_val) if vix_val != "N/A" else 20
            should = (market_phase in ("Panic", "Bear")) or (vix_num > 30)
            result = {
                "should_backtest": should,
                "urgency": "high" if vix_num > 40 else "medium",
                "reasoning": "Heuristic fallback — LLM unavailable",
                "recommended_strategies": "all",
            }

        print(f"\n  [BacktestDecider] should_backtest={result.get('should_backtest')}  "
              f"urgency={result.get('urgency')}  reasoning={result.get('reasoning')}")
        return result

    def decide_rebalance_frequency(self, target_date: str = "") -> dict:
        """Use LLM + current market state to recommend a rebalance frequency.

        Returns a decision dict:
            {recommended_frequency, confidence, reasoning}
        """
        target_date = target_date or datetime.today().strftime("%Y-%m-%d")

        market_phase = self.market_state.get("market_phase", "Unknown")
        risk_appetite = self.market_state.get("risk_appetite", 5)
        risk_level = self.risk_assessment.get("risk_level", "Unknown")

        vix_val = "N/A"
        spy_vol = "N/A"
        trend_strength = "N/A"
        if self.raw_data is not None:
            if "^VIX" in self.raw_data.columns:
                vix_val = str(round(self.raw_data["^VIX"].iloc[-1], 2))
            if "SPY" in self.raw_data.columns:
                spy_ret = self.raw_data["SPY"].pct_change().dropna()
                spy_vol = str(round(spy_ret.tail(20).std() * np.sqrt(252) * 100, 2))
                ret_60d = self.raw_data["SPY"].iloc[-1] / self.raw_data["SPY"].iloc[-60] - 1
                vol_60d = spy_ret.tail(60).std() * np.sqrt(252)
                trend_strength = str(round(abs(ret_60d) / vol_60d, 2)) if vol_60d > 0 else "N/A"

        prompt = FREQUENCY_DECISION_TEMPLATE.format(
            target_date=target_date,
            market_phase=market_phase,
            risk_appetite=risk_appetite,
            risk_level=risk_level,
            vix_val=vix_val,
            spy_vol=spy_vol,
            trend_strength=trend_strength,
        )

        decider = BaseAgent(name="FrequencyDecider", system_prompt=(
            "You are a portfolio rebalancing strategist. Always return valid JSON only."
        ))
        result = decider.call_llm(prompt)

        if result is None:
            # Fallback: simple heuristic
            vix_num = float(vix_val) if vix_val != "N/A" else 20
            if vix_num > 35 or market_phase == "Panic":
                freq = "daily"
            elif vix_num > 25 or market_phase == "Bear":
                freq = "weekly"
            elif vix_num < 15 and market_phase == "Bull":
                freq = "monthly"
            else:
                freq = "biweekly"
            result = {
                "recommended_frequency": freq,
                "confidence": 0.5,
                "reasoning": f"Heuristic fallback — VIX={vix_val}, phase={market_phase}",
            }

        print(f"\n  [FrequencyDecider] recommended={result.get('recommended_frequency')}  "
              f"confidence={result.get('confidence')}  reasoning={result.get('reasoning')}")
        return result

    def _build_backtest_summary(self) -> str:
        """Summarize all backtest results for the risk manager."""
        if not self.backtest_results:
            return "No backtest data available."
        lines = [f"Last backtest: {self.last_backtest_date}", f"Strategies evaluated: {len(self.backtest_results)}"]
        for r in self.backtest_results:
            m = r.get("metrics", {})
            lines.append(
                f"  {r['name']}: Return={m.get('annualized_return_pct','N/A')}%, "
                f"Sharpe={m.get('sharpe_ratio','N/A')}, Calmar={m.get('calmar_ratio','N/A')}, "
                f"MaxDD={m.get('max_drawdown_pct','N/A')}%, Vol={m.get('volatility_ann_pct','N/A')}%, "
                f"WinRate={m.get('win_rate_pct','N/A')}%, PF={m.get('profit_factor','N/A')}"
            )
        return "\n".join(lines)

    def _build_backtest_context(self, ticker: str) -> str:
        """Summarize backtest results for a given ticker, if available."""
        if not self.backtest_results:
            return "No backtest data available."
        lines = []
        for r in self.backtest_results:
            m = r.get("metrics", {})
            bt_tickers = r.get("tickers", [])
            if ticker not in bt_tickers:
                continue
            lines.append(
                f"Strategy '{r['name']}': Sharpe={m.get('sharpe_ratio','N/A')}, "
                f"MaxDD={m.get('max_drawdown_pct','N/A')}%, "
                f"WinRate={m.get('win_rate_pct','N/A')}%, "
                f"ProfitFactor={m.get('profit_factor','N/A')}"
            )
        return "\n".join(lines) if lines else "No backtest data for this ticker."

    def trigger_backtest(self, strategy_indices: list | None = None,
                         config_path: str = "", verbose: bool = True) -> list:
        """Execute backtesting on strategies from strategy_config.json.

        Results are stored in self.backtest_results for use by subsequent run() calls.

        Args:
            strategy_indices: list of 0-based indices, or None for all strategies.
            config_path: path to strategy_config.json (default: auto-detect).
            verbose: print detailed output.

        Returns:
            List of result dicts [{name, metrics, trade_log, tickers}, ...].
        """
        engine = BTEngine(config_path=config_path, verbose=verbose) if config_path else BTEngine(verbose=verbose)
        if strategy_indices is not None:
            results = engine.run_by_indices(strategy_indices)
        else:
            results = engine.run_all()
        # Enrich each result with tickers for per-ticker context lookups
        strategies = StrategyConfig.load(config_path) if config_path else StrategyConfig.load()
        for r in results:
            for s in strategies:
                if s["name"] == r["name"]:
                    r["tickers"] = s.get("tickers", [])
                    break
        self.backtest_results = results
        self.last_backtest_date = datetime.today().strftime("%Y-%m-%d")
        return results

    def auto_backtest(self, target_date: str = "") -> list:
        """Full auto cycle: decide → (if yes) trigger → return results.

        Returns empty list if backtest was deemed unnecessary.
        """
        decision = self.decide_backtest(target_date)
        if not decision.get("should_backtest", False):
            print("  [TradingAgent] Backtest skipped — conditions normal.")
            return []

        rec = decision.get("recommended_strategies", "all")
        if rec == "all" or not rec.strip():
            indices = None
        else:
            try:
                indices = [int(x.strip()) for x in rec.split(",") if x.strip().isdigit()]
            except (ValueError, AttributeError):
                indices = None

        print(f"  [TradingAgent] Triggering backtest (urgency={decision.get('urgency')})"
              f" — strategies: {rec}")
        return self.trigger_backtest(indices)

    # ── best-strategy selection ───────────────────────────────────────

    def select_best_strategy(self, config_path: str = "",
                             verbose: bool = True) -> dict:
        """Backtest all strategies and select the best by composite metric ranking.

        Ranks strategies on 6 metrics (Sharpe, Calmar, Sortino, Profit Factor,
        Max Drawdown, Win Rate), then picks the one with the best average rank.

        Args:
            config_path: path to strategy_config.json (default: auto-detect).
            verbose: print detailed comparison and ranking.

        Returns:
            {index, name, metrics, score, strategy_config, ...}
        """
        results = self.trigger_backtest(config_path=config_path, verbose=verbose)
        if not results:
            raise ValueError("No valid backtest results. Check strategy config.")

        n = len(results)

        # ── single strategy → no ranking needed ──
        if n == 1:
            strategies = StrategyConfig.load(config_path) if config_path else StrategyConfig.load()
            idx = next((i for i, s in enumerate(strategies) if s["name"] == results[0]["name"]), 0)
            print(f"\n  Only 1 strategy available → auto-selected: {results[0]['name']}")
            return {
                "index": idx,
                "name": results[0]["name"],
                "metrics": results[0]["metrics"],
                "score": 1.0,
                "strategy_config": strategies[idx],
            }

        # ── multi-metric ranking ──
        ranking_metrics = [
            ("sharpe_ratio", True),       # higher is better
            ("calmar_ratio", True),
            ("sortino_ratio", True),
            ("profit_factor", True),
            ("max_drawdown_pct", False),  # lower is better
            ("win_rate_pct", True),
        ]

        ranks = {i: [] for i in range(n)}

        for metric, higher_better in ranking_metrics:
            values = []
            for r in results:
                v = r["metrics"].get(metric)
                if not isinstance(v, (int, float)) or v == float("inf") or v == float("-inf"):
                    v = float("-inf") if higher_better else float("inf")
                values.append(v)
            sorted_idx = sorted(range(n), key=lambda i: values[i], reverse=higher_better)
            for rank, idx in enumerate(sorted_idx):
                ranks[idx].append(rank + 1)

        avg_ranks = {i: np.mean(r) for i, r in ranks.items()}
        best_idx = int(min(avg_ranks, key=avg_ranks.get))

        # Composite score [0, 1]: invert the average rank
        max_r = max(avg_ranks.values())
        min_r = min(avg_ranks.values())
        score = 1.0 if max_r == min_r else round(1.0 - (avg_ranks[best_idx] - min_r) / (max_r - min_r), 3)

        strategies = StrategyConfig.load(config_path) if config_path else StrategyConfig.load()
        best_name = results[best_idx]["name"]
        config_idx = next((i for i, s in enumerate(strategies) if s["name"] == best_name), best_idx)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  BEST STRATEGY RANKING (lower avg rank = better)")
            print(f"{'='*70}")
            print(f"  {'Rank':<6} {'Strategy':<22} {'AvgRank':>9} {'Score':>7} "
                  f"{'Sharpe':>8} {'Calmar':>8} {'MaxDD%':>8} {'PF':>7} {'Win%':>7}")
            print(f"  {'─'*6} {'─'*22} {'─'*9} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
            sorted_by_rank = sorted(avg_ranks.items(), key=lambda x: x[1])
            for rank_pos, (idx, avg_r) in enumerate(sorted_by_rank):
                r = results[idx]
                m = r["metrics"]
                marker = " ★" if idx == best_idx else "  "
                pf = m["profit_factor"]
                pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
                s = 1.0 if max_r == min_r else round(1.0 - (avg_r - min_r) / (max_r - min_r), 3)
                print(f"  {rank_pos+1:<6} {r['name']:<22} {avg_r:>8.3f} {s:>6.3f}{marker} "
                      f"{m['sharpe_ratio']:>8.3f} {m['calmar_ratio']:>8.3f} "
                      f"{m['max_drawdown_pct']:>7.2f}% {pf_str:>7} "
                      f"{m['win_rate_pct']:>6.2f}%")
            print(f"\n  ★ Selected: {results[best_idx]['name']} (score={score}, "
                  f"Sharpe={results[best_idx]['metrics']['sharpe_ratio']})")

        return {
            "index": config_idx,
            "name": best_name,
            "metrics": results[best_idx]["metrics"],
            "score": score,
            "strategy_config": strategies[config_idx],
        }

    def auto_select_and_trade(self, target_date: str = "",
                              dry_run: bool = True,
                              use_bracket: bool = False) -> dict:
        """Full auto pipeline: backtest all → select best → paper trade.

        Args:
            target_date: trading date YYYY-MM-DD (default: today).
            dry_run: if True, skip order submission.
            use_bracket: if True, use bracket orders with TP/SL for buys.

        Returns:
            target_weights dict from the winning strategy.
        """
        target_date = target_date or datetime.today().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"  AUTO-SELECT BEST STRATEGY & TRADE")
        print(f"  {target_date}")
        print(f"{'='*60}")

        # Phase 1: Backtest all & select best
        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  PHASE 1: BACKTEST ALL STRATEGIES".ljust(61) + "║")
        print(f"  ╚{'═'*58}╝")
        best = self.select_best_strategy(verbose=True)

        # Phase 2: Trade with best strategy
        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  PHASE 2: TRADE WITH BEST STRATEGY".ljust(61) + "║")
        print(f"  ╚{'═'*58}╝")
        print(f"  Best strategy : {best['name']} (score={best['score']})")
        print(f"  Factors       : {len(best['strategy_config'].get('factors', {}))}")
        print(f"  Tickers       : {', '.join(best['strategy_config']['tickers'][:8])}"
              f"{'...' if len(best['strategy_config'].get('tickers', [])) > 8 else ''}")
        print(f"  Order mode    : {'dry run' if dry_run else 'REAL ORDERS'}")
        if use_bracket:
            print(f"  TP/SL         : enabled (TP={TAKE_PROFIT_PCT*100:.0f}%, SL={STOP_LOSS_PCT*100:.0f}%)")

        weights = self.run_with_strategy(
            best["index"], target_date=target_date, dry_run=dry_run
        )
        self.report()

        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  AUTO-SELECT COMPLETE".ljust(61) + "║")
        print(f"  ╚{'═'*58}╝")
        print(f"  Traded with : {best['name']}")
        print(f"  Score       : {best['score']}")
        print(f"  Sharpe      : {best['metrics']['sharpe_ratio']}")
        print(f"  MaxDD       : {best['metrics']['max_drawdown_pct']}%")

        return weights

    # ── multi-strategy parallel trading ──────────────────────────────

    def run_multi_strategies(self, strategy_indices: list[int],
                             capital_weights: list[float] | None = None,
                             target_date: str = "",
                             dry_run: bool = True,
                             use_bracket: bool = False) -> dict:
        """Run multiple strategies in parallel and combine their target weights.

        Each strategy runs its own factor computation on its own ticker set.
        Capital is split equally (or by capital_weights) across strategies.
        Final target weights are the capital-weighted average of each strategy's weights.

        Args:
            strategy_indices: list of 0-based strategy indices to run.
            capital_weights: capital allocation per strategy (default: equal split).
            target_date: trading date YYYY-MM-DD (default: today).
            dry_run: if True, skip order submission.
            use_bracket: if True, submit buy orders as bracket (TP/SL) orders.

        Returns:
            Combined target_weights dict.
        """
        strategies = StrategyConfig.load()
        n = len(strategy_indices)
        if n == 0:
            raise ValueError("Need at least 1 strategy index.")
        for idx in strategy_indices:
            if idx < 0 or idx >= len(strategies):
                raise IndexError(f"Strategy index {idx} out of range (0-{len(strategies)-1})")

        if capital_weights is None:
            capital_weights = [1.0 / n] * n
        else:
            total = sum(capital_weights)
            capital_weights = [w / total for w in capital_weights]

        target_date = target_date or datetime.today().strftime("%Y-%m-%d")

        print(f"\n╔{'═'*62}╗")
        print(f"║  MULTI-STRATEGY PARALLEL TRADING".ljust(64) + "║")
        print(f"║  {n} strategies | {target_date} | "
              f"Capital: ${INITIAL_CAPITAL:,}".ljust(64) + "║")
        print(f"╠{'═'*62}╣")
        for i, idx in enumerate(strategy_indices):
            s = strategies[idx]
            name = s['name']
            n_factors = len(s.get('factors', {}))
            n_tickers = len(s.get('tickers', []))
            cap = capital_weights[i] * 100
            cap_bar = "█" * int(cap / 5)
            print(f"║  [{i}] #{idx} {name:<22}  {n_factors:>2}f {n_tickers:>2}t  "
                  f"│ {cap:>5.1f}% {cap_bar}".ljust(64) + "║")
        print(f"╚{'═'*62}╝")

        # Step 1 — Fetch all data (union of all tickers across strategies)
        all_tickers = set()
        for idx in strategy_indices:
            s = strategies[idx]
            StrategyConfig.validate(s)
            all_tickers.update(s.get("tickers", []))
        all_tickers = sorted(all_tickers)

        self.raw_data = self.data_agent.fetch_market_data(
            all_tickers, BACKTEST_START, target_date
        )
        end_dt = pd.to_datetime(target_date)
        hist = self.raw_data[self.raw_data.index < end_dt].copy()

        # Step 2 — Run each strategy independently to get target weights
        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  RUNNING {n} STRATEGIES IN PARALLEL".ljust(61) + "║")
        print(f"  ╚{'═'*58}╝")
        all_strategy_weights: list[dict[str, float]] = []
        all_strategy_names: list[str] = []
        for i, idx in enumerate(strategy_indices):
            s = strategies[idx]
            strategy_tickers = [t for t in s["tickers"] if t in hist.columns]
            strategy_name = s['name']
            all_strategy_names.append(strategy_name)
            ticker_prices = hist[strategy_tickers]
            benchmark = hist["SPY"] if "SPY" in hist.columns else hist.iloc[:, 0]

            print(f"\n  ┌── STRATEGY [{i}] : {strategy_name} ".ljust(61) + "┐")
            print(f"  │  Capital: {capital_weights[i]*100:.0f}% | "
                  f"Factors: {', '.join(list(s.get('factors',{}).keys())[:5])}"
                  f"{'...' if len(s.get('factors',{})) > 5 else ''}".ljust(43)[:43] + " │")

            signals = self._compute_strategy_signals(s, ticker_prices, benchmark)

            # Show per-ticker signal scores for this strategy
            if signals:
                sorted_sigs = sorted(signals.items(), key=lambda x: -x[1])
                print(f"  │  Ticker signals:                                    │")
                for t, score in sorted_sigs[:6]:
                    bar = "█" * max(1, int(abs(score) * 10))
                    direction = "+" if score > 0 else " "
                    print(f"  │    {t:<6} {direction}{score:+.3f} {bar} │")

            # Temporarily set self.tickers for _strategy_weights to use
            saved_tickers = self.tickers
            self.tickers = strategy_tickers
            self._raw_strategy_scores = signals
            w = self._strategy_weights(s)
            self.tickers = saved_tickers
            all_strategy_weights.append(w)

            # Show individual strategy weights
            active_positions = {t: wt for t, wt in w.items() if wt > 0.01}
            if active_positions:
                print(f"  │  Target weights ({len(active_positions)} positions):                 │")
                for t, wt in sorted(active_positions.items(), key=lambda x: -x[1]):
                    wbar = "█" * int(wt * 20)
                    print(f"  │    {t:<6} {wt*100:5.1f}% {wbar} │")
            else:
                print(f"  │  (defensive: equal-weight all tickers)             │")
            print(f"  └{'─'*58}┘")

        # Step 3 — Merge weights: capital-weighted average across strategies
        combined_weights: dict[str, float] = {}
        all_tickers_in_use = set()
        for w in all_strategy_weights:
            all_tickers_in_use.update(w.keys())
        all_tickers_in_use = sorted(all_tickers_in_use)

        # Build contribution matrix: contribution[i][ticker] = weight * capital_weight
        contribution: dict[str, dict[str, float]] = {}  # ticker → {strategy_name: contribution}
        for ticker in all_tickers_in_use:
            contribution[ticker] = {}
            combined = 0.0
            for i, w in enumerate(all_strategy_weights):
                contrib = w.get(ticker, 0.0) * capital_weights[i]
                contribution[ticker][all_strategy_names[i]] = contrib
                combined += contrib
            combined_weights[ticker] = combined

        # Renormalize
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {t: w / total for t, w in combined_weights.items()}

        self.target_weights = combined_weights

        # ── Visual contribution breakdown ──
        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  WEIGHT CONTRIBUTION BREAKDOWN".ljust(61) + "║")
        print(f"  ╠{'═'*58}╣")
        # Header
        header = f"  ║ {'Ticker':<7}"
        for name in all_strategy_names:
            header += f"{name[:12]:>12} "
        header += f"{'Combined':>10} ║"
        print(header)
        print(f"  ║{'─'*57}║")
        # Rows
        for ticker, tw in sorted(combined_weights.items(), key=lambda x: -x[1]):
            if tw < 0.01:
                continue
            row = f"  ║ {ticker:<7}"
            for name in all_strategy_names:
                c = contribution[ticker].get(name, 0.0) * 100
                row += f"{c:>10.1f}% "
            row += f"{'→':>2} {tw*100:>5.1f}% ║"
            print(row)
        # Footer: strategy total bar
        print(f"  ║{'─'*57}║")
        for i, name in enumerate(all_strategy_names):
            total_strat = sum(contribution.get(t, {}).get(name, 0.0) for t in all_tickers_in_use) * 100
            print(f"  ║  {name}: allocation contribution = {total_strat:.1f}% of portfolio".ljust(61) + "║")
        print(f"  ╚{'═'*58}╝")

        print(f"\n  ── Final Combined Weights ──")
        for t, w in sorted(combined_weights.items(), key=lambda x: -x[1]):
            if w > 0.01:
                bar = "█" * int(w * 50)
                print(f"    {t:<6} {w*100:5.1f}% {bar}")

        # Step 4 — Risk assessment (LLM, on the combined ticker set)
        print(f"\n  ── Risk (LLM) ──")
        self.tickers = [t for t in all_tickers_in_use if t in hist.columns]
        bt_summary = self._build_backtest_summary()
        self.risk_assessment = self.risk_agent.assess(
            hist, self.tickers, target_date, bt_summary)
        print(f"  Level: {self.risk_assessment['risk_level']}  "
              f"Action: {self.risk_assessment['action']}")
        if self.risk_assessment.get("action") == "Liquidate All Long Positions":
            print(f"    Risk override: LIQUIDATE ALL")
            combined_weights = {t: 0.0 for t in self.tickers}
            self.target_weights = combined_weights

        # Step 5 — Execute
        if ALPACA_AVAILABLE:
            self.executor = AlpacaExecutor()
            self.executor.print_summary()
            self.executor.rebalance(combined_weights, dry_run=dry_run,
                                    use_bracket=use_bracket)
        else:
            print("  [TradingAgent] Alpaca unavailable — skipping execution.")

        return combined_weights

    # ── report helpers ────────────────────────────────────────────────

    def report(self):
        """Pretty-print the full pipeline result."""
        print("\n" + "=" * 60)
        if self.strategy_mode and self.active_strategy:
            print(f"  TRADING AGENT REPORT (Strategy: {self.active_strategy['name']})")
        else:
            print("  TRADING AGENT REPORT")
        print("=" * 60)
        if self.strategy_mode and self.active_strategy:
            s = self.active_strategy
            print(f"\nStrategy Config:")
            print(f"  Name: {s['name']}")
            print(f"  Factors: {len(s.get('factors',{}))} | Logic: {s.get('signal_logic','N/A')}")
            print(f"  Sizing: {s.get('position_sizing','N/A')} | "
                  f"MaxPos: {s.get('max_positions','N/A')}")
            active_factors = list(s.get('factors', {}).keys())
            print(f"  Active factors: {', '.join(active_factors[:8])}"
                  f"{'...' if len(active_factors) > 8 else ''}")
        print(f"\nMarket State:")
        print(json.dumps(self.market_state, indent=2) if self.market_state else "  (not assessed in strategy mode)")
        print(f"\nStock Signals:")
        for t, s in self.stock_signals.items():
            print(f"  {t:<6} {s.get('signal','?'):<5}  {s.get('reasoning','')}")
        print(f"\nRisk Assessment:")
        print(json.dumps(self.risk_assessment, indent=2) if self.risk_assessment else "  (not assessed)")
        if self.frequency_decision:
            fd = self.frequency_decision
            print(f"\nRebalance Frequency (LLM):")
            print(f"  Recommended: {fd.get('recommended_frequency','?')}  "
                  f"Confidence: {fd.get('confidence',0):.2f}")
            print(f"  Reasoning: {fd.get('reasoning','')}")
        print(f"\nFinal Weights:")
        for t, w in sorted(self.target_weights.items(), key=lambda x: -x[1]):
            print(f"  {t:<6} {w*100:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# INTENT RECOGNITION — natural language → menu option mapping
# ═══════════════════════════════════════════════════════════════════════

INTENT_RECOGNITION_SYSTEM = (
    "You are an intent classifier for a paper trading system. "
    "Always return valid JSON only."
)

INTENT_RECOGNITION_TEMPLATE = """
You are an intent classifier for a Paper Trading Agent system.

Available functions (with their option numbers):

[1] LLM Multi-Agent Trading (Dry Run) — Run the full LLM pipeline (Macro → StockAnalyst → RiskManager) and compute target weights, but do NOT submit orders. Keywords: AI分析, LLM分析, 模拟分析, 多智能体分析, dry run, analyze with AI, paper analysis, 看看AI怎么说, 用AI分析一下行情, 试算, 分析一下

[2] LLM Multi-Agent Trading (Real Orders) — Run the full LLM pipeline AND submit real orders to Alpaca paper account. Keywords: AI下单, LLM交易, 多智能体交易, real orders, AI trade, execute with AI, 让AI交易, 用AI实盘, AI实盘交易

[3] Strategy-Based Trading (Dry Run) — Run a single rule-based strategy from strategy_config.json without submitting orders. Keywords: 策略模拟, 策略试算, 因子策略, rule-based dry run, 用策略分析, 跑因子, 因子分析

[4] Strategy-Based Trading (Real Orders) — Run a single rule-based strategy AND submit real orders. Keywords: 策略下单, 因子实盘, 策略实盘交易, rule-based real orders, execute strategy, 用策略交易

[5] Backtest All Strategies — Run backtesting on ALL strategies and show performance comparison. Keywords: 回测所有, 全部回测, 回测全部策略, backtest all, run all backtests, 跑回测, 回测一下, 回测

[6] Backtest Single Strategy — Run backtesting on ONE specific strategy by index. Keywords: 回测单个, 回测某一个, 指定策略回测, backtest one, single backtest, 回测策略N

[7] Live Trading + Auto Backtest — Run LLM live trading plus let the agent decide whether to trigger backtests. Keywords: 自动回测, 交易加回测, auto backtest, live plus backtest

[8] Regenerate Strategy Config — Generate new random strategies in strategy_config.json. Keywords: 生成策略, 重新生成, 创建策略, generate strategies, create config, 生成配置, 新策略, 重新生成配置

[9] View Alpaca Account Summary — Show current Alpaca paper account balance, positions, P&L. Keywords: 查看账户, 账户摘要, 持仓, 余额, account summary, view account, 我的账户, 账户情况, 仓位

[10] LLM Rebalance Frequency Analysis — Use LLM to recommend optimal rebalance frequency. Keywords: 调仓频率, 再平衡频率, rebalance frequency, 多久调仓, 调仓周期

[11] Multi-Strategy Parallel Trading — Run 2+ strategies simultaneously with capital split. Keywords: 多策略并行, 组合策略, 多个策略一起, parallel strategies, multi-strategy, 策略组合, 多策略

[12] Manage TP/SL (Attach + Status Check) — Check TP/SL thresholds and attach orders. Keywords: 止盈止损, 设置止盈, 止损, TP/SL, take profit, stop loss, 盈亏管理

[13] Auto-Select Best Strategy & Trade — Backtest all, rank, pick best, and trade. Keywords: 自动选最优, 最优策略, 自动选择策略, 选最好的, auto select best, best strategy, 智能选策略

Rules:
- Return the MOST SPECIFIC option(s) that match the user's intent.
- matched_options is a list of integers (empty list [] if nothing matches at all).
- If the user mentions "交易" or "下单" or "实盘", prefer Real Orders variants [2], [4] over dry run [1], [3].
- If the user says only "回测" without qualifiers, default to [5] (backtest all).
- If the user mentions a strategy number like "策略3" or "strategy 2", include [3] or [6].
- Confidence should reflect how certain the match is (>0.8 = very clear, 0.5-0.8 = reasonable, <0.5 = uncertain).
- If the request is too vague to determine a specific function, return multiple plausible options.
- If the request has nothing to do with trading/backtesting/strategies, return empty list.

User input: "{user_text}"

Output strictly as JSON:
{{
    "matched_options": [<list of integers>],
    "confidence": <float 0.0-1.0>,
    "reasoning": "<one sentence in Chinese explaining the match>"
}}
"""


def _recognize_intent(user_text: str) -> dict:
    """Use LLM to map natural language to menu option number(s).

    Returns:
        {matched_options: [int, ...], confidence: float, reasoning: str}
    """
    decider = BaseAgent(name="IntentRecognizer", system_prompt=INTENT_RECOGNITION_SYSTEM)
    prompt = INTENT_RECOGNITION_TEMPLATE.format(user_text=user_text)
    result = decider.call_llm(prompt)

    if result is None:
        return {
            "matched_options": [],
            "confidence": 0.0,
            "reasoning": "LLM unavailable — cannot recognize intent.",
        }

    opts = result.get("matched_options", [])
    if isinstance(opts, int):
        opts = [opts]
    opts = [int(o) for o in opts if isinstance(o, (int, float)) and 1 <= int(o) <= 13]

    return {
        "matched_options": opts,
        "confidence": float(result.get("confidence", 0.0)),
        "reasoning": str(result.get("reasoning", "")),
    }


# Option number → (label, handler function)
# Populated inside _interactive_loop so it has access to target_date and loop state.
_OPTION_LABELS: dict[int, str] = {
    1: "LLM Multi-Agent Trading (Dry Run)",
    2: "LLM Multi-Agent Trading (Real Orders)",
    3: "Strategy-Based Trading (Dry Run)",
    4: "Strategy-Based Trading (Real Orders)",
    5: "Backtest All Strategies",
    6: "Backtest Single Strategy",
    7: "Live Trading + Auto Backtest",
    8: "Regenerate Strategy Config",
    9: "View Alpaca Account Summary",
    10: "LLM Rebalance Frequency Analysis",
    11: "Multi-Strategy Parallel Trading",
    12: "Manage TP/SL (Attach + Status Check)",
    13: "Auto-Select Best Strategy & Trade",
}


# ═══════════════════════════════════════════════════════════════════════
# MAIN — unified entry point for live trading + backtesting
# ═══════════════════════════════════════════════════════════════════════

def _show_menu():
    """Display the interactive menu."""
    print(r"""
╔══════════════════════════════════════════════════╗
║              PAPER TRADING AGENT                 ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  [1] LLM Multi-Agent Trading (Dry Run)           ║
║  [2] LLM Multi-Agent Trading (Real Orders)       ║
║  [3] Strategy-Based Trading (Dry Run)            ║
║  [4] Strategy-Based Trading (Real Orders)        ║
║  [5] Backtest All Strategies                     ║
║  [6] Backtest Single Strategy                    ║
║  [7] Live Trading + Auto Backtest                ║
║  [8] Regenerate Strategy Config                  ║
║  [9] View Alpaca Account Summary                 ║
║ [10] LLM Rebalance Frequency Analysis            ║
║ [11] Multi-Strategy Parallel Trading             ║
║ [12] Manage TP/SL (Attach + Status Check)        ║
║ [13] Auto-Select Best Strategy & Trade           ║
║ [14] Natural Language Input (AI Intent Recognition)║
║  [0] Exit                                        ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")


def _list_strategies():
    """Print available strategies from strategy_config.json."""
    try:
        strategies = StrategyConfig.load()
    except Exception as e:
        print(f"  Failed to load strategy config: {e}")
        return []
    if not strategies:
        print("  No strategies found. Run [8] to generate config first.")
        return []
    print(f"\n  Available strategies ({len(strategies)}):")
    for i, s in enumerate(strategies):
        factors = list(s.get("factors", {}).keys())
        tickers = s.get("tickers", [])
        print(f"    [{i}] {s['name']}")
        print(f"        Tickers: {', '.join(tickers[:6])}"
              f"{'...' if len(tickers) > 6 else ''}")
        print(f"        Factors: {len(factors)} | "
              f"Logic: {s.get('signal_logic','?')} | "
              f"Sizing: {s.get('position_sizing','?')}")
    print()
    return strategies


def _pick_strategy(strategies: list) -> int | None:
    """Prompt user to pick a strategy index. Returns index or None."""
    if not strategies:
        return None
    while True:
        try:
            raw = input(f"  Select strategy index [0-{len(strategies)-1}]: ").strip()
            if raw == "":
                return None
            idx = int(raw)
            if 0 <= idx < len(strategies):
                return idx
            print(f"  Invalid index. Choose 0-{len(strategies)-1}.")
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


def _view_account_summary():
    """Print Alpaca paper account summary."""
    if not ALPACA_AVAILABLE:
        print("  alpaca-py not installed. Run: pip install alpaca-py")
        return
    try:
        executor = AlpacaExecutor()
        executor.print_summary()
    except Exception as e:
        print(f"  Failed to fetch account summary: {e}")


def _execute_option(choice: str, target_date: str) -> str:
    """Execute a single menu option. Returns 'continue', 'exit', or 'invalid'.

    This function is called both from the numeric menu and from the intent
    recognition path ([14]) so that natural-language input can dispatch to
    the same handler code without duplication.
    """
    # ── [0] Exit ──────────────────────────────────────────────────
    if choice == "0":
        print("  Goodbye.")
        return "exit"

    # ── [1] LLM Multi-Agent (Dry Run) ──────────────────────────
    elif choice == "1":
        print(f"\n  ── LLM Multi-Agent Trading (Dry Run) | {target_date} ──\n")
        agent = TradingAgent()
        agent.run(target_date=target_date, dry_run=True)
        agent.report()
        return "continue"

    # ── [2] LLM Multi-Agent (Real Orders) ──────────────────────
    elif choice == "2":
        confirm = input("  Submit REAL orders to Alpaca paper account? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
        else:
            print(f"\n  ── LLM Multi-Agent Trading (Real Orders) | {target_date} ──\n")
            agent = TradingAgent()
            agent.run(target_date=target_date, dry_run=False)
            agent.report()
        return "continue"

    # ── [3] Strategy-Based (Dry Run) ───────────────────────────
    elif choice == "3":
        strategies = _list_strategies()
        idx = _pick_strategy(strategies)
        if idx is None:
            print("  Cancelled.")
        else:
            print(f"\n  ── Strategy-Based Trading (Dry Run) | #{idx} | {target_date} ──\n")
            agent = TradingAgent()
            agent.run_with_strategy(idx, target_date=target_date, dry_run=True)
            agent.report()
        return "continue"

    # ── [4] Strategy-Based (Real Orders) ───────────────────────
    elif choice == "4":
        strategies = _list_strategies()
        idx = _pick_strategy(strategies)
        if idx is None:
            print("  Cancelled.")
        else:
            confirm = input("  Submit REAL orders to Alpaca paper account? [y/N]: ").strip().lower()
            if confirm != "y":
                print("  Cancelled.")
            else:
                print(f"\n  ── Strategy-Based Trading (Real Orders) | #{idx} | {target_date} ──\n")
                agent = TradingAgent()
                agent.run_with_strategy(idx, target_date=target_date, dry_run=False)
                agent.report()
        return "continue"

    # ── [5] Backtest All Strategies ────────────────────────────
    elif choice == "5":
        print(f"\n  ── Backtest All Strategies ──\n")
        agent = TradingAgent()
        agent.trigger_backtest(verbose=True)
        return "continue"

    # ── [6] Backtest Single Strategy ───────────────────────────
    elif choice == "6":
        strategies = _list_strategies()
        idx = _pick_strategy(strategies)
        if idx is None:
            print("  Cancelled.")
        else:
            print(f"\n  ── Backtest Strategy [{idx}] ──\n")
            agent = TradingAgent()
            agent.trigger_backtest([idx], verbose=True)
        return "continue"

    # ── [7] Live Trading + Auto Backtest ───────────────────────
    elif choice == "7":
        print(f"\n  ── Live Trading + Auto Backtest | {target_date} ──\n")
        agent = TradingAgent()
        agent.run(target_date=target_date, dry_run=True)
        agent.report()
        print(f"\n  ── Auto Backtest Decision ──")
        results = agent.auto_backtest(target_date)
        if results:
            print(f"\n  {len(results)} strategy(s) backtested.")
        return "continue"

    # ── [8] Regenerate Strategy Config ─────────────────────────
    elif choice == "8":
        try:
            n = input("  Number of strategies to generate [5]: ").strip()
            n = int(n) if n else 5
        except (ValueError, EOFError, KeyboardInterrupt):
            n = 5
        print(f"\n  ── Generating {n} strategies ──\n")
        StrategyConfig.generate_full_config(num_strategies=n)
        print("  Done.")
        return "continue"

    # ── [9] View Alpaca Account Summary ────────────────────────
    elif choice == "9":
        _view_account_summary()
        return "continue"

    # ── [10] LLM Rebalance Frequency Analysis ───────────────────
    elif choice == "10":
        print(f"\n  ── Rebalance Frequency Analysis | {target_date} ──\n")
        agent = TradingAgent()
        agent.raw_data = agent.data_agent.fetch_market_data(
            agent.tickers, BACKTEST_START, target_date
        )
        end_dt = pd.to_datetime(target_date)
        hist = agent.raw_data[agent.raw_data.index < end_dt].copy()
        print(f"  ── Macro ──")
        agent.market_state = agent.macro_agent.assess(hist, target_date)
        print(f"  Phase: {agent.market_state['market_phase']}  "
              f"Appetite: {agent.market_state['risk_appetite']}/10")
        print(f"  ── Risk ──")
        bt_summary = agent._build_backtest_summary()
        agent.risk_assessment = agent.risk_agent.assess(hist, agent.tickers, target_date, bt_summary)
        print(f"  Level: {agent.risk_assessment['risk_level']}  "
              f"Action: {agent.risk_assessment['action']}")
        print(f"  ── Frequency Decision ──")
        agent.decide_rebalance_frequency(target_date)
        return "continue"

    # ── [11] Multi-Strategy Parallel Trading ────────────────────
    elif choice == "11":
        strategies = _list_strategies()
        if not strategies:
            return "continue"
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  Multi-Strategy: select 2+ strategies to run    │")
        print("  │  in parallel. Each gets a capital slice, then   │")
        print("  │  their target weights are merged into one.       │")
        print("  └─────────────────────────────────────────────────┘")
        raw = input("  Strategy indices (comma-separated, e.g. 0,1,2): ").strip()
        if not raw:
            print("  Cancelled.")
            return "continue"
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("  Invalid indices.")
            return "continue"
        if len(indices) < 2:
            print("  Multi-strategy needs at least 2 strategies. Use [3] or [4] for single.")
            return "continue"
        # Show selected strategies and capital allocation
        print(f"\n  ┌── Selected Strategies ({len(indices)}) ──────────────────┐")
        for j, idx in enumerate(indices):
            s = strategies[idx]
            tickers = s.get('tickers', [])
            factors = list(s.get('factors', {}).keys())
            print(f"  │ [{j}] #{idx} {s['name']:<25} "
                  f"{len(factors)} factors, {len(tickers)} tickers │")
        print(f"  └{'─'*54}┘")
        # Capital allocation
        print(f"\n  Capital allocation (default: equal split):")
        custom_cap = input("  Enter custom weights (e.g. 50,30,20) or press Enter for equal: ").strip()
        if custom_cap:
            try:
                cap_weights = [float(x.strip()) for x in custom_cap.split(",") if x.strip()]
                if len(cap_weights) != len(indices):
                    print(f"  Need {len(indices)} weights, got {len(cap_weights)}. Using equal split.")
                    cap_weights = None
                else:
                    total = sum(cap_weights)
                    cap_weights = [w / total for w in cap_weights]
            except ValueError:
                print("  Invalid weights. Using equal split.")
                cap_weights = None
        else:
            cap_weights = None
        # Show allocation
        print(f"\n  ┌── Capital Allocation ────────────────────────────┐")
        for j, idx in enumerate(indices):
            s = strategies[idx]
            cw = cap_weights[j] if cap_weights else 1.0 / len(indices)
            bar = "█" * int(cw * 30)
            print(f"  │ [{j}] {s['name']:<25} {cw*100:5.1f}% {bar} │")
        print(f"  └{'─'*54}┘")
        # Confirm
        print(f"\n  ── Multi-Strategy ({len(indices)} strategies) | {target_date} ──")
        confirm = input("  Submit REAL orders to Alpaca paper account? [y/N]: ").strip().lower()
        dry = confirm != "y"
        use_bracket = input("  Use bracket orders with TP/SL for buys? [y/N]: ").strip().lower() == "y"
        agent = TradingAgent()
        agent.run_multi_strategies(indices, target_date=target_date,
                                   dry_run=dry, use_bracket=use_bracket,
                                   capital_weights=cap_weights)
        return "continue"

    # ── [12] Manage TP/SL (Attach + Status Check) ───────────────
    elif choice == "12":
        if not ALPACA_AVAILABLE:
            print("  alpaca-py not installed.")
        else:
            print(f"\n  ── TP/SL Management ──")
            tp = input(f"  Take-Profit % [{TAKE_PROFIT_PCT*100:.0f}]: ").strip()
            sl = input(f"  Stop-Loss % [{STOP_LOSS_PCT*100:.0f}]: ").strip()
            tp_pct = float(tp) / 100 if tp else TAKE_PROFIT_PCT
            sl_pct = float(sl) / 100 if sl else STOP_LOSS_PCT
            execute = input("  Submit TP/SL orders? [y/N]: ").strip().lower() == "y"
            executor = AlpacaExecutor()
            executor.check_tpsl_status()
            executor.attach_tpsl_all(tp_pct=tp_pct, sl_pct=sl_pct, dry_run=not execute)
        return "continue"

    # ── [13] Auto-Select Best Strategy & Trade ──────────────────
    elif choice == "13":
        print(f"\n  ── Auto-Select Best Strategy & Trade | {target_date} ──\n")
        confirm = input("  Submit REAL orders to Alpaca paper account? [y/N]: ").strip().lower()
        dry = confirm != "y"
        use_bracket = False
        if not dry:
            use_bracket = input("  Use bracket orders with TP/SL for buys? [y/N]: ").strip().lower() == "y"
        agent = TradingAgent()
        try:
            agent.auto_select_and_trade(target_date=target_date,
                                        dry_run=dry, use_bracket=use_bracket)
        except ValueError as e:
            print(f"  Error: {e}")
        return "continue"

    # ── [14] Natural Language Input (AI Intent Recognition) ─────
    elif choice == "14":
        print(f"\n  ── Natural Language Input (AI Intent Recognition) ──")
        print("  Describe what you want to do in plain English or Chinese.")
        try:
            user_text = input("  Your request: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("  Cancelled.")
            return "continue"
        if not user_text:
            print("  Cancelled.")
            return "continue"

        print(f"  Recognizing intent for: \"{user_text}\"")
        intent = _recognize_intent(user_text)
        matched = intent.get("matched_options", [])
        confidence = intent.get("confidence", 0.0)
        reasoning = intent.get("reasoning", "")

        print(f"  Reasoning: {reasoning}")
        print(f"  Confidence: {confidence:.2f}")

        # ── Case 1: no match ──
        if not matched:
            print(f"\n  ✗ Unable to match any existing function.")
            print(f"  Try a more specific description, or select an option from the menu.")
            print(f"  Available functions:")
            for num, label in _OPTION_LABELS.items():
                print(f"    [{num}] {label}")
            return "continue"

        # ── Case 2: single match ──
        if len(matched) == 1:
            opt = matched[0]
            label = _OPTION_LABELS.get(opt, f"Option {opt}")
            print(f"\n  ✓ Matched: [{opt}] {label}")
            if confidence >= 0.7:
                confirm = input(f"  Execute this function? [Y/n]: ").strip().lower()
                if confirm in ("n", "no"):
                    print("  Cancelled.")
                    return "continue"
            else:
                confirm = input(f"  Low confidence. Execute anyway? [y/N]: ").strip().lower()
                if confirm != "y":
                    print("  Cancelled.")
                    return "continue"
            print(f"  → Executing [{opt}] {label}\n")
            return _execute_option(str(opt), target_date)

        # ── Case 3: multiple matches ──
        print(f"\n  ⚠ Matched {len(matched)} possible functions:")
        for i, opt in enumerate(matched):
            label = _OPTION_LABELS.get(opt, f"Option {opt}")
            print(f"    [{i+1}] [{opt}] {label}")
        print(f"    [0] Cancel")
        try:
            pick = input(f"  Select (1-{len(matched)}): ").strip()
            if pick == "0" or not pick:
                print("  Cancelled.")
                return "continue"
            pick_idx = int(pick) - 1
            if 0 <= pick_idx < len(matched):
                opt = matched[pick_idx]
                label = _OPTION_LABELS.get(opt, f"Option {opt}")
                print(f"  → Executing [{opt}] {label}\n")
                return _execute_option(str(opt), target_date)
            else:
                print("  Invalid selection.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("  Cancelled.")
        return "continue"

    else:
        return "invalid"


def _interactive_loop():
    """Run the interactive CLI menu loop."""
    target_date = datetime.today().strftime("%Y-%m-%d")

    while True:
        _show_menu()
        try:
            choice = input("  Select an option [0-14]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        status = _execute_option(choice, target_date)
        if status == "exit":
            break
        elif status == "invalid":
            print(f"  Invalid option. Choose 0-14.")


if __name__ == "__main__":
    # ── Interactive menu (no arguments) ──────────────────────────────────
    if len(sys.argv) == 1:
        _interactive_loop()
        sys.exit(0)

    import argparse

    parser = argparse.ArgumentParser(
        description="Paper Trading Agent — Live trading + Backtesting orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Paper_Trading_Agent.py                                   # interactive menu
  python Paper_Trading_Agent.py --live                             # LLM live trading (real orders)
  python Paper_Trading_Agent.py --use-strategy 2                   # strategy #2 rule-based trading (dry run)
  python Paper_Trading_Agent.py --live --use-strategy 0            # strategy #0 rule-based trading (real orders)
  python Paper_Trading_Agent.py --backtest                         # backtest all strategies
  python Paper_Trading_Agent.py --backtest --strategy 0            # backtest strategy #0 only
  python Paper_Trading_Agent.py --auto-backtest                    # LLM live + agent decides backtest
  python Paper_Trading_Agent.py --auto-select                      # backtest all, select best, dry run trade
  python Paper_Trading_Agent.py --auto-select --live               # backtest all, select best, real orders
  python Paper_Trading_Agent.py --generate-config                   # regenerate strategy_config.json
  python Paper_Trading_Agent.py --generate-config --strategies 8
  python Paper_Trading_Agent.py --summary                          # view Alpaca account summary
        """,
    )

    # ── Mode selection (mutually exclusive) ──────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="Live trading with real Alpaca orders")
    mode.add_argument("--dry-run", action="store_true",
                      help="Live trading pipeline without submitting orders (default)")
    mode.add_argument("--backtest", action="store_true",
                      help="Run backtesting only (no live trading)")
    mode.add_argument("--auto-backtest", action="store_true",
                      help="Live trading + agent decides whether to run backtests")
    mode.add_argument("--generate-config", action="store_true",
                      help="Regenerate strategy_config.json and exit")
    mode.add_argument("--summary", action="store_true",
                      help="View Alpaca paper account summary and exit")

    # ── Options ─────────────────────────────────────────────────────────
    parser.add_argument("--strategy", type=int, default=None,
                        help="For --backtest: run a single strategy by index (0-based)")
    parser.add_argument("--use-strategy", type=int, default=None,
                        help="Use strategy_config.json[N] as the trading rules "
                             "(bypasses LLM, uses 24-factor rule engine)")
    parser.add_argument("--auto-select", action="store_true",
                        help="Backtest all strategies, auto-select the best, and paper trade")
    parser.add_argument("--strategies", type=int, default=5,
                        help="Number of strategies when generating config")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list override")
    parser.add_argument("--date", type=str, default="",
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # ── Summary mode ────────────────────────────────────────────────────
    if args.summary:
        _view_account_summary()
        sys.exit(0)

    # ── Generate config mode ────────────────────────────────────────────
    if args.generate_config:
        StrategyConfig.generate_full_config(num_strategies=args.strategies)
        print("Done. Run with --backtest to execute.")
        sys.exit(0)

    # ── Build agent ─────────────────────────────────────────────────────
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    agent = TradingAgent(tickers=tickers)
    target_date = args.date or datetime.today().strftime("%Y-%m-%d")

    # ── Backtest-only mode ──────────────────────────────────────────────
    if args.backtest:
        print(f"\n{'='*60}")
        print(f"  BACKTEST-ONLY MODE")
        print(f"{'='*60}")
        if args.strategy is not None:
            agent.trigger_backtest([args.strategy], verbose=not args.quiet)
        else:
            agent.trigger_backtest(verbose=not args.quiet)
        sys.exit(0)

    # ── Live + Auto-backtest mode ───────────────────────────────────────
    if args.auto_backtest:
        print(f"\n{'='*60}")
        print(f"  LIVE TRADING + AUTO BACKTEST")
        print(f"{'='*60}")
        if args.use_strategy is not None:
            agent.run_with_strategy(args.use_strategy, target_date=target_date, dry_run=True)
        else:
            agent.run(target_date=target_date, dry_run=True)
        agent.report()

        print(f"\n{'='*60}")
        print(f"  AUTO BACKTEST DECISION")
        print(f"{'='*60}")
        results = agent.auto_backtest(target_date)
        if results:
            print(f"\n  {len(results)} strategy(s) backtested.")
        sys.exit(0)

    # ── Live-only mode (default) ────────────────────────────────────────
    dry_run = not args.live
    use_strat = args.use_strategy
    auto_select = args.auto_select

    # ── Auto-select best strategy branch ────────────────────────────────
    if auto_select:
        mode_label = "AUTO-SELECT BEST STRATEGY"
        order_label = "real orders" if not dry_run else "dry run"
        print(f"\n{'='*60}")
        print(f"  LIVE TRADING — {mode_label} ({order_label})")
        print(f"{'='*60}")
        agent.auto_select_and_trade(target_date=target_date, dry_run=dry_run)
        sys.exit(0)

    if use_strat is not None:
        mode_label = f"STRATEGY #{use_strat} RULE-BASED"
    else:
        mode_label = "LLM MULTI-AGENT"

    order_label = "real orders" if not dry_run else "dry run"
    print(f"\n{'='*60}")
    print(f"  LIVE TRADING — {mode_label} ({order_label})")
    print(f"{'='*60}")

    if use_strat is not None:
        agent.run_with_strategy(use_strat, target_date=target_date, dry_run=dry_run)
    else:
        agent.run(target_date=target_date, dry_run=dry_run)
    agent.report()

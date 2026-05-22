"""
BacktestEngine — Rule-based Backtesting Module (Pure Library)
==============================================================
Reads strategy_config.json, runs deterministic backtests with 24 technical factors,
and outputs performance metrics: Sharpe, Calmar, Sortino, Max Drawdown, Volatility,
Beta, Win Rate, P/L Ratio, Profit Factor.

Usage — import from Paper_Trading_Agent.py:
    from BacktestEngine import BacktestEngine, StrategyConfig
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(MODULE_DIR, "strategy_config.json")
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate

# ── Available factor pool (24 factors) ────────────────────────────────────
FACTOR_POOL = [
    "sma_crossover_20_50",
    "sma_crossover_50_200",
    "ema_crossover_12_26",
    "rsi_14",
    "macd_signal",
    "macd_histogram",
    "bollinger_position",
    "bollinger_squeeze",
    "momentum_20d",
    "momentum_60d",
    "momentum_120d",
    "volatility_20d",
    "volatility_regime",
    "volume_ratio",
    "volume_price_trend",
    "atr_14",
    "stochastic_14",
    "cci_20",
    "williams_r_14",
    "adx_14",
    "obv_trend",
    "beta_spy",
    "return_5d_reversal",
    "drawdown_60d",
]

TICKER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "JNJ", "XOM", "UNH", "BAC", "PG", "V", "MA", "HD", "DIS", "NFLX",
    "ADBE", "CRM", "AMD", "INTC", "PFE", "WMT", "KO", "PEP", "CSCO",
    "QCOM", "TXN", "AVGO", "COST", "ABBV",
]


# ═══════════════════════════════════════════════════════════════════════════
# FACTOR LIBRARY — 24 technical factors
# ═══════════════════════════════════════════════════════════════════════════

class FactorLibrary:
    """Compute 24 technical factors from price/volume data.

    Each factor returns a pandas Series of raw values (typically a z-score or
    normalized signal in [-1, 1]) aligned to the input DataFrame's index.
    """

    def __init__(self, prices: pd.DataFrame, benchmark_prices: pd.Series | None = None):
        """
        Args:
            prices: DataFrame of close prices, columns = tickers
            benchmark_prices: Series of benchmark close prices (e.g. SPY)
        """
        self.prices = prices
        self.benchmark = benchmark_prices
        self._cache: dict[str, pd.DataFrame] = {}

    # ── helpers ──────────────────────────────────────────────────────────

    def _sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=window).mean()

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _roc(self, series: pd.Series, window: int) -> pd.Series:
        return series.pct_change(window)

    def _zscore(self, series: pd.Series, window: int = 252) -> pd.Series:
        roll_mean = series.rolling(window, min_periods=63).mean()
        roll_std = series.rolling(window, min_periods=63).std()
        return (series - roll_mean) / roll_std.replace(0, np.nan)

    def _true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ── factor methods — each returns a DataFrame indexed like self.prices ─

    def sma_crossover_20_50(self) -> pd.DataFrame:
        """1: SMA20 > SMA50 (bullish), -1: SMA20 < SMA50 (bearish)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            sma20 = self._sma(s, 20)
            sma50 = self._sma(s, 50)
            raw = (sma20 - sma50) / sma50.replace(0, np.nan)
            result[col] = raw.apply(lambda x: np.clip(x * 20, -1, 1))
        return result

    def sma_crossover_50_200(self) -> pd.DataFrame:
        """Golden cross / death cross signal."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            sma50 = self._sma(s, 50)
            sma200 = self._sma(s, 200)
            raw = (sma50 - sma200) / sma200.replace(0, np.nan)
            result[col] = raw.apply(lambda x: np.clip(x * 10, -1, 1))
        return result

    def ema_crossover_12_26(self) -> pd.DataFrame:
        """EMA12 vs EMA26 crossover (like MACD without signal line)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            ema12 = self._ema(s, 12)
            ema26 = self._ema(s, 26)
            raw = (ema12 - ema26) / ema26.replace(0, np.nan)
            result[col] = raw.apply(lambda x: np.clip(x * 20, -1, 1))
        return result

    def rsi_14(self) -> pd.DataFrame:
        """RSI-14: overbought >70 (sell -1), oversold <30 (buy +1)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            rsi = self._rsi(self.prices[col], 14)
            result[col] = ((50 - rsi) / 20).clip(-1, 1)
        return result

    def macd_signal(self) -> pd.DataFrame:
        """MACD line vs signal line crossover."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            ema12 = self._ema(s, 12)
            ema26 = self._ema(s, 26)
            macd = ema12 - ema26
            signal = self._ema(macd, 9)
            hist = macd - signal
            hist_std = hist.rolling(63, min_periods=20).std()
            result[col] = (hist / hist_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def macd_histogram(self) -> pd.DataFrame:
        """MACD histogram acceleration/deceleration."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            ema12 = self._ema(s, 12)
            ema26 = self._ema(s, 26)
            macd = ema12 - ema26
            signal = self._ema(macd, 9)
            hist = macd - signal
            hist_diff = hist.diff()
            hist_diff_std = hist_diff.rolling(63, min_periods=20).std()
            result[col] = (hist_diff / hist_diff_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def bollinger_position(self) -> pd.DataFrame:
        """Where price sits within Bollinger Bands (20,2).
        1 = at upper band (overbought), -1 = at lower band (oversold)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            sma20 = self._sma(s, 20)
            std20 = s.rolling(20, min_periods=20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            band_range = upper - lower
            pos = (s - sma20) / band_range.replace(0, np.nan)
            # Flip: high position = sell signal, low = buy signal (mean-reversion bias)
            result[col] = (-pos * 2).clip(-1, 1)
        return result

    def bollinger_squeeze(self) -> pd.DataFrame:
        """Bandwidth contracting (squeeze) = +1, expanding = -1.
        Squeeze often precedes breakout."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            sma20 = self._sma(s, 20)
            std20 = s.rolling(20, min_periods=20).std()
            bw = (2 * std20) / sma20.replace(0, np.nan)
            bw_z = self._zscore(bw, 63)
            result[col] = bw_z.clip(-1, 1)
        return result

    def momentum_20d(self) -> pd.DataFrame:
        """20-day rate-of-change normalized."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            roc = self._roc(self.prices[col], 20) * 100
            roc_std = roc.rolling(252, min_periods=63).std()
            result[col] = (roc / roc_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def momentum_60d(self) -> pd.DataFrame:
        """60-day rate-of-change normalized."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            roc = self._roc(self.prices[col], 60) * 100
            roc_std = roc.rolling(252, min_periods=63).std()
            result[col] = (roc / roc_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def momentum_120d(self) -> pd.DataFrame:
        """120-day rate-of-change normalized."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            roc = self._roc(self.prices[col], 120) * 100
            roc_std = roc.rolling(252, min_periods=63).std()
            result[col] = (roc / roc_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def volatility_20d(self) -> pd.DataFrame:
        """Annualized 20-day volatility z-score. High vol = negative signal."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            vol = self.prices[col].pct_change().rolling(20).std() * np.sqrt(252)
            vol_z = self._zscore(vol, 252)
            result[col] = (-vol_z).clip(-1, 1)
        return result

    def volatility_regime(self) -> pd.DataFrame:
        """Volatility percentile over last 252 days. Low vol regime = +1."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            vol = self.prices[col].pct_change().rolling(20).std() * np.sqrt(252)
            vol_rank = vol.rolling(252, min_periods=63).apply(
                lambda x: (x < x.iloc[-1]).mean() if len(x) > 0 else 0.5
            )
            result[col] = (1 - vol_rank * 2).clip(-1, 1)  # high rank = high vol = bad
        return result

    def volume_ratio(self) -> pd.DataFrame:
        """Volume / 20-day avg volume. >1 = volume surge.
        Requires volume data from yfinance multi-level columns."""
        return pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)

    def volume_price_trend(self) -> pd.DataFrame:
        """Price * volume trend. Simplified: sign of price change * volume ratio."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            ret = self.prices[col].pct_change()
            vpt = ret.rolling(20).sum() * 100
            vpt_std = vpt.rolling(252, min_periods=63).std()
            result[col] = (vpt / vpt_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def atr_14(self) -> pd.DataFrame:
        """ATR(14) normalized by price. High ATR = risk signal."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            tr = self._true_range(s, s, s)
            atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
            atr_pct = atr / s.replace(0, np.nan)
            atr_z = self._zscore(atr_pct, 252)
            result[col] = (-atr_z).clip(-1, 1)
        return result

    def stochastic_14(self) -> pd.DataFrame:
        """Stochastic %K (14,3). <20 oversold = buy(+1), >80 overbought = sell(-1)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            low14 = s.rolling(14, min_periods=14).min()
            high14 = s.rolling(14, min_periods=14).max()
            denom = (high14 - low14).replace(0, np.nan)
            stoch_k = 100 * (s - low14) / denom
            result[col] = ((50 - stoch_k) / 30).clip(-1, 1)
        return result

    def cci_20(self) -> pd.DataFrame:
        """Commodity Channel Index (20). +100 overbought, -100 oversold."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            tp = s  # typical price simplified to close
            sma20 = self._sma(tp, 20)
            mad = tp.rolling(20).apply(lambda x: (x - x.mean()).abs().mean())
            cci = (tp - sma20) / (0.015 * mad.replace(0, np.nan))
            result[col] = (-cci / 100).clip(-1, 1)
        return result

    def williams_r_14(self) -> pd.DataFrame:
        """Williams %R. < -80 oversold, > -20 overbought."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            high14 = s.rolling(14, min_periods=14).max()
            low14 = s.rolling(14, min_periods=14).min()
            denom = (high14 - low14).replace(0, np.nan)
            wr = -100 * (high14 - s) / denom
            result[col] = ((wr + 50) / 30).clip(-1, 1)  # center at -50
        return result

    def adx_14(self) -> pd.DataFrame:
        """ADX(14) strength. High ADX(>25) with trend = strong signal.
        Simplified implementation."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            tr = self._true_range(s, s, s)
            atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
            plus_dm = s.diff().clip(lower=0)
            minus_dm = (-s.diff()).clip(lower=0)
            plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean()) / atr.replace(0, np.nan)
            minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean()) / atr.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
            # Signal: trend direction * adx strength
            raw = (plus_di - minus_di) / 100 * (adx / 50)
            result[col] = raw.clip(-1, 1)
        return result

    def obv_trend(self) -> pd.DataFrame:
        """On-Balance Volume trend simplified (price trend as proxy)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            obv_signal = s.pct_change().rolling(5).sum()
            obv_trend = obv_signal.rolling(20).mean()
            obv_std = obv_signal.rolling(252, min_periods=63).std()
            result[col] = (obv_trend / obv_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def beta_spy(self) -> pd.DataFrame:
        """Rolling 60-day beta to SPY benchmark."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        if self.benchmark is None:
            result[:] = 0.0
            return result
        bench_ret = self.benchmark.pct_change().dropna()
        for col in self.prices.columns:
            asset_ret = self.prices[col].pct_change().dropna()
            common_idx = asset_ret.index.intersection(bench_ret.index)
            a = asset_ret.loc[common_idx]
            b = bench_ret.loc[common_idx]
            rolling_cov = a.rolling(60, min_periods=30).cov(b)
            rolling_var = b.rolling(60, min_periods=30).var()
            beta = (rolling_cov / rolling_var.replace(0, np.nan)).reindex(self.prices.index)
            # Signal: beta > 1.2 bullish in up market, beta < 0.8 defensive
            result[col] = ((beta - 1.0) / 0.5).clip(-1, 1)
        return result

    def return_5d_reversal(self) -> pd.DataFrame:
        """5-day reversal signal. Negative 5d return = buy (+1), positive = sell (-1)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            ret5 = self.prices[col].pct_change(5) * 100
            ret5_std = ret5.rolling(252, min_periods=63).std()
            result[col] = (-ret5 / ret5_std.replace(0, np.nan)).clip(-1, 1)
        return result

    def drawdown_60d(self) -> pd.DataFrame:
        """Recent 60-day max drawdown. Large DD = contrarian buy signal (+1)."""
        result = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        for col in self.prices.columns:
            s = self.prices[col]
            rolling_peak = s.rolling(60, min_periods=30).max()
            dd = (s / rolling_peak - 1) * 100
            dd_std = dd.rolling(252, min_periods=63).std()
            result[col] = (-dd / dd_std.replace(0, np.nan)).clip(-1, 1)
        return result

    # ── dispatcher ───────────────────────────────────────────────────────

    def compute_all_factors(self, factor_names: list[str]) -> dict[str, pd.DataFrame]:
        """Compute and return a dict of factor_name → DataFrame."""
        result = {}
        for name in factor_names:
            method = getattr(self, name, None)
            if method is None:
                print(f"  [FactorLibrary] Unknown factor: {name}")
                continue
            result[name] = method()
        return result


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG
# ═══════════════════════════════════════════════════════════════════════════

class StrategyConfig:
    """Load, validate, and generate strategy configurations."""

    @staticmethod
    def load(path: str = CONFIG_PATH) -> list[dict]:
        """Load strategy configurations from JSON. Returns list of strategy dicts."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        strategies = data.get("strategies", [])
        if not strategies:
            raise ValueError(f"No strategies found in {path}")
        print(f"[StrategyConfig] Loaded {len(strategies)} strategy(s) from {path}")
        return strategies

    @staticmethod
    def validate(strategy: dict) -> tuple[bool, str]:
        """Validate a single strategy dict. Returns (is_valid, error_message)."""
        required = ["name", "start_date", "end_date", "tickers", "initial_capital", "factors"]
        for key in required:
            if key not in strategy:
                return False, f"Missing required key: '{key}'"

        # date validation
        try:
            start = pd.to_datetime(strategy["start_date"])
            end = pd.to_datetime(strategy["end_date"])
        except Exception:
            return False, "Invalid date format. Use YYYY-MM-DD."

        if (end - start).days < 504:  # ~2 years of trading days
            return False, f"Backtest period must be at least 2 years (504 trading days). Got {start} → {end}"

        if len(strategy["tickers"]) < 1:
            return False, "Need at least 1 ticker"

        if strategy["initial_capital"] <= 0:
            return False, "initial_capital must be positive"

        for fname in strategy.get("factors", {}):
            if fname not in FACTOR_POOL:
                return False, f"Unknown factor: '{fname}'. Available: {FACTOR_POOL}"

        return True, "OK"

    @staticmethod
    def generate_random(
        name: str = "random_strategy",
        num_factors: int = 8,
        num_tickers: int = 10,
        start: str = "2022-01-01",
        end: str = "2025-01-01",
    ) -> dict:
        """Generate a single random strategy for testing."""
        rng = np.random.default_rng()
        chosen_factors = list(rng.choice(FACTOR_POOL, size=min(num_factors, len(FACTOR_POOL)), replace=False))

        strategy = {
            "name": name,
            "start_date": start,
            "end_date": end,
            "tickers": list(rng.choice(TICKER_UNIVERSE, size=num_tickers, replace=False)),
            "initial_capital": int(rng.choice([50_000, 100_000, 200_000, 500_000])),
            "signal_logic": str(rng.choice(["weighted_sum", "majority_vote", "top_n"])),
            "rebalance_frequency": str(rng.choice(["daily", "weekly", "biweekly", "monthly"])),
            "position_sizing": str(rng.choice(["equal_weight", "factor_score", "risk_parity"])),
            "max_positions": int(rng.choice([5, 8, 10, 15])),
            "factors": {},
        }

        for f in chosen_factors:
            direction = int(rng.choice([-1, 1]))  # 1 = long signal when positive, -1 = inverse
            strategy["factors"][f] = {
                "weight": round(float(rng.uniform(0.3, 1.5)), 2),
                "direction": direction,
                "threshold_long": round(float(rng.uniform(0.1, 0.5)), 2),
                "threshold_short": round(float(rng.uniform(-0.5, -0.1)), 2),
            }

        return strategy

    @staticmethod
    def generate_full_config(path: str = CONFIG_PATH, num_strategies: int = 5) -> None:
        """Generate a full strategy_config.json with multiple strategies."""
        config = {"strategies": []}
        date_pairs = [
            ("2021-01-01", "2023-06-30"),
            ("2022-01-01", "2025-01-01"),
            ("2020-06-01", "2024-06-01"),
            ("2019-01-01", "2023-12-31"),
            ("2021-06-01", "2024-12-31"),
        ]
        for i in range(num_strategies):
            sd, ed = date_pairs[i % len(date_pairs)]
            strategy = StrategyConfig.generate_random(
                name=f"strategy_{i+1}",
                num_factors=np.random.randint(6, 14),
                num_tickers=np.random.randint(6, 16),
                start=sd,
                end=ed,
            )
            config["strategies"].append(strategy)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[StrategyConfig] Generated {num_strategies} strategies → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BacktestRunner:
    """Core backtesting loop — rule-based, deterministic, LLM-free."""

    def __init__(self, strategy: dict, verbose: bool = True):
        self.strategy = strategy
        self.verbose = verbose

        # Extract params
        self.name = strategy["name"]
        self.start = strategy["start_date"]
        self.end = strategy["end_date"]
        self.tickers = strategy["tickers"]
        self.capital = float(strategy["initial_capital"])
        self.factors_config = strategy.get("factors", {})
        self.signal_logic = strategy.get("signal_logic", "weighted_sum")
        self.rebalance_freq = strategy.get("rebalance_frequency", "monthly")
        self.position_sizing = strategy.get("position_sizing", "equal_weight")
        self.max_positions = strategy.get("max_positions", 10)

        # Frequency mapping
        self._freq_map = {
            "daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21,
        }

        # State (populated by run)
        self.prices: pd.DataFrame | None = None
        self.benchmark: pd.Series | None = None
        self.factor_values: dict[str, pd.DataFrame] = {}
        self.composite_signal: pd.DataFrame | None = None
        self.portfolio_returns: pd.Series | None = None
        self.trade_log: list[dict] = []
        self.daily_positions: pd.DataFrame | None = None

    # ── data fetching ────────────────────────────────────────────────────

    def _fetch_data(self) -> pd.DataFrame:
        """Download price data for tickers + SPY benchmark."""
        symbols = list(set(self.tickers + ["SPY"]))
        fetch_start = (pd.to_datetime(self.start) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        fetch_end = (pd.to_datetime(self.end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        if self.verbose:
            print(f"  Downloading {len(symbols)} symbols {fetch_start} → {fetch_end} ...")

        raw = yf.download(symbols, start=fetch_start, end=fetch_end, progress=False)
        closes = raw["Close"].ffill().dropna(axis=1, thresh=int(len(raw) * 0.9))

        self.benchmark = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
        ticker_closes = closes[[t for t in self.tickers if t in closes.columns]]
        if ticker_closes.empty:
            raise ValueError("No ticker data available after download")
        return ticker_closes

    # ── signal computation ───────────────────────────────────────────────

    def _compute_signals(self) -> None:
        """Compute composite signal from all configured factors."""
        factor_names = list(self.factors_config.keys())
        if self.verbose:
            print(f"  Computing {len(factor_names)} factors ...")

        lib = FactorLibrary(self.prices, self.benchmark)
        all_factors = lib.compute_all_factors(factor_names)

        # Build composite signal
        composite = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        total_weight = 0.0

        for fname, factor_df in all_factors.items():
            cfg = self.factors_config[fname]
            w = cfg.get("weight", 1.0)
            direction = cfg.get("direction", 1)
            thresh_long = cfg.get("threshold_long", 0.2)
            thresh_short = cfg.get("threshold_short", -0.2)

            # Apply thresholds: only strong signals count
            signal = factor_df.copy()
            signal[(signal > thresh_short) & (signal < thresh_long)] = 0.0
            signal = signal * direction * w
            composite = composite.add(signal, fill_value=0)
            total_weight += abs(w)

        # Normalize to [-1, 1]
        if total_weight > 0:
            composite = composite / total_weight

        self.composite_signal = composite
        self.factor_values = all_factors

    # ── position sizing ──────────────────────────────────────────────────

    def _compute_target_weights(self, date_idx: int) -> dict[str, float]:
        """Compute target weights at a given rebalance date."""
        if date_idx >= len(self.composite_signal):
            return {}

        signals = self.composite_signal.iloc[date_idx]
        # Keep only positive-signal tickers
        candidates = signals[signals > 0].sort_values(ascending=False)

        if candidates.empty:
            return {t: 0.0 for t in self.tickers}

        # Limit number of positions
        top_n = min(self.max_positions, len(candidates))
        selected = candidates.head(top_n)

        if self.position_sizing == "equal_weight":
            weights = {t: 1.0 / top_n for t in selected.index}
        elif self.position_sizing == "factor_score":
            total = selected.sum()
            weights = {t: s / total for t, s in selected.items()} if total > 0 else {}
        else:
            # risk_parity simplified: use inverse volatility
            vols = {}
            for t in selected.index:
                rets = self.prices[t].pct_change().dropna()
                if len(rets) > 20:
                    vols[t] = rets.tail(60).std()
                else:
                    vols[t] = 0.02
            inv_vol = {t: 1.0 / v for t, v in vols.items()}
            total_inv = sum(inv_vol.values())
            weights = {t: iv / total_inv for t, iv in inv_vol.items()} if total_inv > 0 else {}

        # Fill zero for non-selected tickers
        full_weights = {t: weights.get(t, 0.0) for t in self.tickers}
        return full_weights

    # ── main run ─────────────────────────────────────────────────────────

    def run(self) -> pd.Series:
        """Execute the backtest. Returns daily portfolio return series."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Backtest: {self.name}")
            print(f"  Period: {self.start} → {self.end}")
            print(f"  Tickers: {len(self.tickers)}, Capital: ${self.capital:,.0f}")
            print(f"{'='*60}")

        # Step 1: Data
        self.prices = self._fetch_data()
        effective_tickers = [t for t in self.tickers if t in self.prices.columns]
        self.tickers = effective_tickers
        self.prices = self.prices[self.tickers]

        # Trim to backtest window
        mask = (self.prices.index >= self.start) & (self.prices.index <= self.end)
        self.prices = self.prices[mask]

        if self.verbose:
            print(f"  Trading days in window: {len(self.prices)}")

        # Step 2: Signals
        self._compute_signals()
        # Align composite signal with price window
        self.composite_signal = self.composite_signal.loc[self.prices.index]

        # Step 3: Run backtest loop
        rebalance_interval = self._freq_map.get(self.rebalance_freq, 21)
        daily_returns = self.prices.pct_change().dropna(how="all")

        portfolio_value = self.capital
        portfolio_values = [self.capital]
        current_weights: dict[str, float] = {t: 0.0 for t in self.tickers}
        trade_log = []
        daily_weight_history = []

        for i in range(1, len(self.prices)):
            date = self.prices.index[i]

            # Rebalance?
            if i % rebalance_interval == 0 or i == 1:
                new_weights = self._compute_target_weights(i)
                # Log trades
                for t in self.tickers:
                    old_w = current_weights.get(t, 0)
                    new_w = new_weights.get(t, 0)
                    if abs(new_w - old_w) > 0.001:
                        trade_log.append({
                            "date": str(date.date()),
                            "ticker": t,
                            "action": "BUY" if new_w > old_w else "SELL",
                            "weight_change": round(new_w - old_w, 4),
                            "new_weight": round(new_w, 4),
                        })
                current_weights = new_weights

            # Daily P&L
            daily_ret = 0.0
            for t, w in current_weights.items():
                if t in daily_returns.columns and i < len(daily_returns):
                    daily_ret += w * daily_returns[t].iloc[i - 1]

            portfolio_value *= (1 + daily_ret)
            portfolio_values.append(portfolio_value)
            daily_weight_history.append(current_weights.copy())

        self.daily_positions = pd.DataFrame(
            daily_weight_history, index=self.prices.index[1:]
        )
        self.trade_log = trade_log

        # Build return series
        self.portfolio_returns = pd.Series(portfolio_values, index=self.prices.index)
        self.portfolio_returns = self.portfolio_returns.pct_change().dropna()

        if self.verbose:
            print(f"  Trades executed: {len(trade_log)}")
            print(f"  Final portfolio value: ${portfolio_value:,.2f}")

        return self.portfolio_returns


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationMetrics:
    """Compute all performance & risk metrics from a daily return series."""

    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series | None = None,
                 risk_free_rate: float = RISK_FREE_RATE):
        """
        Args:
            returns: Daily portfolio return series (decimal, not %)
            benchmark_returns: Daily benchmark return series
            risk_free_rate: Annual risk-free rate (decimal)
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.rf_annual = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

        # Cumulative
        self.cumulative = (1 + self.returns).cumprod()
        self.total_return = self.cumulative.iloc[-1] - 1
        self.annual_return = (1 + self.total_return) ** (252 / len(self.returns)) - 1

        # Basic stats
        self.mean_daily = self.returns.mean()
        self.std_daily = self.returns.std()
        self.annual_vol = self.std_daily * np.sqrt(252)

    # ── individual metrics ───────────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        if self.std_daily == 0:
            return 0.0
        excess = self.mean_daily - self.rf_daily
        return float(excess / self.std_daily * np.sqrt(252))

    def sortino_ratio(self) -> float:
        """Annualized Sortino Ratio (downside deviation only)."""
        downside = self.returns[self.returns < self.rf_daily]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0 if self.mean_daily <= self.rf_daily else float("inf")
        downside_std = downside.std()
        excess = self.mean_daily - self.rf_daily
        return float(excess / downside_std * np.sqrt(252))

    def max_drawdown(self) -> float:
        """Maximum drawdown (negative decimal, e.g. -0.15 = 15% drawdown)."""
        peak = self.cumulative.expanding().max()
        dd = self.cumulative / peak - 1
        return float(dd.min())

    def calmar_ratio(self) -> float:
        """Calmar Ratio = annualized return / |max drawdown|."""
        mdd = self.max_drawdown()
        if mdd == 0:
            return 0.0
        return float(self.annual_return / abs(mdd))

    def volatility(self) -> float:
        """Annualized volatility (standard deviation of returns)."""
        return float(self.annual_vol)

    def beta(self) -> float | None:
        """Beta coefficient vs benchmark."""
        if self.benchmark_returns is None:
            return None
        common = self.returns.index.intersection(self.benchmark_returns.index)
        if len(common) < 30:
            return None
        r = self.returns.loc[common]
        b = self.benchmark_returns.loc[common]
        cov = np.cov(r, b)[0, 1]
        var = np.var(b)
        return float(cov / var) if var != 0 else None

    def win_rate(self) -> float:
        """Proportion of days with positive returns."""
        if len(self.returns) == 0:
            return 0.0
        return float((self.returns > 0).sum() / len(self.returns))

    def profit_loss_ratio(self) -> float | None:
        """Average win / average loss (absolute)."""
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        if len(losses) == 0:
            return float("inf") if len(wins) > 0 else None
        if len(wins) == 0:
            return 0.0
        avg_loss = abs(losses.mean())
        return float(wins.mean() / avg_loss) if avg_loss != 0 else None

    def profit_factor(self) -> float | None:
        """Gross profit / gross loss."""
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        total_profit = wins.sum()
        total_loss = abs(losses.sum())
        if total_loss == 0:
            return float("inf") if total_profit > 0 else None
        return float(total_profit / total_loss)

    def value_at_risk(self, confidence: float = 0.95) -> float:
        """Historical VaR at given confidence level."""
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (expected shortfall)."""
        var = self.value_at_risk(confidence)
        tail = self.returns[self.returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    # ── aggregate report ─────────────────────────────────────────────────

    def compute_all(self) -> dict:
        """Return a dict of all metrics."""
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annual_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "calmar_ratio": round(self.calmar_ratio(), 3),
            "sortino_ratio": round(self.sortino_ratio(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "volatility_ann_pct": round(self.volatility() * 100, 2),
            "beta": round(self.beta(), 3) if self.beta() is not None else "N/A",
            "win_rate_pct": round(self.win_rate() * 100, 2),
            "profit_loss_ratio": round(self.profit_loss_ratio(), 3) if self.profit_loss_ratio() is not None else "N/A",
            "profit_factor": round(self.profit_factor(), 3) if self.profit_factor() is not None else "N/A",
            "var_95_pct": round(self.value_at_risk(0.95) * 100, 3),
            "cvar_95_pct": round(self.cvar(0.95) * 100, 3),
            "num_trading_days": len(self.returns),
        }


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Format and output backtest results."""

    @staticmethod
    def print_report(strategy: dict, metrics: dict, trade_log: list[dict] | None = None) -> None:
        """Pretty-print a single strategy's backtest report."""
        print(f"\n{'='*70}")
        print(f"  BACKTEST REPORT: {strategy['name']}")
        print(f"{'='*70}")
        print(f"  Period          : {strategy['start_date']} → {strategy['end_date']}")
        print(f"  Tickers         : {', '.join(strategy['tickers'])}")
        print(f"  Initial Capital : ${strategy['initial_capital']:,.0f}")
        print(f"  Factors         : {len(strategy.get('factors', {}))}")
        print(f"  Signal Logic    : {strategy.get('signal_logic', 'N/A')}")
        print(f"  Rebalance       : {strategy.get('rebalance_frequency', 'N/A')}")
        print(f"  Position Sizing : {strategy.get('position_sizing', 'N/A')}")
        print()

        print(f"  {'─'*60}")
        print(f"  PERFORMANCE METRICS")
        print(f"  {'─'*60}")
        print(f"  Total Return         : {metrics['total_return_pct']:>10.2f}%")
        print(f"  Annualized Return    : {metrics['annualized_return_pct']:>10.2f}%")
        print(f"  Sharpe Ratio         : {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Calmar Ratio         : {metrics['calmar_ratio']:>10.3f}")
        print(f"  Sortino Ratio        : {metrics['sortino_ratio']:>10.3f}")
        print(f"  Max Drawdown         : {metrics['max_drawdown_pct']:>10.2f}%")
        print(f"  Volatility (ann.)    : {metrics['volatility_ann_pct']:>10.2f}%")
        print(f"  Beta                 : {str(metrics['beta']):>10s}")
        print(f"  Win Rate             : {metrics['win_rate_pct']:>10.2f}%")
        print(f"  Profit/Loss Ratio    : {str(metrics['profit_loss_ratio']):>10s}")
        print(f"  Profit Factor        : {str(metrics['profit_factor']):>10s}")
        print(f"  VaR 95% (daily)      : {metrics['var_95_pct']:>10.3f}%")
        print(f"  CVaR 95% (daily)     : {metrics['cvar_95_pct']:>10.3f}%")
        print(f"  Trading Days         : {metrics['num_trading_days']:>10d}")
        print()

        if trade_log:
            buys = sum(1 for t in trade_log if t["action"] == "BUY")
            sells = sum(1 for t in trade_log if t["action"] == "SELL")
            print(f"  TRADE SUMMARY: {len(trade_log)} rebalance actions ({buys} buys, {sells} sells)")
            print()

    @staticmethod
    def print_comparison(all_results: list[dict]) -> None:
        """Print side-by-side comparison of multiple strategies."""
        if len(all_results) < 2:
            return

        print(f"\n{'='*100}")
        print(f"  STRATEGY COMPARISON")
        print(f"{'='*100}")

        header = (
            f"{'Strategy':<22} {'Return%':>9} {'Sharpe':>8} {'Calmar':>8} "
            f"{'Sortino':>8} {'MaxDD%':>8} {'Vol%':>8} {'Win%':>8} {'PF':>8}"
        )
        print(header)
        print("-" * 100)

        for r in all_results:
            m = r["metrics"]
            pf = m["profit_factor"]
            pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
            print(
                f"{r['name']:<22} {m['annualized_return_pct']:>8.2f}% {m['sharpe_ratio']:>8.3f} "
                f"{m['calmar_ratio']:>8.3f} {m['sortino_ratio']:>8.3f} "
                f"{m['max_drawdown_pct']:>8.2f}% {m['volatility_ann_pct']:>7.2f}% "
                f"{m['win_rate_pct']:>7.2f}% {pf_str:>8}"
            )
        print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Top-level orchestrator: load config → validate → run → evaluate → report."""

    def __init__(self, config_path: str = CONFIG_PATH, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose

    def run_all(self) -> list[dict]:
        """Run all strategies from the config file. Returns list of result dicts."""
        strategies = StrategyConfig.load(self.config_path)
        results = []
        for i, strat in enumerate(strategies):
            valid, msg = StrategyConfig.validate(strat)
            if not valid:
                print(f"  [{strat.get('name', f'#{i}')}] SKIP — {msg}")
                continue
            result = self.run_one(strat)
            results.append(result)
        if len(results) > 1:
            ReportGenerator.print_comparison(results)
        return results

    def run_one(self, strategy: dict) -> dict:
        """Run a single strategy. Returns {name, metrics, trade_log}."""
        if self.verbose:
            print(f"\n▶ Running: {strategy['name']}")

        runner = BacktestRunner(strategy, verbose=self.verbose)
        returns = runner.run()

        # Benchmark returns
        bench_rets = None
        if runner.benchmark is not None:
            bench_rets = runner.benchmark.pct_change().dropna()
            bench_rets = bench_rets.loc[returns.index.intersection(bench_rets.index)]

        evaluator = EvaluationMetrics(returns, bench_rets)
        metrics = evaluator.compute_all()

        ReportGenerator.print_report(strategy, metrics, runner.trade_log)
        return {"name": strategy["name"], "metrics": metrics, "trade_log": runner.trade_log}

    def run_by_indices(self, indices: list[int]) -> list[dict]:
        """Run specific strategies by their 0-based indices in the config."""
        strategies = StrategyConfig.load(self.config_path)
        results = []
        for idx in indices:
            if idx < 0 or idx >= len(strategies):
                print(f"  Index {idx} out of range (0-{len(strategies)-1})")
                continue
            strat = strategies[idx]
            valid, msg = StrategyConfig.validate(strat)
            if not valid:
                print(f"  [{strat['name']}] SKIP — {msg}")
                continue
            results.append(self.run_one(strat))
        if len(results) > 1:
            ReportGenerator.print_comparison(results)
        return results


# BacktestEngine is a pure library module.
# Import and use from Paper_Trading_Agent.py or other scripts.

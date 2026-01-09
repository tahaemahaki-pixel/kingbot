"""
Breakout Optimized Strategy

Entry: Break above swing high + price above upper EVWMA(20) band
Exit: ATR(14) * 2.0 trailing stop
Filters: Volume spike (2x avg), Volume imbalance (10% threshold) - toggleable

Based on backtest results: 69.4% win rate, 4.81 PF with volume filters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BreakoutSignalType(Enum):
    BREAKOUT_LONG = "breakout_long"


class BreakoutStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    STOPPED = "stopped"


@dataclass
class BreakoutSignal:
    """Signal generated when breakout conditions are met."""
    signal_type: BreakoutSignalType
    status: BreakoutStatus
    symbol: str
    setup_key: str
    entry_price: float          # Swing high level (breakout price)
    initial_stop: float         # Entry - ATR * multiplier
    trailing_stop: float        # Updated as price moves
    take_profit: float          # Emergency TP (circuit breaker)
    highest_since_entry: float  # Track highest price for trailing
    vol_ratio: float            # Volume ratio at signal
    imbalance: float            # Volume imbalance at signal
    created_idx: int            # Candle index when created
    created_time: int           # Timestamp when created

    @property
    def direction(self) -> str:
        return "long"

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def to_dict(self) -> dict:
        """Convert signal to dict for JSON serialization."""
        return {
            "signal_type": self.signal_type.value,
            "status": self.status.value,
            "symbol": self.symbol,
            "setup_key": self.setup_key,
            "entry_price": self.entry_price,
            "initial_stop": self.initial_stop,
            "trailing_stop": self.trailing_stop,
            "take_profit": self.take_profit,
            "highest_since_entry": self.highest_since_entry,
            "vol_ratio": self.vol_ratio,
            "imbalance": self.imbalance,
            "created_idx": self.created_idx,
            "created_time": self.created_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BreakoutSignal":
        """Create signal from dict (JSON deserialization)."""
        return cls(
            signal_type=BreakoutSignalType(data["signal_type"]),
            status=BreakoutStatus(data["status"]),
            symbol=data["symbol"],
            setup_key=data["setup_key"],
            entry_price=data["entry_price"],
            initial_stop=data["initial_stop"],
            trailing_stop=data["trailing_stop"],
            take_profit=data["take_profit"],
            highest_since_entry=data["highest_since_entry"],
            vol_ratio=data["vol_ratio"],
            imbalance=data["imbalance"],
            created_idx=data["created_idx"],
            created_time=data["created_time"],
        )


class BreakoutIndicators:
    """Static methods for calculating all required indicators."""

    @staticmethod
    def calculate_evwma(prices: np.ndarray, volumes: np.ndarray, length: int) -> np.ndarray:
        """
        Elastic Volume Weighted Moving Average.
        Gives more weight to price moves on higher volume.
        """
        n = len(prices)
        evwma = np.zeros(n)

        for i in range(n):
            start_idx = max(0, i - length + 1)
            nbfs = np.sum(volumes[start_idx:i+1])
            if nbfs == 0:
                nbfs = 1
            vol = volumes[i] if volumes[i] > 0 else 1

            if i == 0:
                evwma[i] = prices[i]
            else:
                evwma[i] = evwma[i-1] * (nbfs - vol) / nbfs + (vol * prices[i] / nbfs)

        return evwma

    @staticmethod
    def calculate_evwma_bands(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                               volumes: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate EVWMA bands.
        Returns: (middle, upper, lower)
        """
        volumes = np.where(volumes <= 0, 1, volumes)

        evwma_mid = BreakoutIndicators.calculate_evwma(closes, volumes, length)
        evwma_upper = BreakoutIndicators.calculate_evwma(highs, volumes, length)
        evwma_lower = BreakoutIndicators.calculate_evwma(lows, volumes, length)

        return evwma_mid, evwma_upper, evwma_lower

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """
        Average True Range.
        Measures volatility for dynamic stop placement.
        """
        n = len(highs)
        tr = np.zeros(n)
        atr = np.full(n, np.nan)

        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        for i in range(period - 1, n):
            if i == period - 1:
                atr[i] = np.mean(tr[:period])
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def find_pivot_highs(highs: np.ndarray, left: int, right: int) -> np.ndarray:
        """Find swing highs (pivot highs)."""
        n = len(highs)
        pivots = np.full(n, np.nan)

        for i in range(left, n - right):
            is_pivot = True
            current = highs[i]

            for j in range(1, left + 1):
                if highs[i - j] >= current:
                    is_pivot = False
                    break

            if is_pivot:
                for j in range(1, right + 1):
                    if highs[i + j] >= current:
                        is_pivot = False
                        break

            if is_pivot:
                pivots[i] = current

        return pivots

    @staticmethod
    def find_pivot_lows(lows: np.ndarray, left: int, right: int) -> np.ndarray:
        """Find swing lows (pivot lows)."""
        n = len(lows)
        pivots = np.full(n, np.nan)

        for i in range(left, n - right):
            is_pivot = True
            current = lows[i]

            for j in range(1, left + 1):
                if lows[i - j] <= current:
                    is_pivot = False
                    break

            if is_pivot:
                for j in range(1, right + 1):
                    if lows[i + j] <= current:
                        is_pivot = False
                        break

            if is_pivot:
                pivots[i] = current

        return pivots

    @staticmethod
    def get_current_swing_high(pivot_highs: np.ndarray, highs: np.ndarray, pivot_right: int) -> float:
        """Get the most recent confirmed swing high level."""
        n = len(pivot_highs)

        # Look back from the confirmation point (pivot_right bars ago)
        for i in range(n - pivot_right - 1, -1, -1):
            if not np.isnan(pivot_highs[i]):
                return highs[i]

        return np.nan

    @staticmethod
    def calculate_volume_ratio(volumes: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate volume ratio (current volume / average volume).
        Used to identify volume spikes on breakouts.
        """
        n = len(volumes)
        ratio = np.full(n, np.nan)

        for i in range(period, n):
            avg_vol = np.mean(volumes[i - period:i])
            if avg_vol > 0:
                ratio[i] = volumes[i] / avg_vol
            else:
                ratio[i] = 1.0

        return ratio

    @staticmethod
    def calculate_volume_imbalance(opens: np.ndarray, closes: np.ndarray,
                                    volumes: np.ndarray, lookback: int) -> np.ndarray:
        """
        Calculate volume imbalance (buy vs sell pressure).

        Approximation:
        - If close > open: volume is "buy" volume
        - If close < open: volume is "sell" volume

        Imbalance = (buy_vol - sell_vol) / total_vol over lookback period

        Returns: Array of imbalance values (-1 to +1)
        """
        n = len(opens)
        imbalance = np.full(n, np.nan)

        for i in range(lookback - 1, n):
            buy_vol = 0.0
            sell_vol = 0.0

            for j in range(i - lookback + 1, i + 1):
                if closes[j] > opens[j]:
                    buy_vol += volumes[j]
                elif closes[j] < opens[j]:
                    sell_vol += volumes[j]
                else:
                    # Doji - split volume
                    buy_vol += volumes[j] / 2
                    sell_vol += volumes[j] / 2

            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                imbalance[i] = (buy_vol - sell_vol) / total_vol
            else:
                imbalance[i] = 0.0

        return imbalance

    @staticmethod
    def calculate_all(opens: np.ndarray, closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, volumes: np.ndarray, config) -> Dict:
        """
        Calculate all indicators at once.
        Returns dict with all indicator arrays.
        """
        evwma_mid, evwma_upper, evwma_lower = BreakoutIndicators.calculate_evwma_bands(
            closes, highs, lows, volumes, config.evwma_period
        )

        atr = BreakoutIndicators.calculate_atr(highs, lows, closes, config.atr_period)

        pivot_highs = BreakoutIndicators.find_pivot_highs(highs, config.pivot_left, config.pivot_right)
        pivot_lows = BreakoutIndicators.find_pivot_lows(lows, config.pivot_left, config.pivot_right)

        vol_ratio = BreakoutIndicators.calculate_volume_ratio(volumes, config.volume_avg_period)
        vol_imbalance = BreakoutIndicators.calculate_volume_imbalance(
            opens, closes, volumes, config.imbalance_lookback
        )

        return {
            'evwma_mid': evwma_mid,
            'evwma_upper': evwma_upper,
            'evwma_lower': evwma_lower,
            'atr': atr,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows,
            'vol_ratio': vol_ratio,
            'vol_imbalance': vol_imbalance,
        }


class BreakoutStrategy:
    """
    Breakout Trend Follower Strategy.

    Entry: Break above swing high + price above upper EVWMA band
    Exit: ATR trailing stop
    """

    def __init__(self, symbol: str, timeframe: str, config):
        """
        Initialize strategy.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Candle timeframe (e.g., "5" for 5-minute)
            config: BreakoutConfig instance
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config

        # Track last swing high to avoid duplicate signals
        self._last_buy_level: Optional[float] = None

    def scan_for_signals(self, opens: np.ndarray, closes: np.ndarray,
                         highs: np.ndarray, lows: np.ndarray,
                         volumes: np.ndarray, times: np.ndarray) -> Optional[BreakoutSignal]:
        """
        Scan for breakout signals on the latest candle.

        Args:
            opens, closes, highs, lows, volumes: OHLCV arrays
            times: Timestamp array

        Returns:
            BreakoutSignal if conditions met, None otherwise
        """
        n = len(closes)
        if n < 50:  # Need enough data
            return None

        # Calculate all indicators
        indicators = BreakoutIndicators.calculate_all(
            opens, closes, highs, lows, volumes, self.config
        )

        # Get current bar index (last confirmed candle)
        idx = n - 1

        # Get current swing high level
        buy_level = BreakoutIndicators.get_current_swing_high(
            indicators['pivot_highs'], highs, self.config.pivot_right
        )

        if np.isnan(buy_level):
            return None

        # Skip if we already signaled on this level
        if self._last_buy_level is not None and abs(buy_level - self._last_buy_level) < 0.0001:
            return None

        atr = indicators['atr'][idx]
        if np.isnan(atr) or atr <= 0:
            return None

        # === ENTRY CONDITIONS ===

        # 1. Price breaks above swing high
        if highs[idx] <= buy_level:
            return None

        # 2. Close above upper EVWMA band (strong uptrend)
        if closes[idx] <= indicators['evwma_upper'][idx]:
            return None

        # 3. Volume spike filter (optional)
        vol_ratio = indicators['vol_ratio'][idx]
        if self.config.use_volume_filter:
            if np.isnan(vol_ratio) or vol_ratio < self.config.min_vol_ratio:
                logger.debug(f"{self.symbol}: Volume filter failed ({vol_ratio:.2f}x < {self.config.min_vol_ratio}x)")
                return None

        # 4. Volume imbalance filter (optional)
        vol_imbalance = indicators['vol_imbalance'][idx]
        if self.config.use_imbalance_filter:
            if np.isnan(vol_imbalance) or vol_imbalance < self.config.imbalance_threshold:
                logger.debug(f"{self.symbol}: Imbalance filter failed ({vol_imbalance:.2f} < {self.config.imbalance_threshold})")
                return None

        # === ALL CONDITIONS MET - CREATE SIGNAL ===

        entry_price = buy_level
        initial_stop = entry_price - (atr * self.config.atr_multiplier)

        # Ensure valid risk
        if initial_stop >= entry_price:
            return None

        # Calculate emergency take profit (circuit breaker)
        risk = entry_price - initial_stop
        emergency_tp_mult = getattr(self.config, 'emergency_tp_multiplier', 10.0)
        take_profit = entry_price + (risk * emergency_tp_mult)

        # Update last buy level to avoid duplicate signals
        self._last_buy_level = buy_level

        setup_key = f"{self.symbol}_{self.timeframe}"

        signal = BreakoutSignal(
            signal_type=BreakoutSignalType.BREAKOUT_LONG,
            status=BreakoutStatus.PENDING,
            symbol=self.symbol,
            setup_key=setup_key,
            entry_price=entry_price,
            initial_stop=initial_stop,
            trailing_stop=initial_stop,
            take_profit=take_profit,
            highest_since_entry=highs[idx],
            vol_ratio=vol_ratio if not np.isnan(vol_ratio) else 0.0,
            imbalance=vol_imbalance if not np.isnan(vol_imbalance) else 0.0,
            created_idx=idx,
            created_time=int(times[idx]) if len(times) > idx else 0,
        )

        logger.info(
            f"BREAKOUT SIGNAL: {self.symbol} | "
            f"Entry: {entry_price:.4f} | Stop: {initial_stop:.4f} | TP: {take_profit:.4f} | "
            f"Risk: {signal.risk:.4f} | Vol: {vol_ratio:.2f}x | Imb: {vol_imbalance:.2f}"
        )

        return signal

    def update_trailing_stop(self, signal: BreakoutSignal,
                              current_high: float, current_atr: float) -> float:
        """
        Update trailing stop based on new high and current ATR.

        The stop trails up as price makes new highs, but never moves down.

        Args:
            signal: The active signal to update
            current_high: Current candle high
            current_atr: Current ATR value

        Returns:
            New trailing stop value (may be same as current if no update needed)
        """
        # Update highest since entry if we have a new high
        if current_high > signal.highest_since_entry:
            signal.highest_since_entry = current_high

            # Calculate new potential stop
            new_stop = signal.highest_since_entry - (current_atr * self.config.atr_multiplier)

            # Only move stop up, never down
            if new_stop > signal.trailing_stop:
                logger.debug(
                    f"{self.symbol}: Trailing stop updated {signal.trailing_stop:.4f} -> {new_stop:.4f}"
                )
                return new_stop

        return signal.trailing_stop

    def check_stop_hit(self, signal: BreakoutSignal, current_low: float) -> bool:
        """
        Check if trailing stop has been hit.

        Args:
            signal: The active signal
            current_low: Current candle low

        Returns:
            True if stop was hit, False otherwise
        """
        return current_low <= signal.trailing_stop


# === TEST MODE ===

if __name__ == "__main__":
    import sys
    import pandas as pd
    from pathlib import Path

    # Simple test config
    @dataclass
    class TestConfig:
        pivot_left: int = 3
        pivot_right: int = 3
        evwma_period: int = 20
        atr_period: int = 14
        atr_multiplier: float = 2.0
        use_volume_filter: bool = True
        min_vol_ratio: float = 2.0
        volume_avg_period: int = 20
        use_imbalance_filter: bool = True
        imbalance_threshold: float = 0.10
        imbalance_lookback: int = 10
        emergency_tp_multiplier: float = 10.0

    # Test on historical data
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    data_dir = Path("/home/tahae/ai-content/data/Tradingdata/volume-charts")

    # Find data file
    patterns = [
        f"BYBIT_{symbol}.P, 5_*.csv",
        f"*{symbol}*5m*.csv",
        f"*{symbol}*.csv",
    ]

    data_file = None
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            data_file = files[0]
            break

    if not data_file:
        print(f"No data file found for {symbol}")
        sys.exit(1)

    print(f"Testing {symbol} with data from {data_file.name}")

    df = pd.read_csv(data_file)
    df.columns = df.columns.str.lower().str.strip()

    config = TestConfig()
    strategy = BreakoutStrategy(symbol, "5", config)

    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    times = df['time'].values if 'time' in df.columns else np.arange(len(df))

    # Scan through data
    signals_found = 0
    for i in range(100, len(df)):
        signal = strategy.scan_for_signals(
            opens[:i+1], closes[:i+1], highs[:i+1],
            lows[:i+1], volumes[:i+1], times[:i+1]
        )
        if signal:
            signals_found += 1

    print(f"\nFound {signals_found} signals in {len(df)} candles")
    print(f"Signal rate: {signals_found / len(df) * 100:.2f}%")

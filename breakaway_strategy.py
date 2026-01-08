"""
Breakaway Strategy - FVG Trading with Volume Delta Imbalance Filter

Entry conditions for SHORTS (ALL required):
1. Bearish FVG forms (current_high < 2_candles_ago_low)
2. Price cradled 3+ of last 5 candles within EWVMA(20) bands
3. Volume spike >= 2.0x (20-period average)
4. Volume Delta Imbalance < -0.10 (selling pressure)

Entry conditions for LONGS (ALL required):
1. Bullish FVG forms (current_low > 2_candles_ago_high)
2. Price cradled 3+ of last 5 candles within EWVMA(20) bands
3. Volume spike >= 2.0x (20-period average)
4. Volume Delta Imbalance > +0.10 (buying pressure)

Exit:
- SL: FVG boundary + 0.1% buffer
- TP: 3:1 R:R

Aggressive Mode Update (2026-01-08):
- Replaced Tai Index and EWVMA-200 trend filters with Volume Delta Imbalance
- Backtest showed: 3139 trades, 59.3% WR, 1.36R expectancy, 4269R total
- +26% improvement in expectancy vs old filters
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


class BreakawaySignalType(Enum):
    BREAKAWAY_SHORT = "breakaway_short"
    BREAKAWAY_LONG = "breakaway_long"


class BreakawayStatus(Enum):
    READY = "ready"          # Can execute immediately
    FILLED = "filled"        # Order filled
    EXPIRED = "expired"      # Signal expired


@dataclass
class BreakawaySignal:
    """Represents a Breakaway trading signal."""
    signal_type: BreakawaySignalType
    status: BreakawayStatus
    symbol: str
    setup_key: str  # "BTCUSDT_5"

    # Entry levels
    entry_price: float   # FVG boundary
    stop_loss: float     # FVG other boundary + buffer
    target: float        # 3:1 R:R

    # FVG info
    fvg_top: float
    fvg_bottom: float

    # Context at signal
    ewvma_dist: float    # % distance from EWVMA-200
    vol_ratio: float     # Volume spike ratio
    tai_index: float     # Tai value at signal
    cradle_count: int    # Candles cradled
    imbalance: float     # Volume delta imbalance at signal

    # Timing
    created_idx: int     # Candle index when created
    created_time: int    # Timestamp

    @property
    def direction(self) -> str:
        return "short" if self.signal_type == BreakawaySignalType.BREAKAWAY_SHORT else "long"

    @property
    def risk(self) -> float:
        return abs(self.stop_loss - self.entry_price)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0


class BreakawayIndicators:
    """Calculate Breakaway strategy indicators."""

    def __init__(self, ewvma_length: int = 20, ewvma_trend_length: int = 200,
                 vol_lookback: int = 20, rsi_length: int = 100, stoch_length: int = 200):
        self.ewvma_length = ewvma_length
        self.ewvma_trend_length = ewvma_trend_length
        self.vol_lookback = vol_lookback
        self.rsi_length = rsi_length
        self.stoch_length = stoch_length

        # State for incremental calculation
        self._ewvma_20_prev = None
        self._ewvma_20_std_buffer = []
        self._ewvma_200_prev = None
        self._vol_sum_20 = []
        self._vol_sum_200 = []
        self._vol_buffer = []

        # RSI/Tai state
        self._avg_gain = None
        self._avg_loss = None
        self._rsi_buffer = []

    def calculate_ewvma(self, closes: np.ndarray, volumes: np.ndarray, length: int) -> np.ndarray:
        """Calculate EWVMA (Exponentially Weighted Volume Moving Average)."""
        n = len(closes)
        ewvma = np.zeros(n)

        if n == 0:
            return ewvma

        alpha = 2 / (length + 1)

        vol_sum = 0
        pv_sum = 0

        for i in range(n):
            vol = volumes[i] if volumes[i] > 0 else 1

            if i == 0:
                ewvma[i] = closes[i]
                vol_sum = vol
                pv_sum = closes[i] * vol
            else:
                vol_sum = alpha * vol + (1 - alpha) * vol_sum
                pv_sum = alpha * (closes[i] * vol) + (1 - alpha) * pv_sum
                ewvma[i] = pv_sum / vol_sum if vol_sum > 0 else closes[i]

        return ewvma

    def calculate_ewvma_std(self, closes: np.ndarray, ewvma: np.ndarray, length: int = 20) -> np.ndarray:
        """Calculate rolling standard deviation for EWVMA bands."""
        n = len(closes)
        std = np.zeros(n)

        for i in range(length - 1, n):
            window = closes[i - length + 1:i + 1]
            std[i] = np.std(window)

        return std

    def calculate_tai_index(self, closes: np.ndarray) -> np.ndarray:
        """
        Calculate Tai Index (Stochastic RSI).
        RSI(100) with Stochastic(200) applied.
        """
        n = len(closes)
        tai = np.full(n, 50.0)  # Default to neutral

        if n < self.rsi_length + self.stoch_length:
            return tai

        # Calculate RSI(100)
        delta = np.diff(closes, prepend=closes[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        alpha = 2 / (self.rsi_length + 1)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        avg_gain[0] = gains[0]
        avg_loss[0] = losses[0]

        for i in range(1, n):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))

        # Apply Stochastic(200) to RSI
        for i in range(self.stoch_length - 1, n):
            rsi_window = rsi[i - self.stoch_length + 1:i + 1]
            rsi_min = np.min(rsi_window)
            rsi_max = np.max(rsi_window)

            if rsi_max - rsi_min > 0:
                tai[i] = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
            else:
                tai[i] = 50

        return tai

    def calculate_volume_ratio(self, volumes: np.ndarray, lookback: int = 20) -> np.ndarray:
        """Calculate volume ratio (current / SMA)."""
        n = len(volumes)
        vol_ratio = np.ones(n)

        for i in range(lookback - 1, n):
            vol_sma = np.mean(volumes[i - lookback + 1:i + 1])
            if vol_sma > 0:
                vol_ratio[i] = volumes[i] / vol_sma

        return vol_ratio

    def calculate_volume_delta_imbalance(self, opens: np.ndarray, closes: np.ndarray,
                                          volumes: np.ndarray, lookback: int = 10) -> np.ndarray:
        """
        Calculate order book imbalance proxy using volume delta.
        - Bullish candle (close > open): volume = buy pressure
        - Bearish candle (close < open): volume = sell pressure
        Returns: -1.0 (all sells) to +1.0 (all buys)
        """
        n = len(closes)
        imbalance = np.zeros(n)

        is_bullish = closes > opens
        buy_volume = np.where(is_bullish, volumes, 0)
        sell_volume = np.where(~is_bullish, volumes, 0)

        buy_cumsum = np.cumsum(buy_volume)
        sell_cumsum = np.cumsum(sell_volume)

        for i in range(lookback, n):
            buy_sum = buy_cumsum[i] - buy_cumsum[i - lookback]
            sell_sum = sell_cumsum[i] - sell_cumsum[i - lookback]
            total = buy_sum + sell_sum
            if total > 0:
                imbalance[i] = (buy_sum - sell_sum) / total

        return imbalance

    def calculate_all(self, opens: np.ndarray, closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, volumes: np.ndarray,
                      imbalance_lookback: int = 10) -> Dict[str, np.ndarray]:
        """Calculate all indicators."""
        # EWVMA-20 (cradle bands)
        ewvma_20 = self.calculate_ewvma(closes, volumes, self.ewvma_length)
        ewvma_20_std = self.calculate_ewvma_std(closes, ewvma_20, self.ewvma_length)

        # EWVMA-200 (trend)
        ewvma_200 = self.calculate_ewvma(closes, volumes, self.ewvma_trend_length)

        # Tai Index
        tai_index = self.calculate_tai_index(closes)

        # Volume ratio
        vol_ratio = self.calculate_volume_ratio(volumes, self.vol_lookback)

        # Volume delta imbalance
        imbalance = self.calculate_volume_delta_imbalance(opens, closes, volumes, imbalance_lookback)

        # Cradle detection (close within bands)
        upper_band = ewvma_20 + ewvma_20_std
        lower_band = ewvma_20 - ewvma_20_std
        in_cradle = (closes >= lower_band) & (closes <= upper_band)

        return {
            'ewvma_20': ewvma_20,
            'ewvma_20_std': ewvma_20_std,
            'ewvma_20_upper': upper_band,
            'ewvma_20_lower': lower_band,
            'ewvma_200': ewvma_200,
            'tai_index': tai_index,
            'vol_ratio': vol_ratio,
            'imbalance': imbalance,
            'in_cradle': in_cradle,
        }


class BreakawayStrategy:
    """
    Breakaway Strategy - FVG trading with Volume Delta Imbalance filter.

    Scans for bearish/bullish FVGs from EWVMA cradle consolidation
    with volume confirmation and imbalance filter.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "5",
        min_vol_ratio: float = 2.0,
        # NEW: Imbalance filter (replaces Tai/Trend)
        use_imbalance_filter: bool = True,
        imbalance_threshold: float = 0.10,
        imbalance_lookback: int = 10,
        # OLD: Tai/Trend filters (optional, default OFF)
        use_tai_filter: bool = False,
        tai_threshold_short: float = 53.0,
        tai_threshold_long: float = 47.0,
        use_trend_filter: bool = False,
        # Unchanged
        min_cradle_candles: int = 3,
        cradle_lookback: int = 5,
        risk_reward: float = 3.0,
        sl_buffer_pct: float = 0.001,
        trade_direction: str = "both",  # "both", "shorts", "longs"
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.setup_key = f"{symbol}_{timeframe}"

        # Strategy parameters
        self.min_vol_ratio = min_vol_ratio

        # NEW: Imbalance filter
        self.use_imbalance_filter = use_imbalance_filter
        self.imbalance_threshold = imbalance_threshold
        self.imbalance_lookback = imbalance_lookback

        # OLD: Tai/Trend filters (optional)
        self.use_tai_filter = use_tai_filter
        self.tai_threshold_short = tai_threshold_short
        self.tai_threshold_long = tai_threshold_long
        self.use_trend_filter = use_trend_filter

        # Unchanged
        self.min_cradle_candles = min_cradle_candles
        self.cradle_lookback = cradle_lookback
        self.risk_reward = risk_reward
        self.sl_buffer_pct = sl_buffer_pct
        self.trade_direction = trade_direction

        # Indicators calculator
        self.indicators = BreakawayIndicators()

        # State
        self.active_signal: Optional[BreakawaySignal] = None
        self._last_signal_idx = -1

        # Cached indicator values (updated on each scan)
        self._ind_cache: Optional[Dict[str, np.ndarray]] = None

    def calculate_indicators(self, opens: np.ndarray, closes: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, volumes: np.ndarray):
        """Calculate and cache all indicators."""
        self._ind_cache = self.indicators.calculate_all(
            opens, closes, highs, lows, volumes, self.imbalance_lookback
        )

    def _detect_bearish_fvg(self, highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[dict]:
        """
        Detect bearish FVG (gap down).
        Condition: current_high < 2_candles_ago_low
        """
        if idx < 2:
            return None

        if highs[idx] < lows[idx - 2]:
            return {
                'top': lows[idx - 2],      # FVG top (resistance)
                'bottom': highs[idx],       # FVG bottom (entry)
            }
        return None

    def _detect_bullish_fvg(self, highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[dict]:
        """
        Detect bullish FVG (gap up).
        Condition: current_low > 2_candles_ago_high
        """
        if idx < 2:
            return None

        if lows[idx] > highs[idx - 2]:
            return {
                'top': lows[idx],           # FVG top (entry)
                'bottom': highs[idx - 2],   # FVG bottom (support)
            }
        return None

    def _check_cradle(self, idx: int) -> Tuple[bool, int]:
        """Check if price was cradled in EWVMA bands."""
        if self._ind_cache is None:
            return False, 0

        in_cradle = self._ind_cache['in_cradle']

        if idx < self.cradle_lookback:
            return False, 0

        cradle_count = np.sum(in_cradle[idx - self.cradle_lookback:idx])
        return cradle_count >= self.min_cradle_candles, int(cradle_count)

    def _check_short_filters(self, closes: np.ndarray, idx: int) -> Tuple[bool, dict]:
        """Check all short entry filters."""
        if self._ind_cache is None:
            return False, {}

        vol_ratio = self._ind_cache['vol_ratio'][idx]
        tai_index = self._ind_cache['tai_index'][idx]
        ewvma_200 = self._ind_cache['ewvma_200'][idx]
        imbalance = self._ind_cache['imbalance'][idx]

        # Volume spike (always required)
        if vol_ratio < self.min_vol_ratio:
            return False, {}

        # NEW: Imbalance filter - require selling pressure
        if self.use_imbalance_filter:
            if imbalance > -self.imbalance_threshold:
                return False, {}

        # OLD: Tai Index overbought (optional, default OFF)
        if self.use_tai_filter:
            if tai_index <= self.tai_threshold_short:
                return False, {}

        # OLD: Counter-trend (optional, default OFF)
        if self.use_trend_filter:
            if closes[idx] <= ewvma_200:
                return False, {}

        # Cradle check (always required)
        was_cradled, cradle_count = self._check_cradle(idx)
        if not was_cradled:
            return False, {}

        return True, {
            'vol_ratio': vol_ratio,
            'tai_index': tai_index,
            'ewvma_200': ewvma_200,
            'cradle_count': cradle_count,
            'ewvma_dist': (closes[idx] - ewvma_200) / ewvma_200 * 100,
            'imbalance': imbalance,
        }

    def _check_long_filters(self, closes: np.ndarray, idx: int) -> Tuple[bool, dict]:
        """Check all long entry filters."""
        if self._ind_cache is None:
            return False, {}

        vol_ratio = self._ind_cache['vol_ratio'][idx]
        tai_index = self._ind_cache['tai_index'][idx]
        ewvma_200 = self._ind_cache['ewvma_200'][idx]
        imbalance = self._ind_cache['imbalance'][idx]

        # Volume spike (always required)
        if vol_ratio < self.min_vol_ratio:
            return False, {}

        # NEW: Imbalance filter - require buying pressure
        if self.use_imbalance_filter:
            if imbalance < self.imbalance_threshold:
                return False, {}

        # OLD: Tai Index oversold (optional, default OFF)
        if self.use_tai_filter:
            if tai_index >= self.tai_threshold_long:
                return False, {}

        # OLD: Counter-trend (optional, default OFF)
        if self.use_trend_filter:
            if closes[idx] >= ewvma_200:
                return False, {}

        # Cradle check (always required)
        was_cradled, cradle_count = self._check_cradle(idx)
        if not was_cradled:
            return False, {}

        return True, {
            'vol_ratio': vol_ratio,
            'tai_index': tai_index,
            'ewvma_200': ewvma_200,
            'cradle_count': cradle_count,
            'ewvma_dist': (closes[idx] - ewvma_200) / ewvma_200 * 100,
            'imbalance': imbalance,
        }

    def scan_for_signals(self, opens: np.ndarray, closes: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, volumes: np.ndarray,
                        times: np.ndarray) -> Optional[BreakawaySignal]:
        """
        Scan for new Breakaway signals on the latest candle.
        Returns signal if found, None otherwise.
        """
        n = len(closes)
        if n < 300:  # Require minimum history
            return None

        idx = n - 1  # Latest candle

        # Don't signal on same candle twice
        if idx == self._last_signal_idx:
            return None

        # Calculate indicators if needed
        if self._ind_cache is None or len(self._ind_cache['ewvma_20']) != n:
            self.calculate_indicators(opens, closes, highs, lows, volumes)

        signal = None

        # Check for SHORT signal
        if self.trade_direction in ["both", "shorts"]:
            fvg = self._detect_bearish_fvg(highs, lows, idx)
            if fvg is not None:
                passed, context = self._check_short_filters(closes, idx)
                if passed:
                    entry = fvg['bottom']
                    sl = fvg['top'] * (1 + self.sl_buffer_pct)
                    risk = sl - entry
                    tp = entry - (risk * self.risk_reward)

                    signal = BreakawaySignal(
                        signal_type=BreakawaySignalType.BREAKAWAY_SHORT,
                        status=BreakawayStatus.READY,
                        symbol=self.symbol,
                        setup_key=self.setup_key,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        fvg_top=fvg['top'],
                        fvg_bottom=fvg['bottom'],
                        ewvma_dist=context['ewvma_dist'],
                        vol_ratio=context['vol_ratio'],
                        tai_index=context['tai_index'],
                        cradle_count=context['cradle_count'],
                        imbalance=context['imbalance'],
                        created_idx=idx,
                        created_time=int(times[idx]),
                    )

        # Check for LONG signal (only if no short signal)
        if signal is None and self.trade_direction in ["both", "longs"]:
            fvg = self._detect_bullish_fvg(highs, lows, idx)
            if fvg is not None:
                passed, context = self._check_long_filters(closes, idx)
                if passed:
                    entry = fvg['top']
                    sl = fvg['bottom'] * (1 - self.sl_buffer_pct)
                    risk = entry - sl
                    tp = entry + (risk * self.risk_reward)

                    signal = BreakawaySignal(
                        signal_type=BreakawaySignalType.BREAKAWAY_LONG,
                        status=BreakawayStatus.READY,
                        symbol=self.symbol,
                        setup_key=self.setup_key,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        fvg_top=fvg['top'],
                        fvg_bottom=fvg['bottom'],
                        ewvma_dist=context['ewvma_dist'],
                        vol_ratio=context['vol_ratio'],
                        tai_index=context['tai_index'],
                        cradle_count=context['cradle_count'],
                        imbalance=context['imbalance'],
                        created_idx=idx,
                        created_time=int(times[idx]),
                    )

        if signal:
            self._last_signal_idx = idx
            self.active_signal = signal

        return signal

    def clear_signal(self):
        """Clear the active signal."""
        self.active_signal = None

    def get_signal_info(self) -> Optional[str]:
        """Get formatted signal info for logging."""
        if not self.active_signal:
            return None

        s = self.active_signal
        return (
            f"[{s.setup_key}] {s.direction.upper()}: "
            f"Entry={s.entry_price:.4f} SL={s.stop_loss:.4f} TP={s.target:.4f} "
            f"R:R={s.rr_ratio:.1f} Vol={s.vol_ratio:.1f}x Imb={s.imbalance:+.2f}"
        )


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    from pathlib import Path

    # Load sample data
    data_path = Path("/home/tahae/ai-content/data/Tradingdata/volume charts/BTCUSDT_5m_merged.csv")

    if data_path.exists():
        df = pd.read_csv(data_path)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'time'}, inplace=True)

        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        times = df['time'].values if 'time' in df.columns else np.arange(len(df))

        # Create strategy with new aggressive settings
        strategy = BreakawayStrategy(
            symbol="BTCUSDT",
            timeframe="5",
            min_vol_ratio=2.0,
            use_imbalance_filter=True,
            imbalance_threshold=0.10,
            use_tai_filter=False,
            use_trend_filter=False,
            trade_direction="both"
        )

        # Scan historical data for signals
        signals_found = 0
        shorts = 0
        longs = 0

        for i in range(300, len(closes)):
            signal = strategy.scan_for_signals(
                opens[:i+1], closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1], times[:i+1]
            )
            if signal:
                signals_found += 1
                if signal.signal_type == BreakawaySignalType.BREAKAWAY_SHORT:
                    shorts += 1
                else:
                    longs += 1
                strategy.clear_signal()

        print(f"\n=== Breakaway Strategy Test ===")
        print(f"Data: {len(closes)} candles")
        print(f"Total signals: {signals_found}")
        print(f"  Shorts: {shorts}")
        print(f"  Longs: {longs}")
    else:
        print(f"Data file not found: {data_path}")

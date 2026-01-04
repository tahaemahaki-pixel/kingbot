"""
Real-time Data Feed with EVWMA Calculation
"""
import pandas as pd
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from bybit_client import BybitClient, BybitWebSocket
from config import BotConfig


@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool = False

    # EVWMA values (calculated) - for King strategy ribbon
    evwma_mid: float = None
    evwma_upper: float = None
    evwma_lower: float = None

    # SMA for trend filter
    sma300: float = None

    # Double Touch: EMA Ribbon (9/21/50)
    ema9: float = None
    ema21: float = None
    ema50: float = None
    band_color: str = None  # 'green', 'red', 'grey'

    # Double Touch: HH/LL detection
    is_hh: bool = False  # Higher High
    is_ll: bool = False  # Lower Low

    # Double Touch: EWVMA-200 for counter-trend filter
    ewvma_200: float = None


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'high' or 'low'
    candle_open: float
    time: int


@dataclass
class FVG:
    index: int
    type: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    time: int


class DataFeed:
    """Manages real-time candle data and indicators for a single symbol."""

    def __init__(self, config: BotConfig, client: BybitClient, symbol: str, timeframe: str = None):
        self.config = config
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe or config.timeframe  # Allow override
        self.candles: List[Candle] = []
        self.max_candles = 500  # Keep last 500 candles
        self.on_new_candle: Optional[Callable] = None

        # EVWMA state (King strategy)
        self.evwma_mid_prev = None
        self.evwma_upper_prev = None
        self.evwma_lower_prev = None
        self.volume_sum = deque(maxlen=config.evwma_length)

        # Double Touch: EMA state for real-time updates
        self.ema9_prev = None
        self.ema21_prev = None
        self.ema50_prev = None

        # Double Touch: EWVMA-200 state
        self.ewvma_200_prev = None
        self.volume_sum_200 = deque(maxlen=200)

    def load_historical(self, limit: int = 200):
        """Load historical candles from API."""
        klines = self.client.get_klines(
            self.symbol,
            self.timeframe,
            limit=limit
        )

        # Bybit returns newest first, reverse it
        klines.reverse()

        self.candles = []
        for k in klines:
            candle = Candle(
                time=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                confirmed=True
            )
            self.candles.append(candle)

        # Calculate EVWMA for all historical candles (King strategy)
        self._calculate_evwma_all()

        # Calculate Double Touch indicators
        self._calculate_ema_ribbon()
        self._calculate_ewvma_200()
        self._detect_hh_ll(lookback=20)

        print(f"Loaded {len(self.candles)} historical candles")

    def _calculate_evwma(self, price: float, volume: float, prev_evwma: float) -> float:
        """Calculate EVWMA for a single value."""
        self.volume_sum.append(volume)
        nbfs = sum(self.volume_sum)

        if nbfs == 0:
            return price

        if prev_evwma is None:
            return price

        return prev_evwma * (nbfs - volume) / nbfs + (volume * price / nbfs)

    def _calculate_evwma_all(self):
        """Calculate EVWMA and SMA300 for all candles."""
        self.volume_sum.clear()
        self.evwma_mid_prev = None
        self.evwma_upper_prev = None
        self.evwma_lower_prev = None

        for i, candle in enumerate(self.candles):
            vol = candle.volume if candle.volume > 0 else 1

            if self.evwma_mid_prev is None:
                candle.evwma_mid = candle.close
                candle.evwma_upper = candle.high
                candle.evwma_lower = candle.low
            else:
                self.volume_sum.append(vol)
                nbfs = sum(self.volume_sum)

                candle.evwma_mid = self.evwma_mid_prev * (nbfs - vol) / nbfs + (vol * candle.close / nbfs)
                candle.evwma_upper = self.evwma_upper_prev * (nbfs - vol) / nbfs + (vol * candle.high / nbfs)
                candle.evwma_lower = self.evwma_lower_prev * (nbfs - vol) / nbfs + (vol * candle.low / nbfs)

            self.evwma_mid_prev = candle.evwma_mid
            self.evwma_upper_prev = candle.evwma_upper
            self.evwma_lower_prev = candle.evwma_lower

            # Calculate 300 SMA
            if i >= 299:
                sma_sum = sum(c.close for c in self.candles[i-299:i+1])
                candle.sma300 = sma_sum / 300
            else:
                candle.sma300 = None

    def update_candle(self, data: Dict):
        """Update from WebSocket kline data."""
        if not self.candles:
            return

        new_candle = Candle(
            time=data["time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            confirmed=data["confirm"]
        )

        # Check if this is an update to current candle or a new candle
        if self.candles and self.candles[-1].time == new_candle.time:
            # Update current candle
            self.candles[-1] = new_candle
        else:
            # New candle - calculate EVWMA
            vol = new_candle.volume if new_candle.volume > 0 else 1

            if self.evwma_mid_prev is not None:
                self.volume_sum.append(vol)
                nbfs = sum(self.volume_sum)

                new_candle.evwma_mid = self.evwma_mid_prev * (nbfs - vol) / nbfs + (vol * new_candle.close / nbfs)
                new_candle.evwma_upper = self.evwma_upper_prev * (nbfs - vol) / nbfs + (vol * new_candle.high / nbfs)
                new_candle.evwma_lower = self.evwma_lower_prev * (nbfs - vol) / nbfs + (vol * new_candle.low / nbfs)
            else:
                new_candle.evwma_mid = new_candle.close
                new_candle.evwma_upper = new_candle.high
                new_candle.evwma_lower = new_candle.low

            self.evwma_mid_prev = new_candle.evwma_mid
            self.evwma_upper_prev = new_candle.evwma_upper
            self.evwma_lower_prev = new_candle.evwma_lower

            # Calculate SMA300 for new candle
            if len(self.candles) >= 299:
                sma_sum = sum(c.close for c in self.candles[-299:]) + new_candle.close
                new_candle.sma300 = sma_sum / 300
            else:
                new_candle.sma300 = None

            # ===== DOUBLE TOUCH: Calculate EMA ribbon =====
            mult9 = 2 / (9 + 1)
            mult21 = 2 / (21 + 1)
            mult50 = 2 / (50 + 1)

            if self.ema9_prev is not None:
                new_candle.ema9 = new_candle.close * mult9 + self.ema9_prev * (1 - mult9)
                new_candle.ema21 = new_candle.close * mult21 + self.ema21_prev * (1 - mult21)
                new_candle.ema50 = new_candle.close * mult50 + self.ema50_prev * (1 - mult50)
            else:
                new_candle.ema9 = new_candle.close
                new_candle.ema21 = new_candle.close
                new_candle.ema50 = new_candle.close

            self.ema9_prev = new_candle.ema9
            self.ema21_prev = new_candle.ema21
            self.ema50_prev = new_candle.ema50

            # Calculate band color
            new_candle.band_color = self._get_band_color(new_candle.ema9, new_candle.ema21, new_candle.ema50)

            # ===== DOUBLE TOUCH: Calculate EWVMA-200 =====
            self.volume_sum_200.append(vol)
            if self.ewvma_200_prev is not None:
                nbfs_200 = sum(self.volume_sum_200)
                if nbfs_200 > 0:
                    new_candle.ewvma_200 = self.ewvma_200_prev * (nbfs_200 - vol) / nbfs_200 + (vol * new_candle.close / nbfs_200)
                else:
                    new_candle.ewvma_200 = new_candle.close
            else:
                new_candle.ewvma_200 = new_candle.close

            self.ewvma_200_prev = new_candle.ewvma_200

            self.candles.append(new_candle)

            # ===== DOUBLE TOUCH: Detect HH/LL for latest candles =====
            # Re-detect HH/LL for recent candles (last 25 to cover lookback window)
            self._detect_hh_ll_recent(lookback=20)

            # Trim to max candles
            if len(self.candles) > self.max_candles:
                self.candles = self.candles[-self.max_candles:]

            # Notify callback
            if data["confirm"] and self.on_new_candle:
                self.on_new_candle(new_candle)

    def find_swing_points(self, lookback: int = 3) -> List[SwingPoint]:
        """Find swing highs and lows."""
        swings = []

        for i in range(lookback, len(self.candles) - lookback):
            candle = self.candles[i]

            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if candle.high <= self.candles[i-j].high or candle.high <= self.candles[i+j].high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swings.append(SwingPoint(
                    index=i,
                    price=candle.high,
                    type='high',
                    candle_open=candle.open,
                    time=candle.time
                ))

            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if candle.low >= self.candles[i-j].low or candle.low >= self.candles[i+j].low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swings.append(SwingPoint(
                    index=i,
                    price=candle.low,
                    type='low',
                    candle_open=candle.open,
                    time=candle.time
                ))

        swings.sort(key=lambda x: x.index)
        return swings

    def find_fvg(self, start_idx: int, end_idx: int, fvg_type: str) -> Optional[FVG]:
        """Find Fair Value Gap between start and end index."""
        for i in range(start_idx + 1, min(end_idx - 1, len(self.candles) - 1)):
            c1 = self.candles[i - 1]
            c2 = self.candles[i]
            c3 = self.candles[i + 1]

            if fvg_type == 'bullish':
                # Bullish FVG: gap between c1 high and c3 low
                if c1.high < c3.low:
                    return FVG(
                        index=i,
                        type='bullish',
                        top=c3.low,
                        bottom=c1.high,
                        time=c2.time
                    )
            else:  # bearish
                # Bearish FVG: gap between c1 low and c3 high
                if c1.low > c3.high:
                    return FVG(
                        index=i,
                        type='bearish',
                        top=c1.low,
                        bottom=c3.high,
                        time=c2.time
                    )

        return None

    def is_into_ribbon(self, price: float, index: int, buffer: float = 0.002) -> bool:
        """Check if price is into the EVWMA ribbon."""
        candle = self.candles[index]
        if candle.evwma_upper is None or candle.evwma_lower is None:
            return False

        ribbon_range = candle.evwma_upper - candle.evwma_lower
        extended_upper = candle.evwma_upper + ribbon_range * buffer
        extended_lower = candle.evwma_lower - ribbon_range * buffer

        return extended_lower <= price <= extended_upper

    def is_above_ribbon(self, close: float, index: int) -> bool:
        """Check if close is above ribbon upper."""
        candle = self.candles[index]
        return candle.evwma_upper is not None and close > candle.evwma_upper

    def is_below_ribbon(self, close: float, index: int) -> bool:
        """Check if close is below ribbon lower."""
        candle = self.candles[index]
        return candle.evwma_lower is not None and close < candle.evwma_lower

    def check_trend_filter(self, start_idx: int, end_idx: int, direction: str, threshold: float = 0.8) -> bool:
        """
        Check if at least threshold% of candles are above/below 300 SMA.

        Args:
            start_idx: Start index (A point)
            end_idx: End index (G point)
            direction: 'long' or 'short'
            threshold: Percentage required (default 0.8 = 80%)

        Returns:
            True if trend filter passes
        """
        if start_idx >= end_idx or end_idx >= len(self.candles):
            return False

        # Count candles with valid SMA
        valid_count = 0
        trend_count = 0

        for i in range(start_idx, end_idx + 1):
            candle = self.candles[i]
            if candle.sma300 is None:
                continue

            valid_count += 1
            if direction == 'long' and candle.close > candle.sma300:
                trend_count += 1
            elif direction == 'short' and candle.close < candle.sma300:
                trend_count += 1

        if valid_count == 0:
            return False

        return (trend_count / valid_count) >= threshold

    def get_current_price(self) -> float:
        """Get current price (last close)."""
        if self.candles:
            return self.candles[-1].close
        return 0

    # ==================== DOUBLE TOUCH INDICATORS ====================

    def _calculate_ema_ribbon(self):
        """Calculate EMA 9/21/50 and band color for all candles."""
        if not self.candles:
            return

        # EMA multipliers
        mult9 = 2 / (9 + 1)
        mult21 = 2 / (21 + 1)
        mult50 = 2 / (50 + 1)

        # Initialize with first close
        ema9 = self.candles[0].close
        ema21 = self.candles[0].close
        ema50 = self.candles[0].close

        for i, candle in enumerate(self.candles):
            if i == 0:
                candle.ema9 = ema9
                candle.ema21 = ema21
                candle.ema50 = ema50
            else:
                ema9 = candle.close * mult9 + ema9 * (1 - mult9)
                ema21 = candle.close * mult21 + ema21 * (1 - mult21)
                ema50 = candle.close * mult50 + ema50 * (1 - mult50)

                candle.ema9 = ema9
                candle.ema21 = ema21
                candle.ema50 = ema50

            # Determine band color
            candle.band_color = self._get_band_color(candle.ema9, candle.ema21, candle.ema50)

        # Save state for real-time updates
        self.ema9_prev = ema9
        self.ema21_prev = ema21
        self.ema50_prev = ema50

    def _get_band_color(self, ema9: float, ema21: float, ema50: float) -> str:
        """Determine band color from EMA alignment."""
        if ema9 is None or ema21 is None or ema50 is None:
            return 'grey'

        if ema9 > ema21 > ema50:
            return 'green'  # Bullish
        elif ema9 < ema21 < ema50:
            return 'red'    # Bearish
        else:
            return 'grey'   # Neutral/transition

    def _calculate_ewvma_200(self):
        """Calculate EWVMA-200 for counter-trend filter."""
        if not self.candles:
            return

        # Reset state
        self.volume_sum_200.clear()
        ewvma_prev = None

        for i, candle in enumerate(self.candles):
            vol = candle.volume if candle.volume > 0 else 1
            self.volume_sum_200.append(vol)

            if ewvma_prev is None:
                candle.ewvma_200 = candle.close
            else:
                nbfs = sum(self.volume_sum_200)
                if nbfs > 0:
                    candle.ewvma_200 = ewvma_prev * (nbfs - vol) / nbfs + (vol * candle.close / nbfs)
                else:
                    candle.ewvma_200 = candle.close

            ewvma_prev = candle.ewvma_200

        # Save state for real-time updates
        self.ewvma_200_prev = ewvma_prev

    def _detect_hh_ll(self, lookback: int = 20):
        """Detect Higher Highs and Lower Lows for pattern detection."""
        if len(self.candles) < lookback + 1:
            return

        for i in range(lookback, len(self.candles)):
            candle = self.candles[i]

            # Get max high and min low in lookback window (excluding current)
            window_highs = [c.high for c in self.candles[i-lookback:i]]
            window_lows = [c.low for c in self.candles[i-lookback:i]]

            max_high = max(window_highs) if window_highs else 0
            min_low = min(window_lows) if window_lows else float('inf')

            # Higher High: current high exceeds all highs in lookback
            candle.is_hh = candle.high > max_high

            # Lower Low: current low is below all lows in lookback
            candle.is_ll = candle.low < min_low

    def _detect_hh_ll_recent(self, lookback: int = 20):
        """Detect HH/LL for only the most recent candles (for real-time updates)."""
        if len(self.candles) < lookback + 1:
            return

        # Only check the last few candles (saves computation)
        start_idx = max(lookback, len(self.candles) - 5)

        for i in range(start_idx, len(self.candles)):
            candle = self.candles[i]

            # Get max high and min low in lookback window (excluding current)
            window_highs = [c.high for c in self.candles[i-lookback:i]]
            window_lows = [c.low for c in self.candles[i-lookback:i]]

            max_high = max(window_highs) if window_highs else 0
            min_low = min(window_lows) if window_lows else float('inf')

            candle.is_hh = candle.high > max_high
            candle.is_ll = candle.low < min_low

    def get_band_color(self, index: int) -> str:
        """Get band color at specific index."""
        if 0 <= index < len(self.candles):
            return self.candles[index].band_color or 'grey'
        return 'grey'

    def get_ewvma_200(self, index: int) -> Optional[float]:
        """Get EWVMA-200 value at specific index."""
        if 0 <= index < len(self.candles):
            return self.candles[index].ewvma_200
        return None

    def is_hh(self, index: int) -> bool:
        """Check if candle at index is a Higher High."""
        if 0 <= index < len(self.candles):
            return self.candles[index].is_hh
        return False

    def is_ll(self, index: int) -> bool:
        """Check if candle at index is a Lower Low."""
        if 0 <= index < len(self.candles):
            return self.candles[index].is_ll
        return False

    def check_ewvma_counter_trend(self, index: int, direction: str) -> bool:
        """
        Check EWVMA-200 counter-trend filter.

        For LONGS: Price at index must be BELOW EWVMA-200 (mean reversion from oversold)
        For SHORTS: Price at index must be ABOVE EWVMA-200 (mean reversion from overbought)

        Args:
            index: Candle index to check (typically Step 0)
            direction: 'long' or 'short'

        Returns:
            True if counter-trend filter passes
        """
        if index < 0 or index >= len(self.candles):
            return False

        candle = self.candles[index]
        if candle.ewvma_200 is None:
            return False

        if direction == 'long':
            return candle.close < candle.ewvma_200
        else:  # short
            return candle.close > candle.ewvma_200

    # ==================== END DOUBLE TOUCH INDICATORS ====================

    def to_dataframe(self) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        data = []
        for c in self.candles:
            data.append({
                'time': c.time,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume,
                'evwma_mid': c.evwma_mid,
                'evwma_upper': c.evwma_upper,
                'evwma_lower': c.evwma_lower
            })
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test data feed
    config = BotConfig(testnet=True, timeframe="5")
    client = BybitClient(config)
    feed = DataFeed(config, client, "BTCUSDT")

    feed.load_historical(100)

    print(f"\nLast candle:")
    last = feed.candles[-1]
    print(f"  Close: {last.close}")
    print(f"  EVWMA Mid: {last.evwma_mid:.2f}")
    print(f"  EVWMA Upper: {last.evwma_upper:.2f}")
    print(f"  EVWMA Lower: {last.evwma_lower:.2f}")

    swings = feed.find_swing_points(3)
    print(f"\nFound {len(swings)} swing points")

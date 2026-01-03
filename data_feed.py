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

    # EVWMA values (calculated)
    evwma_mid: float = None
    evwma_upper: float = None
    evwma_lower: float = None


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

    def __init__(self, config: BotConfig, client: BybitClient, symbol: str):
        self.config = config
        self.client = client
        self.symbol = symbol
        self.candles: List[Candle] = []
        self.max_candles = 500  # Keep last 500 candles
        self.on_new_candle: Optional[Callable] = None

        # EVWMA state
        self.evwma_mid_prev = None
        self.evwma_upper_prev = None
        self.evwma_lower_prev = None
        self.volume_sum = deque(maxlen=config.evwma_length)

    def load_historical(self, limit: int = 200):
        """Load historical candles from API."""
        klines = self.client.get_klines(
            self.symbol,
            self.config.timeframe,
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

        # Calculate EVWMA for all historical candles
        self._calculate_evwma_all()

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
        """Calculate EVWMA for all candles."""
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

            self.candles.append(new_candle)

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

    def get_current_price(self) -> float:
        """Get current price (last close)."""
        if self.candles:
            return self.candles[-1].close
        return 0

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

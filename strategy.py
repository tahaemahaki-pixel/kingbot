"""
King Strategy Pattern Detection
Long King and Short King with FVG entry
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from data_feed import DataFeed, SwingPoint, FVG, Candle


class SignalType(Enum):
    LONG_KING = "long_king"
    SHORT_KING = "short_king"


class SignalStatus(Enum):
    PENDING_FVG_RETEST = "pending_fvg_retest"  # Waiting for price to retest FVG
    READY = "ready"  # Can execute
    EXPIRED = "expired"  # FVG invalidated
    FILLED = "filled"  # Order filled


@dataclass
class TradeSignal:
    signal_type: SignalType
    status: SignalStatus
    symbol: str  # Trading symbol (e.g., ETHUSDT)

    # Pattern points
    a_index: int
    a_price: float
    c_index: int
    c_price: float  # Target
    d_index: int
    e_index: int
    e_price: float
    e_candle_open: float  # Stop reference
    f_index: int
    fvg: FVG

    # Trade levels
    entry_price: float  # Mid of FVG
    stop_loss: float
    target: float

    # Timing
    created_at: int  # Candle time when signal was created
    max_wait_candles: int = 20
    candles_waited: int = 0
    setup_key: str = ""  # Unique key for setup (e.g., ETHUSDT_5 or ETHUSDT_1)

    def get_risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk > 0 else 0


class KingStrategy:
    """
    Detects Long King and Short King patterns with FVG entry.
    """

    def __init__(self, data_feed: DataFeed, swing_lookback: int = 3, max_wait_candles: int = 20):
        self.feed = data_feed
        self.swing_lookback = swing_lookback
        self.max_wait_candles = max_wait_candles
        self.active_signals: List[TradeSignal] = []

    def scan_for_patterns(self) -> List[TradeSignal]:
        """Scan for new King patterns."""
        signals = []

        swings = self.feed.find_swing_points(self.swing_lookback)
        swing_highs = [s for s in swings if s.type == 'high']
        swing_lows = [s for s in swings if s.type == 'low']

        # Scan for Long King patterns
        long_signals = self._find_long_king_patterns(swings, swing_highs, swing_lows)
        signals.extend(long_signals)

        # Scan for Short King patterns
        short_signals = self._find_short_king_patterns(swings, swing_highs, swing_lows)
        signals.extend(short_signals)

        return signals

    def _find_long_king_patterns(
        self,
        swings: List[SwingPoint],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[TradeSignal]:
        """
        Long King Pattern:
        A: Swing low INTO ribbon
        C: Swing high above A
        D: Close below A + pullback into ribbon
        E: Lower low below D
        F: Close above ribbon
        FVG: Bullish FVG between E and F
        G: Entry at FVG retest
        """
        signals = []
        candles = self.feed.candles

        for a in swing_lows:
            # Skip if too recent (need room for pattern to complete)
            if a.index > len(candles) - 10:
                continue

            # A must be INTO ribbon
            if not self.feed.is_into_ribbon(a.price, a.index, buffer=0.02):
                continue

            # Find C: swing high after A, above A's level
            for c in swing_highs:
                if c.index <= a.index or c.index > a.index + 50:
                    continue
                if c.price <= a.price:
                    continue

                # Find D: close below A, then pullback into ribbon
                d_idx = None
                for i in range(c.index + 1, min(c.index + 30, len(candles))):
                    if candles[i].close < a.price:
                        # Look for pullback into ribbon
                        for j in range(i + 1, min(i + 15, len(candles))):
                            if self.feed.is_into_ribbon(candles[j].close, j, buffer=0.02):
                                d_idx = j
                                break
                        if d_idx:
                            break

                if d_idx is None:
                    continue

                # Find E: lower low after D
                e_swing = None
                d_low = candles[d_idx].low
                for e in swing_lows:
                    if e.index <= d_idx or e.index > d_idx + 30:
                        continue
                    if e.price < d_low:
                        e_swing = e
                        break

                if e_swing is None:
                    continue

                # Find F: close above ribbon after E
                f_idx = None
                for i in range(e_swing.index + 1, min(e_swing.index + 20, len(candles))):
                    if self.feed.is_above_ribbon(candles[i].close, i):
                        f_idx = i
                        break

                if f_idx is None:
                    continue

                # Find bullish FVG between E and F
                fvg = self.feed.find_fvg(e_swing.index, f_idx + 1, 'bullish')
                if fvg is None:
                    continue

                # Create signal
                # Entry at FVG midpoint + 0.03% buffer for better fill rate on longs
                fvg_midpoint = (fvg.top + fvg.bottom) / 2
                entry_price = fvg_midpoint * 1.0003  # Slightly above midpoint

                # ETH uses candle open SL (performs better), others use structure SL
                if self.feed.symbol == "ETHUSDT":
                    stop_loss = e_swing.candle_open - (e_swing.candle_open * 0.001)  # Candle open
                else:
                    stop_loss = e_swing.price - (e_swing.price * 0.001)  # Structure (swing low)

                target = c.price

                signal = TradeSignal(
                    signal_type=SignalType.LONG_KING,
                    status=SignalStatus.PENDING_FVG_RETEST,
                    symbol=self.feed.symbol,
                    setup_key=f"{self.feed.symbol}_{self.feed.timeframe}",
                    a_index=a.index,
                    a_price=a.price,
                    c_index=c.index,
                    c_price=c.price,
                    d_index=d_idx,
                    e_index=e_swing.index,
                    e_price=e_swing.price,
                    e_candle_open=e_swing.candle_open,
                    f_index=f_idx,
                    fvg=fvg,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    created_at=candles[f_idx].time,
                    max_wait_candles=self.max_wait_candles
                )

                # Only add if R:R is favorable
                if signal.get_risk_reward() < 1.0:
                    continue

                # Apply 300 SMA trend filter (80% of candles must be above SMA for longs)
                if not self.feed.check_trend_filter(a.index, f_idx, 'long', threshold=0.8):
                    continue

                signals.append(signal)
                break  # One pattern per A point

        return signals

    def _find_short_king_patterns(
        self,
        swings: List[SwingPoint],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[TradeSignal]:
        """
        Short King Pattern:
        A: Swing high INTO ribbon
        C: Swing low below A
        D: Close above A + pullback into ribbon
        E: Higher high above D
        F: Close below ribbon
        FVG: Bearish FVG between E and F
        G: Entry at FVG retest
        """
        signals = []
        candles = self.feed.candles

        for a in swing_highs:
            if a.index > len(candles) - 10:
                continue

            if not self.feed.is_into_ribbon(a.price, a.index, buffer=0.02):
                continue

            for c in swing_lows:
                if c.index <= a.index or c.index > a.index + 50:
                    continue
                if c.price >= a.price:
                    continue

                # Find D: close above A, then pullback into ribbon
                d_idx = None
                for i in range(c.index + 1, min(c.index + 30, len(candles))):
                    if candles[i].close > a.price:
                        for j in range(i + 1, min(i + 15, len(candles))):
                            if self.feed.is_into_ribbon(candles[j].close, j, buffer=0.02):
                                d_idx = j
                                break
                        if d_idx:
                            break

                if d_idx is None:
                    continue

                # Find E: higher high after D
                e_swing = None
                d_high = candles[d_idx].high
                for e in swing_highs:
                    if e.index <= d_idx or e.index > d_idx + 30:
                        continue
                    if e.price > d_high:
                        e_swing = e
                        break

                if e_swing is None:
                    continue

                # Find F: close below ribbon after E
                f_idx = None
                for i in range(e_swing.index + 1, min(e_swing.index + 20, len(candles))):
                    if self.feed.is_below_ribbon(candles[i].close, i):
                        f_idx = i
                        break

                if f_idx is None:
                    continue

                # Find bearish FVG between E and F
                fvg = self.feed.find_fvg(e_swing.index, f_idx + 1, 'bearish')
                if fvg is None:
                    continue

                # Create signal
                # Entry at FVG midpoint - 0.03% buffer for better fill rate on shorts
                fvg_midpoint = (fvg.top + fvg.bottom) / 2
                entry_price = fvg_midpoint * 0.9997  # Slightly below midpoint

                # ETH uses candle open SL (performs better), others use structure SL
                if self.feed.symbol == "ETHUSDT":
                    stop_loss = e_swing.candle_open + (e_swing.candle_open * 0.001)  # Candle open
                else:
                    stop_loss = e_swing.price + (e_swing.price * 0.001)  # Structure (swing high)

                target = c.price

                signal = TradeSignal(
                    signal_type=SignalType.SHORT_KING,
                    status=SignalStatus.PENDING_FVG_RETEST,
                    symbol=self.feed.symbol,
                    setup_key=f"{self.feed.symbol}_{self.feed.timeframe}",
                    a_index=a.index,
                    a_price=a.price,
                    c_index=c.index,
                    c_price=c.price,
                    d_index=d_idx,
                    e_index=e_swing.index,
                    e_price=e_swing.price,
                    e_candle_open=e_swing.candle_open,
                    f_index=f_idx,
                    fvg=fvg,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    created_at=candles[f_idx].time,
                    max_wait_candles=self.max_wait_candles
                )

                # Only add if R:R is favorable
                if signal.get_risk_reward() < 1.0:
                    continue

                # Apply 300 SMA trend filter (80% of candles must be below SMA for shorts)
                if not self.feed.check_trend_filter(a.index, f_idx, 'short', threshold=0.8):
                    continue

                signals.append(signal)
                break

        return signals

    def check_fvg_retest(self, signal: TradeSignal, current_candle: Candle) -> bool:
        """Check if price has retested the FVG zone."""
        fvg = signal.fvg

        if signal.signal_type == SignalType.LONG_KING:
            # For longs, price should dip into FVG
            if current_candle.low <= fvg.top and current_candle.low >= fvg.bottom:
                return True
            # Check if FVG is invalidated (price closed below)
            if current_candle.close < fvg.bottom:
                signal.status = SignalStatus.EXPIRED
                return False
        else:  # SHORT_KING
            # For shorts, price should rally into FVG
            if current_candle.high >= fvg.bottom and current_candle.high <= fvg.top:
                return True
            # Check if FVG is invalidated (price closed above)
            if current_candle.close > fvg.top:
                signal.status = SignalStatus.EXPIRED
                return False

        return False

    def update_signals(self, new_candle: Candle):
        """Update active signals with new candle data."""
        for signal in self.active_signals:
            if signal.status == SignalStatus.PENDING_FVG_RETEST:
                signal.candles_waited += 1

                # Check for timeout
                if signal.candles_waited >= signal.max_wait_candles:
                    signal.status = SignalStatus.EXPIRED
                    continue

                # Check for FVG retest
                if self.check_fvg_retest(signal, new_candle):
                    signal.status = SignalStatus.READY

        # Remove expired signals
        self.active_signals = [s for s in self.active_signals if s.status != SignalStatus.EXPIRED]

    def get_ready_signals(self) -> List[TradeSignal]:
        """Get signals ready for execution."""
        return [s for s in self.active_signals if s.status == SignalStatus.READY]

    def add_signal(self, signal: TradeSignal):
        """Add a new signal to track."""
        # Check for duplicates (same pattern)
        for existing in self.active_signals:
            if (existing.signal_type == signal.signal_type and
                existing.a_index == signal.a_index and
                existing.e_index == signal.e_index):
                return  # Duplicate

        self.active_signals.append(signal)

    def remove_signal(self, signal: TradeSignal):
        """Remove a signal (after execution or invalidation)."""
        if signal in self.active_signals:
            self.active_signals.remove(signal)


if __name__ == "__main__":
    from bybit_client import BybitClient
    from config import BotConfig

    config = BotConfig(testnet=True, timeframe="5")
    client = BybitClient(config)
    feed = DataFeed(config, client, "BTCUSDT")

    print("Loading historical data...")
    feed.load_historical(200)

    strategy = KingStrategy(feed, swing_lookback=3, max_wait_candles=20)

    print("Scanning for patterns...")
    signals = strategy.scan_for_patterns()

    print(f"\nFound {len(signals)} signals:")
    for s in signals:
        print(f"  [{s.symbol}] {s.signal_type.value}: Entry={s.entry_price:.2f}, SL={s.stop_loss:.2f}, TP={s.target:.2f}, R:R={s.get_risk_reward():.2f}")

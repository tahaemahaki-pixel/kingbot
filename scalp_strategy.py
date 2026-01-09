"""
Scalping Strategy - FVG Breakout with Partial Exits

High-frequency scalping targeting 9+ trades/day with 85%+ win rate.

Entry conditions (ALL required):
- Bearish/Bullish FVG forms
- Price cradled 3+ of last 5 candles within EWVMA(20) bands
- Volume spike >= 1.5x (20-period average)
- Volume Delta Imbalance confirming direction

Exit system:
- TP1: 1.0R - Close 50%, move SL to breakeven
- TP2: 1.5R - Close remaining 50%
- Timeout: 30 candles max hold
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

from breakaway_strategy import BreakawayIndicators


class ScalpSignalType(Enum):
    SCALP_SHORT = "scalp_short"
    SCALP_LONG = "scalp_long"


class ScalpStatus(Enum):
    READY = "ready"              # Can execute immediately
    OPEN = "open"                # Position open, waiting for TP1
    TP1_HIT = "tp1_hit"          # TP1 reached, SL at breakeven
    CLOSED_TP = "closed_tp"      # Closed at full target
    CLOSED_SL = "closed_sl"      # Closed at stop loss
    CLOSED_BE = "closed_be"      # Closed at breakeven after TP1
    CLOSED_TIMEOUT = "timeout"   # Closed on timeout
    EXPIRED = "expired"          # Signal expired without fill


@dataclass
class ScalpSignal:
    """Represents a scalping trading signal with partial exit tracking."""
    signal_type: ScalpSignalType
    status: ScalpStatus
    symbol: str
    setup_key: str  # "BTCUSDT_5"

    # Entry levels
    entry_price: float   # FVG boundary
    stop_loss: float     # FVG other boundary + buffer
    tp1: float           # 1.0R target (close 50%)
    tp2: float           # 1.5R target (close remaining)

    # FVG info
    fvg_top: float
    fvg_bottom: float

    # Context at signal
    vol_ratio: float     # Volume spike ratio
    imbalance: float     # Volume delta imbalance
    cradle_count: int    # Candles cradled

    # Timing
    created_idx: int     # Candle index when created
    created_time: int    # Timestamp
    candles_held: int = 0  # Candles since entry

    # Position tracking
    original_size: float = 0.0
    remaining_size: float = 0.0
    entry_order_id: Optional[str] = None
    realized_pnl: float = 0.0

    @property
    def direction(self) -> str:
        return "short" if self.signal_type == ScalpSignalType.SCALP_SHORT else "long"

    @property
    def risk(self) -> float:
        return abs(self.stop_loss - self.entry_price)

    @property
    def r_multiple_tp1(self) -> float:
        return abs(self.tp1 - self.entry_price) / self.risk if self.risk > 0 else 0

    @property
    def r_multiple_tp2(self) -> float:
        return abs(self.tp2 - self.entry_price) / self.risk if self.risk > 0 else 0


class ScalpStrategy:
    """
    Scalping Strategy - FVG breakout with partial exits.

    Uses same indicator calculations as Breakaway but with:
    - Lower volume threshold (1.5x vs 2.0x)
    - Partial exit system (50% at 1R, 50% at 1.5R)
    - Shorter hold times (~25 min vs ~60 min)
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "5",
        # Entry filters
        min_vol_ratio: float = 1.5,
        imbalance_threshold: float = 0.10,
        imbalance_lookback: int = 10,
        min_cradle_candles: int = 3,
        cradle_lookback: int = 5,
        sl_buffer_pct: float = 0.001,
        # Exit system
        tp1_r_multiple: float = 1.0,
        tp2_r_multiple: float = 1.5,
        tp1_close_pct: float = 0.50,
        max_hold_candles: int = 30,
        # Direction
        trade_direction: str = "both",  # "both", "shorts", "longs"
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.setup_key = f"{symbol}_{timeframe}"

        # Entry filters
        self.min_vol_ratio = min_vol_ratio
        self.imbalance_threshold = imbalance_threshold
        self.imbalance_lookback = imbalance_lookback
        self.min_cradle_candles = min_cradle_candles
        self.cradle_lookback = cradle_lookback
        self.sl_buffer_pct = sl_buffer_pct

        # Exit system
        self.tp1_r_multiple = tp1_r_multiple
        self.tp2_r_multiple = tp2_r_multiple
        self.tp1_close_pct = tp1_close_pct
        self.max_hold_candles = max_hold_candles

        # Direction
        self.trade_direction = trade_direction

        # Indicators calculator (reuse from breakaway)
        self.indicators = BreakawayIndicators()

        # State
        self.active_signal: Optional[ScalpSignal] = None
        self._last_signal_idx = -1

        # Cached indicator values
        self._ind_cache: Optional[Dict[str, np.ndarray]] = None

    def calculate_indicators(self, opens: np.ndarray, closes: np.ndarray,
                            highs: np.ndarray, lows: np.ndarray,
                            volumes: np.ndarray):
        """Calculate and cache all indicators."""
        self._ind_cache = self.indicators.calculate_all(
            opens, closes, highs, lows, volumes, self.imbalance_lookback
        )

    def _detect_bearish_fvg(self, highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[dict]:
        """Detect bearish FVG (gap down). Condition: current_high < 2_candles_ago_low"""
        if idx < 2:
            return None

        if highs[idx] < lows[idx - 2]:
            return {
                'top': lows[idx - 2],      # FVG top (resistance)
                'bottom': highs[idx],       # FVG bottom (entry)
            }
        return None

    def _detect_bullish_fvg(self, highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[dict]:
        """Detect bullish FVG (gap up). Condition: current_low > 2_candles_ago_high"""
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

    def _check_short_filters(self, idx: int) -> Tuple[bool, dict]:
        """Check all short entry filters."""
        if self._ind_cache is None:
            return False, {}

        vol_ratio = self._ind_cache['vol_ratio'][idx]
        imbalance = self._ind_cache['imbalance'][idx]

        # Volume spike
        if vol_ratio < self.min_vol_ratio:
            return False, {}

        # Imbalance - require selling pressure
        if imbalance > -self.imbalance_threshold:
            return False, {}

        # Cradle check
        was_cradled, cradle_count = self._check_cradle(idx)
        if not was_cradled:
            return False, {}

        return True, {
            'vol_ratio': vol_ratio,
            'imbalance': imbalance,
            'cradle_count': cradle_count,
        }

    def _check_long_filters(self, idx: int) -> Tuple[bool, dict]:
        """Check all long entry filters."""
        if self._ind_cache is None:
            return False, {}

        vol_ratio = self._ind_cache['vol_ratio'][idx]
        imbalance = self._ind_cache['imbalance'][idx]

        # Volume spike
        if vol_ratio < self.min_vol_ratio:
            return False, {}

        # Imbalance - require buying pressure
        if imbalance < self.imbalance_threshold:
            return False, {}

        # Cradle check
        was_cradled, cradle_count = self._check_cradle(idx)
        if not was_cradled:
            return False, {}

        return True, {
            'vol_ratio': vol_ratio,
            'imbalance': imbalance,
            'cradle_count': cradle_count,
        }

    def scan_for_signals(self, opens: np.ndarray, closes: np.ndarray,
                        highs: np.ndarray, lows: np.ndarray,
                        volumes: np.ndarray, times: np.ndarray) -> Optional[ScalpSignal]:
        """
        Scan for new scalping signals on the latest candle.
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
                passed, context = self._check_short_filters(idx)
                if passed:
                    entry = fvg['bottom']
                    sl = fvg['top'] * (1 + self.sl_buffer_pct)
                    risk = sl - entry
                    tp1 = entry - (risk * self.tp1_r_multiple)
                    tp2 = entry - (risk * self.tp2_r_multiple)

                    signal = ScalpSignal(
                        signal_type=ScalpSignalType.SCALP_SHORT,
                        status=ScalpStatus.READY,
                        symbol=self.symbol,
                        setup_key=self.setup_key,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp2,
                        fvg_top=fvg['top'],
                        fvg_bottom=fvg['bottom'],
                        vol_ratio=context['vol_ratio'],
                        imbalance=context['imbalance'],
                        cradle_count=context['cradle_count'],
                        created_idx=idx,
                        created_time=int(times[idx]),
                    )

        # Check for LONG signal (only if no short signal)
        if signal is None and self.trade_direction in ["both", "longs"]:
            fvg = self._detect_bullish_fvg(highs, lows, idx)
            if fvg is not None:
                passed, context = self._check_long_filters(idx)
                if passed:
                    entry = fvg['top']
                    sl = fvg['bottom'] * (1 - self.sl_buffer_pct)
                    risk = entry - sl
                    tp1 = entry + (risk * self.tp1_r_multiple)
                    tp2 = entry + (risk * self.tp2_r_multiple)

                    signal = ScalpSignal(
                        signal_type=ScalpSignalType.SCALP_LONG,
                        status=ScalpStatus.READY,
                        symbol=self.symbol,
                        setup_key=self.setup_key,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp2,
                        fvg_top=fvg['top'],
                        fvg_bottom=fvg['bottom'],
                        vol_ratio=context['vol_ratio'],
                        imbalance=context['imbalance'],
                        cradle_count=context['cradle_count'],
                        created_idx=idx,
                        created_time=int(times[idx]),
                    )

        if signal:
            self._last_signal_idx = idx
            self.active_signal = signal

        return signal

    def check_exit_conditions(self, current_price: float, candles_held: int) -> Tuple[str, float]:
        """
        Check if any exit conditions are met.

        Returns:
            Tuple of (exit_type, close_pct) where:
            - exit_type: "tp1", "tp2", "sl", "be", "timeout", or None
            - close_pct: Percentage of position to close (0.5 for TP1, 1.0 for others)
        """
        if self.active_signal is None:
            return None, 0.0

        signal = self.active_signal
        is_short = signal.signal_type == ScalpSignalType.SCALP_SHORT

        # Check timeout
        if candles_held >= self.max_hold_candles:
            return "timeout", 1.0

        # Check TP1 (only if not already hit)
        if signal.status == ScalpStatus.OPEN:
            if is_short and current_price <= signal.tp1:
                return "tp1", self.tp1_close_pct
            elif not is_short and current_price >= signal.tp1:
                return "tp1", self.tp1_close_pct

        # Check TP2 (only after TP1 hit)
        if signal.status == ScalpStatus.TP1_HIT:
            if is_short and current_price <= signal.tp2:
                return "tp2", 1.0
            elif not is_short and current_price >= signal.tp2:
                return "tp2", 1.0

            # Check breakeven stop (after TP1)
            if is_short and current_price >= signal.entry_price:
                return "be", 1.0
            elif not is_short and current_price <= signal.entry_price:
                return "be", 1.0

        # Check stop loss (before TP1)
        if signal.status == ScalpStatus.OPEN:
            if is_short and current_price >= signal.stop_loss:
                return "sl", 1.0
            elif not is_short and current_price <= signal.stop_loss:
                return "sl", 1.0

        return None, 0.0

    def update_after_tp1(self):
        """Update signal state after TP1 is hit."""
        if self.active_signal:
            self.active_signal.status = ScalpStatus.TP1_HIT
            # SL is now at breakeven (entry price)

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
            f"Entry={s.entry_price:.4f} SL={s.stop_loss:.4f} "
            f"TP1={s.tp1:.4f} TP2={s.tp2:.4f} "
            f"Vol={s.vol_ratio:.1f}x Imb={s.imbalance:+.2f}"
        )


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    from pathlib import Path

    data_path = Path("/home/tahae/ai-content/data/Tradingdata/volume-charts/BTCUSDT_5m_merged.csv")

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

        strategy = ScalpStrategy(
            symbol="BTCUSDT",
            timeframe="5",
            min_vol_ratio=1.5,
            imbalance_threshold=0.10,
            trade_direction="both"
        )

        signals_found = 0
        shorts = 0
        longs = 0

        for i in range(300, len(closes)):
            signal = strategy.scan_for_signals(
                opens[:i+1], closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1], times[:i+1]
            )
            if signal:
                signals_found += 1
                if signal.signal_type == ScalpSignalType.SCALP_SHORT:
                    shorts += 1
                else:
                    longs += 1
                strategy.clear_signal()

        print(f"\n=== Scalp Strategy Test ===")
        print(f"Data: {len(closes)} candles")
        print(f"Total signals: {signals_found}")
        print(f"  Shorts: {shorts}")
        print(f"  Longs: {longs}")
    else:
        print(f"Data file not found: {data_path}")

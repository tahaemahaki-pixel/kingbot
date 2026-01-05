"""
Double Touch Strategy Pattern Detection
Detects the 5-step Double Touch sequence with FVG entry
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from data_feed import DataFeed, FVG, Candle


class SignalType(Enum):
    LONG_DOUBLE_TOUCH = "long_double_touch"
    SHORT_DOUBLE_TOUCH = "short_double_touch"


class SignalStatus(Enum):
    PENDING_FVG_RETEST = "pending_fvg_retest"  # Waiting for price to retest FVG
    READY = "ready"  # Can execute
    EXPIRED = "expired"  # FVG invalidated or timeout
    FILLED = "filled"  # Order filled


@dataclass
class SetupState:
    """Tracks progress through the 5-step Double Touch sequence."""
    direction: Optional[str] = None  # 'long' or 'short'
    step: int = -1  # -1 = no setup, 0-3 = in progress
    step_0_idx: Optional[int] = None
    step_0_price: Optional[float] = None  # HH/LL price
    step_1_idx: Optional[int] = None  # First grey
    step_2_idx: Optional[int] = None  # Back to trend color
    step_3_idx: Optional[int] = None  # Second grey
    step_3_price: Optional[float] = None  # SL reference (low for longs, high for shorts)
    active: bool = False

    def reset(self):
        """Reset state to look for new setup."""
        self.direction = None
        self.step = -1
        self.step_0_idx = None
        self.step_0_price = None
        self.step_1_idx = None
        self.step_2_idx = None
        self.step_3_idx = None
        self.step_3_price = None
        self.active = False


@dataclass
class TradeSignal:
    """Trade signal with entry, SL, and TP levels."""
    signal_type: SignalType
    status: SignalStatus
    symbol: str
    setup_key: str  # e.g., "BTCUSDT_5"

    # Setup points
    step_0_idx: int
    step_0_price: float
    step_1_idx: int
    step_2_idx: int
    step_3_idx: int
    step_3_price: float
    fvg: FVG

    # Trade levels
    entry_price: float
    stop_loss: float
    target: float

    # Timing
    created_at: int  # Candle time when signal was created
    max_wait_candles: int = 20
    candles_waited: int = 0

    def get_risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk > 0 else 0


class DoubleTouchStrategy:
    """
    Detects Double Touch patterns with the 5-step sequence:
    Step 0: HH (longs) or LL (shorts) while band is trending
    Step 1: Band goes GREY (first pullback)
    Step 2: Band returns to trend color
    Step 3: Band goes GREY again (second pullback) - defines SL level
    Step 4: FVG appears + band resumes trend = ENTRY
    """

    def __init__(
        self,
        data_feed: DataFeed,
        risk_reward: float = 3.0,
        sl_buffer_pct: float = 0.001,
        use_ewvma_filter: bool = True,
        counter_trend_mode: bool = True,
        max_wait_candles: int = 20,
        htf_feed=None,  # Optional HTFDataFeed for directional filter
        use_htf_filter: bool = True
    ):
        self.feed = data_feed
        self.risk_reward = risk_reward
        self.sl_buffer_pct = sl_buffer_pct
        self.use_ewvma_filter = use_ewvma_filter
        self.counter_trend_mode = counter_trend_mode
        self.max_wait_candles = max_wait_candles
        self.htf_feed = htf_feed
        self.use_htf_filter = use_htf_filter

        self.active_signals: List[TradeSignal] = []

        # State machines for tracking setups
        self.long_state = SetupState()
        self.short_state = SetupState()

        # Track last processed index to avoid reprocessing
        self.last_processed_idx = -1

    def scan_for_patterns(self) -> List[TradeSignal]:
        """Scan for new Double Touch patterns."""
        signals = []
        candles = self.feed.candles

        if len(candles) < 30:
            return signals

        # Start from where we left off (or beginning if first scan)
        start_idx = max(self.last_processed_idx + 1, 25)

        for i in range(start_idx, len(candles)):
            candle = candles[i]
            prev_candle = candles[i - 1] if i > 0 else candle

            band = candle.band_color or 'grey'
            prev_band = prev_candle.band_color or 'grey'

            # Check long setup
            long_signal = self._check_long_setup(i, candle, prev_candle, band, prev_band)
            if long_signal:
                signals.append(long_signal)

            # Check short setup
            short_signal = self._check_short_setup(i, candle, prev_candle, band, prev_band)
            if short_signal:
                signals.append(short_signal)

        self.last_processed_idx = len(candles) - 1
        return signals

    def _check_long_setup(
        self,
        i: int,
        candle: Candle,
        prev_candle: Candle,
        band: str,
        prev_band: str
    ) -> Optional[TradeSignal]:
        """Check and advance long setup state machine."""
        state = self.long_state

        # Step 0: Look for HH while band is green
        if candle.is_hh and band == 'green' and not state.active:
            state.reset()
            state.direction = 'long'
            state.step = 0
            state.step_0_idx = i
            state.step_0_price = candle.high
            state.active = True
            return None

        if not state.active or state.direction != 'long':
            return None

        # Step 1: Band goes grey (pullback)
        if state.step == 0 and band == 'grey':
            state.step = 1
            state.step_1_idx = i
            return None

        # Step 2: Band goes green again (trend resumes)
        if state.step == 1 and band == 'green':
            state.step = 2
            state.step_2_idx = i
            return None

        # Step 3: Band goes grey again (second pullback)
        if state.step == 2 and band == 'grey':
            state.step = 3
            state.step_3_idx = i
            state.step_3_price = candle.low
            return None

        # Update step 3 low if still in grey
        if state.step == 3 and band == 'grey':
            if candle.low < state.step_3_price:
                state.step_3_price = candle.low
            return None

        # Step 4: Look for FVG after step 3, once band goes green
        if state.step == 3 and band == 'green':
            # Check for bullish FVG
            fvg = self.feed.find_fvg(state.step_3_idx, i + 1, 'bullish')

            if fvg is not None:
                # Apply EWVMA filter (counter-trend or trend-aligned based on mode)
                if self.use_ewvma_filter:
                    if self.counter_trend_mode:
                        # Counter-trend: longs when price < EWVMA (mean reversion)
                        if not self.feed.check_ewvma_counter_trend(state.step_0_idx, 'long'):
                            state.reset()
                            return None
                    else:
                        # Trend-aligned: longs when price > EWVMA (with the trend)
                        if not self.feed.check_ewvma_trend_aligned(state.step_0_idx, 'long'):
                            state.reset()
                            return None

                # Apply HTF directional filter - only take longs when HTF is bullish
                if self.use_htf_filter and self.htf_feed is not None:
                    htf_bias = self.htf_feed.get_bias()
                    if htf_bias != 'long':
                        state.reset()
                        return None

                # Calculate trade levels
                entry = fvg.top  # Enter at top of FVG
                sl = state.step_3_price * (1 - self.sl_buffer_pct)
                risk = entry - sl
                target = entry + (risk * self.risk_reward)

                signal = TradeSignal(
                    signal_type=SignalType.LONG_DOUBLE_TOUCH,
                    status=SignalStatus.PENDING_FVG_RETEST,
                    symbol=self.feed.symbol,
                    setup_key=f"{self.feed.symbol}_{self.feed.timeframe}",
                    step_0_idx=state.step_0_idx,
                    step_0_price=state.step_0_price,
                    step_1_idx=state.step_1_idx,
                    step_2_idx=state.step_2_idx,
                    step_3_idx=state.step_3_idx,
                    step_3_price=state.step_3_price,
                    fvg=fvg,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    created_at=candle.time,
                    max_wait_candles=self.max_wait_candles
                )

                state.reset()
                return signal
            else:
                # No FVG yet, check if we've waited too long
                if i - state.step_3_idx > 10:
                    state.reset()
                return None

        # Invalidation: If band goes red, reset
        if band == 'red':
            state.reset()

        return None

    def _check_short_setup(
        self,
        i: int,
        candle: Candle,
        prev_candle: Candle,
        band: str,
        prev_band: str
    ) -> Optional[TradeSignal]:
        """Check and advance short setup state machine."""
        state = self.short_state

        # Step 0: Look for LL while band is red
        if candle.is_ll and band == 'red' and not state.active:
            state.reset()
            state.direction = 'short'
            state.step = 0
            state.step_0_idx = i
            state.step_0_price = candle.low
            state.active = True
            return None

        if not state.active or state.direction != 'short':
            return None

        # Step 1: Band goes grey (rally/pullback)
        if state.step == 0 and band == 'grey':
            state.step = 1
            state.step_1_idx = i
            return None

        # Step 2: Band goes red again (trend resumes)
        if state.step == 1 and band == 'red':
            state.step = 2
            state.step_2_idx = i
            return None

        # Step 3: Band goes grey again (second pullback)
        if state.step == 2 and band == 'grey':
            state.step = 3
            state.step_3_idx = i
            state.step_3_price = candle.high
            return None

        # Update step 3 high if still in grey
        if state.step == 3 and band == 'grey':
            if candle.high > state.step_3_price:
                state.step_3_price = candle.high
            return None

        # Step 4: Look for FVG after step 3, once band goes red
        if state.step == 3 and band == 'red':
            # Check for bearish FVG
            fvg = self.feed.find_fvg(state.step_3_idx, i + 1, 'bearish')

            if fvg is not None:
                # Apply EWVMA filter (counter-trend or trend-aligned based on mode)
                if self.use_ewvma_filter:
                    if self.counter_trend_mode:
                        # Counter-trend: shorts when price > EWVMA (mean reversion)
                        if not self.feed.check_ewvma_counter_trend(state.step_0_idx, 'short'):
                            state.reset()
                            return None
                    else:
                        # Trend-aligned: shorts when price < EWVMA (with the trend)
                        if not self.feed.check_ewvma_trend_aligned(state.step_0_idx, 'short'):
                            state.reset()
                            return None

                # Apply HTF directional filter - only take shorts when HTF is bearish
                if self.use_htf_filter and self.htf_feed is not None:
                    htf_bias = self.htf_feed.get_bias()
                    if htf_bias != 'short':
                        state.reset()
                        return None

                # Calculate trade levels
                entry = fvg.bottom  # Enter at bottom of FVG
                sl = state.step_3_price * (1 + self.sl_buffer_pct)
                risk = sl - entry
                target = entry - (risk * self.risk_reward)

                signal = TradeSignal(
                    signal_type=SignalType.SHORT_DOUBLE_TOUCH,
                    status=SignalStatus.PENDING_FVG_RETEST,
                    symbol=self.feed.symbol,
                    setup_key=f"{self.feed.symbol}_{self.feed.timeframe}",
                    step_0_idx=state.step_0_idx,
                    step_0_price=state.step_0_price,
                    step_1_idx=state.step_1_idx,
                    step_2_idx=state.step_2_idx,
                    step_3_idx=state.step_3_idx,
                    step_3_price=state.step_3_price,
                    fvg=fvg,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    created_at=candle.time,
                    max_wait_candles=self.max_wait_candles
                )

                state.reset()
                return signal
            else:
                # No FVG yet, check if we've waited too long
                if i - state.step_3_idx > 10:
                    state.reset()
                return None

        # Invalidation: If band goes green, reset
        if band == 'green':
            state.reset()

        return None

    def check_fvg_retest(self, signal: TradeSignal, current_candle: Candle) -> bool:
        """Check if price has retested the FVG zone."""
        fvg = signal.fvg

        if signal.signal_type == SignalType.LONG_DOUBLE_TOUCH:
            # For longs, price should dip into FVG
            if current_candle.low <= fvg.top and current_candle.low >= fvg.bottom:
                return True
            # Check if FVG is invalidated (price closed below)
            if current_candle.close < fvg.bottom:
                signal.status = SignalStatus.EXPIRED
                return False
        else:  # SHORT_DOUBLE_TOUCH
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
        # Check for duplicates
        for existing in self.active_signals:
            if (existing.signal_type == signal.signal_type and
                existing.step_0_idx == signal.step_0_idx and
                existing.step_3_idx == signal.step_3_idx):
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
    feed.load_historical(500)

    strategy = DoubleTouchStrategy(
        feed,
        risk_reward=3.0,
        sl_buffer_pct=0.001,
        use_ewvma_filter=True,
        counter_trend_mode=True,
        max_wait_candles=20
    )

    print("Scanning for patterns...")
    signals = strategy.scan_for_patterns()

    print(f"\nFound {len(signals)} signals:")
    for s in signals:
        print(f"  [{s.symbol}] {s.signal_type.value}: Entry={s.entry_price:.2f}, "
              f"SL={s.stop_loss:.2f}, TP={s.target:.2f}, R:R={s.get_risk_reward():.2f}")

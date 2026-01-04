"""
Spread Strategy - MR Double Touch for Cointegrated Pairs

Detects Mean-Reversion Double Touch patterns on the spread between
two cointegrated assets (e.g., ETH/BTC).

Pattern Logic:
1. First extreme: z-score < -2.0 (oversold) or > 2.0 (overbought)
2. Partial recovery: z > -1.0 or < 1.0
3. Second touch: z < -1.5 or > 1.5 (doesn't break first extreme)
4. Entry on second touch, expecting mean reversion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from collections import deque
import numpy as np
import json
import os

from config import BotConfig, SpreadPairConfig
from data_feed import DataFeed, Candle


# Pattern state persistence
PATTERN_STATE_FILE = "data/pattern_state.json"


class SpreadSignalType(Enum):
    LONG_SPREAD = "long_spread"    # Buy ETH, Sell BTC
    SHORT_SPREAD = "short_spread"  # Sell ETH, Buy BTC


class SpreadSignalStatus(Enum):
    PENDING = "pending"      # Pattern detected, waiting for entry
    READY = "ready"          # Ready to execute
    FILLED = "filled"        # Orders placed
    CLOSED = "closed"        # Position closed


@dataclass
class SpreadSignal:
    """A spread trading signal."""
    signal_type: SpreadSignalType
    status: SpreadSignalStatus

    # Pattern info
    first_extreme_z: float
    recovery_z: float
    entry_z: float
    first_extreme_idx: int
    entry_idx: int

    # Current spread values
    spread_value: float
    spread_mean: float
    spread_std: float

    # Prices at entry
    asset_a_price: float  # ETH price
    asset_b_price: float  # BTC price
    hedge_ratio: float

    # Exit levels (z-score based)
    tp_z: float
    sl_z: float

    # Timing
    created_at: int = 0  # Timestamp

    def get_tp_spread(self) -> float:
        """Calculate take profit spread value."""
        # TP at z = tp_z (closer to mean)
        if self.signal_type == SpreadSignalType.LONG_SPREAD:
            return self.spread_mean - self.tp_z * self.spread_std
        else:
            return self.spread_mean + self.tp_z * self.spread_std

    def get_sl_spread(self) -> float:
        """Calculate stop loss spread value."""
        # SL at z = sl_z (further from mean)
        if self.signal_type == SpreadSignalType.LONG_SPREAD:
            return self.spread_mean - self.sl_z * self.spread_std
        else:
            return self.spread_mean + self.sl_z * self.spread_std


@dataclass
class SpreadData:
    """Real-time spread data point."""
    time: int
    spread: float
    zscore: float
    spread_mean: float
    spread_std: float
    asset_a_price: float
    asset_b_price: float


class SpreadStrategy:
    """
    MR Double Touch strategy for spread trading.

    Monitors the spread between two assets and detects mean-reversion
    patterns using the MR Double Touch methodology.
    """

    def __init__(
        self,
        config: BotConfig,
        feed_a: DataFeed,  # Asset A feed (e.g., ETH)
        feed_b: DataFeed,  # Asset B feed (e.g., BTC)
    ):
        self.config = config
        self.pair_config = config.spread_pair
        self.feed_a = feed_a
        self.feed_b = feed_b

        # Spread data
        self.spread_history: deque = deque(maxlen=500)
        self.hedge_ratio: float = self.pair_config.hedge_ratio

        # Pattern state machine (will be loaded from disk if available)
        self.pattern_state: Dict = {'phase': 0}

        # Active signals
        self.active_signals: List[SpreadSignal] = []

        # Calculate initial spread and z-score
        self._initialize_spread()

        # Load persisted pattern state after spread is initialized
        self._load_pattern_state()

    def _initialize_spread(self):
        """Calculate spread from historical data and compute hedge ratio."""
        if not self.feed_a.candles or not self.feed_b.candles:
            return

        # Align candles by time
        times_a = {c.time: c for c in self.feed_a.candles}
        times_b = {c.time: c for c in self.feed_b.candles}

        common_times = sorted(set(times_a.keys()) & set(times_b.keys()))

        if len(common_times) < self.pair_config.zscore_window + 10:
            print(f"Not enough aligned candles for spread: {len(common_times)}")
            return

        # Calculate hedge ratio using OLS on recent data
        prices_a = [times_a[t].close for t in common_times[-200:]]
        prices_b = [times_b[t].close for t in common_times[-200:]]

        # Simple OLS: hedge_ratio = cov(a, b) / var(b)
        mean_a = np.mean(prices_a)
        mean_b = np.mean(prices_b)
        cov = np.mean([(a - mean_a) * (b - mean_b) for a, b in zip(prices_a, prices_b)])
        var_b = np.var(prices_b)

        if var_b > 0:
            self.hedge_ratio = cov / var_b
        print(f"Calculated hedge ratio: {self.hedge_ratio:.6f}")

        # Calculate spread for all aligned candles
        spreads = []
        for t in common_times:
            spread = times_a[t].close - self.hedge_ratio * times_b[t].close
            spreads.append({
                'time': t,
                'spread': spread,
                'price_a': times_a[t].close,
                'price_b': times_b[t].close
            })

        # Calculate rolling z-score
        window = self.pair_config.zscore_window
        for i in range(window, len(spreads)):
            recent = [s['spread'] for s in spreads[i-window:i]]
            mean = np.mean(recent)
            std = np.std(recent)

            if std > 0:
                zscore = (spreads[i]['spread'] - mean) / std
            else:
                zscore = 0

            self.spread_history.append(SpreadData(
                time=spreads[i]['time'],
                spread=spreads[i]['spread'],
                zscore=zscore,
                spread_mean=mean,
                spread_std=std,
                asset_a_price=spreads[i]['price_a'],
                asset_b_price=spreads[i]['price_b']
            ))

        print(f"Initialized spread with {len(self.spread_history)} data points")
        if self.spread_history:
            print(f"Current z-score: {self.spread_history[-1].zscore:.2f}")

    def _get_state_file_path(self) -> str:
        """Get the pattern state file path."""
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        return PATTERN_STATE_FILE

    def _load_pattern_state(self):
        """Load pattern state from disk if available."""
        state_file = self._get_state_file_path()
        pair_name = self.pair_config.name

        if not os.path.exists(state_file):
            return

        try:
            with open(state_file, 'r') as f:
                all_states = json.load(f)

            if pair_name in all_states:
                saved_state = all_states[pair_name]
                # Validate saved state is still relevant
                if saved_state.get('phase', 0) > 0:
                    # Restore the pattern state
                    self.pattern_state = {
                        'phase': saved_state['phase'],
                        'type': saved_state.get('type'),
                        'first_z': saved_state.get('first_z'),
                        'first_time': saved_state.get('first_time'),
                        'recovery_z': saved_state.get('recovery_z'),
                        # Use current history length as first_idx approximation
                        'first_idx': max(0, len(self.spread_history) - saved_state.get('bars_since_first', 50))
                    }
                    print(f"  Restored pattern state for {pair_name}: phase={saved_state['phase']}, type={saved_state.get('type')}")
        except Exception as e:
            print(f"  Could not load pattern state: {e}")

    def _save_pattern_state(self):
        """Save current pattern state to disk."""
        state_file = self._get_state_file_path()
        pair_name = self.pair_config.name

        # Load existing states for other pairs
        all_states = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    all_states = json.load(f)
            except:
                pass

        # Update state for this pair
        if self.pattern_state.get('phase', 0) > 0:
            all_states[pair_name] = {
                'phase': self.pattern_state['phase'],
                'type': self.pattern_state.get('type'),
                'first_z': self.pattern_state.get('first_z'),
                'first_time': self.pattern_state.get('first_time'),
                'recovery_z': self.pattern_state.get('recovery_z'),
                'bars_since_first': len(self.spread_history) - self.pattern_state.get('first_idx', 0),
                'saved_at': int(self.spread_history[-1].time) if self.spread_history else 0
            }
        elif pair_name in all_states:
            # Remove state when pattern resets
            del all_states[pair_name]

        # Save to disk
        try:
            with open(state_file, 'w') as f:
                json.dump(all_states, f, indent=2)
        except Exception as e:
            print(f"  Could not save pattern state: {e}")

    def update(self, candle_a: Candle, candle_b: Candle) -> Optional[SpreadSignal]:
        """
        Update spread with new candles and check for patterns.

        Returns a new signal if pattern completes.
        """
        if candle_a.time != candle_b.time:
            return None  # Candles not aligned

        # Calculate current spread
        spread = candle_a.close - self.hedge_ratio * candle_b.close

        # Calculate z-score from recent history
        if len(self.spread_history) < self.pair_config.zscore_window:
            return None

        recent_spreads = [s.spread for s in list(self.spread_history)[-self.pair_config.zscore_window:]]
        mean = np.mean(recent_spreads)
        std = np.std(recent_spreads)

        if std == 0:
            return None

        zscore = (spread - mean) / std

        # Add to history
        spread_data = SpreadData(
            time=candle_a.time,
            spread=spread,
            zscore=zscore,
            spread_mean=mean,
            spread_std=std,
            asset_a_price=candle_a.close,
            asset_b_price=candle_b.close
        )
        self.spread_history.append(spread_data)

        # Update existing signals
        self._update_signals(spread_data)

        # Check for new pattern
        signal = self._check_pattern(spread_data)

        # Persist pattern state after any changes
        self._save_pattern_state()

        return signal

    def _check_pattern(self, data: SpreadData) -> Optional[SpreadSignal]:
        """Check for MR Double Touch pattern."""
        z = data.zscore
        cfg = self.pair_config
        state = self.pattern_state

        # ========== LONG SPREAD PATTERN ==========
        if state['phase'] == 0:
            # Phase 0: Looking for first extreme (oversold)
            if z < -cfg.first_extreme_z:
                self.pattern_state = {
                    'phase': 1,
                    'type': 'long',
                    'first_idx': len(self.spread_history) - 1,
                    'first_z': z,
                    'first_time': data.time
                }

        elif state['phase'] == 1 and state.get('type') == 'long':
            # Phase 1: Looking for partial recovery
            if z > -cfg.recovery_z:
                state['phase'] = 2
                state['recovery_idx'] = len(self.spread_history) - 1
                state['recovery_z'] = z
            elif z < state['first_z'] - 0.5:
                # Extended too far, reset
                self.pattern_state = {'phase': 0}

        elif state['phase'] == 2 and state.get('type') == 'long':
            # Phase 2: Looking for second touch
            if z < -cfg.second_touch_z:
                # Check for "higher low" - second touch shouldn't break first
                if z > state['first_z'] - 0.3:
                    # Pattern complete! Create signal
                    signal = SpreadSignal(
                        signal_type=SpreadSignalType.LONG_SPREAD,
                        status=SpreadSignalStatus.READY,
                        first_extreme_z=state['first_z'],
                        recovery_z=state['recovery_z'],
                        entry_z=z,
                        first_extreme_idx=state['first_idx'],
                        entry_idx=len(self.spread_history) - 1,
                        spread_value=data.spread,
                        spread_mean=data.spread_mean,
                        spread_std=data.spread_std,
                        asset_a_price=data.asset_a_price,
                        asset_b_price=data.asset_b_price,
                        hedge_ratio=self.hedge_ratio,
                        tp_z=cfg.tp_z,
                        sl_z=cfg.sl_z,
                        created_at=data.time
                    )
                    self.pattern_state = {'phase': 0}
                    return signal
                else:
                    self.pattern_state = {'phase': 0}
            elif z > 0:
                # Went all the way to mean, pattern invalid
                self.pattern_state = {'phase': 0}
            elif len(self.spread_history) - state['first_idx'] > cfg.max_pattern_bars:
                # Took too long
                self.pattern_state = {'phase': 0}

        # ========== SHORT SPREAD PATTERN ==========
        if state['phase'] == 0:
            # Phase 0: Looking for first extreme (overbought)
            if z > cfg.first_extreme_z:
                self.pattern_state = {
                    'phase': 1,
                    'type': 'short',
                    'first_idx': len(self.spread_history) - 1,
                    'first_z': z,
                    'first_time': data.time
                }

        elif state['phase'] == 1 and state.get('type') == 'short':
            # Phase 1: Looking for partial recovery
            if z < cfg.recovery_z:
                state['phase'] = 2
                state['recovery_idx'] = len(self.spread_history) - 1
                state['recovery_z'] = z
            elif z > state['first_z'] + 0.5:
                self.pattern_state = {'phase': 0}

        elif state['phase'] == 2 and state.get('type') == 'short':
            # Phase 2: Looking for second touch
            if z > cfg.second_touch_z:
                # Check for "lower high"
                if z < state['first_z'] + 0.3:
                    signal = SpreadSignal(
                        signal_type=SpreadSignalType.SHORT_SPREAD,
                        status=SpreadSignalStatus.READY,
                        first_extreme_z=state['first_z'],
                        recovery_z=state['recovery_z'],
                        entry_z=z,
                        first_extreme_idx=state['first_idx'],
                        entry_idx=len(self.spread_history) - 1,
                        spread_value=data.spread,
                        spread_mean=data.spread_mean,
                        spread_std=data.spread_std,
                        asset_a_price=data.asset_a_price,
                        asset_b_price=data.asset_b_price,
                        hedge_ratio=self.hedge_ratio,
                        tp_z=cfg.tp_z,
                        sl_z=cfg.sl_z,
                        created_at=data.time
                    )
                    self.pattern_state = {'phase': 0}
                    return signal
                else:
                    self.pattern_state = {'phase': 0}
            elif z < 0:
                self.pattern_state = {'phase': 0}
            elif len(self.spread_history) - state['first_idx'] > cfg.max_pattern_bars:
                self.pattern_state = {'phase': 0}

        return None

    def _update_signals(self, data: SpreadData):
        """Update existing signals with current spread data."""
        z = data.zscore
        cfg = self.pair_config

        for signal in self.active_signals[:]:
            if signal.status != SpreadSignalStatus.FILLED:
                continue

            # Check exit conditions
            if signal.signal_type == SpreadSignalType.LONG_SPREAD:
                # TP: z rises to tp_z
                if z >= -cfg.tp_z:
                    signal.status = SpreadSignalStatus.CLOSED
                # SL: z drops below sl_z
                elif z < -cfg.sl_z:
                    signal.status = SpreadSignalStatus.CLOSED
            else:
                # TP: z drops to tp_z
                if z <= cfg.tp_z:
                    signal.status = SpreadSignalStatus.CLOSED
                # SL: z rises above sl_z
                elif z > cfg.sl_z:
                    signal.status = SpreadSignalStatus.CLOSED

    def get_current_zscore(self) -> float:
        """Get current z-score."""
        if self.spread_history:
            return self.spread_history[-1].zscore
        return 0.0

    def get_current_spread(self) -> Optional[SpreadData]:
        """Get current spread data."""
        if self.spread_history:
            return self.spread_history[-1]
        return None

    def add_signal(self, signal: SpreadSignal):
        """Add a signal to track."""
        self.active_signals.append(signal)

    def remove_signal(self, signal: SpreadSignal):
        """Remove a signal."""
        if signal in self.active_signals:
            self.active_signals.remove(signal)

    def get_ready_signals(self) -> List[SpreadSignal]:
        """Get signals ready for execution."""
        return [s for s in self.active_signals if s.status == SpreadSignalStatus.READY]

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        if not self.spread_history:
            return {}

        current = self.spread_history[-1]
        return {
            'hedge_ratio': self.hedge_ratio,
            'current_zscore': current.zscore,
            'spread_mean': current.spread_mean,
            'spread_std': current.spread_std,
            'current_spread': current.spread,
            'pattern_phase': self.pattern_state.get('phase', 0),
            'pattern_type': self.pattern_state.get('type', None),
            'active_signals': len(self.active_signals),
            'history_length': len(self.spread_history)
        }

"""
Dynamic Spread Scanner - Multi-Pair Cointegration Trading

Periodically checks cointegration between pairs and only trades
those that meet the threshold. Supports multiple spread pairs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from statsmodels.tsa.stattools import coint, adfuller
from collections import deque

from config import BotConfig
from data_feed import DataFeed, Candle
from spread_strategy import SpreadStrategy, SpreadSignal, SpreadSignalType, SpreadSignalStatus


@dataclass
class SpreadPairDef:
    """Definition of a spread pair to monitor."""
    name: str                    # e.g., "ETH_BTC"
    asset_a: str                 # e.g., "ETHUSDT"
    asset_b: str                 # e.g., "BTCUSDT"
    always_active: bool = False  # If True, skip cointegration check

    # Dynamic state
    is_cointegrated: bool = False
    p_value: float = 1.0
    hedge_ratio: float = 0.0
    current_zscore: float = 0.0
    last_check_candle: int = 0

    # Strategy instance (created when cointegrated)
    strategy: Optional[SpreadStrategy] = None


class SpreadScanner:
    """
    Scans multiple pairs for cointegration and manages spread strategies.

    Only trades pairs that are currently cointegrated (p < threshold).
    Rechecks cointegration every N candles.
    """

    # Pairs to monitor: (name, asset_a, asset_b, always_active)
    # All pairs now dynamic - only trade when cointegrated (p < 0.05)
    PAIRS = [
        ("ETH_BTC", "ETHUSDT", "BTCUSDT", False),  # Dynamic
        ("SOL_ETH", "SOLUSDT", "ETHUSDT", False),  # Dynamic
        ("SOL_BTC", "SOLUSDT", "BTCUSDT", False),  # Dynamic
    ]

    def __init__(
        self,
        config: BotConfig,
        feeds: Dict[str, DataFeed],
        check_interval: int = 5,          # Candles between cointegration checks (~25 min)
        p_threshold: float = 0.05,        # Max p-value to enable trading
        p_disable_threshold: float = 0.15 # P-value to disable trading
    ):
        self.config = config
        self.feeds = feeds
        self.check_interval = check_interval
        self.p_threshold = p_threshold
        self.p_disable_threshold = p_disable_threshold

        # Initialize pair definitions
        self.pairs: Dict[str, SpreadPairDef] = {}
        for name, asset_a, asset_b, always_active in self.PAIRS:
            self.pairs[name] = SpreadPairDef(
                name=name,
                asset_a=asset_a,
                asset_b=asset_b,
                always_active=always_active
            )

        # Track candles processed
        self.candles_processed = 0

        # Active signals across all pairs
        self.active_signals: List[Tuple[str, SpreadSignal]] = []  # (pair_name, signal)

        # Run initial cointegration check
        self._check_all_cointegration()

    def _get_feed_key(self, symbol: str) -> str:
        """Get feed key for a symbol."""
        return f"{symbol}_{self.config.timeframe}"

    def _check_cointegration(self, pair: SpreadPairDef) -> Tuple[bool, float, float]:
        """
        Check cointegration for a pair.

        Returns: (is_cointegrated, p_value, hedge_ratio)
        """
        key_a = self._get_feed_key(pair.asset_a)
        key_b = self._get_feed_key(pair.asset_b)

        if key_a not in self.feeds or key_b not in self.feeds:
            return False, 1.0, 0.0

        feed_a = self.feeds[key_a]
        feed_b = self.feeds[key_b]

        if len(feed_a.candles) < 60 or len(feed_b.candles) < 60:
            return False, 1.0, 0.0

        # Align candles by time
        times_a = {c.time: c.close for c in feed_a.candles}
        times_b = {c.time: c.close for c in feed_b.candles}

        common_times = sorted(set(times_a.keys()) & set(times_b.keys()))
        if len(common_times) < 60:
            return False, 1.0, 0.0

        # Get last 60 aligned prices (~5 hours) for cointegration test
        use_times = common_times[-60:]
        prices_a = np.array([times_a[t] for t in use_times])
        prices_b = np.array([times_b[t] for t in use_times])

        # Cointegration test
        try:
            _, p_value, _ = coint(prices_a, prices_b)
        except:
            return False, 1.0, 0.0

        # Calculate hedge ratio
        mean_a, mean_b = np.mean(prices_a), np.mean(prices_b)
        cov = np.cov(prices_a, prices_b)[0, 1]
        var_b = np.var(prices_b)
        hedge_ratio = cov / var_b if var_b > 0 else 0.0

        is_cointegrated = p_value < self.p_threshold

        return is_cointegrated, p_value, hedge_ratio

    def _check_all_cointegration(self):
        """Check cointegration for all pairs."""
        print("\n=== COINTEGRATION SCAN ===")

        for name, pair in self.pairs.items():
            was_cointegrated = pair.is_cointegrated
            is_coint, p_val, hedge = self._check_cointegration(pair)

            pair.p_value = p_val
            pair.hedge_ratio = hedge

            # Always-active pairs skip cointegration check
            if pair.always_active:
                if not pair.is_cointegrated:
                    pair.is_cointegrated = True
                    pair.strategy = self._create_strategy(pair)
                    print(f"  {name}: ALWAYS ACTIVE (p={p_val:.4f}, hedge={pair.hedge_ratio:.6f})")
                else:
                    # Don't update hedge ratio on running strategy - it would break z-score
                    # Just report current p-value for informational purposes
                    actual_hedge = pair.strategy.hedge_ratio if pair.strategy else pair.hedge_ratio
                    print(f"  {name}: ALWAYS ACTIVE (p={p_val:.4f}, hedge={actual_hedge:.6f})")
                continue

            # Dynamic pairs use cointegration threshold
            # Hysteresis: harder to enable than disable
            if is_coint and not pair.is_cointegrated:
                # Enable trading
                pair.is_cointegrated = True
                pair.strategy = self._create_strategy(pair)
                print(f"  {name}: ENABLED (p={p_val:.4f}, hedge={hedge:.6f})")
            elif not is_coint and pair.is_cointegrated and p_val > self.p_disable_threshold:
                # Check if there's an active pattern - don't disable mid-pattern
                has_active_pattern = (pair.strategy and
                    pair.strategy.pattern_state.get('phase', 0) > 0)
                if has_active_pattern:
                    print(f"  {name}: PATTERN ACTIVE (p={p_val:.4f}) - keeping open")
                else:
                    # Disable trading (only if p exceeds higher threshold and no active pattern)
                    pair.is_cointegrated = False
                    pair.strategy = None
                    print(f"  {name}: DISABLED (p={p_val:.4f})")
            elif pair.is_cointegrated:
                # Still cointegrated - don't update hedge ratio on running strategy
                actual_hedge = pair.strategy.hedge_ratio if pair.strategy else pair.hedge_ratio
                print(f"  {name}: ACTIVE (p={p_val:.4f}, hedge={actual_hedge:.6f})")
            else:
                print(f"  {name}: inactive (p={p_val:.4f})")

        active_count = sum(1 for p in self.pairs.values() if p.is_cointegrated)
        print(f"Active pairs: {active_count}/{len(self.pairs)}")
        print("=" * 30)

    def _create_strategy(self, pair: SpreadPairDef) -> Optional[SpreadStrategy]:
        """Create a SpreadStrategy for a pair."""
        key_a = self._get_feed_key(pair.asset_a)
        key_b = self._get_feed_key(pair.asset_b)

        if key_a not in self.feeds or key_b not in self.feeds:
            return None

        # Create a custom config for this pair
        from config import SpreadPairConfig
        pair_config = SpreadPairConfig(
            name=pair.name,
            asset_a=pair.asset_a,
            asset_b=pair.asset_b,
            hedge_ratio=pair.hedge_ratio,
        )

        # Create config copy with this pair
        config_copy = BotConfig(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=self.config.testnet,
            spread_trading_enabled=True,
            spread_pair=pair_config,
            risk_per_trade=self.config.risk_per_trade,
        )

        strategy = SpreadStrategy(
            config_copy,
            self.feeds[key_a],
            self.feeds[key_b]
        )
        # Use the strategy's calculated hedge ratio (from OLS regression)
        # instead of overwriting with scanner's covariance-based ratio
        # This ensures consistency between spread_history and future updates
        pair.hedge_ratio = strategy.hedge_ratio

        return strategy

    def update(self, candle_a: Candle, candle_b: Candle, pair_name: str) -> Optional[SpreadSignal]:
        """
        Update a pair with new candles.

        Returns a signal if pattern completes.
        """
        if pair_name not in self.pairs:
            return None

        pair = self.pairs[pair_name]

        if not pair.is_cointegrated or not pair.strategy:
            return None

        # Update strategy
        signal = pair.strategy.update(candle_a, candle_b)

        # Update current z-score
        pair.current_zscore = pair.strategy.get_current_zscore()

        return signal

    def check_periodic_cointegration(self):
        """Check if it's time to recheck cointegration."""
        self.candles_processed += 1

        if self.candles_processed >= self.check_interval:
            self._check_all_cointegration()
            self.candles_processed = 0

    def get_active_pairs(self) -> List[SpreadPairDef]:
        """Get list of currently active (cointegrated) pairs."""
        return [p for p in self.pairs.values() if p.is_cointegrated]

    def get_pair_for_symbols(self, symbol_a: str, symbol_b: str) -> Optional[SpreadPairDef]:
        """Find pair definition for given symbols."""
        for pair in self.pairs.values():
            if (pair.asset_a == symbol_a and pair.asset_b == symbol_b) or \
               (pair.asset_a == symbol_b and pair.asset_b == symbol_a):
                return pair
        return None

    def get_stats(self) -> Dict:
        """Get scanner statistics."""
        return {
            'total_pairs': len(self.pairs),
            'active_pairs': len(self.get_active_pairs()),
            'pairs': {
                name: {
                    'cointegrated': p.is_cointegrated,
                    'p_value': p.p_value,
                    'hedge_ratio': p.hedge_ratio,
                    'zscore': p.current_zscore,
                }
                for name, p in self.pairs.items()
            },
            'check_interval': self.check_interval,
            'candles_until_check': self.check_interval - self.candles_processed,
        }

    def print_status(self):
        """Print current status of all pairs."""
        print("\n" + "=" * 50)
        print("SPREAD SCANNER STATUS")
        print("=" * 50)

        for name, pair in self.pairs.items():
            if pair.always_active:
                status = "🔒 ALWAYS ON"
            elif pair.is_cointegrated:
                status = "✅ ACTIVE"
            else:
                status = "❌ inactive"

            zscore = f"z={pair.current_zscore:.2f}" if pair.is_cointegrated else ""
            phase = ""
            if pair.strategy:
                phase = f"phase={pair.strategy.pattern_state.get('phase', 0)}"

            print(f"{name}: {status} p={pair.p_value:.4f} {zscore} {phase}")

        print(f"\nNext cointegration check in {self.check_interval - self.candles_processed} candles")
        print("=" * 50)

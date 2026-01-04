"""
Spread Bot Configuration

Configuration for the BTC/ETH spread trading bot using MR Double Touch strategy.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv


@dataclass
class SpreadPair:
    """Configuration for a tradeable spread pair."""
    name: str                    # e.g., "ETH_BTC"
    asset_a: str                 # Long leg symbol (e.g., "ETHUSDT")
    asset_b: str                 # Short leg symbol (e.g., "BTCUSDT")
    hedge_ratio: float = 0.054   # Will be recalculated from data
    zscore_window: int = 60      # Rolling window for z-score

    # MR Double Touch parameters
    first_extreme_z: float = 2.0
    recovery_z: float = 1.0
    second_touch_z: float = 1.5
    max_pattern_bars: int = 50

    # Exit parameters
    tp_z: float = 0.5
    sl_z: float = 4.0


@dataclass
class SpreadBotConfig:
    """Main configuration for the spread trading bot."""

    # API credentials
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # Default to testnet for safety

    # Trading parameters
    timeframe: int = 5                    # Minutes
    risk_per_trade: float = 0.02          # 2% risk per spread trade
    max_positions: int = 1                # Max concurrent spread positions

    # Spread pairs to trade
    pairs: List[SpreadPair] = field(default_factory=lambda: [
        SpreadPair(
            name="ETH_BTC",
            asset_a="ETHUSDT",
            asset_b="BTCUSDT",
            hedge_ratio=0.054,  # Initial estimate, recalculated on startup
        )
    ])

    # Hedge ratio settings
    hedge_ratio_lookback: int = 500       # Candles to calculate hedge ratio
    hedge_ratio_update_interval: int = 100  # Recalculate every N candles

    # Timing
    scan_interval: int = 10               # Seconds between scans

    # Notifications
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "SpreadBotConfig":
        """Load configuration from environment variables."""
        load_dotenv()

        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
            timeframe=int(os.getenv("SPREAD_TIMEFRAME", "5")),
            risk_per_trade=float(os.getenv("SPREAD_RISK_PER_TRADE", "0.02")),
            max_positions=int(os.getenv("SPREAD_MAX_POSITIONS", "1")),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def get_pair(self, name: str) -> SpreadPair:
        """Get a spread pair by name."""
        for pair in self.pairs:
            if pair.name == name:
                return pair
        raise ValueError(f"Spread pair '{name}' not found")


# Default configuration
DEFAULT_CONFIG = SpreadBotConfig(
    testnet=True,
    risk_per_trade=0.02,
    max_positions=1,
    pairs=[
        SpreadPair(
            name="ETH_BTC",
            asset_a="ETHUSDT",
            asset_b="BTCUSDT",
            hedge_ratio=0.054,
            zscore_window=60,
            first_extreme_z=2.0,
            recovery_z=1.0,
            second_touch_z=1.5,
            tp_z=0.5,
            sl_z=4.0,
        )
    ]
)


if __name__ == "__main__":
    # Test configuration loading
    config = SpreadBotConfig.from_env()
    print(f"Testnet: {config.testnet}")
    print(f"Timeframe: {config.timeframe}m")
    print(f"Risk per trade: {config.risk_per_trade * 100}%")
    print(f"Pairs: {[p.name for p in config.pairs]}")

    for pair in config.pairs:
        print(f"\n{pair.name}:")
        print(f"  {pair.asset_a} / {pair.asset_b}")
        print(f"  Hedge ratio: {pair.hedge_ratio}")
        print(f"  Z-score window: {pair.zscore_window}")
        print(f"  Pattern: {pair.first_extreme_z}/{pair.recovery_z}/{pair.second_touch_z}")

"""
Bybit Trading Bot Configuration
Double Touch Strategy Bot
"""
import os
from dataclasses import dataclass
from typing import Optional, List

# ==================== ASSET CLASSIFICATION ====================
# Crypto assets (trade on Bybit)
CRYPTO_ASSETS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "SUIUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "NEARUSDT",
    "FILUSDT",
    "INJUSDT",
]

# Non-crypto assets (for future support - Gold, indices, etc.)
NON_CRYPTO_ASSETS = [
    # "XAUUSD",  # Gold - uncomment when adding non-crypto broker
]


def get_asset_type(symbol: str) -> str:
    """Return 'crypto' or 'non_crypto' for position limit tracking."""
    # Strip any suffixes and check
    base_symbol = symbol.upper().replace(".P", "")
    if base_symbol in CRYPTO_ASSETS:
        return "crypto"
    return "non_crypto"


# Default symbols to trade
DEFAULT_SYMBOLS = CRYPTO_ASSETS.copy()

# Additional timeframe setups (symbol -> list of extra timeframes)
# These are IN ADDITION to the default timeframe
EXTRA_TIMEFRAMES = {
    # "ETHUSDT": ["1"],  # Disabled for Double Touch - 5m only
}


@dataclass
class BotConfig:
    # API Configuration
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True  # Use testnet by default for safety

    # Trading Configuration
    symbols: List[str] = None  # List of symbols to trade
    category: str = "linear"  # linear = USDT perpetual
    timeframe: str = "5"  # 5 minute candles

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = DEFAULT_SYMBOLS.copy()

    # Risk Management
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_positions: int = 5  # Max total concurrent positions
    max_crypto_positions: int = 3  # Max crypto positions
    max_non_crypto_positions: int = 2  # Max non-crypto (gold, indices)
    max_daily_loss: float = 0.05  # 5% max daily loss

    # Strategy Settings (King - legacy)
    evwma_length: int = 20
    swing_lookback: int = 3
    fvg_max_wait_candles: int = 20

    # Double Touch Strategy Settings
    ema_fast: int = 9
    ema_med: int = 21
    ema_slow: int = 50
    hh_ll_lookback: int = 20  # Lookback for HH/LL detection
    ewvma_filter_length: int = 200
    use_ewvma_filter: bool = True
    counter_trend_mode: bool = True  # Default: counter-trend (mean reversion)
    risk_reward: float = 3.0  # Target R:R ratio
    sl_buffer_pct: float = 0.001  # 0.1% buffer beyond step 3 for SL

    # Execution
    use_limit_orders: bool = True  # Use limit orders at FVG
    slippage_buffer: float = 0.001  # 0.1% slippage buffer for market orders

    # Notifications
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        # Parse symbols from env (comma-separated) or use defaults
        symbols_env = os.getenv("TRADING_SYMBOLS", "")
        symbols = [s.strip() for s in symbols_env.split(",") if s.strip()] if symbols_env else None

        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
            symbols=symbols,
            timeframe=os.getenv("TRADING_TIMEFRAME", "5"),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            telegram_token=os.getenv("TELEGRAM_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )


# Bybit API endpoints
BYBIT_MAINNET = "https://api.bybit.com"
BYBIT_TESTNET = "https://api-testnet.bybit.com"
BYBIT_WS_MAINNET = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"
BYBIT_WS_PRIVATE_MAINNET = "wss://stream.bybit.com/v5/private"
BYBIT_WS_PRIVATE_TESTNET = "wss://stream-testnet.bybit.com/v5/private"

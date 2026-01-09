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
class BreakawayConfig:
    """Configuration for Breakaway strategy."""
    # Strategy parameters (updated 2026-01-08 - aggressive imbalance mode)
    ewvma_length: int = 20
    ewvma_trend_length: int = 200
    min_vol_ratio: float = 2.0  # Volume spike threshold

    # NEW: Imbalance filter (replaces Tai/Trend) - 2026-01-08
    use_imbalance_filter: bool = True  # Enable volume delta imbalance
    imbalance_threshold: float = 0.10  # Require ±0.10 imbalance
    imbalance_lookback: int = 10       # Lookback for imbalance calc

    # NEW: Real order book confirmation (optional) - 2026-01-08
    use_orderbook_confirm: bool = False  # Disabled by default - testing
    orderbook_threshold: float = 0.05    # Require ±0.05 OB imbalance

    # OLD: Tai/Trend filters (disabled by default - 2026-01-08)
    use_tai_filter: bool = False       # Disabled - replaced by imbalance
    tai_threshold_short: float = 53.0
    tai_threshold_long: float = 47.0
    use_trend_filter: bool = False     # Disabled - replaced by imbalance

    # Cradle settings
    min_cradle_candles: int = 3
    cradle_lookback: int = 5
    risk_reward: float = 3.0
    sl_buffer_pct: float = 0.001

    # Symbol management
    priority_symbols: List[str] = None
    max_symbols: int = 45  # Top 45 coins by volume
    trade_direction: str = "both"  # "both", "shorts", "longs"

    # Trading - 5-minute (default)
    max_positions: int = 5
    risk_per_trade: float = 0.01  # 1% risk per trade

    # 1-minute timeframe settings
    enable_1m: bool = False  # Disabled - 5-min only
    symbols_1m: int = 22
    risk_per_trade_1m: float = 0.01
    max_positions_1m: int = 5
    min_vol_ratio_1m: float = 2.0
    cooldown_1m_minutes: int = 15
    candles_preload: int = 2000

    def __post_init__(self):
        if self.priority_symbols is None:
            self.priority_symbols = ["SOLUSDT", "BTCUSDT", "PNUTUSDT", "DOGEUSDT"]

    @classmethod
    def from_env(cls):
        """Load from environment variables."""
        priority = os.getenv("BREAKAWAY_PRIORITY_SYMBOLS", "SOLUSDT,BTCUSDT,PNUTUSDT,DOGEUSDT")
        priority_list = [s.strip() for s in priority.split(",") if s.strip()]

        return cls(
            ewvma_length=int(os.getenv("BREAKAWAY_EWVMA_LENGTH", "20")),
            ewvma_trend_length=int(os.getenv("BREAKAWAY_EWVMA_TREND_LENGTH", "200")),
            min_vol_ratio=float(os.getenv("BREAKAWAY_MIN_VOL_RATIO", "2.0")),
            # NEW: Imbalance filter
            use_imbalance_filter=os.getenv("BREAKAWAY_USE_IMBALANCE", "true").lower() == "true",
            imbalance_threshold=float(os.getenv("BREAKAWAY_IMBALANCE_THRESHOLD", "0.10")),
            imbalance_lookback=int(os.getenv("BREAKAWAY_IMBALANCE_LOOKBACK", "10")),
            # NEW: Order book confirmation
            use_orderbook_confirm=os.getenv("BREAKAWAY_USE_ORDERBOOK", "false").lower() == "true",
            orderbook_threshold=float(os.getenv("BREAKAWAY_ORDERBOOK_THRESHOLD", "0.05")),
            # OLD: Tai/Trend filters (disabled by default)
            use_tai_filter=os.getenv("BREAKAWAY_USE_TAI", "false").lower() == "true",
            tai_threshold_short=float(os.getenv("BREAKAWAY_TAI_SHORT", "53.0")),
            tai_threshold_long=float(os.getenv("BREAKAWAY_TAI_LONG", "47.0")),
            use_trend_filter=os.getenv("BREAKAWAY_USE_TREND", "false").lower() == "true",
            # Cradle settings
            min_cradle_candles=int(os.getenv("BREAKAWAY_MIN_CRADLE", "3")),
            cradle_lookback=int(os.getenv("BREAKAWAY_CRADLE_LOOKBACK", "5")),
            risk_reward=float(os.getenv("BREAKAWAY_RISK_REWARD", "3.0")),
            sl_buffer_pct=float(os.getenv("BREAKAWAY_SL_BUFFER", "0.001")),
            priority_symbols=priority_list,
            max_symbols=int(os.getenv("BREAKAWAY_MAX_SYMBOLS", "45")),
            trade_direction=os.getenv("BREAKAWAY_DIRECTION", "both"),
            max_positions=int(os.getenv("BREAKAWAY_MAX_POSITIONS", "5")),
            risk_per_trade=float(os.getenv("BREAKAWAY_RISK_PER_TRADE", "0.01")),
            # 1-minute settings
            enable_1m=os.getenv("BREAKAWAY_ENABLE_1M", "false").lower() == "true",
            symbols_1m=int(os.getenv("BREAKAWAY_SYMBOLS_1M", "22")),
            risk_per_trade_1m=float(os.getenv("BREAKAWAY_RISK_1M", "0.01")),
            max_positions_1m=int(os.getenv("BREAKAWAY_MAX_POSITIONS_1M", "5")),
            min_vol_ratio_1m=float(os.getenv("BREAKAWAY_VOL_RATIO_1M", "2.0")),
            cooldown_1m_minutes=int(os.getenv("BREAKAWAY_COOLDOWN_1M", "15")),
            candles_preload=int(os.getenv("BREAKAWAY_CANDLES_PRELOAD", "2000")),
        )


@dataclass
class BreakoutConfig:
    """
    Configuration for Breakout Optimized Strategy.

    Entry: Break above swing high + price above upper EVWMA band
    Exit: ATR trailing stop
    Filters: Volume spike, Volume imbalance (toggleable)
    """
    # Pivot detection
    pivot_left: int = 3
    pivot_right: int = 3

    # EVWMA filter
    evwma_period: int = 20

    # ATR stops
    atr_period: int = 14
    atr_multiplier: float = 2.0

    # Volume filters (toggleable)
    use_volume_filter: bool = True
    min_vol_ratio: float = 2.0
    volume_avg_period: int = 20

    use_imbalance_filter: bool = True
    imbalance_threshold: float = 0.10
    imbalance_lookback: int = 10

    # Timeframe
    timeframe: str = "5"  # "1" or "5"

    # Symbol management
    priority_symbols: List[str] = None
    max_symbols: int = 45

    # Position sizing
    max_positions: int = 5
    risk_per_trade: float = 0.01

    # Emergency take profit (circuit breaker if bot crashes)
    emergency_tp_multiplier: float = 10.0  # 10R emergency TP

    # State persistence
    state_file: str = "data/breakout_signals.json"

    # Historical data
    candles_preload: int = 2000

    def __post_init__(self):
        if self.priority_symbols is None:
            self.priority_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PNUTUSDT", "INJUSDT"]

    @classmethod
    def from_env(cls):
        """Load from BREAKOUT_* environment variables."""
        priority = os.getenv("BREAKOUT_PRIORITY_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,PNUTUSDT,INJUSDT")
        priority_list = [s.strip() for s in priority.split(",") if s.strip()]

        return cls(
            # Pivot detection
            pivot_left=int(os.getenv("BREAKOUT_PIVOT_LEFT", "3")),
            pivot_right=int(os.getenv("BREAKOUT_PIVOT_RIGHT", "3")),
            # EVWMA filter
            evwma_period=int(os.getenv("BREAKOUT_EVWMA_PERIOD", "20")),
            # ATR stops
            atr_period=int(os.getenv("BREAKOUT_ATR_PERIOD", "14")),
            atr_multiplier=float(os.getenv("BREAKOUT_ATR_MULTIPLIER", "2.0")),
            # Volume filters
            use_volume_filter=os.getenv("BREAKOUT_USE_VOLUME_FILTER", "true").lower() == "true",
            min_vol_ratio=float(os.getenv("BREAKOUT_MIN_VOL_RATIO", "2.0")),
            volume_avg_period=int(os.getenv("BREAKOUT_VOLUME_AVG_PERIOD", "20")),
            use_imbalance_filter=os.getenv("BREAKOUT_USE_IMBALANCE_FILTER", "true").lower() == "true",
            imbalance_threshold=float(os.getenv("BREAKOUT_IMBALANCE_THRESHOLD", "0.10")),
            imbalance_lookback=int(os.getenv("BREAKOUT_IMBALANCE_LOOKBACK", "10")),
            # Timeframe
            timeframe=os.getenv("BREAKOUT_TIMEFRAME", "5"),
            # Symbol management
            priority_symbols=priority_list,
            max_symbols=int(os.getenv("BREAKOUT_MAX_SYMBOLS", "45")),
            # Position sizing
            max_positions=int(os.getenv("BREAKOUT_MAX_POSITIONS", "5")),
            risk_per_trade=float(os.getenv("BREAKOUT_RISK_PER_TRADE", "0.01")),
            # Emergency TP
            emergency_tp_multiplier=float(os.getenv("BREAKOUT_EMERGENCY_TP", "10.0")),
            # State persistence
            state_file=os.getenv("BREAKOUT_STATE_FILE", "data/breakout_signals.json"),
            # Historical data
            candles_preload=int(os.getenv("BREAKOUT_CANDLES_PRELOAD", "2000")),
        )


@dataclass
class SpreadPairConfig:
    """Configuration for a spread trading pair."""
    name: str = "ETH_BTC"
    asset_a: str = "ETHUSDT"      # Long leg (buy for long spread)
    asset_b: str = "BTCUSDT"      # Short leg (sell for long spread)
    hedge_ratio: float = 0.054    # Recalculated on startup
    zscore_window: int = 60       # Rolling window for z-score

    # MR Double Touch parameters
    first_extreme_z: float = 2.0
    recovery_z: float = 1.0
    second_touch_z: float = 1.5
    max_pattern_bars: int = 50

    # Exit parameters
    tp_z: float = 0.5
    sl_z: float = 4.0


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

    # ==================== SPREAD TRADING ====================
    spread_trading_enabled: bool = False  # Enable spread trading mode
    spread_pair: SpreadPairConfig = None  # Spread pair config

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = DEFAULT_SYMBOLS.copy()
        if self.spread_pair is None:
            self.spread_pair = SpreadPairConfig()

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
    hh_ll_lookback: int = 50  # Lookback for HH/LL detection (must be HH/LL in last 50 candles)
    ewvma_filter_length: int = 200
    use_ewvma_filter: bool = True
    counter_trend_mode: bool = False  # False = trend-aligned (trade with EWVMA-200)
    risk_reward: float = 3.0  # Target R:R ratio
    sl_buffer_pct: float = 0.001  # 0.1% buffer beyond step 3 for SL

    # HTF Directional Filter Settings
    use_htf_filter: bool = True  # Enable 4H 50 EMA directional filter
    htf_timeframe_minutes: int = 240  # 4H = 240 minutes
    htf_ema_length: int = 50  # EMA length on HTF

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

        # Spread trading config
        spread_enabled = os.getenv("SPREAD_TRADING_ENABLED", "false").lower() == "true"
        spread_pair = SpreadPairConfig(
            name=os.getenv("SPREAD_PAIR_NAME", "ETH_BTC"),
            asset_a=os.getenv("SPREAD_ASSET_A", "ETHUSDT"),
            asset_b=os.getenv("SPREAD_ASSET_B", "BTCUSDT"),
            first_extreme_z=float(os.getenv("SPREAD_FIRST_EXTREME_Z", "2.0")),
            recovery_z=float(os.getenv("SPREAD_RECOVERY_Z", "1.0")),
            second_touch_z=float(os.getenv("SPREAD_SECOND_TOUCH_Z", "1.5")),
            tp_z=float(os.getenv("SPREAD_TP_Z", "0.5")),
            sl_z=float(os.getenv("SPREAD_SL_Z", "4.0")),
        )

        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
            symbols=symbols,
            timeframe=os.getenv("TRADING_TIMEFRAME", "5"),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            spread_trading_enabled=spread_enabled,
            spread_pair=spread_pair,
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

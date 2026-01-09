"""
Scalping Strategy Configuration

High-frequency FVG scalping with partial exits.
Target: 9+ trades/day, 85%+ win rate, +0.8R expectancy
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScalpConfig:
    """Configuration for the scalping strategy."""

    # === SYMBOLS ===
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "SUIUSDT",
        "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "NEARUSDT", "FILUSDT", "INJUSDT"
    ])
    timeframe: str = "5"  # 5-minute candles

    # === ENTRY FILTERS ===
    min_vol_ratio: float = 1.5          # Volume >= 1.5x average
    imbalance_threshold: float = 0.10   # Order flow imbalance threshold
    imbalance_lookback: int = 10        # Candles for imbalance calculation
    min_cradle_candles: int = 3         # Min candles in EWVMA bands
    cradle_lookback: int = 5            # Lookback for cradle check
    sl_buffer_pct: float = 0.001        # 0.1% buffer on stop loss

    # === EXIT SYSTEM (Partial Take-Profit) ===
    tp1_r_multiple: float = 1.0         # First target (close 50%)
    tp2_r_multiple: float = 1.5         # Second target (close remaining)
    tp1_close_pct: float = 0.50         # Close 50% at TP1
    move_sl_to_be: bool = True          # Move SL to breakeven after TP1
    max_hold_candles: int = 30          # Timeout - close at market

    # === RISK MANAGEMENT ===
    risk_per_trade: float = 0.005       # 0.5% risk per trade
    max_positions: int = 5              # Max concurrent positions
    max_per_symbol: int = 1             # Max positions per symbol
    max_daily_loss: float = 0.03        # 3% daily loss limit
    cooldown_candles: int = 5           # Candles to wait after exit

    # === DIRECTION ===
    trade_direction: str = "both"       # "both", "shorts", "longs"

    # === API ===
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True                # Use testnet by default

    # === NOTIFICATIONS ===
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # === DATA ===
    candles_preload: int = 500          # Historical candles to load

    @classmethod
    def from_env(cls) -> "ScalpConfig":
        """Load configuration from environment variables."""
        default_symbols = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,DOTUSDT,SUIUSDT,LTCUSDT,BCHUSDT,ATOMUSDT,UNIUSDT,APTUSDT,ARBUSDT,OPUSDT,NEARUSDT,FILUSDT,INJUSDT"
        symbols_str = os.getenv("SCALP_SYMBOLS", default_symbols)
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

        return cls(
            # Symbols
            symbols=symbols,
            timeframe=os.getenv("SCALP_TIMEFRAME", "5"),

            # Entry filters
            min_vol_ratio=float(os.getenv("SCALP_MIN_VOL_RATIO", "1.5")),
            imbalance_threshold=float(os.getenv("SCALP_IMBALANCE_THRESHOLD", "0.10")),
            imbalance_lookback=int(os.getenv("SCALP_IMBALANCE_LOOKBACK", "10")),
            min_cradle_candles=int(os.getenv("SCALP_MIN_CRADLE", "3")),
            cradle_lookback=int(os.getenv("SCALP_CRADLE_LOOKBACK", "5")),
            sl_buffer_pct=float(os.getenv("SCALP_SL_BUFFER", "0.001")),

            # Exit system
            tp1_r_multiple=float(os.getenv("SCALP_TP1_R", "1.0")),
            tp2_r_multiple=float(os.getenv("SCALP_TP2_R", "1.5")),
            tp1_close_pct=float(os.getenv("SCALP_TP1_CLOSE_PCT", "0.50")),
            move_sl_to_be=os.getenv("SCALP_MOVE_SL_TO_BE", "true").lower() == "true",
            max_hold_candles=int(os.getenv("SCALP_MAX_HOLD", "30")),

            # Risk management
            risk_per_trade=float(os.getenv("SCALP_RISK_PER_TRADE", "0.005")),
            max_positions=int(os.getenv("SCALP_MAX_POSITIONS", "5")),
            max_per_symbol=int(os.getenv("SCALP_MAX_PER_SYMBOL", "1")),
            max_daily_loss=float(os.getenv("SCALP_MAX_DAILY_LOSS", "0.03")),
            cooldown_candles=int(os.getenv("SCALP_COOLDOWN", "5")),

            # Direction
            trade_direction=os.getenv("SCALP_DIRECTION", "both"),

            # API
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",

            # Notifications
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),

            # Data
            candles_preload=int(os.getenv("SCALP_CANDLES_PRELOAD", "500")),
        )

    def __str__(self) -> str:
        return (
            f"ScalpConfig(\n"
            f"  symbols={self.symbols},\n"
            f"  timeframe={self.timeframe},\n"
            f"  vol_ratio>={self.min_vol_ratio}, imbalance>={self.imbalance_threshold},\n"
            f"  TP1={self.tp1_r_multiple}R ({self.tp1_close_pct*100}%), TP2={self.tp2_r_multiple}R,\n"
            f"  risk={self.risk_per_trade*100}%, max_positions={self.max_positions},\n"
            f"  testnet={self.testnet}\n"
            f")"
        )


# Bybit API endpoints
BYBIT_MAINNET = "https://api.bybit.com"
BYBIT_TESTNET = "https://api-testnet.bybit.com"
BYBIT_WS_MAINNET = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"

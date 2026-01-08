"""
Breakaway Strategy Trading Bot - 5-Minute Scanner

Scans top 45 symbols for counter-trend Breakaway signals on 5-minute timeframe.
- 5-min: Top 45 symbols, 2% risk, max 5 positions
- Preloads 2000 candles of historical data
- Checks for new setups on each candle close
"""

import os
import time
import signal
import sys
import atexit
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load .env file before any config imports
load_dotenv()

# ==================== PID LOCK FILE ====================
# Prevents multiple instances from running simultaneously

LOCKFILE = "/tmp/breakaway_bot.lock"


def check_lock():
    """Check if another instance is running. Exit if so."""
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, 'r') as f:
                old_pid = int(f.read().strip())

            # Check if process is still running
            try:
                os.kill(old_pid, 0)  # Signal 0 = check if process exists
                print(f"ERROR: Another instance is already running (PID {old_pid})")
                print(f"If this is incorrect, delete {LOCKFILE} and try again.")
                sys.exit(1)
            except OSError:
                # Process not running, stale lock file
                print(f"Removing stale lock file (PID {old_pid} not running)")
                os.remove(LOCKFILE)
        except (ValueError, IOError) as e:
            print(f"Warning: Invalid lock file, removing: {e}")
            os.remove(LOCKFILE)

    # Create lock file with current PID
    with open(LOCKFILE, 'w') as f:
        f.write(str(os.getpid()))

    # Register cleanup on exit
    atexit.register(remove_lock)
    print(f"Lock acquired (PID {os.getpid()})")


def remove_lock():
    """Remove the lock file on exit."""
    try:
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
            print("Lock released")
    except Exception as e:
        print(f"Warning: Could not remove lock file: {e}")

from typing import Optional, Dict, List
from config import BotConfig, BreakawayConfig
from bybit_client import BybitClient, BybitWebSocket
from data_feed import DataFeed, Candle
from breakaway_strategy import BreakawayStrategy, BreakawaySignal, BreakawaySignalType, BreakawayStatus
from order_manager import OrderManager
from notifier import TelegramNotifier
from trade_tracker import get_tracker
from symbol_scanner import SymbolScanner


def make_setup_key(symbol: str, timeframe: str) -> str:
    """Create a unique key for a symbol+timeframe setup."""
    return f"{symbol}_{timeframe}"


class BreakawayBot:
    """Breakaway Strategy Bot - Multi-timeframe scanner for counter-trend setups."""

    def __init__(self, config: BotConfig, breakaway_config: BreakawayConfig = None):
        self.config = config
        self.breakaway_config = breakaway_config or BreakawayConfig.from_env()
        self.running = False

        # Initialize client
        self.client = BybitClient(config)

        # Initialize notifier
        self.notifier = TelegramNotifier(config)

        # Symbol scanner
        self.symbol_scanner = SymbolScanner(self.client)

        # ==================== 5-MINUTE TIMEFRAME ====================
        self.symbols_5m: List[str] = []
        self.feeds_5m: Dict[str, DataFeed] = {}
        self.strategies_5m: Dict[str, BreakawayStrategy] = {}

        # Shared components
        self.order_manager = OrderManager(config, self.client, self.notifier)
        self.ws: Optional[BybitWebSocket] = None

        # Active signals (keyed by setup_key)
        self.active_signals: Dict[str, BreakawaySignal] = {}

        # State
        self.account_balance = 0.0
        self.account_equity = 0.0

        # Performance tracking
        self.tracker = get_tracker()

        # Stats
        self.signals_5m = 0
        self.executed_5m = 0

    # ==================== SYMBOL MANAGEMENT ====================

    def _refresh_symbols(self):
        """Fetch symbol list for 5-minute timeframe."""
        print("\nFetching top coins by 24h volume...")

        # 5-minute: Top 45 symbols
        top_coins = self.symbol_scanner.get_top_coins(self.breakaway_config.max_symbols)
        self.symbols_5m = self.symbol_scanner.merge_with_priority(
            top_coins,
            self.breakaway_config.priority_symbols
        )
        print(f"5-min: {len(self.symbols_5m)} symbols")

    # ==================== FEED INITIALIZATION ====================

    def _initialize_feeds(self):
        """Initialize data feeds for all 5-minute symbols."""
        candles_to_load = self.breakaway_config.candles_preload

        print(f"\nLoading 5-min data for {len(self.symbols_5m)} symbols ({candles_to_load} candles)...")
        success_5m = 0

        for i, symbol in enumerate(self.symbols_5m):
            setup_key = make_setup_key(symbol, "5")
            try:
                feed = DataFeed(self.config, self.client, symbol, timeframe="5")
                feed.load_historical(candles_to_load)
                self.feeds_5m[setup_key] = feed

                # Create strategy with 5-min params
                strategy = BreakawayStrategy(
                    symbol=symbol,
                    timeframe="5",
                    min_vol_ratio=self.breakaway_config.min_vol_ratio,
                    # NEW: Imbalance filter
                    use_imbalance_filter=self.breakaway_config.use_imbalance_filter,
                    imbalance_threshold=self.breakaway_config.imbalance_threshold,
                    imbalance_lookback=self.breakaway_config.imbalance_lookback,
                    # OLD: Tai/Trend filters (optional)
                    use_tai_filter=self.breakaway_config.use_tai_filter,
                    tai_threshold_short=self.breakaway_config.tai_threshold_short,
                    tai_threshold_long=self.breakaway_config.tai_threshold_long,
                    use_trend_filter=self.breakaway_config.use_trend_filter,
                    # Unchanged
                    min_cradle_candles=self.breakaway_config.min_cradle_candles,
                    cradle_lookback=self.breakaway_config.cradle_lookback,
                    risk_reward=self.breakaway_config.risk_reward,
                    sl_buffer_pct=self.breakaway_config.sl_buffer_pct,
                    trade_direction=self.breakaway_config.trade_direction,
                )
                self.strategies_5m[setup_key] = strategy
                success_5m += 1

                if (i + 1) % 10 == 0:
                    print(f"  5-min: {i + 1}/{len(self.symbols_5m)} loaded...")

                time.sleep(0.2)

            except Exception as e:
                print(f"  Error loading 5-min {symbol}: {e}")

        print(f"5-min: {success_5m}/{len(self.symbols_5m)} symbols loaded")

    # ==================== WEBSOCKET ====================

    def _connect_websocket(self):
        """Connect WebSocket and subscribe to all 5-minute symbols."""
        subscriptions = []

        # 5-minute subscriptions
        for symbol in self.symbols_5m:
            subscriptions.append((symbol, "5"))

        self.ws = BybitWebSocket(
            self.config,
            on_kline=self._on_kline,
            subscriptions=subscriptions
        )
        self.ws.connect()

        print(f"\nWebSocket connected:")
        print(f"  5-min feeds: {len(self.symbols_5m)}")

    def _on_kline(self, data: Dict):
        """Handle incoming 5-minute kline data."""
        symbol = data.get("symbol")
        timeframe = data.get("timeframe", "5")
        is_new_candle = data.get("confirm", False)

        setup_key = make_setup_key(symbol, timeframe)

        # Process 5-minute candles
        if timeframe == "5" and setup_key in self.feeds_5m:
            self.feeds_5m[setup_key].update_candle(data)
            if is_new_candle:
                self._on_new_candle_5m(symbol, setup_key)

    # ==================== 5-MINUTE CANDLE PROCESSING ====================

    def _on_new_candle_5m(self, symbol: str, setup_key: str):
        """Process new confirmed 5-minute candle."""
        # Log candle close for debugging
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] 5m candle close: {symbol}")

        # Check position limits
        positions_5m = self._get_5m_position_count()
        if positions_5m >= self.breakaway_config.max_positions:
            return

        # Skip if already have position for this symbol
        if self.order_manager.has_position(symbol):
            return

        # Scan for signals
        self._scan_symbol_5m(setup_key)

    def _scan_symbol_5m(self, setup_key: str):
        """Scan a 5-minute symbol for Breakaway signals."""
        if setup_key not in self.feeds_5m or setup_key not in self.strategies_5m:
            return

        feed = self.feeds_5m[setup_key]
        strategy = self.strategies_5m[setup_key]

        if len(feed.candles) < 350:
            return

        # Extract arrays
        opens = np.array([c.open for c in feed.candles])
        closes = np.array([c.close for c in feed.candles])
        highs = np.array([c.high for c in feed.candles])
        lows = np.array([c.low for c in feed.candles])
        volumes = np.array([c.volume for c in feed.candles])
        times = np.array([c.time for c in feed.candles])

        signal = strategy.scan_for_signals(opens, closes, highs, lows, volumes, times)

        if signal:
            self._handle_signal_5m(signal)

    def _handle_signal_5m(self, signal: BreakawaySignal):
        """Handle a new 5-minute Breakaway signal."""
        self.signals_5m += 1

        print(f"\n{'='*60}")
        print(f"[5-MIN] NEW BREAKAWAY SIGNAL - {signal.symbol}")
        print(f"{'='*60}")
        print(f"  Direction: {signal.direction.upper()}")
        print(f"  Entry: {signal.entry_price:.6f}")
        print(f"  Stop Loss: {signal.stop_loss:.6f}")
        print(f"  Target: {signal.target:.6f}")
        print(f"  R:R: {signal.rr_ratio:.1f}")
        print(f"  Volume: {signal.vol_ratio:.1f}x")
        print(f"  Imbalance: {signal.imbalance:+.2f}")
        print(f"{'='*60}")

        self._notify_signal(signal, "5-MIN")
        self._execute_signal_5m(signal)

    def _execute_signal_5m(self, signal: BreakawaySignal):
        """Execute a 5-minute Breakaway signal."""
        self._update_account_balance()

        # Optional: Order book confirmation
        if self.breakaway_config.use_orderbook_confirm:
            ob_imb = self.client.get_orderbook_imbalance(signal.symbol, depth=50)
            if ob_imb is not None:
                threshold = self.breakaway_config.orderbook_threshold
                if signal.direction == "short" and ob_imb > -threshold:
                    print(f"  [5-MIN] Skipped: Order book not confirming SHORT (OB={ob_imb:+.2f})")
                    return
                elif signal.direction == "long" and ob_imb < threshold:
                    print(f"  [5-MIN] Skipped: Order book not confirming LONG (OB={ob_imb:+.2f})")
                    return
                print(f"  [5-MIN] Order book confirmed: {ob_imb:+.2f}")

        setup_key = signal.setup_key
        can_trade, reason = self.order_manager.can_open_trade(
            self.account_balance,
            setup_key=setup_key,
            symbol=signal.symbol
        )
        if not can_trade:
            print(f"  Cannot execute 5-min: {reason}")
            return

        # Calculate position size using 1% risk
        risk_amount = self.account_balance * self.breakaway_config.risk_per_trade
        risk_distance = abs(signal.stop_loss - signal.entry_price)

        if risk_distance <= 0:
            print(f"  Invalid risk distance")
            return

        position_size = risk_amount / risk_distance
        side = "Sell" if signal.direction == "short" else "Buy"

        try:
            order = self.order_manager.place_order(
                symbol=signal.symbol,
                side=side,
                qty=position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.target,
                order_type="Limit",
                signal_type=f"breakaway_5m_{signal.signal_type.value}",
            )

            if order:
                self.executed_5m += 1
                signal.status = BreakawayStatus.FILLED
                self.active_signals[setup_key] = signal
                print(f"  [5-MIN] Order placed: {side} {position_size:.6f} @ {signal.entry_price:.6f}")

        except Exception as e:
            print(f"  [5-MIN] Order failed: {e}")

    # ==================== POSITION TRACKING ====================

    def _get_5m_position_count(self) -> int:
        """Count open 5-minute positions."""
        count = 0
        for trade in self.order_manager.active_trades:
            if hasattr(trade.signal, 'setup_key') and "_5" in str(trade.signal.setup_key):
                count += 1
            elif hasattr(trade.signal, 'setup_key') and "5m" in str(getattr(trade, 'signal_type', '')):
                count += 1
        return count

    # ==================== NOTIFICATIONS ====================

    def _notify_signal(self, signal: BreakawaySignal, timeframe_label: str):
        """Send Telegram notification for new signal."""
        direction_emoji = "🔴" if signal.direction == "short" else "🟢"
        msg = (
            f"{direction_emoji} <b>[{timeframe_label}] BREAKAWAY SIGNAL</b>\n\n"
            f"Symbol: <b>{signal.symbol}</b>\n"
            f"Direction: <b>{signal.direction.upper()}</b>\n"
            f"Entry: {signal.entry_price:.6f}\n"
            f"Stop Loss: {signal.stop_loss:.6f}\n"
            f"Target: {signal.target:.6f}\n"
            f"R:R: {signal.rr_ratio:.1f}\n\n"
            f"Volume: {signal.vol_ratio:.1f}x\n"
            f"Imbalance: {signal.imbalance:+.2f}\n"
            f"Cradle: {signal.cradle_count}/5"
        )
        self.notifier.send(msg)

    # ==================== UTILITY ====================

    def _update_account_balance(self):
        """Update account balance and equity."""
        try:
            self.account_balance = self.client.get_available_balance()
            self.account_equity = self.client.get_equity()
        except Exception as e:
            print(f"Error updating balance: {e}")

    def _print_status(self):
        """Print current bot status."""
        self._update_account_balance()

        print(f"\n{'='*60}")
        print(f"BREAKAWAY BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        print(f"\n5-MIN TIMEFRAME:")
        print(f"  Symbols: {len(self.feeds_5m)}")
        print(f"  Positions: {self._get_5m_position_count()}/{self.breakaway_config.max_positions}")
        print(f"  Signals: {self.signals_5m} | Executed: {self.executed_5m}")

        print(f"\nACCOUNT:")
        print(f"  Balance: ${self.account_balance:.2f}")
        print(f"  Equity: ${self.account_equity:.2f}")
        print(f"  Total Open: {self.order_manager.get_open_count()}")
        print(f"{'='*60}")

    # ==================== MAIN LOOP ====================

    def _run_loop(self):
        """Main event loop."""
        self.running = True
        last_status = 0
        last_sync = 0
        status_interval = 300  # 5 minutes
        sync_interval = 30  # 30 seconds

        print("\nBot running. Press Ctrl+C to stop.")

        while self.running:
            try:
                now = time.time()

                if now - last_status > status_interval:
                    self._print_status()
                    last_status = now

                if now - last_sync > sync_interval:
                    self.order_manager.sync_positions()
                    last_sync = now

                time.sleep(1)

            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n\nShutting down...")
        self.running = False

        if self.ws:
            self.ws.close()

        self.notifier.send("🛑 <b>BREAKAWAY BOT STOPPED</b>")
        print("Shutdown complete.")
        sys.exit(0)

    def start(self):
        """Start the trading bot."""
        print("=" * 60)
        print("BREAKAWAY STRATEGY BOT - 5-Minute Scanner")
        print("=" * 60)
        print(f"\n5-MIN CONFIG:")
        print(f"  Symbols: Top {self.breakaway_config.max_symbols}")
        print(f"  Risk: {self.breakaway_config.risk_per_trade * 100}%")
        print(f"  Max positions: {self.breakaway_config.max_positions}")
        print(f"  Volume filter: {self.breakaway_config.min_vol_ratio}x")
        print(f"  Candles preload: {self.breakaway_config.candles_preload}")
        print(f"\nTestnet: {self.config.testnet}")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount equity: ${self.account_equity:.2f}")

        # Fetch symbols
        self._refresh_symbols()

        # Initialize feeds
        self._initialize_feeds()

        # Connect WebSocket
        self._connect_websocket()

        # Startup notification
        msg = (
            f"🤖 <b>BREAKAWAY BOT STARTED</b>\n\n"
            f"<b>5-MIN:</b> {len(self.symbols_5m)} symbols\n"
            f"Risk: {self.breakaway_config.risk_per_trade*100}%\n"
            f"Max positions: {self.breakaway_config.max_positions}\n\n"
            f"Balance: ${self.account_balance:.2f}"
        )
        self.notifier.send(msg)

        # Initial status
        self._print_status()

        # Run main loop
        self._run_loop()


def main():
    """Main entry point."""
    # Prevent multiple instances
    check_lock()

    config = BotConfig.from_env()
    breakaway_config = BreakawayConfig.from_env()

    bot = BreakawayBot(config, breakaway_config)
    bot.start()


if __name__ == "__main__":
    main()

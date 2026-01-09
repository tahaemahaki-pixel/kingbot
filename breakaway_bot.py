"""
Breakout Optimized Trading Bot

Entry: Break above swing high + price above upper EVWMA(20) band
Exit: ATR(14) * 2.0 trailing stop
Filters: Volume spike (2x avg), Volume imbalance (10% threshold) - toggleable

Configurable timeframe via BREAKOUT_TIMEFRAME env var (default: 5)
"""

import os
import time
import signal
import sys
import atexit
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file before any config imports
load_dotenv()

# ==================== PID LOCK FILE ====================
# Prevents multiple instances from running simultaneously

LOCKFILE = "/tmp/breakout_bot.lock"


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
from config import BotConfig, BreakoutConfig
from bybit_client import BybitClient, BybitWebSocket
from data_feed import DataFeed, Candle
from breakout_strategy import BreakoutStrategy, BreakoutSignal, BreakoutSignalType, BreakoutStatus, BreakoutIndicators
from order_manager import OrderManager
from notifier import TelegramNotifier
from trade_tracker import get_tracker
from symbol_scanner import SymbolScanner


def make_setup_key(symbol: str, timeframe: str) -> str:
    """Create a unique key for a symbol+timeframe setup."""
    return f"{symbol}_{timeframe}"


class BreakoutBot:
    """Breakout Optimized Bot - Trend-following breakout strategy with ATR trailing stops."""

    def __init__(self, config: BotConfig, breakout_config: BreakoutConfig = None):
        self.config = config
        self.breakout_config = breakout_config or BreakoutConfig.from_env()
        self.running = False

        # Default timeframe (5-min for most symbols)
        self.timeframe = self.breakout_config.timeframe

        # 1-minute symbols (if enabled)
        self.symbols_1m: List[str] = []
        if self.breakout_config.enable_1m and self.breakout_config.symbols_1m:
            self.symbols_1m = self.breakout_config.symbols_1m

        # Initialize client
        self.client = BybitClient(config)

        # Initialize notifier
        self.notifier = TelegramNotifier(config)

        # Symbol scanner
        self.symbol_scanner = SymbolScanner(self.client)

        # Data feeds and strategies (keyed by setup_key)
        self.symbols: List[str] = []  # All symbols (5-min)
        self.feeds: Dict[str, DataFeed] = {}
        self.strategies: Dict[str, BreakoutStrategy] = {}

        # Shared components
        self.order_manager = OrderManager(config, self.client, self.notifier)
        self.ws: Optional[BybitWebSocket] = None

        # Active signals (keyed by setup_key) - for trailing stop management
        self.active_signals: Dict[str, BreakoutSignal] = {}

        # State
        self.account_balance = 0.0
        self.account_equity = 0.0

        # Performance tracking
        self.tracker = get_tracker()

        # Stats (per timeframe)
        self.signals_count = 0
        self.executed_count = 0
        self.signals_count_1m = 0
        self.executed_count_1m = 0

        # State persistence path
        self.state_file = Path(self.breakout_config.state_file)

    # ==================== STATE PERSISTENCE ====================

    def _save_signals(self):
        """Save active signals to disk for crash recovery."""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert signals to serializable format
            data = {
                setup_key: sig.to_dict()
                for setup_key, sig in self.active_signals.items()
            }

            # Write atomically (write to temp, then rename)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.state_file)

            logger_msg = f"Saved {len(data)} active signals to {self.state_file}"
            print(f"  [State] {logger_msg}")

        except Exception as e:
            print(f"  [State] Error saving signals: {e}")

    def _load_signals(self):
        """Load active signals from disk on startup."""
        if not self.state_file.exists():
            print(f"  [State] No saved signals found at {self.state_file}")
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            if not data:
                print("  [State] No signals in state file")
                return

            # Import here to avoid circular imports
            from breakout_strategy import BreakoutSignal

            loaded_count = 0
            for setup_key, sig_dict in data.items():
                try:
                    signal = BreakoutSignal.from_dict(sig_dict)
                    self.active_signals[setup_key] = signal
                    loaded_count += 1
                except Exception as e:
                    print(f"  [State] Error loading signal {setup_key}: {e}")

            print(f"  [State] Loaded {loaded_count} signals from {self.state_file}")

        except Exception as e:
            print(f"  [State] Error loading signals: {e}")

    def _sync_signals_with_positions(self):
        """
        Sync loaded signals with actual exchange positions.
        Remove signals for positions that no longer exist.
        """
        try:
            positions = self.client.get_positions()
            position_symbols = {p.symbol for p in positions if p.size > 0}

            stale_keys = []
            for setup_key, signal in self.active_signals.items():
                if signal.symbol not in position_symbols:
                    print(f"  [State] Signal {setup_key} has no matching position - removing")
                    stale_keys.append(setup_key)

            for key in stale_keys:
                del self.active_signals[key]

            if stale_keys:
                self._save_signals()
                print(f"  [State] Removed {len(stale_keys)} stale signals")

            # Log active signals that have matching positions
            active_count = len(self.active_signals)
            if active_count > 0:
                print(f"  [State] {active_count} signals synced with open positions")
                for setup_key, signal in self.active_signals.items():
                    print(f"    - {signal.symbol}: entry={signal.entry_price:.4f}, "
                          f"trailing_stop={signal.trailing_stop:.4f}")

        except Exception as e:
            print(f"  [State] Error syncing with positions: {e}")

    def _cancel_orphan_orders(self):
        """
        Cancel all pending limit orders that don't have matching open positions.
        This prevents order accumulation across bot restarts.
        """
        try:
            # Get all open orders and positions
            open_orders = self.client.get_open_orders()
            positions = self.client.get_positions()
            position_symbols = {p.symbol for p in positions if p.size > 0}

            if not open_orders:
                print("  [Orders] No pending orders found")
                return

            # Cancel orders for symbols without open positions
            cancelled = 0
            for order in open_orders:
                # Keep orders for symbols with open positions (those are SL/TP orders)
                if order.symbol in position_symbols:
                    continue

                # Cancel entry orders for symbols without positions
                try:
                    self.client.cancel_order(order.symbol, order.order_id)
                    print(f"  [Orders] Cancelled orphan order: {order.symbol} {order.side} @ {order.price}")
                    cancelled += 1
                except Exception as e:
                    print(f"  [Orders] Failed to cancel {order.symbol}: {e}")

            if cancelled > 0:
                print(f"  [Orders] Cancelled {cancelled} orphan orders")
            else:
                print(f"  [Orders] No orphan orders to cancel ({len(open_orders)} orders belong to open positions)")

        except Exception as e:
            print(f"  [Orders] Error cancelling orphan orders: {e}")

    # ==================== SYMBOL MANAGEMENT ====================

    def _refresh_symbols(self):
        """Fetch symbol list."""
        print("\nFetching top coins by 24h volume...")

        top_coins = self.symbol_scanner.get_top_coins(self.breakout_config.max_symbols)
        self.symbols = self.symbol_scanner.merge_with_priority(
            top_coins,
            self.breakout_config.priority_symbols
        )
        print(f"{self.timeframe}-min: {len(self.symbols)} symbols")

    # ==================== FEED INITIALIZATION ====================

    def _initialize_feeds(self):
        """Initialize data feeds for all symbols (5-min) and 1-min symbols if enabled."""
        candles_to_load = self.breakout_config.candles_preload

        # ---- 5-MINUTE FEEDS ----
        print(f"\nLoading {self.timeframe}-min data for {len(self.symbols)} symbols ({candles_to_load} candles)...")
        success = 0

        for i, symbol in enumerate(self.symbols):
            setup_key = make_setup_key(symbol, self.timeframe)
            try:
                feed = DataFeed(self.config, self.client, symbol, timeframe=self.timeframe)
                feed.load_historical(candles_to_load)
                self.feeds[setup_key] = feed

                # Create strategy
                strategy = BreakoutStrategy(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    config=self.breakout_config
                )
                self.strategies[setup_key] = strategy
                success += 1

                if (i + 1) % 10 == 0:
                    print(f"  {self.timeframe}-min: {i + 1}/{len(self.symbols)} loaded...")

                time.sleep(0.2)

            except Exception as e:
                print(f"  Error loading {self.timeframe}-min {symbol}: {e}")

        print(f"{self.timeframe}-min: {success}/{len(self.symbols)} symbols loaded")

        # ---- 1-MINUTE FEEDS (for specific symbols) ----
        if self.symbols_1m:
            print(f"\nLoading 1-min data for {len(self.symbols_1m)} symbols ({candles_to_load} candles)...")
            success_1m = 0

            for i, symbol in enumerate(self.symbols_1m):
                setup_key = make_setup_key(symbol, "1")
                try:
                    feed = DataFeed(self.config, self.client, symbol, timeframe="1")
                    feed.load_historical(candles_to_load)
                    self.feeds[setup_key] = feed

                    # Create strategy for 1-min
                    strategy = BreakoutStrategy(
                        symbol=symbol,
                        timeframe="1",
                        config=self.breakout_config
                    )
                    self.strategies[setup_key] = strategy
                    success_1m += 1

                    time.sleep(0.2)

                except Exception as e:
                    print(f"  Error loading 1-min {symbol}: {e}")

            print(f"1-min: {success_1m}/{len(self.symbols_1m)} symbols loaded")

    # ==================== WEBSOCKET ====================

    def _connect_websocket(self):
        """Connect WebSocket and subscribe to all symbols (both timeframes)."""
        # 5-minute subscriptions for all symbols
        subscriptions = [(symbol, self.timeframe) for symbol in self.symbols]

        # Add 1-minute subscriptions for specific symbols
        if self.symbols_1m:
            for symbol in self.symbols_1m:
                subscriptions.append((symbol, "1"))

        self.ws = BybitWebSocket(
            self.config,
            on_kline=self._on_kline,
            subscriptions=subscriptions
        )
        self.ws.connect()

        print(f"\nWebSocket connected:")
        print(f"  {self.timeframe}-min feeds: {len(self.symbols)}")
        if self.symbols_1m:
            print(f"  1-min feeds: {len(self.symbols_1m)} ({', '.join(self.symbols_1m)})")

    def _on_kline(self, data: Dict):
        """Handle incoming kline data."""
        symbol = data.get("symbol")
        timeframe = data.get("timeframe", self.timeframe)
        is_new_candle = data.get("confirm", False)

        setup_key = make_setup_key(symbol, timeframe)

        # Update feed
        if setup_key in self.feeds:
            self.feeds[setup_key].update_candle(data)
            if is_new_candle:
                self._on_new_candle(symbol, setup_key)

    # ==================== CANDLE PROCESSING ====================

    def _on_new_candle(self, symbol: str, setup_key: str):
        """Process new confirmed candle."""
        # Extract actual timeframe from setup_key
        tf = setup_key.split("_")[-1] if "_" in setup_key else self.timeframe

        # Log candle close for debugging
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {tf}m candle close: {symbol}")

        # First, update trailing stops for any open positions
        self._update_trailing_stops()

        # Check if max FILLED positions reached - cancel pending orders if so
        filled_count = self.order_manager.get_filled_position_count()
        if filled_count >= self.breakout_config.max_positions:
            pending_count = self.order_manager.get_pending_order_count()
            if pending_count > 0:
                print(f"\n  Max positions ({filled_count}) filled - cancelling {pending_count} pending orders...")
                cancelled = self.order_manager.cancel_all_pending_entry_orders()
                if cancelled > 0:
                    self._cancel_orphan_orders()  # Also cancel on exchange
            return

        # Skip if already have position for this symbol
        if self.order_manager.has_position(symbol):
            return

        # Scan for new signals
        self._scan_symbol(setup_key)

    def _scan_symbol(self, setup_key: str):
        """Scan a symbol for breakout signals."""
        if setup_key not in self.feeds or setup_key not in self.strategies:
            return

        feed = self.feeds[setup_key]
        strategy = self.strategies[setup_key]

        if len(feed.candles) < 100:
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
            self._handle_signal(signal)

    def _handle_signal(self, signal: BreakoutSignal):
        """Handle a new breakout signal."""
        # Extract timeframe from setup_key
        tf = signal.setup_key.split("_")[-1] if "_" in signal.setup_key else self.timeframe

        if tf == "1":
            self.signals_count_1m += 1
        else:
            self.signals_count += 1

        print(f"\n{'='*60}")
        print(f"[{tf}-MIN] NEW BREAKOUT SIGNAL - {signal.symbol}")
        print(f"{'='*60}")
        print(f"  Direction: LONG")
        print(f"  Entry: {signal.entry_price:.6f}")
        print(f"  Initial Stop: {signal.initial_stop:.6f}")
        print(f"  Emergency TP: {signal.take_profit:.6f} ({self.breakout_config.emergency_tp_multiplier}R)")
        print(f"  Risk: {signal.risk:.6f}")
        print(f"  Volume: {signal.vol_ratio:.1f}x")
        print(f"  Imbalance: {signal.imbalance:+.2f}")
        print(f"{'='*60}")

        self._notify_signal(signal)
        self._execute_signal(signal)

    def _execute_signal(self, signal: BreakoutSignal):
        """Execute a breakout signal."""
        self._update_account_balance()

        setup_key = signal.setup_key
        can_trade, reason = self.order_manager.can_open_trade(
            self.account_balance,
            setup_key=setup_key,
            symbol=signal.symbol
        )
        if not can_trade:
            print(f"  Cannot execute: {reason}")
            return

        # Calculate position size using configured risk
        risk_amount = self.account_balance * self.breakout_config.risk_per_trade
        risk_distance = signal.risk

        if risk_distance <= 0:
            print(f"  Invalid risk distance")
            return

        position_size = risk_amount / risk_distance

        try:
            # Place order with SL and emergency TP (circuit breaker if bot crashes)
            order = self.order_manager.place_order(
                symbol=signal.symbol,
                side="Buy",  # Longs only for now
                qty=position_size,
                price=signal.entry_price,
                stop_loss=signal.initial_stop,
                take_profit=signal.take_profit,  # Emergency TP as circuit breaker
                order_type="Limit",
                signal_type=f"breakout_{self.timeframe}m_long",
            )

            if order:
                # Track execution by timeframe
                tf = setup_key.split("_")[-1] if "_" in setup_key else self.timeframe
                if tf == "1":
                    self.executed_count_1m += 1
                else:
                    self.executed_count += 1

                signal.status = BreakoutStatus.FILLED
                self.active_signals[setup_key] = signal

                # Persist signal to disk for crash recovery
                self._save_signals()

                print(f"  Order placed: BUY {position_size:.6f} @ {signal.entry_price:.6f}")
                print(f"  Initial SL: {signal.initial_stop:.6f}")
                print(f"  Emergency TP: {signal.take_profit:.6f}")

        except Exception as e:
            print(f"  Order failed: {e}")

    # ==================== TRAILING STOP MANAGEMENT ====================

    def _update_trailing_stops(self):
        """Update trailing stops for all open positions."""
        signals_changed = False

        for setup_key, signal in list(self.active_signals.items()):
            if signal.status != BreakoutStatus.FILLED:
                continue

            # Check if we still have position
            symbol = signal.symbol
            if not self.order_manager.has_position(symbol):
                # Position closed (hit stop or TP, or was manually closed)
                print(f"  Position closed for {symbol}")
                del self.active_signals[setup_key]
                signals_changed = True
                continue

            # Get current candle data
            if setup_key not in self.feeds:
                continue

            feed = self.feeds[setup_key]
            if len(feed.candles) < 2:
                continue

            # Calculate current ATR
            highs = np.array([c.high for c in feed.candles])
            lows = np.array([c.low for c in feed.candles])
            closes = np.array([c.close for c in feed.candles])

            atr = BreakoutIndicators.calculate_atr(
                highs, lows, closes,
                self.breakout_config.atr_period
            )
            current_atr = atr[-1]

            if np.isnan(current_atr):
                continue

            current_high = feed.candles[-1].high
            strategy = self.strategies[setup_key]

            # Update trailing stop
            new_stop = strategy.update_trailing_stop(signal, current_high, current_atr)

            if new_stop > signal.trailing_stop:
                # Stop moved up - update on exchange
                signal.trailing_stop = new_stop
                signals_changed = True

                try:
                    success = self.order_manager.modify_stop_loss(symbol, new_stop)
                    if success:
                        print(f"  [{symbol}] Trailing stop updated: {new_stop:.6f}")
                except Exception as e:
                    print(f"  [{symbol}] Failed to update trailing stop: {e}")

        # Persist signal changes to disk
        if signals_changed:
            self._save_signals()

    # ==================== NOTIFICATIONS ====================

    def _notify_signal(self, signal: BreakoutSignal):
        """Send Telegram notification for new signal."""
        vol_filter = "ON" if self.breakout_config.use_volume_filter else "OFF"
        imb_filter = "ON" if self.breakout_config.use_imbalance_filter else "OFF"

        msg = (
            f"🟢 <b>[{self.timeframe}-MIN] BREAKOUT SIGNAL</b>\n\n"
            f"Symbol: <b>{signal.symbol}</b>\n"
            f"Direction: <b>LONG</b>\n"
            f"Entry: {signal.entry_price:.6f}\n"
            f"Initial Stop: {signal.initial_stop:.6f}\n"
            f"Emergency TP: {signal.take_profit:.6f} ({self.breakout_config.emergency_tp_multiplier}R)\n"
            f"Risk: {signal.risk:.6f}\n\n"
            f"Volume: {signal.vol_ratio:.1f}x (filter: {vol_filter})\n"
            f"Imbalance: {signal.imbalance:+.2f} (filter: {imb_filter})\n\n"
            f"<i>Trailing stop will follow price up</i>"
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

        vol_filter = "ON" if self.breakout_config.use_volume_filter else "OFF"
        imb_filter = "ON" if self.breakout_config.use_imbalance_filter else "OFF"

        print(f"\n{'='*60}")
        print(f"BREAKOUT BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        filled = self.order_manager.get_filled_position_count()
        pending = self.order_manager.get_pending_order_count()

        # Count feeds by timeframe
        feeds_5m = len([k for k in self.feeds.keys() if k.endswith(f"_{self.timeframe}")])
        feeds_1m = len([k for k in self.feeds.keys() if k.endswith("_1")])

        print(f"\n{self.timeframe}-MIN TIMEFRAME:")
        print(f"  Symbols: {feeds_5m}")
        print(f"  Signals: {self.signals_count} | Executed: {self.executed_count}")

        if self.symbols_1m:
            print(f"\n1-MIN TIMEFRAME:")
            print(f"  Symbols: {feeds_1m} ({', '.join(self.symbols_1m)})")
            print(f"  Signals: {self.signals_count_1m} | Executed: {self.executed_count_1m}")

        print(f"\nPOSITIONS:")
        print(f"  Filled: {filled}/{self.breakout_config.max_positions}")
        print(f"  Pending orders: {pending}")
        print(f"  Active signals: {len(self.active_signals)}")

        print(f"\nFILTERS:")
        print(f"  Volume spike: {vol_filter} (>= {self.breakout_config.min_vol_ratio}x)")
        print(f"  Imbalance: {imb_filter} (>= {self.breakout_config.imbalance_threshold})")

        print(f"\nACCOUNT:")
        print(f"  Balance: ${self.account_balance:.2f}")
        print(f"  Equity: ${self.account_equity:.2f}")
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

        # Save signals before exit (for crash recovery)
        if self.active_signals:
            print("Saving active signals for recovery...")
            self._save_signals()

        if self.ws:
            self.ws.close()

        self.notifier.send("🛑 <b>BREAKOUT BOT STOPPED</b>")
        print("Shutdown complete.")
        sys.exit(0)

    def start(self):
        """Start the trading bot."""
        vol_filter = "ON" if self.breakout_config.use_volume_filter else "OFF"
        imb_filter = "ON" if self.breakout_config.use_imbalance_filter else "OFF"

        print("=" * 60)
        print("BREAKOUT OPTIMIZED BOT")
        print("=" * 60)
        print(f"\nCONFIG:")
        print(f"  Default timeframe: {self.timeframe}-minute")
        if self.symbols_1m:
            print(f"  1-minute symbols: {', '.join(self.symbols_1m)}")
        print(f"  Symbols: Top {self.breakout_config.max_symbols}")
        print(f"  Risk: {self.breakout_config.risk_per_trade * 100}%")
        print(f"  Max positions: {self.breakout_config.max_positions}")
        print(f"  Candles preload: {self.breakout_config.candles_preload}")
        print(f"\nSTRATEGY:")
        print(f"  Entry: Swing high breakout + above EVWMA({self.breakout_config.evwma_period}) upper band")
        print(f"  Exit: ATR({self.breakout_config.atr_period}) x {self.breakout_config.atr_multiplier} trailing stop")
        print(f"  Emergency TP: {self.breakout_config.emergency_tp_multiplier}R (circuit breaker)")
        print(f"  Volume filter: {vol_filter} (>= {self.breakout_config.min_vol_ratio}x)")
        print(f"  Imbalance filter: {imb_filter} (>= {self.breakout_config.imbalance_threshold})")
        print(f"\nTestnet: {self.config.testnet}")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount equity: ${self.account_equity:.2f}")

        # Load persisted signals from previous session (crash recovery)
        print("\n--- STATE RECOVERY ---")
        self._load_signals()
        self._sync_signals_with_positions()
        self._cancel_orphan_orders()  # Cancel any stale limit orders from previous runs

        # Fetch symbols
        self._refresh_symbols()

        # Initialize feeds
        self._initialize_feeds()

        # Connect WebSocket
        self._connect_websocket()

        # Startup notification
        msg = (
            f"🤖 <b>BREAKOUT BOT STARTED</b>\n\n"
            f"<b>{self.timeframe}-MIN:</b> {len(self.symbols)} symbols\n"
        )
        if self.symbols_1m:
            msg += f"<b>1-MIN:</b> {', '.join(self.symbols_1m)}\n"
        msg += (
            f"Risk: {self.breakout_config.risk_per_trade*100}%\n"
            f"Max positions: {self.breakout_config.max_positions}\n"
            f"Volume filter: {vol_filter}\n"
            f"Imbalance filter: {imb_filter}\n\n"
            f"Balance: ${self.account_balance:.2f}"
        )
        self.notifier.send(msg)

        # Initial status
        self._print_status()

        # Run main loop
        self._run_loop()


# Alias for backwards compatibility
BreakawayBot = BreakoutBot


def main():
    """Main entry point."""
    # Prevent multiple instances
    check_lock()

    config = BotConfig.from_env()
    breakout_config = BreakoutConfig.from_env()

    bot = BreakoutBot(config, breakout_config)
    bot.start()


if __name__ == "__main__":
    main()

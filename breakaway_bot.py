"""
Breakaway Strategy Trading Bot - Multi-Symbol Scanner

Scans 50+ symbols for counter-trend Breakaway signals.
Entry: FVG from EWVMA cradle with volume spike and Tai Index confirmation.
"""

import os
import time
import signal
import sys
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load .env file before any config imports
load_dotenv()

from typing import Optional, Dict, List, Tuple
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
    """Breakaway Strategy Bot - scans multiple symbols for counter-trend setups."""

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

        # Active symbols
        self.symbols: List[str] = []
        self.timeframe = config.timeframe

        # Per-symbol components
        self.feeds: Dict[str, DataFeed] = {}
        self.strategies: Dict[str, BreakawayStrategy] = {}

        # Shared components
        self.order_manager = OrderManager(config, self.client, self.notifier)
        self.ws: Optional[BybitWebSocket] = None

        # Active signals
        self.active_signals: Dict[str, BreakawaySignal] = {}

        # State
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.candles_since_scan: Dict[str, int] = {}

        # Performance tracking
        self.tracker = get_tracker()
        self.last_equity_snapshot = 0.0

        # Stats
        self.total_signals = 0
        self.signals_executed = 0

    def _refresh_symbols(self):
        """Fetch and merge symbol list."""
        print("\nFetching top coins by 24h volume...")
        top_coins = self.symbol_scanner.get_top_coins(self.breakaway_config.max_symbols)
        self.symbols = self.symbol_scanner.merge_with_priority(
            top_coins,
            self.breakaway_config.priority_symbols
        )
        print(f"Trading {len(self.symbols)} symbols")
        print(f"Priority: {self.breakaway_config.priority_symbols}")

    def _initialize_feeds(self):
        """Initialize data feeds for all symbols."""
        print(f"\nLoading historical data for {len(self.symbols)} symbols...")

        success_count = 0
        for i, symbol in enumerate(self.symbols):
            setup_key = make_setup_key(symbol, self.timeframe)
            try:
                # Create feed
                feed = DataFeed(self.config, self.client, symbol, timeframe=self.timeframe)
                feed.load_historical(1000)

                self.feeds[setup_key] = feed
                self.candles_since_scan[setup_key] = 0

                # Create strategy
                strategy = BreakawayStrategy(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    min_vol_ratio=self.breakaway_config.min_vol_ratio,
                    tai_threshold_short=self.breakaway_config.tai_threshold_short,
                    tai_threshold_long=self.breakaway_config.tai_threshold_long,
                    min_cradle_candles=self.breakaway_config.min_cradle_candles,
                    cradle_lookback=self.breakaway_config.cradle_lookback,
                    risk_reward=self.breakaway_config.risk_reward,
                    sl_buffer_pct=self.breakaway_config.sl_buffer_pct,
                    trade_direction=self.breakaway_config.trade_direction,
                )
                self.strategies[setup_key] = strategy

                success_count += 1

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Loaded {i + 1}/{len(self.symbols)} symbols...")

                # Rate limiting
                time.sleep(0.2)

            except Exception as e:
                print(f"  Error loading {symbol}: {e}")

        print(f"Successfully loaded {success_count}/{len(self.symbols)} symbols")

    def _connect_websocket(self):
        """Connect WebSocket and subscribe to all symbols."""
        subscriptions = [(symbol, self.timeframe) for symbol in self.symbols]

        self.ws = BybitWebSocket(
            self.config,
            on_kline=self._on_kline,
            subscriptions=subscriptions
        )
        self.ws.connect()
        print(f"\nWebSocket connected - subscribed to {len(subscriptions)} symbols")

    def _on_kline(self, data: Dict):
        """Handle incoming kline data from WebSocket."""
        symbol = data.get("symbol")
        setup_key = make_setup_key(symbol, self.timeframe)

        if setup_key not in self.feeds:
            return

        # Update feed
        feed = self.feeds[setup_key]
        is_new_candle = data.get("confirm", False)

        feed.update_candle(data)

        # Process new confirmed candle
        if is_new_candle:
            self._on_new_candle(symbol, setup_key)

    def _on_new_candle(self, symbol: str, setup_key: str):
        """Process new confirmed candle."""
        # Skip if at max positions
        open_count = self.order_manager.get_open_count()
        if open_count >= self.breakaway_config.max_positions:
            return

        # Skip if already have position for this symbol
        if self.order_manager.has_position(symbol):
            return

        # Scan for signals
        self._scan_symbol(setup_key)

    def _scan_symbol(self, setup_key: str):
        """Scan a symbol for Breakaway signals."""
        if setup_key not in self.feeds or setup_key not in self.strategies:
            return

        feed = self.feeds[setup_key]
        strategy = self.strategies[setup_key]

        if len(feed.candles) < 300:
            return

        # Extract arrays from candles
        closes = np.array([c.close for c in feed.candles])
        highs = np.array([c.high for c in feed.candles])
        lows = np.array([c.low for c in feed.candles])
        volumes = np.array([c.volume for c in feed.candles])
        times = np.array([c.time for c in feed.candles])

        # Scan for signals
        signal = strategy.scan_for_signals(closes, highs, lows, volumes, times)

        if signal:
            self._handle_signal(signal)

    def _handle_signal(self, signal: BreakawaySignal):
        """Handle a new Breakaway signal."""
        self.total_signals += 1

        print(f"\n{'='*60}")
        print(f"NEW BREAKAWAY SIGNAL - {signal.symbol}")
        print(f"{'='*60}")
        print(f"  Direction: {signal.direction.upper()}")
        print(f"  Entry: {signal.entry_price:.6f}")
        print(f"  Stop Loss: {signal.stop_loss:.6f}")
        print(f"  Target: {signal.target:.6f}")
        print(f"  R:R: {signal.rr_ratio:.1f}")
        print(f"  Volume: {signal.vol_ratio:.1f}x")
        print(f"  Tai Index: {signal.tai_index:.0f}")
        print(f"  Cradle: {signal.cradle_count}/5")
        print(f"{'='*60}")

        # Send Telegram notification
        self._notify_signal(signal)

        # Execute immediately
        self._execute_signal(signal)

    def _execute_signal(self, signal: BreakawaySignal):
        """Execute a Breakaway signal."""
        # Update balance first
        self._update_account_balance()

        # Check if we can trade
        setup_key = f"{signal.symbol}_{self.timeframe}"
        can_trade, reason = self.order_manager.can_open_trade(
            self.account_balance,
            setup_key=setup_key,
            symbol=signal.symbol
        )
        if not can_trade:
            print(f"  Cannot execute: {reason}")
            return

        # Calculate position size
        risk_amount = self.account_balance * self.breakaway_config.risk_per_trade
        risk_distance = abs(signal.stop_loss - signal.entry_price)

        if risk_distance <= 0:
            print(f"  Invalid risk distance")
            return

        position_size = risk_amount / risk_distance

        # Execute trade
        side = "Sell" if signal.direction == "short" else "Buy"

        try:
            # Place order
            order = self.order_manager.place_order(
                symbol=signal.symbol,
                side=side,
                qty=position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.target,
                order_type="Limit",
                signal_type=signal.signal_type.value,
            )

            if order:
                self.signals_executed += 1
                signal.status = BreakawayStatus.FILLED
                self.active_signals[signal.setup_key] = signal
                print(f"  Order placed: {side} {position_size:.6f} @ {signal.entry_price:.6f}")

        except Exception as e:
            print(f"  Order failed: {e}")

    def _notify_signal(self, signal: BreakawaySignal):
        """Send Telegram notification for new signal."""
        direction_emoji = "🔴" if signal.direction == "short" else "🟢"
        msg = (
            f"{direction_emoji} <b>BREAKAWAY SIGNAL</b>\n\n"
            f"Symbol: <b>{signal.symbol}</b>\n"
            f"Direction: <b>{signal.direction.upper()}</b>\n"
            f"Entry: {signal.entry_price:.6f}\n"
            f"Stop Loss: {signal.stop_loss:.6f}\n"
            f"Target: {signal.target:.6f}\n"
            f"R:R: {signal.rr_ratio:.1f}\n\n"
            f"Volume: {signal.vol_ratio:.1f}x\n"
            f"Tai Index: {signal.tai_index:.0f}\n"
            f"Cradle: {signal.cradle_count}/5"
        )
        self.notifier.send(msg)

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
        print(f"Symbols: {len(self.symbols)}")
        print(f"Active feeds: {len(self.feeds)}")
        print(f"Open positions: {self.order_manager.get_open_count()}/{self.breakaway_config.max_positions}")
        print(f"Total signals: {self.total_signals}")
        print(f"Executed: {self.signals_executed}")
        print(f"Balance: ${self.account_balance:.2f}")
        print(f"Equity: ${self.account_equity:.2f}")
        print(f"{'='*60}")

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

                # Print status periodically
                if now - last_status > status_interval:
                    self._print_status()
                    last_status = now

                # Sync positions periodically
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

        # Close WebSocket
        if self.ws:
            self.ws.close()

        # Send shutdown notification
        self.notifier.send("🛑 <b>BREAKAWAY BOT STOPPED</b>")

        print("Shutdown complete.")
        sys.exit(0)

    def start(self):
        """Start the trading bot."""
        print("=" * 60)
        print("BREAKAWAY STRATEGY BOT - Counter-Trend FVG Trading")
        print("=" * 60)
        print(f"Timeframe: {self.timeframe}m")
        print(f"Direction: {self.breakaway_config.trade_direction}")
        print(f"Volume threshold: {self.breakaway_config.min_vol_ratio}x")
        print(f"Tai thresholds: Short > {self.breakaway_config.tai_threshold_short}, Long < {self.breakaway_config.tai_threshold_long}")
        print(f"Risk per trade: {self.breakaway_config.risk_per_trade * 100}%")
        print(f"Max positions: {self.breakaway_config.max_positions}")
        print(f"Risk/Reward: {self.breakaway_config.risk_reward}:1")
        print(f"Testnet: {self.config.testnet}")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount equity: ${self.account_equity:.2f}")

        # Fetch symbols
        self._refresh_symbols()

        # Initialize feeds and strategies
        self._initialize_feeds()

        # Connect WebSocket
        self._connect_websocket()

        # Send startup notification
        self.notifier.send(
            f"🤖 <b>BREAKAWAY BOT STARTED</b>\n\n"
            f"Symbols: {len(self.symbols)}\n"
            f"Direction: {self.breakaway_config.trade_direction}\n"
            f"Balance: ${self.account_balance:.2f}"
        )

        # Initial status
        self._print_status()

        # Run main loop
        self._run_loop()


def main():
    """Main entry point."""
    # Load configs
    config = BotConfig.from_env()
    breakaway_config = BreakawayConfig.from_env()

    # Create and start bot
    bot = BreakawayBot(config, breakaway_config)
    bot.start()


if __name__ == "__main__":
    main()

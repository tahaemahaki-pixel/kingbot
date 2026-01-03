"""
King Strategy Trading Bot - Multi-Symbol Scanner
"""
import time
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, List
from config import BotConfig
from bybit_client import BybitClient, BybitWebSocket
from data_feed import DataFeed, Candle
from strategy import KingStrategy, SignalStatus, TradeSignal
from order_manager import OrderManager
from notifier import TelegramNotifier


class TradingBot:
    """Main trading bot orchestrator - scans multiple symbols."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False

        # Initialize client
        self.client = BybitClient(config)

        # Initialize notifier
        self.notifier = TelegramNotifier(config)

        # Per-symbol components
        self.feeds: Dict[str, DataFeed] = {}
        self.strategies: Dict[str, KingStrategy] = {}

        # Shared components
        self.order_manager = OrderManager(config, self.client, self.notifier)
        self.ws: Optional[BybitWebSocket] = None

        # All active signals across symbols
        self.all_signals: List[TradeSignal] = []

        # State
        self.account_balance = 0.0
        self.candles_since_scan: Dict[str, int] = {}

    def start(self):
        """Start the trading bot."""
        print("=" * 60)
        print("King Strategy Trading Bot - Multi-Symbol Scanner")
        print("=" * 60)
        print(f"Symbols: {len(self.config.symbols)} coins")
        print(f"Timeframe: {self.config.timeframe}m")
        print(f"Testnet: {self.config.testnet}")
        print(f"Risk per trade: {self.config.risk_per_trade * 100}%")
        print(f"Max positions: {self.config.max_positions}")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount balance: ${self.account_balance:.2f}")

        # Initialize feeds and strategies for each symbol
        print(f"\nLoading historical data for {len(self.config.symbols)} symbols...")
        for symbol in self.config.symbols:
            try:
                feed = DataFeed(self.config, self.client, symbol)
                feed.load_historical(200)

                strategy = KingStrategy(
                    feed,
                    swing_lookback=self.config.swing_lookback,
                    max_wait_candles=self.config.fvg_max_wait_candles
                )

                self.feeds[symbol] = feed
                self.strategies[symbol] = strategy
                self.candles_since_scan[symbol] = 0

                print(f"  ✓ {symbol}: {len(feed.candles)} candles loaded")

                # Set max leverage per symbol
                MAX_LEVERAGE = {
                    'BTCUSDT': 100, 'ETHUSDT': 100, 'SOLUSDT': 75, 'XRPUSDT': 75,
                    'DOGEUSDT': 75, 'ADAUSDT': 75, 'AVAXUSDT': 50, 'LINKUSDT': 50,
                    'DOTUSDT': 50, 'SUIUSDT': 50, 'LTCUSDT': 50, 'BCHUSDT': 50,
                    'ATOMUSDT': 50, 'UNIUSDT': 50, 'APTUSDT': 50, 'ARBUSDT': 50,
                    'OPUSDT': 50, 'NEARUSDT': 50, 'FILUSDT': 25, 'INJUSDT': 50,
                }
                try:
                    self.client.set_leverage(symbol, MAX_LEVERAGE.get(symbol, 50))
                except:
                    pass  # Leverage might already be set

                # Rate limit - don't hammer the API
                time.sleep(0.1)

            except Exception as e:
                print(f"  ✗ {symbol}: Failed to load - {e}")

        print(f"\nLoaded {len(self.feeds)} symbols successfully")

        # Initial pattern scan
        print("\nScanning for existing patterns...")
        self._scan_all_patterns()

        # Connect WebSocket
        print("\nConnecting to real-time feed...")
        self.ws = BybitWebSocket(self.config, symbols=list(self.feeds.keys()), on_kline=self._on_kline)
        self.ws.connect()

        self.running = True
        print("\nBot is running. Press Ctrl+C to stop.\n")

        # Send Telegram notification
        self.notifier.notify_bot_started(len(self.feeds), self.account_balance)

        # Main loop
        self._run_loop()

    def _run_loop(self):
        """Main bot loop."""
        last_sync = time.time()
        last_stats = time.time()

        while self.running:
            try:
                current_time = time.time()

                # Sync positions every 30 seconds
                if current_time - last_sync >= 30:
                    self.order_manager.sync_positions()
                    last_sync = current_time

                # Print stats every 5 minutes
                if current_time - last_stats >= 300:
                    self._print_stats()
                    last_stats = current_time

                time.sleep(1)

            except Exception as e:
                print(f"Loop error: {e}")
                time.sleep(5)

    def _on_kline(self, data: dict):
        """Handle new kline data from WebSocket."""
        symbol = data.get("symbol")
        if not symbol or symbol not in self.feeds:
            return

        # Update the correct feed
        self.feeds[symbol].update_candle(data)

        # Only process on confirmed candles
        if not data.get("confirm"):
            return

        new_candle = self.feeds[symbol].candles[-1]
        self._on_new_candle(symbol, new_candle)

    def _on_new_candle(self, symbol: str, candle: Candle):
        """Process a new confirmed candle for a symbol."""
        timestamp = datetime.fromtimestamp(candle.time / 1000).strftime("%H:%M")

        # Update active signals for this symbol
        strategy = self.strategies[symbol]
        strategy.update_signals(candle)

        # Check for ready signals and execute
        ready_signals = strategy.get_ready_signals()
        for sig in ready_signals:
            self._update_account_balance()
            trade = self.order_manager.execute_signal(sig, self.account_balance)
            if trade:
                strategy.remove_signal(sig)
                print(f"[{timestamp}] {symbol} -> Executed {sig.signal_type.value} @ {sig.entry_price:.2f}")

        # Scan for new patterns every 5 candles per symbol
        self.candles_since_scan[symbol] = self.candles_since_scan.get(symbol, 0) + 1
        if self.candles_since_scan[symbol] >= 5:
            self._scan_symbol_patterns(symbol)
            self.candles_since_scan[symbol] = 0

    def _scan_all_patterns(self):
        """Scan for patterns across all symbols."""
        total_signals = 0
        for symbol in self.feeds:
            count = self._scan_symbol_patterns(symbol, quiet=True)
            total_signals += count

        print(f"Found {total_signals} signals across {len(self.feeds)} symbols")
        self._print_active_signals()

    def _scan_symbol_patterns(self, symbol: str, quiet: bool = False) -> int:
        """Scan for new King patterns on a symbol."""
        strategy = self.strategies[symbol]
        new_signals = strategy.scan_for_patterns()

        for sig in new_signals:
            strategy.add_signal(sig)
            if not quiet:
                print(f"  [{symbol}] {sig.signal_type.value}: Entry={sig.entry_price:.2f} SL={sig.stop_loss:.2f} TP={sig.target:.2f} R:R={sig.get_risk_reward():.2f}")

        return len(new_signals)

    def _print_active_signals(self):
        """Print all active signals."""
        all_active = []
        for symbol, strategy in self.strategies.items():
            all_active.extend(strategy.active_signals)

        if not all_active:
            print("No active signals")
            return

        print(f"\nActive signals ({len(all_active)}):")
        # Sort by R:R
        all_active.sort(key=lambda s: s.get_risk_reward(), reverse=True)
        for sig in all_active[:10]:  # Show top 10
            print(f"  [{sig.symbol}] {sig.signal_type.value}: Entry={sig.entry_price:.4f} R:R={sig.get_risk_reward():.2f}")

    def _update_account_balance(self):
        """Update account balance from exchange (uses AVAILABLE balance, not total)."""
        try:
            # Use available balance to account for margin already in use
            self.account_balance = self.client.get_available_balance()
        except Exception as e:
            print(f"Balance update error: {e}")

    def _print_stats(self):
        """Print trading statistics."""
        stats = self.order_manager.get_stats()

        # Count active signals across all symbols
        total_signals = sum(len(s.active_signals) for s in self.strategies.values())

        print("\n" + "=" * 40)
        print("Trading Stats")
        print("=" * 40)
        print(f"Symbols monitored: {len(self.feeds)}")
        print(f"Active signals: {total_signals}")
        print(f"Open trades: {stats.get('open_trades', 0)}")
        print(f"Total trades: {stats.get('total_trades', 0)}")
        print(f"Win rate: {stats.get('win_rate', 0) * 100:.1f}%")
        print(f"Daily P&L: ${stats.get('daily_pnl', 0):.2f}")
        print(f"Total P&L: ${stats.get('total_pnl', 0):.2f}")
        print("=" * 40 + "\n")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\nShutting down...")
        self.stop()

    def stop(self):
        """Stop the trading bot."""
        self.running = False

        # Close WebSocket
        if self.ws:
            self.ws.disconnect()

        # Cancel pending orders for all symbols
        self.order_manager.cancel_pending_orders(list(self.feeds.keys()))

        # Print final stats
        self._print_stats()

        # Send Telegram notification
        self.notifier.notify_bot_stopped()

        print("Bot stopped.")
        sys.exit(0)


def main():
    """Main entry point."""
    # Load config from environment or use defaults
    config = BotConfig.from_env()

    # Override for testing
    if not config.api_key:
        print("WARNING: No API key found. Using testnet defaults.")
        config.testnet = True

    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()

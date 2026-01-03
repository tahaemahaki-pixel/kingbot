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

        # Check for existing positions (recovery after restart)
        self._recover_existing_positions()

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

    def _recover_existing_positions(self):
        """Check for existing positions and pending orders from previous session."""
        from order_manager import Trade, TradeStatus
        from strategy import TradeSignal, SignalType, SignalStatus
        from data_feed import FVG

        recovered_count = 0

        # 1. Recover open positions
        try:
            positions = self.client.get_positions()
            open_positions = [p for p in positions if p.size > 0]

            if open_positions:
                print(f"\nRecovering {len(open_positions)} existing position(s)...")

                for pos in open_positions:
                    symbol = pos.symbol
                    is_long = pos.side == "Buy"

                    dummy_signal = TradeSignal(
                        signal_type=SignalType.LONG_KING if is_long else SignalType.SHORT_KING,
                        status=SignalStatus.FILLED,
                        symbol=symbol,
                        a_index=0, a_price=0,
                        c_index=0, c_price=pos.take_profit or pos.entry_price * (1.05 if is_long else 0.95),
                        d_index=0,
                        e_index=0, e_price=pos.stop_loss or pos.entry_price * (0.95 if is_long else 1.05),
                        e_candle_open=pos.entry_price,
                        f_index=0,
                        fvg=FVG(0, 'bullish' if is_long else 'bearish', pos.entry_price, pos.entry_price, 0),
                        entry_price=pos.entry_price,
                        stop_loss=pos.stop_loss or pos.entry_price * (0.95 if is_long else 1.05),
                        target=pos.take_profit or pos.entry_price * (1.05 if is_long else 0.95),
                        created_at=0
                    )

                    trade = Trade(
                        signal=dummy_signal,
                        status=TradeStatus.OPEN,
                        entry_order_id="recovered",
                        entry_filled_price=pos.entry_price,
                        position_size=pos.size,
                        opened_at=time.time() - 3600
                    )

                    self.order_manager.active_trades.append(trade)
                    direction = "LONG" if is_long else "SHORT"
                    print(f"  ✓ {symbol}: {direction} {pos.size} @ ${pos.entry_price:.4f} (PnL: ${pos.unrealized_pnl:.2f})")
                    recovered_count += 1

        except Exception as e:
            print(f"Position recovery error: {e}")

        # 2. Recover pending limit orders
        try:
            pending_orders = []
            for symbol in self.config.symbols:
                orders = self.client.get_open_orders(symbol)
                # Only recover limit orders with a price (not SL/TP conditional orders)
                for o in orders:
                    if o.price > 0 and o.status in ('New', 'PartiallyFilled'):
                        pending_orders.append(o)

            if pending_orders:
                print(f"\nRecovering {len(pending_orders)} pending order(s)...")

                for order in pending_orders:
                    symbol = order.symbol
                    is_long = order.side == "Buy"

                    dummy_signal = TradeSignal(
                        signal_type=SignalType.LONG_KING if is_long else SignalType.SHORT_KING,
                        status=SignalStatus.FILLED,
                        symbol=symbol,
                        a_index=0, a_price=0,
                        c_index=0, c_price=order.price * (1.05 if is_long else 0.95),
                        d_index=0,
                        e_index=0, e_price=order.price * (0.95 if is_long else 1.05),
                        e_candle_open=order.price,
                        f_index=0,
                        fvg=FVG(0, 'bullish' if is_long else 'bearish', order.price, order.price, 0),
                        entry_price=order.price,
                        stop_loss=order.price * (0.95 if is_long else 1.05),
                        target=order.price * (1.05 if is_long else 0.95),
                        created_at=0
                    )

                    trade = Trade(
                        signal=dummy_signal,
                        status=TradeStatus.PENDING_FILL,
                        entry_order_id=order.order_id,
                        entry_filled_price=None,
                        position_size=order.qty,
                        opened_at=time.time()  # Treat as just placed for expiration tracking
                    )

                    self.order_manager.active_trades.append(trade)
                    direction = "LONG" if is_long else "SHORT"
                    print(f"  ✓ {symbol}: {direction} {order.qty} @ ${order.price:.4f} (pending)")
                    recovered_count += 1

        except Exception as e:
            print(f"Order recovery error: {e}")

        if recovered_count > 0:
            print(f"\nRecovered {recovered_count} total trade(s)")
        else:
            print("\nNo existing positions or orders to recover")

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

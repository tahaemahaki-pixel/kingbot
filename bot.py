"""
Double Touch Strategy Trading Bot - Multi-Symbol, Multi-Timeframe Scanner
Supports both single-asset Double Touch and spread trading modes.
"""
import time
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from config import BotConfig, EXTRA_TIMEFRAMES
from bybit_client import BybitClient, BybitWebSocket
from data_feed import DataFeed, Candle
from double_touch_strategy import DoubleTouchStrategy, SignalStatus, TradeSignal
from order_manager import OrderManager
from notifier import TelegramNotifier
from trade_tracker import get_tracker


def make_setup_key(symbol: str, timeframe: str) -> str:
    """Create a unique key for a symbol+timeframe setup."""
    return f"{symbol}_{timeframe}"


def parse_setup_key(key: str) -> Tuple[str, str]:
    """Parse a setup key into (symbol, timeframe)."""
    parts = key.rsplit("_", 1)
    return parts[0], parts[1]


class TradingBot:
    """Double Touch trading bot - scans multiple symbols and timeframes."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False

        # Initialize client
        self.client = BybitClient(config)

        # Initialize notifier
        self.notifier = TelegramNotifier(config)

        # Build list of all setups: (symbol, timeframe) pairs
        self.setups: List[Tuple[str, str]] = []

        # Spread trading mode
        self.spread_mode = config.spread_trading_enabled
        self.spread_scanner = None  # Multi-pair scanner
        self.last_candles: Dict[str, Candle] = {}  # Last candle per symbol

        # Spread symbols to monitor
        self.spread_symbols = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

        if not self.spread_mode:
            # Normal mode: trade multiple symbols
            for symbol in config.symbols:
                # Default timeframe for all symbols
                self.setups.append((symbol, config.timeframe))
                # Extra timeframes for specific symbols
                if symbol in EXTRA_TIMEFRAMES:
                    for tf in EXTRA_TIMEFRAMES[symbol]:
                        self.setups.append((symbol, tf))
        else:
            # Spread mode: load all spread symbols
            for symbol in self.spread_symbols:
                self.setups.append((symbol, config.timeframe))

        # Per-setup components (keyed by "SYMBOL_TIMEFRAME")
        self.feeds: Dict[str, DataFeed] = {}
        self.strategies: Dict[str, DoubleTouchStrategy] = {}

        # Shared components
        self.order_manager = OrderManager(config, self.client, self.notifier)
        self.ws: Optional[BybitWebSocket] = None

        # All active signals across setups
        self.all_signals: List[TradeSignal] = []

        # State
        self.account_balance = 0.0  # Available balance (for position sizing)
        self.account_equity = 0.0   # Total equity (for display)
        self.candles_since_scan: Dict[str, int] = {}

        # Performance tracking
        self.tracker = get_tracker()
        self.last_equity_snapshot = 0.0

    def start(self):
        """Start the trading bot."""
        print("=" * 60)
        if self.spread_mode:
            print("SPREAD SCANNER BOT - Multi-Pair MR Double Touch")
            print("=" * 60)
            print(f"Mode: DYNAMIC SPREAD TRADING")
            print(f"Pairs: ETH/BTC, SOL/ETH, SOL/BTC")
            print(f"Cointegration check: every 500 candles")
            print(f"P-value threshold: < 0.05 to enable")
            pair = self.config.spread_pair
            print(f"Pattern: z={pair.first_extreme_z} -> z={pair.recovery_z} -> z={pair.second_touch_z}")
            print(f"TP: z={pair.tp_z}, SL: z={pair.sl_z}")
        else:
            print("Double Touch Strategy Bot - Multi-Symbol, Multi-Timeframe")
            print("=" * 60)
            print(f"Setups: {len(self.setups)} (symbol+timeframe combinations)")
            for sym, tf in self.setups:
                print(f"  - {sym} @ {tf}m")
        print(f"Testnet: {self.config.testnet}")
        print(f"Risk per trade: {self.config.risk_per_trade * 100}%")
        if not self.spread_mode:
            print(f"Max positions: {self.config.max_positions} (crypto: {self.config.max_crypto_positions}, non-crypto: {self.config.max_non_crypto_positions})")
            print(f"Risk/Reward: {self.config.risk_reward}:1")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount equity: ${self.account_equity:.2f}")

        # Initialize feeds and strategies for each setup
        print(f"\nLoading historical data for {len(self.setups)} setups...")
        unique_symbols = set()
        for symbol, timeframe in self.setups:
            setup_key = make_setup_key(symbol, timeframe)
            try:
                feed = DataFeed(self.config, self.client, symbol, timeframe=timeframe)
                feed.load_historical(500)

                self.feeds[setup_key] = feed
                self.candles_since_scan[setup_key] = 0

                # Only create Double Touch strategies in normal mode
                if not self.spread_mode:
                    strategy = DoubleTouchStrategy(
                        feed,
                        risk_reward=self.config.risk_reward,
                        sl_buffer_pct=self.config.sl_buffer_pct,
                        use_ewvma_filter=self.config.use_ewvma_filter,
                        counter_trend_mode=self.config.counter_trend_mode,
                        max_wait_candles=self.config.fvg_max_wait_candles
                    )
                    self.strategies[setup_key] = strategy

                print(f"  ✓ {symbol}@{timeframe}m: {len(feed.candles)} candles loaded")

                # Set max leverage per symbol (only once per symbol)
                if symbol not in unique_symbols:
                    unique_symbols.add(symbol)
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
                print(f"  ✗ {symbol}@{timeframe}m: Failed to load - {e}")

        print(f"\nLoaded {len(self.feeds)} setups successfully")

        # Initialize spread strategy if in spread mode
        if self.spread_mode:
            self._init_spread_strategy()

        # Check for existing positions (recovery after restart)
        self._recover_existing_positions()

        # Initial pattern scan
        print("\nScanning for existing patterns...")
        self._scan_all_patterns()

        # Connect WebSocket with all setup subscriptions
        print("\nConnecting to real-time feed...")
        self.ws = BybitWebSocket(self.config, on_kline=self._on_kline, subscriptions=self.setups)
        self.ws.connect()

        self.running = True
        print("\nBot is running. Press Ctrl+C to stop.\n")

        # Send Telegram notification
        self.notifier.notify_bot_started(len(self.feeds), self.account_equity, self.spread_mode)

        # Main loop
        self._run_loop()

    def _run_loop(self):
        """Main bot loop."""
        last_sync = time.time()
        last_stats = time.time()

        # Record initial equity snapshot
        self._record_equity_snapshot()

        while self.running:
            try:
                current_time = time.time()

                # Sync positions every 30 seconds
                if current_time - last_sync >= 30:
                    self.order_manager.sync_positions()
                    last_sync = current_time

                # Print stats and record equity every 5 minutes
                if current_time - last_stats >= 300:
                    self._print_stats()
                    self._record_equity_snapshot()
                    last_stats = current_time

                time.sleep(1)

            except Exception as e:
                print(f"Loop error: {e}")
                time.sleep(5)

    def _on_kline(self, data: dict):
        """Handle new kline data from WebSocket."""
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        if not symbol or not timeframe:
            return

        # Build the setup key
        setup_key = make_setup_key(symbol, timeframe)
        if setup_key not in self.feeds:
            return

        # Update the correct feed
        self.feeds[setup_key].update_candle(data)

        # Only process on confirmed candles
        if not data.get("confirm"):
            return

        new_candle = self.feeds[setup_key].candles[-1]
        self._on_new_candle(setup_key, new_candle)

    def _on_new_candle(self, setup_key: str, candle: Candle):
        """Process a new confirmed candle for a setup (symbol+timeframe)."""
        timestamp = datetime.fromtimestamp(candle.time / 1000).strftime("%H:%M")

        # Handle spread mode differently
        if self.spread_mode:
            self._on_new_candle_spread(setup_key, candle, timestamp)
            return

        # Normal mode: Update active signals for this setup
        strategy = self.strategies[setup_key]
        strategy.update_signals(candle)

        # Check for ready signals and execute
        ready_signals = strategy.get_ready_signals()
        for sig in ready_signals:
            self._update_account_balance()
            trade = self.order_manager.execute_signal(sig, self.account_balance)
            if trade:
                strategy.remove_signal(sig)
                print(f"[{timestamp}] {setup_key} -> Executed {sig.signal_type.value} @ {sig.entry_price:.2f}")

        # Scan for new patterns every 5 candles per setup
        self.candles_since_scan[setup_key] = self.candles_since_scan.get(setup_key, 0) + 1
        if self.candles_since_scan[setup_key] >= 5:
            self._scan_symbol_patterns(setup_key)
            self.candles_since_scan[setup_key] = 0

    def _on_new_candle_spread(self, setup_key: str, candle: Candle, timestamp: str):
        """Handle new candle in spread trading mode (multi-pair)."""
        if not self.spread_scanner:
            return

        # Extract symbol from setup_key
        symbol = setup_key.rsplit("_", 1)[0]

        # Store latest candle for this symbol
        self.last_candles[symbol] = candle

        # Check if we have all symbols with matching timestamps
        if len(self.last_candles) < len(self.spread_symbols):
            return

        # Check all timestamps match
        times = [c.time for c in self.last_candles.values()]
        if len(set(times)) != 1:
            return  # Not all aligned yet

        # Periodic cointegration check
        self.spread_scanner.check_periodic_cointegration()

        # Process each active pair
        for pair in self.spread_scanner.get_active_pairs():
            candle_a = self.last_candles.get(pair.asset_a)
            candle_b = self.last_candles.get(pair.asset_b)

            if not candle_a or not candle_b:
                continue

            # Update pair strategy
            signal = self.spread_scanner.update(candle_a, candle_b, pair.name)

            # Check exits for this pair's z-score
            if pair.strategy:
                current_z = pair.strategy.get_current_zscore()
                self.order_manager.check_spread_exits(current_z)

            # Execute new signal if found
            if signal:
                print(f"[{timestamp}] {pair.name} Spread signal detected!")
                print(f"  Type: {signal.signal_type.value}")
                print(f"  Entry Z: {signal.entry_z:.2f}")
                print(f"  Pattern: {signal.first_extreme_z:.2f} -> {signal.recovery_z:.2f} -> {signal.entry_z:.2f}")

                self._update_account_balance()
                trade = self.order_manager.execute_spread_signal(signal, self.account_balance)
                if trade:
                    pair.strategy.add_signal(signal)

        # Clear candles after processing
        self.last_candles.clear()

    def _scan_all_patterns(self):
        """Scan for patterns across all setups."""
        if self.spread_mode:
            # In spread mode, show scanner status
            if self.spread_scanner:
                self.spread_scanner.print_status()
            return

        total_signals = 0
        for setup_key in self.feeds:
            count = self._scan_symbol_patterns(setup_key, quiet=True)
            total_signals += count

        print(f"Found {total_signals} signals across {len(self.feeds)} setups")
        self._print_active_signals()

    def _scan_symbol_patterns(self, setup_key: str, quiet: bool = False) -> int:
        """Scan for new Double Touch patterns on a setup."""
        strategy = self.strategies[setup_key]
        new_signals = strategy.scan_for_patterns()

        for sig in new_signals:
            strategy.add_signal(sig)
            if not quiet:
                print(f"  [{setup_key}] {sig.signal_type.value}: Entry={sig.entry_price:.2f} SL={sig.stop_loss:.2f} TP={sig.target:.2f} R:R={sig.get_risk_reward():.2f}")

        return len(new_signals)

    def _print_active_signals(self):
        """Print all active signals."""
        all_active = []
        for setup_key, strategy in self.strategies.items():
            for sig in strategy.active_signals:
                sig._setup_key = setup_key  # Tag for display
                all_active.append(sig)

        if not all_active:
            print("No active signals")
            return

        print(f"\nActive signals ({len(all_active)}):")
        # Sort by R:R
        all_active.sort(key=lambda s: s.get_risk_reward(), reverse=True)
        for sig in all_active[:10]:  # Show top 10
            setup_key = getattr(sig, '_setup_key', sig.symbol)
            print(f"  [{setup_key}] {sig.signal_type.value}: Entry={sig.entry_price:.4f} R:R={sig.get_risk_reward():.2f}")

    def _update_account_balance(self):
        """Update account balance and equity from exchange."""
        try:
            # Available balance for position sizing (excludes margin in use)
            self.account_balance = self.client.get_available_balance()
            # Total equity for display (includes margin + unrealized PnL)
            self.account_equity = self.client.get_equity()
        except Exception as e:
            print(f"Balance update error: {e}")

    def _init_spread_strategy(self):
        """Initialize spread scanner with all pair feeds."""
        from spread_scanner import SpreadScanner

        # Check all required feeds exist
        missing = []
        for symbol in self.spread_symbols:
            key = make_setup_key(symbol, self.config.timeframe)
            if key not in self.feeds:
                missing.append(symbol)

        if missing:
            print(f"ERROR: Missing feeds for spread scanner: {missing}")
            return

        # Create the multi-pair scanner
        self.spread_scanner = SpreadScanner(
            config=self.config,
            feeds=self.feeds,
            check_interval=500,      # Check cointegration every 500 candles (~42 hours)
            p_threshold=0.05,        # Enable trading when p < 0.05
            p_disable_threshold=0.15 # Disable trading when p > 0.15
        )

        # Print initial status
        self.spread_scanner.print_status()

    def _record_equity_snapshot(self):
        """Record equity snapshot for performance tracking."""
        try:
            self._update_account_balance()

            # Get unrealized P&L from open positions
            unrealized_pnl = 0.0
            open_positions = 0
            try:
                positions = self.client.get_positions()
                for pos in positions:
                    if pos.size > 0:
                        unrealized_pnl += pos.unrealized_pnl
                        open_positions += 1
            except:
                pass

            # Record the snapshot
            self.tracker.record_equity_snapshot(
                equity=self.account_equity,
                available_balance=self.account_balance,
                unrealized_pnl=unrealized_pnl,
                open_positions=open_positions
            )
        except Exception as e:
            print(f"Equity snapshot error: {e}")

    def _recover_existing_positions(self):
        """Check for existing positions and pending orders from previous session."""
        from order_manager import Trade, TradeStatus
        from double_touch_strategy import TradeSignal, SignalType, SignalStatus
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
                        signal_type=SignalType.LONG_DOUBLE_TOUCH if is_long else SignalType.SHORT_DOUBLE_TOUCH,
                        status=SignalStatus.FILLED,
                        symbol=symbol,
                        setup_key=f"{symbol}_5",
                        step_0_idx=0, step_0_price=0,
                        step_1_idx=0,
                        step_2_idx=0,
                        step_3_idx=0, step_3_price=pos.stop_loss or pos.entry_price * (0.95 if is_long else 1.05),
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
                        signal_type=SignalType.LONG_DOUBLE_TOUCH if is_long else SignalType.SHORT_DOUBLE_TOUCH,
                        status=SignalStatus.FILLED,
                        symbol=symbol,
                        setup_key=f"{symbol}_5",
                        step_0_idx=0, step_0_price=0,
                        step_1_idx=0,
                        step_2_idx=0,
                        step_3_idx=0, step_3_price=order.price * (0.95 if is_long else 1.05),
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
        print("\n" + "=" * 40)

        if self.spread_mode:
            # Spread scanner stats
            stats = self.order_manager.get_spread_stats()

            print("Spread Scanner Stats")
            print("=" * 40)

            if self.spread_scanner:
                scanner_stats = self.spread_scanner.get_stats()
                print(f"Active pairs: {scanner_stats['active_pairs']}/{scanner_stats['total_pairs']}")

                for name, pair_stats in scanner_stats['pairs'].items():
                    status = "✅" if pair_stats['cointegrated'] else "❌"
                    z = f"z={pair_stats['zscore']:.2f}" if pair_stats['cointegrated'] else ""
                    print(f"  {name}: {status} p={pair_stats['p_value']:.3f} {z}")

                print(f"Next coint check: {scanner_stats['candles_until_check']} candles")

            print(f"Active trades: {stats.get('active_spread_trades', 0)}")
            print(f"Total trades: {stats.get('total_spread_trades', 0)}")
            print(f"Win rate: {stats.get('spread_win_rate', 0) * 100:.1f}%")
            print(f"P&L: ${stats.get('spread_pnl', 0):.2f}")
        else:
            # Normal trading stats
            stats = self.order_manager.get_stats()

            # Count active signals across all setups
            total_signals = sum(len(s.active_signals) for s in self.strategies.values())

            print("Trading Stats")
            print("=" * 40)
            print(f"Setups monitored: {len(self.feeds)}")
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

        if self.spread_mode:
            # Close spread trades on shutdown
            print("Closing spread trades...")
            self.order_manager.close_all_spread_trades("shutdown")
        else:
            # Cancel pending orders for all unique symbols
            unique_symbols = list(set(sym for sym, _ in self.setups))
            self.order_manager.cancel_pending_orders(unique_symbols)

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

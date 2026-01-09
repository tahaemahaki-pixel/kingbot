"""
Scalping Strategy Trading Bot - FVG Breakout with Partial Exits

High-frequency scalping bot for BTC, ETH, SOL on 5-minute timeframe.
- Targets 9+ trades/day with 85%+ win rate
- Partial exit system: 50% at 1R, 50% at 1.5R
- Move SL to breakeven after TP1
"""

import os
import time
import signal
import sys
import atexit
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file before any config imports
load_dotenv()

# ==================== PID LOCK FILE ====================
LOCKFILE = "/tmp/scalp_bot.lock"


def check_lock():
    """Check if another instance is running. Exit if so."""
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, 'r') as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"ERROR: Another instance is already running (PID {old_pid})")
                print(f"If this is incorrect, delete {LOCKFILE} and try again.")
                sys.exit(1)
            except OSError:
                print(f"Removing stale lock file (PID {old_pid} not running)")
                os.remove(LOCKFILE)
        except (ValueError, IOError) as e:
            print(f"Warning: Invalid lock file, removing: {e}")
            os.remove(LOCKFILE)

    with open(LOCKFILE, 'w') as f:
        f.write(str(os.getpid()))

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


from config import BotConfig
from scalp_config import ScalpConfig, BYBIT_MAINNET, BYBIT_TESTNET
from bybit_client import BybitClient, BybitWebSocket, Order
from data_feed import DataFeed
from scalp_strategy import ScalpStrategy, ScalpSignal, ScalpSignalType, ScalpStatus
from order_manager import OrderManager, TradeStatus
from notifier import TelegramNotifier
from trade_tracker import get_tracker
import math


# Quantity rules for Bybit USDT Perpetual (min qty and step size)
QTY_RULES = {
    "BTCUSDT": {"min": 0.001, "step": 0.001},
    "ETHUSDT": {"min": 0.01, "step": 0.01},
    "SOLUSDT": {"min": 0.1, "step": 0.1},
    "XRPUSDT": {"min": 1, "step": 1},
    "DOGEUSDT": {"min": 1, "step": 1},
    "ADAUSDT": {"min": 1, "step": 1},
    "AVAXUSDT": {"min": 0.1, "step": 0.1},
    "LINKUSDT": {"min": 0.1, "step": 0.1},
    "DOTUSDT": {"min": 0.1, "step": 0.1},
    "SUIUSDT": {"min": 1, "step": 1},
    "LTCUSDT": {"min": 0.01, "step": 0.01},
    "BCHUSDT": {"min": 0.01, "step": 0.01},
    "ATOMUSDT": {"min": 0.1, "step": 0.1},
    "UNIUSDT": {"min": 0.1, "step": 0.1},
    "APTUSDT": {"min": 0.1, "step": 0.1},
    "ARBUSDT": {"min": 1, "step": 1},
    "OPUSDT": {"min": 0.1, "step": 0.1},
    "NEARUSDT": {"min": 0.1, "step": 0.1},
    "FILUSDT": {"min": 0.1, "step": 0.1},
    "INJUSDT": {"min": 0.1, "step": 0.1},
}


def round_qty(symbol: str, qty: float) -> float:
    """Round quantity to valid step size for the symbol."""
    rules = QTY_RULES.get(symbol, {"min": 0.001, "step": 0.001})
    step = rules["step"]
    min_qty = rules["min"]

    # Round DOWN to nearest step (for closing positions)
    rounded = math.floor(qty / step) * step

    # Ensure minimum
    if rounded < min_qty:
        rounded = min_qty

    return rounded


def make_setup_key(symbol: str, timeframe: str) -> str:
    """Create a unique key for a symbol+timeframe setup."""
    return f"{symbol}_{timeframe}"


@dataclass
class ActiveScalpTrade:
    """Tracks an active scalp trade with partial exit state."""
    signal: ScalpSignal
    entry_order_id: str
    entry_price: float
    original_size: float
    remaining_size: float
    stop_loss: float
    candles_held: int = 0
    tp1_hit: bool = False
    realized_pnl: float = 0.0


class ScalpBot:
    """Scalping Strategy Bot - High-frequency FVG trading with partial exits."""

    def __init__(self, bot_config: BotConfig, scalp_config: ScalpConfig = None):
        self.bot_config = bot_config
        self.config = scalp_config or ScalpConfig.from_env()
        self.running = False

        # Override bot_config with scalp-specific settings
        self.bot_config.api_key = self.config.api_key or self.bot_config.api_key
        self.bot_config.api_secret = self.config.api_secret or self.bot_config.api_secret
        self.bot_config.testnet = self.config.testnet

        # Initialize client
        self.client = BybitClient(self.bot_config)

        # Initialize notifier
        self.bot_config.telegram_token = self.config.telegram_token or self.bot_config.telegram_token
        self.bot_config.telegram_chat_id = self.config.telegram_chat_id or self.bot_config.telegram_chat_id
        self.notifier = TelegramNotifier(self.bot_config)

        # Data feeds and strategies (keyed by setup_key)
        self.feeds: Dict[str, DataFeed] = {}
        self.strategies: Dict[str, ScalpStrategy] = {}

        # Active scalp trades (keyed by symbol)
        self.active_trades: Dict[str, ActiveScalpTrade] = {}

        # Pending limit orders (keyed by symbol) - waiting to fill
        self.pending_orders: Dict[str, dict] = {}  # {symbol: {order_id, signal, size, candles_waiting}}

        # Cooldown tracking (symbol -> candle count since last exit)
        self.cooldowns: Dict[str, int] = {}

        # Order manager for position sync
        self.order_manager = OrderManager(self.bot_config, self.client, self.notifier)

        # WebSocket
        self.ws: Optional[BybitWebSocket] = None

        # Account state
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0

        # Performance tracking
        self.tracker = get_tracker()

        # Stats
        self.signals_count = 0
        self.executed_count = 0
        self.tp1_count = 0
        self.tp2_count = 0
        self.sl_count = 0
        self.be_count = 0

    # ==================== INITIALIZATION ====================

    def _initialize_feeds(self):
        """Initialize data feeds for all symbols."""
        symbols = self.config.symbols
        timeframe = self.config.timeframe
        candles_to_load = self.config.candles_preload

        print(f"\nLoading data for {len(symbols)} symbols ({candles_to_load} candles)...")
        success = 0

        for i, symbol in enumerate(symbols):
            setup_key = make_setup_key(symbol, timeframe)
            try:
                feed = DataFeed(self.bot_config, self.client, symbol, timeframe=timeframe)
                feed.load_historical(candles_to_load)
                self.feeds[setup_key] = feed

                # Create strategy
                strategy = ScalpStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    min_vol_ratio=self.config.min_vol_ratio,
                    imbalance_threshold=self.config.imbalance_threshold,
                    imbalance_lookback=self.config.imbalance_lookback,
                    min_cradle_candles=self.config.min_cradle_candles,
                    cradle_lookback=self.config.cradle_lookback,
                    sl_buffer_pct=self.config.sl_buffer_pct,
                    tp1_r_multiple=self.config.tp1_r_multiple,
                    tp2_r_multiple=self.config.tp2_r_multiple,
                    tp1_close_pct=self.config.tp1_close_pct,
                    max_hold_candles=self.config.max_hold_candles,
                    trade_direction=self.config.trade_direction,
                )
                self.strategies[setup_key] = strategy
                success += 1

                if (i + 1) % 2 == 0:
                    print(f"  {i + 1}/{len(symbols)} loaded...")

                time.sleep(0.2)

            except Exception as e:
                print(f"  Error loading {symbol}: {e}")

        print(f"Loaded {success}/{len(symbols)} symbols")

    def _connect_websocket(self):
        """Connect WebSocket and subscribe to kline streams."""
        subscriptions = []
        timeframe = self.config.timeframe

        for symbol in self.config.symbols:
            subscriptions.append((symbol, timeframe))

        self.ws = BybitWebSocket(
            self.bot_config,
            on_kline=self._on_kline,
            subscriptions=subscriptions
        )
        self.ws.connect()

        print(f"\nWebSocket connected: {len(subscriptions)} streams")

    # ==================== WEBSOCKET HANDLERS ====================

    def _on_kline(self, data: Dict):
        """Handle incoming kline data."""
        symbol = data.get("symbol")
        timeframe = data.get("timeframe", self.config.timeframe)
        is_new_candle = data.get("confirm", False)

        setup_key = make_setup_key(symbol, timeframe)

        if setup_key in self.feeds:
            self.feeds[setup_key].update_candle(data)

            if is_new_candle:
                self._on_candle_close(symbol, setup_key)

            # Check exit conditions on every tick for active trades
            if symbol in self.active_trades:
                current_price = float(data.get("close", 0))
                if current_price > 0:
                    self._check_exits(symbol, current_price)

    def _on_candle_close(self, symbol: str, setup_key: str):
        """Process new confirmed candle."""
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Candle close: {symbol}")

        # Check pending limit orders for fills
        if symbol in self.pending_orders:
            self._check_pending_orders()

        # Update cooldowns
        if symbol in self.cooldowns:
            self.cooldowns[symbol] += 1
            if self.cooldowns[symbol] >= self.config.cooldown_candles:
                del self.cooldowns[symbol]

        # Update candles held for active trades
        if symbol in self.active_trades:
            self.active_trades[symbol].candles_held += 1

        # Check for new signals (if no active position, no pending order, and not in cooldown)
        if symbol not in self.active_trades and symbol not in self.pending_orders and symbol not in self.cooldowns:
            if len(self.active_trades) + len(self.pending_orders) < self.config.max_positions:
                self._scan_for_signal(setup_key)

    # ==================== SIGNAL DETECTION ====================

    def _scan_for_signal(self, setup_key: str):
        """Scan for new scalping signals."""
        if setup_key not in self.feeds or setup_key not in self.strategies:
            return

        feed = self.feeds[setup_key]
        strategy = self.strategies[setup_key]

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
            self._handle_signal(signal)

    def _handle_signal(self, signal: ScalpSignal):
        """Handle a new scalping signal."""
        self.signals_count += 1

        print(f"\n{'='*60}")
        print(f"NEW SCALP SIGNAL - {signal.symbol}")
        print(f"{'='*60}")
        print(f"  Direction: {signal.direction.upper()}")
        print(f"  Entry: {signal.entry_price:.6f}")
        print(f"  Stop Loss: {signal.stop_loss:.6f}")
        print(f"  TP1 (1.0R): {signal.tp1:.6f}")
        print(f"  TP2 (1.5R): {signal.tp2:.6f}")
        print(f"  Volume: {signal.vol_ratio:.1f}x")
        print(f"  Imbalance: {signal.imbalance:+.2f}")
        print(f"{'='*60}")

        self._notify_signal(signal)
        self._execute_signal(signal)

    def _execute_signal(self, signal: ScalpSignal):
        """Execute a scalping signal."""
        self._update_account_balance()

        # Check daily loss limit
        if self.daily_pnl <= -self.account_balance * self.config.max_daily_loss:
            print(f"  Daily loss limit reached (${self.daily_pnl:.2f})")
            return

        # Check max positions
        if len(self.active_trades) >= self.config.max_positions:
            print(f"  Max positions ({self.config.max_positions}) reached")
            return

        # Check max per symbol
        if signal.symbol in self.active_trades:
            print(f"  Already have position in {signal.symbol}")
            return

        # Calculate position size
        risk_amount = self.account_balance * self.config.risk_per_trade
        risk_distance = abs(signal.stop_loss - signal.entry_price)

        if risk_distance <= 0:
            print(f"  Invalid risk distance")
            return

        position_size = risk_amount / risk_distance
        side = "Sell" if signal.direction == "short" else "Buy"

        try:
            # Use LIMIT order at FVG boundary for better entry
            order = self.order_manager.place_order(
                symbol=signal.symbol,
                side=side,
                qty=position_size,
                price=signal.entry_price,  # Limit at FVG boundary
                stop_loss=signal.stop_loss,
                take_profit=None,  # We handle TP manually for partial exits
                order_type="Limit",
                signal_type=f"scalp_{signal.signal_type.value}",
            )

            if order:
                # Track as pending order (not filled yet)
                self.pending_orders[signal.symbol] = {
                    'order_id': order.order_id,
                    'signal': signal,
                    'size': position_size,
                    'side': side,
                    'candles_waiting': 0,
                }

                print(f"  Limit order placed: {side} {position_size:.6f} @ {signal.entry_price:.6f}")
                print(f"  TP1: {signal.tp1:.6f} | TP2: {signal.tp2:.6f} | SL: {signal.stop_loss:.6f}")

        except Exception as e:
            print(f"  Order failed: {e}")

    # ==================== PENDING ORDER MANAGEMENT ====================

    def _check_pending_orders(self):
        """Check if any pending limit orders have been filled."""
        symbols_to_remove = []

        for symbol, pending in list(self.pending_orders.items()):
            order_id = pending['order_id']
            signal = pending['signal']
            size = pending['size']

            try:
                # Check order status
                order_status = self.client.get_order_status(symbol, order_id)

                if order_status == "Filled":
                    # Order filled - move to active trades
                    print(f"\n  Limit order FILLED: {symbol} @ {signal.entry_price:.6f}")

                    self.executed_count += 1
                    self.daily_trades += 1

                    # Create active trade record
                    self.active_trades[symbol] = ActiveScalpTrade(
                        signal=signal,
                        entry_order_id=order_id,
                        entry_price=signal.entry_price,
                        original_size=size,
                        remaining_size=size,
                        stop_loss=signal.stop_loss,
                    )

                    signal.status = ScalpStatus.OPEN
                    signal.original_size = size
                    signal.remaining_size = size
                    signal.entry_order_id = order_id

                    symbols_to_remove.append(symbol)
                    self._notify_fill(symbol, signal, size)

                elif order_status in ["Cancelled", "Rejected", "Deactivated"]:
                    # Order cancelled/rejected - remove from pending
                    print(f"  Limit order {order_status}: {symbol}")
                    symbols_to_remove.append(symbol)

                else:
                    # Still pending - increment wait counter
                    pending['candles_waiting'] += 1

                    # Cancel if waiting too long (3 candles = 15 min)
                    if pending['candles_waiting'] >= 3:
                        print(f"  Cancelling stale order: {symbol} (waited {pending['candles_waiting']} candles)")
                        try:
                            self.client.cancel_order(symbol, order_id)
                        except Exception as e:
                            print(f"  Cancel failed: {e}")
                        symbols_to_remove.append(symbol)

            except Exception as e:
                print(f"  Error checking order {symbol}: {e}")

        # Remove processed orders
        for symbol in symbols_to_remove:
            if symbol in self.pending_orders:
                del self.pending_orders[symbol]

    def _notify_fill(self, symbol: str, signal, size: float):
        """Send notification when limit order fills."""
        direction_emoji = "🔴" if signal.direction == "short" else "🟢"
        msg = (
            f"{direction_emoji} <b>LIMIT ORDER FILLED</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Direction: <b>{signal.direction.upper()}</b>\n"
            f"Entry: {signal.entry_price:.6f}\n"
            f"Size: {size:.4f}\n"
            f"Stop Loss: {signal.stop_loss:.6f}\n"
            f"TP1: {signal.tp1:.6f}\n"
            f"TP2: {signal.tp2:.6f}"
        )
        self.notifier.send(msg)

    # ==================== EXIT MANAGEMENT ====================

    def _check_exits(self, symbol: str, current_price: float):
        """Check exit conditions for an active trade."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        signal = trade.signal
        is_short = signal.signal_type == ScalpSignalType.SCALP_SHORT

        # Check timeout
        if trade.candles_held >= self.config.max_hold_candles:
            self._close_position(symbol, current_price, "timeout", 1.0)
            return

        # Check TP1 (if not already hit)
        if not trade.tp1_hit:
            tp1_hit = (is_short and current_price <= signal.tp1) or \
                      (not is_short and current_price >= signal.tp1)

            if tp1_hit:
                self._handle_tp1(symbol, current_price)
                return

            # Check stop loss (before TP1)
            sl_hit = (is_short and current_price >= trade.stop_loss) or \
                     (not is_short and current_price <= trade.stop_loss)

            if sl_hit:
                self._close_position(symbol, current_price, "sl", 1.0)
                return

        else:
            # After TP1: check TP2 or breakeven stop
            tp2_hit = (is_short and current_price <= signal.tp2) or \
                      (not is_short and current_price >= signal.tp2)

            if tp2_hit:
                self._close_position(symbol, current_price, "tp2", 1.0)
                return

            # Breakeven stop (at entry price)
            be_hit = (is_short and current_price >= signal.entry_price) or \
                     (not is_short and current_price <= signal.entry_price)

            if be_hit:
                self._close_position(symbol, current_price, "be", 1.0)
                return

    def _handle_tp1(self, symbol: str, current_price: float):
        """Handle TP1 hit - close 50%, move SL to breakeven."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        signal = trade.signal
        is_short = signal.signal_type == ScalpSignalType.SCALP_SHORT

        close_size = round_qty(symbol, trade.remaining_size * self.config.tp1_close_pct)

        print(f"\n{'='*60}")
        print(f"TP1 HIT - {symbol}")
        print(f"{'='*60}")
        print(f"  Closing {self.config.tp1_close_pct*100}% ({close_size}) at {current_price:.6f}")
        print(f"  Moving SL to breakeven ({signal.entry_price:.6f})")
        print(f"{'='*60}")

        try:
            # Close partial position
            close_side = "Buy" if is_short else "Sell"
            self.client.place_order(
                symbol=symbol,
                side=close_side,
                qty=close_size,
                order_type="Market",
                reduce_only=True
            )

            # Calculate partial P&L
            if is_short:
                partial_pnl = (trade.entry_price - current_price) * close_size
            else:
                partial_pnl = (current_price - trade.entry_price) * close_size

            trade.realized_pnl += partial_pnl
            trade.remaining_size -= close_size
            trade.tp1_hit = True
            trade.stop_loss = signal.entry_price  # Move to breakeven

            # Update stop loss on exchange
            if self.config.move_sl_to_be:
                self._modify_stop_loss(symbol, signal.entry_price)

            self.tp1_count += 1
            self.daily_pnl += partial_pnl

            self._notify_tp1(symbol, current_price, partial_pnl)

        except Exception as e:
            print(f"  TP1 close failed: {e}")

    def _close_position(self, symbol: str, current_price: float, reason: str, close_pct: float):
        """Close a position (full or partial)."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        signal = trade.signal
        is_short = signal.signal_type == ScalpSignalType.SCALP_SHORT

        close_size = round_qty(symbol, trade.remaining_size * close_pct)

        print(f"\n{'='*60}")
        print(f"CLOSING POSITION - {symbol} ({reason.upper()})")
        print(f"{'='*60}")
        print(f"  Size: {close_size} @ {current_price:.6f}")
        print(f"{'='*60}")

        try:
            close_side = "Buy" if is_short else "Sell"
            self.client.place_order(
                symbol=symbol,
                side=close_side,
                qty=close_size,
                order_type="Market",
                reduce_only=True
            )

            # Calculate P&L
            if is_short:
                final_pnl = (trade.entry_price - current_price) * close_size
            else:
                final_pnl = (current_price - trade.entry_price) * close_size

            trade.realized_pnl += final_pnl
            trade.remaining_size -= close_size
            self.daily_pnl += final_pnl

            # Update stats
            if reason == "tp2":
                self.tp2_count += 1
            elif reason == "sl":
                self.sl_count += 1
            elif reason == "be":
                self.be_count += 1

            # If fully closed, remove from active trades
            if trade.remaining_size <= 0 or close_pct >= 1.0:
                total_pnl = trade.realized_pnl
                del self.active_trades[symbol]

                # Set cooldown
                self.cooldowns[symbol] = 0

                self._notify_close(symbol, reason, total_pnl)

        except Exception as e:
            print(f"  Close failed: {e}")

    def _modify_stop_loss(self, symbol: str, new_sl: float):
        """Modify stop loss for an open position."""
        try:
            # Cancel existing stop loss and set new one
            # Using Bybit's trading-stop endpoint
            self.client.set_trading_stop(
                symbol=symbol,
                stop_loss=new_sl
            )
            print(f"  SL modified to {new_sl:.6f}")
        except Exception as e:
            print(f"  SL modification failed: {e}")

    # ==================== NOTIFICATIONS ====================

    def _notify_signal(self, signal: ScalpSignal):
        """Send Telegram notification for new signal."""
        direction_emoji = "🔴" if signal.direction == "short" else "🟢"
        msg = (
            f"{direction_emoji} <b>SCALP SIGNAL</b>\n\n"
            f"Symbol: <b>{signal.symbol}</b>\n"
            f"Direction: <b>{signal.direction.upper()}</b>\n"
            f"Entry: {signal.entry_price:.6f}\n"
            f"Stop Loss: {signal.stop_loss:.6f}\n"
            f"TP1 (1R): {signal.tp1:.6f}\n"
            f"TP2 (1.5R): {signal.tp2:.6f}\n\n"
            f"Volume: {signal.vol_ratio:.1f}x\n"
            f"Imbalance: {signal.imbalance:+.2f}"
        )
        self.notifier.send(msg)

    def _notify_tp1(self, symbol: str, price: float, pnl: float):
        """Send notification for TP1 hit."""
        msg = (
            f"✅ <b>TP1 HIT</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Price: {price:.6f}\n"
            f"Partial P&L: ${pnl:.2f}\n"
            f"SL moved to breakeven"
        )
        self.notifier.send(msg)

    def _notify_close(self, symbol: str, reason: str, pnl: float):
        """Send notification for position close."""
        emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "⚪"
        msg = (
            f"{emoji} <b>POSITION CLOSED</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Reason: {reason.upper()}\n"
            f"Total P&L: ${pnl:.2f}"
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
        print(f"SCALP BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        print(f"\nACTIVE TRADES: {len(self.active_trades)}/{self.config.max_positions}")
        for symbol, trade in self.active_trades.items():
            status = "TP1 HIT" if trade.tp1_hit else "OPEN"
            print(f"  {symbol}: {trade.signal.direction.upper()} | {status} | "
                  f"Held: {trade.candles_held} candles | P&L: ${trade.realized_pnl:.2f}")

        if self.pending_orders:
            print(f"\nPENDING LIMIT ORDERS: {len(self.pending_orders)}")
            for symbol, pending in self.pending_orders.items():
                print(f"  {symbol}: {pending['side']} @ {pending['signal'].entry_price:.6f} | "
                      f"Waiting: {pending['candles_waiting']} candles")

        print(f"\nSESSION STATS:")
        print(f"  Signals: {self.signals_count}")
        print(f"  Executed: {self.executed_count}")
        print(f"  TP1 Hits: {self.tp1_count}")
        print(f"  TP2 Hits: {self.tp2_count}")
        print(f"  SL Hits: {self.sl_count}")
        print(f"  BE Exits: {self.be_count}")

        print(f"\nACCOUNT:")
        print(f"  Balance: ${self.account_balance:.2f}")
        print(f"  Equity: ${self.account_equity:.2f}")
        print(f"  Daily P&L: ${self.daily_pnl:.2f}")
        print(f"{'='*60}")

    # ==================== MAIN LOOP ====================

    def _run_loop(self):
        """Main event loop."""
        self.running = True
        last_status = 0
        last_sync = 0
        status_interval = 300  # 5 minutes
        sync_interval = 30

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

        # Close all positions at market
        for symbol in list(self.active_trades.keys()):
            try:
                trade = self.active_trades[symbol]
                is_short = trade.signal.signal_type == ScalpSignalType.SCALP_SHORT
                close_side = "Buy" if is_short else "Sell"
                close_qty = round_qty(symbol, trade.remaining_size)

                self.client.place_order(
                    symbol=symbol,
                    side=close_side,
                    qty=close_qty,
                    order_type="Market",
                    reduce_only=True
                )
                print(f"  Closed {symbol} ({close_qty}) at market")
            except Exception as e:
                print(f"  Failed to close {symbol}: {e}")

        if self.ws:
            self.ws.close()

        self.notifier.send("🛑 <b>SCALP BOT STOPPED</b>")
        print("Shutdown complete.")
        sys.exit(0)

    def start(self):
        """Start the scalping bot."""
        print("=" * 60)
        print("SCALPING STRATEGY BOT - FVG Breakout with Partial Exits")
        print("=" * 60)
        print(f"\nCONFIG:")
        print(f"  Symbols: {', '.join(self.config.symbols)}")
        print(f"  Timeframe: {self.config.timeframe}m")
        print(f"  Risk/Trade: {self.config.risk_per_trade * 100}%")
        print(f"  Max positions: {self.config.max_positions}")
        print(f"  Volume filter: {self.config.min_vol_ratio}x")
        print(f"  Imbalance: {self.config.imbalance_threshold}")
        print(f"  TP1: {self.config.tp1_r_multiple}R (close {self.config.tp1_close_pct*100}%)")
        print(f"  TP2: {self.config.tp2_r_multiple}R")
        print(f"  Max hold: {self.config.max_hold_candles} candles")
        print(f"  Cooldown: {self.config.cooldown_candles} candles")
        print(f"\nTestnet: {self.config.testnet}")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Get account balance
        self._update_account_balance()
        print(f"\nAccount equity: ${self.account_equity:.2f}")

        # Initialize feeds
        self._initialize_feeds()

        # Connect WebSocket
        self._connect_websocket()

        # Startup notification
        msg = (
            f"🤖 <b>SCALP BOT STARTED</b>\n\n"
            f"Symbols: {', '.join(self.config.symbols)}\n"
            f"Timeframe: {self.config.timeframe}m\n"
            f"Risk: {self.config.risk_per_trade*100}%\n"
            f"Max positions: {self.config.max_positions}\n\n"
            f"TP1: {self.config.tp1_r_multiple}R | TP2: {self.config.tp2_r_multiple}R\n"
            f"Balance: ${self.account_balance:.2f}"
        )
        self.notifier.send(msg)

        # Initial status
        self._print_status()

        # Run main loop
        self._run_loop()


def main():
    """Main entry point."""
    check_lock()

    bot_config = BotConfig.from_env()
    scalp_config = ScalpConfig.from_env()

    bot = ScalpBot(bot_config, scalp_config)
    bot.start()


if __name__ == "__main__":
    main()

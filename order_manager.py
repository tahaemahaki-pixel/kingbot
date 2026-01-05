"""
Order Manager - Handles order execution, SL/TP, and position tracking
Supports both single-asset Double Touch and dual-leg spread trading.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum
from bybit_client import BybitClient, Position, Order
from double_touch_strategy import TradeSignal, SignalType, SignalStatus
from config import BotConfig, get_asset_type
from trade_tracker import get_tracker

# Import Breakaway types (optional - may not exist)
try:
    from breakaway_strategy import BreakawaySignal, BreakawaySignalType, BreakawayStatus
    BREAKAWAY_AVAILABLE = True
except ImportError:
    BREAKAWAY_AVAILABLE = False


class TradeStatus(Enum):
    PENDING = "pending"  # Order not yet placed
    PENDING_FILL = "pending_fill"  # Limit order placed, waiting for fill
    OPEN = "open"  # Position is open (order filled)
    CLOSED_TP = "closed_tp"  # Closed at take profit
    CLOSED_SL = "closed_sl"  # Closed at stop loss
    CLOSED_MANUAL = "closed_manual"  # Manually closed
    CANCELLED = "cancelled"  # Order cancelled


@dataclass
class Trade:
    """Tracks an individual trade from signal to close."""
    signal: TradeSignal
    status: TradeStatus = TradeStatus.PENDING

    # Order info
    entry_order_id: Optional[str] = None
    entry_filled_price: Optional[float] = None
    position_size: float = 0.0

    # Timing
    opened_at: float = 0.0  # Timestamp when trade was opened

    # P&L tracking
    realized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class SpreadTrade:
    """Tracks a spread trade (two legs)."""
    signal: any  # SpreadSignal (imported dynamically to avoid circular import)
    status: TradeStatus = TradeStatus.PENDING

    # Leg A (e.g., ETH) - Long for long_spread, Short for short_spread
    leg_a_order_id: Optional[str] = None
    leg_a_filled_price: Optional[float] = None
    leg_a_size: float = 0.0
    leg_a_symbol: str = ""

    # Leg B (e.g., BTC) - Short for long_spread, Long for short_spread
    leg_b_order_id: Optional[str] = None
    leg_b_filled_price: Optional[float] = None
    leg_b_size: float = 0.0
    leg_b_symbol: str = ""

    # Timing
    opened_at: float = 0.0

    # P&L tracking (combined)
    realized_pnl: float = 0.0
    exit_reason: Optional[str] = None


class OrderManager:
    """Manages order execution and position lifecycle."""

    # Minimum qty and step size for each symbol (Bybit USDT Perpetual)
    # Default fallback: {"min": 0.01, "step": 0.01}
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
        # Additional coins for Breakaway strategy
        "MATICUSDT": {"min": 1, "step": 1},
        "SHIBUSDT": {"min": 1000, "step": 1000},
        "AAVEUSDT": {"min": 0.01, "step": 0.01},
        "MKRUSDT": {"min": 0.001, "step": 0.001},
        "COMPUSDT": {"min": 0.01, "step": 0.01},
        "ETCUSDT": {"min": 0.1, "step": 0.1},
        "ALGOUSDT": {"min": 1, "step": 1},
        "XLMUSDT": {"min": 1, "step": 1},
        "VETUSDT": {"min": 1, "step": 1},
        "ICPUSDT": {"min": 0.1, "step": 0.1},
        "FTMUSDT": {"min": 1, "step": 1},
        "SANDUSDT": {"min": 1, "step": 1},
        "MANAUSDT": {"min": 1, "step": 1},
        "AXSUSDT": {"min": 0.1, "step": 0.1},
        "GALAUSDT": {"min": 1, "step": 1},
        "TRXUSDT": {"min": 1, "step": 1},
        "APEUSDT": {"min": 0.1, "step": 0.1},
        "LDOUSDT": {"min": 0.1, "step": 0.1},
        "RNDRUSDT": {"min": 0.1, "step": 0.1},
        "GMXUSDT": {"min": 0.01, "step": 0.01},
        "PEPEUSDT": {"min": 100000, "step": 100000},
        "FLOKIUSDT": {"min": 1000, "step": 1000},
        "BONKUSDT": {"min": 100000, "step": 100000},
        "WIFUSDT": {"min": 1, "step": 1},
        "JUPUSDT": {"min": 1, "step": 1},
        "PNUTUSDT": {"min": 1, "step": 1},
        "ONDOUSDT": {"min": 1, "step": 1},
        "ENAUSDT": {"min": 1, "step": 1},
        "EIGENUSDT": {"min": 0.1, "step": 0.1},
        "TRUMPUSDT": {"min": 0.1, "step": 0.1},
    }

    def __init__(self, config: BotConfig, client: BybitClient, notifier=None):
        self.config = config
        self.client = client
        self.notifier = notifier
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.daily_pnl: float = 0.0

        # Spread trading
        self.active_spread_trades: List[SpreadTrade] = []
        self.closed_spread_trades: List[SpreadTrade] = []

        # Performance tracking
        self.tracker = get_tracker()
        self.trade_id_map: Dict[str, str] = {}  # entry_order_id -> tracker_trade_id

    def _round_qty(self, symbol: str, qty: float, price: float) -> float:
        """Round quantity to valid step size for the symbol and ensure min order value."""
        rules = self.QTY_RULES.get(symbol, {"min": 0.001, "step": 0.001})
        step = rules["step"]
        min_qty = rules["min"]

        # Bybit requires minimum $5 order value
        MIN_ORDER_VALUE = 5.0
        min_qty_for_value = MIN_ORDER_VALUE / price if price > 0 else 0

        # Use the larger of: min qty or qty needed for $5 value
        effective_min = max(min_qty, min_qty_for_value)

        # Round UP to nearest step to meet minimum value
        import math
        rounded = math.ceil(qty / step) * step

        # Ensure minimum
        if rounded < effective_min:
            # Round up the minimum to step size
            rounded = math.ceil(effective_min / step) * step

        return rounded

    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """
        Calculate position size based on risk per trade.

        Position Size = Risk Amount / Stop Distance
        Risk Amount = Account Balance * Risk Per Trade
        """
        risk_amount = account_balance * self.config.risk_per_trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)

        if stop_distance == 0:
            return 0

        position_size = risk_amount / stop_distance

        # Round to valid quantity for this symbol (with min $5 order value)
        position_size = self._round_qty(signal.symbol, position_size, signal.entry_price)

        return position_size

    def can_open_trade(self, account_balance: float, setup_key: str = None, symbol: str = None) -> tuple[bool, str]:
        """Check if we can open a new trade based on risk rules."""
        # Check max positions (include pending orders)
        active_trades = [t for t in self.active_trades if t.status in (TradeStatus.OPEN, TradeStatus.PENDING_FILL)]
        if len(active_trades) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

        # Check asset type limits (3 crypto + 2 non-crypto)
        if symbol:
            asset_type = get_asset_type(symbol)
            if asset_type == "crypto":
                crypto_count = len([t for t in active_trades
                                   if get_asset_type(t.signal.symbol) == "crypto"])
                if crypto_count >= self.config.max_crypto_positions:
                    return False, f"Max crypto positions ({self.config.max_crypto_positions}) reached"
            else:
                non_crypto_count = len([t for t in active_trades
                                       if get_asset_type(t.signal.symbol) == "non_crypto"])
                if non_crypto_count >= self.config.max_non_crypto_positions:
                    return False, f"Max non-crypto positions ({self.config.max_non_crypto_positions}) reached"

        # Check if we already have a trade/order for this setup (symbol+timeframe)
        if setup_key:
            existing_for_setup = [t for t in active_trades if (t.signal.setup_key or t.signal.symbol) == setup_key]
            if existing_for_setup:
                return False, f"Already have active trade/order for {setup_key}"

        # Check daily loss limit
        max_daily_loss_amount = account_balance * self.config.max_daily_loss
        if self.daily_pnl <= -max_daily_loss_amount:
            return False, f"Daily loss limit ({self.config.max_daily_loss*100}%) reached"

        return True, ""

    def execute_signal(self, signal: TradeSignal, account_balance: float) -> Optional[Trade]:
        """Execute a ready signal."""
        if signal.status != SignalStatus.READY:
            print(f"Signal not ready: {signal.status}")
            return None

        # Check if we can trade (using setup_key for multi-timeframe support)
        setup_key = signal.setup_key or signal.symbol
        can_trade, reason = self.can_open_trade(account_balance, setup_key, signal.symbol)
        if not can_trade:
            print(f"Cannot open trade: {reason}")
            return None

        # Calculate position size
        position_size = self.calculate_position_size(signal, account_balance)
        if position_size <= 0:
            print("Position size too small")
            return None

        # Determine side - handle both Double Touch and legacy King signal types
        is_long = signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
        side = "Buy" if is_long else "Sell"

        try:
            # Place the order with SL/TP
            if self.config.use_limit_orders:
                order = self.client.place_order(
                    symbol=signal.symbol,
                    side=side,
                    qty=position_size,
                    order_type="Limit",
                    price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target
                )
            else:
                # Market order with slippage buffer
                order = self.client.place_order(
                    symbol=signal.symbol,
                    side=side,
                    qty=position_size,
                    order_type="Market",
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target
                )

            # Create trade record
            # For limit orders, status is PENDING_FILL until filled
            # For market orders, status is OPEN immediately
            initial_status = TradeStatus.PENDING_FILL if self.config.use_limit_orders else TradeStatus.OPEN

            trade = Trade(
                signal=signal,
                status=initial_status,
                entry_order_id=order.order_id,
                entry_filled_price=signal.entry_price if not self.config.use_limit_orders else None,
                position_size=position_size,
                opened_at=time.time()
            )

            self.active_trades.append(trade)
            signal.status = SignalStatus.FILLED

            # Record trade in tracker
            try:
                risk_amount = abs(signal.entry_price - signal.stop_loss) * position_size
                tracker_trade_id = self.tracker.record_trade_opened(
                    symbol=signal.symbol,
                    setup_key=signal.setup_key or signal.symbol,
                    signal_type=signal.signal_type.value,
                    entry_order_id=order.order_id,
                    planned_entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target,
                    risk_amount_usd=risk_amount,
                    position_size=position_size,
                    account_equity=account_balance
                )
                self.trade_id_map[order.order_id] = tracker_trade_id
            except Exception as e:
                print(f"Tracker record failed: {e}")

            print(f"Order placed: {side} {position_size} {signal.symbol} @ {signal.entry_price}")
            print(f"  SL: {signal.stop_loss}, TP: {signal.target}")

            # Send Telegram notification
            if self.notifier:
                if self.config.use_limit_orders:
                    self.notifier.notify_order_placed(
                        symbol=signal.symbol,
                        side=side,
                        entry=signal.entry_price,
                        sl=signal.stop_loss,
                        tp=signal.target,
                        size=position_size,
                        rr=signal.get_risk_reward()
                    )
                else:
                    # Market order - immediately filled
                    self.notifier.notify_order_filled(
                        symbol=signal.symbol,
                        side=side,
                        entry=signal.entry_price,
                        sl=signal.stop_loss,
                        tp=signal.target,
                        size=position_size
                    )

            return trade

        except Exception as e:
            print(f"Order execution failed: {e}")
            return None

    def sync_positions(self):
        """Sync local trade state with actual positions on exchange."""
        try:
            # Get all positions
            all_positions = self.client.get_positions()
            positions_by_symbol = {p.symbol: p for p in all_positions}

            current_time = time.time()

            for trade in self.active_trades[:]:
                symbol = trade.signal.symbol
                exchange_position = positions_by_symbol.get(symbol)

                # Handle PENDING_FILL orders (limit orders waiting to fill)
                if trade.status == TradeStatus.PENDING_FILL:
                    # Skip if order was just placed (give it 10 seconds)
                    if current_time - trade.opened_at < 10:
                        continue

                    # Check if position now exists (order filled)
                    if exchange_position and exchange_position.size > 0:
                        # Order filled! Update status and notify
                        trade.status = TradeStatus.OPEN
                        trade.entry_filled_price = exchange_position.entry_price
                        print(f"Order filled: {symbol} @ {exchange_position.entry_price}")

                        # Update tracker with fill info
                        if trade.entry_order_id in self.trade_id_map:
                            try:
                                position_value = exchange_position.entry_price * trade.position_size
                                slippage = exchange_position.entry_price - trade.signal.entry_price
                                self.tracker.update_trade_fill(
                                    trade_id=self.trade_id_map[trade.entry_order_id],
                                    entry_price=exchange_position.entry_price,
                                    position_value=position_value,
                                    slippage=slippage
                                )
                            except Exception as e:
                                print(f"Tracker fill update failed: {e}")

                        if self.notifier:
                            is_long = trade.signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
                            side = "Buy" if is_long else "Sell"
                            self.notifier.notify_order_filled(
                                symbol=symbol,
                                side=side,
                                entry=exchange_position.entry_price,
                                sl=trade.signal.stop_loss,
                                tp=trade.signal.target,
                                size=trade.position_size
                            )
                        continue

                    # Check if order was cancelled or expired
                    try:
                        open_orders = self.client.get_open_orders(symbol)
                        has_pending_order = any(
                            o.order_id == trade.entry_order_id for o in open_orders
                        )

                        if not has_pending_order:
                            # Order gone but no position = cancelled/expired
                            trade.status = TradeStatus.CANCELLED
                            print(f"Order cancelled/expired: {symbol}")

                            # Record cancellation in tracker
                            if trade.entry_order_id in self.trade_id_map:
                                try:
                                    self.tracker.record_trade_cancelled(
                                        self.trade_id_map[trade.entry_order_id]
                                    )
                                    del self.trade_id_map[trade.entry_order_id]
                                except Exception as e:
                                    print(f"Tracker cancel update failed: {e}")

                            if self.notifier:
                                is_long = trade.signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
                                side = "Buy" if is_long else "Sell"
                                self.notifier.notify_order_cancelled(symbol, side, "price moved away")
                            self.active_trades.remove(trade)
                    except Exception as e:
                        print(f"Error checking orders for {symbol}: {e}")
                    continue

                # Handle OPEN trades (filled positions)
                if trade.status != TradeStatus.OPEN:
                    continue

                # Skip trades opened less than 60 seconds ago
                if current_time - trade.opened_at < 60:
                    continue

                # Check if position still exists
                if exchange_position and exchange_position.size > 0:
                    # Position still open
                    continue

                # No position found - check if there's still a pending order
                try:
                    open_orders = self.client.get_open_orders(symbol)
                    has_pending_order = any(
                        o.order_id == trade.entry_order_id for o in open_orders
                    )

                    if has_pending_order:
                        continue
                except Exception as e:
                    print(f"Error checking orders for {symbol}: {e}")
                    continue

                # Position closed - determine if TP or SL
                # Try to get closed PnL from API to determine outcome
                try:
                    closed_pnl = self.client.get_closed_pnl(symbol, limit=5)
                    if closed_pnl:
                        # Find the most recent close for this symbol
                        recent = closed_pnl[0]
                        trade.realized_pnl = float(recent.get('closedPnl', 0))
                        trade.exit_price = float(recent.get('avgExitPrice', 0))

                        # Determine TP or SL based on P&L
                        if trade.realized_pnl > 0:
                            trade.status = TradeStatus.CLOSED_TP
                        else:
                            trade.status = TradeStatus.CLOSED_SL
                    else:
                        trade.status = TradeStatus.CLOSED_SL
                except Exception as e:
                    print(f"Error getting closed PnL: {e}")
                    trade.status = TradeStatus.CLOSED_SL

                self._finalize_trade(trade, exchange_position)

        except Exception as e:
            print(f"Position sync error: {e}")

    def _finalize_trade(self, trade: Trade, position: Optional[Position]):
        """Finalize a closed trade and calculate P&L."""
        # Move to closed trades
        if trade in self.active_trades:
            self.active_trades.remove(trade)
            self.closed_trades.append(trade)

        # Calculate P&L (simplified - in real implementation, fetch from API)
        if trade.entry_filled_price and trade.exit_price:
            is_long = trade.signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
            if is_long:
                trade.realized_pnl = (trade.exit_price - trade.entry_filled_price) * trade.position_size
            else:
                trade.realized_pnl = (trade.entry_filled_price - trade.exit_price) * trade.position_size

            self.daily_pnl += trade.realized_pnl

        # Record trade close in tracker
        if trade.entry_order_id in self.trade_id_map:
            try:
                exit_reason = trade.status.value.replace('closed_', '')
                self.tracker.record_trade_closed(
                    trade_id=self.trade_id_map[trade.entry_order_id],
                    exit_price=trade.exit_price or 0,
                    realized_pnl=trade.realized_pnl,
                    exit_reason=exit_reason,
                    fees=0  # TODO: Get actual fees from API
                )
                del self.trade_id_map[trade.entry_order_id]
            except Exception as e:
                print(f"Tracker close update failed: {e}")

        print(f"Trade closed: {trade.status.value}, P&L: ${trade.realized_pnl:.2f}")

        # Send Telegram notification
        if self.notifier:
            is_long = trade.signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
            side = "LONG" if is_long else "SHORT"
            self.notifier.notify_trade_closed(
                symbol=trade.signal.symbol,
                side=side,
                pnl=trade.realized_pnl,
                reason=trade.status.value
            )

    def close_trade(self, trade: Trade, reason: str = "manual") -> bool:
        """Manually close a trade."""
        if trade.status != TradeStatus.OPEN:
            return False

        try:
            # Determine close side (opposite of entry)
            is_long = trade.signal.signal_type in (SignalType.LONG_DOUBLE_TOUCH,)
            close_side = "Sell" if is_long else "Buy"

            self.client.place_order(
                symbol=trade.signal.symbol,
                side=close_side,
                qty=trade.position_size,
                order_type="Market",
                reduce_only=True
            )

            trade.status = TradeStatus.CLOSED_MANUAL
            trade.exit_reason = reason
            self._finalize_trade(trade, None)

            return True

        except Exception as e:
            print(f"Close trade failed: {e}")
            return False

    def close_all_trades(self, reason: str = "shutdown"):
        """Close all open trades."""
        for trade in self.active_trades[:]:
            if trade.status == TradeStatus.OPEN:
                self.close_trade(trade, reason)

    def cancel_pending_orders(self, symbols: List[str] = None):
        """Cancel only pending ENTRY orders (not SL/TP orders protecting open positions)."""
        try:
            # Only cancel specific entry orders we're tracking, not all orders
            # This preserves SL/TP orders for open positions
            for trade in self.active_trades[:]:
                if trade.status == TradeStatus.PENDING_FILL and trade.entry_order_id:
                    try:
                        self.client.cancel_order(trade.signal.symbol, trade.entry_order_id)
                        print(f"Cancelled pending entry order {trade.entry_order_id} for {trade.signal.symbol}")
                    except Exception as e:
                        print(f"Failed to cancel order {trade.entry_order_id}: {e}")

                    # Record cancellation in tracker
                    if trade.entry_order_id in self.trade_id_map:
                        try:
                            self.tracker.record_trade_cancelled(
                                self.trade_id_map[trade.entry_order_id]
                            )
                            del self.trade_id_map[trade.entry_order_id]
                        except Exception as e:
                            print(f"Tracker cancel update failed: {e}")

                    trade.status = TradeStatus.CANCELLED
                    self.active_trades.remove(trade)
        except Exception as e:
            print(f"Cancel orders failed: {e}")

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        total_trades = len(self.closed_trades)

        winning_trades = [t for t in self.closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.realized_pnl < 0]

        total_profit = sum(t.realized_pnl for t in winning_trades)
        total_loss = abs(sum(t.realized_pnl for t in losing_trades))

        open_trades = len([t for t in self.active_trades if t.status == TradeStatus.OPEN])
        pending_orders = len([t for t in self.active_trades if t.status == TradeStatus.PENDING_FILL])

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "total_pnl": sum(t.realized_pnl for t in self.closed_trades),
            "daily_pnl": self.daily_pnl,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "open_trades": open_trades,
            "pending_orders": pending_orders
        }

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)."""
        self.daily_pnl = 0.0

    # ==================== BREAKAWAY STRATEGY SUPPORT ====================

    def has_position(self, symbol: str) -> bool:
        """Check if there's an active trade/order for a symbol."""
        for trade in self.active_trades:
            if trade.status in (TradeStatus.OPEN, TradeStatus.PENDING_FILL):
                if trade.signal.symbol == symbol:
                    return True
        return False

    def get_open_count(self) -> int:
        """Get count of open positions and pending orders."""
        return len([t for t in self.active_trades
                    if t.status in (TradeStatus.OPEN, TradeStatus.PENDING_FILL)])

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        order_type: str = "Limit",
        signal_type: str = "breakaway",
    ) -> Optional[Order]:
        """
        Place a generic order for Breakaway strategy.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: "Buy" or "Sell"
            qty: Position size (will be rounded to valid step)
            price: Limit price (required for Limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_type: "Limit" or "Market"
            signal_type: Signal type for tracking

        Returns:
            Order object if successful, None otherwise
        """
        # Round quantity to valid step size
        reference_price = price if price else self.client.get_last_price(symbol)
        rounded_qty = self._round_qty(symbol, qty, reference_price)

        try:
            if order_type == "Limit" and price:
                order = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=rounded_qty,
                    order_type="Limit",
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            else:
                order = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=rounded_qty,
                    order_type="Market",
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

            # Create a minimal TradeSignal for tracking
            # This allows reuse of existing trade tracking infrastructure
            if BREAKAWAY_AVAILABLE:
                from breakaway_strategy import BreakawaySignal, BreakawaySignalType, BreakawayStatus

                # Create a dummy signal for tracking
                direction = "short" if side == "Sell" else "long"
                sig_type = BreakawaySignalType.BREAKAWAY_SHORT if side == "Sell" else BreakawaySignalType.BREAKAWAY_LONG

                # We need to track this as a Trade for position sync
                # Create a minimal TradeSignal-compatible object
                class MinimalSignal:
                    def __init__(self):
                        self.symbol = symbol
                        self.entry_price = price or reference_price
                        self.stop_loss = stop_loss
                        self.target = take_profit
                        self.signal_type = SignalType.SHORT_DOUBLE_TOUCH if side == "Sell" else SignalType.LONG_DOUBLE_TOUCH
                        self.setup_key = f"{symbol}_breakaway"
                        self.status = SignalStatus.FILLED

                    def get_risk_reward(self):
                        if self.stop_loss and self.target and self.entry_price:
                            risk = abs(self.entry_price - self.stop_loss)
                            reward = abs(self.target - self.entry_price)
                            return reward / risk if risk > 0 else 0
                        return 0

                minimal_signal = MinimalSignal()

                # Create trade record for position tracking
                trade = Trade(
                    signal=minimal_signal,
                    status=TradeStatus.PENDING_FILL if order_type == "Limit" else TradeStatus.OPEN,
                    entry_order_id=order.order_id,
                    entry_filled_price=price if order_type == "Market" else None,
                    position_size=rounded_qty,
                    opened_at=time.time()
                )
                self.active_trades.append(trade)

            print(f"Breakaway order placed: {side} {rounded_qty} {symbol} @ {price or 'Market'}")
            return order

        except Exception as e:
            print(f"Breakaway order failed: {e}")
            return None

    # ==================== SPREAD TRADING ====================

    def execute_spread_signal(self, signal, account_balance: float) -> Optional[SpreadTrade]:
        """
        Execute a spread trading signal (two legs).

        For LONG_SPREAD: Buy asset_a (ETH), Sell asset_b (BTC)
        For SHORT_SPREAD: Sell asset_a (ETH), Buy asset_b (BTC)
        """
        from spread_strategy import SpreadSignalType, SpreadSignalStatus

        if signal.status != SpreadSignalStatus.READY:
            print(f"Spread signal not ready: {signal.status}")
            return None

        # Check if we can trade
        if self.active_spread_trades:
            print("Already have active spread trade")
            return None

        # Calculate position sizes based on risk
        # Risk = 2% of account, split between legs based on hedge ratio
        risk_amount = account_balance * self.config.risk_per_trade

        # Position value for each leg (simplified: equal dollar exposure)
        leg_value = risk_amount * 10  # 10x the risk amount for position size

        # Calculate quantities
        asset_a = self.config.spread_pair.asset_a  # e.g., ETHUSDT
        asset_b = self.config.spread_pair.asset_b  # e.g., BTCUSDT

        qty_a = leg_value / signal.asset_a_price
        qty_b = (leg_value * signal.hedge_ratio) / signal.asset_b_price

        qty_a = self._round_qty(asset_a, qty_a, signal.asset_a_price)
        qty_b = self._round_qty(asset_b, qty_b, signal.asset_b_price)

        # Determine sides
        is_long_spread = signal.signal_type == SpreadSignalType.LONG_SPREAD
        side_a = "Buy" if is_long_spread else "Sell"
        side_b = "Sell" if is_long_spread else "Buy"

        try:
            # Execute both legs (market orders for speed)
            order_a = self.client.place_order(
                symbol=asset_a,
                side=side_a,
                qty=qty_a,
                order_type="Market"
            )

            order_b = self.client.place_order(
                symbol=asset_b,
                side=side_b,
                qty=qty_b,
                order_type="Market"
            )

            # Create spread trade record
            spread_trade = SpreadTrade(
                signal=signal,
                status=TradeStatus.OPEN,
                leg_a_order_id=order_a.order_id,
                leg_a_filled_price=signal.asset_a_price,
                leg_a_size=qty_a,
                leg_a_symbol=asset_a,
                leg_b_order_id=order_b.order_id,
                leg_b_filled_price=signal.asset_b_price,
                leg_b_size=qty_b,
                leg_b_symbol=asset_b,
                opened_at=time.time()
            )

            self.active_spread_trades.append(spread_trade)
            signal.status = SpreadSignalStatus.FILLED

            direction = "LONG" if is_long_spread else "SHORT"
            print(f"Spread trade opened: {direction}")
            print(f"  Leg A: {side_a} {qty_a} {asset_a} @ {signal.asset_a_price:.2f}")
            print(f"  Leg B: {side_b} {qty_b} {asset_b} @ {signal.asset_b_price:.2f}")
            print(f"  Entry Z: {signal.entry_z:.2f}")

            # Notify
            if self.notifier:
                self.notifier.send_message(
                    f"🔄 SPREAD TRADE OPENED\n"
                    f"Direction: {direction}\n"
                    f"Entry Z: {signal.entry_z:.2f}\n"
                    f"Leg A: {side_a} {qty_a:.4f} {asset_a}\n"
                    f"Leg B: {side_b} {qty_b:.6f} {asset_b}\n"
                    f"TP Z: {signal.tp_z}, SL Z: {signal.sl_z}"
                )

            return spread_trade

        except Exception as e:
            print(f"Spread order execution failed: {e}")
            # TODO: Handle partial fills (close the leg that filled)
            return None

    def check_spread_exits(self, current_zscore: float):
        """Check if any spread trades should be closed based on z-score."""
        from spread_strategy import SpreadSignalType

        for trade in self.active_spread_trades[:]:
            if trade.status != TradeStatus.OPEN:
                continue

            signal = trade.signal
            is_long = signal.signal_type == SpreadSignalType.LONG_SPREAD

            should_close = False
            exit_reason = None

            if is_long:
                # TP: z rises to -tp_z (closer to 0)
                if current_zscore >= -signal.tp_z:
                    should_close = True
                    exit_reason = "take_profit"
                # SL: z drops below -sl_z
                elif current_zscore < -signal.sl_z:
                    should_close = True
                    exit_reason = "stop_loss"
            else:
                # TP: z drops to +tp_z
                if current_zscore <= signal.tp_z:
                    should_close = True
                    exit_reason = "take_profit"
                # SL: z rises above +sl_z
                elif current_zscore > signal.sl_z:
                    should_close = True
                    exit_reason = "stop_loss"

            if should_close:
                self.close_spread_trade(trade, exit_reason, current_zscore)

    def close_spread_trade(self, trade: SpreadTrade, reason: str, exit_zscore: float = None):
        """Close a spread trade (both legs)."""
        from spread_strategy import SpreadSignalType

        if trade.status != TradeStatus.OPEN:
            return

        signal = trade.signal
        is_long = signal.signal_type == SpreadSignalType.LONG_SPREAD

        # Close sides (opposite of entry)
        close_side_a = "Sell" if is_long else "Buy"
        close_side_b = "Buy" if is_long else "Sell"

        try:
            # Close leg A
            self.client.place_order(
                symbol=trade.leg_a_symbol,
                side=close_side_a,
                qty=trade.leg_a_size,
                order_type="Market",
                reduce_only=True
            )

            # Close leg B
            self.client.place_order(
                symbol=trade.leg_b_symbol,
                side=close_side_b,
                qty=trade.leg_b_size,
                order_type="Market",
                reduce_only=True
            )

            # Calculate approximate P&L
            # For proper P&L, we'd need to fetch actual fill prices
            # This is a rough estimate based on z-score movement
            z_move = abs(signal.entry_z) - abs(exit_zscore or 0)
            # Rough P&L estimate (would need refinement with actual prices)
            trade.realized_pnl = z_move * 100  # Placeholder

            if reason == "take_profit":
                trade.status = TradeStatus.CLOSED_TP
            elif reason == "stop_loss":
                trade.status = TradeStatus.CLOSED_SL
            else:
                trade.status = TradeStatus.CLOSED_MANUAL

            trade.exit_reason = reason
            self.daily_pnl += trade.realized_pnl

            # Move to closed trades
            self.active_spread_trades.remove(trade)
            self.closed_spread_trades.append(trade)

            print(f"Spread trade closed: {reason}")
            print(f"  Exit Z: {exit_zscore:.2f}" if exit_zscore else "")

            # Notify
            if self.notifier:
                emoji = "✅" if reason == "take_profit" else "❌"
                self.notifier.send_message(
                    f"{emoji} SPREAD TRADE CLOSED\n"
                    f"Reason: {reason}\n"
                    f"Exit Z: {exit_zscore:.2f}" if exit_zscore else ""
                )

        except Exception as e:
            print(f"Spread trade close failed: {e}")

    def close_all_spread_trades(self, reason: str = "shutdown"):
        """Close all open spread trades."""
        for trade in self.active_spread_trades[:]:
            if trade.status == TradeStatus.OPEN:
                self.close_spread_trade(trade, reason)

    def get_spread_stats(self) -> Dict:
        """Get spread trading statistics."""
        total = len(self.closed_spread_trades)
        winners = [t for t in self.closed_spread_trades if t.realized_pnl > 0]
        losers = [t for t in self.closed_spread_trades if t.realized_pnl <= 0]

        return {
            "total_spread_trades": total,
            "spread_winners": len(winners),
            "spread_losers": len(losers),
            "spread_win_rate": len(winners) / total if total > 0 else 0,
            "spread_pnl": sum(t.realized_pnl for t in self.closed_spread_trades),
            "active_spread_trades": len(self.active_spread_trades)
        }

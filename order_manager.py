"""
Order Manager - Handles order execution, SL/TP, and position tracking
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from bybit_client import BybitClient, Position, Order
from strategy import TradeSignal, SignalType, SignalStatus
from config import BotConfig


class TradeStatus(Enum):
    PENDING = "pending"  # Order not yet placed
    OPEN = "open"  # Position is open
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

    # P&L tracking
    realized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


class OrderManager:
    """Manages order execution and position lifecycle."""

    # Minimum qty and step size for each symbol (Bybit USDT Perpetual)
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

    def __init__(self, config: BotConfig, client: BybitClient, notifier=None):
        self.config = config
        self.client = client
        self.notifier = notifier
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.daily_pnl: float = 0.0

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

    def can_open_trade(self, account_balance: float) -> tuple[bool, str]:
        """Check if we can open a new trade based on risk rules."""
        # Check max positions
        open_trades = [t for t in self.active_trades if t.status == TradeStatus.OPEN]
        if len(open_trades) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

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

        # Check if we can trade
        can_trade, reason = self.can_open_trade(account_balance)
        if not can_trade:
            print(f"Cannot open trade: {reason}")
            return None

        # Calculate position size
        position_size = self.calculate_position_size(signal, account_balance)
        if position_size <= 0:
            print("Position size too small")
            return None

        # Determine side
        side = "Buy" if signal.signal_type == SignalType.LONG_KING else "Sell"

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
            trade = Trade(
                signal=signal,
                status=TradeStatus.OPEN,
                entry_order_id=order.order_id,
                entry_filled_price=signal.entry_price if self.config.use_limit_orders else None,
                position_size=position_size
            )

            self.active_trades.append(trade)
            signal.status = SignalStatus.FILLED

            print(f"Order placed: {side} {position_size} {signal.symbol} @ {signal.entry_price}")
            print(f"  SL: {signal.stop_loss}, TP: {signal.target}")

            # Send Telegram notification
            if self.notifier:
                self.notifier.notify_trade_opened(
                    symbol=signal.symbol,
                    side=side,
                    entry=signal.entry_price,
                    sl=signal.stop_loss,
                    tp=signal.target,
                    size=position_size,
                    rr=signal.get_risk_reward()
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

            for trade in self.active_trades[:]:
                if trade.status != TradeStatus.OPEN:
                    continue

                # Get position for this trade's symbol
                exchange_position = positions_by_symbol.get(trade.signal.symbol)

                # Check if position still exists
                if exchange_position is None or exchange_position.size == 0:
                    # Position was closed (SL or TP hit)
                    trade.status = TradeStatus.CLOSED_SL  # Assume SL, will refine
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
            if trade.signal.signal_type == SignalType.LONG_KING:
                trade.realized_pnl = (trade.exit_price - trade.entry_filled_price) * trade.position_size
            else:
                trade.realized_pnl = (trade.entry_filled_price - trade.exit_price) * trade.position_size

            self.daily_pnl += trade.realized_pnl

        print(f"Trade closed: {trade.status.value}, P&L: ${trade.realized_pnl:.2f}")

        # Send Telegram notification
        if self.notifier:
            side = "LONG" if trade.signal.signal_type == SignalType.LONG_KING else "SHORT"
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
            close_side = "Sell" if trade.signal.signal_type == SignalType.LONG_KING else "Buy"

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
        """Cancel all pending orders for given symbols."""
        symbols = symbols or []
        try:
            for symbol in symbols:
                self.client.cancel_all_orders(symbol)
            for trade in self.active_trades[:]:
                if trade.status == TradeStatus.PENDING:
                    trade.status = TradeStatus.CANCELLED
                    self.active_trades.remove(trade)
        except Exception as e:
            print(f"Cancel orders failed: {e}")

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return {"total_trades": 0}

        winning_trades = [t for t in self.closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.realized_pnl < 0]

        total_profit = sum(t.realized_pnl for t in winning_trades)
        total_loss = abs(sum(t.realized_pnl for t in losing_trades))

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "total_pnl": sum(t.realized_pnl for t in self.closed_trades),
            "daily_pnl": self.daily_pnl,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "open_trades": len([t for t in self.active_trades if t.status == TradeStatus.OPEN])
        }

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)."""
        self.daily_pnl = 0.0

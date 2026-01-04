"""
Trade Performance Tracker - SQLite-based performance tracking system
"""
import sqlite3
import json
import uuid
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager


@dataclass
class TradeRecord:
    """Represents a complete trade record for storage."""
    trade_id: str
    symbol: str
    setup_key: str
    signal_type: str
    entry_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    planned_entry_price: float = 0.0
    position_size: float = 0.0
    position_value_usd: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_amount_usd: Optional[float] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    entry_slippage: Optional[float] = None
    entry_slippage_pct: Optional[float] = None
    fees_paid: float = 0.0
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    step_0_price: Optional[float] = None
    step_3_price: Optional[float] = None
    fvg_top: Optional[float] = None
    fvg_bottom: Optional[float] = None
    status: str = "pending"
    account_equity_at_entry: Optional[float] = None


class TradeTracker:
    """
    SQLite-based trade performance tracking system.

    Handles:
    - Recording trades (open, update, close)
    - Equity snapshots for drawdown analysis
    - Daily statistics aggregation
    - Performance queries
    """

    def __init__(self, db_path: str = None):
        """Initialize tracker with database path."""
        if db_path is None:
            bot_dir = Path(__file__).parent
            data_dir = bot_dir / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "trading_performance.db")

        self.db_path = db_path
        self._init_database()
        self._peak_equity: Optional[float] = None
        self._load_peak_equity()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    setup_key TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_order_id TEXT,
                    entry_price REAL,
                    planned_entry_price REAL NOT NULL,
                    position_size REAL NOT NULL,
                    position_value_usd REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_amount_usd REAL,
                    realized_pnl REAL DEFAULT 0.0,
                    realized_pnl_pct REAL,
                    r_multiple REAL,
                    entry_slippage REAL,
                    entry_slippage_pct REAL,
                    fees_paid REAL DEFAULT 0.0,
                    opened_at TIMESTAMP,
                    closed_at TIMESTAMP,
                    duration_seconds INTEGER,
                    step_0_price REAL,
                    step_3_price REAL,
                    fvg_top REAL,
                    fvg_bottom REAL,
                    status TEXT NOT NULL,
                    account_equity_at_entry REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Daily stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    cancelled_trades INTEGER DEFAULT 0,
                    gross_profit REAL DEFAULT 0.0,
                    gross_loss REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    largest_win REAL DEFAULT 0.0,
                    largest_loss REAL DEFAULT 0.0,
                    total_r_won REAL DEFAULT 0.0,
                    total_r_lost REAL DEFAULT 0.0,
                    avg_r_per_trade REAL DEFAULT 0.0,
                    starting_equity REAL,
                    ending_equity REAL,
                    equity_high REAL,
                    equity_low REAL,
                    long_trades INTEGER DEFAULT 0,
                    short_trades INTEGER DEFAULT 0,
                    pnl_by_symbol TEXT,
                    trades_by_symbol TEXT,
                    win_streak INTEGER DEFAULT 0,
                    loss_streak INTEGER DEFAULT 0,
                    first_trade_time TIMESTAMP,
                    last_trade_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Equity snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    equity REAL NOT NULL,
                    available_balance REAL,
                    unrealized_pnl REAL,
                    open_positions INTEGER DEFAULT 0,
                    positions_value REAL DEFAULT 0.0,
                    peak_equity REAL,
                    drawdown_amount REAL DEFAULT 0.0,
                    drawdown_pct REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_setup_key ON trades(setup_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_snapshots_timestamp ON equity_snapshots(timestamp)")

    def _load_peak_equity(self):
        """Load peak equity from metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM performance_meta WHERE key = 'peak_equity'")
            row = cursor.fetchone()
            if row:
                self._peak_equity = float(row['value'])

    def _save_peak_equity(self, peak: float):
        """Save peak equity to metadata."""
        self._peak_equity = peak
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_meta (key, value, updated_at)
                VALUES ('peak_equity', ?, CURRENT_TIMESTAMP)
            """, (str(peak),))

    # ==================== TRADE RECORDING ====================

    def record_trade_opened(
        self,
        symbol: str,
        setup_key: str,
        signal_type: str,
        entry_order_id: str,
        planned_entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        account_equity: float,
        step_0_price: float = None,
        step_3_price: float = None,
        fvg_top: float = None,
        fvg_bottom: float = None
    ) -> str:
        """
        Record a new trade when it opens.

        Returns:
            trade_id: Unique identifier for the trade
        """
        trade_id = str(uuid.uuid4())[:8]

        # Calculate risk amount
        is_long = "long" in signal_type.lower()
        if is_long:
            risk_per_unit = planned_entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - planned_entry_price
        risk_amount_usd = abs(risk_per_unit * position_size)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, setup_key, signal_type,
                    entry_order_id, planned_entry_price, position_size,
                    stop_loss, take_profit, risk_amount_usd,
                    step_0_price, step_3_price, fvg_top, fvg_bottom,
                    status, account_equity_at_entry, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, setup_key, signal_type,
                entry_order_id, planned_entry_price, position_size,
                stop_loss, take_profit, risk_amount_usd,
                step_0_price, step_3_price, fvg_top, fvg_bottom,
                "pending_fill", account_equity, datetime.utcnow()
            ))

        return trade_id

    def update_trade_fill(
        self,
        trade_id: str,
        entry_price: float,
        position_value_usd: float = None
    ):
        """Update trade when limit order fills."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get planned entry to calculate slippage
            cursor.execute("SELECT planned_entry_price FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            if row:
                planned = row['planned_entry_price']
                slippage = entry_price - planned
                slippage_pct = (slippage / planned) * 100 if planned else 0

                cursor.execute("""
                    UPDATE trades SET
                        entry_price = ?,
                        position_value_usd = ?,
                        entry_slippage = ?,
                        entry_slippage_pct = ?,
                        status = 'open',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                """, (entry_price, position_value_usd, slippage, slippage_pct, trade_id))

    def record_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str,
        fees_paid: float = 0.0
    ):
        """Update trade record when closed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get trade details for calculations
            cursor.execute("""
                SELECT entry_price, position_size, risk_amount_usd, opened_at,
                       position_value_usd, symbol
                FROM trades WHERE trade_id = ?
            """, (trade_id,))
            row = cursor.fetchone()

            if not row:
                return

            entry_price = row['entry_price'] or 0
            position_value = row['position_value_usd'] or (entry_price * row['position_size'])
            risk_amount = row['risk_amount_usd'] or 1
            opened_at = row['opened_at']
            symbol = row['symbol']

            # Calculate metrics
            closed_at = datetime.utcnow()
            duration_seconds = None
            if opened_at:
                opened_dt = datetime.fromisoformat(opened_at) if isinstance(opened_at, str) else opened_at
                duration_seconds = int((closed_at - opened_dt).total_seconds())

            realized_pnl_pct = (realized_pnl / position_value) * 100 if position_value else 0
            r_multiple = realized_pnl / risk_amount if risk_amount else 0

            # Determine status
            if exit_reason == "tp":
                status = "closed_tp"
            elif exit_reason == "sl":
                status = "closed_sl"
            else:
                status = "closed_manual"

            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    exit_reason = ?,
                    realized_pnl = ?,
                    realized_pnl_pct = ?,
                    r_multiple = ?,
                    fees_paid = ?,
                    closed_at = ?,
                    duration_seconds = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """, (
                exit_price, exit_reason, realized_pnl, realized_pnl_pct,
                r_multiple, fees_paid, closed_at, duration_seconds, status, trade_id
            ))

            # Update daily stats
            self._update_daily_stats_for_trade(
                trade_date=closed_at.date(),
                symbol=symbol,
                pnl=realized_pnl,
                r_multiple=r_multiple,
                is_long="long" in row.get('signal_type', '').lower() if row else True
            )

    def record_trade_cancelled(self, trade_id: str):
        """Mark a pending trade as cancelled."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades SET
                    status = 'cancelled',
                    closed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """, (trade_id,))

    # ==================== EQUITY TRACKING ====================

    def record_equity_snapshot(
        self,
        equity: float,
        available_balance: float = None,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        positions_value: float = 0.0
    ):
        """Record periodic equity snapshot."""
        # Update peak equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._save_peak_equity(equity)

        # Calculate drawdown
        drawdown_amount = self._peak_equity - equity if self._peak_equity else 0
        drawdown_pct = (drawdown_amount / self._peak_equity) * 100 if self._peak_equity else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO equity_snapshots (
                    timestamp, equity, available_balance, unrealized_pnl,
                    open_positions, positions_value, peak_equity,
                    drawdown_amount, drawdown_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow(), equity, available_balance, unrealized_pnl,
                open_positions, positions_value, self._peak_equity,
                drawdown_amount, drawdown_pct
            ))

            # Update daily stats equity
            today = date.today()
            cursor.execute("""
                INSERT INTO daily_stats (date, equity_high, equity_low, ending_equity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    equity_high = MAX(COALESCE(equity_high, 0), ?),
                    equity_low = CASE
                        WHEN equity_low IS NULL THEN ?
                        ELSE MIN(equity_low, ?)
                    END,
                    ending_equity = ?,
                    updated_at = CURRENT_TIMESTAMP
            """, (today, equity, equity, equity, equity, equity, equity, equity))

    def get_current_drawdown(self) -> Dict[str, float]:
        """Get current drawdown from peak."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT equity, peak_equity, drawdown_amount, drawdown_pct
                FROM equity_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()

            if row:
                return {
                    "current_equity": row['equity'],
                    "peak_equity": row['peak_equity'],
                    "drawdown_amount": row['drawdown_amount'],
                    "drawdown_pct": row['drawdown_pct']
                }
            return {"current_equity": 0, "peak_equity": 0, "drawdown_amount": 0, "drawdown_pct": 0}

    def get_max_drawdown(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """Calculate maximum drawdown for period."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT timestamp, equity, drawdown_amount, drawdown_pct
                FROM equity_snapshots
            """
            params = []

            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("DATE(timestamp) >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    conditions.append("DATE(timestamp) <= ?")
                    params.append(end_date.isoformat())
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY drawdown_pct DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            if row:
                return {
                    "date": row['timestamp'],
                    "amount": row['drawdown_amount'],
                    "pct": row['drawdown_pct'],
                    "peak": row['equity'] + row['drawdown_amount']
                }
            return {"date": None, "amount": 0, "pct": 0, "peak": 0}

    # ==================== DAILY STATS ====================

    def _update_daily_stats_for_trade(
        self,
        trade_date: date,
        symbol: str,
        pnl: float,
        r_multiple: float,
        is_long: bool
    ):
        """Update daily stats when a trade closes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current stats
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (trade_date.isoformat(),))
            row = cursor.fetchone()

            if row:
                # Update existing
                total_trades = row['total_trades'] + 1
                winning = row['winning_trades'] + (1 if pnl > 0 else 0)
                losing = row['losing_trades'] + (1 if pnl < 0 else 0)
                gross_profit = row['gross_profit'] + (pnl if pnl > 0 else 0)
                gross_loss = row['gross_loss'] + (abs(pnl) if pnl < 0 else 0)
                net_pnl = row['net_pnl'] + pnl
                largest_win = max(row['largest_win'], pnl) if pnl > 0 else row['largest_win']
                largest_loss = min(row['largest_loss'], pnl) if pnl < 0 else row['largest_loss']
                total_r_won = row['total_r_won'] + (r_multiple if r_multiple > 0 else 0)
                total_r_lost = row['total_r_lost'] + (abs(r_multiple) if r_multiple < 0 else 0)
                long_trades = row['long_trades'] + (1 if is_long else 0)
                short_trades = row['short_trades'] + (0 if is_long else 1)

                # Update pnl_by_symbol
                pnl_by_symbol = json.loads(row['pnl_by_symbol'] or '{}')
                pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0) + pnl

                trades_by_symbol = json.loads(row['trades_by_symbol'] or '{}')
                trades_by_symbol[symbol] = trades_by_symbol.get(symbol, 0) + 1

                # Calculate streaks
                win_streak = row['win_streak']
                loss_streak = row['loss_streak']
                if pnl > 0:
                    win_streak = (win_streak if win_streak > 0 else 0) + 1
                    loss_streak = 0
                elif pnl < 0:
                    loss_streak = (loss_streak if loss_streak > 0 else 0) + 1
                    win_streak = 0

                cursor.execute("""
                    UPDATE daily_stats SET
                        total_trades = ?,
                        winning_trades = ?,
                        losing_trades = ?,
                        gross_profit = ?,
                        gross_loss = ?,
                        net_pnl = ?,
                        largest_win = ?,
                        largest_loss = ?,
                        total_r_won = ?,
                        total_r_lost = ?,
                        avg_r_per_trade = ?,
                        long_trades = ?,
                        short_trades = ?,
                        pnl_by_symbol = ?,
                        trades_by_symbol = ?,
                        win_streak = ?,
                        loss_streak = ?,
                        last_trade_time = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE date = ?
                """, (
                    total_trades, winning, losing, gross_profit, gross_loss, net_pnl,
                    largest_win, largest_loss, total_r_won, total_r_lost,
                    (total_r_won - total_r_lost) / total_trades if total_trades > 0 else 0,
                    long_trades, short_trades,
                    json.dumps(pnl_by_symbol), json.dumps(trades_by_symbol),
                    win_streak, loss_streak, trade_date.isoformat()
                ))
            else:
                # Insert new
                pnl_by_symbol = {symbol: pnl}
                trades_by_symbol = {symbol: 1}

                cursor.execute("""
                    INSERT INTO daily_stats (
                        date, total_trades, winning_trades, losing_trades,
                        gross_profit, gross_loss, net_pnl, largest_win, largest_loss,
                        total_r_won, total_r_lost, avg_r_per_trade,
                        long_trades, short_trades, pnl_by_symbol, trades_by_symbol,
                        win_streak, loss_streak, first_trade_time, last_trade_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    trade_date.isoformat(),
                    1, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0,
                    pnl if pnl > 0 else 0, abs(pnl) if pnl < 0 else 0, pnl,
                    pnl if pnl > 0 else 0, pnl if pnl < 0 else 0,
                    r_multiple if r_multiple > 0 else 0,
                    abs(r_multiple) if r_multiple < 0 else 0,
                    r_multiple,
                    1 if is_long else 0, 0 if is_long else 1,
                    json.dumps(pnl_by_symbol), json.dumps(trades_by_symbol),
                    1 if pnl > 0 else 0, 1 if pnl < 0 else 0
                ))

    # ==================== STATISTICS QUERIES ====================

    def get_stats(
        self,
        period: str = "all",
        symbol: str = None,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        # Calculate date range based on period
        if period == "today":
            start_date = date.today()
            end_date = date.today()
        elif period == "week":
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
        elif period == "month":
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
        elif period == "year":
            end_date = date.today()
            start_date = end_date - timedelta(days=365)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query
            query = """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss,
                    SUM(realized_pnl) as total_pnl,
                    AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
                    AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl END) as avg_loss,
                    MAX(realized_pnl) as largest_win,
                    MIN(realized_pnl) as largest_loss,
                    AVG(r_multiple) as avg_r_multiple,
                    AVG(duration_seconds) as avg_duration_seconds
                FROM trades
                WHERE status IN ('closed_tp', 'closed_sl', 'closed_manual')
            """
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND DATE(closed_at) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND DATE(closed_at) <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
            row = cursor.fetchone()

            total_trades = row['total_trades'] or 0
            winning = row['winning_trades'] or 0
            losing = row['losing_trades'] or 0
            gross_profit = row['gross_profit'] or 0
            gross_loss = row['gross_loss'] or 0

            # Calculate derived metrics
            win_rate = winning / total_trades if total_trades > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            expectancy = (row['total_pnl'] or 0) / total_trades if total_trades > 0 else 0

            # Get max drawdown
            max_dd = self.get_max_drawdown(start_date, end_date)

            # Get streaks
            streaks = self.get_streaks()

            # Format duration
            avg_duration_sec = row['avg_duration_seconds'] or 0
            if avg_duration_sec > 3600:
                avg_duration = f"{avg_duration_sec / 3600:.1f}h"
            elif avg_duration_sec > 60:
                avg_duration = f"{avg_duration_sec / 60:.0f}m"
            else:
                avg_duration = f"{avg_duration_sec:.0f}s"

            return {
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": win_rate,
                "total_pnl": row['total_pnl'] or 0,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor,
                "avg_win": row['avg_win'] or 0,
                "avg_loss": row['avg_loss'] or 0,
                "largest_win": row['largest_win'] or 0,
                "largest_loss": row['largest_loss'] or 0,
                "avg_r_multiple": row['avg_r_multiple'] or 0,
                "expectancy": expectancy,
                "max_drawdown_pct": max_dd['pct'],
                "avg_duration": avg_duration,
                "win_streak": streaks.get('current_win_streak', 0),
                "loss_streak": streaks.get('current_loss_streak', 0),
                "max_win_streak": streaks.get('max_win_streak', 0),
                "max_loss_streak": streaks.get('max_loss_streak', 0)
            }

    def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: str = None,
        direction: str = None,
        status: str = None,
        start_date: date = None,
        end_date: date = None,
        min_pnl: float = None,
        max_pnl: float = None,
        order_by: str = "closed_at",
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """Get filtered list of trades."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if direction:
                if direction.lower() == "long":
                    query += " AND signal_type LIKE '%long%'"
                else:
                    query += " AND signal_type LIKE '%short%'"
            if status:
                if status == "open":
                    query += " AND status IN ('pending_fill', 'open')"
                elif status == "closed":
                    query += " AND status IN ('closed_tp', 'closed_sl', 'closed_manual')"
                else:
                    query += " AND status = ?"
                    params.append(status)
            if start_date:
                query += " AND DATE(COALESCE(closed_at, opened_at)) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND DATE(COALESCE(closed_at, opened_at)) <= ?"
                params.append(end_date.isoformat())
            if min_pnl is not None:
                query += " AND realized_pnl >= ?"
                params.append(min_pnl)
            if max_pnl is not None:
                query += " AND realized_pnl <= ?"
                params.append(max_pnl)

            order_dir = "DESC" if descending else "ASC"
            query += f" ORDER BY {order_by} {order_dir} LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            trades = []
            for row in rows:
                duration_sec = row['duration_seconds'] or 0
                if duration_sec > 3600:
                    duration_str = f"{duration_sec / 3600:.1f}h"
                elif duration_sec > 60:
                    duration_str = f"{duration_sec / 60:.0f}m"
                else:
                    duration_str = f"{duration_sec:.0f}s" if duration_sec > 0 else "-"

                trades.append({
                    "trade_id": row['trade_id'],
                    "symbol": row['symbol'],
                    "signal_type": row['signal_type'],
                    "entry_price": row['entry_price'],
                    "exit_price": row['exit_price'],
                    "position_size": row['position_size'],
                    "realized_pnl": row['realized_pnl'],
                    "r_multiple": row['r_multiple'],
                    "status": row['status'],
                    "opened_at": row['opened_at'],
                    "closed_at": row['closed_at'],
                    "duration_seconds": row['duration_seconds'],
                    "duration_str": duration_str
                })

            return trades

    def get_daily_stats(
        self,
        start_date: date = None,
        end_date: date = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM daily_stats WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())

            query += f" ORDER BY date DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_asset_breakdown(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by asset."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(realized_pnl) as pnl,
                    AVG(r_multiple) as avg_r,
                    SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss
                FROM trades
                WHERE status IN ('closed_tp', 'closed_sl', 'closed_manual')
            """
            params = []

            if start_date:
                query += " AND DATE(closed_at) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND DATE(closed_at) <= ?"
                params.append(end_date.isoformat())

            query += " GROUP BY symbol ORDER BY pnl DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            breakdown = {}
            for row in rows:
                trades = row['trades']
                wins = row['wins'] or 0
                gross_profit = row['gross_profit'] or 0
                gross_loss = row['gross_loss'] or 0

                breakdown[row['symbol']] = {
                    "trades": trades,
                    "wins": wins,
                    "losses": trades - wins,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "pnl": row['pnl'] or 0,
                    "avg_r": row['avg_r'] or 0,
                    "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
                }

            return breakdown

    def get_time_analysis(
        self,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """Get time-based performance analysis."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    strftime('%H', closed_at) as hour,
                    strftime('%w', closed_at) as day_of_week,
                    COUNT(*) as trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(realized_pnl) as pnl
                FROM trades
                WHERE status IN ('closed_tp', 'closed_sl', 'closed_manual')
                    AND closed_at IS NOT NULL
            """
            params = []

            if start_date:
                query += " AND DATE(closed_at) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND DATE(closed_at) <= ?"
                params.append(end_date.isoformat())

            # By hour
            hour_query = query + " GROUP BY hour ORDER BY hour"
            cursor.execute(hour_query, params)
            hour_rows = cursor.fetchall()

            by_hour = {}
            for row in hour_rows:
                hour = int(row['hour'])
                trades = row['trades']
                wins = row['wins'] or 0
                by_hour[hour] = {
                    "hour": hour,
                    "trades": trades,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "pnl": row['pnl'] or 0
                }

            # Sort by P&L for best/worst
            sorted_hours = sorted(by_hour.values(), key=lambda x: x['pnl'], reverse=True)
            best_hours = sorted_hours[:5] if sorted_hours else []
            worst_hours = sorted_hours[-5:][::-1] if len(sorted_hours) >= 5 else sorted_hours[::-1]

            # By day of week
            dow_query = query + " GROUP BY day_of_week ORDER BY day_of_week"
            cursor.execute(dow_query, params)
            dow_rows = cursor.fetchall()

            day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            by_day = {}
            for row in dow_rows:
                day_idx = int(row['day_of_week'])
                trades = row['trades']
                wins = row['wins'] or 0
                by_day[day_names[day_idx]] = {
                    "trades": trades,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "pnl": row['pnl'] or 0
                }

            return {
                "by_hour": by_hour,
                "by_day_of_week": by_day,
                "best_hours": best_hours,
                "worst_hours": worst_hours
            }

    def get_sessions(
        self,
        limit: int = 10,
        sort_by: str = "pnl"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get best and worst trading sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT date, total_trades, winning_trades, net_pnl
                FROM daily_stats
                WHERE total_trades > 0
                ORDER BY net_pnl DESC
                LIMIT ?
            """, (limit,))
            best_rows = cursor.fetchall()

            cursor.execute("""
                SELECT date, total_trades, winning_trades, net_pnl
                FROM daily_stats
                WHERE total_trades > 0
                ORDER BY net_pnl ASC
                LIMIT ?
            """, (limit,))
            worst_rows = cursor.fetchall()

            def format_session(row):
                trades = row['total_trades']
                wins = row['winning_trades'] or 0
                return {
                    "date": row['date'],
                    "trades": trades,
                    "win_rate": wins / trades if trades > 0 else 0,
                    "pnl": row['net_pnl']
                }

            return {
                "best": [format_session(r) for r in best_rows],
                "worst": [format_session(r) for r in worst_rows]
            }

    def get_streaks(self) -> Dict[str, int]:
        """Get current and max win/loss streaks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT realized_pnl
                FROM trades
                WHERE status IN ('closed_tp', 'closed_sl', 'closed_manual')
                ORDER BY closed_at DESC
            """)
            rows = cursor.fetchall()

            current_win = 0
            current_loss = 0
            max_win = 0
            max_loss = 0

            # Calculate current streak
            if rows:
                first_pnl = rows[0]['realized_pnl']
                if first_pnl > 0:
                    for row in rows:
                        if row['realized_pnl'] > 0:
                            current_win += 1
                        else:
                            break
                elif first_pnl < 0:
                    for row in rows:
                        if row['realized_pnl'] < 0:
                            current_loss += 1
                        else:
                            break

            # Calculate max streaks
            win_streak = 0
            loss_streak = 0
            for row in rows:
                if row['realized_pnl'] > 0:
                    win_streak += 1
                    loss_streak = 0
                    max_win = max(max_win, win_streak)
                elif row['realized_pnl'] < 0:
                    loss_streak += 1
                    win_streak = 0
                    max_loss = max(max_loss, loss_streak)

            return {
                "current_win_streak": current_win,
                "current_loss_streak": current_loss,
                "max_win_streak": max_win,
                "max_loss_streak": max_loss
            }

    def get_equity_curve(
        self,
        start_date: date = None,
        end_date: date = None,
        interval: str = "1h"
    ) -> List[Dict[str, Any]]:
        """Get equity curve data points."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT timestamp, equity, drawdown_pct FROM equity_snapshots WHERE 1=1"
            params = []

            if start_date:
                query += " AND DATE(timestamp) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND DATE(timestamp) <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY timestamp"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [{"timestamp": r['timestamp'], "equity": r['equity'], "drawdown_pct": r['drawdown_pct']} for r in rows]

    # ==================== EXPORT ====================

    def export_trades(
        self,
        filepath: str,
        format: str = "csv",
        start_date: date = None,
        end_date: date = None
    ) -> str:
        """Export trades to file."""
        trades = self.get_trades(limit=10000, start_date=start_date, end_date=end_date)

        if format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
        else:
            import csv
            if trades:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    writer.writerows(trades)

        return filepath

    def export_daily_stats(
        self,
        filepath: str,
        format: str = "csv"
    ) -> str:
        """Export daily stats to file."""
        stats = self.get_daily_stats(limit=1000)

        if format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        else:
            import csv
            if stats:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=stats[0].keys())
                    writer.writeheader()
                    writer.writerows(stats)

        return filepath


# Singleton instance for global access
_tracker_instance: Optional[TradeTracker] = None


def get_tracker() -> TradeTracker:
    """Get or create the global TradeTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TradeTracker()
    return _tracker_instance

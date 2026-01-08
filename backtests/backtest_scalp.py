"""
Scalping Strategy Backtest - Multi-Entry with Partial Exit System

Tests 3 entry types:
1. FVG Breakout (proven from Breakaway strategy)
2. Imbalance Flip (new)
3. EWVMA Momentum Touch (new)

Exit System:
- 50% at 1.0R
- 50% at 1.5R (average realized: 1.25R per winner)
- SL to breakeven after TP1

Target: 60%+ win rate, 5-15 trades/day, 0.35R+ expectancy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from breakaway_strategy import BreakawayIndicators


class EntryType(Enum):
    FVG_BREAKOUT = "fvg_breakout"
    IMBALANCE_FLIP = "imbalance_flip"
    EWVMA_TOUCH = "ewvma_touch"


class ExitPhase(Enum):
    INITIAL = "initial"
    PARTIAL_TP1 = "partial_tp1"
    CLOSED = "closed"


@dataclass
class ScalpTrade:
    """Represents a scalping trade."""
    entry_type: EntryType
    direction: str  # "long" or "short"
    entry_idx: int
    entry_price: float
    stop_loss: float
    tp1_price: float  # 1.0R
    tp2_price: float  # 1.5R

    # Context
    volume_ratio: float
    imbalance: float

    # Results (filled after exit)
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_r: Optional[float] = None

    @property
    def risk(self) -> float:
        return abs(self.stop_loss - self.entry_price)

    @property
    def hold_time(self) -> int:
        if self.exit_idx is not None:
            return self.exit_idx - self.entry_idx
        return 0


@dataclass
class ScalpConfig:
    """Scalping backtest configuration."""
    # Entry thresholds
    min_vol_ratio: float = 1.5
    imbalance_threshold: float = 0.10
    imbalance_flip_threshold: float = 0.15
    min_cradle_candles: int = 3
    cradle_lookback: int = 5

    # Exit system
    tp1_r_multiple: float = 1.0
    tp2_r_multiple: float = 1.5
    tp1_close_pct: float = 0.50
    sl_buffer_pct: float = 0.001

    # Timeout
    max_hold_candles: int = 30

    # Direction
    trade_direction: str = "both"  # "both", "shorts", "longs"

    # Entry types to test
    use_fvg_breakout: bool = True
    use_imbalance_flip: bool = True
    use_ewvma_touch: bool = True


class ScalpBacktest:
    """Scalping strategy backtester with partial exits."""

    def __init__(self, config: ScalpConfig):
        self.config = config
        self.indicators = BreakawayIndicators()
        self.trades: List[ScalpTrade] = []

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all required indicators."""
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        return self.indicators.calculate_all(opens, closes, highs, lows, volumes, imbalance_lookback=10)

    def _detect_fvg_breakout(self, df: pd.DataFrame, ind: Dict, idx: int) -> Optional[ScalpTrade]:
        """Entry Type 1: FVG Breakout (from Breakaway strategy)."""
        if idx < 5:
            return None

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        vol_ratio = ind['vol_ratio'][idx]
        imbalance = ind['imbalance'][idx]
        in_cradle = ind['in_cradle']

        # Check cradle
        cradle_count = np.sum(in_cradle[max(0, idx - self.config.cradle_lookback):idx])
        if cradle_count < self.config.min_cradle_candles:
            return None

        # Check volume
        if vol_ratio < self.config.min_vol_ratio:
            return None

        # Check for SHORT: Bearish FVG + selling pressure
        if self.config.trade_direction in ["both", "shorts"]:
            if highs[idx] < lows[idx - 2]:  # Bearish FVG
                if imbalance <= -self.config.imbalance_threshold:
                    fvg_top = lows[idx - 2]
                    fvg_bottom = highs[idx]
                    entry = fvg_bottom
                    sl = fvg_top * (1 + self.config.sl_buffer_pct)
                    risk = sl - entry
                    tp1 = entry - (risk * self.config.tp1_r_multiple)
                    tp2 = entry - (risk * self.config.tp2_r_multiple)

                    return ScalpTrade(
                        entry_type=EntryType.FVG_BREAKOUT,
                        direction="short",
                        entry_idx=idx,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        volume_ratio=vol_ratio,
                        imbalance=imbalance,
                    )

        # Check for LONG: Bullish FVG + buying pressure
        if self.config.trade_direction in ["both", "longs"]:
            if lows[idx] > highs[idx - 2]:  # Bullish FVG
                if imbalance >= self.config.imbalance_threshold:
                    fvg_top = lows[idx]
                    fvg_bottom = highs[idx - 2]
                    entry = fvg_top
                    sl = fvg_bottom * (1 - self.config.sl_buffer_pct)
                    risk = entry - sl
                    tp1 = entry + (risk * self.config.tp1_r_multiple)
                    tp2 = entry + (risk * self.config.tp2_r_multiple)

                    return ScalpTrade(
                        entry_type=EntryType.FVG_BREAKOUT,
                        direction="long",
                        entry_idx=idx,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        volume_ratio=vol_ratio,
                        imbalance=imbalance,
                    )

        return None

    def _detect_imbalance_flip(self, df: pd.DataFrame, ind: Dict, idx: int) -> Optional[ScalpTrade]:
        """Entry Type 2: Imbalance Flip at EWVMA band."""
        if idx < 5:
            return None

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        imbalance = ind['imbalance'][idx]
        prev_imbalance = ind['imbalance'][idx - 1] if idx > 0 else 0
        ewvma_upper = ind['ewvma_20_upper'][idx]
        ewvma_lower = ind['ewvma_20_lower'][idx]
        vol_ratio = ind['vol_ratio'][idx]

        # Volume confirmation
        if vol_ratio < self.config.min_vol_ratio:
            return None

        # SHORT: Imbalance flips to selling + price at upper band
        if self.config.trade_direction in ["both", "shorts"]:
            if prev_imbalance >= 0 and imbalance <= -self.config.imbalance_flip_threshold:
                if highs[idx] >= ewvma_upper * 0.998:  # Near upper band
                    entry = closes[idx]
                    sl = entry * 1.003  # 0.3% stop
                    risk = sl - entry
                    tp1 = entry - (risk * self.config.tp1_r_multiple)
                    tp2 = entry - (risk * self.config.tp2_r_multiple)

                    return ScalpTrade(
                        entry_type=EntryType.IMBALANCE_FLIP,
                        direction="short",
                        entry_idx=idx,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        volume_ratio=vol_ratio,
                        imbalance=imbalance,
                    )

        # LONG: Imbalance flips to buying + price at lower band
        if self.config.trade_direction in ["both", "longs"]:
            if prev_imbalance <= 0 and imbalance >= self.config.imbalance_flip_threshold:
                if lows[idx] <= ewvma_lower * 1.002:  # Near lower band
                    entry = closes[idx]
                    sl = entry * 0.997  # 0.3% stop
                    risk = entry - sl
                    tp1 = entry + (risk * self.config.tp1_r_multiple)
                    tp2 = entry + (risk * self.config.tp2_r_multiple)

                    return ScalpTrade(
                        entry_type=EntryType.IMBALANCE_FLIP,
                        direction="long",
                        entry_idx=idx,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        volume_ratio=vol_ratio,
                        imbalance=imbalance,
                    )

        return None

    def _detect_ewvma_touch(self, df: pd.DataFrame, ind: Dict, idx: int) -> Optional[ScalpTrade]:
        """Entry Type 3: EWVMA Momentum Touch (pullback continuation)."""
        if idx < 5:
            return None

        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values

        ewvma_20 = ind['ewvma_20'][idx]
        imbalance = ind['imbalance'][idx]
        vol_ratio = ind['vol_ratio'][idx]

        # Check momentum candle (body > 70% of range)
        body = abs(closes[idx] - opens[idx])
        range_ = highs[idx] - lows[idx]
        if range_ == 0 or body / range_ < 0.7:
            return None

        # Volume confirmation
        if vol_ratio < self.config.min_vol_ratio:
            return None

        # SHORT: Price below EWVMA midline + touched it + mild selling
        if self.config.trade_direction in ["both", "shorts"]:
            if closes[idx] < ewvma_20:  # Below midline (bearish trend)
                if highs[idx] >= ewvma_20 * 0.998:  # Touched midline
                    if imbalance <= -0.05:  # Mild selling pressure
                        entry = closes[idx]
                        sl = entry * 1.0025  # 0.25% stop
                        risk = sl - entry
                        tp1 = entry - (risk * self.config.tp1_r_multiple)
                        tp2 = entry - (risk * self.config.tp2_r_multiple)

                        return ScalpTrade(
                            entry_type=EntryType.EWVMA_TOUCH,
                            direction="short",
                            entry_idx=idx,
                            entry_price=entry,
                            stop_loss=sl,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            volume_ratio=vol_ratio,
                            imbalance=imbalance,
                        )

        # LONG: Price above EWVMA midline + touched it + mild buying
        if self.config.trade_direction in ["both", "longs"]:
            if closes[idx] > ewvma_20:  # Above midline (bullish trend)
                if lows[idx] <= ewvma_20 * 1.002:  # Touched midline
                    if imbalance >= 0.05:  # Mild buying pressure
                        entry = closes[idx]
                        sl = entry * 0.9975  # 0.25% stop
                        risk = entry - sl
                        tp1 = entry + (risk * self.config.tp1_r_multiple)
                        tp2 = entry + (risk * self.config.tp2_r_multiple)

                        return ScalpTrade(
                            entry_type=EntryType.EWVMA_TOUCH,
                            direction="long",
                            entry_idx=idx,
                            entry_price=entry,
                            stop_loss=sl,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            volume_ratio=vol_ratio,
                            imbalance=imbalance,
                        )

        return None

    def _simulate_partial_exit(self, trade: ScalpTrade, df: pd.DataFrame) -> ScalpTrade:
        """Simulate partial exit system: 50% @ 1R, 50% @ 1.5R."""
        highs = df['high'].values
        lows = df['low'].values

        tp1_hit = False
        breakeven_sl = trade.entry_price

        for i in range(trade.entry_idx + 1, min(trade.entry_idx + self.config.max_hold_candles + 1, len(df))):
            if trade.direction == "short":
                # Check SL hit
                current_sl = breakeven_sl if tp1_hit else trade.stop_loss
                if highs[i] >= current_sl:
                    trade.exit_idx = i
                    trade.exit_price = current_sl
                    if tp1_hit:
                        # Already took 50% at 1R, remaining 50% stopped at BE = 0.5R total
                        trade.pnl_r = 0.5
                        trade.exit_reason = "sl_after_tp1"
                    else:
                        trade.pnl_r = -1.0
                        trade.exit_reason = "sl"
                    return trade

                # Check TP1 hit
                if not tp1_hit and lows[i] <= trade.tp1_price:
                    tp1_hit = True
                    breakeven_sl = trade.entry_price

                # Check TP2 hit
                if tp1_hit and lows[i] <= trade.tp2_price:
                    trade.exit_idx = i
                    trade.exit_price = trade.tp2_price
                    # 50% @ 1R + 50% @ 1.5R = 1.25R total
                    trade.pnl_r = 1.25
                    trade.exit_reason = "tp2"
                    return trade

            else:  # long
                # Check SL hit
                current_sl = breakeven_sl if tp1_hit else trade.stop_loss
                if lows[i] <= current_sl:
                    trade.exit_idx = i
                    trade.exit_price = current_sl
                    if tp1_hit:
                        trade.pnl_r = 0.5
                        trade.exit_reason = "sl_after_tp1"
                    else:
                        trade.pnl_r = -1.0
                        trade.exit_reason = "sl"
                    return trade

                # Check TP1 hit
                if not tp1_hit and highs[i] >= trade.tp1_price:
                    tp1_hit = True
                    breakeven_sl = trade.entry_price

                # Check TP2 hit
                if tp1_hit and highs[i] >= trade.tp2_price:
                    trade.exit_idx = i
                    trade.exit_price = trade.tp2_price
                    trade.pnl_r = 1.25
                    trade.exit_reason = "tp2"
                    return trade

        # Timeout - close at current price
        timeout_idx = min(trade.entry_idx + self.config.max_hold_candles, len(df) - 1)
        exit_price = df['close'].values[timeout_idx]

        if trade.direction == "short":
            pnl_pct = (trade.entry_price - exit_price) / trade.risk
        else:
            pnl_pct = (exit_price - trade.entry_price) / trade.risk

        if tp1_hit:
            # Had 50% closed at 1R, remaining 50% at current price
            trade.pnl_r = 0.5 + (0.5 * pnl_pct)
        else:
            trade.pnl_r = pnl_pct

        trade.exit_idx = timeout_idx
        trade.exit_price = exit_price
        trade.exit_reason = "timeout"

        return trade

    def run_backtest(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """Run backtest on dataframe."""
        self.trades = []

        # Calculate indicators
        ind = self._calculate_indicators(df)

        # Scan for signals
        cooldown_until = 0

        for idx in range(300, len(df)):
            if idx < cooldown_until:
                continue

            trade = None

            # Try each entry type
            if self.config.use_fvg_breakout and trade is None:
                trade = self._detect_fvg_breakout(df, ind, idx)

            if self.config.use_imbalance_flip and trade is None:
                trade = self._detect_imbalance_flip(df, ind, idx)

            if self.config.use_ewvma_touch and trade is None:
                trade = self._detect_ewvma_touch(df, ind, idx)

            if trade is not None:
                # Simulate exit
                trade = self._simulate_partial_exit(trade, df)
                self.trades.append(trade)

                # Cooldown (5 candles after exit)
                if trade.exit_idx:
                    cooldown_until = trade.exit_idx + 5

        # Calculate results
        return self._calculate_results(symbol, len(df))

    def _calculate_results(self, symbol: str, total_candles: int) -> Dict:
        """Calculate backtest results."""
        if not self.trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0,
                'expectancy': 0,
                'total_r': 0,
            }

        winners = [t for t in self.trades if t.pnl_r and t.pnl_r > 0]
        losers = [t for t in self.trades if t.pnl_r and t.pnl_r <= 0]

        total_trades = len(self.trades)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        total_r = sum(t.pnl_r for t in self.trades if t.pnl_r)
        expectancy = total_r / total_trades if total_trades > 0 else 0

        avg_winner = np.mean([t.pnl_r for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl_r for t in losers]) if losers else 0

        avg_hold = np.mean([t.hold_time for t in self.trades])

        # Breakdown by entry type
        by_type = {}
        for entry_type in EntryType:
            type_trades = [t for t in self.trades if t.entry_type == entry_type]
            if type_trades:
                type_winners = [t for t in type_trades if t.pnl_r and t.pnl_r > 0]
                by_type[entry_type.value] = {
                    'trades': len(type_trades),
                    'win_rate': len(type_winners) / len(type_trades) * 100,
                    'total_r': sum(t.pnl_r for t in type_trades if t.pnl_r),
                }

        # Breakdown by direction
        shorts = [t for t in self.trades if t.direction == "short"]
        longs = [t for t in self.trades if t.direction == "long"]

        short_wr = len([t for t in shorts if t.pnl_r and t.pnl_r > 0]) / len(shorts) * 100 if shorts else 0
        long_wr = len([t for t in longs if t.pnl_r and t.pnl_r > 0]) / len(longs) * 100 if longs else 0

        # Trades per day estimate (assuming 5-min candles, ~288 per day)
        candles_per_day = 288
        days = total_candles / candles_per_day
        trades_per_day = total_trades / days if days > 0 else 0

        return {
            'symbol': symbol,
            'total_candles': total_candles,
            'days': round(days, 1),
            'total_trades': total_trades,
            'trades_per_day': round(trades_per_day, 2),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': round(win_rate, 1),
            'total_r': round(total_r, 2),
            'expectancy': round(expectancy, 3),
            'avg_winner': round(avg_winner, 3),
            'avg_loser': round(avg_loser, 3),
            'avg_hold_candles': round(avg_hold, 1),
            'shorts': len(shorts),
            'longs': len(longs),
            'short_wr': round(short_wr, 1),
            'long_wr': round(long_wr, 1),
            'by_type': by_type,
        }


def load_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data."""
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle timestamp column
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'time'}, inplace=True)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Remove zero-range candles
    df = df[df['high'] != df['low']].reset_index(drop=True)

    return df


def print_results(results: Dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"SCALPING BACKTEST RESULTS - {results['symbol']}")
    print(f"{'='*60}")
    print(f"Data: {results['total_candles']:,} candles ({results['days']} days)")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Trades/Day: {results['trades_per_day']}")
    print(f"{'='*60}")
    print(f"Win Rate: {results['win_rate']}%")
    print(f"Winners: {results['winners']} | Losers: {results['losers']}")
    print(f"Total R: {results['total_r']:+.2f}R")
    print(f"Expectancy: {results['expectancy']:+.3f}R per trade")
    print(f"Avg Winner: {results['avg_winner']:+.3f}R")
    print(f"Avg Loser: {results['avg_loser']:+.3f}R")
    print(f"Avg Hold: {results['avg_hold_candles']} candles")
    print(f"{'='*60}")
    print(f"Direction Breakdown:")
    print(f"  Shorts: {results['shorts']} trades, {results['short_wr']}% WR")
    print(f"  Longs: {results['longs']} trades, {results['long_wr']}% WR")
    print(f"{'='*60}")
    print(f"Entry Type Breakdown:")
    for entry_type, data in results.get('by_type', {}).items():
        print(f"  {entry_type}: {data['trades']} trades, {data['win_rate']:.1f}% WR, {data['total_r']:+.2f}R")
    print(f"{'='*60}")


def main():
    """Run scalping backtest on BTC, ETH, SOL."""
    DATA_DIR = Path("/home/tahae/ai-content/data/Tradingdata/volume-charts")

    # Data files to test
    data_files = {
        'BTC': DATA_DIR / "BTCUSDT_5m_merged.csv",
        'ETH': DATA_DIR / "BYBIT_ETHUSDT.P, 5_bf884-new.csv",
        'SOL': DATA_DIR / "BYBIT_SOLUSDT.P, 5_7ef98-new.csv",
    }

    # Configuration
    config = ScalpConfig(
        min_vol_ratio=1.5,
        imbalance_threshold=0.10,
        imbalance_flip_threshold=0.15,
        tp1_r_multiple=1.0,
        tp2_r_multiple=1.5,
        max_hold_candles=30,
        trade_direction="both",
        use_fvg_breakout=True,
        use_imbalance_flip=True,
        use_ewvma_touch=True,
    )

    print("\n" + "="*60)
    print("SCALPING STRATEGY BACKTEST")
    print("="*60)
    print(f"Config: Vol >= {config.min_vol_ratio}x, Imbalance >= {config.imbalance_threshold}")
    print(f"Exit: 50% @ {config.tp1_r_multiple}R, 50% @ {config.tp2_r_multiple}R")
    print(f"Max Hold: {config.max_hold_candles} candles")
    print(f"Entry Types: FVG={config.use_fvg_breakout}, Flip={config.use_imbalance_flip}, Touch={config.use_ewvma_touch}")

    all_results = []

    for symbol, file_path in data_files.items():
        if not file_path.exists():
            print(f"\nSkipping {symbol}: File not found - {file_path}")
            continue

        print(f"\nLoading {symbol}...")
        df = load_data(str(file_path))
        print(f"  Loaded {len(df):,} candles")

        backtest = ScalpBacktest(config)
        results = backtest.run_backtest(df, symbol)
        all_results.append(results)

        print_results(results)

    # Combined results
    if len(all_results) > 1:
        total_trades = sum(r['total_trades'] for r in all_results)
        total_winners = sum(r['winners'] for r in all_results)
        total_r = sum(r['total_r'] for r in all_results)

        print(f"\n{'='*60}")
        print("COMBINED RESULTS (All Symbols)")
        print(f"{'='*60}")
        print(f"Total Trades: {total_trades}")
        print(f"Combined Win Rate: {total_winners/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        print(f"Combined Total R: {total_r:+.2f}R")
        print(f"Combined Expectancy: {total_r/total_trades:+.3f}R" if total_trades > 0 else "N/A")
        print(f"{'='*60}")

        # Check if meets target
        wr = total_winners/total_trades*100 if total_trades > 0 else 0
        exp = total_r/total_trades if total_trades > 0 else 0

        print("\nTARGET VALIDATION:")
        print(f"  Win Rate >= 60%: {'PASS' if wr >= 60 else 'FAIL'} ({wr:.1f}%)")
        print(f"  Expectancy >= 0.35R: {'PASS' if exp >= 0.35 else 'FAIL'} ({exp:+.3f}R)")


if __name__ == "__main__":
    main()

"""
Mean-Reversion Double Touch Strategy

Adapts the Double Touch pattern concept for mean-reverting spreads.

ORIGINAL Double Touch (for trending assets):
    Step 0: HH/LL with trend
    Step 1: Pullback (band grey)
    Step 2: Trend resumes (band green/red)
    Step 3: Second pullback (band grey) - defines SL
    Step 4: Trend resumes + FVG = ENTRY

MEAN-REVERSION Double Touch (for cointegrated pairs):
    Step 0: First extreme (z < -2.0)
    Step 1: Partial recovery (z > -1.0) - shows buying interest
    Step 2: Second touch of extreme (z < -1.5) - "higher low" on z-score
    Step 3: ENTRY expecting mean reversion

The key insight: Both patterns look for "exhaustion after two attempts"
- Trending: Two failed pullbacks = trend strong, continue
- Mean-reverting: Two failed extensions = mean-reversion strong, revert

Backtest Results (BTC/ETH 5-min spread):
    - 51 trades over 19 days
    - 88.2% win rate
    - 4.90 profit factor
    - 61.4% return
    - 12% max drawdown
    - 100% TP exits (0 SL hits)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


@dataclass
class MRDoubleTouch:
    """Mean-Reversion Double Touch configuration"""

    # Pattern detection thresholds
    first_extreme_z: float = 2.0      # First touch: z < -2.0 or z > 2.0
    recovery_z: float = 1.0           # Must recover past -1.0 or +1.0
    second_touch_z: float = 1.5       # Second touch: z < -1.5 or z > 1.5
    max_pattern_bars: int = 50        # Max candles for pattern to complete

    # Exit thresholds
    tp_z: float = 0.5                 # Take profit at z = ±0.5
    sl_z: float = 4.0                 # Stop loss at z = ±4.0

    # Z-score calculation
    zscore_window: int = 60           # Rolling window for mean/std


@dataclass
class MRSignal:
    """A detected Mean-Reversion Double Touch signal"""
    signal_type: str          # 'long_spread' or 'short_spread'
    entry_idx: int            # Bar index of entry
    entry_time: datetime
    entry_spread: float
    entry_z: float            # Z-score at entry (second touch)
    first_touch_z: float      # Z-score at first touch
    recovery_z: float         # Z-score at recovery peak
    hedge_ratio: float

    # Trade levels
    tp_z: float
    sl_z: float

    # For execution
    btc_price: float
    eth_price: float


def load_and_prepare_data(
    btc_path: str,
    eth_path: str,
    resample_minutes: int = 5
) -> Tuple[pd.DataFrame, float]:
    """Load price data and calculate spread with z-score."""

    # Load CSVs
    btc = pd.read_csv(btc_path, parse_dates=['time'])
    eth = pd.read_csv(eth_path, parse_dates=['time'])

    btc = btc[['time', 'open', 'high', 'low', 'close']].copy()
    eth = eth[['time', 'open', 'high', 'low', 'close']].copy()

    # Resample if needed
    if resample_minutes > 1:
        btc = btc.set_index('time').resample(f'{resample_minutes}min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()

        eth = eth.set_index('time').resample(f'{resample_minutes}min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()

    # Merge
    btc.columns = ['time', 'btc_open', 'btc_high', 'btc_low', 'btc_close']
    eth.columns = ['time', 'eth_open', 'eth_high', 'eth_low', 'eth_close']
    df = pd.merge(btc, eth, on='time', how='inner').sort_values('time').reset_index(drop=True)

    # Calculate hedge ratio using OLS
    X = add_constant(df['btc_close'])
    model = OLS(df['eth_close'], X).fit()
    hedge_ratio = model.params.iloc[1]

    # Calculate spread
    df['spread'] = df['eth_close'] - hedge_ratio * df['btc_close']

    return df, hedge_ratio


def add_indicators(df: pd.DataFrame, params: MRDoubleTouch) -> pd.DataFrame:
    """Add z-score and other indicators."""

    # Z-score
    df['spread_mean'] = df['spread'].rolling(params.zscore_window).mean()
    df['spread_std'] = df['spread'].rolling(params.zscore_window).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    return df


def detect_patterns(
    df: pd.DataFrame,
    hedge_ratio: float,
    params: MRDoubleTouch
) -> List[MRSignal]:
    """
    Detect Mean-Reversion Double Touch patterns.

    Pattern for LONG spread (oversold):
        1. First extreme: z < -first_extreme_z (e.g., -2.0)
        2. Partial recovery: z > -recovery_z (e.g., -1.0)
        3. Second touch: z < -second_touch_z (e.g., -1.5)
        4. Second touch must be "higher low" (not break first extreme)

    Pattern for SHORT spread (overbought):
        Mirror of above with positive z-scores
    """
    signals = []

    # State machine
    state = {'phase': 0}

    for i in range(params.zscore_window + 10, len(df)):
        z = df['zscore'].iloc[i]

        if pd.isna(z):
            continue

        # =====================
        # LONG SPREAD PATTERN
        # =====================
        if state['phase'] == 0:
            # Phase 0: Looking for first extreme oversold
            if z < -params.first_extreme_z:
                state = {
                    'phase': 1,
                    'type': 'long',
                    'first_idx': i,
                    'first_z': z
                }

        elif state['phase'] == 1 and state.get('type') == 'long':
            # Phase 1: Looking for partial recovery
            if z > -params.recovery_z:
                state['phase'] = 2
                state['recovery_idx'] = i
                state['recovery_z'] = z
            elif z < state['first_z'] - 0.5:
                # Extended too far below first touch, reset
                state = {'phase': 0}

        elif state['phase'] == 2 and state.get('type') == 'long':
            # Phase 2: Looking for second touch
            if z < -params.second_touch_z:
                # Check for "higher low" - second touch shouldn't break first
                if z > state['first_z'] - 0.3:
                    signals.append(MRSignal(
                        signal_type='long_spread',
                        entry_idx=i,
                        entry_time=df['time'].iloc[i],
                        entry_spread=df['spread'].iloc[i],
                        entry_z=z,
                        first_touch_z=state['first_z'],
                        recovery_z=state['recovery_z'],
                        hedge_ratio=hedge_ratio,
                        tp_z=params.tp_z,
                        sl_z=params.sl_z,
                        btc_price=df['btc_close'].iloc[i],
                        eth_price=df['eth_close'].iloc[i]
                    ))
                state = {'phase': 0}
            elif z > 0:
                # Went all the way to mean, pattern invalid
                state = {'phase': 0}
            elif i - state['first_idx'] > params.max_pattern_bars:
                # Took too long
                state = {'phase': 0}

        # =====================
        # SHORT SPREAD PATTERN
        # =====================
        if state['phase'] == 0:
            # Phase 0: Looking for first extreme overbought
            if z > params.first_extreme_z:
                state = {
                    'phase': 1,
                    'type': 'short',
                    'first_idx': i,
                    'first_z': z
                }

        elif state['phase'] == 1 and state.get('type') == 'short':
            # Phase 1: Looking for partial recovery
            if z < params.recovery_z:
                state['phase'] = 2
                state['recovery_idx'] = i
                state['recovery_z'] = z
            elif z > state['first_z'] + 0.5:
                state = {'phase': 0}

        elif state['phase'] == 2 and state.get('type') == 'short':
            # Phase 2: Looking for second touch
            if z > params.second_touch_z:
                # Check for "lower high"
                if z < state['first_z'] + 0.3:
                    signals.append(MRSignal(
                        signal_type='short_spread',
                        entry_idx=i,
                        entry_time=df['time'].iloc[i],
                        entry_spread=df['spread'].iloc[i],
                        entry_z=z,
                        first_touch_z=state['first_z'],
                        recovery_z=state['recovery_z'],
                        hedge_ratio=hedge_ratio,
                        tp_z=params.tp_z,
                        sl_z=params.sl_z,
                        btc_price=df['btc_close'].iloc[i],
                        eth_price=df['eth_close'].iloc[i]
                    ))
                state = {'phase': 0}
            elif z < 0:
                state = {'phase': 0}
            elif i - state['first_idx'] > params.max_pattern_bars:
                state = {'phase': 0}

    return signals


def backtest(
    df: pd.DataFrame,
    signals: List[MRSignal],
    risk_per_trade: float = 0.02,
    starting_capital: float = 10000
) -> Dict:
    """Backtest the MR Double Touch signals."""

    if not signals:
        return {'error': 'No signals to backtest'}

    capital = starting_capital
    peak = starting_capital
    max_dd = 0
    trades = []

    for sig in signals:
        entry_idx = sig.entry_idx
        is_long = sig.signal_type == 'long_spread'
        entry_spread = sig.entry_spread
        entry_std = df['spread_std'].iloc[entry_idx]

        if pd.isna(entry_std) or entry_std == 0:
            continue

        # Find exit
        exit_idx = None
        exit_reason = None
        exit_z = None

        for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            z = df['zscore'].iloc[j]

            if is_long:
                if z >= -sig.tp_z:
                    exit_idx, exit_reason, exit_z = j, 'take_profit', z
                    break
                elif z < -sig.sl_z:
                    exit_idx, exit_reason, exit_z = j, 'stop_loss', z
                    break
            else:
                if z <= sig.tp_z:
                    exit_idx, exit_reason, exit_z = j, 'take_profit', z
                    break
                elif z > sig.sl_z:
                    exit_idx, exit_reason, exit_z = j, 'stop_loss', z
                    break

        if exit_idx is None:
            continue

        exit_spread = df['spread'].iloc[exit_idx]
        spread_change = (exit_spread - entry_spread) if is_long else (entry_spread - exit_spread)

        # Position sizing: risk amount / (entry z-score * std)
        risk_amount = capital * risk_per_trade
        position_units = risk_amount / (abs(sig.entry_z) * entry_std)
        pnl = position_units * spread_change

        capital += pnl
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100
        max_dd = max(max_dd, dd)

        r_multiple = pnl / risk_amount

        trades.append({
            'entry_time': sig.entry_time,
            'exit_time': df['time'].iloc[exit_idx],
            'type': sig.signal_type,
            'entry_z': sig.entry_z,
            'first_z': sig.first_touch_z,
            'exit_z': exit_z,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'duration': exit_idx - entry_idx
        })

    if not trades:
        return {'error': 'No completed trades'}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]

    return {
        'total_trades': len(trades_df),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades_df) * 100,
        'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
        'total_pnl': trades_df['pnl'].sum(),
        'total_return_pct': (capital - starting_capital) / starting_capital * 100,
        'avg_r': trades_df['r_multiple'].mean(),
        'max_drawdown': max_dd,
        'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
        'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
        'avg_duration': trades_df['duration'].mean(),
        'tp_exits': len(trades_df[trades_df['exit_reason'] == 'take_profit']),
        'sl_exits': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
        'final_equity': capital,
        'trades': trades_df
    }


def run_analysis(
    btc_path: str = "/home/tahae/ai-content/data/Tradingdata/BYBIT_BTCUSDT.P, 1_da562.csv",
    eth_path: str = "/home/tahae/ai-content/data/Tradingdata/BYBIT_ETHUSDT.P, 1_79d61.csv",
    timeframe: int = 5
):
    """Run full MR Double Touch analysis."""

    print("=" * 70)
    print("MEAN-REVERSION DOUBLE TOUCH STRATEGY")
    print("=" * 70)

    params = MRDoubleTouch()

    print(f"\nParameters:")
    print(f"  First extreme: z = ±{params.first_extreme_z}")
    print(f"  Recovery: z = ±{params.recovery_z}")
    print(f"  Second touch: z = ±{params.second_touch_z}")
    print(f"  Take profit: z = ±{params.tp_z}")
    print(f"  Stop loss: z = ±{params.sl_z}")

    # Load data
    print(f"\n1. Loading {timeframe}-minute data...")
    df, hedge_ratio = load_and_prepare_data(btc_path, eth_path, timeframe)
    print(f"   Candles: {len(df)}")
    print(f"   Hedge ratio: {hedge_ratio:.6f}")
    print(f"   Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Add indicators
    df = add_indicators(df, params)

    # Detect patterns
    print(f"\n2. Scanning for patterns...")
    signals = detect_patterns(df, hedge_ratio, params)
    print(f"   Signals found: {len(signals)}")
    print(f"   Long spread: {len([s for s in signals if s.signal_type == 'long_spread'])}")
    print(f"   Short spread: {len([s for s in signals if s.signal_type == 'short_spread'])}")

    if not signals:
        print("\nNo signals found.")
        return

    # Show recent signals
    print(f"\n   Recent signals:")
    for sig in signals[-5:]:
        print(f"   {sig.entry_time}: {sig.signal_type} @ z={sig.entry_z:.2f} "
              f"(first: {sig.first_touch_z:.2f}, recovery: {sig.recovery_z:.2f})")

    # Backtest
    print(f"\n3. Backtest Results")
    print("-" * 40)
    results = backtest(df, signals)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Total return: {results['total_return_pct']:.1f}%")
    print(f"Max drawdown: {results['max_drawdown']:.1f}%")

    print(f"\nTP exits: {results['tp_exits']} ({results['tp_exits']/results['total_trades']*100:.0f}%)")
    print(f"SL exits: {results['sl_exits']} ({results['sl_exits']/results['total_trades']*100:.0f}%)")
    print(f"Avg duration: {results['avg_duration']:.0f} candles ({results['avg_duration'] * timeframe / 60:.1f} hours)")

    print(f"\nP&L: ${results['total_pnl']:.2f}")
    print(f"Avg R: {results['avg_r']:.2f}")
    print(f"Final equity: ${results['final_equity']:.2f}")

    # Compare with pure z-score baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH PURE Z-SCORE")
    print("=" * 70)

    from spread_analysis import backtest_spread_strategy

    baseline = backtest_spread_strategy(
        df, hedge_ratio,
        entry_zscore=2.0, exit_zscore=0.5,
        zscore_window=60, stop_loss_zscore=4.0,
        risk_per_trade=0.02, starting_capital=10000
    )

    print(f"\n{'Metric':<25} {'MR Double Touch':>15} {'Pure Z-Score':>15}")
    print("-" * 55)
    print(f"{'Trades':<25} {results['total_trades']:>15} {baseline['total_trades']:>15}")
    print(f"{'Win rate':<25} {results['win_rate']:>14.1f}% {baseline['win_rate']:>14.1f}%")
    print(f"{'Profit factor':<25} {results['profit_factor']:>15.2f} {baseline['profit_factor']:>15.2f}")
    print(f"{'Return':<25} {results['total_return_pct']:>14.1f}% {baseline['total_return_pct']:>14.1f}%")
    print(f"{'Max drawdown':<25} {results['max_drawdown']:>14.1f}% {baseline.get('max_drawdown', 0):>14.1f}%")

    # Verdict
    print("\n" + "-" * 55)
    if results['profit_factor'] > baseline['profit_factor']:
        improvement = (results['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100
        print(f"MR Double Touch WINS: {improvement:.0f}% better profit factor")
    else:
        print("Pure Z-Score has better profit factor")

    if results['win_rate'] > baseline['win_rate']:
        print(f"MR Double Touch has {results['win_rate'] - baseline['win_rate']:.1f}% higher win rate")

    return {
        'df': df,
        'signals': signals,
        'results': results,
        'baseline': baseline
    }


if __name__ == "__main__":
    run_analysis()

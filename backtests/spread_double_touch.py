"""
Spread Double Touch Strategy - Apply Double Touch pattern detection to BTC/ETH spread

Instead of using simple z-score thresholds, we apply the Double Touch pattern
detection to the spread itself for better entry timing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class BandColor(Enum):
    GREEN = "green"  # Bullish
    RED = "red"      # Bearish
    GREY = "grey"    # Neutral


@dataclass
class SpreadCandle:
    """Represents a candle of the spread."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    # Original prices for execution
    btc_close: float
    eth_close: float


@dataclass
class SpreadSignal:
    """A Double Touch signal on the spread."""
    signal_type: str  # 'long_spread' or 'short_spread'
    entry_spread: float
    stop_loss_spread: float
    target_spread: float
    entry_idx: int
    step_0_idx: int
    step_3_idx: int
    created_at: datetime
    # For execution
    btc_price: float
    eth_price: float
    hedge_ratio: float


def load_and_prepare_data(
    btc_path: str,
    eth_path: str,
    resample_minutes: int = 5
) -> Tuple[pd.DataFrame, float]:
    """Load data and calculate spread."""
    # Load CSVs
    btc = pd.read_csv(btc_path, parse_dates=['time'])
    eth = pd.read_csv(eth_path, parse_dates=['time'])

    # Keep only relevant columns
    btc = btc[['time', 'open', 'high', 'low', 'close']].copy()
    eth = eth[['time', 'open', 'high', 'low', 'close']].copy()

    # Resample
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

    # Calculate hedge ratio
    X = add_constant(df['btc_close'])
    model = OLS(df['eth_close'], X).fit()
    hedge_ratio = model.params.iloc[1]

    # Calculate spread OHLC
    # Spread = ETH - hedge_ratio * BTC
    df['spread_open'] = df['eth_open'] - hedge_ratio * df['btc_open']
    df['spread_high'] = df['eth_high'] - hedge_ratio * df['btc_low']   # Max spread
    df['spread_low'] = df['eth_low'] - hedge_ratio * df['btc_high']    # Min spread
    df['spread_close'] = df['eth_close'] - hedge_ratio * df['btc_close']

    print(f"Loaded {len(df)} candles, hedge ratio: {hedge_ratio:.6f}")

    return df, hedge_ratio


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_ewvma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Calculate EWVMA (Exponentially Weighted Volume Moving Average)."""
    # For spread, we don't have volume, so use SMA of absolute changes as proxy
    vol_proxy = close.diff().abs().rolling(period).mean()
    vol_proxy = vol_proxy.fillna(vol_proxy.mean())

    weights = vol_proxy.ewm(span=period, adjust=False).mean()
    weighted_price = (close * vol_proxy).ewm(span=period, adjust=False).mean()

    return weighted_price / weights


def get_band_color(ema9: float, ema21: float, ema50: float) -> BandColor:
    """Determine EMA ribbon color."""
    if ema9 > ema21 > ema50:
        return BandColor.GREEN
    elif ema9 < ema21 < ema50:
        return BandColor.RED
    else:
        return BandColor.GREY


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA ribbon and other indicators to spread data."""
    close = df['spread_close']

    # EMA Ribbon
    df['ema9'] = calculate_ema(close, 9)
    df['ema21'] = calculate_ema(close, 21)
    df['ema50'] = calculate_ema(close, 50)

    # Band color
    df['band_color'] = [
        get_band_color(e9, e21, e50).value
        for e9, e21, e50 in zip(df['ema9'], df['ema21'], df['ema50'])
    ]

    # EWVMA for counter-trend filter (use close diff as volume proxy)
    vol_proxy = close.diff().abs().fillna(0) + 0.001
    df['ewvma200'] = calculate_ewvma(close, vol_proxy, 200)

    # Z-score for reference
    df['spread_mean'] = close.rolling(60).mean()
    df['spread_std'] = close.rolling(60).std()
    df['zscore'] = (close - df['spread_mean']) / df['spread_std']

    return df


def detect_higher_high(df: pd.DataFrame, idx: int, lookback: int = 20) -> bool:
    """Detect if current candle made a higher high."""
    if idx < lookback:
        return False

    current_high = df['spread_high'].iloc[idx]
    prev_highs = df['spread_high'].iloc[idx-lookback:idx]

    return current_high > prev_highs.max()


def detect_lower_low(df: pd.DataFrame, idx: int, lookback: int = 20) -> bool:
    """Detect if current candle made a lower low."""
    if idx < lookback:
        return False

    current_low = df['spread_low'].iloc[idx]
    prev_lows = df['spread_low'].iloc[idx-lookback:idx]

    return current_low < prev_lows.min()


def detect_fvg(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Detect Fair Value Gap at index."""
    if idx < 2:
        return None

    candle_0 = df.iloc[idx - 2]  # Two candles ago
    candle_2 = df.iloc[idx]      # Current candle

    # Bullish FVG: gap between candle_0 high and candle_2 low
    if candle_2['spread_low'] > candle_0['spread_high']:
        return {
            'type': 'bullish',
            'top': candle_2['spread_low'],
            'bottom': candle_0['spread_high'],
            'idx': idx
        }

    # Bearish FVG: gap between candle_0 low and candle_2 high
    if candle_2['spread_high'] < candle_0['spread_low']:
        return {
            'type': 'bearish',
            'top': candle_0['spread_low'],
            'bottom': candle_2['spread_high'],
            'idx': idx
        }

    return None


def scan_double_touch_patterns(
    df: pd.DataFrame,
    hedge_ratio: float,
    lookback: int = 20,
    use_counter_trend: bool = True
) -> List[SpreadSignal]:
    """
    Scan for Double Touch patterns on the spread.

    Original pattern (rarely triggers on mean-reverting spread).
    """
    signals = []

    # Track pattern state
    long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None,
                  'step_3_idx': None, 'step_3_price': None}
    short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None,
                   'step_3_idx': None, 'step_3_price': None}

    for i in range(lookback + 50, len(df)):
        row = df.iloc[i]
        band = row['band_color']
        spread_close = row['spread_close']
        ewvma = row['ewvma200']

        # ============ LONG SPREAD PATTERN ============
        if long_state['step'] == -1:
            if band == 'green' and detect_higher_high(df, i, lookback):
                if not use_counter_trend or spread_close < ewvma:
                    long_state = {'step': 0, 'step_0_idx': i, 'step_0_price': row['spread_high'],
                                  'step_3_idx': None, 'step_3_price': None}
        elif long_state['step'] == 0:
            if band == 'grey':
                long_state['step'] = 1
            elif band == 'red':
                long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif long_state['step'] == 1:
            if band == 'green':
                long_state['step'] = 2
            elif band == 'red':
                long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif long_state['step'] == 2:
            if band == 'grey':
                long_state['step'] = 3
                long_state['step_3_idx'] = i
                long_state['step_3_price'] = row['spread_low']
            elif band == 'red':
                long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif long_state['step'] == 3:
            if band == 'green':
                fvg = detect_fvg(df, i)
                if fvg and fvg['type'] == 'bullish':
                    entry = fvg['top']
                    sl = long_state['step_3_price'] * 0.999
                    risk = entry - sl
                    tp = entry + (risk * 3.0)
                    signals.append(SpreadSignal(
                        signal_type='long_spread', entry_spread=entry, stop_loss_spread=sl,
                        target_spread=tp, entry_idx=i, step_0_idx=long_state['step_0_idx'],
                        step_3_idx=long_state['step_3_idx'], created_at=row['time'],
                        btc_price=row['btc_close'], eth_price=row['eth_close'], hedge_ratio=hedge_ratio
                    ))
                    long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
            elif band == 'red':
                long_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}

        # ============ SHORT SPREAD PATTERN ============
        if short_state['step'] == -1:
            if band == 'red' and detect_lower_low(df, i, lookback):
                if not use_counter_trend or spread_close > ewvma:
                    short_state = {'step': 0, 'step_0_idx': i, 'step_0_price': row['spread_low'],
                                   'step_3_idx': None, 'step_3_price': None}
        elif short_state['step'] == 0:
            if band == 'grey':
                short_state['step'] = 1
            elif band == 'green':
                short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif short_state['step'] == 1:
            if band == 'red':
                short_state['step'] = 2
            elif band == 'green':
                short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif short_state['step'] == 2:
            if band == 'grey':
                short_state['step'] = 3
                short_state['step_3_idx'] = i
                short_state['step_3_price'] = row['spread_high']
            elif band == 'green':
                short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
        elif short_state['step'] == 3:
            if band == 'red':
                fvg = detect_fvg(df, i)
                if fvg and fvg['type'] == 'bearish':
                    entry = fvg['bottom']
                    sl = short_state['step_3_price'] * 1.001
                    risk = sl - entry
                    tp = entry - (risk * 3.0)
                    signals.append(SpreadSignal(
                        signal_type='short_spread', entry_spread=entry, stop_loss_spread=sl,
                        target_spread=tp, entry_idx=i, step_0_idx=short_state['step_0_idx'],
                        step_3_idx=short_state['step_3_idx'], created_at=row['time'],
                        btc_price=row['btc_close'], eth_price=row['eth_close'], hedge_ratio=hedge_ratio
                    ))
                    short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}
            elif band == 'green':
                short_state = {'step': -1, 'step_0_idx': None, 'step_0_price': None, 'step_3_idx': None, 'step_3_price': None}

    return signals


def scan_ema_ribbon_zscore_signals(
    df: pd.DataFrame,
    hedge_ratio: float,
    zscore_entry: float = 2.0,
    recent_window: int = 10
) -> List[SpreadSignal]:
    """
    Hybrid strategy: Recent extreme z-score + EMA ribbon exhaustion.

    Key insight: Z-score recovers faster than EMA ribbon changes.
    So we check if z-score WAS extreme recently (within N candles),
    and enter when band shows exhaustion (turns GREY).

    Long Spread (buy ETH, sell BTC):
    - Z-score was < -zscore_entry within last N candles
    - Band just turned GREY (exhaustion signal)
    - Enter now, SL at recent low, TP at mean

    Short Spread (sell ETH, buy BTC):
    - Z-score was > +zscore_entry within last N candles
    - Band just turned GREY (exhaustion signal)
    - Enter now, SL at recent high, TP at mean
    """
    signals = []

    lookback = 20
    prev_band = None

    for i in range(100, len(df)):
        row = df.iloc[i]
        zscore = row['zscore']
        band = row['band_color']
        spread_close = row['spread_close']

        if pd.isna(zscore):
            prev_band = band
            continue

        # Check recent z-scores
        recent_zscores = df['zscore'].iloc[max(0, i-recent_window):i+1]
        min_recent_z = recent_zscores.min()
        max_recent_z = recent_zscores.max()

        # ============ LONG SPREAD ============
        # Was recently oversold AND band just turned grey (exhaustion)
        if min_recent_z < -zscore_entry and prev_band == 'red' and band == 'grey':
            sl = df['spread_low'].iloc[i-lookback:i+1].min() * 0.999
            entry = spread_close
            spread_mean = row['spread_mean']
            tp = spread_mean

            risk = entry - sl
            reward = tp - entry
            rr = reward / risk if risk > 0 else 0

            if rr >= 1.0 and risk > 0:
                signals.append(SpreadSignal(
                    signal_type='long_spread',
                    entry_spread=entry,
                    stop_loss_spread=sl,
                    target_spread=tp,
                    entry_idx=i,
                    step_0_idx=i,
                    step_3_idx=i,
                    created_at=row['time'],
                    btc_price=row['btc_close'],
                    eth_price=row['eth_close'],
                    hedge_ratio=hedge_ratio
                ))

        # ============ SHORT SPREAD ============
        # Was recently overbought AND band just turned grey (exhaustion)
        if max_recent_z > zscore_entry and prev_band == 'green' and band == 'grey':
            sl = df['spread_high'].iloc[i-lookback:i+1].max() * 1.001
            entry = spread_close
            spread_mean = row['spread_mean']
            tp = spread_mean

            risk = sl - entry
            reward = entry - tp
            rr = reward / risk if risk > 0 else 0

            if rr >= 1.0 and risk > 0:
                signals.append(SpreadSignal(
                    signal_type='short_spread',
                    entry_spread=entry,
                    stop_loss_spread=sl,
                    target_spread=tp,
                    entry_idx=i,
                    step_0_idx=i,
                    step_3_idx=i,
                    created_at=row['time'],
                    btc_price=row['btc_close'],
                    eth_price=row['eth_close'],
                    hedge_ratio=hedge_ratio
                ))

        prev_band = band

    return signals


def backtest_double_touch_spread(
    df: pd.DataFrame,
    signals: List[SpreadSignal],
    risk_per_trade: float = 0.02,
    starting_capital: float = 10000
) -> Dict:
    """Backtest Double Touch signals on the spread."""
    if not signals:
        return {'error': 'No signals to backtest'}

    trades = []
    capital = starting_capital

    for signal in signals:
        entry_idx = signal.entry_idx

        # Simulate trade from entry
        entry_spread = signal.entry_spread
        sl = signal.stop_loss_spread
        tp = signal.target_spread

        is_long = signal.signal_type == 'long_spread'

        # Risk amount
        spread_risk = abs(entry_spread - sl)
        position_value = capital * risk_per_trade

        # Walk forward to find exit
        exit_idx = None
        exit_spread = None
        exit_reason = None

        for i in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            current_spread = df['spread_close'].iloc[i]
            current_high = df['spread_high'].iloc[i]
            current_low = df['spread_low'].iloc[i]

            if is_long:
                # Check SL
                if current_low <= sl:
                    exit_idx = i
                    exit_spread = sl
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                if current_high >= tp:
                    exit_idx = i
                    exit_spread = tp
                    exit_reason = 'take_profit'
                    break
            else:
                # Check SL
                if current_high >= sl:
                    exit_idx = i
                    exit_spread = sl
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                if current_low <= tp:
                    exit_idx = i
                    exit_spread = tp
                    exit_reason = 'take_profit'
                    break

        if exit_idx is None:
            # Trade still open, use last price
            exit_idx = len(df) - 1
            exit_spread = df['spread_close'].iloc[-1]
            exit_reason = 'open'

        # Calculate P&L
        if is_long:
            spread_pnl = exit_spread - entry_spread
        else:
            spread_pnl = entry_spread - exit_spread

        # Normalize P&L
        r_multiple = spread_pnl / spread_risk if spread_risk > 0 else 0
        pnl = position_value * r_multiple
        capital += pnl

        trades.append({
            'entry_time': signal.created_at,
            'exit_time': df['time'].iloc[exit_idx],
            'direction': signal.signal_type,
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'sl': sl,
            'tp': tp,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'exit_reason': exit_reason,
            'duration_candles': exit_idx - entry_idx
        })

    # Calculate statistics
    trades_df = pd.DataFrame(trades)
    completed = trades_df[trades_df['exit_reason'] != 'open']

    if len(completed) == 0:
        return {'error': 'No completed trades'}

    winners = completed[completed['pnl'] > 0]
    losers = completed[completed['pnl'] <= 0]

    return {
        'total_signals': len(signals),
        'completed_trades': len(completed),
        'open_trades': len(trades_df[trades_df['exit_reason'] == 'open']),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(completed) * 100 if len(completed) > 0 else 0,
        'total_pnl': completed['pnl'].sum(),
        'total_return_pct': (capital - starting_capital) / starting_capital * 100,
        'avg_r': completed['r_multiple'].mean(),
        'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
        'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
        'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
        'largest_win': completed['pnl'].max(),
        'largest_loss': completed['pnl'].min(),
        'avg_duration': completed['duration_candles'].mean(),
        'tp_exits': len(completed[completed['exit_reason'] == 'take_profit']),
        'sl_exits': len(completed[completed['exit_reason'] == 'stop_loss']),
        'final_equity': capital,
        'trades': trades_df
    }


def run_analysis(
    btc_path: str,
    eth_path: str,
    timeframe: int = 5,
    use_counter_trend: bool = True
):
    """Run Double Touch spread analysis."""
    print("=" * 70)
    print("SPREAD DOUBLE TOUCH ANALYSIS")
    print(f"Timeframe: {timeframe}-minute candles")
    print(f"Counter-trend filter: {'ON' if use_counter_trend else 'OFF'}")
    print("=" * 70)

    # Load data
    print("\n1. LOADING DATA & CALCULATING SPREAD")
    print("-" * 40)
    df, hedge_ratio = load_and_prepare_data(btc_path, eth_path, timeframe)

    # Add indicators
    print("\n2. ADDING INDICATORS")
    print("-" * 40)
    df = add_indicators(df)
    print(f"EMA Ribbon: 9/21/50")
    print(f"EWVMA: 200-period")
    print(f"Band colors - GREEN: {(df['band_color'] == 'green').sum()}, "
          f"RED: {(df['band_color'] == 'red').sum()}, "
          f"GREY: {(df['band_color'] == 'grey').sum()}")

    # Scan for patterns - try both methods
    print("\n3. SCANNING FOR PATTERNS")
    print("-" * 40)

    # Original Double Touch (usually few signals on spread)
    dt_signals = scan_double_touch_patterns(df, hedge_ratio, use_counter_trend=use_counter_trend)
    print(f"Pure Double Touch signals: {len(dt_signals)}")

    # Hybrid: Z-score + EMA ribbon confirmation
    hybrid_signals = scan_ema_ribbon_zscore_signals(df, hedge_ratio, zscore_entry=2.0, recent_window=10)
    print(f"Hybrid (Z-score + EMA) signals: {len(hybrid_signals)}")

    # Use hybrid signals (more practical for mean-reverting spread)
    signals = hybrid_signals

    long_signals = [s for s in signals if s.signal_type == 'long_spread']
    short_signals = [s for s in signals if s.signal_type == 'short_spread']

    print(f"\nUsing Hybrid strategy:")
    print(f"  Long spread signals: {len(long_signals)}")
    print(f"  Short spread signals: {len(short_signals)}")

    if not signals:
        print("\nNo signals found. Try adjusting parameters.")
        return

    # Show recent signals
    print("\nRecent signals:")
    for sig in signals[-5:]:
        rr = abs(sig.target_spread - sig.entry_spread) / abs(sig.entry_spread - sig.stop_loss_spread)
        print(f"  {sig.created_at}: {sig.signal_type} @ {sig.entry_spread:.2f}, "
              f"SL={sig.stop_loss_spread:.2f}, TP={sig.target_spread:.2f}, R:R={rr:.1f}")

    # Backtest
    print("\n4. BACKTEST RESULTS")
    print("-" * 40)
    results = backtest_double_touch_spread(df, signals)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print(f"Total signals: {results['total_signals']}")
    print(f"Completed trades: {results['completed_trades']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Average R: {results['avg_r']:.2f}")
    print(f"Total return: {results['total_return_pct']:.2f}%")
    print(f"Final equity: ${results['final_equity']:.2f}")

    print(f"\nTP exits: {results['tp_exits']} ({results['tp_exits']/results['completed_trades']*100:.1f}%)")
    print(f"SL exits: {results['sl_exits']} ({results['sl_exits']/results['completed_trades']*100:.1f}%)")
    print(f"Avg trade duration: {results['avg_duration']:.0f} candles ({results['avg_duration'] * timeframe / 60:.1f} hours)")

    print("\n" + "-" * 40)
    print("P&L BREAKDOWN")
    print("-" * 40)
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Avg win: ${results['avg_win']:.2f}")
    print(f"Avg loss: ${results['avg_loss']:.2f}")
    print(f"Largest win: ${results['largest_win']:.2f}")
    print(f"Largest loss: ${results['largest_loss']:.2f}")

    # Compare with z-score strategy
    print("\n" + "=" * 70)
    print("COMPARISON: Double Touch vs Z-Score")
    print("=" * 70)

    # Quick z-score backtest for comparison
    from spread_analysis import backtest_spread_strategy
    zscore_results = backtest_spread_strategy(
        df, hedge_ratio,
        entry_zscore=2.0, exit_zscore=0.0,
        zscore_window=60, stop_loss_zscore=4.0,
        risk_per_trade=0.02, starting_capital=10000
    )

    print(f"\n{'Metric':<25} {'Double Touch':>15} {'Z-Score':>15}")
    print("-" * 55)
    print(f"{'Total trades':<25} {results['completed_trades']:>15} {zscore_results['total_trades']:>15}")
    print(f"{'Win rate':<25} {results['win_rate']:>14.1f}% {zscore_results['win_rate']:>14.1f}%")
    print(f"{'Profit factor':<25} {results['profit_factor']:>15.2f} {zscore_results['profit_factor']:>15.2f}")
    print(f"{'Total return':<25} {results['total_return_pct']:>14.1f}% {zscore_results['total_return_pct']:>14.1f}%")
    print(f"{'Avg R-multiple':<25} {results['avg_r']:>15.2f} {'N/A':>15}")

    # Determine winner
    dt_score = 0
    zs_score = 0

    if results['win_rate'] > zscore_results['win_rate']:
        dt_score += 1
    else:
        zs_score += 1

    if results['profit_factor'] > zscore_results['profit_factor']:
        dt_score += 1
    else:
        zs_score += 1

    if results['total_return_pct'] > zscore_results['total_return_pct']:
        dt_score += 1
    else:
        zs_score += 1

    print("\n" + "-" * 55)
    if dt_score > zs_score:
        print(f"WINNER: Double Touch ({dt_score}-{zs_score})")
    elif zs_score > dt_score:
        print(f"WINNER: Z-Score ({zs_score}-{dt_score})")
    else:
        print(f"TIE ({dt_score}-{zs_score})")

    # Show trade details
    if len(results['trades']) > 0:
        print("\n" + "-" * 40)
        print("RECENT TRADES")
        print("-" * 40)
        recent = results['trades'].tail(10)
        for _, t in recent.iterrows():
            direction = "LONG" if 'long' in t['direction'] else "SHORT"
            print(f"{t['entry_time']} -> {t['exit_time']}: {direction} spread, "
                  f"R={t['r_multiple']:.2f}, ${t['pnl']:.2f} ({t['exit_reason']})")

    return {
        'df': df,
        'signals': signals,
        'results': results,
        'zscore_results': zscore_results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Spread Double Touch Analysis')
    parser.add_argument('--btc', default="/home/tahae/ai-content/data/Tradingdata/BYBIT_BTCUSDT.P, 1_da562.csv")
    parser.add_argument('--eth', default="/home/tahae/ai-content/data/Tradingdata/BYBIT_ETHUSDT.P, 1_79d61.csv")
    parser.add_argument('--timeframe', '-tf', type=int, default=5)
    parser.add_argument('--no-counter-trend', action='store_true', help='Disable counter-trend filter')

    args = parser.parse_args()

    run_analysis(
        args.btc,
        args.eth,
        timeframe=args.timeframe,
        use_counter_trend=not args.no_counter_trend
    )

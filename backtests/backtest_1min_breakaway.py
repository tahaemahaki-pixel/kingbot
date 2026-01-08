"""
Backtest Breakaway Strategy on 1-Minute Data
Tests counter-trend FVG + EWVMA cradle + volume spike + Tai Index
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def calculate_ewvma(close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    """Calculate Exponentially Weighted Volume Moving Average."""
    ewvma = np.full(len(close), np.nan)

    if len(close) < length:
        return ewvma

    alpha = 2 / (length + 1)

    # Initialize with SMA
    vol_sum = np.sum(volume[:length])
    pv_sum = np.sum(close[:length] * volume[:length])
    ewvma[length-1] = pv_sum / vol_sum if vol_sum > 0 else close[length-1]

    # Calculate EWVMA
    for i in range(length, len(close)):
        vol_sum = alpha * volume[i] + (1 - alpha) * vol_sum
        pv_sum = alpha * (close[i] * volume[i]) + (1 - alpha) * pv_sum
        ewvma[i] = pv_sum / vol_sum if vol_sum > 0 else close[i]

    return ewvma


def calculate_ewvma_std(close: np.ndarray, ewvma: np.ndarray, length: int) -> np.ndarray:
    """Calculate rolling standard deviation for EWVMA bands."""
    std = np.full(len(close), np.nan)

    for i in range(length, len(close)):
        window = close[i-length+1:i+1]
        ewvma_window = ewvma[i-length+1:i+1]
        valid_mask = ~np.isnan(ewvma_window)
        if np.sum(valid_mask) >= length // 2:
            std[i] = np.std(window[valid_mask] - ewvma_window[valid_mask])

    return std


def calculate_tai_index(close: np.ndarray, rsi_len: int = 100, stoch_len: int = 200) -> np.ndarray:
    """Calculate Tai Index (Stochastic RSI)."""
    tai = np.full(len(close), np.nan)

    if len(close) < rsi_len + stoch_len:
        return tai

    # Calculate RSI
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    rsi = np.full(len(close), np.nan)

    # EMA-based RSI
    alpha = 1 / rsi_len
    avg_gain = np.mean(gains[:rsi_len])
    avg_loss = np.mean(losses[:rsi_len])

    for i in range(rsi_len, len(close)):
        avg_gain = alpha * gains[i-1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i-1] + (1 - alpha) * avg_loss

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    # Calculate Stochastic of RSI
    for i in range(rsi_len + stoch_len, len(close)):
        rsi_window = rsi[i-stoch_len+1:i+1]
        valid = rsi_window[~np.isnan(rsi_window)]
        if len(valid) >= stoch_len // 2:
            rsi_min = np.min(valid)
            rsi_max = np.max(valid)
            if rsi_max > rsi_min:
                tai[i] = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
            else:
                tai[i] = 50

    return tai


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio vs moving average."""
    vol_ratio = np.full(len(volume), np.nan)

    for i in range(lookback, len(volume)):
        avg_vol = np.mean(volume[i-lookback:i])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol

    return vol_ratio


def check_cradle(close: np.ndarray, ewvma: np.ndarray, std: np.ndarray,
                 idx: int, lookback: int = 5, min_cradled: int = 3) -> tuple:
    """Check if price has been cradled within EWVMA bands."""
    if idx < lookback:
        return False, 0

    cradled_count = 0
    for i in range(idx - lookback + 1, idx + 1):
        if np.isnan(ewvma[i]) or np.isnan(std[i]):
            continue
        upper = ewvma[i] + std[i]
        lower = ewvma[i] - std[i]
        if lower <= close[i] <= upper:
            cradled_count += 1

    return cradled_count >= min_cradled, cradled_count


def detect_bearish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> dict:
    """Detect bearish FVG (gap down)."""
    if idx < 2:
        return None

    # Bearish FVG: current high < 2 candles ago low
    if highs[idx] < lows[idx - 2]:
        return {
            'top': lows[idx - 2],
            'bottom': highs[idx],
            'size': lows[idx - 2] - highs[idx]
        }
    return None


def detect_bullish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> dict:
    """Detect bullish FVG (gap up)."""
    if idx < 2:
        return None

    # Bullish FVG: current low > 2 candles ago high
    if lows[idx] > highs[idx - 2]:
        return {
            'top': lows[idx],
            'bottom': highs[idx - 2],
            'size': lows[idx] - highs[idx - 2]
        }
    return None


def backtest_breakaway(df: pd.DataFrame, symbol: str,
                       min_vol_ratio: float = 2.5,
                       tai_threshold_short: float = 55.0,
                       tai_threshold_long: float = 45.0,
                       min_cradle: int = 3,
                       cradle_lookback: int = 5,
                       risk_reward: float = 3.0,
                       sl_buffer_pct: float = 0.001,
                       direction: str = "both") -> dict:
    """
    Backtest Breakaway strategy on dataframe.

    Returns dict with results.
    """
    # Extract arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['Volume'].values

    # Calculate indicators
    ewvma_20 = calculate_ewvma(close, volume, 20)
    ewvma_200 = calculate_ewvma(close, volume, 200)
    ewvma_std = calculate_ewvma_std(close, ewvma_20, 20)
    tai = calculate_tai_index(close, 100, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)

    trades = []

    # Start after indicators are warmed up
    start_idx = 350

    for i in range(start_idx, len(df) - 1):
        # Skip if indicators not ready
        if np.isnan(ewvma_20[i]) or np.isnan(ewvma_200[i]) or np.isnan(tai[i]) or np.isnan(vol_ratio[i]):
            continue

        # Check for SHORT signal
        if direction in ["both", "shorts"]:
            bearish_fvg = detect_bearish_fvg(high, low, i)

            if bearish_fvg:
                # Check filters
                is_cradled, cradle_count = check_cradle(close, ewvma_20, ewvma_std, i, cradle_lookback, min_cradle)

                if (is_cradled and
                    vol_ratio[i] >= min_vol_ratio and
                    tai[i] > tai_threshold_short and
                    close[i] > ewvma_200[i]):

                    entry = bearish_fvg['top']
                    sl = entry * (1 + sl_buffer_pct)
                    risk = sl - entry
                    tp = entry - (risk * risk_reward)

                    # Simulate trade
                    result = simulate_trade(df, i, entry, sl, tp, "short")
                    if result:
                        result['vol_ratio'] = vol_ratio[i]
                        result['tai'] = tai[i]
                        result['cradle'] = cradle_count
                        trades.append(result)

        # Check for LONG signal
        if direction in ["both", "longs"]:
            bullish_fvg = detect_bullish_fvg(high, low, i)

            if bullish_fvg:
                # Check filters
                is_cradled, cradle_count = check_cradle(close, ewvma_20, ewvma_std, i, cradle_lookback, min_cradle)

                if (is_cradled and
                    vol_ratio[i] >= min_vol_ratio and
                    tai[i] < tai_threshold_long and
                    close[i] < ewvma_200[i]):

                    entry = bullish_fvg['bottom']
                    sl = entry * (1 - sl_buffer_pct)
                    risk = entry - sl
                    tp = entry + (risk * risk_reward)

                    # Simulate trade
                    result = simulate_trade(df, i, entry, sl, tp, "long")
                    if result:
                        result['vol_ratio'] = vol_ratio[i]
                        result['tai'] = tai[i]
                        result['cradle'] = cradle_count
                        trades.append(result)

    return analyze_trades(trades, symbol)


def simulate_trade(df: pd.DataFrame, entry_idx: int, entry: float, sl: float, tp: float, direction: str) -> dict:
    """Simulate a trade and return result."""

    high = df['high'].values
    low = df['low'].values

    for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
        if direction == "short":
            # Check SL first (high touches SL)
            if high[j] >= sl:
                return {
                    'direction': direction,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry': entry,
                    'exit': sl,
                    'sl': sl,
                    'tp': tp,
                    'pnl_r': -1.0,
                    'outcome': 'SL',
                    'bars_held': j - entry_idx
                }
            # Check TP (low touches TP)
            if low[j] <= tp:
                return {
                    'direction': direction,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry': entry,
                    'exit': tp,
                    'sl': sl,
                    'tp': tp,
                    'pnl_r': 3.0,
                    'outcome': 'TP',
                    'bars_held': j - entry_idx
                }
        else:  # long
            # Check SL first (low touches SL)
            if low[j] <= sl:
                return {
                    'direction': direction,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry': entry,
                    'exit': sl,
                    'sl': sl,
                    'tp': tp,
                    'pnl_r': -1.0,
                    'outcome': 'SL',
                    'bars_held': j - entry_idx
                }
            # Check TP (high touches TP)
            if high[j] >= tp:
                return {
                    'direction': direction,
                    'entry_idx': entry_idx,
                    'exit_idx': j,
                    'entry': entry,
                    'exit': tp,
                    'sl': sl,
                    'tp': tp,
                    'pnl_r': 3.0,
                    'outcome': 'TP',
                    'bars_held': j - entry_idx
                }

    return None  # Trade didn't close in time


def analyze_trades(trades: list, symbol: str) -> dict:
    """Analyze trade results."""
    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'expectancy': 0,
            'total_r': 0,
            'shorts': 0,
            'longs': 0
        }

    winners = [t for t in trades if t['pnl_r'] > 0]
    losers = [t for t in trades if t['pnl_r'] < 0]
    shorts = [t for t in trades if t['direction'] == 'short']
    longs = [t for t in trades if t['direction'] == 'long']

    total_r = sum(t['pnl_r'] for t in trades)
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    expectancy = total_r / len(trades) if trades else 0

    avg_bars = np.mean([t['bars_held'] for t in trades]) if trades else 0

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'expectancy': expectancy,
        'total_r': total_r,
        'shorts': len(shorts),
        'longs': len(longs),
        'short_winners': len([t for t in shorts if t['pnl_r'] > 0]),
        'long_winners': len([t for t in longs if t['pnl_r'] > 0]),
        'avg_bars_held': avg_bars,
        'trades': trades
    }


def simulate_account(all_trades: list, starting_balance: float = 10000,
                     risk_per_trade: float = 0.02) -> dict:
    """
    Simulate account equity with position sizing.

    Args:
        all_trades: List of all trades sorted by entry_idx
        starting_balance: Starting account balance
        risk_per_trade: Risk per trade as decimal (0.02 = 2%)

    Returns:
        Dict with simulation results
    """
    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0
    max_drawdown_pct = 0

    equity_curve = [starting_balance]
    trade_results = []

    for trade in all_trades:
        # Calculate risk amount
        risk_amount = balance * risk_per_trade

        # Calculate P&L based on R-multiple
        pnl = risk_amount * trade['pnl_r']

        # Update balance
        balance += pnl
        equity_curve.append(balance)

        # Track peak and drawdown
        if balance > peak_balance:
            peak_balance = balance

        drawdown = peak_balance - balance
        drawdown_pct = (drawdown / peak_balance) * 100 if peak_balance > 0 else 0

        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown = drawdown

        trade_results.append({
            'pnl': pnl,
            'balance': balance,
            'drawdown_pct': drawdown_pct
        })

    total_return = ((balance - starting_balance) / starting_balance) * 100

    return {
        'starting_balance': starting_balance,
        'ending_balance': balance,
        'total_return_pct': total_return,
        'total_pnl': balance - starting_balance,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'equity_curve': equity_curve,
        'trade_results': trade_results
    }


def main():
    data_dir = Path("/home/tahae/ai-content/data/Tradingdata/new 1min data")

    # Stricter parameters
    VOL_THRESHOLD = 3.0  # Increased from 2.5x
    TAI_SHORT = 55.0
    TAI_LONG = 45.0
    RISK_PER_TRADE = 0.02  # 2%
    STARTING_BALANCE = 10000

    print("=" * 70)
    print("BREAKAWAY STRATEGY - 1 MINUTE BACKTEST (STRICT FILTERS)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Volume threshold: {VOL_THRESHOLD}x (stricter)")
    print(f"  Tai Short: > {TAI_SHORT}")
    print(f"  Tai Long: < {TAI_LONG}")
    print(f"  Cradle: 3/5 candles")
    print(f"  Risk:Reward: 3:1")
    print(f"  Direction: Both (shorts + longs)")
    print(f"  Account: ${STARTING_BALANCE:,}")
    print(f"  Risk per trade: {RISK_PER_TRADE*100}%")
    print("=" * 70)

    all_results = []
    all_trades = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        if "Zone.Identifier" in str(csv_file):
            continue

        # Extract symbol from filename
        symbol = csv_file.stem.split(",")[0].replace("BYBIT_", "").replace(".P", "")

        print(f"\nProcessing {symbol}...")

        # Load data
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={'volume': 'Volume'}, inplace=True)

        # Run backtest with stricter filters
        result = backtest_breakaway(
            df, symbol,
            min_vol_ratio=VOL_THRESHOLD,
            tai_threshold_short=TAI_SHORT,
            tai_threshold_long=TAI_LONG,
            direction="both"
        )
        all_results.append(result)

        # Add symbol to trades for sorting
        for t in result.get('trades', []):
            t['symbol'] = symbol
        all_trades.extend(result.get('trades', []))

        # Print result
        print(f"  Trades: {result['total_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Expectancy: {result['expectancy']:.2f}R")
        print(f"  Total R: {result['total_r']:.1f}R")
        print(f"  Shorts: {result['shorts']} ({result['short_winners']} wins)")
        print(f"  Longs: {result['longs']} ({result['long_winners']} wins)")

    # Sort all trades by entry_idx for proper simulation order
    all_trades.sort(key=lambda x: x['entry_idx'])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL SYMBOLS (3x Volume Filter)")
    print("=" * 70)

    total_trades = sum(r['total_trades'] for r in all_results)
    total_winners = sum(r['winners'] for r in all_results)
    total_r = sum(r['total_r'] for r in all_results)
    total_shorts = sum(r['shorts'] for r in all_results)
    total_longs = sum(r['longs'] for r in all_results)
    total_short_wins = sum(r['short_winners'] for r in all_results)
    total_long_wins = sum(r['long_winners'] for r in all_results)

    print(f"Total Trades: {total_trades}")
    print(f"Overall Win Rate: {total_winners/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
    print(f"Overall Expectancy: {total_r/total_trades:.2f}R" if total_trades > 0 else "N/A")
    print(f"Total R: {total_r:.1f}R")
    print()
    print(f"Shorts: {total_shorts} trades, {total_short_wins} wins ({total_short_wins/total_shorts*100:.1f}% WR)" if total_shorts > 0 else "Shorts: 0")
    print(f"Longs: {total_longs} trades, {total_long_wins} wins ({total_long_wins/total_longs*100:.1f}% WR)" if total_longs > 0 else "Longs: 0")

    # Per-symbol table
    print("\n" + "-" * 70)
    print(f"{'Symbol':<12} {'Trades':>8} {'WR%':>8} {'Exp R':>8} {'Total R':>10} {'Shorts':>8} {'Longs':>8}")
    print("-" * 70)

    for r in sorted(all_results, key=lambda x: x['total_r'], reverse=True):
        print(f"{r['symbol']:<12} {r['total_trades']:>8} {r['win_rate']:>7.1f}% {r['expectancy']:>8.2f} {r['total_r']:>10.1f} {r['shorts']:>8} {r['longs']:>8}")

    # Account simulation
    print("\n" + "=" * 70)
    print(f"ACCOUNT SIMULATION - ${STARTING_BALANCE:,} @ {RISK_PER_TRADE*100}% risk")
    print("=" * 70)

    if all_trades:
        sim = simulate_account(all_trades, STARTING_BALANCE, RISK_PER_TRADE)

        print(f"Starting Balance:  ${sim['starting_balance']:,.2f}")
        print(f"Ending Balance:    ${sim['ending_balance']:,.2f}")
        print(f"Total P&L:         ${sim['total_pnl']:,.2f}")
        print(f"Total Return:      {sim['total_return_pct']:.1f}%")
        print(f"Max Drawdown:      ${sim['max_drawdown']:,.2f} ({sim['max_drawdown_pct']:.1f}%)")

        # Calculate some additional stats
        winning_trades = [t for t in sim['trade_results'] if t['pnl'] > 0]
        losing_trades = [t for t in sim['trade_results'] if t['pnl'] < 0]

        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            print(f"Avg Win:           ${avg_win:,.2f}")
        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            print(f"Avg Loss:          ${avg_loss:,.2f}")

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"Profit Factor:     {profit_factor:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()

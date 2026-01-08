"""
Backtest Breakaway Strategy on MSFT 4-Hour Data
Handles missing volume by using synthetic constant volume
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_ewvma(close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    """Calculate Exponentially Weighted Volume Moving Average."""
    ewvma = np.full(len(close), np.nan)
    if len(close) < length:
        return ewvma

    alpha = 2 / (length + 1)
    vol_sum = np.sum(volume[:length])
    pv_sum = np.sum(close[:length] * volume[:length])
    ewvma[length-1] = pv_sum / vol_sum if vol_sum > 0 else close[length-1]

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

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    rsi = np.full(len(close), np.nan)

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
    if lows[idx] > highs[idx - 2]:
        return {
            'top': lows[idx],
            'bottom': highs[idx - 2],
            'size': lows[idx] - highs[idx - 2]
        }
    return None


def simulate_trade(high: np.ndarray, low: np.ndarray, entry_idx: int,
                   entry: float, sl: float, tp: float, direction: str, max_bars: int = 100) -> dict:
    """Simulate a trade and return result."""
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(high))):
        if direction == "short":
            if high[j] >= sl:
                return {'direction': direction, 'entry_idx': entry_idx, 'exit_idx': j,
                        'entry': entry, 'exit': sl, 'pnl_r': -1.0, 'outcome': 'SL', 'bars_held': j - entry_idx}
            if low[j] <= tp:
                return {'direction': direction, 'entry_idx': entry_idx, 'exit_idx': j,
                        'entry': entry, 'exit': tp, 'pnl_r': 3.0, 'outcome': 'TP', 'bars_held': j - entry_idx}
        else:
            if low[j] <= sl:
                return {'direction': direction, 'entry_idx': entry_idx, 'exit_idx': j,
                        'entry': entry, 'exit': sl, 'pnl_r': -1.0, 'outcome': 'SL', 'bars_held': j - entry_idx}
            if high[j] >= tp:
                return {'direction': direction, 'entry_idx': entry_idx, 'exit_idx': j,
                        'entry': entry, 'exit': tp, 'pnl_r': 3.0, 'outcome': 'TP', 'bars_held': j - entry_idx}

    # Timeout - exit at current price
    return {'direction': direction, 'entry_idx': entry_idx, 'exit_idx': entry_idx + max_bars,
            'entry': entry, 'exit': low[min(entry_idx + max_bars, len(low)-1)],
            'pnl_r': 0, 'outcome': 'TIMEOUT', 'bars_held': max_bars}


def backtest_breakaway(df: pd.DataFrame, symbol: str,
                       tai_threshold_short: float = 53.0,
                       tai_threshold_long: float = 47.0,
                       min_cradle: int = 3,
                       cradle_lookback: int = 5,
                       risk_reward: float = 3.0,
                       sl_buffer_pct: float = 0.001,
                       direction: str = "both") -> dict:
    """Backtest Breakaway strategy (no volume filter for MSFT)."""

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Use synthetic constant volume since MSFT data lacks volume
    volume = np.ones(len(close)) * 1000000

    # Calculate indicators
    ewvma_20 = calculate_ewvma(close, volume, 20)
    ewvma_200 = calculate_ewvma(close, volume, 200)
    ewvma_std = calculate_ewvma_std(close, ewvma_20, 20)
    tai = calculate_tai_index(close, 100, 200)

    trades = []
    start_idx = 350

    for i in range(start_idx, len(df) - 1):
        if np.isnan(ewvma_20[i]) or np.isnan(ewvma_200[i]) or np.isnan(tai[i]):
            continue

        # Check for SHORT signal
        if direction in ["both", "shorts"]:
            bearish_fvg = detect_bearish_fvg(high, low, i)
            if bearish_fvg:
                is_cradled, cradle_count = check_cradle(close, ewvma_20, ewvma_std, i, cradle_lookback, min_cradle)
                if (is_cradled and
                    tai[i] > tai_threshold_short and
                    close[i] > ewvma_200[i]):

                    entry = bearish_fvg['top']
                    sl = entry * (1 + sl_buffer_pct)
                    risk = sl - entry
                    tp = entry - (risk * risk_reward)

                    result = simulate_trade(high, low, i, entry, sl, tp, "short")
                    if result:
                        result['tai'] = tai[i]
                        result['cradle'] = cradle_count
                        trades.append(result)

        # Check for LONG signal
        if direction in ["both", "longs"]:
            bullish_fvg = detect_bullish_fvg(high, low, i)
            if bullish_fvg:
                is_cradled, cradle_count = check_cradle(close, ewvma_20, ewvma_std, i, cradle_lookback, min_cradle)
                if (is_cradled and
                    tai[i] < tai_threshold_long and
                    close[i] < ewvma_200[i]):

                    entry = bullish_fvg['bottom']
                    sl = entry * (1 - sl_buffer_pct)
                    risk = entry - sl
                    tp = entry + (risk * risk_reward)

                    result = simulate_trade(high, low, i, entry, sl, tp, "long")
                    if result:
                        result['tai'] = tai[i]
                        result['cradle'] = cradle_count
                        trades.append(result)

    return analyze_trades(trades, symbol)


def analyze_trades(trades: list, symbol: str) -> dict:
    """Analyze trade results."""
    if not trades:
        return {'symbol': symbol, 'total_trades': 0, 'win_rate': 0, 'expectancy': 0, 'total_r': 0}

    winners = [t for t in trades if t['pnl_r'] > 0]
    losers = [t for t in trades if t['pnl_r'] < 0]
    timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']
    shorts = [t for t in trades if t['direction'] == 'short']
    longs = [t for t in trades if t['direction'] == 'long']

    total_r = sum(t['pnl_r'] for t in trades)
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    expectancy = total_r / len(trades) if trades else 0

    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'timeouts': len(timeouts),
        'win_rate': win_rate,
        'expectancy': expectancy,
        'total_r': total_r,
        'shorts': len(shorts),
        'longs': len(longs),
        'short_winners': len([t for t in shorts if t['pnl_r'] > 0]),
        'long_winners': len([t for t in longs if t['pnl_r'] > 0]),
        'trades': trades
    }


def main():
    """Run backtest on MSFT data."""
    filepath = "/home/tahae/ai-content/data/Tradingdata/chart_data/BATS_MSFT, 240_ae198.csv"

    print("=" * 60)
    print("BREAKAWAY STRATEGY BACKTEST - MSFT (4-Hour)")
    print("=" * 60)
    print("\nNote: Volume filter DISABLED (MSFT data lacks volume)")
    print("Using: FVG + Cradle + Tai Index + Counter-trend filter only\n")

    # Load data
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])

    print(f"Data loaded: {len(df)} candles")
    print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print()

    # Run backtest with deployed parameters
    results = backtest_breakaway(
        df, "MSFT",
        tai_threshold_short=53.0,
        tai_threshold_long=47.0,
        min_cradle=3,
        cradle_lookback=5,
        risk_reward=3.0,
        sl_buffer_pct=0.001,
        direction="both"
    )

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Trades:    {results['total_trades']}")
    print(f"Winners:         {results['winners']}")
    print(f"Losers:          {results['losers']}")
    print(f"Timeouts:        {results['timeouts']}")
    print(f"Win Rate:        {results['win_rate']:.1f}%")
    print(f"Expectancy:      {results['expectancy']:+.2f}R")
    print(f"Total R:         {results['total_r']:+.0f}R")
    print()
    print(f"Shorts:          {results['shorts']} ({results['short_winners']} winners)")
    print(f"Longs:           {results['longs']} ({results['long_winners']} winners)")
    print("=" * 60)

    # Show sample trades
    if results['trades']:
        print("\nSample Trades (first 10):")
        print("-" * 60)
        for t in results['trades'][:10]:
            print(f"  {t['direction'].upper():5} | Entry: ${t['entry']:.2f} | "
                  f"{t['outcome']:7} | R: {t['pnl_r']:+.1f} | Tai: {t['tai']:.0f}")

    return results


if __name__ == "__main__":
    main()

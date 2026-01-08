"""
Compare Breakaway Strategy WITH vs WITHOUT Volume Filter
Tests if removing volume improves win rate on crypto data
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
    if idx < 2:
        return None
    if highs[idx] < lows[idx - 2]:
        return {'top': lows[idx - 2], 'bottom': highs[idx]}
    return None


def detect_bullish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> dict:
    if idx < 2:
        return None
    if lows[idx] > highs[idx - 2]:
        return {'top': lows[idx], 'bottom': highs[idx - 2]}
    return None


def simulate_trade(high: np.ndarray, low: np.ndarray, entry_idx: int,
                   entry: float, sl: float, tp: float, direction: str, max_bars: int = 100) -> dict:
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(high))):
        if direction == "short":
            if high[j] >= sl:
                return {'pnl_r': -1.0, 'outcome': 'SL'}
            if low[j] <= tp:
                return {'pnl_r': 3.0, 'outcome': 'TP'}
        else:
            if low[j] <= sl:
                return {'pnl_r': -1.0, 'outcome': 'SL'}
            if high[j] >= tp:
                return {'pnl_r': 3.0, 'outcome': 'TP'}
    return {'pnl_r': 0, 'outcome': 'TIMEOUT'}


def backtest(df: pd.DataFrame, use_volume_filter: bool = True, min_vol_ratio: float = 2.0):
    """Run backtest with or without volume filter."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Get volume column
    vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
    volume = df[vol_col].values if vol_col in df.columns else np.ones(len(close))

    ewvma_20 = calculate_ewvma(close, volume, 20)
    ewvma_200 = calculate_ewvma(close, volume, 200)
    ewvma_std = calculate_ewvma_std(close, ewvma_20, 20)
    tai = calculate_tai_index(close, 100, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)

    trades = []
    start_idx = 350

    for i in range(start_idx, len(df) - 1):
        if np.isnan(ewvma_20[i]) or np.isnan(ewvma_200[i]) or np.isnan(tai[i]):
            continue

        # Volume check (skip if filter disabled)
        if use_volume_filter and (np.isnan(vol_ratio[i]) or vol_ratio[i] < min_vol_ratio):
            pass  # Will check in conditions below

        # SHORT
        bearish_fvg = detect_bearish_fvg(high, low, i)
        if bearish_fvg:
            is_cradled, _ = check_cradle(close, ewvma_20, ewvma_std, i, 5, 3)
            vol_ok = (not use_volume_filter) or (vol_ratio[i] >= min_vol_ratio)

            if is_cradled and vol_ok and tai[i] > 53 and close[i] > ewvma_200[i]:
                entry = bearish_fvg['top']
                sl = entry * 1.001
                tp = entry - (sl - entry) * 3
                result = simulate_trade(high, low, i, entry, sl, tp, "short")
                result['direction'] = 'short'
                trades.append(result)

        # LONG
        bullish_fvg = detect_bullish_fvg(high, low, i)
        if bullish_fvg:
            is_cradled, _ = check_cradle(close, ewvma_20, ewvma_std, i, 5, 3)
            vol_ok = (not use_volume_filter) or (vol_ratio[i] >= min_vol_ratio)

            if is_cradled and vol_ok and tai[i] < 47 and close[i] < ewvma_200[i]:
                entry = bullish_fvg['bottom']
                sl = entry * 0.999
                tp = entry + (entry - sl) * 3
                result = simulate_trade(high, low, i, entry, sl, tp, "long")
                result['direction'] = 'long'
                trades.append(result)

    if not trades:
        return {'trades': 0, 'winners': 0, 'win_rate': 0, 'expectancy': 0, 'total_r': 0}

    winners = [t for t in trades if t['pnl_r'] > 0]
    total_r = sum(t['pnl_r'] for t in trades)

    return {
        'trades': len(trades),
        'winners': len(winners),
        'losers': len([t for t in trades if t['pnl_r'] < 0]),
        'timeouts': len([t for t in trades if t['outcome'] == 'TIMEOUT']),
        'win_rate': len(winners) / len(trades) * 100,
        'expectancy': total_r / len(trades),
        'total_r': total_r
    }


def main():
    files = [
        "/home/tahae/ai-content/data/Tradingdata/volume charts/BTCUSDT_5m_merged.csv",
        "/home/tahae/ai-content/data/Tradingdata/volume charts/DOTUSDT_5m_merged.csv",
        "/home/tahae/ai-content/data/Tradingdata/volume charts/PNUTUSDT_5m_merged.csv",
    ]

    print("=" * 70)
    print("VOLUME FILTER COMPARISON - Breakaway Strategy")
    print("=" * 70)
    print()

    all_with_vol = {'trades': 0, 'winners': 0, 'losers': 0, 'total_r': 0}
    all_no_vol = {'trades': 0, 'winners': 0, 'losers': 0, 'total_r': 0}

    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}")
            continue

        symbol = path.stem.split('_')[0]
        df = pd.read_csv(filepath)
        print(f"\n{symbol} ({len(df)} candles)")
        print("-" * 50)

        # WITH volume filter
        with_vol = backtest(df, use_volume_filter=True, min_vol_ratio=2.0)

        # WITHOUT volume filter
        no_vol = backtest(df, use_volume_filter=False)

        print(f"{'':15} {'WITH Vol':>12} {'NO Vol':>12} {'Diff':>10}")
        print(f"{'Trades':15} {with_vol['trades']:>12} {no_vol['trades']:>12} {no_vol['trades'] - with_vol['trades']:>+10}")
        print(f"{'Winners':15} {with_vol['winners']:>12} {no_vol['winners']:>12} {no_vol['winners'] - with_vol['winners']:>+10}")
        print(f"{'Win Rate':15} {with_vol['win_rate']:>11.1f}% {no_vol['win_rate']:>11.1f}% {no_vol['win_rate'] - with_vol['win_rate']:>+9.1f}%")
        print(f"{'Expectancy':15} {with_vol['expectancy']:>+11.2f}R {no_vol['expectancy']:>+11.2f}R {no_vol['expectancy'] - with_vol['expectancy']:>+9.2f}R")
        print(f"{'Total R':15} {with_vol['total_r']:>+11.0f}R {no_vol['total_r']:>+11.0f}R {no_vol['total_r'] - with_vol['total_r']:>+9.0f}R")

        # Aggregate
        all_with_vol['trades'] += with_vol['trades']
        all_with_vol['winners'] += with_vol['winners']
        all_with_vol['losers'] += with_vol['losers']
        all_with_vol['total_r'] += with_vol['total_r']

        all_no_vol['trades'] += no_vol['trades']
        all_no_vol['winners'] += no_vol['winners']
        all_no_vol['losers'] += no_vol['losers']
        all_no_vol['total_r'] += no_vol['total_r']

    # Calculate aggregate stats
    if all_with_vol['trades'] > 0:
        all_with_vol['win_rate'] = all_with_vol['winners'] / all_with_vol['trades'] * 100
        all_with_vol['expectancy'] = all_with_vol['total_r'] / all_with_vol['trades']
    else:
        all_with_vol['win_rate'] = 0
        all_with_vol['expectancy'] = 0

    if all_no_vol['trades'] > 0:
        all_no_vol['win_rate'] = all_no_vol['winners'] / all_no_vol['trades'] * 100
        all_no_vol['expectancy'] = all_no_vol['total_r'] / all_no_vol['trades']
    else:
        all_no_vol['win_rate'] = 0
        all_no_vol['expectancy'] = 0

    print()
    print("=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'':15} {'WITH Vol':>12} {'NO Vol':>12} {'Diff':>10}")
    print(f"{'Trades':15} {all_with_vol['trades']:>12} {all_no_vol['trades']:>12} {all_no_vol['trades'] - all_with_vol['trades']:>+10}")
    print(f"{'Winners':15} {all_with_vol['winners']:>12} {all_no_vol['winners']:>12} {all_no_vol['winners'] - all_with_vol['winners']:>+10}")
    print(f"{'Win Rate':15} {all_with_vol['win_rate']:>11.1f}% {all_no_vol['win_rate']:>11.1f}% {all_no_vol['win_rate'] - all_with_vol['win_rate']:>+9.1f}%")
    print(f"{'Expectancy':15} {all_with_vol['expectancy']:>+11.2f}R {all_no_vol['expectancy']:>+11.2f}R {all_no_vol['expectancy'] - all_with_vol['expectancy']:>+9.2f}R")
    print(f"{'Total R':15} {all_with_vol['total_r']:>+11.0f}R {all_no_vol['total_r']:>+11.0f}R {all_no_vol['total_r'] - all_with_vol['total_r']:>+9.0f}R")
    print("=" * 70)

    # Verdict
    print()
    if all_no_vol['win_rate'] > all_with_vol['win_rate']:
        print("VERDICT: Removing volume filter IMPROVES win rate")
    else:
        print("VERDICT: Keeping volume filter is BETTER for win rate")

    if all_no_vol['expectancy'] > all_with_vol['expectancy']:
        print("         Removing volume filter IMPROVES expectancy")
    else:
        print("         Keeping volume filter is BETTER for expectancy")

    if all_no_vol['total_r'] > all_with_vol['total_r']:
        print("         Removing volume filter produces MORE total R")
    else:
        print("         Keeping volume filter produces MORE total R")


if __name__ == "__main__":
    main()

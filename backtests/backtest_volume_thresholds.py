"""
Test Different Volume Thresholds for Breakaway Strategy
Find optimal balance between signal quantity and quality
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_ewvma(close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
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
    std = np.full(len(close), np.nan)
    for i in range(length, len(close)):
        window = close[i-length+1:i+1]
        ewvma_window = ewvma[i-length+1:i+1]
        valid_mask = ~np.isnan(ewvma_window)
        if np.sum(valid_mask) >= length // 2:
            std[i] = np.std(window[valid_mask] - ewvma_window[valid_mask])
    return std


def calculate_tai_index(close: np.ndarray, rsi_len: int = 100, stoch_len: int = 200) -> np.ndarray:
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
    vol_ratio = np.full(len(volume), np.nan)
    for i in range(lookback, len(volume)):
        avg_vol = np.mean(volume[i-lookback:i])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol
    return vol_ratio


def check_cradle(close: np.ndarray, ewvma: np.ndarray, std: np.ndarray,
                 idx: int, lookback: int = 5, min_cradled: int = 3) -> tuple:
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


def backtest_with_threshold(df: pd.DataFrame, min_vol_ratio: float):
    """Run backtest with specific volume threshold."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
    volume = df[vol_col].values

    ewvma_20 = calculate_ewvma(close, volume, 20)
    ewvma_200 = calculate_ewvma(close, volume, 200)
    ewvma_std = calculate_ewvma_std(close, ewvma_20, 20)
    tai = calculate_tai_index(close, 100, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)

    trades = []
    start_idx = 350

    for i in range(start_idx, len(df) - 1):
        if np.isnan(ewvma_20[i]) or np.isnan(ewvma_200[i]) or np.isnan(tai[i]) or np.isnan(vol_ratio[i]):
            continue

        # SHORT
        bearish_fvg = detect_bearish_fvg(high, low, i)
        if bearish_fvg:
            is_cradled, _ = check_cradle(close, ewvma_20, ewvma_std, i, 5, 3)
            if is_cradled and vol_ratio[i] >= min_vol_ratio and tai[i] > 53 and close[i] > ewvma_200[i]:
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
            if is_cradled and vol_ratio[i] >= min_vol_ratio and tai[i] < 47 and close[i] < ewvma_200[i]:
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

    # Test these volume thresholds
    thresholds = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    print("=" * 80)
    print("VOLUME THRESHOLD OPTIMIZATION - Breakaway Strategy")
    print("=" * 80)
    print()

    # Load all data
    all_dfs = []
    for filepath in files:
        path = Path(filepath)
        if path.exists():
            df = pd.read_csv(filepath)
            all_dfs.append((path.stem.split('_')[0], df))
            print(f"Loaded {path.stem.split('_')[0]}: {len(df)} candles")

    print()
    print("=" * 80)
    print(f"{'Vol Thresh':>10} {'Trades':>8} {'Winners':>8} {'Win Rate':>10} {'Expectancy':>12} {'Total R':>10}")
    print("=" * 80)

    results = []

    for threshold in thresholds:
        agg = {'trades': 0, 'winners': 0, 'total_r': 0}

        for symbol, df in all_dfs:
            r = backtest_with_threshold(df, threshold)
            agg['trades'] += r['trades']
            agg['winners'] += r['winners']
            agg['total_r'] += r['total_r']

        if agg['trades'] > 0:
            agg['win_rate'] = agg['winners'] / agg['trades'] * 100
            agg['expectancy'] = agg['total_r'] / agg['trades']
        else:
            agg['win_rate'] = 0
            agg['expectancy'] = 0

        results.append((threshold, agg))

        label = "None" if threshold == 0 else f"{threshold}x"
        print(f"{label:>10} {agg['trades']:>8} {agg['winners']:>8} {agg['win_rate']:>9.1f}% {agg['expectancy']:>+11.2f}R {agg['total_r']:>+9.0f}R")

    print("=" * 80)

    # Find best by different metrics
    best_wr = max(results, key=lambda x: x[1]['win_rate'])
    best_exp = max(results, key=lambda x: x[1]['expectancy'])
    best_r = max(results, key=lambda x: x[1]['total_r'])

    print()
    print("OPTIMAL THRESHOLDS:")
    print(f"  Best Win Rate:    {best_wr[0]}x ({best_wr[1]['win_rate']:.1f}%)")
    print(f"  Best Expectancy:  {best_exp[0]}x ({best_exp[1]['expectancy']:+.2f}R)")
    print(f"  Best Total R:     {best_r[0]}x ({best_r[1]['total_r']:+.0f}R)")

    # Calculate efficiency score (win_rate * expectancy * log(trades))
    print()
    print("EFFICIENCY ANALYSIS (balancing quality vs quantity):")
    print("-" * 60)

    for threshold, agg in results:
        if agg['trades'] > 0:
            # Score that balances all factors
            trades_factor = np.log10(agg['trades'] + 1)
            efficiency = agg['win_rate'] * agg['expectancy'] * trades_factor / 100
            label = "None" if threshold == 0 else f"{threshold}x"
            print(f"  {label:>6}: WR={agg['win_rate']:5.1f}% × Exp={agg['expectancy']:+.2f}R × log(trades)={trades_factor:.2f} = Score: {efficiency:.2f}")

    # Find sweet spot
    print()
    print("RECOMMENDATION:")
    print("-" * 60)

    # Compare 1.5x vs 2.0x specifically
    r_1_5 = next(r for t, r in results if t == 1.5)
    r_2_0 = next(r for t, r in results if t == 2.0)

    print(f"  Current (2.0x): {r_2_0['trades']} trades, {r_2_0['win_rate']:.1f}% WR, {r_2_0['expectancy']:+.2f}R exp")
    print(f"  Alt (1.5x):     {r_1_5['trades']} trades, {r_1_5['win_rate']:.1f}% WR, {r_1_5['expectancy']:+.2f}R exp")
    print()

    if r_1_5['expectancy'] > r_2_0['expectancy'] * 0.9 and r_1_5['trades'] > r_2_0['trades'] * 1.3:
        print("  → Consider lowering to 1.5x for more signals with acceptable quality")
    elif r_2_0['win_rate'] > r_1_5['win_rate'] + 5:
        print("  → Keep 2.0x for higher win rate")
    else:
        print("  → 2.0x is a good balance")


if __name__ == "__main__":
    main()

"""
Filter Variations Backtest - Test adding/removing filters from Breakaway Strategy

Tests:
1. Baseline (all current filters)
2. Remove Tai Index filter
3. Remove EWVMA-200 trend filter
4. Remove Cradle filter
5. Add: Higher volume threshold (3.0x)
6. Add: Imbalance + looser Tai (50/50)
7. Add: Imbalance only (no Tai)
8. Add: Different imbalance lookbacks (5, 10, 20)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeResult:
    signal_type: str
    entry_price: float
    stop_loss: float
    target: float
    entry_idx: int
    exit_idx: int
    exit_price: float
    pnl_r: float
    won: bool


def calculate_ewvma(closes: np.ndarray, volumes: np.ndarray, length: int) -> np.ndarray:
    n = len(closes)
    ewvma = np.zeros(n)
    alpha = 2 / (length + 1)
    vol_sum = 0
    pv_sum = 0
    for i in range(n):
        vol = max(volumes[i], 1)
        if i == 0:
            ewvma[i] = closes[i]
            vol_sum = vol
            pv_sum = closes[i] * vol
        else:
            vol_sum = alpha * vol + (1 - alpha) * vol_sum
            pv_sum = alpha * (closes[i] * vol) + (1 - alpha) * pv_sum
            ewvma[i] = pv_sum / vol_sum if vol_sum > 0 else closes[i]
    return ewvma


def calculate_tai_index(closes: np.ndarray, rsi_length: int = 100, stoch_length: int = 200) -> np.ndarray:
    n = len(closes)
    tai = np.full(n, 50.0)
    if n < rsi_length + stoch_length:
        return tai

    delta = np.diff(closes, prepend=closes[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    alpha = 2 / (rsi_length + 1)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    for i in range(1, n):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    for i in range(stoch_length - 1, n):
        rsi_window = rsi[i - stoch_length + 1:i + 1]
        rsi_min, rsi_max = np.min(rsi_window), np.max(rsi_window)
        if rsi_max - rsi_min > 0:
            tai[i] = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
    return tai


def calculate_volume_ratio(volumes: np.ndarray, lookback: int = 20) -> np.ndarray:
    n = len(volumes)
    vol_ratio = np.ones(n)
    for i in range(lookback - 1, n):
        vol_sma = np.mean(volumes[i - lookback + 1:i + 1])
        if vol_sma > 0:
            vol_ratio[i] = volumes[i] / vol_sma
    return vol_ratio


def calculate_imbalance(opens: np.ndarray, closes: np.ndarray, volumes: np.ndarray, lookback: int = 10) -> np.ndarray:
    n = len(closes)
    imbalance = np.zeros(n)
    is_bullish = closes > opens
    buy_volume = np.where(is_bullish, volumes, 0)
    sell_volume = np.where(~is_bullish, volumes, 0)
    buy_cumsum = np.cumsum(buy_volume)
    sell_cumsum = np.cumsum(sell_volume)

    for i in range(lookback, n):
        buy_sum = buy_cumsum[i] - buy_cumsum[i - lookback]
        sell_sum = sell_cumsum[i] - sell_cumsum[i - lookback]
        total = buy_sum + sell_sum
        if total > 0:
            imbalance[i] = (buy_sum - sell_sum) / total
    return imbalance


def run_backtest(
    df: pd.DataFrame,
    # Filter toggles
    use_tai_filter: bool = True,
    use_trend_filter: bool = True,
    use_cradle_filter: bool = True,
    use_imbalance_filter: bool = False,
    # Parameters
    min_vol_ratio: float = 2.0,
    tai_short: float = 53.0,
    tai_long: float = 47.0,
    imbalance_threshold: float = 0.10,
    imbalance_lookback: int = 10,
) -> Tuple[List[TradeResult], Dict]:

    opens = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    n = len(df)

    # Pre-calculate indicators
    ewvma_20 = calculate_ewvma(closes, volumes, 20)
    ewvma_20_std = pd.Series(closes).rolling(20).std().fillna(0).values
    ewvma_200 = calculate_ewvma(closes, volumes, 200)
    tai_index = calculate_tai_index(closes)
    vol_ratio = calculate_volume_ratio(volumes)
    imbalance = calculate_imbalance(opens, closes, volumes, imbalance_lookback)

    upper_band = ewvma_20 + ewvma_20_std
    lower_band = ewvma_20 - ewvma_20_std
    in_cradle = (closes >= lower_band) & (closes <= upper_band)

    trades = []
    signals_generated = 0
    last_signal_idx = -10

    for i in range(300, n - 50):
        if i <= last_signal_idx + 1:
            continue

        # Volume spike (always required)
        if vol_ratio[i] < min_vol_ratio:
            continue

        # Cradle filter (optional)
        if use_cradle_filter:
            if i < 5:
                continue
            cradle_count = np.sum(in_cradle[i-5:i])
            if cradle_count < 3:
                continue

        signal_type = None
        fvg = None

        # Check SHORT
        if i >= 2 and highs[i] < lows[i - 2]:
            # Tai filter
            if use_tai_filter and tai_index[i] <= tai_short:
                pass
            # Trend filter
            elif use_trend_filter and closes[i] <= ewvma_200[i]:
                pass
            # Imbalance filter
            elif use_imbalance_filter and imbalance[i] > -imbalance_threshold:
                pass
            else:
                tai_ok = (not use_tai_filter) or (tai_index[i] > tai_short)
                trend_ok = (not use_trend_filter) or (closes[i] > ewvma_200[i])
                imb_ok = (not use_imbalance_filter) or (imbalance[i] < -imbalance_threshold)
                if tai_ok and trend_ok and imb_ok:
                    signal_type = 'short'
                    fvg = (lows[i - 2], highs[i])

        # Check LONG
        if signal_type is None and i >= 2 and lows[i] > highs[i - 2]:
            tai_ok = (not use_tai_filter) or (tai_index[i] < tai_long)
            trend_ok = (not use_trend_filter) or (closes[i] < ewvma_200[i])
            imb_ok = (not use_imbalance_filter) or (imbalance[i] > imbalance_threshold)
            if tai_ok and trend_ok and imb_ok:
                signal_type = 'long'
                fvg = (lows[i], highs[i - 2])

        if signal_type is None:
            continue

        signals_generated += 1
        last_signal_idx = i

        # Trade levels
        fvg_top, fvg_bottom = fvg
        sl_buffer = 0.001

        if signal_type == 'short':
            entry = fvg_bottom
            sl = fvg_top * (1 + sl_buffer)
            risk = sl - entry
            tp = entry - (risk * 3.0)
        else:
            entry = fvg_top
            sl = fvg_bottom * (1 - sl_buffer)
            risk = entry - sl
            tp = entry + (risk * 3.0)

        # Simulate
        exit_price, won, exit_idx = None, None, i + 1
        for j in range(i + 1, min(i + 100, n)):
            if signal_type == 'short':
                if highs[j] >= sl:
                    exit_price, won, exit_idx = sl, False, j
                    break
                if lows[j] <= tp:
                    exit_price, won, exit_idx = tp, True, j
                    break
            else:
                if lows[j] <= sl:
                    exit_price, won, exit_idx = sl, False, j
                    break
                if highs[j] >= tp:
                    exit_price, won, exit_idx = tp, True, j
                    break

        if exit_price is None:
            exit_price = closes[min(i + 99, n - 1)]
            exit_idx = min(i + 99, n - 1)
            won = (exit_price < entry) if signal_type == 'short' else (exit_price > entry)

        pnl = (entry - exit_price) if signal_type == 'short' else (exit_price - entry)
        pnl_r = pnl / risk if risk > 0 else 0

        trades.append(TradeResult(
            signal_type=signal_type,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            entry_idx=i,
            exit_idx=exit_idx,
            exit_price=exit_price,
            pnl_r=pnl_r,
            won=won
        ))

    # Stats
    if not trades:
        return [], {'trades': 0, 'win_rate': 0, 'avg_r': 0, 'total_r': 0, 'pf': 0}

    wins = sum(1 for t in trades if t.won)
    gross_profit = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))

    stats = {
        'trades': len(trades),
        'win_rate': wins / len(trades) * 100,
        'avg_r': np.mean([t.pnl_r for t in trades]),
        'total_r': sum(t.pnl_r for t in trades),
        'pf': gross_profit / gross_loss if gross_loss > 0 else float('inf')
    }
    return trades, stats


def main():
    data_dir = Path("/home/tahae/ai-content/data/Tradingdata/volume charts")

    # Load all data
    files = [
        ("BTCUSDT", "BTCUSDT_5m_merged.csv"),
        ("DOTUSDT", "DOTUSDT_5m_merged.csv"),
        ("PNUTUSDT", "PNUTUSDT_5m_merged.csv"),
    ]

    dfs = {}
    for symbol, filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df.columns = [c.lower().strip() for c in df.columns]
            dfs[symbol] = df
            print(f"Loaded {symbol}: {len(df):,} candles")

    print("\n" + "=" * 100)
    print("FILTER VARIATIONS ANALYSIS")
    print("=" * 100)

    # Define test configurations
    configs = [
        # name, use_tai, use_trend, use_cradle, use_imb, vol_ratio, tai_s, tai_l, imb_thresh, imb_lookback
        ("1. Baseline (all filters)", True, True, True, False, 2.0, 53, 47, 0.10, 10),
        ("2. Remove Tai Index", False, True, True, False, 2.0, 53, 47, 0.10, 10),
        ("3. Remove EWVMA-200 Trend", True, False, True, False, 2.0, 53, 47, 0.10, 10),
        ("4. Remove Cradle", True, True, False, False, 2.0, 53, 47, 0.10, 10),
        ("5. Higher volume (3.0x)", True, True, True, False, 3.0, 53, 47, 0.10, 10),
        ("6. Add Imbalance 0.10", True, True, True, True, 2.0, 53, 47, 0.10, 10),
        ("7. Imbalance + Looser Tai (50/50)", True, True, True, True, 2.0, 50, 50, 0.10, 10),
        ("8. Imbalance + No Tai", False, True, True, True, 2.0, 53, 47, 0.10, 10),
        ("9. Imbalance + No Trend", True, False, True, True, 2.0, 53, 47, 0.10, 10),
        ("10. Imbalance only (no Tai/Trend)", False, False, True, True, 2.0, 53, 47, 0.10, 10),
        ("11. Imbalance 0.15 + all filters", True, True, True, True, 2.0, 53, 47, 0.15, 10),
        ("12. Imbalance 0.20 + all filters", True, True, True, True, 2.0, 53, 47, 0.20, 10),
        ("13. Imbalance lookback 5", True, True, True, True, 2.0, 53, 47, 0.10, 5),
        ("14. Imbalance lookback 20", True, True, True, True, 2.0, 53, 47, 0.10, 20),
        ("15. Vol 2.5x + Imbalance 0.10", True, True, True, True, 2.5, 53, 47, 0.10, 10),
    ]

    results = []

    print(f"\n{'Configuration':<40} | {'Trades':>6} | {'Win%':>6} | {'AvgR':>7} | {'TotalR':>8} | {'PF':>5}")
    print("-" * 85)

    for config in configs:
        name = config[0]
        use_tai, use_trend, use_cradle, use_imb = config[1:5]
        vol_ratio, tai_s, tai_l, imb_thresh, imb_lookback = config[5:]

        total_trades = 0
        total_wins = 0
        total_r = 0
        total_profit = 0
        total_loss = 0

        for symbol, df in dfs.items():
            trades, stats = run_backtest(
                df,
                use_tai_filter=use_tai,
                use_trend_filter=use_trend,
                use_cradle_filter=use_cradle,
                use_imbalance_filter=use_imb,
                min_vol_ratio=vol_ratio,
                tai_short=tai_s,
                tai_long=tai_l,
                imbalance_threshold=imb_thresh,
                imbalance_lookback=imb_lookback,
            )
            total_trades += stats['trades']
            total_wins += sum(1 for t in trades if t.won)
            total_r += stats['total_r']
            total_profit += sum(t.pnl_r for t in trades if t.pnl_r > 0)
            total_loss += abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))

        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        avg_r = total_r / total_trades if total_trades > 0 else 0
        pf = total_profit / total_loss if total_loss > 0 else 0

        print(f"{name:<40} | {total_trades:>6} | {win_rate:>5.1f}% | {avg_r:>+6.2f}R | {total_r:>+7.1f}R | {pf:>5.2f}")

        results.append({
            'name': name,
            'trades': total_trades,
            'win_rate': win_rate,
            'avg_r': avg_r,
            'total_r': total_r,
            'pf': pf
        })

    # Find best configs
    print("\n" + "=" * 100)
    print("TOP 5 BY EXPECTANCY (min 50 trades)")
    print("=" * 100)

    valid = [r for r in results if r['trades'] >= 50]
    by_expectancy = sorted(valid, key=lambda x: x['avg_r'], reverse=True)[:5]

    for i, r in enumerate(by_expectancy, 1):
        print(f"{i}. {r['name']:<40} | AvgR: {r['avg_r']:+.3f} | Win: {r['win_rate']:.1f}% | PF: {r['pf']:.2f} | Trades: {r['trades']}")

    print("\n" + "=" * 100)
    print("TOP 5 BY TOTAL R (profit)")
    print("=" * 100)

    by_total = sorted(valid, key=lambda x: x['total_r'], reverse=True)[:5]

    for i, r in enumerate(by_total, 1):
        print(f"{i}. {r['name']:<40} | TotalR: {r['total_r']:+.1f} | AvgR: {r['avg_r']:+.3f} | Trades: {r['trades']}")

    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Compare specific pairs
    baseline = next(r for r in results if '1. Baseline' in r['name'])

    print(f"\nBaseline: {baseline['avg_r']:+.3f}R expectancy, {baseline['trades']} trades")

    for r in results[1:]:
        diff = r['avg_r'] - baseline['avg_r']
        trade_diff = r['trades'] - baseline['trades']
        if abs(diff) > 0.05 or abs(trade_diff) > 50:
            emoji = "✅" if diff > 0 else "❌" if diff < -0.05 else "➖"
            print(f"{emoji} {r['name']}: {diff:+.3f}R change, {trade_diff:+d} trades")


if __name__ == "__main__":
    main()

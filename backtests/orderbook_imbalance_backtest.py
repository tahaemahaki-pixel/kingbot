"""
Order Book Imbalance Backtest for Breakaway Strategy (Optimized)

Since we don't have historical order book data, we use Volume Delta as a proxy:
- Buy Volume: Volume on bullish candles (close > open)
- Sell Volume: Volume on bearish candles (close < open)
- Imbalance = (Buy - Sell) / Total over N candles
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
    """Single trade result."""
    signal_type: str
    entry_price: float
    stop_loss: float
    target: float
    entry_idx: int
    exit_idx: int
    exit_price: float
    pnl_r: float
    won: bool
    vol_ratio: float
    tai_index: float
    imbalance: float


def calculate_ewvma(closes: np.ndarray, volumes: np.ndarray, length: int) -> np.ndarray:
    """Calculate EWVMA."""
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
    """Calculate Tai Index (Stochastic RSI)."""
    n = len(closes)
    tai = np.full(n, 50.0)

    if n < rsi_length + stoch_length:
        return tai

    # RSI
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

    # Stochastic on RSI
    for i in range(stoch_length - 1, n):
        rsi_window = rsi[i - stoch_length + 1:i + 1]
        rsi_min = np.min(rsi_window)
        rsi_max = np.max(rsi_window)
        if rsi_max - rsi_min > 0:
            tai[i] = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
        else:
            tai[i] = 50

    return tai


def calculate_volume_ratio(volumes: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio."""
    n = len(volumes)
    vol_ratio = np.ones(n)

    for i in range(lookback - 1, n):
        vol_sma = np.mean(volumes[i - lookback + 1:i + 1])
        if vol_sma > 0:
            vol_ratio[i] = volumes[i] / vol_sma

    return vol_ratio


def calculate_volume_delta_imbalance(opens: np.ndarray, closes: np.ndarray,
                                      volumes: np.ndarray, lookback: int = 10) -> np.ndarray:
    """Calculate order book imbalance proxy using volume delta."""
    n = len(closes)
    imbalance = np.zeros(n)

    is_bullish = closes > opens
    buy_volume = np.where(is_bullish, volumes, 0)
    sell_volume = np.where(~is_bullish, volumes, 0)

    # Rolling sum
    buy_cumsum = np.cumsum(buy_volume)
    sell_cumsum = np.cumsum(sell_volume)

    for i in range(lookback, n):
        buy_sum = buy_cumsum[i] - buy_cumsum[i - lookback]
        sell_sum = sell_cumsum[i] - sell_cumsum[i - lookback]
        total = buy_sum + sell_sum
        if total > 0:
            imbalance[i] = (buy_sum - sell_sum) / total

    return imbalance


def detect_bearish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
    """Detect bearish FVG (gap down)."""
    if idx < 2:
        return None
    if highs[idx] < lows[idx - 2]:
        return (lows[idx - 2], highs[idx])  # top, bottom
    return None


def detect_bullish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
    """Detect bullish FVG (gap up)."""
    if idx < 2:
        return None
    if lows[idx] > highs[idx - 2]:
        return (lows[idx], highs[idx - 2])  # top, bottom
    return None


def run_optimized_backtest(
    df: pd.DataFrame,
    symbol: str,
    imbalance_threshold: Optional[float] = None,
    min_vol_ratio: float = 2.0,
    tai_short: float = 53.0,
    tai_long: float = 47.0,
    imbalance_lookback: int = 10,
) -> Tuple[List[TradeResult], Dict]:
    """Run optimized breakaway backtest."""

    # Extract arrays
    opens = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    n = len(df)

    # Pre-calculate all indicators once
    ewvma_20 = calculate_ewvma(closes, volumes, 20)
    ewvma_20_std = pd.Series(closes).rolling(20).std().fillna(0).values
    ewvma_200 = calculate_ewvma(closes, volumes, 200)
    tai_index = calculate_tai_index(closes)
    vol_ratio = calculate_volume_ratio(volumes)
    imbalance = calculate_volume_delta_imbalance(opens, closes, volumes, imbalance_lookback)

    # Cradle detection
    upper_band = ewvma_20 + ewvma_20_std
    lower_band = ewvma_20 - ewvma_20_std
    in_cradle = (closes >= lower_band) & (closes <= upper_band)

    trades = []
    signals_generated = 0
    signals_filtered = 0
    last_signal_idx = -10  # Prevent double signals

    for i in range(300, n - 50):
        if i <= last_signal_idx + 1:
            continue

        # Check cradle (3+ of last 5 candles in band)
        if i < 5:
            continue
        cradle_count = np.sum(in_cradle[i-5:i])
        if cradle_count < 3:
            continue

        # Volume spike
        if vol_ratio[i] < min_vol_ratio:
            continue

        signal_type = None
        fvg = None

        # Check for SHORT signal
        fvg_bear = detect_bearish_fvg(highs, lows, i)
        if fvg_bear and tai_index[i] > tai_short and closes[i] > ewvma_200[i]:
            signal_type = 'short'
            fvg = fvg_bear

        # Check for LONG signal
        if signal_type is None:
            fvg_bull = detect_bullish_fvg(highs, lows, i)
            if fvg_bull and tai_index[i] < tai_long and closes[i] < ewvma_200[i]:
                signal_type = 'long'
                fvg = fvg_bull

        if signal_type is None:
            continue

        signals_generated += 1
        current_imbalance = imbalance[i]

        # Apply imbalance filter
        if imbalance_threshold is not None:
            if signal_type == 'short':
                if current_imbalance > -imbalance_threshold:
                    signals_filtered += 1
                    continue
            else:
                if current_imbalance < imbalance_threshold:
                    signals_filtered += 1
                    continue

        last_signal_idx = i

        # Calculate trade levels
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

        # Simulate trade
        exit_idx = i + 1
        exit_price = None
        won = None

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

        # P&L in R
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
            won=won,
            vol_ratio=vol_ratio[i],
            tai_index=tai_index[i],
            imbalance=current_imbalance
        ))

    # Stats
    stats = calculate_stats(trades, signals_generated, signals_filtered)
    return trades, stats


def calculate_stats(trades: List[TradeResult], signals_generated: int, signals_filtered: int) -> Dict:
    """Calculate trading statistics."""
    if not trades:
        return {'total_trades': 0, 'signals_generated': signals_generated,
                'signals_filtered': signals_filtered, 'win_rate': 0, 'avg_r': 0,
                'total_r': 0, 'profit_factor': 0, 'max_drawdown_r': 0,
                'shorts': 0, 'longs': 0, 'short_wr': 0, 'long_wr': 0}

    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    shorts = [t for t in trades if t.signal_type == 'short']
    longs = [t for t in trades if t.signal_type == 'long']
    short_wins = sum(1 for t in shorts if t.won)
    long_wins = sum(1 for t in longs if t.won)

    gross_profit = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))

    equity = np.cumsum([t.pnl_r for t in trades])
    running_max = np.maximum.accumulate(equity)
    max_dd = np.max(running_max - equity) if len(equity) > 0 else 0

    return {
        'total_trades': n,
        'signals_generated': signals_generated,
        'signals_filtered': signals_filtered,
        'filter_rate': signals_filtered / signals_generated * 100 if signals_generated > 0 else 0,
        'win_rate': wins / n * 100,
        'avg_r': np.mean([t.pnl_r for t in trades]),
        'total_r': sum(t.pnl_r for t in trades),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'max_drawdown_r': max_dd,
        'shorts': len(shorts),
        'longs': len(longs),
        'short_wr': short_wins / len(shorts) * 100 if shorts else 0,
        'long_wr': long_wins / len(longs) * 100 if longs else 0,
    }


def main():
    """Run order book imbalance analysis."""

    data_dir = Path("/home/tahae/ai-content/data/Tradingdata/volume charts")

    test_files = [
        ("BTCUSDT", "BTCUSDT_5m_merged.csv"),
        ("DOTUSDT", "DOTUSDT_5m_merged.csv"),
        ("PNUTUSDT", "PNUTUSDT_5m_merged.csv"),
    ]

    thresholds = [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("=" * 80)
    print("ORDER BOOK IMBALANCE ANALYSIS - BREAKAWAY STRATEGY")
    print("=" * 80)
    print("\nProxy Method: Volume Delta Imbalance")
    print("  - Bullish candle volume = buy pressure")
    print("  - Bearish candle volume = sell pressure")
    print("  - Imbalance = (buy - sell) / total [range: -1 to +1]")
    print("\nFilter Logic:")
    print("  - SHORTS: require imbalance < -threshold (selling pressure)")
    print("  - LONGS: require imbalance > +threshold (buying pressure)")
    print("=" * 80)

    all_results = {}

    for symbol, filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"\n[SKIP] {filename} not found")
            continue

        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        df = pd.read_csv(filepath)
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"Loaded {len(df):,} candles")

        # Baseline analysis
        trades_base, _ = run_optimized_backtest(df, symbol, imbalance_threshold=None)

        if trades_base:
            print("\n--- Imbalance at Signal Points ---")
            shorts = [t for t in trades_base if t.signal_type == 'short']
            longs = [t for t in trades_base if t.signal_type == 'long']

            short_winners = [t for t in shorts if t.won]
            short_losers = [t for t in shorts if not t.won]
            long_winners = [t for t in longs if t.won]
            long_losers = [t for t in longs if not t.won]

            print(f"\n  SHORT trades ({len(shorts)} total):")
            if shorts:
                print(f"    Avg imbalance (all):    {np.mean([t.imbalance for t in shorts]):+.3f}")
            if short_winners:
                print(f"    Avg imbalance (wins):   {np.mean([t.imbalance for t in short_winners]):+.3f}")
            if short_losers:
                print(f"    Avg imbalance (losses): {np.mean([t.imbalance for t in short_losers]):+.3f}")

            print(f"\n  LONG trades ({len(longs)} total):")
            if longs:
                print(f"    Avg imbalance (all):    {np.mean([t.imbalance for t in longs]):+.3f}")
            if long_winners:
                print(f"    Avg imbalance (wins):   {np.mean([t.imbalance for t in long_winners]):+.3f}")
            if long_losers:
                print(f"    Avg imbalance (losses): {np.mean([t.imbalance for t in long_losers]):+.3f}")

        # Test thresholds
        print(f"\n--- Threshold Comparison ---")
        print(f"{'Thresh':>8} | {'Trades':>6} | {'Filt%':>6} | {'Win%':>6} | {'AvgR':>7} | {'TotalR':>8} | {'PF':>5} | {'MaxDD':>6}")
        print("-" * 75)

        symbol_results = []

        for thresh in thresholds:
            trades, stats = run_optimized_backtest(df, symbol, imbalance_threshold=thresh)

            thresh_str = "None" if thresh is None else f"{thresh:.2f}"
            filt_pct = stats['filter_rate']
            print(f"{thresh_str:>8} | {stats['total_trades']:>6} | {filt_pct:>5.1f}% | "
                  f"{stats['win_rate']:>5.1f}% | {stats['avg_r']:>+6.2f}R | {stats['total_r']:>+7.1f}R | "
                  f"{stats['profit_factor']:>5.2f} | {stats['max_drawdown_r']:>5.1f}R")

            symbol_results.append({'threshold': thresh, 'stats': stats, 'trades': trades})

        all_results[symbol] = symbol_results

    # Combined summary
    print("\n" + "=" * 80)
    print("COMBINED RESULTS (ALL SYMBOLS)")
    print("=" * 80)

    print(f"\n{'Thresh':>8} | {'Trades':>6} | {'Filt%':>6} | {'Win%':>6} | {'AvgR':>7} | {'TotalR':>8} | {'PF':>5}")
    print("-" * 70)

    best_thresh = None
    best_expectancy = -999
    baseline_expectancy = 0

    for thresh in thresholds:
        total_trades = 0
        total_filtered = 0
        total_generated = 0
        total_wins = 0
        total_r = 0
        gross_profit = 0
        gross_loss = 0

        for symbol, results in all_results.items():
            for r in results:
                if r['threshold'] == thresh:
                    trades = r['trades']
                    stats = r['stats']
                    total_trades += stats['total_trades']
                    total_filtered += stats['signals_filtered']
                    total_generated += stats['signals_generated']
                    total_wins += sum(1 for t in trades if t.won)
                    total_r += stats['total_r']
                    gross_profit += sum(t.pnl_r for t in trades if t.pnl_r > 0)
                    gross_loss += abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))

        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        avg_r = total_r / total_trades if total_trades > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        filt_pct = total_filtered / total_generated * 100 if total_generated > 0 else 0

        thresh_str = "None" if thresh is None else f"{thresh:.2f}"
        print(f"{thresh_str:>8} | {total_trades:>6} | {filt_pct:>5.1f}% | "
              f"{win_rate:>5.1f}% | {avg_r:>+6.2f}R | {total_r:>+7.1f}R | {pf:>5.2f}")

        if thresh is None:
            baseline_expectancy = avg_r

        if avg_r > best_expectancy and total_trades >= 30:
            best_expectancy = avg_r
            best_thresh = thresh

    # Recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nBaseline (no filter): {baseline_expectancy:+.3f}R per trade")

    if best_thresh is not None and best_thresh != thresholds[0]:
        improvement = ((best_expectancy - baseline_expectancy) / abs(baseline_expectancy) * 100
                      if baseline_expectancy != 0 else 0)
        print(f"\nBest threshold: {best_thresh:.2f}")
        print(f"Best expectancy: {best_expectancy:+.3f}R per trade")
        print(f"Improvement: {improvement:+.1f}%")
        print(f"\n*** RECOMMENDATION: Add imbalance filter at {best_thresh:.2f} ***")
        print(f"  - For SHORTS: only enter when imbalance < -{best_thresh:.2f}")
        print(f"  - For LONGS: only enter when imbalance > +{best_thresh:.2f}")
    else:
        print("\n*** RECOMMENDATION: Order book imbalance filter does NOT improve results ***")
        print("The base strategy's volume spike + Tai Index already captures the edge.")


if __name__ == "__main__":
    main()

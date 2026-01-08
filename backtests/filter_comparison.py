#!/usr/bin/env python3
"""
Compare: No filter vs 300 SMA (80%) vs 50 EMA at different thresholds
"""

import pandas as pd
import sys
from king_backtest import (
    load_data, find_swing_points, find_long_king_patterns, find_short_king_patterns,
    simulate_trades, check_ema50_filter, check_trend_filter
)

DATA_FILES = {
    'BTC_1m': '/home/tahae/ai-content/data/Tradingdata/BYBIT_BTCUSDT.P, 1_da562.csv',
    'ETH_1m': '/home/tahae/ai-content/data/Tradingdata/BYBIT_ETHUSDT.P, 1_79d61.csv',
    'BTC_5m': '/home/tahae/ai-content/data/Tradingdata/MEXC_BTCUSDT, 5_64902.csv',
    'ETH_5m': '/home/tahae/ai-content/data/Tradingdata/MEXC_ETHUSDT.P, 5_b72c6.csv',
    'DOGE_5m': '/home/tahae/ai-content/data/Tradingdata/MEXC_DOGEUSDT.P, 5_3ec12.csv',
}


def calculate_metrics(trades):
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_r': 0, 'max_dd': 0}

    wins = sum(1 for t in trades if t.result == 'win')
    total = sum(1 for t in trades if t.result in ['win', 'loss'])

    total_r = 0
    peak = 0
    max_dd = 0

    for t in trades:
        if t.result == 'win':
            total_r += t.risk_reward if t.risk_reward else 1.0
        elif t.result == 'loss':
            total_r -= 1.0
        if total_r > peak:
            peak = total_r
        dd = peak - total_r
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': total,
        'win_rate': wins / total * 100 if total > 0 else 0,
        'total_r': total_r,
        'max_dd': max_dd
    }


def filter_patterns(df, longs, shorts, filter_type, threshold=0.8):
    """Apply filter to patterns."""
    filtered_longs = []
    filtered_shorts = []

    for p in longs:
        if filter_type == 'none':
            filtered_longs.append(p)
        elif filter_type == 'sma300':
            if check_trend_filter(df, p['a'].index, p['g_idx'], 'long', threshold):
                filtered_longs.append(p)
        elif filter_type == 'ema50':
            if check_ema50_filter(df, p['a'].index, p['g_idx'], 'long', threshold):
                filtered_longs.append(p)

    for p in shorts:
        if filter_type == 'none':
            filtered_shorts.append(p)
        elif filter_type == 'sma300':
            if check_trend_filter(df, p['a'].index, p['g_idx'], 'short', threshold):
                filtered_shorts.append(p)
        elif filter_type == 'ema50':
            if check_ema50_filter(df, p['a'].index, p['g_idx'], 'short', threshold):
                filtered_shorts.append(p)

    return filtered_longs, filtered_shorts


def main():
    results = []

    files_to_test = DATA_FILES
    if len(sys.argv) > 1:
        tf = sys.argv[1]
        files_to_test = {k: v for k, v in DATA_FILES.items() if tf in k}

    for name, filepath in files_to_test.items():
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)

        try:
            df = load_data(filepath)
            swings = find_swing_points(df, lookback=3)
            symbol = name.split('_')[0]

            longs = find_long_king_patterns(df, swings, symbol=symbol, filter_type='none')
            shorts = find_short_king_patterns(df, swings, symbol=symbol, filter_type='none')

            print(f"Patterns: {len(longs)} long, {len(shorts)} short")

            # Test different filters
            tests = [
                ('none', 'none', 0.8),
                ('sma300_80', 'sma300', 0.8),
                ('ema50_60', 'ema50', 0.6),
                ('ema50_70', 'ema50', 0.7),
                ('ema50_80', 'ema50', 0.8),
            ]

            for label, ftype, thresh in tests:
                fl, fs = filter_patterns(df, longs, shorts, ftype, thresh)
                trades = simulate_trades(df, fl + fs)
                m = calculate_metrics(trades)
                results.append({'asset': name, 'filter': label, **m})
                print(f"  {label:<12} | {m['trades']:3} trades | {m['win_rate']:5.1f}% | {m['total_r']:+8.2f}R")

        except Exception as e:
            print(f"Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("BEST FILTER PER ASSET")
    print("="*70)
    for asset in sorted(set(r['asset'] for r in results)):
        ar = [r for r in results if r['asset'] == asset]
        best = max(ar, key=lambda x: x['total_r'])
        print(f"{asset:<12}: {best['filter']:<12} ({best['total_r']:+.2f}R, {best['trades']} trades)")


if __name__ == "__main__":
    main()

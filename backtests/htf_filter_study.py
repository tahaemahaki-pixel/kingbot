"""
HTF Filter Study - Test 4H 50 EMA directional filter on Double Touch signals
"""
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_ema(prices: pd.Series, length: int) -> pd.Series:
    """Calculate EMA."""
    return prices.ewm(span=length, adjust=False).mean()

def get_band_color(ema9: float, ema21: float, ema50: float) -> str:
    """Determine EMA ribbon band color."""
    if ema9 > ema21 > ema50:
        return 'green'
    elif ema9 < ema21 < ema50:
        return 'red'
    else:
        return 'grey'

def resample_to_htf(df: pd.DataFrame, htf_minutes: int = 240) -> pd.DataFrame:
    """
    Resample lower timeframe data to higher timeframe.
    Assumes df has a 'timestamp' column and OHLC data.
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)

    htf = df.resample(f'{htf_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    htf['ema50'] = calculate_ema(htf['close'], 50)
    htf.reset_index(inplace=True)

    return htf

def get_htf_bias(timestamp, htf_df: pd.DataFrame) -> str:
    """
    Get HTF bias at a given timestamp.
    Returns 'long' if price > 4H 50 EMA, 'short' if price < 4H 50 EMA, 'neutral' otherwise.
    """
    # Find the most recent completed HTF candle
    mask = htf_df['timestamp'] <= timestamp
    if not mask.any():
        return 'neutral'

    htf_candle = htf_df[mask].iloc[-1]

    if pd.isna(htf_candle['ema50']):
        return 'neutral'

    if htf_candle['close'] > htf_candle['ema50']:
        return 'long'
    elif htf_candle['close'] < htf_candle['ema50']:
        return 'short'
    else:
        return 'neutral'

def detect_signals_with_htf(df: pd.DataFrame, htf_df: pd.DataFrame) -> list:
    """
    Detect Double Touch signals with HTF bias information.
    """
    signals = []

    # Calculate LTF indicators
    df = df.copy()
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['band'] = df.apply(lambda r: get_band_color(r['ema9'], r['ema21'], r['ema50']), axis=1)

    # Find HH/LL
    lookback = 20
    df['hh'] = df['high'].rolling(lookback).max() == df['high']
    df['ll'] = df['low'].rolling(lookback).min() == df['low']

    n = len(df)

    for i in range(100, n - 50):
        timestamp = df['timestamp'].iloc[i]

        # LONG pattern: HH on green -> grey -> green -> grey -> green
        if df['hh'].iloc[i] and df['band'].iloc[i] == 'green':
            step0_idx = i

            for j in range(i+1, min(i+30, n)):
                if df['band'].iloc[j] == 'grey':
                    for k in range(j+1, min(j+30, n)):
                        if df['band'].iloc[k] == 'green':
                            for l in range(k+1, min(k+30, n)):
                                if df['band'].iloc[l] == 'grey':
                                    step3_low = df['low'].iloc[k:l+1].min()

                                    for m in range(l+1, min(l+20, n)):
                                        if df['band'].iloc[m] == 'green':
                                            entry_idx = m
                                            entry_price = df['close'].iloc[m]
                                            entry_time = df['timestamp'].iloc[m]
                                            sl = step3_low * 0.999
                                            tp = entry_price + 3 * (entry_price - sl)

                                            # Get HTF bias at entry
                                            htf_bias = get_htf_bias(entry_time, htf_df)

                                            # Calculate outcome
                                            hit_tp = False
                                            hit_sl = False
                                            for x in range(m+1, min(m+100, n)):
                                                if df['high'].iloc[x] >= tp:
                                                    hit_tp = True
                                                    break
                                                if df['low'].iloc[x] <= sl:
                                                    hit_sl = True
                                                    break

                                            signals.append({
                                                'type': 'long',
                                                'entry_idx': entry_idx,
                                                'htf_bias': htf_bias,
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open')
                                            })
                                            break
                                    break
                            break
                    break

        # SHORT pattern: LL on red -> grey -> red -> grey -> red
        if df['ll'].iloc[i] and df['band'].iloc[i] == 'red':
            step0_idx = i

            for j in range(i+1, min(i+30, n)):
                if df['band'].iloc[j] == 'grey':
                    for k in range(j+1, min(j+30, n)):
                        if df['band'].iloc[k] == 'red':
                            for l in range(k+1, min(k+30, n)):
                                if df['band'].iloc[l] == 'grey':
                                    step3_high = df['high'].iloc[k:l+1].max()

                                    for m in range(l+1, min(l+20, n)):
                                        if df['band'].iloc[m] == 'red':
                                            entry_idx = m
                                            entry_price = df['close'].iloc[m]
                                            entry_time = df['timestamp'].iloc[m]
                                            sl = step3_high * 1.001
                                            tp = entry_price - 3 * (sl - entry_price)

                                            # Get HTF bias at entry
                                            htf_bias = get_htf_bias(entry_time, htf_df)

                                            # Calculate outcome
                                            hit_tp = False
                                            hit_sl = False
                                            for x in range(m+1, min(m+100, n)):
                                                if df['low'].iloc[x] <= tp:
                                                    hit_tp = True
                                                    break
                                                if df['high'].iloc[x] >= sl:
                                                    hit_sl = True
                                                    break

                                            signals.append({
                                                'type': 'short',
                                                'entry_idx': entry_idx,
                                                'htf_bias': htf_bias,
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open')
                                            })
                                            break
                                    break
                            break
                    break

    return signals

def analyze_htf_filter(signals: list) -> dict:
    """Analyze impact of HTF filter."""
    results = {
        'no_filter': {'wins': 0, 'losses': 0, 'total': 0},
        'htf_aligned': {'wins': 0, 'losses': 0, 'total': 0},  # Long when HTF=long, Short when HTF=short
        'htf_counter': {'wins': 0, 'losses': 0, 'total': 0},  # Long when HTF=short, Short when HTF=long
        'by_type': {
            'long_htf_long': {'wins': 0, 'losses': 0},
            'long_htf_short': {'wins': 0, 'losses': 0},
            'long_htf_neutral': {'wins': 0, 'losses': 0},
            'short_htf_long': {'wins': 0, 'losses': 0},
            'short_htf_short': {'wins': 0, 'losses': 0},
            'short_htf_neutral': {'wins': 0, 'losses': 0},
        }
    }

    for sig in signals:
        if sig['outcome'] == 'open':
            continue

        is_win = sig['outcome'] == 'win'
        sig_type = sig['type']
        htf_bias = sig['htf_bias']

        # No filter
        results['no_filter']['total'] += 1
        if is_win:
            results['no_filter']['wins'] += 1
        else:
            results['no_filter']['losses'] += 1

        # HTF aligned (trade with HTF trend)
        if (sig_type == 'long' and htf_bias == 'long') or (sig_type == 'short' and htf_bias == 'short'):
            results['htf_aligned']['total'] += 1
            if is_win:
                results['htf_aligned']['wins'] += 1
            else:
                results['htf_aligned']['losses'] += 1

        # HTF counter (trade against HTF trend)
        if (sig_type == 'long' and htf_bias == 'short') or (sig_type == 'short' and htf_bias == 'long'):
            results['htf_counter']['total'] += 1
            if is_win:
                results['htf_counter']['wins'] += 1
            else:
                results['htf_counter']['losses'] += 1

        # Detailed breakdown
        key = f"{sig_type}_htf_{htf_bias}"
        if key in results['by_type']:
            if is_win:
                results['by_type'][key]['wins'] += 1
            else:
                results['by_type'][key]['losses'] += 1

    return results

def print_results(results: dict):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("HTF 4H 50 EMA FILTER ANALYSIS")
    print("=" * 70)

    for filter_name in ['no_filter', 'htf_aligned', 'htf_counter']:
        stats = results[filter_name]
        total = stats['total']
        wins = stats['wins']
        losses = stats['losses']

        if total == 0:
            print(f"\n{filter_name}: No trades")
            continue

        win_rate = wins / total * 100
        pf = (wins * 3) / losses if losses > 0 else float('inf')
        expectancy = (win_rate/100 * 3) - ((100-win_rate)/100 * 1)

        print(f"\n{filter_name.upper().replace('_', ' ')}:")
        print(f"  Trades: {total:,}")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Expectancy: {expectancy:+.2f}R")

    print("\n" + "-" * 70)
    print("DETAILED BREAKDOWN BY SIGNAL TYPE + HTF BIAS:")
    print("-" * 70)

    for key, stats in results['by_type'].items():
        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses

        if total == 0:
            continue

        win_rate = wins / total * 100
        pf = (wins * 3) / losses if losses > 0 else float('inf')
        expectancy = (win_rate/100 * 3) - ((100-win_rate)/100 * 1)

        print(f"  {key:25} | Trades: {total:5} | WR: {win_rate:5.1f}% | PF: {pf:5.2f} | Exp: {expectancy:+.2f}R")

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()

    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'], utc=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], utc=True)

    return df

def main():
    data_dir = Path("/home/tahae/ai-content/data/Tradingdata")

    # Focus on 5-minute data (can resample to 4H)
    csv_files = list(data_dir.glob("chart_data/*5*.csv"))

    print(f"Found {len(csv_files)} 5-minute data files")

    all_signals = []

    for csv_file in csv_files:
        try:
            df = load_data(str(csv_file))

            required = ['open', 'high', 'low', 'close', 'timestamp']
            if not all(col in df.columns for col in required):
                continue

            # Need at least 5000 candles for meaningful HTF (4H = 48 x 5min candles)
            if len(df) < 5000:
                print(f"Skipping {csv_file.name}: too few candles ({len(df)})")
                continue

            print(f"\nProcessing: {csv_file.name} ({len(df):,} candles)")

            # Create HTF dataframe
            htf_df = resample_to_htf(df, htf_minutes=240)  # 4H
            print(f"  HTF candles: {len(htf_df)}")

            # Detect signals with HTF info
            signals = detect_signals_with_htf(df, htf_df)
            print(f"  Signals found: {len(signals)}")

            all_signals.extend(signals)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if all_signals:
        print(f"\n\nTotal signals across all files: {len(all_signals)}")
        results = analyze_htf_filter(all_signals)
        print_results(results)

        # Calculate improvement
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        no_filter_wr = results['no_filter']['wins'] / results['no_filter']['total'] * 100 if results['no_filter']['total'] > 0 else 0
        aligned_wr = results['htf_aligned']['wins'] / results['htf_aligned']['total'] * 100 if results['htf_aligned']['total'] > 0 else 0

        print(f"\nBaseline Win Rate: {no_filter_wr:.1f}%")
        print(f"HTF Aligned Win Rate: {aligned_wr:.1f}%")
        print(f"Improvement: {aligned_wr - no_filter_wr:+.1f}%")

        trades_filtered = results['no_filter']['total'] - results['htf_aligned']['total']
        pct_filtered = trades_filtered / results['no_filter']['total'] * 100 if results['no_filter']['total'] > 0 else 0
        print(f"\nTrades filtered out: {trades_filtered:,} ({pct_filtered:.1f}%)")
    else:
        print("\nNo signals detected.")

if __name__ == "__main__":
    main()

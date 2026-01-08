"""
Tai Index Study - Analyze relationship between Tai Index and Double Touch signals
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def calculate_rsi(prices: pd.Series, length: int = 100) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()

    # Use Wilder's smoothing after initial SMA
    for i in range(length, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length - 1) + gain.iloc[i]) / length
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length - 1) + loss.iloc[i]) / length

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(src: pd.Series, length: int = 200) -> pd.Series:
    """Calculate Stochastic of a series (where high=low=src)."""
    lowest = src.rolling(window=length, min_periods=length).min()
    highest = src.rolling(window=length, min_periods=length).max()

    stoch = 100 * (src - lowest) / (highest - lowest)
    return stoch

def calculate_tai_index(close: pd.Series, rsi_length: int = 100, stoch_length: int = 200) -> pd.Series:
    """
    Calculate Tai Index (Stochastic RSI with slow parameters).

    Tai Index = SMA(Stochastic(RSI, RSI, RSI, stoch_length), 1)
    Since smoothK=1, it's just the raw stochastic of RSI.
    """
    rsi = calculate_rsi(close, rsi_length)
    tai = calculate_stochastic(rsi, stoch_length)
    return tai

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

def detect_double_touch_signals(df: pd.DataFrame) -> list:
    """
    Simplified Double Touch detection for study purposes.
    Returns list of signal dictionaries with entry index and tai_index value.
    """
    signals = []

    # Calculate indicators
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['tai'] = calculate_tai_index(df['close'])

    # Get band colors
    df['band'] = df.apply(lambda r: get_band_color(r['ema9'], r['ema21'], r['ema50']), axis=1)

    # Find HH/LL
    lookback = 20
    df['hh'] = df['high'].rolling(lookback).max() == df['high']
    df['ll'] = df['low'].rolling(lookback).min() == df['low']

    # State machine for pattern detection
    n = len(df)

    for i in range(300, n - 50):  # Need history for Tai Index
        # Long pattern: HH on green -> grey -> green -> grey -> green
        # Check for Step 0: HH on green band
        if df['hh'].iloc[i] and df['band'].iloc[i] == 'green':
            step0_idx = i
            step0_price = df['high'].iloc[i]

            # Look for Step 1: grey band
            for j in range(i+1, min(i+30, n)):
                if df['band'].iloc[j] == 'grey':
                    step1_idx = j

                    # Look for Step 2: green band
                    for k in range(j+1, min(j+30, n)):
                        if df['band'].iloc[k] == 'green':
                            step2_idx = k

                            # Look for Step 3: grey band (defines SL)
                            for l in range(k+1, min(k+30, n)):
                                if df['band'].iloc[l] == 'grey':
                                    step3_idx = l
                                    step3_low = df['low'].iloc[k:l+1].min()

                                    # Look for Step 4: green band (entry)
                                    for m in range(l+1, min(l+20, n)):
                                        if df['band'].iloc[m] == 'green':
                                            entry_idx = m
                                            entry_price = df['close'].iloc[m]
                                            sl = step3_low * 0.999
                                            tp = entry_price + 3 * (entry_price - sl)

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
                                                'entry_price': entry_price,
                                                'sl': sl,
                                                'tp': tp,
                                                'tai_at_step0': df['tai'].iloc[step0_idx],
                                                'tai_at_entry': df['tai'].iloc[entry_idx],
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open'),
                                                'step0_idx': step0_idx
                                            })
                                            break
                                    break
                            break
                    break

        # Short pattern: LL on red -> grey -> red -> grey -> red
        if df['ll'].iloc[i] and df['band'].iloc[i] == 'red':
            step0_idx = i
            step0_price = df['low'].iloc[i]

            # Look for Step 1: grey band
            for j in range(i+1, min(i+30, n)):
                if df['band'].iloc[j] == 'grey':
                    step1_idx = j

                    # Look for Step 2: red band
                    for k in range(j+1, min(j+30, n)):
                        if df['band'].iloc[k] == 'red':
                            step2_idx = k

                            # Look for Step 3: grey band (defines SL)
                            for l in range(k+1, min(k+30, n)):
                                if df['band'].iloc[l] == 'grey':
                                    step3_idx = l
                                    step3_high = df['high'].iloc[k:l+1].max()

                                    # Look for Step 4: red band (entry)
                                    for m in range(l+1, min(l+20, n)):
                                        if df['band'].iloc[m] == 'red':
                                            entry_idx = m
                                            entry_price = df['close'].iloc[m]
                                            sl = step3_high * 1.001
                                            tp = entry_price - 3 * (sl - entry_price)

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
                                                'entry_price': entry_price,
                                                'sl': sl,
                                                'tp': tp,
                                                'tai_at_step0': df['tai'].iloc[step0_idx],
                                                'tai_at_entry': df['tai'].iloc[entry_idx],
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open'),
                                                'step0_idx': step0_idx
                                            })
                                            break
                                    break
                            break
                    break

    return signals

def analyze_tai_filter(signals: list) -> dict:
    """Analyze how Tai Index filtering would affect results."""
    results = {
        'total': len(signals),
        'no_filter': {'wins': 0, 'losses': 0},
        'tai_counter_trend': {'wins': 0, 'losses': 0},  # Longs when Tai < 45, Shorts when Tai > 55
        'tai_extreme': {'wins': 0, 'losses': 0},  # Longs when Tai < 35, Shorts when Tai > 75
        'tai_with_trend': {'wins': 0, 'losses': 0},  # Longs when Tai > 55, Shorts when Tai < 45
    }

    for sig in signals:
        if sig['outcome'] == 'open':
            continue

        is_win = sig['outcome'] == 'win'
        tai_step0 = sig['tai_at_step0']
        tai_entry = sig['tai_at_entry']

        # No filter
        if is_win:
            results['no_filter']['wins'] += 1
        else:
            results['no_filter']['losses'] += 1

        # Counter-trend filter (trade against momentum)
        if sig['type'] == 'long' and tai_step0 < 45:
            if is_win:
                results['tai_counter_trend']['wins'] += 1
            else:
                results['tai_counter_trend']['losses'] += 1
        elif sig['type'] == 'short' and tai_step0 > 55:
            if is_win:
                results['tai_counter_trend']['wins'] += 1
            else:
                results['tai_counter_trend']['losses'] += 1

        # Extreme counter-trend filter
        if sig['type'] == 'long' and tai_step0 < 35:
            if is_win:
                results['tai_extreme']['wins'] += 1
            else:
                results['tai_extreme']['losses'] += 1
        elif sig['type'] == 'short' and tai_step0 > 75:
            if is_win:
                results['tai_extreme']['wins'] += 1
            else:
                results['tai_extreme']['losses'] += 1

        # With-trend filter (trade with momentum)
        if sig['type'] == 'long' and tai_step0 > 55:
            if is_win:
                results['tai_with_trend']['wins'] += 1
            else:
                results['tai_with_trend']['losses'] += 1
        elif sig['type'] == 'short' and tai_step0 < 45:
            if is_win:
                results['tai_with_trend']['wins'] += 1
            else:
                results['tai_with_trend']['losses'] += 1

    return results

def print_results(results: dict):
    """Print analysis results."""
    print("\n" + "=" * 60)
    print("TAI INDEX FILTER ANALYSIS")
    print("=" * 60)
    print(f"Total signals detected: {results['total']}")
    print()

    for filter_name, stats in results.items():
        if filter_name == 'total':
            continue

        wins = stats['wins']
        losses = stats['losses']
        total = wins + losses

        if total == 0:
            print(f"{filter_name}: No trades")
            continue

        win_rate = wins / total * 100
        # With 3:1 R:R, profit factor = (wins * 3) / losses
        pf = (wins * 3) / losses if losses > 0 else float('inf')
        expectancy = (win_rate/100 * 3) - ((100-win_rate)/100 * 1)

        print(f"{filter_name}:")
        print(f"  Trades: {total} | Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {win_rate:.1f}% | Profit Factor: {pf:.2f} | Expectancy: {expectancy:.2f}R")
        print()

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Handle different column naming conventions
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])

    return df

def main():
    # Find data files
    data_dir = Path("/home/tahae/ai-content/data/Tradingdata")

    # Look for CSV files with price data in chart_data and Strategy 1 folders
    csv_files = (
        list(data_dir.glob("chart_data/*.csv")) +
        list(data_dir.glob("Strategy 1/*.csv")) +
        list(data_dir.glob("doubletouch/*.csv"))
    )

    print(f"Found {len(csv_files)} data files")

    all_signals = []

    # Process each file
    for csv_file in csv_files:
        try:
            df = load_data(str(csv_file))

            # Check if it has required columns
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                continue

            print(f"\nProcessing: {csv_file.name} ({len(df)} candles)")

            signals = detect_double_touch_signals(df)
            print(f"  Found {len(signals)} signals")

            for sig in signals:
                sig['file'] = csv_file.name

            all_signals.extend(signals)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if all_signals:
        results = analyze_tai_filter(all_signals)
        print_results(results)

        # Additional analysis: Tai distribution for wins vs losses
        print("\n" + "=" * 60)
        print("TAI INDEX DISTRIBUTION ANALYSIS")
        print("=" * 60)

        wins = [s for s in all_signals if s['outcome'] == 'win']
        losses = [s for s in all_signals if s['outcome'] == 'loss']

        if wins:
            win_tai = [s['tai_at_step0'] for s in wins if not np.isnan(s['tai_at_step0'])]
            print(f"\nWinning trades Tai @ Step0:")
            print(f"  Mean: {np.mean(win_tai):.1f} | Median: {np.median(win_tai):.1f}")
            print(f"  Min: {np.min(win_tai):.1f} | Max: {np.max(win_tai):.1f}")

        if losses:
            loss_tai = [s['tai_at_step0'] for s in losses if not np.isnan(s['tai_at_step0'])]
            print(f"\nLosing trades Tai @ Step0:")
            print(f"  Mean: {np.mean(loss_tai):.1f} | Median: {np.median(loss_tai):.1f}")
            print(f"  Min: {np.min(loss_tai):.1f} | Max: {np.max(loss_tai):.1f}")

        # By signal type
        for sig_type in ['long', 'short']:
            type_signals = [s for s in all_signals if s['type'] == sig_type and s['outcome'] != 'open']
            if type_signals:
                type_wins = [s for s in type_signals if s['outcome'] == 'win']
                type_losses = [s for s in type_signals if s['outcome'] == 'loss']

                print(f"\n{sig_type.upper()} signals:")
                print(f"  Total: {len(type_signals)} | Wins: {len(type_wins)} | Losses: {len(type_losses)}")

                if type_wins:
                    win_tai = [s['tai_at_step0'] for s in type_wins if not np.isnan(s['tai_at_step0'])]
                    print(f"  Winners Tai mean: {np.mean(win_tai):.1f}")
                if type_losses:
                    loss_tai = [s['tai_at_step0'] for s in type_losses if not np.isnan(s['tai_at_step0'])]
                    print(f"  Losers Tai mean: {np.mean(loss_tai):.1f}")
    else:
        print("\nNo signals detected in any file.")

if __name__ == "__main__":
    main()

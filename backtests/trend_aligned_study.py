"""
Trend-Aligned EWVMA Study
Compare counter-trend vs trend-aligned with 50-candle HH/LL lookback
"""
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_ema(prices: pd.Series, length: int) -> pd.Series:
    """Calculate EMA."""
    return prices.ewm(span=length, adjust=False).mean()

def calculate_ewvma(df: pd.DataFrame, length: int = 200) -> pd.Series:
    """Calculate EWVMA (Elastic Volume Weighted Moving Average)."""
    ewvma = pd.Series(index=df.index, dtype=float)
    volume_sum = []
    prev_ewvma = None

    for i in range(len(df)):
        vol = df['volume'].iloc[i] if df['volume'].iloc[i] > 0 else 1
        volume_sum.append(vol)
        if len(volume_sum) > length:
            volume_sum.pop(0)

        if prev_ewvma is None:
            ewvma.iloc[i] = df['close'].iloc[i]
        else:
            nbfs = sum(volume_sum)
            if nbfs > 0:
                ewvma.iloc[i] = prev_ewvma * (nbfs - vol) / nbfs + (vol * df['close'].iloc[i] / nbfs)
            else:
                ewvma.iloc[i] = df['close'].iloc[i]

        prev_ewvma = ewvma.iloc[i]

    return ewvma

def get_band_color(ema9: float, ema21: float, ema50: float) -> str:
    """Determine EMA ribbon band color."""
    if ema9 > ema21 > ema50:
        return 'green'
    elif ema9 < ema21 < ema50:
        return 'red'
    else:
        return 'grey'

def calculate_tai_index(df: pd.DataFrame, rsi_length: int = 100, stoch_length: int = 200) -> pd.Series:
    """
    Calculate Tai Index (Stochastic RSI with slow parameters).

    Tai Index = Stochastic(RSI(close, 100), 200)
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(span=rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Stochastic of RSI
    rsi_min = rsi.rolling(stoch_length).min()
    rsi_max = rsi.rolling(stoch_length).max()

    tai_index = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100

    return tai_index

def calculate_zscore(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    """
    Calculate z-score (deviations from rolling mean).

    z = (price - mean) / std
    """
    rolling_mean = df['close'].rolling(lookback).mean()
    rolling_std = df['close'].rolling(lookback).std()

    zscore = (df['close'] - rolling_mean) / rolling_std

    return zscore

def resample_to_htf(df: pd.DataFrame, htf_minutes: int = 240) -> pd.DataFrame:
    """Resample to higher timeframe and calculate 50 EMA."""
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
    """Get HTF bias at a given timestamp."""
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

def detect_signals(df: pd.DataFrame, hh_ll_lookback: int = 50, counter_trend: bool = False,
                   htf_df: pd.DataFrame = None, use_htf_filter: bool = False,
                   use_tai_filter: bool = False, tai_threshold: float = 45.0,
                   use_zscore_filter: bool = False, zscore_lookback: int = 200,
                   zscore_max: float = 1.5) -> list:
    """
    Detect Double Touch signals.

    Args:
        df: DataFrame with OHLCV data
        hh_ll_lookback: Lookback for HH/LL detection (20 or 50)
        counter_trend: If True, trade against EWVMA. If False, trade with EWVMA.
        htf_df: Higher timeframe dataframe for directional filter
        use_htf_filter: If True, only take longs when HTF bullish, shorts when HTF bearish
        use_tai_filter: If True, filter signals based on Tai Index
        tai_threshold: Tai Index threshold (longs when > threshold, shorts when < 100-threshold)
        use_zscore_filter: If True, filter based on z-score from mean
        zscore_lookback: Lookback for z-score calculation (200 or 500)
        zscore_max: Max z-score allowed (e.g., 1.5 = within 1.5 std dev)
    """
    signals = []

    # Calculate indicators
    df = df.copy()
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['ema50'] = calculate_ema(df['close'], 50)
    df['ewvma200'] = calculate_ewvma(df, 200)
    df['band'] = df.apply(lambda r: get_band_color(r['ema9'], r['ema21'], r['ema50']), axis=1)

    # Calculate Tai Index
    df['tai_index'] = calculate_tai_index(df)

    # Calculate z-score
    df['zscore'] = calculate_zscore(df, zscore_lookback)

    # Find HH/LL with specified lookback
    df['hh'] = df['high'].rolling(hh_ll_lookback).max() == df['high']
    df['ll'] = df['low'].rolling(hh_ll_lookback).min() == df['low']

    n = len(df)

    for i in range(100, n - 50):
        # LONG pattern: HH on green -> grey -> green -> grey -> green + FVG
        if df['hh'].iloc[i] and df['band'].iloc[i] == 'green':
            step0_idx = i
            step0_price = df['high'].iloc[i]
            ewvma_at_step0 = df['ewvma200'].iloc[i]
            close_at_step0 = df['close'].iloc[i]

            # EWVMA filter
            if counter_trend:
                # Counter-trend: price < EWVMA for longs
                if close_at_step0 >= ewvma_at_step0:
                    continue
            else:
                # Trend-aligned: price > EWVMA for longs
                if close_at_step0 <= ewvma_at_step0:
                    continue

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

                                            # HTF filter
                                            if use_htf_filter and htf_df is not None:
                                                htf_bias = get_htf_bias(entry_time, htf_df)
                                                if htf_bias != 'long':
                                                    break  # Skip this signal

                                            # Tai Index filter - longs when index > threshold
                                            if use_tai_filter:
                                                tai_value = df['tai_index'].iloc[m]
                                                if pd.isna(tai_value) or tai_value < tai_threshold:
                                                    break  # Skip this signal

                                            # Z-score filter - within X std dev of mean
                                            if use_zscore_filter:
                                                z_value = df['zscore'].iloc[m]
                                                if pd.isna(z_value) or abs(z_value) > zscore_max:
                                                    break  # Skip if too far from mean

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
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open')
                                            })
                                            break
                                    break
                            break
                    break

        # SHORT pattern: LL on red -> grey -> red -> grey -> red
        if df['ll'].iloc[i] and df['band'].iloc[i] == 'red':
            step0_idx = i
            step0_price = df['low'].iloc[i]
            ewvma_at_step0 = df['ewvma200'].iloc[i]
            close_at_step0 = df['close'].iloc[i]

            # EWVMA filter
            if counter_trend:
                # Counter-trend: price > EWVMA for shorts
                if close_at_step0 <= ewvma_at_step0:
                    continue
            else:
                # Trend-aligned: price < EWVMA for shorts
                if close_at_step0 >= ewvma_at_step0:
                    continue

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

                                            # HTF filter
                                            if use_htf_filter and htf_df is not None:
                                                htf_bias = get_htf_bias(entry_time, htf_df)
                                                if htf_bias != 'short':
                                                    break  # Skip this signal

                                            # Tai Index filter - shorts when index < (100 - threshold)
                                            if use_tai_filter:
                                                tai_value = df['tai_index'].iloc[m]
                                                if pd.isna(tai_value) or tai_value > (100 - tai_threshold):
                                                    break  # Skip this signal

                                            # Z-score filter - within X std dev of mean
                                            if use_zscore_filter:
                                                z_value = df['zscore'].iloc[m]
                                                if pd.isna(z_value) or abs(z_value) > zscore_max:
                                                    break  # Skip if too far from mean

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
                                                'outcome': 'win' if hit_tp else ('loss' if hit_sl else 'open')
                                            })
                                            break
                                    break
                            break
                    break

    return signals

def analyze_signals(signals: list, name: str) -> dict:
    """Analyze signals and print results."""
    closed = [s for s in signals if s['outcome'] != 'open']
    wins = len([s for s in closed if s['outcome'] == 'win'])
    losses = len([s for s in closed if s['outcome'] == 'loss'])
    total = len(closed)

    if total == 0:
        return {'wins': 0, 'losses': 0, 'total': 0, 'wr': 0, 'pf': 0, 'exp': 0}

    wr = wins / total * 100
    pf = (wins * 3) / losses if losses > 0 else float('inf')
    exp = (wr/100 * 3) - ((100-wr)/100 * 1)

    longs = [s for s in closed if s['type'] == 'long']
    shorts = [s for s in closed if s['type'] == 'short']
    long_wins = len([s for s in longs if s['outcome'] == 'win'])
    short_wins = len([s for s in shorts if s['outcome'] == 'win'])

    print(f"\n{name}:")
    print(f"  Total Trades: {total}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Expectancy: {exp:+.2f}R")
    print(f"  Longs: {len(longs)} ({long_wins} wins = {long_wins/len(longs)*100:.1f}% WR)" if longs else "  Longs: 0")
    print(f"  Shorts: {len(shorts)} ({short_wins} wins = {short_wins/len(shorts)*100:.1f}% WR)" if shorts else "  Shorts: 0")

    return {'wins': wins, 'losses': losses, 'total': total, 'wr': wr, 'pf': pf, 'exp': exp}

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

    # Focus on 5-minute data
    csv_files = list(data_dir.glob("chart_data/*5*.csv"))

    print(f"Found {len(csv_files)} 5-minute data files")
    print("=" * 70)
    print("ANALYZING WHY SHORTS FAIL - TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 70)

    # Different configurations to test for shorts
    all_baseline = []          # No filters (raw pattern)
    all_htf_only = []          # HTF filter only
    all_tai_only = []          # Tai filter only
    all_htf_tai = []           # HTF + Tai (current)
    all_tai_inverted = []      # Tai < 55 for shorts (inverted)
    all_no_htf_tai55 = []      # No HTF, Tai < 55 for shorts

    for csv_file in csv_files:  # Test on all files
        try:
            df = load_data(str(csv_file))

            required = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            if not all(col in df.columns for col in required):
                continue

            if len(df) < 5000:  # Need enough for HTF
                continue

            print(f"\nProcessing: {csv_file.name} ({len(df):,} candles)")

            # Create HTF dataframe
            htf_4h = resample_to_htf(df, htf_minutes=240)

            # 1. Baseline - no filters
            baseline = detect_signals(df, hh_ll_lookback=50, counter_trend=False)

            # 2. HTF only
            htf_only = detect_signals(df, hh_ll_lookback=50, counter_trend=False,
                                       htf_df=htf_4h, use_htf_filter=True)

            # 3. Tai only (longs > 45, shorts < 55)
            tai_only = detect_signals(df, hh_ll_lookback=50, counter_trend=False,
                                       use_tai_filter=True, tai_threshold=45.0)

            # 4. HTF + Tai (current config)
            htf_tai = detect_signals(df, hh_ll_lookback=50, counter_trend=False,
                                      htf_df=htf_4h, use_htf_filter=True,
                                      use_tai_filter=True, tai_threshold=45.0)

            all_baseline.extend(baseline)
            all_htf_only.extend(htf_only)
            all_tai_only.extend(tai_only)
            all_htf_tai.extend(htf_tai)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ==================== ANALYZE SHORTS SPECIFICALLY ====================
    print("\n" + "=" * 70)
    print("SHORTS ANALYSIS - WHY ARE THEY FAILING?")
    print("=" * 70)

    def analyze_by_type(signals, name):
        """Analyze longs and shorts separately."""
        closed = [s for s in signals if s['outcome'] != 'open']
        longs = [s for s in closed if s['type'] == 'long']
        shorts = [s for s in closed if s['type'] == 'short']

        long_wins = len([s for s in longs if s['outcome'] == 'win'])
        short_wins = len([s for s in shorts if s['outcome'] == 'win'])

        print(f"\n{name}:")
        if longs:
            long_wr = long_wins / len(longs) * 100
            long_exp = (long_wr/100 * 3) - ((100-long_wr)/100 * 1)
            print(f"  LONGS:  {len(longs):3} trades, {long_wins:3} wins = {long_wr:5.1f}% WR, {long_exp:+.2f}R")
        else:
            print(f"  LONGS:  0 trades")

        if shorts:
            short_wr = short_wins / len(shorts) * 100
            short_exp = (short_wr/100 * 3) - ((100-short_wr)/100 * 1)
            print(f"  SHORTS: {len(shorts):3} trades, {short_wins:3} wins = {short_wr:5.1f}% WR, {short_exp:+.2f}R")
        else:
            print(f"  SHORTS: 0 trades")

        return {
            'long_trades': len(longs),
            'long_wins': long_wins,
            'long_wr': long_wins / len(longs) * 100 if longs else 0,
            'short_trades': len(shorts),
            'short_wins': short_wins,
            'short_wr': short_wins / len(shorts) * 100 if shorts else 0,
        }

    print("\n--- FILTER COMPARISON FOR SHORTS ---")
    r1 = analyze_by_type(all_baseline, "1. NO FILTERS (raw pattern)")
    r2 = analyze_by_type(all_htf_only, "2. HTF ONLY (4H EMA50)")
    r3 = analyze_by_type(all_tai_only, "3. TAI ONLY (>45 longs, <55 shorts)")
    r4 = analyze_by_type(all_htf_tai, "4. HTF + TAI (current config)")

    print("\n" + "=" * 70)
    print("SHORTS SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Config':<30} {'Trades':>8} {'Wins':>6} {'WR':>8} {'Exp':>8}")
    print("-" * 62)
    print(f"{'No filters':<30} {r1['short_trades']:>8} {r1['short_wins']:>6} {r1['short_wr']:>7.1f}% {(r1['short_wr']/100*3 - (100-r1['short_wr'])/100*1):>+7.2f}R")
    print(f"{'HTF only':<30} {r2['short_trades']:>8} {r2['short_wins']:>6} {r2['short_wr']:>7.1f}% {(r2['short_wr']/100*3 - (100-r2['short_wr'])/100*1):>+7.2f}R")
    print(f"{'Tai only':<30} {r3['short_trades']:>8} {r3['short_wins']:>6} {r3['short_wr']:>7.1f}% {(r3['short_wr']/100*3 - (100-r3['short_wr'])/100*1):>+7.2f}R")
    print(f"{'HTF + Tai':<30} {r4['short_trades']:>8} {r4['short_wins']:>6} {r4['short_wr']:>7.1f}% {(r4['short_wr']/100*3 - (100-r4['short_wr'])/100*1):>+7.2f}R")

    print("\n" + "=" * 70)
    print("LONGS SUMMARY TABLE (for comparison)")
    print("=" * 70)
    print(f"\n{'Config':<30} {'Trades':>8} {'Wins':>6} {'WR':>8} {'Exp':>8}")
    print("-" * 62)
    print(f"{'No filters':<30} {r1['long_trades']:>8} {r1['long_wins']:>6} {r1['long_wr']:>7.1f}% {(r1['long_wr']/100*3 - (100-r1['long_wr'])/100*1):>+7.2f}R")
    print(f"{'HTF only':<30} {r2['long_trades']:>8} {r2['long_wins']:>6} {r2['long_wr']:>7.1f}% {(r2['long_wr']/100*3 - (100-r2['long_wr'])/100*1):>+7.2f}R")
    print(f"{'Tai only':<30} {r3['long_trades']:>8} {r3['long_wins']:>6} {r3['long_wr']:>7.1f}% {(r3['long_wr']/100*3 - (100-r3['long_wr'])/100*1):>+7.2f}R")
    print(f"{'HTF + Tai':<30} {r4['long_trades']:>8} {r4['long_wins']:>6} {r4['long_wr']:>7.1f}% {(r4['long_wr']/100*3 - (100-r4['long_wr'])/100*1):>+7.2f}R")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"\nShorts with NO filters: {r1['short_wr']:.1f}% WR")
    print(f"Longs with NO filters:  {r1['long_wr']:.1f}% WR")
    print(f"\nDifference: {r1['long_wr'] - r1['short_wr']:.1f}% (longs naturally better)")

    if r1['short_wr'] < 25:
        print("\n>>> SHORTS ARE FUNDAMENTALLY BROKEN - Even without filters!")
        print(">>> This suggests the SHORT pattern itself doesn't work well on crypto.")
        print(">>> Crypto has a long bias - markets tend to go UP over time.")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

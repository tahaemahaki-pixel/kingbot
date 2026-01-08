"""
Backtest Double Touch Strategy on 1-minute data with 60-minute HTF
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Data directory
DATA_DIR = Path("/home/tahae/ai-content/data/Tradingdata/chart_data")

# 1-minute data files
DATA_FILES = [
    "BYBIT_BTCUSDT.P, 1_da562.csv",
    "BYBIT_ETHUSDT.P, 1_79d61.csv",
]


def load_data(filepath):
    """Load and prepare OHLCV data."""
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename columns if needed
    col_map = {'timestamp': 'time'}
    df.rename(columns=col_map, inplace=True)

    # Ensure we have required columns
    required = ['time', 'open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            print(f"  Warning: Missing column {col}")
            return None

    # Convert time to unix timestamp if it's a datetime string
    if df['time'].dtype == 'object':
        df['time'] = pd.to_datetime(df['time']).astype(int) // 10**9

    # Add volume column if missing (use 1 as placeholder)
    if 'volume' not in df.columns:
        df['volume'] = 1

    df = df.sort_values('time').reset_index(drop=True)
    return df


def calculate_emas(df, fast=9, med=21, slow=50):
    """Calculate EMA ribbon."""
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_med'] = df['close'].ewm(span=med, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    return df


def calculate_band_color(df):
    """Determine band color based on EMA order."""
    conditions = [
        (df['ema_fast'] > df['ema_med']) & (df['ema_med'] > df['ema_slow']),  # green
        (df['ema_fast'] < df['ema_med']) & (df['ema_med'] < df['ema_slow']),  # red
    ]
    choices = ['green', 'red']
    df['band_color'] = np.select(conditions, choices, default='grey')
    return df


def calculate_ewvma(df, length=200):
    """Calculate EWVMA-200 (or EMA-200 if no volume data)."""
    if df['volume'].sum() > len(df):  # Has real volume data
        volume_sum = df['volume'].ewm(span=length, adjust=False).mean() * length
        pv_sum = (df['close'] * df['volume']).ewm(span=length, adjust=False).mean() * length
        df['ewvma_200'] = pv_sum / volume_sum
    else:
        # No volume data, use simple EMA-200
        df['ewvma_200'] = df['close'].ewm(span=length, adjust=False).mean()
    df['ewvma_200'] = df['ewvma_200'].fillna(df['close'])
    return df


def detect_hh_ll(df, lookback=50):
    """Detect Higher Highs and Lower Lows."""
    df['is_hh'] = False
    df['is_ll'] = False

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']

        if current_high > window['high'].max():
            df.loc[df.index[i], 'is_hh'] = True
        if current_low < window['low'].min():
            df.loc[df.index[i], 'is_ll'] = True

    return df


def detect_fvg(df, start_idx, end_idx, fvg_type):
    """Find Fair Value Gap in range."""
    for i in range(start_idx + 2, min(end_idx, len(df))):
        if i < 2:
            continue

        candle_0 = df.iloc[i - 2]  # Two candles ago
        candle_2 = df.iloc[i]      # Current candle

        if fvg_type == 'bullish':
            # Bullish FVG: gap between candle_0 high and candle_2 low
            if candle_2['low'] > candle_0['high']:
                return {
                    'type': 'bullish',
                    'top': candle_2['low'],
                    'bottom': candle_0['high'],
                    'idx': i
                }
        else:
            # Bearish FVG: gap between candle_0 low and candle_2 high
            if candle_2['high'] < candle_0['low']:
                return {
                    'type': 'bearish',
                    'top': candle_0['low'],
                    'bottom': candle_2['high'],
                    'idx': i
                }

    return None


def resample_to_htf(df, htf_minutes=60):
    """Resample 1-minute data to HTF."""
    df_htf = df.copy()
    df_htf['time_dt'] = pd.to_datetime(df_htf['time'], unit='s')
    df_htf.set_index('time_dt', inplace=True)

    htf = df_htf.resample(f'{htf_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna()

    htf['ema_50'] = htf['close'].ewm(span=50, adjust=False).mean()
    htf.reset_index(inplace=True)

    return htf


def get_htf_bias(df_htf, ltf_time):
    """Get HTF bias (long/short/neutral) at a given LTF timestamp."""
    # Find the HTF candle that contains this LTF time
    htf_times = df_htf['time'].values
    idx = np.searchsorted(htf_times, ltf_time, side='right') - 1

    if idx < 1 or idx >= len(df_htf):
        return 'neutral'

    # Use previous completed HTF candle
    htf_candle = df_htf.iloc[idx - 1]

    if htf_candle['close'] > htf_candle['ema_50']:
        return 'long'
    elif htf_candle['close'] < htf_candle['ema_50']:
        return 'short'
    return 'neutral'


def calculate_tai_index(df, rsi_len=100, stoch_len=200):
    """Calculate Tai Index (Stochastic RSI)."""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic of RSI
    rsi_min = rsi.rolling(stoch_len).min()
    rsi_max = rsi.rolling(stoch_len).max()
    df['tai_index'] = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    df['tai_index'] = df['tai_index'].fillna(50)

    return df


def detect_signals(df, df_htf, use_htf=True, use_tai=True, use_ewvma=True,
                   trend_aligned=True, rr=3.0):
    """
    Detect Double Touch signals.

    Args:
        use_htf: Use 60min HTF EMA50 filter
        use_tai: Use Tai Index filter (>45 for longs, <55 for shorts)
        use_ewvma: Use EWVMA-200 filter
        trend_aligned: True = trade WITH trend, False = counter-trend
        rr: Risk/Reward ratio
    """
    signals = []

    # State machines
    long_state = {'active': False, 'step': -1}
    short_state = {'active': False, 'step': -1}

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        band = row['band_color']
        prev_band = prev_row['band_color']

        # ===== LONG SETUP =====
        # Step 0: HH while green
        if row['is_hh'] and band == 'green' and not long_state['active']:
            long_state = {
                'active': True, 'step': 0,
                'step_0_idx': i, 'step_0_price': row['high']
            }

        if long_state['active']:
            # Step 1: Grey (first pullback)
            if long_state['step'] == 0 and band == 'grey':
                long_state['step'] = 1
                long_state['step_1_idx'] = i

            # Step 2: Green again
            elif long_state['step'] == 1 and band == 'green':
                long_state['step'] = 2
                long_state['step_2_idx'] = i

            # Step 3: Grey again (second pullback)
            elif long_state['step'] == 2 and band == 'grey':
                long_state['step'] = 3
                long_state['step_3_idx'] = i
                long_state['step_3_low'] = row['low']

            # Update step 3 low
            elif long_state['step'] == 3 and band == 'grey':
                if row['low'] < long_state['step_3_low']:
                    long_state['step_3_low'] = row['low']

            # Step 4: Green + FVG = signal
            elif long_state['step'] == 3 and band == 'green':
                fvg = detect_fvg(df, long_state['step_3_idx'], i + 1, 'bullish')

                if fvg is not None:
                    valid = True
                    step_0_row = df.iloc[long_state['step_0_idx']]

                    # EWVMA filter
                    if use_ewvma and valid:
                        if trend_aligned:
                            # Longs when price > EWVMA
                            if step_0_row['close'] <= step_0_row['ewvma_200']:
                                valid = False
                        else:
                            # Longs when price < EWVMA (counter-trend)
                            if step_0_row['close'] >= step_0_row['ewvma_200']:
                                valid = False

                    # HTF filter
                    if use_htf and valid:
                        htf_bias = get_htf_bias(df_htf, step_0_row['time'])
                        if htf_bias != 'long':
                            valid = False

                    # Tai Index filter
                    if use_tai and valid:
                        if step_0_row['tai_index'] >= 45:
                            valid = False

                    if valid:
                        entry = fvg['top']
                        sl = long_state['step_3_low'] * 0.999
                        risk = entry - sl
                        tp = entry + (risk * rr)

                        signals.append({
                            'type': 'long',
                            'entry_idx': i,
                            'entry_price': entry,
                            'sl': sl,
                            'tp': tp,
                            'step_0_idx': long_state['step_0_idx'],
                            'fvg': fvg
                        })

                    long_state = {'active': False, 'step': -1}

                elif i - long_state['step_3_idx'] > 10:
                    long_state = {'active': False, 'step': -1}

            # Invalidation
            if band == 'red':
                long_state = {'active': False, 'step': -1}

        # ===== SHORT SETUP =====
        # Step 0: LL while red
        if row['is_ll'] and band == 'red' and not short_state['active']:
            short_state = {
                'active': True, 'step': 0,
                'step_0_idx': i, 'step_0_price': row['low']
            }

        if short_state['active']:
            # Step 1: Grey (first rally)
            if short_state['step'] == 0 and band == 'grey':
                short_state['step'] = 1
                short_state['step_1_idx'] = i

            # Step 2: Red again
            elif short_state['step'] == 1 and band == 'red':
                short_state['step'] = 2
                short_state['step_2_idx'] = i

            # Step 3: Grey again (second rally)
            elif short_state['step'] == 2 and band == 'grey':
                short_state['step'] = 3
                short_state['step_3_idx'] = i
                short_state['step_3_high'] = row['high']

            # Update step 3 high
            elif short_state['step'] == 3 and band == 'grey':
                if row['high'] > short_state['step_3_high']:
                    short_state['step_3_high'] = row['high']

            # Step 4: Red + FVG = signal
            elif short_state['step'] == 3 and band == 'red':
                fvg = detect_fvg(df, short_state['step_3_idx'], i + 1, 'bearish')

                if fvg is not None:
                    valid = True
                    step_0_row = df.iloc[short_state['step_0_idx']]

                    # EWVMA filter
                    if use_ewvma and valid:
                        if trend_aligned:
                            # Shorts when price < EWVMA
                            if step_0_row['close'] >= step_0_row['ewvma_200']:
                                valid = False
                        else:
                            # Shorts when price > EWVMA (counter-trend)
                            if step_0_row['close'] <= step_0_row['ewvma_200']:
                                valid = False

                    # HTF filter
                    if use_htf and valid:
                        htf_bias = get_htf_bias(df_htf, step_0_row['time'])
                        if htf_bias != 'short':
                            valid = False

                    # Tai Index filter
                    if use_tai and valid:
                        if step_0_row['tai_index'] <= 55:
                            valid = False

                    if valid:
                        entry = fvg['bottom']
                        sl = short_state['step_3_high'] * 1.001
                        risk = sl - entry
                        tp = entry - (risk * rr)

                        signals.append({
                            'type': 'short',
                            'entry_idx': i,
                            'entry_price': entry,
                            'sl': sl,
                            'tp': tp,
                            'step_0_idx': short_state['step_0_idx'],
                            'fvg': fvg
                        })

                    short_state = {'active': False, 'step': -1}

                elif i - short_state['step_3_idx'] > 10:
                    short_state = {'active': False, 'step': -1}

            # Invalidation
            if band == 'green':
                short_state = {'active': False, 'step': -1}

    return signals


def simulate_trades(df, signals, rr=3.0):
    """Simulate trades and calculate outcomes."""
    results = []

    for sig in signals:
        entry_idx = sig['entry_idx']
        entry_price = sig['entry_price']
        sl = sig['sl']
        tp = sig['tp']

        outcome = None
        exit_idx = None
        exit_price = None

        # Simulate forward from entry
        for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            candle = df.iloc[j]

            if sig['type'] == 'long':
                # Check SL first (conservative)
                if candle['low'] <= sl:
                    outcome = 'loss'
                    exit_idx = j
                    exit_price = sl
                    break
                # Check TP
                if candle['high'] >= tp:
                    outcome = 'win'
                    exit_idx = j
                    exit_price = tp
                    break
            else:  # short
                # Check SL first
                if candle['high'] >= sl:
                    outcome = 'loss'
                    exit_idx = j
                    exit_price = sl
                    break
                # Check TP
                if candle['low'] <= tp:
                    outcome = 'win'
                    exit_idx = j
                    exit_price = tp
                    break

        if outcome is None:
            outcome = 'open'

        r_multiple = rr if outcome == 'win' else (-1 if outcome == 'loss' else 0)

        results.append({
            'type': sig['type'],
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'outcome': outcome,
            'exit_idx': exit_idx,
            'r_multiple': r_multiple
        })

    return results


def analyze_results(results, label=""):
    """Analyze trade results."""
    if not results:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'expectancy': 0}

    completed = [r for r in results if r['outcome'] in ['win', 'loss']]
    if not completed:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'expectancy': 0}

    wins = sum(1 for r in completed if r['outcome'] == 'win')
    total = len(completed)
    wr = wins / total * 100
    expectancy = sum(r['r_multiple'] for r in completed) / total

    return {
        'trades': total,
        'wins': wins,
        'wr': wr,
        'expectancy': expectancy
    }


def run_backtest():
    """Run the full backtest."""
    print("=" * 70)
    print("DOUBLE TOUCH BACKTEST - 1 MINUTE DATA WITH 60 MIN HTF")
    print("=" * 70)

    all_results = {
        'no_filter': {'long': [], 'short': []},
        'htf_only': {'long': [], 'short': []},
        'tai_only': {'long': [], 'short': []},
        'htf_tai': {'long': [], 'short': []},
        'full': {'long': [], 'short': []},
    }

    for filename in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  File not found: {filename}")
            continue

        symbol = filename.split(",")[0].replace("BYBIT_", "").replace(".P", "")
        print(f"\nProcessing: {symbol} ({filename})")

        # Load data
        df = load_data(filepath)
        if df is None:
            continue

        print(f"  Loaded {len(df):,} candles")

        # Calculate indicators
        df = calculate_emas(df)
        df = calculate_band_color(df)
        df = calculate_ewvma(df)
        df = detect_hh_ll(df, lookback=50)
        df = calculate_tai_index(df)

        # Create HTF data (60 minutes)
        df_htf = resample_to_htf(df, htf_minutes=60)
        print(f"  HTF: {len(df_htf)} 1H candles")

        # Test different configurations
        configs = [
            ('no_filter', {'use_htf': False, 'use_tai': False, 'use_ewvma': False}),
            ('htf_only', {'use_htf': True, 'use_tai': False, 'use_ewvma': True}),
            ('tai_only', {'use_htf': False, 'use_tai': True, 'use_ewvma': True}),
            ('htf_tai', {'use_htf': True, 'use_tai': True, 'use_ewvma': True}),
            ('full', {'use_htf': True, 'use_tai': True, 'use_ewvma': True}),
        ]

        for config_name, config in configs:
            signals = detect_signals(df, df_htf, trend_aligned=True, **config)
            results = simulate_trades(df, signals)

            for r in results:
                all_results[config_name][r['type']].append(r)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - 1 MINUTE DATA WITH 60 MIN HTF")
    print("=" * 70)

    print("\n--- NO FILTERS (raw pattern) ---")
    for direction in ['long', 'short']:
        stats = analyze_results(all_results['no_filter'][direction])
        exp_str = f"+{stats['expectancy']:.2f}R" if stats['expectancy'] >= 0 else f"{stats['expectancy']:.2f}R"
        print(f"  {direction.upper():6s}: {stats['trades']:3d} trades, {stats['wins']:3d} wins = {stats['wr']:5.1f}% WR, {exp_str}")

    print("\n--- HTF ONLY (60min EMA50) ---")
    for direction in ['long', 'short']:
        stats = analyze_results(all_results['htf_only'][direction])
        exp_str = f"+{stats['expectancy']:.2f}R" if stats['expectancy'] >= 0 else f"{stats['expectancy']:.2f}R"
        print(f"  {direction.upper():6s}: {stats['trades']:3d} trades, {stats['wins']:3d} wins = {stats['wr']:5.1f}% WR, {exp_str}")

    print("\n--- TAI ONLY (Stoch RSI filter) ---")
    for direction in ['long', 'short']:
        stats = analyze_results(all_results['tai_only'][direction])
        exp_str = f"+{stats['expectancy']:.2f}R" if stats['expectancy'] >= 0 else f"{stats['expectancy']:.2f}R"
        print(f"  {direction.upper():6s}: {stats['trades']:3d} trades, {stats['wins']:3d} wins = {stats['wr']:5.1f}% WR, {exp_str}")

    print("\n--- HTF + TAI (combined) ---")
    for direction in ['long', 'short']:
        stats = analyze_results(all_results['htf_tai'][direction])
        exp_str = f"+{stats['expectancy']:.2f}R" if stats['expectancy'] >= 0 else f"{stats['expectancy']:.2f}R"
        print(f"  {direction.upper():6s}: {stats['trades']:3d} trades, {stats['wins']:3d} wins = {stats['wr']:5.1f}% WR, {exp_str}")

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Config':<20} {'Longs':<25} {'Shorts':<25}")
    print("-" * 70)

    for config_name in ['no_filter', 'htf_only', 'tai_only', 'htf_tai']:
        long_stats = analyze_results(all_results[config_name]['long'])
        short_stats = analyze_results(all_results[config_name]['short'])

        long_str = f"{long_stats['trades']}t {long_stats['wr']:.1f}% {long_stats['expectancy']:+.2f}R"
        short_str = f"{short_stats['trades']}t {short_stats['wr']:.1f}% {short_stats['expectancy']:+.2f}R"

        print(f"{config_name:<20} {long_str:<25} {short_str:<25}")

    print("=" * 70)


if __name__ == "__main__":
    run_backtest()

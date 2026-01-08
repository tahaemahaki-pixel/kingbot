"""
Compare Double Touch Strategy:
- 1-minute data with 60-minute HTF
- 5-minute data with 4-hour (240min) HTF
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/tahae/ai-content/data/Tradingdata/chart_data")

# Data files by timeframe
DATA_1MIN = [
    "BYBIT_BTCUSDT.P, 1_da562.csv",
    "BYBIT_ETHUSDT.P, 1_79d61.csv",
]

DATA_5MIN = [
    "BYBIT_SOLUSDT.P, 5_7ef98-new.csv",
    "BYBIT_ETHUSDT.P, 5_bf884-new.csv",
    "BYBIT_LINKUSDT.P, 5_4a3e0.csv",
]


def load_data(filepath):
    """Load and prepare OHLCV data."""
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    col_map = {'timestamp': 'time'}
    df.rename(columns=col_map, inplace=True)

    required = ['time', 'open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            return None

    if df['time'].dtype == 'object':
        df['time'] = pd.to_datetime(df['time']).astype(int) // 10**9

    if 'volume' not in df.columns:
        df['volume'] = 1

    df = df.sort_values('time').reset_index(drop=True)
    return df


def calculate_indicators(df):
    """Calculate all indicators."""
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_med'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()

    # Band color
    conditions = [
        (df['ema_fast'] > df['ema_med']) & (df['ema_med'] > df['ema_slow']),
        (df['ema_fast'] < df['ema_med']) & (df['ema_med'] < df['ema_slow']),
    ]
    df['band_color'] = np.select(conditions, ['green', 'red'], default='grey')

    # EMA-200 (using EMA since no volume in some files)
    df['ewvma_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # HH/LL detection
    lookback = 50
    df['is_hh'] = False
    df['is_ll'] = False
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        if df.iloc[i]['high'] > window['high'].max():
            df.loc[df.index[i], 'is_hh'] = True
        if df.iloc[i]['low'] < window['low'].min():
            df.loc[df.index[i], 'is_ll'] = True

    # Tai Index
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=100, adjust=False).mean()
    avg_loss = loss.ewm(span=100, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_min = rsi.rolling(200).min()
    rsi_max = rsi.rolling(200).max()
    df['tai_index'] = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    df['tai_index'] = df['tai_index'].fillna(50)

    return df


def resample_to_htf(df, htf_minutes):
    """Resample to HTF."""
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
    """Get HTF bias at LTF timestamp."""
    htf_times = df_htf['time'].values
    idx = np.searchsorted(htf_times, ltf_time, side='right') - 1
    if idx < 1 or idx >= len(df_htf):
        return 'neutral'
    htf_candle = df_htf.iloc[idx - 1]
    if htf_candle['close'] > htf_candle['ema_50']:
        return 'long'
    elif htf_candle['close'] < htf_candle['ema_50']:
        return 'short'
    return 'neutral'


def detect_fvg(df, start_idx, end_idx, fvg_type):
    """Find FVG in range."""
    for i in range(start_idx + 2, min(end_idx, len(df))):
        if i < 2:
            continue
        c0 = df.iloc[i - 2]
        c2 = df.iloc[i]
        if fvg_type == 'bullish' and c2['low'] > c0['high']:
            return {'top': c2['low'], 'bottom': c0['high'], 'idx': i}
        if fvg_type == 'bearish' and c2['high'] < c0['low']:
            return {'top': c0['low'], 'bottom': c2['high'], 'idx': i}
    return None


def detect_signals(df, df_htf, use_htf=True, use_tai=True, rr=3.0):
    """Detect Double Touch signals."""
    signals = []
    long_state = {'active': False, 'step': -1}
    short_state = {'active': False, 'step': -1}

    for i in range(50, len(df)):
        row = df.iloc[i]
        band = row['band_color']

        # LONG SETUP
        if row['is_hh'] and band == 'green' and not long_state['active']:
            long_state = {'active': True, 'step': 0, 'step_0_idx': i, 'step_0_price': row['high']}

        if long_state['active']:
            if long_state['step'] == 0 and band == 'grey':
                long_state['step'] = 1
                long_state['step_1_idx'] = i
            elif long_state['step'] == 1 and band == 'green':
                long_state['step'] = 2
            elif long_state['step'] == 2 and band == 'grey':
                long_state['step'] = 3
                long_state['step_3_idx'] = i
                long_state['step_3_low'] = row['low']
            elif long_state['step'] == 3 and band == 'grey':
                if row['low'] < long_state['step_3_low']:
                    long_state['step_3_low'] = row['low']
            elif long_state['step'] == 3 and band == 'green':
                fvg = detect_fvg(df, long_state['step_3_idx'], i + 1, 'bullish')
                if fvg:
                    valid = True
                    s0 = df.iloc[long_state['step_0_idx']]
                    # Trend-aligned: longs when price > EMA200
                    if s0['close'] <= s0['ewvma_200']:
                        valid = False
                    if use_htf and valid:
                        if get_htf_bias(df_htf, s0['time']) != 'long':
                            valid = False
                    if use_tai and valid:
                        if s0['tai_index'] >= 45:
                            valid = False
                    if valid:
                        entry = fvg['top']
                        sl = long_state['step_3_low'] * 0.999
                        risk = entry - sl
                        signals.append({
                            'type': 'long', 'entry_idx': i, 'entry': entry,
                            'sl': sl, 'tp': entry + risk * rr
                        })
                    long_state = {'active': False, 'step': -1}
                elif i - long_state['step_3_idx'] > 10:
                    long_state = {'active': False, 'step': -1}
            if band == 'red':
                long_state = {'active': False, 'step': -1}

        # SHORT SETUP
        if row['is_ll'] and band == 'red' and not short_state['active']:
            short_state = {'active': True, 'step': 0, 'step_0_idx': i, 'step_0_price': row['low']}

        if short_state['active']:
            if short_state['step'] == 0 and band == 'grey':
                short_state['step'] = 1
            elif short_state['step'] == 1 and band == 'red':
                short_state['step'] = 2
            elif short_state['step'] == 2 and band == 'grey':
                short_state['step'] = 3
                short_state['step_3_idx'] = i
                short_state['step_3_high'] = row['high']
            elif short_state['step'] == 3 and band == 'grey':
                if row['high'] > short_state['step_3_high']:
                    short_state['step_3_high'] = row['high']
            elif short_state['step'] == 3 and band == 'red':
                fvg = detect_fvg(df, short_state['step_3_idx'], i + 1, 'bearish')
                if fvg:
                    valid = True
                    s0 = df.iloc[short_state['step_0_idx']]
                    # Trend-aligned: shorts when price < EMA200
                    if s0['close'] >= s0['ewvma_200']:
                        valid = False
                    if use_htf and valid:
                        if get_htf_bias(df_htf, s0['time']) != 'short':
                            valid = False
                    if use_tai and valid:
                        if s0['tai_index'] <= 55:
                            valid = False
                    if valid:
                        entry = fvg['bottom']
                        sl = short_state['step_3_high'] * 1.001
                        risk = sl - entry
                        signals.append({
                            'type': 'short', 'entry_idx': i, 'entry': entry,
                            'sl': sl, 'tp': entry - risk * rr
                        })
                    short_state = {'active': False, 'step': -1}
                elif i - short_state['step_3_idx'] > 10:
                    short_state = {'active': False, 'step': -1}
            if band == 'green':
                short_state = {'active': False, 'step': -1}

    return signals


def simulate_trades(df, signals, rr=3.0):
    """Simulate trades."""
    results = []
    for sig in signals:
        outcome = None
        for j in range(sig['entry_idx'] + 1, min(sig['entry_idx'] + 500, len(df))):
            c = df.iloc[j]
            if sig['type'] == 'long':
                if c['low'] <= sig['sl']:
                    outcome = 'loss'
                    break
                if c['high'] >= sig['tp']:
                    outcome = 'win'
                    break
            else:
                if c['high'] >= sig['sl']:
                    outcome = 'loss'
                    break
                if c['low'] <= sig['tp']:
                    outcome = 'win'
                    break
        if outcome:
            results.append({
                'type': sig['type'],
                'outcome': outcome,
                'r': rr if outcome == 'win' else -1
            })
    return results


def analyze(results):
    """Analyze results."""
    if not results:
        return {'t': 0, 'w': 0, 'wr': 0, 'exp': 0}
    wins = sum(1 for r in results if r['outcome'] == 'win')
    total = len(results)
    exp = sum(r['r'] for r in results) / total
    return {'t': total, 'w': wins, 'wr': wins/total*100, 'exp': exp}


def run_backtest(files, htf_minutes, label):
    """Run backtest on a set of files."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    all_results = {
        'no_filter': {'long': [], 'short': []},
        'htf_only': {'long': [], 'short': []},
        'htf_tai': {'long': [], 'short': []},
    }

    for filename in files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  Not found: {filename}")
            continue

        symbol = filename.split(",")[0].replace("BYBIT_", "").replace(".P", "")
        df = load_data(filepath)
        if df is None:
            continue

        print(f"  {symbol}: {len(df):,} candles", end="")
        df = calculate_indicators(df)
        df_htf = resample_to_htf(df, htf_minutes)
        print(f" -> {len(df_htf)} HTF candles")

        configs = [
            ('no_filter', {'use_htf': False, 'use_tai': False}),
            ('htf_only', {'use_htf': True, 'use_tai': False}),
            ('htf_tai', {'use_htf': True, 'use_tai': True}),
        ]

        for name, cfg in configs:
            signals = detect_signals(df, df_htf, **cfg)
            results = simulate_trades(df, signals)
            for r in results:
                all_results[name][r['type']].append(r)

    # Print results
    print(f"\n{'Config':<15} {'Longs':<30} {'Shorts':<30}")
    print("-" * 75)

    for cfg in ['no_filter', 'htf_only', 'htf_tai']:
        l = analyze(all_results[cfg]['long'])
        s = analyze(all_results[cfg]['short'])
        l_str = f"{l['t']:3d}t {l['wr']:5.1f}% {l['exp']:+.2f}R"
        s_str = f"{s['t']:3d}t {s['wr']:5.1f}% {s['exp']:+.2f}R"
        print(f"{cfg:<15} {l_str:<30} {s_str:<30}")

    return all_results


def resample_ltf_data(df, target_minutes):
    """Resample LTF data to a higher timeframe for testing."""
    df_copy = df.copy()
    df_copy['time_dt'] = pd.to_datetime(df_copy['time'], unit='s')
    df_copy.set_index('time_dt', inplace=True)

    resampled = df_copy.resample(f'{target_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna()

    resampled.reset_index(inplace=True)
    resampled = resampled.drop(columns=['time_dt'])
    return resampled


def run_backtest_resampled(files, source_tf_min, target_tf_min, htf_minutes, label):
    """Run backtest by resampling source data to target timeframe."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    all_results = {
        'no_filter': {'long': [], 'short': []},
        'htf_only': {'long': [], 'short': []},
        'htf_tai': {'long': [], 'short': []},
    }

    for filename in files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  Not found: {filename}")
            continue

        symbol = filename.split(",")[0].replace("BYBIT_", "").replace(".P", "")
        df_raw = load_data(filepath)
        if df_raw is None:
            continue

        # Resample to target timeframe
        df = resample_ltf_data(df_raw, target_tf_min)
        print(f"  {symbol}: {len(df_raw):,} -> {len(df):,} candles ({target_tf_min}min)", end="")

        df = calculate_indicators(df)
        df_htf = resample_to_htf(df, htf_minutes)
        print(f" -> {len(df_htf)} HTF candles")

        configs = [
            ('no_filter', {'use_htf': False, 'use_tai': False}),
            ('htf_only', {'use_htf': True, 'use_tai': False}),
            ('htf_tai', {'use_htf': True, 'use_tai': True}),
        ]

        for name, cfg in configs:
            signals = detect_signals(df, df_htf, **cfg)
            results = simulate_trades(df, signals)
            for r in results:
                all_results[name][r['type']].append(r)

    # Print results
    print(f"\n{'Config':<15} {'Longs':<30} {'Shorts':<30}")
    print("-" * 75)

    for cfg in ['no_filter', 'htf_only', 'htf_tai']:
        l = analyze(all_results[cfg]['long'])
        s = analyze(all_results[cfg]['short'])
        l_str = f"{l['t']:3d}t {l['wr']:5.1f}% {l['exp']:+.2f}R"
        s_str = f"{s['t']:3d}t {s['wr']:5.1f}% {s['exp']:+.2f}R"
        print(f"{cfg:<15} {l_str:<30} {s_str:<30}")

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("DOUBLE TOUCH STRATEGY - TIMEFRAME COMPARISON")
    print("=" * 70)

    # 1-minute with 60-minute HTF
    results_1m = run_backtest(DATA_1MIN, 60, "1-MINUTE DATA with 60-MIN (1H) HTF")

    # 5-minute with 240-minute (4H) HTF
    results_5m = run_backtest(DATA_5MIN, 240, "5-MINUTE DATA with 240-MIN (4H) HTF")

    # 15-minute with 120-minute (2H) HTF - resampled from 5-minute data
    results_15m = run_backtest_resampled(DATA_5MIN, 5, 15, 120, "15-MINUTE DATA with 120-MIN (2H) HTF (resampled from 5min)")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON - ALL TIMEFRAMES")
    print("=" * 70)

    print("\n--- HTF ONLY filter ---")
    print(f"{'Timeframe':<25} {'Longs':<25} {'Shorts':<25}")
    print("-" * 75)

    for label, results in [("1min + 1H HTF", results_1m),
                           ("5min + 4H HTF", results_5m),
                           ("15min + 2H HTF", results_15m)]:
        l = analyze(results['htf_only']['long'])
        s = analyze(results['htf_only']['short'])
        print(f"{label:<25} {l['t']:3d}t {l['wr']:5.1f}% {l['exp']:+.2f}R       {s['t']:3d}t {s['wr']:5.1f}% {s['exp']:+.2f}R")

    print("\n--- HTF + TAI filter ---")
    print(f"{'Timeframe':<25} {'Longs':<25} {'Shorts':<25}")
    print("-" * 75)

    for label, results in [("1min + 1H HTF", results_1m),
                           ("5min + 4H HTF", results_5m),
                           ("15min + 2H HTF", results_15m)]:
        l = analyze(results['htf_tai']['long'])
        s = analyze(results['htf_tai']['short'])
        print(f"{label:<25} {l['t']:3d}t {l['wr']:5.1f}% {l['exp']:+.2f}R       {s['t']:3d}t {s['wr']:5.1f}% {s['exp']:+.2f}R")

    print("\n" + "=" * 70)

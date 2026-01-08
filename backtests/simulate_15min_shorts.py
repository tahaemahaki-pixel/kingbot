"""
Simulate $10k account trading 15-minute shorts with 2H HTF
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/tahae/ai-content/data/Tradingdata/chart_data")

DATA_5MIN = [
    "BYBIT_SOLUSDT.P, 5_7ef98-new.csv",
    "BYBIT_ETHUSDT.P, 5_bf884-new.csv",
    "BYBIT_LINKUSDT.P, 5_4a3e0.csv",
]


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'time'}, inplace=True)
    if df['time'].dtype == 'object':
        df['time'] = pd.to_datetime(df['time']).astype(int) // 10**9
    if 'volume' not in df.columns:
        df['volume'] = 1
    return df.sort_values('time').reset_index(drop=True)


def resample_to_tf(df, minutes):
    df_copy = df.copy()
    df_copy['time_dt'] = pd.to_datetime(df_copy['time'], unit='s')
    df_copy.set_index('time_dt', inplace=True)
    resampled = df_copy.resample(f'{minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', 'time': 'first'
    }).dropna()
    resampled.reset_index(inplace=True)
    return resampled.drop(columns=['time_dt'])


def calculate_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_med'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()

    conditions = [
        (df['ema_fast'] > df['ema_med']) & (df['ema_med'] > df['ema_slow']),
        (df['ema_fast'] < df['ema_med']) & (df['ema_med'] < df['ema_slow']),
    ]
    df['band_color'] = np.select(conditions, ['green', 'red'], default='grey')
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # HH/LL
    lookback = 50
    df['is_hh'] = False
    df['is_ll'] = False
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        if df.iloc[i]['high'] > window['high'].max():
            df.loc[df.index[i], 'is_hh'] = True
        if df.iloc[i]['low'] < window['low'].min():
            df.loc[df.index[i], 'is_ll'] = True

    return df


def get_htf_bias(df_htf, ltf_time):
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
    for i in range(start_idx + 2, min(end_idx, len(df))):
        if i < 2:
            continue
        c0 = df.iloc[i - 2]
        c2 = df.iloc[i]
        if fvg_type == 'bearish' and c2['high'] < c0['low']:
            return {'top': c0['low'], 'bottom': c2['high'], 'idx': i}
    return None


def detect_short_signals(df, df_htf, rr=3.0):
    """Detect SHORT signals only."""
    signals = []
    state = {'active': False, 'step': -1}

    for i in range(50, len(df)):
        row = df.iloc[i]
        band = row['band_color']

        # Step 0: LL while red
        if row['is_ll'] and band == 'red' and not state['active']:
            state = {'active': True, 'step': 0, 'step_0_idx': i, 'step_0_price': row['low']}

        if state['active']:
            if state['step'] == 0 and band == 'grey':
                state['step'] = 1
            elif state['step'] == 1 and band == 'red':
                state['step'] = 2
            elif state['step'] == 2 and band == 'grey':
                state['step'] = 3
                state['step_3_idx'] = i
                state['step_3_high'] = row['high']
            elif state['step'] == 3 and band == 'grey':
                if row['high'] > state['step_3_high']:
                    state['step_3_high'] = row['high']
            elif state['step'] == 3 and band == 'red':
                fvg = detect_fvg(df, state['step_3_idx'], i + 1, 'bearish')
                if fvg:
                    valid = True
                    s0 = df.iloc[state['step_0_idx']]

                    # Trend-aligned: shorts when price < EMA200
                    if s0['close'] >= s0['ema_200']:
                        valid = False

                    # HTF filter
                    if valid and get_htf_bias(df_htf, s0['time']) != 'short':
                        valid = False

                    if valid:
                        entry = fvg['bottom']
                        sl = state['step_3_high'] * 1.001
                        risk = sl - entry
                        tp = entry - (risk * rr)

                        signals.append({
                            'entry_idx': i,
                            'entry_time': row['time'],
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'risk_pct': (sl - entry) / entry * 100
                        })

                    state = {'active': False, 'step': -1}
                elif i - state['step_3_idx'] > 10:
                    state = {'active': False, 'step': -1}

            if band == 'green':
                state = {'active': False, 'step': -1}

    return signals


def simulate_account(df, signals, starting_balance=10000, risk_pct=0.02, rr=3.0):
    """Simulate account with compounding."""
    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0

    trades = []
    equity_curve = [(0, starting_balance)]

    for sig in signals:
        entry_idx = sig['entry_idx']
        entry = sig['entry']
        sl = sig['sl']
        tp = sig['tp']

        # Calculate position size
        risk_amount = balance * risk_pct
        risk_per_unit = sl - entry  # For shorts: SL > entry
        position_size = risk_amount / risk_per_unit

        outcome = None
        exit_price = None
        exit_idx = None

        for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            c = df.iloc[j]
            # Short: SL hit if price goes up, TP hit if price goes down
            if c['high'] >= sl:
                outcome = 'loss'
                exit_price = sl
                exit_idx = j
                break
            if c['low'] <= tp:
                outcome = 'win'
                exit_price = tp
                exit_idx = j
                break

        if outcome is None:
            continue

        # Calculate P&L
        if outcome == 'win':
            pnl = position_size * (entry - tp)  # Short profit
            r_multiple = rr
        else:
            pnl = -position_size * (sl - entry)  # Short loss
            r_multiple = -1

        balance += pnl

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        current_dd = (peak_balance - balance) / peak_balance * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        trades.append({
            'entry_idx': entry_idx,
            'entry': entry,
            'exit': exit_price,
            'outcome': outcome,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'balance': balance,
            'drawdown': current_dd
        })

        equity_curve.append((len(trades), balance))

    return trades, equity_curve, max_drawdown


def main():
    print("=" * 70)
    print("15-MINUTE SHORTS SIMULATION - $10,000 STARTING BALANCE")
    print("=" * 70)

    all_signals = []
    all_dfs = []

    for filename in DATA_5MIN:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue

        symbol = filename.split(",")[0].replace("BYBIT_", "").replace(".P", "")
        print(f"\nProcessing {symbol}...")

        # Load and resample to 15-minute
        df_5m = load_data(filepath)
        df = resample_to_tf(df_5m, 15)
        print(f"  {len(df_5m):,} 5m candles -> {len(df):,} 15m candles")

        # Calculate indicators
        df = calculate_indicators(df)

        # Create 2H HTF
        df_htf = resample_to_tf(df, 120)
        df_htf['ema_50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
        print(f"  HTF: {len(df_htf)} 2H candles")

        # Detect signals
        signals = detect_short_signals(df, df_htf)
        print(f"  Found {len(signals)} short signals")

        for sig in signals:
            sig['symbol'] = symbol
            sig['df_ref'] = len(all_dfs)

        all_signals.extend(signals)
        all_dfs.append(df)

    # Sort signals by time
    all_signals.sort(key=lambda x: x['entry_time'])

    print(f"\n{'='*70}")
    print(f"TOTAL SIGNALS: {len(all_signals)}")
    print(f"{'='*70}")

    # Test different risk levels
    risk_levels = [0.01, 0.02, 0.03, 0.05]

    print(f"\n{'Risk %':<10} {'Final Balance':<15} {'Return':<12} {'Max DD':<10} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 70)

    for risk_pct in risk_levels:
        # Simulate with merged timeline
        balance = 10000
        peak = 10000
        max_dd = 0
        wins = 0
        losses = 0

        for sig in all_signals:
            df = all_dfs[sig['df_ref']]
            entry_idx = sig['entry_idx']
            entry = sig['entry']
            sl = sig['sl']
            tp = sig['tp']

            risk_amount = balance * risk_pct
            risk_per_unit = sl - entry
            position_size = risk_amount / risk_per_unit

            outcome = None
            for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
                c = df.iloc[j]
                if c['high'] >= sl:
                    outcome = 'loss'
                    pnl = -position_size * (sl - entry)
                    break
                if c['low'] <= tp:
                    outcome = 'win'
                    pnl = position_size * (entry - tp)
                    break

            if outcome is None:
                continue

            balance += pnl
            if outcome == 'win':
                wins += 1
            else:
                losses += 1

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        total_trades = wins + losses
        wr = wins / total_trades * 100 if total_trades > 0 else 0
        ret = (balance - 10000) / 10000 * 100

        print(f"{risk_pct*100:.0f}%        ${balance:,.0f}       {ret:+.1f}%       {max_dd:.1f}%      {total_trades:<8} {wr:.1f}%")

    # Detailed simulation at 2% risk
    print(f"\n{'='*70}")
    print("DETAILED SIMULATION @ 2% RISK PER TRADE")
    print(f"{'='*70}")

    balance = 10000
    peak = 10000
    max_dd = 0
    trades_detail = []

    for sig in all_signals:
        df = all_dfs[sig['df_ref']]
        entry_idx = sig['entry_idx']
        entry = sig['entry']
        sl = sig['sl']
        tp = sig['tp']

        risk_amount = balance * 0.02
        risk_per_unit = sl - entry
        position_size = risk_amount / risk_per_unit

        outcome = None
        exit_price = None

        for j in range(entry_idx + 1, min(entry_idx + 500, len(df))):
            c = df.iloc[j]
            if c['high'] >= sl:
                outcome = 'loss'
                exit_price = sl
                pnl = -position_size * (sl - entry)
                break
            if c['low'] <= tp:
                outcome = 'win'
                exit_price = tp
                pnl = position_size * (entry - tp)
                break

        if outcome is None:
            continue

        balance += pnl
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd

        trades_detail.append({
            'symbol': sig['symbol'],
            'entry': entry,
            'exit': exit_price,
            'outcome': outcome,
            'pnl': pnl,
            'balance': balance,
            'dd': dd
        })

    # Print trade log
    print(f"\n{'#':<4} {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Result':<8} {'P&L':<12} {'Balance':<12} {'DD':<8}")
    print("-" * 80)

    for i, t in enumerate(trades_detail, 1):
        result_str = "WIN" if t['outcome'] == 'win' else "LOSS"
        print(f"{i:<4} {t['symbol']:<8} {t['entry']:<12.4f} {t['exit']:<12.4f} {result_str:<8} ${t['pnl']:>+10.2f} ${t['balance']:>10.2f} {t['dd']:>6.1f}%")

    # Summary
    wins = sum(1 for t in trades_detail if t['outcome'] == 'win')
    total = len(trades_detail)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY @ 2% RISK")
    print(f"{'='*70}")
    print(f"Starting Balance:  $10,000.00")
    print(f"Final Balance:     ${balance:,.2f}")
    print(f"Total Return:      {(balance-10000)/100:+.1f}%")
    print(f"Max Drawdown:      {max_dd:.1f}%")
    print(f"Total Trades:      {total}")
    print(f"Winners:           {wins} ({wins/total*100:.1f}%)")
    print(f"Losers:            {total-wins} ({(total-wins)/total*100:.1f}%)")
    print(f"Avg Win:           ${sum(t['pnl'] for t in trades_detail if t['outcome']=='win')/wins:.2f}" if wins > 0 else "N/A")
    print(f"Avg Loss:          ${sum(t['pnl'] for t in trades_detail if t['outcome']=='loss')/(total-wins):.2f}" if total-wins > 0 else "N/A")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

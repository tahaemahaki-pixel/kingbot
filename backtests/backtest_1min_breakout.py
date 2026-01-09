#!/usr/bin/env python3
"""
Backtest Breakout Optimized strategy on 1-minute vs 5-minute timeframes.
Compare performance to determine if 1-min is viable.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from breakout_strategy import BreakoutIndicators
from config import BreakoutConfig

# Simulation Constants
FEE_PCT = 0.0006  # 0.06% per round trip
STARTING_BALANCE = 10000
RISK_PER_TRADE = 0.01

DATA_DIR = Path("/home/tahae/ai-content/data/Tradingdata/volume-charts")

@dataclass
class TradeRecord:
    symbol: str
    entry_time: any
    entry_price: float
    exit_time: any
    exit_price: float
    pnl_pct: float
    pnl_r: float
    bars_held: int
    hold_minutes: float = 0

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        if 'time' in df.columns:
            df['time_dt'] = pd.to_datetime(df['time'], utc=True)
        else:
            df['time_dt'] = pd.Series(range(len(df)))
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def run_backtest(df, symbol, config, timeframe_minutes=5):
    """Run backtest on data with given config."""

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    times = df['time_dt'].values

    # Pre-calculate indicators
    inds = BreakoutIndicators.calculate_all(opens, closes, highs, lows, volumes, config)
    atr = inds['atr']
    pivot_highs = inds['pivot_highs']
    evwma_upper = inds['evwma_upper']
    vol_ratio = inds['vol_ratio']
    vol_imbalance = inds['vol_imbalance']

    trades = []
    active_signal = None

    def get_buy_level(idx):
        for j in range(idx - config.pivot_right, -1, -1):
            if not np.isnan(pivot_highs[j]):
                return highs[j]
        return np.nan

    last_buy_level = None
    lookback = max(100, config.evwma_period * 2)

    for i in range(lookback, len(df)):
        if active_signal is None:
            buy_level = get_buy_level(i)
            if np.isnan(buy_level):
                continue

            # Skip duplicate levels
            if last_buy_level is not None and abs(buy_level - last_buy_level) < 0.0001:
                continue

            # Entry Conditions
            if highs[i] > buy_level and \
               closes[i] > evwma_upper[i] and \
               (not config.use_volume_filter or (vol_ratio[i] >= config.min_vol_ratio)) and \
               (not config.use_imbalance_filter or (vol_imbalance[i] >= config.imbalance_threshold)):

                entry_price = max(buy_level, opens[i])
                initial_stop = buy_level - (atr[i] * config.atr_multiplier)

                if initial_stop < entry_price:
                    active_signal = {
                        'entry_price': entry_price,
                        'initial_stop': initial_stop,
                        'trailing_stop': initial_stop,
                        'highest_since_entry': highs[i],
                        'entry_idx': i,
                        'entry_time': times[i]
                    }
                    last_buy_level = buy_level
        else:
            # Update trailing stop
            if highs[i] > active_signal['highest_since_entry']:
                active_signal['highest_since_entry'] = highs[i]
                new_stop = active_signal['highest_since_entry'] - (atr[i] * config.atr_multiplier)
                if new_stop > active_signal['trailing_stop']:
                    active_signal['trailing_stop'] = new_stop

            # Check stop hit
            if lows[i] <= active_signal['trailing_stop']:
                exit_price = min(active_signal['trailing_stop'], opens[i])

                raw_pnl = (exit_price - active_signal['entry_price']) / active_signal['entry_price']
                pnl_pct = (raw_pnl - FEE_PCT) * 100

                initial_risk = active_signal['entry_price'] - active_signal['initial_stop']
                pnl_r = (exit_price - active_signal['entry_price']) / initial_risk if initial_risk > 0 else 0

                bars_held = i - active_signal['entry_idx']
                hold_minutes = bars_held * timeframe_minutes

                trades.append(TradeRecord(
                    symbol=symbol,
                    entry_time=active_signal['entry_time'],
                    entry_price=active_signal['entry_price'],
                    exit_time=times[i],
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_r=pnl_r,
                    bars_held=bars_held,
                    hold_minutes=hold_minutes
                ))
                active_signal = None

    return trades

def get_files_for_timeframe(timeframe):
    """Get data files for a specific timeframe."""
    files = {}
    for f in DATA_DIR.glob("*.csv"):
        name = f.name.lower()
        if timeframe == "1":
            if ", 1_" in name or "_1m_" in name or "_1m.csv" in name:
                # Extract symbol
                if "bybit_" in name:
                    parts = name.replace("bybit_", "").split(",")[0].split("_")[0]
                    symbol = parts.upper().replace(".P", "")
                    if symbol not in files:  # Take first match
                        files[symbol] = f
        elif timeframe == "5":
            if ", 5_" in name or "_5m_" in name:
                if "bybit_" in name:
                    parts = name.replace("bybit_", "").split(",")[0].split("_")[0]
                    symbol = parts.upper().replace(".P", "")
                    if symbol not in files:
                        files[symbol] = f
    return files

def print_stats(trades, label):
    """Print statistics for a list of trades."""
    if not trades:
        print(f"\n{label}: No trades")
        return

    winners = [t for t in trades if t.pnl_r > 0]
    losers = [t for t in trades if t.pnl_r <= 0]

    wr = len(winners) / len(trades) * 100
    total_r = sum(t.pnl_r for t in trades)
    avg_r = total_r / len(trades)
    avg_winner = sum(t.pnl_r for t in winners) / len(winners) if winners else 0
    avg_loser = sum(t.pnl_r for t in losers) / len(losers) if losers else 0
    avg_hold = sum(t.hold_minutes for t in trades) / len(trades)

    # Calculate max drawdown
    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0
    for t in trades:
        balance += balance * RISK_PER_TRADE * t.pnl_r
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100
        max_dd = max(max_dd, dd)

    final_balance = STARTING_BALANCE
    for t in trades:
        final_balance += final_balance * RISK_PER_TRADE * t.pnl_r

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Total Trades:      {len(trades)}")
    print(f"Win Rate:          {wr:.1f}%")
    print(f"Total R:           {total_r:.1f}R")
    print(f"Avg R per Trade:   {avg_r:.2f}R")
    print(f"Avg Winner:        {avg_winner:.2f}R")
    print(f"Avg Loser:         {avg_loser:.2f}R")
    print(f"Avg Hold Time:     {avg_hold:.0f} minutes")
    print(f"Max Drawdown:      {max_dd:.1f}%")
    print(f"Final Balance:     ${final_balance:,.2f}")
    print(f"Return:            {((final_balance/STARTING_BALANCE)-1)*100:.1f}%")

def main():
    print("="*60)
    print("BREAKOUT OPTIMIZED: 1-MIN vs 5-MIN COMPARISON")
    print("="*60)

    # Get files for both timeframes
    files_1m = get_files_for_timeframe("1")
    files_5m = get_files_for_timeframe("5")

    print(f"\n1-minute data files: {len(files_1m)}")
    for sym in sorted(files_1m.keys()):
        print(f"  - {sym}")

    print(f"\n5-minute data files: {len(files_5m)}")

    # Find common symbols
    common_symbols = set(files_1m.keys()) & set(files_5m.keys())
    print(f"\nCommon symbols for comparison: {sorted(common_symbols)}")

    # Config with current settings
    config = BreakoutConfig()

    # Run backtests
    trades_1m = []
    trades_5m = []

    print("\n" + "-"*60)
    print("RUNNING BACKTESTS...")
    print("-"*60)

    for symbol in sorted(common_symbols):
        # 1-minute
        df_1m = load_data(files_1m[symbol])
        if df_1m is not None:
            t1 = run_backtest(df_1m, symbol, config, timeframe_minutes=1)
            trades_1m.extend(t1)
            print(f"{symbol} 1m: {len(t1)} trades")

        # 5-minute
        df_5m = load_data(files_5m[symbol])
        if df_5m is not None:
            t5 = run_backtest(df_5m, symbol, config, timeframe_minutes=5)
            trades_5m.extend(t5)
            print(f"{symbol} 5m: {len(t5)} trades")

    # Print comparison
    print_stats(trades_1m, "1-MINUTE TIMEFRAME")
    print_stats(trades_5m, "5-MINUTE TIMEFRAME")

    # Summary comparison
    if trades_1m and trades_5m:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        wr_1m = len([t for t in trades_1m if t.pnl_r > 0]) / len(trades_1m) * 100
        wr_5m = len([t for t in trades_5m if t.pnl_r > 0]) / len(trades_5m) * 100

        avg_1m = sum(t.pnl_r for t in trades_1m) / len(trades_1m)
        avg_5m = sum(t.pnl_r for t in trades_5m) / len(trades_5m)

        print(f"                    1-MIN      5-MIN")
        print(f"Trades:             {len(trades_1m):>5}      {len(trades_5m):>5}")
        print(f"Win Rate:           {wr_1m:>5.1f}%     {wr_5m:>5.1f}%")
        print(f"Avg R:              {avg_1m:>5.2f}R     {avg_5m:>5.2f}R")

        if avg_1m > avg_5m:
            print("\n>>> 1-MINUTE appears more profitable per trade")
        else:
            print("\n>>> 5-MINUTE appears more profitable per trade")

        # Trade frequency comparison
        print(f"\n1-MIN generates {len(trades_1m)/len(trades_5m):.1f}x more trades than 5-MIN")

if __name__ == "__main__":
    main()

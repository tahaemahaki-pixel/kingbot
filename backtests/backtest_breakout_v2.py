import sys
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path to import bybit_bot modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from breakout_strategy import BreakoutStrategy, BreakoutStatus
from config import BreakoutConfig

# Simulation Constants
FEE_PCT = 0.0006  # 0.06% per round trip
STARTING_BALANCE = 10000
RISK_PER_TRADE = 0.01

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

def run_backtest(df, symbol, config):
    strategy = BreakoutStrategy(symbol, config.timeframe, config)
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    times = df['time_dt'].values
    
    trades = []
    active_signal = None
    
    # We need a decent amount of data before starting
    start_idx = 100 
    
    for i in range(start_idx, len(df)):
        # If not in position, scan for signals
        if active_signal is None:
            # We pass the data up to the current candle
            signal = strategy.scan_for_signals(
                opens[:i+1], closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1], 
                np.arange(i+1) # Using index as time for the internal signal tracker
            )
            
            if signal:
                # GAP HANDLING: Enter at max(entry_price, open)
                entry_price = max(signal.entry_price, opens[i])
                
                # Recalculate stop based on actual entry if needed, 
                # but the strategy uses fixed entry - ATR*mult
                # Here we stick to the signal's logic but adjust entry price
                signal.entry_price = entry_price
                signal.status = BreakoutStatus.FILLED
                active_signal = signal
                active_signal.entry_idx = i
                active_signal.entry_time = times[i]
        
        else:
            # In position: Check for exit or update trailing stop
            
            # Get current ATR for trailing stop
            # Note: Strategy.calculate_all is heavy to call every bar, 
            # but for backtest accuracy we use it or pre-calculate.
            # Here we'll pre-calculate indicators for speed.
            pass

    # Actually, to be more efficient and match the logic, 
    # I'll pre-calculate indicators and then loop.
    return [] # Placeholder

# Revised run_backtest for efficiency
def run_backtest_efficient(df, symbol, config):
    from breakout_strategy import BreakoutIndicators
    
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
    
    # Pivot High tracker
    def get_buy_level(idx):
        # Look back from confirmation point
        for j in range(idx - config.pivot_right, -1, -1):
            if not np.isnan(pivot_highs[j]):
                return highs[j]
        return np.nan

    last_buy_level = None

    for i in range(100, len(df)):
        if active_signal is None:
            buy_level = get_buy_level(i)
            if np.isnan(buy_level): continue
            
            # Skip duplicate levels
            if last_buy_level is not None and abs(buy_level - last_buy_level) < 0.0001:
                continue
                
            # Entry Conditions
            if highs[i] > buy_level and \
               closes[i] > evwma_upper[i] and \
               (not config.use_volume_filter or (vol_ratio[i] >= config.min_vol_ratio)) and \
               (not config.use_imbalance_filter or (vol_imbalance[i] >= config.imbalance_threshold)):
                
                # Signal found!
                entry_price = max(buy_level, opens[i]) # Gap handling
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
            # Exit logic
            # Update trailing stop
            if highs[i] > active_signal['highest_since_entry']:
                active_signal['highest_since_entry'] = highs[i]
                new_stop = active_signal['highest_since_entry'] - (atr[i] * config.atr_multiplier)
                if new_stop > active_signal['trailing_stop']:
                    active_signal['trailing_stop'] = new_stop
            
            # Check stop hit
            if lows[i] <= active_signal['trailing_stop']:
                exit_price = min(active_signal['trailing_stop'], opens[i]) # Gap handling
                
                # Calculate PnL
                raw_pnl = (exit_price - active_signal['entry_price']) / active_signal['entry_price']
                pnl_pct = (raw_pnl - FEE_PCT) * 100
                
                initial_risk = active_signal['entry_price'] - active_signal['initial_stop']
                pnl_r = (exit_price - active_signal['entry_price']) / initial_risk if initial_risk > 0 else 0
                
                trades.append(TradeRecord(
                    symbol=symbol,
                    entry_time=active_signal['entry_time'],
                    entry_price=active_signal['entry_price'],
                    exit_time=times[i],
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_r=pnl_r,
                    bars_held=i - active_signal['entry_idx']
                ))
                active_signal = None
                
    return trades

def main():
    data_dir = Path("volume-charts")
    timeframe = "5"
    
    # Matching logic from previous script
    all_csvs = list(data_dir.glob("*.csv"))
    valid_files = []
    for f in all_csvs:
        name = f.name.lower()
        if f"_{timeframe}m_" in name or f", {timeframe}_" in name or f"_{timeframe}m.csv" in name:
             valid_files.append(f)
             
    if not valid_files:
        print("No valid data files found.")
        return

    # Select 10 random
    if len(valid_files) > 10:
        selected_files = random.sample(valid_files, 10)
    else:
        selected_files = valid_files
        
    print(f"Starting backtest on {len(selected_files)} pairs...")
    
    config = BreakoutConfig()
    all_trades = []
    
    for f in selected_files:
        symbol = f.name.split('_')[1] if '_' in f.name else f.name
        print(f"Testing {symbol}...", end=" ")
        
        df = load_data(f)
        if df is not None:
            trades = run_backtest_efficient(df, symbol, config)
            all_trades.extend(trades)
            
            if trades:
                wr = len([t for t in trades if t.pnl_r > 0]) / len(trades) * 100
                total_r = sum([t.pnl_r for t in trades])
                print(f"{len(trades)} trades, {wr:.1f}% WR, {total_r:.1f}R")
            else:
                print("no trades")
                
    if not all_trades:
        print("No trades found in entire set.")
        return
        
    # Aggregate Stats
    winners = [t for t in all_trades if t.pnl_r > 0]
    wr = len(winners) / len(all_trades) * 100
    total_r = sum([t.pnl_r for t in all_trades])
    avg_r = total_r / len(all_trades)
    
    print("\n" + "="*30)
    print("AGGREGATE RESULTS (Realistic)")
    print("="*30)
    print(f"Total Trades: {len(all_trades)}")
    print(f"Win Rate:     {wr:.1f}%")
    print(f"Total R:      {total_r:.1f}")
    print(f"Avg R:        {avg_r:.2f}")
    
    # Account Simulation
    balance = STARTING_BALANCE
    for t in all_trades:
        balance += balance * RISK_PER_TRADE * t.pnl_r
    
    print(f"Account Growth ($10k): ${balance:,.2f}")
    print(f"Return: {((balance/STARTING_BALANCE)-1)*100:.1f}%")

if __name__ == "__main__":
    main()

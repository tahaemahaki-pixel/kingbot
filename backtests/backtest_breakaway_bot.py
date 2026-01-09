import sys
import os
import random
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from breakaway_bot import BreakoutBot
from config import BotConfig, BreakoutConfig
from data_feed import Candle
from breakout_strategy import BreakoutStatus

# ==================== MOCKS ====================

class MockBybitClient:
    def __init__(self, balance=10000.0):
        self.balance = balance
        self.positions = {} # symbol -> Position
    
    def get_available_balance(self):
        return self.balance
        
    def get_equity(self):
        # Simplified: balance + unrealized pnl
        return self.balance # We won't track unrealized pnl perfectly here
        
    def set_trading_stop(self, symbol, stop_loss, take_profit=None):
        # Simulate API call success
        return True
        
    def get_last_price(self, symbol):
        return 0.0 # Not used in main loop if we inject prices

class MockNotifier:
    def send(self, msg):
        pass

@dataclass
class MockOrder:
    order_id: str
    symbol: str
    status: str = "Filled"

class MockOrderManager:
    def __init__(self, config, client, notifier):
        self.config = config
        self.client = client
        self.positions = {} # symbol -> {entry, size, sl}
        self.trade_history = []
        
    def get_open_count(self):
        return len(self.positions)
        
    def has_position(self, symbol):
        return symbol in self.positions
        
    def can_open_trade(self, balance, setup_key=None, symbol=None):
        if len(self.positions) >= self.config.max_positions:
            return False, "Max positions reached"
        return True, "OK"
        
    def place_order(self, symbol, side, qty, price, stop_loss, take_profit, order_type, signal_type):
        # Simulate filling an order
        if side == "Buy":
            self.positions[symbol] = {
                'entry': price,
                'size': qty,
                'sl': stop_loss,
                'tp': take_profit,
                'symbol': symbol
            }
            return MockOrder("mock_id", symbol)
        return None
        
    def modify_stop_loss(self, symbol, new_sl):
        if symbol in self.positions:
            self.positions[symbol]['sl'] = new_sl
            return True
        return False
        
    def check_exits(self, symbol, low, high, open_price):
        """Custom method for backtest to simulate exits"""
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        sl = pos['sl']
        
        # Check SL hit
        # GAP HANDLING: If low <= SL, we exit.
        # Exit Price is min(SL, Open) if Open < SL (Gap Down)
        # Otherwise SL.
        if low <= sl:
            # Determine exit price
            # If the bar opened BELOW the SL, we get filled at Open (bad slippage)
            # If the bar opened ABOVE the SL but wicked down, we get filled at SL
            exit_price = min(sl, open_price)
            
            pnl = (exit_price - pos['entry']) * pos['size']
            # Fees: 0.06%
            fees = (pos['entry'] * pos['size'] + exit_price * pos['size']) * 0.0006
            net_pnl = pnl - fees
            
            pnl_pct = (exit_price - pos['entry']) / pos['entry'] * 100
            
            self.trade_history.append({
                'symbol': symbol,
                'entry': pos['entry'],
                'exit': exit_price,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct
            })
            
            del self.positions[symbol]
            return "SL"
            
        return None

# ==================== BACKTEST BOT ====================

class BacktestBreakoutBot(BreakoutBot):
    def __init__(self, config, breakout_config, data_files):
        # Initialize parent but don't call super().__init__ directly 
        # because we want to replace components *before* they are used
        self.config = config
        self.breakout_config = breakout_config
        self.running = False
        self.timeframe = self.breakout_config.timeframe
        
        # Inject Mocks
        self.client = MockBybitClient()
        self.notifier = MockNotifier()
        self.order_manager = MockOrderManager(config, self.client, self.notifier)
        
        # Standard components
        from symbol_scanner import SymbolScanner
        # Mock scanner just to avoid errors, we manually set symbols
        self.symbol_scanner = None 
        
        self.symbols = []
        self.feeds = {}
        self.strategies = {}
        self.active_signals = {}
        self.ws = None
        self.tracker = None
        self.signals_count = 0
        self.executed_count = 0
        self.account_balance = 10000.0
        self.account_equity = 10000.0
        
        # Data
        self.data_files = data_files
        self.data_map = {} # symbol -> DataFrame
        
    def _initialize_feeds(self):
        # Override to load from CSVs
        for filepath in self.data_files:
            # Extract symbol name
            # Format: BYBIT_BTCUSDT_5m_20000.csv -> BTCUSDT
            filename = filepath.name
            parts = filename.split('_')
            # Handle standard "BYBIT_BTCUSDT..."
            if parts[0] == "BYBIT" and len(parts) > 1:
                symbol = parts[1]
            else:
                symbol = filename.split('.')[0] # Fallback
            
            # Remove .P if present
            symbol = symbol.replace('.P', '')
            
            self.symbols.append(symbol)
            setup_key = f"{symbol}_{self.timeframe}"
            
            # Load Data
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip().str.lower()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], utc=True).astype(int) // 10**9 # Convert to unix timestamp
            else:
                df['time'] = np.arange(len(df))
            
            self.data_map[setup_key] = df
            
            # Initialize Feed (Empty initially)
            from data_feed import DataFeed
            # We mock DataFeed slightly to avoid real API calls in init
            feed = DataFeed(self.config, self.client, symbol, self.timeframe)
            feed.candles = [] # Start empty
            self.feeds[setup_key] = feed
            
            # Initialize Strategy
            from breakout_strategy import BreakoutStrategy
            self.strategies[setup_key] = BreakoutStrategy(symbol, self.timeframe, self.breakout_config)
            
        print(f"Initialized {len(self.symbols)} pairs from CSVs.")

    def run_backtest(self):
        self._initialize_feeds()
        
        # Find the common time range or just iterate based on index if data is aligned
        # For simplicity, we assume we want to run through the data of each symbol.
        # BUT, to simulate portfolio limits correctly, we should align by time.
        # Given the random random selection, timestamps might overlap.
        # We will iterate row-by-row for the symbol with the most data,
        # and match others by timestamp if possible, or index if not.
        
        # To simplify: We'll align by index since we pulled them all "now" for 20k candles.
        max_len = 0
        for df in self.data_map.values():
            max_len = max(max_len, len(df))
            
        print(f"Running simulation for {max_len} candles...")
        
        # Pre-load data into memory for speed
        dfs = {k: v.to_dict('records') for k,v in self.data_map.items()}
        
        # START SIMULATION
        # Need at least 100 candles for indicators
        # SPEED UP: Process last 5000 candles only
        start_idx = max(100, max_len - 5000)
        
        for i in range(start_idx, max_len):
            # 1. Update Feeds & Check Exits for current candle
            current_timestamp = None
            
            for setup_key in self.symbols:
                setup_key = f"{setup_key}_{self.timeframe}"
                if setup_key not in dfs: continue
                
                data = dfs[setup_key]
                if i >= len(data): continue
                
                row = data[i]
                
                # Convert row to Candle object
                candle = Candle(
                    time=row['time'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                
                # Update feed
                self.feeds[setup_key].candles.append(candle)
                # Keep feed size manageable (optional, but good for memory)
                # self.feeds[setup_key].candles = self.feeds[setup_key].candles[-500:] 
                
                # CHECK EXITS (Simulated)
                # We check exit based on the current candle's Low/Open vs SL
                self.order_manager.check_exits(
                    self.feeds[setup_key].symbol, 
                    candle.low, 
                    candle.high, 
                    candle.open
                )

            # 2. Update Account (Mock)
            # Update equity based on open positions (skipped for simplicity, using balance)
            self.account_balance = 10000.0 + sum(t['pnl'] for t in self.order_manager.trade_history)

            # 3. Logic: Update Trailing Stops & Scan
            # This calls the actual bot logic
            self._update_trailing_stops()
            
            for setup_key in self.symbols:
                setup_key = f"{setup_key}_{self.timeframe}"
                if setup_key not in dfs or i >= len(dfs[setup_key]): continue
                
                # Run scan logic
                # Only if we don't have a position
                symbol = self.feeds[setup_key].symbol
                if not self.order_manager.has_position(symbol):
                    self._scan_symbol(setup_key)
                    
            if i % 1000 == 0:
                print(f"Processed {i}/{max_len} candles...")

        # End of simulation
        print("\nBacktest Complete.")
        self.print_results()

    def print_results(self):
        trades = self.order_manager.trade_history
        if not trades:
            print("No trades executed.")
            return
            
        total_trades = len(trades)
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winners) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades
        
        print("\n" + "="*40)
        print("BREAKAWAY BOT BACKTEST RESULTS")
        print("="*40)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate:     {win_rate:.2f}%")
        print(f"Total PnL:    ${total_pnl:.2f}")
        print(f"Avg PnL:      ${avg_pnl:.2f}")
        print(f"Final Balance: ${10000 + total_pnl:.2f}")
        
        print("\nNote: This backtest exercised the actual 'BreakoutBot' class logic,")
        print("mocking only the external API calls and order management.")

# ==================== RUNNER ====================

def main():
    # 1. Select Data
    data_dir = Path("volume-charts")
    all_csvs = list(data_dir.glob("*_5m_*.csv")) + list(data_dir.glob("*, 5_*.csv"))
    
    if not all_csvs:
        print("No data files found.")
        return
        
    # Select 10 random
    if len(all_csvs) > 10:
        selected_files = random.sample(all_csvs, 10)
    else:
        selected_files = all_csvs
        
    print(f"Selected {len(selected_files)} files for backtest.")
    
    # 2. Config
    config = BotConfig(testnet=True)
    breakout_config = BreakoutConfig(
        timeframe="5",
        risk_per_trade=0.01,
        max_positions=5
    )
    
    # 3. Run
    bot = BacktestBreakoutBot(config, breakout_config, selected_files)
    bot.run_backtest()

if __name__ == "__main__":
    main()

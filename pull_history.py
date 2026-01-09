import requests
import time
import pandas as pd
import os
import argparse
from datetime import datetime

# Configuration
BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"  # or 'inverse' for inverse contracts
OUTPUT_DIR = "../volume-charts"

def get_klines(symbol, interval, limit=1000, end_time=None):
    """
    Fetch klines from Bybit V5 API.
    """
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["end"] = end_time
        
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["retCode"] != 0:
            print(f"Error: {data['retMsg']}")
            return []
            
        return data["result"]["list"]
    except Exception as e:
        print(f"Request failed: {e}")
        return []

def pull_data(symbol, interval, total_candles):
    print(f"Starting fetch for {symbol} ({interval}m) - Target: {total_candles} candles")
    
    all_kline_data = []
    end_time = None
    
    # Estimate batches
    batch_size = 1000 # Max for Bybit V5
    batches = (total_candles + batch_size - 1) // batch_size
    
    start_time_perf = time.time()
    
    for i in range(batches):
        print(f"Fetching batch {i+1}/{batches}...", end="\r")
        
        klines = get_klines(symbol, interval, limit=batch_size, end_time=end_time)
        
        if not klines:
            print("\nNo more data available.")
            break
            
        # Bybit returns newest first
        all_kline_data.extend(klines)
        
        # Update end_time for next batch (oldest timestamp - 1ms)
        # kline structure: [startTime, open, high, low, close, volume, turnover]
        last_candle_time = int(klines[-1][0])
        end_time = last_candle_time - 1
        
        time.sleep(0.1) # Rate limit safety
        
    duration = time.time() - start_time_perf
    print(f"\nFetched {len(all_kline_data)} candles in {duration:.2f} seconds.")
    
    # Process Data
    if not all_kline_data:
        return

    columns = ["startTime", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(all_kline_data, columns=columns)
    
    # Convert types
    df["startTime"] = pd.to_numeric(df["startTime"])
    df["time"] = pd.to_datetime(df["startTime"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col])
        
    # Sort by time (oldest first)
    df = df.sort_values("time").reset_index(drop=True)
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=["time"])
    
    # Trim to exact requested amount (from newest back)
    if len(df) > total_candles:
        df = df.iloc[-total_candles:]
        
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    filename = f"{OUTPUT_DIR}/BYBIT_{symbol}_{interval}m_{len(df)}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")
    print(f"Time Range: {df['time'].min()} to {df['time'].max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull historical data from Bybit")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="5", help="Timeframe (e.g., 1, 5, 15, 60)")
    parser.add_argument("--limit", type=int, default=20000, help="Number of candles to fetch")
    
    args = parser.parse_args()
    
    pull_data(args.symbol, args.interval, args.limit)

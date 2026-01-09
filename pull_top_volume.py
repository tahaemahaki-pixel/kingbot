import requests
import time
from pull_history import pull_data

BASE_URL = "https://api.bybit.com"
CATEGORY = "linear"

def get_top_volume_symbols(limit=20, exclude=None):
    if exclude is None:
        exclude = []
        
    print("Fetching market tickers to identify top volume pairs...")
    url = f"{BASE_URL}/v5/market/tickers"
    params = {
        "category": CATEGORY,
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["retCode"] != 0:
            print(f"Error fetching tickers: {data['retMsg']}")
            return []
            
        tickers = data["result"]["list"]
        
        # Sort by 24h turnover (volume in USDT)
        # Filter only USDT pairs to be safe (though linear usually implies USDT perps mostly)
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        
        sorted_tickers = sorted(usdt_tickers, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        selected_symbols = []
        for t in sorted_tickers:
            sym = t['symbol']
            if sym not in exclude:
                selected_symbols.append(sym)
                if len(selected_symbols) >= limit:
                    break
                    
        return selected_symbols
        
    except Exception as e:
        print(f"Failed to fetch tickers: {e}")
        return []

def main():
    # Exclude BTCUSDT and XRPUSDT as requested
    exclusions = ["BTCUSDT", "XRPUSDT"]
    
    top_symbols = get_top_volume_symbols(limit=20, exclude=exclusions)
    
    if not top_symbols:
        print("No symbols found.")
        return
        
    print(f"\nTop 20 Symbols by Volume (excluding {', '.join(exclusions)}):")
    print(", ".join(top_symbols))
    print("-" * 50)
    
    for i, symbol in enumerate(top_symbols):
        print(f"\n[{i+1}/{len(top_symbols)}] Processing {symbol}...")
        try:
            pull_data(symbol, interval="5", total_candles=20000)
            # Small buffer between symbols
            time.sleep(1)
        except Exception as e:
            print(f"Failed to pull data for {symbol}: {e}")

if __name__ == "__main__":
    main()

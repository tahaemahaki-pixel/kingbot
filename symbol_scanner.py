"""
Symbol Scanner - Fetch top coins from Bybit by 24h volume.

Dynamically fetches the top N USDT perpetual contracts
sorted by 24-hour trading volume.
"""

import time
from typing import List, Optional, Dict
from bybit_client import BybitClient


class SymbolScanner:
    """Scans Bybit for top trading symbols."""

    # Symbols to exclude (stablecoins, wrapped tokens, unavailable)
    EXCLUDED_SYMBOLS = [
        "USDCUSDT",
        "DAIUSDT",
        "TUSDUSDT",
        "BUSDUSDT",
        "USTUSDT",
        "WBTCUSDT",
        "STETHUSDT",
        "WETHUSDT",
        # Not available on Bybit perpetuals
        "SHIBUSDT",
        "1000SHIBUSDT",
    ]

    # Meme coins that use 1000XXXUSDT format on Bybit perpetuals
    # Maps common name -> Bybit perpetual symbol
    MEME_COIN_CONVERSIONS = {
        "PEPEUSDT": "1000PEPEUSDT",
        "FLOKIUSDT": "1000FLOKIUSDT",
        "BONKUSDT": "1000BONKUSDT",
        "LUNCUSDT": "1000LUNCUSDT",
    }

    # Minimum 24h turnover in USD to be considered
    MIN_TURNOVER_USD = 10_000_000  # $10M minimum

    def __init__(self, client: BybitClient):
        self.client = client
        self._cache: Optional[List[str]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 3600  # Cache for 1 hour

    def convert_to_bybit_symbol(self, symbol: str) -> str:
        """Convert symbol to Bybit perpetual format (handles meme coins)."""
        return self.MEME_COIN_CONVERSIONS.get(symbol, symbol)

    def get_top_coins(self, count: int = 50) -> List[str]:
        """
        Fetch top coins by 24h trading volume.

        Uses Bybit API: GET /v5/market/tickers?category=linear
        Returns list of symbols sorted by 24h turnover.
        """
        # Check cache
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache[:count]

        try:
            # Fetch all linear tickers
            tickers = self._fetch_tickers()

            if not tickers:
                print("Warning: No tickers received from Bybit API")
                return self._get_fallback_list(count)

            # Filter and sort
            usdt_pairs = []

            for ticker in tickers:
                symbol = ticker.get('symbol', '')

                # Only USDT perpetuals
                if not symbol.endswith('USDT'):
                    continue

                # Exclude stablecoins and wrapped tokens
                if symbol in self.EXCLUDED_SYMBOLS:
                    continue

                # Get 24h turnover
                turnover = float(ticker.get('turnover24h', 0))

                # Minimum volume filter
                if turnover < self.MIN_TURNOVER_USD:
                    continue

                usdt_pairs.append({
                    'symbol': symbol,
                    'turnover': turnover,
                    'price': float(ticker.get('lastPrice', 0)),
                    'volume': float(ticker.get('volume24h', 0)),
                })

            # Sort by turnover descending
            usdt_pairs.sort(key=lambda x: x['turnover'], reverse=True)

            # Extract symbols
            symbols = [p['symbol'] for p in usdt_pairs[:count]]

            # Update cache
            self._cache = symbols
            self._cache_time = time.time()

            print(f"Fetched top {len(symbols)} coins by 24h volume")
            return symbols

        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return self._get_fallback_list(count)

    def _fetch_tickers(self) -> List[Dict]:
        """Fetch all tickers from Bybit API."""
        try:
            response = self.client._request(
                "GET",
                "/v5/market/tickers",
                {"category": "linear"}
            )

            if response and 'result' in response:
                return response['result'].get('list', [])

            return []

        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []

    def _get_fallback_list(self, count: int) -> List[str]:
        """Return fallback list if API fails (uses correct Bybit symbols)."""
        fallback = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "SUIUSDT",
            "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "APTUSDT",
            "ARBUSDT", "OPUSDT", "NEARUSDT", "FILUSDT", "INJUSDT",
            "MATICUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT", "ETCUSDT",
            "ALGOUSDT", "XLMUSDT", "VETUSDT", "ICPUSDT", "FTMUSDT",
            "SANDUSDT", "MANAUSDT", "AXSUSDT", "GALAUSDT", "TRXUSDT",
            "APEUSDT", "LDOUSDT", "RNDRUSDT", "GMXUSDT", "WIFUSDT",
            "1000PEPEUSDT", "1000FLOKIUSDT", "1000BONKUSDT", "JUPUSDT",
            "PNUTUSDT", "ONDOUSDT", "ENAUSDT", "EIGENUSDT", "TRUMPUSDT",
        ]
        return fallback[:count]

    def merge_with_priority(self, top_coins: List[str], priority_coins: List[str]) -> List[str]:
        """
        Merge top coins with priority list.
        Priority coins always included even if not in top N.
        Converts meme coin symbols to Bybit format.
        Excludes unavailable symbols.

        Args:
            top_coins: List of top coins by volume
            priority_coins: List of priority symbols to always include

        Returns:
            Merged list with priority coins first (all in Bybit format)
        """
        result = []

        # Add priority coins first (convert to Bybit format)
        for symbol in priority_coins:
            bybit_symbol = self.convert_to_bybit_symbol(symbol)
            if bybit_symbol not in result and bybit_symbol not in self.EXCLUDED_SYMBOLS:
                result.append(bybit_symbol)

        # Add top coins (excluding duplicates and unavailable, convert to Bybit format)
        for symbol in top_coins:
            bybit_symbol = self.convert_to_bybit_symbol(symbol)
            if bybit_symbol not in result and bybit_symbol not in self.EXCLUDED_SYMBOLS:
                result.append(bybit_symbol)

        return result

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed info for a specific symbol."""
        try:
            response = self.client._request(
                "GET",
                "/v5/market/tickers",
                {"category": "linear", "symbol": symbol}
            )

            if response and 'result' in response:
                tickers = response['result'].get('list', [])
                if tickers:
                    return tickers[0]

            return None

        except Exception as e:
            print(f"Error fetching symbol info for {symbol}: {e}")
            return None

    def refresh_cache(self):
        """Force refresh the symbol cache."""
        self._cache = None
        self._cache_time = 0


if __name__ == "__main__":
    from config import BotConfig

    # Test symbol scanner
    config = BotConfig.from_env()
    client = BybitClient(config)
    scanner = SymbolScanner(client)

    print("Fetching top 50 coins...")
    top_coins = scanner.get_top_coins(50)

    print(f"\nTop 50 coins by 24h volume:")
    for i, symbol in enumerate(top_coins, 1):
        print(f"  {i:2}. {symbol}")

    # Test merge with priority
    priority = ["SOLUSDT", "BTCUSDT", "PNUTUSDT", "DOGEUSDT"]
    merged = scanner.merge_with_priority(top_coins, priority)

    print(f"\nAfter merging with priority symbols:")
    print(f"  First 10: {merged[:10]}")

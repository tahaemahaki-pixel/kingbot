"""
Bybit API Client - REST and WebSocket
"""
import time
import hmac
import hashlib
import json
import requests
import websocket
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from config import BotConfig, BYBIT_MAINNET, BYBIT_TESTNET, BYBIT_WS_MAINNET, BYBIT_WS_TESTNET


@dataclass
class Position:
    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    order_type: str
    price: float
    qty: float
    status: str


class BybitClient:
    """Bybit REST API Client."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.base_url = BYBIT_TESTNET if config.testnet else BYBIT_MAINNET
        self.session = requests.Session()
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 500ms between requests (conservative for multiple bots)
        self._request_lock = threading.Lock()

    def _generate_signature(self, param_str: str, timestamp: int) -> str:
        """Generate HMAC SHA256 signature."""
        sign_str = str(timestamp) + self.config.api_key + "5000" + param_str
        return hmac.new(
            self.config.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False, retries: int = 3) -> Dict:
        """Make API request with retry logic and rate limiting."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        last_error = None
        for attempt in range(retries):
            try:
                # Rate limiting with lock
                with self._request_lock:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self._min_request_interval:
                        time.sleep(self._min_request_interval - elapsed)
                    self._last_request_time = time.time()

                headers = {"Content-Type": "application/json"}

                if signed:
                    timestamp = int(time.time() * 1000)

                    if method == "GET":
                        # For GET: use query string format
                        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                    else:
                        # For POST: use JSON string
                        param_str = json.dumps(params)

                    signature = self._generate_signature(param_str, timestamp)
                    headers.update({
                        "X-BAPI-API-KEY": self.config.api_key,
                        "X-BAPI-SIGN": signature,
                        "X-BAPI-TIMESTAMP": str(timestamp),
                        "X-BAPI-RECV-WINDOW": "5000"
                    })

                # Use fresh session for each request to avoid threading issues
                if method == "GET":
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                else:
                    response = requests.post(url, json=params, headers=headers, timeout=10)

                # Handle empty or invalid responses
                if not response.text or response.text.strip() == "":
                    raise Exception(f"Empty response from API (status={response.status_code}): {endpoint}")

                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise Exception(f"Invalid JSON from API: {endpoint} - {response.text[:100]}")

                if data.get("retCode") != 0:
                    raise Exception(f"Bybit API Error: {data.get('retMsg')}")

                return data.get("result", {})

            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff: 0.5s, 1s, 1.5s
                    continue
                raise last_error

    # ===== Market Data =====

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        """Get historical klines/candlesticks."""
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        result = self._request("GET", "/v5/market/kline", params)
        return result.get("list", [])

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker info."""
        params = {"category": self.config.category, "symbol": symbol}
        result = self._request("GET", "/v5/market/tickers", params)
        tickers = result.get("list", [])
        return tickers[0] if tickers else {}

    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook."""
        params = {"category": self.config.category, "symbol": symbol, "limit": limit}
        return self._request("GET", "/v5/market/orderbook", params)

    def get_orderbook_imbalance(self, symbol: str, depth: int = 50) -> Optional[float]:
        """
        Calculate order book imbalance from bid/ask depth.
        Returns: -1.0 (all asks) to +1.0 (all bids), None on error.
        Positive = more bids (buying pressure), Negative = more asks (selling pressure)
        """
        try:
            data = self.get_orderbook(symbol, depth)
            bids = data.get("b", [])
            asks = data.get("a", [])

            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            total = bid_volume + ask_volume

            if total > 0:
                return (bid_volume - ask_volume) / total
            return 0.0
        except Exception as e:
            print(f"Error getting orderbook imbalance for {symbol}: {e}")
            return None

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for a symbol.
        Positive = longs pay shorts, Negative = shorts pay longs.
        """
        try:
            params = {"category": self.config.category, "symbol": symbol, "limit": 1}
            data = self._request("GET", "/v5/market/funding/history", params)
            rates = data.get("list", [])
            if rates:
                return float(rates[0].get("fundingRate", 0))
            return None
        except Exception as e:
            print(f"Error getting funding rate for {symbol}: {e}")
            return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        """Get current open interest for a symbol."""
        try:
            params = {"category": self.config.category, "symbol": symbol, "intervalTime": "5min", "limit": 1}
            data = self._request("GET", "/v5/market/open-interest", params)
            oi_list = data.get("list", [])
            if oi_list:
                return float(oi_list[0].get("openInterest", 0))
            return None
        except Exception as e:
            print(f"Error getting open interest for {symbol}: {e}")
            return None

    def get_long_short_ratio(self, symbol: str) -> Optional[Dict]:
        """
        Get long/short ratio (account-based).
        Returns: {'buyRatio': 0.65, 'sellRatio': 0.35} or None
        """
        try:
            params = {"category": self.config.category, "symbol": symbol, "period": "5min", "limit": 1}
            data = self._request("GET", "/v5/market/account-ratio", params)
            ratios = data.get("list", [])
            if ratios:
                return {
                    'buyRatio': float(ratios[0].get("buyRatio", 0.5)),
                    'sellRatio': float(ratios[0].get("sellRatio", 0.5))
                }
            return None
        except Exception as e:
            print(f"Error getting long/short ratio for {symbol}: {e}")
            return None

    # ===== Account =====

    def get_wallet_balance(self) -> Dict:
        """Get wallet balance."""
        params = {"accountType": "UNIFIED"}
        return self._request("GET", "/v5/account/wallet-balance", params, signed=True)

    def get_available_balance(self) -> float:
        """Get available balance (not locked in margin)."""
        wallet = self.get_wallet_balance()
        try:
            account = wallet.get("list", [{}])[0]
            return float(account.get("totalAvailableBalance", 0))
        except (IndexError, KeyError, ValueError):
            return 0.0

    def get_equity(self) -> float:
        """Get total equity (includes margin in use + unrealized PnL)."""
        wallet = self.get_wallet_balance()
        try:
            account = wallet.get("list", [{}])[0]
            return float(account.get("totalEquity", 0))
        except (IndexError, KeyError, ValueError):
            return 0.0

    def get_positions(self, symbol: str = None) -> List[Position]:
        """Get open positions."""
        params = {"category": self.config.category, "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol

        result = self._request("GET", "/v5/position/list", params, signed=True)
        positions = []

        for p in result.get("list", []):
            if float(p.get("size", 0)) > 0:
                positions.append(Position(
                    symbol=p["symbol"],
                    side=p["side"],
                    size=float(p["size"]),
                    entry_price=float(p["avgPrice"]),
                    unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                    leverage=int(p.get("leverage", 1)),
                    stop_loss=float(p.get("stopLoss", 0) or 0),
                    take_profit=float(p.get("takeProfit", 0) or 0)
                ))

        return positions

    # ===== Orders =====

    def place_order(
        self,
        symbol: str,
        side: str,  # "Buy" or "Sell"
        qty: float,
        order_type: str = "Market",  # "Market" or "Limit"
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> Order:
        """Place an order."""
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": "GTC",
            "reduceOnly": reduce_only
        }

        if order_type == "Limit" and price:
            params["price"] = str(price)

        if stop_loss:
            params["stopLoss"] = str(stop_loss)
            params["slTriggerBy"] = "LastPrice"

        if take_profit:
            params["takeProfit"] = str(take_profit)
            params["tpTriggerBy"] = "LastPrice"

        result = self._request("POST", "/v5/order/create", params, signed=True)

        return Order(
            order_id=result.get("orderId", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price or 0,
            qty=qty,
            status="New"
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "orderId": order_id
        }
        self._request("POST", "/v5/order/cancel", params, signed=True)
        return True

    def get_order_status(self, symbol: str, order_id: str) -> str:
        """Get order status. Returns: New, PartiallyFilled, Filled, Cancelled, Rejected, etc."""
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "orderId": order_id
        }
        result = self._request("GET", "/v5/order/realtime", params, signed=True)
        orders = result.get("list", [])

        if orders:
            return orders[0].get("orderStatus", "Unknown")

        # Order not in open orders - check history
        result = self._request("GET", "/v5/order/history", params, signed=True)
        orders = result.get("list", [])

        if orders:
            return orders[0].get("orderStatus", "Unknown")

        return "Unknown"

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all orders for a symbol."""
        params = {"category": self.config.category, "symbol": symbol}
        self._request("POST", "/v5/order/cancel-all", params, signed=True)
        return True

    def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders."""
        params = {"category": self.config.category}
        if symbol:
            params["symbol"] = symbol
        else:
            # When no symbol specified, Bybit requires settleCoin
            params["settleCoin"] = "USDT"

        result = self._request("GET", "/v5/order/realtime", params, signed=True)
        orders = []

        for o in result.get("list", []):
            orders.append(Order(
                order_id=o["orderId"],
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["orderType"],
                price=float(o.get("price", 0)),
                qty=float(o["qty"]),
                status=o["orderStatus"]
            ))

        return orders

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        self._request("POST", "/v5/position/set-leverage", params, signed=True)
        return True

    def get_closed_pnl(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Get closed P&L records."""
        params = {
            "category": self.config.category,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol

        result = self._request("GET", "/v5/position/closed-pnl", params, signed=True)
        return result.get("list", [])

    def get_last_price(self, symbol: str) -> float:
        """Get the last traded price for a symbol."""
        ticker = self.get_ticker(symbol)
        return float(ticker.get("lastPrice", 0))

    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: float = None,
        take_profit: float = None,
        position_idx: int = 0  # 0 = one-way mode
    ) -> bool:
        """
        Modify stop loss and/or take profit for an existing position.

        Args:
            symbol: Trading pair
            stop_loss: New stop loss price (None to keep current)
            take_profit: New take profit price (None to keep current)
            position_idx: 0 for one-way mode, 1 for buy-side (hedge), 2 for sell-side (hedge)

        Returns:
            True if successful
        """
        params = {
            "category": self.config.category,
            "symbol": symbol,
            "positionIdx": position_idx
        }

        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            params["slTriggerBy"] = "LastPrice"

        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            params["tpTriggerBy"] = "LastPrice"

        self._request("POST", "/v5/position/trading-stop", params, signed=True)
        return True

    def close_partial_position(
        self,
        symbol: str,
        qty: float,
        side: str  # Side to close: "Buy" to close short, "Sell" to close long
    ) -> Order:
        """
        Close a partial position using a market order with reduce_only.

        Args:
            symbol: Trading pair
            qty: Quantity to close
            side: "Buy" to close short, "Sell" to close long

        Returns:
            Order object
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="Market",
            reduce_only=True
        )


class BybitWebSocket:
    """
    Bybit WebSocket Client for real-time data.

    Features:
    - Auto-reconnect on disconnect
    - Ping/pong keepalive every 20 seconds (Bybit requirement)
    - Multi-timeframe subscription support

    Bybit WebSocket limits:
    - Ping required every 20 seconds
    - Max 500 connections per 5 min
    - Max args array: 21,000 characters
    """

    def __init__(self, config: BotConfig, symbols: List[str] = None, on_kline: Callable = None, on_trade: Callable = None, subscriptions: List[tuple] = None):
        self.config = config
        self.symbols = symbols or config.symbols
        # subscriptions: list of (symbol, timeframe) tuples for multi-timeframe support
        self.subscriptions = subscriptions
        self.ws_url = BYBIT_WS_TESTNET if config.testnet else BYBIT_WS_MAINNET
        self.ws = None
        self.on_kline = on_kline
        self.on_trade = on_trade
        self.running = False
        self.thread = None
        self.ping_thread = None
        self.last_pong_time = 0
        self.ping_interval = 20  # Bybit requires ping every 20 seconds

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        data = json.loads(message)

        # Handle pong response
        if data.get("op") == "pong" or data.get("ret_msg") == "pong":
            self.last_pong_time = time.time()
            return

        if "topic" in data:
            topic = data["topic"]

            if "kline" in topic:
                # Extract symbol and timeframe from topic: kline.5.BTCUSDT -> timeframe=5, symbol=BTCUSDT
                parts = topic.split(".")
                timeframe = parts[1] if len(parts) >= 2 else None
                symbol = parts[2] if len(parts) >= 3 else None

                kline_data = data.get("data", [])
                for kline in kline_data:
                    if self.on_kline:
                        self.on_kline({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "time": int(kline["start"]),
                            "open": float(kline["open"]),
                            "high": float(kline["high"]),
                            "low": float(kline["low"]),
                            "close": float(kline["close"]),
                            "volume": float(kline["volume"]),
                            "confirm": kline["confirm"]  # True if candle is closed
                        })

            elif "trade" in topic:
                if self.on_trade:
                    for trade in data.get("data", []):
                        self.on_trade(trade)

    def _ping_loop(self):
        """Send periodic pings to keep connection alive."""
        while self.running:
            try:
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    ping_msg = json.dumps({"op": "ping"})
                    self.ws.send(ping_msg)
                time.sleep(self.ping_interval)
            except Exception as e:
                # Connection might be closed, will reconnect
                if self.running:
                    time.sleep(1)

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        print(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        """Handle WebSocket close."""
        print(f"WebSocket Closed: {close_status} - {close_msg}")
        if self.running:
            print("Reconnecting...")
            time.sleep(5)
            self.connect()

    def _on_open(self, ws):
        """Handle WebSocket open."""
        print("WebSocket Connected")
        self.last_pong_time = time.time()

        # Start ping thread if not already running
        if self.ping_thread is None or not self.ping_thread.is_alive():
            self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
            self.ping_thread.start()

        # Subscribe to klines - use subscriptions if provided, otherwise default behavior
        if self.subscriptions:
            # Multi-timeframe: subscribe to each (symbol, timeframe) pair
            topics = [f"kline.{tf}.{sym}" for sym, tf in self.subscriptions]
        else:
            # Legacy: single timeframe for all symbols
            topics = [f"kline.{self.config.timeframe}.{symbol}" for symbol in self.symbols]

        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }
        ws.send(json.dumps(subscribe_msg))
        print(f"Subscribed to {len(topics)} feeds: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")

    def connect(self):
        """Connect to WebSocket."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

        self.running = True
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()

    def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        # Ping thread will exit on next iteration since self.running is False
        if self.ws:
            self.ws.close()
        # Wait for ping thread to finish
        if self.ping_thread and self.ping_thread.is_alive():
            self.ping_thread.join(timeout=2)

    # Alias for compatibility
    close = disconnect


if __name__ == "__main__":
    # Test the client
    config = BotConfig(testnet=True)
    client = BybitClient(config)

    # Test public endpoints
    print("Testing Bybit API...")
    ticker = client.get_ticker("BTCUSDT")
    print(f"BTC Price: ${ticker.get('lastPrice', 'N/A')}")

    klines = client.get_klines("BTCUSDT", "5", limit=5)
    print(f"Last 5 candles: {len(klines)} received")

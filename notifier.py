"""
Telegram Notification System
"""
import requests
from typing import Optional
from config import BotConfig


class TelegramNotifier:
    """Sends notifications via Telegram bot."""

    def __init__(self, config: BotConfig):
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)

        if self.enabled:
            print(f"Telegram notifications enabled")
        else:
            print("Telegram notifications disabled (no token/chat_id)")

    def send(self, message: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False

    def notify_trade_opened(self, symbol: str, side: str, entry: float, sl: float, tp: float, size: float, rr: float):
        """Notify when a trade is opened."""
        emoji = "🟢" if side == "Buy" else "🔴"
        direction = "LONG" if side == "Buy" else "SHORT"

        msg = f"""
{emoji} <b>NEW TRADE OPENED</b>

<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction}
<b>Entry:</b> ${entry:.4f}
<b>Stop Loss:</b> ${sl:.4f}
<b>Take Profit:</b> ${tp:.4f}
<b>Size:</b> {size:.4f}
<b>R:R:</b> {rr:.2f}
"""
        self.send(msg.strip())

    def notify_trade_closed(self, symbol: str, side: str, pnl: float, reason: str):
        """Notify when a trade is closed."""
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = f"""
{emoji} <b>TRADE CLOSED</b>

<b>Symbol:</b> {symbol}
<b>P&L:</b> {pnl_str}
<b>Reason:</b> {reason}
"""
        self.send(msg.strip())

    def notify_signal_found(self, symbol: str, signal_type: str, entry: float, rr: float):
        """Notify when a new signal is found."""
        emoji = "📈" if "long" in signal_type.lower() else "📉"

        msg = f"""
{emoji} <b>NEW SIGNAL</b>

<b>Symbol:</b> {symbol}
<b>Type:</b> {signal_type}
<b>Entry:</b> ${entry:.4f}
<b>R:R:</b> {rr:.2f}
"""
        self.send(msg.strip())

    def notify_daily_summary(self, stats: dict):
        """Send daily summary."""
        msg = f"""
📊 <b>DAILY SUMMARY</b>

<b>Total Trades:</b> {stats.get('total_trades', 0)}
<b>Win Rate:</b> {stats.get('win_rate', 0) * 100:.1f}%
<b>Daily P&L:</b> ${stats.get('daily_pnl', 0):.2f}
<b>Open Trades:</b> {stats.get('open_trades', 0)}
<b>Active Signals:</b> {stats.get('active_signals', 0)}
"""
        self.send(msg.strip())

    def notify_bot_started(self, symbols_count: int, balance: float):
        """Notify when bot starts."""
        msg = f"""
🤖 <b>BOT STARTED</b>

<b>Symbols:</b> {symbols_count}
<b>Balance:</b> ${balance:.2f}
<b>Status:</b> Scanning for King patterns...
"""
        self.send(msg.strip())

    def notify_bot_stopped(self):
        """Notify when bot stops."""
        self.send("🛑 <b>BOT STOPPED</b>")

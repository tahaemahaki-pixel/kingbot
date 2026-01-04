# Double Touch Bot VPS Quick-Start Guide

## How to Connect to Your VPS

1. Open Terminal (Mac/Linux) or PowerShell (Windows)
2. Type this and press Enter:
   ```
   ssh root@209.38.84.47
   ```
3. You're now connected when you see: `root@ubuntu-s-1vcpu-1gb-syd1-01:~#`

---

## Bot Commands (run these on the VPS)

| What you want to do | Command |
|---------------------|---------|
| **See live logs** | `journalctl -u doubletouchbot -f` |
| **Stop watching logs** | Press `Ctrl+C` |
| **Check if bot is running** | `systemctl status doubletouchbot` |
| **Stop the bot** | `systemctl stop doubletouchbot` |
| **Start the bot** | `systemctl start doubletouchbot` |
| **Restart the bot** | `systemctl restart doubletouchbot` |
| **Disconnect from VPS** | Type `exit` and press Enter |

---

## Common Tasks

### Check if the bot is running
```bash
systemctl status doubletouchbot
```
- **Active (running)** = Bot is working
- **Inactive (dead)** = Bot is stopped

### View what the bot is doing (live)
```bash
journalctl -u doubletouchbot -f
```
This shows real-time updates. Press `Ctrl+C` to stop watching (bot keeps running).

### View last 50 lines of logs
```bash
journalctl -u doubletouchbot -n 50
```

### View logs from last hour
```bash
journalctl -u doubletouchbot --since "1 hour ago"
```

### Restart after making changes
```bash
systemctl restart doubletouchbot
```

---

## Important Info

| Item | Value |
|------|-------|
| VPS IP | `209.38.84.47` |
| Bot location | `/root/kingbot/` |
| Config file | `/root/kingbot/.env` |
| Service name | `doubletouchbot` |
| Strategy | Double Touch (counter-trend) |
| Timeframe | 5-minute candles |
| Symbols | 20 crypto coins |

---

## Strategy Settings

| Setting | Value |
|---------|-------|
| EMA Ribbon | 9/21/50 |
| EWVMA Filter | 200-period (counter-trend) |
| Risk:Reward | 3:1 |
| Risk per trade | 1% |
| Max positions | 5 (3 crypto + 2 non-crypto) |

---

## Edit API Keys

If you need to change your Bybit or Telegram keys:

1. Connect to VPS
2. Run: `nano /root/kingbot/.env`
3. Make changes
4. Save: `Ctrl+O` then `Enter` then `Ctrl+X`
5. Restart bot: `systemctl restart doubletouchbot`

### .env File Contents
```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=false
TRADING_TIMEFRAME=5
RISK_PER_TRADE=0.01
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Troubleshooting

### Bot not starting?
Check the logs for errors:
```bash
journalctl -u doubletouchbot -n 100
```

### Need to update the bot?
```bash
cd /root/kingbot
git pull
systemctl restart doubletouchbot
```

### Check for Python errors
```bash
cd /root/kingbot
python3 -c "from bot import TradingBot; print('OK')"
```

### VPS not responding?
Go to DigitalOcean dashboard and use the web console, or reboot the droplet.

### Bot keeps restarting?
Check if there's a crash loop:
```bash
journalctl -u doubletouchbot --since "10 minutes ago" | grep -i error
```

---

## Understanding Log Output

### Startup Messages
```
Double Touch Strategy Bot - Multi-Symbol, Multi-Timeframe
Setups: 20 (symbol+timeframe combinations)
Max positions: 5 (crypto: 3, non-crypto: 2)
Risk/Reward: 3.0:1
```

### Signal Found
```
[BTCUSDT_5] long_double_touch: Entry=97500.00 SL=97000.00 TP=99000.00 R:R=3.00
```
- `BTCUSDT_5` = Symbol @ 5min timeframe
- `long_double_touch` = Buy signal (short = sell)
- `Entry` = Where order will be placed
- `SL` = Stop loss level
- `TP` = Take profit level
- `R:R` = Risk:Reward ratio

### Trade Executed
```
[09:15] BTCUSDT_5 -> Executed long_double_touch @ 97500.00
```

### Stats Update (every 5 minutes)
```
========================================
Trading Stats
========================================
Setups monitored: 20
Active signals: 5
Open trades: 2
Total trades: 15
Win rate: 40.0%
Daily P&L: $125.50
Total P&L: $450.00
========================================
```

---

## Quick Reference Card

```
Connect:     ssh root@209.38.84.47
Logs:        journalctl -u doubletouchbot -f
Status:      systemctl status doubletouchbot
Stop:        systemctl stop doubletouchbot
Start:       systemctl start doubletouchbot
Restart:     systemctl restart doubletouchbot
Update:      cd /root/kingbot && git pull && systemctl restart doubletouchbot
Edit config: nano /root/kingbot/.env
Disconnect:  exit
```

---

## Telegram Notifications

The bot sends alerts for:
- Bot started/stopped
- Limit order placed (waiting for fill)
- Order filled (trade active)
- Trade closed (TP/SL hit)
- Order cancelled (expired)

### Setup Telegram
1. Message @BotFather on Telegram
2. Create new bot, get token
3. Message your bot, then visit:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find your chat_id in the response
5. Add to .env file and restart bot

---

## File Structure on VPS

```
/root/kingbot/
├── bot.py                    # Main bot
├── double_touch_strategy.py  # Strategy logic
├── data_feed.py              # Price data + indicators
├── order_manager.py          # Trade execution
├── config.py                 # Settings
├── bybit_client.py           # Bybit API
├── notifier.py               # Telegram
├── start.py                  # Entry point
├── .env                      # API keys (secret!)
└── CLAUDE.md                 # Documentation
```

---

## Systemd Service Location

```
/etc/systemd/system/doubletouchbot.service
```

To view the service file:
```bash
cat /etc/systemd/system/doubletouchbot.service
```

To reload after editing:
```bash
systemctl daemon-reload
systemctl restart doubletouchbot
```

# Trading Bot VPS Guide

Quick reference for managing trading bots on VPS.

**Updated:** 2026-01-07 - Bot now runs 5-minute only (1-min disabled)

---

## Quick Reference Card

### Breakaway Bot (ACTIVE) - 5-Min Scanner

| Action | Command |
|--------|---------|
| Connect to VPS | `ssh root@209.38.84.47` |
| Go to bot folder | `cd /root/kingbot` |
| Check if running | `pgrep -f breakaway_bot` |
| Start bot | `nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &` |
| Stop bot | `pkill -f breakaway_bot` |
| Restart bot | `pkill -f breakaway_bot && sleep 2 && nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &` |
| Watch live logs | `tail -f breakaway_bot.log` |
| Last 50 lines | `tail -50 breakaway_bot.log` |
| Search signals | `grep -i "signal" breakaway_bot.log` |
| Search errors | `grep -i "error" breakaway_bot.log` |

### Double Touch Bot (Legacy - systemd)

| Action | Command |
|--------|---------|
| Check status | `systemctl status doubletouchbot` |
| Start | `systemctl start doubletouchbot` |
| Stop | `systemctl stop doubletouchbot` |
| Restart | `systemctl restart doubletouchbot` |
| View logs | `journalctl -u doubletouchbot -f` |

---

## Connect to VPS

```bash
ssh root@209.38.84.47
```

Then go to bot folder:
```bash
cd /root/kingbot
```

---

# Breakaway Bot Commands

## Check if Running

```bash
pgrep -f breakaway_bot
```

- **Numbers shown** = Bot IS running (process IDs)
- **Blank/empty** = Bot is NOT running

## Start the Bot

```bash
cd /root/kingbot
nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &
```

Verify:
```bash
pgrep -f breakaway_bot
```

## Stop the Bot

```bash
pkill -f breakaway_bot
```

## Restart the Bot

```bash
pkill -f breakaway_bot && sleep 2 && nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &
```

---

## View Logs

### Last 50 lines
```bash
tail -50 breakaway_bot.log
```

### Watch live (Ctrl+C to stop)
```bash
tail -f breakaway_bot.log
```

### Search for signals
```bash
grep -i "signal" breakaway_bot.log | tail -20
```

### Search for trades/orders
```bash
grep -i "trade\|order\|executed" breakaway_bot.log | tail -20
```

### Search for errors
```bash
grep -i "error" breakaway_bot.log | tail -20
```

### Count WebSocket errors (should be 0)
```bash
grep -c "WebSocket" breakaway_bot.log
```

---

## Check Account

### Balance
```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv()
from bybit_client import BybitClient
from config import BotConfig
c = BybitClient(BotConfig.from_env())
print(f'Balance: \${c.get_available_balance():.2f}')
print(f'Equity: \${c.get_equity():.2f}')
"
```

### Open Positions
```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv()
from bybit_client import BybitClient
from config import BotConfig
c = BybitClient(BotConfig.from_env())
pos = c.get_positions()
if pos:
    for p in pos:
        print(f'{p.symbol}: {p.side} {p.size} @ {p.entry_price:.4f} PnL: \${p.unrealized_pnl:.2f}')
else:
    print('No open positions')
"
```

### Recent Closed Trades
```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv()
from bybit_client import BybitClient
from config import BotConfig
c = BybitClient(BotConfig.from_env())
trades = c.get_closed_pnl(limit=10)
total = 0
for t in trades:
    pnl = float(t.get('closedPnl', 0))
    total += pnl
    print(f\"{t.get('symbol')}: {t.get('side')} PnL: \${pnl:.2f}\")
print(f'---\\nTotal: \${total:.2f}')
"
```

---

## Understanding Log Output

### Normal Status Update (every 5 min)
```
============================================================
BREAKAWAY BOT STATUS - 13:27:45
============================================================

5-MIN TIMEFRAME:
  Symbols: 42
  Positions: 0/5
  Signals: 0 | Executed: 0

ACCOUNT:
  Balance: $215.46
  Equity: $215.46
  Total Open: 0
============================================================
```

### When a Signal Triggers
```
============================================================
[5-MIN] NEW BREAKAWAY SIGNAL - SOLUSDT
============================================================
  Direction: SHORT
  Entry: 187.450000
  Stop Loss: 188.637450
  Target: 183.887850
  R:R: 3.0
  Volume: 3.2x
  Tai Index: 67
============================================================
```

---

## Breakaway Strategy Settings (5-Min Only)

| Parameter | Value |
|-----------|-------|
| Symbols | 45 pairs |
| Risk per trade | 2% |
| Max positions | 5 |
| Volume filter | ≥1.5x |
| Tai short | >53 |
| Tai long | <47 |
| R:R Target | 3:1 |
| Candles preload | 2000 |
| Scan frequency | Every 5-min candle close |

**Note:** 1-minute trading disabled 2026-01-07 (5-min outperforms: 1.13R vs 0.33R expectancy)

### Pairs Being Scanned (45 Total)

**Priority Symbols (always included):**
```
SOLUSDT, BTCUSDT, PNUTUSDT, DOGEUSDT
```

**Full List:**
```
 1. SOLUSDT        16. APTUSDT        31. FTMUSDT
 2. BTCUSDT        17. ARBUSDT        32. SANDUSDT
 3. PNUTUSDT       18. OPUSDT         33. MANAUSDT
 4. DOGEUSDT       19. NEARUSDT       34. AXSUSDT
 5. ETHUSDT        20. FILUSDT        35. GALAUSDT
 6. XRPUSDT        21. INJUSDT        36. TRXUSDT
 7. ADAUSDT        22. MATICUSDT      37. APEUSDT
 8. AVAXUSDT       23. AAVEUSDT       38. LDOUSDT
 9. LINKUSDT       24. MKRUSDT        39. RNDRUSDT
10. DOTUSDT        25. COMPUSDT       40. GMXUSDT
11. SUIUSDT        26. ETCUSDT        41. WIFUSDT
12. LTCUSDT        27. ALGOUSDT       42. 1000PEPEUSDT
13. BCHUSDT        28. XLMUSDT        43. 1000FLOKIUSDT
14. ATOMUSDT       29. VETUSDT        44. 1000BONKUSDT
15. UNIUSDT        30. ICPUSDT        45. JUPUSDT
```

**Note:** Meme coins use `1000XXXUSDT` format on Bybit perpetuals

### Signal Conditions (ALL must align)

**For SHORT:**
1. Bearish FVG (price gap down)
2. Tai Index > 53 (overbought)
3. Volume ≥ 2.0x average
4. Cradle: 3+ of 5 candles in EWVMA(20) bands
5. Price ABOVE EWVMA-200 (counter-trend)

**For LONG:**
1. Bullish FVG (price gap up)
2. Tai Index < 47 (oversold)
3. Volume ≥ 2.0x average
4. Cradle: 3+ of 5 candles in EWVMA(20) bands
5. Price BELOW EWVMA-200 (counter-trend)

---

## One-Liner Commands (from local machine)

### Full status check
```bash
ssh root@209.38.84.47 "cd /root/kingbot && pgrep -f breakaway_bot && tail -30 breakaway_bot.log"
```

### Quick balance check
```bash
ssh root@209.38.84.47 "cd /root/kingbot && python3 -c \"from dotenv import load_dotenv; load_dotenv(); from bybit_client import BybitClient; from config import BotConfig; c=BybitClient(BotConfig.from_env()); print(f'Balance: \\\${c.get_available_balance():.2f}')\""
```

### Restart bot remotely
```bash
ssh root@209.38.84.47 "cd /root/kingbot && pkill -f breakaway_bot; sleep 2; nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &; sleep 1; pgrep -f breakaway_bot && echo 'Bot started'"
```

---

## Deploy Updates

From local machine:
```bash
# Copy files to VPS
scp breakaway_bot.py breakaway_strategy.py config.py bybit_client.py root@209.38.84.47:/root/kingbot/

# Restart bot
ssh root@209.38.84.47 "cd /root/kingbot && pkill -f breakaway_bot && sleep 2 && nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &"
```

---

## Troubleshooting

### Bot not starting?
Run in foreground to see errors:
```bash
python3 breakaway_bot.py
```
(Ctrl+C to stop, then start with nohup)

### Check if .env is loaded
```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv()
import os
key = os.getenv('BYBIT_API_KEY', '')
print(f'API Key: {\"Yes\" if key else \"No\"} ({len(key)} chars)')
print(f'Testnet: {os.getenv(\"BYBIT_TESTNET\", \"not set\")}')
"
```

### WebSocket disconnecting?
Check error count:
```bash
grep -c "WebSocket\|Disconn\|Reconnect" breakaway_bot.log
```
Should be 0 or very low.

### No signals for a long time?
Normal! Strategy needs 5 conditions to align:
- FVG + Tai extreme + Volume spike + Cradle + Counter-trend

---

## VPS Details

| Item | Value |
|------|-------|
| IP | 209.38.84.47 |
| Provider | DigitalOcean |
| Specs | 1 CPU, 1GB RAM, 25GB disk |
| OS | Ubuntu 22.04 LTS |
| Bot path | `/root/kingbot/` |
| Log file | `/root/kingbot/breakaway_bot.log` |
| Config | `/root/kingbot/.env` |

---

## Edit Configuration

```bash
nano /root/kingbot/.env
```

Save: `Ctrl+O` → `Enter` → `Ctrl+X`

Then restart bot.

### .env Contents
```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false
TELEGRAM_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## File Structure

```
/root/kingbot/
├── breakaway_bot.py          # Main bot (ACTIVE)
├── breakaway_strategy.py     # Signal detection
├── bybit_client.py           # API client + WebSocket
├── config.py                 # Settings
├── symbol_scanner.py         # Top coin fetcher
├── .env                      # API keys (secret!)
├── breakaway_bot.log         # Bot logs
│
├── bot.py                    # Double Touch bot (legacy)
├── double_touch_strategy.py  # DT pattern detection
└── CLAUDE.md                 # Full documentation
```

---

## Need Help?

1. Stop bot: `pkill -f breakaway_bot`
2. Check log: `cat breakaway_bot.log`
3. Look for errors
4. Ask Claude!

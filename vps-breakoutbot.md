# VPS Deployment - Breakout Optimized Bot

## Server Details

| Property | Value |
|----------|-------|
| Provider | DigitalOcean |
| IP | `209.38.84.47` |
| OS | Ubuntu 22.04 LTS |
| Specs | 1 CPU, 1GB RAM, 25GB disk |
| Bot Directory | `/root/kingbot` |

---

## Quick Reference

| Action | Command |
|--------|---------|
| Connect to VPS | `ssh root@209.38.84.47` |
| Go to bot folder | `cd /root/kingbot` |
| Start bot | `nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &` |
| Check if running | `pgrep -f breakaway_bot` |
| Watch logs | `tail -f breakout_bot.log` |
| Stop bot | `pkill -f breakaway_bot` |
| Pull latest code | `git pull origin main` |

---

## Initial Deployment

### 1. Connect to VPS

```bash
ssh root@209.38.84.47
cd /root/kingbot
```

### 2. Pull Latest Code

```bash
git pull origin main
```

### 3. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Edit `.env` file with your credentials:

```bash
nano .env
```

Required variables:
```bash
# API Credentials
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false

# Telegram Notifications
TELEGRAM_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Breakout Strategy (defaults shown)
BREAKOUT_TIMEFRAME=5
BREAKOUT_USE_VOLUME_FILTER=true
BREAKOUT_USE_IMBALANCE_FILTER=true
BREAKOUT_MIN_VOL_RATIO=2.0
BREAKOUT_IMBALANCE_THRESHOLD=0.10
BREAKOUT_ATR_PERIOD=14
BREAKOUT_ATR_MULTIPLIER=2.0
BREAKOUT_MAX_POSITIONS=5
BREAKOUT_RISK_PER_TRADE=0.01
BREAKOUT_EMERGENCY_TP=10.0
BREAKOUT_STATE_FILE=data/breakout_signals.json
```

### 5. Start the Bot

```bash
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

### 6. Verify Running

```bash
pgrep -f breakaway_bot
tail -20 breakout_bot.log
```

---

## Managing the Bot

### View Logs

```bash
# Last 50 lines
tail -50 breakout_bot.log

# Watch live
tail -f breakout_bot.log

# Search for signals
grep "BREAKOUT SIGNAL" breakout_bot.log

# Search for errors
grep -i error breakout_bot.log
```

### Stop the Bot

```bash
pkill -f breakaway_bot
```

### Restart the Bot

```bash
pkill -f breakaway_bot
sleep 2
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

### Update to Latest Version

```bash
pkill -f breakaway_bot
git pull origin main
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

---

## State Recovery

The bot automatically persists active signals to `data/breakout_signals.json`.

### On Crash/Restart:
1. Bot loads signals from disk
2. Syncs with actual exchange positions
3. Removes signals for closed positions
4. Resumes trailing stop management

### Manual State Check

```bash
cat data/breakout_signals.json
```

### Clear State (fresh start)

```bash
rm data/breakout_signals.json
```

---

## Strategy Configuration

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Timeframe | 5m | Candle timeframe |
| Risk per trade | 1% | Account risk per position |
| Max positions | 5 | Concurrent positions |
| ATR period | 14 | Trailing stop ATR |
| ATR multiplier | 2.0 | Stop = entry - ATR*2 |
| Emergency TP | 10R | Circuit breaker TP |
| Volume filter | ON | >= 2x average volume |
| Imbalance filter | ON | >= 10% buy pressure |

### Adjust Settings

Edit `.env` and restart:

```bash
nano .env
pkill -f breakaway_bot
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

---

## Monitoring

### Check Bot Status

```bash
# Process running?
pgrep -f breakaway_bot

# Memory usage
ps aux | grep breakaway_bot

# Disk space
df -h
```

### Check Positions

The bot logs position count every 5 minutes:
```bash
grep "Positions:" breakout_bot.log | tail -5
```

### Check Signals

```bash
# Recent signals
grep "BREAKOUT SIGNAL" breakout_bot.log | tail -10

# Signal count
grep -c "BREAKOUT SIGNAL" breakout_bot.log
```

---

## Troubleshooting

### Bot Not Starting

```bash
# Check for errors
python3 breakaway_bot.py

# Check lock file
cat /tmp/breakout_bot.lock
rm /tmp/breakout_bot.lock  # Remove stale lock
```

### Connection Issues

```bash
# Test API connection
python3 -c "
from bybit_client import BybitClient
from config import BotConfig
c = BybitClient(BotConfig.from_env())
print('Balance:', c.get_available_balance())
"
```

### High Memory Usage

```bash
# Check memory
free -m

# Restart bot (clears memory)
pkill -f breakaway_bot
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

---

## Systemd Service (Optional)

For auto-restart on crash/reboot:

### Create Service File

```bash
sudo nano /etc/systemd/system/breakout-bot.service
```

```ini
[Unit]
Description=Breakout Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/kingbot
ExecStart=/usr/bin/python3 -u breakaway_bot.py
Restart=always
RestartSec=10
StandardOutput=append:/root/kingbot/breakout_bot.log
StandardError=append:/root/kingbot/breakout_bot.log

[Install]
WantedBy=multi-user.target
```

### Enable Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable breakout-bot
sudo systemctl start breakout-bot
```

### Manage Service

```bash
sudo systemctl status breakout-bot
sudo systemctl stop breakout-bot
sudo systemctl restart breakout-bot
sudo journalctl -u breakout-bot -f
```

---

## Backup

### Backup State

```bash
cp data/breakout_signals.json data/breakout_signals.json.bak
```

### Backup Logs

```bash
cp breakout_bot.log breakout_bot.log.$(date +%Y%m%d)
```

### Backup Database

```bash
cp data/trading_performance.db data/trading_performance.db.bak
```

---

## Performance Tracking

### View Stats

```bash
python3 start.py stats
python3 start.py trades -n 20
python3 start.py equity
```

### Export Trades

```bash
python3 start.py export trades --format csv
```

---

## Contact

For issues or questions:
- Check logs first: `tail -100 breakout_bot.log`
- GitHub Issues: https://github.com/anthropics/claude-code/issues

# King Strategy Trading Bot

A crypto trading bot that scans 20 coins on Bybit for "King" patterns and automatically executes trades.

---

## Quick Start Guide (For Beginners)

### Step 1: Open Terminal
Open your terminal/command line on your computer.

### Step 2: Go to the Bot Folder
```bash
cd /home/tahae/ai-content/data/Tradingdata/bybit_bot
```

---

## How to START the Bot

```bash
nohup python3 -u -c "
from dotenv import load_dotenv
load_dotenv()
from bot import TradingBot
from config import BotConfig
TradingBot(BotConfig.from_env()).start()
" > bot.log 2>&1 &
```

After running this, you'll see something like:
```
[1] 12345
```
That number (12345) is the bot's ID (called PID).

---

## How to CHECK if Bot is Running

```bash
pgrep -f TradingBot
```

- If you see a number → Bot IS running
- If blank/empty → Bot is NOT running

---

## How to SEE What the Bot is Doing

### See the last 20 lines:
```bash
tail -20 bot.log
```

### Watch live (updates in real-time):
```bash
tail -f bot.log
```
Press `Ctrl+C` to stop watching.

### See everything:
```bash
cat bot.log
```

---

## How to STOP the Bot

### Option 1: Kill by name (easiest)
```bash
pkill -f TradingBot
```

### Option 2: Kill by PID
First find the PID:
```bash
pgrep -f TradingBot
```
Then kill it:
```bash
kill 12345
```
(Replace 12345 with the actual number)

---

## Quick Reference Card

| What you want to do | Command |
|---------------------|---------|
| Go to bot folder | `cd /home/tahae/ai-content/data/Tradingdata/bybit_bot` |
| Start the bot | See "How to START" section above |
| Check if running | `pgrep -f TradingBot` |
| See last 20 lines | `tail -20 bot.log` |
| Watch live | `tail -f bot.log` |
| Stop watching | `Ctrl+C` |
| Stop the bot | `pkill -f TradingBot` |

---

## What the Bot Does

1. **Scans 20 top coins** every 5 minutes:
   - BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, SUI
   - LTC, BCH, ATOM, UNI, APT, ARB, OP, NEAR, FIL, INJ

2. **Looks for "King" patterns** (special price patterns that predict reversals)

3. **Waits for FVG retest** (price must come back to a specific zone)

4. **Executes trades automatically** with:
   - Entry at FVG midpoint
   - Stop loss at pattern low/high
   - Take profit at pattern target

5. **Risk management**:
   - 1% risk per trade
   - Max 3 positions at once
   - 5% max daily loss limit

---

## Understanding the Log Output

```
Active signals (12):
  [APTUSDT] long_king: Entry=1.8910 R:R=23.51
```

This means:
- `APTUSDT` = the coin (Aptos)
- `long_king` = buy signal (short_king = sell signal)
- `Entry=1.8910` = will buy at $1.89
- `R:R=23.51` = reward is 23.5x the risk (very good!)

---

## Settings

Settings are in the `.env` file. To edit:
```bash
nano .env
```

Current settings:
```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false        # false = real money, true = test money
TRADING_TIMEFRAME=5        # 5 minute candles
RISK_PER_TRADE=0.01        # 1% risk per trade
```

---

## Telegram Notifications (Optional)

To get alerts on your phone when trades happen:

### Step 1: Create a Telegram Bot
1. Open Telegram app
2. Search for `@BotFather`
3. Send `/newbot`
4. Follow instructions to get your **token**

### Step 2: Get Your Chat ID
1. Message your new bot
2. Go to: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
3. Find `"chat":{"id":123456789}` - that's your **chat ID**

### Step 3: Add to .env
```bash
nano .env
```
Add these lines:
```
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Step 4: Restart the bot
```bash
pkill -f TradingBot
```
Then start it again (see "How to START" section).

---

## File Structure

```
bybit_bot/
├── bot.py           # Main bot (starts everything)
├── config.py        # Settings and coin list
├── bybit_client.py  # Connects to Bybit exchange
├── data_feed.py     # Gets price data
├── strategy.py      # Finds King patterns
├── order_manager.py # Places trades
├── notifier.py      # Telegram alerts
├── .env             # Your API keys (secret!)
├── bot.log          # Bot's diary (what it's doing)
└── CLAUDE.md        # This file
```

---

## Troubleshooting

### Bot won't start
1. Make sure you're in the right folder:
   ```bash
   cd /home/tahae/ai-content/data/Tradingdata/bybit_bot
   ```
2. Check if another bot is already running:
   ```bash
   pkill -f TradingBot
   ```
3. Try starting again

### Log file is empty
The bot buffers output. Wait 5 minutes for the first stats update.

### "No module named" error
Install the missing package:
```bash
pip install python-dotenv requests websocket-client --break-system-packages
```

### Bot stops randomly
Check the log for errors:
```bash
cat bot.log
```

---

## Safety Notes

1. **Start with small amounts** - The bot uses real money!
2. **Monitor regularly** - Check the log daily
3. **API keys are secret** - Never share your .env file
4. **Testnet first** - Set `BYBIT_TESTNET=true` to practice with fake money

---

## Current Account Info

- **Exchange**: Bybit (Mainnet)
- **Balance**: ~$271 USDT
- **Risk per trade**: $2.71 (1%)
- **Max positions**: 3 at a time

---

## Need Help?

If something goes wrong:
1. Stop the bot: `pkill -f TradingBot`
2. Check the log: `cat bot.log`
3. Look at the error message
4. Ask Claude for help!

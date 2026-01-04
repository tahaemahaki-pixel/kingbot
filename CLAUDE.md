# King Strategy Trading Bot

A crypto trading bot that scans multiple coins on Bybit for "King" patterns across multiple timeframes and automatically executes trades.

**Multi-Timeframe Support**: The bot can trade the same symbol on different timeframes simultaneously. Currently configured for:
- All 20 coins on 5-minute timeframe
- ETH also on 1-minute timeframe (total 21 setups)

---

## Strategy Overview

### King Patterns
The bot detects two pattern types:
- **Long King**: Bullish reversal pattern (buy signal)
- **Short King**: Bearish reversal pattern (sell signal)

### Pattern Structure (Long King Example)
```
A: Swing low INTO the EVWMA ribbon
C: Swing high above A (this becomes the target)
D: Close below A, then pullback into ribbon
E: Lower low below D (this defines stop loss)
F: Close above ribbon
FVG: Bullish Fair Value Gap between E and F
G: Entry when price retests the FVG
```

### Stop Loss Logic (Split Strategy)
Based on backtesting, we use different SL methods per asset:
- **ETH**: Candle Open SL (E candle open - 0.1%)
- **All other coins**: Structure SL (E swing low/high - 0.1%)

This split approach was determined by comparing both methods across multiple assets.

### Pattern Detection Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Swing Lookback | 3 | Candles on each side to confirm swing |
| EVWMA Length | 20 | Ribbon calculation period |
| Ribbon Buffer | 2% | "Into ribbon" tolerance |
| FVG Max Wait | 20 candles | Max bars to wait for FVG retest |
| Min R:R | 1.5 | Minimum risk-reward to take trade |

**Swing Lookback Testing (SPY):**
| Lookback | Swings | Trades | Win Rate | Return |
|----------|--------|--------|----------|--------|
| 3 | 10,836 | 224 | 70.5% | +6.80% |
| 5 | 6,879 | 100 | 67.0% | -1.74% |
| 10+ | <3,600 | <15 | - | Too few trades |

Lookback of 3 is optimal - finds the most patterns while maintaining good win rate.

### 300 SMA Trend Filter
The bot applies a trend filter before taking trades:
- **For Longs**: 80% of candles from A to F must be ABOVE the 300 SMA
- **For Shorts**: 80% of candles from A to F must be BELOW the 300 SMA

This filter improves results by ensuring trades align with the overall trend:
| Asset | Without Filter | With Filter | Improvement |
|-------|----------------|-------------|-------------|
| BTC 1m | -0.43R | +6.05R | +6.48R |
| Gold 5m | +3.44R | +11.10R | +7.66R |
| SPY 1m | +6.80% return | +5.16% return | Lower DD (4.9% vs 12%) |

Note: The filter reduces trade frequency but significantly reduces drawdown.

---

## Backtest Results Summary

### Timeframe Comparison
| Timeframe | Trades | Win Rate | Total P&L |
|-----------|--------|----------|-----------|
| 1-minute  | 246    | 44.7%    | +75.68R   |
| 5-minute  | 372    | 42.2%    | +324.98R  |
| 60-minute | 29     | 55.2%    | +18.27R   |
| 1-week    | 7      | 57.1%    | +25.67R   |

**5-minute is optimal** - best balance of opportunity and quality.

### Stop Loss Method Comparison (5-min)
| Asset | Structure SL | Candle Open SL | Winner |
|-------|--------------|----------------|--------|
| BTC   | +19.94R      | -162.48R       | Structure |
| ETH   | +5.51R       | +260.65R       | Candle Open |
| DOGE  | +20.35R      | +3.92R         | Structure |
| PNUT  | +21.83R      | +4.92R         | Structure |
| SPY   | +6.80%       | +0.40%         | Structure |

### SPY Backtest Results (1-min, 7 months)
| Config | Trades | Win Rate | Return | Max DD |
|--------|--------|----------|--------|--------|
| Structure SL + No Filter | 224 | 70.5% | +6.80% | 11.95% |
| Structure SL + 300 SMA | 111 | 75.7% | +5.16% | 4.90% |
| Candle Open SL + No Filter | 224 | 66.1% | +0.40% | 13.41% |

**Conclusion**: Structure SL performs better for most assets. ETH is the exception.

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

1. **Scans 21 setups** across multiple timeframes:
   - 20 coins on 5-minute: BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, SUI, LTC, BCH, ATOM, UNI, APT, ARB, OP, NEAR, FIL, INJ
   - ETH also on 1-minute (can trade independently from 5m)

2. **Looks for "King" patterns** (special price patterns that predict reversals)

3. **Applies filters**:
   - Minimum R:R of 1.5 (reward must be at least 1.5x the risk)
   - 300 SMA trend filter (80% of candles must align with trade direction)

4. **Waits for FVG retest** (price must come back to a specific zone)

5. **Executes trades automatically** with:
   - Entry at FVG midpoint (+ 0.03% buffer for better fills)
   - Stop loss: Structure-based (swing low/high) for most coins, Candle Open for ETH
   - Take profit at pattern target (Point C)

6. **Risk management**:
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
├── strategy.py      # Finds King patterns (split SL logic)
├── order_manager.py # Places trades
├── notifier.py      # Telegram alerts
├── start.py         # Entry point with single-instance check
├── setup_vps.sh     # VPS deployment script
├── .env             # Your API keys (secret!)
├── bot.log          # Bot's diary (what it's doing)
├── CLAUDE.md        # This file
└── VPS_GUIDE.md     # VPS commands quick reference
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

## VPS Deployment

The bot runs 24/7 on a DigitalOcean VPS. See `VPS_GUIDE.md` for commands.

Quick reference:
```bash
ssh root@209.38.84.47           # Connect to VPS
journalctl -u kingbot -f        # View live logs
systemctl restart kingbot       # Restart bot
```

---

## Current Setup

- **Exchange**: Bybit (Mainnet)
- **Timeframes**:
  - 5-minute candles (all 20 coins)
  - 1-minute candles (ETH only)
- **Total setups**: 21 (20 coins @ 5m + ETH @ 1m)
- **Multi-TF Independence**: Same symbol can trade on different timeframes simultaneously (e.g., ETH 5m and ETH 1m can both have open positions)
- **Filters**:
  - Min R:R: 1.5 (only takes trades with reward ≥ 1.5x risk)
  - 300 SMA Trend Filter (80% threshold)
- **Risk per trade**: 1%
- **Max positions**: 3 at a time
- **Leverage**: Max per symbol (BTC/ETH: 100x, most alts: 50x)
- **Entry buffer**: 0.03% above/below FVG midpoint
- **SL/TP Preservation**: On restart, only pending entry orders are cancelled - SL/TP orders protecting open positions are preserved
- **VPS**: DigitalOcean (209.38.84.47)

---

## Need Help?

If something goes wrong:
1. Stop the bot: `pkill -f TradingBot`
2. Check the log: `cat bot.log`
3. Look at the error message
4. Ask Claude for help!

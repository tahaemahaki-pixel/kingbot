# Double Touch Strategy Trading Bot

A crypto trading bot that scans multiple coins on Bybit for "Double Touch" patterns and automatically executes trades.

---

## Strategy Overview

### Double Touch Pattern
The bot detects counter-trend reversal patterns using a 5-step sequence with EMA ribbon confirmation.

### Pattern Types
- **Long Double Touch**: Bullish reversal (buy signal)
- **Short Double Touch**: Bearish reversal (sell signal)

### Pattern Structure (Long Example)
```
Step 0: HH (Higher High) while EMA band is GREEN
Step 1: Band goes GREY (first pullback)
Step 2: Band goes GREEN again (trend resumes)
Step 3: Band goes GREY again (second pullback) - defines SL level
Step 4: Band goes GREEN + Bullish FVG appears = ENTRY
```

### EMA Ribbon & Band Colors
The bot uses a 9/21/50 EMA ribbon to determine trend state:
- **GREEN**: Fast EMA (9) > Medium EMA (21) > Slow EMA (50) = Uptrend
- **RED**: Fast EMA (9) < Medium EMA (21) < Slow EMA (50) = Downtrend
- **GREY**: Mixed order = Consolidation/Transition

### EWVMA-200 Counter-Trend Filter
Trades are filtered using a 200-period EWVMA (Exponentially Weighted Volume Moving Average):
- **For Longs**: Price at Step 0 must be BELOW EWVMA-200 (counter-trend)
- **For Shorts**: Price at Step 0 must be ABOVE EWVMA-200 (counter-trend)

This filter ensures we trade against the longer-term trend, capturing mean reversion moves.

### Trade Levels
| Component | Calculation |
|-----------|-------------|
| Entry | FVG zone (top for longs, bottom for shorts) |
| Stop Loss | Step 3 extreme - 0.1% buffer |
| Take Profit | 3:1 Risk:Reward ratio |

### Pattern Detection Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| EMA Fast | 9 | Fast EMA for ribbon |
| EMA Medium | 21 | Medium EMA for ribbon |
| EMA Slow | 50 | Slow EMA for ribbon |
| HH/LL Lookback | 20 | Candles to detect Higher Highs/Lower Lows |
| EWVMA Length | 200 | Counter-trend filter period |
| FVG Max Wait | 20 candles | Max bars to wait for FVG retest |
| SL Buffer | 0.1% | Buffer beyond Step 3 for stop loss |
| Risk:Reward | 3.0 | Target R:R ratio |

---

## Backtest Results

### Asset Performance (2% risk per trade, $10k starting)
| Asset | Trades | Return | Max Drawdown | Win Rate |
|-------|--------|--------|--------------|----------|
| BTC | 20 | +25.07% | 9.61% | 40% |
| ETH | 21 | +4.77% | 13.19% | 28.6% |
| Gold | 57 | +29.81% | 22.26% | 31.6% |

### Comparison vs King Strategy (with EWVMA filter)
| Asset | Double Touch | King (EWVMA) |
|-------|--------------|--------------|
| BTC | +25.07% | +15.17% |
| ETH | +4.77% | -6.02% |
| Gold | +29.81% | +18.17% |
| **Total** | **+59.65%** | **+27.32%** |

Double Touch outperforms on all tested assets.

---

## Quick Start Guide

### Step 1: Go to Bot Folder
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

---

## How to CHECK if Bot is Running

```bash
pgrep -f TradingBot
```

- If you see a number: Bot IS running
- If blank: Bot is NOT running

---

## How to SEE What the Bot is Doing

```bash
tail -20 bot.log    # Last 20 lines
tail -f bot.log     # Watch live (Ctrl+C to stop)
cat bot.log         # See everything
```

---

## How to STOP the Bot

```bash
pkill -f TradingBot
```

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| Go to bot folder | `cd /path/to/bybit_bot` |
| Start the bot | See "How to START" section |
| Check if running | `pgrep -f TradingBot` |
| See last 20 lines | `tail -20 bot.log` |
| Watch live | `tail -f bot.log` |
| Stop the bot | `pkill -f TradingBot` |

---

## What the Bot Does

1. **Scans 20 crypto symbols** on 5-minute timeframe:
   BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, SUI, LTC, BCH, ATOM, UNI, APT, ARB, OP, NEAR, FIL, INJ

2. **Detects Double Touch patterns** using the 5-step sequence

3. **Applies filters**:
   - EMA ribbon band color confirmation
   - EWVMA-200 counter-trend filter

4. **Waits for FVG retest** (price must enter the Fair Value Gap zone)

5. **Executes trades automatically** with:
   - Entry at FVG boundary
   - Stop loss at Step 3 extreme - 0.1%
   - Take profit at 3:1 R:R

6. **Risk management**:
   - 1% risk per trade
   - Max 5 positions total (3 crypto + 2 non-crypto)
   - 5% max daily loss limit

---

## Position Limits

| Type | Max Positions |
|------|---------------|
| Crypto | 3 |
| Non-Crypto (Gold, indices) | 2 |
| **Total** | **5** |

---

## Understanding the Log Output

```
Active signals (12):
  [APTUSDT_5] long_double_touch: Entry=1.8910 R:R=3.00
```

- `APTUSDT_5` = Symbol @ Timeframe
- `long_double_touch` = Buy signal
- `Entry=1.8910` = Entry price
- `R:R=3.00` = Risk:Reward ratio

---

## Settings (.env file)

```bash
nano .env
```

```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false        # false = real money
TRADING_TIMEFRAME=5        # 5 minute candles
RISK_PER_TRADE=0.01        # 1% risk per trade
```

---

## Telegram Notifications

### Setup
1. Create bot with @BotFather, get token
2. Message your bot, get chat ID from: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
3. Add to .env:
```
TELEGRAM_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```
4. Restart bot

---

## File Structure

```
bybit_bot/
├── bot.py                    # Main bot orchestrator
├── config.py                 # Settings and coin list
├── bybit_client.py           # Bybit API client
├── data_feed.py              # Price data + indicators
├── double_touch_strategy.py  # Double Touch pattern detection
├── order_manager.py          # Trade execution
├── notifier.py               # Telegram alerts
├── start.py                  # Entry point
├── .env                      # API keys (secret!)
├── bot.log                   # Bot logs
└── CLAUDE.md                 # This file
```

---

## Current Configuration

- **Exchange**: Bybit (Mainnet)
- **Timeframe**: 5-minute candles
- **Symbols**: 20 crypto coins
- **Strategy**: Double Touch (counter-trend)
- **EMA Ribbon**: 9/21/50
- **Filter**: EWVMA-200 counter-trend
- **Risk:Reward**: 3:1
- **Risk per trade**: 1%
- **Max positions**: 5 (3 crypto + 2 non-crypto)
- **Leverage**: Max per symbol (BTC/ETH: 100x, most alts: 50x)

---

## VPS Deployment

```bash
ssh root@209.38.84.47           # Connect to VPS
journalctl -u doubletouchbot -f # View live logs
systemctl restart doubletouchbot # Restart bot
```

---

## Performance Tracking CLI

The bot includes a comprehensive performance tracking system with SQLite storage and CLI interface.

### CLI Commands

| Command | Description |
|---------|-------------|
| `python start.py stats` | Show overall trading statistics |
| `python start.py stats today` | Today's stats only |
| `python start.py stats week --symbol BTCUSDT` | Weekly stats for BTC |
| `python start.py trades -n 20` | Last 20 trades |
| `python start.py trades --winners` | Winners only |
| `python start.py trades --losers` | Losers only |
| `python start.py equity` | Equity curve and drawdowns |
| `python start.py assets --sort pnl` | Per-asset breakdown |
| `python start.py sessions -n 10` | Best/worst trading days |
| `python start.py time` | Performance by hour/day of week |
| `python start.py streaks` | Win/loss streak analysis |
| `python start.py export trades --format csv` | Export to CSV |

### Example Output: Stats
```
============================================================
TRADING PERFORMANCE STATISTICS
============================================================
Total Trades:                         45
Win Rate:                          42.2%
Profit Factor:                      1.85
Net P&L:                       +$567.89
Max Drawdown:                     12.5%
Avg R-Multiple:                  +0.45R
============================================================
```

### Database Location
```
data/trading_performance.db
```

---

## Safety Notes

1. **Start with small amounts** - Uses real money!
2. **Monitor regularly** - Check logs daily
3. **API keys are secret** - Never share .env file
4. **Testnet first** - Set `BYBIT_TESTNET=true` to practice

---

## Need Help?

1. Stop the bot: `pkill -f TradingBot`
2. Check the log: `cat bot.log`
3. Look at error messages
4. Ask Claude for help!

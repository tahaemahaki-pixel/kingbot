# Double Touch Strategy Trading Bot

A crypto trading bot that scans multiple coins on Bybit for "Double Touch" patterns and automatically executes trades.

---

## Project Priorities

### Spread Trading - MR Double Touch Strategy
We successfully adapted the Double Touch strategy for **pairs/spread trading** on cointegrated assets.

- **Status:** Backtested & Ready (`mr_double_touch.py`)
- **Pair Tested:** BTC/ETH spread (cointegrated, p=0.001)
- **Results:** 88.2% win rate, 4.90 profit factor, 61.4% return
- **Reference:** See `SPREAD_TRADING_RESULTS.md` for full analysis

**Key Finding:** Original Double Touch doesn't work on mean-reverting spreads (EMA ribbon too slow). We created **Mean-Reversion Double Touch** - same concept, inverted logic.

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

**Simple method (bot.py now loads .env automatically):**
```bash
cd /root/kingbot
nohup python3 -u bot.py > double_touch_bot.log 2>&1 &
```

**Alternative method (explicit dotenv load):**
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
├── bot.py                    # Double Touch bot orchestrator
├── config.py                 # Settings and coin list
├── bybit_client.py           # Bybit API client
├── data_feed.py              # Price data + indicators
├── double_touch_strategy.py  # Double Touch pattern detection
├── order_manager.py          # Trade execution
├── notifier.py               # Telegram alerts
├── start.py                  # Entry point
├── trade_tracker.py          # Performance tracking (SQLite)
├── performance_cli.py        # CLI for stats/trades/equity
├── .env                      # API keys (secret!)
├── bot.log                   # Bot logs
├── CLAUDE.md                 # This file
├── PERFORMANCE_CLI.md        # CLI documentation
│
├── # Breakaway Strategy (ACTIVE)
├── breakaway_bot.py          # Breakaway bot orchestrator
├── breakaway_strategy.py     # Breakaway signal detection & indicators
├── symbol_scanner.py         # Top 50 coin fetcher by volume
│
├── # Spread Trading
├── spread_strategy.py        # MR Double Touch pattern detection for spreads
├── spread_scanner.py         # Multi-pair dynamic cointegration scanner
├── spread_config.py          # Spread bot configuration
│
├── # Spread Trading Research
├── spread_analysis.py        # Cointegration analysis & z-score backtest
├── mr_double_touch.py        # Mean-Reversion Double Touch backtest
├── spread_double_touch.py    # Original DT on spread (doesn't work)
└── SPREAD_TRADING_RESULTS.md # Full spread trading analysis
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
cd /root/kingbot                # Go to bot folder
```

---

## Spread Scanner Bot (Multi-Pair MR Double Touch)

The spread scanner trades multiple cointegrated pairs using the MR Double Touch strategy with dynamic cointegration checking.

### Pairs Monitored
| Pair | Assets | Mode |
|------|--------|------|
| ETH/BTC | ETHUSDT / BTCUSDT | 🔒 Always Active |
| SOL/ETH | SOLUSDT / ETHUSDT | Dynamic (p < 0.05) |
| SOL/BTC | SOLUSDT / BTCUSDT | Dynamic (p < 0.05) |

### How it Works
1. **Dynamic Cointegration**: Checks p-value every 500 candles (~42 hours)
2. **Auto Enable/Disable**: Pairs activate when p < 0.05, disable when p > 0.15
3. **ETH/BTC Always On**: Trades regardless of cointegration status
4. **MR Double Touch Pattern**:
   - **Phase 0**: Wait for z < -2.0 or z > 2.0 (first extreme)
   - **Phase 1**: Wait for recovery (z > -1.0 or z < 1.0)
   - **Phase 2**: Wait for second touch (z < -1.5 or z > 1.5) → **ENTRY**
5. **Dual-leg execution**: Buy one asset + Sell the other simultaneously
6. **Exits**: TP at z = ±0.5, SL at z = ±4.0

### Start Spread Scanner

```bash
cd /root/kingbot
pkill -f TradingBot  # Stop any existing bot first

nohup python3 -u -c "
import os
os.environ['SPREAD_TRADING_ENABLED'] = 'true'
from dotenv import load_dotenv
load_dotenv()
os.environ['SPREAD_TRADING_ENABLED'] = 'true'
from bot import TradingBot
from config import BotConfig
config = BotConfig.from_env()
config.spread_trading_enabled = True
config.risk_per_trade = 0.05  # 5% risk per trade
TradingBot(config).start()
" > spread_bot.log 2>&1 &
```

### Monitor Spread Scanner

| Action | Command |
|--------|---------|
| Check if running | `pgrep -f TradingBot` |
| Watch live | `tail -f spread_bot.log` |
| Last 50 lines | `tail -50 spread_bot.log` |
| Stop the bot | `pkill -f TradingBot` |

### Understanding Scanner Output

```
==================================================
SPREAD SCANNER STATUS
==================================================
ETH_BTC: 🔒 ALWAYS ON p=0.4625 z=0.96 phase=1
SOL_ETH: ❌ inactive p=0.1995
SOL_BTC: ❌ inactive p=0.2688

Next cointegration check in 450 candles
==================================================
```

- **🔒 ALWAYS ON**: Pair trades regardless of p-value
- **✅ ACTIVE**: Pair enabled (p < 0.05)
- **❌ inactive**: Pair disabled (p > 0.05)
- **z=X.XX**: Current z-score
- **phase=N**: Pattern phase (0=waiting, 1=extreme seen, 2=recovery)

### Spread Scanner Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Check Interval | 500 candles | Cointegration recheck frequency |
| Enable Threshold | p < 0.05 | P-value to activate pair |
| Disable Threshold | p > 0.15 | P-value to deactivate pair |
| First Extreme | z = ±2.0 | Entry to pattern |
| Recovery | z = ±1.0 | Partial mean reversion |
| Second Touch | z = ±1.5 | Entry trigger |
| Take Profit | z = ±0.5 | Close near mean |
| Stop Loss | z = ±4.0 | Max adverse move |
| Risk per Trade | 5% | Position sizing |

### Future Improvements
- [ ] Add max concurrent trades limit to prevent over-exposure when multiple pairs trigger

### Switch Between Bots

**To run Double Touch bot (normal):**
```bash
pkill -f TradingBot
nohup python3 -u -c "
from dotenv import load_dotenv
load_dotenv()
from bot import TradingBot
from config import BotConfig
TradingBot(BotConfig.from_env()).start()
" > bot.log 2>&1 &
```

**To run Spread Scanner bot:**
```bash
pkill -f TradingBot
# Use the spread scanner start command above
```

---

## Breakaway Strategy Bot (ACTIVE)

The Breakaway bot trades counter-trend FVG setups across **50 symbols** using volume spikes and Tai Index confirmation.

### Strategy Overview

**Entry Conditions (ALL required):**

| Condition | SHORT | LONG |
|-----------|-------|------|
| FVG | Bearish (gap down) | Bullish (gap up) |
| EWVMA Cradle | 3+ of 5 candles within EWVMA(20) bands | Same |
| Volume Spike | >= 2.5x 20-period average | Same |
| Tai Index | > 55 (overbought) | < 45 (oversold) |
| Trend Filter | Price > EWVMA-200 | Price < EWVMA-200 |

**Exit Rules:**
- Stop Loss: FVG boundary + 0.1% buffer
- Take Profit: 3:1 R:R ratio

### Backtest Results

| Direction | Win Rate | Expectancy |
|-----------|----------|------------|
| Shorts | 76-93% | +2.0R |
| Longs | 53-55% | +1.1R |

SOL 5min shorts: 92.9% WR, +2.71R expectancy

### Start Breakaway Bot

```bash
ssh root@209.38.84.47
cd /root/kingbot

# Stop any existing bot
pkill -f breakaway_bot
pkill -f TradingBot

# Start Breakaway bot
nohup python3 -u breakaway_bot.py > breakaway_bot.log 2>&1 &
```

### Monitor Breakaway Bot

| Action | Command |
|--------|---------|
| Check if running | `pgrep -f breakaway_bot` |
| Watch live | `tail -f breakaway_bot.log` |
| Last 50 lines | `tail -50 breakaway_bot.log` |
| Stop the bot | `pkill -f breakaway_bot` |

### Understanding Breakaway Output

```
============================================================
BREAKAWAY BOT STATUS - 12:05:15
============================================================
Symbols: 50
Active feeds: 46
Open positions: 0/5
Total signals: 0
Executed: 0
Balance: $215.41
Equity: $215.41
============================================================
```

When a signal triggers:
```
============================================================
NEW BREAKAWAY SIGNAL - SOLUSDT
============================================================
  Direction: SHORT
  Entry: 187.450000
  Stop Loss: 188.637450
  Target: 183.887850
  R:R: 3.0
  Volume: 3.2x
  Tai Index: 67
  Cradle: 4/5
============================================================
```

### Breakaway Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Symbols | Top 50 by volume | Dynamically fetched |
| Priority Symbols | SOL, BTC, PNUT, DOGE | Always included |
| Timeframe | 5 minutes | Default |
| EWVMA Length | 20 | Cradle detection |
| EWVMA Trend | 200 | Counter-trend filter |
| Volume Threshold | 2.5x | Minimum spike |
| Tai Short | > 55 | Overbought for shorts |
| Tai Long | < 45 | Oversold for longs |
| Risk per Trade | 2% | Position sizing |
| Max Positions | 5 | Concurrent trades |
| Risk:Reward | 3:1 | Target ratio |
| Historical Candles | 1000 | Loaded on startup |

### Breakaway Files

| File | Description |
|------|-------------|
| `breakaway_bot.py` | Main bot orchestrator |
| `breakaway_strategy.py` | Signal detection & indicators |
| `symbol_scanner.py` | Top 50 coin fetcher |

### Environment Variables

Add to `.env` for custom configuration:
```
BREAKAWAY_PRIORITY_SYMBOLS=SOLUSDT,BTCUSDT,PNUTUSDT,DOGEUSDT
BREAKAWAY_MAX_SYMBOLS=50
BREAKAWAY_DIRECTION=both
BREAKAWAY_MAX_POSITIONS=5
BREAKAWAY_RISK_PER_TRADE=0.02
BREAKAWAY_MIN_VOL_RATIO=2.5
BREAKAWAY_TAI_SHORT=55.0
BREAKAWAY_TAI_LONG=45.0
BREAKAWAY_RISK_REWARD=3.0
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

## Changelog / Fixes

### 2026-01-05: API Authentication & Reliability Fixes

**Problem:** Bot was getting "Position sync error: Empty response from API" and 401 Unauthorized errors.

**Root Cause:** When running `python3 bot.py` directly, the `.env` file wasn't being loaded, so `BYBIT_TESTNET` defaulted to `true`. This caused the bot to use testnet URLs (`api-testnet.bybit.com`) with mainnet API keys, resulting in 401 errors.

**Fixes Applied:**

1. **Added `load_dotenv()` to bot.py** (line 10-13)
   - Bot now loads `.env` automatically when run directly
   - No longer need the wrapper script with explicit `load_dotenv()`

2. **Added API retry logic with exponential backoff** (`bybit_client.py`)
   - 3 retries per request with 0.5s, 1s, 1.5s delays
   - Handles transient network failures gracefully

3. **Added rate limiting** (`bybit_client.py`)
   - 500ms minimum between API requests
   - Thread-safe with locking to prevent concurrent request issues
   - Prevents hitting Bybit rate limits when running multiple bots

4. **Switched to fresh requests per API call**
   - Replaced `self.session.get/post` with `requests.get/post`
   - Avoids session state issues in multi-threaded environment

5. **Removed leverage calls during startup**
   - Previously set leverage for all 20 symbols on startup (20 API calls)
   - Leverage is usually already set, so this was unnecessary API load

**Running Both Bots Simultaneously:**

Both Double Touch and Spread Scanner can now run at the same time:
```bash
# Terminal 1: Double Touch bot
cd /root/kingbot
nohup python3 -u bot.py > double_touch_bot.log 2>&1 &

# Terminal 2: Spread Scanner bot
nohup python3 -u -c "
import os
os.environ['SPREAD_TRADING_ENABLED'] = 'true'
from dotenv import load_dotenv
load_dotenv()
from bot import TradingBot
from config import BotConfig
config = BotConfig.from_env()
config.spread_trading_enabled = True
config.risk_per_trade = 0.02
TradingBot(config).start()
" > spread_bot.log 2>&1 &
```

**Verify both are running:**
```bash
ps aux | grep python | grep -v grep | grep -E 'bot.py|TradingBot'
```

---

## Need Help?

1. Stop the bot: `pkill -f TradingBot`
2. Check the log: `cat bot.log`
3. Look at error messages
4. Ask Claude for help!

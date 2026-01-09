# Bybit Trading Bot

A crypto trading bot that scans multiple coins on Bybit and automatically executes trades.

**Active Strategy:** Breakout Optimized (swing high breakout with ATR trailing stops)

**Legacy Strategies:** Double Touch, Breakaway FVG, Spread Trading

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
├── config.py                 # Settings (BotConfig, BreakawayConfig, BreakoutConfig)
├── bybit_client.py           # Bybit API + WebSocket client
├── data_feed.py              # Price data + indicators
├── double_touch_strategy.py  # Double Touch pattern detection
├── order_manager.py          # Trade execution + position management
├── notifier.py               # Telegram alerts
├── start.py                  # Entry point
├── trade_tracker.py          # Performance tracking (SQLite)
├── performance_cli.py        # CLI for stats/trades/equity
├── .env                      # API keys (secret!)
├── bot.log                   # Bot logs
├── CLAUDE.md                 # This file
├── PERFORMANCE_CLI.md        # CLI documentation
│
├── # Breakout Optimized Strategy (ACTIVE)
├── breakaway_bot.py          # Main bot orchestrator (uses breakout strategy)
├── breakout_strategy.py      # Breakout signal detection, ATR trailing stops
├── symbol_scanner.py         # Top 50 coin fetcher by volume
├── sync_trades.sh            # Sync trade CSVs from VPS to local
│
├── data/
│   ├── trading_performance.db  # SQLite trade history
│   └── breakout_signals.json   # Persisted signal state
│
├── exports/                  # Trade CSV exports (synced from VPS)
│   ├── trades_latest.csv     # Most recent export
│   └── trades_YYYY-MM-DD.csv # Daily exports
│
├── # Legacy Strategies
├── breakaway_strategy.py     # Breakaway FVG signal detection (legacy)
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

## Scalping Strategy Bot (NEW) - FVG Breakout with Partial Exits

High-frequency scalping bot for BTC, ETH, SOL targeting **9+ trades/day** with **85%+ win rate**.

### Key Differences from Breakaway

| Aspect | Breakaway | **Scalping** |
|--------|-----------|--------------|
| Win Rate | 60-70% | **85-90%** |
| R:R Ratio | 3:1 (full) | **1.5:1 (partial)** |
| Avg Winner | 3.0R | **1.25R** |
| Expectancy | +0.8-1.0R | **+0.8R** |
| Trades/Day | ~3 | **~9** |
| Hold Time | 60-80 min | **~25 min** |
| Exit System | Single TP | **Partial (50%@1R, 50%@1.5R)** |

### Configuration

| Parameter | Value |
|-----------|-------|
| **Symbols** | BTCUSDT, ETHUSDT, SOLUSDT |
| **Timeframe** | 5-minute |
| **Risk per trade** | 0.5% |
| **Max positions** | 5 |
| **Volume filter** | ≥1.5x average |
| **TP1** | 1.0R (close 50%) |
| **TP2** | 1.5R (close remaining) |
| **Move SL to BE** | Yes, after TP1 |
| **Max hold** | 30 candles |
| **Cooldown** | 5 candles after exit |

### Entry Conditions (ALL Required)

| Condition | SHORT | LONG |
|-----------|-------|------|
| FVG | Bearish: `high < low[-2]` | Bullish: `low > high[-2]` |
| EWVMA Cradle | 3+ of 5 candles in bands | Same |
| Volume Spike | ≥ 1.5x 20-period average | Same |
| Imbalance | ≤ -0.10 (selling) | ≥ +0.10 (buying) |

### Exit System

```
Position Entry (100%)
        │
        ├──► TP1 @ 1.0R reached
        │    ├── Close 50% of position
        │    └── Move Stop Loss to Entry (breakeven)
        │
        └──► TP2 @ 1.5R reached
             └── Close remaining 50%

Average realized per full winner: 1.25R
```

### Backtest Results (FVG Breakout Only)

| Asset | Trades | Win Rate | Expectancy | Trades/Day |
|-------|--------|----------|------------|------------|
| BTC | 1,751 | 84.8% | +0.784R | 6.3 |
| ETH | 427 | 88.8% | +0.870R | 6.1 |
| SOL | 429 | 86.5% | +0.849R | 6.2 |
| **Total** | **2,607** | **85.7%** | **+0.809R** | **9.3** |

### Start Scalping Bot

```bash
cd /root/kingbot
pkill -f scalp_bot  # Stop any existing instance

nohup python3 -u scalp_bot.py > scalp_bot.log 2>&1 &
```

### Monitor Scalping Bot

| Action | Command |
|--------|---------|
| Check if running | `pgrep -f scalp_bot` |
| Watch live | `tail -f scalp_bot.log` |
| Last 50 lines | `tail -50 scalp_bot.log` |
| Stop the bot | `pkill -f scalp_bot` |

### Scalping Files

| File | Description |
|------|-------------|
| `scalp_bot.py` | Main bot orchestrator |
| `scalp_strategy.py` | Signal detection with partial exits |
| `scalp_config.py` | Configuration dataclass |
| `backtests/backtest_scalp.py` | Backtest validation |
| `SCALPING_STRATEGY.md` | Full strategy documentation |

### Environment Variables

```bash
SCALP_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
SCALP_TIMEFRAME=5
SCALP_MIN_VOL_RATIO=1.5
SCALP_IMBALANCE_THRESHOLD=0.10
SCALP_TP1_R=1.0
SCALP_TP2_R=1.5
SCALP_TP1_CLOSE_PCT=0.50
SCALP_MOVE_SL_TO_BE=true
SCALP_MAX_HOLD=30
SCALP_RISK_PER_TRADE=0.005
SCALP_MAX_POSITIONS=5
SCALP_COOLDOWN=5
SCALP_DIRECTION=both
```

---

## Breakout Optimized Strategy Bot (ACTIVE) - 5-Minute Scanner

The Breakout Optimized bot trades **swing high breakouts** on the 5-minute timeframe with ATR trailing stops. This strategy showed **+3,135R** in backtesting with 69.4% win rate.

### Strategy Overview

**Entry Conditions (ALL required):**

| Condition | LONG |
|-----------|------|
| Price Action | Break above most recent pivot high |
| EVWMA Filter | Close > upper EVWMA(20) band (strong uptrend) |
| Volume Spike | ≥ 2.0x 20-period average (toggleable) |
| Volume Imbalance | ≥ +0.10 buying pressure (toggleable) |

**Exit Rules:**
- Initial Stop: Entry - (ATR(14) * 2.0)
- Trailing Stop: Moves up with price on candle close
- Emergency TP: 10R circuit breaker (if bot crashes)

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Symbols** | 45 pairs | Top by volume + priority |
| **Timeframe** | 5-minute | Candle close trigger |
| **Risk per trade** | 1% | Position sizing |
| **Max positions** | 5 | Filled positions limit |
| **Pending orders** | Unlimited | Cancel when 5 fill |
| **ATR period** | 14 | Trailing stop calculation |
| **ATR multiplier** | 2.0 | Stop distance |
| **Emergency TP** | 10R | Circuit breaker |

### State Persistence & Crash Recovery

The bot persists active signals to disk for crash recovery:

| Feature | Description |
|---------|-------------|
| **State file** | `data/breakout_signals.json` |
| **Save trigger** | Every signal update |
| **Load on startup** | Restores active signals |
| **Orphan cleanup** | Cancels stale limit orders |
| **Position sync** | Matches signals to open positions |

### Position Management

The bot allows **unlimited pending limit orders** but cancels all unfilled orders when 5 positions are filled:

```
Signal 1 → Limit order placed (pending)
Signal 2 → Limit order placed (pending)
...
Signal 5 fills → 5th position opened
Signal 6 fires → Check filled count (5) → Cancel all pending orders
```

### Start Breakout Bot

```bash
ssh root@209.38.84.47
cd /root/kingbot

# Via systemd (recommended - auto-restart)
sudo systemctl restart breakout-bot
sudo systemctl status breakout-bot

# Or manually
pkill -f breakaway_bot || true
nohup python3 -u breakaway_bot.py > breakout_bot.log 2>&1 &
```

### Monitor Breakout Bot

| Action | Command |
|--------|---------|
| Service status | `sudo systemctl status breakout-bot` |
| View logs | `journalctl -u breakout-bot -f` |
| Check if running | `pgrep -f breakaway_bot` |
| Watch log file | `tail -f breakout_bot.log` |
| Stop the bot | `sudo systemctl stop breakout-bot` |

### Understanding Breakout Output

```
============================================================
BREAKOUT BOT STATUS - 13:27:45
============================================================

5-MIN TIMEFRAME:
  Symbols: 45
  Filled Positions: 3/5
  Pending Orders: 7
  Signals: 10 | Executed: 3

ACCOUNT:
  Balance: $215.46
  Equity: $218.92
  Total Open: 3
============================================================
```

When a signal triggers:
```
============================================================
[5-MIN] NEW BREAKOUT SIGNAL - SOLUSDT
============================================================
  Direction: LONG
  Entry: 187.450000 (limit at swing high)
  Stop Loss: 185.012550 (ATR trailing)
  Emergency TP: 211.825500 (10R circuit breaker)
  Volume: 3.2x
  Imbalance: +0.25
============================================================
```

### Breakout Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pivot Left/Right | 3 | Swing high detection lookback |
| EVWMA Period | 20 | Trend bands |
| ATR Period | 14 | Stop calculation |
| ATR Multiplier | 2.0 | Stop distance (2x ATR) |
| Volume Threshold | 2.0x | Min volume spike |
| Imbalance Threshold | ±0.10 | Volume delta imbalance |
| Emergency TP | 10R | Circuit breaker |
| State File | `data/breakout_signals.json` | Persistence |

### Breakout Environment Variables

```bash
# Core settings
BREAKOUT_TIMEFRAME=5
BREAKOUT_MAX_POSITIONS=5
BREAKOUT_RISK_PER_TRADE=0.01
BREAKOUT_MAX_SYMBOLS=45
BREAKOUT_PRIORITY_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,PNUTUSDT,INJUSDT

# Pivot detection
BREAKOUT_PIVOT_LEFT=3
BREAKOUT_PIVOT_RIGHT=3

# EVWMA bands
BREAKOUT_EVWMA_PERIOD=20

# ATR trailing stops
BREAKOUT_ATR_PERIOD=14
BREAKOUT_ATR_MULTIPLIER=2.0

# Volume filters (toggleable)
BREAKOUT_USE_VOLUME_FILTER=true
BREAKOUT_MIN_VOL_RATIO=2.0
BREAKOUT_VOLUME_AVG_PERIOD=20

# Imbalance filter (toggleable)
BREAKOUT_USE_IMBALANCE_FILTER=true
BREAKOUT_IMBALANCE_THRESHOLD=0.10
BREAKOUT_IMBALANCE_LOOKBACK=10

# Emergency take profit
BREAKOUT_EMERGENCY_TP_MULTIPLIER=10.0
```

### Breakout Files

| File | Description |
|------|-------------|
| `breakaway_bot.py` | Main bot orchestrator (now uses breakout strategy) |
| `breakout_strategy.py` | Signal detection, indicators, trailing stops |
| `config.py` | BreakoutConfig dataclass |
| `data/breakout_signals.json` | Persisted signal state |

---

## Trade Export System

### Daily CSV Export

Trades are automatically exported to CSV daily at midnight UTC via cron job on VPS.

**VPS Export Script:** `/root/kingbot/export_trades.py`
**Local Sync Script:** `sync_trades.sh`
**Export Location (VPS):** `/root/kingbot/exports/`
**Export Location (Local):** `exports/`

### Sync Trades Locally

```bash
cd /home/tahae/ai-content/data/Tradingdata/bybit_bot
./sync_trades.sh
```

This will:
1. SSH to VPS and run export_trades.py
2. Rsync CSV files to local `exports/` folder
3. Show trade count

### Cron Job (VPS)

```bash
# View cron jobs
crontab -l

# Runs at midnight UTC daily
0 0 * * * cd /root/kingbot && /usr/bin/python3 export_trades.py >> /root/kingbot/export.log 2>&1
```

### Export Files

| File | Description |
|------|-------------|
| `trades_latest.csv` | Most recent export (symlinked) |
| `trades_YYYY-MM-DD.csv` | Daily timestamped exports |

---

## Systemd Services

### Breakout Bot Service

The bot runs as a systemd service with auto-restart on crash.

**Service File:** `/etc/systemd/system/breakout-bot.service`

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

### Service Commands

| Action | Command |
|--------|---------|
| Start bot | `sudo systemctl start breakout-bot` |
| Stop bot | `sudo systemctl stop breakout-bot` |
| Restart bot | `sudo systemctl restart breakout-bot` |
| Check status | `sudo systemctl status breakout-bot` |
| View logs | `journalctl -u breakout-bot -f` |
| Enable on boot | `sudo systemctl enable breakout-bot` |

---

## Breakaway Strategy Bot (LEGACY) - 5-Minute Scanner

The Breakaway bot trades counter-trend FVG setups on the **5-minute timeframe** using volume spikes and **Volume Delta Imbalance** confirmation.

### Configuration (Updated 2026-01-08)

| Parameter | Value |
|-----------|-------|
| **Symbols** | 45 pairs |
| **Risk per trade** | 1% |
| **Max positions** | 5 |
| **Volume filter** | ≥2.0x average |
| **Candles preload** | 2000 |
| **Scan frequency** | Every 5-min candle close |

**Note:** 1-minute trading was disabled based on backtest results showing 5-min significantly outperforms (1.13R vs 0.33R expectancy).

### Pairs Being Scanned (45 Total)

**Priority Symbols:** `SOLUSDT, BTCUSDT, PNUTUSDT, DOGEUSDT`

| # | Symbol | # | Symbol | # | Symbol |
|---|--------|---|--------|---|--------|
| 1 | SOLUSDT | 16 | APTUSDT | 31 | FTMUSDT |
| 2 | BTCUSDT | 17 | ARBUSDT | 32 | SANDUSDT |
| 3 | PNUTUSDT | 18 | OPUSDT | 33 | MANAUSDT |
| 4 | DOGEUSDT | 19 | NEARUSDT | 34 | AXSUSDT |
| 5 | ETHUSDT | 20 | FILUSDT | 35 | GALAUSDT |
| 6 | XRPUSDT | 21 | INJUSDT | 36 | TRXUSDT |
| 7 | ADAUSDT | 22 | MATICUSDT | 37 | APEUSDT |
| 8 | AVAXUSDT | 23 | AAVEUSDT | 38 | LDOUSDT |
| 9 | LINKUSDT | 24 | MKRUSDT | 39 | RNDRUSDT |
| 10 | DOTUSDT | 25 | COMPUSDT | 40 | GMXUSDT |
| 11 | SUIUSDT | 26 | ETCUSDT | 41 | WIFUSDT |
| 12 | LTCUSDT | 27 | ALGOUSDT | 42 | 1000PEPEUSDT |
| 13 | BCHUSDT | 28 | XLMUSDT | 43 | 1000FLOKIUSDT |
| 14 | ATOMUSDT | 29 | VETUSDT | 44 | 1000BONKUSDT |
| 15 | UNIUSDT | 30 | ICPUSDT | 45 | JUPUSDT |

**Note:** Meme coins use `1000XXXUSDT` format on Bybit perpetuals (PEPE, FLOKI, BONK)

### Strategy Overview

**Entry Conditions (ALL required):**

| Condition | SHORT | LONG |
|-----------|-------|------|
| FVG | Bearish (gap down) | Bullish (gap up) |
| EWVMA Cradle | 3+ of 5 candles within EWVMA(20) bands | Same |
| Volume Spike | ≥ 2.0x 20-period average | Same |
| **Imbalance** | ≤ -0.10 (selling pressure) | ≥ +0.10 (buying pressure) |

**Note:** Tai Index and EWVMA-200 trend filters were replaced by Volume Delta Imbalance on 2026-01-08 based on backtest results showing +26% improvement in expectancy.

**Exit Rules:**
- Stop Loss: FVG boundary + 0.1% buffer
- Take Profit: 3:1 R:R ratio

### Backtest Results (5-Minute) - Imbalance Filter

**Live Bybit Data (17 days, 5000 candles per symbol):**

| Symbol | Trades | Win Rate | Total R | Expectancy |
|--------|--------|----------|---------|------------|
| BTCUSDT | 45 | 53.3% | +51R | +1.13R |
| ETHUSDT | 55 | 47.3% | +49R | +0.89R |
| SOLUSDT | 47 | 44.7% | +37R | +0.79R |
| **TOTAL** | **147** | **48.3%** | **+137R** | **+0.93R** |

- Avg trades/day: ~2.9 per symbol (~8.6 total)
- Avg hold time: 60-80 minutes
- Long/Short split: ~50/50

**Historical comparison (Tai+Trend vs Imbalance filter):**

| Metric | Before (Tai+Trend) | After (Imbalance) | Change |
|--------|-------------------|-------------------|--------|
| Trades | 314 | ~3,139 | +10x |
| Win Rate | 52.2% | ~59.3% | +7% |
| Expectancy | +1.08R | ~1.36R | +26% |

See `volume charts/BACKTEST_RESULTS.md` for full analysis.

### Start Breakaway Bot

```bash
ssh root@209.38.84.47
cd /root/kingbot

# Bot runs as systemd service
systemctl restart kingbot
systemctl status kingbot
```

Or manually:
```bash
pkill -f breakaway_bot
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

When a signal triggers:
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
  Imbalance: -0.25
============================================================
```

### Breakaway Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Priority Symbols | SOL, BTC, PNUT, DOGE | Always included |
| Max Symbols | 45 | Top coins by 24h volume |
| Risk per Trade | 1% | Per trade risk |
| Max Positions | 5 | Concurrent positions |
| Volume Threshold | 2.0x | Min volume spike |
| EWVMA Length | 20 | Cradle detection |
| **Imbalance Threshold** | ±0.10 | Volume delta imbalance |
| **Imbalance Lookback** | 10 | Candles for imbalance calc |
| Risk:Reward | 3:1 | Target ratio |
| Candles Preload | 2000 | Historical data loaded |

**Disabled Filters (2026-01-08):** Tai Index and EWVMA-200 trend filters are now OFF by default.

### Breakaway Files

| File | Description |
|------|-------------|
| `breakaway_bot.py` | Main bot orchestrator (5-min only) |
| `breakaway_strategy.py` | Signal detection & indicators |
| `symbol_scanner.py` | Top coin fetcher by volume |

### Environment Variables

Add to `.env` for custom configuration:
```
BREAKAWAY_PRIORITY_SYMBOLS=SOLUSDT,BTCUSDT,PNUTUSDT,DOGEUSDT
BREAKAWAY_DIRECTION=both
BREAKAWAY_RISK_REWARD=3.0
BREAKAWAY_CANDLES_PRELOAD=2000
BREAKAWAY_MAX_SYMBOLS=45
BREAKAWAY_MAX_POSITIONS=5
BREAKAWAY_RISK_PER_TRADE=0.01
BREAKAWAY_MIN_VOL_RATIO=2.0

# Imbalance filter (NEW - 2026-01-08)
BREAKAWAY_USE_IMBALANCE=true
BREAKAWAY_IMBALANCE_THRESHOLD=0.10
BREAKAWAY_IMBALANCE_LOOKBACK=10

# Legacy filters (disabled by default)
BREAKAWAY_USE_TAI=false
BREAKAWAY_USE_TREND=false
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

### 2026-01-09: Breakout Optimized Strategy Implementation

**Overview:** Replaced Breakaway FVG strategy with Breakout Optimized strategy based on backtest showing +3,135R (69.4% win rate).

**Strategy Changes:**

| Aspect | Breakaway (Old) | Breakout (New) |
|--------|-----------------|----------------|
| Entry | FVG (Fair Value Gap) | Swing high breakout |
| Trend Filter | EWVMA cradle (inside bands) | EVWMA bands (above upper) |
| Exit | Fixed 3:1 R:R target | ATR trailing stop |
| Direction | Both (longs + shorts) | Longs only |
| Stop Management | Set at entry, never changes | Trails up with price |

**New Features:**

1. **State Persistence:**
   - Signals saved to `data/breakout_signals.json`
   - Survives bot restarts/crashes
   - Syncs with open positions on startup

2. **Emergency Take Profit:**
   - 10R circuit breaker TP sent to exchange
   - Protects against bot crash scenarios

3. **Orphan Order Cleanup:**
   - Cancels stale limit orders on startup
   - Prevents order accumulation across restarts

4. **Position Limit Logic:**
   - Unlimited pending limit orders allowed
   - When 5 positions fill, all pending orders cancelled
   - Prevents over-exposure

5. **Trade Logging:**
   - All breakout trades logged to SQLite
   - Accessible via `python start.py trades`

6. **Daily CSV Export:**
   - Cron job exports trades at midnight UTC
   - `sync_trades.sh` syncs to local machine

7. **Systemd Service:**
   - `breakout-bot.service` for auto-restart
   - Replaces manual nohup startup

**Files Created:**
- `breakout_strategy.py` - New strategy with ATR trailing stops
- `sync_trades.sh` - Local trade sync script
- `vps-breakoutbot.md` - VPS deployment guide (on GitHub)

**Files Modified:**
- `breakaway_bot.py` - State persistence, orphan cleanup, position logic
- `config.py` - Added BreakoutConfig dataclass
- `order_manager.py` - Added `get_filled_position_count()`, `get_pending_order_count()`, `cancel_all_pending_entry_orders()`
- `bybit_client.py` - Fixed `get_open_orders()` API call (settleCoin param)

**VPS Deployment:**
- Created `/etc/systemd/system/breakout-bot.service`
- Disabled old `kingbot.service`
- Set up cron for daily exports

---

### 2026-01-08: Aggressive Breakaway Strategy - Imbalance Filter

**Problem:** Tai Index and EWVMA-200 trend filters were too restrictive, limiting trade opportunities.

**Analysis:** Tested 15 different filter configurations via `filter_variations_backtest.py`. Key findings:
- Tai Index + Trend filters: 314 trades, 52.2% WR, +1.08R expectancy
- Imbalance filter only: 3,139 trades, 59.3% WR, +1.36R expectancy

**Solution:** Replaced Tai Index and EWVMA-200 trend filters with **Volume Delta Imbalance** filter.

**Volume Delta Imbalance Calculation:**
```python
# For each candle: bullish (close > open) = buy, bearish = sell
buy_volume = volume if bullish else 0
sell_volume = volume if bearish else 0
imbalance = (buy_sum - sell_sum) / total_sum  # Over 10-candle lookback
# Returns -1.0 (all sells) to +1.0 (all buys)
```

**Filter Logic:**
| Direction | Old Filters | New Filter |
|-----------|-------------|------------|
| SHORT | Tai > 53 AND Price > EWVMA-200 | Imbalance ≤ -0.10 |
| LONG | Tai < 47 AND Price < EWVMA-200 | Imbalance ≥ +0.10 |

**Performance Improvement:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Trades | 314 | ~3,139 | +10x |
| Win Rate | 52.2% | ~59.3% | +7% |
| Expectancy | +1.08R | ~1.36R | +26% |
| Total R | +340R | ~4,269R | +12.5x |

**Files Modified:**
- `breakaway_strategy.py`: Added `calculate_volume_delta_imbalance()`, updated filter logic
- `config.py`: Added `use_imbalance_filter`, `imbalance_threshold`, `imbalance_lookback` params
- `breakaway_bot.py`: Pass imbalance parameters to strategy, updated signal output

**Rollback (if needed):**
```bash
BREAKAWAY_USE_IMBALANCE=false
BREAKAWAY_USE_TAI=true
BREAKAWAY_USE_TREND=true
```

---

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

### 2026-01-06: Breakaway Strategy Threshold Optimization

**Problem:** Bot was generating 0 signals overnight - thresholds too strict.

**Analysis:** Ran backtest comparison on 1-minute data (7 symbols: BTC, DOGE, DOT, ETH, INJ, PNUT, SOL):
- Volume spike 2.5-3.0x requirement was blocking 86% of FVGs
- Tai Index thresholds (>55/<45) blocking signals even when volume/FVG aligned

**Backtest Results:**

| Parameter Set | Trades | Win Rate | Expectancy | Total R |
|---------------|--------|----------|------------|---------|
| Old (strict) | 185 | 48.6% | 0.95R | 175R |
| **New (moderate)** | **349** | **46.1%** | **0.85R** | **295R** |
| Relaxed | 593 | 44.4% | 0.77R | 459R |

**Changes Applied:**

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| `min_vol_ratio` | 3.0x | **2.0x** | ~2x more signals |
| `tai_threshold_short` | >55 | **>53** | Slightly less extreme |
| `tai_threshold_long` | <45 | **<47** | Slightly less extreme |
| `min_vol_ratio_1m` | 3.0x | **2.0x** | Same as 5-min |

**Trade-offs:**
- Win rate drops 2.5% (48.6% → 46.1%)
- Expectancy drops 10% (0.95R → 0.85R)
- **Total R increases 69%** (175R → 295R)
- Max drawdown unchanged (~11.4%)

**Files Modified:**
- `config.py`: Updated default thresholds and env var defaults
- `breakaway_strategy.py`: Updated docstring with new values

### 2026-01-06: WebSocket Ping/Pong Keepalive Fix

**Problem:** WebSocket disconnecting every ~1 minute (35 disconnections in 35 minutes), causing potential missed signals.

**Root Cause:** Missing ping/pong heartbeat. Bybit requires ping every 20 seconds to keep connection alive.

**Bybit WebSocket Requirements:**
| Requirement | Value |
|-------------|-------|
| Ping interval | Every 20 seconds |
| Timeout without ping | 10 minutes |
| Max connections | 500 per 5 min per domain |
| Max args array | 21,000 characters |

**Fix Applied:**
Added ping thread to `bybit_client.py`:
```python
def _ping_loop(self):
    """Send periodic pings to keep connection alive."""
    while self.running:
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps({"op": "ping"}))
        time.sleep(20)  # Bybit requires ping every 20 seconds
```

**Changes:**
- Added `ping_thread` that sends `{"op": "ping"}` every 20 seconds
- Handle pong responses in `_on_message`
- Auto-start ping thread on connection open
- Proper cleanup on disconnect

**Results:**
| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Errors in 2 min | ~2 | **0** |
| Connection stability | Disconnecting every 1 min | **Stable** |

**VPS Specs (DigitalOcean):**
- 1 CPU, 1GB RAM, 25GB disk
- Ubuntu 22.04 LTS
- Adequate for trading bot

**Files Modified:**
- `bybit_client.py`: Added ping/pong keepalive mechanism

### 2026-01-08: WebSocket Field Name Mismatch Fix

**Problem:** Bot's `_on_kline` handler was reading wrong field name for timeframe.

**Root Cause:** Field name mismatch between WebSocket client and bot handler:
```python
# bybit_client.py sends:
"timeframe": timeframe

# breakaway_bot.py was reading:
interval = data.get("interval", "5")  # Wrong key!
```

**Impact Assessment:**
- **Signal detection was NOT affected** in current configuration
- The default value `"5"` happened to match the only timeframe being used
- Bug would have caused issues if multiple timeframes (1m + 5m) were added later
- All candles would have been treated as "5" regardless of actual timeframe

**Why it worked despite the bug:**
```python
interval = data.get("interval", "5")  # Returns "5" (default, key missing)
if interval == "5":  # True, so code proceeded correctly
```

**Fix Applied:**
```python
# Before (wrong):
interval = data.get("interval", "5")

# After (correct):
timeframe = data.get("timeframe", "5")
```

**Additional Change:**
Added debug logging to confirm candle close detection:
```python
def _on_new_candle_5m(self, symbol: str, setup_key: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] 5m candle close: {symbol}")
```

**Verification:**
```
[03:40:00] 5m candle close: SOLUSDT
[03:40:00] 5m candle close: BTCUSDT
[03:40:04] 5m candle close: ETHUSDT
... (all 45 symbols at :00, :05, :10, etc.)
```

**Lesson Learned:**
- Always verify field names match between producer (WebSocket) and consumer (handler)
- Add debug logging for critical event detection (candle closes)
- Test with logging before assuming timing is correct

**Files Modified:**
- `breakaway_bot.py`: Fixed field name `interval` → `timeframe`, added candle close logging

---

## Need Help?

1. Stop the bot: `pkill -f TradingBot`
2. Check the log: `cat bot.log`
3. Look at error messages
4. Ask Claude for help!

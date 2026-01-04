# Performance Tracking CLI - Quick Start Guide

## Overview

The bot includes a built-in performance tracking system that records all trades to a local SQLite database and provides CLI commands to analyze your trading performance.

---

## Basic Usage

All commands are run via `start.py`:

```bash
python3 start.py <command> [options]
```

---

## Commands

### 1. Stats - Trading Statistics

View overall trading performance metrics.

```bash
# All-time stats
python3 start.py stats

# Today only
python3 start.py stats today

# This week
python3 start.py stats week

# This month
python3 start.py stats month

# Filter by symbol
python3 start.py stats --symbol BTCUSDT

# Custom date range
python3 start.py stats --from 2024-01-01 --to 2024-01-31
```

**Output includes:**
- Total trades, wins, losses
- Win rate, profit factor
- Net P&L, gross profit/loss
- Largest win/loss, average win/loss
- Max drawdown, current drawdown
- Average R-multiple, expectancy

---

### 2. Trades - Trade History

View individual trade records.

```bash
# Last 20 trades (default)
python3 start.py trades

# Last 50 trades
python3 start.py trades -n 50

# Only winners
python3 start.py trades --winners

# Only losers
python3 start.py trades --losers

# Filter by symbol
python3 start.py trades --symbol ETHUSDT

# Filter by direction
python3 start.py trades --direction long
python3 start.py trades --direction short

# Date range
python3 start.py trades --from 2024-01-01 --to 2024-01-31

# Combine filters
python3 start.py trades -n 10 --symbol BTCUSDT --winners
```

---

### 3. Equity - Equity Curve & Drawdowns

Analyze equity changes and drawdown metrics.

```bash
# Last week (default)
python3 start.py equity

# Last day
python3 start.py equity --period day

# Last month
python3 start.py equity --period month

# All time
python3 start.py equity --period all
```

**Output includes:**
- Current equity & peak equity
- Current drawdown percentage
- Maximum drawdown (amount, percentage, date)
- Equity curve summary (start, end, high, low)

---

### 4. Assets - Per-Asset Breakdown

See performance broken down by trading symbol.

```bash
# Default (sorted by P&L)
python3 start.py assets

# Sort by number of trades
python3 start.py assets --sort trades

# Sort by win rate
python3 start.py assets --sort winrate

# Sort by average R-multiple
python3 start.py assets --sort r
```

**Output shows for each symbol:**
- Number of trades
- Win count & win rate
- Net P&L
- Average R-multiple
- Profit factor

---

### 5. Sessions - Best & Worst Trading Days

Identify your best and worst trading sessions.

```bash
# Top 10 best and worst days (default)
python3 start.py sessions

# Top 5 days
python3 start.py sessions -n 5

# Sort by P&L (default)
python3 start.py sessions --sort pnl

# Sort by number of trades
python3 start.py sessions --sort trades

# Sort by win rate
python3 start.py sessions --sort winrate
```

---

### 6. Time - Time-Based Analysis

Analyze performance by hour of day and day of week.

```bash
# All-time analysis
python3 start.py time

# Last week only
python3 start.py time --period week

# Last month only
python3 start.py time --period month
```

**Output includes:**
- Performance by hour (UTC): trades, win rate, P&L, avg R
- Performance by day of week: trades, win rate, P&L, avg R

---

### 7. Streaks - Win/Loss Streaks

View current and historical win/loss streaks.

```bash
python3 start.py streaks
```

**Output includes:**
- Current win streak
- Current loss streak
- Maximum win streak (all-time)
- Maximum loss streak (all-time)

---

### 8. Export - Export Data

Export trade data to CSV or JSON files.

```bash
# Export trades to CSV (auto-named)
python3 start.py export trades

# Export to JSON
python3 start.py export trades --format json

# Custom output file
python3 start.py export trades --output my_trades.csv

# Export daily stats
python3 start.py export daily --format csv

# Export equity snapshots
python3 start.py export equity --format csv

# With date range
python3 start.py export trades --from 2024-01-01 --to 2024-01-31
```

---

## Quick Reference

| Command | Description | Common Options |
|---------|-------------|----------------|
| `stats` | Trading statistics | `today`, `week`, `month`, `--symbol`, `--from`, `--to` |
| `trades` | Trade history | `-n`, `--winners`, `--losers`, `--symbol`, `--direction` |
| `equity` | Drawdown analysis | `--period day/week/month/all` |
| `assets` | Per-symbol breakdown | `--sort pnl/trades/winrate/r` |
| `sessions` | Best/worst days | `-n`, `--sort pnl/trades/winrate` |
| `time` | Hour/day analysis | `--period week/month/all` |
| `streaks` | Win/loss streaks | (none) |
| `export` | Export to file | `trades/daily/equity`, `--format csv/json`, `--output` |

---

## Examples

### Daily Check-in
```bash
# Quick overview of today's performance
python3 start.py stats today

# What trades happened today?
python3 start.py trades -n 10
```

### Weekly Review
```bash
# This week's stats
python3 start.py stats week

# Best/worst days this week
python3 start.py sessions -n 5

# Which assets performed best?
python3 start.py assets --sort pnl
```

### Performance Audit
```bash
# Full stats
python3 start.py stats

# Check drawdown history
python3 start.py equity --period month

# Analyze timing patterns
python3 start.py time

# Export for spreadsheet analysis
python3 start.py export trades --format csv --output audit.csv
```

### Debugging a Bad Streak
```bash
# Check current streaks
python3 start.py streaks

# View recent losers
python3 start.py trades -n 20 --losers

# Which assets are underperforming?
python3 start.py assets --sort pnl
```

---

## VPS Usage

On the VPS, run commands from the bot directory:

```bash
ssh root@209.38.84.47
cd /root/kingbot
python3 start.py stats
```

---

## Database Location

All data is stored in:
```
data/trading_performance.db
```

This SQLite database is automatically created and maintained by the bot.

---

## Notes

- Stats start fresh - historical trades from before the tracking system was installed are not included
- Equity snapshots are recorded every 5 minutes while the bot is running
- All times are in UTC
- The CLI can be run while the bot is trading (separate process)

# Scalping Strategy - FVG Breakout with Partial Exits

A high-frequency scalping strategy for BTC, ETH, SOL targeting **9+ trades/day** with **85%+ win rate**.

---

## Strategy Comparison

| Aspect | Double Touch | Breakaway (Current Bot) | **Scalping (New)** |
|--------|--------------|------------------------|-------------------|
| **Pattern** | 5-step EMA ribbon sequence | FVG from EWVMA cradle | FVG from EWVMA cradle |
| **Win Rate** | 30-40% | 60-70% | **85-90%** |
| **R:R Ratio** | 3:1 | 3:1 | **1.5:1 (partial)** |
| **Avg Winner** | 3.0R | 3.0R | **1.25R** |
| **Expectancy** | +0.2-0.3R | +0.8-1.0R | **+0.8R** |
| **Trades/Day** | ~1 | ~3 | **~9** |
| **Hold Time** | 2-4 hours | 60-80 min | **~25 min** |
| **Complexity** | High (5 steps) | Medium | **Low** |
| **Best For** | Swing trading | Day trading | **Scalping** |

### Key Difference: Exit System

**Breakaway (3:1 R:R):**
```
Entry ────────────────────────────────► Target (3R)
  │
  └── Stop Loss (-1R)

Win: +3R | Loss: -1R
Need 25% WR to break even
```

**Scalping (Partial Exits):**
```
Entry ──────► TP1 (1R) ──────► TP2 (1.5R)
  │              │
  │              └── Close 50%, SL → Breakeven
  │
  └── Stop Loss (-1R)

Full Win: +1.25R | Partial Win: +0.5R | Loss: -1R
Need 44% WR to break even (much safer)
```

---

## How The Scalping Strategy Works

### Core Concept

Trade Fair Value Gaps (FVGs) that form after price consolidates in EWVMA bands. Use volume and order flow confirmation to filter for high-probability setups. Take partial profits quickly.

### What is a Fair Value Gap?

A gap between candles where price "skipped" a zone - indicates strong institutional momentum.

```
Bearish FVG (SHORT signal):

  Bar -2:  ████████           Low = $100.00
  Bar -1:     ████
  Bar  0:        ████████     High = $99.50

           ════════           ← GAP: $99.50 to $100.00

Condition: Current HIGH < 2-bars-ago LOW
```

```
Bullish FVG (LONG signal):

  Bar -2:     ████████        High = $100.00
  Bar -1:        ████
  Bar  0:  ████████           Low = $100.50

           ════════           ← GAP: $100.00 to $100.50

Condition: Current LOW > 2-bars-ago HIGH
```

### What is EWVMA Cradle?

EWVMA = Exponentially Weighted Volume Moving Average. It weighs price by volume, so high-volume candles have more influence.

**Cradle** = Price contained within EWVMA ± 1 standard deviation bands (consolidation).

```
                    Upper Band
─────────────────────────────────────
      ●     ●
         ●     ●   ●                 ← Candles "cradled" in bands
─────────────────────────────────────
                    EWVMA Midline
─────────────────────────────────────
                    Lower Band
                              │
                              └──── Then FVG breakout here
```

**Requirement:** 3+ of last 5 candles must close within the bands before FVG forms.

**Why:** Consolidation builds energy. Breakout from consolidation = high probability move.

---

## Entry Conditions (ALL Required)

### For SHORT:

| # | Condition | Check |
|---|-----------|-------|
| 1 | **Bearish FVG** | `high[0] < low[-2]` |
| 2 | **EWVMA Cradle** | 3+ of last 5 candles within bands |
| 3 | **Volume Spike** | Current volume >= 1.5x 20-period average |
| 4 | **Selling Pressure** | Volume Delta Imbalance <= -0.10 |

**Entry Price:** FVG bottom (current candle high)
**Stop Loss:** FVG top + 0.1% buffer
**TP1:** Entry - 1.0 × Risk
**TP2:** Entry - 1.5 × Risk

### For LONG:

| # | Condition | Check |
|---|-----------|-------|
| 1 | **Bullish FVG** | `low[0] > high[-2]` |
| 2 | **EWVMA Cradle** | 3+ of last 5 candles within bands |
| 3 | **Volume Spike** | Current volume >= 1.5x 20-period average |
| 4 | **Buying Pressure** | Volume Delta Imbalance >= +0.10 |

**Entry Price:** FVG top (current candle low)
**Stop Loss:** FVG bottom - 0.1% buffer
**TP1:** Entry + 1.0 × Risk
**TP2:** Entry + 1.5 × Risk

---

## Volume Delta Imbalance Explained

Measures buying vs selling pressure over the last 10 candles:

```python
For each candle:
  If close > open (bullish): volume counted as BUY
  If close < open (bearish): volume counted as SELL

Imbalance = (buy_volume - sell_volume) / total_volume
```

**Returns:** -1.0 (all selling) to +1.0 (all buying)

| Imbalance | Meaning | Action |
|-----------|---------|--------|
| <= -0.10 | Sellers dominating | SHORT allowed |
| >= +0.10 | Buyers dominating | LONG allowed |
| -0.10 to +0.10 | Neutral | No trade |

---

## Exit System: Partial Take-Profit

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

### All Exit Scenarios:

| Scenario | What Happens | Result |
|----------|--------------|--------|
| **Full Winner** | TP1 hit → TP2 hit | **+1.25R** |
| **Partial Winner** | TP1 hit → SL at breakeven | **+0.50R** |
| **Full Loser** | SL hit before TP1 | **-1.00R** |
| **Timeout** | 30 candles, close at market | Variable |

### Why Partial Exits Work:

At 85% win rate:
```
Expected per trade:
  85% × 1.25R = +1.0625R (winners)
  15% × -1.0R = -0.1500R (losers)
  ─────────────────────────
  Net: +0.91R per trade
```

Even accounting for some partial winners (TP1 only):
```
Realistic mix (70% full win, 15% partial, 15% loss):
  70% × 1.25R = +0.875R
  15% × 0.50R = +0.075R
  15% × -1.0R = -0.150R
  ─────────────────────────
  Net: +0.80R per trade
```

---

## Visual Trade Example

### SHORT Trade on BTC 5-minute:

```
Price
$100.60 ─┬─ Stop Loss (FVG top + 0.1% buffer)
         │
$100.00 ─┼─ FVG Top (2-bars-ago low)
         │  ════════════════════════ ← Fair Value Gap
$99.50 ──┼─ Entry (FVG bottom = current high)
         │
         │  Risk = $100.60 - $99.50 = $1.10
         │
$98.40 ──┼─ TP1 (Entry - 1.0R) → Close 50%, SL to $99.50
         │
$97.85 ──┼─ TP2 (Entry - 1.5R) → Close remaining 50%
         │

Context at entry:
  - Volume: 2.3x average ✓
  - Imbalance: -0.18 (selling) ✓
  - Cradle: 4 of 5 candles in bands ✓

Result: +1.25R in ~25 minutes
```

---

## Backtest Results

### Individual Assets (5-minute, ~280 days of data):

| Asset | Trades | Win Rate | Expectancy | Trades/Day | Total R |
|-------|--------|----------|------------|------------|---------|
| BTC | 1,751 | 84.8% | +0.784R | 6.3 | +1,372R |
| ETH | 427 | 88.8% | +0.870R | 6.1 | +372R |
| SOL | 429 | 86.5% | +0.849R | 6.2 | +364R |

### Combined Performance:

| Metric | Value |
|--------|-------|
| **Total Trades** | 2,607 |
| **Win Rate** | 85.7% |
| **Expectancy** | +0.809R per trade |
| **Total R Profit** | +2,108R |
| **Trades/Day** | 9.3 |
| **Avg Hold Time** | 5.6 candles (~28 min) |

### Projected Returns (0.5% risk per trade):

| Period | Trades | Expected R | Return |
|--------|--------|------------|--------|
| Daily | 9 | +7.3R | +3.6% |
| Weekly | 63 | +51R | +25.5% |
| Monthly | 270 | +218R | +109% |

*Note: These are backtested results. Live trading may differ due to slippage, fees, and market conditions.*

---

## Risk Management

### Position Sizing:

| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.5% of account |
| Max positions | 5 concurrent |
| Max per symbol | 1 position |
| Daily loss limit | 3% → stop trading |

### Cooldown Rules:

- After exit: Wait 5 candles before new entry on same symbol
- After 4 consecutive losses: Reduce size by 50%
- After 2 winners: Resume normal size

### Session Sizing (Optional):

| Session (UTC) | Size |
|---------------|------|
| NY Morning (14:00-16:00) | 100% |
| NY Afternoon (16:00-21:00) | 75% |
| Asian (00:00-08:00) | 50% |

---

## Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_vol_ratio` | 1.5 | Minimum volume spike |
| `imbalance_threshold` | 0.10 | Min order flow imbalance |
| `min_cradle_candles` | 3 | Min candles in EWVMA bands |
| `cradle_lookback` | 5 | Lookback for cradle check |
| `tp1_r_multiple` | 1.0 | First take-profit level |
| `tp2_r_multiple` | 1.5 | Second take-profit level |
| `tp1_close_pct` | 0.50 | % to close at TP1 |
| `sl_buffer_pct` | 0.001 | 0.1% buffer on stop loss |
| `max_hold_candles` | 30 | Timeout (close at market) |

---

## Why This Strategy Works

### 1. FVG = Institutional Footprint
Fair Value Gaps occur when large orders move price so fast that no trades happen in between. Trading FVGs means trading with institutional flow.

### 2. Consolidation Before Expansion
The EWVMA cradle requirement ensures we only trade breakouts from tight ranges. Random FVGs in trending markets are filtered out.

### 3. Volume Confirms Conviction
A gap without volume often fills immediately. We require 1.5x+ volume to ensure real money is behind the move.

### 4. Order Flow Alignment
The imbalance filter ensures sellers are in control for shorts (and buyers for longs). We don't fight the dominant flow.

### 5. Partial Exits Compound Faster
By taking profits at 1R and 1.5R instead of waiting for 3R:
- More trades reach target (85% vs 60%)
- Faster capital turnover
- Lower drawdowns
- Psychological ease (frequent wins)

---

## Comparison: When to Use Each Strategy

| Scenario | Best Strategy |
|----------|---------------|
| Want highest win rate | **Scalping** (85%) |
| Want fewer, larger wins | Breakaway (3:1 R:R) |
| Limited screen time | Breakaway (3 trades/day) |
| Active monitoring available | **Scalping** (9 trades/day) |
| Lower drawdown tolerance | **Scalping** (more winners) |
| Want to compound faster | **Scalping** (more trades) |

---

## Files Reference

| File | Purpose |
|------|---------|
| `backtests/backtest_scalp.py` | Backtest script with partial exits |
| `scalp_strategy.py` | Signal detection (to be created) |
| `scalp_bot.py` | Live bot orchestrator (to be created) |
| `scalp_config.py` | Configuration (to be created) |
| `breakaway_strategy.py` | Indicator calculations (reused) |

---

## Quick Start (Once Implemented)

```bash
# Run backtest
python3 backtests/backtest_scalp.py

# Start scalping bot (future)
python3 scalp_bot.py

# Monitor
tail -f scalp_bot.log
```

---

## Changelog

### 2026-01-09: Initial Design
- Created scalping strategy based on Breakaway FVG detection
- Added partial exit system (50% @ 1R, 50% @ 1.5R)
- Backtested on BTC, ETH, SOL: 85.7% WR, +0.81R expectancy
- Disabled Imbalance Flip and EWVMA Touch entries (underperformed)
- Target: 5-15 trades/day achieved (9.3 actual)

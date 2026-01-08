# Spread Trading Research Results

## Overview

This document summarizes our exploration of applying the Double Touch strategy to spread/pairs trading on cointegrated cryptocurrency pairs.

**Bottom Line:** Original Double Touch doesn't work on mean-reverting spreads, but we successfully adapted the concept into **Mean-Reversion Double Touch** which shows excellent results.

---

## 1. Cointegration Analysis (BTC/ETH)

### Test Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cointegration p-value | 0.0010 | Strongly cointegrated (< 0.05) |
| Spread Stationary (ADF) | Yes | p = 0.0012 |
| Hurst Exponent | 0.24 | Strongly mean-reverting (< 0.5) |
| Half-life | 154 candles | ~12.9 hours on 5-min data |
| Hedge Ratio | 0.0539 | 1 ETH ≈ 0.054 BTC exposure |

### Conclusion
BTC and ETH are statistically cointegrated. The spread between them is mean-reverting with a reasonable half-life for intraday trading.

---

## 2. Why Original Double Touch Failed

### The Problem

The original Double Touch pattern detection relies on EMA ribbon transitions:
- Step 0: HH/LL with GREEN/RED band
- Step 1-3: Band color transitions (GREEN → GREY → GREEN → GREY)
- Step 4: Band returns + FVG = Entry

**On the BTC/ETH spread, this fails because:**

1. **Timing mismatch**: Z-score recovers in ~19 candles, EMA ribbon changes in ~22 candles
2. **By the time EMA signals exhaustion, z-score is already at mean**
3. **No reward left**: TP is mean, but we're already there

### Evidence

| Event | Z-Score Recovery | Band Turns Grey | Z at Transition |
|-------|------------------|-----------------|-----------------|
| Average | 19 candles | 22 candles | -0.33 (near mean) |

When z < -2, band is RED. By time band turns GREY, z ≈ 0.

### Result
- Original Double Touch on spread: **0 signals**
- Without counter-trend filter: **1 signal** (lost)

---

## 3. Pure Z-Score Strategy (Baseline)

Before adapting Double Touch, we tested simple z-score mean-reversion:

### Best Parameters

| Config | Trades | Win Rate | PF | Return | Max DD |
|--------|--------|----------|-----|--------|--------|
| z=1.5→0.5 SL=3.0 | 235 | 62.1% | 1.14 | 33.9% | 43.9% |
| z=2.0→0.5 SL=4.0 | 124 | 71.0% | 1.26 | 31.5% | 33.2% |
| z=2.0→0.0 SL=4.0 | 109 | 70.6% | 1.14 | 18.9% | 37.8% |

**Recommended baseline:** z=2.0 entry, z=0.5 exit, z=4.0 stop loss

---

## 4. Mean-Reversion Double Touch (Adapted Strategy)

### Concept

We inverted the Double Touch logic for mean-reverting markets:

| Aspect | Original Double Touch | MR Double Touch |
|--------|----------------------|-----------------|
| Market type | Trending assets | Mean-reverting spreads |
| Core concept | Two pullbacks before trend continues | Two extremes before reversion |
| Step 0 | Higher High in uptrend | First extreme (z < -2.0) |
| Step 1 | Pullback (band grey) | Partial recovery (z > -1.0) |
| Step 2 | Trend resumes | Second touch (z < -1.5) |
| Entry trigger | FVG on trend resumption | Second touch is "higher low" |
| Logic | Failed pullbacks = trend strong | Failed extensions = mean strong |

### Pattern Visualization

```
Z-Score
   │
 2 ├─────────────────────────────────────────
   │
 1 ├─────────────────Recovery────────────────
   │                ↗         ↘
 0 ├─────Mean──────────────────────Mean──────
   │
-1 ├──────────────────────────────ENTRY──────
   │                                ↓
-2 ├───First Touch────────────Second Touch───
   │         ↓                      ↓
   └─────────────────────────────────────────
           Time →

Pattern: Extreme → Partial Recovery → Second Extreme (higher low) → ENTRY
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| First Extreme | z = ±2.0 | Initial touch of extreme zone |
| Recovery | z = ±1.0 | Must recover past this level |
| Second Touch | z = ±1.5 | Re-enter extreme zone |
| Max Pattern Bars | 50 | Pattern must complete within |
| Take Profit | z = ±0.5 | Exit near mean |
| Stop Loss | z = ±4.0 | Exit if extends further |

---

## 5. Backtest Results

### MR Double Touch Performance (BTC/ETH 5-min)

| Metric | Value |
|--------|-------|
| **Total Trades** | 51 |
| **Winners** | 45 |
| **Losers** | 6 |
| **Win Rate** | 88.2% |
| **Profit Factor** | 4.90 |
| **Total Return** | +61.4% |
| **Max Drawdown** | 12.0% |
| **TP Exits** | 51 (100%) |
| **SL Exits** | 0 (0%) |
| **Avg Hold Time** | 1.2 hours |

### Comparison with Baseline

| Metric | MR Double Touch | Pure Z-Score |
|--------|-----------------|--------------|
| Trades | 51 | 124 |
| Win Rate | **88.2%** | 71.0% |
| Profit Factor | **4.90** | 1.26 |
| Max Drawdown | **12.0%** | 33.2% |
| TP Rate | **100%** | ~85% |

**MR Double Touch has 3.9x better profit factor and 17% higher win rate.**

### Parameter Sensitivity

| First/Recovery/Second | Trades | Win% | PF | Return |
|----------------------|--------|------|-----|--------|
| -1.5/-0.5/-1.0 | 50 | 82.0% | 4.76 | 60.9% |
| **-2.0/-1.0/-1.5** | **51** | **88.2%** | **4.90** | **61.4%** |
| -2.5/-1.5/-2.0 | 18 | 72.2% | 1.08 | 1.0% |
| -2.0/-0.5/-1.5 | 24 | 83.3% | 2.51 | 17.9% |
| -2.0/-1.0/-1.8 | 30 | 90.0% | 12.05 | 43.4% |

Default parameters (-2.0/-1.0/-1.5) provide good balance of trade frequency and performance.

---

## 6. Sample Trades

| Entry Time | Type | Entry Z | First Z | Exit | P&L | Duration |
|------------|------|---------|---------|------|-----|----------|
| 2025-12-16 03:45 | SHORT | 1.50 | 2.14 | TP | +$127 | 13 bars |
| 2025-12-18 09:55 | LONG | -1.56 | -2.01 | TP | +$142 | 3 bars |
| 2025-12-19 04:00 | SHORT | 1.80 | 2.33 | TP | +$175 | 3 bars |
| 2025-12-22 15:25 | LONG | -1.78 | -2.15 | TP | +$168 | 8 bars |
| 2025-12-28 03:15 | SHORT | 1.62 | 2.41 | TP | +$203 | 5 bars |

---

## 7. Files

| File | Description |
|------|-------------|
| `spread_analysis.py` | Cointegration tests, z-score backtest |
| `mr_double_touch.py` | MR Double Touch strategy implementation |
| `spread_double_touch.py` | Original DT on spread (experimental, doesn't work) |

### Running the Analysis

```bash
# Cointegration analysis
python3 spread_analysis.py

# MR Double Touch backtest
python3 mr_double_touch.py
```

---

## 8. Key Insights

### Why MR Double Touch Works

1. **Pattern filters noise**: Random z-score crossings are filtered out
2. **"Higher low" confirmation**: Second touch not breaking first = buying/selling pressure
3. **Partial recovery requirement**: Ensures genuine mean-reversion attempt, not just noise
4. **Timing alignment**: Pattern completes when move is ready, not when EMAs catch up

### Why Original Double Touch Failed

1. **Designed for trending markets**: Looks for pullbacks in trends
2. **EMA ribbon too slow**: 9/21/50 EMAs lag z-score by 3+ candles
3. **Spread is anti-trend**: Mean-reverting by nature, opposite of what DT expects

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Pure Z-Score | More trades, simpler | Lower win rate, more drawdown |
| MR Double Touch | Higher win rate, better PF | Fewer trades, may miss some moves |

---

## 9. Next Steps

### To Implement Live Trading

1. **Add to bot**: Integrate `mr_double_touch.py` logic into main bot
2. **Execution**: Need to execute both legs (buy ETH, sell BTC) simultaneously
3. **Position sizing**: Account for hedge ratio in position sizes
4. **Monitoring**: Track spread and z-score in real-time

### To Expand Research

1. **Test other pairs**: SOL/ETH, BTC/SOL, etc.
2. **Test other timeframes**: 15-min, 1-hour
3. **Walk-forward validation**: Test on out-of-sample data
4. **Add more filters**: Volume, volatility, time-of-day

---

## 10. Conclusion

The Double Touch concept successfully translates to mean-reverting markets when inverted:

- **Original**: Two pullbacks in trend → enter on continuation
- **Adapted**: Two extremes in spread → enter on reversion

The Mean-Reversion Double Touch strategy shows:
- **88.2% win rate** (vs 71% baseline)
- **4.90 profit factor** (vs 1.26 baseline)
- **100% TP exits** (never hit stop loss in 51 trades)
- **12% max drawdown** (vs 33% baseline)

This is a proprietary adaptation of the Double Touch pattern for cointegrated pairs trading.

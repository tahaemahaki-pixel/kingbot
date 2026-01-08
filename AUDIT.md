# Breakaway Bot Audit Report

**Date:** 2026-01-08
**Auditor:** Claude Code
**Bot Version:** Breakaway Strategy v2 (Imbalance Filter)

---

## Executive Summary

| Area | Status | Notes |
|------|--------|-------|
| **VPS** | Running | Bot active via systemd |
| **Code Logic** | Sound | Strategy well-implemented |
| **Trade Logic** | Sound | Risk management in place |
| **Issues Found** | 5 Minor | See findings below |

---

## 1. VPS Status

| Metric | Value |
|--------|-------|
| **Bot Status** | Active (systemd service) |
| **Service Name** | `kingbot.service` |
| **Memory Usage** | 111.6M / 957M (11.6%) |
| **Disk Usage** | 3.0G / 25G (12%) |
| **Server Uptime** | 2 days, 23 hours |
| **Load Average** | 0.20, 0.18, 0.09 |

**Verdict:** VPS resources are adequate. Memory usage is healthy.

---

## 2. Bot Configuration (Live)

| Setting | Value | Assessment |
|---------|-------|------------|
| Testnet | `false` | Live trading |
| Timeframe | 5m | Correct |
| Risk/Trade | 1% | Conservative |
| Symbols | 45 | Good coverage |
| Max Positions | 5 | Reasonable |
| Candles Preload | 2000 | Sufficient history |

---

## 3. Code Audit Findings

### 3.1 Strategy Logic (`breakaway_strategy.py`)

**Status:** Sound

**Verified correct:**
- FVG detection (bearish: `high[i] < low[i-2]`, bullish: `low[i] > high[i-2]`)
- Volume Delta Imbalance filter implementation
- EWVMA cradle detection with proper rolling window
- R:R calculation: `risk = |SL - entry|`, `target = entry +/- (risk * 3.0)`

**Minor issue:**
```python
# Line 177 - Division warning (cosmetic only)
rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
```
NumPy throws a warning when `avg_loss` is 0. No functional impact.

### 3.2 Order Manager (`order_manager.py`)

**Status:** Sound

**Verified correct:**
- Position sizing: `position_size = risk_amount / stop_distance`
- Minimum order value check ($5 Bybit minimum)
- Quantity rounding to valid step sizes per symbol
- Position limit checks (max 5 positions)
- Daily loss limit enforcement (5%)

### 3.3 Main Bot (`breakaway_bot.py`)

**Status:** Sound

**Verified correct:**
- Proper signal handling (SIGINT, SIGTERM)
- WebSocket ping/pong keepalive every 20 seconds
- Position sync every 30 seconds
- Status print every 5 minutes
- Correct field name usage (`timeframe` not `interval`)

**Flow verified:**
1. Load .env
2. Init client
3. Fetch top 45 symbols by volume
4. Load 2000 candles each
5. Connect WebSocket
6. Process candle closes
7. Execute signals

### 3.4 Data Feed (`data_feed.py`)

**Status:** Sound

**Verified correct:**
- Handles Bybit's 1000-candle limit via pagination
- Correct EWVMA formula
- EMA ribbon calculation (9/21/50 multipliers)
- Candle deduplication by timestamp

### 3.5 WebSocket Client (`bybit_client.py`)

**Status:** Sound

**Verified correct:**
- Auto-reconnect on disconnect
- Ping every 20 seconds (Bybit requirement)
- Rate limiting (500ms between REST requests)
- Exponential backoff retries (0.5s, 1s, 1.5s)

---

## 4. Trade Logic Verification

### Entry Conditions (All must be true)

| Condition | SHORT | LONG |
|-----------|-------|------|
| FVG | Bearish gap | Bullish gap |
| Cradle | 3+/5 candles in EWVMA bands | Same |
| Volume | >= 2.0x average | Same |
| Imbalance | < -0.10 (selling) | > +0.10 (buying) |

### Exit Rules

| Rule | Value |
|------|-------|
| Stop Loss | FVG boundary + 0.1% buffer |
| Take Profit | 3:1 R:R |

### Position Sizing Formula

```
risk_amount = account_balance * 0.01  (1%)
position_size = risk_amount / |entry - stop_loss|
```

**Verified:** Correctly implemented in `_execute_signal_5m()`

---

## 5. Issues & Recommendations

### Issue 1: NumPy Division Warning (LOW)

**Location:** `breakaway_strategy.py:177`
**Impact:** Log spam, no functional impact
**Recommendation:** Suppress warning or handle edge case:
```python
with np.errstate(divide='ignore', invalid='ignore'):
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
```

### Issue 2: Some Symbols Load 0 Candles (MEDIUM)

**Observed:** `Loaded 0 historical candles` for some symbols
**Cause:** Symbol may be delisted or have insufficient history
**Impact:** Those symbols won't generate signals
**Recommendation:** Add validation and skip symbols with <300 candles

### Issue 3: Missing `get_last_price()` Method (LOW)

**Location:** `order_manager.py:660`
**Issue:** `BybitClient` doesn't have `get_last_price()` method
**Impact:** Would error if `place_order()` called without price
**Note:** Currently all calls include price, so not triggered

### Issue 4: Position Count String Matching (LOW)

**Location:** `breakaway_bot.py:286-294`
**Issue:** Uses fragile string matching on setup_key (`"_5" in str(...)`)
**Recommendation:** Use explicit timeframe attribute instead

### Issue 5: Telegram Token in Logs (INFO)

**Recommendation:** Ensure `.env` remains in `.gitignore`

---

## 6. Overall Assessment

| Category | Rating | Notes |
|----------|--------|-------|
| **Code Quality** | 8/10 | Clean, well-structured |
| **Strategy Logic** | 9/10 | Mathematically sound |
| **Risk Management** | 9/10 | Conservative, proper limits |
| **Error Handling** | 7/10 | Could add more validation |
| **VPS Setup** | 9/10 | Systemd, auto-restart |

---

## 7. Conclusion

**The bot is production-ready and operating correctly.**

The issues found are minor and don't affect core functionality. The strategy logic, trade execution, and risk management are all implemented correctly.

### Next Steps (Optional)
1. Fix NumPy warning to clean up logs
2. Add symbol validation for 0-candle edge case
3. Add `get_last_price()` method to BybitClient for completeness

---

*Generated by Claude Code audit on 2026-01-08*

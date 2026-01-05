#!/usr/bin/env python3
"""
Long King & Short King Pattern Backtest
Identifies and backtests the king patterns on OHLC data with EMA ribbon.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SwingPoint:
    index: int
    price: float
    time: str
    type: str  # 'high' or 'low'
    candle_open: float

@dataclass
class Trade:
    pattern: str  # 'long_king' or 'short_king'
    entry_time: str
    entry_price: float
    stop_loss: float
    target: float
    exit_time: Optional[str]
    exit_price: Optional[float]
    pnl: Optional[float]
    pnl_percent: Optional[float]
    result: Optional[str]  # 'win', 'loss', 'open'
    risk_reward: Optional[float]
    # Pattern points for reference
    a_idx: int
    c_idx: int
    d_idx: int
    e_idx: int
    f_idx: int
    g_idx: int


def calc_evwma(price: pd.Series, volume: pd.Series, length: int = 20, use_cumulative: bool = False) -> pd.Series:
    """
    Calculate Elastic Volume Weighted Moving Average (EVWMA).

    Formula: evwma = prev_evwma * (nbfs - volume)/nbfs + (volume * price / nbfs)

    Args:
        price: Price series (close, high, or low)
        volume: Volume series
        length: Rolling window for volume sum (if not using cumulative)
        use_cumulative: Use cumulative volume instead of rolling sum

    Returns:
        EVWMA series
    """
    if use_cumulative:
        nbfs = volume.cumsum()
    else:
        nbfs = volume.rolling(window=length, min_periods=1).sum()

    evwma = pd.Series(index=price.index, dtype=float)
    evwma.iloc[0] = price.iloc[0]  # Initialize with first price

    for i in range(1, len(price)):
        if nbfs.iloc[i] == 0 or pd.isna(nbfs.iloc[i]):
            evwma.iloc[i] = evwma.iloc[i-1]
        else:
            prev = evwma.iloc[i-1] if not pd.isna(evwma.iloc[i-1]) else price.iloc[i]
            vol = volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
            nb = nbfs.iloc[i]

            # EVWMA formula from PineScript
            evwma.iloc[i] = prev * (nb - vol) / nb + (vol * price.iloc[i] / nb)

    return evwma


def load_data(filepath: str, evwma_length: int = 20) -> pd.DataFrame:
    """Load and prepare the CSV data."""
    df = pd.read_csv(filepath)

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Handle duplicate column names - pandas adds .1, .2 etc for duplicates
    # Look for evwma+/evwma- columns (could be original or .1 suffix)
    cols = df.columns.tolist()

    ribbon_upper = None
    ribbon_lower = None

    # Try to find evwma+ and evwma- columns with data
    # Check both original and duplicate columns (.1 suffix)
    for suffix in ['', '.1', '.2']:
        upper_col = f'evwma+{suffix}' if suffix else 'evwma+'
        lower_col = f'evwma-{suffix}' if suffix else 'evwma-'

        if upper_col in cols and lower_col in cols:
            upper_data = pd.to_numeric(df[upper_col], errors='coerce')
            lower_data = pd.to_numeric(df[lower_col], errors='coerce')

            # Check if this column has mostly valid data
            if upper_data.notna().sum() > len(df) * 0.3:
                ribbon_upper = upper_data
                ribbon_lower = lower_data
                print(f"Using EVWMA ribbon from columns: {upper_col}, {lower_col}")
                break

    # If we found ribbon data, use it
    if ribbon_upper is not None:
        df['ribbon_upper'] = ribbon_upper
        df['ribbon_lower'] = ribbon_lower
    else:
        # Check if we have volume data to calculate proper EVWMA
        has_volume = 'volume' in [c.lower() for c in df.columns]

        if has_volume:
            # Find the volume column (case-insensitive)
            vol_col = [c for c in df.columns if c.lower() == 'volume'][0]
            df['vol'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0)

            # Calculate proper EVWMA ribbon
            print(f"Calculating EVWMA ribbon (length={evwma_length})...")
            df['evwma_mid'] = calc_evwma(df['close'], df['vol'], evwma_length)
            df['ribbon_upper'] = calc_evwma(df['high'], df['vol'], evwma_length)  # evwma+
            df['ribbon_lower'] = calc_evwma(df['low'], df['vol'], evwma_length)   # evwma-
        else:
            # No volume data - fall back to EMA-based approximation
            print("No volume/EVWMA data - calculating EVWMA approximation...")

            # Use price-based EVWMA approximation (treat each bar as equal volume)
            # This gives a simple moving average behavior
            df['vol'] = 1.0  # Treat each bar as having volume of 1
            df['evwma_mid'] = calc_evwma(df['close'], df['vol'], evwma_length)
            df['ribbon_upper'] = calc_evwma(df['high'], df['vol'], evwma_length)
            df['ribbon_lower'] = calc_evwma(df['low'], df['vol'], evwma_length)

    # Calculate 300 SMA for trend filter
    df['sma300'] = df['close'].rolling(window=300).mean()

    # Calculate 50 EMA for trend filter
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Calculate EWVMA-200 for counter-trend filter (like Double Touch)
    if 'vol' in df.columns:
        df['ewvma_200'] = calc_evwma(df['close'], df['vol'], 200)
    else:
        # Fallback to SMA-200 if no volume
        df['ewvma_200'] = df['close'].rolling(window=200).mean()

    return df


def get_short_wick_target(df: pd.DataFrame, swing_idx: int, swing_type: str) -> float:
    """
    Get the short wick level from the 2 candles forming the swing point.

    For swing HIGH (long TP): Use body top (max of open/close) - more conservative than absolute high
    For swing LOW (short TP): Use body bottom (min of open/close) - more conservative than absolute low

    Args:
        df: DataFrame with OHLC data
        swing_idx: Index of the swing point candle
        swing_type: 'high' or 'low'

    Returns:
        The short wick target price
    """
    # Get the swing candle and the one before it (the 2 candles forming the swing)
    idx1 = max(0, swing_idx - 1)
    idx2 = swing_idx

    candle1 = df.iloc[idx1]
    candle2 = df.iloc[idx2]

    if swing_type == 'high':
        # For swing high: use body top (short wick level)
        # Body top = max(open, close)
        body_top1 = max(candle1['open'], candle1['close'])
        body_top2 = max(candle2['open'], candle2['close'])
        # Use the lower of the two for more conservative TP
        return min(body_top1, body_top2)
    else:
        # For swing low: use body bottom (short wick level)
        # Body bottom = min(open, close)
        body_bottom1 = min(candle1['open'], candle1['close'])
        body_bottom2 = min(candle2['open'], candle2['close'])
        # Use the higher of the two for more conservative TP
        return max(body_bottom1, body_bottom2)


def check_trend_filter(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str, threshold: float = 0.8) -> bool:
    """
    Check if at least threshold% of candles are above/below 300 SMA.

    Args:
        df: DataFrame with 'close' and 'sma300' columns
        start_idx: Start index of the pattern (A point)
        end_idx: End index of the pattern (G point)
        direction: 'long' or 'short'
        threshold: Percentage of candles required (default 0.8 = 80%)

    Returns:
        True if trend filter passes, False otherwise
    """
    if start_idx >= end_idx:
        return False

    subset = df.iloc[start_idx:end_idx + 1]

    # Skip if SMA not calculated yet (not enough data)
    if subset['sma300'].isna().all():
        return False

    valid_rows = subset.dropna(subset=['sma300'])
    if len(valid_rows) == 0:
        return False

    if direction == 'long':
        # For longs: count candles where close > sma300
        above_count = (valid_rows['close'] > valid_rows['sma300']).sum()
        pct_above = above_count / len(valid_rows)
        return pct_above >= threshold
    else:
        # For shorts: count candles where close < sma300
        below_count = (valid_rows['close'] < valid_rows['sma300']).sum()
        pct_below = below_count / len(valid_rows)
        return pct_below >= threshold


def check_ema50_filter(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str, threshold: float = 0.8) -> bool:
    """
    Check if at least threshold% of candles are above/below the 50 EMA.
    """
    if start_idx >= end_idx:
        return False

    subset = df.iloc[start_idx:end_idx + 1]

    if subset['ema50'].isna().all():
        return False

    valid_rows = subset.dropna(subset=['ema50'])
    if len(valid_rows) == 0:
        return False

    if direction == 'long':
        above_count = (valid_rows['close'] > valid_rows['ema50']).sum()
        return (above_count / len(valid_rows)) >= threshold
    else:
        below_count = (valid_rows['close'] < valid_rows['ema50']).sum()
        return (below_count / len(valid_rows)) >= threshold


def check_evwma_filter(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str, threshold: float = 0.8) -> bool:
    """
    Check if at least threshold% of candles are above/below the EVWMA ribbon midpoint.

    Args:
        df: DataFrame with 'close', 'ribbon_upper', 'ribbon_lower' columns
        start_idx: Start index of the pattern (A point)
        end_idx: End index of the pattern (G point)
        direction: 'long' or 'short'
        threshold: Percentage of candles required (default 0.8 = 80%)

    Returns:
        True if EVWMA filter passes, False otherwise
    """
    if start_idx >= end_idx:
        return False

    subset = df.iloc[start_idx:end_idx + 1]

    # Skip if ribbon not calculated yet
    if subset['ribbon_upper'].isna().all() or subset['ribbon_lower'].isna().all():
        return False

    valid_rows = subset.dropna(subset=['ribbon_upper', 'ribbon_lower'])
    if len(valid_rows) == 0:
        return False

    # Calculate ribbon midpoint
    ribbon_mid = (valid_rows['ribbon_upper'] + valid_rows['ribbon_lower']) / 2

    if direction == 'long':
        # For longs: count candles where close > ribbon midpoint
        above_count = (valid_rows['close'] > ribbon_mid).sum()
        pct_above = above_count / len(valid_rows)
        return pct_above >= threshold
    else:
        # For shorts: count candles where close < ribbon midpoint
        below_count = (valid_rows['close'] < ribbon_mid).sum()
        pct_below = below_count / len(valid_rows)
        return pct_below >= threshold


def check_ewvma_counter_trend(df: pd.DataFrame, a_idx: int, direction: str) -> bool:
    """
    Counter-trend filter using EWVMA-200 (like Double Touch strategy).

    For longs: A point (step 0) must be BELOW EWVMA-200 (mean reversion from oversold)
    For shorts: A point (step 0) must be ABOVE EWVMA-200 (mean reversion from overbought)

    Args:
        df: DataFrame with 'close' and 'ewvma_200' columns
        a_idx: Index of point A (the start of the pattern)
        direction: 'long' or 'short'

    Returns:
        True if counter-trend filter passes, False otherwise
    """
    if a_idx >= len(df):
        return False

    # Get the close price and EWVMA-200 at point A
    a_close = df.iloc[a_idx]['close']
    ewvma_200 = df.iloc[a_idx]['ewvma_200']

    # Skip if EWVMA not calculated yet
    if pd.isna(ewvma_200):
        return False

    if direction == 'long':
        # Counter-trend: A must be BELOW EWVMA-200 (catching oversold reversals)
        return a_close < ewvma_200
    else:
        # Counter-trend: A must be ABOVE EWVMA-200 (catching overbought reversals)
        return a_close > ewvma_200


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """
    Identify swing highs and lows.
    A swing high has lower highs on both sides.
    A swing low has higher lows on both sides.
    """
    swings = []

    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['high'] <= df.iloc[i-j]['high'] or df.iloc[i]['high'] <= df.iloc[i+j]['high']:
                is_swing_high = False
                break

        if is_swing_high:
            swings.append(SwingPoint(
                index=i,
                price=df.iloc[i]['high'],
                time=str(df.iloc[i]['time']),
                type='high',
                candle_open=df.iloc[i]['open']
            ))

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if df.iloc[i]['low'] >= df.iloc[i-j]['low'] or df.iloc[i]['low'] >= df.iloc[i+j]['low']:
                is_swing_low = False
                break

        if is_swing_low:
            swings.append(SwingPoint(
                index=i,
                price=df.iloc[i]['low'],
                time=str(df.iloc[i]['time']),
                type='low',
                candle_open=df.iloc[i]['open']
            ))

    # Sort by index
    swings.sort(key=lambda x: x.index)
    return swings


def is_into_ribbon(price: float, ribbon_upper: float, ribbon_lower: float, buffer: float = 0.005) -> bool:
    """Check if price is into/touching the ribbon (within buffer %)."""
    ribbon_range = ribbon_upper - ribbon_lower
    extended_upper = ribbon_upper + ribbon_range * buffer
    extended_lower = ribbon_lower - ribbon_range * buffer
    return extended_lower <= price <= extended_upper


@dataclass
class FVG:
    """Fair Value Gap"""
    index: int  # Index of the middle (impulse) candle
    type: str   # 'bullish' or 'bearish'
    top: float  # Upper bound of gap
    bottom: float  # Lower bound of gap
    time: str


def find_fvg(df: pd.DataFrame, start_idx: int, end_idx: int, fvg_type: str) -> Optional[FVG]:
    """
    Find a Fair Value Gap between start_idx and end_idx.

    Bullish FVG: Candle 1 high < Candle 3 low (gap up)
    Bearish FVG: Candle 1 low > Candle 3 high (gap down)

    Returns the first valid FVG found, or None.
    """
    for i in range(start_idx + 1, end_idx - 1):
        c1 = df.iloc[i - 1]  # Candle before impulse
        c2 = df.iloc[i]      # Impulse candle
        c3 = df.iloc[i + 1]  # Candle after impulse

        if fvg_type == 'bullish':
            # Bullish FVG: gap between c1 high and c3 low
            if c1['high'] < c3['low']:
                return FVG(
                    index=i,
                    type='bullish',
                    top=c3['low'],
                    bottom=c1['high'],
                    time=str(c2['time'])
                )
        else:  # bearish
            # Bearish FVG: gap between c1 low and c3 high
            if c1['low'] > c3['high']:
                return FVG(
                    index=i,
                    type='bearish',
                    top=c1['low'],
                    bottom=c3['high'],
                    time=str(c2['time'])
                )

    return None


def find_fvg_entry(df: pd.DataFrame, fvg: FVG, start_idx: int, max_candles: int = 20) -> Optional[int]:
    """
    Find when price returns to the FVG zone for entry.

    For bullish FVG: wait for price to dip into the gap (low touches gap)
    For bearish FVG: wait for price to rally into the gap (high touches gap)

    Returns the candle index for entry, or None if no retest occurs.
    """
    for i in range(start_idx, min(start_idx + max_candles, len(df))):
        if fvg.type == 'bullish':
            # Price dips into the FVG zone
            if df.iloc[i]['low'] <= fvg.top and df.iloc[i]['low'] >= fvg.bottom:
                return i
            # FVG fully filled (invalidated) - price closed below gap
            if df.iloc[i]['close'] < fvg.bottom:
                return None
        else:  # bearish
            # Price rallies into the FVG zone
            if df.iloc[i]['high'] >= fvg.bottom and df.iloc[i]['high'] <= fvg.top:
                return i
            # FVG fully filled (invalidated) - price closed above gap
            if df.iloc[i]['close'] > fvg.top:
                return None

    return None


def is_above_ribbon(close: float, ribbon_upper: float) -> bool:
    """Check if close is above the ribbon upper bound."""
    return close > ribbon_upper


def is_below_ribbon(close: float, ribbon_lower: float) -> bool:
    """Check if close is below the ribbon lower bound."""
    return close < ribbon_lower


def find_short_king_patterns(df: pd.DataFrame, swings: List[SwingPoint], symbol: str = "", filter_type: str = "sma300") -> List[Dict]:
    """
    Find Short King patterns:
    A: Swing high INTO ribbon (resistance)
    C: Swing low below A
    D: Close above A + pullback into ribbon
    E: Higher high above D
    F: Close below ribbon
    G: Entry zone

    filter_type: 'none', 'sma300', or 'evwma'
    """
    patterns = []
    swing_highs = [s for s in swings if s.type == 'high']
    swing_lows = [s for s in swings if s.type == 'low']

    for a in swing_highs:
        # Skip if no ribbon data
        if pd.isna(df.iloc[a.index]['ribbon_upper']):
            continue

        ribbon_upper = df.iloc[a.index]['ribbon_upper']
        ribbon_lower = df.iloc[a.index]['ribbon_lower']

        # A must be INTO the ribbon
        if not is_into_ribbon(a.price, ribbon_upper, ribbon_lower, buffer=0.02):
            continue

        # Find C: swing low after A, below A's level
        for c in swing_lows:
            if c.index <= a.index:
                continue
            if c.index > a.index + 50:  # Don't look too far
                break
            if c.price >= a.price:  # C must be below A
                continue

            # Find D: candle that closes above A's high, then pulls back into ribbon
            d_idx = None
            for i in range(c.index + 1, min(c.index + 30, len(df))):
                if df.iloc[i]['close'] > a.price:  # Closes above A
                    # Now look for pullback into ribbon
                    for j in range(i + 1, min(i + 15, len(df))):
                        if pd.isna(df.iloc[j]['ribbon_upper']):
                            continue
                        rb_up = df.iloc[j]['ribbon_upper']
                        rb_lo = df.iloc[j]['ribbon_lower']
                        if is_into_ribbon(df.iloc[j]['close'], rb_up, rb_lo, buffer=0.02):
                            d_idx = j
                            break
                    if d_idx:
                        break

            if d_idx is None:
                continue

            # Find E: Higher high after D
            e_swing = None
            d_high = df.iloc[d_idx]['high']
            for e in swing_highs:
                if e.index <= d_idx:
                    continue
                if e.index > d_idx + 30:
                    break
                if e.price > d_high:  # Higher high
                    e_swing = e
                    break

            if e_swing is None:
                continue

            # Find F: Close below ribbon after E
            f_idx = None
            for i in range(e_swing.index + 1, min(e_swing.index + 20, len(df))):
                if pd.isna(df.iloc[i]['ribbon_lower']):
                    continue
                if is_below_ribbon(df.iloc[i]['close'], df.iloc[i]['ribbon_lower']):
                    f_idx = i
                    break

            if f_idx is None:
                continue

            # NEW: Find bearish FVG between E and F
            fvg = find_fvg(df, e_swing.index, f_idx + 1, 'bearish')
            if fvg is None:
                continue  # No FVG = no trade

            # G is entry when price retests the FVG
            g_idx = find_fvg_entry(df, fvg, f_idx + 1, max_candles=20)
            if g_idx is None:
                continue  # Price never retested FVG or FVG was invalidated

            # Entry price is middle of FVG zone - 0.03% buffer for better fills
            fvg_midpoint = (fvg.top + fvg.bottom) / 2
            entry_price = fvg_midpoint * 0.9997  # Slightly below midpoint for shorts

            # ETH uses candle open SL, others use structure SL
            if "ETH" in symbol.upper():
                stop_loss = e_swing.candle_open * 1.001  # Just above E candle open
            else:
                stop_loss = e_swing.price * 1.001  # Just above E's high (structure)

            # Apply trend filter based on filter_type
            if filter_type == 'sma300':
                if not check_trend_filter(df, a.index, g_idx, 'short', threshold=0.8):
                    continue
            elif filter_type == 'evwma':
                if not check_evwma_filter(df, a.index, g_idx, 'short', threshold=0.8):
                    continue
            elif filter_type == 'ewvma_counter':
                if not check_ewvma_counter_trend(df, a.index, 'short'):
                    continue
            # filter_type == 'none' - no filter applied

            patterns.append({
                'type': 'short_king',
                'a': a,
                'c': c,
                'd_idx': d_idx,
                'e': e_swing,
                'f_idx': f_idx,
                'g_idx': g_idx,
                'fvg': fvg,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target': c.price,
                'entry_time': str(df.iloc[g_idx]['time'])
            })
            break  # Found pattern starting from this A, move on

    return patterns


def find_long_king_patterns(df: pd.DataFrame, swings: List[SwingPoint], symbol: str = "", filter_type: str = "sma300") -> List[Dict]:
    """
    Find Long King patterns:
    A: Swing low INTO ribbon (support)
    C: Swing high above A
    D: Close below A + pullback into ribbon
    E: Lower low below D
    F: Close above ribbon
    G: Entry zone

    filter_type: 'none', 'sma300', or 'evwma'
    """
    patterns = []
    swing_highs = [s for s in swings if s.type == 'high']
    swing_lows = [s for s in swings if s.type == 'low']

    for a in swing_lows:
        # Skip if no ribbon data
        if pd.isna(df.iloc[a.index]['ribbon_upper']):
            continue

        ribbon_upper = df.iloc[a.index]['ribbon_upper']
        ribbon_lower = df.iloc[a.index]['ribbon_lower']

        # A must be INTO the ribbon
        if not is_into_ribbon(a.price, ribbon_upper, ribbon_lower, buffer=0.02):
            continue

        # Find C: swing high after A, above A's level
        for c in swing_highs:
            if c.index <= a.index:
                continue
            if c.index > a.index + 50:
                break
            if c.price <= a.price:  # C must be above A
                continue

            # Find D: candle that closes below A's low, then pulls back into ribbon
            d_idx = None
            for i in range(c.index + 1, min(c.index + 30, len(df))):
                if df.iloc[i]['close'] < a.price:  # Closes below A
                    # Now look for pullback into ribbon
                    for j in range(i + 1, min(i + 15, len(df))):
                        if pd.isna(df.iloc[j]['ribbon_upper']):
                            continue
                        rb_up = df.iloc[j]['ribbon_upper']
                        rb_lo = df.iloc[j]['ribbon_lower']
                        if is_into_ribbon(df.iloc[j]['close'], rb_up, rb_lo, buffer=0.02):
                            d_idx = j
                            break
                    if d_idx:
                        break

            if d_idx is None:
                continue

            # Find E: Lower low after D
            e_swing = None
            d_low = df.iloc[d_idx]['low']
            for e in swing_lows:
                if e.index <= d_idx:
                    continue
                if e.index > d_idx + 30:
                    break
                if e.price < d_low:  # Lower low
                    e_swing = e
                    break

            if e_swing is None:
                continue

            # Find F: Close above ribbon after E
            f_idx = None
            for i in range(e_swing.index + 1, min(e_swing.index + 20, len(df))):
                if pd.isna(df.iloc[i]['ribbon_upper']):
                    continue
                if is_above_ribbon(df.iloc[i]['close'], df.iloc[i]['ribbon_upper']):
                    f_idx = i
                    break

            if f_idx is None:
                continue

            # NEW: Find bullish FVG between E and F
            fvg = find_fvg(df, e_swing.index, f_idx + 1, 'bullish')
            if fvg is None:
                continue  # No FVG = no trade

            # G is entry when price retests the FVG
            g_idx = find_fvg_entry(df, fvg, f_idx + 1, max_candles=20)
            if g_idx is None:
                continue  # Price never retested FVG or FVG was invalidated

            # Entry price is middle of FVG zone + 0.03% buffer for better fills
            fvg_midpoint = (fvg.top + fvg.bottom) / 2
            entry_price = fvg_midpoint * 1.0003  # Slightly above midpoint for longs

            # ETH uses candle open SL, others use structure SL
            if "ETH" in symbol.upper():
                stop_loss = e_swing.candle_open * 0.999  # Just below E candle open
            else:
                stop_loss = e_swing.price * 0.999  # Just below E's low (structure)

            # Apply trend filter based on filter_type
            if filter_type == 'sma300':
                if not check_trend_filter(df, a.index, g_idx, 'long', threshold=0.8):
                    continue
            elif filter_type == 'evwma':
                if not check_evwma_filter(df, a.index, g_idx, 'long', threshold=0.8):
                    continue
            elif filter_type == 'ewvma_counter':
                if not check_ewvma_counter_trend(df, a.index, 'long'):
                    continue
            # filter_type == 'none' - no filter applied

            patterns.append({
                'type': 'long_king',
                'a': a,
                'c': c,
                'd_idx': d_idx,
                'e': e_swing,
                'f_idx': f_idx,
                'g_idx': g_idx,
                'fvg': fvg,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target': c.price,
                'entry_time': str(df.iloc[g_idx]['time'])
            })
            break

    return patterns


def simulate_trades(df: pd.DataFrame, patterns: List[Dict], min_rr: float = 0.0) -> List[Trade]:
    """Simulate trades based on detected patterns."""
    trades = []

    for p in patterns:
        # Calculate R:R before taking trade
        entry_price = p['entry_price']
        stop_loss = p['stop_loss']
        target = p['target']

        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0

        # Filter by minimum R:R
        if rr < min_rr:
            continue
        g_idx = p['g_idx']
        entry_price = p['entry_price']
        stop_loss = p['stop_loss']
        target = p['target']

        is_long = p['type'] == 'long_king'

        exit_time = None
        exit_price = None
        result = None

        # Walk forward from entry to find exit
        for i in range(g_idx + 1, len(df)):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            if is_long:
                # Check stop loss first (hit if low goes below stop)
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = str(df.iloc[i]['time'])
                    result = 'loss'
                    break
                # Check target (hit if high reaches target)
                if high >= target:
                    exit_price = target
                    exit_time = str(df.iloc[i]['time'])
                    result = 'win'
                    break
            else:  # Short
                # Check stop loss first (hit if high goes above stop)
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = str(df.iloc[i]['time'])
                    result = 'loss'
                    break
                # Check target (hit if low reaches target)
                if low <= target:
                    exit_price = target
                    exit_time = str(df.iloc[i]['time'])
                    result = 'win'
                    break

        # Calculate P&L
        if exit_price is not None:
            if is_long:
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            pnl_percent = (pnl / entry_price) * 100

            # Risk/Reward
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr = reward / risk if risk > 0 else 0
        else:
            pnl = None
            pnl_percent = None
            rr = None
            result = 'open'

        trades.append(Trade(
            pattern=p['type'],
            entry_time=p['entry_time'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            exit_time=exit_time,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            result=result,
            risk_reward=rr,
            a_idx=p['a'].index,
            c_idx=p['c'].index if hasattr(p['c'], 'index') else p['c'],
            d_idx=p['d_idx'],
            e_idx=p['e'].index,
            f_idx=p['f_idx'],
            g_idx=p['g_idx']
        ))

    return trades


def simulate_equity(trades: List[Trade], starting_equity: float = 10000.0, risk_per_trade: float = 0.01):
    """
    Simulate equity curve with fixed risk per trade.

    Args:
        trades: List of trades
        starting_equity: Starting capital
        risk_per_trade: Risk per trade as decimal (0.01 = 1%)

    Returns:
        Dict with equity stats
    """
    equity = starting_equity
    peak_equity = starting_equity
    max_drawdown = 0
    max_drawdown_pct = 0
    equity_curve = [starting_equity]
    trade_results = []

    for t in trades:
        if t.result not in ['win', 'loss']:
            continue

        # Calculate position size based on risk
        risk_amount = equity * risk_per_trade  # Dollar risk per trade
        stop_distance = abs(t.entry_price - t.stop_loss)

        if stop_distance == 0:
            continue

        # Position size = risk amount / stop distance
        position_size = risk_amount / stop_distance

        # Calculate actual P&L with position sizing
        if t.pattern == 'long_king':
            trade_pnl = (t.exit_price - t.entry_price) * position_size
        else:  # short
            trade_pnl = (t.entry_price - t.exit_price) * position_size

        # Update equity
        equity += trade_pnl
        equity_curve.append(equity)

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        drawdown_pct = drawdown / peak_equity * 100 if peak_equity > 0 else 0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown = drawdown

        trade_results.append({
            'equity': equity,
            'trade_pnl': trade_pnl,
            'position_size': position_size,
            'r_multiple': trade_pnl / risk_amount if risk_amount > 0 else 0
        })

    total_return = (equity - starting_equity) / starting_equity * 100
    num_trades = len(trade_results)

    return {
        'starting_equity': starting_equity,
        'ending_equity': equity,
        'total_return_pct': total_return,
        'risk_per_trade': risk_per_trade * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'num_trades': num_trades,
        'equity_curve': equity_curve,
        'trade_results': trade_results
    }


def print_results(trades: List[Trade], starting_equity: float = 10000.0, risk_per_trade: float = 0.01):
    """Print backtest results summary."""
    if not trades:
        print("No trades found.")
        return

    # Separate by pattern type
    long_trades = [t for t in trades if t.pattern == 'long_king']
    short_trades = [t for t in trades if t.pattern == 'short_king']

    print("=" * 70)
    print("KING PATTERN BACKTEST RESULTS")
    print("=" * 70)

    for name, subset in [("LONG KING", long_trades), ("SHORT KING", short_trades), ("COMBINED", trades)]:
        if not subset:
            continue

        completed = [t for t in subset if t.result in ['win', 'loss']]
        wins = [t for t in completed if t.result == 'win']
        losses = [t for t in completed if t.result == 'loss']

        print(f"\n{name}")
        print("-" * 40)
        print(f"Total Patterns Found: {len(subset)}")
        print(f"Completed Trades: {len(completed)}")
        print(f"Open Trades: {len(subset) - len(completed)}")

        if completed:
            win_rate = len(wins) / len(completed) * 100
            print(f"Wins: {len(wins)}")
            print(f"Losses: {len(losses)}")
            print(f"Win Rate: {win_rate:.1f}%")

            total_pnl = sum(t.pnl for t in completed if t.pnl)
            avg_pnl = total_pnl / len(completed)
            print(f"Total P&L (raw): ${total_pnl:.2f}")
            print(f"Avg P&L per trade: ${avg_pnl:.2f}")

            avg_win = sum(t.pnl for t in wins if t.pnl) / len(wins) if wins else 0
            avg_loss = sum(t.pnl for t in losses if t.pnl) / len(losses) if losses else 0
            print(f"Avg Win: ${avg_win:.2f}")
            print(f"Avg Loss: ${avg_loss:.2f}")

            if avg_loss != 0:
                profit_factor = abs(sum(t.pnl for t in wins if t.pnl) / sum(t.pnl for t in losses if t.pnl)) if losses else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")

            avg_rr = sum(t.risk_reward for t in completed if t.risk_reward) / len(completed)
            print(f"Avg Risk/Reward: {avg_rr:.2f}")

    # Equity simulation
    print("\n" + "=" * 70)
    print("EQUITY SIMULATION")
    print(f"Starting Equity: ${starting_equity:,.2f}")
    print(f"Risk Per Trade: {risk_per_trade*100:.1f}%")
    print("=" * 70)

    for name, subset in [("LONG KING", long_trades), ("SHORT KING", short_trades), ("COMBINED", trades)]:
        if not subset:
            continue

        eq_stats = simulate_equity(subset, starting_equity, risk_per_trade)

        print(f"\n{name}")
        print("-" * 40)
        print(f"Ending Equity: ${eq_stats['ending_equity']:,.2f}")
        print(f"Total Return: {eq_stats['total_return_pct']:.2f}%")
        print(f"Max Drawdown: ${eq_stats['max_drawdown']:,.2f} ({eq_stats['max_drawdown_pct']:.2f}%)")

        if eq_stats['trade_results']:
            avg_r = sum(tr['r_multiple'] for tr in eq_stats['trade_results']) / len(eq_stats['trade_results'])
            print(f"Avg R-Multiple: {avg_r:.2f}R")

    print("\n" + "=" * 70)
    print("INDIVIDUAL TRADES")
    print("=" * 70)

    for i, t in enumerate(trades, 1):
        print(f"\n[{i}] {t.pattern.upper()}")
        print(f"    Entry: {t.entry_time} @ ${t.entry_price:.2f}")
        print(f"    Stop: ${t.stop_loss:.2f} | Target: ${t.target:.2f}")
        if t.exit_time:
            print(f"    Exit: {t.exit_time} @ ${t.exit_price:.2f}")
            print(f"    Result: {t.result.upper()} | P&L: ${t.pnl:.2f} ({t.pnl_percent:.2f}%)")
        else:
            print(f"    Status: OPEN")


def save_results_csv(trades: List[Trade], filepath: str):
    """Save trade results to CSV."""
    rows = []
    for t in trades:
        rows.append({
            'pattern': t.pattern,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'stop_loss': t.stop_loss,
            'target': t.target,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'result': t.result,
            'risk_reward': t.risk_reward
        })

    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


def extract_symbol(filepath: str) -> str:
    """Extract symbol from filepath (e.g., 'BYBIT_ETHUSDT.P, 5_xxx.csv' -> 'ETHUSDT')."""
    import os
    filename = os.path.basename(filepath)
    # Handle formats like "BYBIT_ETHUSDT.P, 5_xxx.csv" or "ETHUSDT_5min.csv"
    if "BYBIT_" in filename:
        # Extract between BYBIT_ and .P or ,
        symbol = filename.split("BYBIT_")[1].split(".P")[0].split(",")[0]
    elif "_" in filename:
        symbol = filename.split("_")[0]
    else:
        symbol = filename.split(".")[0]
    return symbol.upper()


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='King Pattern Backtest')
    parser.add_argument('filepath', nargs='?', default="BATS_MSFT, 240_ae198.csv",
                        help='Path to CSV file with OHLCV data')
    parser.add_argument('--filter', '-f', choices=['none', 'sma300', 'evwma', 'ewvma_counter'],
                        default='sma300', help='Filter type (default: sma300)')
    parser.add_argument('--min-rr', type=float, default=0.0, help='Minimum R:R ratio (default: 0)')

    args = parser.parse_args()
    filepath = args.filepath
    filter_type = args.filter

    # Extract symbol for SL logic
    symbol = extract_symbol(filepath)
    print(f"Loading data from: {filepath}")
    print(f"Symbol detected: {symbol} (ETH uses candle open SL, others use structure SL)")
    print(f"Filter type: {filter_type}")

    df = load_data(filepath)
    print(f"Loaded {len(df)} candles from {df.iloc[0]['time']} to {df.iloc[-1]['time']}")

    # Find swing points
    print("\nFinding swing points...")
    swings = find_swing_points(df, lookback=3)
    print(f"Found {len(swings)} swing points ({len([s for s in swings if s.type == 'high'])} highs, {len([s for s in swings if s.type == 'low'])} lows)")

    # Find patterns (pass symbol for SL logic)
    print(f"\nScanning for Short King patterns (filter: {filter_type})...")
    short_patterns = find_short_king_patterns(df, swings, symbol, filter_type=filter_type)
    print(f"Found {len(short_patterns)} Short King patterns")

    print(f"\nScanning for Long King patterns (filter: {filter_type})...")
    long_patterns = find_long_king_patterns(df, swings, symbol, filter_type=filter_type)
    print(f"Found {len(long_patterns)} Long King patterns")

    all_patterns = short_patterns + long_patterns
    all_patterns.sort(key=lambda x: x['g_idx'])

    # Simulate trades (FVG filter is now built into pattern detection)
    print(f"\nSimulating trades (FVG entry required, min R:R={args.min_rr})...")
    trades = simulate_trades(df, all_patterns, min_rr=args.min_rr)

    # Print results
    print_results(trades)

    # Save to CSV
    save_results_csv(trades, "king_backtest_results.csv")


if __name__ == "__main__":
    main()

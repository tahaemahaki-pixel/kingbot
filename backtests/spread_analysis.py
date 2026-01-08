"""
Spread/Pairs Trading Analysis - BTC/ETH Cointegration Backtest

Tests cointegration between BTC and ETH and backtests a simple spread trading strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def load_data(btc_path: str, eth_path: str, resample_minutes: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align BTC and ETH price data."""
    # Load CSVs
    btc = pd.read_csv(btc_path, parse_dates=['time'])
    eth = pd.read_csv(eth_path, parse_dates=['time'])

    # Keep only relevant columns
    btc = btc[['time', 'open', 'high', 'low', 'close']].copy()
    eth = eth[['time', 'open', 'high', 'low', 'close']].copy()

    # Resample if requested
    if resample_minutes and resample_minutes > 1:
        print(f"Resampling to {resample_minutes}-minute candles...")

        btc = btc.set_index('time')
        eth = eth.set_index('time')

        btc = btc.resample(f'{resample_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()

        eth = eth.resample(f'{resample_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()

    # Rename for clarity
    btc.columns = ['time', 'btc_open', 'btc_high', 'btc_low', 'btc_close']
    eth.columns = ['time', 'eth_open', 'eth_high', 'eth_low', 'eth_close']

    # Merge on time
    df = pd.merge(btc, eth, on='time', how='inner')
    df = df.sort_values('time').reset_index(drop=True)

    timeframe_str = f"{resample_minutes}-minute" if resample_minutes else "1-minute"
    print(f"Loaded {len(df)} aligned {timeframe_str} candles")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    return df


def test_stationarity(series: pd.Series, name: str) -> Dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag='AIC')

    return {
        'name': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_1%': result[4]['1%'],
        'critical_5%': result[4]['5%'],
        'critical_10%': result[4]['10%'],
        'is_stationary': result[1] < 0.05
    }


def test_cointegration(btc_prices: pd.Series, eth_prices: pd.Series) -> Dict:
    """Test for cointegration between BTC and ETH."""
    # Engle-Granger cointegration test
    coint_result = coint(btc_prices, eth_prices)

    return {
        'coint_statistic': coint_result[0],
        'p_value': coint_result[1],
        'critical_1%': coint_result[2][0],
        'critical_5%': coint_result[2][1],
        'critical_10%': coint_result[2][2],
        'is_cointegrated': coint_result[1] < 0.05
    }


def calculate_hedge_ratio(btc_prices: pd.Series, eth_prices: pd.Series) -> Tuple[float, pd.Series]:
    """Calculate the hedge ratio using OLS regression."""
    # ETH = alpha + beta * BTC + epsilon
    # hedge_ratio = beta (how much ETH to short for each BTC longed)

    X = add_constant(btc_prices)
    model = OLS(eth_prices, X).fit()

    hedge_ratio = model.params.iloc[1]  # beta coefficient
    intercept = model.params.iloc[0]

    # Calculate spread: ETH - hedge_ratio * BTC
    spread = eth_prices - hedge_ratio * btc_prices

    return hedge_ratio, spread, intercept, model.rsquared


def calculate_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Calculate rolling z-score of the spread."""
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore


def calculate_half_life(spread: pd.Series) -> float:
    """Calculate half-life of mean reversion using Ornstein-Uhlenbeck."""
    # Regress spread(t) - spread(t-1) on spread(t-1)
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    # Remove NaN
    valid = ~(spread_lag.isna() | spread_diff.isna())
    spread_lag = spread_lag[valid]
    spread_diff = spread_diff[valid]

    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()

    # Half-life = -ln(2) / theta where theta is the mean reversion coefficient
    theta = model.params.iloc[1]
    if theta >= 0:
        return np.inf  # Not mean-reverting

    half_life = -np.log(2) / theta
    return half_life


def calculate_hurst_exponent(series, max_lag: int = 100) -> float:
    """Calculate Hurst exponent to test for mean reversion.
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    # Convert to numpy array if needed
    if hasattr(series, 'values'):
        series = series.values

    lags = range(2, min(max_lag, len(series) // 2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

    # Linear fit to log-log plot
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def backtest_spread_strategy(
    df: pd.DataFrame,
    hedge_ratio: float,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    zscore_window: int = 60,
    stop_loss_zscore: float = 4.0,
    risk_per_trade: float = 0.02,
    starting_capital: float = 10000
) -> Dict:
    """
    Backtest a simple spread trading strategy.

    Entry: When z-score crosses ±entry_zscore
    Exit: When z-score crosses back to ±exit_zscore
    Stop: When z-score reaches ±stop_loss_zscore

    Long spread = Long ETH, Short BTC (when z-score < -entry)
    Short spread = Short ETH, Long BTC (when z-score > +entry)
    """
    # Calculate spread and z-score
    spread = df['eth_close'] - hedge_ratio * df['btc_close']
    zscore = calculate_zscore(spread, window=zscore_window)

    # Trading state
    position = 0  # 1 = long spread, -1 = short spread, 0 = flat
    entry_idx = None
    entry_spread = None
    entry_zscore_val = None

    # Results tracking
    trades = []
    equity_curve = [starting_capital]
    capital = starting_capital

    for i in range(zscore_window, len(df)):
        z = zscore.iloc[i]
        current_spread = spread.iloc[i]

        if pd.isna(z):
            equity_curve.append(capital)
            continue

        # Entry logic
        if position == 0:
            if z < -entry_zscore:
                # Long spread (buy ETH, sell BTC)
                position = 1
                entry_idx = i
                entry_spread = current_spread
                entry_zscore_val = z

            elif z > entry_zscore:
                # Short spread (sell ETH, buy BTC)
                position = -1
                entry_idx = i
                entry_spread = current_spread
                entry_zscore_val = z

        # Exit logic
        elif position == 1:  # Long spread
            # Exit on mean reversion or stop loss
            if z >= -exit_zscore or z < -stop_loss_zscore:
                spread_pnl = current_spread - entry_spread
                # Normalize P&L relative to spread volatility
                spread_std = spread.iloc[i-zscore_window:i].std()
                pnl_pct = (spread_pnl / spread_std) * risk_per_trade
                pnl = capital * pnl_pct
                capital += pnl

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_time': df['time'].iloc[entry_idx],
                    'exit_time': df['time'].iloc[i],
                    'direction': 'long_spread',
                    'entry_zscore': entry_zscore_val,
                    'exit_zscore': z,
                    'entry_spread': entry_spread,
                    'exit_spread': current_spread,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': 'mean_reversion' if z >= -exit_zscore else 'stop_loss'
                })
                position = 0

        elif position == -1:  # Short spread
            if z <= exit_zscore or z > stop_loss_zscore:
                spread_pnl = entry_spread - current_spread
                spread_std = spread.iloc[i-zscore_window:i].std()
                pnl_pct = (spread_pnl / spread_std) * risk_per_trade
                pnl = capital * pnl_pct
                capital += pnl

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_time': df['time'].iloc[entry_idx],
                    'exit_time': df['time'].iloc[i],
                    'direction': 'short_spread',
                    'entry_zscore': entry_zscore_val,
                    'exit_zscore': z,
                    'entry_spread': entry_spread,
                    'exit_spread': current_spread,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': 'mean_reversion' if z <= exit_zscore else 'stop_loss'
                })
                position = 0

        equity_curve.append(capital)

    # Calculate statistics
    if not trades:
        return {'error': 'No trades generated'}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]

    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100

    return {
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'total_pnl': trades_df['pnl'].sum(),
        'total_return_pct': (capital - starting_capital) / starting_capital * 100,
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
        'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
        'largest_win': trades_df['pnl'].max(),
        'largest_loss': trades_df['pnl'].min(),
        'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf'),
        'max_drawdown_pct': drawdown.min(),
        'final_equity': capital,
        'mean_reversion_exits': len(trades_df[trades_df['exit_reason'] == 'mean_reversion']),
        'stop_loss_exits': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
        'avg_trade_duration': (trades_df['exit_idx'] - trades_df['entry_idx']).mean(),
        'trades': trades_df
    }


def run_analysis(btc_path: str, eth_path: str, resample_minutes: int = None):
    """Run full cointegration analysis and backtest."""
    print("=" * 70)
    print("BTC/ETH SPREAD TRADING ANALYSIS")
    if resample_minutes:
        print(f"Timeframe: {resample_minutes}-minute candles")
    print("=" * 70)

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)
    df = load_data(btc_path, eth_path, resample_minutes=resample_minutes)

    btc_prices = df['btc_close']
    eth_prices = df['eth_close']

    # Test stationarity of individual price series
    print("\n2. STATIONARITY TESTS (ADF)")
    print("-" * 40)

    btc_adf = test_stationarity(btc_prices, "BTC")
    eth_adf = test_stationarity(eth_prices, "ETH")

    print(f"BTC: ADF={btc_adf['adf_statistic']:.4f}, p={btc_adf['p_value']:.4f} -> {'Stationary' if btc_adf['is_stationary'] else 'Non-stationary'}")
    print(f"ETH: ADF={eth_adf['adf_statistic']:.4f}, p={eth_adf['p_value']:.4f} -> {'Stationary' if eth_adf['is_stationary'] else 'Non-stationary'}")

    if not btc_adf['is_stationary'] and not eth_adf['is_stationary']:
        print("\nBoth series are non-stationary (expected for prices) - proceeding with cointegration test")

    # Test cointegration
    print("\n3. COINTEGRATION TEST (Engle-Granger)")
    print("-" * 40)

    coint_result = test_cointegration(btc_prices, eth_prices)

    print(f"Cointegration statistic: {coint_result['coint_statistic']:.4f}")
    print(f"P-value: {coint_result['p_value']:.4f}")
    print(f"Critical values: 1%={coint_result['critical_1%']:.2f}, 5%={coint_result['critical_5%']:.2f}, 10%={coint_result['critical_10%']:.2f}")
    print(f"\nResult: {'COINTEGRATED' if coint_result['is_cointegrated'] else 'NOT COINTEGRATED'} (at 5% significance)")

    # Calculate hedge ratio and spread
    print("\n4. HEDGE RATIO & SPREAD")
    print("-" * 40)

    hedge_ratio, spread, intercept, r_squared = calculate_hedge_ratio(btc_prices, eth_prices)

    print(f"Hedge ratio (beta): {hedge_ratio:.6f}")
    print(f"Intercept (alpha): {intercept:.2f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"\nInterpretation: For every 1 BTC, hedge with {hedge_ratio:.4f} ETH")

    # Test spread stationarity
    print("\n5. SPREAD STATIONARITY")
    print("-" * 40)

    spread_adf = test_stationarity(spread, "Spread")
    print(f"Spread ADF: {spread_adf['adf_statistic']:.4f}, p={spread_adf['p_value']:.6f}")
    print(f"Result: {'STATIONARY' if spread_adf['is_stationary'] else 'Non-stationary'}")

    # Calculate half-life
    half_life = calculate_half_life(spread)
    candle_minutes = resample_minutes if resample_minutes else 1
    half_life_minutes = half_life * candle_minutes
    half_life_hours = half_life_minutes / 60
    print(f"Half-life of mean reversion: {half_life:.1f} candles ({half_life_hours:.1f} hours)")

    # Calculate Hurst exponent
    hurst = calculate_hurst_exponent(spread.dropna().values)
    print(f"Hurst exponent: {hurst:.4f} ({'Mean-reverting' if hurst < 0.5 else 'Trending' if hurst > 0.5 else 'Random walk'})")

    # Spread statistics
    print("\n6. SPREAD STATISTICS")
    print("-" * 40)

    zscore = calculate_zscore(spread, window=60)

    print(f"Spread mean: {spread.mean():.2f}")
    print(f"Spread std: {spread.std():.2f}")
    print(f"Current spread: {spread.iloc[-1]:.2f}")
    print(f"Current z-score: {zscore.iloc[-1]:.2f}")
    print(f"Times z-score > 2: {(zscore.abs() > 2).sum()} ({(zscore.abs() > 2).sum() / len(zscore) * 100:.1f}%)")
    print(f"Times z-score > 3: {(zscore.abs() > 3).sum()} ({(zscore.abs() > 3).sum() / len(zscore) * 100:.1f}%)")

    # Backtest
    print("\n7. BACKTEST RESULTS")
    print("-" * 40)

    results = backtest_spread_strategy(
        df,
        hedge_ratio=hedge_ratio,
        entry_zscore=2.0,
        exit_zscore=0.0,
        zscore_window=60,
        stop_loss_zscore=4.0,
        risk_per_trade=0.02,
        starting_capital=10000
    )

    if 'error' in results:
        print(f"Backtest error: {results['error']}")
    else:
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.1f}%")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print(f"Total return: {results['total_return_pct']:.2f}%")
        print(f"Final equity: ${results['final_equity']:.2f}")
        print(f"Max drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Avg trade duration: {results['avg_trade_duration']:.0f} candles")
        print(f"Mean reversion exits: {results['mean_reversion_exits']}")
        print(f"Stop loss exits: {results['stop_loss_exits']}")

        print("\n" + "-" * 40)
        print("P&L BREAKDOWN")
        print("-" * 40)
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Avg win: ${results['avg_win']:.2f}")
        print(f"Avg loss: ${results['avg_loss']:.2f}")
        print(f"Largest win: ${results['largest_win']:.2f}")
        print(f"Largest loss: ${results['largest_loss']:.2f}")

        # Show recent trades
        if len(results['trades']) > 0:
            print("\n" + "-" * 40)
            print("RECENT TRADES")
            print("-" * 40)
            recent = results['trades'].tail(10)
            for _, t in recent.iterrows():
                direction = "LONG" if t['direction'] == 'long_spread' else "SHORT"
                print(f"{t['entry_time']} -> {t['exit_time']}: {direction} spread, P&L: ${t['pnl']:.2f} ({t['exit_reason']})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    is_viable = (
        coint_result['is_cointegrated'] and
        spread_adf['is_stationary'] and
        hurst < 0.5 and
        half_life < 200
    )

    print(f"Cointegrated: {'YES' if coint_result['is_cointegrated'] else 'NO'}")
    print(f"Spread stationary: {'YES' if spread_adf['is_stationary'] else 'NO'}")
    print(f"Mean-reverting (Hurst < 0.5): {'YES' if hurst < 0.5 else 'NO'}")
    print(f"Reasonable half-life: {'YES' if half_life < 200 else 'NO'}")
    print(f"\nOVERALL: {'VIABLE FOR SPREAD TRADING' if is_viable else 'NOT RECOMMENDED'}")

    return {
        'df': df,
        'coint_result': coint_result,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'zscore': zscore,
        'half_life': half_life,
        'hurst': hurst,
        'backtest': results
    }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='BTC/ETH Spread Trading Analysis')
    parser.add_argument('--btc', default="/home/tahae/ai-content/data/Tradingdata/BYBIT_BTCUSDT.P, 1_da562.csv",
                        help='Path to BTC CSV file')
    parser.add_argument('--eth', default="/home/tahae/ai-content/data/Tradingdata/BYBIT_ETHUSDT.P, 1_79d61.csv",
                        help='Path to ETH CSV file')
    parser.add_argument('--timeframe', '-tf', type=int, default=1,
                        help='Timeframe in minutes (default: 1, use 5 for 5-min)')
    parser.add_argument('--entry', type=float, default=2.0,
                        help='Z-score entry threshold (default: 2.0)')
    parser.add_argument('--exit', type=float, default=0.0,
                        help='Z-score exit threshold (default: 0.0)')
    parser.add_argument('--window', type=int, default=60,
                        help='Z-score lookback window in candles (default: 60)')

    args = parser.parse_args()

    resample = args.timeframe if args.timeframe > 1 else None
    results = run_analysis(args.btc, args.eth, resample_minutes=resample)

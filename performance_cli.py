"""
Performance CLI - Command-line interface for trading performance analysis.

Usage:
    python start.py stats [period] [--symbol SYMBOL] [--from DATE] [--to DATE]
    python start.py trades [-n NUM] [--symbol SYMBOL] [--winners] [--losers]
    python start.py equity [--period PERIOD] [--interval INTERVAL]
    python start.py assets [--sort FIELD]
    python start.py sessions [-n NUM] [--sort FIELD]
    python start.py time [--period PERIOD]
    python start.py export TYPE [--format FORMAT] [--output FILE]
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Optional
from trade_tracker import get_tracker


class PerformanceCLI:
    """Argparse-based CLI for performance queries."""

    def __init__(self):
        self.tracker = get_tracker()

    def run(self, args: List[str]) -> None:
        """Parse and execute CLI command."""
        parser = argparse.ArgumentParser(
            prog='performance',
            description='Trading Performance Analysis CLI'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show trading statistics')
        stats_parser.add_argument('period', nargs='?', default='all',
                                  choices=['today', 'week', 'month', 'year', 'all'],
                                  help='Time period (default: all)')
        stats_parser.add_argument('--symbol', '-s', help='Filter by symbol')
        stats_parser.add_argument('--from', dest='from_date', help='Start date (YYYY-MM-DD)')
        stats_parser.add_argument('--to', dest='to_date', help='End date (YYYY-MM-DD)')

        # Trades command
        trades_parser = subparsers.add_parser('trades', help='List trades')
        trades_parser.add_argument('-n', type=int, default=20, help='Number of trades (default: 20)')
        trades_parser.add_argument('--symbol', '-s', help='Filter by symbol')
        trades_parser.add_argument('--direction', '-d', choices=['long', 'short'], help='Filter by direction')
        trades_parser.add_argument('--winners', '-w', action='store_true', help='Show only winners')
        trades_parser.add_argument('--losers', '-l', action='store_true', help='Show only losers')
        trades_parser.add_argument('--from', dest='from_date', help='Start date (YYYY-MM-DD)')
        trades_parser.add_argument('--to', dest='to_date', help='End date (YYYY-MM-DD)')

        # Equity command
        equity_parser = subparsers.add_parser('equity', help='Show equity curve and drawdowns')
        equity_parser.add_argument('--period', '-p', default='week',
                                   choices=['day', 'week', 'month', 'year', 'all'],
                                   help='Time period (default: week)')
        equity_parser.add_argument('--interval', '-i', default='1h',
                                   choices=['5m', '15m', '1h', '4h', '1d'],
                                   help='Interval (default: 1h)')

        # Assets command
        assets_parser = subparsers.add_parser('assets', help='Show per-asset breakdown')
        assets_parser.add_argument('--sort', default='pnl',
                                   choices=['pnl', 'trades', 'winrate', 'r'],
                                   help='Sort field (default: pnl)')

        # Sessions command
        sessions_parser = subparsers.add_parser('sessions', help='Show best/worst trading sessions')
        sessions_parser.add_argument('-n', type=int, default=10, help='Number of sessions (default: 10)')
        sessions_parser.add_argument('--sort', default='pnl',
                                     choices=['pnl', 'trades', 'winrate'],
                                     help='Sort field (default: pnl)')

        # Time command
        time_parser = subparsers.add_parser('time', help='Show time-based analysis')
        time_parser.add_argument('--period', '-p', default='all',
                                 choices=['week', 'month', 'year', 'all'],
                                 help='Time period (default: all)')

        # Export command
        export_parser = subparsers.add_parser('export', help='Export data to file')
        export_parser.add_argument('type', choices=['trades', 'daily', 'equity'],
                                   help='Data type to export')
        export_parser.add_argument('--format', '-f', default='csv',
                                   choices=['csv', 'json'],
                                   help='Output format (default: csv)')
        export_parser.add_argument('--output', '-o', help='Output file path')
        export_parser.add_argument('--from', dest='from_date', help='Start date (YYYY-MM-DD)')
        export_parser.add_argument('--to', dest='to_date', help='End date (YYYY-MM-DD)')

        # Streaks command
        subparsers.add_parser('streaks', help='Show win/loss streaks')

        parsed = parser.parse_args(args)

        if not parsed.command:
            parser.print_help()
            return

        # Route to handler
        handler = getattr(self, f'cmd_{parsed.command}', None)
        if handler:
            handler(parsed)
        else:
            print(f"Unknown command: {parsed.command}")

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {date_str} (expected YYYY-MM-DD)")
            return None

    def _format_currency(self, value: float) -> str:
        """Format currency value."""
        if value >= 0:
            return f"+${value:,.2f}"
        else:
            return f"-${abs(value):,.2f}"

    def _format_percent(self, value: float) -> str:
        """Format percentage value."""
        if value >= 0:
            return f"+{value:.2f}%"
        else:
            return f"{value:.2f}%"

    def _format_r(self, value: Optional[float]) -> str:
        """Format R-multiple."""
        if value is None:
            return "N/A"
        if value >= 0:
            return f"+{value:.2f}R"
        else:
            return f"{value:.2f}R"

    def cmd_stats(self, args) -> None:
        """Show trading statistics."""
        start_date = self._parse_date(args.from_date)
        end_date = self._parse_date(args.to_date)

        stats = self.tracker.get_stats(
            period=args.period,
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date
        )

        # Header
        print("\n" + "=" * 60)
        print("TRADING PERFORMANCE STATISTICS")
        if args.symbol:
            print(f"Symbol: {args.symbol}")
        if args.period != 'all':
            print(f"Period: {args.period}")
        if start_date or end_date:
            date_range = f"{start_date.strftime('%Y-%m-%d') if start_date else 'start'}"
            date_range += f" to {end_date.strftime('%Y-%m-%d') if end_date else 'now'}"
            print(f"Date Range: {date_range}")
        print("=" * 60)

        # Overview
        print("\n" + "-" * 20 + "Overview" + "-" * 20)
        print(f"{'Total Trades:':<25}{stats['total_trades']:>15}")
        print(f"{'Winning Trades:':<25}{stats['winning_trades']:>15}")
        print(f"{'Losing Trades:':<25}{stats['losing_trades']:>15}")
        print(f"{'Win Rate:':<25}{stats['win_rate']:>14.1f}%")
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "INF"
        print(f"{'Profit Factor:':<25}{pf_str:>15}")

        # P&L
        print("\n" + "-" * 17 + "Profit & Loss" + "-" * 18)
        print(f"{'Net P&L:':<25}{self._format_currency(stats.get('total_pnl', 0)):>15}")
        print(f"{'Gross Profit:':<25}{self._format_currency(stats.get('gross_profit', 0)):>15}")
        print(f"{'Gross Loss:':<25}{self._format_currency(stats.get('gross_loss', 0)):>15}")
        print(f"{'Largest Win:':<25}{self._format_currency(stats.get('largest_win', 0)):>15}")
        print(f"{'Largest Loss:':<25}{self._format_currency(stats.get('largest_loss', 0)):>15}")
        print(f"{'Average Win:':<25}{self._format_currency(stats.get('avg_win', 0)):>15}")
        print(f"{'Average Loss:':<25}{self._format_currency(stats.get('avg_loss', 0)):>15}")

        # Risk Metrics
        print("\n" + "-" * 17 + "Risk Metrics" + "-" * 19)
        print(f"{'Max Drawdown:':<25}{stats.get('max_drawdown_pct', 0):>14.2f}%")
        # Get current drawdown
        drawdown = self.tracker.get_current_drawdown()
        print(f"{'Current Drawdown:':<25}{drawdown.get('drawdown_pct', 0):>14.2f}%")
        print(f"{'Avg R-Multiple:':<25}{self._format_r(stats.get('avg_r_multiple')):>15}")
        print(f"{'Expectancy:':<25}{self._format_currency(stats.get('expectancy', 0)):>15}")
        print(f"{'Win Streak (max):':<25}{stats.get('max_win_streak', 0):>15}")
        print(f"{'Loss Streak (max):':<25}{stats.get('max_loss_streak', 0):>15}")

        print("=" * 60 + "\n")

    def cmd_trades(self, args) -> None:
        """List trades."""
        # Determine winner/loser filter
        winners_only = None
        if args.winners:
            winners_only = True
        elif args.losers:
            winners_only = False

        start_date = self._parse_date(args.from_date)
        end_date = self._parse_date(args.to_date)

        # Map direction filter
        direction = None
        if args.direction:
            direction = 'long' if args.direction == 'long' else 'short'

        # Set min/max PnL for winners/losers filter
        min_pnl = None
        max_pnl = None
        if winners_only is True:
            min_pnl = 0.01  # Winners only
        elif winners_only is False:
            max_pnl = -0.01  # Losers only

        trades = self.tracker.get_trades(
            limit=args.n,
            symbol=args.symbol,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            min_pnl=min_pnl,
            max_pnl=max_pnl
        )

        if not trades:
            print("\nNo trades found matching criteria.\n")
            return

        # Header
        print("\n" + "=" * 95)
        print("TRADE HISTORY")
        print("=" * 95)

        # Column headers
        print(f"{'Time':<16} {'Symbol':<12} {'Type':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'R':>8} {'Reason':<10}")
        print("-" * 95)

        for trade in trades:
            time_str = trade['closed_at'].strftime('%m/%d %H:%M') if trade['closed_at'] else trade['opened_at'].strftime('%m/%d %H:%M')
            symbol = trade['symbol'][:10]
            direction = 'LONG' if 'long' in trade['signal_type'] else 'SHORT'
            entry = f"${trade['entry_price']:.4f}" if trade['entry_price'] else "PENDING"
            exit_p = f"${trade['exit_price']:.4f}" if trade['exit_price'] else "-"
            pnl = self._format_currency(trade['realized_pnl'])
            r_mult = self._format_r(trade['r_multiple'])
            reason = trade['exit_reason'][:10] if trade['exit_reason'] else trade['status']

            print(f"{time_str:<16} {symbol:<12} {direction:<6} {entry:>10} {exit_p:>10} {pnl:>12} {r_mult:>8} {reason:<10}")

        print("=" * 95)
        print(f"Showing {len(trades)} trades")
        print()

    def cmd_equity(self, args) -> None:
        """Show equity curve and drawdowns."""
        # Calculate time range
        now = datetime.now()
        period_map = {
            'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            'year': timedelta(days=365),
            'all': timedelta(days=3650)
        }
        start_date = now - period_map.get(args.period, timedelta(weeks=1))

        # Get equity curve
        curve = self.tracker.get_equity_curve(start_date=start_date)

        if not curve:
            print("\nNo equity data available.\n")
            return

        # Get drawdown info
        drawdown = self.tracker.get_current_drawdown()
        max_dd = self.tracker.get_max_drawdown(start_date=start_date)

        print("\n" + "=" * 60)
        print("EQUITY & DRAWDOWN ANALYSIS")
        print(f"Period: Last {args.period}")
        print("=" * 60)

        # Current status
        print("\n" + "-" * 18 + "Current Status" + "-" * 18)
        print(f"{'Current Equity:':<25}${drawdown.get('current_equity', 0):>14,.2f}")
        print(f"{'Peak Equity:':<25}${drawdown.get('peak_equity', 0):>14,.2f}")
        print(f"{'Current Drawdown:':<25}{drawdown.get('drawdown_pct', 0):>14.2f}%")
        print(f"{'Open Positions:':<25}{drawdown.get('open_positions', 0):>15}")

        # Max drawdown
        print("\n" + "-" * 17 + "Maximum Drawdown" + "-" * 17)
        print(f"{'Max Drawdown:':<25}{max_dd.get('pct', 0):>14.2f}%")
        print(f"{'Peak Equity:':<25}${max_dd.get('peak', 0):>14,.2f}")
        print(f"{'Drawdown Amount:':<25}${max_dd.get('amount', 0):>14,.2f}")
        if max_dd.get('date'):
            print(f"{'Date:':<25}{max_dd['date'].strftime('%Y-%m-%d %H:%M') if hasattr(max_dd['date'], 'strftime') else str(max_dd['date']):>25}")

        # Equity curve summary
        if len(curve) > 1:
            print("\n" + "-" * 17 + "Equity Summary" + "-" * 18)
            start_eq = curve[0]['equity']
            end_eq = curve[-1]['equity']
            change = end_eq - start_eq
            change_pct = (change / start_eq * 100) if start_eq > 0 else 0
            high_eq = max(c['equity'] for c in curve)
            low_eq = min(c['equity'] for c in curve)

            print(f"{'Starting Equity:':<25}${start_eq:>14,.2f}")
            print(f"{'Ending Equity:':<25}${end_eq:>14,.2f}")
            print(f"{'Change:':<25}{self._format_currency(change):>15} ({self._format_percent(change_pct)})")
            print(f"{'High:':<25}${high_eq:>14,.2f}")
            print(f"{'Low:':<25}${low_eq:>14,.2f}")
            print(f"{'Data Points:':<25}{len(curve):>15}")

        print("=" * 60 + "\n")

    def cmd_assets(self, args) -> None:
        """Show per-asset breakdown."""
        breakdown = self.tracker.get_asset_breakdown()

        if not breakdown:
            print("\nNo asset data available.\n")
            return

        # Sort
        sort_key = {
            'pnl': lambda x: x[1]['net_pnl'],
            'trades': lambda x: x[1]['total_trades'],
            'winrate': lambda x: x[1]['win_rate'],
            'r': lambda x: x[1]['avg_r_multiple'] or 0
        }
        sorted_assets = sorted(breakdown.items(), key=sort_key.get(args.sort, sort_key['pnl']), reverse=True)

        print("\n" + "=" * 85)
        print("ASSET BREAKDOWN")
        print(f"Sorted by: {args.sort}")
        print("=" * 85)

        # Column headers
        print(f"{'Symbol':<12} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'Net P&L':>14} {'Avg R':>10} {'PF':>8}")
        print("-" * 85)

        total_trades = 0
        total_pnl = 0

        for symbol, stats in sorted_assets:
            pf = stats['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float('inf') else "INF"

            print(f"{symbol:<12} {stats['total_trades']:>8} {stats['winning_trades']:>6} "
                  f"{stats['win_rate']:>7.1f}% {self._format_currency(stats['net_pnl']):>14} "
                  f"{self._format_r(stats['avg_r_multiple']):>10} {pf_str:>8}")

            total_trades += stats['total_trades']
            total_pnl += stats['net_pnl']

        print("-" * 85)
        print(f"{'TOTAL':<12} {total_trades:>8} {'':<6} {'':<8} {self._format_currency(total_pnl):>14}")
        print("=" * 85 + "\n")

    def cmd_sessions(self, args) -> None:
        """Show best/worst trading sessions."""
        sessions = self.tracker.get_sessions(limit=args.n, sort_by=args.sort)

        if not sessions['best'] and not sessions['worst']:
            print("\nNo session data available.\n")
            return

        print("\n" + "=" * 75)
        print("TRADING SESSIONS")
        print(f"Sorted by: {args.sort}")
        print("=" * 75)

        # Best sessions
        if sessions['best']:
            print("\n" + "-" * 20 + " BEST DAYS " + "-" * 20)
            print(f"{'Date':<12} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'Net P&L':>14}")
            print("-" * 50)

            for day in sessions['best']:
                win_pct = (day['winning_trades'] / day['total_trades'] * 100) if day['total_trades'] > 0 else 0
                print(f"{day['date'].strftime('%Y-%m-%d'):<12} {day['total_trades']:>8} "
                      f"{day['winning_trades']:>6} {win_pct:>7.1f}% {self._format_currency(day['net_pnl']):>14}")

        # Worst sessions
        if sessions['worst']:
            print("\n" + "-" * 20 + " WORST DAYS " + "-" * 19)
            print(f"{'Date':<12} {'Trades':>8} {'Wins':>6} {'Win%':>8} {'Net P&L':>14}")
            print("-" * 50)

            for day in sessions['worst']:
                win_pct = (day['winning_trades'] / day['total_trades'] * 100) if day['total_trades'] > 0 else 0
                print(f"{day['date'].strftime('%Y-%m-%d'):<12} {day['total_trades']:>8} "
                      f"{day['winning_trades']:>6} {win_pct:>7.1f}% {self._format_currency(day['net_pnl']):>14}")

        print("=" * 75 + "\n")

    def cmd_time(self, args) -> None:
        """Show time-based analysis."""
        # Calculate time range
        now = datetime.now()
        period_map = {
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            'year': timedelta(days=365),
            'all': None
        }
        start_date = None
        if args.period != 'all':
            start_date = now - period_map.get(args.period, timedelta(days=30))

        analysis = self.tracker.get_time_analysis(start_date=start_date)

        print("\n" + "=" * 70)
        print("TIME-BASED ANALYSIS")
        if args.period != 'all':
            print(f"Period: Last {args.period}")
        print("=" * 70)

        # By hour
        if analysis['by_hour']:
            print("\n" + "-" * 20 + " BY HOUR (UTC) " + "-" * 20)
            print(f"{'Hour':<8} {'Trades':>8} {'Win%':>8} {'Net P&L':>14} {'Avg R':>10}")
            print("-" * 55)

            for hour_data in sorted(analysis['by_hour'], key=lambda x: x['hour']):
                print(f"{hour_data['hour']:02d}:00   {hour_data['trades']:>8} "
                      f"{hour_data['win_rate']:>7.1f}% {self._format_currency(hour_data['net_pnl']):>14} "
                      f"{self._format_r(hour_data['avg_r']):>10}")

        # By day of week
        if analysis['by_day_of_week']:
            print("\n" + "-" * 18 + " BY DAY OF WEEK " + "-" * 18)
            print(f"{'Day':<12} {'Trades':>8} {'Win%':>8} {'Net P&L':>14} {'Avg R':>10}")
            print("-" * 55)

            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day_data in sorted(analysis['by_day_of_week'], key=lambda x: x['day_of_week']):
                day_name = day_names[day_data['day_of_week']]
                print(f"{day_name:<12} {day_data['trades']:>8} "
                      f"{day_data['win_rate']:>7.1f}% {self._format_currency(day_data['net_pnl']):>14} "
                      f"{self._format_r(day_data['avg_r']):>10}")

        print("=" * 70 + "\n")

    def cmd_streaks(self, args) -> None:
        """Show win/loss streaks."""
        streaks = self.tracker.get_streaks()

        print("\n" + "=" * 50)
        print("WIN/LOSS STREAKS")
        print("=" * 50)

        print("\n" + "-" * 18 + " Current " + "-" * 18)
        print(f"{'Current Win Streak:':<25}{streaks['current_win_streak']:>10}")
        print(f"{'Current Loss Streak:':<25}{streaks['current_loss_streak']:>10}")

        print("\n" + "-" * 18 + " All-Time " + "-" * 17)
        print(f"{'Max Win Streak:':<25}{streaks['max_win_streak']:>10}")
        print(f"{'Max Loss Streak:':<25}{streaks['max_loss_streak']:>10}")

        print("=" * 50 + "\n")

    def cmd_export(self, args) -> None:
        """Export data to file."""
        start_date = self._parse_date(args.from_date)
        end_date = self._parse_date(args.to_date)

        # Generate default filename
        if args.output:
            filepath = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"{args.type}_{timestamp}.{args.format}"

        try:
            if args.type == 'trades':
                self.tracker.export_trades(filepath, args.format, start_date, end_date)
            elif args.type == 'daily':
                self.tracker.export_daily_stats(filepath, args.format, start_date, end_date)
            elif args.type == 'equity':
                # Export equity snapshots
                curve = self.tracker.get_equity_curve(start_date=start_date, end_date=end_date)
                if args.format == 'csv':
                    import csv
                    with open(filepath, 'w', newline='') as f:
                        if curve:
                            writer = csv.DictWriter(f, fieldnames=curve[0].keys())
                            writer.writeheader()
                            writer.writerows(curve)
                else:
                    import json
                    with open(filepath, 'w') as f:
                        # Convert datetime objects
                        for item in curve:
                            if 'timestamp' in item:
                                item['timestamp'] = item['timestamp'].isoformat()
                        json.dump(curve, f, indent=2)

            print(f"\nExported {args.type} to: {filepath}\n")

        except Exception as e:
            print(f"\nError exporting data: {e}\n")


def main():
    """Entry point for CLI."""
    import sys
    cli = PerformanceCLI()
    cli.run(sys.argv[1:])


if __name__ == '__main__':
    main()

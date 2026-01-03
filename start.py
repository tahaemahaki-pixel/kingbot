#!/usr/bin/env python3
"""Start script for King Trading Bot"""
import os
import sys
import atexit

# Set working directory and load env FIRST
BOT_DIR = '/home/tahae/ai-content/data/Tradingdata/bybit_bot'
PID_FILE = os.path.join(BOT_DIR, 'bot.pid')

os.chdir(BOT_DIR)
sys.path.insert(0, BOT_DIR)


def check_single_instance():
    """Ensure only one instance of the bot is running."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())

            # Check if process is still running
            try:
                os.kill(old_pid, 0)  # Signal 0 = check if process exists
                print(f"ERROR: Bot is already running (PID {old_pid})")
                print(f"To stop it: kill {old_pid}")
                print(f"Or: pkill -f start.py")
                sys.exit(1)
            except OSError:
                # Process not running, remove stale PID file
                os.remove(PID_FILE)
                print(f"Removed stale PID file (old PID {old_pid})")
        except (ValueError, IOError):
            os.remove(PID_FILE)

    # Write current PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    print(f"Bot starting with PID {os.getpid()}")


def cleanup_pid():
    """Remove PID file on exit."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except:
        pass


# Register cleanup
atexit.register(cleanup_pid)

# Check for existing instance
check_single_instance()

# Load environment BEFORE any imports
from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, '.env'), override=True)

# Now import and start
from bot import TradingBot
from config import BotConfig

config = BotConfig.from_env()
print(f"Telegram notifications {'enabled' if config.telegram_token else 'disabled'}")

bot = TradingBot(config)
bot.start()

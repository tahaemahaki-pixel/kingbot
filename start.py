#!/usr/bin/env python3
"""Start script for King Trading Bot"""
import os
import sys

# Set working directory and load env FIRST
os.chdir('/home/tahae/ai-content/data/Tradingdata/bybit_bot')
sys.path.insert(0, '/home/tahae/ai-content/data/Tradingdata/bybit_bot')

# Load environment BEFORE any imports
from dotenv import load_dotenv
load_dotenv('/home/tahae/ai-content/data/Tradingdata/bybit_bot/.env', override=True)

# Debug
print(f"DEBUG: TELEGRAM_TOKEN = {os.getenv('TELEGRAM_TOKEN', 'NONE')[:20]}..." if os.getenv('TELEGRAM_TOKEN') else "DEBUG: TELEGRAM_TOKEN = NONE")
print(f"DEBUG: TELEGRAM_CHAT_ID = {os.getenv('TELEGRAM_CHAT_ID', 'NONE')}")

# Now import and start
from bot import TradingBot
from config import BotConfig

config = BotConfig.from_env()
print(f"DEBUG: config.telegram_token = {config.telegram_token[:20] if config.telegram_token else 'None'}...")
print(f"DEBUG: config.telegram_chat_id = {config.telegram_chat_id}")

bot = TradingBot(config)
bot.start()

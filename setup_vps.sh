#!/bin/bash
# King Trading Bot - VPS Setup Script
# Run this on a fresh Ubuntu VPS

set -e

echo "=========================================="
echo "King Trading Bot - VPS Setup"
echo "=========================================="

# Update system
echo "[1/5] Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "[2/5] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Create bot directory
echo "[3/5] Setting up bot directory..."
mkdir -p ~/kingbot
cd ~/kingbot

# Clone repository
echo "[4/5] Cloning bot from GitHub..."
if [ -d ".git" ]; then
    git pull
else
    git clone https://github.com/tahaemahaki-pixel/kingbot.git .
fi

# Install Python packages
echo "[5/5] Installing Python packages..."
pip3 install --break-system-packages python-dotenv requests websocket-client

# Create .env file template
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Bybit API Credentials
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=false

# Trading Settings
TRADING_SYMBOL=BTCUSDT
TRADING_TIMEFRAME=5
RISK_PER_TRADE=0.01

# Telegram Notifications
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
EOF
    echo ""
    echo "IMPORTANT: Edit .env with your API keys!"
    echo "Run: nano ~/kingbot/.env"
fi

# Create systemd service for auto-start
echo "Creating systemd service..."
sudo tee /etc/systemd/system/kingbot.service > /dev/null << EOF
[Unit]
Description=King Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/kingbot
ExecStart=/usr/bin/python3 -u $HOME/kingbot/start.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit your API keys:  nano ~/kingbot/.env"
echo "2. Start the bot:       sudo systemctl start kingbot"
echo "3. Enable auto-start:   sudo systemctl enable kingbot"
echo "4. View logs:           sudo journalctl -u kingbot -f"
echo "5. Stop bot:            sudo systemctl stop kingbot"
echo ""

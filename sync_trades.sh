#!/bin/bash
# Sync trade CSVs from VPS to local machine
# Run manually or add to cron: 0 1 * * * /path/to/sync_trades.sh

VPS_HOST="root@209.38.84.47"
VPS_PATH="/root/kingbot/exports"
LOCAL_PATH="/home/tahae/ai-content/data/Tradingdata/bybit_bot/exports"

# Create local exports directory
mkdir -p "$LOCAL_PATH"

# First, trigger an export on VPS to get latest data
echo "Exporting trades on VPS..."
ssh "$VPS_HOST" "cd /root/kingbot && python3 export_trades.py"

# Sync all CSV files
echo "Syncing to local machine..."
rsync -avz --progress "$VPS_HOST:$VPS_PATH/*.csv" "$LOCAL_PATH/"

# Show latest file
echo ""
echo "Latest export:"
ls -la "$LOCAL_PATH"/trades_latest.csv 2>/dev/null || echo "No exports yet"

# Show trade count
if [ -f "$LOCAL_PATH/trades_latest.csv" ]; then
    LINES=$(wc -l < "$LOCAL_PATH/trades_latest.csv")
    echo "Total trades: $((LINES - 1))"
fi

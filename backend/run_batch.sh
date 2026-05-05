#!/bin/bash
# Daily batch runner for AeroCast time series forecasting
# Add to cron with: crontab -e
# Run at 11 PM daily: 0 23 * * * /home/<user>/aerocast/backend/run_batch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/logs/batch_$(date +\%Y-\%m-\%d).log"
mkdir -p "$SCRIPT_DIR/logs"

echo "=== Batch run started: $(date) ===" >> "$LOG_FILE"

source venv/bin/activate

python -m app.features.time_series_forecasting.batch_runner >> "$LOG_FILE" 2>&1

echo "=== Batch run finished: $(date) ===" >> "$LOG_FILE"

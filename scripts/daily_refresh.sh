#!/bin/bash
# Daily refresh script for FinanGPT
# Phase 7: Unified Workflow & Automation
#
# This script performs automated daily data refresh:
# 1. Activates virtual environment
# 2. Runs incremental data refresh
# 3. Transforms data to DuckDB
# 4. Generates status report
# 5. Optional: Sends email notification on failure
#
# Usage:
#   ./scripts/daily_refresh.sh
#
# Cron example (weekdays at 6 PM):
#   0 18 * * 1-5 /path/to/FinanGPT/scripts/daily_refresh.sh >> /path/to/FinanGPT/logs/cron.log 2>&1

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
TICKERS_FILE="$PROJECT_DIR/tickers.csv"  # Default tickers file
EMAIL_ON_FAILURE=""  # Set to email address to enable notifications

# Timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$TIMESTAMP]${NC} $1"
}

log_error() {
    echo -e "${RED}[$TIMESTAMP] ERROR:${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[$TIMESTAMP] WARNING:${NC} $1"
}

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"

    # Send email notification if configured
    if [[ -n "$EMAIL_ON_FAILURE" ]]; then
        echo "FinanGPT daily refresh failed at line $line_number" | \
            mail -s "FinanGPT Refresh Failed" "$EMAIL_ON_FAILURE" 2>/dev/null || true
    fi

    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Main execution
log "Starting FinanGPT daily refresh..."

# Change to project directory
cd "$PROJECT_DIR"
log "Working directory: $PROJECT_DIR"

# Activate virtual environment
if [[ -d "$VENV_DIR" ]]; then
    log "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    log_warning "Virtual environment not found at $VENV_DIR"
    log_warning "Using system Python"
fi

# Create logs directory if needed
mkdir -p "$LOG_DIR"

# Check if tickers file exists
if [[ ! -f "$TICKERS_FILE" ]]; then
    log_warning "Tickers file not found: $TICKERS_FILE"
    log_warning "Using all previously ingested tickers"
    TICKERS_ARG=""
else
    TICKERS_ARG="--tickers-file $TICKERS_FILE"
    log "Using tickers from: $TICKERS_FILE"
fi

# Step 1: Run incremental refresh (only update stale data)
log "Step 1: Running incremental data refresh..."
python finangpt.py ingest --refresh $TICKERS_ARG

# Step 2: Transform data to DuckDB
log "Step 2: Transforming data to DuckDB..."
python finangpt.py transform

# Step 3: Generate status report
log "Step 3: Generating status report..."
python finangpt.py status --json > "$LOG_DIR/status_$(date '+%Y%m%d').json"

# Success
log "Daily refresh completed successfully!"

# Optional: Send success notification
if [[ -n "$EMAIL_ON_FAILURE" ]]; then
    # Only send email on failure, not success (to reduce noise)
    true
fi

exit 0

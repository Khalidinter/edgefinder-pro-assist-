#!/bin/bash
# Daily NBA prediction + resolution runner. Invoked by launchd
# (com.edgefinder.daily.plist) from the user's Mac, replacing the GHA cron
# which was killed 2026-04-25 by stats.nba.com WAF rate-limits on GitHub
# Actions runner IPs.
#
# Order matters:
#   1. Resolve yesterday's bets first (writes actuals + pnl)
#   2. Generate today's predictions (assists, rebounds)
#
# Logs roll daily into ops/logs/.

set -u  # don't die on individual command failure — keep going so partial
        # success still lands data; rely on exit codes per step in the log.

PROJECT_DIR="/Users/khalidhassan/Projects/edgefinder-assist-pro"
LOG_DIR="$PROJECT_DIR/ops/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).log"

cd "$PROJECT_DIR" || exit 1

# Load .env so SUPABASE_URL / SUPABASE_KEY / SUPABASE_SERVICE_ROLE_KEY /
# ODDS_API_KEY are present for the python scripts.
if [ -f .env ]; then
    set -o allexport
    # shellcheck disable=SC1091
    . .env
    set +o allexport
fi

PY=/usr/local/bin/python3

{
    echo "=========================================="
    echo "Daily run starting at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "=========================================="

    echo
    echo "[1/4] Resolve assists"
    "$PY" scripts/paper_trade.py --resolve
    echo "  → exit $?"

    echo
    echo "[2/4] Resolve rebounds"
    "$PY" scripts/rebound_paper_trade.py --resolve
    echo "  → exit $?"

    echo
    echo "[3/4] Predict assists"
    "$PY" scripts/paper_trade.py --threshold 0.60
    echo "  → exit $?"

    echo
    echo "[4/4] Predict rebounds"
    "$PY" scripts/rebound_paper_trade.py
    echo "  → exit $?"

    echo
    echo "Daily run complete at $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
} >> "$LOG_FILE" 2>&1

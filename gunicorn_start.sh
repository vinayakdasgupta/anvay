#!/bin/bash
# =============================================================================
# gunicorn_start.sh — anvay production startup
#
# Usage:
#   bash gunicorn_start.sh          # run directly (testing)
#   (wire into systemd for actual deployment — see anvay.service)
#
# Requirements:
#   - ANVAY_SECRET_KEY must be set in the environment (or .env loaded before)
#   - A virtualenv must exist at $APP_DIR/venv
#   - The logs/ directory will be created automatically
# =============================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$APP_DIR/venv"
LOG_DIR="$APP_DIR/logs"

# ── Bind address ──────────────────────────────────────────────────────────────
# Nginx proxies from port 443 to this. Never expose 5001 directly to the internet.
BIND="127.0.0.1:5001"

# ── Worker count ──────────────────────────────────────────────────────────────
# Keep this LOW. LDA training is CPU and memory heavy.
# 2 workers = 2 concurrent jobs. Raise only if you have ≥ 8 GB RAM per worker.
WORKERS=2

# ── Worker timeout (seconds) ──────────────────────────────────────────────────
# Gunicorn kills any worker that does not respond within this window.
# 300 s (5 min) is enough for large corpora on a decent CPU.
# Raise to 600 if users regularly time out on very large jobs.
TIMEOUT=300

# ── Log level ─────────────────────────────────────────────────────────────────
# "warning" in production keeps logs clean. Use "info" or "debug" when diagnosing.
LOG_LEVEL="warning"

# =============================================================================

mkdir -p "$LOG_DIR"

# Load .env if present (contains ANVAY_SECRET_KEY and other secrets).
# Never commit .env to git.
if [ -f "$APP_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$APP_DIR/.env"
    set +a
fi

# Activate virtualenv
# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "Starting anvay on $BIND with $WORKERS workers (timeout ${TIMEOUT}s)"

exec gunicorn wsgi:app \
    --bind          "$BIND"                         \
    --workers       "$WORKERS"                      \
    --worker-class  sync                            \
    --timeout       "$TIMEOUT"                      \
    --log-level     "$LOG_LEVEL"                    \
    --access-logfile  "$LOG_DIR/access.log"         \
    --error-logfile   "$LOG_DIR/gunicorn_error.log" \
    --capture-output                                \
    --chdir         "$APP_DIR"

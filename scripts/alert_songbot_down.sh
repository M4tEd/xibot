#!/usr/bin/env bash
set -euo pipefail

WEBHOOK_URL="$(cat "$HOME/.config/songbot/webhook.url")"

MESSAGE="SongBot is DOWN! The service failed on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S %Z')."

curl -fsS -H "Content-Type: application/json" \
  -d "{\"content\": \"$MESSAGE\"}" \
  "$WEBHOOK_URL"
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE=/etc/systemd/system/nano-speaker.service

if [[ $# -ne 0 && "${1:-}" != "gpio" ]]; then
  echo "Usage: $0 [gpio]" >&2
  exit 2
fi

sed -e "s|@USER@|$USER|g" -e "s|@PROJECT_ROOT@|$ROOT|g" \
  "$ROOT/systemd/nano-speaker.service.template" | sudo tee "$SERVICE" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable nano-speaker.service
sudo systemctl restart nano-speaker.service
sudo systemctl --no-pager status nano-speaker.service

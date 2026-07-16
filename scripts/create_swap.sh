#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-4G}"
if swapon --show=NAME --noheadings | grep -qx '/swapfile'; then
  echo "/swapfile is already active."
  exit 0
fi
if [[ ! -f /swapfile ]]; then
  sudo fallocate -l "$SIZE" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
fi
sudo swapon /swapfile
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
free -h

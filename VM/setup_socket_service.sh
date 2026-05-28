#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
SERVICE_PATH="/etc/systemd/system/2026project-socket.service"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python virtual environment not found. Run ./VM/setup_ubuntu.sh first."
  exit 1
fi

echo "[1/3] Writing systemd service"
sudo tee "${SERVICE_PATH}" >/dev/null <<SERVICE
[Unit]
Description=2026 Project mRNA Stability Prediction Socket
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PYTHON_BIN} ${PROJECT_ROOT}/Server/prediction_socket_server.py --host 127.0.0.1 --port 16888
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

echo "[2/3] Enabling service"
sudo systemctl daemon-reload
sudo systemctl enable 2026project-socket.service

echo "[3/3] Starting service"
sudo systemctl restart 2026project-socket.service
sudo systemctl --no-pager --full status 2026project-socket.service

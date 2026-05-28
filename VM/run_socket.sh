#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" Server/prediction_socket_server.py --host 127.0.0.1 --port 16888

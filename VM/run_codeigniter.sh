#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEIGNITER_DIR="${PROJECT_ROOT}/Server/CodeIgniterApp"

if [ ! -d "${CODEIGNITER_DIR}" ]; then
  echo "CodeIgniter app not found. Run ./VM/setup_ubuntu.sh first."
  exit 1
fi

if [ ! -f /etc/apache2/sites-available/2026project.conf ]; then
  echo "Apache site is not configured. Run ./VM/setup_apache.sh first."
  exit 1
fi

sudo apache2ctl configtest
sudo systemctl restart apache2
sudo systemctl --no-pager --full status apache2

echo "CodeIgniter is served by Apache at http://localhost:17888/2026Project/"

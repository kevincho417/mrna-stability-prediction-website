#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEIGNITER_DIR="${PROJECT_ROOT}/Server/CodeIgniterApp"
PUBLIC_DIR="${CODEIGNITER_DIR}/public"
CONF_PATH="/etc/apache2/sites-available/2026project.conf"

if [ ! -d "${PUBLIC_DIR}" ]; then
  echo "CodeIgniter public directory not found. Run ./VM/setup_ubuntu.sh first."
  exit 1
fi

echo "[1/5] Enabling Apache modules"
sudo a2enmod rewrite headers >/dev/null

echo "[2/5] Ensuring Apache listens on port 17888"
if ! grep -qE "^[[:space:]]*Listen[[:space:]]+17888\b" /etc/apache2/ports.conf; then
  echo "Listen 17888" | sudo tee -a /etc/apache2/ports.conf >/dev/null
fi

echo "[3/5] Writing Apache VirtualHost"
sudo tee "${CONF_PATH}" >/dev/null <<APACHECONF
<VirtualHost *:17888>
    ServerName localhost
    ServerAdmin webmaster@localhost

    DocumentRoot "${PUBLIC_DIR}"

    <Directory "${PUBLIC_DIR}">
        Options FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    ErrorLog \${APACHE_LOG_DIR}/2026project_error.log
    CustomLog \${APACHE_LOG_DIR}/2026project_access.log combined
</VirtualHost>
APACHECONF

echo "[4/5] Granting Apache read/traverse access to the project folder"
if command -v setfacl >/dev/null 2>&1; then
  sudo setfacl -m u:www-data:rx "$(dirname "${PROJECT_ROOT}")" || true
  sudo setfacl -R -m u:www-data:rx "${PROJECT_ROOT}"
  sudo setfacl -R -m u:www-data:rwx "${CODEIGNITER_DIR}/writable"
else
  sudo chmod -R o+rX "${PROJECT_ROOT}"
  sudo chmod -R o+rwX "${CODEIGNITER_DIR}/writable"
fi

echo "[5/5] Enabling site and restarting Apache"
sudo a2ensite 2026project.conf >/dev/null
sudo apache2ctl configtest
sudo systemctl restart apache2

echo "Apache is serving CodeIgniter at http://localhost:17888/2026Project/"

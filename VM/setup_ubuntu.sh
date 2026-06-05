#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEIGNITER_DIR="${PROJECT_ROOT}/Server/CodeIgniterApp"
OVERLAY_DIR="${PROJECT_ROOT}/Server/codeigniter_overlay"

echo "[1/6] Installing Ubuntu packages"
sudo apt-get update
sudo apt-get install -y \
  acl \
  apache2 \
  curl \
  libapache2-mod-php \
  sqlite3 \
  unzip \
  python3 \
  python3-pip \
  python3-venv \
  php \
  php-cli \
  php-curl \
  php-intl \
  php-mbstring \
  php-sqlite3 \
  php-xml \
  php-zip \
  composer

echo "[2/6] Creating Python virtual environment"
python3 -m venv "${PROJECT_ROOT}/.venv"
"${PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${PROJECT_ROOT}/.venv/bin/python" -m pip install --no-cache-dir \
  flask \
  matplotlib \
  numpy \
  pandas \
  scikit-learn
"${PROJECT_ROOT}/.venv/bin/python" -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch

echo "[3/6] Creating CodeIgniter 4 app"
if [ ! -d "${CODEIGNITER_DIR}" ]; then
  composer create-project codeigniter4/appstarter "${CODEIGNITER_DIR}"
fi

echo "[4/6] Applying project MVC files"
mkdir -p \
  "${CODEIGNITER_DIR}/app/Controllers" \
  "${CODEIGNITER_DIR}/app/Models" \
  "${CODEIGNITER_DIR}/app/Views/layout" \
  "${CODEIGNITER_DIR}/app/Views/project" \
  "${CODEIGNITER_DIR}/public/assets"

cp "${OVERLAY_DIR}/app/Controllers/Project.php" "${CODEIGNITER_DIR}/app/Controllers/Project.php"
cp "${OVERLAY_DIR}/app/Models/PredictionModel.php" "${CODEIGNITER_DIR}/app/Models/PredictionModel.php"
cp "${OVERLAY_DIR}/app/Views/layout/header.php" "${CODEIGNITER_DIR}/app/Views/layout/header.php"
cp "${OVERLAY_DIR}/app/Views/layout/footer.php" "${CODEIGNITER_DIR}/app/Views/layout/footer.php"
cp "${OVERLAY_DIR}/app/Views/project/index.php" "${CODEIGNITER_DIR}/app/Views/project/index.php"
cp "${PROJECT_ROOT}/Server/static/styles.css" "${CODEIGNITER_DIR}/public/assets/styles.css"

ROUTES_FILE="${CODEIGNITER_DIR}/app/Config/Routes.php"
if ! grep -q "Project::index" "${ROUTES_FILE}"; then
  {
    echo ""
    echo "// 2026 DL Big Data project routes"
    cat "${OVERLAY_DIR}/app/Config/Routes.php.append"
  } >> "${ROUTES_FILE}"
fi

echo "[5/6] Setting CodeIgniter base URL and environment"
ENV_FILE="${CODEIGNITER_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
  cp "${CODEIGNITER_DIR}/env" "${ENV_FILE}"
fi
sed -i "s/# CI_ENVIRONMENT = production/CI_ENVIRONMENT = development/" "${ENV_FILE}"
if grep -q "^app.baseURL" "${ENV_FILE}"; then
  sed -i "s|^app.baseURL.*|app.baseURL = 'http://localhost:17888/'|" "${ENV_FILE}"
else
  echo "app.baseURL = 'http://localhost:17888/'" >> "${ENV_FILE}"
fi

echo "[6/6] Done"
echo "Configure Apache server: ./VM/setup_apache.sh"
echo "Install socket systemd service: ./VM/setup_socket_service.sh"
echo "Run local socket manually: ./VM/run_socket.sh"

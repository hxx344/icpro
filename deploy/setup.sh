#!/usr/bin/env bash
# ==============================================================
# Iron Condor Trader — Linux VPS 部署脚本
# ==============================================================
# 用法: bash deploy/setup.sh
# 前提: Ubuntu 20.04+ / Debian 11+, Python 3.9+
# ==============================================================
set -euo pipefail

APP_DIR="/opt/project_Options"
VENV_DIR="$APP_DIR/.venv"
SERVICE_USER="trader"

echo "========================================="
echo "  Iron Condor Trader — 部署脚本"
echo "========================================="

# --- 1. 系统依赖 ---
echo "[1/6] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git sqlite3

# --- 2. 创建运行用户 ---
echo "[2/6] 创建服务用户 '$SERVICE_USER'..."
if ! id "$SERVICE_USER" &>/dev/null; then
    sudo useradd -r -m -s /bin/bash "$SERVICE_USER"
    echo "  用户 '$SERVICE_USER' 已创建"
else
    echo "  用户 '$SERVICE_USER' 已存在，跳过"
fi

# --- 3. 部署代码 ---
echo "[3/6] 部署代码到 $APP_DIR..."
sudo mkdir -p "$APP_DIR"
sudo rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    "$(dirname "$(dirname "$(realpath "$0")")")/" "$APP_DIR/"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

# --- 4. Python 虚拟环境 ---
echo "[4/6] 创建 Python 虚拟环境并安装依赖..."
sudo -u "$SERVICE_USER" bash -c "
    cd $APP_DIR
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -e '.[trader]' -q
"

# --- 5. 创建数据目录 ---
echo "[5/6] 创建数据和日志目录..."
sudo -u "$SERVICE_USER" mkdir -p "$APP_DIR/data" "$APP_DIR/logs"

# --- 6. 安装 systemd 服务 ---
echo "[6/6] 安装 systemd 服务..."
sudo cp "$APP_DIR/deploy/trader-engine.service" /etc/systemd/system/
sudo cp "$APP_DIR/deploy/trader-dashboard.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trader-engine trader-dashboard

echo ""
echo "========================================="
echo "  部署完成!"
echo "========================================="
echo ""
echo "后续步骤:"
echo "  1. 配置 Binance API 密钥（环境变量或 .env）:"
echo "     sudo -u $SERVICE_USER nano $APP_DIR/configs/trader/weekend_vol_btc.yaml"
echo ""
echo "  2. 启动服务:"
echo "     sudo systemctl start trader-engine"
echo "     sudo systemctl start trader-dashboard"
echo ""
echo "  3. 查看日志:"
echo "     journalctl -u trader-engine -f"
echo "     journalctl -u trader-dashboard -f"
echo ""
echo "  4. 访问面板:"
echo "     http://<VPS_IP>:8501"
echo ""
echo "  5. 防火墙 (如需):"
echo "     sudo ufw allow 8501/tcp"
echo ""

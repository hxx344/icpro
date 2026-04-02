#!/usr/bin/env bash
# ==============================================================
# Iron Condor Trader — Linux VPS 部署脚本
# ==============================================================
# 用法:
#   bash deploy/setup.sh
#   BINANCE_API_KEY=... BINANCE_API_SECRET=... DASHBOARD_READONLY_USER=... \
#   DASHBOARD_READONLY_PASS=... DASHBOARD_TRADER_USER=... DASHBOARD_TRADER_PASS=... \
#   bash deploy/setup.sh
#   bash deploy/setup.sh --api-key xxx --api-secret yyy --dashboard-readonly-user ro \
#     --dashboard-readonly-pass ro_pass --dashboard-trader-user trader \
#     --dashboard-trader-pass trader_pass
#   # 仅在需要独立行情监控面板时再加 --with-monitor 及 monitor 凭据
# 前提: Ubuntu 22.04+ / Debian 12+，或已预装 Python 3.10+
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

APP_DIR="${APP_DIR:-$SOURCE_DIR}"
VENV_DIR="$APP_DIR/.venv"
SERVICE_USER="${SERVICE_USER:-$(id -un)}"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn "$SERVICE_USER" 2>/dev/null || id -gn)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRADER_CONFIG_PATH="${TRADER_CONFIG_PATH:-configs/trader/weekend_vol_btc.yaml}"
DASHBOARD_PUBLIC="${DASHBOARD_PUBLIC:-false}"
START_SERVICES="${START_SERVICES:-1}"
WITH_MONITOR="${WITH_MONITOR:-0}"
BINANCE_API_KEY_VALUE="${BINANCE_API_KEY:-}"
BINANCE_API_SECRET_VALUE="${BINANCE_API_SECRET:-}"
CDD_API_TOKEN_VALUE="${CDD_API_TOKEN:-}"
DASHBOARD_READONLY_USER_VALUE="${DASHBOARD_READONLY_USER:-readonly}"
DASHBOARD_READONLY_PASS_VALUE="${DASHBOARD_READONLY_PASS:-}"
DASHBOARD_TRADER_USER_VALUE="${DASHBOARD_TRADER_USER:-trader}"
DASHBOARD_TRADER_PASS_VALUE="${DASHBOARD_TRADER_PASS:-}"
MONITOR_USER_VALUE="${MONITOR_USER:-monitor}"
MONITOR_PASS_VALUE="${MONITOR_PASS:-}"

usage() {
    cat <<'EOF'
Usage: bash deploy/setup.sh [options]

Options:
  --app-dir PATH
  --python-bin PATH
  --config PATH
    --with-monitor
  --api-key VALUE
  --api-secret VALUE
  --cdd-token VALUE
  --dashboard-public true|false
  --dashboard-readonly-user VALUE
  --dashboard-readonly-pass VALUE
  --dashboard-trader-user VALUE
  --dashboard-trader-pass VALUE
  --monitor-user VALUE
  --monitor-pass VALUE
  --no-start
  -h, --help
EOF
}

prompt_value() {
    local var_name="$1"
    local prompt_text="$2"
    local default_value="${3:-}"
    local secret="${4:-0}"
    local current_value="${!var_name:-}"
    local effective_default="$default_value"
    local input=""

    if [[ -n "$current_value" ]]; then
        effective_default="$current_value"
    fi

    if [[ "$secret" == "1" ]]; then
        if [[ -n "$effective_default" ]]; then
            printf "%s [%s]: " "$prompt_text" "已设置"
        else
            printf "%s: " "$prompt_text"
        fi
        read -r -s input
        printf "\n"
    else
        if [[ -n "$effective_default" ]]; then
            printf "%s [%s]: " "$prompt_text" "$effective_default"
        else
            printf "%s: " "$prompt_text"
        fi
        read -r input
    fi

    if [[ -z "$input" ]]; then
        input="$effective_default"
    fi

    printf -v "$var_name" '%s' "$input"
}

mask_value() {
    local value="$1"
    if [[ -z "$value" ]]; then
        printf "<empty>"
    elif [[ ${#value} -le 4 ]]; then
        printf "****"
    else
        printf "%s****%s" "${value:0:2}" "${value: -2}"
    fi
}

prompt_yes_no() {
    local var_name="$1"
    local prompt_text="$2"
    local default_value="${3:-N}"
    local current_value="${!var_name:-}"
    local effective_default="$default_value"
    local input=""

    if [[ -n "$current_value" ]]; then
        case "$(printf '%s' "$current_value" | tr '[:lower:]' '[:upper:]')" in
            Y|YES|1|TRUE)
                effective_default="Y"
                ;;
            N|NO|0|FALSE)
                effective_default="N"
                ;;
        esac
    fi

    printf "%s [%s]: " "$prompt_text" "$effective_default"
    read -r input
    input="${input:-$effective_default}"
    input="$(printf '%s' "$input" | tr '[:lower:]' '[:upper:]')"

    case "$input" in
        Y|YES|1|TRUE)
            printf -v "$var_name" '1'
            ;;
        N|NO|0|FALSE)
            printf -v "$var_name" '0'
            ;;
        *)
            echo "输入无效，请输入 Y 或 N。"
            prompt_yes_no "$var_name" "$prompt_text" "$default_value"
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app-dir)
            APP_DIR="$2"; VENV_DIR="$APP_DIR/.venv"; shift 2 ;;
        --python-bin)
            PYTHON_BIN="$2"; shift 2 ;;
        --config)
            TRADER_CONFIG_PATH="$2"; shift 2 ;;
        --with-monitor)
            WITH_MONITOR="1"; shift ;;
        --api-key)
            BINANCE_API_KEY_VALUE="$2"; shift 2 ;;
        --api-secret)
            BINANCE_API_SECRET_VALUE="$2"; shift 2 ;;
        --cdd-token)
            CDD_API_TOKEN_VALUE="$2"; shift 2 ;;
        --dashboard-public)
            DASHBOARD_PUBLIC="$2"; shift 2 ;;
        --dashboard-readonly-user)
            DASHBOARD_READONLY_USER_VALUE="$2"; shift 2 ;;
        --dashboard-readonly-pass)
            DASHBOARD_READONLY_PASS_VALUE="$2"; shift 2 ;;
        --dashboard-trader-user)
            DASHBOARD_TRADER_USER_VALUE="$2"; shift 2 ;;
        --dashboard-trader-pass)
            DASHBOARD_TRADER_PASS_VALUE="$2"; shift 2 ;;
        --monitor-user)
            MONITOR_USER_VALUE="$2"; shift 2 ;;
        --monitor-pass)
            MONITOR_PASS_VALUE="$2"; shift 2 ;;
        --no-start)
            START_SERVICES="0"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "错误: 未知参数 $1" >&2
            usage
            exit 1 ;;
    esac
done

escape_env_value() {
    local value="$1"
    value=${value//\'/\'\"\'\"\'}
    printf "'%s'" "$value"
}

write_env_line() {
    local key="$1"
    local value="$2"
    printf "%s=%s\n" "$key" "$(escape_env_value "$value")"
}

escape_sed_replacement() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//&/\\&}
    printf '%s' "$value"
}

install_rendered_service() {
    local template_path="$1"
    local target_path="$2"
    local app_dir_escaped
    local service_user_escaped
    local service_group_escaped
    local tmp_service

    app_dir_escaped="$(escape_sed_replacement "$APP_DIR")"
    service_user_escaped="$(escape_sed_replacement "$SERVICE_USER")"
    service_group_escaped="$(escape_sed_replacement "$SERVICE_GROUP")"
    tmp_service="$(mktemp)"

    sed \
        -e "s|/opt/project_Options|${app_dir_escaped}|g" \
        -e "s|^User=.*$|User=${service_user_escaped}|" \
        -e "s|^Group=.*$|Group=${service_group_escaped}|" \
        "$template_path" > "$tmp_service"

    sudo install -o root -g root -m 644 "$tmp_service" "$target_path"
    rm -f "$tmp_service"
}

validate_required_inputs() {
    prompt_value APP_DIR "部署目录" "$APP_DIR"
    VENV_DIR="$APP_DIR/.venv"
    prompt_value PYTHON_BIN "Python 解释器" "$PYTHON_BIN"
    prompt_value TRADER_CONFIG_PATH "策略配置文件路径" "$TRADER_CONFIG_PATH"
    prompt_yes_no WITH_MONITOR "是否部署独立 Monitor 面板？" "N"
    prompt_yes_no START_SERVICES "部署完成后是否自动启动服务？" "Y"
    prompt_yes_no DASHBOARD_PUBLIC "是否公开暴露 Dashboard 到 0.0.0.0？" "N"

    if [[ "$DASHBOARD_PUBLIC" == "1" ]]; then
        DASHBOARD_PUBLIC="true"
    else
        DASHBOARD_PUBLIC="false"
    fi

    prompt_value BINANCE_API_KEY_VALUE "请输入 Binance API Key"
    prompt_value BINANCE_API_SECRET_VALUE "请输入 Binance API Secret" "" 1
    prompt_value DASHBOARD_READONLY_USER_VALUE "Dashboard 只读用户名" "$DASHBOARD_READONLY_USER_VALUE"
    prompt_value DASHBOARD_READONLY_PASS_VALUE "Dashboard 只读密码" "" 1
    prompt_value DASHBOARD_TRADER_USER_VALUE "Dashboard 交易用户名" "$DASHBOARD_TRADER_USER_VALUE"
    prompt_value DASHBOARD_TRADER_PASS_VALUE "Dashboard 交易密码" "" 1
    prompt_value CDD_API_TOKEN_VALUE "可选：CDD API Token（可留空）" "$CDD_API_TOKEN_VALUE" 1

    if [[ "$WITH_MONITOR" == "1" ]]; then
        prompt_value MONITOR_USER_VALUE "Monitor 用户名" "$MONITOR_USER_VALUE"
        prompt_value MONITOR_PASS_VALUE "Monitor 密码" "" 1
    fi

    local missing=()
    [[ -n "$BINANCE_API_KEY_VALUE" ]] || missing+=("Binance API Key")
    [[ -n "$BINANCE_API_SECRET_VALUE" ]] || missing+=("Binance API Secret")
    [[ -n "$DASHBOARD_READONLY_PASS_VALUE" ]] || missing+=("Dashboard 只读密码")
    [[ -n "$DASHBOARD_TRADER_PASS_VALUE" ]] || missing+=("Dashboard 交易密码")
    if [[ "$WITH_MONITOR" == "1" ]]; then
        [[ -n "$MONITOR_PASS_VALUE" ]] || missing+=("Monitor 密码")
    fi

    if (( ${#missing[@]} > 0 )); then
        echo "错误: 以下必填项不能为空:"
        for item in "${missing[@]}"; do
            echo "  - $item"
        done
        exit 1
    fi
}

print_configuration_summary() {
    echo ""
    echo "================ 配置摘要 ================"
    echo "部署目录              : $APP_DIR"
    echo "运行用户              : $SERVICE_USER"
    echo "Python 解释器         : $PYTHON_BIN"
    echo "策略配置              : $TRADER_CONFIG_PATH"
    echo "部署 Monitor          : $WITH_MONITOR"
    echo "自动启动服务          : $START_SERVICES"
    echo "Dashboard 对外暴露    : $DASHBOARD_PUBLIC"
    echo "Binance API Key       : $(mask_value "$BINANCE_API_KEY_VALUE")"
    echo "Binance API Secret    : $(mask_value "$BINANCE_API_SECRET_VALUE")"
    echo "CDD API Token         : $(mask_value "$CDD_API_TOKEN_VALUE")"
    echo "Dashboard 只读用户名  : $DASHBOARD_READONLY_USER_VALUE"
    echo "Dashboard 只读密码    : $(mask_value "$DASHBOARD_READONLY_PASS_VALUE")"
    echo "Dashboard 交易用户名  : $DASHBOARD_TRADER_USER_VALUE"
    echo "Dashboard 交易密码    : $(mask_value "$DASHBOARD_TRADER_PASS_VALUE")"
    if [[ "$WITH_MONITOR" == "1" ]]; then
        echo "Monitor 用户名        : $MONITOR_USER_VALUE"
        echo "Monitor 密码          : $(mask_value "$MONITOR_PASS_VALUE")"
    else
        echo "Monitor               : 不部署"
    fi
    echo "将写入文件            :"
    echo "  - $APP_DIR/deploy/trader-engine.env"
    echo "  - $APP_DIR/deploy/trader-dashboard.env"
    if [[ "$WITH_MONITOR" == "1" ]]; then
        echo "  - $APP_DIR/deploy/monitor-dashboard.env"
    fi
    echo "将启用服务            : trader-engine trader-dashboard$( [[ "$WITH_MONITOR" == "1" ]] && printf ' monitor-dashboard' )"
    if [[ "$START_SERVICES" == "1" ]]; then
        echo "将自动启动服务        : 是"
    else
        echo "将自动启动服务        : 否"
    fi
    echo "========================================="
    echo ""
}

confirm_configuration() {
    local confirmed=""
    prompt_yes_no confirmed "确认以上配置并开始部署？" "Y"
    if [[ "$confirmed" != "1" ]]; then
        echo "已取消部署，未写入任何 env 文件，也未启动任何服务。"
        exit 0
    fi
}

validate_required_inputs
print_configuration_summary
confirm_configuration

echo "========================================="
echo "  Iron Condor Trader — 部署脚本"
echo "========================================="
echo "  APP_DIR=$APP_DIR"
echo "  TRADER_CONFIG_PATH=$TRADER_CONFIG_PATH"
echo "  WITH_MONITOR=$WITH_MONITOR"

# --- 1. 系统依赖 ---
echo "[1/6] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git sqlite3 rsync

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误: 未找到 Python 解释器: $PYTHON_BIN"
    echo "请安装 Python 3.10+，或以 PYTHON_BIN 指定解释器路径。"
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "错误: 检测到 $PYTHON_BIN 版本低于 3.10。"
    echo "请升级系统 Python，或使用 PYTHON_BIN=/path/to/python3.10 bash deploy/setup.sh"
    exit 1
fi

# --- 2. 创建运行用户 ---
echo "[2/6] 创建服务用户 '$SERVICE_USER'..."
if ! id "$SERVICE_USER" &>/dev/null; then
    sudo useradd -r -m -s /bin/bash "$SERVICE_USER"
    echo "  用户 '$SERVICE_USER' 已创建"
else
    echo "  用户 '$SERVICE_USER' 已存在，跳过"
fi

# --- 3. 部署代码 ---
SOURCE_DIR_REAL="$(realpath "$SOURCE_DIR")"
APP_DIR_REAL="$(realpath -m "$APP_DIR")"
sudo mkdir -p "$APP_DIR"
if [[ "$SOURCE_DIR_REAL" == "$APP_DIR_REAL" ]]; then
    echo "  检测到原地部署模式，直接使用当前仓库目录"
else
    sudo rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
        "$SOURCE_DIR/" "$APP_DIR/"
fi
sudo chown -R "$SERVICE_USER:$SERVICE_GROUP" "$APP_DIR"

# --- 4. Python 虚拟环境 ---
echo "[4/6] 创建 Python 虚拟环境并安装依赖..."
sudo -u "$SERVICE_USER" bash -c "
    cd $APP_DIR
    $PYTHON_BIN -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    if [ -f requirements.lock.txt ]; then
        .venv/bin/pip install -r requirements.lock.txt -q
    else
        .venv/bin/pip install -e '.[trader]' -q
    fi
    .venv/bin/pip install -e '.[trader]' -q
"

# --- 5. 创建数据目录 ---
echo "[5/6] 创建数据和日志目录..."
sudo -u "$SERVICE_USER" mkdir -p "$APP_DIR/data" "$APP_DIR/logs"

# --- 6. 安装 systemd 服务 ---
echo "[6/6] 安装 systemd 服务..."
install_rendered_service "$APP_DIR/deploy/trader-engine.service" /etc/systemd/system/trader-engine.service
install_rendered_service "$APP_DIR/deploy/trader-dashboard.service" /etc/systemd/system/trader-dashboard.service
if [[ "$WITH_MONITOR" == "1" ]]; then
    install_rendered_service "$APP_DIR/deploy/monitor-dashboard.service" /etc/systemd/system/monitor-dashboard.service
fi

tmp_engine_env=$(mktemp)
{
    echo "# Generated by deploy/setup.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    write_env_line "TRADER_CONFIG_PATH" "$TRADER_CONFIG_PATH"
    write_env_line "BINANCE_API_KEY" "$BINANCE_API_KEY_VALUE"
    write_env_line "BINANCE_API_SECRET" "$BINANCE_API_SECRET_VALUE"
    if [[ -n "$CDD_API_TOKEN_VALUE" ]]; then
        write_env_line "CDD_API_TOKEN" "$CDD_API_TOKEN_VALUE"
    fi
} > "$tmp_engine_env"
sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 600 "$tmp_engine_env" "$APP_DIR/deploy/trader-engine.env"
rm -f "$tmp_engine_env"

tmp_dashboard_env=$(mktemp)
{
    echo "# Generated by deploy/setup.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    write_env_line "TRADER_CONFIG_PATH" "$TRADER_CONFIG_PATH"
    write_env_line "DASHBOARD_PUBLIC" "$DASHBOARD_PUBLIC"
    write_env_line "DASHBOARD_READONLY_USER" "$DASHBOARD_READONLY_USER_VALUE"
    write_env_line "DASHBOARD_READONLY_PASS" "$DASHBOARD_READONLY_PASS_VALUE"
    write_env_line "DASHBOARD_TRADER_USER" "$DASHBOARD_TRADER_USER_VALUE"
    write_env_line "DASHBOARD_TRADER_PASS" "$DASHBOARD_TRADER_PASS_VALUE"
} > "$tmp_dashboard_env"
sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 600 "$tmp_dashboard_env" "$APP_DIR/deploy/trader-dashboard.env"
rm -f "$tmp_dashboard_env"

tmp_monitor_env=$(mktemp)
if [[ "$WITH_MONITOR" == "1" ]]; then
    {
        echo "# Generated by deploy/setup.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        write_env_line "MONITOR_USER" "$MONITOR_USER_VALUE"
        write_env_line "MONITOR_PASS" "$MONITOR_PASS_VALUE"
    } > "$tmp_monitor_env"
    sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 600 "$tmp_monitor_env" "$APP_DIR/deploy/monitor-dashboard.env"
fi
rm -f "$tmp_monitor_env"

sudo systemctl daemon-reload
sudo systemctl enable trader-engine trader-dashboard
if [[ "$WITH_MONITOR" == "1" ]]; then
    sudo systemctl enable monitor-dashboard
else
    sudo systemctl disable monitor-dashboard >/dev/null 2>&1 || true
fi

if [[ "$START_SERVICES" == "1" ]]; then
    sudo systemctl restart trader-engine trader-dashboard
    if [[ "$WITH_MONITOR" == "1" ]]; then
        sudo systemctl restart monitor-dashboard
    fi
fi

echo ""
echo "========================================="
echo "  部署完成!"
echo "========================================="
echo ""
echo "已写入环境文件:"
echo "  - $APP_DIR/deploy/trader-engine.env"
echo "  - $APP_DIR/deploy/trader-dashboard.env"
if [[ "$WITH_MONITOR" == "1" ]]; then
    echo "  - $APP_DIR/deploy/monitor-dashboard.env"
fi
echo ""
echo "服务状态命令:"
if [[ "$WITH_MONITOR" == "1" ]]; then
    echo "  sudo systemctl status trader-engine trader-dashboard monitor-dashboard"
else
    echo "  sudo systemctl status trader-engine trader-dashboard"
fi
echo ""
echo "查看日志:"
echo "     journalctl -u trader-engine -f"
echo "     journalctl -u trader-dashboard -f"
if [[ "$WITH_MONITOR" == "1" ]]; then
    echo "     journalctl -u monitor-dashboard -f"
fi
echo ""
echo "访问面板:"
echo "     http://<VPS_IP>:8501"
if [[ "$WITH_MONITOR" == "1" ]]; then
    echo "     http://<VPS_IP>:8502"
fi
echo "     当前策略配置: $TRADER_CONFIG_PATH"
if [[ "$START_SERVICES" == "1" ]]; then
    echo "     服务已自动启动。"
else
    if [[ "$WITH_MONITOR" == "1" ]]; then
        echo "     服务未自动启动，可手动执行: sudo systemctl start trader-engine trader-dashboard monitor-dashboard"
    else
        echo "     服务未自动启动，可手动执行: sudo systemctl start trader-engine trader-dashboard"
    fi
fi
echo ""
echo "防火墙 (如需):"
echo "     sudo ufw allow 8501/tcp"
if [[ "$WITH_MONITOR" == "1" ]]; then
    echo "     sudo ufw allow 8502/tcp"
fi
echo ""

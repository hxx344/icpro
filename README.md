# IC Pro

加密货币期权回测 + 自动交易系统。基于 Binance European Options API（USD 保证金）。

## 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/hxx344/icpro.git
cd icpro

# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -U pip setuptools wheel
pip install -e ".[dev,trader]"
```

### 2. 配置 API 密钥

```bash
# 设置环境变量
# Windows PowerShell
$env:BINANCE_API_KEY = "your_key"
$env:BINANCE_API_SECRET = "your_secret"

# Linux/macOS
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

也可以在 `.env` 文件中配置：

```bash
cp .env.example .env
# 编辑 .env 填写 BINANCE_API_KEY / BINANCE_API_SECRET
```

### 3. 启动交易

```bash
# 推荐：Dashboard + 引擎（Streamlit Web UI）
streamlit run trader/dashboard.py -- --config configs/trader/weekend_vol_btc.yaml

# 仅后端引擎（无 UI）
python -m trader.main run -c configs/trader/weekend_vol_btc.yaml
```

> **注意**：默认配置 `simulate_private: true`，引擎只模拟下单不实际成交，适合调试。正式交易前改为 `false`，并确认 `testnet: false`。

### Ubuntu VPS 命令清单（从 0 到启动）

```bash
# 1) 基础依赖
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 2) 克隆项目
git clone https://github.com/hxx344/icpro.git
cd icpro

# 3) 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 4) 安装项目依赖
pip install -U pip setuptools wheel
pip install -e ".[trader]"

# 5) 配置 API 密钥
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# 6) 启动 Dashboard + 引擎
streamlit run trader/dashboard.py -- --config configs/trader/weekend_vol_btc.yaml

# 7) 或使用 systemd 服务（参考 deploy/setup.sh）
sudo bash deploy/setup.sh
```

## 交易策略

### Weekend Vol（推荐）

周末波动率卖权策略 — 周五开仓 Iron Condor，周日到期结算。基于回测优化参数：

| 参数 | 值 | 说明 |
|------|----|------|
| target_delta | 0.40 | Call/Put 卖方 delta |
| wing_delta | 0.05 | 翼保护 delta |
| leverage | 3.0 | 杠杆倍数 |
| entry_day | friday | 开仓日（UTC） |
| entry_time_utc | 16:00 | 开仓时间 |
| settlement | Sunday 08:00 UTC | 最优结算日 |

回测表现（BTC, 3x, Sunday）：APR 113%, MaxDD 18.7%, Sharpe 3.10, Win Rate 81.7%

```bash
python -m trader.main run -c configs/trader/weekend_vol_btc.yaml
```

### 其他策略配置

| 配置文件 | 策略 |
|----------|------|
| `configs/trader/weekend_vol_btc.yaml` | Weekend Vol BTC Iron Condor（推荐） |
| `configs/trader/iron_condor_0dte.yaml` | 0DTE Iron Condor |
| `configs/trader/short_strangle_7dte.yaml` | 7DTE Short Strangle |

## 核心特性

- **交易所 Delta**：优先使用 Binance API 返回的 Greeks（delta / markIV），Black-76 仅作 fallback
- **限价追单引擎**：窗口内自适应漂移限价单，二次曲线从最优价渐移至市价，兜底市价成交
- **交易所持仓检查**：开仓前查询 Binance 实际持仓，防止与手动单冲突
- **仓位恢复**：引擎重启自动从 SQLite 恢复未平仓位
- **Streamlit Dashboard**：一键启动/停止，实时查看持仓、权益曲线、成交记录
- **USD 保证金**：Binance 欧式期权 USDT 保证金模式

## CLI 命令

```bash
# 启动交易引擎
python -m trader.main run -c configs/trader/weekend_vol_btc.yaml

# 查看当前持仓
python -m trader.main status -c configs/trader/weekend_vol_btc.yaml

# 查看成交记录
python -m trader.main trades -c configs/trader/weekend_vol_btc.yaml

# 查看权益快照
python -m trader.main equity -c configs/trader/weekend_vol_btc.yaml

# 查看统计
python -m trader.main stats -c configs/trader/weekend_vol_btc.yaml

# 平掉所有仓位
python -m trader.main close-all -c configs/trader/weekend_vol_btc.yaml
```

## 回测系统

支持 BTC/ETH 多策略回测，配置驱动：

```bash
options-bt run --config configs/backtest/iron_condor.yaml
```

| 策略 | 示例配置 |
|------|----------|
| Iron Condor 0DTE | `ic_0dte_8pct_direct.yaml` |
| Short Strangle | `short_strangle_0dte_12m.yaml` |
| Covered Call | `covered_call_eth_compound.yaml` |
| Short Put | `short_put_eth_compound.yaml` |
| Naked Call | `nc_eth_12m.yaml` |

功能：复利/非复利、周末跳过、IV 条件翼跳过、Binance 费率模型。

## 项目结构

```
src/options_backtest/    # 回测引擎（策略、撮合、定价、分析）
trader/                  # 自动交易系统
  ├── binance_client.py  #   Binance API 客户端 + Greeks 获取
  ├── limit_chaser.py    #   限价追单引擎
  ├── position_manager.py#   仓位管理
  ├── strategy.py        #   WeekendVol / IronCondor0DTE / Strangle 策略
  ├── engine.py          #   交易引擎（后台线程）
  ├── dashboard.py       #   Streamlit 管理面板
  ├── config.py          #   YAML 配置加载
  └── storage.py         #   SQLite 持久化
configs/                 # YAML 配置（回测 + 交易）
monitor/                 # 行情监控面板
tests/                   # 299 个单元测试
```

## 测试

```bash
pytest tests/ -v
```

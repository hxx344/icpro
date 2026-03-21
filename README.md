# IC Pro

加密货币期权回测 + Iron Condor 0DTE 自动交易系统。基于 Binance European Options API。

## 安装

```bash
pip install -e ".[dev]"
```

### Ubuntu 命令清单（从 0 到启动）

```bash
# 1) 基础依赖
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 2) 进入项目
cd ~/icpro

# 3) 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 4) 安装项目依赖
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"

# 5) 配置 API 环境变量
cp .env.example .env
# 编辑 .env 填写 BINANCE_API_KEY / BINANCE_API_SECRET

# 6) 启动（Dashboard + 引擎）
streamlit run trader/dashboard.py -- --config configs/trader/iron_condor_0dte.yaml

# 7) 仅后端运行（可选）
python -m trader.main run -c configs/trader/iron_condor_0dte.yaml
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

功能：复利/非复利、周末跳过、IV 条件翼跳过、多交易所费率对比（Deribit/OKX/Binance）。

## 自动交易系统

基于 Binance 欧式期权 API 的 Iron Condor 0DTE 自动交易，核心特性：

- **限价追单引擎**：30 分钟窗口内自适应漂移限价单，二次曲线从最优价渐移至市价，最后 60s 兜底市价成交
- **交易所持仓检查**：开仓前查询 Binance 实际持仓，防止与手动单冲突
- **仓位恢复**：引擎重启自动从 SQLite 恢复未平仓位
- **Streamlit Dashboard**：一键启动/停止，实时查看持仓、权益曲线、成交记录

### 启动

```bash
# Dashboard + 引擎（推荐）
streamlit run trader/dashboard.py -- --config configs/trader/iron_condor_0dte.yaml

# 仅后端
python -m trader.main run -c configs/trader/iron_condor_0dte.yaml
```

### 环境变量

```bash
# 复制模板后填值
cp .env.example .env

# 或直接导出
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### CLI

```bash
python -m trader.main status  -c configs/trader/iron_condor_0dte.yaml
python -m trader.main trades  -c configs/trader/iron_condor_0dte.yaml
python -m trader.main close-all -c configs/trader/iron_condor_0dte.yaml
```

> 配置文件默认 `testnet: true`，正式交易前改为 `false`。

## 项目结构

```
src/options_backtest/    # 回测引擎（策略、撮合、定价、分析）
trader/                  # 自动交易系统
  ├── binance_client.py  #   Binance API 客户端
  ├── limit_chaser.py    #   限价追单引擎
  ├── position_manager.py#   仓位管理
  ├── strategy.py        #   0DTE Iron Condor 策略
  ├── engine.py          #   交易引擎（后台线程）
  ├── dashboard.py       #   Streamlit 管理面板
  └── storage.py         #   SQLite 持久化
configs/                 # YAML 配置（回测 + 交易）
monitor/                 # 跨交易所行情监控面板
tests/                   # 216 个单元测试
```

## 测试

```bash
pytest tests/ -v
```

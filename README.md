# Deribit Options Backtest

加密货币期权回测系统 + 实时监控面板。支持 BTC/ETH，覆盖 Deribit / OKX / Binance。

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 获取数据
options-bt fetch -u BTC -s 2025-01-01 -e 2025-12-31 -r 60

# 运行回测
options-bt run --config configs/covered_call.yaml

# 启动实时监控面板
streamlit run monitor/app.py --server.port 8501
```

## 项目结构

```
project_Options/
├── README.md                   # 本文件
├── pyproject.toml              # 项目元数据 & 依赖
│
├── src/options_backtest/       # 核心库代码
│   ├── data/                   #   数据获取、加载、模型
│   ├── pricing/                #   Black-76 定价、IV 求解
│   ├── engine/                 #   回测引擎、撮合、仓位、结算
│   ├── strategy/               #   策略（CC / SP / Strangle / IC / DualInvest）
│   ├── analytics/              #   绩效指标（BTC/USD）、Plotly 图表
│   ├── cli.py                  #   命令行入口
│   ├── config.py               #   配置解析（Pydantic）
│   └── utils.py                #   通用工具函数
│
├── configs/                    # 回测策略配置 YAML
│   ├── covered_call*.yaml      #   Covered Call 系列
│   ├── short_put*.yaml         #   Short Put 系列
│   ├── short_strangle*.yaml    #   Short Strangle / 0DTE 双卖
│   ├── iron_condor.yaml        #   Iron Condor
│   ├── nc_eth*.yaml            #   Naked Call ETH 系列
│   └── default.yaml            #   默认配置
│
├── tests/                      # 单元测试（pytest）
│   ├── test_engine.py
│   ├── test_pricing.py
│   └── test_strategy.py
│
├── scripts/                    # 可复用分析脚本
│   ├── compare_cc_nc*.py       #   CC vs NC 策略对比
│   ├── compare_hedged_unhedged.py
│   ├── generate_report.py      #   HTML 报告生成
│   ├── analyze_losses.py       #   亏损日分析
│   ├── validate_hedge.py       #   对冲验证
│   ├── test_entry_hour.py      #   开仓时间优化
│   └── test_optimizations.py   #   参数优化批量测试
│
├── tools/                      # 独立工具
│   ├── analyze_results.py      #   回测结果分析
│   ├── analyze_k2.py           #   Call Spread K2 行权价分析
│   ├── extract_analysis.py     #   从 HTML 报告提取数据
│   └── filter_instruments_by_coverage.py  # 合约覆盖率过滤
│
├── monitor/                    # Streamlit 实时监控面板
│   ├── app.py                  #   面板主程序
│   └── exchanges.py            #   Deribit / OKX / Binance API
│
├── sandbox/                    # 临时实验区（不影响主流程）
│   ├── experiments/            #   一次性参数探索脚本
│   ├── api_tests/              #   API 端点测试脚本
│   ├── benchmarks/             #   性能基准 / profiling
│   └── logs/                   #   实验输出日志（gitignored）
│
├── data/                       # 数据存储（gitignored）
│   ├── underlying/             #   BTC/ETH 现货指数
│   ├── market_data/60/         #   1H 期权 OHLCV（Deribit）
│   ├── instruments/            #   合约元数据
│   └── settlements/            #   到期结算记录
│
├── reports/                    # 回测报告输出（gitignored）
│   ├── comparison/             #   策略对比报告
│   ├── entry_hour/             #   开仓时间分析
│   ├── loss_analysis/          #   亏损分析
│   ├── optimizations/          #   参数优化结果
│   └── validation/             #   对冲验证报告
│
└── docs/                       # 项目文档
    └── requirements_design.md  #   需求设计文档（MVP）
```

## 实时交易系统 (Trader)

基于 Deribit 期权 API 的铁鹰 0DTE 自动交易系统，内置 Streamlit 管理面板。

### 启动（前后端一体化）

```bash
# 一条命令启动 Dashboard + 交易引擎
.\.venv\Scripts\python.exe -m streamlit run trader/dashboard.py -- --config configs/trader_iron_condor_0dte.yaml
```

浏览器访问 `http://localhost:8501`，点击侧边栏 **🚀 启动引擎** 即可开始自动交易。

### 环境变量

```bash
# Deribit API 密钥（也可写入 YAML 配置）
$env:DERIBIT_CLIENT_ID = "your_client_id"
$env:DERIBIT_CLIENT_SECRET = "your_client_secret"
```

### CLI 命令（仅后端）

```bash
# 直接运行交易程序（无前端）
.\.venv\Scripts\python.exe -m trader.main run -c configs/trader_iron_condor_0dte.yaml

# 查看状态 / 成交 / 资产曲线 / 统计
.\.venv\Scripts\python.exe -m trader.main status -c configs/trader_iron_condor_0dte.yaml
.\.venv\Scripts\python.exe -m trader.main trades -c configs/trader_iron_condor_0dte.yaml
.\.venv\Scripts\python.exe -m trader.main equity -c configs/trader_iron_condor_0dte.yaml
.\.venv\Scripts\python.exe -m trader.main stats  -c configs/trader_iron_condor_0dte.yaml

# 紧急平仓
.\.venv\Scripts\python.exe -m trader.main close-all -c configs/trader_iron_condor_0dte.yaml
```

> 配置文件默认 `testnet: true`（Deribit 测试网），正式交易前在 `configs/trader_iron_condor_0dte.yaml` 中改为 `false`。

## 实时监控面板

跨交易所期权行情监控，功能包括：

- **多交易所对比**：Deribit / OKX / Binance 实时行情
- **智能 APR 排序**：基于时间价值（extrinsic value）计算，排除 ITM 内在价值虚增
- **TOP 20 机会表**：按 APR 排序的最佳 CC/SP 机会
- **收益预估器**：输入资金量，计算预期日/周/月/年收益
- **到期收益曲线**：Plotly 交互图表
- **自动刷新**：可设置 30s / 60s / 300s 自动更新

```bash
streamlit run monitor/app.py --server.port 8501
```

## 回测策略

| 策略 | 配置文件 | 说明 |
|------|----------|------|
| Covered Call | `covered_call.yaml` | 备兑卖 Call（0-DTE ATM 每日滚动） |
| Covered Call 复利 | `covered_call_compound.yaml` | 收益复投 |
| Short Put | `short_put_btc.yaml` | 卖 Put（0-DTE ATM） |
| Short Put 复利 | `short_put_btc_compound.yaml` | 收益复投 |
| Short Strangle | `short_strangle_0dte_12m.yaml` | 卖出宽跨式（0-DTE 1% OTM） |
| Iron Condor | `iron_condor.yaml` | 有限风险宽跨式（保护翼） |
| Dual Invest | `dual_invest.yaml` | OKX 双币赢模拟 |
| Naked Call | `nc_eth_12m.yaml` | 裸卖 Call（ETH 12 个月） |

ETH 版本：`covered_call_eth.yaml`、`short_put_eth.yaml` 及对应 `_compound` / `_okx` 后缀。

```bash
# 示例
options-bt run --config configs/covered_call_compound.yaml
options-bt run --config configs/short_strangle_0dte_12m.yaml
```

## 数据获取

```bash
options-bt fetch -u BTC -s 2025-01-01 -e 2025-12-31 -r 60
options-bt fetch -u ETH -s 2025-01-01 -e 2025-12-31 -r 60
```

**数据源层次**：引擎优先使用真实期权 OHLCV（`data/market_data/60/`），无数据时回退到 Black-76 + Proxy IV（已实现波动率 + 微笑/期限调整）。

## IV 定价模式

| 模式 | 配置值 | 说明 |
|------|--------|------|
| **Proxy** | `iv_mode: proxy` | 已实现波动率 → ATM IV → moneyness/term 调整（默认） |
| **Surface** | `iv_mode: surface` | 从历史期权价格反推 IV 曲面 |
| **Fixed** | `iv_mode: fixed` | 固定 IV（`fixed_iv: 0.60`） |

## 运行测试

```bash
pytest tests/ -v
```

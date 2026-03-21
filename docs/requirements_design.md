# Deribit 期权策略回测系统 — 需求设计文档 (MVP)

> 版本：v0.1 MVP  
> 日期：2026-03-07  
> 状态：MVP 精简版  

---

## 目录

1. [项目概述](#1-项目概述)
2. [MVP 范围定义](#2-mvp-范围定义)
3. [数据需求](#3-数据需求)
4. [系统架构](#4-系统架构)
5. [核心模块](#5-核心模块)
6. [策略框架](#6-策略框架)
7. [回测引擎](#7-回测引擎)
8. [绩效与可视化](#8-绩效与可视化)
9. [技术选型](#9-技术选型)
10. [项目结构](#10-项目结构)
11. [版本路线图](#11-版本路线图)
12. [附录](#12-附录)

---

## 1. 项目概述

基于 Deribit 历史行情数据的加密货币期权策略回测系统。支持 BTC/ETH 欧式期权，使用 Black-76 定价模型，币本位计价。

**核心价值：** 用真实历史行情验证期权策略，快速迭代策略参数。

---

## 2. MVP 范围定义

### 2.1 MVP 包含 ✅

| 模块 | MVP 范围 |
|------|----------|
| 数据获取 | Deribit API 获取 BTC 期权历史数据 |
| 数据存储 | Parquet 文件本地存储 |
| 数据加载 | Pandas 直接加载，构建期权链快照 |
| 定价模型 | Black-76 定价 + Greeks 解析解 |
| 回测引擎 | 固定时间步长（1h），顺序遍历 |
| 撮合 | Mid-price 成交 + 固定滑点 + Deribit 费率 |
| 仓位管理 | 开/平仓、持仓盈亏、到期结算 |
| 策略 | 策略基类 + 2 个内置策略（Short Strangle, Long Call） |
| 绩效 | 收益率、最大回撤、Sharpe、胜率 |
| 可视化 | 权益曲线、回撤图、交易标记 |
| 接口 | CLI 命令行 |

### 2.2 MVP 不包含 ❌（后续版本）

- 波动率曲面建模
- PnL 归因分析（Delta/Gamma/Theta/Vega 分解）
- 复杂风控规则引擎
- 多种滑点模型
- 策略注册表 / 热插拔
- 参数优化 / 网格搜索
- Web Dashboard
- ETH 期权（架构支持，MVP 只测 BTC）

---

## 3. 数据需求

### 3.1 数据源：Deribit API

| 数据类型 | API 端点 | 必要性 |
|----------|----------|--------|
| 期权合约列表 | `GET /public/get_instruments` | 必须 |
| 历史 K 线 | `GET /public/get_tradingview_chart_data` | 必须 |
| 指数价格 | `GET /public/get_index_price` | 必须 |
| 结算记录 | `GET /public/get_delivery_prices` | 必须 |
| 历史成交 | `GET /public/get_last_trades_by_instrument` | 可选 |
| 历史波动率 | `GET /public/get_historical_volatility` | 可选 |

### 3.2 数据模型

#### 期权合约 (OptionInstrument)

```
instrument_name     str         "BTC-26MAR26-80000-C"
underlying          str         "BTC" / "ETH"
strike_price        float       行权价
option_type         str         "call" / "put"
expiration_date     datetime    到期日（UTC）
creation_date       datetime    创建日期
contract_size       float       合约面值
tick_size           float       最小变动
is_active           bool        是否活跃
```

#### 行情快照 (OptionMarketData)

```
timestamp           datetime    时间戳（UTC）
instrument_name     str         合约名称
underlying_price    float       标的指数价格
mark_price          float       标记价格（BTC 计价）
mark_iv             float       标记隐含波动率 (%)
bid_price           float       最优买价
ask_price           float       最优卖价
last_price          float       最新成交价
volume_24h          float       24 小时成交量
open_interest       float       未平仓量
delta               float       Delta
gamma               float       Gamma
theta               float       Theta
vega                float       Vega
```

#### 标的价格 (UnderlyingOHLCV)

```
timestamp           datetime    时间戳
underlying          str         "BTC" / "ETH"
open                float       开盘价
high                float       最高价
low                 float       最低价
close               float       收盘价
volume              float       成交量
```

### 3.3 存储方案

- **格式：** Parquet 文件（按数据类型分目录）
- **读取：** Pandas DataFrame 直接加载
- **目录结构：**

```
data/
├── instruments/          # 合约信息
│   └── btc_instruments.parquet
├── market_data/          # 期权行情
│   └── BTC-26MAR26-80000-C.parquet
├── underlying/           # 标的价格
│   └── btc_index_1h.parquet
└── settlements/          # 结算价格
    └── btc_settlements.parquet
```

---

## 4. 系统架构

```
┌──────────────┐
│   CLI 入口    │
└──────┬───────┘
       │
┌──────▼───────┐    ┌──────────────┐
│  回测引擎     │◄───│  策略实例     │
│ (时间步循环)  │    └──────────────┘
└──────┬───────┘
       │
  ┌────┴─────┬──────────┬─────────────┐
  ▼          ▼          ▼             ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌────────────┐
│ 撮合  │ │ 仓位  │ │ 账户   │ │ 数据加载器  │
│ 模块  │ │ 管理  │ │ 管理   │ │            │
└──────┘ └──────┘ └────────┘ └─────┬──────┘
                                   │
                              ┌────▼─────┐
                              │ Parquet  │
                              │ 文件     │
                              └──────────┘

  独立模块：
  ┌──────────────┐  ┌──────────────┐
  │ Black-76 定价 │  │ 绩效 & 可视化 │
  └──────────────┘  └──────────────┘
```

---

## 5. 核心模块

### 5.1 数据获取 (DataFetcher)

- 异步获取 Deribit API 数据（aiohttp）
- 自动速率限制（每秒 20 次）+ 指数退避重试
- 支持断点续传（记录已拉取的时间范围）
- 数据落盘为 Parquet 文件

### 5.2 数据加载 (DataLoader)

- 加载指定时间范围的标的价格序列
- 构建"期权链快照"— 某一时刻所有可用合约及行情
- 时间戳对齐：标的价格与期权行情按时间步对齐

### 5.3 Black-76 定价

Deribit 使用 Black-76 模型（期货期权），无风险利率通常设 0：

$$C = e^{-rT}[F \cdot N(d_1) - K \cdot N(d_2)]$$
$$P = e^{-rT}[K \cdot N(-d_2) - F \cdot N(-d_1)]$$

$$d_1 = \frac{\ln(F/K) + \frac{\sigma^2 T}{2}}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**MVP 实现：**
- 理论价格计算
- Greeks 解析解（Delta, Gamma, Theta, Vega）
- IV 反算（scipy.optimize.brentq）

### 5.4 仓位管理 (PositionManager)

```
Position:
  instrument_name     合约名称
  direction           "long" / "short"
  quantity            持仓量
  entry_price         开仓均价 (BTC)
  entry_time          开仓时间
  current_mark_price  当前标记价
  unrealized_pnl      未实现盈亏 (BTC)
  realized_pnl        已实现盈亏 (BTC)
```

**核心逻辑：**
- 开仓 / 平仓 / 部分平仓
- 按 mark_price 计算未实现盈亏
- 到期自动结算：ITM 行权计算 PnL，OTM 作废

### 5.5 账户管理 (Account)

```
Account:
  initial_balance     初始余额 (BTC)
  balance             当前余额 (BTC)
  equity              权益 = balance + 未实现盈亏
  total_fee_paid      累计手续费
```

### 5.6 撮合 (Matcher)

- **成交价：** `(bid + ask) / 2`（mid-price）
- **滑点：** 固定值（默认 0.0001 BTC）
- **手续费：** Deribit 费率
  - Taker: 0.03% of underlying
  - 最小: 0.0003 BTC / 张
  - 最大: 合约价值的 12.5%
  - 交割: 0.015% (ITM)

---

## 6. 策略框架

### 6.1 策略基类

```python
class BaseStrategy:
    def initialize(self, context):
        """策略初始化，设置参数"""
        pass

    def on_step(self, context):
        """每个时间步调用，核心逻辑入口"""
        pass

    def on_expiry(self, context, expired_positions):
        """期权到期事件"""
        pass

    def on_fill(self, context, fill):
        """订单成交回调"""
        pass
```

### 6.2 策略上下文

```python
context:
  current_time          当前时间
  underlying_price      当前标的价格
  option_chain          当前可用期权链（DataFrame）
  positions             持仓列表
  account               账户信息
  
  # 动作方法
  buy(instrument, quantity)
  sell(instrument, quantity)
  close(instrument)
  close_all()
```

### 6.3 MVP 内置策略

**策略 1：Long Call**
- 买入 ATM/OTM Call，持有到期或止盈止损
- 参数：Delta 范围、持有天数、止盈/止损比例

**策略 2：Short Strangle**
- 卖出 OTM Call + OTM Put
- 参数：目标 Delta、最小/最大到期天数、换月天数
- 到期前自动平仓换月

---

## 7. 回测引擎

### 7.1 回测循环（固定步长）

```
初始化 → 加载数据 → 设定时间范围

for each time_step in time_range:
    1. 更新标的价格
    2. 更新所有持仓期权的 mark_price / Greeks
    3. 检查到期事件 → 结算到期持仓
    4. 构建 context（期权链快照）
    5. 调用 strategy.on_step(context)
    6. 处理策略发出的订单 → 撮合 → 更新仓位
    7. 记录权益快照

回测结束 → 计算绩效 → 输出报告
```

### 7.2 到期结算

```
到期日 UTC 08:00 触发:
  获取 Deribit 结算价格 (delivery_price)
  
  对每个到期持仓:
    if Call:
      intrinsic = max(0, settlement_price - strike) / settlement_price
    if Put:
      intrinsic = max(0, strike - settlement_price) / settlement_price
    
    if intrinsic > 0 (ITM):
      pnl = intrinsic × quantity × direction_sign - entry_price × quantity × direction_sign
      fee = delivery_fee
    else (OTM):
      pnl = -entry_price × quantity (买方) / +entry_price × quantity (卖方)
      fee = 0
    
    更新 realized_pnl，移除持仓
```

---

## 8. 绩效与可视化

### 8.1 绩效指标（MVP）

| 指标 | 计算方式 |
|------|----------|
| 总收益率 | (最终权益 - 初始资金) / 初始资金 |
| 年化收益率 | 总收益率按回测时长年化 |
| 最大回撤 | 权益曲线峰值到谷值最大跌幅 |
| Sharpe Ratio | 年化收益 / 年化波动率 |
| 胜率 | 盈利交易数 / 总交易数 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 总交易次数 | 完成的交易轮次 |
| 总手续费 | 累计交易费用 |

### 8.2 可视化（MVP）

| 图表 | 说明 |
|------|------|
| 权益曲线 | BTC 计价权益随时间变化 |
| 回撤曲线 | 回撤深度随时间变化 |
| 标的价格 + 交易标记 | 标的 K 线图叠加开平仓点位 |
| 每笔交易 PnL | 柱状图展示每笔交易盈亏 |

输出格式：Plotly HTML 交互式图表

---

## 9. 技术选型

| 库 | 用途 |
|----|------|
| `pandas` >= 2.0 | 数据处理 |
| `numpy` >= 1.24 | 数值计算 |
| `scipy` >= 1.10 | IV 求解 |
| `aiohttp` >= 3.8 | 异步 HTTP |
| `pyarrow` >= 12.0 | Parquet 读写 |
| `plotly` >= 5.15 | 交互式图表 |
| `pydantic` >= 2.0 | 数据校验 |
| `loguru` >= 0.7 | 日志 |
| `click` >= 8.0 | CLI |
| `tqdm` >= 4.65 | 进度条 |
| `pytest` >= 7.0 | 测试 |

---

## 10. 项目结构

```
project_Options/
├── docs/
│   └── requirements_design.md
│
├── src/
│   └── options_backtest/
│       ├── __init__.py
│       ├── cli.py                # CLI 入口
│       ├── config.py             # 配置管理
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetcher.py        # Deribit API 数据获取
│       │   ├── loader.py         # 数据加载器
│       │   └── models.py         # 数据模型 (Pydantic)
│       │
│       ├── pricing/
│       │   ├── __init__.py
│       │   ├── black76.py        # Black-76 定价 + Greeks
│       │   └── iv_solver.py      # IV 反算
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── backtest.py       # 回测主循环
│       │   ├── matcher.py        # 撮合模块
│       │   ├── position.py       # 仓位管理
│       │   ├── account.py        # 账户管理
│       │   └── settlement.py     # 到期结算
│       │
│       ├── strategy/
│       │   ├── __init__.py
│       │   ├── base.py           # 策略基类
│       │   ├── long_call.py      # Long Call 策略
│       │   └── short_strangle.py # Short Strangle 策略
│       │
│       └── analytics/
│           ├── __init__.py
│           ├── metrics.py        # 绩效指标
│           └── plotting.py       # 可视化
│
├── configs/
│   └── default.yaml              # 默认配置
│
├── tests/
│   ├── test_pricing.py
│   ├── test_engine.py
│   └── test_strategy.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 11. 版本路线图

| 版本 | 范围 | 预计周期 |
|------|------|----------|
| **v0.1 MVP** ← 当前 | 数据获取 + 定价 + 简单回测 + 2 策略 + 基础报告 | 3-4 周 |
| **v0.2** | 多策略模板 + 组合 Greeks 监控 + 基础风控 + 保证金计算 | 3-4 周 |
| **v1.0** | 波动率曲面 + PnL 归因 + 参数优化 + 事件驱动引擎 + 完整可视化 | 4-5 周 |

---

## 12. 附录

### Deribit 合约命名

```
{标的}-{到期日}-{行权价}-{类型}
BTC-26MAR26-80000-C  →  BTC 2026/3/26 行权价80000 Call
```

### Deribit 结算规则

- 结算时间：UTC 08:00
- 结算价格：到期前 30 分钟 TWAP
- 欧式期权，自动行权 ITM
- 币本位结算（PnL 和保证金均以 BTC/ETH 计价）

### 回测配置示例

```yaml
backtest:
  name: "BTC Short Strangle"
  start_date: "2025-01-01"
  end_date: "2025-12-31"
  time_step: "1h"
  underlying: "BTC"

account:
  initial_balance: 1.0   # BTC

execution:
  slippage: 0.0001        # 固定滑点 BTC
  taker_fee: 0.0003       # 0.03%
  delivery_fee: 0.00015   # 0.015%

strategy:
  name: "ShortStrangle"
  params:
    target_delta: 0.25
    min_days_to_expiry: 14
    max_days_to_expiry: 45
    roll_days_before_expiry: 3
    take_profit_pct: 50
    stop_loss_pct: 200
```

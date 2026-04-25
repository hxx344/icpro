# IC Pro

加密货币期权回测与实盘交易系统，当前实盘部分基于 Bybit Options。

## 交易程序概览

- 默认实盘配置：`configs/trader/weekend_vol_btc.yaml`
- 当前实盘策略：`weekend_vol`（可通过 `wing_delta` 在带翼/无翼结构之间切换）
- Dashboard 页面：总览、资产曲线、成交历史、策略配置、引擎状态
- 当前已接入执行事件日志、执行健康指标与“执行风控锁”

## 快速开始

### 1. 安装

```bash
git clone https://github.com/hxx344/icpro.git
cd icpro

python3 -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -U -e ".[dev,trader]"
```

> Dashboard 使用了 `st.fragment`，请确保 `streamlit>=1.37`。

### 2. 配置 API 密钥

```bash
# Windows PowerShell
$env:BYBIT_API_KEY = "your_key"
$env:BYBIT_API_SECRET = "your_secret"

# Linux/macOS
export BYBIT_API_KEY="your_key"
export BYBIT_API_SECRET="your_secret"
```

也可以写入 `.env`。

### 3. 启动

```bash
# Dashboard + 后台引擎
streamlit run trader/dashboard.py -- --config configs/trader/weekend_vol_btc.yaml

# 仅命令行引擎
python -m trader.main run -c configs/trader/weekend_vol_btc.yaml
```

> 当前示例实盘配置 `configs/trader/weekend_vol_btc.yaml` 使用 `simulate_private: false`、`testnet: false`。如需先做私有接口模拟，请手动改为 `simulate_private: true`。

## 推荐实盘策略

### BTC 7 因子分层 Delta Covered Call / Collar（研究 + 推荐面板）

新增的 BTC 期权覆盖性卖 Call / Collar 研究策略，用于在持有 BTC 底仓的同时，根据多因子牛熊强弱动态调整卖 Call 的目标 Delta，并在弱势阶段买入保护性 Put。

> 当前实现是**行情推荐与回测研究面板**，不会自动下单；实盘执行前仍需人工确认流动性、保证金、滑点和风控。

核心结构：

- `1x BTC` 现货/线性底仓敞口
- `2x short call`，目标 Delta 由牛熊强弱分层决定
- `2x protective put`，弱势/回撤条件下买入保护性 Put
- Delta 选合约方式：`at_least_target`，即优先选择不低于目标 Delta 的最近合约
- USD margin 口径，包含期权手续费与交割费估算

7 个牛熊投票因子：

| 因子 | 含义 |
|---|---|
| `rsi_14d_gt_50_lt_80` | 14 日 RSI 在 50~80，过滤弱势与过热 |
| `price_gt_sma_30d` | BTC 价格高于 30 日均线 |
| `roc_30d_gt_-5%` | 30 日涨跌幅大于 -5% |
| `onchain_sopr_proxy_155d_0p95_to_1p5` | SOPR 代理：价格 / 155 日前价格，位于 0.95~1.5 |
| `fear_greed_25_to_80` | 恐惧贪婪指数在 25~80 |
| `roc_730d_gt_+0%` | 两年动量为正 |
| `onchain_mvrv_1_to_3p5` | CoinMetrics MVRV 位于 1~3.5 |

投票强弱与卖 Call Delta 分层：

| 投票数量 | 状态 | 目标 Call Delta |
|---:|---|---:|
| 7/7 | 极强牛 | `0.01` |
| 6/7 | 牛市 | `0.01` |
| 4~5/7 | 中性偏弱 | `0.30` |
| 0~3/7 | 弱势/熊市 | `0.48` |

主要文件：

| 文件 | 说明 |
|---|---|
| `scripts/optimize_btc_covered_call.py` | 覆盖性卖 Call / Collar 历史回测核心逻辑 |
| `scripts/bybit_cc_recommender_panel.py` | 独立 Bybit 行情推荐面板 |
| `tmp/scan_independent_7of6_vote_count_delta_layers.py` | 7 因子投票数量分层 Delta 网格优化 |
| `tmp/backtest_layered_delta_natural_monthly_compound.py` | 自然到期后再月度复利回测 |

独立面板启动：

```bash
# 推荐先设置面板登录凭据；未设置时默认 admin / change-me-now
$env:CC_PANEL_USERNAME = "your_user"
$env:CC_PANEL_PASSWORD = "your_strong_password"
streamlit run scripts/bybit_cc_recommender_panel.py
```

也可以使用 Streamlit secrets 配置：

```toml
cc_panel_username = "your_user"
cc_panel_password = "your_strong_password"
```

面板功能：

- 拉取 Bybit BTC 现货、永续资金费率与期权盘口
- 自动计算 7 因子投票、牛熊强弱和目标 Call Delta
- 推荐当前最接近策略条件的卖 Call 与保护性 Put
- 展示权利金、最大潜在赔付、到期收益情景和 Greeks
- 支持在侧边栏调整 Call/Put 倍数、Delta 分层和筛选条件

研究回测结果（2023-04-25 至 2026-04-25，BTC 小时级期权数据）：

| 版本 | 最终权益 | 总收益 | 最大回撤 | Sharpe |
|---|---:|---:|---:|---:|
| 固定 1 BTC 非复利 | `$170,350` | `+520.54%` | `-15.84%` | `2.189` |
| 强制月度复利 | `$186,279` | `+578.56%` | `-28.68%` | `2.060` |
| 自然到期后月度复利 | `$221,729` | `+707.70%` | `-20.35%` | `2.225` |

自然复利版本不会在月末强制平仓，而是在跨月后等待原有期权腿自然到期，账户空仓后再按当前权益重置下一轮 BTC 名义本金，因此不会破坏保护性 Put 的连续性。

### Weekend Vol BTC（推荐）

当前推荐配置是 BTC 周末波动率卖方策略；默认结构是**无翼结构**，但也可通过 `wing_delta > 0` 切到带翼结构：

- `mode: weekend_vol`
- `underlying: BTC`
- `target_delta: 0.45`
- `wing_delta: 0.0`（即无保护翼、无翼结构）
- `entry_day: friday`
- `entry_time_utc: 18:00`
- `check_interval_sec: 5`
- `stop_loss_pct: 200`
- `stop_loss_underlying_move_pct: 5`

对应配置文件：`configs/trader/weekend_vol_btc.yaml`

### 其他配置

| 配置文件 | 说明 |
|---|---|
| `configs/trader/weekend_vol_btc.yaml` | Weekend Vol BTC |

## 当前实盘特性

- 优先使用交易所返回的 `delta / mark_iv`
- 支持同步分批市价执行，也保留可选限价追单能力
- 开仓前检查交易所真实持仓，避免与手动单冲突
- 启动时可从交易所持仓恢复到本地账本
- 恢复仓位可按默认费率估算手续费并回填
- 已接入执行事件日志、执行健康指标、执行风控锁

## 常用 CLI

```bash
python -m trader.main run -c configs/trader/weekend_vol_btc.yaml
python -m trader.main status -c configs/trader/weekend_vol_btc.yaml
python -m trader.main trades -c configs/trader/weekend_vol_btc.yaml
python -m trader.main equity -c configs/trader/weekend_vol_btc.yaml
python -m trader.main stats -c configs/trader/weekend_vol_btc.yaml
python -m trader.main close-all -c configs/trader/weekend_vol_btc.yaml
```

## 数据构建工具 (build_hourly_parquet)

从远程服务器下载 Deribit 期权逐笔 `.zst` 数据，重采样为**小时快照**并保存为 Parquet 文件。

### 数据源与输出

| 项目 | 说明 |
|------|------|
| 数据源 | `data.yutsing.work` (Cloudflare CDN)，每日一个 `OPTIONS.csv.zst` (~922 MB, 解压 ~5.7 GB) |
| 时间范围 | 2023-03-20 至今 |
| 输出路径 | `data/options_hourly/{BTC,ETH}/{YYYY-MM}.parquet` |
| 输出大小 | ~1–1.5 GB（3 年全量） |

### 快速开始

```bash
# 推荐默认：polars-stream + 64MB chunk
python scripts/build_hourly_parquet.py --csv-read-mode polars-stream --polars-stream-chunk-mb 64

# 自定义日期范围
python scripts/build_hourly_parquet.py --start 2024-01-01 --end 2024-06-30 --csv-read-mode polars-stream --polars-stream-chunk-mb 64

# 3 线程并发下载 + 8 段分段下载（推荐大批量使用）
python scripts/build_hourly_parquet.py --download-workers 3 --segments 8 --csv-read-mode polars-stream --polars-stream-chunk-mb 64

# 处理本地 .zst 文件（调试用）
python scripts/build_hourly_parquet.py --local tmp/OPTIONS.csv.zst --csv-read-mode polars-stream --polars-stream-chunk-mb 64
```

> 当前推荐默认工作流：显式使用 `polars-stream`，并保持 `--polars-stream-chunk-mb 64`。

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start` | `2023-03-20` | 起始日期 `YYYY-MM-DD` |
| `--end` | `2025-12-31` | 结束日期 `YYYY-MM-DD` |
| `--download-workers N` | `1` | 并发下载线程数，建议 2–3 |
| `--segments N` | `8` | 单文件 curl 分段并行下载数（设 1 为单连接） |
| `--csv-read-mode` | `auto` | CSV 读取模式：`auto`(按文件大小自动)、`bulk`(最快但吃内存)、`stream`(分段读取省内存)、`hybrid`、`polars-stream`(当前推荐) |
| `--csv-block-mb N` | `1024` | `stream` 模式每批读取块大小（MB） |
| `--polars-stream-chunk-mb N` | `64` | `polars-stream` 模式 chunk 大小（MB），当前推荐保持 64 |
| `--cpu-limit N` | `0` | 限制 Arrow/BLAS 线程数（`0`=自动，`1`=最低占用） |
| `--keep-cache` | 关 | 处理完毕后保留 `.zst` 缓存文件 |
| `--cache-only` | 关 | 仅处理本地已有的缓存，不下载 |
| `--force` | 关 | 忽略 checkpoint，重新下载并处理全部日期 |
| `--dry-run` | 关 | 列出远程可用日期，不做处理 |
| `--hourly-pick` | `both` | 小时快照选取: `first`=开盘, `last`=收盘, `both`=两者都保留 |
| `--proxy URL` | 无 | HTTP 代理，如 `http://127.0.0.1:7890` |
| `--local FILE` | 无 | 处理单个本地 `.zst` 文件（测试用） |
| `-v` / `--verbose` | 关 | 详细日志 |

### 核心机制

- **子进程隔离**：每天数据在独立子进程中处理，进程退出后 OS 回收全部内存，避免多日运行的内存累积
- **并发下载流水线**：`--download-workers N` 启动 N 个下载线程，通过有界队列 (back-pressure) 与主线程串行处理衔接，避免过度预下载占用磁盘
- **分段下载**：`--segments N` 使用 curl 多连接并行下载单个大文件，支持 Range 分段加速
- **推荐主链路**：`--csv-read-mode polars-stream --polars-stream-chunk-mb 64`，走 `zstd-cli/python-zstd -> chunked Polars parse -> 小时聚合`，当前综合速度最佳
- **分段读 CSV**：`--csv-read-mode stream` 使用 `open_csv` 分批读取，显著降低峰值内存；`auto` 仍可按内存环境自动选择传统路径
- **可控 CPU 线程**：`--cpu-limit N` 可限制 Arrow/BLAS 线程，减少抢占
- **断点续传**：每日处理完写入 checkpoint (`data/options_hourly/.checkpoint`)，程序重启自动跳过已完成日期
- **中断可回收**：按 `Ctrl+C` 会终止主进程并回收已启动的下载/处理子进程（Windows 也处理 `SIGBREAK`）
- **实时进度**：运行时每 2 秒刷新状态行，显示下载/处理进度、队列深度、预估剩余时间

```
[01:23] DL 15/30 (2 active: 04-05,04-06) | Proc 12/30 (04-04) | Q:2  ETA 38m
```

### 性能参考（典型）

| 模式 | 单日耗时 | 内存峰值 | 说明 |
|------|----------|----------|------|
| `polars-stream (64MB)` | ~104s | 中低 | 当前推荐默认，综合速度/内存最优 |
| `bulk` | 更快（机器相关） | 高 | 适合内存充足、追求速度 |
| `stream` | 略慢（机器相关） | 低 | 适合内存紧张，避免一次性读满 |

> 提示：若系统安装了 `zstd` 命令行工具，解压会优先走 `zstd-cli`（通常比 Python 解压更快）。

### 使用示例

```bash
# 预览有哪些日期可下载
python scripts/build_hourly_parquet.py --dry-run --start 2024-01-01

# 仅下载不处理（配合 --keep-cache 先攒缓存）
# → 之后用 --cache-only 离线处理
python scripts/build_hourly_parquet.py --start 2024-06-01 --end 2024-06-30 --keep-cache
python scripts/build_hourly_parquet.py --cache-only --start 2024-06-01 --end 2024-06-30

# 带代理下载
python scripts/build_hourly_parquet.py --proxy http://127.0.0.1:7890

# 强制重新处理某个范围（忽略 checkpoint）
python scripts/build_hourly_parquet.py --start 2024-03-01 --end 2024-03-31 --force

# 高性能全量构建（内存充足时）
python scripts/build_hourly_parquet.py --download-workers 3 --segments 8 --keep-cache -v

# 内存优先模式（分段读取 CSV，避免一次性读满内存）
python scripts/build_hourly_parquet.py --download-workers 3 --segments 8 --csv-read-mode stream --csv-block-mb 384 --keep-cache -v

# 中断测试（运行后按 Ctrl+C，应快速退出并回收子进程）
python scripts/build_hourly_parquet.py --start 2024-06-01 --end 2024-06-02 --download-workers 3 --segments 8 -v
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

功能：复利/非复利、周末跳过、IV 条件翼跳过、Bybit/通用费率模型。

## 项目结构

```
src/options_backtest/    # 回测引擎（策略、撮合、定价、分析）
trader/                  # 自动交易系统
  ├── bybit_client/      #   Bybit API 客户端 + Greeks 获取
  ├── position_manager.py#   仓位管理
  ├── strategy.py        #   WeekendVolStrategy（支持带翼/无翼结构）
  ├── engine.py          #   交易引擎（后台线程）
  ├── dashboard.py       #   Streamlit 管理面板
  ├── config.py          #   YAML 配置加载
  └── storage.py         #   SQLite 持久化
configs/                 # YAML 配置（回测 + 交易）
monitor/                 # 行情监控面板
tests/                   # 单元测试与回归测试
```

## 测试

```bash
pytest tests/ -v
```

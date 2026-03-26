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

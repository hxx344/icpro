"""
CryptoDataDownload API — 精准下载 0DTE 回测所需数据
====================================================
策略: 只下载每个到期日当天、ATM ±10% 行权价范围的期权 OHLCV。
优化: 887次 API 调用 (vs 逐合约 40,000+), 每次返回 ~33 行。

用法:
    python scripts/cdd_download_0dte.py                # BTC 全量
    python scripts/cdd_download_0dte.py --year 2024    # 只下 2024
    python scripts/cdd_download_0dte.py --test         # 测试 3 个日期
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from typing import Optional, Union

from cdd_secrets import get_cdd_api_token

# ── 配置 ──────────────────────────────────────────────────────────────
BASE_URL = "https://api.cryptodatadownload.com/v1"
HEADERS = {"Authorization": f"Token {get_cdd_api_token()}"}

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MARKET_DIR = DATA_DIR / "market_data" / "1D"       # 日频 OHLCV
CHECKPOINT_FILE = PROJECT_DIR / "scripts" / ".cdd_0dte_checkpoint.json"

# 速率: 保守 10s 间隔, 429 后自适应
BASE_DELAY = 10.0
MAX_DELAY = 120.0


def load_expiry_plan(currency: str = "BTC", year: int = None,
                     strike_range: float = 0.10) -> list:
    """
    构建下载计划: 每个到期日需要哪些合约。
    返回: [{expiry_code, expiry_date, instruments: [name,...], index_price}, ...]
    """
    inst_path = DATA_DIR / "instruments" / f"{currency.lower()}_instruments.parquet"
    stl_path = DATA_DIR / "settlements" / f"{currency.lower()}_settlements.parquet"

    inst = pd.read_parquet(inst_path)
    stl = pd.read_parquet(stl_path)

    # 解析到期日
    inst["exp_dt"] = pd.to_datetime(inst["expiration_date"], utc=True)
    inst["exp_date"] = inst["exp_dt"].dt.date

    # settlement index_price (结算时的现货价, 用于确定 ATM)
    stl["stl_date"] = pd.to_datetime(stl["settlement_timestamp"], utc=True).dt.date
    ref_prices = stl.groupby("stl_date")["index_price"].first().to_dict()

    # 下载 2023-01-01 ~ 2025-12-31 全部到期日
    start = date(2023, 1, 1)
    end = date(2025, 12, 31)

    plan = []
    for exp_date in sorted(inst["exp_date"].unique()):
        if exp_date < start or exp_date > end:
            continue
        if year and exp_date.year != year:
            continue

        idx_price = ref_prices.get(exp_date)
        if not idx_price:
            continue

        # ATM ± range 过滤
        lo = idx_price * (1 - strike_range)
        hi = idx_price * (1 + strike_range)
        exp_inst = inst[inst["exp_date"] == exp_date]
        filtered = exp_inst[
            (exp_inst["strike_price"] >= lo) & (exp_inst["strike_price"] <= hi)
        ]
        if filtered.empty:
            continue

        # 提取 expiry code (e.g. "1JAN23" from "BTC-1JAN23-16000-C")
        sample_name = filtered["instrument_name"].iloc[0]
        expiry_code = sample_name.split("-")[1]  # "1JAN23"

        plan.append({
            "expiry_code": expiry_code,
            "expiry_date": exp_date.isoformat(),
            "index_price": idx_price,
            "instrument_count": len(filtered),
            "instruments": filtered["instrument_name"].tolist(),
        })

    return plan


def load_checkpoint() -> set:
    """返回已完成的 expiry_date 集合"""
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set):
    CHECKPOINT_FILE.write_text(json.dumps({
        "completed": sorted(completed),
        "updated": datetime.now().isoformat(),
    }, indent=2))


def fetch_expiry_day(currency: str, expiry_code: str,
                     expiry_date: str, max_retries: int = 3) -> Union[list, int, None]:
    """
    调用 CDD API 获取某个到期日当天的所有合约 OHLCV.
    GET /data/ohlc/deribit/options/?currency=BTC&expiry=1JAN23&date=2023-01-01
    内置连接错误重试机制.
    """
    url = f"{BASE_URL}/data/ohlc/deribit/options/"
    params = {
        "currency": currency,
        "expiry": expiry_code,
        "date": expiry_date,
        "limit": 5000,
        "return": "JSON",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"\n  ⚠ 连接错误 (retry {attempt+1}/{max_retries}), "
                      f"等待 {wait}s: {type(e).__name__}")
                time.sleep(wait)
                continue
            else:
                print(f"\n  ✗ 连接错误, 已重试 {max_retries} 次: {type(e).__name__}")
                return None

        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", data.get("data", []))
            if isinstance(result, list):
                return result
            return []
        elif resp.status_code == 429:
            # 解析冷却时间
            try:
                body = resp.json()
                msg = str(body.get("detail", body.get("message", "")))
                import re
                m = re.search(r"(\d+)\s*seconds?", msg)
                wait = int(m.group(1)) + 5 if m else 60
            except Exception:
                wait = 60
            return wait  # 返回 int 表示需要等待
        elif resp.status_code >= 500 and attempt < max_retries - 1:
            wait = 30 * (attempt + 1)
            print(f"\n  ⚠ 服务端错误 {resp.status_code} (retry {attempt+1}), "
                  f"等待 {wait}s")
            time.sleep(wait)
            continue
        else:
            print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
            return None

    return None


def save_instrument_ohlcv(name: str, row: dict):
    """
    将一条 API 记录保存为与引擎兼容的 parquet 文件。
    CDD 返回: {unix, date, symbol, open, high, low, close, volume}
    引擎期望: timestamp(datetime), open, high, low, close, volume
    """
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKET_DIR / f"{name}.parquet"

    # 如果已存在且有数据, 跳过 (不覆盖 fetcher 数据)
    if path.exists():
        return

    # 构建 DataFrame
    ts = pd.Timestamp(row["date"], tz="UTC") if isinstance(row["date"], str) else \
         pd.Timestamp(int(row["unix"]), unit="ms", tz="UTC")
    # Deribit 日频 OHLCV 从 08:00 UTC 开始
    if ts.hour == 0:
        ts = ts.replace(hour=8)

    df = pd.DataFrame([{
        "timestamp": ts,
        "open": float(row.get("open", 0)),
        "high": float(row.get("high", 0)),
        "low": float(row.get("low", 0)),
        "close": float(row.get("close", 0)),
        "volume": float(row.get("volume", 0)),
    }])
    df.to_parquet(path, index=False)


def run_download(currency: str = "BTC", year: int = None,
                 test_mode: bool = False, strike_range: float = 0.10):
    print(f"{'='*60}")
    print(f"CDD 0DTE 精准下载 — {currency} {'year='+str(year) if year else 'ALL'}")
    print(f"ATM ± {strike_range*100:.0f}% | 只下载到期日当天数据")
    print(f"{'='*60}")

    plan = load_expiry_plan(currency, year, strike_range)
    completed = load_checkpoint()

    # 过滤已完成
    remaining = [p for p in plan if p["expiry_date"] not in completed]

    if test_mode:
        remaining = remaining[:3]
        print("TEST MODE: 只处理 3 个到期日")

    total = len(plan)
    done = len(plan) - len(remaining)
    print(f"计划: {total} 个到期日, 已完成: {done}, 待下载: {len(remaining)}")
    total_instruments = sum(p["instrument_count"] for p in remaining)
    print(f"预计合约数: {total_instruments} (ATM ±{strike_range*100:.0f}%)")
    est_hours = len(remaining) * BASE_DELAY / 3600
    print(f"预计耗时: {est_hours:.1f} 小时 ({BASE_DELAY:.0f}s/请求)")
    print()

    delay = BASE_DELAY
    consecutive_ok = 0
    saved_count = 0
    skipped_count = 0
    api_calls = 0

    for i, entry in enumerate(remaining):
        exp_code = entry["expiry_code"]
        exp_date = entry["expiry_date"]
        idx_price = entry["index_price"]
        needed = set(entry["instruments"])

        pct = (done + i + 1) / total * 100
        print(f"[{done+i+1}/{total}] {exp_date} ({exp_code}) "
              f"ATM=${idx_price:,.0f} | {len(needed)}合约 | {pct:.1f}%",
              end=" ", flush=True)

        # API 调用
        result = fetch_expiry_day(currency, exp_code, exp_date)
        api_calls += 1

        if isinstance(result, int):
            # 429 — 需要等待
            wait = result
            print(f"⏱ RATE LIMIT, 等待 {wait}s...")
            time.sleep(wait)
            # 重试
            result = fetch_expiry_day(currency, exp_code, exp_date)
            api_calls += 1
            if isinstance(result, int):
                print(f"  再次限制, 等待 {result}s...")
                time.sleep(result)
                result = fetch_expiry_day(currency, exp_code, exp_date)
                api_calls += 1

            delay = min(delay * 1.5, MAX_DELAY)
            consecutive_ok = 0

        if result is None or isinstance(result, int):
            print("✗ FAILED")
            # 也记录失败日期到 checkpoint，避免重启时重复请求
            completed.add(exp_date)
            if (i + 1) % 20 == 0:
                save_checkpoint(completed)
            continue

        # 保存数据
        batch_saved = 0
        batch_skipped = 0
        for row in result:
            symbol = row.get("symbol", "")
            if symbol in needed:
                path = MARKET_DIR / f"{symbol}.parquet"
                if path.exists():
                    batch_skipped += 1
                    skipped_count += 1
                else:
                    save_instrument_ohlcv(symbol, row)
                    batch_saved += 1
                    saved_count += 1

        print(f"→ API返回{len(result)}行, 保存{batch_saved}, "
              f"跳过{batch_skipped} | delay={delay:.0f}s")

        # 更新 checkpoint
        completed.add(exp_date)
        if (i + 1) % 5 == 0 or i == len(remaining) - 1:
            save_checkpoint(completed)

        # 自适应速率
        consecutive_ok += 1
        if consecutive_ok >= 5 and delay > BASE_DELAY:
            delay = max(delay * 0.8, BASE_DELAY)
            consecutive_ok = 0

        time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"完成! API调用: {api_calls}, 保存: {saved_count}, 跳过: {skipped_count}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    save_checkpoint(completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDD 0DTE 精准数据下载")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--year", type=int, default=None, help="只下载指定年份")
    parser.add_argument("--test", action="store_true", help="测试模式(3个日期)")
    parser.add_argument("--range", type=float, default=0.10,
                        help="ATM ± 范围 (默认 0.10 = ±10%%)")
    parser.add_argument("--delay", type=float, default=None,
                        help="请求间隔秒数 (默认 10)")
    args = parser.parse_args()

    if args.delay:
        BASE_DELAY = args.delay

    run_download(args.currency, args.year, args.test, args.range)

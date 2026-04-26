"""WeekendVol 参数网格优化。

基于当前 `configs/backtest/weekend_vol_btc_hourly.yaml`，扫描：
- `target_delta`
- `wing_delta`
- `leverage`
- `max_delta_diff`

输出：
- `reports/optimizations/weekend_vol_grid.json`
- `reports/optimizations/weekend_vol_grid.csv`
- `reports/optimizations/weekend_vol_grid_top10.json`
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.utils import to_utc_timestamp


logger.remove()
logger.add(sys.stderr, level="WARNING")

BASE_CONFIG = Path("configs/backtest/weekend_vol_btc_hourly.yaml")
OUTPUT_DIR = Path("reports/optimizations")
OUTPUT_JSON = OUTPUT_DIR / "weekend_vol_grid.json"
OUTPUT_CSV = OUTPUT_DIR / "weekend_vol_grid.csv"
OUTPUT_TOP = OUTPUT_DIR / "weekend_vol_grid_top10.json"
OUTPUT_CACHE = OUTPUT_DIR / "weekend_vol_grid_cache.json"

CACHE_VERSION = 1

TARGET_DELTAS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
WING_DELTAS = [0.00, 0.03, 0.05, 0.07, 0.10]
LEVERAGES = [2.0, 3.0, 4.0]
MAX_DELTA_DIFFS = [0.10, 0.15, 0.20]


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _month_starts(start_ts, end_ts) -> list[pd.Timestamp]:
    start_month = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC")
    return list(pd.date_range(start=start_month, end=end_month, freq="MS", tz="UTC"))


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.as_posix()}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}"


def _build_cache_namespace(cfg: Config) -> str:
    underlying = str(cfg.backtest.underlying).upper()
    start_ts = to_utc_timestamp(cfg.backtest.start_date)
    end_ts = to_utc_timestamp(cfg.backtest.end_date)

    parts = [
        f"v{CACHE_VERSION}",
        BASE_CONFIG.read_text(encoding="utf-8"),
    ]

    data_root = Path("data")
    instruments_path = data_root / "instruments" / f"{underlying.lower()}_instruments.parquet"
    parts.append(_file_fingerprint(instruments_path))

    underlying_candidates = [
        data_root / "underlying" / f"{underlying.lower()}_index_60.parquet",
        data_root / "underlying" / f"{underlying.lower()}_index_60m.parquet",
    ]
    for candidate in underlying_candidates:
        if candidate.exists():
            parts.append(_file_fingerprint(candidate))
            break

    hourly_root = data_root / "options_hourly" / underlying
    for month in _month_starts(start_ts, end_ts):
        month_path = hourly_root / f"{month.year:04d}-{month.month:02d}.parquet"
        if month_path.exists():
            parts.append(_file_fingerprint(month_path))

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _load_result_cache(namespace: str) -> dict[str, dict]:
    if not OUTPUT_CACHE.exists():
        return {}
    try:
        payload = json.loads(OUTPUT_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if int(payload.get("version", -1)) != CACHE_VERSION:
        return {}
    if str(payload.get("namespace", "")) != namespace:
        return {}
    rows = payload.get("rows", {})
    return rows if isinstance(rows, dict) else {}


def _save_result_cache(namespace: str, rows: dict[str, dict]) -> None:
    _atomic_write_json(
        OUTPUT_CACHE,
        {
            "version": CACHE_VERSION,
            "namespace": namespace,
            "rows": rows,
        },
    )


def _combo_cache_key(target_delta: float, wing_delta: float, leverage: float, max_delta_diff: float) -> str:
    return json.dumps(
        {
            "target_delta": round(float(target_delta), 8),
            "wing_delta": round(float(wing_delta), 8),
            "leverage": round(float(leverage), 8),
            "max_delta_diff": round(float(max_delta_diff), 8),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def run_one(
    target_delta: float,
    wing_delta: float,
    leverage: float,
    max_delta_diff: float,
    cached_rows: dict[str, dict],
) -> tuple[dict, bool]:
    cache_key = _combo_cache_key(target_delta, wing_delta, leverage, max_delta_diff)
    cached = cached_rows.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached), True

    cfg = Config.from_yaml(BASE_CONFIG)
    params = cfg.strategy.params

    params["target_delta"] = target_delta
    params["wing_delta"] = wing_delta
    params["leverage"] = leverage
    params["max_delta_diff"] = max_delta_diff

    label = (
        f"WeekendVol Δ={target_delta:.2f} "
        f"wing={wing_delta:.2f} lev={leverage:.1f} diff={max_delta_diff:.2f}"
    )
    cfg.backtest.name = label
    cfg.report.output_dir = f"./reports/optimizations/weekend_vol_runs/d{int(target_delta*100):02d}_w{int(wing_delta*100):02d}_l{int(leverage*10):02d}_md{int(max_delta_diff*100):02d}"

    strategy = _load_strategy(cfg.strategy.name, params)
    engine = BacktestEngine(cfg, strategy)

    t0 = time.perf_counter()
    results = engine.run()
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(results)

    total_return = float(metrics.get("total_return", 0.0)) * 100
    annualized_return = float(metrics.get("annualized_return", 0.0)) * 100
    max_dd = float(metrics.get("max_drawdown", 0.0)) * 100
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0)) * 100
    profit_factor = float(metrics.get("profit_factor", 0.0))
    total_trades = int(metrics.get("total_trades", 0))
    total_fees = float(metrics.get("total_fees", 0.0))
    final_equity = float(metrics.get("final_equity", 0.0))

    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0
    return_over_dd = total_return / abs(max_dd) if max_dd != 0 else 0.0

    row = {
        "target_delta": target_delta,
        "wing_delta": wing_delta,
        "leverage": leverage,
        "max_delta_diff": max_delta_diff,
        "total_return_pct": total_return,
        "annualized_return_pct": annualized_return,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "calmar_like": calmar,
        "return_over_dd": return_over_dd,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_fees": total_fees,
        "final_equity": final_equity,
        "elapsed_sec": elapsed,
    }
    cached_rows[cache_key] = row
    return row, False


def print_top(title: str, rows: list[dict], key: str, top_n: int = 10) -> list[dict]:
    top = sorted(rows, key=lambda r: r[key], reverse=True)[:top_n]
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    print(
        f"{'Rank':>4} {'Δ':>5} {'Wing':>5} {'Lev':>5} {'Diff':>5} "
        f"{'Ret%':>10} {'Ann%':>10} {'DD%':>8} {'Sharpe':>8} {'PF':>7} {'WR%':>7} {'Trades':>7}"
    )
    print("-" * 110)
    for i, r in enumerate(top, 1):
        print(
            f"{i:>4} "
            f"{r['target_delta']:>5.2f} {r['wing_delta']:>5.2f} {r['leverage']:>5.1f} {r['max_delta_diff']:>5.2f} "
            f"{r['total_return_pct']:>10.2f} {r['annualized_return_pct']:>10.2f} {r['max_drawdown_pct']:>8.2f} "
            f"{r['sharpe']:>8.2f} {r['profit_factor']:>7.2f} {r['win_rate_pct']:>7.2f} {r['total_trades']:>7}"
        )
    return top


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = Config.from_yaml(BASE_CONFIG)
    cache_namespace = _build_cache_namespace(base_cfg)
    cached_rows = _load_result_cache(cache_namespace)

    combos = list(itertools.product(TARGET_DELTAS, WING_DELTAS, LEVERAGES, MAX_DELTA_DIFFS))
    total = len(combos)
    rows: list[dict] = []

    print(f"总组合数: {total}")
    print(f"target_delta: {TARGET_DELTAS}")
    print(f"wing_delta:   {WING_DELTAS}")
    print(f"leverage:     {LEVERAGES}")
    print(f"max_delta_diff: {MAX_DELTA_DIFFS}")
    print("=" * 110)

    global_start = time.perf_counter()

    for idx, (target_delta, wing_delta, leverage, max_delta_diff) in enumerate(combos, 1):
        row, cache_hit = run_one(
            target_delta,
            wing_delta,
            leverage,
            max_delta_diff,
            cached_rows,
        )
        rows.append(row)
        if not cache_hit:
            _save_result_cache(cache_namespace, cached_rows)

        avg_time = (time.perf_counter() - global_start) / idx
        eta_min = avg_time * (total - idx) / 60
        src_label = "cache" if cache_hit else f"{row['elapsed_sec']:.2f}s"
        print(
            f"[{idx:>3}/{total}] Δ={target_delta:.2f} wing={wing_delta:.2f} lev={leverage:.1f} diff={max_delta_diff:.2f} -> "
            f"Ret={row['total_return_pct']:>8.2f}% DD={row['max_drawdown_pct']:>7.2f}% "
            f"Sharpe={row['sharpe']:>5.2f} PF={row['profit_factor']:>5.2f} "
            f"WR={row['win_rate_pct']:>6.2f}% ({src_label}, ETA {eta_min:.1f}m)"
        )

        if idx % 15 == 0:
            _atomic_write_json(OUTPUT_JSON, rows)

    total_elapsed = time.perf_counter() - global_start

    df = pd.DataFrame(rows).sort_values(
        ["sharpe", "calmar_like", "total_return_pct"],
        ascending=[False, False, False],
    )
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    _atomic_write_json(OUTPUT_JSON, df.to_dict(orient="records"))

    top_sharpe = print_top("Top 10 by Sharpe", rows, "sharpe")
    top_calmar = print_top("Top 10 by Calmar-like", rows, "calmar_like")
    top_return = print_top("Top 10 by Total Return", rows, "total_return_pct")

    summary = {
        "total_combinations": total,
        "total_elapsed_sec": total_elapsed,
        "best_by_sharpe": top_sharpe[0] if top_sharpe else None,
        "best_by_calmar_like": top_calmar[0] if top_calmar else None,
        "best_by_total_return": top_return[0] if top_return else None,
        "top10_by_sharpe": top_sharpe,
        "top10_by_calmar_like": top_calmar,
        "top10_by_total_return": top_return,
    }
    _atomic_write_json(OUTPUT_TOP, summary)

    print(f"\n完成，共 {total} 组，耗时 {total_elapsed / 60:.1f} 分钟")
    print(f"结果已保存到: {OUTPUT_JSON}")
    print(f"CSV 已保存到: {OUTPUT_CSV}")
    print(f"汇总已保存到: {OUTPUT_TOP}")


if __name__ == "__main__":
    main()

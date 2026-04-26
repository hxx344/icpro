from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.weekend_vol import WeekendVolStrategy
from scripts.research.run_cached_hourly_backtest import (
    _apply_initial_usd_balance,
    _choose_date_window,
    detect_hourly_coverage,
    load_run_config,
)

logger.remove()
logger.add(sys.stderr, level="WARNING")

DEFAULT_RV_GRID = "0,0.85,0.90,0.95"
DEFAULT_IV_GRID = "0,0.80,0.85"
DEFAULT_PREMIUM_GRID = "0,0.80,0.85"
DEFAULT_ABS_MOVE_GRID = "0,0.80,0.85"
DEFAULT_LOOKBACK_GRID = "52"
DEFAULT_MIN_HISTORY_GRID = "12"
DEFAULT_ABS_MOVE_LOOKBACK_HOURS = 24
DEFAULT_BATCH_SIZE = 5


class PercentileFilterWeekendVolStrategy(WeekendVolStrategy):
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.percentile_lookback_entries = int(self.params.get("percentile_lookback_entries", 52))
        self.percentile_min_history = int(self.params.get("percentile_min_history", 12))
        self.percentile_abs_move_lookback_hours = int(
            self.params.get("percentile_abs_move_lookback_hours", DEFAULT_ABS_MOVE_LOOKBACK_HOURS)
        )

        self.entry_rv_percentile_max = float(self.params.get("entry_rv_percentile_max", 0.0))
        self.entry_iv_percentile_max = float(self.params.get("entry_iv_percentile_max", 0.0))
        self.entry_premium_percentile_max = float(self.params.get("entry_premium_percentile_max", 0.0))
        self.entry_abs_move_percentile_max = float(self.params.get("entry_abs_move_percentile_max", 0.0))

        self._percentile_history: dict[str, list[float]] = {
            "rv_24h": [],
            "avg_short_iv": [],
            "combined_short_premium": [],
            "abs_move_24h": [],
        }
        self._percentile_filter_skips: list[dict[str, Any]] = []

    def get_diagnostics(self) -> dict[str, Any]:
        diagnostics = super().get_diagnostics()
        diagnostics["percentile_filter_skips"] = list(self._percentile_filter_skips)
        return diagnostics

    def _history_window(self, metric_name: str) -> np.ndarray:
        values = [float(v) for v in self._percentile_history.get(metric_name, []) if np.isfinite(v)]
        if self.percentile_lookback_entries > 0:
            values = values[-self.percentile_lookback_entries :]
        return np.asarray(values, dtype=float)

    def _append_percentile_metrics(self, metrics: dict[str, float | None]) -> None:
        for key in self._percentile_history:
            value = metrics.get(key)
            if value is None or not np.isfinite(value):
                continue
            self._percentile_history[key].append(float(value))

    def _metric_percentile(self, metric_name: str, value: float | None) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        hist = self._history_window(metric_name)
        if len(hist) < self.percentile_min_history:
            return None
        return float(np.mean(hist <= float(value)))

    def _preview_percentile_metrics(self, context, now: pd.Timestamp) -> dict[str, float | None] | None:
        chain = context.option_chain
        if chain.empty:
            return None
        chain_df = self._filter_target_expiry(chain, now)
        if chain_df.empty:
            return None

        spot = float(context.underlying_price)
        short_call = self._select_leg(chain_df, spot, now, "call", self.target_call_delta)
        short_put = self._select_leg(chain_df, spot, now, "put", self.target_put_delta)
        if short_call is None or short_put is None:
            return None

        realized_vol = None
        if self.entry_realized_vol_lookback_hours > 1:
            realized_vol = self._recent_realized_vol(context, now, self.entry_realized_vol_lookback_hours)

        abs_move_24h = None
        if self.percentile_abs_move_lookback_hours > 0:
            abs_move_24h = self._recent_abs_move_pct(context, now, self.percentile_abs_move_lookback_hours)

        short_call_iv = self._normalize_iv(short_call.get("mark_iv", self.default_iv))
        short_put_iv = self._normalize_iv(short_put.get("mark_iv", self.default_iv))
        short_call_premium = self._quote_reference_price(short_call)
        short_put_premium = self._quote_reference_price(short_put)

        return {
            "rv_24h": None if realized_vol is None else float(realized_vol),
            "avg_short_iv": float((short_call_iv + short_put_iv) / 2.0),
            "combined_short_premium": float(short_call_premium + short_put_premium),
            "abs_move_24h": None if abs_move_24h is None else float(abs_move_24h),
        }

    def _passes_entry_filters(self, context, now: pd.Timestamp) -> bool:
        if not super()._passes_entry_filters(context, now):
            return False

        metrics = self._preview_percentile_metrics(context, now)
        if metrics is None:
            return False

        checks = [
            ("rv_24h", self.entry_rv_percentile_max),
            ("avg_short_iv", self.entry_iv_percentile_max),
            ("combined_short_premium", self.entry_premium_percentile_max),
            ("abs_move_24h", self.entry_abs_move_percentile_max),
        ]
        reasons: list[str] = []
        percentile_values: dict[str, float | None] = {}
        for metric_name, threshold in checks:
            pct = self._metric_percentile(metric_name, metrics.get(metric_name))
            percentile_values[f"{metric_name}_pctile"] = pct
            if threshold > 0 and pct is not None and pct > threshold:
                reasons.append(f"{metric_name} pctile {pct:.3f} > {threshold:.3f}")

        allow_entry = len(reasons) == 0
        if not allow_entry:
            self.log(f"Skip entry: percentile filter | {'; '.join(reasons)}")
            self._percentile_filter_skips.append(
                {
                    "entry_time": pd.Timestamp(now),
                    **metrics,
                    **percentile_values,
                    "reasons": reasons,
                }
            )

        self._append_percentile_metrics(metrics)
        return allow_entry


def _parse_float_grid(value: str) -> list[float]:
    return [float(v.strip()) for v in str(value).split(",") if str(v).strip()]


def _parse_int_grid(value: str) -> list[int]:
    return [int(v.strip()) for v in str(value).split(",") if str(v).strip()]


def _default_workers() -> int:
    # This workload is memory-bound, not CPU-bound.
    # Each worker process may hold a full copy of the large hourly option store,
    # so a conservative default is much safer than auto-scaling by core count.
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-optimize WeekendVol percentile-based entry filters to improve drawdown.",
    )
    parser.add_argument("--config", default="configs/trader/weekend_vol_btc.yaml")
    parser.add_argument("--start", default="2023-04-16")
    parser.add_argument("--end", default="2026-04-16")
    parser.add_argument("--output-dir", default="reports/optimizations/weekend_vol_percentile_filters")
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--rv-grid", default=DEFAULT_RV_GRID)
    parser.add_argument("--iv-grid", default=DEFAULT_IV_GRID)
    parser.add_argument("--premium-grid", default=DEFAULT_PREMIUM_GRID)
    parser.add_argument("--abs-move-grid", default=DEFAULT_ABS_MOVE_GRID)
    parser.add_argument("--lookback-grid", default=DEFAULT_LOOKBACK_GRID)
    parser.add_argument("--min-history-grid", default=DEFAULT_MIN_HISTORY_GRID)
    parser.add_argument("--abs-move-lookback-hours", type=int, default=DEFAULT_ABS_MOVE_LOOKBACK_HOURS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV/JSON results if present.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests.")
    return parser


def _combo_key(combo: tuple[float, float, float, float, int, int], abs_move_lookback_hours: int) -> str:
    rv_pct_max, iv_pct_max, premium_pct_max, abs_move_pct_max, lookback_entries, min_history = combo
    return json.dumps(
        {
            "rv_pct_max": float(rv_pct_max),
            "iv_pct_max": float(iv_pct_max),
            "premium_pct_max": float(premium_pct_max),
            "abs_move_pct_max": float(abs_move_pct_max),
            "lookback_entries": int(lookback_entries),
            "min_history": int(min_history),
            "abs_move_lookback_hours": int(abs_move_lookback_hours),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _run_one(
    config_path: str,
    start: str,
    end: str,
    abs_move_lookback_hours: int,
    combo: tuple[float, float, float, float, int, int],
) -> dict[str, Any]:
    rv_pct_max, iv_pct_max, premium_pct_max, abs_move_pct_max, lookback_entries, min_history = combo

    loaded = load_run_config(Path(config_path))
    cfg = copy.deepcopy(loaded.cfg)
    coverage = detect_hourly_coverage(str(cfg.backtest.underlying or "BTC").upper())
    start_date, end_date = _choose_date_window(coverage, start, end)

    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.backtest.show_progress = False
    cfg.report.generate_plots = False

    cfg.strategy.params = dict(cfg.strategy.params or {})
    cfg.strategy.params.update(
        {
            "entry_rv_percentile_max": float(rv_pct_max),
            "entry_iv_percentile_max": float(iv_pct_max),
            "entry_premium_percentile_max": float(premium_pct_max),
            "entry_abs_move_percentile_max": float(abs_move_pct_max),
            "percentile_lookback_entries": int(lookback_entries),
            "percentile_min_history": int(min_history),
            "percentile_abs_move_lookback_hours": int(abs_move_lookback_hours),
        }
    )

    _apply_initial_usd_balance(cfg, loaded.source_type, start_date, end_date)

    t0 = time.perf_counter()
    strategy = PercentileFilterWeekendVolStrategy(cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(results)
    diagnostics = results.get("strategy_diagnostics") or {}
    skip_rows = diagnostics.get("percentile_filter_skips") or []

    row = {
        "rv_pct_max": float(rv_pct_max),
        "iv_pct_max": float(iv_pct_max),
        "premium_pct_max": float(premium_pct_max),
        "abs_move_pct_max": float(abs_move_pct_max),
        "lookback_entries": int(lookback_entries),
        "min_history": int(min_history),
        "abs_move_lookback_hours": int(abs_move_lookback_hours),
        "final_equity": float(metrics.get("final_equity", 0.0)),
        "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "annualized_return_pct": float(metrics.get("annualized_return", 0.0)) * 100.0,
        "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)) * 100.0,
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
        "total_trades": int(metrics.get("total_trades", 0)),
        "total_fees": float(metrics.get("total_fees", 0.0)),
        "skip_count": int(len(skip_rows)),
        "elapsed_sec": float(elapsed),
    }
    return row


def _sort_for_drawdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r["max_drawdown_pct"]),
            float(r["annualized_return_pct"]),
            float(r["sharpe"]),
            float(r["profit_factor"]),
        ),
        reverse=True,
    )


def _save_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    cpu_count: int,
    workers: int,
    combos: list[tuple[float, float, float, float, int, int]],
    args,
    elapsed: float,
) -> None:
    rows_sorted = _sort_for_drawdown(rows)
    df = pd.DataFrame(rows_sorted)
    csv_path = output_dir / "percentile_filter_grid.csv"
    json_path = output_dir / "percentile_filter_grid.json"
    top_path = output_dir / "percentile_filter_grid_top10.json"
    meta_path = output_dir / "percentile_filter_grid_meta.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(_top_summary(rows_sorted, top_n=10), f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cpu_logical_cores": cpu_count,
                "workers": workers,
                "combo_count": len(combos),
                "elapsed_sec": elapsed,
                "config": args.config,
                "start": args.start,
                "end": args.end,
                "rv_grid": _parse_float_grid(args.rv_grid),
                "iv_grid": _parse_float_grid(args.iv_grid),
                "premium_grid": _parse_float_grid(args.premium_grid),
                "abs_move_grid": _parse_float_grid(args.abs_move_grid),
                "lookback_grid": _parse_int_grid(args.lookback_grid),
                "min_history_grid": _parse_int_grid(args.min_history_grid),
                "abs_move_lookback_hours": int(args.abs_move_lookback_hours),
                "batch_size": int(args.batch_size),
                "resume": bool(args.resume),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    csv_path = output_dir / "percentile_filter_grid.csv"
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if df.empty:
        return []
    records = df.to_dict("records")
    return [{str(k): v for k, v in row.items()} for row in records]


def _top_summary(rows: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    sorted_rows = _sort_for_drawdown(rows)
    return sorted_rows[:top_n]


def _print_top(title: str, rows: list[dict[str, Any]], top_n: int = 10) -> None:
    top = _top_summary(rows, top_n=top_n)
    print("\n" + "=" * 140)
    print(f"{title}")
    print("=" * 140)
    print(
        f"{'Rank':>4} {'RV%':>6} {'IV%':>6} {'Prem%':>6} {'Move%':>6} {'Look':>5} {'Hist':>5} "
        f"{'Ret%':>10} {'Ann%':>10} {'DD%':>9} {'Sharpe':>8} {'PF':>7} {'Trades':>7} {'Skips':>7} {'Sec':>7}"
    )
    print("-" * 140)
    for idx, row in enumerate(top, start=1):
        print(
            f"{idx:>4} {row['rv_pct_max']:>6.2f} {row['iv_pct_max']:>6.2f} {row['premium_pct_max']:>6.2f} "
            f"{row['abs_move_pct_max']:>6.2f} {row['lookback_entries']:>5} {row['min_history']:>5} "
            f"{row['total_return_pct']:>10.2f} {row['annualized_return_pct']:>10.2f} {row['max_drawdown_pct']:>9.2f} "
            f"{row['sharpe']:>8.2f} {row['profit_factor']:>7.2f} {row['total_trades']:>7} {row['skip_count']:>7} {row['elapsed_sec']:>7.2f}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rv_grid = _parse_float_grid(args.rv_grid)
    iv_grid = _parse_float_grid(args.iv_grid)
    premium_grid = _parse_float_grid(args.premium_grid)
    abs_move_grid = _parse_float_grid(args.abs_move_grid)
    lookback_grid = _parse_int_grid(args.lookback_grid)
    min_history_grid = _parse_int_grid(args.min_history_grid)

    combos = list(itertools.product(rv_grid, iv_grid, premium_grid, abs_move_grid, lookback_grid, min_history_grid))
    if args.limit > 0:
        combos = combos[: args.limit]

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(args.workers), len(combos))) if combos else 1
    if workers > 1:
        print(
            "[warn] This optimizer is memory-heavy: each worker may load a full copy of the hourly option store. "
            "If memory usage spikes, rerun with --workers 1."
        )
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = _load_existing_rows(output_dir) if args.resume else []
    done_keys = {
        _combo_key(
            (
                float(row["rv_pct_max"]),
                float(row["iv_pct_max"]),
                float(row["premium_pct_max"]),
                float(row["abs_move_pct_max"]),
                int(row["lookback_entries"]),
                int(row["min_history"]),
            ),
            int(row.get("abs_move_lookback_hours", args.abs_move_lookback_hours)),
        )
        for row in rows
    }
    pending_combos = [
        combo for combo in combos if _combo_key(combo, int(args.abs_move_lookback_hours)) not in done_keys
    ]

    print(f"CPU logical cores: {cpu_count}")
    print(f"Workers: {workers}")
    print(f"Combos: {len(combos)}")
    print(f"Pending combos: {len(pending_combos)}")
    print(f"RV grid: {rv_grid}")
    print(f"IV grid: {iv_grid}")
    print(f"Premium grid: {premium_grid}")
    print(f"Abs move grid: {abs_move_grid}")
    print(f"Lookback grid: {lookback_grid}")
    print(f"Min history grid: {min_history_grid}")
    started = time.perf_counter()

    config_path = str((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    completed = len(rows)
    save_every = max(1, int(args.batch_size))

    if workers == 1:
        print("[mode] single-process serial mode (memory-safe)")
        for combo in pending_combos:
            row = _run_one(
                config_path,
                args.start,
                args.end,
                int(args.abs_move_lookback_hours),
                combo,
            )
            rows.append(row)
            completed += 1
            print(
                f"[{completed:>3}/{len(combos)}] "
                f"rv={row['rv_pct_max']:.2f} iv={row['iv_pct_max']:.2f} prem={row['premium_pct_max']:.2f} move={row['abs_move_pct_max']:.2f} "
                f"look={row['lookback_entries']} hist={row['min_history']} | "
                f"ret={row['total_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% sharpe={row['sharpe']:.2f} skips={row['skip_count']}"
            )
            if completed % save_every == 0 or completed == len(combos):
                elapsed = time.perf_counter() - started
                _save_outputs(rows, output_dir, cpu_count, workers, combos, args, elapsed)
                gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    config_path,
                    args.start,
                    args.end,
                    int(args.abs_move_lookback_hours),
                    combo,
                ): combo
                for combo in pending_combos
            }

            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[{completed:>3}/{len(combos)}] "
                    f"rv={row['rv_pct_max']:.2f} iv={row['iv_pct_max']:.2f} prem={row['premium_pct_max']:.2f} move={row['abs_move_pct_max']:.2f} "
                    f"look={row['lookback_entries']} hist={row['min_history']} | "
                    f"ret={row['total_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% sharpe={row['sharpe']:.2f} skips={row['skip_count']}"
                )
                if completed % save_every == 0 or completed == len(combos):
                    elapsed = time.perf_counter() - started
                    _save_outputs(rows, output_dir, cpu_count, workers, combos, args, elapsed)

    elapsed = time.perf_counter() - started
    _save_outputs(rows, output_dir, cpu_count, workers, combos, args, elapsed)

    rows_sorted = _sort_for_drawdown(rows)
    _print_top("Top 10 by smallest drawdown (then higher return/sharpe)", rows_sorted, top_n=10)
    print(f"\nSaved CSV: {output_dir / 'percentile_filter_grid.csv'}")
    print(f"Saved JSON: {output_dir / 'percentile_filter_grid.json'}")
    print(f"Saved Top10: {output_dir / 'percentile_filter_grid_top10.json'}")
    print(f"Saved Meta: {output_dir / 'percentile_filter_grid_meta.json'}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

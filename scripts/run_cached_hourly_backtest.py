"""Run a cached-hourly backtest over the real parquet coverage and save reports.

Purpose
-------
This script exists so the exact workflow does not get forgotten again.
It always follows the same method:

1. Read the backtest YAML.
2. Detect the actual coverage of `data/options_hourly/<UNDERLYING>/*.parquet`.
3. Use the real cache coverage as the backtest range by default.
4. Run the strategy backtest directly against the hourly cache.
5. Export trades CSV, summary JSON, standard plots, and PnL histograms.

Default usage
-------------
    python scripts/run_cached_hourly_backtest.py

Explicit config
---------------
    python scripts/run_cached_hourly_backtest.py \
        --config configs/backtest/weekend_vol_btc_hourly.yaml

Current trader config
---------------------
    python scripts/run_cached_hourly_backtest.py \
        --config configs/trader/weekend_vol_btc.yaml

Optional manual window
----------------------
    python scripts/run_cached_hourly_backtest.py \
        --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.analytics.plotting import generate_all_plots
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from trader.config import load_config as load_trader_config


DEFAULT_TRADER_INITIAL_USD = 10_000.0


@dataclass(frozen=True)
class CoverageWindow:
    underlying: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    file_count: int

    @property
    def start_date(self) -> str:
        return self.start_ts.strftime("%Y-%m-%d")

    @property
    def end_date(self) -> str:
        return self.end_ts.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class LoadedRunConfig:
    cfg: Config
    source_type: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached-hourly backtest and generate reusable PnL reports.",
    )
    parser.add_argument(
        "--config",
        default="configs/backtest/weekend_vol_btc_hourly.yaml",
        help="Backtest YAML config path.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Optional manual start date YYYY-MM-DD. Default: cache coverage start.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Optional manual end date YYYY-MM-DD. Default: cache coverage end.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional override for report output directory.",
    )
    return parser.parse_args()


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def _looks_like_backtest_config(raw: dict[str, Any]) -> bool:
    return "backtest" in raw or "account" in raw or "report" in raw


def _map_trader_config_to_backtest(trader_config_path: Path) -> Config:
    trader_cfg = load_trader_config(trader_config_path)
    if str(trader_cfg.strategy.mode).lower() != "weekend_vol":
        raise ValueError(
            f"Trader config strategy.mode must be 'weekend_vol', got {trader_cfg.strategy.mode!r}"
        )

    cfg = Config()
    cfg.backtest.name = trader_cfg.name or "WeekendVol from trader config"
    cfg.backtest.start_date = "2023-01-01"
    cfg.backtest.end_date = "2023-12-31"
    cfg.backtest.time_step = "1h"
    cfg.backtest.underlying = str(trader_cfg.strategy.underlying or "BTC").upper()
    cfg.backtest.margin_mode = "USD"
    cfg.backtest.use_bs_only = False
    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.option_snapshot_pick = "close"
    cfg.backtest.iv_mode = "fixed"
    cfg.backtest.fixed_iv = float(trader_cfg.strategy.default_iv)
    cfg.backtest.show_progress = False

    cfg.account.initial_balance = 1.0

    cfg.execution.slippage = 0.0001
    cfg.execution.taker_fee = 0.00024
    cfg.execution.maker_fee = 0.00024
    cfg.execution.min_fee = 0.00024
    cfg.execution.max_fee_pct = 0.10
    cfg.execution.delivery_fee = 0.00015
    cfg.execution.delivery_fee_max_pct = 0.10

    cfg.strategy.name = "WeekendVol"
    cfg.strategy.params = {
        "target_delta": float(trader_cfg.strategy.target_delta),
        "wing_delta": float(trader_cfg.strategy.wing_delta),
        "default_iv": float(trader_cfg.strategy.default_iv),
        "entry_day": str(trader_cfg.strategy.entry_day).lower(),
        "entry_time_utc": str(trader_cfg.strategy.entry_time_utc),
        "close_day": "sunday",
        "close_time_utc": "08:00",
        "expire_day": "sunday",
        "expiry_selection": "exact",
        "leverage": float(trader_cfg.strategy.leverage),
        "quantity": float(trader_cfg.strategy.quantity),
        "quantity_step": float(trader_cfg.strategy.quantity),
        "max_positions": int(trader_cfg.strategy.max_positions),
        "compound": bool(trader_cfg.strategy.compound),
        "max_delta_diff": float(trader_cfg.strategy.max_delta_diff),
        "expiry_tolerance_hours": 2.0,
        "entry_realized_vol_lookback_hours": int(trader_cfg.strategy.entry_realized_vol_lookback_hours),
        "entry_realized_vol_max": float(trader_cfg.strategy.entry_realized_vol_max),
        "stop_loss_pct": float(trader_cfg.strategy.stop_loss_pct),
    }

    cfg.report.output_dir = f"./reports/{trader_config_path.stem}_hourly_backtest"
    cfg.report.generate_plots = False
    return cfg


def load_run_config(config_path: Path) -> LoadedRunConfig:
    raw = _read_yaml_dict(config_path)
    if _looks_like_backtest_config(raw):
        return LoadedRunConfig(cfg=Config.from_yaml(config_path), source_type="backtest")
    return LoadedRunConfig(cfg=_map_trader_config_to_backtest(config_path), source_type="trader")


def _to_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    if not isinstance(value, (str, pd.Timestamp)) and hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (int, float)):
        abs_value = abs(value)
        if abs_value >= 10**17:
            ts = pd.to_datetime(value, unit="ns", utc=True)
        elif abs_value >= 10**14:
            ts = pd.to_datetime(value, unit="us", utc=True)
        elif abs_value >= 10**11:
            ts = pd.to_datetime(value, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(value, unit="s", utc=True)
        return pd.Timestamp(ts)

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def detect_hourly_coverage(underlying: str, hourly_root: Path | None = None) -> CoverageWindow:
    base = hourly_root or (REPO_ROOT / "data" / "options_hourly" / underlying.upper())
    parquet_files = sorted(base.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No hourly cache parquet files found in: {base}")

    first_df = pd.read_parquet(parquet_files[0], columns=["timestamp"])
    last_df = pd.read_parquet(parquet_files[-1], columns=["timestamp"])
    start_ts = _to_timestamp(first_df["timestamp"].min())
    end_ts = _to_timestamp(last_df["timestamp"].max())
    return CoverageWindow(
        underlying=underlying.upper(),
        start_ts=start_ts,
        end_ts=end_ts,
        file_count=len(parquet_files),
    )


def _choose_date_window(coverage: CoverageWindow, start: str, end: str) -> tuple[str, str]:
    start_ts = _to_timestamp(start) if start else coverage.start_ts
    end_ts = _to_timestamp(end) if end else coverage.end_ts

    if start_ts < coverage.start_ts:
        start_ts = coverage.start_ts
    if end_ts > coverage.end_ts:
        end_ts = coverage.end_ts
    if start_ts > end_ts:
        raise ValueError(
            f"Invalid range after coverage clipping: {start_ts.date()} > {end_ts.date()}"
        )

    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def _apply_initial_usd_balance(
    cfg: Config,
    source_type: str,
    start_date: str,
    end_date: str,
) -> float | None:
    initial_usd = None
    if source_type == "trader":
        initial_usd = DEFAULT_TRADER_INITIAL_USD
    else:
        params = getattr(cfg.strategy, "params", {}) or {}
        explicit_initial_usd = params.get("backtest_initial_usd")
        if explicit_initial_usd is not None:
            initial_usd = float(explicit_initial_usd)

    if initial_usd is None or initial_usd <= 0:
        return None
    if str(cfg.backtest.margin_mode or "").upper() != "USD":
        return None

    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.backtest.start_date = start_date
    probe_cfg.backtest.end_date = end_date
    probe_cfg.backtest.show_progress = False
    probe_strategy = _load_strategy(probe_cfg.strategy.name, probe_cfg.strategy.params)
    probe_engine = BacktestEngine(probe_cfg, probe_strategy)
    probe_engine._load_data(
        probe_cfg.backtest.underlying,
        start_date,
        end_date,
        probe_cfg.backtest.time_step,
    )
    underlying_df = getattr(probe_engine, "_underlying_df", pd.DataFrame())
    if underlying_df.empty:
        return None

    first_price = float(underlying_df["close"].iloc[0])
    if first_price <= 0:
        return None

    cfg.account.initial_balance = initial_usd / first_price
    return first_price


def _prepare_equity_history_df(results: dict[str, Any]) -> pd.DataFrame:
    equity_history = results.get("equity_history", []) or []
    if not equity_history:
        return pd.DataFrame()

    sample = equity_history[0]
    if len(sample) >= 5:
        eq_df = pd.DataFrame(
            equity_history,
            columns=["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"],
        )
    else:
        eq_df = pd.DataFrame(
            equity_history,
            columns=["timestamp", "equity", "balance", "unrealized_pnl"],
        )
        eq_df["underlying_price"] = 0.0

    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True, errors="coerce")
    eq_df = eq_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return eq_df


def _prepare_trades_df(results: dict[str, Any]) -> pd.DataFrame:
    trades = results.get("closed_trades", []) or []
    df = pd.DataFrame(trades)
    if df.empty:
        return df

    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "exit_time" in df.columns:
        df["exit_year"] = df["exit_time"].dt.year
    elif "entry_time" in df.columns:
        df["exit_year"] = df["entry_time"].dt.year
    else:
        df["exit_year"] = pd.Series(dtype="Int64")

    base_balance = float(results.get("initial_balance") or 0.0)
    eq_df = _prepare_equity_history_df(results)
    df["balance_before_entry"] = base_balance
    df["equity_before_entry"] = base_balance
    df["equity_snapshot_time"] = pd.NaT

    if not eq_df.empty and "entry_time" in df.columns:
        sorted_df = df.reset_index(names="_orig_index").sort_values("entry_time")
        merged = pd.merge_asof(
            sorted_df,
            eq_df[["timestamp", "balance", "equity"]],
            left_on="entry_time",
            right_on="timestamp",
            direction="backward",
            allow_exact_matches=False,
        )
        df = merged.sort_values("_orig_index").drop(columns=["_orig_index"])
        df["balance_before_entry"] = df["balance"].fillna(base_balance).astype(float)
        df["equity_before_entry"] = df["equity"].fillna(base_balance).astype(float)
        df["equity_snapshot_time"] = df["timestamp"]
        df = df.drop(columns=["balance", "equity", "timestamp"], errors="ignore")

    current_balance = df["balance_before_entry"].replace(0.0, pd.NA)
    if "pnl" in df.columns:
        df["pnl_pct_current_balance"] = (
            df["pnl"].astype(float).div(current_balance).fillna(0.0) * 100.0
        )
    else:
        df["pnl_pct_current_balance"] = 0.0

    exit_diag = pd.DataFrame(
        ((results.get("strategy_diagnostics") or {}).get("weekend_vol_exits") or [])
    )
    if not exit_diag.empty and "entry_time" in df.columns and "entry_time" in exit_diag.columns:
        for col in ("entry_time", "trigger_time", "exit_time"):
            if col in exit_diag.columns:
                exit_diag[col] = pd.to_datetime(exit_diag[col], utc=True, errors="coerce")
        keep_cols = [
            col
            for col in (
                "entry_time",
                "reason",
                "trigger_time",
                "exit_time",
                "basket_pnl_pct",
                "stop_loss_pct",
            )
            if col in exit_diag.columns
        ]
        exit_diag = exit_diag.loc[:, keep_cols].drop_duplicates(subset=["entry_time"], keep="last")
        df = df.merge(exit_diag, on="entry_time", how="left")
    return df


def _build_stop_loss_cap_details(results: dict[str, Any]) -> pd.DataFrame:
    trades = pd.DataFrame(results.get("closed_trades", []) or [])
    exits = pd.DataFrame(
        ((results.get("strategy_diagnostics") or {}).get("weekend_vol_exits") or [])
    )
    if trades.empty or exits.empty:
        return pd.DataFrame()

    for col in ("entry_time", "exit_time"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")
        if col in exits.columns:
            exits[col] = pd.to_datetime(exits[col], utc=True, errors="coerce")
    if "trigger_time" in exits.columns:
        exits["trigger_time"] = pd.to_datetime(exits["trigger_time"], utc=True, errors="coerce")

    for col in ("entry_price", "pnl", "quantity"):
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    if "basket_pnl_pct" in exits.columns:
        exits["basket_pnl_pct"] = pd.to_numeric(exits["basket_pnl_pct"], errors="coerce")
    if "stop_loss_pct" in exits.columns:
        exits["stop_loss_pct"] = pd.to_numeric(exits["stop_loss_pct"], errors="coerce")

    trades = trades.dropna(subset=["entry_time"]).copy()
    exits = exits.dropna(subset=["entry_time"]).copy()
    if trades.empty or exits.empty:
        return pd.DataFrame()

    signs = np.where(trades["direction"].astype(str).str.lower() == "short", 1.0, -1.0)
    trades["net_entry_credit_component"] = trades["entry_price"].astype(float) * trades["quantity"].astype(float) * signs
    grouped = (
        trades.groupby("entry_time", as_index=False)
        .agg(
            observed_group_pnl=("pnl", "sum"),
            net_entry_credit=("net_entry_credit_component", "sum"),
            observed_exit_time=("exit_time", "max"),
            leg_count=("instrument_name", "count"),
        )
    )
    if "reason" not in exits.columns:
        return pd.DataFrame()
    exits = exits.loc[exits["reason"].astype(str) == "stop_loss"].copy()
    if exits.empty:
        return pd.DataFrame()

    details = exits.merge(grouped, on="entry_time", how="inner")
    if details.empty:
        return pd.DataFrame()

    details["capped_group_pnl"] = -details["net_entry_credit"] * details["stop_loss_pct"] / 100.0
    details["pnl_adjustment"] = details["capped_group_pnl"] - details["observed_group_pnl"]
    details = details.loc[
        (details["net_entry_credit"] > 0)
        & np.isfinite(details["basket_pnl_pct"])
        & np.isfinite(details["stop_loss_pct"])
        & (details["basket_pnl_pct"] < -details["stop_loss_pct"])
        & (details["pnl_adjustment"] > 1e-9)
    ].copy()
    return details.sort_values("entry_time").reset_index(drop=True)


def _apply_stop_loss_caps(results: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    cap_details = _build_stop_loss_cap_details(results)
    adjusted_results = copy.deepcopy(results)
    if cap_details.empty:
        return adjusted_results, cap_details

    trades = pd.DataFrame(adjusted_results.get("closed_trades", []) or [])
    if trades.empty:
        return adjusted_results, cap_details
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
    trades["observed_pnl"] = trades["pnl"].astype(float)
    trades["stop_loss_capped"] = False

    for row in cap_details.itertuples(index=False):
        mask = trades["entry_time"] == row.entry_time
        if not mask.any():
            continue
        loss_mask = mask & (trades["pnl"] < 0)
        target_adjustment = float(getattr(row, "pnl_adjustment"))
        alloc_mask = loss_mask if loss_mask.any() else mask
        alloc_base = trades.loc[alloc_mask, "pnl"].abs().astype(float)
        alloc_sum = float(alloc_base.sum())
        if alloc_sum <= 0:
            alloc_base = pd.Series(1.0, index=trades.index[alloc_mask])
            alloc_sum = float(alloc_base.sum())
        alloc = alloc_base / alloc_sum * target_adjustment
        trades.loc[alloc.index, "pnl"] = trades.loc[alloc.index, "pnl"].astype(float) + alloc.to_numpy()
        trades.loc[mask, "stop_loss_capped"] = True

    adjusted_results["closed_trades"] = trades.drop(columns=["observed_pnl", "stop_loss_capped"]).to_dict("records")

    eq_df = _prepare_equity_history_df(adjusted_results)
    if not eq_df.empty:
        adj_events = (
            cap_details.groupby("observed_exit_time", as_index=False)
            .agg(equity_adjustment=("pnl_adjustment", "sum"))
            .rename(columns={"observed_exit_time": "timestamp"})
        )
        eq_df = eq_df.merge(adj_events, on="timestamp", how="left")
        eq_df["equity_adjustment"] = eq_df["equity_adjustment"].fillna(0.0)
        eq_df["cumulative_adjustment"] = eq_df["equity_adjustment"].cumsum()
        eq_df["equity"] = eq_df["equity"].astype(float) + eq_df["cumulative_adjustment"]
        eq_df["balance"] = eq_df["balance"].astype(float) + eq_df["cumulative_adjustment"]
        adjusted_results["equity_history"] = []
        for timestamp, equity, balance, unrealized_pnl, underlying_price in eq_df[
            ["timestamp", "equity", "balance", "unrealized_pnl", "underlying_price"]
        ].itertuples(index=False, name=None):
            adjusted_results["equity_history"].append(
                [
                    timestamp,
                    float(equity),
                    float(balance),
                    float(unrealized_pnl),
                    float(underlying_price),
                ]
            )

    diagnostics = copy.deepcopy(adjusted_results.get("strategy_diagnostics") or {})
    diagnostics["stop_loss_cap_details"] = cap_details.to_dict("records")
    adjusted_results["strategy_diagnostics"] = diagnostics
    return adjusted_results, cap_details


def _write_dataset_outputs(
    dataset_name: str,
    dataset_label: str,
    cfg_path: Path,
    cfg: Config,
    source_type: str,
    coverage: CoverageWindow,
    start_date: str,
    end_date: str,
    results: dict[str, Any],
    output_dir: Path,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(results)
    plot_paths = generate_all_plots(results, str(output_dir))
    trades_df = _prepare_trades_df(results)
    trades_df.attrs["initial_balance"] = float(results.get("initial_balance") or 0.0)
    dated_stem = f"{start_date}_{end_date}"

    csv_path = output_dir / f"trades_{dated_stem}.csv"
    trades_df.to_csv(csv_path, index=False)
    full_plot_paths = list(plot_paths)
    if not trades_df.empty:
        full_plot_paths.append(
            plot_pnl_histogram(
                trades_df,
                output_dir / f"pnl_histogram_{dated_stem}.html",
                f"{dataset_label}: PnL Histogram as % of Current Balance Before Entry ({cfg.backtest.underlying}, {start_date} to {end_date})",
            )
        )
        full_plot_paths.append(
            plot_profit_loss_histogram(
                trades_df,
                output_dir / f"pnl_histogram_profit_loss_{dated_stem}.html",
                f"{dataset_label}: PnL % of Current Balance Before Entry ({cfg.backtest.underlying})",
            )
        )
        full_plot_paths.append(
            plot_yearly_histograms(
                trades_df,
                output_dir / f"pnl_histogram_by_year_{dated_stem}.html",
                f"{dataset_label}: PnL % of Current Balance Before Entry by Exit Year ({cfg.backtest.underlying})",
            )
        )

    summary = build_summary(
        cfg_path,
        cfg,
        source_type,
        coverage,
        start_date,
        end_date,
        trades_df,
        metrics,
        full_plot_paths,
    )
    summary["dataset"] = {"name": dataset_name, "label": dataset_label}
    if extra_summary:
        summary["dataset_details"] = extra_summary
    summary_path = output_dir / f"backtest_summary_{dated_stem}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    return {
        "name": dataset_name,
        "label": dataset_label,
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "plot_paths": full_plot_paths,
        "metrics": metrics,
        "trade_count": int(len(trades_df)),
    }


def _add_stat_lines(fig: go.Figure, values: pd.Series, suffix: str = "") -> None:
    if values.empty:
        return

    mean_v = float(values.mean())
    median_v = float(values.median())
    fig.add_vline(x=0, line_dash="dot", line_color="#6c757d")
    fig.add_vline(x=mean_v, line_dash="dash", line_color="#2e7d32")
    fig.add_vline(x=median_v, line_dash="dash", line_color="#1565c0")
    fig.add_annotation(x=mean_v, y=1, yref="paper", text=f"mean={mean_v:,.2f}{suffix}", showarrow=False)
    fig.add_annotation(
        x=median_v,
        y=0.93,
        yref="paper",
        text=f"median={median_v:,.2f}{suffix}",
        showarrow=False,
    )


def plot_pnl_histogram(trades_df: pd.DataFrame, output_path: Path, title: str) -> str:
    fig = go.Figure()
    pnls = trades_df["pnl_pct_current_balance"].astype(float)
    fig.add_trace(
        go.Histogram(
            x=pnls,
            nbinsx=min(60, max(20, int(math.sqrt(len(trades_df)) * 2))),
            marker_color="#4c78a8",
            opacity=0.9,
            name="PnL % of Balance",
        )
    )
    _add_stat_lines(fig, pnls, suffix="%")
    fig.update_layout(
        title=title,
        xaxis_title="PnL (% of current balance before entry)",
        yaxis_title="Trade Count",
        bargap=0.05,
        template="plotly_white",
        height=520,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return str(output_path)


def plot_profit_loss_histogram(trades_df: pd.DataFrame, output_path: Path, title: str) -> str:
    pnls = trades_df["pnl_pct_current_balance"].astype(float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Profits", "Losses"))
    fig.add_trace(
        go.Histogram(
            x=wins,
            nbinsx=min(40, max(10, int(math.sqrt(max(len(wins), 1)) * 2))),
            marker_color="#2e7d32",
            name="Profits",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=losses,
            nbinsx=min(40, max(10, int(math.sqrt(max(len(losses), 1)) * 2))),
            marker_color="#c62828",
            name="Losses",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(title_text="PnL (% of current balance before entry)", row=1, col=1)
    fig.update_xaxes(title_text="PnL (% of current balance before entry)", row=1, col=2)
    fig.update_yaxes(title_text="Trade Count", row=1, col=1)
    fig.update_yaxes(title_text="Trade Count", row=1, col=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return str(output_path)


def plot_yearly_histograms(trades_df: pd.DataFrame, output_path: Path, title: str) -> str:
    years = [int(y) for y in sorted(trades_df["exit_year"].dropna().unique())]
    cols = 2
    rows = max(1, math.ceil(len(years) / cols))
    subplot_titles = [str(y) for y in years] or ["No Trades"]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    if not years:
        fig.update_layout(title=title, template="plotly_white", height=400)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        return str(output_path)

    for idx, year in enumerate(years, start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        year_pnls = trades_df.loc[trades_df["exit_year"] == year, "pnl_pct_current_balance"].astype(float)
        fig.add_trace(
            go.Histogram(
                x=year_pnls,
                nbinsx=min(30, max(10, int(math.sqrt(len(year_pnls)) * 2))),
                marker_color="#7b61ff",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="PnL (% of current balance before entry)", row=row, col=col)
        fig.update_yaxes(title_text="Trades", row=row, col=col)

    fig.update_layout(title=title, template="plotly_white", height=max(500, rows * 320))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return str(output_path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def build_summary(
    cfg_path: Path,
    cfg: Config,
    source_type: str,
    coverage: CoverageWindow,
    start_date: str,
    end_date: str,
    trades_df: pd.DataFrame,
    metrics: dict[str, Any],
    plot_paths: list[str],
) -> dict[str, Any]:
    pnls = trades_df["pnl"].astype(float) if not trades_df.empty else pd.Series(dtype=float)
    pnl_pcts = trades_df["pnl_pct_current_balance"].astype(float) if not trades_df.empty else pd.Series(dtype=float)
    balances = trades_df["balance_before_entry"].astype(float) if not trades_df.empty else pd.Series(dtype=float)
    wins = int((pnls > 0).sum())
    losses = int((pnls < 0).sum())
    return {
        "config": {
            "path": str(cfg_path),
            "source_type": source_type,
            "strategy": cfg.strategy.name,
            "underlying": cfg.backtest.underlying,
            "margin_mode": cfg.backtest.margin_mode,
            "option_data_source": cfg.backtest.option_data_source,
        },
        "coverage": {
            "underlying": coverage.underlying,
            "start": coverage.start_ts,
            "end": coverage.end_ts,
            "file_count": coverage.file_count,
        },
        "run_window": {
            "start": start_date,
            "end": end_date,
        },
        "trade_stats": {
            "count": int(len(trades_df)),
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / len(trades_df)) if len(trades_df) else 0.0,
            "starting_balance": float(results_initial_balance := (trades_df.attrs.get("initial_balance") or 0.0)),
            "avg_balance_before_entry": float(balances.mean()) if len(balances) else 0.0,
            "mean_pnl": float(pnls.mean()) if len(pnls) else 0.0,
            "median_pnl": float(pnls.median()) if len(pnls) else 0.0,
            "min_pnl": float(pnls.min()) if len(pnls) else 0.0,
            "max_pnl": float(pnls.max()) if len(pnls) else 0.0,
            "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
            "mean_pnl_pct_current_balance": float(pnl_pcts.mean()) if len(pnl_pcts) else 0.0,
            "median_pnl_pct_current_balance": float(pnl_pcts.median()) if len(pnl_pcts) else 0.0,
            "min_pnl_pct_current_balance": float(pnl_pcts.min()) if len(pnl_pcts) else 0.0,
            "max_pnl_pct_current_balance": float(pnl_pcts.max()) if len(pnl_pcts) else 0.0,
        },
        "metrics": metrics,
        "plots": plot_paths,
    }


def main() -> None:
    args = _parse_args()
    cfg_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    loaded = load_run_config(cfg_path)
    cfg = loaded.cfg

    underlying = str(cfg.backtest.underlying or "BTC").upper()
    coverage = detect_hourly_coverage(underlying)
    start_date, end_date = _choose_date_window(coverage, args.start, args.end)

    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    initial_spot = _apply_initial_usd_balance(cfg, loaded.source_type, start_date, end_date)
    if args.output_dir:
        cfg.report.output_dir = args.output_dir

    output_dir = (REPO_ROOT / cfg.report.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[coverage] {coverage.underlying}: {coverage.start_ts} -> {coverage.end_ts} ({coverage.file_count} parquet files)")
    print(f"[run] config={cfg_path}")
    print(f"[run] config_type={loaded.source_type}")
    print(f"[run] backtest window={start_date} -> {end_date}")
    if initial_spot is not None:
        initial_usd = float(cfg.account.initial_balance) * float(initial_spot)
        print(
            f"[run] initial_usd={initial_usd:,.2f} "
            f"(coin_equiv={cfg.account.initial_balance:.12f}, first_spot={initial_spot:,.2f})"
        )
    print(f"[run] output_dir={output_dir}")

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    observed_results = engine.run()
    capped_results, cap_details = _apply_stop_loss_caps(observed_results)

    observed_output = _write_dataset_outputs(
        dataset_name="observed_hour",
        dataset_label="Observed hourly stop-loss fills",
        cfg_path=cfg_path,
        cfg=cfg,
        source_type=loaded.source_type,
        coverage=coverage,
        start_date=start_date,
        end_date=end_date,
        results=observed_results,
        output_dir=output_dir / "observed_hour",
        extra_summary={
            "stop_loss_fill_mode": "observed_hour",
            "stop_loss_cap_applied": False,
        },
    )
    capped_output = _write_dataset_outputs(
        dataset_name="cap_at_threshold",
        dataset_label="Stop-loss capped at threshold",
        cfg_path=cfg_path,
        cfg=cfg,
        source_type=loaded.source_type,
        coverage=coverage,
        start_date=start_date,
        end_date=end_date,
        results=capped_results,
        output_dir=output_dir / "cap_at_threshold",
        extra_summary={
            "stop_loss_fill_mode": "cap_at_threshold",
            "stop_loss_cap_applied": True,
            "capped_stop_events": int(len(cap_details)),
            "total_pnl_adjustment": float(cap_details["pnl_adjustment"].sum()) if not cap_details.empty else 0.0,
        },
    )

    dated_stem = f"{start_date}_{end_date}"
    cap_details_path = output_dir / f"stop_loss_cap_details_{dated_stem}.csv"
    if not cap_details.empty:
        cap_details.to_csv(cap_details_path, index=False)

    comparison = {
        "config": str(cfg_path),
        "source_type": loaded.source_type,
        "run_window": {"start": start_date, "end": end_date},
        "datasets": {
            "observed_hour": {
                "summary_path": observed_output["summary_path"],
                "csv_path": observed_output["csv_path"],
                "metrics": observed_output["metrics"],
            },
            "cap_at_threshold": {
                "summary_path": capped_output["summary_path"],
                "csv_path": capped_output["csv_path"],
                "metrics": capped_output["metrics"],
            },
        },
        "stop_loss_cap_details_csv": str(cap_details_path) if not cap_details.empty else "",
        "capped_stop_events": int(len(cap_details)),
        "total_pnl_adjustment": float(cap_details["pnl_adjustment"].sum()) if not cap_details.empty else 0.0,
    }
    comparison_path = output_dir / f"backtest_comparison_{dated_stem}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[done] dataset=observed_hour trades={observed_output['trade_count']}")
    print(f"[done] csv={observed_output['csv_path']}")
    print(f"[done] summary={observed_output['summary_path']}")
    for path in observed_output["plot_paths"]:
        print(f"[done] plot={path}")

    print(f"[done] dataset=cap_at_threshold trades={capped_output['trade_count']}")
    print(f"[done] csv={capped_output['csv_path']}")
    print(f"[done] summary={capped_output['summary_path']}")
    for path in capped_output["plot_paths"]:
        print(f"[done] plot={path}")

    if not cap_details.empty:
        print(f"[done] stop_loss_caps={cap_details_path}")
    print(f"[done] comparison={comparison_path}")


if __name__ == "__main__":
    main()

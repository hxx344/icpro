from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from options_backtest.data.loader import DataLoader
from scripts.optimize_btc_covered_call import _ensure_quote_index, run_covered_call
from scripts.run_cached_hourly_backtest import _choose_date_window, detect_hourly_coverage

SRC = REPO / "tmp" / "scan_onchain_voting_bull_bear.py"
spec_obj = importlib.util.spec_from_file_location("onchain_vote", SRC)
if spec_obj is None or spec_obj.loader is None:
    raise RuntimeError(f"Cannot load module from {SRC}")
spec = spec_obj
mod = importlib.util.module_from_spec(spec)
loader = cast(Any, spec.loader)
loader.exec_module(mod)

OUT_DIR = REPO / "reports" / "optimizations" / "btc_covered_call_independent_7of6_vote_count_delta_layers_scan"
VOTE_MEMBERS = [
    "rsi_14d_gt_50_lt_80",
    "price_gt_sma_30d",
    "roc_30d_gt_-5%",
    "onchain_sopr_proxy_155d_0p95_to_1p5",
    "fear_greed_25_to_80",
    "roc_730d_gt_+0%",
    "onchain_mvrv_1_to_3p5",
]


def build_vote_count(daily: pd.DataFrame) -> pd.DataFrame:
    signals: dict[str, pd.Series] = mod.build_atomic_signals(daily)
    vote_mat = pd.concat([signals[name].astype(int).rename(name) for name in VOTE_MEMBERS], axis=1)
    out = pd.DataFrame({"date": daily["date"], "vote_count": vote_mat.sum(axis=1).astype(int)})
    return out


def run_with_layers(
    name: str,
    daily_votes: pd.DataFrame,
    underlying: pd.DataFrame,
    store: Any,
    d7: float,
    d6: float,
    d45: float,
    d03: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    v = daily_votes.copy()
    vc = v["vote_count"].astype(int)
    v["call_delta_override"] = d03
    v.loc[vc.between(4, 5), "call_delta_override"] = d45
    v.loc[vc == 6, "call_delta_override"] = d6
    v.loc[vc == 7, "call_delta_override"] = d7
    v["signal"] = vc >= 6

    u = underlying.copy()
    u["date"] = pd.to_datetime(u["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    u = u.merge(v[["date", "vote_count", "call_delta_override", "signal"]], on="date", how="left")
    u["vote_count"] = u["vote_count"].ffill().fillna(0).astype(int)
    u["call_delta_override"] = u["call_delta_override"].ffill().fillna(d03).astype(float)
    u["signal"] = u["signal"].ffill().fillna(False).astype(bool)
    close = u["close"].astype(float)
    # votes >= 6: bull/no put; votes <= 5: weak or bear/put enabled.
    u["trend_sma"] = close.where(~u["signal"], close * 0.999)
    u["trend_sma"] = u["trend_sma"].where(u["signal"], close * 1.001)

    metrics, trades, equity = run_covered_call(
        u,
        store,
        expiry_days=7,
        selection_mode="delta",
        otm_pct=0.0,
        call_delta=0.10,
        call_delta_pick_mode="at_least_target",
        protective_put_otm_pct=None,
        protective_put_delta=0.08,
        protective_put_rule="below_sma_or_drawdown",
        call_rule="static",
        call_otm_adjust=0.0,
        drawdown_trigger_pct=0.10,
        initial_btc=1.0,
        initial_capital_usd=None,
        short_call_multiplier=2.0,
        protective_call_multiplier=0.0,
        protective_call_otm_add=None,
        protective_put_multiplier=2.0,
        option_fee_pct=0.0003,
        delivery_fee_pct=0.0003,
    )
    metrics["indicator_or_vote_group"] = name
    metrics["vote_members"] = ",".join(VOTE_MEMBERS)
    metrics["delta_votes_7"] = d7
    metrics["delta_votes_6"] = d6
    metrics["delta_votes_4_5"] = d45
    metrics["delta_votes_0_3"] = d03
    metrics["bull_pct_daily"] = float((daily_votes["vote_count"] >= 6).mean())
    return metrics, trades, equity


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = mod.load_daily_indicators()
    daily_votes = build_vote_count(daily)
    daily_votes.to_csv(OUT_DIR / "daily_vote_count.csv", index=False)

    coverage = detect_hourly_coverage("BTC")
    start, end = _choose_date_window(coverage, "2023-04-25", "2026-04-25")
    loader = DataLoader("data")
    underlying = loader.load_underlying("BTC", resolution="60", start_date=start, end_date=end)
    store = loader.load_hourly_option_store("BTC", start_date=start, end_date=end)
    _ensure_quote_index(store)
    underlying = loader.align_underlying_to_hourly_store(underlying, store, pick="close").copy()
    underlying["timestamp"] = pd.to_datetime(underlying["timestamp"], utc=True)
    close = underlying["close"].astype(float)
    underlying["rolling_peak"] = close.rolling(24 * 30, min_periods=24).max()

    d7_grid = [0.001, 0.005, 0.01, 0.02]
    d6_grid = [0.005, 0.01, 0.02, 0.03, 0.05]
    d45_grid = [0.15, 0.20, 0.25, 0.30, 0.40]
    d03_grid = [0.30, 0.40, 0.48, 0.55]
    jobs = [x for x in itertools.product(d7_grid, d6_grid, d45_grid, d03_grid) if x[0] <= x[1] <= x[2] <= x[3]]

    rows: list[dict[str, Any]] = []
    best_payload = None
    best_sharpe = -1e18
    for i, (d7, d6, d45, d03) in enumerate(jobs, start=1):
        name = f"vote_count_layers_7_{d7:.3f}_6_{d6:.3f}_45_{d45:.2f}_03_{d03:.2f}"
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        metrics, trades, equity = run_with_layers(name, daily_votes, underlying, store, d7, d6, d45, d03)
        rows.append(metrics)
        if metrics.get("sharpe_usd", -1e18) > best_sharpe:
            best_sharpe = metrics.get("sharpe_usd", -1e18)
            best_payload = (name, metrics, trades, equity)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "vote_count_delta_layers_scan_all.csv", index=False)
    df.sort_values(["sharpe_usd", "final_usd"], ascending=[False, False]).to_csv(OUT_DIR / "vote_count_delta_layers_scan_by_sharpe.csv", index=False)
    df.sort_values(["final_usd", "sharpe_usd"], ascending=[False, False]).to_csv(OUT_DIR / "vote_count_delta_layers_scan_by_final.csv", index=False)
    df.sort_values(["max_drawdown_usd", "final_usd"], ascending=[False, False]).to_csv(OUT_DIR / "vote_count_delta_layers_scan_by_drawdown.csv", index=False)

    if best_payload is not None:
        name, metrics, trades, equity = best_payload
        pd.DataFrame(trades).to_csv(OUT_DIR / "best_sharpe_trades.csv", index=False)
        equity.to_csv(OUT_DIR / "best_sharpe_equity.csv", index=False)
        (OUT_DIR / "best_sharpe_meta.json").write_text(json.dumps({"best_indicator": name, "metrics": metrics}, indent=2), encoding="utf-8")

    cols = ["delta_votes_7", "delta_votes_6", "delta_votes_4_5", "delta_votes_0_3", "final_usd", "total_return_usd", "max_drawdown_usd", "sharpe_usd", "short_call_net_pnl_usd", "short_call_payoff_usd", "long_put_net_pnl_usd", "trade_count"]
    print("Vote count distribution")
    print(daily_votes["vote_count"].value_counts().sort_index().to_string())
    print("\nTop by sharpe")
    print(df.sort_values(["sharpe_usd", "final_usd"], ascending=[False, False])[cols].head(25).to_string(index=False))
    print("\nTop by final")
    print(df.sort_values(["final_usd", "sharpe_usd"], ascending=[False, False])[cols].head(25).to_string(index=False))
    print(f"Saved: {OUT_DIR.as_posix()}")


if __name__ == "__main__":
    main()

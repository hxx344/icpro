"""Weekly BTC ATM short strangle with optional hourly delta hedging.

Runs two backtests:
1. baseline weekly ATM short strangle
2. same option strategy with an hourly virtual perp hedge overlay

The hedge is rebalanced every hourly bar to offset current option delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.engine.settlement import check_and_settle
from options_backtest.pricing.black76 import delta as bs_delta

CONFIG_PATH = Path("configs/backtest/weekly_short_monfri_btc_atm.yaml")
HEDGE_FEE_PCT = 0.0003
OUTPUT_PATH = Path("reports/weekly_short_monfri_btc_atm_delta_hedge_summary.json")


def build_engine(name: str) -> tuple[Config, BacktestEngine]:
    cfg = Config.from_yaml(CONFIG_PATH)
    cfg.backtest.name = name
    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    return cfg, BacktestEngine(cfg, strategy)


def portfolio_delta(engine: BacktestEngine, positions: dict, price: float, ts_np) -> float:
    net = 0.0
    for name, pos in positions.items():
        inst = engine._instrument_dict.get(name)
        if inst is None:
            continue
        strike = float(inst.get("strike_price", inst.get("strike", 0.0)) or 0.0)
        opt_raw = str(inst.get("option_type", "call")).lower()
        option_type = "call" if opt_raw.startswith("c") else "put"
        exp_ns = int(inst.get("_expiry_ns", 0) or 0)
        if exp_ns <= 0 or strike <= 0:
            continue

        cur_ns = int(ts_np.astype("int64")) if hasattr(ts_np, "astype") else int(ts_np)
        dte_ns = exp_ns - cur_ns
        if dte_ns <= 0:
            if option_type == "call":
                delta_val = 1.0 if price > strike else 0.0
            else:
                delta_val = -1.0 if price < strike else 0.0
        else:
            t_years = dte_ns / (365.25 * 86400 * 1e9)
            sigma = engine._resolve_proxy_iv(ts_np, price, strike, t_years)
            try:
                delta_val = float(bs_delta(price, strike, t_years, sigma, option_type=option_type, r=0.0))
            except Exception:
                delta_val = 0.5 if option_type == "call" else -0.5

        dir_sign = 1.0 if pos.direction.value == "long" else -1.0
        net += delta_val * float(pos.quantity) * dir_sign
    return net


def run_delta_hedged(engine: BacktestEngine) -> tuple[dict, dict]:
    bcfg = engine.config.backtest
    engine._load_data(bcfg.underlying, bcfg.start_date, bcfg.end_date, bcfg.time_step)

    ts_values = engine._underlying_df["timestamp"].values
    close_values = engine._underlying_df["close"].values.astype(float)
    n_steps = len(ts_values)
    if n_steps == 0:
        raise RuntimeError("No underlying data for delta-hedged backtest")

    if engine._margin_usd:
        first_price = float(close_values[0])
        if first_price > 0:
            usd_balance = engine.account.initial_balance * first_price
            engine.account.initial_balance = usd_balance
            engine.account.balance = usd_balance

    ts0 = pd.Timestamp(ts_values[0])
    initial_ctx = engine._build_context(ts0, float(close_values[0]))
    engine.strategy.initialize(initial_ctx)

    position_mgr = engine.position_mgr
    account = engine.account
    matcher = engine.matcher
    instrument_dict = engine._instrument_dict
    settlements_df = engine._settlements_df
    pending_orders = engine._pending_orders
    ts_pd_all = pd.DatetimeIndex(ts_values).tz_localize("UTC")

    perp_qty = 0.0
    prev_price = float(close_values[0])
    hedge_pnl = 0.0
    hedge_fees = 0.0
    hedge_rebalances = 0

    for i in tqdm(range(n_steps), desc="Delta-Hedged Backtest"):
        ts_np = ts_values[i]
        price = float(close_values[i])
        ts_pd = ts_pd_all[i]

        if position_mgr.positions:
            mark_prices = engine._get_mark_prices_fast(ts_np, price)
            position_mgr.update_marks(mark_prices)
            check_and_settle(
                ts_np,
                position_mgr,
                account,
                matcher,
                instrument_dict,
                settlements_df,
                margin_usd=engine._margin_usd,
                settlement_index=engine._settlement_index,
            )

        ctx = engine._build_context(ts_pd, price)
        pending_orders.clear()
        engine.strategy.on_step(ctx)

        if pending_orders:
            engine._process_orders(ts_np, price, ctx)

        dp = price - prev_price
        if abs(perp_qty) > 1e-12:
            step_pnl = perp_qty * dp
            hedge_pnl += step_pnl
            account.balance += step_pnl

        if position_mgr.positions:
            port_delta = portfolio_delta(engine, position_mgr.positions, price, ts_np)
            target_perp = -port_delta
            trade_size = abs(target_perp - perp_qty)
            if trade_size > 1e-6:
                fee = trade_size * price * HEDGE_FEE_PCT
                account.balance -= fee
                hedge_fees += fee
                hedge_rebalances += 1
                perp_qty = target_perp
        elif abs(perp_qty) > 1e-6:
            fee = abs(perp_qty) * price * HEDGE_FEE_PCT
            account.balance -= fee
            hedge_fees += fee
            hedge_rebalances += 1
            perp_qty = 0.0

        prev_price = price
        account.record_equity(ts_pd, position_mgr.total_unrealized_pnl, price)

    if position_mgr.positions:
        last_ts = ts_pd_all[-1]
        last_price = float(close_values[-1])
        ctx = engine._build_context(last_ts, last_price)
        ctx.close_all()
        engine._process_orders(ts_values[-1], last_price, ctx)
        if abs(perp_qty) > 1e-6:
            fee = abs(perp_qty) * last_price * HEDGE_FEE_PCT
            account.balance -= fee
            hedge_fees += fee
            hedge_rebalances += 1
            perp_qty = 0.0
        account.record_equity(last_ts, position_mgr.total_unrealized_pnl, last_price)

    results = engine._build_results()
    stats = {
        "hedge_fee_pct": HEDGE_FEE_PCT,
        "hedge_rebalances": hedge_rebalances,
        "hedge_pnl": hedge_pnl,
        "hedge_fees": hedge_fees,
        "hedge_net": hedge_pnl - hedge_fees,
    }
    return results, stats


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "final_equity",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "profit_factor",
        "total_trades",
        "total_fees",
        "final_usd",
        "max_drawdown_usd",
        "sharpe_ratio_usd",
    ]
    return {k: metrics.get(k) for k in keys}


def main() -> None:
    print("=" * 72)
    print("Weekly BTC ATM short strangle: baseline vs hourly delta hedge")
    print("=" * 72)

    _, baseline_engine = build_engine("BTC Weekly ATM Short Strangle (baseline)")
    baseline_results = baseline_engine.run()
    baseline_metrics = compute_metrics(baseline_results)

    _, hedged_engine = build_engine("BTC Weekly ATM Short Strangle + hourly delta hedge")
    hedged_results, hedge_stats = run_delta_hedged(hedged_engine)
    hedged_metrics = compute_metrics(hedged_results)

    print(f"Baseline     | Return={baseline_metrics.get('total_return', 0)*100:+.2f}% | DD={baseline_metrics.get('max_drawdown', 0)*100:.2f}% | Sharpe={baseline_metrics.get('sharpe_ratio', 0):.2f} | WR={baseline_metrics.get('win_rate', 0)*100:.1f}% | Trades={baseline_metrics.get('total_trades', 0)}")
    print(f"Delta hedged | Return={hedged_metrics.get('total_return', 0)*100:+.2f}% | DD={hedged_metrics.get('max_drawdown', 0)*100:.2f}% | Sharpe={hedged_metrics.get('sharpe_ratio', 0):.2f} | WR={hedged_metrics.get('win_rate', 0)*100:.1f}% | Trades={hedged_metrics.get('total_trades', 0)}")
    print(f"Hedge stats  | Rebalances={hedge_stats['hedge_rebalances']} | HedgePnL=${hedge_stats['hedge_pnl']:,.2f} | HedgeFees=${hedge_stats['hedge_fees']:,.2f} | Net=${hedge_stats['hedge_net']:,.2f}")

    payload = {
        "config": CONFIG_PATH.as_posix(),
        "baseline": compact_metrics(baseline_metrics),
        "delta_hedged": compact_metrics(hedged_metrics),
        "hedge_stats": hedge_stats,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()

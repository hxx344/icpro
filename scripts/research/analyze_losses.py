"""Analyze per-trade losses for the unhedged Naked Call strategy.

Runs the backtest with buy_protective_call=false, then extracts and
analyses every trade to identify where the biggest losses occurred.

Usage: python scripts/research/analyze_losses.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.cli import _load_strategy


def parse_instrument_fields(name: str):
    """Extract strike, expiry, option_type from instrument name like ETH-28MAR25-3200-C."""
    m = re.match(r"(\w+)-(\d+\w+\d+)-(\d+)-([CP])", name, re.IGNORECASE)
    if m:
        return {
            "underlying": m.group(1),
            "expiry_str": m.group(2),
            "strike": float(m.group(3)),
            "option_type": m.group(4).upper(),
        }
    return {}


def main():
    # --- Run unhedged backtest ---
    cfg_path = Path("configs/backtest/naked_call_usd.yaml")
    cfg = Config.from_yaml(cfg_path)

    # Force unhedged
    cfg.strategy.params["buy_protective_call"] = False

    strategy = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()

    closed_trades = results["closed_trades"]
    equity_history = results["equity_history"]
    coin = cfg.backtest.underlying  # "ETH" or "BTC"

    # --- Build underlying price series ---
    # equity_history items: (timestamp, equity, unrealized_pnl, ?, underlying_price)
    ul_times = []
    ul_prices = []
    for eh in equity_history:
        ts = eh[0]
        if len(eh) > 4 and eh[4] > 0:
            ul_times.append(pd.Timestamp(ts))
            ul_prices.append(eh[4])

    ul_series = pd.Series(ul_prices, index=pd.DatetimeIndex(ul_times))

    def _closest_price(ts):
        """Find the closest underlying price at or before ts."""
        try:
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            mask = ul_series.index <= ts
            if mask.any():
                return float(ul_series.loc[mask].iloc[-1])
        except Exception:
            pass
        return None

    # --- Build per-trade DataFrame ---
    rows = []
    for t in closed_trades:
        info = parse_instrument_fields(t["instrument_name"])
        entry_time = t["entry_time"]
        exit_time = t["exit_time"]

        entry_ul = _closest_price(entry_time)
        exit_ul = _closest_price(exit_time)

        row = {
            "instrument": t["instrument_name"],
            "direction": t["direction"],
            "quantity": t["quantity"],
            "entry_price_coin": t["entry_price"],
            "exit_price_coin": t["exit_price"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl_coin": t["pnl"],
            "fee_coin": t["fee"],
            "close_type": t["close_type"],
            "strike": info.get("strike"),
            "option_type": info.get("option_type"),
            "expiry_str": info.get("expiry_str"),
            "underlying_at_entry": entry_ul,
            "underlying_at_exit": exit_ul,
        }

        # Compute distance from strike at entry
        if entry_ul and info.get("strike"):
            row["strike_vs_entry_pct"] = (info["strike"] - entry_ul) / entry_ul * 100
        else:
            row["strike_vs_entry_pct"] = None

        # For calls: loss happens when underlying > strike at settlement
        if exit_ul and info.get("strike") and info.get("option_type") == "C":
            row["underlying_vs_strike_at_exit_pct"] = (exit_ul - info["strike"]) / info["strike"] * 100
        else:
            row["underlying_vs_strike_at_exit_pct"] = None

        # Convert PnL to USD at exit
        if exit_ul:
            row["pnl_usd"] = t["pnl"] * exit_ul
        else:
            row["pnl_usd"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("pnl_coin", ascending=True, inplace=True)

    # --- Summary ---
    out_dir = Path("reports/loss_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full trade log
    csv_path = out_dir / "all_trades.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all {len(df)} trades to {csv_path}")

    # --- Print worst losing trades ---
    losses = df[df["pnl_coin"] < 0].copy()
    wins = df[df["pnl_coin"] > 0].copy()

    print(f"\n{'='*80}")
    print(f"总交易数: {len(df)}")
    print(f"盈利交易: {len(wins)}  ({len(wins)/len(df)*100:.1f}%)")
    print(f"亏损交易: {len(losses)}  ({len(losses)/len(df)*100:.1f}%)")
    print(f"{'='*80}")

    print(f"\n�?PnL ({coin}):   {df['pnl_coin'].sum():.6f}")
    print(f"总亏�?({coin}):   {losses['pnl_coin'].sum():.6f}")
    print(f"总盈�?({coin}):   {wins['pnl_coin'].sum():.6f}")
    print(f"平均盈利 ({coin}): {wins['pnl_coin'].mean():.6f}")
    print(f"平均亏损 ({coin}): {losses['pnl_coin'].mean():.6f}")

    if losses["pnl_usd"].notna().any():
        print(f"\n总亏�?(USD):   ${losses['pnl_usd'].sum():.2f}")
        print(f"总盈�?(USD):   ${wins['pnl_usd'].sum():.2f}")

    # Top 20 worst trades
    print(f"\n{'='*80}")
    print(f"TOP 20 最大亏损交�?")
    print(f"{'='*80}")
    top_losses = losses.head(20)
    for i, (_, row) in enumerate(top_losses.iterrows(), 1):
        print(f"\n#{i}  {row['instrument']}")
        print(f"   方向: {row['direction']}, 数量: {row['quantity']}")
        print(f"   入场时间: {row['entry_time']}")
        print(f"   结算时间: {row['exit_time']}")
        print(f"   入场权利�?{coin}): {row['entry_price_coin']:.6f}")
        print(f"   结算内在价�?     {row['exit_price_coin']:.6f}")
        print(f"   PnL ({coin}):       {row['pnl_coin']:.6f}")
        if row['pnl_usd'] is not None and not pd.isna(row['pnl_usd']):
            print(f"   PnL (USD):       ${row['pnl_usd']:.2f}")
        if row['underlying_at_entry']:
            print(f"   标的入场�? ${row['underlying_at_entry']:.2f}")
        if row['underlying_at_exit']:
            print(f"   标的结算�? ${row['underlying_at_exit']:.2f}")
        if row['strike'] and row['underlying_at_exit']:
            print(f"   行权�?     ${row['strike']:.0f}")
            if row['underlying_at_exit'] > row['strike']:
                loss_depth = row['underlying_at_exit'] - row['strike']
                print(f"   ITM深度:    ${loss_depth:.2f}  ({loss_depth/row['strike']*100:.2f}%)")

    # --- Aggregate by date to find worst days ---
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.date
    daily_pnl = df.groupby("exit_date").agg(
        total_pnl_coin=("pnl_coin", "sum"),
        n_trades=("pnl_coin", "count"),
        worst_trade=("pnl_coin", "min"),
    ).sort_values("total_pnl_coin")

    print(f"\n{'='*80}")
    print(f"TOP 10 最差结算日:")
    print(f"{'='*80}")
    for date, row_data in daily_pnl.head(10).iterrows():
        print(f"  {date}  PnL={row_data['total_pnl_coin']:.6f} {coin}  "
              f"交易�?{row_data['n_trades']}  最差单�?{row_data['worst_trade']:.6f}")

    # --- Loss pattern analysis ---
    print(f"\n{'='*80}")
    print(f"亏损原因分析:")
    print(f"{'='*80}")

    if not losses.empty and losses["underlying_at_exit"].notna().any():
        # Group by how deep ITM the call was at settlement
        losses_itm = losses[losses["underlying_vs_strike_at_exit_pct"].notna()].copy()
        if not losses_itm.empty:
            bins = [0, 1, 2, 3, 5, 10, 100]
            labels = ["0-1%", "1-2%", "2-3%", "3-5%", "5-10%", ">10%"]
            losses_itm["itm_bucket"] = pd.cut(
                losses_itm["underlying_vs_strike_at_exit_pct"],
                bins=bins, labels=labels, right=True,
            )
            bucket_stats = losses_itm.groupby("itm_bucket", observed=True).agg(
                count=("pnl_coin", "count"),
                total_loss_coin=("pnl_coin", "sum"),
                avg_loss_coin=("pnl_coin", "mean"),
            )
            print("\n�?ITM 深度分组:")
            print(bucket_stats.to_string())

        # Price movement analysis: how much did underlying move between entry and exit
        losses_with_ul = losses[losses["underlying_at_entry"].notna() & losses["underlying_at_exit"].notna()].copy()
        if not losses_with_ul.empty:
            losses_with_ul["ul_move_pct"] = (
                (losses_with_ul["underlying_at_exit"] - losses_with_ul["underlying_at_entry"])
                / losses_with_ul["underlying_at_entry"] * 100
            )
            print(f"\n亏损交易期间标的涨幅统计:")
            print(f"  均�? {losses_with_ul['ul_move_pct'].mean():.2f}%")
            print(f"  中位: {losses_with_ul['ul_move_pct'].median():.2f}%")
            print(f"  最�? {losses_with_ul['ul_move_pct'].max():.2f}%")

    # Save summary
    summary = {
        "underlying": coin,
        "total_trades": len(df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(df) * 100 if len(df) > 0 else 0,
        f"total_pnl_{coin.lower()}": float(df["pnl_coin"].sum()),
        f"total_loss_{coin.lower()}": float(losses["pnl_coin"].sum()),
        f"total_profit_{coin.lower()}": float(wins["pnl_coin"].sum()),
        f"avg_win_{coin.lower()}": float(wins["pnl_coin"].mean()) if len(wins) > 0 else 0,
        f"avg_loss_{coin.lower()}": float(losses["pnl_coin"].mean()) if len(losses) > 0 else 0,
        f"worst_trade_{coin.lower()}": float(losses["pnl_coin"].min()) if len(losses) > 0 else 0,
        "worst_trade_instrument": losses.iloc[0]["instrument"] if len(losses) > 0 else "",
    }

    summary_path = out_dir / "loss_summary.json"
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved summary to {summary_path}")

    daily_pnl.to_csv(out_dir / "daily_pnl.csv")
    print(f"Saved daily PnL to {out_dir / 'daily_pnl.csv'}")


if __name__ == "__main__":
    main()

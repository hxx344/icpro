"""娴嬭瘯涓嶅悓鍏ュ満鏃堕棿锛坋ntry_hour锛夊瑁稿崠 Call 鏀剁泭鐨勫奖鍝嶃€?

Usage: python scripts/debug/test_entry_hour.py
"""
from __future__ import annotations

import json
from pathlib import Path

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

# 娴嬭瘯鐨勫叆鍦哄皬鏃?(UTC)
# Deribit 0-DTE 鍚堢害鍦?08:00 UTC 鍒版湡/涓婃柊
# 8=鍒版湡鍗冲埢鍗栧嚭, 9=寤惰繜1h, ... 16=寤惰繜8h, 0=鎻愬墠8h(鍓嶄竴鏅?
ENTRY_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def run_one(entry_hour: int) -> dict:
    cfg = Config.from_yaml(Path("configs/backtest/naked_call_usd.yaml"))
    params = dict(cfg.strategy.params)
    params["buy_protective_call"] = False
    params["entry_hour"] = entry_hour
    # 娓呴櫎浼樺寲鍙傛暟锛岀敤绾?baseline
    params["strike_offset_pct"] = 0.0
    params["stop_loss_pct"] = 0.0
    params["take_profit_pct"] = 0.0
    params["vol_lookback_hours"] = 0
    params["max_vol_pct"] = 0.0
    params["dynamic_otm"] = False
    params["max_loss_equity_pct"] = 0.0
    cfg.strategy.params = params

    strategy = _load_strategy(cfg.strategy.name, params)
    engine = BacktestEngine(cfg, strategy)
    results = engine.run()
    metrics = compute_metrics(results)

    closed = results["closed_trades"]
    pnls = [t["pnl"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "entry_hour": entry_hour,
        "trades": len(pnls),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
        "total_return": metrics["total_return"],
        "final_coin": metrics["final_equity"],
        "final_usd": metrics.get("final_usd", 0),
        "hedged_usd": metrics.get("final_hedged_usd", 0),
        "max_dd": metrics["max_drawdown"],
        "max_dd_hedged": metrics.get("max_drawdown_hedged", 0),
        "sharpe": metrics["sharpe_ratio"],
        "sharpe_hedged": metrics.get("sharpe_ratio_hedged", 0),
        "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        "total_profit": sum(wins),
        "total_loss": sum(losses),
        "avg_premium": sum(pnls) / len(pnls) if pnls else 0,
    }


def main():
    results = []
    for h in ENTRY_HOURS:
        print(f"  Entry Hour = {h:02d}:00 UTC ...", end="", flush=True)
        r = run_one(h)
        results.append(r)
        print(f"  Return={r['total_return']:.2%}  MaxDD={r['max_dd']:.2%}  "
              f"Sharpe={r['sharpe']:.2f}  PF={r['profit_factor']:.2f}  Wins={r['win_rate']:.1f}%")

    # Baseline = hour 8
    baseline = next(r for r in results if r["entry_hour"] == 8)

    print(f"\n\n{'='*120}")
    print(f"  鍏ュ満鏃堕棿鏁忔劅鎬у垎鏋? (ETH Naked Call 0-DTE, 90 days, Baseline = 08:00 UTC)")
    print(f"{'='*120}")
    hdr = (f"{'Hour':>6s} {'#':>4s} {'Win%':>6s} {'Return':>8s} {'vs 8h':>8s} "
           f"{'HgdUSD':>8s} {'螖$':>7s} "
           f"{'MaxDD':>7s} {'DDHgd':>7s} {'Sharpe':>7s} {'PF':>6s} {'AvgPnl':>10s}")
    print(hdr)
    print("-" * 120)

    for r in results:
        delta_ret = r["total_return"] - baseline["total_return"]
        delta_usd = r["hedged_usd"] - baseline["hedged_usd"]
        marker = " 鈼€ baseline" if r["entry_hour"] == 8 else ""
        line = (
            f"{r['entry_hour']:>4d}:00 "
            f"{r['trades']:>4d} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['total_return']:>7.2%} "
            f"{delta_ret:>+7.2%} "
            f"${r['hedged_usd']:>7,.0f} "
            f"{delta_usd:>+7,.0f} "
            f"{r['max_dd']:>6.2%} "
            f"{r['max_dd_hedged']:>6.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['profit_factor']:>5.2f} "
            f"{r['avg_premium']:>9.6f}"
            f"{marker}"
        )
        print(line)

    # 鍏抽敭鏃舵鍒嗘瀽
    print(f"\n{'='*80}")
    print("  鍏抽敭鏃舵瀵规瘮 (vs 08:00 baseline)")
    print(f"{'='*80}")
    key_hours = [8, 9, 10, 11, 12, 14, 16, 20, 0, 4]
    for h in key_hours:
        r = next((x for x in results if x["entry_hour"] == h), None)
        if r is None:
            continue
        delta = r["total_return"] - baseline["total_return"]
        delta_pct = delta / baseline["total_return"] * 100 if baseline["total_return"] != 0 else 0
        delta_usd = r["hedged_usd"] - baseline["hedged_usd"]
        delay = (h - 8) % 24
        print(f"  {h:02d}:00 UTC (寤惰繜{delay:>2d}h): 鏀剁泭 {r['total_return']:>7.2%} "
              f"({delta:>+7.2%}, {delta_pct:>+5.1f}%)  "
              f"HgdUSD ${r['hedged_usd']:>7,.0f} ({delta_usd:>+6,.0f})  "
              f"MaxDD {r['max_dd_hedged']:>6.2%}  PF {r['profit_factor']:.2f}")

    # Best / worst
    best = max(results, key=lambda x: x["total_return"])
    worst = min(results, key=lambda x: x["total_return"])
    best_pf = max(results, key=lambda x: x["profit_factor"])
    least_dd = min(results, key=lambda x: abs(x["max_dd_hedged"]))

    print(f"\n  鏈€浣冲叆鍦烘椂闂?(鏀剁泭): {best['entry_hour']:02d}:00 UTC 鈫?{best['total_return']:.2%}")
    print(f"  鏈€宸叆鍦烘椂闂?(鏀剁泭): {worst['entry_hour']:02d}:00 UTC 鈫?{worst['total_return']:.2%}")
    print(f"  鏈€浣?PF:            {best_pf['entry_hour']:02d}:00 UTC 鈫?PF {best_pf['profit_factor']:.2f}")
    print(f"  鏈€灏忓洖鎾?            {least_dd['entry_hour']:02d}:00 UTC 鈫?{least_dd['max_dd_hedged']:.2%}")
    print(f"{'='*80}")

    # Save
    out_dir = Path("reports/entry_hour")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

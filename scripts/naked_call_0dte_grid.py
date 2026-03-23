"""0DTE Naked Short Call — OTM% grid comparison."""
import sys, time
sys.path.insert(0, "src")

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy

OTM_PCTS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

rows = []
for otm in OTM_PCTS:
    cfg = Config.from_yaml("configs/backtest/ic_0dte_8pct_direct.yaml")
    cfg.strategy.params["otm_pct"] = otm
    cfg.strategy.params["hedge_otm_pct"] = 0.0
    cfg.strategy.params["min_days_to_expiry"] = 0.0
    cfg.strategy.params["max_days_to_expiry"] = 1.5
    cfg.strategy.params["roll_daily"] = True
    cfg.strategy.params["compound"] = True
    cfg.strategy.params["take_profit_pct"] = 9999
    cfg.strategy.params["stop_loss_pct"] = 9999
    cfg.strategy.params["max_positions"] = 1
    cfg.backtest.name = f"0DTE Naked Call OTM={otm*100:.0f}%"

    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)

    # Monkey-patch: force call-only
    _orig = s._open_strangle
    def _call_only(context, chain, legs="both", _orig=_orig):
        _orig(context, chain, legs="call_only")
    s._open_strangle = _call_only

    e = BacktestEngine(cfg, s)
    t0 = time.perf_counter()
    r = e.run()
    elapsed = time.perf_counter() - t0

    m = compute_metrics(r)
    trades = e.position_mgr.closed_trades
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    row = {
        "otm": otm,
        "return_pct": m.get("total_return", 0) * 100,
        "max_dd_pct": m.get("max_drawdown", 0) * 100,
        "sharpe": m.get("sharpe_ratio", 0),
        "win_rate": m.get("win_rate", 0) * 100,
        "profit_factor": m.get("profit_factor", 0),
        "trades": m.get("total_trades", 0),
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "market_q": e._quote_source_market,
        "synth_q": e._quote_source_synth,
    }
    rows.append(row)
    total_q = row["market_q"] + row["synth_q"]
    mkt_pct = row["market_q"] / total_q * 100 if total_q else 0

    print(f"  OTM={otm*100:5.1f}%  Return={row['return_pct']:>+7.1f}%  "
          f"DD={row['max_dd_pct']:>6.1f}%  Sharpe={row['sharpe']:>5.2f}  "
          f"WR={row['win_rate']:>5.1f}%  PF={row['profit_factor']:>5.2f}  "
          f"Trades={row['trades']:>4.0f}  "
          f"AvgW={row['avg_win']:>8.4f}  AvgL={row['avg_loss']:>9.4f}  "
          f"Mkt={mkt_pct:.0f}%  ({elapsed:.1f}s)")

# Summary
print("\n" + "=" * 90)
print("  0DTE 裸卖 Call — OTM Grid Summary")
print("=" * 90)
print(f"  {'OTM':>6s}  {'Return':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'WinRate':>8s}  {'PF':>6s}  {'Trades':>6s}  {'Mkt%':>5s}")
print("-" * 90)
for row in rows:
    total_q = row["market_q"] + row["synth_q"]
    mkt_pct = row["market_q"] / total_q * 100 if total_q else 0
    print(f"  {row['otm']*100:5.1f}%  {row['return_pct']:>+7.1f}%  {row['max_dd_pct']:>6.1f}%  "
          f"{row['sharpe']:>7.2f}  {row['win_rate']:>7.1f}%  {row['profit_factor']:>5.2f}  "
          f"{row['trades']:>6.0f}  {mkt_pct:>4.0f}%")

# Best
print("\n" + "-" * 90)
best_ret = max(rows, key=lambda r: r["return_pct"])
best_sh = max(rows, key=lambda r: r["sharpe"])
print(f"  最高收益: OTM={best_ret['otm']*100:.0f}% → {best_ret['return_pct']:+.1f}%")
print(f"  最高Sharpe: OTM={best_sh['otm']*100:.0f}% → Sharpe={best_sh['sharpe']:.2f}, Return={best_sh['return_pct']:+.1f}%")

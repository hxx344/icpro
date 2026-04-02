"""Diagnose ETH TP80/SL200+W4% vs other strategies."""
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")
from options_backtest.config import Config
from options_backtest.engine.backtest import BacktestEngine
from options_backtest.analytics.metrics import compute_metrics
from options_backtest.cli import _load_strategy
import numpy as np

BASE_YAML = "configs/backtest/ic_0dte_8pct_direct.yaml"
BASE_PARAMS = {
    "otm_pct": 0.001, "hedge_otm_pct": 0.0,
    "min_days_to_expiry": 0.0, "max_days_to_expiry": 1.5,
    "roll_daily": True, "compound": True,
    "take_profit_pct": 9999, "stop_loss_pct": 9999, "max_positions": 1,
}

configs = {
    "Naked":          {},
    "Wing4%":         {"hedge_otm_pct": 0.04},
    "TP80/SL200":     {"take_profit_pct": 80, "stop_loss_pct": 200},
    "TP80/SL200+W4%": {"take_profit_pct": 80, "stop_loss_pct": 200, "hedge_otm_pct": 0.04},
}

for label, overrides in configs.items():
    cfg = Config.from_yaml(BASE_YAML)
    for k, v in BASE_PARAMS.items():
        cfg.strategy.params[k] = v
    for k, v in overrides.items():
        cfg.strategy.params[k] = v
    cfg.backtest.name = label
    s = _load_strategy(cfg.strategy.name, cfg.strategy.params)
    e = BacktestEngine(cfg, s)
    r = e.run()
    m = compute_metrics(r)
    trades = e.position_mgr.closed_trades
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pnl_arr = np.array(pnls)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Return: {m.get('total_return',0)*100:+.1f}%  DD: {m.get('max_drawdown',0)*100:.1f}%  Sharpe: {m.get('sharpe_ratio',0):.2f}")
    print(f"  Trades: {len(trades)}  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    print(f"  Avg Win:  {np.mean(wins):.4f}" if wins else "  Avg Win: N/A")
    print(f"  Avg Loss: {np.mean(losses):.4f}" if losses else "  Avg Loss: N/A")
    print(f"  Total Gain: {sum(wins):.2f}  Total Loss: {sum(losses):.2f}  Net: {sum(pnls):.2f}")
    print(f"  Max Win:  {max(pnls):.4f}  Max Loss: {min(pnls):.4f}")
    print(f"  Win/Loss ratio: {abs(np.mean(wins)/np.mean(losses)):.2f}" if wins and losses else "")
    
    # PnL distribution
    print(f"  PnL percentiles: P1={np.percentile(pnl_arr,1):.2f} P5={np.percentile(pnl_arr,5):.2f} P10={np.percentile(pnl_arr,10):.2f} P25={np.percentile(pnl_arr,25):.2f} P50={np.percentile(pnl_arr,50):.2f} P75={np.percentile(pnl_arr,75):.2f} P90={np.percentile(pnl_arr,90):.2f} P95={np.percentile(pnl_arr,95):.2f} P99={np.percentile(pnl_arr,99):.2f}")
    
    # Equity curve milestones
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] + p)
    eq = np.array(equity)
    
    # Monthly-ish breakdown (every 100 trades)
    chunk = max(len(trades)//10, 1)
    print(f"  Equity progression (every ~{chunk} trades):")
    for i in range(0, len(trades), chunk):
        end = min(i+chunk, len(trades))
        chunk_pnl = sum(pnls[i:end])
        chunk_wins = sum(1 for p in pnls[i:end] if p > 0)
        chunk_losses = sum(1 for p in pnls[i:end] if p < 0)
        dt_start = trades[i].get("open_time", "?")
        dt_end = trades[end-1].get("open_time", "?")
        print(f"    [{i:>4d}-{end:>4d}] {str(dt_start)[:10]} to {str(dt_end)[:10]}  PnL={chunk_pnl:>+8.2f}  Eq={eq[end]:>8.2f}  W={chunk_wins} L={chunk_losses}")

    # Check: how many times did TP/SL actually trigger?
    if "take_profit_pct" in overrides or "stop_loss_pct" in overrides:
        # Look at trade close reasons if available
        tp_count = 0
        sl_count = 0
        expire_count = 0
        for t in trades:
            reason = t.get("close_reason", "")
            if "tp" in reason.lower() or "profit" in reason.lower():
                tp_count += 1
            elif "sl" in reason.lower() or "stop" in reason.lower() or "loss" in reason.lower():
                sl_count += 1
            else:
                expire_count += 1
        if tp_count + sl_count > 0:
            print(f"  Close reasons: TP={tp_count} SL={sl_count} Expire={expire_count}")
        
        # Approximate: wins with PnL near TP threshold, losses near SL threshold
        tp_pct = overrides.get("take_profit_pct", 9999)
        sl_pct = overrides.get("stop_loss_pct", 9999)
        # Count large wins vs small wins
        if wins:
            large_wins = [w for w in wins if w > np.mean(wins) * 1.5]
            small_wins = [w for w in wins if w <= np.mean(wins) * 1.5]
            print(f"  Win distribution: Large(>{np.mean(wins)*1.5:.2f})={len(large_wins)}  Small={len(small_wins)}")
        if losses:
            large_losses = [l for l in losses if l < np.mean(losses) * 1.5]
            small_losses = [l for l in losses if l >= np.mean(losses) * 1.5]
            print(f"  Loss distribution: Large(<{np.mean(losses)*1.5:.2f})={len(large_losses)}  Small={len(small_losses)}")

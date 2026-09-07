"""Test Sunday whole-package take-profit targets against expiry settlement.

Targets measure executable gross PnL / net opening credit. Opening and closing
fees are deducted from reported PnL. Four valid, adequately sized closing quotes
are required; the strategy does not discard unsellable long protection legs.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.backtest_btc_report_ic import (
    Config, ReportEngine, ReportStrategy, entry_metadata, prefetch_entry_metadata, summarize,
)


def evaluate_premium_target(positions, quotes, target):
    """Use actual opening fills and opposite-side close quotes, all in USD."""
    if len(positions) != 4:
        return {"reason": "incomplete_package"}
    credit = -sum(p.entry_price * p.quantity * p.direction_sign for p in positions.values())
    if credit <= 0:
        return {"reason": "nonpositive_credit", "net_entry_credit_usd": credit}
    close_cost = 0.0
    for name, pos in positions.items():
        bid, ask, mark = quotes.get(name, (None, None, None))
        price = ask if pos.direction_sign < 0 else bid
        if (price is None or not np.isfinite(price) or price <= 0
                or mark is None or not np.isfinite(mark) or mark <= 0):
            return {"reason": "missing_close_quote", "net_entry_credit_usd": credit,
                    "first_missing_instrument": name,
                    "first_missing_side": "ask" if pos.direction_sign < 0 else "bid"}
        if bid is not None and ask is not None and ask < bid:
            return {"reason": "crossed_close_quote", "net_entry_credit_usd": credit}
        close_cost -= pos.direction_sign * price * pos.quantity
    fraction = (credit - close_cost) / credit
    return {"reason": "target_reached" if fraction >= target else "below_target",
            "net_entry_credit_usd": credit, "close_cost_usd": close_cost,
            "gross_pnl_usd": credit - close_cost, "achieved_fraction": fraction}


def sufficient_close_depth(positions, metadata):
    for name, pos in positions.items():
        if name not in metadata.index:
            return False
        size = metadata.loc[name, "ask_amount" if pos.direction_sign < 0 else "bid_amount"]
        if not np.isfinite(size) or size < pos.quantity:
            return False
    return True


def sunday_hours(cfg, timezone):
    start = pd.to_datetime(cfg.backtest.start_date, utc=True)
    end = pd.to_datetime(cfg.backtest.end_date, utc=True)
    days = pd.date_range(start.normalize(), end.normalize(), freq="W-SUN")
    hours = []
    for day in days:
        local_midnight = day.tz_localize(None).tz_localize(timezone).tz_convert("UTC")
        expiry = day + pd.Timedelta(hours=8)
        hours.extend(t for t in pd.date_range(local_midnight, expiry, freq="h", inclusive="left")
                     if start <= t <= end)
    return hours


class SundayTakeProfitStrategy(ReportStrategy):
    def __init__(self, params, target, timezone="UTC"):
        super().__init__(params, require_depth=True)
        self.target = target
        self.timezone = timezone
        self.exits = []
        self.exit_fills = []
        self.exit_checks = []
        self.exit_audit = []
        self._closing = False

    def on_step(self, context):
        now = pd.Timestamp(context.current_time)
        if self.target is None or not context.positions:
            return super().on_step(context)
        if now.tz_convert(self.timezone).weekday() != 6 or now.minute != 0:
            return
        if any(now >= self.expiries[name] for name in context.positions):
            return
        quotes = {name: context._engine._get_quotes_fast(
            name, now, context.underlying_price, for_execution=True)
            for name in context.positions}
        check = {"timestamp": now, **evaluate_premium_target(context.positions, quotes, self.target)}
        entry_time = pd.to_datetime(next(iter(context.positions.values())).entry_time, utc=True)
        check["entry_time"] = entry_time
        if check["reason"] == "target_reached":
            metadata = entry_metadata(now)
            if not sufficient_close_depth(context.positions, metadata):
                check["reason"] = "insufficient_close_size"
            else:
                for name, pos in context.positions.items():
                    side = "ask" if pos.direction_sign < 0 else "bid"
                    row = metadata.loc[name]
                    source_time = pd.to_datetime(row.timestamp, unit="us", utc=True)
                    delay = (source_time - now).total_seconds()
                    assert 0 <= delay < 3600, "Snapshot outside closing hour"
                    self.exit_audit.append({
                        "entry_time": entry_time, "exit_time": now, "instrument_name": name,
                        "quantity": pos.quantity, "side": side,
                        "visible_size": row[f"{side}_amount"], "source_time": source_time,
                        "snapshot_delay_seconds": delay,
                        "close_price_usd": quotes[name][1 if side == "ask" else 0],
                    })
                self.exits.append({"entry_time": entry_time, "exit_time": now,
                                   "target_fraction": self.target, **check})
                self._closing = True
                context.close_all()
        self.exit_checks.append(check)

    def on_fill(self, context, fill):
        if not self._closing:
            return super().on_fill(context, fill)
        self.exit_fills.append({"exit_time": pd.to_datetime(fill.timestamp, utc=True),
                                "instrument_name": fill.instrument_name,
                                "exit_price": fill.fill_price, "quantity": fill.quantity,
                                "fee": fill.fee})
        if not context.positions:
            self._closing = False


def validate_exits(strategy):
    if not strategy.exits:
        assert not strategy.exit_fills
        return
    audit = pd.DataFrame(strategy.exit_audit)
    fills = pd.DataFrame(strategy.exit_fills)
    merged = audit.merge(fills, on=["exit_time", "instrument_name"], validate="one_to_one")
    assert len(merged) == len(strategy.exits) * 4 == len(audit) == len(fills)
    np.testing.assert_allclose(merged.close_price_usd, merged.exit_price, rtol=0, atol=1e-10)
    assert merged.quantity_x.eq(merged.quantity_y).all()
    assert (merged.visible_size >= merged.quantity_x).all()


def write_sweep_report(output, comparisons, yearly, cfg, thresholds, timezone):
    table = pd.DataFrame(comparisons)
    baseline = table.iloc[0]
    lines = ["# BTC 周日净权利金止盈阈值回测", "",
             f"区间：{cfg.backtest.start_date} 至 {cfg.backtest.end_date} UTC。"
             f"卖腿 Delta {cfg.strategy.params['short_delta']:g}，保护翼 Delta {cfg.strategy.params['wing_delta']:g}，每腿固定 {cfg.strategy.params['quantity']:g} BTC。", "",
             f"从 {timezone} 周日 00:00 起逐小时检查，未能整组平仓则于 UTC 周日 08:00 到期结算。", "",
             "## 结果", "",
             "| 止盈目标 | 净盈利 USD | 较持有到期 USD | 日度 PnL Sharpe | 小时最大回撤 USD | 组合胜率 | 提前平仓组数 | 平仓手续费 USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['label']} | {row['total_pnl_usd']:,.2f} | "
                     f"{row['total_pnl_usd'] - baseline.total_pnl_usd:+,.2f} | "
                     f"{row['daily_pnl_sharpe_365']:.3f} | {row['hourly_max_drawdown_usd']:,.2f} | "
                     f"{row['package_win_rate']:.2%} | {row['early_close_packages']} | {row['close_fees_usd']:,.2f} |")
    lines += ["", "## 分年净盈亏 USD", "",
              "| 止盈目标 | " + " | ".join(str(y) for y in sorted(yearly.year.unique())) + " |",
              "|---|" + "---:|" * yearly.year.nunique()]
    for row in comparisons:
        values = yearly.loc[yearly.label == row["label"]].set_index("year").total_pnl_usd
        lines.append(f"| {row['label']} | " + " | ".join(f"{v:,.2f}" for v in values.sort_index()) + " |")
    lines += ["", "2023、2026 为不完整年度。", "", "## 逐小时平仓检查", "",
              "| 止盈目标 | 缺少完整报价 | 未达目标 | 达标但数量不足 | 完成平仓 | 首个缺失为保护翼 bid |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in comparisons[1:]:
        reasons = row["exit_check_reasons"]
        lines.append(f"| {row['label']} | {reasons.get('missing_close_quote', 0)} | "
                     f"{reasons.get('below_target', 0)} | {reasons.get('insufficient_close_size', 0)} | "
                     f"{row['early_close_packages']} | {row['first_missing_close_sides'].get('bid', 0)} |")
    lines += ["", "次数是仍持仓时的小时检查次数。同一组合可多次失败，提前平仓后停止检查。缺失方向仅记录每次检查遇到的第一条腿。",
              "", "## 判定与成交口径", "",
              "- 开仓净收权利金 C = 两条卖腿实际成交权利金 − 两条保护翼实际成本，不含手续费。",
              "- 平仓成本 D = 买回卖腿的 ask 成本 − 卖出保护翼的 bid 收入。达成比例 = (C − D) / C；C 必须大于 0。",
              "- 例如净收 1,000 USD，按盘口平仓需 200 USD，达成比例为 80%。不会将小时观测到的收益截断到目标值。",
              "- 四条腿全部有可成交的对应方向报价、有效标记价及足够盘口数量才整组平仓；不假设没有买价的保护翼可以免费卖掉，也不改为只平卖腿。",
              "- 原始报价仍是小时 open 快照；该时点四腿的报价时间存在秒级差异，不模拟排队及跨腿冲击。未触发或无法完整成交均继续持有。",
              "- 最终净盈亏扣除开仓费、主动平仓费及适用的到期交割费。主动平仓按配置的 0.00024 × BTC 指数价/腿计费，上限为对应权利金的 10%。",
              "- 沿用此前固定数量、USD 线性盈亏及现金＋持仓市值的权益口径；不复利、不对冲、不添加其他止损。",
              "- Sharpe 使用每日 21:00 UTC 的美元 PnL，保留空仓日，以 sqrt(365) 年化。胜率为整组费后胜率。",
              "- 使用本地挂牌期权及离散 Delta，未复现原 SABR/WV4 合成 48 小时期权；不含保证金和强平模型。", "",
              "## 验证与解释", "",
              f"各档开仓合约、数量、价格、时间均与持有到期版本完全相同，共 {int(baseline.packages)} 组。",
              "提前平仓均核对完整四腿成交、对应方向盘口数量、成交价格、费用和逐腿账本；所有组合净盈亏与最终权益核对一致。",
              "不同阈值是在同一历史样本内比较，排序不等于样本外最优参数。", "",
              "## 复现", "", "```powershell",
              r".\.venv-1\Scripts\python.exe scripts/research/sweep_sunday_premium_take_profit.py",
              "```", "",
              f"阈值：{thresholds}%；时区：`{timezone}`。支持 `--config`、`--thresholds`、`--sunday-timezone`、`--start`、`--end`、`--output-dir`。", "",
              "`comparison.csv`、`yearly_comparison.csv` 为汇总；`comparison.html` 为权益及回撤曲线。各阈值子目录保留成交、权益、平仓尝试、触发记录和原始报价审计。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/backtest/btc_report_ic_45_10.yaml")
    parser.add_argument("--thresholds", default="25,50,60,70,80,90,95")
    parser.add_argument("--sunday-timezone", choices=["UTC", "Asia/Shanghai"], default="UTC")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir", default="reports/btc_sunday_take_profit_45_10")
    args = parser.parse_args()
    thresholds = sorted(set(float(x) for x in args.thresholds.split(",")))
    if any(not 0 < x <= 100 for x in thresholds):
        parser.error("Thresholds must be percentages in (0, 100]")
    cfg = Config.from_yaml(args.config)
    if args.start:
        cfg.backtest.start_date = args.start
    if args.end:
        cfg.backtest.end_date = args.end
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(output / "effective_config.yaml")
    (output / "sweep_config.json").write_text(json.dumps({
        "thresholds_pct": thresholds, "sunday_timezone": args.sunday_timezone,
        "base_config": args.config, "target_basis": "gross_executable_pnl_over_net_entry_credit",
    }, indent=2), encoding="utf-8")
    logger.remove()
    logger.add(str(output / "run.log"), level="INFO")
    logger.add(sys.stderr, level="WARNING")
    started = perf_counter()
    prefetch_entry_metadata(cfg, additional_hours=sunday_hours(cfg, args.sunday_timezone))
    comparisons, years, baseline_entries = [], [], None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Cumulative net PnL (USD)", "Hourly drawdown (USD)"))
    for target_pct in [None, *thresholds]:
        label = "持有到期" if target_pct is None else f"{target_pct:g}%"
        name = "hold_to_expiry" if target_pct is None else f"tp_{target_pct:g}"
        print(f"Running {label}", flush=True)
        strategy = SundayTakeProfitStrategy(cfg.strategy.params,
            None if target_pct is None else target_pct / 100, args.sunday_timezone)
        engine = ReportEngine(copy.deepcopy(cfg), strategy)
        result = engine.run()
        validate_exits(strategy)
        stats, equity, yearly = summarize(engine, result, output / name,
                                          allow_early_close=target_pct is not None)
        entry_cols = ["entry_time", "instrument_name", "quantity", "entry_price", "entry_fee"]
        entries = pd.read_csv(output / name / "legs.csv")[entry_cols].sort_values(
            ["entry_time", "instrument_name"]).reset_index(drop=True)
        if baseline_entries is None:
            baseline_entries = entries
        else:
            pd.testing.assert_frame_equal(entries, baseline_entries, check_exact=True)
        pd.DataFrame(strategy.exit_checks).to_csv(output / name / "exit_checks.csv", index=False)
        pd.DataFrame(strategy.exits).to_csv(output / name / "exit_events.csv", index=False)
        pd.DataFrame(strategy.exit_audit).to_csv(output / name / "exit_quote_audit.csv", index=False)
        stats.update(label=label, target_pct=target_pct, early_close_packages=len(strategy.exits),
                     close_fees_usd=stats.get("close_fees_usd", 0.0),
                     exit_check_reasons=pd.Series([e["reason"] for e in strategy.exit_checks]).value_counts().to_dict(),
                     first_missing_close_sides=pd.Series([
                         e["first_missing_side"] for e in strategy.exit_checks
                         if "first_missing_side" in e]).value_counts().to_dict())
        comparisons.append(stats)
        for row in yearly:
            years.append({"label": label, **row})
        print(json.dumps({k: stats[k] for k in ["label", "total_pnl_usd", "hourly_max_drawdown_usd",
            "daily_pnl_sharpe_365", "early_close_packages", "close_fees_usd", "exit_check_reasons"]}, ensure_ascii=False), flush=True)
        fig.add_trace(go.Scatter(x=equity.index, y=equity.cumulative_pnl_usd, name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=equity.index, y=equity.drawdown_usd, name=label, showlegend=False), row=2, col=1)
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
    (output / "comparison.json").write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")
    yearly = pd.DataFrame(years)
    yearly.to_csv(output / "yearly_comparison.csv", index=False)
    fig.update_layout(template="plotly_white", height=850, title="BTC Sunday take profit | Short 0.45 / Wing 0.10 | Fixed 1 BTC/leg")
    fig.write_html(output / "comparison.html", include_plotlyjs=True)
    write_sweep_report(output, comparisons, yearly, cfg, thresholds, args.sunday_timezone)
    print(f"Done in {perf_counter() - started:.2f}s: {output}", flush=True)


if __name__ == "__main__":
    main()

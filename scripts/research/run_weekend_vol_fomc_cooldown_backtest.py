from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from options_backtest.engine.backtest import BacktestEngine
from options_backtest.strategy.weekend_vol import WeekendVolStrategy
from scripts.research.run_cached_hourly_backtest import (
    _apply_initial_usd_balance,
    _apply_stop_loss_caps,
    _json_default,
    _write_dataset_outputs,
    _choose_date_window,
    detect_hourly_coverage,
    load_run_config,
)

# Official FOMC statement / meeting-end dates from the Federal Reserve calendar.
FOMC_END_DATES_UTC = [
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
]


class FomcCooldownWeekendVolStrategy(WeekendVolStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.fomc_end_dates = {
            pd.Timestamp(v).tz_localize("UTC").normalize()
            for v in self.params.get("fomc_end_dates_utc", [])
        }
        self.fomc_blackout_days_after = int(self.params.get("fomc_blackout_days_after", 0))

    def _is_in_fomc_blackout(self, now: pd.Timestamp) -> bool:
        if self.fomc_blackout_days_after <= 0 or not self.fomc_end_dates:
            return False
        entry_day = now.normalize()
        for fomc_day in self.fomc_end_dates:
            day_diff = (entry_day - fomc_day).days
            if 0 <= day_diff <= self.fomc_blackout_days_after:
                return True
        return False

    def _should_enter(self, now: pd.Timestamp) -> bool:
        if self._is_in_fomc_blackout(now):
            self.log(
                f"Skip entry: FOMC blackout active for {now.strftime('%Y-%m-%d')} "
                f"(within {self.fomc_blackout_days_after} days after meeting)"
            )
            return False
        return super()._should_enter(now)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WeekendVol backtest with FOMC post-meeting blackout.")
    parser.add_argument("--config", default="configs/trader/weekend_vol_btc.yaml")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--output-dir", default="reports/validation/weekend_vol_btc_fomc_3d_blackout")
    parser.add_argument("--blackout-days-after", type=int, default=3)
    return parser.parse_args()


def _compute_blocked_fridays(start_date: str, end_date: str, blackout_days_after: int) -> list[str]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    blocked: list[str] = []
    for value in FOMC_END_DATES_UTC:
        fomc_day = pd.Timestamp(value, tz="UTC")
        for offset in range(blackout_days_after + 1):
            candidate = fomc_day + pd.Timedelta(days=offset)
            if candidate.weekday() == 4 and start_ts <= candidate <= end_ts:
                blocked.append(candidate.strftime("%Y-%m-%d"))
    return sorted(set(blocked))


def main() -> None:
    args = _parse_args()
    cfg_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    loaded = load_run_config(cfg_path)
    cfg = copy.deepcopy(loaded.cfg)

    underlying = str(cfg.backtest.underlying or "BTC").upper()
    coverage = detect_hourly_coverage(underlying)
    start_date, end_date = _choose_date_window(coverage, args.start, args.end)

    cfg.backtest.option_data_source = "options_hourly"
    cfg.backtest.start_date = start_date
    cfg.backtest.end_date = end_date
    cfg.strategy.params = dict(cfg.strategy.params or {})
    cfg.strategy.params.update(
        {
            "fomc_end_dates_utc": FOMC_END_DATES_UTC,
            "fomc_blackout_days_after": int(args.blackout_days_after),
        }
    )
    cfg.report.output_dir = args.output_dir

    initial_spot = _apply_initial_usd_balance(cfg, loaded.source_type, start_date, end_date)
    output_dir = (REPO_ROOT / cfg.report.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    blocked_fridays = _compute_blocked_fridays(start_date, end_date, int(args.blackout_days_after))

    print(f"[coverage] {coverage.underlying}: {coverage.start_ts} -> {coverage.end_ts} ({coverage.file_count} parquet files)")
    print(f"[run] config={cfg_path}")
    print(f"[run] config_type={loaded.source_type}")
    print(f"[run] backtest window={start_date} -> {end_date}")
    print(f"[run] fomc_blackout_days_after={int(args.blackout_days_after)}")
    print(f"[run] blocked_fridays={blocked_fridays}")
    if initial_spot is not None:
        initial_usd = float(cfg.account.initial_balance) * float(initial_spot)
        print(
            f"[run] initial_usd={initial_usd:,.2f} "
            f"(coin_equiv={cfg.account.initial_balance:.12f}, first_spot={initial_spot:,.2f})"
        )
    print(f"[run] output_dir={output_dir}")

    strategy = FomcCooldownWeekendVolStrategy(cfg.strategy.params)
    engine = BacktestEngine(cfg, strategy)
    observed_results = engine.run()
    capped_results, cap_details = _apply_stop_loss_caps(observed_results)

    observed_output = _write_dataset_outputs(
        dataset_name="observed_hour",
        dataset_label="Observed hourly stop-loss fills (FOMC blackout)",
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
            "fomc_blackout_days_after": int(args.blackout_days_after),
            "blocked_fridays": blocked_fridays,
        },
    )
    capped_output = _write_dataset_outputs(
        dataset_name="cap_at_threshold",
        dataset_label="Stop-loss capped at threshold (FOMC blackout)",
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
            "fomc_blackout_days_after": int(args.blackout_days_after),
            "blocked_fridays": blocked_fridays,
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
        "fomc_blackout_days_after": int(args.blackout_days_after),
        "blocked_fridays": blocked_fridays,
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
    print(f"[done] summary={observed_output['summary_path']}")
    print(f"[done] dataset=cap_at_threshold trades={capped_output['trade_count']}")
    print(f"[done] summary={capped_output['summary_path']}")
    print(f"[done] comparison={comparison_path}")


if __name__ == "__main__":
    main()

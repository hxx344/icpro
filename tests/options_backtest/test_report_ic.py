"""Research safeguards: complete packages, premium accounting, daily PnL basis."""
from types import SimpleNamespace
import os

import numpy as np
import pandas as pd
import pytest

from scripts.research import backtest_btc_report_ic as report_ic

from scripts.research.backtest_btc_report_ic import (
    CashMarketValueAccount,
    ReportEngine,
    daily_statistics,
    has_package_depth,
    select_package,
)


def chain():
    return pd.DataFrame([
        {"instrument_name": name, "option_type": kind, "strike_price": strike,
         "delta": delta, "expiration_date": pd.Timestamp("2025-01-05 08:00Z"),
         "bid_price": 0.01, "ask_price": 0.012, "mark_price": 0.011}
        for name, kind, strike, delta in [
            ("SC", "call", 100, 0.46), ("LC", "call", 110, 0.16),
            ("SP", "put", 95, -0.44), ("LP", "put", 90, -0.14),
        ]
    ])


def test_select_complete_package_and_never_mix_expiries():
    data = chain()
    wrong_expiry = data.copy()
    wrong_expiry.expiration_date += pd.Timedelta(days=1)
    wrong_expiry.delta = [0.45, 0.15, -0.45, -0.15]
    legs, reason = select_package(pd.concat([data, wrong_expiry]),
                                  pd.Timestamp("2025-01-03 21:00Z"), 0.45, 0.15)
    assert reason == "entered"
    assert [leg.instrument_name for leg in legs] == ["SC", "SP", "LC", "LP"]
    assert all(leg.expiration_date == pd.Timestamp("2025-01-05 08:00Z") for leg in legs)


@pytest.mark.parametrize("bad_quote", [0.0, np.nan])
def test_no_partial_package_when_wing_cannot_execute(bad_quote):
    data = chain()
    data.loc[data.instrument_name == "LC", "ask_price"] = bad_quote
    legs, reason = select_package(data, pd.Timestamp("2025-01-03 21:00Z"), 0.45, 0.15)
    assert not legs
    assert reason == "no_executable_call_wing"


def test_equity_does_not_count_received_premium_twice():
    positions = {"short": SimpleNamespace(current_mark_price=100.0,
                                           quantity=1.0, direction_sign=-1)}
    account = CashMarketValueAccount(10000.0, positions)
    account.deposit(100.0)
    assert account.equity() == 10000.0
    positions["short"].current_mark_price = 60.0
    assert account.equity(40.0) == 10040.0
    account.pay_fee(5.0)
    assert account.equity() == 10035.0


def test_daily_statistics_keep_flat_days_and_initial_loss_drawdown():
    pnl = pd.Series([-100.0, 0.0, 50.0, 100.0])
    result = daily_statistics(pnl)
    assert result["days"] == 4
    assert result["nonzero_days"] == 3
    assert result["total_pnl_usd"] == 50.0
    assert result["daily_max_drawdown_usd"] == -100.0
    assert result["nonzero_daily_win_rate"] == pytest.approx(2 / 3)
    assert result["daily_pnl_sharpe_365"] == pytest.approx(
        pnl.mean() / pnl.std(ddof=1) * np.sqrt(365))


def test_mark_new_contract_without_instrument_catalogue_entry():
    engine = object.__new__(ReportEngine)
    engine.strategy = SimpleNamespace(expiries={"new": pd.Timestamp("2026-02-01 08:00Z")})
    engine.position_mgr = SimpleNamespace(positions={"new": SimpleNamespace(current_mark_price=50)})
    engine._options_hourly_store = SimpleNamespace(get_quote=lambda *args: (0.01, 0.012, 0.011))
    engine.market_marks = 0
    engine.stale_marks = []
    assert engine._get_mark_prices_fast(pd.Timestamp("2026-01-30 21:00Z"), 10000) == {"new": 110}
    assert engine._get_mark_prices_fast(pd.Timestamp("2026-02-01 08:00Z"), 10000) == {}


def test_all_four_legs_must_have_sufficient_directional_depth():
    legs, _ = select_package(chain(), pd.Timestamp("2025-01-03 21:00Z"), 0.45, 0.15)
    metadata = pd.DataFrame({"bid_amount": [1., 1., 0.1, 0.1],
                             "ask_amount": [0.1, 0.1, 1., 1.]}, index=["SC", "SP", "LC", "LP"])
    assert has_package_depth(legs, metadata, 1.)
    metadata.loc["LP", "ask_amount"] = 0.9
    assert not has_package_depth(legs, metadata, 1.)
    assert not has_package_depth(legs, metadata.drop("LC"), 1.)


@pytest.fixture
def metadata_source(tmp_path, monkeypatch):
    monkeypatch.setattr(report_ic, "ROOT", tmp_path)
    monkeypatch.setattr(report_ic, "_ENTRY_METADATA_PREFETCH", {})
    report_ic._read_entry_metadata.cache_clear()
    path = tmp_path / "data/options_hourly/BTC/2025-01.parquet"
    path.parent.mkdir(parents=True)
    times = pd.to_datetime(["2025-01-03 21:00Z", "2025-01-03 22:00Z", "2025-01-04 21:00Z"])
    frame = pd.DataFrame({"hour": times, "hourly_pick": ["open"] * 3,
                          "symbol": ["A"] * 3, "timestamp": [100, 200, 300],
                          "bid_amount": [1., 2., 3.], "ask_amount": [4., 5., 6.]})
    frame.to_parquet(path, index=False)
    cfg = report_ic.Config()
    cfg.backtest.start_date = "2025-01-03"
    cfg.backtest.end_date = "2025-01-04 23:00:00"
    cfg.strategy.params = {"entry_hour_utc": 21, "entry_retry_hours": 2}
    yield path, frame, cfg
    report_ic._read_entry_metadata.cache_clear()


def test_prefetch_is_one_read_per_month_and_does_not_expose_future_rows(metadata_source, monkeypatch):
    _, _, cfg = metadata_source
    real_read = pd.read_parquet
    reads = []

    def read(*args, **kwargs):
        reads.append(args[0])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    report_ic.prefetch_entry_metadata(cfg)
    assert len(reads) == 1
    first = report_ic.entry_metadata(pd.Timestamp("2025-01-03 21:00Z"))
    later = report_ic.entry_metadata(pd.Timestamp("2025-01-03 22:00Z"))
    assert first.loc["A", "bid_amount"] == 1.
    assert later.loc["A", "bid_amount"] == 2.
    assert report_ic.entry_metadata(pd.Timestamp("2025-01-03 23:00Z")).empty
    assert len(reads) == 1
    # Unscheduled callers retain the direct-read fallback.
    assert report_ic.entry_metadata(pd.Timestamp("2025-01-04 21:00Z")).loc["A", "bid_amount"] == 3.
    assert len(reads) == 2


def test_metadata_cache_invalidates_when_raw_file_changes(metadata_source):
    path, frame, cfg = metadata_source
    now = pd.Timestamp("2025-01-03 21:00Z")
    report_ic.prefetch_entry_metadata(cfg)
    assert report_ic.entry_metadata(now).loc["A", "bid_amount"] == 1.
    previous_mtime = path.stat().st_mtime_ns
    frame.loc[0, "bid_amount"] = 9.
    frame.to_parquet(path, index=False)
    os.utime(path, ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000))
    assert report_ic.entry_metadata(now).loc["A", "bid_amount"] == 9.


def test_unused_duplicate_snapshot_does_not_break_prefetch(metadata_source):
    path, frame, cfg = metadata_source
    pd.concat([frame, frame.iloc[[1]]]).to_parquet(path, index=False)
    report_ic.prefetch_entry_metadata(cfg)
    assert report_ic.entry_metadata(pd.Timestamp("2025-01-03 21:00Z")).loc["A", "bid_amount"] == 1.
    with pytest.raises(RuntimeError, match="Duplicate opening snapshots"):
        report_ic.entry_metadata(pd.Timestamp("2025-01-03 22:00Z"))

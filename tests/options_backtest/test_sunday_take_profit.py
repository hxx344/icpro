"""Executable target, Sunday timing, whole-package depth and fee accounting."""
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.research import backtest_btc_report_ic as report
from scripts.research import sweep_sunday_premium_take_profit as sweep


def package():
    return {name: SimpleNamespace(entry_price=price, direction_sign=sign, quantity=1.,
                                  entry_time=pd.Timestamp("2025-01-03 21:00Z"))
            for name, price, sign in [("SC", 600, -1), ("SP", 600, -1),
                                      ("LC", 100, 1), ("LP", 100, 1)]}


def quotes():
    # Net credit 1000; close debit 2*110 - 2*10 = 200, hence 80% achieved.
    return {"SC": (90., 110., 100.), "SP": (90., 110., 100.),
            "LC": (10., 20., 15.), "LP": (10., 20., 15.)}


def test_target_uses_net_credit_and_executable_sides_without_capping_profit():
    result = sweep.evaluate_premium_target(package(), quotes(), .5)
    assert result["reason"] == "target_reached"
    assert result["net_entry_credit_usd"] == 1000
    assert result["close_cost_usd"] == 200
    assert result["achieved_fraction"] == .8
    assert sweep.evaluate_premium_target(package(), quotes(), .81)["reason"] == "below_target"


@pytest.mark.parametrize("bad", [None, 0., float("nan")])
def test_no_free_disposal_of_long_wing(bad):
    book = quotes()
    book["LP"] = (bad, 20., 15.)
    assert sweep.evaluate_premium_target(package(), book, .25)["reason"] == "missing_close_quote"


def test_nonpositive_credit_and_missing_contract_cannot_trigger():
    positions = package()
    positions["LC"].entry_price = 1500
    assert sweep.evaluate_premium_target(positions, quotes(), .25)["reason"] == "nonpositive_credit"
    book = quotes()
    del book["LP"]
    assert sweep.evaluate_premium_target(package(), book, .25)["reason"] == "missing_close_quote"


def test_invalid_mark_and_crossed_book_cannot_trigger():
    book = quotes()
    book["LP"] = (10., 20., float("nan"))
    result = sweep.evaluate_premium_target(package(), book, .25)
    assert result["reason"] == "missing_close_quote"
    assert result["first_missing_side"] == "bid"
    book["LP"] = (20., 10., 15.)
    assert sweep.evaluate_premium_target(package(), book, .25)["reason"] == "crossed_close_quote"


def test_close_depth_is_opposite_of_entry_depth():
    depth = pd.DataFrame({"ask_amount": [1., 1., .1, .1],
                          "bid_amount": [.1, .1, 1., 1.]}, index=package())
    assert sweep.sufficient_close_depth(package(), depth)
    depth.loc["LP", "bid_amount"] = .9
    assert not sweep.sufficient_close_depth(package(), depth)


@pytest.mark.parametrize("time,timezone,expected", [
    ("2025-01-04 23:00Z", "UTC", 0), ("2025-01-05 00:00Z", "UTC", 1),
    ("2025-01-05 07:00Z", "UTC", 1), ("2025-01-05 08:00Z", "UTC", 0),
    ("2025-01-04 16:00Z", "Asia/Shanghai", 1),
])
def test_sunday_gate_and_whole_package_order(time, timezone, expected, monkeypatch):
    strategy = sweep.SundayTakeProfitStrategy({}, .8, timezone)
    strategy.expiries = {name: pd.Timestamp("2025-01-05 08:00Z") for name in package()}
    depth = pd.DataFrame({"ask_amount": [1.] * 4, "bid_amount": [1.] * 4,
                          "timestamp": [pd.Timestamp(time).value // 1000] * 4}, index=package())
    monkeypatch.setattr(sweep, "entry_metadata", lambda now: depth)
    calls = []
    context = SimpleNamespace(current_time=pd.Timestamp(time), positions=package(),
        underlying_price=10000., close_all=lambda: calls.append("close"),
        _engine=SimpleNamespace(_get_quotes_fast=lambda name, *a, **kw: quotes()[name]))
    strategy.on_step(context)
    assert len(calls) == expected
    assert len(strategy.exit_audit) == 4 * expected
    if expected:
        depth.loc["LP", "bid_amount"] = .5
        strategy.on_step(context)
        assert len(calls) == 1
        assert strategy.exit_checks[-1]["reason"] == "insufficient_close_size"


def test_sunday_hours_end_before_expiry():
    cfg = report.Config()
    cfg.backtest.start_date = "2025-01-03"
    cfg.backtest.end_date = "2025-01-06"
    hours = sweep.sunday_hours(cfg, "UTC")
    assert len(hours) == 8
    assert hours[0] == pd.Timestamp("2025-01-05 00:00Z")
    assert hours[-1] == pd.Timestamp("2025-01-05 07:00Z")


def test_mixed_exit_ledger_charges_each_fee_exactly_once(tmp_path, monkeypatch):
    entries, fills, trades = [], [], []
    for day, kind in [(3, "trade"), (10, "settlement")]:
        entry = pd.Timestamp(f"2025-01-{day:02} 21:00Z")
        expiry = entry.normalize() + pd.Timedelta(days=2, hours=8)
        for name in package():
            entries.append(dict(entry_time=entry, instrument_name=name, dte_hours=35., delta_error=0.))
            fills.append(dict(entry_time=entry, instrument_name=name, entry_fee=2.))
            trades.append(dict(entry_time=entry, exit_time=expiry, instrument_name=name,
                quantity=1., pnl=100. if kind == "trade" else 97., fee=3., close_type=kind))
    # Each leg nets 100 - 2 opening fee - 3 closing/delivery fee = 95.
    strategy = SimpleNamespace(entries=entries, fills=fills, require_depth=True,
        params={"quantity": 1.}, attempts=[{"reason": "entered"}],
        exits=[{"entry_time": entries[0]["entry_time"], "exit_time": trades[0]["exit_time"]}])
    audit = pd.DataFrame({"touch_size": [1.] * 8, "quantity": [1.] * 8,
                          "snapshot_delay_seconds": [0.] * 8})
    monkeypatch.setattr(report, "audit_raw_entries", lambda frame: audit)
    results = {"initial_balance": 10000., "closed_trades": trades,
        "equity_history": [(pd.Timestamp("2025-01-03 21:00Z"), 10000., 10000., 0., 10000.),
                           (pd.Timestamp("2025-01-13 23:00Z"), 10760., 10760., 0., 10000.),
                           (pd.Timestamp("2025-01-13 21:00Z"), 10760., 10760., 0., 10000.)]}
    # Input history must be chronological, just as the engine emits it.
    results["equity_history"].sort(key=lambda row: row[0])
    engine = SimpleNamespace(strategy=strategy, stale_marks=[], market_marks=0)
    stats, _, _ = report.summarize(engine, results, tmp_path, allow_early_close=True)
    assert stats["total_pnl_usd"] == 760.
    assert stats["gross_pnl_usd"] == 800.
    assert stats["entry_fees_usd"] == 16.
    assert stats["close_fees_usd"] == 12.
    assert stats["settlement_fees_usd"] == 12.
    assert stats["ledger_reconciliation_error_usd"] == 0.

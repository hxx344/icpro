"""Research safeguards: complete packages, premium accounting, daily PnL basis."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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

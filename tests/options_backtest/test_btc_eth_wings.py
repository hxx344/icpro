"""Cross-asset sizing, quote conversion, fees and settlement must use own asset."""
import pandas as pd
import pytest
import numpy as np

from scripts.research.backtest_btc_eth_wings import (
    Config, Market, equal_notional_quantity, select_eth_wings, simulate,
)


def test_equal_notional_is_fixed_entry_quantity_not_equal_option_premium():
    quantity = equal_notional_quantity(100000., 2000.)
    assert quantity == 50.
    assert quantity * 2000. == 100000.
    with pytest.raises(ValueError):
        equal_notional_quantity(100000., 0.)


def test_eth_wings_require_matching_expiry_otm_and_valid_two_sided_quotes():
    expiry = pd.Timestamp("2025-01-05 08:00Z")
    frame = pd.DataFrame([
        {"instrument_name": name, "option_type": kind, "strike_price": strike,
         "delta": delta, "expiration_date": expiry, "bid_price": 1., "ask_price": 2., "mark_price": 1.5}
        for name, kind, strike, delta in [("C", "call", 2200., .11), ("P", "put", 1800., -.12),
                                          ("ITM", "call", 1900., .10)]])
    wrong_expiry = frame.iloc[[0]].copy()
    wrong_expiry.expiration_date += pd.Timedelta(days=1)
    wrong_expiry.delta = .1
    wings, reason = select_eth_wings(pd.concat([frame, wrong_expiry]), expiry, 2000.)
    assert reason == "eligible"
    assert [w.instrument_name for w in wings] == ["C", "P"]
    frame.loc[frame.instrument_name == "P", "bid_price"] = 0.
    assert select_eth_wings(frame, expiry, 2000.)[1] == "missing_eth_put_wing"


def test_missing_eth_settlement_falls_back_to_eth_spot_only():
    market = object.__new__(Market)
    now = pd.Timestamp("2025-01-05 08:00Z")
    market.settlement_index = {"ETH": {}}
    market.settlements = {"ETH": pd.DataFrame()}
    market.spots = {"BTC": {now: 100000.}, "ETH": {now: 2500.}}
    assert market.settlement("ETH-5JAN25-2400-C", now, now) == (2500., "same_asset_hour_open")


def test_eth_quotes_convert_with_eth_price():
    class Store:
        def get_quote(self, *args):
            return .001, .002, .0015
    market = object.__new__(Market)
    now = pd.Timestamp("2025-01-03 21:00Z")
    market.spots = {"BTC": {now: 100000.}, "ETH": {now: 2000.}}
    market.stores = {"BTC": Store(), "ETH": Store()}
    assert market.quote("ETH-5JAN25-2400-C", now) == (2., 4., 3.)
    assert market.quote("BTC-5JAN25-110000-C", now) == (100., 200., 150.)
    market.quote.cache_clear()


@pytest.mark.parametrize("fractional", [False, True])
def test_cross_asset_ledger_and_fees_use_each_asset_price(tmp_path, fractional):
    entry, expiry = pd.Timestamp("2025-01-03 21:00Z"), pd.Timestamp("2025-01-05 08:00Z")
    eth_entry = 2100. if fractional else 2000.
    eth_settle = 2600.1 if fractional else 2600.
    quantity = equal_notional_quantity(100000., eth_entry)
    class FakeMarket:
        timeline = pd.date_range(entry, "2025-01-06 21:00Z", freq="h")

        def spot(self, coin, now):
            return 100000. if coin == "BTC" else eth_entry

        def quote(self, name, now):
            return (1000., 1002., 1001.) if name.startswith("BTC") else (1., 2., 1.5)

        def depth(self, name, now, side, qty):
            return {"source_time": now, "visible_size": 1000., "snapshot_delay_seconds": 0.}

        def settlement(self, name, expiry, now):
            return (100000. if name.startswith("BTC") else eth_settle), "instrument_record"

    plan = [{"instrument_name": f"{coin}-5JAN25-{strike}-{label[-1]}", "leg": label,
             "quantity": qty, "strike": strike, "expiry": expiry, "entry_time": entry, "delta": delta}
            for coin, label, qty, strike, delta in [
                ("BTC", "SC", 1., 110000., .45), ("BTC", "SP", 1., 90000., -.45),
                ("ETH", "LC", quantity, np.float32(2500.), .10),
                ("ETH", "LP", quantity, np.float32(1500.), -.10)]]
    cfg = Config.from_yaml("configs/backtest/btc_report_ic_45_10.yaml")
    stats, _, trades, _ = simulate({entry: plan}, FakeMarket(), cfg, None, tmp_path)
    entry_fees = 48. + .4 * quantity
    delivery_fees = .00015 * eth_settle * quantity
    assert stats["entry_fees_usd"] == pytest.approx(entry_fees)
    assert stats["settlement_fees_usd"] == pytest.approx(delivery_fees)
    assert stats["total_pnl_usd"] == pytest.approx(
        2000. - 4. * quantity + (eth_settle - 2500.) * quantity - entry_fees - delivery_fees)
    assert stats["ledger_error_usd"] == pytest.approx(0.)
    assert trades.loc[trades.leg == "LC", "exit_price"].iloc[0] == pytest.approx(eth_settle - 2500.)

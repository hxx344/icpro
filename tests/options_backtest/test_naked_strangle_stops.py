"""Execution gaps and fee-inclusive package stops must preserve actual losses."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.sweep_btc_naked_strangle_stops import (
    Week, choose_exit, fee, rank, run_variant, select_shorts,
)


def exit_path(costs, threshold=.5, *, fees=None, valid=None, depth=None):
    n = len(costs)
    return choose_exit(1000., np.array(costs, dtype=float),
                       np.zeros(n) if fees is None else np.array(fees),
                       np.ones(n, dtype=bool) if valid is None else np.array(valid),
                       np.ones(n, dtype=bool) if depth is None else np.array(depth), threshold)


def test_fee_inclusive_net_credit_boundary_ignores_entry_and_expiry_quotes():
    result = exit_path([9999., 1490., 9999.], fees=[0., 10., 0.])
    assert result['exit_i'] == 1
    assert result['exit_loss_fraction'] == pytest.approx(.5)
    assert not exit_path([9999., 1490., 9999.])['stopped']
    assert not exit_path([9999., 9999., 9999.], None)['stopped']


def test_missing_quote_cannot_trigger_and_gap_loss_is_not_capped():
    result = exit_path([1000., 1700., 2500., 0.], valid=[True, False, True, False])
    assert result['exit_i'] == 2
    assert result['trigger_i'] == 2
    assert result['missing_checks'] == 1
    assert result['exit_loss_fraction'] == 1.5


def test_trigger_latches_through_missing_quotes_and_price_recovery():
    result = exit_path([1000., 1600., np.nan, 1100., 0.],
                       valid=[True, True, False, True, False],
                       depth=[True, False, False, True, False])
    assert (result['trigger_i'], result['exit_i']) == (1, 3)
    assert result['insufficient_size_checks'] == 1
    assert result['missing_checks'] == 1
    assert result['exit_loss_fraction'] == pytest.approx(.1)


def test_unfilled_stop_still_settles_and_nonpositive_credit_is_rejected():
    result = exit_path([1000., 1600., 1700., 0.], depth=[True, False, False, False])
    assert (result['trigger_i'], result['exit_i'], result['stopped']) == (1, 3, False)
    assert result['insufficient_size_checks'] == 2
    with pytest.raises(ValueError, match='positive'):
        choose_exit(0., [1., 2.], [0., 0.], [True, True], [True, True], .5)


def test_fees_cap_by_each_option_premium_and_otm_delivery_is_zero():
    assert fee([1000., 10.], 100000., 1., .0003, .07) == pytest.approx([30., .7])
    assert fee([1000., 10., 0.], 100000., 1., .00015, .125) == pytest.approx([15., 1.25, 0.])


def chain():
    now = pd.Timestamp('2025-01-03 21:00Z')
    expiry = pd.Timestamp('2025-01-05 08:00Z')
    rows = [dict(instrument_name=name, option_type=kind, strike_price=strike,
                 delta=delta, mark_price=.01, bid_price=.009, ask_price=.011,
                 expiration_date=expiry)
            for name, kind, strike, delta in [('C', 'call', 101000., .45),
                                              ('P', 'put', 99000., -.45),
                                              ('C2', 'call', 102000., .40)]]
    return now, pd.DataFrame(rows)


def test_selection_requires_same_sunday_distinct_strikes_and_nearest_before_liquidity():
    now, frame = chain()
    legs, reason = select_shorts(frame, now)
    assert reason == 'candidate'
    assert [x.instrument_name for x in legs] == ['C', 'P']
    frame.loc[frame.instrument_name == 'C', 'bid_price'] = 0.
    assert select_shorts(frame, now)[1] == 'missing_entry_quote'
    now, frame = chain()
    frame.loc[frame.instrument_name == 'P', 'strike_price'] = 101000.
    assert select_shorts(frame, now)[1] == 'overlapping_short_strikes'
    now, frame = chain()
    frame.loc[frame.instrument_name == 'P', 'expiration_date'] += pd.Timedelta(days=1)
    assert select_shorts(frame, now)[1] == 'no_valid_put'


@pytest.mark.parametrize('blocked', [False, True])
def test_two_leg_execution_and_independent_cash_ledger(blocked):
    hours = pd.date_range('2025-01-03 21:00Z', '2025-01-05 08:00Z', freq='h')
    n = len(hours)
    cfg = dict(quantity=1., trading_fee_rate=.0003, trading_fee_cap=.07,
               delivery_fee_rate=.00015, delivery_fee_cap=.125)
    asks = np.full((n, 2), 500.)
    asks[1] = [900., 600.]  # net credit=940; close including fees=1560; loss=620
    depth = np.ones((n, 2), dtype=bool)
    if blocked:
        depth[1:-1, 1] = False
    week = Week(hours[0], hours[-1],
                [dict(instrument_name='C', quantity=1.), dict(instrument_name='P', quantity=1.)],
                hours, np.full(n, 100000.), np.full((n, 2), 500.), asks,
                np.full((n, 2), 510.), np.ones((n, 2), dtype=bool), depth,
                depth.astype(float), [[t, t] for t in hours], np.array([500., 500.]),
                np.array([30., 30.]), np.array([1200., 0.]), np.array([100000., 100000.]),
                ['instrument_record'] * 2, np.zeros((n, 2), dtype=bool))
    packs, legs, curve = run_variant([week], hours, cfg, .5)
    expected = -275. if blocked else -620.
    assert len(legs) == 2
    assert packs.net_pnl_usd.iloc[0] == pytest.approx(expected)
    assert curve.iloc[-1] == pytest.approx(expected)
    assert legs.net_pnl_usd.sum() == pytest.approx(expected)
    assert bool(packs.unfilled_stop_at_expiry.iloc[0]) == blocked
    assert packs.exit_time.iloc[0] == (hours[-1] if blocked else hours[1])
    assert packs.close_fees_usd.iloc[0] == pytest.approx(0. if blocked else 60.)
    assert packs.delivery_fees_usd.iloc[0] == pytest.approx(15. if blocked else 0.)
    assert packs.stale_mark_observations.iloc[0] == 0


def test_training_rank_ties_prefer_smaller_drawdown_then_tighter_stop():
    table = pd.DataFrame(dict(variant=['a', 'b', 'c'], total_pnl_usd=[100., 100., 100.],
                              hourly_max_drawdown_usd=[-20., -10., -10.], threshold_pct=[5., 50., 25.]))
    assert rank(table).variant.tolist() == ['c', 'b', 'a']

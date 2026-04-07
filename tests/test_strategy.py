"""Basic tests for strategies (instantiation and param handling)."""

import pytest

from options_backtest.strategy.long_call import LongCallStrategy
from options_backtest.strategy.iron_condor import IronCondorStrategy
from options_backtest.strategy.short_strangle import ShortStrangleStrategy


class TestLongCall:
    def test_default_params(self):
        s = LongCallStrategy()
        assert s.name == "LongCall"
        assert s.target_delta == 0.40
        assert s.quantity == 1.0

    def test_custom_params(self):
        s = LongCallStrategy(params={"target_delta": 0.30, "quantity": 5.0})
        assert s.target_delta == 0.30
        assert s.quantity == 5.0


class TestShortStrangle:
    def test_default_params(self):
        s = ShortStrangleStrategy()
        assert s.name == "ShortStrangle"
        assert s.target_delta == 0.25
        assert s.selection_mode == "otm"
        assert s.entry_weekdays == []

    def test_custom_params(self):
        s = ShortStrangleStrategy(params={
            "target_delta": 0.15,
            "target_call_delta": 0.18,
            "target_put_delta": 0.12,
            "max_delta_diff": 0.05,
            "selection_mode": "delta",
            "stop_loss_pct": 300,
            "entry_weekdays": [0, 2, 4],
        })
        assert s.target_delta == 0.15
        assert s.target_call_delta == 0.18
        assert s.target_put_delta == 0.12
        assert s.max_delta_diff == 0.05
        assert s.selection_mode == "delta"
        assert s.stop_loss_pct == 300
        assert s.entry_weekdays == [0, 2, 4]


class TestIronCondor:
    def test_default_params(self):
        s = IronCondorStrategy()
        assert s.name == "IronCondor"
        assert s.short_otm_pct == 0.10
        assert s.long_otm_pct == 0.15

    def test_custom_params(self):
        s = IronCondorStrategy(params={"short_otm_pct": 0.08, "long_otm_pct": 0.20, "quantity": 2.0})
        assert s.short_otm_pct == 0.08
        assert s.long_otm_pct == 0.20
        assert s.quantity == 2.0

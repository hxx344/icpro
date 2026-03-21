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

    def test_custom_params(self):
        s = ShortStrangleStrategy(params={"target_delta": 0.15, "stop_loss_pct": 300})
        assert s.target_delta == 0.15
        assert s.stop_loss_pct == 300


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

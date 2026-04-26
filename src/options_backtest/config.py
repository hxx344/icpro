"""Configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    """Backtest time range and general settings."""

    name: str = "Backtest"
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    time_step: str = "1h"
    underlying: str = "BTC"
    margin_mode: str = "USD"  # "USD" (USDT-style margin) | "coin" (Deribit inverse)
    use_bs_only: bool = False
    option_data_source: str = "auto"  # auto | market_data | options_hourly
    option_snapshot_pick: str = "close"  # open | close
    iv_mode: str = "fixed"  # fixed | surface | proxy
    fixed_iv: float = 0.60
    dvol_path: str = ""
    iv_proxy_lookback_days: int = 30
    iv_smile_slope: float = 0.35
    iv_term_slope: float = 0.20
    iv_min: float = 0.05
    iv_max: float = 3.00
    show_progress: bool = True


class AccountConfig(BaseModel):
    """Account / capital settings."""

    initial_balance: float = 1.0  # coin units (auto-converted to USD if margin_mode=USD)


class ExecutionConfig(BaseModel):
    """Order execution / fee settings (Deribit rates)."""

    slippage: float = 0.0001
    market_quote_spread_pct: float = 0.10
    require_touch_quote: bool = False
    require_real_quote_source: bool = True
    taker_fee: float = 0.0003
    maker_fee: float = 0.0003
    min_fee: float = 0.0003
    max_fee_pct: float = 0.125
    delivery_fee: float = 0.00015
    delivery_fee_max_pct: float = 0.0  # 0 = no cap; e.g. 0.10 = 10% of option value


class StrategyConfig(BaseModel):
    """Strategy selection and parameters."""

    name: str = "ShortStrangle"
    params: dict[str, Any] = Field(default_factory=dict)


class ReportConfig(BaseModel):
    """Report output settings."""

    output_dir: str = "./reports"
    generate_plots: bool = True


class Config(BaseModel):
    """Root configuration."""

    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    account: AccountConfig = Field(default_factory=AccountConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)

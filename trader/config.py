"""Configuration management – 配置管理.

Loads YAML config, merges with environment variables, and provides
typed access to all trader settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExchangeConfig:
    """Exchange API connection settings (Binance European Options)."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    base_url: str = ""
    timeout: int = 10
    account_currency: str = "USDT"
    simulate_private: bool = False

    def __post_init__(self):
        self.api_key = os.environ.get("BINANCE_API_KEY", self.api_key)
        self.api_secret = os.environ.get("BINANCE_API_SECRET", self.api_secret)

        if not self.base_url:
            self.base_url = (
                "https://testnet.binancefuture.com"
                if self.testnet
                else "https://eapi.binance.com"
            )


# Backward-compatible alias
DeribitConfig = ExchangeConfig


@dataclass
class StrategyConfig:
    """Iron Condor 0DTE strategy parameters."""
    underlying: str = "ETH"
    otm_pct: float = 0.08               # 8% OTM for short legs
    wing_width_pct: float = 0.02         # additional 2% for long legs (protection)
    entry_time_utc: str = "08:00"         # daily entry at HH:MM UTC
    quantity: float = 0.01               # base order quantity (in contracts)
    max_positions: int = 1               # max concurrent iron condors
    max_capital_pct: float = 0.30        # max 30% of account for positions
    compound: bool = True                # scale quantity to equity
    wait_for_midpoint: bool = False      # wait for spot to reach strike midpoint before entry


@dataclass
class StorageConfig:
    """Persistence settings."""
    db_path: str = "./data/trader.db"
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"


@dataclass
class MonitorConfig:
    """Runtime monitoring."""
    check_interval_sec: int = 60         # position check loop interval
    heartbeat_interval_sec: int = 300    # log heartbeat every 5 min
    equity_snapshot_interval_sec: int = 3600  # hourly equity snapshot


@dataclass
class ChaserConfig:
    """Limit-order chaser settings."""
    window_seconds: int = 1800          # 30 minutes total window
    poll_interval_sec: int = 60         # check / amend every 60 seconds
    tick_size_usdt: float = 0.01        # min price increment in USDT
    market_fallback_sec: int = 60       # switch to market order last N seconds
    max_amend_attempts: int = 180       # safety cap on re-pricing loops


@dataclass
class TraderConfig:
    """Top-level trader configuration."""
    name: str = "Iron Condor 0DTE +8%"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    chaser: ChaserConfig = field(default_factory=ChaserConfig)

    # Backward-compat alias so old code using cfg.deribit still works
    @property
    def deribit(self) -> ExchangeConfig:
        return self.exchange


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _merge_section(dc_instance: Any, section: dict | None) -> None:
    """Merge a dict into a dataclass instance (shallow)."""
    if not section:
        return
    for k, v in section.items():
        if hasattr(dc_instance, k):
            setattr(dc_instance, k, v)


def load_config(path: str | Path | None = None) -> TraderConfig:
    """Load trader configuration from YAML file.

    Parameters
    ----------
    path : path to YAML config. If None, uses defaults.

    Returns
    -------
    TraderConfig instance with all settings merged.
    """
    cfg = TraderConfig()

    if path is not None:
        path = Path(path)
        if path.exists():
            logger.info(f"Loading config from {path}")
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "name" in raw:
                cfg.name = raw["name"]

            # Support both 'exchange' (new) and 'deribit' (legacy) keys
            raw_exchange = raw.get("exchange") or raw.get("deribit")
            _merge_section(cfg.exchange, raw_exchange)
            _merge_section(cfg.strategy, raw.get("strategy"))
            _merge_section(cfg.storage, raw.get("storage"))
            _merge_section(cfg.monitor, raw.get("monitor"))
            _merge_section(cfg.chaser, raw.get("chaser"))

            # Map legacy deribit field names to new ones
            if hasattr(cfg.exchange, "client_id") and not cfg.exchange.api_key:
                cfg.exchange.api_key = getattr(cfg.exchange, "client_id", "")
            if hasattr(cfg.exchange, "client_secret") and not cfg.exchange.api_secret:
                cfg.exchange.api_secret = getattr(cfg.exchange, "client_secret", "")

            # If YAML did not explicitly set base_url, clear it so
            # __post_init__ recalculates from the (possibly changed) testnet flag.
            if not (raw_exchange or {}).get("base_url"):
                cfg.exchange.base_url = ""
        else:
            logger.warning(f"Config file not found: {path}, using defaults")

    # Re-trigger __post_init__ for env vars and base_url
    cfg.exchange.__post_init__()

    return cfg

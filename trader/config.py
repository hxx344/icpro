"""Configuration management – 配置管理.

Loads YAML config, merges with environment variables, and provides
typed access to all trader settings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _load_env_file(env_path: Path) -> bool:
    """Load .env key-values into process env (without overriding existing vars)."""
    if not env_path.exists():
        return False

    loaded_any = False
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value)

        if not key:
            continue

        if key not in os.environ:
            os.environ[key] = value
            loaded_any = True

    return loaded_any


def _auto_load_dotenv(config_path: Path | None) -> None:
    """Load .env from a single controlled location.

    Precedence:
    1. `PROJECT_OPTIONS_ENV_FILE` if explicitly provided
    2. project-root `.env`
    """
    explicit_env = os.environ.get("PROJECT_OPTIONS_ENV_FILE", "").strip()
    candidates: list[Path] = []

    if explicit_env:
        candidates.append(Path(explicit_env))
    else:
        project_root = Path(__file__).resolve().parent.parent
        candidates.append(project_root / ".env")

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if _load_env_file(p):
            logger.info(f"Loaded environment from {p}")
            return


@dataclass
class ExchangeConfig:
    """Exchange API connection settings (Binance USD margin)."""
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


@dataclass
class StrategyConfig:
    """Options selling strategy parameters.

    Modes:
      - strangle      : 2-leg naked sell (OTM%-based strikes)
      - iron_condor   : 4-leg IC  (OTM%-based strikes)
      - weekend_vol   : Weekend vol selling IC (delta-based, Friday→Sunday)
    """
    mode: str = "strangle"               # "strangle" / "iron_condor" / "weekend_vol"
    underlying: str = "ETH"

    # --- OTM%-based strike selection (strangle / iron_condor) ---
    otm_pct: float = 0.10                # 10% OTM for short legs
    wing_width_pct: float = 0.02         # additional 2% for long legs
    target_dte_days: int = 7             # target DTE in days (0 = 0DTE)
    dte_window_hours: int = 48           # accept options within ±window of target DTE

    # --- Delta-based strike selection (weekend_vol) ---
    target_delta: float = 0.40           # |delta| for short legs
    wing_delta: float = 0.05             # |delta| for protection legs (0 = no wings)
    max_delta_diff: float = 0.20         # max allowed |actual_delta-target_delta|
    leverage: float = 1.0                # notional leverage multiplier
    entry_day: str = "friday"            # day of week to enter (lowercase)
    default_iv: float = 0.60             # fallback IV if mark_iv unavailable
    entry_realized_vol_lookback_hours: int = 0  # realized vol lookback in hours (0=off)
    entry_realized_vol_max: float = 0.0   # max allowed annualized RV for entry (0=off)
    stop_loss_pct: float = 0.0            # close all when basket pnl% <= -stop_loss_pct
    stop_loss_underlying_move_pct: float = 0.0  # require one-way underlying move >= pct before stop loss can fire

    # --- Common ---
    entry_time_utc: str = "08:00"         # entry at HH:MM UTC
    quantity: float = 0.01               # base order quantity (non-weekend_vol or weekend_vol非复利)
    max_positions: int = 1               # max concurrent positions
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
    """Spread-gated market execution settings."""
    window_seconds: int = 1800          # 30 minutes total window
    poll_interval_sec: int = 60         # check / amend every 60 seconds
    tick_size_usdt: float = 5.0         # min price increment in USD
    market_fallback_sec: int = 60       # switch to market order last N seconds
    market_trigger_spread_ticks: int = 1  # trigger market when all legs spread <= N ticks
    max_amend_attempts: int = 180       # safety cap on re-pricing loops


@dataclass
class TraderConfig:
    """Top-level trader configuration."""
    name: str = "Short Strangle 7DTE ±10%"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    chaser: ChaserConfig = field(default_factory=ChaserConfig)




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


def _validate_config(cfg: TraderConfig) -> None:
    """Validate critical config fields early so startup fails fast."""
    allowed_modes = {"strangle", "iron_condor", "weekend_vol"}
    if cfg.strategy.mode not in allowed_modes:
        raise ValueError(f"strategy.mode must be one of {sorted(allowed_modes)}, got {cfg.strategy.mode!r}")

    if not re.match(r"^\d{2}:\d{2}$", cfg.strategy.entry_time_utc):
        raise ValueError("strategy.entry_time_utc must be in HH:MM format")
    hh, mm = [int(x) for x in cfg.strategy.entry_time_utc.split(":", 1)]
    if hh not in range(24) or mm not in range(60):
        raise ValueError("strategy.entry_time_utc must be a valid UTC time")

    if not (0 < cfg.strategy.target_delta <= 0.95):
        raise ValueError("strategy.target_delta must be within (0, 0.95]")
    if not (0 <= cfg.strategy.wing_delta <= 0.95):
        raise ValueError("strategy.wing_delta must be within [0, 0.95]")
    if not (0 <= cfg.strategy.max_delta_diff <= 0.95):
        raise ValueError("strategy.max_delta_diff must be within [0, 0.95]")
    if cfg.strategy.leverage <= 0:
        raise ValueError("strategy.leverage must be > 0")
    if cfg.strategy.mode != "weekend_vol" and cfg.strategy.quantity <= 0:
        raise ValueError("strategy.quantity must be > 0")
    if cfg.strategy.max_positions < 1:
        raise ValueError("strategy.max_positions must be >= 1")
    if not (0 < cfg.strategy.max_capital_pct <= 1.0):
        raise ValueError("strategy.max_capital_pct must be within (0, 1]")
    if cfg.strategy.default_iv <= 0:
        raise ValueError("strategy.default_iv must be > 0")
    if cfg.strategy.entry_realized_vol_lookback_hours < 0:
        raise ValueError("strategy.entry_realized_vol_lookback_hours must be >= 0")
    if cfg.strategy.entry_realized_vol_lookback_hours == 1:
        raise ValueError("strategy.entry_realized_vol_lookback_hours must be 0 or >= 2")
    if cfg.strategy.entry_realized_vol_max < 0:
        raise ValueError("strategy.entry_realized_vol_max must be >= 0")
    if cfg.strategy.stop_loss_pct < 0:
        raise ValueError("strategy.stop_loss_pct must be >= 0")
    if cfg.strategy.stop_loss_underlying_move_pct < 0:
        raise ValueError("strategy.stop_loss_underlying_move_pct must be >= 0")
    if cfg.strategy.target_dte_days < 0:
        raise ValueError("strategy.target_dte_days must be >= 0")
    if cfg.strategy.dte_window_hours <= 0:
        raise ValueError("strategy.dte_window_hours must be > 0")

    if cfg.exchange.timeout < 1:
        raise ValueError("exchange.timeout must be >= 1")
    if cfg.monitor.check_interval_sec < 1:
        raise ValueError("monitor.check_interval_sec must be >= 1")
    if cfg.monitor.heartbeat_interval_sec < 1:
        raise ValueError("monitor.heartbeat_interval_sec must be >= 1")
    if cfg.monitor.equity_snapshot_interval_sec < 1:
        raise ValueError("monitor.equity_snapshot_interval_sec must be >= 1")

    if cfg.chaser.window_seconds <= 0:
        raise ValueError("chaser.window_seconds must be > 0")
    if cfg.chaser.poll_interval_sec <= 0:
        raise ValueError("chaser.poll_interval_sec must be > 0")
    if cfg.chaser.tick_size_usdt <= 0:
        raise ValueError("chaser.tick_size_usdt must be > 0")
    if cfg.chaser.market_fallback_sec < 0:
        raise ValueError("chaser.market_fallback_sec must be >= 0")
    if cfg.chaser.market_fallback_sec >= cfg.chaser.window_seconds:
        raise ValueError("chaser.market_fallback_sec must be smaller than window_seconds")
    if cfg.chaser.market_trigger_spread_ticks < 1:
        raise ValueError("chaser.market_trigger_spread_ticks must be >= 1")
    if cfg.chaser.max_amend_attempts < 1:
        raise ValueError("chaser.max_amend_attempts must be >= 1")

    if not str(cfg.storage.db_path).strip():
        raise ValueError("storage.db_path must not be empty")
    if not str(cfg.storage.log_dir).strip():
        raise ValueError("storage.log_dir must not be empty")


def load_config(path: str | Path | None = None) -> TraderConfig:
    """Load trader configuration from YAML file.

    Parameters
    ----------
    path : path to YAML config. If None, uses defaults.

    Returns
    -------
    TraderConfig instance with all settings merged.
    """
    path_obj = Path(path) if path is not None else None
    _auto_load_dotenv(path_obj)

    cfg = TraderConfig()

    if path_obj is not None:
        if path_obj.exists():
            logger.info(f"Loading config from {path_obj}")
            with open(path_obj, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "name" in raw:
                cfg.name = raw["name"]

            raw_exchange = raw.get("exchange")
            _merge_section(cfg.exchange, raw_exchange)
            _merge_section(cfg.strategy, raw.get("strategy"))
            _merge_section(cfg.storage, raw.get("storage"))
            _merge_section(cfg.monitor, raw.get("monitor"))
            _merge_section(cfg.chaser, raw.get("chaser"))

            # If YAML did not explicitly set base_url, clear it so
            # __post_init__ recalculates from the (possibly changed) testnet flag.
            if not (raw_exchange or {}).get("base_url"):
                cfg.exchange.base_url = ""
        else:
            logger.warning(f"Config file not found: {path_obj}, using defaults")

    # Re-trigger __post_init__ for env vars and base_url
    cfg.exchange.__post_init__()
    _validate_config(cfg)

    return cfg

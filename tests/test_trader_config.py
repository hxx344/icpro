"""Tests for trader.config — 配置管理单元测试.

覆盖:
  - ExchangeConfig 默认值 & __post_init__
  - 环境变量覆盖
  - load_config YAML 加载
  - 缺失文件处理
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from trader.config import (
    ExchangeConfig,
    StrategyConfig,
    StorageConfig,
    MonitorConfig,
    TraderConfig,
    load_config,
)


# ======================================================================
# 1. ExchangeConfig
# ======================================================================


class TestExchangeConfig:
    def test_defaults(self):
        cfg = ExchangeConfig()
        assert cfg.testnet is False
        assert cfg.timeout == 15
        assert cfg.account_currency == "USDT"
        assert cfg.simulate_private is False

    def test_testnet_url(self):
        cfg = ExchangeConfig(testnet=True)
        assert "testnet" in cfg.base_url

    def test_prod_url(self):
        cfg = ExchangeConfig(testnet=False)
        assert "eapi.binance.com" in cfg.base_url

    def test_custom_base_url_not_overwritten(self):
        """显式指定 base_url 时不被 __post_init__ 覆盖."""
        cfg = ExchangeConfig(base_url="https://custom.api.com", testnet=True)
        assert cfg.base_url == "https://custom.api.com"

    def test_env_vars_override(self, monkeypatch):
        """环境变量 BINANCE_API_KEY / BINANCE_API_SECRET 应覆盖字段."""
        monkeypatch.setenv("BINANCE_API_KEY", "env_key_123")
        monkeypatch.setenv("BINANCE_API_SECRET", "env_secret_456")
        cfg = ExchangeConfig(api_key="yaml_key", api_secret="yaml_secret")
        assert cfg.api_key == "env_key_123"
        assert cfg.api_secret == "env_secret_456"

    def test_env_vars_not_set(self, monkeypatch):
        """环境变量不存在时保留 YAML 值."""
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
        cfg = ExchangeConfig(api_key="from_yaml", api_secret="from_yaml_secret")
        assert cfg.api_key == "from_yaml"
        assert cfg.api_secret == "from_yaml_secret"


# ======================================================================
# 2. StrategyConfig / StorageConfig / MonitorConfig
# ======================================================================


class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig()
        assert cfg.mode == "weekend_vol"
        assert cfg.underlying == "BTC"
        assert cfg.target_delta == 0.45
        assert cfg.wing_delta == 0.0
        assert cfg.entry_time_utc == "18:00"
        assert cfg.compound is True


class TestStorageConfig:
    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.log_level == "INFO"


class TestMonitorConfig:
    def test_defaults(self):
        cfg = MonitorConfig()
        assert cfg.check_interval_sec == 5
        assert cfg.heartbeat_interval_sec == 300


# ======================================================================
# 3. TraderConfig
# ======================================================================


class TestTraderConfig:
    def test_defaults(self):
        cfg = TraderConfig()
        assert isinstance(cfg.exchange, ExchangeConfig)
        assert isinstance(cfg.strategy, StrategyConfig)


# ======================================================================
# 4. load_config
# ======================================================================


class TestLoadConfig:
    def test_load_none(self):
        """path=None 返回全部默认值."""
        cfg = load_config(None)
        assert cfg.name == "Weekend Vol BTC (Binance 3x USD)"
        assert cfg.exchange.testnet is False

    def test_load_missing_file(self, tmp_path):
        """文件不存在时使用默认值 (不报错)."""
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.name == "Weekend Vol BTC (Binance 3x USD)"

    def test_load_yaml(self, tmp_path, monkeypatch):
        """从 YAML 加载配置并合并."""
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

        config_data = {
            "name": "Test Strategy",
            "exchange": {
                "api_key": "yaml_key",
                "api_secret": "yaml_secret",
                "testnet": False,
                "account_currency": "USDT",
            },
            "strategy": {
                "underlying": "BTC",
                "target_delta": 0.35,
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(yaml_path)
        assert cfg.name == "Test Strategy"
        assert cfg.exchange.api_key == "yaml_key"
        assert cfg.exchange.testnet is False
        assert "eapi.binance.com" in cfg.exchange.base_url
        assert cfg.strategy.underlying == "BTC"
        assert cfg.strategy.target_delta == 0.35

    def test_load_empty_yaml(self, tmp_path):
        """空 YAML 文件不报错."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = load_config(yaml_path)
        assert cfg.name == "Weekend Vol BTC (Binance 3x USD)"

    def test_all_sections_merge(self, tmp_path, monkeypatch):
        """所有配置节都能正确合并."""
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

        config_data = {
            "strategy": {"quantity": 0.5, "max_positions": 3},
            "storage": {"log_level": "DEBUG"},
            "monitor": {"check_interval_sec": 30},
        }
        yaml_path = tmp_path / "full.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(yaml_path)
        assert cfg.strategy.quantity == 0.5
        assert cfg.strategy.max_positions == 3
        assert cfg.storage.log_level == "DEBUG"
        assert cfg.monitor.check_interval_sec == 30

    @pytest.mark.parametrize(
        "config_data, expected_message",
        [
            ({"strategy": {"mode": "bad_mode"}}, "strategy.mode"),
            ({"strategy": {"entry_time_utc": "25:00"}}, "entry_time_utc"),
            ({"strategy": {"compound": False, "quantity": 0}}, "strategy.quantity"),
            ({"chaser": {"window_seconds": 10, "market_fallback_sec": 10}}, "market_fallback_sec"),
        ],
    )
    def test_invalid_config_raises(self, tmp_path, config_data, expected_message):
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text(yaml.dump(config_data), encoding="utf-8")

        with pytest.raises(ValueError, match=expected_message):
            load_config(yaml_path)

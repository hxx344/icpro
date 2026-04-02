"""Deribit Options Strategy Backtesting System."""

from __future__ import annotations

import os
import sys

from loguru import logger

__version__ = "0.1.0"


def _configure_default_logging() -> None:
	"""Configure package-wide logging once with a sane default level.

	Loguru defaults to ``DEBUG``, which is very expensive for backtests because
	fills and settlements generate大量日志。默认改为 ``INFO``，并允许通过
	``OPTIONS_BT_LOG_LEVEL`` 覆盖。
	"""
	level = str(os.getenv("OPTIONS_BT_LOG_LEVEL", "INFO") or "INFO").upper()
	logger.remove()
	logger.add(sys.stderr, level=level)


_configure_default_logging()

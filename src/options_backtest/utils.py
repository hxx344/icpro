"""Shared utility helpers."""

from __future__ import annotations

import pandas as pd


def to_utc_timestamp(val) -> pd.Timestamp:
    """Convert *val* to a timezone‑aware UTC ``pd.Timestamp``.

    Handles datetime, Timestamp (with or without tz), and date‑strings.
    """
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

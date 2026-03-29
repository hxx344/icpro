#!/usr/bin/env python3
"""
build_hourly_parquet.py
=======================
Download Deribit options_chain tick-level .zst files from remote server,
resample to hourly snapshots (last tick per hour per symbol), and save as
compact Parquet files partitioned by underlying / year-month.

Source: https://data.yutsing.work/0324-charles/tardis/deribit/options_chain/
Format: {base}/{YYYY}/{MM}/{DD}/OPTIONS.csv.zst  (~922 MB each, ~5.68 GB decompressed)

Output: data/options_hourly/{BTC,ETH}/{YYYY-MM}.parquet
        ~1-1.5 GB total for 3 years

Usage examples:
    # Process all available dates (2023-03 to 2025-12)
    python scripts/build_hourly_parquet.py

    # Process a specific date range
    python scripts/build_hourly_parquet.py --start 2023-03-20 --end 2023-03-25

    # Process a local .zst file (for testing)
    python scripts/build_hourly_parquet.py --local tmp/OPTIONS.csv.zst

    # Dry run - list available dates without processing
    python scripts/build_hourly_parquet.py --dry-run

    # Skip download, only process already-downloaded local cache
    python scripts/build_hourly_parquet.py --cache-only
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import date, datetime, timedelta, timezone
import queue
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import requests
import zstandard as zstd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://data.yutsing.work/0324-charles/tardis/deribit/options_chain"
OUTPUT_DIR = Path("data/options_hourly")
CACHE_DIR = Path("tmp/zst_cache")
MIN_DAILY_AVAILABILITY_PCT = 90.0

# Columns to read from CSV (skip exchange, local_timestamp, rho)
USE_COLS = [
    "timestamp",       # epoch μs
    "symbol",
    "type",            # call / put
    "strike_price",
    "expiration",      # epoch μs
    "open_interest",
    "last_price",
    "bid_price",
    "bid_amount",
    "bid_iv",
    "ask_price",
    "ask_amount",
    "ask_iv",
    "mark_price",
    "mark_iv",
    "underlying_index",
    "underlying_price",
    "delta",
    "gamma",
    "vega",
    "theta",
]

# PyArrow column types for fast CSV parsing
PA_COLUMN_TYPES = {
    "timestamp": pa.int64(),
    "symbol": pa.utf8(),
    "type": pa.utf8(),
    "strike_price": pa.float32(),
    "expiration": pa.int64(),
    "open_interest": pa.float32(),
    "last_price": pa.float64(),
    "bid_price": pa.float64(),
    "bid_amount": pa.float64(),
    "bid_iv": pa.float32(),
    "ask_price": pa.float64(),
    "ask_amount": pa.float64(),
    "ask_iv": pa.float32(),
    "mark_price": pa.float64(),
    "mark_iv": pa.float32(),
    "underlying_index": pa.utf8(),
    "underlying_price": pa.float64(),
    "delta": pa.float32(),
    "gamma": pa.float32(),
    "vega": pa.float32(),
    "theta": pa.float32(),
}

# Fallback pandas dtypes (for --local mode compatibility)
DTYPES = {
    "exchange": "category",
    "symbol": "category",
    "type": "category",
    "underlying_index": "category",
    "strike_price": "float32",
    "open_interest": "float32",
    "last_price": "float64",
    "bid_price": "float64",
    "bid_amount": "float64",
    "bid_iv": "float32",
    "ask_price": "float64",
    "ask_amount": "float64",
    "ask_iv": "float32",
    "mark_price": "float64",
    "mark_iv": "float32",
    "underlying_price": "float64",
    "delta": "float32",
    "gamma": "float32",
    "vega": "float32",
    "theta": "float32",
    "rho": "float32",
}

PA_BLOCK_SIZE = 256 * 1024 * 1024  # 256 MB per PyArrow CSV read block

log = logging.getLogger("build_hourly")

# Runtime CPU/thread controls
_cpu_limit: Optional[int] = None
_stream_merge_every_batches: int = 2
_mem_soft_limit_mb: int = 0
_mem_hard_limit_mb: int = 0
_mem_diagnostics: bool = False
_mem_diag_every_batches: int = 2
_mem_peak_interval_ms: int = 100
_pandas_slice_rows: int = 0
_stream_queue_depth: int = 2
_csv_aggressive_parse: bool = False
_polars_stream_chunk_mb: int = 64
_mem_diag_records: List[Dict[str, Any]] = []
_mem_diag_lock = threading.Lock()


def _apply_cpu_limits(cpu_limit: Optional[int]):
    """Apply process-level CPU/thread limits for Arrow/BLAS stacks.

    This keeps peak CPU usage controllable on shared/dev machines.
    """
    global _cpu_limit
    _cpu_limit = cpu_limit if cpu_limit and cpu_limit > 0 else None
    if _cpu_limit is None:
        return

    n = str(_cpu_limit)
    os.environ["ARROW_NUM_THREADS"] = n
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n

    try:
        pa.set_cpu_count(_cpu_limit)
    except Exception:
        pass

    log.info("CPU limit enabled: %d thread(s)", _cpu_limit)


def _log_mem(label: str = ""):
    """Log current process RSS (resident set size) in MB."""
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        log.info("  [MEM] %s: %.0f MB RSS", label, rss)
    except ImportError:
        pass  # psutil not installed — silently skip


def _get_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _get_available_mem_mb() -> Optional[float]:
    """Return currently available system memory in MB, or None if unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None


def _is_memory_pressure_error(exc: Exception) -> bool:
    """Best-effort check whether an exception indicates memory pressure."""
    if isinstance(exc, MemoryError):
        return True
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if "memory" in name or "alloc" in name:
        return True
    markers = [
        "out of memory",
        "cannot allocate",
        "std::bad_alloc",
        "memoryerror",
        "malloc",
    ]
    return any(m in msg for m in markers)


def _get_arrow_pool_mb() -> Optional[float]:
    """Return current PyArrow memory-pool allocation in MB, or None."""
    try:
        if hasattr(pa, "total_allocated_bytes"):
            return pa.total_allocated_bytes() / (1024 * 1024)
        pool = pa.default_memory_pool()
        return pool.bytes_allocated() / (1024 * 1024)
    except Exception:
        return None


def _mem_diag_point(label: str, **fields):
    """Emit detailed memory diagnostics when enabled."""
    if not _mem_diagnostics:
        return

    rss = _get_rss_mb()
    arrow_mb = _get_arrow_pool_mb()
    parts: List[str] = []
    if rss is not None:
        parts.append(f"rss={rss:.0f}MB")
    if arrow_mb is not None:
        parts.append(f"arrow={arrow_mb:.0f}MB")
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    detail = " ".join(parts) if parts else "mem-unavailable"
    log.info("  [MEM-DIAG] %s %s", label, detail)

    with _mem_diag_lock:
        _mem_diag_records.append({
            "ts": time.time(),
            "label": label,
            "rss": rss,
            "arrow": arrow_mb,
            "fields": dict(fields),
        })


def _mem_diag_delta(label: str, rss_before: Optional[float], arrow_before: Optional[float], **fields):
    """Emit memory delta diagnostics when enabled."""
    if not _mem_diagnostics:
        return

    rss_after = _get_rss_mb()
    arrow_after = _get_arrow_pool_mb()
    extra: Dict[str, str] = {}
    if rss_before is not None and rss_after is not None:
        extra["rss_delta_mb"] = f"{rss_after - rss_before:+.0f}"
    if arrow_before is not None and arrow_after is not None:
        extra["arrow_delta_mb"] = f"{arrow_after - arrow_before:+.0f}"
    extra.update({k: str(v) for k, v in fields.items()})
    _mem_diag_point(label, **extra)


def _reset_mem_diag_records():
    with _mem_diag_lock:
        _mem_diag_records.clear()


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("MB", "").replace("mb", "").replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def _report_mem_diag_hotspots(context: str = ""):
    if not _mem_diagnostics:
        return

    with _mem_diag_lock:
        records = list(_mem_diag_records)

    if not records:
        return

    rss_records = [r for r in records if r.get("rss") is not None]
    arrow_records = [r for r in records if r.get("arrow") is not None]

    peak_rss = max(rss_records, key=lambda r: r["rss"]) if rss_records else None
    peak_arrow = max(arrow_records, key=lambda r: r["arrow"]) if arrow_records else None

    delta_by_label: Dict[str, Dict[str, float]] = {}
    peak_delta_by_label: Dict[str, Dict[str, float]] = {}
    for r in records:
        label = str(r.get("label", "unknown"))
        fields = r.get("fields", {}) or {}

        rss_delta = _to_float(fields.get("rss_delta_mb"))
        if rss_delta is not None and rss_delta > 0:
            d = delta_by_label.setdefault(label, {"sum": 0.0, "max": 0.0, "count": 0.0})
            d["sum"] += rss_delta
            d["max"] = max(d["max"], rss_delta)
            d["count"] += 1.0

        rss_peak_delta = _to_float(fields.get("rss_peak_delta_mb"))
        if rss_peak_delta is not None and rss_peak_delta > 0:
            d2 = peak_delta_by_label.setdefault(label, {"max": 0.0, "count": 0.0})
            d2["max"] = max(d2["max"], rss_peak_delta)
            d2["count"] += 1.0

    top_delta = sorted(
        delta_by_label.items(),
        key=lambda kv: (kv[1]["max"], kv[1]["sum"]),
        reverse=True,
    )[:3]
    top_peak_delta = sorted(
        peak_delta_by_label.items(),
        key=lambda kv: kv[1]["max"],
        reverse=True,
    )[:3]

    wall = timer.wall_elapsed
    top_stages = sorted(timer._totals.items(), key=lambda kv: kv[1], reverse=True)[:3]

    prefix = f"{context} " if context else ""
    log.info("  [MEM-HOTSPOT] %ssummary: %d mem events", prefix, len(records))
    if peak_rss is not None:
        log.info(
            "  [MEM-HOTSPOT] %speak RSS: %.0f MB @ %s",
            prefix,
            peak_rss["rss"],
            peak_rss["label"],
        )
    if peak_arrow is not None:
        log.info(
            "  [MEM-HOTSPOT] %speak Arrow pool: %.0f MB @ %s",
            prefix,
            peak_arrow["arrow"],
            peak_arrow["label"],
        )

    if top_peak_delta:
        for idx, (label, stats) in enumerate(top_peak_delta, start=1):
            log.info(
                "  [MEM-HOTSPOT] %speak-driver #%d: %s (max +%.0f MB, samples=%d)",
                prefix,
                idx,
                label,
                stats["max"],
                int(stats["count"]),
            )

    if top_delta:
        for idx, (label, stats) in enumerate(top_delta, start=1):
            log.info(
                "  [MEM-HOTSPOT] %sdelta-driver #%d: %s (max +%.0f MB, sum +%.0f MB, hits=%d)",
                prefix,
                idx,
                label,
                stats["max"],
                stats["sum"],
                int(stats["count"]),
            )

    if top_stages and wall > 0:
        for idx, (stage, sec) in enumerate(top_stages, start=1):
            pct = sec / wall * 100
            log.info(
                "  [EFF-HOTSPOT] %stime-driver #%d: %s %.1fs (%.1f%% wall)",
                prefix,
                idx,
                stage,
                sec,
                pct,
            )


class _PeakSampler:
    """Sample RSS/Arrow memory at fixed interval and report stage peak."""

    def __init__(
        self,
        label: str,
        enabled: bool,
        interval_ms: int = 100,
        **fields,
    ):
        self.label = label
        self.enabled = enabled
        self.interval_sec = max(0.02, interval_ms / 1000.0)
        self.fields = {k: str(v) for k, v in fields.items()}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.rss_start: Optional[float] = None
        self.rss_peak: Optional[float] = None
        self.rss_end: Optional[float] = None
        self.arrow_start: Optional[float] = None
        self.arrow_peak: Optional[float] = None
        self.arrow_end: Optional[float] = None

    def _sample_once(self):
        rss = _get_rss_mb()
        arr = _get_arrow_pool_mb()

        if rss is not None:
            if self.rss_peak is None or rss > self.rss_peak:
                self.rss_peak = rss
        if arr is not None:
            if self.arrow_peak is None or arr > self.arrow_peak:
                self.arrow_peak = arr

    def start(self):
        if not self.enabled:
            return

        self.rss_start = _get_rss_mb()
        self.arrow_start = _get_arrow_pool_mb()
        self.rss_peak = self.rss_start
        self.arrow_peak = self.arrow_start

        def _loop():
            while not self._stop_event.wait(self.interval_sec):
                self._sample_once()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self, **fields):
        if not self.enabled:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        self._sample_once()
        self.rss_end = _get_rss_mb()
        self.arrow_end = _get_arrow_pool_mb()

        out: Dict[str, str] = dict(self.fields)
        out.update({k: str(v) for k, v in fields.items()})

        if self.rss_start is not None:
            out["rss_start_mb"] = f"{self.rss_start:.0f}"
        if self.rss_peak is not None:
            out["rss_peak_mb"] = f"{self.rss_peak:.0f}"
        if self.rss_end is not None:
            out["rss_end_mb"] = f"{self.rss_end:.0f}"
        if self.rss_start is not None and self.rss_peak is not None:
            out["rss_peak_delta_mb"] = f"{self.rss_peak - self.rss_start:+.0f}"

        if self.arrow_start is not None:
            out["arrow_start_mb"] = f"{self.arrow_start:.0f}"
        if self.arrow_peak is not None:
            out["arrow_peak_mb"] = f"{self.arrow_peak:.0f}"
        if self.arrow_end is not None:
            out["arrow_end_mb"] = f"{self.arrow_end:.0f}"
        if self.arrow_start is not None and self.arrow_peak is not None:
            out["arrow_peak_delta_mb"] = f"{self.arrow_peak - self.arrow_start:+.0f}"

        _mem_diag_point(f"{self.label}.peak", **out)


def _auto_memory_limits() -> Tuple[int, int]:
    """Return (soft_mb, hard_mb) memory thresholds based on host RAM.

    Soft threshold starts cleanup pressure; hard threshold adds brief sleeps.
    """
    try:
        import psutil
        total_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        soft = int(total_mb * 0.70)
        hard = int(total_mb * 0.85)
        if hard <= soft:
            hard = soft + 512
        return soft, hard
    except Exception:
        # Conservative fallback for unknown environments
        return 8192, 10240


def _maybe_memory_backpressure(label: str = ""):
    """Apply memory backpressure based on configured RSS thresholds."""
    if _mem_soft_limit_mb <= 0 and _mem_hard_limit_mb <= 0:
        return

    rss = _get_rss_mb()
    if rss is None:
        return

    if _mem_soft_limit_mb > 0 and rss >= _mem_soft_limit_mb:
        log.info(
            "  [MEM] %s RSS %.0f MB >= soft %d MB, releasing memory",
            label, rss, _mem_soft_limit_mb,
        )
        _release_memory()
        rss = _get_rss_mb() or rss

    if _mem_hard_limit_mb > 0 and rss >= _mem_hard_limit_mb:
        log.warning(
            "  [MEM] %s RSS %.0f MB >= hard %d MB, throttling briefly",
            label, rss, _mem_hard_limit_mb,
        )
        _release_memory()
        time.sleep(0.25)


def _release_memory():
    """Aggressively free Python + PyArrow memory back to the OS."""
    gc.collect()
    # PyArrow keeps a memory pool that doesn't auto-return to OS
    try:
        pool = pa.default_memory_pool()
        pool.release_unused()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Timing profiler
# ---------------------------------------------------------------------------


class StageTimer:
    """Accumulate wall-clock time across named processing stages."""

    def __init__(self):
        self._totals: Dict[str, float] = {}   # stage → cumulative seconds
        self._counts: Dict[str, int] = {}     # stage → invocation count
        self._rows: Dict[str, int] = {}       # stage → cumulative rows
        self._bytes: Dict[str, int] = {}      # stage → cumulative bytes
        self._active: Dict[int, Tuple[str, float]] = {}  # tid -> (stage, start)
        self._lock = threading.Lock()
        self._wall_start: float = time.time()

    def start(self, stage: str):
        tid = threading.get_ident()
        now = time.time()
        with self._lock:
            self._active[tid] = (stage, now)

    def stop(self, rows: int = 0, nbytes: int = 0):
        tid = threading.get_ident()
        now = time.time()
        with self._lock:
            active = self._active.pop(tid, None)
            if active is None:
                return
            s, started = active
            elapsed = now - started
            self._totals[s] = self._totals.get(s, 0.0) + elapsed
            self._counts[s] = self._counts.get(s, 0) + 1
            self._rows[s] = self._rows.get(s, 0) + rows
            self._bytes[s] = self._bytes.get(s, 0) + nbytes

    # -- serialisation for subprocess isolation -------------------------
    def to_json(self) -> str:
        """Serialise accumulated data to a JSON string."""
        return json.dumps({
            "totals": self._totals,
            "counts": self._counts,
            "rows": self._rows,
            "bytes": self._bytes,
        })

    def merge(self, raw):
        """Merge timer data (JSON string or dict) from a child process."""
        data = json.loads(raw) if isinstance(raw, str) else raw
        for k, v in data.get("totals", {}).items():
            self._totals[k] = self._totals.get(k, 0.0) + v
        for k, v in data.get("counts", {}).items():
            self._counts[k] = self._counts.get(k, 0) + v
        for k, v in data.get("rows", {}).items():
            self._rows[k] = self._rows.get(k, 0) + v
        for k, v in data.get("bytes", {}).items():
            self._bytes[k] = self._bytes.get(k, 0) + v

    @property
    def wall_elapsed(self) -> float:
        return time.time() - self._wall_start

    def report(self) -> str:
        """Return a formatted timing breakdown table."""
        wall = self.wall_elapsed
        lines = []
        lines.append("")
        lines.append("=" * 72)
        lines.append("  TIMING BREAKDOWN")
        lines.append("=" * 72)
        lines.append(f"  {'Stage':<22} {'Time':>8} {'%':>6} {'Count':>6} {'Rows':>12} {'Throughput':>14}")
        lines.append(f"  {'-'*22} {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")

        for stage in self._totals:
            t = self._totals[stage]
            pct = t / wall * 100 if wall > 0 else 0
            cnt = self._counts[stage]
            rows = self._rows[stage]
            nbytes = self._bytes[stage]

            # Throughput
            if nbytes > 0 and t > 0:
                tp = f"{nbytes / 1024 / 1024 / t:.0f} MB/s"
            elif rows > 0 and t > 0:
                tp = f"{rows / 1000 / t:.0f}k rows/s"
            else:
                tp = ""

            rows_str = f"{rows:,}" if rows else ""
            lines.append(
                f"  {stage:<22} {t:>7.1f}s {pct:>5.1f}% {cnt:>6} {rows_str:>12} {tp:>14}"
            )

        # Unaccounted time
        accounted = sum(self._totals.values())
        other = wall - accounted
        if other > 0.5:
            pct = other / wall * 100
            lines.append(
                f"  {'(other/overhead)':<22} {other:>7.1f}s {pct:>5.1f}%"
            )

        lines.append(f"  {'-'*22} {'-'*8}")
        lines.append(f"  {'TOTAL WALL TIME':<22} {wall:>7.1f}s")
        lines.append("=" * 72)

        # Estimate full-run time
        if self._counts.get("decompress", 0) > 0:
            days_done = self._counts["decompress"]
            per_day = wall / days_done
            est_1000 = per_day * 1000
            h = int(est_1000 // 3600)
            m = int((est_1000 % 3600) // 60)
            lines.append(
                f"  Avg per day: {per_day:.1f}s  |  "
                f"Est. 1000 days: {h}h {m}m  ({per_day * 1000 / 3600:.1f}h)"
            )
            lines.append("=" * 72)

        return "\n".join(lines)


# Global timer instance (reset per run)
timer = StageTimer()


# ---------------------------------------------------------------------------
# Pipeline real-time status tracker
# ---------------------------------------------------------------------------


class _PipelineStatus:
    """Thread-safe tracker for concurrent download + serial process pipeline.

    Displays a compact status line like:
        [Pipeline] DL 5/100 (3 active: 04-01,04-02,04-03) | Proc 2/100 (04-01) | Queue 3
    """

    def __init__(self, total: int):
        self.total = total
        self._lock = threading.Lock()

        # Download tracking
        self.dl_done = 0                    # finished downloads
        self.dl_active: List[str] = []      # active keys (YYYY-MM-DD)
        self.dl_progress: Dict[str, float] = {}  # key -> 0..1
        self.dl_failed = 0

        # Process tracking
        self.proc_done = 0                  # finished processing
        self.proc_current: Optional[str] = None  # date being processed
        self.proc_failed = 0

        # Queue depth (updated from main thread)
        self.queue_depth = 0

        # Timing
        self._start_time = time.time()
        self._last_print = 0.0
        self._last_display_lines = 0
        self._ticker_stop = threading.Event()
        self._ticker_paused = threading.Event()  # set = paused (don't print)
        self._ticker_thread: Optional[threading.Thread] = None

    # -- download events (called from download threads) --
    def dl_start(self, d: date):
        with self._lock:
            key = d.isoformat()
            self.dl_active.append(key)
            self.dl_progress[key] = 0.0

    def dl_progress_update(self, d: date, downloaded: int, total: int):
        if total <= 0:
            return
        with self._lock:
            key = d.isoformat()
            if key in self.dl_progress:
                ratio = downloaded / total
                ratio = max(0.0, min(1.0, ratio))
                # Allow rollback (e.g. multi-seg fallback to single connection)
                self.dl_progress[key] = ratio

    def dl_finish(self, d: date, ok: bool = True):
        with self._lock:
            key = d.isoformat()
            if key in self.dl_active:
                self.dl_active.remove(key)
            self.dl_progress.pop(key, None)
            if ok:
                self.dl_done += 1
            else:
                self.dl_failed += 1

    # -- process events (called from main thread) --
    def proc_start(self, d: date):
        with self._lock:
            self.proc_current = d.strftime("%m-%d")

    def proc_finish(self, ok: bool = True):
        with self._lock:
            self.proc_current = None
            if ok:
                self.proc_done += 1
            else:
                self.proc_failed += 1

    def set_queue_depth(self, n: int):
        with self._lock:
            self.queue_depth = n

    # -- formatted status line --
    def _status_line(self) -> str:
        with self._lock:
            elapsed = time.time() - self._start_time
            h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
            time_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

            # -- Overall progress bar --
            proc_total = self.proc_done + self.proc_failed
            ratio = proc_total / self.total if self.total > 0 else 0
            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            pct = ratio * 100

            # ETA
            eta_str = ""
            if self.proc_done > 0:
                per_day = elapsed / self.proc_done
                remaining_days = self.total - proc_total
                eta_sec = per_day * remaining_days
                eh, em = int(eta_sec // 3600), int(eta_sec % 3600 // 60)
                eta_str = f"{eh}h{em:02d}m" if eh else f"{em}m"

            # Failures
            fail_str = ""
            total_fail = self.dl_failed + self.proc_failed
            if total_fail:
                fail_str = f"  \033[91m\u2716 {total_fail} failed\033[0m"

            # Speed (days/hour)
            speed_str = ""
            if self.proc_done > 0 and elapsed > 0:
                days_per_hour = self.proc_done / (elapsed / 3600)
                speed_str = f"  {days_per_hour:.1f} days/h"

            # Line 1: overall progress
            lines = []
            lines.append(
                f"\033[1m[{time_str}]\033[0m "
                f"{bar} {pct:5.1f}%  "
                f"\033[32m\u2714{self.proc_done}\033[0m/{self.total}"
                f"{fail_str}"
                f"{speed_str}"
                f"  ETA {eta_str}" if eta_str else
                f"\033[1m[{time_str}]\033[0m "
                f"{bar} {pct:5.1f}%  "
                f"\033[32m\u2714{self.proc_done}\033[0m/{self.total}"
                f"{fail_str}"
                f"{speed_str}"
            )

            # Line 2+: download status
            dl_total_done = self.dl_done + self.dl_failed
            lines.append(
                f"  \033[36mDL\033[0m {dl_total_done}/{self.total}  Q:{self.queue_depth}"
            )
            if self.dl_active:
                # Sort active downloads by progress (highest first)
                active_sorted = sorted(
                    self.dl_active,
                    key=lambda k: self.dl_progress.get(k, 0.0),
                    reverse=True,
                )
                for key in active_sorted:
                    p = self.dl_progress.get(key, 0.0)
                    label = key[5:]  # MM-DD
                    mini_len = 18
                    mini_filled = int(mini_len * p)
                    mini_bar = "\u2588" * mini_filled + "\u2591" * (mini_len - mini_filled)
                    lines.append(f"    {label} {mini_bar} {p*100:5.1f}%")
            else:
                lines.append("    (idle)")

            # Final line: processing status
            if self.proc_current:
                proc_line = f"  \033[33mProc\033[0m {self.proc_current} ..."
            else:
                proc_line = f"  \033[33mProc\033[0m idle"

            lines.append(proc_line)
            return "\n".join(lines)

    @property
    def _display_lines(self) -> int:
        """Number of lines the status display uses."""
        return max(1, self._last_display_lines)

    def print_status(self, force: bool = False):
        """Print multi-line status (throttled to every 2s unless *force*)."""
        now = time.time()
        if not force and (now - self._last_print) < 2.0:
            return
        # Move cursor up to overwrite previous status block
        n = self._display_lines
        if self._last_print > 0:
            # Move up N lines and clear each
            print(f"\033[{n}A", end="", flush=True)
        self._last_print = now
        block = self._status_line()
        self._last_display_lines = len(block.split("\n"))
        # Clear each line before printing
        for line in block.split("\n"):
            print(f"\033[K{line}", flush=True)

    def print_final(self):
        """Print final status (no overwrite)."""
        n = self._display_lines
        if self._last_print > 0:
            print(f"\033[{n}A", end="", flush=True)
        block = self._status_line()
        self._last_display_lines = len(block.split("\n"))
        for line in block.split("\n"):
            print(f"\033[K{line}", flush=True)

    # -- background ticker (auto-refresh every 2s) --
    def start_ticker(self):
        def _tick():
            while not self._ticker_stop.wait(2.0):
                if not self._ticker_paused.is_set():
                    self.print_status()
        self._ticker_thread = threading.Thread(target=_tick, daemon=True)
        self._ticker_thread.start()

    def stop_ticker(self):
        self._ticker_stop.set()
        if self._ticker_thread:
            self._ticker_thread.join(timeout=3)

    def pause_ticker(self):
        """Pause status line output (e.g. while subprocess is running)."""
        self._ticker_paused.set()
        # Clear the status block so subprocess output starts clean
        n = self._display_lines
        if self._last_print > 0:
            print(f"\033[{n}A", end="", flush=True)
            for _ in range(n):
                print(f"\033[K", flush=True)
            # Reset so resume_ticker won't cursor-up over subprocess output
            self._last_print = 0
            self._last_display_lines = 0

    def resume_ticker(self):
        """Resume status line output after subprocess finishes."""
        self._ticker_paused.clear()
        self.print_status(force=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


_CHILD_PROCS: set = set()
_CHILD_PROCS_LOCK = threading.Lock()
_SHUTDOWN_EVENT = threading.Event()
_SIGINT_COUNT = 0
_SIGINT_LOCK = threading.Lock()


def _register_child_process(proc: subprocess.Popen):
    with _CHILD_PROCS_LOCK:
        _CHILD_PROCS.add(proc)


def _unregister_child_process(proc: subprocess.Popen):
    with _CHILD_PROCS_LOCK:
        _CHILD_PROCS.discard(proc)


def _popen_tracked(cmd, **kwargs) -> subprocess.Popen:
    """Spawn subprocess in its own process group/session and track it."""
    if sys.platform == "win32":
        flags = kwargs.pop("creationflags", 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs.setdefault("start_new_session", True)

    proc = subprocess.Popen(cmd, **kwargs)
    _register_child_process(proc)
    return proc


def _terminate_process_tree(proc: subprocess.Popen, timeout: float = 3.0):
    """Terminate a subprocess and (best-effort) its children."""
    if proc.poll() is not None:
        _unregister_child_process(proc)
        return

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()

            try:
                proc.wait(timeout=timeout)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
    except Exception:
        pass
    finally:
        _unregister_child_process(proc)


def _terminate_all_children(timeout: float = 3.0):
    """Terminate all tracked child processes."""
    with _CHILD_PROCS_LOCK:
        children = list(_CHILD_PROCS)
    for proc in children:
        _terminate_process_tree(proc, timeout=timeout)


def _force_exit(code: int = 130):
    """Kill all child processes then exit immediately."""
    try:
        _terminate_all_children(timeout=2.0)

        # Windows: use taskkill to kill the entire process tree
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(os.getpid())],
                capture_output=True, timeout=5,
            )
        else:
            import signal as _sig
            os.killpg(os.getpgid(os.getpid()), _sig.SIGKILL)
    except Exception:
        pass
    os._exit(code)


def _handle_sigint(signum, frame):
    """Handle Ctrl+C: first press requests graceful stop, second press forces exit."""
    global _SIGINT_COUNT
    with _SIGINT_LOCK:
        _SIGINT_COUNT += 1
        count = _SIGINT_COUNT

    _SHUTDOWN_EVENT.set()
    if count <= 1:
        print(
            "\n\nCtrl+C detected, stopping... (press Ctrl+C again to force exit)",
            file=sys.stderr,
            flush=True,
        )
    else:
        print("\n\nForce exiting...", file=sys.stderr, flush=True)
        _force_exit(130)


def install_interrupt_handler():
    """Install process-wide SIGINT handler (single Ctrl+C exits whole program)."""
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _handle_sigint)
    except Exception:
        pass


def _wait_process_interruptible(proc: subprocess.Popen, timeout: float = 0) -> int:
    """Wait for process while remaining responsive to shutdown events/KeyboardInterrupt."""
    start = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc

        if _SHUTDOWN_EVENT.is_set():
            _terminate_process_tree(proc, timeout=1.0)
            raise KeyboardInterrupt()

        if timeout > 0 and (time.time() - start) > timeout:
            _terminate_process_tree(proc, timeout=1.0)
            raise subprocess.TimeoutExpired(proc.args, timeout)

        time.sleep(0.2)


def discover_dates(start: date, end: date) -> List[date]:
    """Generate all dates in [start, end]."""
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates

def discover_remote_dates_for_month(year: int, month: int) -> List[int]:
    """Scrape available day folders for a given year/month."""
    url = f"{BASE_URL}/{year:04d}/{month:02d}/"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        # Parse directory listing - look for day links like "20/"
        days = re.findall(r'href="(\d{1,2})/"', resp.text)
        return sorted(set(int(d) for d in days))
    except Exception as e:
        log.warning("Failed to list %s: %s", url, e)
        return []


def discover_all_remote_dates() -> List[date]:
    """Crawl the server to find all available dates."""
    log.info("Discovering available dates on remote server...")
    all_dates = []

    # Known range: 2023-03 to 2025-12 (plus possibly 2026)
    for year in range(2023, 2027):
        for month in range(1, 13):
            days = discover_remote_dates_for_month(year, month)
            if days:
                log.info("  %04d-%02d: %d days", year, month, len(days))
                for day in days:
                    try:
                        all_dates.append(date(year, month, day))
                    except ValueError:
                        pass

    log.info("Total available dates: %d", len(all_dates))
    return sorted(all_dates)


def file_url(d: date) -> str:
    return f"{BASE_URL}/{d.year:04d}/{d.month:02d}/{d.day:02d}/OPTIONS.csv.zst"


def cache_path(d: date) -> Path:
    return CACHE_DIR / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}.csv.zst"


def output_path(underlying: str, year: int, month: int) -> Path:
    return OUTPUT_DIR / underlying / f"{year:04d}-{month:02d}.parquet"


# ---------------------------------------------------------------------------
# Download — multi-segment curl with progress tracking
# ---------------------------------------------------------------------------

# Windows system curl uses Schannel (same TLS as browsers), which avoids
# ISP SNI-based throttling that affects OpenSSL-based tools.
_WIN_CURL = Path(r"C:\Windows\System32\curl.exe")
_HAS_WIN_CURL = _WIN_CURL.exists()
_HAS_CURL = _HAS_WIN_CURL or shutil.which("curl") is not None
_CURL_BIN = str(_WIN_CURL) if _HAS_WIN_CURL else "curl"
_ZSTD_BIN = shutil.which("zstd")

# Proxy setting (set via --proxy CLI or HTTPS_PROXY env)
_proxy: Optional[str] = None

# Number of parallel segments for downloading one file
_download_segments: int = 8

# CSV read strategy
_csv_read_mode: str = "auto"   # auto | bulk | stream
_csv_block_mb: int = 1024
_hybrid_chunk_mb: int = 1024
_hybrid_staging: str = "memory"  # memory | disk
_CSV_STREAM_AUTO_THRESHOLD_MB: int = 2200

_UA_VALUE = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/131.0.0.0 Safari/537.36")
_CURL_UA = f"User-Agent: {_UA_VALUE}"


# ---------------------------------------------------------------------------
# Network route detection (VPN / direct)
# ---------------------------------------------------------------------------

_route_label: Optional[str] = None   # cached result


def _detect_vpn_local() -> bool:
    """Detect VPN conservatively from local tunnel adapters only.

    Avoid process/proxy heuristics because they frequently cause false positives.
    """
    # Check for VPN / tunnel network adapters via ipconfig (always available)
    try:
        r = subprocess.run(
            ["ipconfig", "/all"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            output = r.stdout.lower()
            vpn_adapter_keywords = [
                "tap-windows", "tap-win", "tun ", "wintun",
                "wireguard", "wg tunnel", "letstap", "lets tap",
                "vpn", "clash", "singbox", "sing-box",
                "v2ray", "xray", "hysteria", "trojan",
                "shadowsocks", "nekoray",
            ]
            # Only check adapter descriptions/names, not DNS/IP lines
            # ipconfig sections start with adapter name lines
            for line in r.stdout.splitlines():
                line_l = line.lower().strip()
                # Adapter header lines (e.g. "未知适配器 LetsTAP:" or "Ethernet adapter ...")
                if ("适配器" in line or "adapter" in line_l) and ":" in line:
                    if any(kw in line_l for kw in vpn_adapter_keywords):
                        return True
                # Description lines (e.g. "Description: TAP-Windows Adapter V9")
                if ("description" in line_l or "描述" in line_l) and ":" in line:
                    if any(kw in line_l for kw in vpn_adapter_keywords):
                        return True
    except Exception:
        pass

    return False


def _detect_route() -> str:
    """Detect whether traffic goes via VPN or direct connection.

        Strategy:
            1. Check explicit proxy setting or local tunnel adapters
            2. Query exit IP via ipinfo.io for location info
            3. Combine: conservative local detection determines route label
    """
    global _route_label
    if _route_label is not None:
        return _route_label

    has_proxy = _proxy is not None
    has_tunnel = _detect_vpn_local()

    # Get exit IP + location
    ip_info = ""
    try:
        if _HAS_CURL:
            cmd = [_CURL_BIN, "-s", "--connect-timeout", "5",
                   "https://ipinfo.io/json"]
            if _proxy:
                cmd.extend(["--proxy", _proxy])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if r.returncode == 0 and r.stdout.strip():
                info = json.loads(r.stdout.strip())
                ip = info.get("ip", "?")
                city = info.get("city", "")
                country = info.get("country", "")
                loc_parts = [p for p in [city, country] if p]
                ip_info = f"{ip}, {', '.join(loc_parts)}" if loc_parts else ip
    except Exception:
        pass

    if not ip_info:
        ip_info = "IP unknown"

    if has_proxy:
        _route_label = f"Proxy -> {ip_info}"
    elif has_tunnel:
        _route_label = f"Tunnel adapter -> {ip_info}"
    else:
        _route_label = f"Direct -> {ip_info}"
    return _route_label


# ---------------------------------------------------------------------------
# curl multi-segment downloader (with progress + strict validation)
# ---------------------------------------------------------------------------

# Zstandard magic number (first 4 bytes of any valid .zst frame)
_ZST_MAGIC = b'\x28\xb5\x2f\xfd'


def _validate_zst_magic(path: Path) -> bool:
    """Return True if *path* starts with the zstd magic number."""
    try:
        with open(path, 'rb') as f:
            return f.read(4) == _ZST_MAGIC
    except Exception:
        return False


def _get_file_size(url: str) -> int:
    """Get remote file size via HEAD request.

    Tries curl (Schannel) first, then requests as fallback.
    Returns 0 if size cannot be determined.
    """
    if _HAS_CURL:
        cmd = [
            _CURL_BIN, "-sI", "-L",
            "-H", _CURL_UA,
            "-H", "Accept-Encoding: identity",
            url,
        ]
        if _proxy:
            cmd.extend(["--proxy", _proxy])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            for line in r.stdout.splitlines():
                if line.lower().startswith("content-length:"):
                    size = int(line.split(":", 1)[1].strip())
                    if size > 0:
                        return size
        except Exception:
            pass

    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=15,
            headers={"User-Agent": _UA_VALUE, "Accept-Encoding": "identity"},
        )
        return int(resp.headers.get("content-length", 0))
    except Exception:
        return 0


def _validate_range_support(url: str) -> bool:
    """Verify the server returns HTTP 206 for a byte-range GET request.

    Uses an actual GET (not HEAD) with ``-r 0-1023`` because Cloudflare
    CDN may answer HEAD differently from GET.  The downloaded 1 KB is
    discarded to NUL.
    """
    if not _HAS_CURL:
        return False
    nul = "NUL" if sys.platform == "win32" else "/dev/null"
    def _probe(range_spec: str) -> bool:
        cmd = [
            _CURL_BIN, "-s", "-L",
            "-r", range_spec,
            "-o", nul,
            "-w", "%{http_code}",
            "--connect-timeout", "10",
            "-H", _CURL_UA,
            "-H", "Accept-Encoding: identity",
            url,
        ]
        if _proxy:
            cmd.extend(["--proxy", _proxy])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        code = r.stdout.strip()
        if code != "206":
            log.debug("Range support check %s: HTTP %s (expected 206)", range_spec, code)
            return False
        return True

    try:
        # Probe both head and non-zero offsets; some endpoints behave
        # differently outside the first bytes.
        return _probe("0-1023") and _probe("1048576-1049599")
    except Exception as e:
        log.debug("Range support check failed: %s", e)
        return False


class _SpeedTracker:
    """Sliding-window speed tracker (real-time, not average)."""

    def __init__(self, window: float = 3.0):
        self._window = window
        self._samples: deque = deque()  # (time, bytes)

    def update(self, downloaded: int):
        now = time.time()
        self._samples.append((now, downloaded))
        # Purge samples older than window
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def speed(self) -> float:
        """Current speed in bytes/s."""
        if len(self._samples) < 2:
            return 0.0
        dt = self._samples[-1][0] - self._samples[0][0]
        db = self._samples[-1][1] - self._samples[0][1]
        return db / dt if dt > 0.1 else 0.0


def _format_progress_bar(
    downloaded: int, total: int, speed_bps: float, bar_len: int = 30,
) -> str:
    """Build a progress bar string.  Never exceeds 99.9% until merge."""
    ratio = min(downloaded / total, 0.999) if total > 0 else 0
    pct = ratio * 100
    filled = int(bar_len * ratio)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    speed_mb = speed_bps / (1024 * 1024)
    eta = (
        (total - downloaded) / speed_bps if speed_bps > 0 else 0
    )
    return (
        f"  {bar} {pct:5.1f}% "
        f"({downloaded // (1024*1024)}/{total // (1024*1024)} MB) "
        f"{speed_mb:.1f} MB/s  ETA {eta:.0f}s"
    )


def _format_segment_progress(
    seg_files: List[Path],
    segments: List[Tuple[int, int, int]],
) -> str:
    """Return compact per-segment progress text (one entry per child curl)."""
    parts: List[str] = []
    for idx, (_, _, expected) in enumerate(segments):
        sf = seg_files[idx]
        actual = sf.stat().st_size if sf.exists() else 0
        ratio = (actual / expected) if expected > 0 else 0.0
        ratio = max(0.0, min(ratio, 1.0))
        parts.append(f"S{idx}:{ratio * 100:5.1f}%")
    return "  " + " | ".join(parts)


def _download_multi_segment_curl(
    url: str,
    dst: Path,
    file_size: int,
    quiet: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Download *url* via parallel curl byte-range segments, merge, validate.

    Progress is tracked by monitoring the total bytes written to all
    segment files on disk — this is inherently bounded by *file_size* so
    the displayed percentage can never exceed 100%.

    Returns True on success, False on any failure (caller falls back).
    """
    n = max(2, _download_segments)
    seg_size = file_size // n

    # Build (start, end, expected_bytes) for each segment
    segments: List[Tuple[int, int, int]] = []
    for i in range(n):
        start = i * seg_size
        end = (i + 1) * seg_size - 1 if i < n - 1 else file_size - 1
        segments.append((start, end, end - start + 1))

    seg_files = [dst.parent / f"{dst.stem}.seg{i}" for i in range(n)]

    if not quiet:
        log.info(
            "  Multi-segment: %d x ~%d MB, total %d MB",
            n, seg_size // (1024 * 1024), file_size // (1024 * 1024),
        )

    t0 = time.time()
    procs: List[subprocess.Popen] = []
    tracker = _SpeedTracker(window=3.0)

    try:
        # --- Phase 1: launch all curl processes --------------------------
        for i, (start, end, _) in enumerate(segments):
            seg_file = seg_files[i]
            if seg_file.exists():
                seg_file.unlink()

            cmd = [
                _CURL_BIN,
                "-s", "-S",          # silent but show errors
                "-L",                # follow redirects
                "-f",                # fail fast on HTTP >= 400
                "-o", str(seg_file),
                "-r", f"{start}-{end}",
                "--retry", "3",
                "--retry-delay", "2",
                "--connect-timeout", "15",
                "-H", _CURL_UA,
                "-H", "Accept-Encoding: identity",
            ]
            if _proxy:
                cmd.extend(["--proxy", _proxy])
            cmd.append(url)

            proc = _popen_tracked(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            procs.append(proc)

        # --- Phase 2: monitor progress + early-abort on bad range ------
        _EARLY_ABORT_SECS = 5  # start strict checks after warm-up
        while any(p.poll() is None for p in procs):
            if _SHUTDOWN_EVENT.is_set():
                raise KeyboardInterrupt()
            downloaded = sum(
                sf.stat().st_size if sf.exists() else 0
                for sf in seg_files
            )
            if progress_cb:
                progress_cb(downloaded, file_size)
            elapsed = time.time() - t0

            # Early-abort: if any segment already exceeds 1.5x expected
            # size, the server returned 200 (full file) instead of 206.
            if elapsed >= _EARLY_ABORT_SECS:
                for idx, (_, _, expected) in enumerate(segments):
                    sf = seg_files[idx]
                    if sf.exists() and sf.stat().st_size > expected * 1.5:
                        raise RuntimeError(
                            f"Early abort: segment {idx} is "
                            f"{sf.stat().st_size:,} bytes after {elapsed:.0f}s "
                            f"(expected ~{expected:,}).  "
                            f"Server returned 200 instead of 206."
                        )

            if not quiet:
                tracker.update(downloaded)
                seg_text = _format_segment_progress(seg_files, segments)
                print(
                    "\r" + _format_progress_bar(
                        downloaded, file_size, tracker.speed,
                    ) + seg_text,
                    end="", flush=True,
                )
            time.sleep(0.5)

        # Final progress snapshot
        if not quiet:
            downloaded = sum(
                sf.stat().st_size if sf.exists() else 0 for sf in seg_files
            )
            tracker.update(downloaded)
            seg_text = _format_segment_progress(seg_files, segments)
            print(
                "\r" + _format_progress_bar(
                    downloaded, file_size, tracker.speed,
                ) + seg_text,
                end="", flush=True,
            )
            print()  # newline
        if progress_cb:
            progress_cb(file_size, file_size)

        # --- Phase 3: check curl exit codes ------------------------------
        any_curl_fail = False
        for i, proc in enumerate(procs):
            rc = proc.returncode
            stderr_text = (
                proc.stderr.read().decode(errors="replace").strip()
                if proc.stderr else ""
            )
            if rc != 0:
                log.warning("  Segment %d curl exited %d: %s", i, rc, stderr_text or "(no stderr)")
                any_curl_fail = True
        if any_curl_fail:
            raise RuntimeError("One or more curl segments failed (see warnings above)")

        # --- Phase 4: segment size validation ----------------------------
        # HTTP Range responses should return EXACTLY the requested bytes.
        # A mismatch means: truncated download, server ignoring Range, or
        # size changed between HEAD and GET.
        seg_sizes = []
        for i, (start, end, expected) in enumerate(segments):
            seg_file = seg_files[i]
            if not seg_file.exists():
                raise RuntimeError(f"Segment {i} file missing: {seg_file}")
            actual = seg_file.stat().st_size
            seg_sizes.append(actual)
            if actual != expected:
                ratio = actual / expected if expected > 0 else 0
                log.warning(
                    "  Segment %d size mismatch: expected %s, got %s (ratio %.4f)",
                    i, f"{expected:,}", f"{actual:,}", ratio,
                )
                # Accept segments as long as at least 90% of requested bytes arrived.
                # Still reject oversized responses because that often means the server
                # ignored Range and returned the full file.
                if actual < expected * 0.90 or actual > expected * 1.05:
                    raise RuntimeError(
                        f"Segment {i}: expected {expected:,} bytes, "
                        f"got {actual:,} (ratio {ratio:.4f}). "
                        f"Server may have returned 200 instead of 206."
                    )
                # Minor/moderate shortfall (>=90%) — accept degraded segment.
                log.info("  Segment %d: acceptable size diff (%.2f%%), accepting",
                         i, abs(ratio - 1) * 100)

        # Double-check: total of all segments vs expected file size
        total_seg_bytes = sum(seg_sizes)
        if total_seg_bytes != file_size:
            ratio = total_seg_bytes / file_size if file_size > 0 else 0.0
            diff_pct = abs(total_seg_bytes - file_size) / file_size * 100
            if ratio < 0.90 or ratio > 1.05:
                raise RuntimeError(
                    f"Total segment bytes {total_seg_bytes:,} != "
                    f"expected {file_size:,} (diff {diff_pct:.2f}%, ratio {ratio:.4f})"
                )
            log.info("  Total segment bytes %s vs expected %s (diff %.3f%%, ratio %.4f, accepting)",
                     f"{total_seg_bytes:,}", f"{file_size:,}", diff_pct, ratio)

        # --- Phase 5: merge segments into destination --------------------
        with open(dst, "wb") as fout:
            for seg_file in seg_files:
                with open(seg_file, "rb") as fin:
                    while True:
                        chunk = fin.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)

        # Final file validation
        final_size = dst.stat().st_size
        if final_size < file_size * 0.90 or final_size > file_size * 1.05:
            raise RuntimeError(
                f"Merged file {final_size:,} bytes != expected {file_size:,}"
            )

        elapsed = time.time() - t0
        speed = file_size / (1024 * 1024) / (elapsed + 0.001)
        if not quiet:
            log.info(
                "  Downloaded %d MB in %.1fs (%.1f MB/s, %d segments)",
                file_size // (1024 * 1024), elapsed, speed, n,
            )
        return True

    except Exception as e:
        if not quiet:
            log.warning("  Multi-segment download failed: %s", e)
            # Dump per-segment diagnostic info
            for idx, sf in enumerate(seg_files):
                sz = sf.stat().st_size if sf.exists() else -1
                exp = segments[idx][2] if idx < len(segments) else 0
                rc = procs[idx].returncode if idx < len(procs) else "?"
                log.warning("    S%d: %s bytes (expected %s), curl rc=%s",
                            idx, f"{sz:,}" if sz >= 0 else "MISSING",
                            f"{exp:,}", rc)
        # Kill any still-running curl processes
        for proc in procs:
            try:
                _terminate_process_tree(proc, timeout=1.0)
            except OSError:
                pass
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        return False

    finally:
        for proc in procs:
            _unregister_child_process(proc)

        # Always clean up segment files
        for seg_file in seg_files:
            try:
                if seg_file.exists():
                    seg_file.unlink()
            except OSError:
                pass


def _download_with_curl(
    url: str,
    dst: Path,
    quiet: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Download using single-connection curl (Schannel)."""
    total = _get_file_size(url)
    max_attempts = 6

    for attempt in range(1, max_attempts + 1):
        try:
            cmd = [
                _CURL_BIN,
                "-L", "-o", str(dst),
                "--silent" if quiet else "--progress-bar",
                "--retry", "6",
                "--retry-delay", "2",
                "--retry-all-errors",
                "--retry-connrefused",
                "--connect-timeout", "20",
                "-H", _CURL_UA,
                "-H", "Accept-Encoding: identity",
            ]
            if dst.exists() and dst.stat().st_size > 0:
                cmd.extend(["-C", "-"])
            if _proxy:
                cmd.extend(["--proxy", _proxy])
            cmd.append(url)

            proc = _popen_tracked(cmd)
            try:
                while True:
                    if _SHUTDOWN_EVENT.is_set():
                        _terminate_process_tree(proc, timeout=1.0)
                        raise KeyboardInterrupt()
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if progress_cb and total > 0 and dst.exists():
                        try:
                            progress_cb(dst.stat().st_size, total)
                        except Exception:
                            pass
                    time.sleep(0.2)
            finally:
                _unregister_child_process(proc)

            if rc == 0 and dst.exists() and dst.stat().st_size > 0:
                if progress_cb and total > 0:
                    try:
                        progress_cb(total, total)
                    except Exception:
                        pass
                return True

            partial_mb = (dst.stat().st_size / (1024 * 1024)) if dst.exists() else 0.0
            log.warning(
                "curl attempt %d/%d failed (rc=%s, partial=%.1f MB): %s",
                attempt, max_attempts, rc, partial_mb, url,
            )
        except FileNotFoundError as e:
            log.warning("curl not found: %s", e)
            break
        except subprocess.TimeoutExpired as e:
            log.warning("curl timeout on attempt %d/%d: %s", attempt, max_attempts, e)

        if attempt < max_attempts:
            time.sleep(min(2 * attempt, 10))

    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    return False


# Reusable session (fallback only)
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": _CURL_UA.split(": ", 1)[1],
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        })
        if _proxy:
            _session.proxies = {"http": _proxy, "https": _proxy}
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
            max_retries=Retry(total=3, backoff_factor=1,
                              status_forcelist=[502, 503, 504]),
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def _download_with_requests(
    url: str,
    dst: Path,
    quiet: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
):
    """Fallback download using requests."""
    session = _get_session()
    resp = session.get(url, stream=True, timeout=(10, 300))
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    tracker = _SpeedTracker(window=3.0)

    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if _SHUTDOWN_EVENT.is_set():
                raise KeyboardInterrupt()
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb and total > 0:
                progress_cb(downloaded, total)
            if total and not quiet:
                tracker.update(downloaded)
                spd = tracker.speed
                pct = min(downloaded / total, 0.999) * 100
                speed_mb = spd / (1024 * 1024)
                eta = (total - downloaded) / spd if spd > 0 else 0
                print(
                    f"\r  {pct:5.1f}% ({downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB) "
                    f"{speed_mb:.1f} MB/s  ETA {eta:.0f}s",
                    end="",
                    flush=True,
                )
    if not quiet:
        print()
    if progress_cb and total > 0:
        progress_cb(total, total)


def download_file(
    d: date,
    force: bool = False,
    quiet: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download a .zst file for a given date.

    Strategy (ordered by priority):
        1. Multi-segment curl  (if --segments > 1 AND server supports Range)
        2. Single-connection curl (Schannel)
        3. Python requests (last-resort fallback)

    The multi-segment path validates HTTP 206 support *before* starting,
    which prevents the >100% progress bug (segments downloading full file).

    If *quiet* is True, suppress all progress output (used for prefetch).
    """
    dst = cache_path(d)
    if dst.exists() and not force:
        # Validate cached file isn't a truncated partial download.
        # A valid .zst must be at least 50 MB (smallest daily file ~600 MB).
        # Also do a quick HEAD check when possible to compare exact size.
        cached_size = dst.stat().st_size
        if cached_size < 50 * 1024 * 1024:
            log.warning("Cache %s too small (%d bytes), re-downloading", dst, cached_size)
            try:
                dst.unlink()
            except OSError:
                pass
        elif not _validate_zst_magic(dst):
            log.warning(
                "Cache %s has unexpected zstd header, keeping file and trying to process it",
                dst,
            )
            log.debug("Cache header check relaxed for %s", dst)
            return dst
        else:
            log.debug("Cache hit: %s (%d MB)", dst, cached_size // (1024 * 1024))
            return dst
    if force and dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass

    url = file_url(d)
    dst.parent.mkdir(parents=True, exist_ok=True)

    timer.start("download")
    t0 = time.time()

    # Show network route on first download
    if not quiet:
        route = _detect_route()
        log.info("  Network: %s", route)

    ok = False
    fallback_reason: Optional[str] = None

    if _SHUTDOWN_EVENT.is_set():
        raise KeyboardInterrupt()

    # --- Attempt 1: multi-segment curl ----------------------------------
    if not ok and _HAS_CURL and _download_segments > 1:
        file_size = _get_file_size(url)
        if file_size > 10 * 1024 * 1024:
            # Pre-flight: confirm HTTP 206 Range support
            if _validate_range_support(url):
                if not quiet:
                    log.info(
                        "Downloading (curl x%d/Schannel) %s ...",
                        _download_segments, url,
                    )
                ok = _download_multi_segment_curl(
                    url, dst, file_size, quiet=quiet, progress_cb=progress_cb,
                )
                if not ok:
                    fallback_reason = "multi-segment transfer/validation failed"
            else:
                fallback_reason = "server does not support HTTP 206 range requests"
                if not quiet:
                    log.info(
                        "  Server does not support Range (HTTP 206), "
                        "falling back to single-connection curl."
                    )
        else:
            fallback_reason = (
                f"remote file too small for segmented mode ({file_size} bytes)"
            )

    # --- Attempt 2: single-connection curl -------------------------------
    if not ok and _HAS_CURL:
        if _download_segments > 1:
            log.warning(
                "Falling back to single-connection curl for %s: %s",
                d,
                fallback_reason or "unknown reason",
            )
        if not quiet:
            log.info("Downloading (curl/Schannel) %s ...", url)
        ok = _download_with_curl(url, dst, quiet=quiet, progress_cb=progress_cb)

    # --- Attempt 3: Python requests fallback (only if curl unavailable) --
    if not ok and not _HAS_CURL:
        if not quiet:
            log.info("Downloading (requests) %s ...", url)
        _download_with_requests(url, dst, quiet=quiet, progress_cb=progress_cb)
    elif not ok:
        raise RuntimeError(f"curl download failed after retries: {url}")

    elapsed = time.time() - t0
    fsize = dst.stat().st_size if dst.exists() else 0
    timer.stop(nbytes=fsize)

    # Post-download integrity check: verify zstd magic bytes.
    # Be tolerant here as well: some edge-case files may fail the quick
    # header probe but still decompress successfully in the real pipeline.
    if dst.exists() and not _validate_zst_magic(dst):
        log.warning(
            "Downloaded file %s has unexpected zstd header, accepting and deferring validation to processing stage",
            dst,
        )

    if not quiet:
        log.info(
            "Downloaded %s in %.1fs (%.1f MB/s)",
            dst.name, elapsed,
            fsize / 1024 / 1024 / (elapsed + 0.001),
        )
    return dst


# ---------------------------------------------------------------------------
# Core processing: .zst → hourly DataFrame
# ---------------------------------------------------------------------------


def _decompress_to_tmpfile(zst_path: Path) -> Path:
    """Decompress .zst to a temp CSV file for PyArrow to read.
    Returns the temp file path. Caller must delete it."""
    tmp_csv = zst_path.with_suffix(".csv.tmp")
    log.info("  Decompressing %s → %s ...", zst_path.name, tmp_csv.name)
    timer.start("decompress")

    used_external = False
    if _ZSTD_BIN:
        try:
            # Prefer native zstd binary when available (typically faster).
            cmd = [_ZSTD_BIN, "-d", "-f", "-q", "-o", str(tmp_csv), str(zst_path)]
            proc = _popen_tracked(cmd)
            try:
                rc = _wait_process_interruptible(proc, timeout=3600)
            finally:
                _unregister_child_process(proc)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            used_external = True
        except Exception as e:
            log.debug("External zstd decompress failed, fallback to python-zstd: %s", e)

    if not used_external:
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as fin, open(tmp_csv, "wb") as fout:
            dctx.copy_stream(
                fin,
                fout,
                read_size=32 * 1024 * 1024,
                write_size=32 * 1024 * 1024,
            )

    nbytes = tmp_csv.stat().st_size
    timer.stop(nbytes=nbytes)
    size_mb = nbytes / 1024 / 1024
    elapsed = timer._totals.get("decompress", 0)
    mode = "zstd-cli" if used_external else "python-zstd"
    log.info("  Decompressed (mode=%s) %.0f MB in %.1fs (%.0f MB/s)",
             mode,
             size_mb, timer._totals["decompress"] / timer._counts["decompress"],
             size_mb / (timer._totals["decompress"] / timer._counts["decompress"] + 0.001))
    return tmp_csv


def _decompress_to_buffer(zst_path: Path) -> bytes:
    """Decompress .zst entirely into an in-memory bytes buffer.

    Eliminates disk I/O for the temp CSV.  Peak memory ≈ decompressed size
    (~5-6 GB), acceptable under subprocess-per-date isolation.
    """
    log.info("  Decompressing %s → memory ...", zst_path.name)
    timer.start("decompress")

    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f:
        with dctx.stream_reader(f, read_size=32 * 1024 * 1024) as reader:
            csv_data = reader.readall()

    nbytes = len(csv_data)
    timer.stop(nbytes=nbytes)
    size_mb = nbytes / (1024 * 1024)
    dec_time = timer._totals["decompress"] / timer._counts["decompress"]
    log.info("  Decompressed (memory) %.0f MB in %.1fs (%.0f MB/s)",
             size_mb, dec_time, size_mb / (dec_time + 0.001))
    return csv_data


def _open_streaming_zstd_source(
    zst_path: Path,
) -> Tuple[Any, Optional[BinaryIO], Optional[Any], Optional[subprocess.Popen], str]:
    """Open a streaming CSV source from a .zst file.

    Prefer the system `zstd` binary when available, falling back to
    `python-zstd` stream_reader.
    """
    if _ZSTD_BIN:
        proc: Optional[subprocess.Popen] = None
        try:
            cmd = [_ZSTD_BIN, "-d", "-q", "-c", str(zst_path)]
            proc = _popen_tracked(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024 * 1024,
            )
            if proc.stdout is None:
                raise RuntimeError("zstd stdout pipe unavailable")
            return proc.stdout, None, None, proc, "zstd-cli"
        except Exception as e:
            if proc is not None:
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                _terminate_process_tree(proc, timeout=1.0)
            log.debug("External zstd stream init failed, fallback to python-zstd: %s", e)

    zst_fh: Optional[BinaryIO] = None
    zst_stream: Optional[Any] = None
    try:
        zst_fh = open(zst_path, "rb")
        dctx = zstd.ZstdDecompressor()
        zst_stream = dctx.stream_reader(zst_fh, read_size=32 * 1024 * 1024)
        return zst_stream, zst_fh, zst_stream, None, "python-zstd"
    except Exception:
        if zst_stream is not None:
            try:
                zst_stream.close()
            except Exception:
                pass
        if zst_fh is not None:
            try:
                zst_fh.close()
            except Exception:
                pass
        raise


def _finalize_streaming_zstd_proc(proc: subprocess.Popen, timeout: float = 30.0):
    """Wait for a streaming `zstd` subprocess and raise on non-zero exit."""
    try:
        rc = _wait_process_interruptible(proc, timeout=timeout)
    finally:
        _unregister_child_process(proc)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)


def _iter_csv_chunk_files(csv_path: Path, chunk_mb: int, chunk_dir: Path):
    """Yield CSV chunk files one-by-one, each including the original header."""
    target_mb = max(16, int(chunk_mb))
    target_bytes = target_mb * 1024 * 1024

    idx = 0

    def _new_chunk() -> Tuple[Path, BinaryIO]:
        nonlocal idx
        path = chunk_dir / f"part_{idx:04d}.csv"
        idx += 1
        return path, open(path, "wb")

    with open(csv_path, "rb") as fin:
        header = fin.readline()
        if not header:
            return
        header_len = len(header)

        chunk_path, fout = _new_chunk()
        fout.write(header)
        chunk_size = header_len
        data_size = 0

        pending = b""
        while True:
            block = fin.read(16 * 1024 * 1024)
            if not block:
                break

            data = pending + block
            lines = data.splitlines(keepends=True)
            if lines and not data.endswith(b"\n"):
                pending = lines.pop()
            else:
                pending = b""

            for line in lines:
                if chunk_size + len(line) > target_bytes and data_size > 0:
                    fout.close()
                    yield chunk_path
                    chunk_path, fout = _new_chunk()
                    fout.write(header)
                    chunk_size = header_len
                    data_size = 0
                fout.write(line)
                chunk_size += len(line)
                data_size += len(line)

        if pending:
            if chunk_size + len(pending) > target_bytes and data_size > 0:
                fout.close()
                yield chunk_path
                chunk_path, fout = _new_chunk()
                fout.write(header)
                chunk_size = header_len
                data_size = 0
            fout.write(pending)
            chunk_size += len(pending)
            data_size += len(pending)

        fout.close()
        if data_size > 0:
            yield chunk_path
        else:
            try:
                chunk_path.unlink()
            except OSError:
                pass


def _iter_csv_chunk_bytes(csv_path: Path, chunk_mb: int):
    """Yield CSV chunks as bytes (with header), avoiding chunk temp files."""
    target_mb = max(16, int(chunk_mb))
    target_bytes = target_mb * 1024 * 1024

    with open(csv_path, "rb") as fin:
        header = fin.readline()
        if not header:
            return
        header_len = len(header)
        target_data_bytes = max(1, target_bytes - header_len)

        pending = bytearray()
        while True:
            block = fin.read(32 * 1024 * 1024)
            if not block:
                break

            pending.extend(block)

            while len(pending) >= target_data_bytes:
                cut = pending.rfind(b"\n", 0, target_data_bytes + 1)
                if cut < 0:
                    cut = pending.find(b"\n", target_data_bytes)
                    if cut < 0:
                        break

                body_end = cut + 1
                yield header + bytes(pending[:body_end])
                del pending[:body_end]

        if pending:
            yield header + bytes(pending)


def _iter_csv_chunk_bytes_from_stream(source: Any, chunk_mb: int):
    """Yield CSV chunks as bytes (with header) from a decompressed byte stream."""
    target_mb = max(16, int(chunk_mb))
    target_bytes = target_mb * 1024 * 1024

    header = source.readline()
    if not header:
        return

    target_data_bytes = max(1, target_bytes - len(header))
    pending = bytearray()

    while True:
        block = source.read(32 * 1024 * 1024)
        if not block:
            break

        pending.extend(block)

        while len(pending) >= target_data_bytes:
            cut = pending.rfind(b"\n", 0, target_data_bytes + 1)
            if cut < 0:
                cut = pending.find(b"\n", target_data_bytes)
                if cut < 0:
                    break

            body_end = cut + 1
            yield header + bytes(pending[:body_end])
            del pending[:body_end]

    if pending:
        yield header + bytes(pending)


def _import_polars():
    try:
        import polars as pl  # type: ignore
        return pl
    except ImportError as exc:
        raise RuntimeError(
            "polars-stream mode requires the 'polars' package to be installed"
        ) from exc


def _polars_schema_overrides(pl):
    return {
        "timestamp": pl.Int64,
        "symbol": pl.Utf8,
        "type": pl.Utf8,
        "strike_price": pl.Float32,
        "expiration": pl.Int64,
        "open_interest": pl.Float32,
        "last_price": pl.Float64,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "bid_iv": pl.Float32,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
        "ask_iv": pl.Float32,
        "mark_price": pl.Float64,
        "mark_iv": pl.Float32,
        "underlying_index": pl.Utf8,
        "underlying_price": pl.Float64,
        "delta": pl.Float32,
        "gamma": pl.Float32,
        "vega": pl.Float32,
        "theta": pl.Float32,
    }


def _merge_hourly_polars(accum, chunk, hourly_pick: str):
    pl = _import_polars()

    if accum is None or accum.height == 0:
        return chunk

    merged = pl.concat([accum, chunk], how="vertical_relaxed")

    if hourly_pick == "first":
        return (
            merged
            .sort(["symbol", "hour_us", "timestamp"])
            .unique(subset=["symbol", "hour_us"], keep="first")
        )
    if hourly_pick == "last":
        return (
            merged
            .sort(["symbol", "hour_us", "timestamp"])
            .unique(subset=["symbol", "hour_us"], keep="last")
        )

    open_rows = (
        merged
        .filter(pl.col("hourly_pick") == "open")
        .sort(["symbol", "hour_us", "timestamp"])
        .unique(subset=["symbol", "hour_us", "hourly_pick"], keep="first")
    )
    close_rows = (
        merged
        .filter(pl.col("hourly_pick") == "close")
        .sort(["symbol", "hour_us", "timestamp"])
        .unique(subset=["symbol", "hour_us", "hourly_pick"], keep="last")
    )
    return pl.concat([open_rows, close_rows], how="vertical_relaxed")


def _append_polars_hourly_part(parts: List[Any], part: Any, row_count: int) -> int:
    if part is None or part.height == 0:
        return row_count
    parts.append(part)
    return row_count + int(part.height)


def _polars_hour_us_max(df: Any) -> Optional[int]:
    if df is None or df.height == 0:
        return None
    value = df["hour_us"].max()
    if value is None:
        return None
    return int(value)


def _finalize_hourly_output_polars(hourly, hourly_pick: str) -> pd.DataFrame:
    pl = _import_polars()

    hourly = (
        hourly
        .with_columns([
            pl.from_epoch(pl.col("hour_us"), time_unit="us").dt.replace_time_zone("UTC").alias("hour"),
            pl.col("underlying_index").str.extract(r"^(\w+)", 1).str.to_uppercase().alias("underlying"),
        ])
        .filter(pl.col("underlying").is_in(sorted(_VALID_UNDERLYINGS)))
        .drop("hour_us")
    )

    if hourly_pick == "first":
        hourly = hourly.with_columns(pl.lit("open").alias("hourly_pick"))
    elif hourly_pick == "last":
        hourly = hourly.with_columns(pl.lit("close").alias("hourly_pick"))

    return hourly.to_pandas()


def _process_zst_to_hourly_polars_stream(
    zst_path: Path,
    hourly_pick: str,
) -> Tuple[pd.DataFrame, int]:
    pl = _import_polars()
    schema_overrides = _polars_schema_overrides(pl)

    source = None
    zst_fh = None
    zst_stream = None
    zstd_proc: Optional[subprocess.Popen] = None
    stream_mode = ""
    chunk_count = 0
    total_rows = 0
    hourly_parts: List[Any] = []
    hourly_rows = 0
    pending_tail = None
    pending_hour_us: Optional[int] = None

    try:
        source, zst_fh, zst_stream, zstd_proc, stream_mode = _open_streaming_zstd_source(zst_path)
        log.info(
            "  Polars-stream mode: %s → chunked CSV parse (%d MB chunks)",
            stream_mode,
            _polars_stream_chunk_mb,
        )

        chunk_iter = iter(_iter_csv_chunk_bytes_from_stream(source, _polars_stream_chunk_mb))
        while True:
            timer.start("decompress")
            try:
                chunk_blob = next(chunk_iter)
            except StopIteration:
                timer.stop()
                break
            timer.stop(nbytes=len(chunk_blob))

            chunk_count += 1

            timer.start("csv_read")
            chunk_df = pl.read_csv(
                io.BytesIO(chunk_blob),
                columns=USE_COLS,
                schema_overrides=schema_overrides,
                infer_schema_length=0,
                low_memory=True,
                batch_size=50_000,
            )
            timer.stop(rows=chunk_df.height)
            total_rows += chunk_df.height

            if chunk_df.height == 0:
                continue

            timer.start("resample")
            chunk_df = (
                chunk_df
                .filter(
                    pl.col("symbol").str.starts_with("BTC-") |
                    pl.col("symbol").str.starts_with("ETH-")
                )
                .with_columns(((pl.col("timestamp") // 3_600_000_000) * 3_600_000_000).alias("hour_us"))
            )

            if chunk_df.height > 0:
                if hourly_pick == "first":
                    chunk_hourly = chunk_df.group_by(["symbol", "hour_us"]).first()
                elif hourly_pick == "last":
                    chunk_hourly = chunk_df.group_by(["symbol", "hour_us"]).last()
                else:
                    chunk_open = chunk_df.group_by(["symbol", "hour_us"]).first().with_columns(
                        pl.lit("open").alias("hourly_pick")
                    )
                    chunk_close = chunk_df.group_by(["symbol", "hour_us"]).last().with_columns(
                        pl.lit("close").alias("hourly_pick")
                    )
                    chunk_hourly = pl.concat([chunk_open, chunk_close], how="vertical_relaxed")

                if chunk_hourly.height > 0:
                    chunk_max_hour_us = _polars_hour_us_max(chunk_hourly)
                    if chunk_max_hour_us is None:
                        pending_rows = pending_tail.height if pending_tail is not None else 0
                        timer.stop(rows=(hourly_rows + pending_rows))
                        continue

                    if pending_tail is not None and pending_tail.height > 0 and pending_hour_us is not None:
                        overlap_part = chunk_hourly.filter(pl.col("hour_us") == pending_hour_us)
                        if overlap_part.height > 0:
                            pending_tail = _merge_hourly_polars(pending_tail, overlap_part, hourly_pick)

                        future_part = chunk_hourly.filter(pl.col("hour_us") > pending_hour_us)
                        if future_part.height > 0:
                            hourly_rows = _append_polars_hourly_part(hourly_parts, pending_tail, hourly_rows)

                            future_max_hour_us = _polars_hour_us_max(future_part)
                            if future_max_hour_us is None:
                                pending_rows = pending_tail.height if pending_tail is not None else 0
                                timer.stop(rows=(hourly_rows + pending_rows))
                                continue
                            stable_part = future_part.filter(pl.col("hour_us") < future_max_hour_us)
                            hourly_rows = _append_polars_hourly_part(hourly_parts, stable_part, hourly_rows)

                            pending_tail = future_part.filter(pl.col("hour_us") == future_max_hour_us)
                            pending_hour_us = future_max_hour_us
                    else:
                        stable_part = chunk_hourly.filter(pl.col("hour_us") < chunk_max_hour_us)
                        hourly_rows = _append_polars_hourly_part(hourly_parts, stable_part, hourly_rows)

                        pending_tail = chunk_hourly.filter(pl.col("hour_us") == chunk_max_hour_us)
                        pending_hour_us = chunk_max_hour_us

            pending_rows = pending_tail.height if pending_tail is not None else 0
            timer.stop(rows=(hourly_rows + pending_rows))

        if pending_tail is not None and pending_tail.height > 0:
            hourly_rows = _append_polars_hourly_part(hourly_parts, pending_tail, hourly_rows)

        if not hourly_parts:
            hourly = pd.DataFrame()
        else:
            timer.start("to_pandas")
            hourly = _finalize_hourly_output_polars(
                pl.concat(hourly_parts, how="vertical_relaxed"),
                hourly_pick,
            )
            timer.stop(rows=len(hourly))

        log.info(
            "  Polars-stream processed %dk rows across %d chunk(s) via %s",
            total_rows // 1000,
            chunk_count,
            stream_mode,
        )
        return hourly, total_rows
    finally:
        if zst_stream is not None:
            try:
                zst_stream.close()
            except Exception:
                pass
        if zst_fh is not None:
            try:
                zst_fh.close()
            except Exception:
                pass
        if zstd_proc is not None:
            if zstd_proc.stdout is not None:
                try:
                    zstd_proc.stdout.close()
                except Exception:
                    pass
            _finalize_streaming_zstd_proc(zstd_proc, timeout=30.0)


# Only keep these underlyings (filter out SYN, INDEX_PRICE, etc.)
_VALID_UNDERLYINGS = {"BTC", "ETH"}


def _extract_underlying(series: pd.Series) -> pd.Series:
    """Extract underlying (BTC/ETH) from underlying_index column."""
    return (
        series.astype(str)
        .str.extract(r"^(\w+)", expand=False)
        .str.upper()
    )


def _resample_df_to_hourly(df: pd.DataFrame, hourly_pick: str) -> pd.DataFrame:
    """Resample one dataframe to hourly rows.

    Uses sort + drop_duplicates instead of groupby — typically 2-5x faster
    for first/last row selection per group.
    """
    if "hour_us" not in df.columns:
        hour_us = 3_600_000_000
        df["hour_us"] = (df["timestamp"] // hour_us) * hour_us

    # Keep fast path for already time-ordered input (common in tick files)
    if not df["timestamp"].is_monotonic_increasing:
        df.sort_values("timestamp", kind="mergesort", inplace=True)

    if hourly_pick == "first":
        hourly = df.drop_duplicates(subset=["symbol", "hour_us"], keep="first").copy()
        hourly["hourly_pick"] = "open"
    elif hourly_pick == "last":
        hourly = df.drop_duplicates(subset=["symbol", "hour_us"], keep="last").copy()
        hourly["hourly_pick"] = "close"
    else:
        h_first = df.drop_duplicates(subset=["symbol", "hour_us"], keep="first").copy()
        h_first["hourly_pick"] = "open"
        h_last = df.drop_duplicates(subset=["symbol", "hour_us"], keep="last").copy()
        h_last["hourly_pick"] = "close"
        hourly = pd.concat([h_first, h_last], ignore_index=True)
        del h_first, h_last

    return hourly


def _finalize_hourly_output(hourly: pd.DataFrame) -> pd.DataFrame:
    """Finalize hourly output columns and filter to BTC/ETH."""
    hourly["hour"] = pd.to_datetime(hourly["hour_us"], unit="us", utc=True)
    hourly.drop(columns=["hour_us"], inplace=True)

    if "underlying_index" in hourly.columns:
        hourly["underlying"] = _extract_underlying(hourly["underlying_index"])
    else:
        hourly["underlying"] = hourly["symbol"].str.split("-").str[0].str.upper()

    n_before = len(hourly)
    mask_valid = hourly["underlying"].isin(_VALID_UNDERLYINGS)
    hourly = hourly.loc[mask_valid, :].copy()
    n_dropped = n_before - len(hourly)
    if n_dropped:
        log.info("  Filtered out %d non-BTC/ETH rows (%d -> %d)",
                 n_dropped, n_before, len(hourly))
    return hourly


def _merge_hourly_chunk(
    accum: Optional[pd.DataFrame],
    chunk: pd.DataFrame,
    hourly_pick: str,
) -> pd.DataFrame:
    """Incrementally merge one hourly chunk into accumulated hourly rows.

    This keeps only one accumulated dataframe in memory, avoiding the
    high-peak ``hourly_parts -> concat`` pattern in stream mode.
    """
    if accum is None or accum.empty:
        return chunk.copy()

    merged = pd.concat([accum, chunk], ignore_index=True)

    # Stream path is typically append-only in time; avoid sort when possible.
    need_sort = True
    try:
        need_sort = bool(chunk["timestamp"].iloc[0] < accum["timestamp"].iloc[-1])
    except Exception:
        need_sort = True
    if need_sort:
        merged.sort_values("timestamp", kind="mergesort", inplace=True)

    if hourly_pick == "first":
        merged = merged.drop_duplicates(
            subset=["symbol", "hour_us"], keep="first").copy()
    elif hourly_pick == "last":
        merged = merged.drop_duplicates(
            subset=["symbol", "hour_us"], keep="last").copy()
    else:
        opens = merged[merged["hourly_pick"] == "open"]
        closes = merged[merged["hourly_pick"] == "close"]
        h_open = opens.drop_duplicates(
            subset=["symbol", "hour_us"], keep="first").copy()
        h_open["hourly_pick"] = "open"
        h_close = closes.drop_duplicates(
            subset=["symbol", "hour_us"], keep="last").copy()
        h_close["hourly_pick"] = "close"
        merged = pd.concat([h_open, h_close], ignore_index=True)
        del opens, closes, h_open, h_close

    return merged


def process_zst_to_hourly(
    zst_path: Path,
    data_date: Optional[date] = None,
    hourly_pick: str = "both",
) -> pd.DataFrame:
    """
        Decompress → CSV read → Arrow pre-filter → Pandas resample.

        Bulk pipeline (fastest, default):
      1. zstd decompress to memory buffer  (no temp file I/O)
      2. PyArrow bulk read_csv from buffer  (multi-threaded)
      3. Arrow pre-filter (BTC/ETH only) + compute hour_us
      4. Arrow→Pandas conversion
      5. sort + drop_duplicates resample (faster than groupby)

                Stream pipeline (lower peak memory):
      1. zstd stream_reader → open_csv per-batch
      2. Per-batch Arrow pre-filter + to_pandas + resample
            3. Incrementally merge/deduplicate across batches (no hourly_parts list)

        Fallback: decompress to temp file (if memory mode fails).
    """
    log.info("Processing %s ...", zst_path)
    t0 = time.time()
    total_rows = 0
    process_peak_sampler: Optional[_PeakSampler] = None
    if _mem_diagnostics:
        _reset_mem_diag_records()
        process_peak_sampler = _PeakSampler(
            "process.total",
            enabled=True,
            interval_ms=_mem_peak_interval_ms,
            mode=_csv_read_mode,
        )
        process_peak_sampler.start()

    # Common PyArrow CSV options
    convert_kwargs = {
        "column_types": PA_COLUMN_TYPES,
        "include_columns": USE_COLS,
        "strings_can_be_null": False,
    }
    parse_kwargs = {
        "newlines_in_values": False,
    }
    if _csv_aggressive_parse:
        convert_kwargs["check_utf8"] = False
        parse_kwargs.update({
            "quote_char": False,
            "double_quote": False,
            "escape_char": False,
        })

    convert_options = pa_csv.ConvertOptions(**convert_kwargs)
    parse_options = pa_csv.ParseOptions(**parse_kwargs)

    # Determine read mode
    zst_mb = zst_path.stat().st_size / (1024 * 1024)
    # Typical compression ratio ~6.3x (very rough estimate)
    est_csv_mb = zst_mb * 6.5
    read_mode = _csv_read_mode
    auto_mode = (read_mode == "auto")
    if read_mode == "auto":
        avail_mb = _get_available_mem_mb()
        if avail_mb is not None:
            # Rough working-set estimate for bulk path (Arrow table + Pandas + overhead)
            est_working_mb = est_csv_mb * 2.2
            reserve_mb = max(2048.0, avail_mb * 0.20)
            budget_mb = max(0.0, avail_mb - reserve_mb)
            read_mode = "bulk" if est_working_mb <= budget_mb else "stream"
            log.info(
                "  CSV read mode: %s (auto; zst %.0f MB, est csv ~%.0f MB, avail %.0f MB, budget %.0f MB)",
                read_mode, zst_mb, est_csv_mb, avail_mb, budget_mb,
            )
        else:
            # Fallback heuristic when memory telemetry is unavailable
            read_mode = "stream" if est_csv_mb >= _CSV_STREAM_AUTO_THRESHOLD_MB else "bulk"
            log.info("  CSV read mode: %s (auto-fallback; zst %.0f MB, est csv ~%.0f MB)",
                     read_mode, zst_mb, est_csv_mb)
    else:
        log.info("  CSV read mode: %s (zst %.0f MB, est csv ~%.0f MB)",
                 read_mode, zst_mb, est_csv_mb)

    # ====================================================================
    # POLARS-STREAM MODE: streaming zstd → chunked Polars parse/aggregate
    # ====================================================================
    if read_mode == "polars-stream":
        hourly, total_rows = _process_zst_to_hourly_polars_stream(
            zst_path,
            hourly_pick=hourly_pick,
        )

        elapsed_read = timer._totals.get("csv_read", 0.0)
        if elapsed_read > 0:
            log.info("  Polars-stream read %dk rows in %.1fs (%.0fk rows/s)",
                     total_rows // 1000, elapsed_read,
                     total_rows / 1000 / (elapsed_read + 0.001))

        elapsed_rs = timer._totals.get("resample", 0)
        rs_count = timer._counts.get("resample", 1)
        log.info("  Resampled to %d hourly rows in %.1fs",
                 len(hourly), elapsed_rs / rs_count)

        elapsed_total = time.time() - t0
        log.info("  Total: %dk → %d rows in %.1fs",
                 total_rows // 1000, len(hourly), elapsed_total)
        if process_peak_sampler is not None:
            process_peak_sampler.stop(total_rows=total_rows, output_rows=len(hourly))
        _log_mem("after process_zst_to_hourly")
        _report_mem_diag_hotspots(context=zst_path.name)
        return hourly

    # ====================================================================
    # BULK MODE: decompress to memory → PyArrow bulk read
    # ====================================================================
    bulk_fallback_exc: Optional[Exception] = None
    if read_mode == "bulk":
        try:
            # --- Step 1: Decompress (memory preferred, temp file fallback) ---
            csv_data: Optional[bytes] = None
            tmp_csv: Optional[Path] = None
            try:
                csv_data = _decompress_to_buffer(zst_path)
            except (MemoryError, zstd.ZstdError) as e:
                log.warning("  Memory decompress failed (%s), falling back to temp file", e)
                tmp_csv = _decompress_to_tmpfile(zst_path)

            try:
                # --- Step 2: PyArrow bulk read ---
                read_options = pa_csv.ReadOptions(use_threads=(_cpu_limit != 1))
                timer.start("csv_read")
                if csv_data is not None:
                    buf_reader = pa.BufferReader(csv_data)
                    table = pa_csv.read_csv(
                        buf_reader,
                        convert_options=convert_options,
                        read_options=read_options,
                        parse_options=parse_options,
                    )
                    del csv_data, buf_reader  # free large buffer asap
                else:
                    table = pa_csv.read_csv(
                        tmp_csv,
                        convert_options=convert_options,
                        read_options=read_options,
                        parse_options=parse_options,
                    )
                total_rows = len(table)
                timer.stop(rows=total_rows)
                elapsed_read = timer._totals["csv_read"] / timer._counts["csv_read"]
                log.info("  Read %dk rows in %.1fs (%.0fk rows/s)",
                         total_rows // 1000, elapsed_read,
                         total_rows / 1000 / (elapsed_read + 0.001))

                # --- Step 3: Arrow pre-filter + hour_us computation ---
                timer.start("arrow_filter")
                symbol_col = table.column("symbol")
                keep_mask = pc.or_(
                    pc.starts_with(symbol_col, "BTC-"),
                    pc.starts_with(symbol_col, "ETH-"),
                )
                n_before = len(table)
                table = table.filter(keep_mask)
                n_after = len(table)

                _HOUR_US_SCALAR = pa.scalar(3_600_000_000, type=pa.int64())
                ts_col = table.column("timestamp")
                hour_us_arr = pc.multiply(
                    pc.divide(ts_col, _HOUR_US_SCALAR),
                    _HOUR_US_SCALAR,
                )
                table = table.append_column("hour_us", hour_us_arr)
                timer.stop(rows=n_after)
                if n_before != n_after:
                    log.info("  Arrow pre-filter: %dk → %dk rows (-%dk non-BTC/ETH)",
                             n_before // 1000, n_after // 1000,
                             (n_before - n_after) // 1000)

                # --- Step 4: Arrow → Pandas ---
                timer.start("to_pandas")
                df = table.to_pandas()
                del table
                timer.stop(rows=n_after)
                elapsed_pd = timer._totals["to_pandas"] / timer._counts["to_pandas"]
                log.info("  Arrow→Pandas %dk rows in %.1fs (%.0fk rows/s)",
                         n_after // 1000, elapsed_pd,
                         n_after / 1000 / (elapsed_pd + 0.001))

                # --- Step 5: Resample + finalize ---
                timer.start("resample")
                hourly = _resample_df_to_hourly(df, hourly_pick=hourly_pick)
                del df
                hourly = _finalize_hourly_output(hourly)
                timer.stop(rows=len(hourly))

            finally:
                # Cleanup temp CSV if fallback was used
                if tmp_csv and tmp_csv.exists():
                    for _retry in range(5):
                        try:
                            tmp_csv.unlink()
                            break
                        except PermissionError:
                            _release_memory()
                            time.sleep(0.3)

        except Exception as exc:
            if auto_mode and _is_memory_pressure_error(exc):
                bulk_fallback_exc = exc
                log.warning("  Bulk path hit memory pressure (%s), switching to stream mode", exc)
                _release_memory()
                read_mode = "stream"
            else:
                raise

    # ====================================================================
    # HYBRID MODE: split file first, then bulk-read each chunk
    # ====================================================================
    if read_mode == "hybrid":
        tmp_csv = _decompress_to_tmpfile(zst_path)
        chunk_dir = tmp_csv.parent / f"{tmp_csv.stem}.chunks"
        use_disk_staging = (_hybrid_staging == "disk")
        if use_disk_staging:
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_dir.mkdir(parents=True, exist_ok=True)
        try:
            read_options = pa_csv.ReadOptions(use_threads=(_cpu_limit != 1))
            hourly_accum: Optional[pd.DataFrame] = None
            _HOUR_US_SCALAR_H = pa.scalar(3_600_000_000, type=pa.int64())
            chunk_count = 0
            total_chunk_bytes = 0
            target_chunk_mb = max(16, _hybrid_chunk_mb)

            if use_disk_staging:
                chunk_iter = _iter_csv_chunk_files(tmp_csv, target_chunk_mb, chunk_dir)
            else:
                chunk_iter = _iter_csv_chunk_bytes(tmp_csv, target_chunk_mb)

            chunk_iter_obj = iter(chunk_iter)
            while True:
                timer.start("csv_split")
                try:
                    chunk = next(chunk_iter_obj)
                except StopIteration:
                    timer.stop()
                    break

                chunk_count += 1

                if use_disk_staging:
                    chunk_path = chunk
                    chunk_bytes = chunk_path.stat().st_size
                    total_chunk_bytes += chunk_bytes
                    timer.stop(nbytes=chunk_bytes)

                    timer.start("csv_read")
                    table = pa_csv.read_csv(
                        chunk_path,
                        convert_options=convert_options,
                        read_options=read_options,
                        parse_options=parse_options,
                    )
                else:
                    chunk_blob = chunk
                    chunk_bytes = len(chunk_blob)
                    total_chunk_bytes += chunk_bytes
                    timer.stop(nbytes=chunk_bytes)

                    timer.start("csv_read")
                    table = pa_csv.read_csv(
                        pa.BufferReader(chunk_blob),
                        convert_options=convert_options,
                        read_options=read_options,
                        parse_options=parse_options,
                    )

                rows_chunk = len(table)
                total_rows += rows_chunk
                timer.stop(rows=rows_chunk)

                timer.start("arrow_filter")
                symbol_col = table.column("symbol")
                keep_mask = pc.or_(
                    pc.starts_with(symbol_col, "BTC-"),
                    pc.starts_with(symbol_col, "ETH-"),
                )
                table = table.filter(keep_mask)
                rows_after = len(table)
                ts_col = table.column("timestamp")
                hour_us_arr = pc.multiply(
                    pc.divide(ts_col, _HOUR_US_SCALAR_H),
                    _HOUR_US_SCALAR_H,
                )
                table = table.append_column("hour_us", hour_us_arr)
                timer.stop(rows=rows_after)

                timer.start("to_pandas")
                chunk_df = table.to_pandas()
                del table
                timer.stop(rows=len(chunk_df))

                timer.start("resample")
                hourly_part = _resample_df_to_hourly(chunk_df, hourly_pick=hourly_pick)
                del chunk_df
                hourly_accum = _merge_hourly_chunk(hourly_accum, hourly_part, hourly_pick)
                timer.stop(rows=(len(hourly_accum) if hourly_accum is not None else 0))
                del hourly_part

                if chunk_count % 2 == 0:
                    _maybe_memory_backpressure("hybrid")

                if use_disk_staging:
                    try:
                        chunk_path.unlink()
                    except OSError:
                        pass
                else:
                    del chunk_blob

            log.info("  Hybrid split: %d chunk(s), %.0f MB total, target chunk=%d MB (process-as-you-split, staging=%s)",
                     chunk_count, total_chunk_bytes / (1024 * 1024), target_chunk_mb, _hybrid_staging)

            if hourly_accum is None or hourly_accum.empty:
                hourly = pd.DataFrame()
            else:
                hourly = _finalize_hourly_output(hourly_accum)
                del hourly_accum

            elapsed_read = timer._totals.get("csv_read", 0.0)
            if elapsed_read > 0:
                log.info("  Hybrid-read %dk rows in %.1fs (%.0fk rows/s)",
                         total_rows // 1000, elapsed_read,
                         total_rows / 1000 / (elapsed_read + 0.001))
        finally:
            if tmp_csv.exists():
                try:
                    tmp_csv.unlink()
                except OSError:
                    pass
            if use_disk_staging and chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)

    # ====================================================================
    # STREAM MODE: streaming decompress → per-batch processing
    # ====================================================================
    if read_mode == "stream":
        if bulk_fallback_exc is not None:
            log.info("  Stream fallback activated after bulk memory error")
        # block_size is C long — cap at 2 GB on Windows (2^31-1)
        _MAX_BLOCK = 2 * 1024 * 1024 * 1024 - 1  # 2 147 483 647
        _configured_mb = _csv_block_mb
        _min_mb = 4
        _block = min(max(_min_mb * 1024 * 1024, int(_configured_mb * 1024 * 1024)), _MAX_BLOCK)
        log.info("  Stream block_size: %d MB (configured %d MB, capped to 2 GB)",
                 _block // (1024 * 1024), _configured_mb)
        read_options = pa_csv.ReadOptions(
            use_threads=(_cpu_limit != 1),
            block_size=_block,
        )

        # Try streaming decompression (no temp file); fall back if needed
        tmp_csv: Optional[Path] = None
        zst_fh = None
        zst_stream = None
        zstd_proc: Optional[subprocess.Popen] = None
        csv_source = None
        reader = None
        stream_mode = "temp-file"

        try:
            csv_source, zst_fh, zst_stream, zstd_proc, stream_mode = _open_streaming_zstd_source(zst_path)
            if zstd_proc is not None:
                log.info("  Stream mode: system zstd pipe → PyArrow (no temp file)")
            else:
                log.info("  Stream mode: python-zstd stream → PyArrow (no temp file)")
        except Exception as e:
            log.warning("  Streaming decompress init failed (%s), using temp file", e)
            if zst_stream:
                try: zst_stream.close()
                except Exception: pass
            if zst_fh:
                try: zst_fh.close()
                except Exception: pass
            if zstd_proc and zstd_proc.stdout:
                try: zstd_proc.stdout.close()
                except Exception: pass
            if zstd_proc:
                _terminate_process_tree(zstd_proc, timeout=1.0)
            zst_fh = None
            zst_stream = None
            zstd_proc = None
            tmp_csv = _decompress_to_tmpfile(zst_path)
            csv_source = tmp_csv
            stream_mode = "temp-file"

        try:
            reader = pa_csv.open_csv(
                csv_source,
                convert_options=convert_options,
                read_options=read_options,
                parse_options=parse_options,
            )

            batch_queue: queue.Queue = queue.Queue(maxsize=_stream_queue_depth)
            part_queue: queue.Queue = queue.Queue(maxsize=_stream_queue_depth)
            _SENTINEL = object()
            producer_error: Dict[str, Exception] = {}
            consumer_error: Dict[str, Exception] = {}
            stop_event = threading.Event()
            total_rows_counter = [0]
            _HOUR_US_SCALAR_S = pa.scalar(3_600_000_000, type=pa.int64())

            def _queue_put_with_stop(q: queue.Queue, item: object) -> bool:
                while True:
                    if stop_event.is_set() and item is not _SENTINEL:
                        return False
                    try:
                        q.put(item, timeout=0.2)
                        return True
                    except queue.Full:
                        continue

            def _producer():
                try:
                    reader_iter = iter(reader)
                    producer_batch_idx = 0
                    while True:
                        if stop_event.is_set():
                            break
                        timer.start("csv_read")
                        read_peak = _PeakSampler(
                            "stream.csv_read",
                            enabled=_mem_diagnostics,
                            interval_ms=_mem_peak_interval_ms,
                            batch=(producer_batch_idx + 1),
                        )
                        read_peak.start()
                        try:
                            batch = next(reader_iter)
                        except StopIteration:
                            read_peak.stop(event="eof")
                            timer.stop()
                            break
                        except Exception:
                            read_peak.stop(event="error")
                            raise
                        read_peak.stop(rows=len(batch))
                        timer.stop(rows=len(batch))

                        if len(batch) == 0:
                            continue

                        timer.start("arrow_filter")
                        sym_col = batch.column("symbol")
                        bm = pc.or_(
                            pc.starts_with(sym_col, "BTC-"),
                            pc.starts_with(sym_col, "ETH-"),
                        )
                        batch = batch.filter(bm)
                        if len(batch) == 0:
                            timer.stop()
                            continue

                        ts_b = batch.column("timestamp")
                        hu_b = pc.multiply(
                            pc.divide(ts_b, _HOUR_US_SCALAR_S), _HOUR_US_SCALAR_S,
                        )
                        batch = batch.append_column("hour_us", hu_b)
                        timer.stop(rows=len(batch))
                        producer_batch_idx += 1
                        if not _queue_put_with_stop(batch_queue, batch):
                            break
                except Exception as e:
                    producer_error["err"] = e
                    stop_event.set()
                finally:
                    _queue_put_with_stop(batch_queue, _SENTINEL)

            def _consumer():
                try:
                    while True:
                        if stop_event.is_set() and batch_queue.empty():
                            break
                        try:
                            item = batch_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if item is _SENTINEL:
                            break
                        batch = item

                        batch_rows = len(batch)
                        total_rows_counter[0] += batch_rows

                        if _pandas_slice_rows > 0 and batch_rows > _pandas_slice_rows:
                            slice_count = 0
                            for offset in range(0, batch_rows, _pandas_slice_rows):
                                if stop_event.is_set():
                                    break
                                slice_count += 1
                                slice_len = min(_pandas_slice_rows, batch_rows - offset)
                                batch_slice = batch.slice(offset, slice_len)

                                timer.start("to_pandas")
                                rss_before_pd = _get_rss_mb() if _mem_diagnostics else None
                                arrow_before_pd = _get_arrow_pool_mb() if _mem_diagnostics else None
                                pd_peak = _PeakSampler(
                                    "stream.to_pandas_slice",
                                    enabled=_mem_diagnostics,
                                    interval_ms=_mem_peak_interval_ms,
                                )
                                pd_peak.start()
                                batch_df = batch_slice.to_pandas()
                                pd_peak.stop(rows=len(batch_df))
                                timer.stop(rows=len(batch_df))
                                _mem_diag_delta(
                                    "stream.to_pandas_slice",
                                    rss_before_pd,
                                    arrow_before_pd,
                                    slice=slice_count,
                                    rows=len(batch_df),
                                )

                                timer.start("resample")
                                hourly_part = _resample_df_to_hourly(batch_df, hourly_pick=hourly_pick)
                                timer.stop(rows=len(hourly_part))
                                if not _queue_put_with_stop(part_queue, hourly_part):
                                    del hourly_part
                                    break
                                del hourly_part
                                del batch_df, batch_slice
                            if stop_event.is_set():
                                del batch
                                break
                        else:
                            timer.start("to_pandas")
                            rss_before_pd = _get_rss_mb() if _mem_diagnostics else None
                            arrow_before_pd = _get_arrow_pool_mb() if _mem_diagnostics else None
                            pd_peak = _PeakSampler(
                                "stream.to_pandas",
                                enabled=_mem_diagnostics,
                                interval_ms=_mem_peak_interval_ms,
                            )
                            pd_peak.start()
                            batch_df = batch.to_pandas()
                            pd_peak.stop(rows=len(batch_df))
                            timer.stop(rows=len(batch_df))
                            _mem_diag_delta(
                                "stream.to_pandas",
                                rss_before_pd,
                                arrow_before_pd,
                                rows=len(batch_df),
                            )

                            timer.start("resample")
                            hourly_part = _resample_df_to_hourly(batch_df, hourly_pick=hourly_pick)
                            timer.stop(rows=len(hourly_part))
                            if not _queue_put_with_stop(part_queue, hourly_part):
                                del hourly_part
                                del batch_df
                                del batch
                                break
                            del hourly_part
                            del batch_df

                        del batch
                except Exception as e:
                    consumer_error["err"] = e
                    stop_event.set()
                finally:
                    _queue_put_with_stop(part_queue, _SENTINEL)

            producer_t = threading.Thread(target=_producer, daemon=True)
            consumer_t = threading.Thread(target=_consumer, daemon=True)
            producer_t.start()
            consumer_t.start()

            hourly_accum: Optional[pd.DataFrame] = None
            pending_parts: List[pd.DataFrame] = []
            merge_round = 0

            while True:
                part = part_queue.get()
                if part is _SENTINEL:
                    break
                pending_parts.append(part)

                if len(pending_parts) >= max(1, _stream_merge_every_batches):
                    merge_round += 1
                    timer.start("resample")
                    rss_before_merge = _get_rss_mb() if _mem_diagnostics else None
                    arrow_before_merge = _get_arrow_pool_mb() if _mem_diagnostics else None
                    merge_peak = _PeakSampler(
                        "stream.merge_pending",
                        enabled=_mem_diagnostics,
                        interval_ms=_mem_peak_interval_ms,
                        merge_round=merge_round,
                    )
                    merge_peak.start()
                    chunk_to_merge = pd.concat(pending_parts, ignore_index=True)
                    pending_parts.clear()
                    hourly_accum = _merge_hourly_chunk(hourly_accum, chunk_to_merge, hourly_pick)
                    merge_peak.stop(accum_rows=(len(hourly_accum) if hourly_accum is not None else 0))
                    timer.stop(rows=(len(hourly_accum) if hourly_accum is not None else 0))
                    del chunk_to_merge
                    _mem_diag_delta(
                        "stream.merge_pending",
                        rss_before_merge,
                        arrow_before_merge,
                        merge_round=merge_round,
                        accum_rows=(len(hourly_accum) if hourly_accum is not None else 0),
                    )
                    _maybe_memory_backpressure("stream")

                if producer_error.get("err") or consumer_error.get("err"):
                    stop_event.set()
                    break

            stop_event.set()
            producer_t.join(timeout=10)
            consumer_t.join(timeout=10)

            if producer_error.get("err"):
                raise producer_error["err"]
            if consumer_error.get("err"):
                raise consumer_error["err"]

            if pending_parts:
                timer.start("resample")
                rss_before_final_merge = _get_rss_mb() if _mem_diagnostics else None
                arrow_before_final_merge = _get_arrow_pool_mb() if _mem_diagnostics else None
                final_merge_peak = _PeakSampler(
                    "stream.final_merge",
                    enabled=_mem_diagnostics,
                    interval_ms=_mem_peak_interval_ms,
                )
                final_merge_peak.start()
                chunk_to_merge = pd.concat(pending_parts, ignore_index=True)
                pending_parts.clear()
                hourly_accum = _merge_hourly_chunk(hourly_accum, chunk_to_merge, hourly_pick)
                final_merge_peak.stop(accum_rows=(len(hourly_accum) if hourly_accum is not None else 0))
                timer.stop(rows=(len(hourly_accum) if hourly_accum is not None else 0))
                del chunk_to_merge
                _mem_diag_delta(
                    "stream.final_merge",
                    rss_before_final_merge,
                    arrow_before_final_merge,
                    accum_rows=(len(hourly_accum) if hourly_accum is not None else 0),
                )

            total_rows = total_rows_counter[0]
            if hourly_accum is None or hourly_accum.empty:
                hourly = pd.DataFrame()
            else:
                hourly = _finalize_hourly_output(hourly_accum)
                del hourly_accum

            elapsed_read = timer._totals.get("csv_read", 0.0)
            if elapsed_read > 0:
                log.info("  Stream-read %dk rows in %.1fs (%.0fk rows/s)",
                         total_rows // 1000, elapsed_read,
                         total_rows / 1000 / (elapsed_read + 0.001))

            if zstd_proc is not None:
                if zstd_proc.stdout is not None:
                    try:
                        zstd_proc.stdout.close()
                    except Exception:
                        pass
                _finalize_streaming_zstd_proc(zstd_proc, timeout=30.0)
                zstd_proc = None
                log.info("  Stream source finalized via %s", stream_mode)

        finally:
            if reader is not None and hasattr(reader, "close"):
                try:
                    reader.close()
                except Exception:
                    pass
            # Close streaming handles
            if zst_stream:
                try: zst_stream.close()
                except Exception: pass
            if zst_fh:
                try: zst_fh.close()
                except Exception: pass
            if zstd_proc is not None:
                if zstd_proc.stdout is not None:
                    try:
                        zstd_proc.stdout.close()
                    except Exception:
                        pass
                try:
                    _finalize_streaming_zstd_proc(zstd_proc, timeout=5.0)
                except Exception:
                    _terminate_process_tree(zstd_proc, timeout=1.0)
            # Cleanup temp CSV if fallback was used
            if tmp_csv and tmp_csv.exists():
                for _retry in range(5):
                    try:
                        tmp_csv.unlink()
                        break
                    except PermissionError:
                        _release_memory()
                        time.sleep(0.3)

    elapsed_rs = timer._totals.get("resample", 0)
    rs_count = timer._counts.get("resample", 1)
    log.info("  Resampled to %d hourly rows in %.1fs",
             len(hourly), elapsed_rs / rs_count)

    elapsed_total = time.time() - t0
    log.info("  Total: %dk → %d rows in %.1fs",
             total_rows // 1000, len(hourly), elapsed_total)
    if process_peak_sampler is not None:
        process_peak_sampler.stop(total_rows=total_rows, output_rows=len(hourly))
    _log_mem("after process_zst_to_hourly")
    _report_mem_diag_hotspots(context=zst_path.name)
    return hourly


# ---------------------------------------------------------------------------
# Write Parquet (append-friendly: merge with existing monthly file)
# ---------------------------------------------------------------------------


def append_to_parquet(df: pd.DataFrame):
    """
    Write hourly data to Parquet, partitioned by underlying/year-month.
    If month file already exists, merge (replacing overlap dates).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    # Derive year-month from hour column (vectorised integer, not strftime)
    df["_year"] = df["hour"].dt.year
    df["_month"] = df["hour"].dt.month

    for underlying in df["underlying"].unique():
        mask_ul = df["underlying"] == underlying
        ul_dir = OUTPUT_DIR / underlying
        ul_dir.mkdir(parents=True, exist_ok=True)

        sub = df.loc[mask_ul]
        for (year, month), grp in sub.groupby(["_year", "_month"]):
            new_data = grp.drop(columns=["_year", "_month", "underlying"], errors="ignore")

            out = output_path(underlying, int(year), int(month))

            if out.exists():
                # Merge: existing data for other dates + new data
                rss_before_read = _get_rss_mb() if _mem_diagnostics else None
                arrow_before_read = _get_arrow_pool_mb() if _mem_diagnostics else None
                existing = pd.read_parquet(out)
                _mem_diag_delta(
                    "parquet.read_existing",
                    rss_before_read,
                    arrow_before_read,
                    file=out.name,
                    existing_rows=len(existing),
                )
                if "hourly_pick" in new_data.columns and "hourly_pick" not in existing.columns:
                    existing["hourly_pick"] = "close"
                # Identify dates in new data
                new_dates = new_data["hour"].dt.date.unique()
                # Remove those dates from existing
                if "hour" in existing.columns:
                    keep_mask = ~existing["hour"].dt.date.isin(new_dates)
                    existing = existing[keep_mask]
                rss_before_month_merge = _get_rss_mb() if _mem_diagnostics else None
                arrow_before_month_merge = _get_arrow_pool_mb() if _mem_diagnostics else None
                merged = pd.concat([existing, new_data], ignore_index=True)
                del existing  # free immediately
                sort_keys = ["symbol", "hour"]
                dedup_keys = ["symbol", "hour"]
                if "hourly_pick" in merged.columns:
                    sort_keys.append("hourly_pick")
                    dedup_keys.append("hourly_pick")
                merged.sort_values(sort_keys, inplace=True)
                merged.drop_duplicates(subset=dedup_keys, keep="last", inplace=True)
                _mem_diag_delta(
                    "parquet.month_merge",
                    rss_before_month_merge,
                    arrow_before_month_merge,
                    file=out.name,
                    merged_rows=len(merged),
                )
            else:
                sort_keys = ["symbol", "hour"]
                if "hourly_pick" in new_data.columns:
                    sort_keys.append("hourly_pick")
                merged = new_data.sort_values(sort_keys)

            # Write with compression
            timer.start("parquet_write")
            merged.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
            fsize = out.stat().st_size
            timer.stop(rows=len(merged), nbytes=fsize)
            log.info("  Wrote %s: %d rows (%.1f MB)",
                     out, len(merged), fsize / 1024 / 1024)
            del merged  # free after each write

    # Drop temp columns added to caller's df
    df.drop(columns=["_year", "_month"], inplace=True, errors="ignore")


def _summarize_target_date_availability(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Summarize hourly availability for one processed date by underlying."""
    if df.empty or "hour" not in df.columns or "underlying" not in df.columns:
        return pd.DataFrame(columns=["underlying", "unique_hours", "rows", "unique_symbols", "availability_pct"])

    target_mask = df["hour"].dt.date == target_date
    target_df = df.loc[target_mask, ["underlying", "hour", "symbol"]].copy()
    if target_df.empty:
        return pd.DataFrame(columns=["underlying", "unique_hours", "rows", "unique_symbols", "availability_pct"])

    summary = target_df.groupby("underlying", sort=True).agg(
        unique_hours=("hour", lambda s: int(s.nunique())),
        rows=("symbol", "size"),
        unique_symbols=("symbol", lambda s: int(s.nunique())),
    ).reset_index()
    summary["availability_pct"] = summary["unique_hours"] / 24.0 * 100.0
    return summary.sort_values("underlying").reset_index(drop=True)


def _delete_output_for_date(target_date: date, underlyings: list[str]) -> None:
    """Delete one date's rows from existing monthly parquet outputs."""
    for underlying in sorted({str(u).upper() for u in underlyings if str(u).strip()}):
        out = output_path(underlying, target_date.year, target_date.month)
        if not out.exists():
            continue
        try:
            existing = pd.read_parquet(out)
        except Exception as e:
            log.warning("Could not read existing output %s for cleanup: %s", out, e)
            continue
        if "hour" not in existing.columns:
            continue
        keep_mask = existing["hour"].dt.date != target_date
        removed = int((~keep_mask).sum())
        if removed <= 0:
            continue

        cleaned = existing.loc[keep_mask].copy()
        if cleaned.empty:
            try:
                out.unlink()
                log.warning("Deleted output %s after removing %s (%d rows)", out, target_date, removed)
            except OSError as e:
                log.warning("Failed to delete empty output %s: %s", out, e)
            continue

        cleaned.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        log.warning("Removed %d existing row(s) for %s from %s", removed, target_date, out)


def _scan_existing_daily_availability(processed_dates: set[str]) -> pd.DataFrame:
    """Scan existing monthly parquet outputs and summarize daily availability."""
    columns = ["date", "underlying", "unique_hours", "rows", "unique_symbols", "availability_pct"]
    if not processed_dates:
        return pd.DataFrame(columns=columns)

    processed_dates_sorted = sorted({str(d) for d in processed_dates if str(d).strip()})
    processed_dates_set = set(processed_dates_sorted)
    rows: list[pd.DataFrame] = []

    for underlying in sorted(_VALID_UNDERLYINGS):
        ul_dir = OUTPUT_DIR / underlying
        if not ul_dir.exists():
            continue
        for pq_path in sorted(ul_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pq_path, columns=["hour", "symbol"])
            except Exception as e:
                log.warning("Availability scan skipped %s: %s", pq_path, e)
                continue
            if df.empty or "hour" not in df.columns:
                continue
            day_str = df["hour"].dt.strftime("%Y-%m-%d")
            mask = day_str.isin(processed_dates_set)
            if not mask.any():
                continue
            sub = pd.DataFrame({
                "date": day_str[mask],
                "hour": df.loc[mask, "hour"],
                "symbol": df.loc[mask, "symbol"],
            })
            summary = sub.groupby("date", sort=True).agg(
                unique_hours=("hour", lambda s: int(s.nunique())),
                rows=("symbol", "size"),
                unique_symbols=("symbol", lambda s: int(s.nunique())),
            ).reset_index()
            summary["underlying"] = underlying
            rows.append(summary)

    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "unique_hours", "rows", "unique_symbols", "underlying"])

    missing_rows = []
    for underlying in sorted(_VALID_UNDERLYINGS):
        seen_dates = set(report.loc[report["underlying"] == underlying, "date"]) if not report.empty else set()
        for d_str in processed_dates_sorted:
            if d_str not in seen_dates:
                missing_rows.append({
                    "date": d_str,
                    "unique_hours": 0,
                    "rows": 0,
                    "unique_symbols": 0,
                    "underlying": underlying,
                })
    if missing_rows:
        report = pd.concat([report, pd.DataFrame(missing_rows)], ignore_index=True)

    report["availability_pct"] = report["unique_hours"].astype(float) / 24.0 * 100.0
    report = report[columns]
    return report.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Progress tracking (checkpoint)
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = OUTPUT_DIR / ".checkpoint"


def load_checkpoint() -> set:
    """Load set of already-processed date strings."""
    if CHECKPOINT_FILE.exists():
        return set(CHECKPOINT_FILE.read_text().strip().split("\n"))
    return set()


def save_checkpoint(processed: set):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text("\n".join(sorted(processed)))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_single_date(
    d: date,
    force: bool = False,
    use_cache: bool = True,
    cache_only: bool = False,
    keep_cache: bool = False,
    hourly_pick: str = "both",
) -> bool:
    """Download (if needed) and process one date. Returns True on success."""
    max_attempts = 1 if cache_only else 2  # retry once on corrupt cache
    zst: Optional[Path] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if cache_only:
                zst = cache_path(d)
                if not zst.exists():
                    log.debug("No cache for %s, skipping", d)
                    return False
            else:
                # On retry, force re-download (cache was corrupt)
                zst = download_file(d, force=(force or attempt > 1))

            df = process_zst_to_hourly(
                zst,
                data_date=d,
                hourly_pick=hourly_pick,
            )
            if df.empty:
                log.warning("No data for %s", d)
                return False

            availability = _summarize_target_date_availability(df, d)
            if availability.empty:
                _delete_output_for_date(d, sorted(df["underlying"].astype(str).str.upper().unique()))
                raise RuntimeError(f"No hourly rows found for target date {d}")

            bad_availability = availability[availability["availability_pct"] < MIN_DAILY_AVAILABILITY_PCT]
            if not bad_availability.empty:
                _delete_output_for_date(d, bad_availability["underlying"].astype(str).tolist())
                details = ", ".join(
                    f"{row.underlying}={row.availability_pct:.2f}% ({int(row.unique_hours)}/24h)"
                    for row in bad_availability.itertuples(index=False)
                )
                raise RuntimeError(
                    f"Daily availability below {MIN_DAILY_AVAILABILITY_PCT:.0f}% for {d}: {details}"
                )

            details = ", ".join(
                f"{row.underlying}={row.availability_pct:.2f}% ({int(row.unique_hours)}/24h)"
                for row in availability.itertuples(index=False)
            )
            log.info("  Availability %s: %s", d, details)

            append_to_parquet(df)
            del df
            _release_memory()
            _log_mem("after release")

            # Auto-delete downloaded .zst after successful processing
            if not keep_cache and zst.exists() and CACHE_DIR in zst.parents:
                zst.unlink()
                log.info("  Deleted cache %s", zst)
                # Clean up empty parent dirs
                for parent in [zst.parent, zst.parent.parent]:
                    try:
                        parent.rmdir()  # only removes if empty
                    except OSError:
                        break

            return True

        except requests.HTTPError as e:
            log.error("HTTP error for %s: %s", d, e)
            return False
        except (zstd.ZstdError, RuntimeError) as e:
            # Decompression / processing error — likely corrupt cache
            if zst and zst.exists() and CACHE_DIR in zst.parents:
                try:
                    zst.unlink()
                    log.warning("Deleted corrupted cache file: %s", zst)
                except OSError:
                    pass
            if attempt < max_attempts:
                log.warning(
                    "Attempt %d/%d failed for %s (%s), re-downloading...",
                    attempt, max_attempts, d, e,
                )
                continue  # retry with fresh download
            log.exception("Failed to process %s: %s", d, e)
            return False
        except Exception as e:
            if zst and zst.exists() and CACHE_DIR in zst.parents:
                try:
                    zst.unlink()
                    log.warning("Deleted corrupted cache file: %s", zst)
                except OSError:
                    pass
            log.exception("Failed to process %s: %s", d, e)
            return False

    return False  # should not reach here


def process_local_file(path: str):
    """Process a single local .zst file (for testing)."""
    global timer
    timer = StageTimer()  # fresh timer

    zst = Path(path)
    if not zst.exists():
        log.error("File not found: %s", zst)
        sys.exit(1)

    df = process_zst_to_hourly(zst, hourly_pick="both")
    if df.empty:
        log.error("No data extracted")
        sys.exit(1)

    append_to_parquet(df)
    log.info("Done! Output in %s", OUTPUT_DIR)

    # Print summary
    print("\n=== Summary ===")
    print(f"Input:  {zst} ({zst.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Rows:   {len(df):,}")
    print(f"Hours:  {df['hour'].nunique()}")
    print(f"Symbols: {df['symbol'].nunique()}")
    if "underlying" in df.columns:
        for ul in sorted(df["underlying"].unique()):
            n = (df["underlying"] == ul).sum()
            print(f"  {ul}: {n:,} rows")

    # Show output files
    print(f"\nOutput files:")
    for p in sorted(OUTPUT_DIR.rglob("*.parquet")):
        print(f"  {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")

    # Timing report
    print(timer.report())


def main():
    parser = argparse.ArgumentParser(
        description="Build hourly Parquet files from Deribit options tick data"
    )
    parser.add_argument("--local", type=str,
                        help="Process a single local .zst file (for testing)")
    parser.add_argument("--start", type=str, default="2023-03-20",
                        help="Start date YYYY-MM-DD (default: 2023-03-20)")
    parser.add_argument("--end", type=str, default="2026-03-24",
                        help="End date YYYY-MM-DD (default: 2026-03-24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List available dates without processing")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only process already-downloaded files")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep downloaded .zst files after processing")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP proxy for download, e.g. http://127.0.0.1:7890")
    parser.add_argument("--segments", type=int, default=8,
                        help="Number of parallel curl segments per file (default: 8, set 1 for single)")
    parser.add_argument("--hourly-pick", type=str, default="both",
                        choices=["first", "last", "both"],
                        help="Hourly snapshot selection: first=open, last=close, both=keep both (default: both)")
    parser.add_argument("--download-workers", type=int, default=1, dest="download_workers",
                        help="Number of concurrent download threads (default: 1, try 3 for faster pipeline)")
    parser.add_argument("--csv-read-mode", type=str, default="auto",
                        choices=["auto", "bulk", "stream", "hybrid", "polars-stream"],
                        help="CSV read strategy: auto=memory-aware choose, bulk=fastest/high-memory, stream=chunked low-memory, hybrid=split file then bulk-read each chunk, polars-stream=stream zstd into chunked Polars aggregation")
    parser.add_argument("--csv-block-mb", type=int, default=1024,
                        help="Chunk size for stream mode in MB (default: 1024)")
    parser.add_argument("--polars-stream-chunk-mb", type=int, default=64,
                        help="Chunk size for polars-stream mode in MB (default: 64)")
    parser.add_argument("--hybrid-chunk-mb", type=int, default=1024,
                        help="Target chunk size for hybrid mode in MB (default: 1024, minimum 16)")
    parser.add_argument("--hybrid-staging", type=str, default="memory",
                        choices=["memory", "disk"],
                        help="Hybrid chunk staging backend: memory=no chunk temp files, disk=chunk files")
    parser.add_argument("--stream-merge-every", type=int, default=2,
                        help="Merge stream batches every N chunks (default: 2, lower=less memory, higher=less CPU)")
    parser.add_argument("--stream-queue-depth", type=int, default=2,
                        help="Bounded queue depth for stream pipeline stages (default: 2)")
    parser.add_argument("--mem-soft-limit-mb", type=int, default=0,
                        help="RSS soft limit in MB for memory cleanup backpressure (0=auto)")
    parser.add_argument("--mem-hard-limit-mb", type=int, default=0,
                        help="RSS hard limit in MB for throttle backpressure (0=auto)")
    parser.add_argument("--mem-diagnostics", action="store_true",
                        help="Enable detailed memory diagnostics for stream/parquet hotspots")
    parser.add_argument("--mem-diag-every-batches", type=int, default=2,
                        help="Emit stream memory heartbeat every N batches when diagnostics enabled (default: 2)")
    parser.add_argument("--mem-peak-interval-ms", type=int, default=100,
                        help="Sampling interval (ms) for stage peak memory diagnostics (default: 100)")
    parser.add_argument("--pandas-slice-rows", type=int, default=0,
                        help="Split each Arrow batch into smaller row chunks before to_pandas in stream mode (0=disable)")
    parser.add_argument("--csv-aggressive-parse", action="store_true",
                        help="Enable aggressive CSV parsing options (check_utf8=False, quote parsing off) for faster csv_read")
    parser.add_argument("--cpu-limit", type=int, default=0,
                        help="Limit CPU thread usage for Arrow/BLAS (0=auto, 1=lowest CPU, 2-4 recommended)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and reprocess all dates")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Hidden args for subprocess worker isolation
    parser.add_argument("--_worker-date", dest="_worker_date", type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_timer-file", dest="_timer_file", type=str,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()
    setup_logging(args.verbose)
    install_interrupt_handler()

    _apply_cpu_limits(args.cpu_limit)

    global _proxy, _session, _download_segments, timer, _csv_read_mode, _csv_block_mb, _hybrid_chunk_mb, _hybrid_staging
    global _stream_merge_every_batches, _stream_queue_depth, _mem_soft_limit_mb, _mem_hard_limit_mb
    global _mem_diagnostics, _mem_diag_every_batches, _mem_peak_interval_ms, _pandas_slice_rows, _csv_aggressive_parse
    global _polars_stream_chunk_mb

    _csv_read_mode = args.csv_read_mode
    _csv_block_mb = max(16, args.csv_block_mb)
    _polars_stream_chunk_mb = max(16, args.polars_stream_chunk_mb)
    _hybrid_chunk_mb = max(16, args.hybrid_chunk_mb)
    _hybrid_staging = args.hybrid_staging
    _stream_merge_every_batches = max(1, args.stream_merge_every)
    _stream_queue_depth = max(1, args.stream_queue_depth)
    _mem_diagnostics = args.mem_diagnostics
    _mem_diag_every_batches = max(1, args.mem_diag_every_batches)
    _mem_peak_interval_ms = max(20, args.mem_peak_interval_ms)
    _pandas_slice_rows = max(0, args.pandas_slice_rows)
    _csv_aggressive_parse = bool(args.csv_aggressive_parse)

    soft_auto, hard_auto = _auto_memory_limits()
    _mem_soft_limit_mb = args.mem_soft_limit_mb if args.mem_soft_limit_mb > 0 else soft_auto
    _mem_hard_limit_mb = args.mem_hard_limit_mb if args.mem_hard_limit_mb > 0 else hard_auto
    if _mem_hard_limit_mb <= _mem_soft_limit_mb:
        _mem_hard_limit_mb = _mem_soft_limit_mb + 512

    log.info(
        "Resource scheduler: stream-merge-every=%d, stream-queue-depth=%d, mem soft/hard=%d/%d MB",
        _stream_merge_every_batches,
        _stream_queue_depth,
        _mem_soft_limit_mb,
        _mem_hard_limit_mb,
    )
    if _mem_diagnostics:
        log.info(
            "Memory diagnostics: enabled (heartbeat every %d stream batch(es), peak sample=%dms)",
            _mem_diag_every_batches,
            _mem_peak_interval_ms,
        )
    if _pandas_slice_rows > 0:
        log.info("Pandas slice mode: enabled (%d rows per slice in stream mode)", _pandas_slice_rows)
    if _csv_aggressive_parse:
        log.info("CSV aggressive parse: enabled (check_utf8=False, quote parsing off)")
    if _csv_read_mode == "polars-stream":
        log.info("Polars-stream mode: enabled (%d MB chunks)", _polars_stream_chunk_mb)

    # ------------------------------------------------------------------
    # Subprocess worker mode: process ONE date then exit.
    # The OS reclaims ALL memory when this process terminates, solving
    # CPython pymalloc arena fragmentation across multi-day runs.
    # ------------------------------------------------------------------
    if args._worker_date:
        if args.proxy:
            _proxy = args.proxy
        elif os.environ.get("HTTPS_PROXY"):
            _proxy = os.environ["HTTPS_PROXY"]
        _download_segments = max(1, args.segments)
        timer = StageTimer()

        d = date.fromisoformat(args._worker_date)
        ok = process_single_date(
            d,
            force=args.force,
            cache_only=args.cache_only,
            keep_cache=args.keep_cache,
            hourly_pick=args.hourly_pick,
        )
        # Write timer data to file so parent can aggregate
        if args._timer_file:
            try:
                Path(args._timer_file).write_text(timer.to_json())
            except Exception:
                pass
        sys.exit(0 if ok else 1)

    # Set proxy (CLI > env)
    if args.proxy:
        _proxy = args.proxy
        _session = None  # reset session to pick up proxy
        log.info("Using proxy: %s", _proxy)
    elif os.environ.get("HTTPS_PROXY"):
        _proxy = os.environ["HTTPS_PROXY"]
        log.info("Using proxy from HTTPS_PROXY: %s", _proxy)

    _download_segments = max(1, args.segments)
    if _download_segments > 1:
        log.info("Downloader: curl/Schannel multi-segment (x%d)", _download_segments)
    else:
        log.info("Downloader: curl/Schannel single-connection")

    # --- Local file mode ---
    if args.local:
        process_local_file(args.local)
        return

    # --- Batch mode ---
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.dry_run:
        dates = discover_all_remote_dates()
        dates = [d for d in dates if start <= d <= end]
        print(f"\nAvailable dates in [{start} .. {end}]: {len(dates)}")
        for d in dates[:20]:
            print(f"  {d}")
        if len(dates) > 20:
            print(f"  ... and {len(dates) - 20} more")
        return

    # Reset timer for batch run
    timer = StageTimer()

    # Load checkpoint
    processed = load_checkpoint() if not args.force else set()

    if processed:
        log.info(
            "Startup availability scan: checking processed dates against %.0f%% threshold...",
            MIN_DAILY_AVAILABILITY_PCT,
        )
        availability_report = _scan_existing_daily_availability(processed)
        bad_rows = availability_report[
            availability_report["availability_pct"] < MIN_DAILY_AVAILABILITY_PCT
        ].copy()
        if not bad_rows.empty:
            bad_dates = sorted(set(str(x) for x in bad_rows["date"]))
            for d_str in bad_dates:
                target_date = date.fromisoformat(d_str)
                underlyings = bad_rows.loc[
                    bad_rows["date"] == d_str,
                    "underlying",
                ].astype(str).tolist()
                _delete_output_for_date(target_date, underlyings)
                processed.discard(d_str)
            save_checkpoint(processed)
            sample_details = ", ".join(
                f"{row.date}:{row.underlying}={row.availability_pct:.2f}%"
                for row in bad_rows.head(10).itertuples(index=False)
            )
            log.warning(
                "Startup availability scan found %d low-availability sample(s) across %d date(s); they will be re-downloaded. Samples: %s",
                len(bad_rows),
                len(bad_dates),
                sample_details or "-",
            )
        else:
            log.info("Startup availability scan passed: no date below %.0f%%", MIN_DAILY_AVAILABILITY_PCT)

    dates = discover_dates(start, end)
    remaining = [d for d in dates if d.isoformat() not in processed]

    log.info("Total dates: %d, already processed: %d, remaining: %d",
             len(dates), len(processed), len(remaining))

    success = 0
    failed = 0

    # ---------------------------------------------------------------
    # Pipeline: concurrent download → queue → serial subprocess
    # ---------------------------------------------------------------
    # Download threads fetch .zst files in parallel.  As each finishes
    # it puts the date onto a bounded queue.  The main thread pulls
    # from the queue and spawns one subprocess worker at a time (to
    # keep memory isolation). A bounded queue (max_ahead) prevents
    # downloading too far ahead, which would waste disk space.
    # ---------------------------------------------------------------

    n_dl_workers = max(1, args.download_workers)
    # Allow at most 2× workers items queued ahead of the processor
    max_ahead = max(2, n_dl_workers * 2)
    _SENTINEL = None  # signals "all downloads done"

    ready_q: queue.Queue = queue.Queue(maxsize=max_ahead)
    dl_error_dates: list = []  # dates that failed to download
    _pipeline_cancel = threading.Event()

    # -- real-time progress tracker --
    pstatus = _PipelineStatus(total=len(remaining))

    def _download_worker(dates_to_dl: list, force: bool, cache_only: bool):
        """Thread target: download each date's .zst and enqueue it."""
        for d in dates_to_dl:
            if _pipeline_cancel.is_set() or _SHUTDOWN_EVENT.is_set():
                break
            pstatus.dl_start(d)
            try:
                if cache_only:
                    zst = cache_path(d)
                    if not zst.exists():
                        log.debug("No cache for %s, skipping", d)
                        dl_error_dates.append(d)
                        pstatus.dl_finish(d, ok=False)
                        continue
                else:
                    # download_file is thread-safe (uses per-call curl process)
                    # Batch mode uses `_PipelineStatus` to render per-file download
                    # progress in a dedicated multi-line status block.
                    quiet = True
                    download_file(
                        d,
                        force=force,
                        quiet=quiet,
                        progress_cb=lambda done, total, _d=d: pstatus.dl_progress_update(_d, done, total),
                    )
            except Exception as e:
                log.error("Download failed for %s: %s", d, e)
                dl_error_dates.append(d)
                pstatus.dl_finish(d, ok=False)
                continue
            pstatus.dl_finish(d, ok=True)
            while not (_pipeline_cancel.is_set() or _SHUTDOWN_EVENT.is_set()):
                try:
                    ready_q.put(d, timeout=0.5)  # back-pressure + interruptible wait
                    break
                except queue.Full:
                    continue
            pstatus.set_queue_depth(ready_q.qsize())
        while True:
            try:
                ready_q.put(_SENTINEL, timeout=0.5)
                break
            except queue.Full:
                if _pipeline_cancel.is_set() or _SHUTDOWN_EVENT.is_set():
                    continue

    def _build_worker_cmd(d, args, timer_file):
        """Build subprocess command for one date."""
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker-date", d.isoformat(),
            "--_timer-file", str(timer_file),
            "--segments", str(args.segments),
            "--hourly-pick", args.hourly_pick,
            "--csv-read-mode", args.csv_read_mode,
            "--csv-block-mb", str(args.csv_block_mb),
            "--polars-stream-chunk-mb", str(args.polars_stream_chunk_mb),
            "--hybrid-chunk-mb", str(args.hybrid_chunk_mb),
            "--hybrid-staging", args.hybrid_staging,
            "--stream-merge-every", str(args.stream_merge_every),
            "--stream-queue-depth", str(args.stream_queue_depth),
            "--mem-soft-limit-mb", str(args.mem_soft_limit_mb),
            "--mem-hard-limit-mb", str(args.mem_hard_limit_mb),
            "--mem-diag-every-batches", str(args.mem_diag_every_batches),
            "--mem-peak-interval-ms", str(args.mem_peak_interval_ms),
            "--pandas-slice-rows", str(args.pandas_slice_rows),
            "--cpu-limit", str(args.cpu_limit),
        ]
        if args.csv_aggressive_parse:
            cmd.append("--csv-aggressive-parse")
        if args.mem_diagnostics:
            cmd.append("--mem-diagnostics")
        # Worker should never re-download; file is already in cache
        # So we do NOT pass --force here.
        if args.cache_only:
            cmd.append("--cache-only")
        if args.keep_cache:
            cmd.append("--keep-cache")
        if args.verbose:
            cmd.append("-v")
        if args.proxy:
            cmd.extend(["--proxy", args.proxy])
        return cmd

    # Split remaining dates into chunks for download threads
    if n_dl_workers == 1:
        dl_chunks = [remaining]
    else:
        dl_chunks = [[] for _ in range(n_dl_workers)]
        for idx, d in enumerate(remaining):
            dl_chunks[idx % n_dl_workers].append(d)

    log.info("Pipeline: %d download threads, queue depth %d, %d dates",
             n_dl_workers, max_ahead, len(remaining))

    # Start download threads
    dl_threads = []
    for chunk in dl_chunks:
        if not chunk:
            continue
        t = threading.Thread(
            target=_download_worker,
            args=(chunk, args.force, args.cache_only),
            daemon=True,
        )
        t.start()
        dl_threads.append(t)

    # Start background status ticker (auto-prints every 2s)
    pstatus.start_ticker()

    # How many SENTINEL values to expect (one per thread)
    sentinels_expected = len(dl_threads)
    sentinels_received = 0
    processed_count = 0

    try:
        while sentinels_received < sentinels_expected:
            if _SHUTDOWN_EVENT.is_set():
                raise KeyboardInterrupt()
            try:
                item = ready_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                sentinels_received += 1
                continue

            d = item
            processed_count += 1
            pstatus.set_queue_depth(ready_q.qsize())
            pstatus.proc_start(d)
            # Print final status before pausing ticker for subprocess
            pstatus.print_status(force=True)
            print(flush=True)  # newline to preserve status line
            log.info("=== [%d/%d] Processing %s ===",
                     processed_count, len(remaining), d)
            pstatus.pause_ticker()  # pause during subprocess

            # --- Spawn isolated subprocess per date ---
            timer_file = Path(tempfile.gettempdir()) / (
                f"_bhp_timer_{os.getpid()}_{d.isoformat()}.json"
            )
            cmd = _build_worker_cmd(d, args, timer_file)
            worker_proc = _popen_tracked(cmd)
            try:
                rc = _wait_process_interruptible(worker_proc, timeout=0)
            finally:
                _unregister_child_process(worker_proc)
            ok = rc == 0

            pstatus.proc_finish(ok=ok)
            pstatus.set_queue_depth(ready_q.qsize())
            pstatus.resume_ticker()  # resume after subprocess

            # Merge timing data from child
            if timer_file.exists():
                try:
                    timer.merge(timer_file.read_text())
                except Exception:
                    pass
                finally:
                    try:
                        timer_file.unlink()
                    except OSError:
                        pass

            if ok:
                processed.add(d.isoformat())
                save_checkpoint(processed)
                success += 1
            else:
                failed += 1

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected, stopping...", file=sys.stderr, flush=True)
        _SHUTDOWN_EVENT.set()
        _pipeline_cancel.set()
        pstatus.stop_ticker()
        for t in dl_threads:
            t.join(timeout=5)
        _force_exit(130)

    # Stop the status ticker
    pstatus.stop_ticker()
    pstatus.print_final()

    # Wait for download threads to finish cleanly
    for t in dl_threads:
        t.join(timeout=10)

    # Count download failures as processing failures
    failed += len(dl_error_dates)
    if dl_error_dates:
        log.warning("Download failed for %d dates: %s",
                    len(dl_error_dates),
                    ", ".join(d.isoformat() for d in dl_error_dates[:10]))
    log.info("=== Completed: %d success, %d failed ===", success, failed)

    # Final output summary
    total_size = sum(p.stat().st_size for p in OUTPUT_DIR.rglob("*.parquet"))
    total_files = len(list(OUTPUT_DIR.rglob("*.parquet")))
    log.info("Output: %d files, %.1f MB total in %s",
             total_files, total_size / 1024 / 1024, OUTPUT_DIR)

    # Timing report
    print(timer.report())


if __name__ == "__main__":
    main()

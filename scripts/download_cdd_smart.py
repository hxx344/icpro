"""Smart CDD downloader: batch by expiry, minimal API calls.

Key optimizations vs. previous approach:
1. Download by EXPIRY: 1 API call = all strikes for that expiry (50-100 instruments)
   Old approach: 1 API call per instrument = 50-100x more calls!
2. Daily expiries (0DTE) fit in 1 page (~127 rows) = exactly 1 call each
3. Adaptive rate limiting with 429 retry-after parsing
4. Full resume support via JSON checkpoint
5. Priority queue: most recent data first

Data gaps:
  2023: 365 expiry dates, 0% have OHLCV data
  2024: 341 expiry dates missing
  Total: ~706 expiry dates = ~706 API calls (vs ~34,000 instrument calls!)

Usage:
    python scripts/download_cdd_smart.py                 # All missing BTC
    python scripts/download_cdd_smart.py --year 2024     # Just 2024
    python scripts/download_cdd_smart.py --currency ETH  # ETH options
    python scripts/download_cdd_smart.py --status        # Show progress
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from cdd_secrets import get_cdd_api_token

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://api.cryptodatadownload.com/v1"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market_data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
INST_DIR = Path(__file__).resolve().parent.parent / "data" / "instruments"

# Rate limiting - conservative for CDD
INITIAL_DELAY = 20.0       # 20s between requests
MIN_DELAY = 12.0           # Never below 12s
MAX_DELAY = 300.0          # Cap at 5 minutes
BACKOFF_FACTOR = 1.5       # Increase delay by 50% on 429
RECOVERY_FACTOR = 0.95     # Reduce by 5% on success streak
SUCCESS_STREAK_FOR_SPEEDUP = 10

PAGE_LIMIT = 5000

# ---------------------------------------------------------------------------
# Expiry code helpers
# ---------------------------------------------------------------------------

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
MONTH_NAMES = {v: k for k, v in MONTHS.items()}
EXPIRY_RE = re.compile(r"^(BTC|ETH)-(\d{1,2})([A-Z]{3})(\d{2})-")


def expiry_code_to_date(code):
    """'27DEC24' -> date(2024, 12, 27)"""
    i = 0
    while i < len(code) and code[i].isdigit():
        i += 1
    day = int(code[:i])
    month_str = code[i:i + 3]
    year = 2000 + int(code[i + 3:])
    return date(year, MONTHS[month_str], day)


def date_to_expiry_code(d):
    """date(2024, 12, 27) -> '27DEC24'"""
    return f"{d.day}{MONTH_NAMES[d.month]}{d.year % 100:02d}"


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class CDDSession:
    """Synchronous CDD API client with adaptive rate limiting."""

    def __init__(self, token: str | None = None):
        token = token or get_cdd_api_token()
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {token}"
        self.delay = INITIAL_DELAY
        self.success_streak = 0
        self.stats = {
            "requests": 0, "pages": 0, "rows": 0,
            "rate_limited": 0, "errors": 0,
            "instruments_saved": 0, "expiries_done": 0,
        }
        self._last_request = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_request
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get(self, endpoint, params):
        """Rate-limited GET with retry on 429."""
        url = f"{API_BASE}{endpoint}"
        max_retries = 5

        for attempt in range(max_retries):
            self._wait()
            self._last_request = time.time()
            self.stats["requests"] += 1

            try:
                r = self.session.get(url, params=params, timeout=90)
            except requests.RequestException as e:
                self.stats["errors"] += 1
                wait = min(60 * (attempt + 1), 300)
                print(f"    Network error: {e}, retry in {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 200:
                self.success_streak += 1
                if self.success_streak >= SUCCESS_STREAK_FOR_SPEEDUP:
                    self.delay = max(MIN_DELAY, self.delay * RECOVERY_FACTOR)
                    self.success_streak = 0
                return r.json()

            elif r.status_code == 429:
                self.success_streak = 0
                self.stats["rate_limited"] += 1
                # Parse "Expected available in X seconds" from body
                retry_after = 60
                try:
                    body = r.json()
                    detail = body.get("detail", "")
                    m = re.search(r"(\d+)\s*seconds?", detail)
                    if m:
                        retry_after = int(m.group(1)) + 10
                except Exception:
                    pass
                self.delay = min(MAX_DELAY, self.delay * BACKOFF_FACTOR)
                wait = max(retry_after, self.delay)
                print(f"    429! wait {wait:.0f}s (delay={self.delay:.0f}s)")
                time.sleep(wait)

            elif r.status_code == 400:
                return None

            else:
                self.stats["errors"] += 1
                if attempt < max_retries - 1:
                    time.sleep(15 * (attempt + 1))
                else:
                    return None

        return None

    def get_paginated(self, endpoint, params):
        """GET with automatic pagination."""
        all_rows = []
        offset = 0
        while True:
            p = {**params, "limit": PAGE_LIMIT, "offset": offset}
            data = self.get(endpoint, p)
            if not data:
                break
            rows = data.get("result", [])
            all_rows.extend(rows)
            self.stats["pages"] += 1
            self.stats["rows"] += len(rows)
            pagination = data.get("pagination", {})
            if pagination.get("has_more") and pagination.get("next_offset"):
                offset = pagination["next_offset"]
            else:
                break
        return all_rows


# ---------------------------------------------------------------------------
# Data conversion and saving
# ---------------------------------------------------------------------------

def rows_to_instrument_dfs(rows):
    """Convert CDD rows to per-instrument DataFrames."""
    if not rows:
        return {}
    grouped = defaultdict(list)
    for row in rows:
        sym = row.get("symbol", "")
        if sym:
            grouped[sym].append(row)

    result = {}
    for sym, sym_rows in grouped.items():
        df = pd.DataFrame(sym_rows)
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["unix"], unit="ms", utc=True),
            "open": pd.to_numeric(df["open"], errors="coerce").fillna(0),
            "high": pd.to_numeric(df["high"], errors="coerce").fillna(0),
            "low": pd.to_numeric(df["low"], errors="coerce").fillna(0),
            "close": pd.to_numeric(df["close"], errors="coerce").fillna(0),
            "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0),
            "instrument_name": sym,
        }).sort_values("timestamp").reset_index(drop=True)
        result[sym] = out
    return result


def save_instruments(instrument_dfs, out_dir):
    """Save per-instrument parquet files, merging with existing."""
    saved = 0
    for sym, df in instrument_dfs.items():
        if df.empty:
            continue
        out_path = out_dir / f"{sym}.parquet"
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["timestamp"]
                ).sort_values("timestamp").reset_index(drop=True)
            except Exception:
                pass
        df.to_parquet(out_path, index=False)
        saved += 1
    return saved


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, currency):
        self.path = CACHE_DIR / f"cdd_smart_{currency.lower()}.json"
        self.currency = currency
        self.completed = set()
        self.load()

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.completed = set(data.get("completed_expiries", []))
            except Exception:
                self.completed = set()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "currency": self.currency,
            "completed_expiries": sorted(self.completed),
            "count": len(self.completed),
            "updated": datetime.utcnow().isoformat(),
        }, indent=2))

    def is_done(self, code):
        return code in self.completed

    def mark_done(self, code):
        self.completed.add(code)
        if len(self.completed) % 5 == 0:
            self.save()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_missing_expiries(currency, years=None):
    """Find expiry dates that need downloading."""
    inst_file = INST_DIR / f"{currency.lower()}_instruments.parquet"
    if not inst_file.exists():
        print(f"  No catalogue: {inst_file}")
        return []

    df = pd.read_parquet(inst_file)
    df["exp_date"] = pd.to_datetime(df["expiration_date"], utc=True).dt.date

    if years:
        df = df[df["exp_date"].apply(lambda d: d.year in years)]

    unique_dates = sorted(df["exp_date"].unique())
    print(f"  Catalogue: {len(df)} instruments, {len(unique_dates)} expiry dates")

    existing_files = {f.stem for f in DATA_DIR.glob(f"{currency}-*.parquet")}

    missing = []
    partial = []
    for exp_date in unique_dates:
        code = date_to_expiry_code(exp_date)
        exp_insts = df[df["exp_date"] == exp_date]["instrument_name"].tolist()
        with_data = sum(1 for inst in exp_insts if inst in existing_files)
        total = len(exp_insts)
        if with_data == 0:
            missing.append(code)
        elif with_data < total * 0.8:
            partial.append((code, with_data, total))

    print(f"  Fully missing: {len(missing)} expiry dates")
    print(f"  Partial (<80%): {len(partial)} expiry dates")

    all_needed = missing + [c for c, _, _ in partial]
    return all_needed


def download_expiries(client, currency, expiry_codes, checkpoint, out_dir):
    """Download OHLCV data by expiry code."""
    out_dir.mkdir(parents=True, exist_ok=True)

    remaining = [c for c in expiry_codes if not checkpoint.is_done(c)]
    total = len(expiry_codes)
    done = total - len(remaining)

    print(f"\n  {currency}: {total} total, {done} done, {len(remaining)} to go")
    if not remaining:
        print("  All done!")
        return

    # Most recent first
    remaining.sort(key=lambda c: expiry_code_to_date(c), reverse=True)

    start_time = time.time()

    for i, code in enumerate(remaining):
        exp_date = expiry_code_to_date(code)
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1) * 3600

        print(f"  [{done+i+1}/{total}] {code} ({exp_date}) ", end="", flush=True)

        rows = client.get_paginated(
            "/data/ohlc/deribit/options/",
            {"currency": currency, "expiry": code},
        )

        if rows:
            dfs = rows_to_instrument_dfs(rows)
            saved = save_instruments(dfs, out_dir)
            client.stats["instruments_saved"] += saved
            client.stats["expiries_done"] += 1
            print(f"OK {len(rows)} rows, {len(dfs)} instr "
                  f"[d={client.delay:.0f}s ~{rate:.0f}/hr]")
        else:
            print(f"empty [d={client.delay:.0f}s]")

        checkpoint.mark_done(code)

        # Progress report every 20
        if (i + 1) % 20 == 0:
            left = len(remaining) - i - 1
            avg = elapsed / (i + 1)
            eta_h = left * avg / 3600
            print(f"    === {done+i+1}/{total}, "
                  f"ETA {eta_h:.1f}h, "
                  f"429s: {client.stats['rate_limited']} ===")

    checkpoint.save()


def show_status(currency):
    """Show download progress."""
    ckpt = Checkpoint(currency)
    files = list(DATA_DIR.glob(f"{currency}-*.parquet"))
    print(f"\n{currency} Status:")
    print(f"  Completed expiries: {len(ckpt.completed)}")
    print(f"  Parquet files: {len(files)}")

    inst_file = INST_DIR / f"{currency.lower()}_instruments.parquet"
    if inst_file.exists():
        df = pd.read_parquet(inst_file)
        existing = {f.stem for f in files}
        have = sum(1 for n in df["instrument_name"] if n in existing)
        print(f"  Coverage: {have}/{len(df)} ({100*have/len(df):.1f}%)")

    year_re = re.compile(r"^[A-Z]+-\d+[A-Z]{3}(\d{2})-")
    yc = Counter()
    for f in files:
        m = year_re.match(f.name)
        if m:
            yc[2000 + int(m.group(1))] += 1
    for y in sorted(yc):
        print(f"    {y}: {yc[y]} files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart CDD batch downloader")
    parser.add_argument("--currency", "-c", default="BTC")
    parser.add_argument("--year", "-y", type=str, default=None,
                        help="Years (comma-sep), e.g. 2023,2024")
    parser.add_argument("--status", "-s", action="store_true")
    parser.add_argument("--delay", "-d", type=float, default=INITIAL_DELAY)
    parser.add_argument("--wait", "-w", type=int, default=0,
                        help="Wait N seconds before starting (for cooldown)")
    args = parser.parse_args()

    currency = args.currency.upper()

    if args.status:
        show_status(currency)
        return

    if args.wait > 0:
        print(f"Waiting {args.wait}s for API cooldown...")
        time.sleep(args.wait)

    print("=" * 60)
    print("CDD Smart Downloader - Batch by Expiry")
    print("=" * 60)
    print(f"Currency: {currency}")
    print(f"Delay: {args.delay}s")

    years = None
    if args.year:
        years = [int(y.strip()) for y in args.year.split(",")]
        print(f"Years: {years}")

    print("\nStep 1: Analyzing gaps...")
    codes = get_missing_expiries(currency, years)
    if not codes:
        print("All data present!")
        return

    est = len(codes) * args.delay / 3600
    print(f"\nStep 2: Downloading {len(codes)} expiries (~{est:.1f}h)")

    client = CDDSession()
    client.delay = args.delay
    ckpt = Checkpoint(currency)

    start = time.time()
    try:
        download_expiries(client, currency, codes, ckpt, DATA_DIR)
    except KeyboardInterrupt:
        print("\nInterrupted! Progress saved.")
        ckpt.save()
    finally:
        elapsed = time.time() - start
        s = client.stats
        print(f"\n{'='*60}")
        print(f"Duration: {elapsed/60:.1f}min ({elapsed/3600:.1f}h)")
        print(f"Requests: {s['requests']}, 429s: {s['rate_limited']}, "
              f"Errors: {s['errors']}")
        print(f"Expiries: {s['expiries_done']}, "
              f"Instruments: {s['instruments_saved']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

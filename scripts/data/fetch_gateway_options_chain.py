#!/usr/bin/env python3
"""Fetch Deribit ``options_chain`` daily files from the crypto data gateway and
fold them into ``data/options_hourly`` via ``build_hourly_parquet.py``.

Each day is downloaded into the cache layout ``build_hourly_parquet.py``
expects, converted by its single-date worker, then deleted, so disk usage stays
bounded to a few files.

Credentials are read from ``CDG_USER`` / ``CDG_TOKEN`` or ``--token-file``.

Usage (run from the directory that holds ``data/``):
    python scripts/data/fetch_gateway_options_chain.py --start 2026-03-25 --dry-run
    python scripts/data/fetch_gateway_options_chain.py --start 2026-03-25
    python scripts/data/fetch_gateway_options_chain.py --start 2026-03-25 --proxy http://127.0.0.1:7897
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://alcoholic.lovecnvpn.com"
USER_AGENT = "crypto-data-gateway-client/1.0"
EXCHANGE = "deribit"
DATA_TYPE = "options_chain"
SYMBOL = "OPTIONS"
# Signed links live for one hour; re-issuing one charges the file size again.
LINK_REUSE_SECONDS = 50 * 60
BHP_SCRIPT = Path(__file__).resolve().parents[1] / "build_hourly_parquet.py"


class QuotaExceeded(RuntimeError):
    pass


class LinkExpired(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def load_bhp():
    spec = importlib.util.spec_from_file_location("build_hourly_parquet", BHP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_hourly_parquet"] = module
    spec.loader.exec_module(module)
    return module


def read_credentials(args) -> tuple[str, str]:
    user = args.user or os.environ.get("CDG_USER", "")
    token = os.environ.get("CDG_TOKEN", "")
    token_file = Path(args.token_file).expanduser()
    if not token and token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
    if not user or not token:
        sys.exit("Missing gateway credentials: set CDG_USER/CDG_TOKEN or use --user/--token-file")
    return user, token


def make_session(proxy: str | None) -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


class Gateway:
    def __init__(self, user: str, token: str, proxy: str | None):
        self.user = user
        self.session = make_session(proxy)
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}

    def _get(self, path: str, params: dict, **kw) -> requests.Response:
        last: Exception | None = None
        for attempt in range(5):
            try:
                return self.session.get(BASE_URL + path, params=params, headers=self.headers, timeout=60, **kw)
            except requests.RequestException as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"gateway request failed: {last}")

    def list_files(self, start: date, end: date) -> dict[date, int]:
        params = {
            "user": self.user, "exchange": EXCHANGE, "type": DATA_TYPE, "symbol": SYMBOL,
            "start": start.isoformat(), "end": end.isoformat(), "limit": 1000,
        }
        out: dict[date, int] = {}
        while True:
            r = self._get("/list", params)
            r.raise_for_status()
            page = r.json()
            for f in page["files"]:
                out[date.fromisoformat(f["date"])] = int(f["size"])
            if page.get("cursor") is None:
                return out
            params["cursor"] = page["cursor"]

    def issue_link(self, d: date) -> str:
        """Request a signed download link. Charges the file size on every call."""
        params = {"user": self.user, "exchange": EXCHANGE, "type": DATA_TYPE,
                  "symbol": SYMBOL, "date": d.isoformat()}
        r = self._get("/file", params, allow_redirects=False)
        if r.status_code == 307:
            return r.headers["Location"]
        if r.status_code == 429:
            raise QuotaExceeded(f"{r.text[:200]} (Retry-After={r.headers.get('Retry-After')})")
        raise RuntimeError(f"GET /file {d} -> HTTP {r.status_code}: {r.text[:200]}")


class Downloader:
    """Ranged multi-connection transfer of one signed link.

    Transfers go through the system curl: its Schannel TLS stack is not hit by
    the per-connection throttling that OpenSSL clients see on direct routes.
    """

    def __init__(self, gateway: Gateway, proxy: str | None, connections: int, chunk_bytes: int):
        self.gateway = gateway
        self.proxy = proxy
        self.connections = connections
        self.chunk_bytes = chunk_bytes
        win_curl = Path(r"C:\Windows\System32\curl.exe")
        self.curl = str(win_curl) if win_curl.exists() else "curl"

    def _fetch_chunk(self, loc: str, seg: Path, start: int, end: int) -> None:
        want = end - start + 1
        route = ["-x", self.proxy] if self.proxy else ["--noproxy", "*"]
        last = ""
        for attempt in range(8):
            cmd = [self.curl, "-s", "-S", "-o", str(seg), "-r", f"{start}-{end}",
                   "--connect-timeout", "20", "--speed-limit", "20000", "--speed-time", "45",
                   "-w", "%{http_code}", *route, loc]
            r = subprocess.run(cmd, capture_output=True, text=True)
            code = r.stdout.strip()
            if code in ("401", "403"):
                seg.unlink(missing_ok=True)
                raise LinkExpired(f"HTTP {code}")
            if r.returncode == 0 and code == "206" and seg.exists() and seg.stat().st_size == want:
                return
            last = f"curl rc={r.returncode} http={code} {r.stderr.strip()[:120]}"
            seg.unlink(missing_ok=True)
            time.sleep(min(30, 2 * (attempt + 1)))
        raise RuntimeError(f"chunk {start}-{end} failed: {last}")

    def download(self, d: date, size: int, final: Path) -> None:
        if final.exists() and final.stat().st_size == size:
            log(f"[dl] {d} already cached")
            return
        final.parent.mkdir(parents=True, exist_ok=True)
        seg_dir = final.with_name(final.name + ".parts")
        state_path = seg_dir / "state.json"
        state: dict = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except ValueError:
                state = {}
        if state.get("size") != size or state.get("chunk") != self.chunk_bytes:
            shutil.rmtree(seg_dir, ignore_errors=True)
            state = {"size": size, "chunk": self.chunk_bytes, "loc": None, "issued_at": 0}
        seg_dir.mkdir(parents=True, exist_ok=True)

        ranges = [(seg_dir / f"{i:05d}.seg", s, min(s + self.chunk_bytes, size) - 1)
                  for i, s in enumerate(range(0, size, self.chunk_bytes))]

        def missing() -> list:
            return [r for r in ranges if not (r[0].exists() and r[0].stat().st_size == r[2] - r[1] + 1)]

        for _ in range(3):
            todo = missing()
            if not todo:
                break
            if not state["loc"] or time.time() - state["issued_at"] > LINK_REUSE_SECONDS:
                state["loc"] = self.gateway.issue_link(d)
                state["issued_at"] = time.time()
                state_path.write_text(json.dumps(state))
                log(f"[dl] {d} link issued ({size / 2**30:.2f} GiB charged)")
            total = sum(e - s + 1 for _, s, e in todo)
            t0 = last_report = time.time()
            finished = 0
            expired = False
            errors: list[Exception] = []
            with ThreadPoolExecutor(self.connections) as pool:
                futs = [pool.submit(self._fetch_chunk, state["loc"], seg, s, e) for seg, s, e in todo]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                        finished += 1
                    except LinkExpired:
                        expired = True
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
                    now = time.time()
                    if now - last_report >= 20:
                        last_report = now
                        log(f"[dl] {d} {finished}/{len(todo)} chunks, "
                            f"{finished * self.chunk_bytes / (now - t0) / 1e6:.1f} MB/s")
            if expired:
                state["issued_at"] = 0
                continue
            if errors:
                raise RuntimeError(f"download {d} failed: {errors[0]}")
            log(f"[dl] {d} done {total / 2**30:.2f} GiB in {time.time() - t0:.0f}s "
                f"({total / (time.time() - t0) / 1e6:.1f} MB/s)")
        if missing():
            raise RuntimeError(f"download {d} incomplete")
        part = final.with_name(final.name + ".part")
        with open(part, "wb") as out:
            for seg, _, _ in ranges:
                with open(seg, "rb") as src:
                    shutil.copyfileobj(src, out, 8 << 20)
        if part.stat().st_size != size:
            raise RuntimeError(f"download {d} size mismatch after assembly")
        os.replace(part, final)
        shutil.rmtree(seg_dir, ignore_errors=True)

def process_day(d: date, keep_cache: bool, chunk_mb: int) -> int:
    cmd = [sys.executable, "-u", str(BHP_SCRIPT), "--_worker-date", d.isoformat(), "--cache-only",
           "--csv-read-mode", "polars-stream", "--polars-stream-chunk-mb", str(chunk_mb)]
    if keep_cache:
        cmd.append("--keep-cache")
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--data-root", default=".", help="directory holding data/ and tmp/ (default: cwd)")
    ap.add_argument("--proxy", default=None, help="route gateway and file transfers through this proxy")
    ap.add_argument("--connections", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--max-ahead", type=int, default=2, help="downloaded days allowed to wait for processing")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--user", default=None)
    ap.add_argument("--token-file", default="~/.cdg_token")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--polars-stream-chunk-mb", type=int, default=64)
    args = ap.parse_args()

    os.chdir(args.data_root)
    bhp = load_bhp()
    user, token = read_credentials(args)
    gateway = Gateway(user, token, args.proxy)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    files = gateway.list_files(start, end)
    processed = bhp.load_checkpoint()
    todo = sorted(d for d in files if d.isoformat() not in processed)
    total = sum(files[d] for d in todo)
    log(f"gateway has {len(files)} day(s) in [{start}..{end}]; {len(todo)} to fetch, {total / 2**30:.1f} GiB")
    if args.dry_run or not todo:
        for d in todo:
            print(f"  {d}  {files[d] / 2**30:6.2f} GiB")
        return 0

    downloader = Downloader(gateway, args.proxy, args.connections, args.chunk_mb << 20)
    ready: queue.Queue = queue.Queue(maxsize=max(1, args.max_ahead))
    stop = threading.Event()
    quota_hit: list[str] = []
    dl_failed: list[date] = []

    def download_loop() -> None:
        for d in todo:
            if stop.is_set():
                break
            path = bhp.cache_path(d)
            while shutil.disk_usage(".").free < files[d] + args.min_free_gb * 2**30 and not stop.is_set():
                log(f"[dl] waiting for free disk space before {d}")
                time.sleep(30)
            try:
                downloader.download(d, files[d], path)
            except QuotaExceeded as e:
                quota_hit.append(f"{d}: {e}")
                log(f"[dl] download allowance exhausted at {d}: {e}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"[dl] {d} FAILED: {e}")
                dl_failed.append(d)
                continue
            ready.put(d)
        ready.put(None)

    threading.Thread(target=download_loop, daemon=True).start()

    ok: list[date] = []
    proc_failed: list[date] = []
    try:
        while True:
            d = ready.get()
            if d is None:
                break
            if args.download_only:
                ok.append(d)
                continue
            t0 = time.time()
            rc = process_day(d, args.keep_cache, args.polars_stream_chunk_mb)
            if rc == 0:
                processed.add(d.isoformat())
                bhp.save_checkpoint(processed)
                ok.append(d)
                log(f"[proc] {d} ok in {time.time() - t0:.0f}s ({len(ok)}/{len(todo)})")
            else:
                proc_failed.append(d)
                log(f"[proc] {d} FAILED (exit {rc})")
    except KeyboardInterrupt:
        stop.set()
        log("interrupted; partial downloads are resumable while their link is valid")
        return 130

    log(f"finished: {len(ok)} ok, {len(dl_failed)} download failure(s), {len(proc_failed)} processing failure(s)")
    for label, items in (("download failed", dl_failed), ("processing failed", proc_failed)):
        if items:
            log(f"  {label}: {', '.join(x.isoformat() for x in items)}")
    if quota_hit:
        log(f"  stopped by gateway allowance: {quota_hit[0]}")
        return 3
    return 1 if (dl_failed or proc_failed) else 0


if __name__ == "__main__":
    sys.exit(main())

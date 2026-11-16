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
import collections
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
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
GIB = 2**30
STAGE_LABELS = {
    "starting": "启动", "prepare": "准备", "read": "解压解析", "validate": "校验",
    "write": "写入 parquet", "cleanup": "清理缓存", "done": "完成", "retry": "重试",
}


class QuotaExceeded(RuntimeError):
    pass


class LinkExpired(RuntimeError):
    pass


def fmt_dur(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def bar(frac: float, width: int = 24) -> str:
    filled = int(max(0.0, min(1.0, frac)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clip(text: str, width: int) -> str:
    """Cut to a terminal width, counting CJK characters as two columns."""
    used = 0
    for i, ch in enumerate(text):
        used += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used > width:
            return text[:i]
    return text


class Status:
    """Shared pipeline state plus a live terminal panel.

    On a terminal the panel is redrawn in place and events scroll above it;
    when output is redirected, events and a periodic one-line summary are
    written instead.
    """

    def __init__(self, total_days: int, total_bytes: int, max_ahead: int, log_path: Path):
        self.lock = threading.Lock()
        self.total_days = total_days
        self.total_bytes = total_bytes
        self.max_ahead = max_ahead
        self.started = time.time()
        self.done_days = 0
        self.fetched_bytes = 0  # bytes of fully downloaded files
        self.dl_date: date | None = None
        self.dl_size = 0
        self.dl_phase = "准备中"
        self.dl_bytes_fn = None
        self.dl_samples: collections.deque = collections.deque(maxlen=12)
        self.waiting: list[date] = []
        self.proc_date: date | None = None
        self.proc_started = 0.0
        self.proc_file: Path | None = None
        self.proc_phase = "等待下载完成"
        self._events: list[str] = []
        self._panel_lines = 0
        self._stop = threading.Event()
        self._log = open(log_path, "a", encoding="utf-8")
        self.tty = sys.stdout.isatty()
        if self.tty and os.name == "nt":
            os.system("")  # turns on ANSI escape handling in legacy consoles
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def event(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        with self.lock:
            self._events.append(line)
            self._log.write(line + "\n")
            self._log.flush()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._render(final=True)
        self._log.close()

    def _loop(self) -> None:
        last_plain = 0.0
        while not self._stop.wait(1.0):
            if self.tty:
                self._render()
            else:
                self._flush_events()
                if time.time() - last_plain >= 30:
                    last_plain = time.time()
                    print(" | ".join(self._panel()[:4]), flush=True)

    def _flush_events(self) -> None:
        with self.lock:
            events, self._events = self._events, []
        for line in events:
            print(line, flush=True)

    def _dl_progress(self) -> tuple[int, float]:
        fn = self.dl_bytes_fn
        if fn is None:
            return 0, 0.0
        try:
            got = fn()
        except OSError:
            got = 0
        now = time.time()
        self.dl_samples.append((now, got))
        t0, b0 = self.dl_samples[0]
        speed = (got - b0) / (now - t0) if now - t0 > 0.5 and got >= b0 else 0.0
        return got, speed

    def _read_proc(self) -> tuple[float, str]:
        if self.proc_file is None:
            return 0.0, ""
        try:
            payload = json.loads(self.proc_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0.0, "启动"
        stage = str(payload.get("stage", ""))
        label = STAGE_LABELS.get(stage, stage)
        if stage == "read" and payload.get("message", "").startswith("hour "):
            label += f"（第 {payload['message'][5:]} 小时）"
        return float(payload.get("percent", 0.0)), label

    def _panel(self) -> list[str]:
        now = time.time()
        got, speed = self._dl_progress()
        with self.lock:
            dl_date, dl_size, dl_phase = self.dl_date, self.dl_size, self.dl_phase
            waiting = list(self.waiting)
            proc_date, proc_started, proc_phase = self.proc_date, self.proc_started, self.proc_phase
            done_days, fetched = self.done_days, self.fetched_bytes
        moved = fetched + (got if dl_date else 0)
        elapsed = now - self.started
        rate = moved / elapsed if elapsed > 5 else 0.0
        eta = fmt_dur((self.total_bytes - moved) / rate) if rate > 0 else "--"
        lines = [
            f"总进度 {bar(done_days / max(1, self.total_days))} {done_days}/{self.total_days} 天"
            f"  已下载 {moved / GIB:.1f}/{self.total_bytes / GIB:.1f} GiB"
            f"  已用 {fmt_dur(elapsed)}  预计剩余 {eta}",
        ]
        if dl_date:
            left = fmt_dur((dl_size - got) / speed) if speed > 0 else "--"
            lines.append(
                f"下载   {bar(got / max(1, dl_size))} {dl_date} {got / max(1, dl_size) * 100:5.1f}%"
                f"  {got / GIB:.2f}/{dl_size / GIB:.2f} GiB  {speed / 1e6:5.1f} MB/s  剩余 {left}  [{dl_phase}]"
            )
        else:
            lines.append(f"下载   [{dl_phase}]")
        queue_txt = ", ".join(d.isoformat() for d in waiting) or "空"
        lines.append(f"待处理队列 {len(waiting)}/{self.max_ahead}: {queue_txt}")
        if proc_date:
            pct, stage = self._read_proc()
            lines.append(
                f"处理   {bar(pct / 100)} {proc_date} {pct:5.1f}%  {stage}  已用 {fmt_dur(now - proc_started)}"
            )
        else:
            lines.append(f"处理   [{proc_phase}]")
        return lines

    def _render(self, final: bool = False) -> None:
        if not self.tty:
            self._flush_events()
            return
        width = max(40, shutil.get_terminal_size((120, 30)).columns - 1)
        out = []
        if self._panel_lines:
            out.append(f"\x1b[{self._panel_lines}F")
        out.append("\x1b[J")
        with self.lock:
            events, self._events = self._events, []
        out.extend(clip(e, width) + "\n" for e in events)
        panel = ["-" * min(width, 100)] + self._panel()
        out.extend(clip(line, width) + "\n" for line in panel)
        self._panel_lines = 0 if final else len(panel)
        sys.stdout.write("".join(out))
        sys.stdout.flush()


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

    def __init__(self, gateway: Gateway, proxy: str | None, connections: int, chunk_bytes: int, status: Status):
        self.gateway = gateway
        self.proxy = proxy
        self.connections = connections
        self.chunk_bytes = chunk_bytes
        self.status = status
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

    def _phase(self, text: str) -> None:
        with self.status.lock:
            self.status.dl_phase = text

    def download(self, d: date, size: int, final: Path) -> None:
        st = self.status
        if final.exists() and final.stat().st_size == size:
            st.event(f"[dl] {d} 已在本地缓存，跳过下载")
            with st.lock:
                st.fetched_bytes += size
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

        def bytes_on_disk() -> int:
            total = 0
            with os.scandir(seg_dir) as it:
                for entry in it:
                    if entry.name.endswith(".seg"):
                        total += entry.stat().st_size
            return total

        with st.lock:
            st.dl_date, st.dl_size, st.dl_bytes_fn = d, size, bytes_on_disk
            st.dl_samples.clear()
        for _ in range(3):
            todo = missing()
            if not todo:
                break
            if not state["loc"] or time.time() - state["issued_at"] > LINK_REUSE_SECONDS:
                self._phase("申请下载链接")
                state["loc"] = self.gateway.issue_link(d)
                state["issued_at"] = time.time()
                state_path.write_text(json.dumps(state))
                st.event(f"[dl] {d} 已申请下载链接（扣除额度 {size / GIB:.2f} GiB）")
            elif len(todo) < len(ranges):
                st.event(f"[dl] {d} 续传：剩余 {len(todo)}/{len(ranges)} 个分块，复用原链接")
            self._phase(f"下载中，{self.connections} 连接")
            total = sum(e - s + 1 for _, s, e in todo)
            t0 = time.time()
            expired = False
            errors: list[Exception] = []
            with ThreadPoolExecutor(self.connections) as pool:
                futs = [pool.submit(self._fetch_chunk, state["loc"], seg, s, e) for seg, s, e in todo]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except LinkExpired:
                        expired = True
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
            if expired:
                st.event(f"[dl] {d} 下载链接已过期，需要重新申请")
                state["issued_at"] = 0
                continue
            if errors:
                raise RuntimeError(f"download {d} failed: {errors[0]}")
            st.event(f"[dl] {d} 下载完成 {total / GIB:.2f} GiB，用时 {time.time() - t0:.0f}s"
                     f"（{total / (time.time() - t0) / 1e6:.1f} MB/s）")
        if missing():
            raise RuntimeError(f"download {d} incomplete")
        self._phase("合并分块")
        part = final.with_name(final.name + ".part")
        with open(part, "wb") as out:
            for seg, _, _ in ranges:
                with open(seg, "rb") as src:
                    shutil.copyfileobj(src, out, 8 << 20)
        if part.stat().st_size != size:
            raise RuntimeError(f"download {d} size mismatch after assembly")
        os.replace(part, final)
        with st.lock:
            st.dl_date, st.dl_bytes_fn = None, None
            st.fetched_bytes += size
        shutil.rmtree(seg_dir, ignore_errors=True)


def process_day(d: date, args, status: Status, worker_log: Path) -> int:
    progress_file = Path(tempfile.gettempdir()) / f"_gateway_fetch_progress_{os.getpid()}_{d.isoformat()}.json"
    cmd = [sys.executable, "-u", str(BHP_SCRIPT), "--_worker-date", d.isoformat(), "--cache-only",
           "--csv-read-mode", "polars-stream", "--polars-stream-chunk-mb", str(args.polars_stream_chunk_mb),
           "--underlyings", args.underlyings, "--_progress-file", str(progress_file)]
    if args.keep_cache:
        cmd.append("--keep-cache")
    with status.lock:
        status.proc_date, status.proc_started, status.proc_file = d, time.time(), progress_file
    try:
        with open(worker_log, "a", encoding="utf-8") as out:
            out.write(f"\n===== {d} =====\n")
            out.flush()
            return subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    finally:
        with status.lock:
            status.proc_date, status.proc_file = None, None
        progress_file.unlink(missing_ok=True)


def tail(path: Path, lines: int = 6) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


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
    ap.add_argument("--underlyings", default="BTC,ETH", help="comma-separated underlyings to keep")
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
    print(f"网关在 [{start}..{end}] 有 {len(files)} 天；待获取 {len(todo)} 天，共 {total / GIB:.1f} GiB", flush=True)
    if args.dry_run or not todo:
        for d in todo:
            print(f"  {d}  {files[d] / GIB:6.2f} GiB")
        return 0

    max_ahead = max(1, args.max_ahead)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    worker_log = log_dir / f"gateway_fetch_{stamp}_worker.log"
    status = Status(len(todo), total, max_ahead, log_dir / f"gateway_fetch_{stamp}.log")
    print(f"事件日志: {log_dir / f'gateway_fetch_{stamp}.log'}   处理明细: {worker_log}", flush=True)

    downloader = Downloader(gateway, args.proxy, args.connections, args.chunk_mb << 20, status)
    ready: queue.Queue = queue.Queue(maxsize=max_ahead)
    stop = threading.Event()
    quota_hit: list[str] = []
    dl_failed: list[date] = []

    def set_dl_phase(text: str, clear: bool = False) -> None:
        with status.lock:
            status.dl_phase = text
            if clear:
                status.dl_date, status.dl_bytes_fn = None, None

    def download_loop() -> None:
        for d in todo:
            if stop.is_set():
                break
            path = bhp.cache_path(d)
            while shutil.disk_usage(".").free < files[d] + args.min_free_gb * GIB and not stop.is_set():
                set_dl_phase(f"等待磁盘空间（{d} 需要 {files[d] / GIB:.1f} GiB）", clear=True)
                time.sleep(10)
            try:
                downloader.download(d, files[d], path)
            except QuotaExceeded as e:
                quota_hit.append(f"{d}: {e}")
                status.event(f"[dl] 网关下载额度已用完，停在 {d}: {e}")
                break
            except Exception as e:  # noqa: BLE001
                status.event(f"[dl] {d} 下载失败: {e}")
                dl_failed.append(d)
                continue
            set_dl_phase(f"{d} 已下完，等待处理队列空位", clear=True)
            while not stop.is_set():
                try:
                    ready.put(d, timeout=0.5)
                    break
                except queue.Full:
                    continue
            with status.lock:
                status.waiting.append(d)
        set_dl_phase("额度用完，已停止下载" if quota_hit else "全部下载完成", clear=True)
        ready.put(None)

    status.start()
    threading.Thread(target=download_loop, daemon=True).start()

    ok: list[date] = []
    proc_failed: list[date] = []
    interrupted = False
    try:
        while True:
            try:
                d = ready.get(timeout=0.5)
            except queue.Empty:
                continue
            if d is None:
                break
            # The downloader appends to `waiting` just after its put() returns.
            while True:
                with status.lock:
                    if d in status.waiting:
                        status.waiting.remove(d)
                        break
                time.sleep(0.05)
            if args.download_only:
                ok.append(d)
                with status.lock:
                    status.done_days += 1
                continue
            t0 = time.time()
            rc = process_day(d, args, status, worker_log)
            if rc == 0:
                processed.add(d.isoformat())
                bhp.save_checkpoint(processed)
                ok.append(d)
                with status.lock:
                    status.done_days += 1
                status.event(f"[proc] {d} 处理完成，用时 {time.time() - t0:.0f}s（{len(ok)}/{len(todo)}）")
            else:
                proc_failed.append(d)
                status.event(f"[proc] {d} 处理失败（exit {rc}），明细见 {worker_log}")
                for line in tail(worker_log):
                    status.event("    " + line)
        with status.lock:
            status.proc_phase = "全部处理完成"
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        status.event("已中断；50 分钟内重新运行可复用下载链接续传，不重复扣额度")

    status.event(f"结束：成功 {len(ok)} 天，下载失败 {len(dl_failed)} 天，处理失败 {len(proc_failed)} 天")
    for label, items in (("下载失败", dl_failed), ("处理失败", proc_failed)):
        if items:
            status.event(f"  {label}: {', '.join(x.isoformat() for x in items)}")
    if quota_hit:
        status.event(f"  因网关额度停止: {quota_hit[0]}")
    status.close()
    if interrupted:
        return 130
    if quota_hit:
        return 3
    return 1 if (dl_failed or proc_failed) else 0


if __name__ == "__main__":
    sys.exit(main())

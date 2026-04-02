"""Test CDD API rate limits to find sustainable request interval."""
import requests
import time
import json
from pathlib import Path

from cdd_secrets import get_cdd_api_token

TOKEN = get_cdd_api_token()
BASE = "https://api.cryptodatadownload.com/v1"
s = requests.Session()
s.headers["Authorization"] = f"Token {TOKEN}"

results = []

def log(msg):
    print(msg, flush=True)
    results.append(msg)

log("Waiting 180s for cooldown...")
time.sleep(180)

# First request
r = s.get(f"{BASE}/data/ohlc/deribit/options/",
          params={"currency": "BTC", "expiry": "17APR24", "limit": 5000}, timeout=60)
log(f"First request: {r.status_code}")

if r.status_code == 200:
    cnt = len(r.json().get("result", []))
    log(f"  -> {cnt} rows")
    
    # Test at 15s intervals
    ok = 0
    for d in range(18, 30):
        time.sleep(15)
        exp = f"{d}APR24"
        r2 = s.get(f"{BASE}/data/ohlc/deribit/options/",
                   params={"currency": "BTC", "expiry": exp, "limit": 5000}, timeout=60)
        if r2.status_code == 200:
            ok += 1
            c = len(r2.json().get("result", []))
            log(f"  [{ok}] {exp}: OK {c} rows")
        else:
            log(f"  After {ok} OK: {r2.status_code} on {exp}")
            # If 429, wait and try 30s interval
            if r2.status_code == 429:
                log("  Switching to 30s interval after 60s cooldown...")
                time.sleep(60)
                for d2 in range(d, min(d+5, 30)):
                    time.sleep(30)
                    exp2 = f"{d2}APR24"
                    r3 = s.get(f"{BASE}/data/ohlc/deribit/options/",
                              params={"currency": "BTC", "expiry": exp2, "limit": 5000}, timeout=60)
                    if r3.status_code == 200:
                        ok += 1
                        c3 = len(r3.json().get("result", []))
                        log(f"  [{ok}] {exp2}: OK {c3} rows (30s)")
                    else:
                        log(f"  30s also fails: {r3.status_code}")
                        break
            break
    
    log(f"\nResult: {ok} sustained at 15s/30s intervals")
else:
    log(f"Still rate limited: {r.status_code}")
    # Show all headers for debugging
    for k, v in r.headers.items():
        log(f"  {k}: {v}")

Path("scripts/rate_result.json").write_text(json.dumps(results, indent=2))
log("Done!")

"""Test CDD API rate limits to find sustainable request interval."""

import requests
import time

TOKEN = "368ca0bd2ccf5620aa35c50f9a11a65943589b49"
BASE = "https://api.cryptodatadownload.com/v1"
s = requests.Session()
s.headers["Authorization"] = f"Token {TOKEN}"

# Wait 3 minutes for full cooldown
print("Waiting 180s for full cooldown...")
time.sleep(180)

# Single request
r = s.get(
    f"{BASE}/data/ohlc/deribit/options/",
    params={"currency": "BTC", "expiry": "17APR24", "limit": 5000},
    timeout=60,
)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    result = r.json().get("result", [])
    print(f"OK! {len(result)} rows")

    # Test at 15s intervals
    success_count = 0
    expiries = [f"{d}APR24" for d in range(18, 30)]
    for exp in expiries:
        time.sleep(15)
        r2 = s.get(
            f"{BASE}/data/ohlc/deribit/options/",
            params={"currency": "BTC", "expiry": exp, "limit": 5000},
            timeout=60,
        )
        if r2.status_code == 200:
            cnt = len(r2.json().get("result", []))
            success_count += 1
            print(f"  [{success_count}] {exp}: OK {cnt} rows")
        elif r2.status_code == 429:
            print(f"  After {success_count} OK: 429 on {exp}!")
            break
        else:
            print(f"  {exp}: {r2.status_code}")
    print(f"\nSustained {success_count} requests at 15s intervals")

elif r.status_code == 429:
    print("Still rate limited! Headers:")
    for k, v in r.headers.items():
        print(f"  {k}: {v}")

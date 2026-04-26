"""Test Tardis.dev free tier availability."""
import requests

# Test: free download for 1st of month (no API key needed)
url = "https://datasets.tardis.dev/v1/deribit/trades/2024/01/01/OPTIONS.csv.gz"
r = requests.get(url, stream=True)
ct = r.headers.get("Content-Type", "?")
cl = r.headers.get("Content-Length", "?")
print(f"1st of month: status={r.status_code}, type={ct}, size={cl}")

if r.status_code == 200:
    chunk = next(r.iter_content(2048))
    print(f"Got {len(chunk)} bytes of data")

# Try options_chain (mark price, IV, greeks)
url_oc = "https://datasets.tardis.dev/v1/deribit/options_chain/2024/01/01/OPTIONS.csv.gz"
r_oc = requests.get(url_oc, stream=True)
ct_oc = r_oc.headers.get("Content-Type", "?")
cl_oc = r_oc.headers.get("Content-Length", "?")
print(f"\noptions_chain 1st of month: status={r_oc.status_code}, type={ct_oc}, size={cl_oc}")

# Try 2nd day (should require API key)
url2 = "https://datasets.tardis.dev/v1/deribit/trades/2024/01/02/OPTIONS.csv.gz"
r2 = requests.get(url2, stream=True)
print(f"\n2nd of month: status={r2.status_code}")
if r2.status_code != 200:
    print(f"Response: {r2.text[:300]}")

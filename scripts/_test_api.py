import requests
s = requests.Session()
s.headers["Authorization"] = "Token 368ca0bd2ccf5620aa35c50f9a11a65943589b49"
r = s.get(
    "https://api.cryptodatadownload.com/v1/data/ohlc/deribit/options/",
    params={"currency": "BTC", "expiry": "1JAN24", "limit": 10},
    timeout=60,
)
print(f"Status: {r.status_code}")
print(f"Headers: {dict(r.headers)}")
if r.status_code == 429:
    print(f"Body: {r.text[:500]}")
elif r.status_code == 200:
    data = r.json()
    rows = data.get("result", [])
    print(f"Rows: {len(rows)}")
    print("API is ready!")
else:
    print(f"Body: {r.text[:300]}")

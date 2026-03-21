"""Test more Deribit API endpoints for historical data."""
import asyncio
import aiohttp
from datetime import datetime

BASE = "https://www.deribit.com/api/v2/public"


async def test():
    async with aiohttp.ClientSession() as s:
        # Test 1: get_last_settlements_by_currency
        url = f"{BASE}/get_last_settlements_by_currency"
        params = {"currency": "BTC", "type": "delivery", "count": 20}
        async with s.get(url, params=params) as r:
            d = await r.json()
            result = d.get("result", {})
            settlements = result.get("settlements", [])
            print("=== Settlements by Currency ===")
            print(f"Count: {len(settlements)}")
            if settlements:
                for st in settlements[:3]:
                    ts = st.get("timestamp", 0)
                    dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
                    print(f"  {dt} | {st.get('instrument_name', 'N/A')} | "
                          f"index={st.get('index_price', 'N/A')} | "
                          f"type={st.get('type', 'N/A')}")

        # Test 2: get_funding_chart_data for BTC - shows index history
        url2 = f"{BASE}/get_funding_chart_data"
        params2 = {"instrument_name": "BTC-PERPETUAL", "length": "8h"}
        async with s.get(url2, params=params2) as r:
            d = await r.json()
            print(f"\n=== Funding Chart Data ===")
            print("status:", r.status)
            result = d.get("result", {})
            if isinstance(result, dict):
                print("Keys:", list(result.keys()))

        # Test 3: get_index_price 
        url3 = f"{BASE}/get_index_price"
        params3 = {"index_name": "btc_usd"}
        async with s.get(url3, params=params3) as r:
            d = await r.json()
            print(f"\n=== Index Price ===")
            print("status:", r.status)
            print("result:", d.get("result", d.get("error")))

        # Test 4: Try tradingview chart data with deribit_btc index
        url4 = f"{BASE}/get_tradingview_chart_data"
        params4 = {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": 1735689600000,  # 2025-01-01
            "end_timestamp": 1740787200000,    # 2025-03-01
            "resolution": "1D",
        }
        async with s.get(url4, params=params4) as r:
            d = await r.json()
            print(f"\n=== BTC-PERP Daily OHLCV (Jan-Mar 2025) ===")
            print("status:", r.status)
            if "result" in d:
                ticks = d["result"].get("ticks", [])
                opens = d["result"].get("open", [])
                print(f"Got {len(ticks)} bars")
                if ticks:
                    t0 = datetime.utcfromtimestamp(ticks[0]/1000).strftime("%Y-%m-%d")
                    t1 = datetime.utcfromtimestamp(ticks[-1]/1000).strftime("%Y-%m-%d")
                    print(f"Range: {t0} -> {t1}")
                    print(f"Open[0]={opens[0]}, Open[-1]={opens[-1]}")
            else:
                print("Error:", d.get("error"))

        # Test 5: Try getting mark price history for a known option
        url5 = f"{BASE}/get_tradingview_chart_data"
        params5 = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "start_timestamp": 1735689600000,  # 2025-01-01
            "end_timestamp": 1737504000000,    # 2025-01-22
            "resolution": "1D",
        }
        async with s.get(url5, params=params5) as r:
            d = await r.json()
            print(f"\n=== Option OHLCV (BTC-28MAR25-100000-C) ===")
            print("status:", r.status)
            if "result" in d:
                ticks = d["result"].get("ticks", [])
                opens = d["result"].get("open", [])
                print(f"Got {len(ticks)} bars")
                if ticks and opens:
                    print(f"Open prices (BTC): {opens[:5]}")
            else:
                print("Error:", d.get("error"))

        # Test 6: Try to get mark_price_history
        url6 = f"{BASE}/get_mark_price_history"
        params6 = {
            "instrument_name": "BTC-28MAR25-100000-C",
            "start_timestamp": 1735689600000,
            "end_timestamp": 1737504000000,
        }
        async with s.get(url6, params=params6) as r:
            d = await r.json()
            print(f"\n=== Mark Price History (BTC-28MAR25-100000-C) ===")
            print("status:", r.status)
            if "result" in d:
                data = d["result"]
                if isinstance(data, list):
                    print(f"Got {len(data)} records")
                    if data:
                        print(f"Sample: {data[0]}")
                else:
                    print(f"Result: {data}")
            else:
                print("Error:", d.get("error"))

asyncio.run(test())

"""Quick test of Deribit API calls."""
import asyncio
import aiohttp
from datetime import datetime


async def test():
    async with aiohttp.ClientSession() as s:
        # Test 1: underlying OHLCV with BTC-PERPETUAL
        url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        params = {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": 1735689600000,  # 2025-01-01
            "end_timestamp": 1735776000000,    # 2025-01-02
            "resolution": "60",
        }
        async with s.get(url, params=params) as r:
            d = await r.json()
            print("=== BTC-PERPETUAL OHLCV ===")
            print("status:", r.status)
            if "result" in d:
                ticks = d["result"].get("ticks", [])
                print(f"Got {len(ticks)} bars")
                if ticks:
                    print("First tick:", ticks[0], " Last:", ticks[-1])
            else:
                print("Error:", d.get("error"))

        # Test 2: expired instruments
        url2 = "https://www.deribit.com/api/v2/public/get_instruments"
        params2 = {"currency": "BTC", "kind": "option", "expired": "true"}
        async with s.get(url2, params=params2) as r:
            d = await r.json()
            instruments = d.get("result", [])
            print(f"\n=== Expired Instruments ===")
            print(f"Total: {len(instruments)}")
            if instruments:
                exps = set()
                for i in instruments:
                    ts = i.get("expiration_timestamp", 0)
                    dt = datetime.utcfromtimestamp(ts / 1000)
                    exps.add(dt.strftime("%Y-%m-%d"))
                sorted_exp = sorted(exps)
                print(f"Expiry range: {sorted_exp[0]} -> {sorted_exp[-1]}")
                print(f"Unique expiry dates: {len(sorted_exp)}")
                names = [i["instrument_name"] for i in instruments[:5]]
                print(f"Sample names: {names}")

        # Test 3: delivery prices
        url3 = "https://www.deribit.com/api/v2/public/get_delivery_prices"
        params3 = {"currency": "BTC", "count": 10}
        async with s.get(url3, params=params3) as r:
            d = await r.json()
            print(f"\n=== Delivery Prices ===")
            print("status:", r.status)
            result = d.get("result", {})
            if isinstance(result, dict):
                data = result.get("data", [])
                print(f"Records: {len(data)}")
                if data:
                    print("Sample:", data[0])
            else:
                print("Result:", result)


asyncio.run(test())

"""Test settlement pagination and instrument name discovery."""
import asyncio
import aiohttp
from datetime import datetime

BASE = "https://www.deribit.com/api/v2/public"


async def test():
    async with aiohttp.ClientSession() as s:
        # Test: get_last_settlements_by_currency with continuation
        url = f"{BASE}/get_last_settlements_by_currency"
        params = {
            "currency": "BTC",
            "type": "delivery",
            "count": 1000,
        }
        async with s.get(url, params=params) as r:
            d = await r.json()
            result = d.get("result", {})
            settlements = result.get("settlements", [])
            cont = result.get("continuation")
            print(f"=== Settlements Page 1 ===")
            print(f"Count: {len(settlements)}, continuation: {cont}")
            
            if settlements:
                # Group by timestamp (each expiry date has multiple instruments)
                dates = set()
                for st in settlements:
                    ts = st.get("timestamp", 0)
                    dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                    dates.add(dt)
                sorted_dates = sorted(dates)
                print(f"Date range: {sorted_dates[0]} -> {sorted_dates[-1]}")
                print(f"Unique dates: {len(sorted_dates)}")
                print(f"Dates: {sorted_dates}")

        # Page 2 with continuation
        if cont:
            params["continuation"] = cont
            async with s.get(url, params=params) as r:
                d = await r.json()
                result = d.get("result", {})
                settlements2 = result.get("settlements", [])
                cont2 = result.get("continuation")
                print(f"\n=== Settlements Page 2 ===")
                print(f"Count: {len(settlements2)}, continuation: {cont2}")
                if settlements2:
                    dates2 = set()
                    for st in settlements2:
                        ts = st.get("timestamp", 0)
                        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                        dates2.add(dt)
                    sorted_dates2 = sorted(dates2)
                    print(f"Date range: {sorted_dates2[0]} -> {sorted_dates2[-1]}")
                    print(f"Unique dates: {len(sorted_dates2)}")
                    print(f"Dates: {sorted_dates2}")

            # Page 3
            if cont2:
                params["continuation"] = cont2
                async with s.get(url, params=params) as r:
                    d = await r.json()
                    result = d.get("result", {})
                    settlements3 = result.get("settlements", [])
                    cont3 = result.get("continuation")
                    print(f"\n=== Settlements Page 3 ===")
                    print(f"Count: {len(settlements3)}, continuation: {cont3}")
                    if settlements3:
                        dates3 = set()
                        instruments3 = set()
                        for st in settlements3:
                            ts = st.get("timestamp", 0)
                            dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                            dates3.add(dt)
                            instruments3.add(st.get("instrument_name", ""))
                        sorted_dates3 = sorted(dates3)
                        print(f"Date range: {sorted_dates3[0]} -> {sorted_dates3[-1]}")
                        print(f"Unique dates: {len(sorted_dates3)}")
                        print(f"Unique instruments: {len(instruments3)}")
                        print(f"Sample instruments: {sorted(instruments3)[:10]}")

asyncio.run(test())

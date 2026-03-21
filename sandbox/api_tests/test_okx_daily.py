"""Test OKX API for daily options - active vs history, and mark-price vs regular candles."""
import asyncio
import aiohttp

async def check():
    async with aiohttp.ClientSession() as s:
        print("=== 1. History mark-price candles (expired) ===")
        expired = [
            "ETH-USD-260314-2000-C",  # yesterday
            "ETH-USD-260313-2000-C",  # 2 days ago
        ]
        for inst in expired:
            url = "https://www.okx.com/api/v5/market/history-mark-price-candles"
            params = {"instId": inst, "bar": "1H", "limit": "100"}
            async with s.get(url, params=params) as r:
                d = await r.json()
                bars = d.get("data", [])
                msg = d.get("msg", "")
                status = f"{len(bars)} bars" if not msg else msg[:60]
                print(f"  {inst}: {status}")
                if bars:
                    print(f"    oldest={bars[-1][0]}, newest={bars[0][0]}")

        print("\n=== 2. Active mark-price candles (listed now) ===")
        active = [
            "ETH-USD-260315-2000-C",  # today
            "ETH-USD-260316-2000-C",  # tomorrow
        ]
        for inst in active:
            url = "https://www.okx.com/api/v5/market/mark-price-candles"
            params = {"instId": inst, "bar": "1H", "limit": "100"}
            async with s.get(url, params=params) as r:
                d = await r.json()
                bars = d.get("data", [])
                msg = d.get("msg", "")
                status = f"{len(bars)} bars" if not msg else msg[:60]
                print(f"  {inst}: {status}")
                if bars:
                    from datetime import datetime, timezone
                    oldest = datetime.fromtimestamp(int(bars[-1][0])/1000, tz=timezone.utc)
                    newest = datetime.fromtimestamp(int(bars[0][0])/1000, tz=timezone.utc)
                    print(f"    range: {oldest} -> {newest}")

        print("\n=== 3. History regular candles (expired daily) ===")
        for inst in expired:
            url = "https://www.okx.com/api/v5/market/history-candles"
            params = {"instId": inst, "bar": "1H", "limit": "100"}
            async with s.get(url, params=params) as r:
                d = await r.json()
                bars = d.get("data", [])
                msg = d.get("msg", "")
                status = f"{len(bars)} bars" if not msg else msg[:60]
                print(f"  {inst}: {status}")

asyncio.run(check())

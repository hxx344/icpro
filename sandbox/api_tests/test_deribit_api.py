"""Test Deribit API for expired ETH option OHLCV data."""
import asyncio
import aiohttp

BASE = "https://www.deribit.com/api/v2"

async def test():
    async with aiohttp.ClientSession() as s:
        # 1. Test tradingview chart data for expired options
        print("=== Deribit get_tradingview_chart_data (expired ETH options) ===")
        tests = [
            # Recent daily-expiry options
            ("ETH-14MAR26-2000-C", "14MAR26 (yesterday)"),
            ("ETH-13MAR26-2000-C", "13MAR26 (2 days ago)"),
            ("ETH-10MAR26-2000-C", "10MAR26 (5 days ago)"),
            ("ETH-1MAR26-2200-C",  "1MAR26 (2 weeks ago)"),
            ("ETH-15FEB26-2800-C", "15FEB26 (1 month ago)"),
            ("ETH-15JAN26-3500-C", "15JAN26 (2 months ago)"),
            ("ETH-27DEC25-3500-C", "27DEC25 (quarterly, ~3 months ago)"),
        ]
        
        for inst, label in tests:
            url = f"{BASE}/public/get_tradingview_chart_data"
            params = {
                "instrument_name": inst,
                "resolution": "60",
                "start_timestamp": 1734048000000,  # 2024-12-13
                "end_timestamp": 1741910400000,    # 2026-03-14
            }
            async with s.get(url, params=params) as r:
                d = await r.json()
                result = d.get("result", {})
                if isinstance(result, dict) and result.get("status") == "ok":
                    ticks = result.get("ticks", [])
                    print(f"  {inst} ({label}): {len(ticks)} bars OK")
                else:
                    error = d.get("error", {})
                    msg = error.get("message", str(d.get("result", "unknown")))
                    print(f"  {inst} ({label}): ERROR - {msg}")

        # 2. Check what active ETH instruments exist
        print("\n=== Deribit active ETH instruments ===")
        url = f"{BASE}/public/get_instruments"
        params = {"currency": "ETH", "kind": "option", "expired": "false"}
        async with s.get(url, params=params) as r:
            d = await r.json()
            insts = d.get("result", [])
            expiries = set()
            for i in insts:
                name = i.get("instrument_name", "")
                parts = name.split("-")
                if len(parts) >= 2:
                    expiries.add(parts[1])
            print(f"  Active instruments: {len(insts)}")
            daily = sorted([e for e in expiries if len(e) <= 7])
            print(f"  Expiry dates ({len(expiries)} total): {sorted(expiries)}")

        # 3. Test expired=true
        print("\n=== Deribit get_instruments(expired=true) ===")
        params2 = {"currency": "ETH", "kind": "option", "expired": "true"}
        async with s.get(url, params=params2) as r:
            d = await r.json()
            insts2 = d.get("result", [])
            print(f"  Expired instruments returned: {len(insts2)}")
            if insts2:
                names = [i.get("instrument_name", "") for i in insts2[:5]]
                print(f"  Sample: {names}")

asyncio.run(test())

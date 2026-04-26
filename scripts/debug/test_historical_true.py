"""Test Deribit API with historical=true parameter for expired instruments.

Per community info, adding historical=true to get_last_trades_by_instrument_and_time
may enable retrieving trades for expired options.
"""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
import json

BASE = "https://www.deribit.com/api/v2"


async def call_api(session, endpoint, params, label):
    url = f"{BASE}{endpoint}"
    async with session.get(url, params=params) as r:
        data = await r.json()
        result = data.get("result", {})
        error = data.get("error")
        
        if error:
            print(f"\n[ERROR] {label}")
            print(f"  Endpoint: {endpoint}")
            print(f"  Params: {params}")
            print(f"  Error: {error}")
            return None
        
        if isinstance(result, dict):
            trades = result.get("trades", [])
            has_more = result.get("has_more", False)
        elif isinstance(result, list):
            trades = result
            has_more = False
        else:
            trades = []
            has_more = False
        
        print(f"\n[{'OK' if trades else 'EMPTY'}] {label}")
        print(f"  Endpoint: {endpoint}")
        print(f"  Trades: {len(trades)}, has_more: {has_more}")
        if trades:
            for t in trades[:3]:
                inst = t.get("instrument_name", "?")
                price = t.get("price", 0)
                amount = t.get("amount", 0)
                ts = t.get("timestamp", 0)
                dt = datetime.utcfromtimestamp(ts / 1000)
                print(f"    {inst}  price={price}  amount={amount}  time={dt}")
        return trades


async def main():
    async with aiohttp.ClientSession() as s:
        print("=" * 80)
        print("Testing Deribit historical=true parameter")
        print("=" * 80)

        # ============================================================
        # Step 1: Get expired instruments to find valid names
        # ============================================================
        print("\n--- Step 1: Get expired instruments ---")
        r = await s.get(f"{BASE}/public/get_instruments",
                        params={"currency": "BTC", "kind": "option", "expired": "true"})
        data = await r.json()
        expired = data.get("result", [])
        print(f"Expired instruments returned: {len(expired)}")
        
        if expired:
            # Show some examples
            for inst in expired[:5]:
                exp_ts = inst.get("expiration_timestamp", 0)
                created_ts = inst.get("creation_timestamp", 0)
                print(f"  {inst['instrument_name']}  "
                      f"created={datetime.utcfromtimestamp(created_ts/1000)}  "
                      f"expired={datetime.utcfromtimestamp(exp_ts/1000)}")

        # ============================================================
        # Step 2: Test get_last_trades_by_instrument_and_time 
        #         WITH historical=true for expired instruments
        # ============================================================
        print("\n" + "=" * 80)
        print("Step 2: Test with historical=true on EXPIRED instruments")
        print("=" * 80)

        # Test cases: recently expired + older expired
        test_instruments = [
            # Recently expired (within last few days)
            ("BTC-22MAR26-85000-C", "2026-03-21", "2026-03-22", "Recently expired call"),
            ("BTC-22MAR26-85000-P", "2026-03-21", "2026-03-22", "Recently expired put"),
            # Expired 1 week ago
            ("BTC-14MAR26-85000-C", "2026-03-13", "2026-03-14", "Expired ~1 week ago"),
            # Expired 1 month ago  
            ("BTC-28FEB26-85000-C", "2026-02-27", "2026-02-28", "Expired ~1 month ago"),
            # Expired 3 months ago
            ("BTC-27DEC25-100000-C", "2025-12-26", "2025-12-27", "Expired ~3 months ago"),
            # Expired 1 year ago
            ("BTC-29MAR25-85000-C", "2025-03-28", "2025-03-29", "Expired ~1 year ago"),
            # Expired 2 years ago
            ("BTC-31MAR24-70000-C", "2024-03-30", "2024-03-31", "Expired ~2 years ago"),
            # Expired 3 years ago
            ("BTC-31MAR23-28000-C", "2023-03-30", "2023-03-31", "Expired ~3 years ago"),
        ]

        for inst, start_date, end_date, desc in test_instruments:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            # WITHOUT historical=true
            await call_api(s,
                "/public/get_last_trades_by_instrument_and_time",
                {"instrument_name": inst,
                 "start_timestamp": start_ms,
                 "end_timestamp": end_ms,
                 "count": 5, "sorting": "desc"},
                f"{desc}: {inst} (NO historical param)")

            # WITH historical=true
            await call_api(s,
                "/public/get_last_trades_by_instrument_and_time",
                {"instrument_name": inst,
                 "start_timestamp": start_ms,
                 "end_timestamp": end_ms,
                 "count": 5, "sorting": "desc",
                 "include_old": "true"},
                f"{desc}: {inst} (include_old=true)")

            # Also try with historical=true as a param name variant
            await call_api(s,
                "/public/get_last_trades_by_instrument_and_time",
                {"instrument_name": inst,
                 "start_timestamp": start_ms,
                 "end_timestamp": end_ms,
                 "count": 5, "sorting": "desc",
                 "historical": "true"},
                f"{desc}: {inst} (historical=true)")

        # ============================================================
        # Step 3: Test get_last_trades_by_currency_and_time with historical
        # ============================================================
        print("\n" + "=" * 80)
        print("Step 3: Test by_currency_and_time with historical=true")
        print("=" * 80)

        # Try 1 year ago 
        dt_1y = datetime(2025, 3, 22, 8, 0, tzinfo=timezone.utc)
        start_ms = int((dt_1y - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((dt_1y + timedelta(minutes=30)).timestamp() * 1000)

        await call_api(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 10, "sorting": "desc"},
            "BTC options 1 year ago (no historical)")

        await call_api(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 10, "sorting": "desc",
             "include_old": "true"},
            "BTC options 1 year ago (include_old=true)")

        await call_api(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 10, "sorting": "desc",
             "historical": "true"},
            "BTC options 1 year ago (historical=true)")

        # ============================================================
        # Step 4: Test get_tradingview_chart_data for expired instruments
        # ============================================================
        print("\n" + "=" * 80)
        print("Step 4: Test get_tradingview_chart_data for expired instruments")
        print("=" * 80)
        
        for inst, start_date, end_date, desc in test_instruments[:4]:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            url = f"{BASE}/public/get_tradingview_chart_data"
            params = {
                "instrument_name": inst,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": "60",
            }
            async with s.get(url, params=params) as r:
                data = await r.json()
                result = data.get("result", {})
                error = data.get("error")
                if error:
                    print(f"\n[ERROR] OHLCV {desc}: {inst}")
                    print(f"  Error: {error}")
                else:
                    ticks = result.get("ticks", [])
                    print(f"\n[{'OK' if ticks else 'EMPTY'}] OHLCV {desc}: {inst}")
                    print(f"  Data points: {len(ticks)}")
                    if ticks:
                        print(f"  First: {datetime.utcfromtimestamp(ticks[0]/1000)}")
                        print(f"  Last: {datetime.utcfromtimestamp(ticks[-1]/1000)}")

        # ============================================================
        # Step 5: Test get_candles for expired instruments
        # ============================================================
        print("\n" + "=" * 80)
        print("Step 5: Test get_candles (alternative OHLCV endpoint)")
        print("=" * 80)

        # get_candles might be the newer endpoint
        for inst, start_date, end_date, desc in test_instruments[:4]:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            url = f"{BASE}/public/get_candles"
            params = {
                "instrument_name": inst,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": "60",
            }
            async with s.get(url, params=params) as r:
                data = await r.json()
                result = data.get("result", {})
                error = data.get("error")
                if error:
                    print(f"\n[ERROR] Candles {desc}: {inst}")
                    print(f"  Error: {error}")
                else:
                    if isinstance(result, list):
                        print(f"\n[{'OK' if result else 'EMPTY'}] Candles {desc}: {inst}")
                        print(f"  Data points: {len(result)}")
                        if result:
                            print(f"  Sample: {result[0]}")
                    elif isinstance(result, dict):
                        ticks = result.get("ticks", [])
                        print(f"\n[{'OK' if ticks else 'EMPTY'}] Candles {desc}: {inst}")
                        print(f"  Data points: {len(ticks)}")

        # ============================================================
        # Step 6: Also check if there's a "get_book_summary_by_instrument" 
        #         that might work for expired
        # ============================================================
        print("\n" + "=" * 80)
        print("Step 6: Quick check - recent active instrument with historical=true")
        print("=" * 80)

        # As a control test: use an ACTIVE instrument 
        r = await s.get(f"{BASE}/public/get_instruments",
                        params={"currency": "BTC", "kind": "option", "expired": "false"})
        data = await r.json()
        active = data.get("result", [])
        if active:
            name = active[0]["instrument_name"]
            now = datetime.now(timezone.utc)
            start_ms = int((now - timedelta(hours=4)).timestamp() * 1000)
            end_ms = int(now.timestamp() * 1000)

            trades = await call_api(s,
                "/public/get_last_trades_by_instrument_and_time",
                {"instrument_name": name,
                 "start_timestamp": start_ms,
                 "end_timestamp": end_ms,
                 "count": 5, "sorting": "desc",
                 "include_old": "true"},
                f"ACTIVE {name} (include_old=true, control test)")

        print("\n" + "=" * 80)
        print("DONE - All tests complete")
        print("=" * 80)


asyncio.run(main())

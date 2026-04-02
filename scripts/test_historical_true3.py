"""Test with verified instrument names and by_currency endpoint.
Focus: does historical=true / include_old=true actually unlock older data?
"""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
import json

BASE = "https://www.deribit.com/api/v2"


async def call(session, endpoint, params):
    url = f"{BASE}{endpoint}"
    async with session.get(url, params=params) as r:
        return await r.json()


async def main():
    async with aiohttp.ClientSession() as s:

        # ============================================================
        # Part A: Use by_currency_and_time (NO instrument name needed)
        #         This is the cleanest test - if data exists, it will show up
        # ============================================================
        print("=" * 80)
        print("Part A: get_last_trades_by_currency_and_time")
        print("  Does NOT require instrument name - tests pure data availability")
        print("=" * 80)

        test_dates = [
            ("2026-03-22 07:00", "2026-03-22 08:00", "Yesterday 7-8AM (just expired)"),
            ("2026-03-21 07:00", "2026-03-21 08:00", "2 days ago 7-8AM"),
            ("2026-03-15 07:00", "2026-03-15 08:00", "1 week ago"),
            ("2026-03-08 07:00", "2026-03-08 08:00", "2 weeks ago"),
            ("2026-03-01 07:00", "2026-03-01 08:00", "3 weeks ago"),
            ("2026-02-15 07:00", "2026-02-15 08:00", "1 month ago"),
            ("2026-01-15 07:00", "2026-01-15 08:00", "2 months ago"),
            ("2025-12-15 07:00", "2025-12-15 08:00", "3 months ago"),
            ("2025-09-15 07:00", "2025-09-15 08:00", "6 months ago"),
            ("2025-06-15 07:00", "2025-06-15 08:00", "9 months ago"),
            ("2025-03-15 07:00", "2025-03-15 08:00", "1 year ago"),
            ("2024-09-15 07:00", "2024-09-15 08:00", "1.5 years ago"),
            ("2024-03-15 07:00", "2024-03-15 08:00", "2 years ago"),
            ("2023-06-15 07:00", "2023-06-15 08:00", "2.5 years ago"),
            ("2023-01-15 07:00", "2023-01-15 08:00", "3 years ago"),
        ]

        for start_str, end_str, desc in test_dates:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            results = {}
            for label, extra in [
                ("plain", {}),
                ("include_old", {"include_old": "true"}),
                ("historical", {"historical": "true"}),
            ]:
                params = {
                    "currency": "BTC", "kind": "option",
                    "start_timestamp": start_ms, "end_timestamp": end_ms,
                    "count": 5, "sorting": "desc",
                    **extra
                }
                data = await call(s, "/public/get_last_trades_by_currency_and_time", params)
                error = data.get("error")
                result = data.get("result", {})
                trades = result.get("trades", []) if isinstance(result, dict) else []
                has_more = result.get("has_more", False) if isinstance(result, dict) else False
                results[label] = (len(trades), has_more, error)
                
                # If any variant has trades, show them
                if trades and label == "plain":
                    sample = trades[0]
                    inst = sample.get("instrument_name", "?")
                    price = sample.get("price", 0)

            p, i, h = results["plain"], results["include_old"], results["historical"]
            status = f"plain={p[0]}({'+' if p[1] else '-'})  include_old={i[0]}({'+' if i[1] else '-'})  historical={h[0]}({'+' if h[1] else '-'})"
            
            if p[2]:  # error
                status = f"ERROR: {p[2]}"
            
            emoji = "OK" if p[0] > 0 else "--"
            print(f"  [{emoji}] {desc:25s}  {status}")
            
            # Show sample instrument if trades found
            if p[0] > 0:
                data = await call(s, "/public/get_last_trades_by_currency_and_time", {
                    "currency": "BTC", "kind": "option",
                    "start_timestamp": start_ms, "end_timestamp": end_ms,
                    "count": 3, "sorting": "desc",
                })
                trades = data.get("result", {}).get("trades", [])
                for t in trades[:2]:
                    print(f"       {t['instrument_name']}  price={t['price']}  "
                          f"amount={t['amount']}  "
                          f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # ============================================================
        # Part B: Get verified instrument names from settlements data
        # ============================================================
        print("\n" + "=" * 80)
        print("Part B: Use our local settlement data to find REAL instrument names")
        print("=" * 80)

        try:
            import pandas as pd
            df = pd.read_parquet("data/settlements/btc_settlements.parquet")
            print(f"Settlement records: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            
            # Get unique instrument names at various dates
            if "instrument_name" in df.columns and "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
                
                sample_dates = [
                    datetime(2025, 3, 28).date(),   # ~1 year ago
                    datetime(2024, 12, 27).date(),   # ~1.2 years ago
                    datetime(2024, 6, 28).date(),    # ~1.7 years ago
                    datetime(2024, 3, 29).date(),    # ~2 years ago
                    datetime(2023, 12, 29).date(),   # ~2.2 years ago
                    datetime(2023, 6, 30).date(),    # ~2.7 years ago
                    datetime(2023, 3, 31).date(),    # ~3 years ago
                ]
                
                for d in sample_dates:
                    matches = df[df["date"] == d]
                    if len(matches) > 0:
                        names = matches["instrument_name"].unique()
                        print(f"\n  Date {d}: {len(names)} settled instruments")
                        # Show a few ATM-ish ones
                        for n in names[:5]:
                            row = matches[matches["instrument_name"] == n].iloc[0]
                            print(f"    {n}  settlement={row.get('settlement_price', '?')}")
                        
                        # Test trades for these REAL instruments
                        test_name = names[0]
                        # Query 24h window around settlement
                        d_dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
                        start_ms = int((d_dt - timedelta(hours=24)).timestamp() * 1000)
                        end_ms = int(d_dt.replace(hour=8).timestamp() * 1000)
                        
                        for label, extra in [
                            ("plain", {}),
                            ("include_old", {"include_old": "true"}),
                            ("historical", {"historical": "true"}),
                        ]:
                            data = await call(s, 
                                "/public/get_last_trades_by_instrument_and_time",
                                {"instrument_name": test_name,
                                 "start_timestamp": start_ms,
                                 "end_timestamp": end_ms,
                                 "count": 5, "sorting": "desc",
                                 **extra})
                            error = data.get("error")
                            if error:
                                print(f"    -> [{label}] ERROR: {error}")
                                break  # no point testing other variants
                            else:
                                trades = data.get("result", {}).get("trades", [])
                                has_more = data.get("result", {}).get("has_more", False)
                                print(f"    -> [{label}] trades={len(trades)}, has_more={has_more}")
                                if trades:
                                    t = trades[0]
                                    print(f"       price={t['price']}  amount={t['amount']}  "
                                          f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")
                    else:
                        # Try nearby dates
                        nearby = df[(df["date"] >= d - timedelta(days=3)) & 
                                   (df["date"] <= d + timedelta(days=3))]
                        if len(nearby) > 0:
                            actual_date = nearby["date"].iloc[0]
                            print(f"\n  Date {d}: no exact match, nearest={actual_date} ({len(nearby)} records)")
                        else:
                            print(f"\n  Date {d}: no settlement data nearby")
            else:
                print(f"  Available columns: {list(df.columns)}")
                print(f"  Sample row: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"  Error reading settlements: {e}")

        # ============================================================
        # Part C: Also check get_tradingview_chart_data with real names
        # ============================================================
        print("\n" + "=" * 80)
        print("Part C: OHLCV for real expired instruments (from part B names)")
        print("=" * 80)
        
        # Use the 62 recently expired instruments we know about
        data = await call(s, "/public/get_instruments", 
                         {"currency": "BTC", "kind": "option", "expired": "true"})
        expired = data.get("result", [])
        
        # Find ones that actually had trades (higher strikes = more liquid)
        liquid_tests = [i for i in expired if i["strike"] >= 80000 and "C" in i["instrument_name"]]
        liquid_tests.sort(key=lambda x: x["strike"])
        
        print(f"Testing OHLCV for {len(liquid_tests)} recently expired instruments:")
        for inst in liquid_tests[:5]:
            name = inst["instrument_name"]
            created_ts = inst["creation_timestamp"]
            exp_ts = inst["expiration_timestamp"]
            
            data = await call(s, "/public/get_tradingview_chart_data", {
                "instrument_name": name,
                "start_timestamp": created_ts,
                "end_timestamp": exp_ts,
                "resolution": "60",
            })
            error = data.get("error")
            result = data.get("result", {})
            if error:
                print(f"  {name}: ERROR {error.get('message', error)}")
            else:
                ticks = result.get("ticks", [])
                if ticks:
                    close = result.get("close", [])
                    vol = result.get("volume", [])
                    print(f"  {name}: {len(ticks)} bars, "
                          f"close[0]={close[0] if close else '?'}, "
                          f"vol_sum={sum(vol) if vol else '?'}")
                else:
                    print(f"  {name}: 0 bars")

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)


asyncio.run(main())

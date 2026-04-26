"""Test with REAL instrument names from get_instruments API."""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta

BASE = "https://www.deribit.com/api/v2"


async def get_trades(session, inst, start_ms, end_ms, extra_params=None):
    params = {
        "instrument_name": inst,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": 10, "sorting": "desc",
    }
    if extra_params:
        params.update(extra_params)
    url = f"{BASE}/public/get_last_trades_by_instrument_and_time"
    async with session.get(url, params=params) as r:
        data = await r.json()
        error = data.get("error")
        if error:
            return None, error
        result = data.get("result", {})
        trades = result.get("trades", [])
        has_more = result.get("has_more", False)
        return trades, has_more


async def main():
    async with aiohttp.ClientSession() as s:
        # ============================================================
        # 1. Get REAL recently expired instrument names
        # ============================================================
        print("=" * 80)
        print("1. Get real expired instruments from API")
        print("=" * 80)
        r = await s.get(f"{BASE}/public/get_instruments",
                        params={"currency": "BTC", "kind": "option", "expired": "true"})
        data = await r.json()
        expired = data.get("result", [])
        print(f"Expired count: {len(expired)}")
        
        # Pick a few real ones
        test_expired = expired[:6]
        for inst in test_expired:
            name = inst["instrument_name"]
            exp_ts = inst["expiration_timestamp"]
            created_ts = inst["creation_timestamp"]
            print(f"  {name}  created={datetime.utcfromtimestamp(created_ts/1000)}  "
                  f"expired={datetime.utcfromtimestamp(exp_ts/1000)}")

            # Query 24h before expiry
            start_ms = exp_ts - 48 * 3600 * 1000
            end_ms = exp_ts

            for label, extra in [
                ("no extra", {}),
                ("include_old=true", {"include_old": "true"}),
                ("historical=true", {"historical": "true"}),
            ]:
                trades, info = await get_trades(s, name, start_ms, end_ms, extra)
                if trades is None:
                    print(f"    [{label}] ERROR: {info}")
                else:
                    print(f"    [{label}] trades={len(trades)}, has_more={info}")
                    if trades:
                        for t in trades[:2]:
                            print(f"      price={t['price']}  amount={t['amount']}  "
                                  f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # ============================================================
        # 2. Try the by_currency endpoint for the same time window
        # ============================================================
        print("\n" + "=" * 80)
        print("2. by_currency_and_time around last expiry (22MAR26 08:00 UTC)")
        print("=" * 80)
        
        # 22MAR26 was yesterday, try 8AM UTC on March 21
        dt = datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000)
        end_ms = int((dt + timedelta(hours=1)).timestamp() * 1000)

        for label, extra in [
            ("no extra", {}),
            ("include_old=true", {"include_old": "true"}),
            ("historical=true", {"historical": "true"}),
        ]:
            params = {
                "currency": "BTC", "kind": "option",
                "start_timestamp": start_ms, "end_timestamp": end_ms,
                "count": 10, "sorting": "desc",
            }
            params.update(extra)
            url = f"{BASE}/public/get_last_trades_by_currency_and_time"
            async with s.get(url, params=params) as r:
                data = await r.json()
                error = data.get("error")
                result = data.get("result", {})
                trades = result.get("trades", []) if isinstance(result, dict) else []
                has_more = result.get("has_more", False) if isinstance(result, dict) else False
                
                if error:
                    print(f"  [{label}] ERROR: {error}")
                else:
                    print(f"  [{label}] trades={len(trades)}, has_more={has_more}")
                    if trades:
                        for t in trades[:3]:
                            print(f"    {t['instrument_name']}  price={t['price']}  "
                                  f"amount={t['amount']}  "
                                  f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # ============================================================
        # 3. Try ACTIVE instrument's trades as control
        # ============================================================
        print("\n" + "=" * 80)
        print("3. Control: Active instrument trades (should work)")
        print("=" * 80)
        
        r = await s.get(f"{BASE}/public/get_instruments",
                        params={"currency": "BTC", "kind": "option", "expired": "false"})
        data = await r.json()
        active = data.get("result", [])
        # Find one that's likely liquid (near ATM, nearest expiry)
        now = datetime.now(timezone.utc)
        nearest = [i for i in active if i["expiration_timestamp"]/1000 - now.timestamp() < 7*86400]
        if nearest:
            # Sort by how close strike is to ~84000
            nearest.sort(key=lambda x: abs(x["strike"] - 84000))
            name = nearest[0]["instrument_name"]
            print(f"Testing: {name}")
            
            start_ms = int((now - timedelta(hours=24)).timestamp() * 1000)
            end_ms = int(now.timestamp() * 1000)
            trades, info = await get_trades(s, name, start_ms, end_ms)
            if trades is None:
                print(f"  ERROR: {info}")
            else:
                print(f"  trades={len(trades)}, has_more={info}")
                if trades:
                    for t in trades[:3]:
                        print(f"    price={t['price']}  amount={t['amount']}  "
                              f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # Also by currency for NOW
        start_ms = int((now - timedelta(hours=1)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        params = {
            "currency": "BTC", "kind": "option",
            "start_timestamp": start_ms, "end_timestamp": end_ms,
            "count": 10, "sorting": "desc",
        }
        url = f"{BASE}/public/get_last_trades_by_currency_and_time"
        async with s.get(url, params=params) as r:
            data = await r.json()
            result = data.get("result", {})
            trades = result.get("trades", []) if isinstance(result, dict) else []
            print(f"\n  by_currency last 1h: trades={len(trades)}")
            if trades:
                for t in trades[:5]:
                    print(f"    {t['instrument_name']}  price={t['price']}  "
                          f"amount={t['amount']}  "
                          f"time={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # ============================================================
        # 4. Specifically try OHLCV for a real recently expired instrument
        # ============================================================
        print("\n" + "=" * 80)
        print("4. OHLCV (tradingview_chart_data) for real expired instruments")
        print("=" * 80)
        
        for inst in test_expired[:3]:
            name = inst["instrument_name"]
            created_ts = inst["creation_timestamp"]
            exp_ts = inst["expiration_timestamp"]
            
            url = f"{BASE}/public/get_tradingview_chart_data"
            params = {
                "instrument_name": name,
                "start_timestamp": created_ts,
                "end_timestamp": exp_ts,
                "resolution": "60",
            }
            async with s.get(url, params=params) as r:
                data = await r.json()
                error = data.get("error")
                result = data.get("result", {})
                if error:
                    print(f"  {name}: ERROR {error}")
                else:
                    ticks = result.get("ticks", [])
                    if ticks:
                        print(f"  {name}: {len(ticks)} bars  "
                              f"from {datetime.utcfromtimestamp(ticks[0]/1000)} "
                              f"to {datetime.utcfromtimestamp(ticks[-1]/1000)}")
                    else:
                        print(f"  {name}: 0 bars (empty)")

        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)


asyncio.run(main())

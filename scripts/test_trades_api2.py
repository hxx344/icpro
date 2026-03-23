"""Test alternative Deribit endpoints for historical option data."""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta

BASE = "https://www.deribit.com/api/v2"


async def test_endpoint(session, endpoint, params, label):
    url = f"{BASE}{endpoint}"
    async with session.get(url, params=params) as r:
        data = await r.json()
        result = data.get("result", {})
        if isinstance(result, dict):
            trades = result.get("trades", [])
            has_more = result.get("has_more", False)
        elif isinstance(result, list):
            trades = result
            has_more = False
        else:
            trades = []
            has_more = False

        print(f"\n--- {label} ---")
        print(f"  Endpoint: {endpoint}")
        print(f"  Status: {r.status}, Trades: {len(trades)}, has_more: {has_more}")
        if trades:
            for t in trades[:3]:
                inst = t.get("instrument_name", "?")
                price = t.get("price", 0)
                amount = t.get("amount", 0)
                ts = t.get("timestamp", 0)
                print(f"    {inst}  price={price}  amount={amount}  ts={ts}")
        else:
            err = data.get("error", {})
            if err:
                print(f"  Error: {err}")


async def main():
    async with aiohttp.ClientSession() as s:
        now = datetime.now(timezone.utc)

        # Test 1: get_last_trades_by_currency_and_time for ALL BTC options at a historical 8AM
        # Try yesterday
        yesterday = now - timedelta(days=1)
        yd8am = yesterday.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((yd8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((yd8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades yesterday 8AM ({yd8am.date()})")

        # Test 2: Same but 1 week ago
        week_ago = now - timedelta(days=7)
        wa8am = week_ago.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((wa8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((wa8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades 1 week ago 8AM ({wa8am.date()})")

        # Test 3: 1 month ago
        month_ago = now - timedelta(days=30)
        ma8am = month_ago.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((ma8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((ma8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades 1 month ago 8AM ({ma8am.date()})")

        # Test 4: 6 months ago
        six_months = now - timedelta(days=180)
        sm8am = six_months.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((sm8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((sm8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades 6 months ago 8AM ({sm8am.date()})")

        # Test 5: 1 year ago
        one_year = now - timedelta(days=365)
        oy8am = one_year.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((oy8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((oy8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades 1 year ago 8AM ({oy8am.date()})")

        # Test 6: 2 years ago
        two_years = now - timedelta(days=730)
        ty8am = two_years.replace(hour=8, minute=0, second=0, microsecond=0)
        start_ms = int((ty8am - timedelta(minutes=30)).timestamp() * 1000)
        end_ms = int((ty8am + timedelta(minutes=30)).timestamp() * 1000)

        await test_endpoint(s,
            "/public/get_last_trades_by_currency_and_time",
            {"currency": "BTC", "kind": "option",
             "start_timestamp": start_ms, "end_timestamp": end_ms,
             "count": 5, "sorting": "asc"},
            f"All BTC option trades 2 years ago 8AM ({ty8am.date()})")

        # Test 7: Recently expired instrument (find one that expired this month)
        # Use get_instruments to find recently expired
        print("\n\n--- Testing recently expired instruments ---")
        result = await s.get(f"{BASE}/public/get_instruments",
                            params={"currency": "BTC", "kind": "option", "expired": "true"})
        data = await result.json()
        expired = data.get("result", [])
        print(f"Recently expired instruments: {len(expired)}")
        if expired:
            # Pick the first one
            inst = expired[0]
            name = inst["instrument_name"]
            exp_ts = inst.get("expiration_timestamp", 0)
            exp_dt = datetime.utcfromtimestamp(exp_ts / 1000).replace(tzinfo=timezone.utc)
            print(f"  Testing: {name} (expired {exp_dt})")

            # Try trades for the day before expiry
            day_before = exp_dt - timedelta(days=1)
            start_ms = int(day_before.replace(hour=7).timestamp() * 1000)
            end_ms = int(day_before.replace(hour=9).timestamp() * 1000)

            await test_endpoint(s,
                "/public/get_last_trades_by_instrument_and_time",
                {"instrument_name": name,
                 "start_timestamp": start_ms, "end_timestamp": end_ms,
                 "count": 5, "sorting": "asc"},
                f"Recently expired: {name} on {day_before.date()}")


asyncio.run(main())

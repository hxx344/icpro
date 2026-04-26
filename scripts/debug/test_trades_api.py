"""Quick test: can Deribit return historical trades for expired instruments?"""
import asyncio
import aiohttp
from datetime import datetime, timezone

BASE = "https://www.deribit.com/api/v2"


async def fetch_trades(session, instrument, start_ms, end_ms):
    url = f"{BASE}/public/get_last_trades_by_instrument_and_time"
    params = {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": 5,
        "sorting": "asc",
    }
    async with session.get(url, params=params) as r:
        data = await r.json()
        result = data.get("result", {})
        trades = result.get("trades", [])
        return trades


async def main():
    async with aiohttp.ClientSession() as s:
        tests = [
            # BTC price was ~$87K on 2025-03-27
            ("BTC-28MAR25-87000-C", "2025-03-27", "ATM call, day before expiry"),
            ("BTC-28MAR25-87000-P", "2025-03-27", "ATM put, day before expiry"),
            ("BTC-28MAR25-85000-C", "2025-03-27", "Near ATM call"),
            # Full day window
            ("BTC-28MAR25-87000-C", "2025-03-26", "ATM call, 2 days before"),
            # Earlier date: BTC ~$100K in Jan 2025
            ("BTC-31JAN25-100000-C", "2025-01-30", "Jan 2025 ATM call"),
            # 2024: BTC ~$43K in Jan 2024
            ("BTC-26JAN24-43000-C", "2024-01-25", "Jan 2024 ATM call"),
            # 2023: BTC ~$23K in Jan 2023
            ("BTC-27JAN23-23000-C", "2023-01-26", "Jan 2023 ATM call"),
            # Daily 0DTE option - BTC ~$97K on Dec 15, 2025
            ("BTC-15DEC25-97000-C", "2025-12-14", "Dec 2025 daily"),
        ]

        for inst, date_str, desc in tests:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # Window: 07:00 - 09:00 UTC
            start_ms = int(dt.replace(hour=7).timestamp() * 1000)
            end_ms = int(dt.replace(hour=9).timestamp() * 1000)

            trades = await fetch_trades(s, inst, start_ms, end_ms)
            if trades:
                prices = [t["price"] for t in trades]
                amounts = [t["amount"] for t in trades]
                print(f"OK  {inst:30s} {date_str}  {desc:30s}  trades={len(trades)}  "
                      f"price={prices[0]:.4f}  amt={amounts[0]}")
            else:
                print(f"--  {inst:30s} {date_str}  {desc:30s}  NO TRADES")

            # Also try FULL day window (00:00 - 23:59)
            start_full = int(dt.replace(hour=0).timestamp() * 1000)
            end_full = int(dt.replace(hour=23, minute=59).timestamp() * 1000)
            trades_full = await fetch_trades(s, inst, start_full, end_full)
            if trades_full and not trades:
                print(f"    ^ but FULL DAY has {len(trades_full)} trades "
                      f"(price={trades_full[0]['price']:.4f})")


asyncio.run(main())

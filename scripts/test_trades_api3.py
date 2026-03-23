"""Test trades API with ACTIVE instruments to verify it works at all."""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta

BASE = "https://www.deribit.com/api/v2"


async def main():
    async with aiohttp.ClientSession() as s:
        # Get active instruments
        r = await s.get(f"{BASE}/public/get_instruments",
                        params={"currency": "BTC", "kind": "option", "expired": "false"})
        data = await r.json()
        instruments = data.get("result", [])
        print(f"Active instruments: {len(instruments)}")
        if not instruments:
            return

        # Pick the first one
        name = instruments[0]["instrument_name"]
        print(f"Testing active: {name}")

        # Get recent trades (last 4 hours)
        now = datetime.now(timezone.utc)
        start_ms = int((now - timedelta(hours=4)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        r2 = await s.get(f"{BASE}/public/get_last_trades_by_instrument_and_time",
                         params={"instrument_name": name,
                                 "start_timestamp": start_ms,
                                 "end_timestamp": end_ms,
                                 "count": 10, "sorting": "desc"})
        data2 = await r2.json()
        trades = data2.get("result", {}).get("trades", [])
        print(f"Trades by instrument in last 4h: {len(trades)}")
        for t in trades[:5]:
            print(f"  {t['instrument_name']}  price={t['price']}  amount={t['amount']}  "
                  f"ts={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # Try by currency - last 4 hours
        r3 = await s.get(f"{BASE}/public/get_last_trades_by_currency_and_time",
                         params={"currency": "BTC", "kind": "option",
                                 "start_timestamp": start_ms,
                                 "end_timestamp": end_ms,
                                 "count": 10, "sorting": "desc"})
        data3 = await r3.json()
        trades3 = data3.get("result", {}).get("trades", [])
        print(f"\nAll BTC option trades in last 4h: {len(trades3)}")
        for t in trades3[:5]:
            print(f"  {t['instrument_name']}  price={t['price']}  amount={t['amount']}  "
                  f"ts={datetime.utcfromtimestamp(t['timestamp']/1000)}")

        # Test an expired instrument
        print(f"\n--- Expired instrument test ---")
        r4 = await s.get(f"{BASE}/public/get_instruments",
                         params={"currency": "BTC", "kind": "option", "expired": "true"})
        data4 = await r4.json()
        expired = data4.get("result", [])
        print(f"Today's expired: {len(expired)}")
        if expired:
            exp_name = expired[0]["instrument_name"]
            exp_ts = expired[0].get("expiration_timestamp", 0)
            print(f"Testing expired: {exp_name} (exp={datetime.utcfromtimestamp(exp_ts/1000)})")

            # Query trades 24h before expiry
            start_ms = exp_ts - 24 * 3600 * 1000
            end_ms = exp_ts
            r5 = await s.get(f"{BASE}/public/get_last_trades_by_instrument_and_time",
                             params={"instrument_name": exp_name,
                                     "start_timestamp": start_ms,
                                     "end_timestamp": end_ms,
                                     "count": 10, "sorting": "desc"})
            data5 = await r5.json()
            trades5 = data5.get("result", {}).get("trades", [])
            print(f"Trades 24h before expiry: {len(trades5)}")
            for t in trades5[:3]:
                print(f"  price={t['price']}  amount={t['amount']}  "
                      f"ts={datetime.utcfromtimestamp(t['timestamp']/1000)}")

            if not trades5:
                # Try get_last_trades_by_instrument (no time filter)
                r6 = await s.get(f"{BASE}/public/get_last_trades_by_instrument",
                                 params={"instrument_name": exp_name, "count": 5, "sorting": "desc"})
                data6 = await r6.json()
                trades6 = data6.get("result", {}).get("trades", [])
                print(f"Last trades (no time filter): {len(trades6)}")
                for t in trades6[:3]:
                    print(f"  price={t['price']}  amount={t['amount']}  "
                          f"ts={datetime.utcfromtimestamp(t['timestamp']/1000)}")


asyncio.run(main())

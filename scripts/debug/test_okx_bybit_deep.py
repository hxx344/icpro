"""Test OKX and Bybit for any available historical option trade data."""
import asyncio
import aiohttp


async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:

        # ==================== OKX ====================
        print("=" * 60)
        print("  OKX - Historical Option Data")
        print("=" * 60)

        # 1. Try active option candles (not the expired one that failed)
        url = "https://www.okx.com/api/v5/public/instruments"
        params = {"instType": "OPTION", "uly": "BTC-USD"}
        async with s.get(url, params=params) as r:
            data = await r.json()
            insts = data.get("data", [])
            active_names = [i["instId"] for i in insts[:3]]
            print(f"Active BTC options: {len(insts)}")

        # Try candles for an active option
        if active_names:
            for name in active_names[:2]:
                url = "https://www.okx.com/api/v5/market/candles"
                params = {"instId": name, "bar": "1H", "limit": "3"}
                async with s.get(url, params=params) as r:
                    data = await r.json()
                    candles = data.get("data", [])
                    print(f"  {name}: {len(candles)} candles")
                    if candles:
                        print(f"    Sample: {candles[0][:6]}")

        # 2. OKX option trades history
        url = "https://www.okx.com/api/v5/market/trades"
        if active_names:
            params = {"instId": active_names[0], "limit": "5"}
            async with s.get(url, params=params) as r:
                data = await r.json()
                trades = data.get("data", [])
                print(f"  Trades for {active_names[0]}: {len(trades)}")

        # 3. OKX history-trades - uses trade ID pagination, goes further back
        url = "https://www.okx.com/api/v5/market/history-trades"
        if active_names:
            params = {"instId": active_names[0], "limit": "5", "type": "1"}
            async with s.get(url, params=params) as r:
                data = await r.json()
                trades = data.get("data", [])
                code = data.get("code", "?")
                msg = data.get("msg", "")
                print(f"  History-trades: code={code}, count={len(trades)}, msg={msg}")

        # 4. OKX option summary / open interest
        url = "https://www.okx.com/api/v5/public/opt-summary"
        params = {"uly": "BTC-USD"}
        async with s.get(url, params=params) as r:
            data = await r.json()
            items = data.get("data", [])
            print(f"  Option summary (mark/IV/greeks): {len(items)} instruments")
            if items:
                i0 = items[0]
                print(f"    {i0.get('instId')}: mark={i0.get('markVol')}, "
                      f"delta={i0.get('delta')}, gamma={i0.get('gamma')}")

        # ==================== BYBIT ====================
        print("\n" + "=" * 60)
        print("  BYBIT - Historical Option Data")
        print("=" * 60)

        # 1. Get instruments
        url = "https://api.bybit.com/v5/market/instruments-info"
        params = {"category": "option", "baseCoin": "BTC", "limit": "100"}
        async with s.get(url, params=params) as r:
            data = await r.json()
            items = data.get("result", {}).get("list", [])
            print(f"Active BTC options: {len(items)}")
            bybit_names = [i["symbol"] for i in items[:5]]

        # 2. Try kline
        if bybit_names:
            for name in bybit_names[:2]:
                url = "https://api.bybit.com/v5/market/kline"
                params = {"category": "option", "symbol": name, "interval": "60", "limit": "3"}
                async with s.get(url, params=params) as r:
                    data = await r.json()
                    klines = data.get("result", {}).get("list", [])
                    ret_msg = data.get("retMsg", "")
                    print(f"  {name}: {len(klines)} bars, msg={ret_msg}")

        # 3. Bybit tickers - option chain snapshot
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "option", "baseCoin": "BTC"}
        async with s.get(url, params=params) as r:
            data = await r.json()
            items = data.get("result", {}).get("list", [])
            print(f"  Option tickers: {len(items)}")
            if items:
                i0 = items[0]
                print(f"    {i0.get('symbol')}: mark={i0.get('markPrice')}, "
                      f"markIv={i0.get('markIv')}, delta={i0.get('delta')}")

        # 4. Bybit historical volatility
        url = "https://api.bybit.com/v5/market/historical-volatility"
        params = {"category": "option", "baseCoin": "BTC"}
        async with s.get(url, params=params) as r:
            data = await r.json()
            items = data.get("result", [])
            if isinstance(items, list) and items:
                print(f"  Historical vol data points: {len(items)}")
                print(f"    Latest: {items[0]}")
                print(f"    Oldest: {items[-1]}")
            else:
                print(f"  Historical vol: {data.get('retMsg', '?')}")

        # ==================== SUMMARY ====================
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)


asyncio.run(main())

"""
Test historical option data availability from Binance, OKX, and Bybit.
"""
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta


async def test_binance(session):
    """Binance European Options - data.binance.vision bulk downloads + API."""
    print("=" * 60)
    print("  BINANCE OPTIONS")
    print("=" * 60)

    # 1. Check public data listing for options
    # Binance provides data.binance.vision for historical downloads
    url = "https://data.binance.vision/?prefix=data/option/daily/klines/"
    async with session.get(url) as r:
        text = await r.text()
        # Count instrument folders
        items = text.count("<Key>")
        print(f"  data.binance.vision option klines: status={r.status}")

    # 2. Try Binance Options API - get available instruments
    url = "https://eapi.binance.com/eapi/v1/exchangeInfo"
    async with session.get(url) as r:
        if r.status == 200:
            data = await r.json()
            symbols = data.get("optionSymbols", [])
            print(f"  Active option symbols: {len(symbols)}")
            if symbols:
                # Show some examples
                btc_opts = [s for s in symbols if s.get("underlying", "").startswith("BTC")]
                eth_opts = [s for s in symbols if s.get("underlying", "").startswith("ETH")]
                print(f"  BTC options: {len(btc_opts)}, ETH options: {len(eth_opts)}")
                if btc_opts:
                    s0 = btc_opts[0]
                    print(f"  Example: {s0.get('symbol')} exp={s0.get('expiryDate')}")
        else:
            print(f"  exchangeInfo: status={r.status}")

    # 3. Try historical klines for a specific option
    # Binance options klines endpoint
    url = "https://eapi.binance.com/eapi/v1/klines"
    now = datetime.now(timezone.utc)
    start_ms = int((now - timedelta(days=30)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # Try to get klines for an active option
    url2 = "https://eapi.binance.com/eapi/v1/exchangeInfo"
    async with session.get(url2) as r:
        if r.status == 200:
            data = await r.json()
            symbols = data.get("optionSymbols", [])
            btc_opts = [s for s in symbols if "BTC" in s.get("underlying", "")]
            if btc_opts:
                sym = btc_opts[0]["symbol"]
                params = {"symbol": sym, "interval": "1h", "limit": 5}
                async with session.get(url, params=params) as kr:
                    if kr.status == 200:
                        kdata = await kr.json()
                        print(f"  Klines for {sym}: {len(kdata)} bars")
                        if kdata:
                            print(f"    First bar: {kdata[0][:6]}")
                    else:
                        print(f"  Klines status: {kr.status}")

    # 4. Check if expired options data is available via data.binance.vision
    # Try downloading a specific historical file
    test_url = "https://data.binance.vision/data/option/daily/klines/BTC-250328-90000-C/1h/BTC-250328-90000-C-1h-2025-03-01.zip"
    async with session.head(test_url) as r:
        print(f"  Historical zip (expired option): status={r.status}")
        if r.status == 200:
            cl = r.headers.get("Content-Length", "?")
            print(f"    Size: {cl} bytes")

    # Try a 2024 option
    test_url2 = "https://data.binance.vision/data/option/daily/klines/BTC-240329-50000-C/1h/BTC-240329-50000-C-1h-2024-03-01.zip"
    async with session.head(test_url2) as r:
        print(f"  2024 expired option zip: status={r.status}")

    # Try a 2023 option
    test_url3 = "https://data.binance.vision/data/option/daily/klines/BTC-230331-30000-C/1h/BTC-230331-30000-C-1h-2023-03-01.zip"
    async with session.head(test_url3) as r:
        print(f"  2023 expired option zip: status={r.status}")


async def test_okx(session):
    """OKX Options - REST API historical data."""
    print("\n" + "=" * 60)
    print("  OKX OPTIONS")
    print("=" * 60)

    # 1. Get active instruments
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "OPTION", "uly": "BTC-USD"}
    async with session.get(url, params=params) as r:
        if r.status == 200:
            data = await r.json()
            instruments = data.get("data", [])
            print(f"  Active BTC options: {len(instruments)}")
            if instruments:
                print(f"  Example: {instruments[0].get('instId')}")
        else:
            print(f"  instruments: status={r.status}")

    # 2. Try historical candles for an active option
    if instruments:
        inst_id = instruments[0]["instId"]
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": inst_id, "bar": "1H", "limit": "5"}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                data = await r.json()
                candles = data.get("data", [])
                print(f"  Active candles for {inst_id}: {len(candles)} bars")
            else:
                print(f"  candles: status={r.status}")

    # 3. Try history-candles (for older data / expired instruments)
    # OKX provides /api/v5/market/history-candles for history up to recent months
    url = "https://www.okx.com/api/v5/market/history-candles"
    # Try an expired instrument
    params = {"instId": "BTC-USD-250328-90000-C", "bar": "1H", "limit": "5"}
    async with session.get(url, params=params) as r:
        if r.status == 200:
            data = await r.json()
            candles = data.get("data", [])
            code = data.get("code", "?")
            msg = data.get("msg", "")
            print(f"  Expired instrument history: code={code}, bars={len(candles)}, msg={msg}")
        else:
            print(f"  history-candles (expired): status={r.status}")

    # 4. Try a 2024 expired option
    params = {"instId": "BTC-USD-240329-50000-C", "bar": "1H", "limit": "5"}
    async with session.get(url, params=params) as r:
        if r.status == 200:
            data = await r.json()
            candles = data.get("data", [])
            code = data.get("code", "?")
            msg = data.get("msg", "")
            print(f"  2024 expired option: code={code}, bars={len(candles)}, msg={msg}")
        else:
            print(f"  history-candles (2024): status={r.status}")


async def test_bybit(session):
    """Bybit Options - REST API."""
    print("\n" + "=" * 60)
    print("  BYBIT OPTIONS")
    print("=" * 60)

    # 1. Get active instruments
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": "option", "baseCoin": "BTC", "limit": "10"}
    async with session.get(url, params=params) as r:
        if r.status == 200:
            data = await r.json()
            items = data.get("result", {}).get("list", [])
            print(f"  Active BTC options: {len(items)}")
            if items:
                print(f"  Example: {items[0].get('symbol')}")
        else:
            print(f"  instruments: status={r.status}")

    # 2. Try kline for an active option
    url = "https://api.bybit.com/v5/market/kline"
    if items:
        sym = items[0]["symbol"]
        params = {"category": "option", "symbol": sym, "interval": "60", "limit": "5"}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                data = await r.json()
                klines = data.get("result", {}).get("list", [])
                print(f"  Active klines for {sym}: {len(klines)} bars")
            else:
                print(f"  kline: status={r.status}")

    # 3. Try historical kline for expired option
    params = {"category": "option", "symbol": "BTC-28MAR25-90000-C", "interval": "60", "limit": "5"}
    async with session.get(url, params=params) as r:
        if r.status == 200:
            data = await r.json()
            klines = data.get("result", {}).get("list", [])
            ret_code = data.get("retCode", "?")
            ret_msg = data.get("retMsg", "")
            print(f"  Expired option kline: retCode={ret_code}, bars={len(klines)}, msg={ret_msg}")
        else:
            print(f"  expired kline: status={r.status}")

    # 4. Bybit public data download
    # https://public.bybit.com/option/
    url = "https://public.bybit.com/option/"
    async with session.get(url) as r:
        if r.status == 200:
            text = await r.text()
            # Count date folders
            import re
            dates = re.findall(r'href="(\d{4}-\d{2}-\d{2})/"', text)
            if dates:
                print(f"  Bybit public data: {len(dates)} date folders")
                print(f"    Range: {min(dates)} → {max(dates)}")
            else:
                folders = re.findall(r'href="([^"]+)/"', text)
                print(f"  Bybit public data: {len(folders)} folders")
                if folders[:5]:
                    print(f"    Samples: {folders[:5]}")
        else:
            print(f"  public data: status={r.status}")


async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        try:
            await test_binance(s)
        except Exception as e:
            print(f"  Binance error: {e}")

        try:
            await test_okx(s)
        except Exception as e:
            print(f"  OKX error: {e}")

        try:
            await test_bybit(s)
        except Exception as e:
            print(f"  Bybit error: {e}")


asyncio.run(main())

"""Deep test Binance options data availability via data.binance.vision."""
import asyncio
import aiohttp
import re


async def main():
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        print("=== Binance data.binance.vision - Option data structure ===\n")

        # 1. List what's under data/option/
        url = "https://data.binance.vision/?prefix=data/option/"
        async with s.get(url) as r:
            text = await r.text()
            prefixes = re.findall(r'<Prefix>([^<]+)</Prefix>', text)
            print(f"Option prefixes: {prefixes}")

        # 2. List daily klines instruments
        url = "https://data.binance.vision/?prefix=data/option/daily/klines/&delimiter=/"
        async with s.get(url) as r:
            text = await r.text()
            prefixes = re.findall(r'<Prefix>([^<]+)</Prefix>', text)
            btc_items = [p for p in prefixes if 'BTC' in p]
            print(f"\nTotal option kline folders: {len(prefixes)}")
            print(f"BTC option folders: {len(btc_items)}")
            if btc_items:
                # Show some examples
                print(f"First 5: {btc_items[:5]}")
                print(f"Last 5: {btc_items[-5:]}")

        # 3. Try to list files for a specific instrument
        if btc_items:
            # Pick one that looks like it might have data
            test_inst = btc_items[0]
            url = f"https://data.binance.vision/?prefix={test_inst}1h/&delimiter=/"
            async with s.get(url) as r:
                text = await r.text()
                keys = re.findall(r'<Key>([^<]+)</Key>', text)
                print(f"\nFiles in {test_inst}1h/: {len(keys)}")
                for k in keys[:5]:
                    print(f"  {k}")

        # 4. Try to directly access a known recent option kline file
        # Active options should have data
        # BTC-260327-100000-C is active (expires Mar 27, 2026)
        url = "https://data.binance.vision/?prefix=data/option/daily/klines/BTC-260327-100000-C/1h/&delimiter=/"
        async with s.get(url) as r:
            text = await r.text()
            keys = re.findall(r'<Key>([^<]+)</Key>', text)
            print(f"\nActive option BTC-260327-100000-C 1h files: {len(keys)}")
            for k in keys[:5]:
                print(f"  {k}")

        # 5. Try monthly data (might have aggregated historical)
        url = "https://data.binance.vision/?prefix=data/option/monthly/&delimiter=/"
        async with s.get(url) as r:
            text = await r.text()
            prefixes = re.findall(r'<Prefix>([^<]+)</Prefix>', text)
            print(f"\nOption monthly prefixes: {prefixes}")

        # 6. Check Binance API for historical mark price
        # eapi/v1/mark endpoint
        url = "https://eapi.binance.com/eapi/v1/mark"
        async with s.get(url) as r:
            if r.status == 200:
                data = await r.json()
                btc_marks = [d for d in data if 'BTC' in d.get('symbol', '')]
                print(f"\nBinance mark prices: {len(data)} total, {len(btc_marks)} BTC")
            else:
                print(f"\nBinance mark: status={r.status}")

        # 7. Check if Binance has historical option index data
        url = "https://eapi.binance.com/eapi/v1/index"
        params = {"underlying": "BTCUSDT"}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                data = await r.json()
                print(f"\nBinance option index: {data}")

        # 8. Check Binance Options historical volatility index
        url = "https://eapi.binance.com/eapi/v1/historicalVolatility"
        params = {"underlying": "BTCUSDT"}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                data = await r.json()
                print(f"\nBinance historical vol: {len(data)} data points")
                if data:
                    print(f"  Latest: {data[-1]}")
                    print(f"  Oldest: {data[0]}")


asyncio.run(main())

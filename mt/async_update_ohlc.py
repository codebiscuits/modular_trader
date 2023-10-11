import asyncio
import pandas as pd
from binance import Client, AsyncClient
from time import perf_counter as perf
from pprint import pformat
from resources.loggers import create_logger

# TODO maybe work out a workflow where it loads maybe all files from disk, checks how much new data they each need,
#  separates the ones which need 500 periods or less (get_klines will be sufficient for them) and does them in batches
#  of 20 or 30 or whatever - download the data, stitch together, save the file, then move on to the next batch, then it
#  does others that need longer updates and any new ones it doesn't have at all.

sync_client = Client()

logger = create_logger('async_update_ohlc', 'async_update_ohlc')

async def main():
    client = await AsyncClient.create()

    # Get all symbols
    info = await client.get_exchange_info()

    pairs = [pair['symbol'] for pair in info.get('symbols') if ((pair['quoteAsset'] == 'USDT') and (pair['status'] == 'TRADING'))]
    logger.debug(f"{len(pairs) = }")

    # Create a list of tasks to download the klines
    tasks = []
    for symbol in pairs:
        tasks.append(asyncio.create_task(client.get_historical_klines(symbol=symbol,
                                                                      interval=Client.KLINE_INTERVAL_1HOUR,
                                                                      start_str="4 years ago UTC")))
        # tasks.append(asyncio.create_task(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE)))

    # Await the completion of all tasks
    results = await asyncio.gather(*tasks)
    print(f"used-weight: {client.response.headers['x-mbx-used-weight']}")
    print(f"used-weight-1m: {client.response.headers['x-mbx-used-weight-1m']}")

    # Create a dictionary to store the dataframes
    df_dict = {}
    for pair, klines in zip(pairs, results):
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
                'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = df['timestamp'] * 1000000
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # check df is localised to UTC
        try:
            print(f"funcs get_bin_ohlc - {pair} ohlc data wasn't timezone aware, fixing now.")
            df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
        except TypeError:
            pass

        df = df.drop(['close_time', 'ignore'], axis=1)

        print(f"{pair} {len(df) = }")

        df.to_parquet(f"bin_ohlc_5m/{pair}_5m.parquet")

    await client.close_connection()

start = perf()
asyncio.run(main())
elapsed = perf() - start
print(f"Time taken: {int(elapsed // 60)}m {elapsed % 60:.2f}s")
import asyncio
import pandas as pd
from binance import Client, AsyncClient
from time import perf_counter as perf
from mt.resources.loggers import create_logger
from pyarrow import ArrowInvalid
from datetime import datetime, timezone
from mt.sessions import TradingSession
from pathlib import Path

sync_client = Client()
session = TradingSession(0.1, True)
logger = create_logger('async_update_ohlc', 'async_update_ohlc')



async def stitch(pair, klines, all_data):
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    new_df = pd.DataFrame(klines, columns=cols)
    new_df['timestamp'] = new_df['timestamp'] * 1000000
    new_df = new_df.astype(float)
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    # check df is localised to UTC
    try:
        new_df['timestamp'] = new_df.timestamp.dt.tz_localize('UTC')
    except TypeError:
        pass

    new_df = new_df.drop(['close_time', 'ignore'], axis=1)

    old_data = all_data[pair]['ohlc']

    if old_data is not None:
        old_data = old_data.loc[old_data.timestamp < new_df.timestamp.iloc[0]]
        df = pd.concat([old_data, new_df], axis=0, ignore_index=True)
    else:
        df = new_df

    extra_data = {'pair': pair}
    extra_data['roc_1d'] = df.close.rolling(12).mean().ffill().pct_change(288).iloc[-1]
    extra_data['roc_1w'] = df.close.rolling(84).mean().ffill().pct_change(2016).iloc[-1]
    extra_data['roc_1m'] = df.close.rolling(360).mean().ffill().pct_change(8640).iloc[-1]

    extra_data['volume_1d'] = df.tail(288).quote_vol.sum()
    extra_data['volume_1w'] = df.tail(2016).quote_vol.sum()
    extra_data['volatility_1d'] = df.tail(288).close.ffill().pct_change().std()
    extra_data['volatility_1w'] = df.tail(2016).close.ffill().pct_change().std()

    extra_data['length'] = len(df)

    ohlc_w = Path(f'{session.ohlc_w}/{pair}.parquet')
    df.to_parquet(ohlc_w)

    return extra_data

async def main(pairs):
    async_client = await AsyncClient.create()

    logger.debug(f"{len(pairs) = }")
    all_data = {}
    for pair in pairs:
        ohlc_r = Path(f'{session.ohlc_r}/{pair}.parquet')
        ohlc_w = Path(f'{session.ohlc_w}/{pair}.parquet')
        try:
            old_df = pd.read_parquet(ohlc_r)
            last_timestamp = old_df.timestamp.iloc[-1].timestamp()
            now = datetime.now(timezone.utc).timestamp()
            data_age = int((now - last_timestamp) / 300)  # dividing by 300 shows how many 5min periods have passed
        except (ArrowInvalid, OSError):
            logger.exception(f"Problem reading {pair} parquet file, downloading from scratch.")
            ohlc_r.unlink()
            old_df = None
            data_age = 1001
            print(f"failed to load {pair} ohlc")

        all_data[pair] = {'ohlc': old_df, 'age': max(data_age, 5)}

    # Create a list of tasks to download the klines
    tasks = []
    for symbol in pairs:
        if all_data[symbol]['age'] <= 1000:
            # print(f"quick downloading {symbol}")
            tasks.append(asyncio.create_task(async_client.get_klines(symbol=symbol,
                                                                     interval=Client.KLINE_INTERVAL_5MINUTE,
                                                                     limit=all_data[symbol]['age'])))
        else:
            print(f"slow downloading {symbol}")
            tasks.append(asyncio.create_task(async_client.get_historical_klines(symbol=symbol,
                                                                                interval=Client.KLINE_INTERVAL_5MINUTE,
                                                                                start_str="6 years ago UTC")))

    # Await the completion of all tasks
    results = await asyncio.gather(*tasks)
    print(f"used-weight: {async_client.response.headers['x-mbx-used-weight']}")
    print(f"used-weight-1m: {async_client.response.headers['x-mbx-used-weight-1m']}")

    # stitch them all together
    stitch_tasks = []
    for pair, klines in zip(pairs, results):
        stitch_tasks.append(asyncio.create_task(stitch(pair, klines, all_data)))
    extras = await asyncio.gather(*stitch_tasks)

    await async_client.close_connection()

    return pd.DataFrame(extras)


start = perf()

# Get all symbols
for sym in session.info['symbols']:
    if sym['status'] != 'TRADING':
        dead_symbol = sym['symbol']
        fp = Path(f"{session.ohlc_w}/{dead_symbol}.parquet")
        if fp.exists():
            fp.unlink()
all_pairs = list(session.pairs_data.keys())

pairs_0 = all_pairs[::4]
pairs_1 = all_pairs[1::4]
pairs_2 = all_pairs[2::4]
pairs_3 = all_pairs[3::4]

extra_df_1 = asyncio.run(main(pairs_0))
extra_df_2 = asyncio.run(main(pairs_1))
extra_df_3 = asyncio.run(main(pairs_2))
extra_df_4 = asyncio.run(main(pairs_3))

extra_df = pd.concat([extra_df_1, extra_df_2, extra_df_3, extra_df_4], ignore_index=True)
extra_df['rank_1d'] = extra_df['roc_1d'].rank(pct=True)
extra_df['rank_1w'] = extra_df['roc_1w'].rank(pct=True)
extra_df['rank_1m'] = extra_df['roc_1m'].rank(pct=True)
print(extra_df)
extra_df.to_parquet()

elapsed = perf() - start
print(f"Time taken: {int(elapsed // 60)}m {elapsed % 60:.2f}s")

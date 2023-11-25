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
session = TradingSession(0.1, True, True)
logger = create_logger('async_update_ohlc', 'async_update_ohlc')


async def stitch(pair, klines, all_data, tf):
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

    max_dict = {'1m': 1_051_200, '5m': 420_000}  # 2 years worth of 1m periods, 4 years worth of 5m periods
    max_len = max_dict[tf]
    if len(df) > max_len:
        # print(f"trimming ohlc from {len(df)} to {max_len}")
        df = df.tail(max_len).reset_index(drop=True)

    extra_data = {'pair': pair,
                  'roc_1d': df.close.rolling(12).mean().ffill().pct_change(288).iloc[-1],
                  'roc_1w': df.close.rolling(84).mean().ffill().pct_change(2016).iloc[-1],
                  'roc_1m': df.close.rolling(360).mean().ffill().pct_change(8640).iloc[-1],
                  'volume_1d': df.tail(288).quote_vol.sum(),
                  'volume_1w': df.tail(2016).quote_vol.sum(),
                  'volatility_1d': df.tail(288).close.ffill().pct_change().std(),
                  'volatility_1w': df.tail(2016).close.ffill().pct_change().std(),
                  'length': len(df)}

    ohlc_w = Path(f'/home/ross/coding/modular_trader/bin_ohlc_{tf}/{pair}.parquet')
    df.to_parquet(ohlc_w)

    return extra_data


async def main(pairs, tf):
    async_client = await AsyncClient.create()

    all_data = {}
    for pair in pairs:
        ohlc_r = Path(f'{session.ohlc_r}/{pair}.parquet')
        try:
            old_df = pd.read_parquet(ohlc_r)
            last_timestamp = old_df.timestamp.iloc[-1].timestamp()
            now = datetime.now(timezone.utc).timestamp()
            data_age = int((now - last_timestamp) / 300)  # dividing by 300 shows how many 5min periods have passed
        except FileNotFoundError:
            logger.error(f"{pair} parquet file missing, downloading from scratch.")
            old_df = None
            data_age = 1001
            logger.info(f"failed to load {pair} ohlc")
        except (ArrowInvalid, OSError):
            ohlc_r.unlink()
            logger.exception(f"Problem reading {pair} parquet file, downloading from scratch.")
            old_df = None
            data_age = 1001
            logger.info(f"failed to load {pair} ohlc")

        all_data[pair] = {'ohlc': old_df, 'age': max(data_age, 5)}

    # Create a list of tasks to download the klines
    if tf == '1m':
        interval = Client.KLINE_INTERVAL_1MINUTE
    elif tf == '5m':
        interval = Client.KLINE_INTERVAL_5MINUTE

    tasks = []
    for symbol in pairs:
        if all_data[symbol]['age'] <= 1000:
            # print(f"quick downloading {symbol}")
            tasks.append(asyncio.create_task(async_client.get_klines(symbol=symbol,
                                                                     interval=interval,
                                                                     limit=all_data[symbol]['age'])))
        else:
            logger.info(f"slow downloading {symbol}")
            tasks.append(asyncio.create_task(async_client.get_historical_klines(symbol=symbol,
                                                                                interval=interval,
                                                                                start_str="6 years ago UTC")))

    # Await the completion of all tasks
    results = await asyncio.gather(*tasks)
    logger.info(f"used-weight: {async_client.response.headers['x-mbx-used-weight']}")
    logger.info(f"used-weight-1m: {async_client.response.headers['x-mbx-used-weight-1m']}")

    # stitch them all together
    stitch_tasks = []
    for pair, klines in zip(pairs, results):
        stitch_tasks.append(asyncio.create_task(stitch(pair, klines, all_data, tf)))
    extras = await asyncio.gather(*stitch_tasks)

    await async_client.close_connection()

    return pd.DataFrame(extras)


start = perf()
now_start = datetime.now(tz=timezone.utc).strftime("%d-%m-%y %H:%M:%S")
logger.info(f"Async Update OHLC starting at {now_start}\n")

# Get all symbols
for sym in session.info['symbols']:
    if sym['status'] != 'TRADING':
        dead_symbol = sym['symbol']
        fp = Path(f"{session.ohlc_w}/{dead_symbol}.parquet")
        if fp.exists():
            fp.unlink()
all_pairs = list(session.pairs_data.keys())

divs = 6
extra_dfs = []
for div in range(divs):
    logger.info(f"division {div + 1} of {divs}")
    pairs = all_pairs[div::divs]
    extra_dfs.append(asyncio.run(main(pairs, '5m')))

extra_df = pd.concat(extra_dfs, ignore_index=True)
extra_df['rank_1d'] = extra_df['roc_1d'].rank(pct=True)
extra_df['rank_1w'] = extra_df['roc_1w'].rank(pct=True)
extra_df['rank_1m'] = extra_df['roc_1m'].rank(pct=True)
extra_df = extra_df.set_index('pair', drop=True)
# print(extra_df)
mkt_info_path = Path("/home/ross/coding/modular_trader/market_data/market_info.parquet")
extra_df.to_parquet(mkt_info_path)

elapsed = perf() - start
logger.debug(f"Time taken: {int(elapsed // 60)}m {elapsed % 60:.2f}s")
logger.info(f"Time taken: {int(elapsed // 60)}m {elapsed % 60:.2f}s\n\n")

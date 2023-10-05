import pandas as pd
import polars as pl
from resources import binance_funcs as funcs
import time
from datetime import datetime, timezone
from pathlib import Path
from sessions import TradingSession
from pyarrow import ArrowInvalid
import json
from resources.loggers import create_logger

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

logger = create_logger('update_ohlc', 'update_ohlc')

now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')

# session = LightSession()
session = TradingSession(0.1)

logger.debug(f"{'-:-' * 10} {now} UTC running update_ohlc {'-:-' * 10}")
logger.debug(f"ohlc read path: {session.ohlc_r}")
logger.debug(f"ohlc write path: {session.ohlc_w}")

start = time.perf_counter()

for sym in session.info['symbols']:
    if sym['status'] != 'TRADING':
        dead_symbol = sym['symbol']
        fp = Path(f"{session.ohlc_w}/{dead_symbol}.parquet")
        if fp.exists():
            fp.unlink()

pairs = list(session.pairs_data.keys())
rocs = {}
volumes_1d = {}
volumes_1w = {}
volatilities_1d = {}
volatilities_1w = {}

def from_scratch(session, pair, tf):
    df_start = time.perf_counter()
    df = funcs.get_ohlc(pair, tf, '2 years ago UTC', session)
    df_end = time.perf_counter()
    elapsed = df_end - df_start
    logger.info(f'downloaded {pair} from scratch, took {int(elapsed // 60)}m {elapsed % 60:.1f}s')
    return df


def iterations(n, session, pair, tf):
    # print(f"{n} {pair} {tf}")
    session.set_ohlc_tf(tf)
    # print(session.ohlc_data)
    ohlc_r = Path(f'{session.ohlc_r}/{pair}.parquet')
    ohlc_w = Path(f'{session.ohlc_w}/{pair}.parquet')
    # print(filepath)
    # -------------------- if theres already some local data -------------------------#
    if ohlc_r.exists():

        try:
            pldf = pl.read_parquet(source=ohlc_r, use_pyarrow=True)
            df = pldf.to_pandas()

        except (ArrowInvalid, OSError) as e:
            logger.error('Error:\n', e)
            logger.error(f"Problem reading {pair} parquet file, downloading from scratch.")
            logger.exception(e)
            ohlc_r.unlink()
            df = from_scratch(session, pair, tf)

        # df = pd.read_parquet(filepath)

        if len(df) > 2:
            df = funcs.update_ohlc(pair, tf, df, session)
    # -------------------- if theres no local data yet -------------------------#
    else:
        df = from_scratch(session, pair, tf)
        # print(f'{n} downloaded {pair} from scratch')

    max_dict = {'1m': 1051200, '5m': 210240, '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        # print(f"trimming ohlc from {len(df)} to {max_len}")
        df = df.tail(max_len).reset_index(drop=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # save to file
    pldf = pl.from_pandas(df)
    pldf.write_parquet(ohlc_w, row_group_size=10512, use_pyarrow=True)

    return df


def mkt_rank(rocs_dict):
    rocs_df = pd.DataFrame.from_dict(rocs_dict, orient='index')

    rocs_df['rank_1d'] = rocs_df['1d'].rank(pct=True)
    rocs_df['rank_1w'] = rocs_df['1w'].rank(pct=True)
    rocs_df['rank_1m'] = rocs_df['1m'].rank(pct=True)

    rocs_df = rocs_df.drop(['1d', '1w', '1m'], axis=1)

    filepath = session.mkt_data_w / 'market_ranks.parquet'
    rocs_df.to_parquet(path=filepath)

# TODO maybe combine mkt_rank, volumes and volatilities into one dataframe and save as one parquet file
def save_volumes(vols, period):
    vol_path = Path(f'recent_1{period}_volumes.json')
    vol_path.touch(exist_ok=True)

    with open(vol_path, 'w') as file:
        json.dump(vols, file)


def save_volatilities(vols, period):
    vol_path = Path(f'recent_1{period}_volatilities.json')
    vol_path.touch(exist_ok=True)

    with open(vol_path, 'w') as file:
        json.dump(vols, file)


iterations(0, session, 'BTCUSDT', '1m')
iterations(1, session, 'ETHUSDT', '1m')

for n, pair in enumerate(pairs):
    # print(n, pair)
    try:
        logger.debug(pair)
        df = iterations(n, session, pair, '5m')
    except Exception as e:
        logger.error(f"*** {pair} exception during download, data not downloaded ***")
        logger.exception(e)
        continue

    rocs[pair] = {}
    rocs[pair]['1d'] = df.close.rolling(12).mean().pct_change(288).iloc[-1]
    rocs[pair]['1w'] = df.close.rolling(84).mean().pct_change(2016).iloc[-1]
    rocs[pair]['1m'] = df.close.rolling(360).mean().pct_change(8640).iloc[-1]

    volumes_1d[pair] = df.tail(288).quote_vol.sum()
    volumes_1w[pair] = df.tail(2016).quote_vol.sum()
    volatilities_1d[pair] = df.tail(288).close.pct_change().std()
    volatilities_1w[pair] = df.tail(2016).close.pct_change().std()

mkt_rank(rocs)
save_volumes(volumes_1d, 'd')
save_volumes(volumes_1w, 'w')
save_volatilities(volatilities_1d, 'd')
save_volatilities(volatilities_1w, 'w')

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round(all_time // 60)}m {round(all_time % 60)}s'

logger.info(f"used-weight-1m: {session.client.response.headers['x-mbx-used-weight-1m']}")
logger.info(f'update_ohlc complete, {elapsed_str}')
logger.debug(f'update_ohlc complete, {elapsed_str}')

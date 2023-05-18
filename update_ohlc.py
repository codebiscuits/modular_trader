import pandas as pd
import polars as pl
import keys
import time
from datetime import datetime, timezone
import binance_funcs as funcs
from binance.client import Client
from pushbullet import Pushbullet
from pathlib import Path
from sessions import LightSession
from pyarrow import ArrowInvalid
import json

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')

session = LightSession()

print('-:-' * 10, f' {now} running update_ohlc ', '-:-' * 10)

start = time.perf_counter()

for sym in session.info['symbols']:
    if sym['status'] != 'TRADING':
        dead_symbol = sym['symbol']
        fp = Path(f"{session.ohlc_data}/{dead_symbol}.parquet")
        if fp.exists():
            fp.unlink()

pairs = list(session.pairs_data.keys())
rocs = {}
volumes = {}

def from_scratch(pair, tf):
    df_start = time.perf_counter()
    df = funcs.get_ohlc(pair, tf, '2 years ago UTC')
    df_end = time.perf_counter()
    elapsed = df_end - df_start
    print(f'downloaded {pair} from scratch, took {int(elapsed // 60)}m {elapsed % 60:.1f}s')
    return df


def iterations(n, pair, tf):
    # print(f"{n} {pair} {tf}")
    session.set_ohlc_tf(tf)
    # print(session.ohlc_data)
    filepath = Path(f'{session.ohlc_data}/{pair}.parquet')
    # print(filepath)
    # -------------------- if theres already some local data -------------------------#
    if filepath.exists():

        try:
            pldf = pl.read_parquet(source=filepath, use_pyarrow=True)
            df = pldf.to_pandas()

        except (ArrowInvalid, OSError) as e:
            print('Error:\n', e)
            print(f"Problem reading {pair} parquet file, downloading from scratch.")
            filepath.unlink()
            df = from_scratch(pair, tf)

        # df = pd.read_parquet(filepath)

        if len(df) > 2:
            df = funcs.update_ohlc(pair, tf, df)
    # -------------------- if theres no local data yet -------------------------#
    else:
        df = from_scratch(pair, tf)
        # print(f'{n} downloaded {pair} from scratch')

    max_dict = {'1m': 1051200, '5m': 210240, '15m': 70080, '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        # print(f"trimming ohlc from {len(df)} to {max_len}")
        df = df.tail(max_len).reset_index(drop=True)

    # df.to_parquet(filepath, compression=None)
    pldf = pl.from_pandas(df)
    pldf.write_parquet(filepath, row_group_size=10512, use_pyarrow=True)

    return df


def mkt_rank(rocs_dict):
    rocs_df = pd.DataFrame.from_dict(rocs_dict, orient='index')

    rocs_df['rank_1d'] = rocs_df['1d'].rank(pct=True)
    rocs_df['rank_1w'] = rocs_df['1w'].rank(pct=True)
    rocs_df['rank_1m'] = rocs_df['1m'].rank(pct=True)

    rocs_df = rocs_df.drop(['1d', '1w', '1m'], axis=1)

    filepath = session.market_data_write / 'market_ranks.parquet'
    rocs_df.to_parquet(path=filepath)


def save_vols(vols):
    vol_path = Path('recent_1d_volumes.json')
    vol_path.touch(exist_ok=True)

    with open(vol_path, 'w') as file:
        json.dump(vols, file)


iterations(0, 'BTCUSDT', '1m')
iterations(1, 'ETHUSDT', '1m')

for n, pair in enumerate(pairs):
    # print(n, pair)
    try:
        df = iterations(n, pair, '5m')
    except Exception as e:
        print(f"*** {pair} exception during download, data not downloaded ***")
        print(e)
        continue

    rocs[pair] = {}
    rocs[pair]['1d'] = df.close.rolling(12).mean().pct_change(288).iloc[-1]
    rocs[pair]['1w'] = df.close.rolling(84).mean().pct_change(2016).iloc[-1]
    rocs[pair]['1m'] = df.close.rolling(360).mean().pct_change(8640).iloc[-1]

    volumes[pair] = df.quote_vol.sum()

mkt_rank(rocs)
save_vols(volumes)

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round(all_time // 60)}m {round(all_time % 60)}s'

print(f"used-weight-1m: {client.response.headers['x-mbx-used-weight-1m']}")
print(f'update_ohlc complete, {elapsed_str}')

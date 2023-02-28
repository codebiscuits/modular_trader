import pandas as pd
import polars as pl
import keys
import time
from datetime import datetime
import binance_funcs as funcs
from binance.client import Client
from pushbullet import Pushbullet
from pathlib import Path
from sessions import LightSession
from pyarrow import ArrowInvalid

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

now = datetime.now().strftime('%d/%m/%y %H:%M')

session = LightSession()

if session.live:
    print('-:-' * 10, f' {now} running update_ohlc ', '-:-' * 10)
else:
    print('*** Warning: Not Live ***')


start = time.perf_counter()

for sym in session.info['symbols']:
    if sym['status'] != 'TRADING':
        dead_symbol = sym['symbol']
        fp = Path(f"{session.ohlc_data}/{dead_symbol}.parquet")
        if fp.exists():
            fp.unlink()

pairs = list(session.pairs_data.keys())

def from_scratch(pair):
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
    #-------------------- if theres already some local data -------------------------#
    if filepath.exists():

        try:
            pldf = pl.read_parquet(source=filepath, use_pyarrow=True)
            df = pldf.to_pandas()
        except ArrowInvalid as e:
            print('Error:\n', e)
            print(f"Problem reading {pair} parquet file, downloading from scratch.")
            filepath.unlink()
            df = from_scratch(pair)



        # df = pd.read_parquet(filepath)

        if len(df) > 2:
            df = funcs.update_ohlc(pair, tf, df)
    # -------------------- if theres no local data yet -------------------------#
    else:
        df = from_scratch(pair)

    # print(pair, df.timestamp.iloc[-1])

    max_dict = {'1m': 1051200, '5m': 210240, '15m': 70080, '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        df = df.tail(max_len).reset_index(drop=True)

    # print(pair, df.timestamp.dtype)

    # df.to_parquet(filepath, compression=None)
    pldf = pl.from_pandas(df)
    pldf.write_parquet(filepath, row_group_size=10512, use_pyarrow=True)

    # print(f"{n} {pair} ohlc length: {len(df)}")


iterations(0, 'BTCUSDT', '1m')
iterations(1, 'ETHUSDT', '1m')

for n, pair in enumerate(pairs):
    try:
        iterations(n, pair, '5m')
    except:
        continue

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round(all_time // 60)}m {round(all_time % 60)}s'

print(f'update_ohlc complete, {elapsed_str}')

import pandas as pd
import keys
import time
from datetime import datetime
import binance_funcs as funcs
from binance.client import Client
from pushbullet import Pushbullet
from pathlib import Path
from sessions import LightSession

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


# def get_filepath(pair, tf):
#     folder = f"bin_ohlc_{tf}"
#     if live:
#         ohlc_data = Path('/media/coding/') / folder
#     else:
#         ohlc_data = Path('/home/ross/Documents/backtester_2021/') / folder
#     ohlc_data.mkdir(exist_ok=True)
#     return Path(f'{ohlc_data}/{pair}.parquet')


start = time.perf_counter()

pairs = list(session.pairs_data.keys())

# TODO i want to try running a function which records the roc of different timeframes for each pair in a separate file
#  for setup_scanner to use as a ranking metric (relative strength)

# TODO once i have a market cap column, i can make a 24h volume / market cap column

# TODO i also want really high timeframe stuff like 200 week sma and emas. it might be possible to download daily or
#  weekly ohlc data from exchange to get that, or i might be able to get those kind of metrics from coingecko. these
#  will need to be interpolated too

# TODO it would also be nice to have some on-chain metrics being included in this script as well but i will have to work
#  out where to get that data from


def iterations(n, pair, tf):
    # print(f"{n} {pair} {tf}")
    session.set_ohlc_tf(tf)
    # print(session.ohlc_data)
    filepath = Path(f'{session.ohlc_data}/{pair}.parquet')
    # print(filepath)
    #-------------------- if theres already some local data -------------------------#
    if filepath.exists():
        df = pd.read_parquet(filepath)
        if len(df) > 2:
            df = funcs.update_ohlc(pair, tf, df)
    # -------------------- if theres no local data yet -------------------------#
    else:
        df_start = time.perf_counter()
        df = funcs.get_ohlc(pair, tf, '2 years ago UTC')
        df_end = time.perf_counter()
        elapsed = df_end - df_start
        print(f'downloaded {pair} from scratch, took {int(elapsed // 60)}m {elapsed % 60:.1f}s')

    # print(pair, df.timestamp.iloc[-1])

    max_dict = {'1m': 1051200, '5m': 210240, '15m': 70080, '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        df = df.tail(max_len).reset_index(drop=True)
    df.to_parquet(filepath)
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

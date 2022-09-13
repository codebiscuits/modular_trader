import pandas as pd
import keys
import time
import binance_funcs as funcs
from binance.client import Client
from pushbullet import Pushbullet
from pathlib import Path
from config import not_pairs

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

# check to see if this is running on the raspberry pi or not
pi2path = Path('/home/ubuntu/rpi_2.txt')
if live := pi2path.exists():
    print('-:-' * 10, ' running update_ohlc ', '-:-' * 10)
else:
    print('*** Warning: Not Live ***')

start = time.perf_counter()

pairs = funcs.get_pairs()
good_pairs = [pair for pair in pairs if pair not in not_pairs]


def iterations(pair, tf):
    if live:
        ohlc_data = Path(f'/media/coding/ohlc_binance_{tf}')
    else:
        ohlc_data = Path(f'/home/ross/Documents/backtester_2021/bin_ohlc_{tf}')
    ohlc_data.mkdir(exist_ok=True)
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        if len(df) > 2:
            df = df.iloc[:-1, :]
            df = funcs.update_ohlc(pair, tf, df)

    else:
        df_start = time.perf_counter()
        df = funcs.get_ohlc(pair, tf, '1 year ago UTC')
        df_end = time.perf_counter()
        elapsed = df_end - df_start
        print(f'downloaded {pair} from scratch, took {elapsed % 60:.1f}s')

    max_dict = {'1m': 1051200,
                '15m': 70080,
                '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        df = df.tail(max_len)
        df.reset_index(drop=True, inplace=True)
    df.to_pickle(filepath)
    # print(f"{pair} ohlc length: {len(df)}")


iterations('BTCUSDT', '1m')

for pair in pairs:
    if pair in not_pairs:
        continue

    iterations(pair, '15m')

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round(all_time // 60)}m {round(all_time % 60)}s'

print(f'update_ohlc complete, {elapsed_str}')

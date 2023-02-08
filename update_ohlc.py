import pandas as pd
import keys
import time
from datetime import datetime
import binance_funcs as funcs
from binance.client import Client
from pushbullet import Pushbullet
from pathlib import Path
from config import not_pairs

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# TODO after migrating to polars, i want to incorporate market cap data from coingecko/coinmarketcap

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

now = datetime.now().strftime('%d/%m/%y %H:%M')

# check to see if this is running on the raspberry pi or not
pi2path = Path('/home/ubuntu/rpi_2.txt')
if live := pi2path.exists():
    print('-:-' * 10, f' {now} running update_ohlc ', '-:-' * 10)
else:
    print('*** Warning: Not Live ***')

start = time.perf_counter()

pairs = funcs.get_pairs()


def iterations(n, pair, tf):
    folder = f"bin_ohlc_{tf}"
    if live:
        ohlc_data = Path('/media/coding/') / folder
    else:
        ohlc_data = Path('/home/ross/Documents/backtester_2021/') / folder
    ohlc_data.mkdir(exist_ok=True)
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        if len(df) > 2:
            df = funcs.update_ohlc(pair, tf, df)

    else:
        df_start = time.perf_counter()
        df = funcs.get_ohlc(pair, tf, '2 years ago UTC')
        df_end = time.perf_counter()
        elapsed = df_end - df_start
        print(f'downloaded {pair} from scratch, took {int(elapsed // 60)}m {elapsed % 60:.1f}s')

    # print(pair, df.timestamp.iloc[-1])

    max_dict = {'1m': 1051200,
                '5m': 210240,
                '15m': 70080,
                '1h': 17520}
    max_len = max_dict[tf]  # returns 2 years worth of timeframe periods
    if len(df) > max_len:
        df = df.tail(max_len).reset_index(drop=True)
    df.to_pickle(filepath)
    print(f"{n} {pair} ohlc length: {len(df)}")


iterations(0, 'BTCUSDT', '1m')
iterations(1, 'ETHUSDT', '1m')

for n, pair in enumerate(pairs):
    iterations(n, pair, '5m')
    iterations(n, pair, '15m')
    # iterations(n, pair, '1h')

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round(all_time // 60)}m {round(all_time % 60)}s'

print(f'update_ohlc complete, {elapsed_str}')

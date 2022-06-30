import pandas as pd
import keys, time
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
live = pi2path.exists()

if live:
    print('-:-' * 10, ' running update_ohlc ', '-:-' * 10)
    ohlc_data = Path('/media/coding/ohlc_binance_1h')
else:
    print('*** Warning: Not Live ***')
    ohlc_data = Path('/home/ross/Documents/backtester_2021/bin_ohlc')

start = time.perf_counter()

pairs = funcs.get_pairs()

for pair in pairs:
    if not pair in not_pairs:
        filepath = Path(f'{ohlc_data}/{pair}.pkl')
        if filepath.exists():
            df = pd.read_pickle(filepath)
            if len(df) > 2:
                df = df.iloc[:-1, :]
                df = funcs.update_ohlc(pair, '1h', df)
            
        else:
            df = funcs.get_ohlc(pair, '1h', '1 year ago UTC')
            print(f'downloaded {pair} from scratch')
        
        max_len = 17520
        if len(df) > max_len:  # 17520 is 2 year's worth of 1h periods
            df = df.tail(max_len)
            df.reset_index(drop=True, inplace=True)
        df.to_pickle(filepath)
        print(f"{pair} ohlc length: {len(df)}")

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s'
        
print(f'update_ohlc complete, {elapsed_str}')        

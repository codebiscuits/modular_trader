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
else:
    print('*** Warning: Not Live ***')

start = time.perf_counter()

pairs = funcs.get_pairs()

for pair in pairs:
    if not pair in not_pairs:
        df = funcs.prepare_ohlc(pair, live)

end = time.perf_counter()
all_time = end - start
elapsed_str = f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s'
        
print(f'update_ohlc complete, {elapsed_str}')        

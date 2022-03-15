import json
from json.decoder import JSONDecodeError
from pathlib import Path
from config import market_data
from pprint import pprint
import utility_funcs as uf

strat = 'double_st_lo' # 'rsi_st_ema'

pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()

c_path = Path(f'{market_data}/{strat}_closed_trades.json')

with open(c_path, 'r') as file:
    try:
        c_data = json.load(file)
        if c_data.keys():
            key_ints = [int(x) for x in c_data.keys()]
            next_id = sorted(key_ints)[-1] + 1
        else:
            next_id = 0
    except JSONDecodeError:
        c_data = []
        next_id = 0
        print('no closed trades yet')

# work out which records have discrepancies
bad_keys = uf.find_bad_keys(c_data)

for b in bad_keys:
    print(b)

bad_key = '1644508866339'

pprint(c_data.get(bad_key))

# c_data[bad_key][-1]['base_size'] = 72.2

# pprint(c_data.get(bad_key))

# c_data[bad_key][-1]['quote_size'] = 136.5302

# pprint(c_data.get(bad_key))
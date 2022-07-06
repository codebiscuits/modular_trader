import json
from pathlib import Path
from pprint import pprint
from statistics import mean
import matplotlib.pyplot as plt
from binance.client import Client
import keys

client = Client(keys.bPkey, keys.bSkey)

mkt_data_path = Path("/mnt/pi_2/market_data")
folders = list(mkt_data_path.glob('*'))

def inspect_data(data_path):
    with open(data_path, 'r') as file:
        return file.readlines()

def split_name(name):
    strats = ['double_st', 'ema_cross', 'ema_cross_hma']
    for strat_name in strats:
        if strat_name in name:
            strat = strat_name
    parts = name.split('_')
    
    tf = parts[-4]
    offset = parts[-3]
    var1 = parts[-2]
    var2 = parts[-1]
    
    return strat, tf, offset, var1, var2

#%%

pair = None

for folder in folders:
    if not folder.is_dir():
        continue
    data_path = mkt_data_path / folder.name / Path('open_trades.json')
    try:
        records = inspect_data(data_path)
    except FileNotFoundError:
        continue
    if records:
        print(folder.name)
        # strat, tf, offset, var1, var2 = split_name(folder.name)
        # for record in records:
        #     rec = json.loads(record)
        #     if pair:
        #         pprint(rec[pair])
        #     else:
        #         pprint(rec)

#%%

pair = ''

for folder in folders:
    if not folder.is_dir():
        continue
    data_path = mkt_data_path / folder.name / Path('open_trades.json')
    try:
        records = inspect_data(data_path)
    except FileNotFoundError:
        continue
    if records:
        print(folder.name)
        strat, tf, offset, var1, var2 = split_name(folder.name)
        for record in records:
            rec = json.loads(record)
            if pair:
                for x in rec.values():
                    if x[0]['pair'] == pair:
                        pprint(x)
            else:
                pprint(rec)
                
                
                
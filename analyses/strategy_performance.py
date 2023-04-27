import pandas as pd
# from pandas import errors
# import keys
# import time
# from datetime import datetime
# import binance_funcs as funcs
# from binance.client import Client
# from pushbullet import Pushbullet
from pathlib import Path
# from pycoingecko import CoinGeckoAPI
# import update_ohlc as uo
from pprint import pprint
import json
# from timers import Timer
# import sys
# import sessions
import plotly.express as px
import statistics as stats

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# client = Client(keys.bPkey, keys.bSkey)

records_path = Path('/home/ross/coding/pi_down/modular_trader/records')

def get_data(data_from: str):

    ot_path = rp / f'{data_from}_trades.json'

    try:
        with open(ot_path, 'r') as file:
            ot = json.load(file)
    except json.JSONDecodeError:
        # print('decode error')
        ot = None
    except FileNotFoundError:
        # print(f"file not found")
        ot = None

    return ot


def analyse(trades):

    all_r = []
    all_d = []

    for t in trades.values():
        # pprint(t)

        bought = float(t[0]['quote_size'])
        sold = float(t[1]['quote_size'])
        if 'long' in t[0]['type']:
            profit = sold - bought
        elif 'short' in t[0]['type']:
            profit = bought - sold
        all_d.append(profit)

        for x in t:
            rpnl = x.get('rpnl')
            if rpnl is not None:
                all_r.append(rpnl)

    return all_d, all_r


def print_results(title, all_d, all_r):
    wins = len(list(filter(lambda x: x > 0, all_d)))
    winrate = wins / len(all_d)

    print(title)
    print(f"Total rpnl: ${sum(all_d):.2f} / {sum(all_r):.2f}R from {len(trades)} trades, win rate: {winrate:.2%}")
    print(f"Max: {max(all_r)}R, min: {min(all_r)}R, mean: {stats.mean(all_r):.2f}R, median: {stats.median(all_r):.2f}R")
    print(
        f"Max: ${max(all_d):.2f}, min: ${min(all_d):.2f}, mean: ${stats.mean(all_d):.2f}, median: ${stats.median(all_d):.2f}")


for rp in records_path.glob('*'):
    if not rp.is_dir():
        continue

    real = False
    sim = False

    trades = get_data('closed')
    if trades:
        real = True
        real_d, real_r = analyse(trades)

    trades = get_data('closed_sim')
    if trades:
        sim = True
        sim_d, sim_r = analyse(trades)

    if real or sim:
        print(f"\n{rp}\n")
        if real:
            print_results('Real Trades', real_d, real_r)
        if sim:
            print_results('Sim Trades', sim_d, sim_r)
        print('')

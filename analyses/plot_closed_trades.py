from binance.client import Client
from binance.exceptions import BinanceAPIException
import binance_funcs as funcs
from config import ohlc_data, market_data
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import indicators as ind
import time, json, keys
from datetime import datetime as dt
from datetime import timedelta as delt

client = Client(keys.bPkey, keys.bSkey)

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

tf = '4H'
lb = 3
mult = 1.5
strat = 'double_st_lo'

def get_trades(market_data):
    c_path = Path(f'{market_data}/{strat}_closed_trades.json')

    with open(c_path, 'r') as file:
        return json.load(file)

def get_ohlc(pair, start, end):
    klines = client.get_historical_klines(symbol=pair, 
                                          interval=Client.KLINE_INTERVAL_5MINUTE, 
                                          start_str = start, 
                                          end_str=end, 
                                          limit=1000)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
            'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    return df

def prepare_data(df, pair, start, end, lb, mult, timeframe='4H') -> pd.DataFrame:
    # copy data and resample copy
    df2 = df.copy()
    df2 = df2.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                       'high': 'max',
                                                       'low': 'min', 
                                                       'close': 'last', 
                                                       'volume': 'sum'})
    # calculate supertrend at higher timeframe
    df2.reset_index(inplace=True)
    ind.supertrend_new(df2, 10, 3)
    df2.rename(columns={'st': 'st_loose', 'st_u': 'st_loose_u', 'st_d': 'st_loose_d'}, inplace=True)
    ind.supertrend_new(df2, lb, mult)
    df2['ema200'] = df2.close.ewm(200).mean()
    df2 = df2.loc[:, ['timestamp', 'st_u', 'st_d', 'st_loose_u', 'st_loose_d', 'ema200']]
    
    # # resample back to lower timeframe
    df2.set_index('timestamp', drop=True, inplace=True)
    # df2 = df2.resample('5T').asfreq().pad()
    
    # join them back together and trim the excess data
    df.set_index('timestamp', drop=True, inplace=True)
    # TODO having real trouble joining the two dataframes properly
    df = df.loc[df.index > start]
    df2 = df2.loc[df2.index > start]
    
    return df, df2


####################################################


trades = get_trades(market_data)
extend_long = delt(days=17)
extend = delt(days=2)


if trades:
    trade_keys = list(trades.keys())
    # pprint(c_data)
    for x, k in enumerate(trade_keys[-50:]):
        trade = trades.get(k)
        trade_add = None
        trade_tp = None
        for n, i in enumerate(trade):
            if i.get('type') in ['open_long', 'open_short']:
                trade_entry = trade[n]
            elif i.get('type') in ['add_long', 'add_short']:
                trade_add = trade[n]
                # print('TODO need to add calculations for adding and tping now')
            elif i.get('type') in ['tp_long', 'tp_short']:
                trade_tp = trade[n]
                # print('TODO need to add calculations for adding and tping now')
            elif i.get('type') in ['close_long', 'close_short', 
                                   'stop_long', 'stop_short']:
                trade_exit = trade[n]    
    
        ################################################
    
        pair = trade_entry.get('pair')
        
        start = dt.fromtimestamp(int(trade_entry.get('timestamp')) / 1000)
        start_stamp = int(trade_entry.get('timestamp')) / 1000
        entry_price = float(trade_entry.get('exe_price'))
        init_stop = trade_entry.get('hard_stop')
        r = 100 * (entry_price-init_stop) / init_stop
        
        if trade_add:
            add_time = dt.fromtimestamp(int(trade_add.get('timestamp')) / 1000)
            add_price = float(trade_add.get('exe_price'))
        
        if trade_tp:
            tp_time = dt.fromtimestamp(int(trade_tp.get('timestamp')) / 1000)
            tp_price = float(trade_tp.get('exe_price'))
        
        end = dt.fromtimestamp(int(trade_exit.get('timestamp')) / 1000)
        end_stamp = int(trade_exit.get('timestamp')) / 1000
        exit_price = float(trade_exit.get('exe_price'))
        exit_type = i.get('type')
        
        ################################################
        
        df = get_ohlc(pair, str(start - extend_long), str(end + extend))
        df, df2 = prepare_data(df, pair, start - extend, end + extend, lb, mult)
        if len(df) == 0:
            continue
        
        trade_entry = trade[0].get('exe_price')
        
        # no_display = ['timestamp', 'open', 'high', 'low', 'volume']
        # print(df.drop(no_display, axis=1).head())
        # print(df.drop(no_display, axis=1).tail())
        
        trade_line = 'g--' if exit_price > entry_price else 'r--'
        
        duration = round((end_stamp - start_stamp) / 3600, 1)
        
        ent_str = f'entry: {entry_price:.4},  '
        stop_str = f'stop: {init_stop:.4},  '
        exit_str = f'exit: {exit_price:.4}'
        price_str = ent_str + stop_str
        if trade_add:
            add_str = f'add: {add_price:.4},  '
            price_str += add_str
        if trade_tp:
            tp_str = f'tp: {tp_price:.4},  '
            price_str += tp_str
        price_str += exit_str
    
        plt.plot(df.open, linewidth=1)
        plt.plot(df.low, linewidth=1, linestyle='dashed')
        plt.plot(df2.st_loose_u, color='g', linewidth=1)
        plt.plot(df2.st_loose_d, color='r', linewidth=1)
        plt.plot(df2.st_u, color='g', linewidth=1)
        plt.plot(df2.st_d, color='r', linewidth=1)
        plt.plot(df2.ema200, color='b', linewidth=1)
        plt.scatter(start, entry_price, s=200, color='g', marker='.')
        plt.scatter(end, exit_price, s=200, color='r', marker='.')
        plt.plot([start, end], [entry_price, exit_price], trade_line, linewidth=1)
        if trade_add:
            plt.scatter(add_time, add_price, s=200, color='g', marker='.')
        if trade_tp:
            plt.scatter(tp_time, tp_price, s=200, color='r', marker='.')
        plt.title(f'{x+1} - {pair}  |  {duration} hours  |  {price_str}')
        plt.show()
        
        

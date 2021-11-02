'''this script is an analysis where i can input a single set of settings 
and then plot the backtest for one pair at those settings, with buys and sells 
on the chart and pnl evolution underneath, so i can cycle through all pairs 
and see whats happening pair by pair, trade by trade. that might give me some 
insight into why so few pairs are producing 30+ trades in a year'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import keys
import talib
import time
import json
from datetime import datetime
from pathlib import Path
from rsi_optimising import get_pairs, get_ohlc, update_ohlc, get_supertrend, get_signals, get_results
from binance.client import Client
from pushbullet import Pushbullet

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


pairs = get_pairs('usdt')
not_pairs = ['GPBUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 'USDCUSDT', 'PAXUSDT', 'COCOSUSDT',
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

rsi_length = 4
oversold = 45
overbought = 96

max_length = 250

with open('positions.json', 'r') as read_pos:
    positions = json.load(read_pos)


# pairs = ['BTCUSDT']

quick_start = time.perf_counter()
short_pairs = []

# quicker check every 4 hours just for ema uptrend
dropped = 0
for pair in pairs:
    if pair in not_pairs:
        continue
    # get data
    filepath = Path(f'ohlc_data/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
    else:
        df = get_ohlc(pair, '4h', '35 days ago UTC')
    if len(df) <= 200:
        continue
    df['20ema'] = talib.EMA(df.close, 20)
    df['200ema'] = talib.EMA(df.close, 200)
    ema_ratio = df.at[len(df)-1, '20ema'] / df.at[len(df)-1, '200ema']
    if ema_ratio < 0.9:
        dropped += 1
        continue
    else:
        short_pairs.append(pair)
quick_end = time.perf_counter()
quick_time = quick_end - quick_start
print(f'Shortlist time taken: {round((quick_time) // 60)}m {(quick_time) % 60:.3}s')
print(f'pairs in shortlist: {len(short_pairs)}, pairs dropped: {dropped}')
print('\n----------------------------------\n')

# full check on shortlisted pairs that loops
for _ in range(50):    
    all_start = time.perf_counter()
    for pair in short_pairs:
        if pair in not_pairs:
            continue
        in_pos = positions.get(pair)
        # get data
        filepath = Path(f'ohlc_data/{pair}.pkl')
        if filepath.exists():
            df = pd.read_pickle(filepath)
            df = update_ohlc(pair, '4h', df)
        else:
            df = get_ohlc(pair, '4h', '35 days ago UTC')
        if len(df) <= 200:
            continue
        if len(df) > max_length:
            df = df.iloc[-1*max_length:,]
        
        # compute indicators
        df['20ema'] = talib.EMA(df.close, 20)
        df['200ema'] = talib.EMA(df.close, 200)
        if df.at[len(df)-1, '20ema'] < df.at[len(df)-1, '200ema']:
            continue
        df['st'], df['st_u'], df['st_d'] = get_supertrend(df.high, df.low, df.close, 10, 3)
        df['rsi'] = talib.RSI(df.close, rsi_length)
        hodl = df['close'].iloc[-1] / df['close'].iloc[0]
        
        # generate signals
        buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = get_signals(df, oversold, overbought)
        
        # calculate results
        pnl, pnl_list = get_results(df)
        pnl_bth = pnl / hodl
        if not pd.isna(df.at[len(df)-1, 'signals']):
            buy_sig = df.at[len(df)-1, 'signals'][:3] == 'buy'
            sell_sig = df.at[len(df)-1, 'signals'][:4] == 'sell'
            stop_sig = df.at[len(df)-1, 'signals'][:4] == 'stop'
            now = datetime.now().strftime('%d/%m/%y %H:%M')
            price = df.at[len(df)-1, 'close']
            
            if in_pos:
                if sell_sig:
                    positions['pair'] = 0
                    note = f"{now} sell {pair} @ {price}"
                    push = pb.push_note(now, note)
                    print(note)
                elif stop_sig:
                    positions['pair'] = 0
                    note = f"{now} sell (stop) {pair} @ {price}"
                    push = pb.push_note(now, note)
                    print(note)
            else:
                if buy_sig:
                    msg = df.at[len(df)-1, 'signals'][5:24]
                    positions['pair'] = 1
                    note = f"{now} buy {pair} @ {price}, {msg}"
                    push = pb.push_note(now, note)
                    print(note)
            
                
        
        df = df.iloc[:-1,]
        df.to_pickle(filepath)
       
        # # plot results
        # df = df.iloc[1:,]    
        # df.set_index('timestamp', inplace=True)
        # ind = df[['st_u', 'st_d', '20ema', '200ema']]
        # apds = [mpf.make_addplot(ind)]
        # if df['s_buy'].any():
        #     apds.append(mpf.make_addplot(df['s_buy'], type='scatter', 
        #                                  color='g', markersize=200, marker='.'))
        # if df['s_sell'].any():
        #     apds.append(mpf.make_addplot(df['s_sell'], type='scatter', 
        #                                  color='r', markersize=200, marker='.'))
        # if df['s_stop'].any():
        #     apds.append(mpf.make_addplot(df['s_stop'], type='scatter', 
        #                                  color='b', markersize=200, marker='.'))
        # mpf.plot(df, title=f'{pair} - {rsi_length}, {oversold}, {overbought}',
        #     addplot=apds, figscale=2, figratio=(2, 1), 
        #     tight_layout=True,
        #     warn_too_much_data=2200)
    
    
    all_end = time.perf_counter()
    all_time = all_end - all_start
    print(f'Total time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')

with open('positions.json', 'w') as write_pos:
    json.dump(positions, write_pos)

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
import statistics as stats
import time
from pathlib import Path
from rsi_optimising import get_results
import binance_funcs as funcs
import indicators
import strategies as strats
from config import not_pairs, ohlc_data
from binance.client import Client

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

all_start = time.perf_counter()


pairs = funcs.get_pairs('USDT', 'SPOT')

rsi_length = 4
oversold = 45
overbought = 96

# pairs = ['BTCUSDT']

for pair in pairs:
    start = time.perf_counter()

    #get data
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    df = pd.read_pickle(filepath)
    if len(df) <= 200:
        continue
    if pair in not_pairs:
        continue
    
    # compute indicators
    
    if len(df) > 2190: # 2190 is 1 year's worth of 4h periods
        df = df.iloc[-2190:,]
        df.reset_index(drop=True, inplace=True)
    df['st'], df['st_u'], df['st_d'] = indicators.supertrend(df.high, df.low, df.close, 10, 3)
    df['20ema'] = talib.EMA(df.close, 20)
    df['200ema'] = talib.EMA(df.close, 200)
    df['rsi'] = talib.RSI(df.close, rsi_length)
    df = df.iloc[200:,]
    df.reset_index(drop=True, inplace=True)
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    # generate signals
    buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = strats.get_signals(df, oversold, overbought)
    
    # calculate results
    pnl, pnl_list, r_list = get_results(df)
    # print(f'pnl list: {pnl_list}')
    # print(df.drop(['open', 'high', 'low', 'volume', 'st_u', 'st_d', 
                    # '20ema', '200ema', 'rsi'], axis=1).tail(200))
    pnl_bth = pnl / hodl
    print(f'{pair} trades: {sells+stops}, pnl: {pnl:.1f}x, hodl returns: {hodl:.1f}x, avg r multiple: {stats.mean(r_list):.3}')

    # plot results
    # TODO i want the current plotting method to be one of two options. the other
    # option is to plot all individual trades on one chart, aligned by the timestamp
    # when they opened. it might also be useful to split the winners and losers 
    # into two charts

    df = df.iloc[1:,]
    
    df.set_index('timestamp', inplace=True)
    ind = df[['st_u', 'st_d', '20ema', '200ema']]
    apds = [mpf.make_addplot(ind), 
            mpf.make_addplot(df[['pnl_evo', 'hodl_evo']], panel=1, secondary_y=False), 
            mpf.make_addplot(df['s_buy'], type='scatter', color='g', markersize=200, marker='.'),
            mpf.make_addplot(df['s_stop'], type='scatter', color='b', markersize=200, marker='.'),
            ]
    # sometimes a backtest produces 0 sell signals. conditionally including sells
    # in the plot avoids errors when there are none.
    if df['s_sell'].any():
        apds.append(mpf.make_addplot(df['s_sell'], type='scatter', 
                                     color='r', markersize=200, marker='.'),
        )
    mpf.plot(df, title=f'{pair} - {rsi_length}, {oversold}, {overbought}',
        addplot=apds, figscale=2, figratio=(2, 1), 
        tight_layout=True,
        warn_too_much_data=2200)


all_end = time.perf_counter()
all_time = all_end - all_start
print(f'Total time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')

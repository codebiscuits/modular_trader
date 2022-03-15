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
import indicators as ind
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

# rsi_length = 6
# oversold = 55
# overbought = 84
lb = 10
mult = 3
tf = '1H'
    
def smrsi_st_ema_indicators(df, rsi_length):
    df['st'], df['st_u'], df['st_d'] = ind.supertrend(df.high, df.low, df.close, 10, 3)
    df['20ema'] = df.close.ewm(20).mean()
    df['200ema'] = df.close.ewm(200).mean()
    df['k_close'] = df.close.ewm(2).mean()
    df['rsi'] = talib.RSI(df.k_close, rsi_length)
    df = df.iloc[200:,]
    df.reset_index(drop=True, inplace=True)

def ha_st_lo_indicators(df, lb, mult):
    df = ind.heikin_ashi(df)
    df['ema200'] = df.close.ewm(200).mean()
    # df.drop(['20ema', '200ema'], axis=1, inplace=True)
    df['st'], df['st_u'], df['st_d'] = ind.supertrend(df.ha_high, df.ha_low, df.ha_close, lb, mult)
    df['st2'], df['st_u2'], df['st_d2'] = ind.supertrend(df.high, df.low, df.close, 1, 1)
    df['ratio'] = df.ha_close / df.st
    df['abs_ratio'] = (((df.ratio - 1) ** 2) ** 0.5) + 1
    df['mam_ratio'] = df.abs_ratio / (df.abs_ratio.rolling(100).mean())
    df = df.iloc[100:,]
    df.reset_index(drop=True, inplace=True)

pairs = ['ETHUSDT']

for pair in pairs:
    start = time.perf_counter()

    ### get data
    # filepath = Path(f'{ohlc_data}/{pair}.pkl')
    filepath = Path(f'bin_ohlc/{pair}.pkl')
    df = pd.read_pickle(filepath)
    if tf != '1H':
        funcs.resample(df, tf)
    
    if len(df) <= 200:
        continue
    if pair in not_pairs:
        continue
    
    if len(df) > 2190: # 2190 is 1 year's worth of 4h periods
        df = df.iloc[-2190:,]
        df.reset_index(drop=True, inplace=True)
    
    # # just test a short sample of recent data
    # df = df.tail(100)
    # df.reset_index(drop=True, inplace=True)
    
    ### generate indicators and signals
    
    # # smoothed rsi, supertrend, and EMAs
    # smrsi_st_ema_indicators(df, rsi_length)
    # buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = strats.get_signals(df, oversold, overbought)
    
    # heikin-ashi supertrend, long only
    ha_st_lo_indicators(df, lb, mult)
    counters, sig_data = strats.ha_st_lo_bt(df)
    
    buys, adds, sells, tps, stops = counters
    print(f'{buys = }, {adds = }, {sells = }, {tps = }, {stops = }')
    df['s_buy'], df['s_add'], df['s_sell'], df['s_tp'], df['s_stop'] = sig_data
    
    ### calculate results
    _, pnl_list, r_list = get_results(df)
    pnl = df.at[len(df)-1, 'pnl_evo']
    # print('pnl_evo', df.pnl_evo)
    
    # print(f'pnl list: {pnl_list}')
    # print(f'sorted r list: {sorted(r_list)}')
    # print(df.drop(['open', 'high', 'low', 'volume', 'st_u', 'st_d', 
                    # '20ema', '200ema', 'rsi'], axis=1).tail(200))
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    pnl_bth = pnl / hodl
    
    if pnl_list:
        print(f'{pair} trades: {sells+stops}, pnl: {pnl:.1f}x, hodl returns: {hodl:.1f}x, avg r multiple: {stats.median(r_list):.3}')

        ### plot results
        # TODO i want the current plotting method to be one of two options. the other
        # option is to plot all individual trades on one chart, aligned by the timestamp
        # when they opened. it might also be useful to split the winners and losers 
        # into two charts
    
        df = df.iloc[1:,]
        
        # df = df.tail(1650)
        # df = df.head(800)
        
        df.set_index('timestamp', inplace=True)
        # indis = df[['st_u', 'st_d', '20ema', '200ema']]
        indis = df[['st_u', 'st_d', 'st_u2', 'st_d2', 'ema200']]
        apds = [mpf.make_addplot(indis), 
                mpf.make_addplot(df[['pnl_evo', 'hodl_evo']], panel=1, secondary_y=False), 
                mpf.make_addplot(df[['in_pos']], panel=2, secondary_y=False), 
                mpf.make_addplot(df[['ratio']], color='g', panel=3, secondary_y=False), 
                mpf.make_addplot(df['s_buy'], type='scatter', color='g', markersize=200, marker='.'),
                # mpf.make_addplot(df['ha_close'], type='scatter', color='y', markersize=200, marker='_'),
                ]
        # sometimes a backtest produces 0 sell signals. conditionally including sells
        # in the plot avoids errors when there are none.
        if df['s_add'].any():
            apds.append(mpf.make_addplot(df['s_add'], type='scatter', color='aqua', markersize=200, marker='.'))
        if df['s_sell'].any():
            apds.append(mpf.make_addplot(df['s_sell'], type='scatter', color='r', markersize=200, marker='.'))
        if df['s_tp'].any():
            apds.append(mpf.make_addplot(df['s_tp'], type='scatter', color='#ff00c2', markersize=200, marker='.'))
        if df['s_stop'].any():    
            apds.append(mpf.make_addplot(df['s_stop'], type='scatter', color='b', markersize=200, marker='.'))
        mpf.plot(df, title=f'{pair} {tf} - st multiplier: {mult}',
            addplot=apds, figscale=2, figratio=(2, 1), 
            tight_layout=True,
            warn_too_much_data=2200)


all_end = time.perf_counter()
all_time = all_end - all_start
print(f'Total time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')

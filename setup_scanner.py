'''this script is an analysis where i can input a single set of settings 
and then plot the backtest for one pair at those settings, with buys and sells 
on the chart and pnl evolution underneath, so i can cycle through all pairs 
and see whats happening pair by pair, trade by trade. that might give me some 
insight into why so few pairs are producing 30+ trades in a year'''

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import keys
import talib
import time
from pathlib import Path
from rsi_optimising import get_pairs, get_ohlc, update_ohlc, get_supertrend, get_signals, get_results
from binance.client import Client

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

all_start = time.perf_counter()


pairs = get_pairs('usdt')
not_pairs = ['GPBUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 'USDCUSDT', 'PAXUSDT', 'COCOSUSDT',
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

rsi_length = 4
oversold = 45
overbought = 96

max_length = 250


# pairs = ['BTCUSDT']

for pair in pairs:
    if pair in not_pairs:
        continue
    # get data
    filepath = Path(f'ohlc_data/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        df = update_ohlc(pair, '4h', df)
    else:
        df = get_ohlc(pair, '4h', '35 days ago UTC')
    if len(df) <= 200:
        continue
    print(len(df))
    if len(df) > max_length:
        df = df.iloc[-1*max_length:,]
    
    # compute indicators
    df['st'], df['st_u'], df['st_d'] = get_supertrend(df.high, df.low, df.close, 10, 3)
    df['20ema'] = talib.EMA(df.close, 20)
    df['200ema'] = talib.EMA(df.close, 200)
    df['rsi'] = talib.RSI(df.close, rsi_length)
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    
    # generate signals
    buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = get_signals(df, oversold, overbought)
    
    # calculate results
    pnl, pnl_list = get_results(df)
    print(df.drop(['open', 'high', 'low', 'volume', 'st_u', 'st_d', 
                    '20ema', '200ema', 'rsi'], axis=1).tail(1))
    pnl_bth = pnl / hodl
    print('---')
    print(f'{pair} trades: {sells+stops}, pnl: {pnl:.1f}x, hodl returns: {hodl:.1f}x, pnl compared to hodl: {pnl_bth:.1f}x')
    
    print('-' * 100)
    
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

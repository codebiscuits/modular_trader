import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import keys
import talib
import statistics as stats
import time
client = Client(keys.bPkey, keys.bSkey)
from pprint import pprint
import json
from pathlib import Path

all_start = time.perf_counter()

# functions
def get_pairs(quote):
    info = client.get_exchange_info()
    symbols = info.get('symbols')
    btc_pairs = []
    usdt_pairs = []
    for sym in symbols:
        if sym.get('symbol')[-3:] == 'BTC':
            btc_pairs.append(sym.get('symbol'))
        elif sym.get('symbol')[-4:] == 'USDT':
            usdt_pairs.append(sym.get('symbol'))
    
    if quote == 'usdt':
        return usdt_pairs
    elif quote == 'btc':
        return btc_pairs

def get_ohlc(pair, timeframe):
    client = Client(keys.bPkey, keys.bSkey)
    tf = {'1m': Client.KLINE_INTERVAL_1MINUTE, 
          '5m': Client.KLINE_INTERVAL_5MINUTE, 
          '15m': Client.KLINE_INTERVAL_15MINUTE,
          '30m': Client.KLINE_INTERVAL_30MINUTE,
          '1h': Client.KLINE_INTERVAL_1HOUR,
          '4h': Client.KLINE_INTERVAL_4HOUR,
          '6h': Client.KLINE_INTERVAL_6HOUR,
          '8h': Client.KLINE_INTERVAL_8HOUR,
          '12h': Client.KLINE_INTERVAL_12HOUR,
          '1d': Client.KLINE_INTERVAL_1DAY,
          '3d': Client.KLINE_INTERVAL_3DAY,
          '1w': Client.KLINE_INTERVAL_1WEEK,
          }
    klines = client.get_historical_klines(pair, tf.get(timeframe), "6 months ago UTC")
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time', 
                'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol', 
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    return df

def v_candles(df, v):
    df['vol_group'] = df['volume'].cumsum().floordiv(v)
    
    df_v = df.groupby('vol_group').agg(
        timestamp=pd.NamedAgg(column='timestamp', aggfunc='last'),
        open=pd.NamedAgg(column='open', aggfunc='first'),
        high=pd.NamedAgg(column='high', aggfunc=max),
        low=pd.NamedAgg(column='low', aggfunc=min),
        close=pd.NamedAgg(column='close', aggfunc='last')  
        )
    
    df_v.reset_index(drop=True, inplace=True)
    
    return df_v

def get_supertrend(high, low, close, lookback, multiplier):
    # ATR
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FILL DATAFRAME WITH ZEROS TO MAKE IT THE RIGHT SIZE
    
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    
    # FINAL UPPER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    # supertrend = supertrend.dropna()[1:]
    # supertrend.reset_index(drop=True, inplace=True)
    
    # ST UPTREND/DOWNTREND
    
    upt = [0]
    dt = [0]
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(1, len(supertrend)):
        # print('testing', close[i], supertrend.iloc[i, 0])
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    
    upt.index, dt.index = supertrend.index, supertrend.index
    
    return st, upt, dt
    
def st_dist(close, supertrend, upt, dt):
    # ST Multiple / ST Distance
    
    dist_u = [0]
    dist_d = [0]
    
    nonzero = 1 # this is updated with the most recent nonzero value of upt so
    # it can be used as a substitute when upt[j] == 0
    for j in range(1, len(supertrend)):
        if close[j] > supertrend.iloc[j]:
            if upt[j] == 0:
                dist_u.append(close[j] - upt[j] / nonzero)
            else:
                dist_u.append((close[j] - upt[j]) / upt[j])
                dist_d.append(np.nan)
                nonzero = upt[j]
        elif close[j] < supertrend.iloc[j]:
            dist_u.append(np.nan)
            dist_d.append(abs(close[j] - dt[j]) / dt[j])
        else:
            dist_u.append(np.nan)
            dist_d.append(np.nan)
    
    st_dist_u, st_dist_d = pd.Series(dist_u), pd.Series(dist_d)
    
    # st_mult_u = (close - upt) / upt
    # st_mult_d = (close - dt) / dt
    # st_dist_u = st_mult_u.abs()
    # st_dist_d = st_mult_d.abs()

    st_dist_u.index, st_dist_d.index = supertrend.index, supertrend.index
    
    
    
    return st_dist_u, st_dist_d

def volatility(df, lookback):
    df[f'volatil{lookback}'] = df['close'].rolling(lookback).stdev()

def get_signals(df, buy_thresh, sell_thresh):
    '''during an uptrend as defined by price being above 200ema, when price closes 
    above st line after previously closing below, record distance. if distance 
    subsequently prints a value between 0 and x% of recorded value, record 
    distance value again and trigger a buy.
    if distance subsequently prints a value greater than y% of new recorded value
    OR if price closes below st line, trigger a sell'''
    signals = [np.nan]
    buy_thresh_list = [0]
    init_dist = None
    exe_dist = None
    trade_ready = 0
    in_pos = 0
    buys = 0
    sells = 0
    stops = 0
    for i in range(1, len(df)):
        low_dist = (df.low[i] - df.st[i]) / df.st[i]
        trend_up = (df.loc[i, '20ema'] > df.loc[i, '200ema'])
        cross_up = (df.close[i] > df.st[i]) and(df.close[i-1] <= df.st[i-1])
        cross_down = (df.close[i] < df.st[i]) and (df.close[i-1] >= df.st[i-1])
        if cross_down and in_pos == 1:
            signals.append('stop')
            in_pos = 0
            stops += 1
            buy_thresh_list.append(buy_thresh_list[-1])
        elif trend_up and cross_up:
            signals.append(np.nan)
            init_dist = df.dist_u[i]
            trade_ready = 1
            buy_thresh_list.append(df.dist_u[i] * buy_thresh)
        elif trade_ready == 1:
            dist_trig_buy = (low_dist / init_dist) <= buy_thresh
            if in_pos == 0 and dist_trig_buy:
                signals.append('buy, init stop @ {df.st[i]}')
                exe_dist = df.dist_u[i]
                in_pos = 1
                trade_ready = 0
                buys += 1
                buy_thresh_list.append(buy_thresh_list[-1])
            else:
                signals.append(np.nan)
                buy_thresh_list.append(buy_thresh_list[-1])
        elif in_pos == 1:
            dist_trig_sell = (df.dist_u[i] / exe_dist) >= sell_thresh
            if dist_trig_sell:
                signals.append(f'sell {df.dist_u[i] / exe_dist:.3}')
                in_pos = 0
                sells += 1
                buy_thresh_list.append(buy_thresh_list[-1])
                init_dist = df.dist_u[i]
                trade_ready = 1
            else:
                signals.append(np.nan)
                buy_thresh_list.append(buy_thresh_list[-1])
        elif cross_down and trade_ready == 1:
            trade_ready = 0
            signals.append(np.nan)
            buy_thresh_list.append(buy_thresh_list[-1])
        else:
            signals.append(np.nan)
            buy_thresh_list.append(buy_thresh_list[-1])
    
    
    df['signals'] = signals
    df['buy_thresh'] = buy_thresh_list
    # print('buys:', buys, 'sells:', sells, 'stops:', stops)
    return buys, sells, stops

def get_results(df):
    results = df.loc[df['signals'].notna(), ['close', 'signals', 'st_u', 'buy_thresh']]
    results.reset_index(drop=True, inplace=True)
    
    exe_prices = []
    for s in range(len(results.signals)):
        exe_p = (1 + results.at[s, 'buy_thresh']) * results.at[s, 'st_u']
        if results.at[s, 'signals'] == 'buy':
            exe_prices.append(exe_p)
        else:
            exe_prices.append(results.at[s, 'close'])
    results['exe_price'] = exe_prices
    
    results['roc'] = results['exe_price'].pct_change()
    results['roc'] = results['roc'] - 0.0015 # subtract two * binance fees from 
    results['pnl'] = results['roc'] * 100
    results['adj'] = results['roc'] + 1 # each entry in this column will represent 
    # the proportional adjustment that trade makes to the account balance
    results = results[1::2] # if the strat is just 
    # opening and closing positions in full, every second signal will 
    # be a position close. they are the only ones im interested in
    results.reset_index(drop=True, inplace=True)
    results.drop('roc', axis=1, inplace=True)
    # print(results)
    bal = 1.0
    for a in results['adj']:
        bal *= a
    # print(f'final pnl: {bal:.4}')
    # med_pnl = results['pnl'].median()
    pnl_list = list(results['pnl'])
    return bal, pnl_list

# assign variables
# pair = 'SOLUSDT'
timeframe = '5m'
# agg_vol = 500
lookback = 10
multiplier = 3
comm = 0.00075

#TODO make it save to file after every pair, and record risk factor

pairs = get_pairs('usdt')
done_pairs = [x.stem for x in Path('results/').glob('*.*')]
not_pairs = ['TUSDUSDT', 'USDCUSDT', 'PAXUSDT']

for pair in pairs:
    if pair in done_pairs:
        continue
    if pair in not_pairs:
        continue
    # download data
    data = get_ohlc(pair, timeframe)
    if len(data) == 0:
        continue
    all_results = [data.volume.sum()]
    print('------------', pair, 'total volume:', data.volume.sum())
    print('num ohlc periods:', len(data))
    agg_div = len(data) * 20
    init_agg = data.volume.sum() / agg_div
    for agg_vol in [(av**2 * init_agg) for av in range(1, 31)]:
        start = time.perf_counter()
        df = v_candles(data, agg_vol)
        # compute indicators
        df['st'], df['st_u'], df['st_d'] = get_supertrend(df.high, df.low, df.close, lookback, multiplier)
        df['dist_u'], df['dist_d'] = st_dist(df.close, df.st, df.st_u, df.st_d)
        df['20ema'] = talib.EMA(df.close, 20)
        df['200ema'] = talib.EMA(df.close, 200)
        
        # backtest
        b_list = [x/10 for x in range(1, 10)]
        # avg_pnl_list = []
        
        for x in b_list:
            for y in [round(1 + (q**2 / 16), 3) for q in range(5)]:
                try:
                    buys, sells, stops = get_signals(df, x, y)
                    bal, pnl_list = get_results(df)
                    pnl = (bal - 1) * 100
                    num_trades = (sells + stops)
                    res_dict = {'pair': pair, 'agg vol': agg_vol, 
                                'buy_thr': x, 'sell_thr': y, 
                                'tot_pnl': pnl, 'pnl_list': pnl_list,
                                'buys': buys, 'sells': sells, 'stops': stops, 
                                'tot_volu': data.volume.sum(), 
                                'tot_stdev': stats.stdev(data.close), 
                                'ohlc_len': len(data), 
                                'v_cand_len': len(df)
                                }
                    all_results.append(res_dict)
                    # print(f'{x}-{y}: total profit {pnl:.3}%, buys {buys}, sells {sells}, stops {stops}')
                except:
                    continue        
        end = time.perf_counter()
        elapsed = f'{round((end - start) // 60)}m {(end - start) % 60:.3}s'
        print(f'{pair} {agg_vol:.1f} num candles: {len(df)}, time taken: {elapsed}')
    
    with open(f'vol_results/{pair}.txt', 'w') as outfile:
        json.dump(all_results, outfile)

all_end = time.perf_counter()
all_time = all_end - all_start
print(f'Total time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')

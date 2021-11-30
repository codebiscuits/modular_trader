import pandas as pd
import numpy as np
from binance.client import Client
import keys
import talib
import statistics as stats
import time
import json
from pathlib import Path
# from execution import binance_spreads
import binance_funcs as funcs
from multiprocessing import Pool
import indicators as ind
import strategies as strats
from config import not_pairs, ohlc_data, results_data


client = Client(keys.bPkey, keys.bSkey)

all_start = time.perf_counter()

def get_results(df):
    results = df.loc[df['signals'].notna(), ['timestamp', 'close', 'signals']]
    results.reset_index(drop=True, inplace=True)
    
    # TODO create a 'stop_loss' column at the point where these signals are 
    # generated instead of reconstructing it here
    sl = []
    for r in results['signals']:
        if r[:3] == 'buy':
            sl.append(float(r[17:]))
        else:
            sl.append(np.nan)
    results['stop-loss'] = sl
    
    results['r'] = (results['close'] - results['stop-loss']) / results['close']
    results.drop('stop-loss', axis=1)
    results['r'] = results['r'].shift(1)
    results['roc'] = results['close'].pct_change()
    results['roc'] = results['roc'] - 0.0015 # subtract two * binance fees from 
    results['pnl'] = results['roc'] * 100
    results['adj'] = results['roc'] + 1 # each entry in this column will represent 
    # the proportional adjustment that trade makes to the account balance
    results = results[1::2] # if the strat is just 
    # opening and closing positions in full, every second signal will 
    # be a position close. they are the only ones im interested in
    results.reset_index(drop=True, inplace=True)
    results['r_multiple'] = results['roc'] / results['r']
    results.drop('roc', axis=1, inplace=True)
    # print(results)
    bal = 1.0
    for a in results['adj']:
        bal *= a
    # print(f'final pnl: {bal:.4}')
    # med_pnl = results['pnl'].median()
    pnl_list = list(results['pnl'])
    r_list = list(results['r_multiple'])
    return bal, pnl_list, r_list

def mp_opt_old(pair, df, rsi_len, pars):
    os, ob = pars
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    buys, sells, stops, _, _, _ = strats.get_signals(df, x, y)
    bal, pnl_list, r_list = get_results(df)
    pnl = (bal - 1) * 100
    pnl_bth = pnl / hodl
    # calculate sqn here
    res_dict = {'pair': pair, 'rsi_len': rsi_len, 
                'rsi_os': x, 'rsi_ob': y, 
                'tot_pnl': pnl, 'pnl_bth': pnl_bth, 
                'pnl_list': pnl_list,# 'sqn': sqn, 
                'r_list': r_list, 
                'buys': buys, 'sells': sells, 'stops': stops, 
                'avg_volu': df.volume.mean(), 
                'tot_stdev': stats.stdev(df.close), 
                'ohlc_len': len(df), 
                'v_cand_len': len(df)
                }
    
    return res_dict

def mp_opt_new(pair, df, rsi_len, pars):
    os, ob = pars
    buys, sells, stops, _, _, _ = strats.get_signals(df, x, y)
    bal, pnl_list, r_list = get_results(df)
    pnl = (bal - 1) * 100
    avg_r = stats.mean(r_list)
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    avg_hodl = hodl / len(df.index)
    r_bth = avg_r / avg_hodl
    # calculate sqn here
    res_dict = {'pair': pair, 'rsi_len': rsi_len, 
                'rsi_os': x, 'rsi_ob': y, 
                'tot_pnl': pnl, 'r_bth': r_bth, 
                'pnl_list': pnl_list,# 'sqn': sqn, 
                'r_list': r_list, 
                'buys': buys, 'sells': sells, 'stops': stops, 
                'avg_volu': df.volume.mean(), 
                'tot_stdev': stats.stdev(df.close), 
                'ohlc_len': len(df), 
                'v_cand_len': len(df)
                }
    
    return res_dict

if __name__ == '__main__':
    
    # assign variables
    strat_name = 'smoothed_rsi_4h_mult-3_5'
    timeframe = '4h'
    lookback = 10
    multiplier = 3.5
    comm = 0.00075
    
    print(f'Backtest Optimising {strat_name}, st lookback: {lookback}')
    
    # TODO sort out all the paths to match setup_scanner
    results_folder = Path(f'rsi_st_ema/{strat_name}')
    res_path = results_data / results_folder
    res_path.mkdir(parents=True, exist_ok=True)
    
    #TODO make it record risk factor
    
    pairs_usdt = funcs.get_pairs('USDT')
    spreads_usdt = funcs.binance_spreads('USDT')
    pairs_u = [p for p in pairs_usdt if spreads_usdt.get(p) < 0.01]
    pairs_btc = funcs.get_pairs('BTC')
    spreads_btc = funcs.binance_spreads('BTC')
    pairs_b = [p for p in pairs_btc if spreads_btc.get(p) < 0.01]
    all_pairs = pairs_u + pairs_b
    done_pairs = [x.stem for x in res_path.glob('*.*')]
    bad_pairs = done_pairs + not_pairs
    pairs = [p for p in all_pairs if not p in bad_pairs]
    
    for pair in pairs:
        print(pair)
        ohlc_path = Path(f'{ohlc_data}/{pair}.pkl')
        # download data
        if ohlc_path.exists():
            df_full = pd.read_pickle(ohlc_path)
            print('ohlc path exists')
        else:
            continue
        if len(df_full) <= 200:
            continue
        print('length > 200')
        all_results = [df_full.volume.sum()]
        print(f'{pair} num ohlc periods: {len(df_full)}')
        for rsi_len in [3, 4, 5, 6, 7]:
            df = df_full.copy()
            start = time.perf_counter()
            # compute indicators
            df['st'], df['st_u'], df['st_d'] = ind.supertrend(df.high, df.low, df.close, lookback, multiplier)
            df['20ema'] = talib.EMA(df.close, 20)
            df['200ema'] = talib.EMA(df.close, 200)
            df['k_close'] = talib.EMA(df.close, 2)
            df['rsi'] = talib.RSI(df.k_close, rsi_len)
            # df['volatil20'] = df['close'].rolling(20).stdev()
            # df['50ma_volu'] =  df['volume'].rolling(50).mean()
            df = df.iloc[200:,]
            df.reset_index(drop=True, inplace=True)
            
            # backtest
            # avg_pnl_list = []
            
            os = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
            ob = [100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60]

            params = []
            for x in os:
                for y in ob:
                    params.append((x, y))
                    
            pair_list = [pair] * len(params)
            df_list = [df] * len(params)
            rsi_list = [rsi_len] * len(params)

            inputs = zip(pair_list, df_list, rsi_list, params) # pair, df, rsi_len, (os, ob)
            # try:
            print(f'multiprocessing begin, rsi_len: {rsi_len}')
            pool = Pool(processes=7)
            mp_results = pool.starmap(mp_opt_old, inputs)
            pool.close()
            pool.join()
            all_results.extend(mp_results)
            # print(f'{x}-{y}: total profit {pnl:.3}%, buys {buys}, sells {sells}, stops {stops}')
            # except:
            #     continue        
            end = time.perf_counter()
            elapsed = f'{round((end - start) // 60)}m {(end - start) % 60:.3}s'
            print(f'{pair}, rsi {rsi_len}, time taken: {elapsed}')
        
        # with open(f'{res_path}/{pair}.txt', 'w') as outfile:
        #     json.dump(all_results, outfile)
    
    all_end = time.perf_counter()
    all_time = all_end - all_start
    print(f'Total time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')

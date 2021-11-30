import pandas as pd
import numpy as np
import json
import statistics as stats
from pathlib import Path
import matplotlib.pyplot as plt
from config import results_data

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

folder = f'{results_data}/rsi_st_ema/smoothed_rsi_4h_mult-2_5'

quote = 'USDT'

usdt_results = []

all_pairs = [x.stem for x in Path(folder).glob('*.*')]
pairs = []

for p in all_pairs:
    if p[-1*len(quote):] == quote:
        pairs.append(p)

for pair in pairs:
    try:
        file = open(f'{folder}/{pair}.txt', 'r')
        data = json.load(file)
        usdt_results.extend(data[1:])
    except:
        print(f"skipping {pair}, didn't work")
        continue

df_full = pd.DataFrame(usdt_results)
# print(df_full.head().drop('pnl_list', axis=1))
# print(df.columns)
# print(df.head())

for l in [3, 4, 5, 6, 7]:
    rsi_length = l
    oversold = None
    overbought = None
    min_trades = 5
    min_total_trades = 500
    
    print('\n', '-:-' * 30)
    print(f'\nresults for rsi length {rsi_length}')
    
    df = df_full.copy()
    
    df.drop(df.index[df['pair'] == 'COCOSUSDT'], inplace=True) # some very weird price action
    
    df.drop(df.index[df['rsi_len'] != rsi_length], inplace=True)
    
    if oversold:
        df.drop(df.index[df['rsi_os'] != oversold], inplace=True)
    
    if overbought:
        df.drop(df.index[df['rsi_ob'] != overbought], inplace=True)
    
    df.drop(df.index[df['buys'] < min_trades], inplace=True)
    
    wins = df.loc[df['tot_pnl'] > 0]
    losses = df.loc[df['tot_pnl'] <= 0]
    
    df['pnl_var'] = df['pnl_list'].apply(stats.stdev)
    
    df['r_var'] = df['r_list'].apply(stats.stdev)
    
    df['exit_ratio'] = df['sells'] / df['stops']
    
    df['avg_pnl'] = df['pnl_list'].apply(stats.mean)
    
    df['avg_r'] = df['r_list'].apply(stats.median)
    
    df['num_trades'] = df['pnl_list'].apply(len)
    
    # df['win_rate'] = + values / - values in pnl_list
    
    # df.drop(df.index[df['tot_pnl'] > 1000], inplace=True)
    
    print(f'# of results after dropping rows: {len(df)}')
    
    pair_groups = df.groupby('pair')['avg_r'].median()
    print(pair_groups.sort_values(ascending=False).head())
    # print(f'pairs left: {len(pair_groups.index)}')
    print(f'avg r per trade: {pair_groups.median():.4}')
    
    # print('med tot pnl', df.tot_pnl.median())
    print(f'med avg r: {df.avg_r.median():.3}')
    
    print('max trades:', df.buys.max())
    
    # print(df)
    
    rsi_groups1 = df.groupby(['rsi_os', 'rsi_ob'])[['avg_r', 'r_var']].median()
    rsi_groups2 = df.groupby(['rsi_os', 'rsi_ob'])[['num_trades']].sum()
    group_size = df.groupby(['rsi_os', 'rsi_ob']).size().to_frame(name='count')
    rsi_groups = pd.concat([rsi_groups1, rsi_groups2, group_size], axis=1)
    rsi_groups.drop(rsi_groups.index[rsi_groups['avg_r'] < 0], inplace=True)
    rsi_groups.drop(rsi_groups.index[rsi_groups['num_trades'] < min_total_trades], inplace=True)
    rsi_groups['score'] = rsi_groups['avg_r'] / rsi_groups['r_var']
    
    print(rsi_groups.sort_values('avg_r', ascending=False).head())
    
    # heatmap
    rsi_2d = rsi_groups.reset_index().pivot(columns='rsi_os',index='rsi_ob',values='score')
    plt.imshow(rsi_2d)
    plt.xlabel('rsi oversold value')
    plt.ylabel('rsi overbought value')
    plt.xticks(ticks=range(len(rsi_2d.columns)), labels=rsi_2d.columns)
    plt.yticks(ticks=range(len(rsi_2d.index)), labels=rsi_2d.index)
    
    # histogram
    # plt.hist(wins.rsi_ob, bins=100, rwidth=0.5, color='g', align='left')
    # plt.hist(losses.rsi_ob, bins=100, rwidth=0.5, color='r')
    # # plt.xticks(df.pair, size = 8, rotation=90)
    
    # scatter plot
    # plt.scatter(df.exit_ratio, df.tot_pnl)
    
    plt.title(f'rsi length: {rsi_length}')
    plt.show()

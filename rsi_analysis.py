import pandas as pd
import numpy as np
import json
import statistics as stats
from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

folder = 'results/rsi_results_1h'

quote = 'USDT'
rsi_length = 3
oversold = 45
overbought = 96
min_trades = 30

usdt_results = []

all_pairs = [x.stem for x in Path(folder).glob('*.*')]
pairs = []

for p in all_pairs:
    if p[-1*len(quote):] == quote:
        pairs.append(p)

for pair in pairs:
    file = open(f'{folder}/{pair}.txt', 'r')
    data = json.load(file)
    usdt_results.extend(data[1:])

df = pd.DataFrame(usdt_results)

# print(df.columns)

# print(df.head())

print(f'results for rsi length {rsi_length}')

df.drop(df.index[df['pair'] == 'COCOSUSDT'], inplace=True)

df.drop(df.index[df['rsi_len'] != rsi_length], inplace=True)

# df.drop(df.index[df['rsi_os'] != oversold], inplace=True)

# df.drop(df.index[df['rsi_ob'] != overbought], inplace=True)

df.drop(df.index[df['buys'] < min_trades], inplace=True)

wins = df.loc[df['tot_pnl'] > 0]
losses = df.loc[df['tot_pnl'] <= 0]

df['var'] = df['pnl_list'].apply(stats.stdev)

df['exit_ratio'] = df['sells'] / df['stops']

df['avg_pnl'] = df['pnl_list'].apply(stats.mean)

df['num_trades'] = df['pnl_list'].apply(len)

# df['win_rate'] = + values / - values in pnl_list

# df.drop(df.index[df['tot_pnl'] > 1000], inplace=True)

print(f'results after dropping: {len(df)}')

pair_groups = df.groupby('pair')['avg_pnl'].median()
# print(pair_groups)
# print(f'pairs left: {len(pair_groups.index)}')
print(f'avg trade pnl: {pair_groups.median():.4}')

print('med tot pnl', df.tot_pnl.median())
print('med avg pnl', df.avg_pnl.median())

print('max trades', df.buys.max())

# print(df)

rsi_groups1 = df.groupby(['rsi_os', 'rsi_ob'])[['avg_pnl', 'var']].median()
rsi_groups2 = df.groupby(['rsi_os', 'rsi_ob'])[['num_trades']].sum()
group_size = df.groupby(['rsi_os', 'rsi_ob']).size().to_frame(name='count')
rsi_groups = pd.concat([rsi_groups1, rsi_groups2, group_size], axis=1)
rsi_groups.drop(rsi_groups.index[rsi_groups['avg_pnl'] < 0], inplace=True)
rsi_groups['score'] = rsi_groups['avg_pnl'] / rsi_groups['var']

print(rsi_groups.sort_values('score', ascending=False).head())

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

import pandas as pd
import json
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

btc_results = []
usdt_results = []

all_pairs = [x.stem for x in Path('rsi_results/').glob('*.*')]
btc_pairs = []
usdt_pairs = []

for p in all_pairs:
    if p[-3:] == 'BTC':
        btc_pairs.append(p)
    else:
        usdt_pairs.append(p)

for pair in btc_pairs:
    file = open(f'rsi_results/{pair}.txt', 'r')
    data = json.load(file)
    btc_results.extend(data[1:])

for pair in usdt_pairs:
    file = open(f'rsi_results/{pair}.txt', 'r')
    data = json.load(file)
    usdt_results.extend(data[1:])

btc_df = pd.DataFrame(btc_results)
usdt_df = pd.DataFrame(usdt_results)

print(f'len btc: {len(btc_df)}, len usdt: {len(usdt_df)}')

btc_df.drop(btc_df.index[btc_df['buys'] < 30], inplace=True)
usdt_df.drop(usdt_df.index[usdt_df['buys'] < 30], inplace=True)

print(f'len btc: {len(btc_df)}, len usdt: {len(usdt_df)}')

print(btc_df.tot_pnl.median(), usdt_df.tot_pnl.median())
# df.drop(df.index[df['buy_thr'] != 0.3], inplace=True)
# df.drop(df.index[df['sell_thr'] != 1.062], inplace=True)
# # print(df.columns)

# # calculate agg_vol bands
# min_agg = df['agg vol'].min()
# max_agg = df['agg vol'].max()
# b = 30
# s = max_agg / b**2
# bs = [x**2 * s for x in range(b)]
# bs.append(max_agg+1) # +1 so that value == max_agg will be incliuded in last band

# for i in range(b-1):
#     min, max = int(bs[i]), int(bs[i+1])
#     # print('band', i, ':', min, max)
#     sub_df = df[df['agg vol'] >= min]
#     sub_df = df[df['agg vol'] < max]
#     low_volatility = sub_df.drop(sub_df.index[sub_df['tot_stdev'] > sub_df['tot_stdev'].median()])
#     high_volatility = sub_df.drop(sub_df.index[sub_df['tot_stdev'] < sub_df['tot_stdev'].median()])
#     low_volume = sub_df.drop(sub_df.index[sub_df['tot_volu'] > sub_df['tot_volu'].median()])
#     high_volume = sub_df.drop(sub_df.index[sub_df['tot_volu'] < sub_df['tot_volu'].median()])
#     t = f'band {i}, agg_vol range: {min}-{max}, low volatility'
#     # print(t)
#     plt.hist(low_volatility['tot_pnl'], bins=100, histtype='step', color='r', label='low_volatility')
#     plt.hist(high_volatility['tot_pnl'], bins=100, histtype='step', color='g', label='high_volatility')
#     plt.hist(low_volume['tot_pnl'], bins=100, histtype='step', color='b', label='low_volume')
#     plt.hist(high_volume['tot_pnl'], bins=100, histtype='step', color='y', label='high_volume')
#     plt.title(t)
#     plt.legend()
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import not_pairs
import keys, time
from datetime import datetime
import binance_funcs as funcs
import math
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
client = Client(keys.bPkey, keys.bSkey)
now = datetime.now().strftime('%d/%m/%y %H:%M')
start = time.perf_counter()

pair = 'BTCUSDT'

roc_df = pd.DataFrame()
wick_df = pd.DataFrame()

all_pairs = funcs.get_pairs()
pairs = [p for p in all_pairs if p not in not_pairs]

for i, pair in enumerate(pairs):
    if i%10 == 0:
        print(f'{i} of {len(pairs)} done')
    # download 15min ohlc
    tf = '4h'
    span = '1 month'
    if tf == '15m':
        span_periods = {'6 months': 17368, '1 month': 2688, '1 week': 672, '1 day': 96}
        period_modulus = 16
    elif tf == '4h':
        span_periods = {'6 months': 1085, '1 month': 168, '1 week': 42, '1 day': 6}
        period_modulus = 6
    df = funcs.get_ohlc(pair, tf, f'{span} ago UTC')
    if len(df) < span_periods.get(span):
        # print(f'{len(df) = } span = {span_periods.get(span)}')
        continue
    # calculate 1 period roc and avg lower wick length
    df['group'] = df.index % period_modulus
    df['roc'] = df.open.pct_change() * 100
    df['wick'] = 100 * (df.low - df.open) / df.open
    df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)
    # print(df.head(1))
    df_groups = df.groupby(['group']).median()
    roc_df = roc_df.append(df_groups.roc)
    wick_df = wick_df.append(df_groups.wick)

roc_list = list(roc_df.median())
wick_list = list(wick_df.median())

x = np.arange(len(roc_list))
width = 0.2
fig, ax = plt.subplots()
roc_medians = ax.bar([y-width/2 for y in x], roc_list, width, label='roc medians')
wick_medians = ax.bar([y+width/2 for y in x], wick_list, width, label='wick medians')

ax.bar_label(roc_medians, padding=3)
ax.bar_label(wick_medians, padding=3)

ax.legend()
ax.set_title(f'avg open-close roc and lower wicks from {span} of {tf} data')

fig.tight_layout()

plt.show()

print(df.loc[df.index == 0, 'timestamp'])

t = time.perf_counter() - start
print(f'Time taken: {round((t) // 60)}m {round((t) % 60)}s')


import pandas as pd
import matplotlib.pyplot as plt
from config import not_pairs, ohlc_data
import time
from datetime import datetime
import statistics as stats
from pathlib import Path
from functions import indicators as ind, binance_funcs as funcs

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
start = time.perf_counter()

all_pairs = funcs.get_pairs()
pairs = [p for p in all_pairs if p not in not_pairs]

lows = []
highs = []
all_wicks = []
test = 0

for short in [x**2 for x in range(2, 7)]:
    for long in [x**2 for x in range(7, 11)]:
        test += 1
        for i, pair in enumerate(pairs):
            # get data
            filepath = Path(f'{ohlc_data}/{pair}.pkl')
            if filepath.exists():
                df = pd.read_pickle(filepath)
            if len(df) <= 2160:
                continue
            if len(df) > 2160:
                df = df.tail(2160)
                df.reset_index(inplace=True)
            
            # resample to 4h
            df = df.resample('4H', on='timestamp').agg({'open': 'first',
                                                             'high': 'max',
                                                             'low': 'min',
                                                             'close': 'last',
                                                             'volume': 'sum'})
            df.reset_index(inplace=True)
            
            # calculations
            df['snr'] = ind.signal_noise_ratio(df, 10)
            # snr not currently working
            df['momentum'] = df.close.ewm(short).mean() / df.close.ewm(long).mean()
            df['low_wick'] = 100 * (df.low - df.open) / df.open
            df['high_wick'] = 100 * (df.high - df.open) / df.open
            
            df_up = df[df.momentum > 1]
            lows.extend(df_up.low_wick)
            highs.extend(df_up.high_wick)
            all_wicks.extend(df_up.low_wick)
            all_wicks.extend(df_up.high_wick)
            
            longest = round(max(all_wicks), 1)
            all_mean = stats.mean(all_wicks)
            all_stdev = stats.stdev(all_wicks)
            q1_idx = int(len(all_wicks)*0.25)
            q3_idx = int(len(all_wicks)*0.75)
            q1 = round(all_wicks[q1_idx], 3)
            q3 = round(all_wicks[q3_idx], 3)


        print(f'Test {test}, {short}:{long}, mean: {all_mean:.3}, stdev: {all_stdev:.3}, q delta: {q1 + q3}')
        # print(f'Test {test}, {short}:{long}, mean: {all_mean:.3}, stdev: {all_stdev:.3}, q delta: {all_q1 + all_q3:.3}')
        
        # plt.hist(all_wicks, 40, (-20, 20), density=True)
        # plt.title(f'Test {test}, {short} | {long}')
        # plt.show()
        
t = time.perf_counter() - start
print(f'Time taken: {round((t) // 60)}m {round((t) % 60)}s')

import binance_funcs as funcs
from config import ohlc_data
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import indicators as ind
import time

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

tf = '4H'
lb = 10
mult = 3

def get_pairs() -> list:
    filepath = Path(f'{ohlc_data}')
    files = filepath.glob('*.pkl')
    return [x.stem for x in files]

def get_data(pair, length, timeframe='4H') -> pd.DataFrame:
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    df = pd.read_pickle(filepath)
    if len(df) > length: # 8760 is 1 year's worth of 1h periods
        df = df.tail(length)
        df.reset_index(drop=True, inplace=True)
    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                'high': 'max',
                                                'low': 'min', 
                                                'close': 'last', 
                                                'volume': 'sum'})
    df.reset_index(inplace=True)
    return df


####################################################

# TODO get it to pull the open and closed trades records and make two dictionaries
# with pairs as keys and trade start/end datetimes as values, then plot all trades
# and automatically fit the plots to the timeline of each trade

pairs = get_pairs()
# pairs = ['SLPUSDT']
# print(pairs)
for pair in pairs:
    df = get_data(pair, 100, tf)
    if len(df.index) < 25:
        continue
    ind.supertrend_new(df, 10, 3)
    
    no_display = ['timestamp', 'open', 'high', 'low', 'volume']
    # print(df.drop(no_display, axis=1).head())
    # print(df.drop(no_display, axis=1).tail())
    
    df.set_index('timestamp', inplace=True)
    
    apds = []
    if df['st_u'].any():
        apds.append(mpf.make_addplot(df[['st_u']], panel=0, secondary_y=False))
    if df['st_d'].any():
        apds.append(mpf.make_addplot(df[['st_d']], panel=0, secondary_y=False))
            
    mpf.plot(df, title=f'{pair} {tf} - supertrend: {lb},{mult}',
        addplot=apds, figscale=2, figratio=(2, 1), 
        tight_layout=True,
        warn_too_much_data=2200)        
        
        
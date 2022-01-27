import pandas as pd
import numpy as np

def atr(df, lb):
    df['tr1'] = df.high - df.low
    df['tr2'] = abs(df.high - df.close.shift(1))
    df['tr3'] = abs(df.low - df.close.shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(lb).mean()
    df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
    
def supertrend_new(df, lb, mult):
    atr(df, lb)
    
    df['hl_avg'] = (df.high + df.low) / 2
    df['upper_band'] = (df.hl_avg + mult * df.atr)#.dropna()
    df['lower_band'] = (df.hl_avg - mult * df.atr)#.dropna()
    df.drop(['hl_avg', 'atr'], axis=1, inplace=True)
    
    df['final_upper'] = 0
    df['final_lower'] = 0
    
    # i have to use for loops to calculate the next columns because other methods using conditional
    # statements work on only 1 row at a time, and these steps base the current value on the previous one
    for i in df.index:
        if i == 0:
            df.at[i, 'final_upper'] = 0
        elif (df.at[i, 'upper_band'] < df.at[i-1, 'final_upper']) | (df.at[i-1, 'close'] > df.at[i-1, 'final_upper']):
            df.at[i, 'final_upper'] = df.at[i, 'upper_band']
        else:
            df.at[i, 'final_upper'] = df.at[i-1, 'final_upper']
    
    for i in df.index:
        if i == 0:
            df.at[i, 'final_lower'] = 0
        elif (df.at[i, 'lower_band'] > df.at[i-1, 'final_lower']) | (df.at[i-1, 'close'] < df.at[i-1, 'final_lower']):
            df.at[i, 'final_lower'] = df.at[i, 'lower_band']
        else:
            df.at[i, 'final_lower'] = df.at[i-1, 'final_lower']
    
    df.drop(['upper_band', 'lower_band'], axis=1, inplace=True)
    
    df['st'] = 0
    
    for j in df.index:
        if j == 0:
            df.at[j, 'st'] = 0
        elif df.at[j-1, 'st'] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] < df.at[j, 'final_upper']:
            df.at[j, 'st'] = df.at[j, 'final_upper']
        elif df.at[j-1, 'st'] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] > df.at[j, 'final_upper']:
            df.at[j, 'st'] = df.at[j, 'final_lower']
        elif df.at[j-1, 'st'] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] > df.at[j, 'final_lower']:
            df.at[j, 'st'] = df.at[j, 'final_lower']
        elif df.at[j-1, 'st'] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] < df.at[j, 'final_lower']:
            df.at[j, 'st'] = df.at[j, 'final_upper']
        
    df.drop(['final_upper', 'final_lower'], axis=1, inplace=True)
    
    # this next step doesn't involve data from previous row in calculating current row,
    # so i can use np.where which is much faster than a for loop
    df['st_u'] = np.where(df.close >= df.st, df.st, np.nan)
    df['st_d'] = np.where(df.close < df.st, df.st, np.nan)
    
    
    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)

def supertrend(high, low, close, lookback, multiplier):
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
    
    try:
        for i in range(len(final_bands)):
            if i == 0:
                final_bands.at[i, 'upper'] = 0
            else:
                if (upper_band[i] < final_bands.at[i-1,'upper']) | (close[i-1] > final_bands.at[i-1,'upper']):
                    final_bands.at[i,'upper'] = upper_band[i]
                else:
                    final_bands.at[i,'upper'] = final_bands.at[i-1,'upper']
    except KeyError as e:
        print('key error', e)
        print(final_bands.head(20))
    
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

def heikin_ashi(df):
    df['ha_close'] = (df.open + df.high + df.low + df.close) / 4
    df['ha_open'] = (df.open + df.close) / 2
    df.ha_open = df.ha_open.shift(1)
    df['ha_high'] = df.loc[:, ['high', 'ha_close', 'ha_open']].max(axis=1)
    df['ha_low'] = df.loc[:, ['low', 'ha_close', 'ha_open']].min(axis=1)

    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
    
def k_candles(df):
    df['k_open'] = (df.open + df.open.shift(1)) / 2
    df['k_high'] = (df.high + df.high.shift(1)) / 2
    df['k_low'] = (df.low + df.low.shift(1)) / 2
    df['k_close'] = (df.close + df.close.shift(1)) / 2

    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def stochastic(data, lookback):
    pass
    
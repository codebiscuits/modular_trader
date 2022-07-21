import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, lb: int) -> None:
    '''calculates the average true range on an ohlc dataframe'''
    df['tr1'] = df.high - df.low
    df['tr2'] = abs(df.high - df.close.shift(1))
    df['tr3'] = abs(df.low - df.close.shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df[f'atr-{lb}'] = df['tr'].ewm(lb).mean()
    df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
    
def atr_bands(df: pd.DataFrame, lb: int, mult: float) -> None:
    '''calculates bands at a specified multiple of atr above and below price 
    on a given dataframe'''
    atr(df, lb)
    
    df['hl_avg'] = (df.high + df.low) / 2
    df[f'atr-{lb}-{mult}-upper'] = (df.hl_avg + mult * df[f'atr-{lb}'])
    df[f'atr-{lb}-{mult}-lower'] = (df.hl_avg - mult * df[f'atr-{lb}'])
    df.drop(['hl_avg', f'atr-{lb}'], axis=1, inplace=True)  

def supertrend_new(df: pd.DataFrame, lb: int, mult: float) -> None:
    '''calculates supertrend indicator and adds it to the input dataframe with the 
    column names following the format: st-{lb}-{mult}, st-{lb}-{mult}-up, st-{lb}-{mult}-dn'''
    atr(df, lb)
    
    df['hl_avg'] = (df.high + df.low) / 2
    df['upper_band'] = (df.hl_avg + mult * df[f'atr-{lb}'])
    df['lower_band'] = (df.hl_avg - mult * df[f'atr-{lb}'])
    df.drop(['hl_avg', f'atr-{lb}'], axis=1, inplace=True)
    
    df['final_upper'] = 0.0
    df['final_lower'] = 0.0
    
    # i have to use for loops to calculate the next columns because other methods using conditional
    # statements work on only 1 row at a time, and these steps base the current value on the previous one
    for i in df.index:
        if i == 0.0:
            df.at[i, 'final_upper'] = 0.0
        elif (df.at[i, 'upper_band'] < df.at[i-1, 'final_upper']) | (df.at[i-1, 'close'] > df.at[i-1, 'final_upper']):
            df.at[i, 'final_upper'] = df.at[i, 'upper_band']
        else:
            df.at[i, 'final_upper'] = df.at[i-1, 'final_upper']
    
    for i in df.index:
        if i == 0.0:
            df.at[i, 'final_lower'] = 0.0
        elif (df.at[i, 'lower_band'] > df.at[i-1, 'final_lower']) | (df.at[i-1, 'close'] < df.at[i-1, 'final_lower']):
            df.at[i, 'final_lower'] = df.at[i, 'lower_band']
        else:
            df.at[i, 'final_lower'] = df.at[i-1, 'final_lower']
    
    df.drop(['upper_band', 'lower_band'], axis=1, inplace=True)
    
    st = f"st-{lb}-{mult}"
    stu = f"st-{lb}-{mult}-up"
    std = f"st-{lb}-{mult}-dn"
    df[st] = 0.0
    
    for j in df.index:
        if j == 0.0:
            df.at[j, st] = 0.0
        elif df.at[j-1, st] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] < df.at[j, 'final_upper']:
            df.at[j, st] = df.at[j, 'final_upper']
        elif df.at[j-1, st] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] > df.at[j, 'final_upper']:
            df.at[j, st] = df.at[j, 'final_lower']
        elif df.at[j-1, st] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] > df.at[j, 'final_lower']:
            df.at[j, st] = df.at[j, 'final_lower']
        elif df.at[j-1, st] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] < df.at[j, 'final_lower']:
            df.at[j, st] = df.at[j, 'final_upper']
        
    df.drop(['final_upper', 'final_lower'], axis=1, inplace=True)
    
    # this next step doesn't involve data from previous row in calculating current row,
    # so i can use np.where which is much faster than a for loop
    df[stu] = np.where(df.close >= df[st], df[st], np.nan)
    df[std] = np.where(df.close < df[st], df[st], np.nan)
    
    
    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)

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

def signal_noise_ratio(df, periods):
    df['hh'] = df.high.rolling(periods).max()
    df['ll'] = df.low.rolling(periods).min()
    df['mid'] = (df.hh + df.ll) / 2
    df['signal'] = (df.hh - df.ll) / df.mid
    
    df['abs_range'] = df.high - df.low
    df['range_mid'] = (df.high + df.low) / 2
    df['pct_range'] = df.abs_range / df.range_mid
    df['noise'] = df.pct_range.rolling(periods).mean()
    df['snr'] = df.signal / df.noise
    
    df.drop(['hh', 'll', 'mid', 'signal', 'abs_range', 'noise'], axis=1, inplace=True)

def stochastic(data, lookback):
    pass

def wma(s: pd.Series, lb: int) -> pd.Series:
    '''calculates the weighted moving average on an input series'''
    return s.rolling(lb).apply(lambda x: ((np.arange(lb)+1)*x).sum()/(np.arange(lb)+1).sum(), raw=True)

def hma(s: pd.Series, lb: int) -> pd.Series:
    '''calculates the Hull moving average on an input series. typically applied to closing prices.
    actually shortens the lookback period by its own square root so that the indicator can be 
    calculated within the stated lookback period, because i usually want to truncate the ohlc data to 
    save resources, so this implementation is technically a shorter HMA than it seems'''
    lb = int(lb - (lb**0.5))
    return wma(wma(s, lb//2).multiply(2).sub(wma(s, lb)), int(np.sqrt(lb)))


def trend_score(df: pd.DataFrame, column: str) -> None:

    df['ema20'] = df[column].ewm(20).mean()
    df['ema20_up'] = (df['ema20'] > df['ema20'].shift(1)).astype(int)
    df['ema20_cross'] = (df[column] > df['ema20']).astype(int)

    df['ema50'] = df[column].ewm(50).mean()
    df['ema50_up'] = (df['ema50'] > df['ema50'].shift(1)).astype(int)
    df['ema50_cross'] = (df[column] > df['ema50']).astype(int)

    df['ema100'] = df[column].ewm(100).mean()
    df['ema100_up'] = (df['ema100'] > df['ema100'].shift(1)).astype(int)
    df['ema100_cross'] = (df[column] > df['ema100']).astype(int)

    df['ema200'] = df[column].ewm(200).mean()
    df['ema200_up'] = (df['ema200'] > df['ema200'].shift(1)).astype(int)
    df['ema200_cross'] = (df[column] > df['ema200']).astype(int)

    df['hma20'] = hma(df[column], 20)
    df['hma20_up'] = (df['hma20'] > df['hma20'].shift(1)).astype(int)
    df['hma20_cross'] = (df[column] > df['hma20']).astype(int)

    df['hma50'] = hma(df[column], 50)
    df['hma50_up'] = (df['hma50'] > df['hma50'].shift(1)).astype(int)
    df['hma50_cross'] = (df[column] > df['hma50']).astype(int)

    df['hma100'] = hma(df[column], 100)
    df['hma100_up'] = (df['hma100'] > df['hma100'].shift(1)).astype(int)
    df['hma100_cross'] = (df[column] > df['hma100']).astype(int)

    df['trend_score'] = (df['ema20_up'] + df['ema20_cross'] +
                         df['ema50_up'] + df['ema50_cross'] +
                         df['ema100_up'] + df['ema100_cross'] +
                         df['ema200_up'] + df['ema200_cross'] +
                         df['hma20_up'] + df['hma20_cross'] +
                         df['hma50_up'] + df['hma50_cross'] +
                         df['hma100_up'] + df['hma100_cross']) / 14

    df.drop(['ema20', 'ema20_up', 'ema20_cross',
             'ema50', 'ema50_up', 'ema50_cross',
             'ema100', 'ema100_up', 'ema100_cross',
             'ema200', 'ema200_up', 'ema200_cross',
             'hma20', 'hma20_up', 'hma20_cross',
             'hma50', 'hma50_up', 'hma50_cross',
             'hma100', 'hma100_up', 'hma100_cross'],
             axis=1, inplace=True)

def trend_score_new(df: pd.DataFrame, column: str) -> None:

    all = ['ema50', 'ema100', 'ema200',
           'sma50', 'sma100', 'sma200',
           'hma50', 'hma100', 'hma200']

    def indiv_ma_score(df: pd.DataFrame, column: str, ma: str):
        if 'ema' in ma:
            df[ma] = df[column].ewm(int(ma[3:])).mean()
        elif 'sma' in ma:
            df[ma] = df[column].rolling(int(ma[3:])).mean()
        elif 'hma' in ma:
            df[ma] = hma(df[column], int(ma[3:]))

        df[f"{ma}_up"] = (df[ma] > df[ma].shift(1)).astype(int)
        df[f"{ma}_cross"] = (df[column] > df[ma]).astype(int)
        df[f"{ma}_score"] = df[f"{ma}_up"] + df[f"{ma}_cross"] / 2
        df.drop([ma, f"{ma}_up", f"{ma}_cross"], axis=1, inplace=True)

    for i in all:
        indiv_ma_score(df, 'close', i)

    all_scores = [f"{x}_score" for x in all]

    df['trend_score'] = df[all_scores].sum(axis=1) / len(all_scores)
    df.drop(all_scores, axis=1, inplace=True)


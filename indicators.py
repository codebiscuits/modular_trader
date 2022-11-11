import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, lb: int) -> None:
    '''calculates the average true range on an ohlc dataframe'''
    df['tr1'] = df.high - df.low
    df['tr2'] = abs(df.high - df.close.shift(1))
    df['tr3'] = abs(df.low - df.close.shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df[f'atr-{lb}'] = df['tr'].ewm(lb).mean()
    df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)

    return df


def atr_bands(df: pd.DataFrame, lb: int, mult: float) -> None:
    '''calculates bands at a specified multiple of atr above and below price 
    on a given dataframe'''
    df = atr(df, lb)
    
    df['hl_avg'] = (df.high + df.low) / 2
    df[f'atr-{lb}-{mult}-upper'] = (df.hl_avg + mult * df[f'atr-{lb}'])
    df[f'atr-{lb}-{mult}-lower'] = (df.hl_avg - mult * df[f'atr-{lb}'])
    df.drop(['hl_avg', f'atr-{lb}'], axis=1, inplace=True)  


def supertrend(df: pd.DataFrame, lb: int, mult: float) -> None:
    '''calculates supertrend indicator and adds it to the input dataframe with the
        column names following the format: st-{lb}-{mult}, st-{lb}-{mult}-up, st-{lb}-{mult}-dn'''

    df = atr(df, lb)

    df['hl_avg'] = (df.high + df.low) / 2
    df['upper_band'] = (df.hl_avg + mult * df[f'atr-{lb}'])
    df['lower_band'] = (df.hl_avg - mult * df[f'atr-{lb}'])
    df.drop(['hl_avg', f'atr-{lb}'], axis=1, inplace=True)

    # i have to use iteration to calculate final_upper/lower because
    # each value is affected by the previously calculated value
    final_upper = []
    final_lower = []
    last_upper = 0
    last_lower = 0
    df['close_1'] = df['close'].shift(1)

    for row in df.itertuples(index=True):
        if row.index == 0:
            final_upper.append(0.0)
        elif (row.upper_band < last_upper) | (row.close_1 > last_upper):
            final_upper.append(row.upper_band)
            last_upper = row.upper_band
        else:
            final_upper.append(last_upper)

    for row in df.itertuples(index=True):
        if row.index == 0:
            final_lower.append(0.0)
        elif (row.lower_band > last_lower) | (row.close_1 < last_lower):
            final_lower.append(row.lower_band)
            last_lower = row.lower_band
        else:
            final_lower.append(last_lower)

    df['final_upper'] = final_upper
    df['final_lower'] = final_lower

    df = df.drop(['upper_band', 'lower_band', 'close_1'], axis=1)

    st_vals = []
    last_st = 0
    df['final_upper_1'] = df.final_upper.shift(1)
    df['final_lower_1'] = df.final_lower.shift(1)

    for row in df.itertuples(index=True):
        if row.index == 0.0:
            st_vals.append(0.0)
        elif last_st == row.final_upper_1 and row.close < row.final_upper:
            st_vals.append(row.final_upper)
            last_st = row.final_upper
        elif last_st == row.final_upper_1 and row.close > row.final_upper:
            st_vals.append(row.final_lower)
            last_st = row.final_lower
        elif last_st == row.final_lower_1 and row.close > row.final_lower:
            st_vals.append(row.final_lower)
            last_st = row.final_lower
        elif last_st == row.final_lower_1 and row.close < row.final_lower:
            st_vals.append(row.final_upper)
            last_st = row.final_upper
        else:
            st_vals.append(last_st)

    df = df.drop(['final_upper', 'final_lower', 'final_upper_1', 'final_lower_1'], axis=1)

    st = f"st-{lb}-{mult}"
    df[st] = st_vals

    stu = f"st-{lb}-{mult}-up"
    std = f"st-{lb}-{mult}-dn"

    # this next step doesn't involve data from previous row in calculating current row,
    # so i can use np.where which is much faster than a for loop
    df[stu] = np.where(df.close >= df[st], df[st], np.nan)
    df[std] = np.where(df.close < df[st], df[st], np.nan)

    try:
        df = df.drop(0)
        df = df.reset_index(drop=True)
    except KeyError as e:
        print(e)
        print(df.head())

    return df


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


def hidden_flow(df, lookback):
    df['hlc3'] = df[['high', 'low', 'close']].mean(axis=1)
    df['sma3'] = df.hlc3.rolling(lookback).mean()
    df['rolling_pricevol'] = (df.hlc3 * df.volume).rolling(lookback).sum()
    df['rolling_vol'] = df.volume.rolling(lookback).sum()
    df['vwap'] = df.rolling_pricevol / df.rolling_vol

    df['hidden_flow'] = df.vwap / df.sma3
    df['hf_avg'] = df.hidden_flow.rolling(lookback * 5).mean()
    df['hf_std'] = df.hidden_flow.rolling(lookback * 5).std()
    df['hf_upper'] = df.hf_avg + df.hf_std
    df['hf_lower'] = df.hf_avg - df.hf_std

    df['hidden_flow_hi'] = df.hidden_flow < df.hf_upper
    df['hidden_flow_lo'] = df.hidden_flow > df.hf_lower

    return df


def vwma(df, lookback):
    df['hlc3'] = df[['high', 'low', 'close']].mean(axis=1)
    df['rolling_pricevol'] = (df.hlc3 * df.volume).rolling(lookback).sum()
    df['rolling_vol'] = df.volume.rolling(lookback).sum()
    df['vwma'] = df.rolling_pricevol / df.rolling_vol

    return df


def williams_fractals(df, frac_width, atr_spacing=0):
    """calculates williams fractals either on the highs and lows or spaced according to average true range.
    if the spacing value is left at the default 0, no atr spacing will be implemented. if spacing is set to an integer
    above 0, the atr will be calculated with a lookback length equal to the spacing value, and the resulting atr values
    will then be multiplied by one tenth of the spacing value. eg if spacing is set to 5, a 5 period atr series will be
    calculated, and the fractals will be spaced 0.5*atr from the highs and lows of the ohlc candles
    frac_width determines how many candles are used to decide if the current candle is a local high/low, so a frac_width
    of five will look at the current candle, the two previous candles, and the two subsequent ones"""

    if atr_spacing:
        df = atr(df, atr_spacing)
        mult = atr_spacing / 10
        df['fractal_high'] = np.where(df.high == df.high.rolling(frac_width, center=True).max(),
                                      df.high + (mult * df[f'atr-{atr_spacing}']), np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(frac_width, center=True).min(),
                                     df.low - (mult * df[f'atr-{atr_spacing}']), np.nan)
    else:
        df['fractal_high'] = np.where(df.high == df.high.rolling(frac_width, center=True).max(), df.high, np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(frac_width, center=True).min(), df.low, np.nan)

    df['inval_high'] = df.fractal_high.interpolate('pad')
    df['inval_low'] = df.fractal_low.interpolate('pad')

    return df


def fractal_density(df, lookback, frac_width):
    """a way of detecting when price is trending based on how frequently williams fractals are printed, since they are
    much more common during choppy conditions and spaced further apart during trending conditions.
    the calculation is simply the total number of fractals (high+low) in a given lookback period divided by the lookback.
    i could modify it by working out how to normalise the output to a range of 0-1, but for now the range of possible
    values is from 0 to some number between 0 and 1, since the most fractals you could possibly have in any lookback
    period is going to be significantly less than the period itself and dependent on the frac_width parameter"""

    df = williams_fractals(df, frac_width)

    return (df.fractal_high.rolling(lookback).count() + df.fractal_low.rolling(lookback).count()) / lookback


def inside_bars(df):
    df['inside_bar'] = (df.high < df.high.shift(1)) & (df.low > df.low.shift(1))

    return df


def doji(df):
    span = df.high - df.low
    upper_wick = df.high - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df.low
    df['bullish_doji'] = lower_wick / span
    df['bearish_doji'] = upper_wick / span

    return df


def bull_bear_bar(df):
    df['bullish_bar'] = df.close > df.open
    df['bearish_bar'] = df.close < df.open

    return df


def trend_rate(df, z, bars, source):
    """returns True for any ohlc period which follows a strong trend as defined by the rate-of-change and bars params.
    if price has moved at least a certain percentage within the set number of bars, it meets the criteria"""

    df[f"roc_{bars}"] = df[f"{source}"].pct_change(bars)
    m = df[f"roc_{bars}"].abs().rolling(bars * 9).mean()
    s = df[f"roc_{bars}"].abs().rolling(bars * 9).std()
    df['thresh'] = m + (z * s)

    df['trend_up'] = df[f"roc_{bars}"] > df['thresh']
    df['trend_down'] = df[f"roc_{bars}"] < 0 - df['thresh']

    return df


def trend_consec_bars(df, bars):
    """returns True for any ohlc period which follows a strong trend as defined by a sequence of consecutive periods
    that all move in the same direction"""

    df['up_bar'] = df.close.pct_change() > 0
    df['dir_diff'] = df.up_bar.diff().cumsum()
    df['trend_consec'] = df.dir_diff.groupby(df.dir_diff).cumcount()
    df = df.drop(0)
    df = df.reset_index(drop=True)
    df.trend_consec = df.trend_consec.astype(int) + 1

    df['trend_up'] = df.up_bar & (df.trend_consec >= bars)
    df['trend_down'] = ~df.up_bar & (df.trend_consec >= bars)

    return df


def ema_breakout(df, length, lookback):
    df[f"ema_{length}"] = df.close.ewm(length).mean()
    df['ema_high'] = df[f"ema_{length}"].shift(1).rolling(lookback).max()
    df['ema_low'] = df[f"ema_{length}"].shift(1).rolling(lookback).min()

    df['ema_up'] = df[f"ema_{length}"] > df.ema_high
    df['ema_down'] = df[f"ema_{length}"] < df.ema_low

    return df


def ema_trend(df, length):
    lookback = max(1, int(length/100))

    df[f"ema_{length}"] = df.close.ewm(length).mean()

    df['ema_up'] = df[f"ema_{length}"] > df[f"ema_{length}"].shift(lookback)
    df['ema_down'] = df[f"ema_{length}"] < df[f"ema_{length}"].shift(lookback)

    return df


def vol_delta_div(df):
    df['roc'] = df.close.pct_change(1)
    # df['vol_delta'] = (2 * df.)


# def supertrend_new(df: pd.DataFrame, lb: int, mult: float) -> None:
#     '''calculates supertrend indicator and adds it to the input dataframe with the
#     column names following the format: st-{lb}-{mult}, st-{lb}-{mult}-up, st-{lb}-{mult}-dn'''
#     atr(df, lb)
#
#     df['hl_avg'] = (df.high + df.low) / 2
#     df['upper_band'] = (df.hl_avg + mult * df[f'atr-{lb}'])
#     df['lower_band'] = (df.hl_avg - mult * df[f'atr-{lb}'])
#     df.drop(['hl_avg', f'atr-{lb}'], axis=1, inplace=True)
#
#     df['final_upper'] = 0.0
#     df['final_lower'] = 0.0
#
#     # i have to use for loops to calculate the next columns because other methods using conditional
#     # statements work on only 1 row at a time, and these steps base the current value on the previous one
#     for i in df.index:
#         if i == 0.0:
#             df.at[i, 'final_upper'] = 0.0
#         elif (df.at[i, 'upper_band'] < df.at[i-1, 'final_upper']) | (df.at[i-1, 'close'] > df.at[i-1, 'final_upper']):
#             df.at[i, 'final_upper'] = df.at[i, 'upper_band']
#         else:
#             df.at[i, 'final_upper'] = df.at[i-1, 'final_upper']
#
#     for i in df.index:
#         if i == 0.0:
#             df.at[i, 'final_lower'] = 0.0
#         elif (df.at[i, 'lower_band'] > df.at[i-1, 'final_lower']) | (df.at[i-1, 'close'] < df.at[i-1, 'final_lower']):
#             df.at[i, 'final_lower'] = df.at[i, 'lower_band']
#         else:
#             df.at[i, 'final_lower'] = df.at[i-1, 'final_lower']
#
#     df.drop(['upper_band', 'lower_band'], axis=1, inplace=True)
#
#     st = f"st-{lb}-{mult}"
#     stu = f"st-{lb}-{mult}-up"
#     std = f"st-{lb}-{mult}-dn"
#     df[st] = 0.0
#
#     for j in df.index:
#         if j == 0.0:
#             df.at[j, st] = 0.0
#         elif df.at[j-1, st] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] < df.at[j, 'final_upper']:
#             df.at[j, st] = df.at[j, 'final_upper']
#         elif df.at[j-1, st] == df.at[j-1, 'final_upper'] and df.at[j, 'close'] > df.at[j, 'final_upper']:
#             df.at[j, st] = df.at[j, 'final_lower']
#         elif df.at[j-1, st] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] > df.at[j, 'final_lower']:
#             df.at[j, st] = df.at[j, 'final_lower']
#         elif df.at[j-1, st] == df.at[j-1, 'final_lower'] and df.at[j, 'close'] < df.at[j, 'final_lower']:
#             df.at[j, st] = df.at[j, 'final_upper']
#
#     df.drop(['final_upper', 'final_lower'], axis=1, inplace=True)
#
#     # this next step doesn't involve data from previous row in calculating current row,
#     # so i can use np.where which is much faster than a for loop
#     df[stu] = np.where(df.close >= df[st], df[st], np.nan)
#     df[std] = np.where(df.close < df[st], df[st], np.nan)
#
#     try:
#         df.drop(0, inplace=True)
#     except KeyError:
#         print(df.head())
#     df.reset_index(drop=True, inplace=True)



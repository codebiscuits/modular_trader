from resources import indicators as ind
import pandas as pd
import numpy as np

def add_feature(df, name, timeframe):
    feature_lookup = {
        'atr_5_pct': {'call': atr_pct, 'params': (df, 5)},
        'atr_10_pct': {'call': atr_pct, 'params': (df, 10)},
        'atr_25_pct': {'call': atr_pct, 'params': (df, 25)},
        'atr_50_pct': {'call': atr_pct, 'params': (df, 50)},
        'ats_z_25': {'call': ats_z, 'params': (df, 25)},
        'ats_z_50': {'call': ats_z, 'params': (df, 50)},
        'ats_z_100': {'call': ats_z, 'params': (df, 100)},
        'ats_z_200': {'call': ats_z, 'params': (df, 200)},
        'bullish_bar': {'call': bull_bear_bar, 'params': (df,)},
        'bearish_bar': {'call': bull_bear_bar, 'params': (df,)},
        'weighted_50_bull_doji': {'call': doji, 'params': (df, 0.5, 2, True)},
        'weighted_50_bear_doji': {'call': doji, 'params': (df, 0.5, 2, True)},
        'weighted_100_bull_doji': {'call': doji, 'params': (df, 1, 2, True)},
        'weighted_100_bear_doji': {'call': doji, 'params': (df, 1, 2, True)},
        'weighted_150_bull_doji': {'call': doji, 'params': (df, 1.5, 2, True)},
        'weighted_150_bear_doji': {'call': doji, 'params': (df, 1.5, 2, True)},
        'unweighted_bull_doji': {'call': doji, 'params': (df, 0.5, 2, False)},
        'unweighted_bear_doji': {'call': doji, 'params': (df, 0.5, 2, False)},
        'bearish_engulf_1': {'call': engulfing, 'params': (df, 1)},
        'bearish_engulf_2': {'call': engulfing, 'params': (df, 2)},
        'bearish_engulf_3': {'call': engulfing, 'params': (df, 3)},
        'bullish_engulf_1': {'call': engulfing, 'params': (df, 1)},
        'bullish_engulf_2': {'call': engulfing, 'params': (df, 2)},
        'bullish_engulf_3': {'call': engulfing, 'params': (df, 3)},
        'chan_mid_ratio_25': {'call': channel_mid_ratio, 'params': (df, 25)},
        'chan_mid_ratio_50': {'call': channel_mid_ratio, 'params': (df, 50)},
        'chan_mid_ratio_100': {'call': channel_mid_ratio, 'params': (df, 100)},
        'chan_mid_ratio_200': {'call': channel_mid_ratio, 'params': (df, 200)},
        'chan_mid_width_25': {'call': channel_mid_width, 'params': (df, 25)},
        'chan_mid_width_50': {'call': channel_mid_width, 'params': (df, 50)},
        'chan_mid_width_100': {'call': channel_mid_width, 'params': (df, 100)},
        'chan_mid_width_200': {'call': channel_mid_width, 'params': (df, 200)},
        'daily_open_ratio': {'call': daily_open_ratio, 'params': (df,)},
        'roc_1d': {'call': daily_roc, 'params': (df, timeframe)},
        'day_of_week': {'call': day_of_week, 'params': (df,)},
        'day_of_week_180': {'call': day_of_week_180, 'params': (df,)},
        'ema_12_break_up': {'call': ema_breakout, 'params': (df, 12, 25)},
        'ema_12_break_down': {'call': ema_breakout, 'params': (df, 12, 25)},
        'ema_25_break_up': {'call': ema_breakout, 'params': (df, 25, 50)},
        'ema_25_break_down': {'call': ema_breakout, 'params': (df, 25, 50)},
        'ema_50_break_up': {'call': ema_breakout, 'params': (df, 50, 100)},
        'ema_50_break_down': {'call': ema_breakout, 'params': (df, 50, 100)},
        'ema_100_break_up': {'call': ema_breakout, 'params': (df, 100, 200)},
        'ema_100_break_down': {'call': ema_breakout, 'params': (df, 100, 200)},
        'ema_25_ratio': {'call': ema_ratio, 'params': (df, 25)},
        'ema_50_ratio': {'call': ema_ratio, 'params': (df, 50)},
        'ema_100_ratio': {'call': ema_ratio, 'params': (df, 100)},
        'ema_200_ratio': {'call': ema_ratio, 'params': (df, 200)},
        'ema_25_roc': {'call': ema_roc, 'params': (df, 25)},
        'ema_50_roc': {'call': ema_roc, 'params': (df, 50)},
        'ema_100_roc': {'call': ema_roc, 'params': (df, 100)},
        'ema_200_roc': {'call': ema_roc, 'params': (df, 200)},
        'hma_25_ratio': {'call': hma_ratio, 'params': (df, 25)},
        'hma_50_ratio': {'call': hma_ratio, 'params': (df, 50)},
        'hma_100_ratio': {'call': hma_ratio, 'params': (df, 100)},
        'hma_200_ratio': {'call': hma_ratio, 'params': (df, 200)},
        'hma_25_roc': {'call': hma_roc, 'params': (df, 25)},
        'hma_50_roc': {'call': hma_roc, 'params': (df, 50)},
        'hma_100_roc': {'call': hma_roc, 'params': (df, 100)},
        'hma_200_roc': {'call': hma_roc, 'params': (df, 200)},
        'hour': {'call': hour, 'params': (df,)},
        'hour_180': {'call': hour_180, 'params': (df,)},
        'inside_bar': {'call': inside_bar, 'params': (df,)},
        'kurtosis_6': {'call': kurtosis, 'params': (df, 6)},
        'kurtosis_12': {'call': kurtosis, 'params': (df, 12)},
        'kurtosis_25': {'call': kurtosis, 'params': (df, 25)},
        'kurtosis_50': {'call': kurtosis, 'params': (df, 50)},
        'kurtosis_100': {'call': kurtosis, 'params': (df, 100)},
        'kurtosis_200': {'call': kurtosis, 'params': (df, 200)},
        'prev_daily_open_ratio': {'call': prev_daily_open_ratio, 'params': (df,)},
        'prev_daily_high_ratio': {'call': prev_daily_high_ratio, 'params': (df,)},
        'prev_daily_low_ratio': {'call': prev_daily_low_ratio, 'params': (df,)},
        'recent_vd_div_1': {'call': vol_delta_div, 'params': (df, 1)},
        'recent_vd_div_2': {'call': vol_delta_div, 'params': (df, 2)},
        'recent_vd_div_3': {'call': vol_delta_div, 'params': (df, 3)},
        'recent_vd_div_4': {'call': vol_delta_div, 'params': (df, 4)},
        'rsi_14': {'call': rsi, 'params': (df, 14)},
        'rsi_25': {'call': rsi, 'params': (df, 25)},
        'rsi_50': {'call': rsi, 'params': (df, 50)},
        'rsi_100': {'call': rsi, 'params': (df, 100)},
        'rsi_200': {'call': rsi, 'params': (df, 200)},
        'skew_6': {'call': skew, 'params': (df, 6)},
        'skew_12': {'call': skew, 'params': (df, 12)},
        'skew_25': {'call': skew, 'params': (df, 25)},
        'skew_50': {'call': skew, 'params': (df, 50)},
        'skew_100': {'call': skew, 'params': (df, 100)},
        'skew_200': {'call': skew, 'params': (df, 200)},
        'stoch_base_vol_25': {'call': stoch_base_vol, 'params': (df, 25)},
        'stoch_base_vol_50': {'call': stoch_base_vol, 'params': (df, 50)},
        'stoch_base_vol_100': {'call': stoch_base_vol, 'params': (df, 100)},
        'stoch_base_vol_200': {'call': stoch_base_vol, 'params': (df, 200)},
        'stoch_num_trades_25': {'call': stoch_num_trades, 'params': (df, 25)},
        'stoch_num_trades_50': {'call': stoch_num_trades, 'params': (df, 50)},
        'stoch_num_trades_100': {'call': stoch_num_trades, 'params': (df, 100)},
        'stoch_num_trades_200': {'call': stoch_num_trades, 'params': (df, 200)},
        'stoch_vwma_ratio_25': {'call': stoch_vwma_ratio, 'params': (df, 25)},
        'stoch_vwma_ratio_50': {'call': stoch_vwma_ratio, 'params': (df, 50)},
        'stoch_vwma_ratio_100': {'call': stoch_vwma_ratio, 'params': (df, 100)},
        'fractal_trend_age_long': {'call': fractal_trend_age, 'params': (df, )},
        'fractal_trend_age_short': {'call': fractal_trend_age, 'params': (df, )},
        'vol_delta': {'call': vol_delta, 'params': (df,)},
        'vol_delta_pct': {'call': vol_delta_pct, 'params': (df,)},
        'vol_denom_roc_2': {'call': vol_denom_roc, 'params': (df, 2, 25)},
        'vol_denom_roc_5': {'call': vol_denom_roc, 'params': (df, 5, 50)},
        'week_of_year': {'call': week_of_year, 'params': (df,)},
        'week_of_year_180': {'call': week_of_year_180, 'params': (df,)},
        'roc_1w': {'call': weekly_roc, 'params': (df, timeframe)}
    }
    feature = feature_lookup[name]
    df = feature['call'](*feature['params'])

    return df

# new features: round_numbers_proximity (powers of 10, single digit integer multiples of powers of 10,
# numerology numbers, numerology multiples of powers of 10


def htf_fractals_proximity(df: pd.DataFrame, orig_tf: str, htf: str='W', frac_width: int=5) -> pd.DataFrame:
    """resamples to weekly or monthly timeframe, calculates williams fractals on that, works out which is closest to the
    current price and returns the pct difference"""

    closes = df.close.resample(htf).shift(1) # make sure the close price for each week is recorded on the following week

    # only need to resample close column
    fractal_high = np.where(closes == closes.rolling(frac_width, center=True).max(), closes, np.nan)
    fractal_low = np.where(closes == closes.rolling(frac_width, center=True).min(), closes, np.nan)

    # make a list of all the levels with their timestamps (they are only valid levels after they have been established)

    # for each period in the original df, record which is the closest VALID level, then create a column that represents
    # how far those prices are from the open price or vwma

    return df


def daily_open_ratio(df: pd.DataFrame):
    # for each period in the df, find the daily open, then divide each period's close price by that daily open
    pass


def daily_sfp(df: pd.DataFrame):
    # group by day (not day of week or day of year but every unique day) and calculate for each period whether the price
    # has started on one side of the open and then crossed to the other side.
    # maybe include a volume filter to ignore when a tiny percentage of normal volume was one one side to avoid falsly
    # classifying a retest as a cross
    pass


def atr_pct(df, lb):
    df = ind.atr(df, lb)
    return df.drop(f'atr-{lb}', axis=1)


def stoch_vwma_ratio(df, lookback: int) -> pd.DataFrame:
    df[f"stoch_vwma_ratio_{lookback}"] = ind.stochastic(df.close / df.vwma, lookback)
    return df


def ema_roc(df, length) -> pd.DataFrame:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    df[f"ema_{length}_roc"] = (df[f"ema_{length}"].pct_change())

    return df


def ema_ratio(df, length) -> pd.DataFrame:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    df[f"ema_{length}_ratio"] = (df.close / df[f"ema_{length}"])

    return df


def ema_breakout(df: pd.DataFrame, length: int, lookback: int) -> pd.DataFrame:
    """creates two columns 'ema_break_up' and 'ema_break_down' which represent whether the ema of close prices is above or below the
    range it occupied over the lookback period. if both are false, it is within the range."""

    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()
    ema_high = df[f"ema_{length}"].shift(1).rolling(lookback).max()
    ema_low = df[f"ema_{length}"].shift(1).rolling(lookback).min()

    if f'ema_{length}_break_up' not in df.columns:
        df[f'ema_{length}_break_up'] = (df[f"ema_{length}"] > ema_high)
        df[f'ema_{length}_break_down'] = (df[f"ema_{length}"] < ema_low)

    return df


def engulfing(df, lookback: int = 1) -> pd.DataFrame:
    if f'bullish_engulf_{lookback}' not in df.columns:
        df = ind.engulfing(df, lookback)

    return df


def doji(df: pd.DataFrame, thresh: float, lookback: int, weighted: bool) -> pd.DataFrame:
    """returns booleans which represent whether there was an upper or lower wick which met the threshold requirement in
    the lookback window. If weighted == True, a rolling 50 period z-score of base volume is calculated and the wick size
    is multiplied by the volume z-score, so higher than average volume will increase the chance of the wick being
    considered over the threshold"""

    w = f'weighted_{int(thresh*100)}' if weighted else 'unweighted'

    if f'{w}_bull_doji' in df.columns:
        return df

    df = ind.doji(df)

    if weighted:
        z_volume = ind.z_score(df.base_vol, 50)
        bull_doji_bool = df.bullish_doji * z_volume >= thresh
        bear_doji_bool = df.bearish_doji * z_volume >= thresh
    else:
        bull_doji_bool = df.bullish_doji >= thresh
        bear_doji_bool = df.bearish_doji >= thresh

    # df = df.drop(['bullish_doji', 'bearish_doji'], axis=1)

    bull_bool_window = bull_doji_bool.rolling(lookback).sum() > 0
    bear_bool_window = bear_doji_bool.rolling(lookback).sum() > 0

    df[f'{w}_bull_doji'] = bull_bool_window
    df[f'{w}_bear_doji'] = bear_bool_window

    return df


def vol_doji(df: pd.DataFrame, thresh: float, lookback: int, weighted: bool) -> pd.DataFrame:
    """same as doji, but the wicks are measured in terms of recent atr volatility, so """
    pass


def bull_bear_bar(df) -> pd.DataFrame:
    if 'bullish_bar' not in df.columns:
        df = ind.bull_bear_bar(df)

    return df


def hour(df:pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df.timestamp.dt.hour
    return df


def hour_180(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_180'] = (df.timestamp.dt.hour + 12) % 24
    return df


def day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df.timestamp.dt.dayofweek
    return df


def day_of_week_180(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week_180'] = (df.timestamp.dt.dayofweek + 3) % 7
    return df


def week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    df['week_of_year'] = df.timestamp.dt.dayofyear // 7
    return df


def week_of_year_180(df: pd.DataFrame) -> pd.DataFrame:
    df['week_of_year_180'] = ((df.timestamp.dt.dayofyear // 7) + 26) % 52
    return df


def vol_denom_roc(df: pd.DataFrame, roc_lb: int, atr_lb: int) -> pd.DataFrame:
    """returns the roc of price over the specified lookback period divided by the percentage denominated atr"""
    if f'atr_{atr_lb}_pct' not in df.columns:
        df = atr_pct(df, atr_lb)

    df[f'vol_denom_roc_{roc_lb}'] = (df.close.pct_change(roc_lb) / df[f'atr_{atr_lb}_pct'])

    return df


def vol_delta_div(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """adds a boolean series to the dataframe representing whether there was a volume delta divergence at any time
    during the lookback period"""
    roc: pd.Series = df.close.pct_change(1)
    if 'vol_delta' not in df.columns:
        df['vol_delta'] = ind.vol_delta(df)

    vd_div_a = (roc > 0) & (df.vol_delta < 0)
    vd_div_b = (roc < 0) & (df.vol_delta > 0)
    vd_div = vd_div_a | vd_div_b

    df[f'recent_vd_div_{lookback}'] = vd_div.rolling(lookback).sum() > 0

    return df


def ats_z(df: pd.DataFrame, lookback: int):
    avg_trade_size_sm = (df.base_vol / df.num_trades).ewm(5).mean()
    ats_long_mean = avg_trade_size_sm.ewm(lookback).mean()
    ats_std = avg_trade_size_sm.ewm(lookback).std()
    df[f'ats_z_{lookback}'] = ((avg_trade_size_sm - ats_long_mean) / ats_std)

    return df


def hma_roc(df, length) -> pd.DataFrame:
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    df[f"hma_{length}_roc"] = (df[f"hma_{length}"].pct_change())

    return df


def hma_ratio(df, length) -> pd.DataFrame:
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    df[f"hma_{length}_ratio"] = (df.close / df[f"hma_{length}"])

    return df


def channel_mid_ratio(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    df[f'chan_mid_ratio_{lookback}'] = (df.close / (chan_hi + chan_lo) / 2)

    return df


def channel_mid_width(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    chan_mid = (chan_hi + chan_lo) / 2

    df[f'chan_mid_width_{lookback}'] = ((chan_hi - chan_lo) / chan_mid)

    return df


def daily_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'daily_open' not in df.columns:
        df = ind.daily_open(df)

    df['daily_open_ratio'] = (df.close / df.daily_open)
    return df


def prev_daily_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_open' not in df.columns:
        df = ind.prev_daily_open(df)

    df['prev_daily_open_ratio'] = (df.close / df.prev_daily_open).shift()
    return df


def prev_daily_high_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_high' not in df.columns:
        df = ind.prev_daily_high(df)

    df['prev_daily_high_ratio'] = (df.close / df.prev_daily_high)
    return df


def prev_daily_low_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_low' not in df.columns:
        df = ind.prev_daily_low(df)

    df['prev_daily_low_ratio'] = (df.close / df.prev_daily_low)
    return df


def vol_delta_pct(df: pd.DataFrame) -> pd.DataFrame:
    df['vol_delta_pct'] = ind.vol_delta_pct(df)
    return df


def stoch_base_vol(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f'stoch_base_vol_{lookback}'] = ind.stochastic(df.base_vol, lookback)
    return df


def stoch_num_trades(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f'stoch_num_trades_{lookback}'] = ind.stochastic(df.num_trades, lookback)
    return df


def inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    df['inside_bar'] = ind.inside_bars(df)
    return df


def rsi(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f"rsi_{lookback}"] = ind.rsi(df.close, lookback)
    return df


def daily_roc(df: pd.DataFrame, timeframe) -> pd.DataFrame:
    periods_1d = {'1h': 24, '4h': 6, '12h': 2, '1d': 1}
    df['roc_1d'] = df.close.pct_change(periods_1d[timeframe])
    return df


def weekly_roc(df: pd.DataFrame, timeframe) -> pd.DataFrame:
    periods_1w = {'1h': 168, '4h': 42, '12h': 14, '1d': 7}
    df['roc_1w'] = df.close.pct_change(periods_1w[timeframe])
    return df


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['log_returns'] = np.log(df.close.pct_change() + 1)
    return df


def kurtosis(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if 'log_returns' not in df.columns:
        df = log_returns(df)

    df[f'kurtosis_{lookback}'] = df.log_returns.rolling(lookback).kurt()

    return df


def skew(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if 'log_returns' not in df.columns:
        df = log_returns(df)

    df[f'skew_{lookback}'] = df.log_returns.rolling(lookback).skew()

    return df


def vol_delta(df: pd.DataFrame) -> pd.DataFrame:
    df['vol_delta'] = ind.vol_delta(df)
    return df


def fractal_trend_age(df: pd.DataFrame, width: int=5, spacing: int=2) -> pd.DataFrame:
    if 'frac_low' not in df.columns:
        df = ind.williams_fractals(df, width, spacing)
        df = df.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1)

    long_trend_condition = df.open > df.frac_low
    df['fractal_trend_age_long'] = ind.consec_condition(long_trend_condition)

    short_trend_condition = df.open < df.frac_high
    df['fractal_trend_age_short'] = ind.consec_condition(short_trend_condition)

    return df


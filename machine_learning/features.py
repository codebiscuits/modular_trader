import indicators as ind
import pandas as pd
import numpy as np

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
    df[f"atr_{lb}_pct"] = df[f"atr_{lb}_pct"].shift(1)
    return df.drop(f'atr-{lb}', axis=1)


def stoch_vwma_ratio(df, lookback: int) -> pd.Series:
    return ind.stochastic(df.close / df.vwma, lookback).shift(1)

def ema_roc(df, length) -> pd.Series:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    return (df[f"ema_{length}"].pct_change()).shift(1)

def ema_ratio(df, length):
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    return (df.close / df[f"ema_{length}"]).shift(1)


def ema_breakout(df: pd.DataFrame, length: int, lookback: int) -> pd.DataFrame:
    """creates two columns 'ema_break_up' and 'ema_break_down' which represent whether the ema of close prices is above or below the
    range it occupied over the lookback period. if both are false, it is within the range."""

    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()
    ema_high = df[f"ema_{length}"].shift(1).rolling(lookback).max()
    ema_low = df[f"ema_{length}"].shift(1).rolling(lookback).min()

    df[f'ema_{length}_break_up'] = (df[f"ema_{length}"] > ema_high).shift()
    df[f'ema_{length}_break_down'] = (df[f"ema_{length}"] < ema_low).shift()

    return df


def engulfing(df, lookback: int = 1) -> pd.DataFrame:
    df = ind.engulfing(df, lookback)
    df[f'bullish_engulf_{lookback}'] = df[f'bullish_engulf_{lookback}'].shift(1)
    df[f'bearish_engulf_{lookback}'] = df[f'bearish_engulf_{lookback}'].shift(1)

    return df


def doji(df: pd.DataFrame, thresh: float, lookback: int) -> pd.DataFrame:
    """returns booleans which represent whether or not there was an upper or lower wick which met the threshold
    requirement in the lookback window"""

    df = ind.doji(df)
    bull_doji_bool = df.bullish_doji >= thresh
    bear_doji_bool = df.bearish_doji >= thresh
    bull_bool_window = bull_doji_bool.rolling(lookback).sum() > 0
    bear_bool_window = bear_doji_bool.rolling(lookback).sum() > 0

    df['recent_bull_doji'] = bull_bool_window.shift(1)
    df['recent_bear_doji'] = bear_bool_window.shift(1)

    return df


def bull_bear_bar(df) -> pd.DataFrame:
    df = ind.bull_bear_bar(df)
    df['bullish_bar'] = df.bullish_bar.shift(1)
    df['bearish_bar'] = df.bearish_bar.shift(1)

    return df


def hour(df:pd.DataFrame) -> pd.Series:
    return df.timestamp.dt.hour


def hour_180(df: pd.DataFrame) -> pd.Series:
    return (df.timestamp.dt.hour + 12) % 24


def day_of_week(df: pd.DataFrame) -> pd.Series:
    return df.timestamp.dt.dayofweek


def day_of_week_180(df: pd.DataFrame) -> pd.Series:
    return (df.timestamp.dt.dayofweek + 3) % 7


def week_of_year(df: pd.DataFrame) -> pd.Series:
    return df.timestamp.dt.dayofyear // 7


def week_of_year_180(df: pd.DataFrame) -> pd.Series:
    return ((df.timestamp.dt.dayofyear // 7) + 26) % 52


def vol_denom_roc(df: pd.DataFrame, roc_lb: int, atr_lb: int) -> pd.Series:
    """returns the roc of price over the specified lookback period divided by the percentage denominated atr"""
    if f'atr_{atr_lb}_pct' not in df.columns:
        df = atr_pct(df, atr_lb)

    return (df.close.pct_change(roc_lb) / df[f'atr_{atr_lb}_pct']).shift(1)


def vol_delta_div(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """adds a boolean series to the dataframe representing whether there was a volume delta divergence at any time
    during the lookback period"""
    roc: pd.Series = df.close.pct_change(1)
    if 'vol_delta' not in df.columns:
        df['vol_delta'] = ind.vol_delta(df)

    vd_div_a = (roc > 0) & (df.vol_delta < 0)
    vd_div_b = (roc < 0) & (df.vol_delta > 0)
    vd_div = vd_div_a | vd_div_b

    df[f'recent_vd_div_{lookback}'] = vd_div.shift(1).rolling(lookback).sum() > 0

    return df


def ats_z(df: pd.DataFrame, lookback: int):
    avg_trade_size_sm = (df.base_vol / df.num_trades).ewm(5).mean()
    ats_long_mean = avg_trade_size_sm.ewm(lookback).mean()
    ats_std = avg_trade_size_sm.ewm(lookback).std()
    df[f'ats_z_{lookback}'] = ((avg_trade_size_sm - ats_long_mean) / ats_std).shift(1)

    return df


def hma_roc(df, length) -> pd.Series:
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    return (df[f"hma_{length}"].pct_change()).shift(1)

def hma_ratio(df, length):
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    return (df.close / df[f"hma_{length}"]).shift(1)

def channel_mid_ratio(df: pd.DataFrame, lookback: int) -> pd.Series:
    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    return (df.close / (chan_hi + chan_lo) / 2).shift()

def channel_mid_width(df: pd.DataFrame, lookback: int) -> pd.Series:
    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    chan_mid = (chan_hi + chan_lo) / 2

    return ((chan_hi - chan_lo) / chan_mid).shift()


def daily_open_ratio(df: pd.DataFrame) -> pd.Series:
    if 'daily_open' not in df.columns:
        df = ind.daily_open(df)

    return (df.close / df.daily_open).shift()


def prev_daily_open_ratio(df: pd.DataFrame) -> pd.Series:
    if 'prev_daily_open' not in df.columns:
        df = ind.prev_daily_open(df)

    return (df.close / df.prev_daily_open).shift()


def prev_daily_high_ratio(df: pd.DataFrame) -> pd.Series:
    if 'prev_daily_high' not in df.columns:
        df = ind.prev_daily_high(df)

    return (df.close / df.prev_daily_high).shift()


def prev_daily_low_ratio(df: pd.DataFrame) -> pd.Series:
    if 'prev_daily_low' not in df.columns:
        df = ind.prev_daily_low(df)

    return (df.close / df.prev_daily_low).shift()




import indicators as ind
import pandas as pd
import numpy as np

def stoch_vwma_ratio(df, lookback: int) -> pd.Series:
    return ind.stochastic(df.close / df.vwma, lookback)

def ema_roc(df, length) -> pd.Series:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    return df[f"ema_{length}"].pct_change()

def ema_ratio(df, length):
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    return df.close / df[f"ema_{length}"]


def engulfing(df, lookback: int = 1) -> pd.DataFrame:
    df = ind.engulfing(df, lookback)
    df['bullish_engulf'] = df.bullish_engulf.shift(1)
    df['bearish_engulf'] = df.bearish_engulf.shift(1)

    return df


def doji(df) -> pd.DataFrame:
    df = ind.doji(df)
    df['bullish_doji'] = df.bullish_doji.shift(1)
    df['bearish_doji'] = df.bearish_doji.shift(1)

    return df


def bull_bear_bar(df) -> pd.DataFrame:
    df = ind.bull_bear_bar(df)
    df['bullish_bar'] = df.bullish_bar.shift(1)
    df['bearish_bar'] = df.bearish_bar.shift(1)

    return df


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


import indicators as ind
import pandas as pd

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


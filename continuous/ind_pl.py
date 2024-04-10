import polars as pl
import math


def hma(data, length):
    half_len = length / 2
    root_len = round(math.sqrt(length))
    data = data.with_columns(
        ((pl.col('close').ewm_mean(span=half_len) * 2) - pl.col('close').ewm_mean(span=length))
        .ewm_mean(span=root_len)
        .alias(f'hma_{length}')
    )
    return data


def ema(data, length):
    data = data.with_columns(pl.col('close').ewm_mean(span=length).alias(f'ema_{length}'))

    return data


def ichimoku(data, f=9, s=26):
    """fast and slow periods make up all the different components of the ichimoku system"""

    # indicators
    data = data.with_columns(
        ((pl.col('high').rolling_max(f, min_periods=2) + pl.col('low').rolling_min(f, min_periods=2)) / 2).shift(1).alias(f'tenkan_{f}'),
        ((pl.col('high').rolling_max(s, min_periods=2) + pl.col('low').rolling_min(s, min_periods=2)) / 2).shift(1).alias(f'kijun_{s}'),
        ((pl.col('high').rolling_max(s, min_periods=2) + pl.col('low').rolling_min(s * 2, min_periods=2)) / 2).shift(1 + s).alias(f'senkou_b_{s*2}'),
        pl.col('close').shift(1 - s).alias(f'chikou_{1-s}')
    )
    data = data.with_columns(
        ((pl.col(f'tenkan_{f}') + pl.col(f'kijun_{s}')) / 2).shift(1 + s).alias(f'senkou_a_{f}_{s}')
    )

    return data

def rsi_clip(series: pl.Series, lookback: int=14) -> pl.Series:
    """transforms input series into rsi of that series"""

    ups = series.clip(lower_bound=0).ewm_mean(lookback)
    downs = series.clip(upper_bound=0).abs().ewm_mean(lookback)

    return pl.Series(pl.when(downs > 0.0).then((ups - downs) / downs).otherwise(1.0))

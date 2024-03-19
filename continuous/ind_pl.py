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
        ((pl.col('high').rolling_max(f) + pl.col('low').rolling_min(f)) / 2).shift(1).alias(f'tenkan_{f}'),
        ((pl.col('high').rolling_max(s) + pl.col('low').rolling_min(s)) / 2).shift(1).alias(f'kijun_{s}'),
        ((pl.col('high').rolling_max(s) + pl.col('low').rolling_min(s * 2)) / 2).shift(1 + s).alias(f'senkou_b_{s*2}'),
        pl.col('close').shift(1 - s).alias(f'chikou_{1-s}')
    )
    data = data.with_columns(
        ((pl.col(f'tenkan_{f}') + pl.col(f'kijun_{s}')) / 2).shift(1 + s).alias(f'senkou_a_{f}_{s}')
    )

    return data

import polars as pl
import math
import plotly.express as px


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


def rsi(series: pl.Series, lookback: int=14) -> pl.Series:
    """transforms input series into rsi of that series"""

    ups = series.pct_change().clip(lower_bound=0).ewm_mean(lookback)
    downs = series.pct_change().clip(upper_bound=0).abs().ewm_mean(lookback)

    df = pl.DataFrame({'ups': ups, 'downs': downs})

    df = df.with_columns(
        pl.when(pl.col("downs").gt(0.0))
        .then(100 - (100 / (1.0 + pl.col("ups") / pl.col("downs"))))
        .otherwise(50)
        .alias(f"rsi_{lookback}")
    )

    return df[f"rsi_{lookback}"]


def stochastic(series: pl.Series, lookback: int=14) -> pl.Series:
    """transforms input series into stochastic oscillator of that series"""

    df = pl.DataFrame(
        {'input': series,
         'highs': series.rolling_max(lookback, min_periods=1),
         'lows': series.rolling_min(lookback, min_periods=1)}
    )

    df = df.with_columns(
        pl.col('input')
        .sub(pl.col('lows'))
        .truediv(pl.col('highs').sub(pl.col('lows')))
        .mul(100)
        .alias(f"stoch_{lookback}")
    )

    return df[f"stoch_{lookback}"]


def ema_balance(input: pl.DataFrame, lookback: int=14) -> pl.Series:
    """an oscillator that reperesents what proportion of recent price action or volume was above or below the moving
    average. values close to 1 indicate a strong uptrend, values close to -1 indicate a strong downtrend, and values
    close to 0 indicate choppy conditions with no trend"""

    input = ema(input, lookback)

    return input.with_columns(
        pl.col('close')
        .sub(pl.col(f"ema_{lookback}"))
        .rolling_mean(lookback)
        # .rolling_quantile(quantile=1.0, window_size=lookback)
        .alias('ema_balance')
    )

def atr(df: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """calculates true range of the input data, then calculates a rolling average of the true range according to the
    specified lookback period"""

    return df.with_columns(
        pl.col('high').sub(pl.col('low'))
        .truediv(pl.col('high').add(pl.col('low')).truediv(pl.lit(2)))
        .ewm_mean(span=lookback)
        .alias(f'atr_{lookback}'),
    )

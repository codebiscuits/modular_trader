import polars as pl
from continuous import ind_pl as ind
from datetime import datetime

# TODO check the code that manages decaying signals. it should be possible for successive signals to accumulate in some
#  way so 2 or more successive signals make the allocation stronger. maybe i could allow them to sum together, then
#  clip the magnitude of the resulting signal, so it becomes a truncated larger signal, and therefore longer but not
#  taller. or maybe allow successive signals to make the peak slightly higher than a single signal would be, but not
#  full double or triplle height.

class SubStrat:
    """this class contains one specific set of trading rules and a designated timeframe, and analyses the market data
    to produce a sub-size for the Trader object to use in trading decisions"""

    def __init__(self, data, timeframe: str):
        self.timeframe = timeframe
        self.data_start = data.item(0, 'timestamp')
        self.data_end = data.item(-1, 'timestamp')
        if self.timeframe != '1h':
            self.data = self.resample_data(data)
        else:
            self.data = data
        self.forecast_col = str()

    def resample_data(self, data):
        self.data = (data.group_by_dynamic(pl.col('timestamp'), every=self.timeframe).agg(
            pl.first('open'),
            pl.max('high'),
            pl.min('low'),
            pl.last('close'),
            pl.sum('quote_vol'),
            pl.sum('base_vol_delta'),
            pl.last('vwma_1h'),
            pl.last('vwma_25h'),
            pl.last('vwma_50h'),
            pl.last('vwma_100h'),
            pl.last('vwma_200h'),
        ))

        self.data = self.data.sort('timestamp')

        self.data = self.data.with_columns(
            pl.col('close').pct_change().fill_null(0).alias(f'pct_change_{self.timeframe}'))

        # TODO maybe try optimising the rolling_std window, 50 was just a guess
        annualiser = {'1h': 94, '4h': 47, '8h': 33, '1d': 19, '2d': 14, '3d': 11, '1w': 7}[self.timeframe]
        self.data = self.data.with_columns(
            (pl.col(f'pct_change_{self.timeframe}').rolling_std(50, min_periods=3) * annualiser).alias(f'dyn_std')
        )

        return (
            self.data
            .select(['timestamp', 'open', 'high', 'low', 'close', 'quote_vol', 'base_vol_delta', 'dyn_std',
                     f'pct_change_{self.timeframe}', 'vwma_1h', 'vwma_25h', 'vwma_50h', 'vwma_100h', 'vwma_200h'])
            .set_sorted('timestamp')
        )

    # def process_forecast(self,
    #                      long_only: bool = False,
    #                      standardise: bool = False,
    #                      normalise: bool = False,
    #                      flip: bool = False,
    #                      smooth: bool = False,
    #                      quantise: float = 0.5,
    #                      shift: bool = True) -> pl.Series:
    #
    #     # fit range to long and short
    #     if not long_only:
    #         self.data = self.data.with_columns(pl.col(self.forecast_col).mul(2).sub(1).alias(self.forecast_col))
    #
    #     # standardise forecast at local timescale
    #     if standardise:
    #         self.data = (
    #             self.data.with_columns(
    #                 pl.col(self.forecast_col)
    #                 .truediv(pl.col('dyn_std'))
    #                 .alias(self.forecast_col)
    #             )
    #         )
    #
    #     if normalise:
    #         self.data = (
    #             self.data.with_columns(
    #                 pl.col(self.forecast_col)
    #                 .truediv(pl.col(self.forecast_col).abs().mean())
    #                 .alias(self.forecast_col)
    #             )
    #         )
    #
    #     if flip:
    #         self.data = (
    #             self.data.with_columns(
    #                 pl.col(self.forecast_col)
    #                 .mul(-1)
    #                 .alias(self.forecast_col)
    #             )
    #         )
    #
    #     if smooth:
    #         self.data = (
    #             self.data.with_columns(
    #                 pl.col(self.forecast_col)
    #                 .ewm_mean(span=3)
    #                 .alias(self.forecast_col)
    #             )
    #         )
    #
    #     if quantise:
    #         self.data = (
    #             self.data.with_columns(
    #                 pl.col(self.forecast_col)
    #                 .truediv(quantise)
    #                 .round()
    #                 .mul(quantise)
    #                 .alias(self.forecast_col)
    #             )
    #         )
    #
    #     # clip
    #     self.data = (
    #         self.data.with_columns(
    #             pl.col(self.forecast_col)
    #             .clip(lower_bound=-1, upper_bound=1)
    #             .alias(self.forecast_col)
    #         )
    #     )

    def shift_and_resample(self):
        # clip values
        self.data = self.data.with_columns(
            pl.col(self.forecast_col).cast(pl.Float64).clip(lower_bound=-1, upper_bound=1).alias(self.forecast_col))

        # shift forecast one period at local timescale. this prevents look-ahead bias because i'm using closing prices
        self.data = self.data.with_columns(
            pl.col(self.forecast_col).shift(n=1, fill_value=0).alias(self.forecast_col))

        # resample to 1h
        if self.timeframe != '1h':
            self.data = (
                self.data.upsample(time_column='timestamp', every='1h')
                .interpolate().fill_null(strategy='forward')
            )

        self.data = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start=self.data_start,
                    end=self.data_end,
                    interval="1h",
                    time_unit='ns',
                    eager=True,
                )
            }
        ).lazy().join(self.data.lazy(), on="timestamp", how="left").collect().fill_null(strategy='forward')

        return self.data.get_column(self.forecast_col).fill_null(0).ewm_mean(6)

    def impulse_to_forecast(self, decay: float):
        fc_list = self.data[self.forecast_col].to_list()
        new_fc_list = []
        for i, x in enumerate(fc_list):
            if i == 0:
                new_fc_list.append(0)
            elif abs(fc_list[i]) < 1:
                new_fc_list.append(new_fc_list[i-1] * decay)
            else:
                new_fc_list.append(fc_list[i])

        return pl.Series(new_fc_list)


############################################## Mean Reversion ##############################################


class EmaVwmaZScore(SubStrat):
    """Compares the the EMA to the VWMA to identify genuinely overbought or oversold conditions"""


class KsRevInd3(SubStrat):
    """Uses 14 period autocorrelation of the diff of closing prices, combined the 14 period RSI. A bullish signal is
    generated when ac is above 60% while RSI is below 40%, bearish signals are generated when ac is above 60% and RSI is
    above 60%"""


class KsRevInd2(SubStrat):
    """"""


class KsRevInd1(SubStrat):
    """Uses BBands and MACD to identify high probability reversals. A bullish signal is generated when the close price
    is at or below the 100 period bbands while the macd line crosses over the signal line, the bearish signal is the
    opposite."""


class VolatilityBands(SubStrat):
    """Calculates z-scores of price movements around a moving average, then applies two moving averages to the z-score
    series. A buy signal is triggered when the fast ma crosses above the slower ma and a sell signal is triggered in the
    opposite scenario. The magnitude of the signal is determined by the z-score at the moment the signal is triggered.

    Possible values for ma_type include: sma, ema, hma or vwma.
    Possible values for lb include: 25, 50, 100 or 200.
    Fast ma must be less than slow ma.
    Slow ma must be no greater than lb."""

    def __init__(self, data, timeframe: str, ma_type: str, lb: int, fast_ma: int, slow_ma: int):
        super().__init__(data, timeframe)
        self.ma_type = ma_type
        self.lb = lb
        self.fast_ma = fast_ma
        self.slow_ma = int(lb * slow_ma)
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def calc_forecast(self):
        self.forecast_col = f"vol_bands_{self.timeframe}_{self.ma_type}_{self.lb}_{self.fast_ma}_{self.slow_ma}"

        # create the main moving average
        if self.ma_type == 'sma':
            self.data = self.data.with_columns(
                pl.col('close').rolling_mean(self.lb).alias('ma')
            )
        elif self.ma_type == 'ema':
            self.data = self.data.with_columns(
                pl.col('close').ewm_mean(self.lb).alias('ma')
            )
        elif self.ma_type == 'hma':
            self.data = ind.hma(self.data, self.lb)
            self.data = self.data.with_columns(
                pl.col(f"hma_{self.lb}").alias('ma')
            )
        elif self.ma_type == 'vwma':
            self.data = self.data.with_columns(
                pl.col(f'vwma_{self.lb}h').alias('ma')
            )
        else:
            raise TypeError('Not a valid ma_type')

        # create the z-score series
        self.data = self.data.with_columns(
            pl.col('vwma_1h').sub(pl.col('ma'))
            .truediv(
                pl.col('vwma_1h').sub(pl.col('ma'))
                .rolling_std(self.lb)
            )
            .alias('z_score')
        )

        # create the two moving averages of the z-score
        self.data = self.data.with_columns(
            pl.col('z_score').ewm_mean(self.fast_ma).alias('fast'),
            pl.col('z_score').ewm_mean(self.slow_ma).alias('slow')
        )

        # calculate signals
        self.data = self.data.with_columns(
            pl.when(
                pl.col('fast').gt(pl.col('slow'))
                .and_(pl.col('fast').shift().lt(pl.col('slow').shift()))
                .and_(pl.col('fast').lt(0.0))
            ).then(pl.col('fast'))
            .when(
                pl.col('fast').lt(pl.col('slow'))
                .and_(pl.col('fast').shift().gt(pl.col('slow').shift()))
                .and_(pl.col('fast').gt(0.0))
            ).then(pl.col('fast'))
            .fill_null(0)
            .alias(self.forecast_col)
        )

        # add decay to impulse signals
        self.data = self.data.with_columns(
            pl.Series(self.impulse_to_forecast(0.9))
            .alias(self.forecast_col),
        )

        # normalise
        self.data = self.data.with_columns(
            pl.col(self.forecast_col)
            .mul(2.0)
            .clip(lower_bound=-1.0, upper_bound=1.0)
            .alias(self.forecast_col),
        )

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class StochRSIReversal(SubStrat):
    """Mean reversion strategy which attempts to time reversals based on the rsi."""
    def __init__(self, data, timeframe: str, lb: int, col: str= 'close'):
        super().__init__(data, timeframe)
        self.lb = lb
        self.col = col
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def calc_forecast(self):
        extr_hi = 90
        extr_lo = 10

        self.forecast_col = f"stoch_rsi_rev_{self.timeframe}_{self.lb}_{self.col}"

        rsi_series = ind.rsi(self.data[self.col], self.lb)
        stoch_rsi_series = ind.stochastic(rsi_series, self.lb)

        self.data = self.data.with_columns(
            pl.Series(stoch_rsi_series).alias(f"stoch_rsi_{self.lb}"),
        )
        self.data = self.data.with_columns(
            # using rolling mean because ewm mean didn't work for some reason
            pl.col(f"stoch_rsi_{self.lb}").rolling_mean(5).alias('stoch_rsi_ema'),
            pl.col(f"stoch_rsi_{self.lb}").pct_change(n=5).alias('stoch_rsi_roc')
        )

        # px.line(data_frame=self.data.select([f"stoch_rsi_{self.lb}", 'stoch_rsi_ema']), title=f'stoch_rsi_{self.lb}').show()

        self.data = self.data.with_columns(
            pl.when(pl.col(f'stoch_rsi_{self.lb}').gt(extr_hi).or_(pl.col(f'stoch_rsi_{self.lb}').lt(extr_lo))).then(pl.lit(0))
            .when(pl.col('stoch_rsi_ema').gt(extr_hi).and_(pl.col(f'stoch_rsi_{self.lb}').lt(extr_hi))).then(pl.lit(-1))
            .when(pl.col('stoch_rsi_ema').lt(extr_lo).and_(pl.col(f'stoch_rsi_{self.lb}').gt(extr_lo))).then(pl.lit(1))
            # .fill_null(strategy='forward')
            .fill_null(0)
            .alias(self.forecast_col)
        )

        # px.line(data_frame=self.data.select(self.forecast_col), title=f'stoch_rsi_{self.lb} pre forecast').show()
        # print(self.data.tail(20))

        self.data = self.data.with_columns(
            pl.Series(self.impulse_to_forecast(0.9)).alias(self.forecast_col),
        )

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class RSIReversal(SubStrat):
    """Mean reversion strategy which attempts to time reversals based on the rsi."""
    def __init__(self, data, timeframe: str, lb: int, col: str= 'close'):
        super().__init__(data, timeframe)
        self.lb = lb
        self.col = col
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def calc_forecast(self):
        extr_hi = 75
        extr_lo = 25

        self.forecast_col = f"rsi_rev_{self.timeframe}_{self.lb}_{self.col}"

        rsi_series = ind.rsi(self.data[self.col], self.lb)

        self.data = self.data.with_columns(
            pl.Series(rsi_series),
            pl.Series(rsi_series).ewm_mean(span=5).alias('rsi_ema'),
            pl.Series(rsi_series).pct_change(n=5).alias('rsi_roc')
        )

        self.data = self.data.with_columns(
            pl.when(pl.col(f'rsi_{self.lb}').gt(extr_hi).or_(pl.col(f'rsi_{self.lb}').lt(extr_lo))).then(pl.lit(0))
            .when(pl.col('rsi_ema').gt(extr_hi).and_(pl.col(f'rsi_{self.lb}').lt(extr_hi))).then(pl.lit(-1))
            .when(pl.col('rsi_ema').lt(extr_lo).and_(pl.col(f'rsi_{self.lb}').gt(extr_lo))).then(pl.lit(1))
            # .fill_null(strategy='forward')
            .fill_null(0)
            .alias(self.forecast_col)
        )

        self.data = self.data.with_columns(
            pl.Series(self.impulse_to_forecast(0.9)).alias(self.forecast_col),
        )

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


############################################## Trend/Momentum ##############################################


class ChanBreak(SubStrat):
    """Momentum strategy which is always long or short, never flat. The signal flips long when the recent price range is
    broken to the upside and flips short when the recent price range is broken to the downside."""
    def __init__(self, data, timeframe: str, lb: int, col: str = 'close'):
        super().__init__(data, timeframe)
        self.lb = lb
        self.col = col
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def calc_forecast(self):
        self.forecast_col = f'chan_break_{self.timeframe}_{self.lb}_{self.col}'

        # is the current price above last period's channel high or below last periods channel low?
        # if so, was this not the case one period before that?

        self.data = self.data.with_columns(
            # these two are just for plotting, they aren't necessary for the forecast
            # pl.col(self.col).rolling_max(self.lb).shift(1).alias('chan_hi'),
            # pl.col(self.col).rolling_min(self.lb).shift(1).alias('chan_lo'),

            pl.col(self.col).rolling_max(self.lb).shift(1).lt(pl.col(self.col)).cast(pl.Int64).diff().alias('above'),
            pl.col(self.col).rolling_min(self.lb).shift(1).gt(pl.col(self.col)).cast(pl.Int64).diff().alias('below'),
        )

        # fill forecast_col with either 1 or -1 depending on which side of the channel was most recently broken
        self.data = self.data.with_columns(
            pl.when(pl.col('above') == 1).then(pl.lit(1))
            .when(pl.col('below') == 1).then(pl.lit(-1))
            .fill_null(strategy='forward')
            .fill_null(0)
            .alias(self.forecast_col)
        )

        # print(self.data.tail(20))

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class IchiTrend(SubStrat):

    def __init__(self, data, timeframe, f, s, input):
        super().__init__(data, timeframe)
        self.f = f
        self.s = s
        self.input = input
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def __str__(self):
        return f"IchiTrend ({self.f}, {self.s}), {self.input}, {self.timeframe}"

    def calc_forecast(self, standardise: bool = True, shift: bool = True) -> pl.Series:
        self.data = ind.ichimoku(self.data, self.f, self.s)

        # trend following strategy
        self.data = self.data.with_columns(
            pl.col(f'tenkan_{self.f}').gt(pl.col(f'kijun_{self.s}')).alias(f'tk_up_{self.f}_{self.s}'),

            pl.col(f'senkou_a_{self.f}_{self.s}').gt(pl.col(f'senkou_b_{self.s * 2}')).alias(
                f'cloud_up_{self.f}_{self.s}'),

            (pl.col(self.input).gt(pl.col(f'senkou_a_{self.f}_{self.s}')) & pl.col(self.input).gt(
                pl.col(f'tenkan_{self.f}'))).alias(f'price_above_tenkan_{self.f}_{self.s}'),

            (pl.col(self.input).gt(pl.col(f'senkou_a_{self.f}_{self.s}')) & pl.col(self.input).gt(
                pl.col(f'kijun_{self.s}'))).alias(f'price_above_kijun_{self.f}_{self.s}'),

            (pl.col(self.input).gt(pl.col(f'senkou_a_{self.f}_{self.s}')) & pl.col(self.input).gt(
                pl.col(f'senkou_b_{self.s * 2}'))).alias(f'price_above_cloud_{self.f}_{self.s}'),

            pl.col(self.input).gt(pl.col(self.input).shift(self.s)).alias(f'chikou_above_price_{self.f}_{self.s}')
        )

        self.forecast_col = f'ichi_trend_{self.timeframe}_{self.f}_{self.s}'

        self.data = self.data.with_columns(
            self.data.select(
                [f'tk_up_{self.f}_{self.s}',
                 f'cloud_up_{self.f}_{self.s}',
                 f'price_above_tenkan_{self.f}_{self.s}',
                 f'price_above_kijun_{self.f}_{self.s}',
                 f'price_above_cloud_{self.f}_{self.s}',
                 f'chikou_above_price_{self.f}_{self.s}']
            )
            .mean_horizontal()
            .alias(self.forecast_col)
        )

        self.data = self.data.with_columns(pl.col(self.forecast_col).mul(2.0).sub(1.0).alias(self.forecast_col))

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class EmaRoc(SubStrat):
    def __init__(self, data, timeframe, lb):
        super().__init__(data, timeframe)
        self.lb = lb
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def __str__(self):
        return f"EmaRoc {self.timeframe} {self.lb}"

    def calc_forecast(self) -> pl.DataFrame:
        self.forecast_col = f'EmaRoc_{self.timeframe}_{self.lb}'

        self.data = ind.ema(self.data, self.lb)

        # trend following strategy
        self.data = self.data.with_columns(
            pl.col(f'ema_{self.lb}').pct_change().gt(0).alias(self.forecast_col),
        )

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class HmaRoc(SubStrat):
    def __init__(self, data, timeframe, lb):
        super().__init__(data, timeframe)
        self.lb = lb
        self.data = self.calc_forecast()
        self.forecast = self.shift_and_resample()

    def __str__(self):
        return f"HmaRoc {self.timeframe} {self.lb}"

    def calc_forecast(self) -> pl.DataFrame:
        self.forecast_col = f'HmaRoc_{self.timeframe}'

        self.data = ind.hma(self.data, self.lb)

        # trend following strategy
        self.data = self.data.with_columns(
            pl.col(f'hma_{self.lb}').pct_change().gt(0).alias(self.forecast_col)
        )

        return self.data.select(['timestamp', 'dyn_std', self.forecast_col])


class EmaTrend(SubStrat):

    def __init__(self, market, timeframe, fast, slow):
        super().__init__(market, timeframe)
        self.fast = fast
        self.slow = slow

    def __str__(self):
        return f"EmaTrend ({self.fast}, {self.slow}), {self.market}, {self.timeframe}"

    def ideal_position(self):
        pass

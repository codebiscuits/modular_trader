from wootrade import Client as Client_w
from binance.client import Client as Client_b
import binance.enums as be
import mt.resources.keys as keys
from datetime import datetime, timedelta, timezone
import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics as stats
from pathlib import Path
import ind_pl as ind

# client = Client_w(keys.woo_key, keys.woo_secret, keys.woo_app_id, testnet=True)
client = Client_b(keys.bPkey, keys.bSkey)

class Session:
    """this class loads and distributes all data that relates to the trading account and the current trading session
    and handles record-keeping"""

    pass

class SubStrat:
    """this class contains one specific set of trading rules and a designated timeframe, and analyses the market data
    to produce a sub-size for the Trader object to use in trading decisions"""

    def __init__(self, market: str, timeframe: str):
        self.market = market
        self.timeframe = timeframe
        self.data = self.load_data()

    def load_data(self):
        time_aggs = {'1h': 12, '4h': 48, '8h': 96, '12h': 144, '1d':288, '3d':864, '1w':2016}

        datapath = Path(f"/home/ross/coding/modular_trader/bin_ohlc_5m/{self.market}.parquet")
        df = pl.scan_parquet(datapath).set_sorted('timestamp')

        df = (
            df.with_columns(
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=time_aggs[self.timeframe], min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=time_aggs[self.timeframe], min_periods=1))
                .alias('vwma_1'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=time_aggs[self.timeframe] * 25, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=time_aggs[self.timeframe] * 25, min_periods=1))
                .alias('vwma_25'),
                pl.col('close').mul(pl.col('base_vol'))
                .rolling_sum(window_size=time_aggs[self.timeframe] * 50, min_periods=1)
                .truediv(pl.col('base_vol').rolling_sum(window_size=time_aggs[self.timeframe] * 50, min_periods=1))
                .alias('vwma_50'),
            )
        )

        df = (df.group_by_dynamic(pl.col('timestamp'), every=self.timeframe).agg(
                pl.first('open'),
                pl.max('high'),
                pl.min('low'),
                pl.last('close'),
                pl.sum('base_vol'),
                pl.sum('quote_vol'),
                pl.sum('num_trades'),
                pl.sum('taker_buy_base_vol'),
                pl.sum('taker_buy_quote_vol'),
                pl.last('vwma_1'),
                pl.last('vwma_25'),
                pl.last('vwma_50'),
            ))

        df = df.sort('timestamp')

        df = df.with_columns(
            pl.col('taker_buy_base_vol').mul(2).sub(pl.col('base_vol')).alias('base_vol_delta'),
            pl.col('close').pct_change().fill_null(0).alias('pct_change')
        )

        annualiser = {'1h': 94, '4h': 47, '8h': 33, '1d': 19, '2d': 14, '3d': 11, '1w': 7}[self.timeframe]
        df = df.with_columns((pl.col('pct_change').rolling_std(50, min_periods=3) * annualiser).alias(f'dyn_std'))

        return df.select(['timestamp', 'open', 'high', 'low', 'close', 'quote_vol', 'base_vol_delta', 'pct_change',
                          'vwma_1', 'vwma_25', 'vwma_50', 'dyn_std'])

    def standard_forecast(self, name, flip):
        """scales a forecast by dynamic standard deviation"""

        flipper = -1 if flip else 1

        self.data = self.data.with_columns(pl.col(name).truediv(pl.col('dyn_std')))
        self.data = self.data.with_columns(
            pl.col(name).mul(flipper).truediv(pl.col(name).abs().mean()).clip(lower_bound=-2, upper_bound=2).alias(
                name))


class IchiTrend(SubStrat):

    def __init__(self, market, timeframe, a, b):
        super().__init__(market, timeframe)
        self.a = a
        self.b = b

    def __str__(self):
        return f"IchiTrend ({self.a}, {self.b}), {self.market}, {self.timeframe}"

    def ideal_position(self):
        self.data = ind.ichimoku(self.data, self.a, self.b)

        # trend following strategy
        self.data = self.data.with_columns(
            pl.col('tenkan').gt(pl.col('kijun')).alias('tk_up'),
            pl.col('senkou_a').gt(pl.col('senkou_b')).alias('cloud_up'),
            (pl.col('close').gt(pl.col('senkou_a')) & pl.col('close').gt(pl.col('tenkan'))).alias('price_above_tenkan'),
            (pl.col('close').gt(pl.col('senkou_a')) & pl.col('close').gt(pl.col('kijun'))).alias('price_above_kijun'),
            (pl.col('close').gt(pl.col('senkou_a')) & pl.col('close').gt(pl.col('senkou_b'))).alias('price_above_cloud'),
            pl.col('close').gt(pl.col('close').shift(self.b)).alias('chikou_above_price')
        )
        self.data = self.data.with_columns(
            self.data.select(
                ['tk_up', 'cloud_up', 'price_above_tenkan', 'price_above_kijun', 'price_above_cloud', 'chikou_above_price']
            ).mean_horizontal().alias(f'ichi_trend_{self.a}_{self.b}')
        )

        self.data = self.standard_forecast(self.data, f'ichi_trend_{self.a}_{self.b}', False)

class EmaTrend(SubStrat):

    def __init__(self, market, timeframe, fast, slow):
        super().__init__(market, timeframe)
        self.fast = fast
        self.slow = slow

    def __str__(self):
        return f"EmaTrend ({self.fast}, {self.slow}), {self.market}, {self.timeframe}"

    def ideal_position(self):
        pass

class Trader:
    """this is the class that holds the market data and all strategies and strategy variations which apply to a given
    trading pair. they all combine to ultimately produce an ideal current size and therefore a trading action for a
    trading pair"""
    def __init__(self, market: str):
        self.market = market
        self.strats = []

    def add_substrat(self, strat, params: dict):
        self.strats.append(strat(**params))

    def __str__(self):
         return f"Trader object: {self.market}, substrats: {[str(s) for s in self.strats]}"

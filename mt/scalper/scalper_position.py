from pushbullet import Pushbullet
import numpy as np

# from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


class Position():

    def __init__(self, df, pair, tf, type):
        self.df = df
        self.pair = pair
        self.tf = tf
        self.type = type
        self.entry = self.df.close.iat[-1]
        self.direction = 1 if self.df.long_signal.iat[-1] else -1
        self.calc_inval()
        self.init_stop = self.df.inval.iat[-1]
        self.r = abs(self.entry - self.init_stop) / self.entry

        dir = 'long' if self.direction > 0 else 'short'
        print(f"{self.pair} {self.tf} {dir} position opened @ ${self.entry}, init stop @ {self.init_stop}")

    def calc_inval(self):
        if self.type == 'atr':
            self.atr_bands(10, 1)
            self.df['inval'] = (self.df.atr_lower
                                if self.direction > 0
                                else self.df.atr_upper)
        elif self.type == 'fractal':
            self.williams_fractals(5)
            self.df['inval'] = (self.df.fractal_low.interpolate('pad')
                                if self.direction > 0
                                else self.df.fractal_high.interpolate('pad'))
        elif self.type == 'oco':
            self.df['inval'] = (self.df.low.shift(1).rolling(2).min()
                                if self.direction > 0
                                else self.df.high.shift(1).rolling(2).max())

    def calc_atr(self, lb) -> None:
        """calculates the average true range on an ohlc dataframe"""
        self.df['tr1'] = self.df.high - self.df.low
        self.df['tr2'] = abs(self.df.high - self.df.close.shift(1))
        self.df['tr3'] = abs(self.df.low - self.df.close.shift(1))
        self.df['tr'] = self.df[['tr1', 'tr2', 'tr3']].max(axis=1)
        self.df['atr'] = self.df['tr'].ewm(lb).mean()
        self.df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    def atr_bands(self, lb: int, mult: float) -> None:
        '''calculates bands at a specified multiple of atr above and below price
        on a given dataframe'''
        self.df = self.calc_atr(lb)

        self.df['atr_upper'] = (self.df.vwma + mult * self.df.atr)
        self.df['atr_lower'] = (self.df.vwma - mult * self.df.atr)
        self.df = self.df.drop('atr')

    def williams_fractals(self, width):
        """calculates williams fractals on the highs and lows.
        frac_width determines how many candles are used to decide if the current candle is a local high/low, so a frac_width
        of five will look at the current candle, the two previous candles, and the two subsequent ones"""

        self.df['fractal_high'] = np.where(self.df.high == self.df.high.rolling(width, center=True).max(),
                                           self.df.high, np.nan)
        self.df['fractal_low'] = np.where(self.df.low == self.df.low.rolling(width, center=True).min(),
                                          self.df.low, np.nan)

    def check_position(self, df):
        pass


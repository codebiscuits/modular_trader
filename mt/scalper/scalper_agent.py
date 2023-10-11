import datetime
from functions import keys
import pandas as pd
from binance import Client
from pushbullet import Pushbullet

# from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class Agent():

    def __init__(self, params, id, live):

        self.pair = params['pair']
        self.timeframe = params['tf']
        self.stream = f"{self.pair.lower()}@kline_{self.timeframe}"
        self.bias_lb = params['bias_lb']
        self.source = params['source']
        self.bars = params['bars']
        self.z = params['z']
        self.width = params['width']
        self.name = f"{self.pair}_{self.timeframe}_{self.bias_lb}_{self.source}_{self.bars}_{self.z}_{self.width}"
        self.short_name = f"{self.pair}_{self.timeframe}_{id}"
        self.position = 0
        self.inval_atr = None
        self.inval_frac = None
        self.entry = None
        self.r = None
        self.open_time = None
        self.candle = None

        if live:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey)
        else:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey, testnet=True)

        # max_lb = the highest of bias_lb and bars*mult, but not more than 1000
        self.max_lb = min(max(self.bias_lb, (self.bars * 9)), 1000)

        print(f"init agent {id}: {self.name}")

    def make_dataframe(self, ohlc_data):
        self.df = pd.DataFrame(ohlc_data)
        self.df['timestamp'] = pd.to_datetime(self.df.timestamp, utc=True)

    def inside_bars(self):
        self.df['inside_bar'] = (self.df.high < self.df.high.shift(1)) & (self.df.low > self.df.low.shift(1))
        last_idx = self.df.index[-1]
        # if self.df.at[last_idx, 'inside_bar']:
        #     print(f"{self.stream} inside bar detected. {datetime.datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')}")

    def bar_direction(self):
        self.df['up_bar'] = self.df.close > self.df.open
        self.df['down_bar'] = self.df.open > self.df.close

    def doji(self):
        span = self.df.high - self.df.low
        upper_wick = self.df.high - self.df[['open', 'close']].max(axis=1)
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df.low
        self.df['bullish_doji'] = (lower_wick / span) > 0.5
        self.df['bearish_doji'] = (upper_wick / span) > 0.5

    def ema_trend(self):
        length = self.bias_lb
        span = int(length/100)
        self.df[f"ema_{length}"] = self.df.close.ewm(length).mean()
        self.df['ema_up'] = self.df[f"ema_{length}"] > self.df[f"ema_{length}"].shift(span)
        self.df['ema_down'] = self.df[f"ema_{length}"] < self.df[f"ema_{length}"].shift(span)

    def trend_rate(self):
        """returns True for any ohlc period which follows a strong trend as defined by the rate-of-change and bars params.
        if source series has moved at least a certain percentage within the set number of bars, it meets the criteria"""

        self.df[f"roc_{self.bars}"] = self.df[f"{self.source}"].pct_change(self.bars)
        m = self.df[f"roc_{self.bars}"].abs().rolling(self.bars * 9).mean()
        s = self.df[f"roc_{self.bars}"].abs().rolling(self.bars * 9).std()
        self.df['thresh'] = m + (self.z * s)

        self.df['trend_up'] = self.df[f"roc_{self.bars}"] > self.df['thresh']
        self.df['trend_down'] = self.df[f"roc_{self.bars}"] < 0 - self.df['thresh']

    def calc_atr(self, lb) -> None:
        """calculates the average true range on an ohlc dataframe"""
        self.df['tr1'] = self.df.high - self.df.low
        self.df['tr2'] = abs(self.df.high - self.df.close.shift(1))
        self.df['tr3'] = abs(self.df.low - self.df.close.shift(1))
        self.df['tr'] = self.df[['tr1', 'tr2', 'tr3']].max(axis=1)
        self.df['atr'] = self.df['tr'].ewm(lb).mean()
        self.df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    def entry_signals(self):
        self.df['long_signal'] = (self.df[['inside_bar', 'up_bar', 'bullish_doji']].any(axis=1)
                                  & self.df.trend_down & self.df.ema_up)

        self.df['short_signal'] = (self.df[['inside_bar', 'down_bar', 'bearish_doji']].any(axis=1)
                                   & self.df.trend_up & self.df.ema_down)

    def open_trade(self):
        self.df['long_entry_inval'] = self.df.low.shift(1).rolling(2).min()
        self.df['short_entry_inval'] = self.df.high.shift(1).rolling(2).max()

        last = self.df.to_dict('records')[-1]
        if 'inval_low' in last:
            long_inval = min(last['inval_low'], last['long_entry_inval'])
        else:
            long_inval = last['long_entry_inval']

        if 'inval_high' in last:
            short_inval = max(last['inval_high'], last['short_entry_inval'])
        else:
            short_inval = last['short_entry_inval']

        vol_delta = 'positive vol delta' if last['vol_delta'] > 0 else 'negative vol delta'
        price = last['close']
        now = datetime.datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
        now_stamp = datetime.datetime.now(timezone.utc)

        if last['long_signal']:
            self.position = 1
            self.inval = long_inval
            self.entry = price
            self.r = (price - long_inval) / price
            self.open_time = now_stamp
            note = f"{self.pair} {self.timeframe} long signal @ ${price}, {long_inval = }, {vol_delta}"
            print(now, note)
            # pb.push_note(title=now, body=note)
            if last['inside_bar']:
                self.candle = 'inside_bar'
            elif last['bullish_doji']:
                self.candle = 'doji'
            elif last['up_bar']:
                self.candle = 'up_bar'

        if last['short_signal']:
            self.position = -1
            self.inval = short_inval
            self.entry = price
            self.r = (short_inval - price) / price
            self.open_time = now_stamp
            note = f"{self.pair} {self.timeframe} short signal @ ${price}, {short_inval = }, {vol_delta}"
            print(now, note)
            # pb.push_note(title=now, body=note)
            if last['inside_bar']:
                self.candle = 'inside_bar'
            elif last['bearish_doji']:
                self.candle = 'doji'
            elif last['down_bar']:
                self.candle = 'down_bar'

    def trail_stop(self, method, mult=1):
        if method == 'fractals':
            self.williams_fractals()
            self.df['inval_high'] = self.df.fractal_high.interpolate('pad')
            self.df['inval_low'] = self.df.fractal_low.interpolate('pad')
        elif method == 'atr':
            self.calc_atr(10)
            self.df['inval_high'] = self.df.vwma + (self.df.atr * mult)
            self.df['inval_low'] = self.df.vwma - (self.df.atr * mult)

        last_idx = self.df.index[-1]
        if self.position > 0:
            new_inval = self.df.at[last_idx, 'inval_low']
            if self.inval < new_inval:
                print(f'moved {self.pair} stop up to {new_inval}')
            return max(self.inval, new_inval)
        elif self.position < 0:
            new_inval = self.df.at[last_idx, 'inval_high']
            if self.inval > new_inval:
                print(f'moved {self.pair} stop down to {new_inval}')
            return min(self.inval, new_inval)

    def stopped(self, last):
        """this will probably need to be rewritten when its actually live, to check the exchange for stop orders"""
        if self.position > 0:
            return last['close'] < self.inval
        elif self.position < 0:
            return last['close'] > self.inval

    def run_calcs(self, data, ohlc_data):
        self.make_dataframe(ohlc_data)
        if self.position == 0: # check for entry signals
            # print(1)
            self.inside_bars()
            # print(2)
            self.doji()
            # print(3)
            self.bar_direction()
            # print(4)
            self.ema_trend()
            # print(5)
            self.trend_rate()
            # print(6)
            self.entry_signals()
            # print(7)
            self.open_trade()
            # print(8)
        else: # manage position
            last = self.df.to_dict('records')[-1]
            if self.position > 0:
                pos = 'long'
                # print(f"run calcs manage pos: {self.entry = }")
                pnl = (last['close'] - self.entry) / self.entry
            else:
                pos = 'short'
                pnl = (self.entry - last['close']) / self.entry

            if self.stopped(last):
                now = datetime.datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
                now_stamp = datetime.datetime.now(timezone.utc)
                duration = f"{(now_stamp - self.open_time).seconds//60} minutes"
                # print(f"run calcs if stopped: {self.r = }")
                note = (f"{self.short_name} {pos} stopped out @ {pnl:.3%} PnL ({pnl/self.r:.1f}R), "
                        f"duration: {duration}, signal candle: {self.candle}")
                print(now, note)
                pb.push_note(title=now, body=note)

                self.position = 0
                self.inval = None
                self.entry = None
                self.r = None
                self.open_time = None
                self.candle = None
                print('all attributes reset')
            else:
                self.inval = self.trail_stop('atr')
                last_idx = self.df.index[-1]
                # print(f"run calcs if not stopped: {self.df.at[last_idx, 'close'] = }")
                dist = abs(self.df.at[last_idx, 'close'] - self.inval) / self.df.at[last_idx, 'close']
                # print(f"run calcs if not stopped: {self.r = }")
                print(f"{self.short_name} currently in {pos} position @ {pnl:.3%} PnL ({pnl/self.r:.1f}R), "
                      f"{dist:.3%} ({dist/self.r:.1f}R) from trailing stop")
        # print(self.df.tail())

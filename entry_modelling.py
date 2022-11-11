import pandas as pd
import numpy as np
from pprint import pprint
import keys
from binance import Client
from pathlib import Path
import time
import indicators as ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from typing import Union, List, Tuple, Dict, Set, Optional, Any
from config import not_pairs
import itertools as it

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
client = Client(keys.bPkey, keys.bSkey)


def get_pairs(quote: str = 'USDT', market: str = 'SPOT') -> List[str]:
    """returns all active pairs for a given quote currency. possible values for
    quote are USDT, BTC, BNB etc. possible values for market are SPOT or CROSS"""

    if market == 'SPOT':
        info = client.get_exchange_info()
        symbols = info.get('symbols')
        pairs = []
        for sym in symbols:
            right_quote = sym.get('quoteAsset') == quote
            right_market = market in sym.get('permissions')
            trading = sym.get('status') == 'TRADING'
            allowed = sym.get('symbol') not in not_pairs
            if right_quote and right_market and trading and allowed:
                pairs.append(sym.get('symbol'))
    elif market == 'CROSS':
        pairs = []
        info = client.get_margin_all_pairs()
        for i in info:
            if i.get('quote') == quote:
                pairs.append(i.get('symbol'))
    return pairs


def get_ohlc(pair):
    print('runnning get_ohlc')
    try:
        # pair_path = ohlc_path / f"{pair}.pkl"
        pair_path = f"{pair}_1m.pkl"
        return pd.read_pickle(pair_path)
    except (FileNotFoundError, OSError):
        klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, '1 year ago UTC')
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = df['timestamp'] * 1000000
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
                      'taker buy quote vol', 'ignore'], axis=1)

        return df


def update_ohlc(pair: str, timeframe: str, old_df: pd.DataFrame) -> pd.DataFrame:
    print('runnning update_ohlc')
    """takes an ohlc dataframe, works out when the data ends, then requests from
    binance all data from the end to the current moment. It then joins the new
    data onto the old data and returns the updated dataframe"""

    # client = Client(keys.bPkey, keys.bSkey)
    tf = {'1m': Client.KLINE_INTERVAL_1MINUTE,
          '5m': Client.KLINE_INTERVAL_5MINUTE,
          '15m': Client.KLINE_INTERVAL_15MINUTE,
          '30m': Client.KLINE_INTERVAL_30MINUTE,
          '1h': Client.KLINE_INTERVAL_1HOUR,
          '4h': Client.KLINE_INTERVAL_4HOUR,
          '6h': Client.KLINE_INTERVAL_6HOUR,
          '8h': Client.KLINE_INTERVAL_8HOUR,
          '12h': Client.KLINE_INTERVAL_12HOUR,
          '1d': Client.KLINE_INTERVAL_1DAY,
          '3d': Client.KLINE_INTERVAL_3DAY,
          '1w': Client.KLINE_INTERVAL_1WEEK,
          }
    old_end = int(old_df.at[len(old_df) - 1, 'timestamp'].timestamp()) * 1000
    klines = client.get_klines(symbol=pair, interval=tf.get(timeframe),
                               startTime=old_end)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
            'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    df_new = pd.concat([old_df[:-1], df], copy=True, ignore_index=True)
    return df_new


def ohlc_1yr(pair):
    fp = Path(f"bin_ohlc_1m/{pair}_1m.pkl")
    if fp.exists():
        df = pd.read_pickle(fp)
        df = update_ohlc(pair, '1m', df)
        df = df.tail(525600)
        df = df.reset_index(drop=True)
    else:
        df = get_ohlc(pair)

    df.to_pickle(fp)

    return df


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """resamples a dataframe and resets the datetime index"""

    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum',
                                                     'vwma': 'last'
                                                     })
    df = df.dropna(how='any')
    df = df.reset_index()  # don't use drop=True because i want the
    # timestamp index back as a column

    return df


def ib_signals(df, side, z, bars, source, ema_len):
    df = ind.inside_bars(df)

    df = ind.ema_trend(df, ema_len)

    df = ind.trend_rate(df, z, bars, source)

    if side == 'long':
        df['long_signal_ib'] = df.inside_bar & df.trend_down & df.ema_up
    else:
        df['short_signal_ib'] = df.inside_bar & df.trend_up & df.ema_down

    return df


def doji_signals(df, side, z, bars, source, ema_len):
    df = ind.doji(df)

    df = ind.ema_trend(df, ema_len)

    df = ind.trend_rate(df, z, bars, source)

    if side == 'long':
        df['long_signal_doji'] = (df.bullish_doji > 0.5) & df.trend_down & df.ema_up
    else:
        df['short_signal_doji'] = (df.bearish_doji > 0.5) & df.trend_up & df.ema_down

    return df


def bbb_signals(df, side, z, bars, source, ema_len):
    df = ind.bull_bear_bar(df)

    df = ind.ema_trend(df, ema_len)

    df = ind.trend_rate(df, z, bars, source)

    if side == 'long':
        df['long_signal_bbb'] = df.bullish_bar & df.trend_down & df.ema_up
    else:
        df['short_signal_bbb'] = df.bearish_bar & df.trend_up & df.ema_down

    return df


def project_pnl(df, side, method):
    # fetch fees for pair

    # work out a vectorised way to do this for very row in the dataframe:

    # calculate r by setting init stop based on last 2 bars ll/hh
    # trail stop by atr
    # trail stop by fractals
    # set oco orders for 1R
    # set oco orders for 2R
    # set oco orders for 3R
    pass


# set conditions
# pairs = get_pairs()
pairs = ['BTCUSDT']
# timeframes = {'5min': 5, '15min': 15, '30min': 30, '1h': 60}
timeframes = {'5min': 5}
# sides = ['long', 'short']
sides = ['long']

# ema_lengths = [200, 400, 600, 800, 1000]
ema_lengths = [600]
# z_scores = [1, 1.5, 2, 2.5, 3]
z_scores = [2]
# bars = [5, 10, 15, 20]
bars = [10]
# methods = ['atr', 'fractal', 'oco_1', 'oco_2', 'oco_3']
methods = ['atr']

# import and prepare data
for pair in pairs:
    df_orig = ohlc_1yr(pair)
    print(df_orig.head())
    for tf in timeframes.keys():
        data = df_orig.copy()
        # data = hidden_flow(data, 100)
        data = ind.vwma(data, timeframes[tf])
        data = resample(data, tf)
        for side, z_score, bar, length, method in it.product(sides, z_scores, bars, ema_lengths, methods):
            print(f"Testing {pair} {tf} {side} trades, {z_score = }, {bar = }, {length = }, {method = }")
            # create features
            data = ib_signals(data, side, z_score, bar, 'vwma', length)
            data = doji_signals(data, side, z_score, bar, 'vwma', length)
            data = bbb_signals(data, side, z_score, bar, 'vwma', length)

            # create dependent variable (pnl)
            data = project_pnl(data, side, method)

            # split data into features and labels
            X = data.drop('pnl', axis=1)
            y = data.pnl

            # split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

            # instantiate and train model
            rf_model = RandomForestClassifier()

            param_dict = dict(
                n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)],
                max_features = ['auto', 'sqrt'],
                max_depth = [2, 4],
                min_samples_split = [2, 5],
                min_samples_leaf = [1, 2],
                bootstrap = [True, False]
                )
            rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_dict, cv=3, verbose=2, n_jobs=4)
            rf_grid.fit(X_train, y_train)

            print(rf_grid.best_params_)

            print(f"Train Accuracy - : {rf_grid.score(X_train, y_train):.3f}")
            print(f"Test Accuracy - : {rf_grid.score(X_test, y_test):.3f}")

            # rf_model_2 = RandomForestClassifier(bootstrap=True,
            #                                     max_depth=2,
            #                                     max_features='sqrt',
            #                                     min_samples_leaf=1,
            #                                     min_samples_split=2,
            #                                     n_estimators=10)
            # rf_model_2.fit(X_train, y_train)
            #
            # print(f"Train Accuracy - : {rf_model_2.score(X_train, y_train):.3f}")
            # print(f"Test Accuracy - : {rf_model_2.score(X_test, y_test):.3f}")

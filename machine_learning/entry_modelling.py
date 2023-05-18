import time
all_start = time.perf_counter()
import pandas as pd
import numpy as np
from pprint import pprint
import keys
from binance import Client
from pathlib import Path
import indicators as ind
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import itertools as it
import binance_funcs as funcs
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
client = Client(keys.bPkey, keys.bSkey)

timeframe = '4h'
vwma_lengths = {'1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
vwma_periods = 24 # vwma_lengths just accounts for timeframe resampling, vwma_periods is a multiplier on that
side = 'long'
inval_lookback = 2 # lowest low / highest high for last 2 bars
# exit_method = {'type': 'trail_atr', 'len': 2, 'mult': 2}
exit_method = {'type': 'trail_fractal', 'width': 9, 'atr_spacing': 3}
# exit_method = {'type': 'oco', 'r_multiple': 2}

ohlc_folder = Path('../bin_ohlc_5m')
pairs = [p.stem for p in ohlc_folder.glob('*.parquet')]
pairs = ['BTCUSDT']
trim_ohlc = 1000


def get_data(pair):
    ohlc_path = ohlc_folder / f"{pair}.parquet"
    df = pd.read_parquet(ohlc_path)

    vwma = ind.vwma(df, vwma_lengths[timeframe] * vwma_periods)
    vwma = vwma[int(vwma_lengths[timeframe] / 2)::vwma_lengths[timeframe]].reset_index(drop=True)
    df = funcs.resample_ohlc(timeframe, None, df).tail(len(vwma)).reset_index(drop=True)
    df['vwma'] = vwma

    return df


def trail_atr(df, atr_len, atr_mult):
    pass


def trail_fractal(df_0, width, spacing, side):
    df_0 = ind.williams_fractals(df_0, width, spacing)
    df_0 = df_0.drop(['fractal_high', 'fractal_low'], axis=1).dropna(axis=0).reset_index(drop=True)

    condition = (df_0.open > df_0.frac_low) if side == 'long' else (df_0.open < df_0.frac_high)
    rows = list(df_0.loc[condition].index)
    # print(rows[-5:])
    results = []

    # loop through potential entries
    for row in rows:
        df = df_0[row:row+trim_ohlc].copy().reset_index(drop=True)
        entry_price = df.open.iloc[0]

        if side == 'long':
            df['inval'] = df.frac_low.cummax()
            r_pct = abs(entry_price - df.frac_low.iloc[0]) / entry_price
            exit_row = df.loc[df.low < df.inval].index.min()
        else:
            df['inval'] = df.frac_high.cummin()
            r_pct = abs(entry_price - df.frac_high.iloc[0]) / entry_price
            exit_row = df.loc[df.high > df.inval].index.min()

        if not isinstance(exit_row, np.int64):
            continue

        lifespan = exit_row
        exit_price = df.inval.iloc[exit_row]
        trade_diff = exit_price / entry_price

        pnl_pct = (trade_diff - 1.0015) if side == 'long' else (0.9985 - trade_diff) # accounting for 15bps fees
        pnl_r = pnl_pct / r_pct

        row_data = df.iloc[0].to_dict()

        row_res = dict(
            # idx=row,
            r_pct=r_pct,
            lifespan=lifespan,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r
        )

        results.append(row_data | row_res)

        if lifespan / trim_ohlc > 0.5:
            print("warning: trade lifespans getting close to trimmed ohlc length, increase trim ohlc")

    res_df = pd.DataFrame(results)

    return res_df


def oco(df, r_mult, inval_lb, side):

    # method 1
    # i want to ask how many rows from current_row till row.high / row.low exceeds current_row.stop / current_row.profit
    # then i can compare those umber to find which will be hit first

    # method 2
    # i want to find the index of the first high / low to exceed the profit value and the index of the first low / high
    # to exceed the stop value, then i can see which is first. i can use idxmax and idxmin for this but i first need to
    # use clip to make sure that the first values to exceed my limits will be considered the first min/max value

    # calculate r by setting init stop based on last 2 bars ll/hh
    if side == 'long':
        df['r'] = abs((df.close - df.low.rolling(inval_lb).min()) / df.close).shift(1)
    else:
        df['r'] = abs((df.close - df.high.rolling(inval_lb).max()) / df.close).shift(1)
    pass


def add_features(df):

    df['stoch_vwma_ratio'] = ind.stochastic(df.close / df.vwma, 50)
    df['ema_200'] = df.close.ewm(200).mean()
    df['ema_200_roc'] = df.ema_200.pct_change()
    df['ema_ratio'] = df.close / df.ema_200
    df = ind.ema_breakout(df, 50, 50)

    return df


def project_pnl(df, side, method, inval_lb):
    # fetch fees for pair
    fees = 0.0015

    if method['type'] == 'trail_atr':
        df = trail_atr(df, method['len'], method['mult'])
    if method['type'] == 'trail_fractal':
        df = trail_fractal(df, method['width'], method['atr_spacing'], side)
    if method['type'] == 'oco':
        df = oco(df, method['r_multiple'], inval_lb, side)

    return df


for pair in pairs:
    df = get_data(pair)
    df = add_features(df)
    df = project_pnl(df, side, exit_method, inval_lookback)
    df = df.dropna(axis=0).reset_index(drop=True)
    # print(df.tail(100))

    # split data into features and labels
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'pnl_pct', 'pnl_r', 'ema_200', 'lifespan'], axis=1)
    y = df.pnl_r

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    pipe = Pipeline([
        ('scale', QuantileTransformer()),
        ('model', RandomForestRegressor())
    ])
    print('')
    pprint(pipe.get_params())

    param_dict = dict(
        model__n_estimators = [int(x) for x in np.linspace(start=60, stop=100, num=5)],
        model__max_features = ['sqrt'],
        model__max_depth = [5, 6, 7, 8, 9],
        model__min_samples_split = [2, 3],
        model__min_samples_leaf = [1, 2, 3],
        model__bootstrap = [True, False]
    )
    rf_grid = GridSearchCV(estimator=pipe, param_grid=param_dict, cv=5, n_jobs=4)
    rf_grid.fit(X_train, y_train)

    print('Best Params:')
    print(rf_grid.best_params_)

    print(f"Train Accuracy - : {rf_grid.score(X_train, y_train):.3f}")
    print(f"Test Accuracy - : {rf_grid.score(X_test, y_test):.3f}")

    results = pd.DataFrame(rf_grid.cv_results_).sort_values('')
    print(results)


# # import and prepare data
# for pair in pairs:
#     df_orig = ohlc_1yr(pair)
#     print(df_orig.head())
#     for tf in timeframes.keys():
#         data = df_orig.copy()
#         # data = hidden_flow(data, 100)
#         data['vwma'] = ind.vwma(data, timeframes[tf])
#         data = resample(data, tf)
#         for side, z_score, bar, length, method in it.product(sides, z_scores, bars, ema_lengths, methods):
#             print(f"Testing {pair} {tf} {side} trades, {z_score = }, {bar = }, {length = }, {method = }")
#             # create features
#             data = ib_signals(data, side, z_score, bar, 'vwma', length)
#             data = doji_signals(data, side, z_score, bar, 'vwma', length)
#             data = bbb_signals(data, side, z_score, bar, 'vwma', length)
#
#             # create dependent variable (pnl)
#             data = project_pnl(data, side, method)
#
#             # split data into features and labels
#             X = data.drop('pnl', axis=1)
#             y = data.pnl
#
#             # split into train and test sets
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#
#             # instantiate and train model
#             rf_model = RandomForestClassifier()
#
#             param_dict = dict(
#                 n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)],
#                 max_features = ['auto', 'sqrt'],
#                 max_depth = [2, 4],
#                 min_samples_split = [2, 5],
#                 min_samples_leaf = [1, 2],
#                 bootstrap = [True, False]
#                 )
#             rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_dict, cv=3, verbose=2, n_jobs=4)
#             rf_grid.fit(X_train, y_train)
#
#             print(rf_grid.best_params_)
#
#             print(f"Train Accuracy - : {rf_grid.score(X_train, y_train):.3f}")
#             print(f"Test Accuracy - : {rf_grid.score(X_test, y_test):.3f}")
#
#             # rf_model_2 = RandomForestClassifier(bootstrap=True,
#             #                                     max_depth=2,
#             #                                     max_features='sqrt',
#             #                                     min_samples_leaf=1,
#             #                                     min_samples_split=2,
#             #                                     n_estimators=10)
#             # rf_model_2.fit(X_train, y_train)
#             #
#             # print(f"Train Accuracy - : {rf_model_2.score(X_train, y_train):.3f}")
#             # print(f"Test Accuracy - : {rf_model_2.score(X_test, y_test):.3f}")

all_end = time.perf_counter()
elapsed = all_end - all_start
print(f"Total time taken: {int(elapsed // 60)}m {elapsed % 60:.1f}s")

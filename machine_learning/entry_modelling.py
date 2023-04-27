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
import itertools as it
import binance_funcs as funcs

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
client = Client(keys.bPkey, keys.bSkey)

timeframe = '1h'
vwma_lengths = {'1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
side = 'long'
init_inval = 'll_3' # lowest low for last 3 bars
exit = 'trail_atr_2_2' # len 2 mult 2

ohlc_folder = Path('../bin_ohlc_5m')
pairs = [p.stem for p in ohlc_folder.glob('*.parquet')]
pairs = ['BTCUSDT']

def project_pnl(df, side, method):
    # fetch fees for pair

    # work out a vectorised way to do this for every row in the dataframe:

    # calculate r by setting init stop based on last 2 bars ll/hh
    # trail stop by atr
    # trail stop by fractals
    # set oco orders for 1R
    # set oco orders for 2R
    # set oco orders for 3R
    pass


for pair in pairs:
    ohlc_path = ohlc_folder / f"{pair}.parquet"
    df = pd.read_parquet(ohlc_path)

    vwma = ind.vwma(df, vwma_lengths[timeframe])
    vwma = vwma.iloc[::vwma_lengths[timeframe]]

    df = funcs.resample_ohlc(timeframe, None, df)
    df['vwma'] = vwma
    print(df.tail())


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

import time
import pandas as pd
from pprint import pprint
import keys
from binance import Client
from pathlib import Path
import indicators as ind
import features
import binance_funcs as funcs
import numpy as np
import json

from sklearnex import get_patch_names, patch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline

all_start = time.perf_counter()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
client = Client(keys.bPkey, keys.bSkey)

timeframe = '1h'
vwma_lengths = {'1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
vwma_periods = 24  # vwma_lengths just accounts for timeframe resampling, vwma_periods is a multiplier on that
inval_lookback = 2  # lowest low / the highest high for last 2 bars
# exit_method = {'type': 'trail_atr', 'len': 2, 'mult': 2}
exit_method = {'type': 'trail_fractal', 'width': 9, 'atr_spacing': 3}
# exit_method = {'type': 'oco', 'r_multiple': 2}

ohlc_folder = Path('../bin_ohlc_5m')
# pairs = [p.stem for p in ohlc_folder.glob('*.parquet')]
# pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
trim_ohlc = 1000

# TODO i still have to deal with the unbalanced labels
# TODO also need to investigate scoring algorithms inside the estimator - i need to penalise false positives much more
#  than false negatives
# TODO Using R-denominated pnl might be making prediction more dificult. i might find i get better results with % pnl and
#  then i can use R as a strategy filter further down the line

def rank_pairs(n, start=0):
    with open('../recent_1d_volumes.json', 'r') as file:
        vols = json.load(file)

    vol_sorted_pairs = sorted(vols, key=lambda x: vols[x], reverse=True)

    return vol_sorted_pairs[start:n]


def get_data(pair):
    start = time.perf_counter()
    ohlc_path = ohlc_folder / f"{pair}.parquet"

    if ohlc_path.exists():
        df = pd.read_parquet(ohlc_path)
        # print("Loaded OHLC from file")
    else:
        df = funcs.get_ohlc(pair, '5m', '2 years ago UTC')
        ohlc_folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ohlc_path)
        # print("Downloaded OHLC from internet")

    vwma = ind.vwma(df, vwma_lengths[timeframe] * vwma_periods)
    vwma = vwma[int(vwma_lengths[timeframe] / 2)::vwma_lengths[timeframe]].reset_index(drop=True)
    df = funcs.resample_ohlc(timeframe, None, df).tail(len(vwma)).reset_index(drop=True)
    df['vwma'] = vwma

    elapsed = time.perf_counter() - start
    # print(f"get_data took {int(elapsed // 60)}m {elapsed % 60:.1f}s")

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
        df = df_0[row:row + trim_ohlc].copy().reset_index(drop=True)
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

        pnl_pct = (trade_diff - 1.0015) if side == 'long' else (0.9985 - trade_diff)  # accounting for 15bps fees
        pnl_r = pnl_pct / r_pct
        pnl_cat = 0 if (pnl_r <= 0) else 1

        row_data = df.iloc[0].to_dict()

        row_res = dict(
            # idx=row,
            r_pct=r_pct,
            lifespan=lifespan,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r,
            pnl_cat=pnl_cat
        )

        results.append(row_data | row_res)

        if lifespan / trim_ohlc > 0.5:
            print("warning: trade lifespans getting close to trimmed ohlc length, increase trim ohlc")

    return results


def oco(df, r_mult, inval_lb, side):
    # method 1
    # I want to ask how many rows from current_row till row.high / row.low exceeds current_row.stop / current_row.profit
    # then I can compare those umber to find which will be hit first

    # method 2
    # I want to find the index of the first high / low to exceed the profit value and the index of the first low / high
    # to exceed the stop value, then I can see which is first. I can use idxmax and idxmin for this, but I first need to
    # use clip to make sure that the first values to exceed my limits will be considered the first min/max value

    # calculate r by setting init stop based on last 2 bars ll/hh
    if side == 'long':
        df['r'] = abs((df.close - df.low.rolling(inval_lb).min()) / df.close).shift(1)
    else:
        df['r'] = abs((df.close - df.high.rolling(inval_lb).max()) / df.close).shift(1)
    pass


def add_features(df):
    df['vol_delta_div'] = ind.vol_delta_div(df)
    df['stoch_vwma_ratio_20'] = features.stoch_vwma_ratio(df, 20)
    df['stoch_vwma_ratio_50'] = features.stoch_vwma_ratio(df, 50)
    df['stoch_vwma_ratio_100'] = features.stoch_vwma_ratio(df, 100)
    df['ema_20_roc'] = features.ema_roc(df, 20)
    df['ema_50_roc'] = features.ema_roc(df, 50)
    df['ema_100_roc'] = features.ema_roc(df, 100)
    df['ema_200_roc'] = features.ema_roc(df, 200)
    df['ema_200_ratio'] = features.ema_ratio(df, 200)
    df = ind.ema_breakout(df, 50, 50)
    df = ind.atr(df, 5)
    df = ind.atr(df, 10)
    df = ind.atr(df, 20)
    df = ind.atr(df, 50)
    df['stoch_base_vol_20'] = ind.stochastic(df.base_vol, 20)
    df['stoch_base_vol_50'] = ind.stochastic(df.base_vol, 50)
    df['stoch_base_vol_100'] = ind.stochastic(df.base_vol, 100)
    df['stoch_base_vol_200'] = ind.stochastic(df.base_vol, 200)
    df['stoch_num_trades_20'] = ind.stochastic(df.num_trades, 20)
    df['stoch_num_trades_50'] = ind.stochastic(df.num_trades, 50)
    df['stoch_num_trades_100'] = ind.stochastic(df.num_trades, 100)
    df['stoch_num_trades_200'] = ind.stochastic(df.num_trades, 200)
    df['inside_bar'] = ind.inside_bars(df).shift(1)
    df = features.engulfing(df, 1)
    df = features.doji(df)
    df = features.bull_bear_bar(df)

    return df


def project_pnl(df, side, method, inval_lb) -> list[dict]:
    start = time.perf_counter()

    res_list = []
    if method['type'] == 'trail_atr':
        res_list = trail_atr(df, method['len'], method['mult'])
    if method['type'] == 'trail_fractal':
        res_list = trail_fractal(df, method['width'], method['atr_spacing'], side)
    if method['type'] == 'oco':
        res_list = oco(df, method['r_multiple'], inval_lb, side)

    elapsed = time.perf_counter() - start
    # print(f"project_pnl took {int(elapsed // 60)}m {elapsed % 60:.1f}s")

    return res_list


def train_ml(df):
    # split data into features and labels
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'num_trades',
                 'taker_buy_base_vol', 'taker_buy_quote_vol', 'vwma', 'pnl_pct', 'pnl_r', 'pnl_cat', 'ema_20',
                 'ema_50', 'ema_100', 'ema_200', 'lifespan'], axis=1)
    y = df.pnl_cat

    # print(X.describe())

    # split into train and test sets for hold-out validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    pipe = Pipeline([
        ('scale', QuantileTransformer()),
        ('model', RandomForestClassifier())
    ])
    # pprint(pipe.get_params())

    param_dict = dict(
        model__n_estimators=[int(x) for x in np.linspace(start=50, stop=200, num=7)],
        model__max_features=['sqrt'],
        model__max_depth=[5, 7, 9, 11],
        model__min_samples_split=[2, 3], # never less than 2
        model__min_samples_leaf=[1, 2, 3],
        model__bootstrap=[True, False]
    )
    rf_grid = GridSearchCV(estimator=pipe, param_grid=param_dict, cv=5, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    # print('Best Params:')
    # print(rf_grid.best_params_)

    print(f"Train Accuracy - : {rf_grid.score(X_train, y_train):.3f}")
    print(f"Test Accuracy - : {rf_grid.score(X_test, y_test):.3f}")

    best_features(rf_grid, X_train)


def best_features(grid, X_train):
    # Get the best estimator from the grid search
    best_estimator = grid.best_estimator_

    # Get feature importances from the best estimator
    importances = best_estimator.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame(importances, index=X_train.columns).sort_values(0, ascending=False)
    print(f'Best Features: 1 {imp_df.index[0]}, 2 {imp_df.index[1]}, 3 {imp_df.index[2]}')

    # # Print the top K features
    # print(f"\nTop {top_k} features:")
    # for f in selected_features:
    #     print(f"{X_train.columns[f]}: {importances[f]:.2%}")

side = 'long'
# for side in ['long', 'short']:
group_size = 1
for i in range(0, 3, group_size):
    pairs = rank_pairs(100)
    print(f"\nTesting {side} setups on pairs {pairs[i:i+group_size]}\n")
    all_results = []
    for pair in pairs[i:i+group_size]:
        df = get_data(pair)
        df = add_features(df)
        results = project_pnl(df, side, exit_method, inval_lookback)
        all_results.extend(results)

    res_df = pd.DataFrame(all_results)
    res_df = res_df.dropna(axis=0).reset_index(drop=True)
    # print(f'Fitting Model on pairs {pairs[i:i+group_size]}')
    train_ml(res_df)

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

all_end = time.perf_counter()
elapsed = all_end - all_start
print(f"Total time taken: {int(elapsed // 60)}m {elapsed % 60:.1f}s")

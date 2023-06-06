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
from itertools import product
from collections import Counter
import joblib
from datetime import datetime
import plotly.express as px

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn

patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, make_scorer

# print(get_patch_names())

# TODO i want to try ensembling different models together, either by voting or by stacking
# TODO i also want to try bagging on models that don't normally support bagging like knn or svc
#  so the architecture i would like to work towards would be something like: bagged logistic regression and knn and
#  svc all stacked into a random forest, or maybe those four or others all going into a voting classifier
# TODO to speed everything up, i should try to use nvidia RAPIDS

all_start = time.perf_counter()

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
client = Client(keys.bPkey, keys.bSkey)


def rank_pairs():
    with open('../recent_1d_volumes.json', 'r') as file:
        vols = json.load(file)

    return sorted(vols, key=lambda x: vols[x], reverse=True)


def get_data(pair, timeframe, vwma_periods=24):
    """loads the ohlc data from file or downloads it from binance if necessary, then calculates vwma at the correct
    scale before resampling to the desired timeframe.
    vwma_lengths just accounts for timeframe resampling, vwma_periods is a multiplier on that"""
    start = time.perf_counter()

    ohlc_folder = Path('../bin_ohlc_5m')
    ohlc_path = ohlc_folder / f"{pair}.parquet"

    if ohlc_path.exists():
        df = pd.read_parquet(ohlc_path)
        # print("Loaded OHLC from file")
    else:
        df = funcs.get_ohlc(pair, '5m', '2 years ago UTC')
        ohlc_folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ohlc_path)
        # print("Downloaded OHLC from internet")

    vwma_lengths = {'1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
    vwma = ind.vwma(df, vwma_lengths[timeframe] * vwma_periods)
    vwma = vwma[int(vwma_lengths[timeframe] / 2)::vwma_lengths[timeframe]].reset_index(drop=True)

    df = funcs.resample_ohlc(timeframe, None, df).tail(len(vwma)).reset_index(drop=True)
    df['vwma'] = vwma

    if timeframe == '1h':
        df = df.tail(8760).reset_index(drop=True)

    elapsed = time.perf_counter() - start
    # print(f"get_data took {int(elapsed // 60)}m {elapsed % 60:.1f}s")

    return df


def trail_atr(df, atr_len, atr_mult):
    pass


def trail_fractal(df_0, width, spacing, side, trim_ohlc=1000):
    df_0 = ind.williams_fractals(df_0, width, spacing)
    df_0 = df_0.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1).dropna(
        axis=0).reset_index(drop=True)

    condition = (df_0.open > df_0.frac_low) if side == 'long' else (df_0.open < df_0.frac_high)
    rows = list(df_0.loc[condition].index)
    df_0['trend_age'] = ind.consec_condition(condition)
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
            print(
                f"warning: trade lifespans getting close to trimmed ohlc length ({lifespan / trim_ohlc:.1%}), increase trim ohlc")

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


def add_features(df, tf):
    periods_1d = {'1h': 24, '4h': 6, '12h': 2, '1d': 1}
    periods_1w = {'1h': 168, '4h': 42, '12h': 14, '1d': 7}

    df['vol_delta_pct'] = ind.vol_delta_pct(df).shift(1)
    df['stoch_vwma_ratio_25'] = features.stoch_vwma_ratio(df, 25)
    df['stoch_vwma_ratio_50'] = features.stoch_vwma_ratio(df, 50)
    df['stoch_vwma_ratio_100'] = features.stoch_vwma_ratio(df, 100)
    df['ema_25_roc'] = features.ema_roc(df, 25)
    df['ema_50_roc'] = features.ema_roc(df, 50)
    df['ema_100_roc'] = features.ema_roc(df, 100)
    df['ema_200_roc'] = features.ema_roc(df, 200)
    df['ema_25_ratio'] = features.ema_ratio(df, 25)
    df['ema_50_ratio'] = features.ema_ratio(df, 50)
    df['ema_100_ratio'] = features.ema_ratio(df, 100)
    df['ema_200_ratio'] = features.ema_ratio(df, 200)
    df['hma_25_roc'] = features.hma_roc(df, 25)
    df['hma_50_roc'] = features.hma_roc(df, 50)
    df['hma_100_roc'] = features.hma_roc(df, 100)
    df['hma_200_roc'] = features.hma_roc(df, 200)
    df['hma_25_ratio'] = features.hma_ratio(df, 25)
    df['hma_50_ratio'] = features.hma_ratio(df, 50)
    df['hma_100_ratio'] = features.hma_ratio(df, 100)
    df['hma_200_ratio'] = features.hma_ratio(df, 200)

    df = df.copy()

    df['stoch_base_vol_25'] = ind.stochastic(df.base_vol, 25).shift(1)
    df['stoch_base_vol_50'] = ind.stochastic(df.base_vol, 50).shift(1)
    df['stoch_base_vol_100'] = ind.stochastic(df.base_vol, 100).shift(1)
    df['stoch_base_vol_200'] = ind.stochastic(df.base_vol, 200).shift(1)
    df['stoch_num_trades_25'] = ind.stochastic(df.num_trades, 25).shift(1)
    df['stoch_num_trades_50'] = ind.stochastic(df.num_trades, 50).shift(1)
    df['stoch_num_trades_100'] = ind.stochastic(df.num_trades, 100).shift(1)
    df['stoch_num_trades_200'] = ind.stochastic(df.num_trades, 200).shift(1)
    df['inside_bar'] = ind.inside_bars(df).shift(1)

    df = df.copy()

    df['hour'] = features.hour(df)
    df['hour_180'] = features.hour_180(df)
    df['day_of_week'] = features.day_of_week(df)
    df['day_of_week_180'] = features.day_of_week_180(df)
    # df['week_of_year'] = features.week_of_year(df)
    # df['week_of_year_180'] = features.week_of_year_180(df)
    df['vol_denom_roc_2'] = features.vol_denom_roc(df, 2, 25)
    df['vol_denom_roc_5'] = features.vol_denom_roc(df, 5, 50)
    df['rsi_14'] = ind.rsi(df.close).shift(1)
    df['rsi_25'] = ind.rsi(df.close, 25).shift(1)
    df['rsi_50'] = ind.rsi(df.close, 50).shift(1)
    df['rsi_100'] = ind.rsi(df.close, 100).shift(1)
    df['rsi_200'] = ind.rsi(df.close, 200).shift(1)

    df = df.copy()

    df['roc_1d'] = df.close.pct_change(periods_1d[tf]).shift(1)
    df['roc_1w'] = df.close.pct_change(periods_1w[tf]).shift(1)
    df['log_returns'] = np.log(df.close.pct_change() + 1).shift(1)
    df['kurtosis_6'] = df.log_returns.rolling(6).kurt()
    df['kurtosis_12'] = df.log_returns.rolling(12).kurt()
    df['kurtosis_25'] = df.log_returns.rolling(25).kurt()
    df['kurtosis_50'] = df.log_returns.rolling(50).kurt()
    df['kurtosis_100'] = df.log_returns.rolling(100).kurt()
    df['kurtosis_200'] = df.log_returns.rolling(200).kurt()
    df['skew_6'] = df.log_returns.rolling(6).skew()
    df['skew_12'] = df.log_returns.rolling(12).skew()
    df['skew_25'] = df.log_returns.rolling(25).skew()
    df['skew_50'] = df.log_returns.rolling(50).skew()
    df['skew_100'] = df.log_returns.rolling(100).skew()
    df['skew_200'] = df.log_returns.rolling(200).skew()

    df = df.copy()

    df = features.vol_delta_div(df, 1)
    df = features.vol_delta_div(df, 2)
    df = features.vol_delta_div(df, 3)
    df = features.vol_delta_div(df, 4)
    df = features.ats_z(df, 25)
    df = features.ats_z(df, 50)
    df = features.ats_z(df, 100)
    df = features.ats_z(df, 200)
    df = features.engulfing(df, 1)
    df = features.engulfing(df, 2)
    df = features.engulfing(df, 3)
    df = features.doji(df, 0.5, 2)
    df = features.bull_bear_bar(df)
    df = ind.ema_breakout(df, 12, 25)
    df = ind.ema_breakout(df, 25, 50)
    df = ind.ema_breakout(df, 50, 100)
    df = ind.ema_breakout(df, 100, 200)
    df = features.atr_pct(df, 5)
    df = features.atr_pct(df, 10)
    df = features.atr_pct(df, 25)
    df = features.atr_pct(df, 50)

    df = df.copy()

    return df


def project_pnl(df, side, method, inval_lb) -> pd.DataFrame:
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

    return pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)


def prepare_data(df, split_pct):
    # split data into features and labels
    # df = df.dropna(axis=0).reset_index(drop=True)
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'num_trades',
                 'taker_buy_base_vol', 'taker_buy_quote_vol', 'vwma', 'pnl_pct', 'pnl_r', 'pnl_cat',
                 'atr-25', 'atr-50', 'atr-100', 'atr-200', 'ema_12', 'ema_25', 'ema_50', 'ema_100', 'ema_200',
                 'hma_25', 'hma_50', 'hma_100', 'hma_200', 'lifespan', 'frac_high', 'frac_low', 'inval'],
                axis=1, errors='ignore')
    y = df.pnl_cat
    z = df.pnl_r
    # print(X.describe())

    # print(f"{len(y)} setups to test")

    # split into train and test sets for hold-out validation
    # train_size = int(split_pct*len(X))
    # X_train = X[:train_size, :]
    # X_test = X[train_size:, :]
    # y_train = y[:train_size]
    # y_test = y[train_size:]
    # z_test = z[train_size:]

    cols = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_pct, random_state=11)
    _, _, _, z_test = train_test_split(X, z, train_size=split_pct, random_state=11)

    # column transformation
    transformers = [
        ('minmax', MinMaxScaler(),
         ['vol_delta_pct', 'ema_25_roc', 'ema_50_roc', 'ema_100_roc', 'ema_200_roc', 'hma_25_roc', 'hma_50_roc',
          'hma_100_roc', 'hma_200_roc', 'hour', 'hour_180', 'day_of_week', 'day_of_week_180']),
        ('quantile', QuantileTransformer(),
         ['r_pct', 'ema_25_ratio', 'ema_50_ratio', 'ema_100_ratio', 'ema_200_ratio', 'hma_25_ratio', 'hma_50_ratio',
          'hma_100_ratio', 'hma_200_ratio', 'atr_5_pct', 'atr_10_pct', 'atr_25_pct', 'atr_50_pct'])
    ]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    # # limiting number of unique values in each feature
    # n_limit = 512
    # discretised = [col for col in range(X_train.shape[1]) if np.unique(X_train[:, col]).size > n_limit]
    # discretizer = KBinsDiscretizer(n_bins=n_limit, encode='ordinal', strategy='uniform')
    # X_train[:, discretised] = discretizer.fit_transform(X_train[:, discretised])
    # X_test[:, discretised] = discretizer.transform(X_test[:, discretised])

    return X_train, X_test, y_train, y_test, z_test, cols


def train_knn(X_train, y_train):
    param_dict = dict(
        # model__n_estimators=[100, 200, 300],#[int(x) for x in np.linspace(start=10, stop=200, num=7)],
        n_neighbors=[2, 4, 6, 8, 10],
        weights=['uniform', 'distance']
    )
    # rf_grid = GridSearchCV(estimator=pipe, param_grid=param_dict, scoring='precision', cv=3, n_jobs=-1)
    rf_grid = RandomizedSearchCV(estimator=KNeighborsClassifier(n_jobs=-1),
                                 param_distributions=param_dict,
                                 n_iter=10,
                                 scoring='precision',
                                 cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    return rf_grid


def train_forest(X_train, y_train):
    param_dict = dict(
        # model__n_estimators=[100, 200, 300],#[int(x) for x in np.linspace(start=10, stop=200, num=7)],
        max_features=[4, 6, 8, 10],
        max_depth=[int(x) for x in np.linspace(start=10, stop=20, num=5)],
        min_samples_split=[2, 3, 4],  # must be 2 or more
    )
    fb_scorer = make_scorer(fbeta_score, beta=0.5, zero_division=0)
    # rf_grid = GridSearchCV(estimator=pipe, param_grid=param_dict, scoring='precision', cv=3, n_jobs=-1)
    rf_grid = RandomizedSearchCV(estimator=RandomForestClassifier(class_weight='balanced',
                                                                  n_estimators=300,
                                                                  min_samples_leaf=2),
                                 param_distributions=param_dict,
                                 n_iter=60,
                                 scoring='precision',#fb_scorer, #
                                 cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    return rf_grid


def train_vc(X_train, y_train):
    param_dict = dict(
        knn__n_neighbors=[2, 4, 6, 8, 10],
        knn__weights=['uniform', 'distance'],
        rf__max_features=[4, 6, 8, 10],
        rf__max_depth=[int(x) for x in np.linspace(start=10, stop=20, num=5)],
        rf__min_samples_split=[2, 3, 4],  # must be 2 or more
    )

    vc = VotingClassifier(estimators=
    [
        ('knn', KNeighborsClassifier(n_jobs=-1)),
        ('rf', RandomForestClassifier(class_weight='balanced',
                                      n_estimators=300,
                                      min_samples_leaf=2))
    ], voting='soft')

    rf_grid = RandomizedSearchCV(estimator=vc,
                                 param_distributions=param_dict,
                                 n_iter=120,
                                 scoring='precision',
                                 cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    return rf_grid


def calc_scores(model, X_test, y_test, guess=False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_guess = np.zeros_like(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    acc_guess = accuracy_score(y_test, y_guess)
    cm = confusion_matrix(y_test, y_pred)

    # print(f"Confusion Matrix: TP: {cm[1, 1]}, TN: {cm[0, 0]}, FP: {cm[0, 1]}, FN: {cm[1, 0]}")

    return {
        'precision': precision_score(y_test, y_guess if guess else y_pred, zero_division=0),
        # what % of trades taken were good
        'recall': recall_score(y_test, y_guess if guess else y_pred, zero_division=0),
        # what % of good trades were taken
        'f1': f1_score(y_test, y_guess if guess else y_pred, zero_division=0),  # harmonic mean of precision and recall
        'f_beta': fbeta_score(y_test, y_guess if guess else y_pred, beta=0.5, zero_division=0),
        'auroc': roc_auc_score(y_test, y_guess if guess else y_proba),
        'accuracy_better_than_guess': accuracy > acc_guess,
        'true_pos': cm[1, 1],
        'false_pos': cm[0, 1],
        'true_neg': cm[0, 0],
        'false_neg': cm[1, 0],
    }


def best_features(grid, cols):
    # Get the best estimator from the grid search
    best_estimator = grid.best_estimator_

    # Get feature importances from the best estimator
    importances = best_estimator.feature_importances_
    imp_df = pd.Series(importances, index=cols)  # .sort_values(0, ascending=False)
    # print(f'Best Features: '
    #       f'1 {imp_df.index[0]}: {imp_df.iat[0, 0]:.2%}, '
    #       f'2 {imp_df.index[1]}: {imp_df.iat[1, 0]:.2%}, '
    #       f'3 {imp_df.index[2]}: {imp_df.iat[2, 0]:.2%}, '
    #       f'4 {imp_df.index[3]}: {imp_df.iat[3, 0]:.2%}, '
    #       f'5 {imp_df.index[4]}: {imp_df.iat[4, 0]:.2%}')

    return imp_df


def backtest(model, X_test, y_test, z_test, min_conf=0.75):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame(
        {'labels': y_test,
         'predictions': y_pred,
         'confidence': y_prob,
         'pnl_r': z_test}
    ).reset_index(drop=True)

    fr = 0.01
    start_cash = 1

    results['confidence'] = results.confidence * results.predictions
    results['confidence'] = results.confidence.where(results.confidence >= min_conf, 0)
    results['in_trade'] = results.confidence > 0
    results['open_trade'] = results.in_trade.diff()
    results['trades'] = results.confidence * results.pnl_r * results.open_trade
    results['trade_pnl_mult'] = ((results.trades * fr) + 1).fillna(1)
    results['pnl_curve'] = results.trade_pnl_mult.cumprod() * start_cash

    trades_taken = results.in_trade.sum()
    winners = len(results.loc[results.trades > 0])
    win_rate = winners / trades_taken

    final_pnl = results.pnl_curve.iloc[-1] - 1

    # print(
    #     f"Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, from {trades_taken} trades, {len(results)} signals")

    return {'pnl': final_pnl, 'win_rate': win_rate}


if __name__ == '__main__':
    print(f"Starting at {datetime.now().strftime('%d/%m/%y %H:%M')}")

    features_dict = {}

    inval_lookback = 2  # lowest low / the highest high for last 2 bars
    # exit_method = {'type': 'trail_atr', 'len': 2, 'mult': 2}
    exit_method = {'type': 'trail_fractal', 'width': 11, 'atr_spacing': 15}
    # exit_method = {'type': 'oco', 'r_multiple': 2}

    # pairs = ['ETHUSDT']
    # sides = ['short']
    # timeframes = ['1d']

    sides = ['long', 'short']
    timeframes = {
        '1d': {'frac_widths': [3, 5, 7], 'atr_spacings': [1, 2, 3, 4], 'num_pairs': 60, 'data_len': 50},
        '12h': {'frac_widths': [3, 5, 7], 'atr_spacings': [1, 2, 4, 8], 'num_pairs': 50, 'data_len': 75},
        '4h': {'frac_widths': [3, 5, 7, 9], 'atr_spacings': [1, 2, 4, 8, 16], 'num_pairs': 40, 'data_len': 100},
        '1h': {'frac_widths': [3, 5, 7, 9], 'atr_spacings': [1, 2, 4, 8, 16], 'num_pairs': 30, 'data_len': 200},
    }


    for side, timeframe in product(sides, timeframes):
        print(f"\nTesting {side} {timeframe}")
        loop_start = time.perf_counter()

        frac_widths = timeframes[timeframe]['frac_widths']
        atr_spacings = timeframes[timeframe]['atr_spacings']
        # frac_widths = [3]
        # atr_spacings = [2]
        data_len = timeframes[timeframe]['data_len']
        num_pairs = timeframes[timeframe]['num_pairs']
        pairs = rank_pairs()[:num_pairs]
        print(pairs)

        res_path = Path(f'results/{side}_{timeframe}_top{num_pairs}.parquet')
        if res_path.exists():
            print('Results already present, skipping tests')
            continue

        res_list = []
        for frac_width, spacing in product(frac_widths, atr_spacings):
            all_res = pd.DataFrame()
            for pair in pairs:

                df = get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
                df = add_features(df, timeframe).tail(data_len).reset_index(drop=True)
                # print(f"data length: {len(df)}")

                df_loop = df.copy()
                exit_method['width'] = frac_width
                exit_method['atr_spacing'] = spacing
                res_df = project_pnl(df_loop, side, exit_method, inval_lookback)
                all_res = pd.concat([all_res, res_df], axis=0)

            X_train, X_test, y_train, y_test, z_test, cols = prepare_data(all_res, 0.9)

            try:
                # model = train_knn(X_train, y_train)
                model = train_forest(X_train, y_train)
                # model = train_vc(X_train, y_train)
            except ValueError as e:
                # print(
                #     f"{side}, {timeframe}, {frac_width = }, {spacing = }, ValueError raised, skipping to next test.\n")
                continue

            test_balance = Counter(y_test)
            test_balance = {f"test_{k}": v for k, v in test_balance.items()}

            best_params = model.best_params_
            try:
                scores = calc_scores(model, X_test, y_test)
            except ValueError as e:
                # print(f'ValueError while calculating scores on {pair}, skipping to next test.')
                continue
            # guess_scores = analyse_results(model, X_test, y_test, guess=True)
            imp_df = best_features(model, cols)

            # print(f"\n{side}, {timeframe}, {frac_width = }, {spacing = }, "
            #       f"precision: {scores['precision']:.1%}, "
            #       f"AUC: {scores['auroc']:.1%}, "
            #       f"f beta: {scores['f_beta']:.1%}, "
            #       f"abtg: {scores['accuracy_better_than_guess']}, "
            #       f"pos predictions: {scores['true_pos'] + scores['false_pos']}, "
            #       f"neg predictions: {scores['true_neg'] + scores['false_neg']}"
            #       )
            test_pnl = backtest(model, X_test, y_test, z_test)

            res_dict = dict(
                timeframe=timeframe,
                frac_width=frac_width,
                spacing=spacing,
                side=side,
                feature_1=imp_df.index[0],
                feature_2=imp_df.index[1],
                feature_3=imp_df.index[2],
                feature_4=imp_df.index[3],
                feature_5=imp_df.index[4],
                feature_6=imp_df.index[5],
                feature_7=imp_df.index[6],
                feature_8=imp_df.index[7],
                pos_preds=scores['true_pos'] + scores['false_pos']
            ) | scores | model.best_params_ | test_balance | test_pnl

            res_list.append(res_dict)

        final_results = pd.DataFrame(res_list)
        final_results.to_parquet(path=res_path)
        print(final_results.sort_values('precision').head())

        loop_end = time.perf_counter()
        loop_elapsed = loop_end - loop_start
        print(f"Loop took {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s\n")

    all_end = time.perf_counter()
    elapsed = all_end - all_start
    print(f"\n\nTotal time taken: {int(elapsed // 60)}m {elapsed % 60:.1f}s")

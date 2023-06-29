import time
import pandas as pd
from pprint import pprint
import keys
from binance import Client
from pathlib import Path
import indicators as ind
import machine_learning.features as features
import binance_funcs as funcs
import numpy as np
import json
from itertools import product
from collections import Counter
import joblib
from datetime import datetime
import plotly.express as px
from pyarrow import ArrowInvalid

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn

patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, make_scorer
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler

# print(get_patch_names())

# TODO i want to try ensembling different models together, either by voting or by stacking
# TODO i also want to try bagging on models that don't normally support bagging like knn or svc
#  so the architecture i would like to work towards would be something like: bagged logistic regression and knn and
#  svc all stacked into a random forest, or maybe those four or others all going into a voting classifier
# TODO to speed everything up, i should try to use nvidia RAPIDS

# TODO add features based on alternative data like spread, book imbalance, on-chain metrics etc
# TODO add features from higher timeframes like weekly rsi, weekly s/r channel etc

all_start = time.perf_counter()

# I must remember that i am performing column transformation outside cross-validation. if real-world performance ends up
# being significantly worse than this stage's performance, i should try to get the column transformer inside the cv.

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

    ohlc_folder = Path('/home/ross/coding/modular_trader/bin_ohlc_5m')
    ohlc_path = ohlc_folder / f"{pair}.parquet"

    if ohlc_path.exists():
        try:
            df = pd.read_parquet(ohlc_path)
            # print("Loaded OHLC from file")
        except (ArrowInvalid, OSError) as e:
            print('Error:\n', e)
            print(f"Problem reading {pair} parquet file, downloading from scratch.")
            ohlc_path.unlink()
            df = funcs.get_ohlc(pair, '5m', '2 years ago UTC')
            df.to_parquet(ohlc_path)
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

        row_data = df_0.iloc[row-1].to_dict()

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
    """i want this function to either set the invalidation by looking at recent lows/highs or by using recent volatility
    as a multiplier"""

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
    df = features.atr_pct(df, 5)
    df = features.atr_pct(df, 10)
    df = features.atr_pct(df, 25)
    df = features.atr_pct(df, 50)
    df = features.ats_z(df, 25)
    df = features.ats_z(df, 50)
    df = features.ats_z(df, 100)
    df = features.ats_z(df, 200)
    df = features.bull_bear_bar(df)
    df = features.channel_mid_ratio(df, 25)
    df = features.channel_mid_ratio(df, 50)
    df = features.channel_mid_ratio(df, 100)
    df = features.channel_mid_ratio(df, 200)
    df = features.channel_mid_width(df, 25)
    df = features.channel_mid_width(df, 50)
    df = features.channel_mid_width(df, 100)
    df = features.channel_mid_width(df, 200)
    df = features.daily_open_ratio(df)
    df = features.daily_roc(df, tf)
    df = features.day_of_week(df)
    df = features.day_of_week_180(df)
    df = features.doji(df, 0.5, 2, weighted=True)
    df = features.doji(df, 1, 2, weighted=True)
    df = features.doji(df, 2, 2, weighted=True)
    df = features.doji(df, 0.5, 2, weighted=False)
    df = features.engulfing(df, 1)
    df = features.engulfing(df, 2)
    df = features.engulfing(df, 3)
    df = features.ema_breakout(df, 12, 25)
    df = features.ema_breakout(df, 25, 50)
    df = features.ema_breakout(df, 50, 100)
    df = features.ema_breakout(df, 100, 200)
    df = features.ema_roc(df, 25)
    df = features.ema_roc(df, 50)
    df = features.ema_roc(df, 100)
    df = features.ema_roc(df, 200)
    df = features.ema_ratio(df, 25)
    df = features.ema_ratio(df, 50)
    df = features.ema_ratio(df, 100)
    df = features.ema_ratio(df, 200)
    df = features.hma_roc(df, 25)
    df = features.hma_roc(df, 50)
    df = features.hma_roc(df, 100)
    df = features.hma_roc(df, 200)
    df = features.hma_ratio(df, 25)
    df = features.hma_ratio(df, 50)
    df = features.hma_ratio(df, 100)
    df = features.hma_ratio(df, 200)
    df = features.hour(df)
    df = features.hour_180(df)
    df = features.inside_bar(df)
    df = features.kurtosis(df, 6)
    df = features.kurtosis(df, 12)
    df = features.kurtosis(df, 25)
    df = features.kurtosis(df, 50)
    df = features.kurtosis(df, 100)
    df = features.kurtosis(df, 200)
    df = features.prev_daily_open_ratio(df)
    df = features.prev_daily_high_ratio(df)
    df = features.prev_daily_low_ratio(df)
    df = features.rsi(df, 14)
    df = features.rsi(df, 25)
    df = features.rsi(df, 50)
    df = features.rsi(df, 100)
    df = features.rsi(df, 200)
    df = features.skew(df, 6)
    df = features.skew(df, 12)
    df = features.skew(df, 25)
    df = features.skew(df, 50)
    df = features.skew(df, 100)
    df = features.skew(df, 200)
    df = features.stoch_base_vol(df, 25)
    df = features.stoch_base_vol(df, 50)
    df = features.stoch_base_vol(df, 100)
    df = features.stoch_base_vol(df, 200)
    df = features.stoch_num_trades(df, 25)
    df = features.stoch_num_trades(df, 50)
    df = features.stoch_num_trades(df, 100)
    df = features.stoch_num_trades(df, 200)
    df = features.stoch_vwma_ratio(df, 25)
    df = features.stoch_vwma_ratio(df, 50)
    df = features.stoch_vwma_ratio(df, 100)
    df = features.vol_delta_div(df, 1)
    df = features.vol_delta_div(df, 2)
    df = features.vol_delta_div(df, 3)
    df = features.vol_delta_div(df, 4)
    df = features.vol_delta_pct(df)
    df = features.vol_denom_roc(df, 2, 25)
    df = features.vol_denom_roc(df, 5, 50)
    # df = features.week_of_year(df)
    # df = features.week_of_year_180(df)
    df = features.weekly_roc(df, tf)

    return df


def project_pnl(df, side, method) -> pd.DataFrame:
    start = time.perf_counter()

    res_list = []
    if method['type'] == 'trail_atr':
        res_list = trail_atr(df, method['len'], method['mult'])
    if method['type'] == 'trail_fractal':
        res_list = trail_fractal(df, method['width'], method['atr_spacing'], side)
    if method['type'] == 'oco':
        res_list = oco(df, method['r_multiple'], side)

    elapsed = time.perf_counter() - start
    # print(f"project_pnl took {int(elapsed // 60)}m {elapsed % 60:.1f}s")

    return pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)


def features_labels_split(df):
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'num_trades',
                 'taker_buy_base_vol', 'taker_buy_quote_vol', 'vwma', 'pnl_pct', 'pnl_r', 'pnl_cat',
                 'atr-25', 'atr-50', 'atr-100', 'atr-200', 'ema_12', 'ema_25', 'ema_50', 'ema_100', 'ema_200',
                 'hma_25', 'hma_50', 'hma_100', 'hma_200', 'lifespan', 'frac_high', 'frac_low', 'inval', 'daily_open',
                 'prev_daily_open', 'prev_daily_high', 'prev_daily_low'],
                axis=1, errors='ignore')
    y = df.pnl_cat
    z = df.pnl_r

    return X, y, z


def tt_split_rand(X: pd.DataFrame, y: pd.Series, z: pd.Series, split_pct: float) -> tuple:
    """split into train and test sets for hold-out validation"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_pct, random_state=11)
    _, _, _, z_test = train_test_split(X, z, train_size=split_pct, random_state=11)

    return X_train, X_test, y_train, y_test, z_test


def tt_split_bifurcate(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, z: pd.Series | np.ndarray, split_pct: float) -> tuple:
    """split into train and test sets for hold-out validation"""
    train_size = int(split_pct*len(X))
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
    else:
        X_train = X[:train_size, :]
        X_test = X[train_size:, :]
    if isinstance(y, pd.Series):
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
    else:
        y_train = y[:train_size, :]
        y_test = y[train_size:, :]
    if isinstance(z, pd.Series):
        z_test = z.iloc[train_size:]
    else:
        z_test = z[train_size:, :]

    return X_train, X_test, y_train, y_test, z_test


def tt_split_idx(X, y, z, train_idxs, test_idxs):
    """split into train and test sets for hold-out validation"""

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idxs[0]:train_idxs[1]+1]
        X_test = X.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        X_train = X[train_idxs[0]:train_idxs[1] + 1, :]
        X_test = X[test_idxs[0]:test_idxs[1] + 1, :]
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idxs[0]:train_idxs[1]+1]
        y_test = y.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        y_train = y[train_idxs[0]:train_idxs[1] + 1]
        y_test = y[test_idxs[0]:test_idxs[1] + 1]
    if isinstance(z, pd.Series):
        z_test = z.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        z_test = z[test_idxs[0]:test_idxs[1] + 1]

    return X_train, X_test, y_train, y_test, z_test


def transform_columns(X_train, X_test):
    # column transformation
    min_max_cols = ['vol_delta_pct', 'ema_25_roc', 'ema_50_roc', 'ema_100_roc', 'ema_200_roc', 'hma_25_roc', 'hma_50_roc',
          'hma_100_roc', 'hma_200_roc', 'hour', 'hour_180', 'day_of_week', 'day_of_week_180', 'chan_mid_ratio_25',
          'chan_mid_ratio_50', 'chan_mid_ratio_100', 'chan_mid_ratio_200', 'chan_mid_width_25', 'chan_mid_width_50',
          'chan_mid_width_100', 'chan_mid_width_200']
    min_max_cols = [mmc for mmc in min_max_cols if mmc in X_train.columns]

    quant_cols = ['r_pct', 'ema_25_ratio', 'ema_50_ratio', 'ema_100_ratio', 'ema_200_ratio', 'hma_25_ratio', 'hma_50_ratio',
          'hma_100_ratio', 'hma_200_ratio', 'atr_5_pct', 'atr_10_pct', 'atr_25_pct', 'atr_50_pct']
    quant_cols = [qc for qc in quant_cols if qc in X_train.columns]

    transformers = [('minmax', MinMaxScaler(), min_max_cols), ('quantile', QuantileTransformer(), quant_cols)]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    feature_cols = [f.split('__')[1] for f in ct.get_feature_names_out()]

    return X_train, X_test, feature_cols


def train_knn(X_train, y_train):
    param_dict = dict(
        # model__n_estimators=[100, 200, 300],#[int(x) for x in np.linspace(start=10, stop=200, num=7)],
        estimator__n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        estimator__metric=['euclidean', 'manhattan', 'minkowski'],
        estimator__weights=['uniform', 'distance']
    )
    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
    model = KNeighborsClassifier(n_jobs=-1)
    rf_grid = RandomizedSearchCV(estimator=model,
                                 param_distributions=param_dict,
                                 n_iter=60,
                                 scoring=fb_scorer,
                                 cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    return rf_grid


def train_rfc(X_train, y_train):
    param_dict = dict(
        estimator__max_features=[4, 6, 8, 10],
        estimator__max_depth=[int(x) for x in np.linspace(start=15, stop=30, num=4)],
        estimator__min_samples_split=[2, 3, 4],  # must be 2 or more
    )
    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
    model = RandomForestClassifier(class_weight='balanced', n_estimators=300, min_samples_leaf=2)
    rf_grid = RandomizedSearchCV(estimator=model,
                                 param_distributions=param_dict,
                                 n_iter=60,
                                 scoring=fb_scorer,
                                 cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    return rf_grid


def train_gbc(X_train, y_train):
    param_dict = dict(
        # max_features=[2, 4, 8, 16, 32],
        estimator__max_depth=[int(x) for x in np.linspace(start=5, stop=20, num=4)],
        estimator__min_samples_split=[2, 4, 8],  # must be 2 or more
        estimator__learning_rate=[0.05, 0.1],
        estimator__subsample=[0.125, 0.25, 0.5, 1.0]
    )
    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
    model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5)
    rf_grid = RandomizedSearchCV(estimator=model,
                                 param_distributions=param_dict,
                                 scoring=fb_scorer,
                                 n_iter=120, # full test = 3600
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
    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

    # # Get MDI feature importances from the best estimator
    # importances = best_estimator.feature_importances_
    # imp_df = pd.Series(importances, index=cols).sort_values(ascending=False)

    # get permutation-based feature importances
    importances = permutation_importance(best_estimator, X_test, y_test, scoring=fb_scorer, n_repeats=100, random_state=42, n_jobs=-1)
    imp_df = pd.Series(importances.importances_mean, index=cols).sort_values(ascending=False)

    return imp_df


def backtest(y_test, y_pred, y_prob, z_test, track_perf = False, min_conf=0.75, fr=0.01):

    results = pd.DataFrame(
        {'labels': y_test,
         'predictions': y_pred,
         'confidence': y_prob,
         'pnl_r': z_test}
    ).reset_index(drop=True)

    start_cash = 1

    results['pnl_r'] = results.pnl_r.clip(lower=-1)
    results['confidence'] = results.confidence * results.predictions
    results['confidence'] = results.confidence.where(results.confidence >= min_conf, 0)
    results['in_trade'] = results.confidence > 0
    results['open_trade'] = results.in_trade & results.in_trade.diff()
    results['trades'] = results.confidence * results.pnl_r * results.open_trade
    if track_perf:
        results['perf_score'] = (results.pnl_r > 0).astype(int).rolling(100).mean().shift()
        results['trade_pnl_mult'] = ((results.trades * fr * results.perf_score) + 1).fillna(1)
    else:
        results['trade_pnl_mult'] = ((results.trades * fr) + 1).fillna(1)
    results['pnl_curve'] = (results.trade_pnl_mult.cumprod() * start_cash) - 1

    return results


if __name__ == '__main__':
    print(f"Starting at {datetime.now().strftime('%d/%m/%y %H:%M')}")

    features_dict = {}

    # exit_method = {'type': 'trail_atr', 'len': 2, 'mult': 2}
    exit_method = {'type': 'trail_fractal', 'width': 11, 'atr_spacing': 15}
    # exit_method = {'type': 'oco', 'r_multiple': 2}

    # pairs = ['ETHUSDT']
    sides = ['short']
    # timeframes = ['1d']

    algorithms = [
        # 'knn',
        'rfc', 'gbc']
    sides = ['long', 'short']
    timeframes = {
        # '1d': {'frac_widths': [3, 5], 'atr_spacings': [1, 2], 'num_pairs': 100, 'data_len': 100},
        # '12h': {'frac_widths': [3, 5], 'atr_spacings': [1, 2], 'num_pairs': 66, 'data_len': 150},
        # '4h': {'frac_widths': [3, 5], 'atr_spacings': [1, 2], 'num_pairs': 50, 'data_len': 200},
        '1h': {'frac_widths': [3, 5], 'atr_spacings': [1, 2], 'num_pairs': 25, 'data_len': 400},
    }


    for algo, balanced, side, timeframe in product(algorithms, [True, False], sides, timeframes):
        print(f"\nTesting {algo} {'balanced' if balanced else 'unbalanced'} {side} {timeframe}")
        loop_start = time.perf_counter()

        frac_widths = timeframes[timeframe]['frac_widths']
        atr_spacings = timeframes[timeframe]['atr_spacings']
        # frac_widths = [3]
        # atr_spacings = [2]
        data_len = timeframes[timeframe]['data_len']
        num_pairs = timeframes[timeframe]['num_pairs']
        pairs = rank_pairs()[:num_pairs]
        # print(pairs)

        res_folders = Path(f"sfs/{algo}_results/{'balanced' if balanced else 'unbalanced'}")
        res_folders.mkdir(parents=True, exist_ok=True)
        res_path = Path(f"{res_folders}/{side}_{timeframe}_top{num_pairs}.parquet")
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
                res_df = project_pnl(df_loop, side, exit_method)
                all_res = pd.concat([all_res, res_df], axis=0)

            X, y, z = features_labels_split(all_res)
            X_train, X_test, y_train, y_test, z_test = tt_split_bifurcate(X, y, z, 0.75)
            X_train, X_test, cols = transform_columns(X_train, X_test)

            # balancing classes/prototype selection
            if balanced:
                rus = RandomUnderSampler(random_state=0)
                X_train, y_train = rus.fit_resample(X_train, y_train)

            if y_test.value_counts().loc[1] < 30:
                print(f'{side} {timeframe} {frac_width} {spacing} '
                      f'Not enough positive values to reliably predict, skipping training')
                continue

            # print(f"Fitting model for {algo} {'balanced' if balanced else 'unbalanced'} {frac_width = } {spacing = }. "
            #       f"{len(y_train)} observations in training set.")

            try:
                train_start = time.perf_counter()
                if algo == 'knn':
                    model = train_knn(X_train, y_train)
                elif algo == 'rfc':
                    model = train_rfc(X_train, y_train)
                elif algo == 'gbc':
                    model = train_gbc(X_train, y_train)
                # model = train_vc(X_train, y_train)
                train_end = time.perf_counter()
                train_elapsed = train_end - train_start
                # print(f"Training time taken: {int(train_elapsed // 60)}m {train_elapsed % 60:.1f}s")
            except ValueError as e:
                print(
                    f"{side}, {timeframe}, {frac_width = }, {spacing = }, ValueError raised, skipping to next test.\n")
                print(e.args)
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
            # print(imp_df.head(8))

            # print(f"\n{side}, {timeframe}, {frac_width = }, {spacing = }, "
            #       f"precision: {scores['precision']:.1%}, "
            #       f"AUC: {scores['auroc']:.1%}, "
            #       f"f beta: {scores['f_beta']:.1%}, "
            #       f"abtg: {scores['accuracy_better_than_guess']}, "
            #       f"pos predictions: {scores['true_pos'] + scores['false_pos']}, "
            #       f"neg predictions: {scores['true_neg'] + scores['false_neg']}"
            #       )

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            bt_results = backtest(y_test, y_pred, y_prob, z_test)
            trades_taken = bt_results.in_trade.sum()
            winners = len(bt_results.loc[bt_results.trades > 0])
            if trades_taken:
                win_rate = winners / trades_taken
            else:
                win_rate = 0
            mean_pnl = bt_results.trades.loc[bt_results.open_trade.astype(bool)].mean()
            med_pnl = bt_results.trades.loc[bt_results.open_trade.astype(bool)].median()
            final_pnl = bt_results.pnl_curve.iloc[-1]
            # print(f"Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, from {trades_taken} trades, "
            #       f"{len(bt_results)} signals")

            res_dict = dict(
                timeframe=timeframe,
                frac_width=frac_width,
                spacing=spacing,
                side=side,
                pnl=final_pnl,
                mean_trade=mean_pnl,
                med_trade=med_pnl,
                win_rate=win_rate,
                trades_taken=trades_taken,
                pos_preds=scores['true_pos'] + scores['false_pos'],
                feature_1=imp_df.index[0],
                imp_1=imp_df.iloc[0],
                feature_2=imp_df.index[1],
                imp_2=imp_df.iloc[1],
                feature_3=imp_df.index[2],
                imp_3=imp_df.iloc[2],
                feature_4=imp_df.index[3],
                imp_4=imp_df.iloc[3],
                feature_5=imp_df.index[4],
                imp_5=imp_df.iloc[4],
                feature_6=imp_df.index[5],
                imp_6=imp_df.iloc[5],
                feature_7=imp_df.index[6],
                imp_7=imp_df.iloc[6],
                feature_8=imp_df.index[7],
                imp_8=imp_df.iloc[7]
            ) | scores | model.best_params_ | test_balance

            res_list.append(res_dict)

        final_results = pd.DataFrame(res_list)
        final_results.to_parquet(path=res_path)
        try:
            print(final_results.loc[final_results.trades_taken > 30].sort_values('pnl', ascending=False).head(1))
        except KeyError:
            print("KeyError raised, probably an empty results df. moving on")
            continue

        loop_end = time.perf_counter()
        loop_elapsed = loop_end - loop_start
        print(f"Loop took {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s\n")

    all_end = time.perf_counter()
    elapsed = all_end - all_start
    print(f"\n\nTotal time taken: {int(elapsed // 60)}m {elapsed % 60:.1f}s")


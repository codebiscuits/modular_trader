import time
import pandas as pd
from pathlib import Path
from mt.resources import binance_funcs as funcs, indicators as ind, features as features
import numpy as np
import json
from pyarrow import ArrowInvalid
import logging
from binance import Client
import binance.exceptions as bx
import mt.resources.keys as keys
from datetime import datetime, timezone
import joblib

if not Path('/pi_2.txt').exists():
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# import lightgbm as lgbm
import xgboost as xgb
from optuna import logging as op_logging, integration, pruners, create_study
from optuna.samplers import TPESampler

logger = logging.getLogger('ml_funcs')
logger.setLevel(logging.DEBUG)
scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

def init_client(max_retries: int = 360, delay: int = 5):
    for i in range(max_retries):
        try:
            client = Client(keys.bPkey, keys.bSkey)
            if i > 0:
                print(f'initialising binance client worked on attempt number {i + 1}')
            return client
        except bx.BinanceAPIException as e:
            if e.code != -3044:
                raise e
            print(f"System busy, retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Max retries exceeded. Request still failed after {max_retries} attempts.")


def save_models(strategy, params, sel_method, num_pairs, side, tf, data_len, selected, pairs, model, scaler, validity):
    folder = Path(f"/home/ross/coding/modular_trader/machine_learning/"
                  f"models/{strategy}_{params}/{sel_method}_{num_pairs}")
    pi_folder = Path(f"/home/ross/coding/pi_2/modular_trader/machine_learning/"
                      f"models/{strategy}_{params}/{sel_method}_{num_pairs}")
    model_file = f"{side}_{tf}_model_1a.sav"
    model_info = f"{side}_{tf}_info_1a.json"
    scaler_file = f"{side}_{tf}_scaler_1a.sav"

    info_dict = {'data_length': data_len, 'features': selected, 'pair_selection': sel_method,
                 'pairs': pairs, 'created': int(datetime.now(timezone.utc).timestamp()), 'validity': validity}

    # save local copy
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / model_file)
    info_path = folder / model_info
    info_path.touch(exist_ok=True)
    with open(info_path, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # save on pi
    pi_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, pi_folder / model_file)
    info_path_pi = pi_folder / model_info
    info_path_pi.touch(exist_ok=True)
    with open(info_path_pi, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = pi_folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)


def rank_pairs(selection):
    with open(f'../recent_{selection}.json', 'r') as file:
        vols = json.load(file)

    return sorted(vols, key=lambda x: vols[x], reverse=True)


def get_data(pair, timeframe, vwma_periods=24):
    """loads the ohlc data from file or downloads it from binance if necessary, then calculates vwma at the correct
    scale before resampling to the desired timeframe.
    vwma_lengths just accounts for timeframe resampling, vwma_periods is a multiplier on that"""

    ohlc_folder = Path('../../bin_ohlc_5m')
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

    vwma_lengths = {'15m': 3, '30m': 6, '1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
    vwma = ind.vwma(df, vwma_lengths[timeframe] * vwma_periods)
    vwma = vwma[int(vwma_lengths[timeframe] / 2)::vwma_lengths[timeframe]].reset_index(drop=True)

    df = funcs.resample_ohlc(timeframe, None, df).tail(len(vwma)).reset_index(drop=True)
    df['vwma'] = vwma

    if timeframe == '1h':
        df = df.tail(8760).reset_index(drop=True)

    return df


def get_margin_pairs(method, num_pairs):
    start_pair = 0
    pairs = rank_pairs(method)

    with open('/home/ross/coding/modular_trader/ohlc_lengths.json', 'r') as file:
        ohlc_lengths = json.load(file)

    client = init_client()
    exc_info = client.get_exchange_info()
    symbol_margin = {i['symbol']: i['isMarginTradingAllowed'] for i in exc_info['symbols'] if
                     i['quoteAsset'] == 'USDT'}
    pairs = [p for p in pairs if symbol_margin[p] and (ohlc_lengths[p] > 4032)]

    return pairs[start_pair:start_pair + num_pairs]


def feature_selection_1a(X: np.ndarray, y, limit: int, cols: list[str]):
    """uses a combination of methods to select the best performing subset of features. returns the trained selector
    object that can be used to transform a new dataset"""
    fs_start = time.perf_counter()

    # drop any columns with the same value in every row, and adjust cols to match
    keep_cols = np.array(-1 * (np.all(X == X[0, :], axis=0)) + 1, dtype=bool)
    X = X[:, keep_cols]
    cols = [col for i, col in enumerate(cols) if keep_cols[i]]

    # Selection stage 1
    pre_selector_model = GradientBoostingClassifier(random_state=42,
                                                    n_estimators=10000,
                                                    validation_fraction=0.1,
                                                    n_iter_no_change=50,
                                                    subsample=0.5,
                                                    min_samples_split=8,
                                                    max_depth=12,
                                                    learning_rate=0.3)

    selector_1 = SFS(estimator=pre_selector_model, k_features=10, forward=True,
                     floating=False, verbose=0, scoring=scorer, n_jobs=-1)
    selector_2 = SelectKBest(f_classif, k=10)
    selector_3 = SelectKBest(mutual_info_classif, k=10)
    # selector_4 = SelectKBest(chi2, k=8)

    selector_1 = selector_1.fit(X, y)
    selector_2.fit(X, y)
    selector_3.fit(X, y)
    # selector_4.fit(X, y)

    cols_idx_1 = list(selector_1.k_feature_idx_)
    cols_idx_2 = list(selector_2.get_support(indices=True))
    cols_idx_3 = list(selector_3.get_support(indices=True))
    # cols_4 = list(selector_4.get_support(indices=True))
    all_cols_idx = list(set(cols_idx_1 + cols_idx_2 + cols_idx_3))

    # selected_columns = [cols[i] for i in all_cols]
    selected_columns = [col for i, col in enumerate(cols) if i in all_cols_idx]

    X = pd.DataFrame(X[:, all_cols_idx], columns=selected_columns)

    if X.shape[0] > limit:
        X_train, _, y_train, _ = train_test_split(X, y, train_size=limit, random_state=99)
    else:
        X_train, y_train = X, y

    # Selection stage 2
    selector_model = GradientBoostingClassifier(random_state=42,
                                                n_estimators=10000,
                                                validation_fraction=0.1,
                                                n_iter_no_change=50,
                                                subsample=0.5,
                                                min_samples_split=8,
                                                max_depth=12,
                                                learning_rate=0.3)
    selector = SFS(estimator=selector_model,
                   k_features='best',
                   forward=False,
                   floating=True,
                   verbose=0,
                   scoring=scorer,
                   n_jobs=-1)
    selector = selector.fit(X_train, y_train)
    print(f"Number of features selected: {len(selector.k_feature_names_)}")

    # sel_score = selector.score(X_train, y_train)
    # print(f"Model score after feature selection: {sel_score:.1%}")

    fs_end = time.perf_counter()
    fs_elapsed = fs_end - fs_start
    print(f"Feature selection time taken: {int(fs_elapsed // 60)}m {fs_elapsed % 60:.1f}s")

    return selector


def trail_fractal(df_0: pd.DataFrame, width: int, spacing: int, side: str, trim_ohlc: int=1000, r_threshold: float=0.5):
    """r_threshold is how much pnl a trade must make for the model to consider it a profitable trade.
    higher values will train the model to target only the trades which produce higher profits, but will also limit
    the number of true positives to train the model on """

    # identify potential entries
    df_0 = ind.williams_fractals(df_0, width, spacing)
    df_0 = df_0.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1).dropna(
        axis=0).reset_index(drop=True)

    trend_condition = (df_0.open > df_0.frac_low) if side == 'long' else (df_0.open < df_0.frac_high)
    rows = list(df_0.loc[trend_condition].index)
    df_0[f'fractal_trend_age_{side}'] = ind.consec_condition(trend_condition)

    # loop through potential entries
    results = []
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

        pnl_pct = (trade_diff - 1.003) if side == 'long' else (0.997 - trade_diff)  # accounting for 15bps fees and 15bps slippage
        pnl_r = pnl_pct / r_pct

        pnl_cat = 0 if (pnl_r <= r_threshold) else 1

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

        msg = f"trade lifespans getting close to trimmed ohlc length ({lifespan / trim_ohlc:.1%}), increase trim ohlc"
        if lifespan / trim_ohlc > 0.5:
            print(msg)

    return results





    # calculate r by setting init stop based on last 2 bars ll/hh
    if side == 'long':
        df['r'] = abs((df.close - df.low.rolling(inval_lb).min()) / df.close).shift(1)
    else:
        df['r'] = abs((df.close - df.high.rolling(inval_lb).max()) / df.close).shift(1)
    pass


def trail_atr(df, atr_len, atr_mult):
    pass


def add_features(df, tf):
    df = features.atr_zscore(df, 25)
    df = features.atr_zscore(df, 50)
    df = features.atr_zscore(df, 100)
    df = features.atr_zscore(df, 200)
    df = features.atr_pct(df, 5)
    df = features.atr_pct(df, 10)
    df = features.atr_pct(df, 25)
    df = features.atr_pct(df, 50)
    df = features.atr_pct(df, 100)
    df = features.atr_pct(df, 200)
    df = features.ats_z(df, 12)
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
    df = features.dd_zscore(df, 12)
    df = features.dd_zscore(df, 25)
    df = features.dd_zscore(df, 50)
    df = features.dd_zscore(df, 100)
    df = features.dd_zscore(df, 200)
    df = features.doji(df, 0.5, 2, weighted=True)
    df = features.doji(df, 1, 2, weighted=True)
    df = features.doji(df, 1.5, 2, weighted=True)
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
    df = features.fractal_trend_age(df)
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
    df = features.num_trades_z(df, 12)
    df = features.num_trades_z(df, 25)
    df = features.num_trades_z(df, 50)
    df = features.num_trades_z(df, 100)
    df = features.num_trades_z(df, 200)
    df = features.prev_daily_open_ratio(df)
    df = features.prev_daily_high_ratio(df)
    df = features.prev_daily_low_ratio(df)
    df = features.prev_weekly_open_ratio(df)
    df = features.prev_weekly_high_ratio(df)
    df = features.prev_weekly_low_ratio(df)
    df = features.round_numbers_proximity(df)
    df = features.big_round_nums_proximity(df)
    df = features.spooky_nums_proximity(df)
    df = features.round_numbers_close(df, 1)
    df = features.round_numbers_close(df, 2)
    df = features.round_numbers_close(df, 3)
    df = features.round_numbers_close(df, 4)
    df = features.round_numbers_close(df, 5)
    df = features.big_round_nums_close(df, 1)
    df = features.big_round_nums_close(df, 5)
    df = features.big_round_nums_close(df, 10)
    df = features.big_round_nums_close(df, 15)
    df = features.big_round_nums_close(df, 20)
    df = features.rsi(df, 14)
    df = features.rsi(df, 25)
    df = features.rsi(df, 50)
    df = features.rsi(df, 100)
    df = features.rsi(df, 200)
    df = features.rsi_above(df, 14, 30)
    df = features.rsi_above(df, 25, 30)
    df = features.rsi_above(df, 50, 30)
    df = features.rsi_above(df, 100, 30)
    df = features.rsi_above(df, 200, 30)
    df = features.rsi_above(df, 14, 50)
    df = features.rsi_above(df, 25, 50)
    df = features.rsi_above(df, 50, 50)
    df = features.rsi_above(df, 100, 50)
    df = features.rsi_above(df, 200, 50)
    df = features.rsi_above(df, 14, 70)
    df = features.rsi_above(df, 25, 70)
    df = features.rsi_above(df, 50, 70)
    df = features.rsi_above(df, 100, 70)
    df = features.rsi_above(df, 200, 70)
    df = df.copy()
    df = features.rsi_timing_long(df, 3)
    df = features.rsi_timing_long(df, 5)
    df = features.rsi_timing_long(df, 7)
    df = features.rsi_timing_long(df, 9)
    df = features.rsi_timing_short(df, 3)
    df = features.rsi_timing_short(df, 5)
    df = features.rsi_timing_short(df, 7)
    df = features.rsi_timing_short(df, 9)
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
    df = features.stoch_m(df, 12)
    df = features.stoch_m(df, 25)
    df = features.stoch_m(df, 50)
    df = features.stoch_m(df, 100)
    df = features.stoch_w(df, 12)
    df = features.stoch_w(df, 25)
    df = features.stoch_w(df, 50)
    df = features.stoch_w(df, 100)
    df = features.stoch_num_trades(df, 25)
    df = features.stoch_num_trades(df, 50)
    df = features.stoch_num_trades(df, 100)
    df = features.stoch_num_trades(df, 200)
    df = features.stoch_vwma_ratio(df, 25)
    df = features.stoch_vwma_ratio(df, 50)
    df = features.stoch_vwma_ratio(df, 100)
    df = features.two_emas(df, 12, 24)
    df = features.two_emas(df, 12, 48)
    df = features.two_emas(df, 24, 96)
    df = features.two_emas(df, 48, 192)
    df = features.volume_climax_up(df, 12)
    df = features.volume_climax_up(df, 25)
    df = features.volume_climax_up(df, 50)
    df = features.volume_climax_down(df, 12)
    df = features.volume_climax_down(df, 25)
    df = features.volume_climax_down(df, 50)
    df = features.high_volume_churn(df, 12)
    df = features.high_volume_churn(df, 25)
    df = features.high_volume_churn(df, 50)
    df = features.low_volume_bar(df, 12)
    df = features.low_volume_bar(df, 25)
    df = features.low_volume_bar(df, 50)
    df = features.vol_delta(df)
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
    df = features.weekly_open_ratio(df)

    return df.copy()


# def fit_lgbm(X, y, num):
#     X = X.astype('float64')
#     y = y.astype('int32')
#
#     def objective(trial, X, y):
#         params = {
#             'n_estimators': 50000,
#             'lambda_l1': trial.suggest_int('lambda_l1', 0, 100, step=5),
#             'lambda_l2': trial.suggest_int('lambda_l2', 0, 100, step=5),
#             'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
#             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 0.95, step=0.1),
#             'bagging_freq': trial.suggest_categorical('bagging_freq', [1]),
#             'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.95, step=0.1),
#             'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 3000, log=True),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50, log=True),
#             'max_bin': trial.suggest_int('max_bin', 200, 300),
#             'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
#             'early_stopping_round': 100,
#             'n_thread': -1,
#             'verbosity': -1
#         }
#
#         pruning_callback = integration.LightGBMPruningCallback(trial, 'binary_logloss')
#
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
#         cv_scores = np.empty(5)
#         for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#             model = lgbm.LGBMClassifier(objective='binary', boosting_type='gbdt', **params)
#             model.fit(X_train, y_train,
#                       eval_set=[(X_test, y_test)],
#                       eval_metric='binary_logloss',
#                       # early_stopping_round=50,
#                       callbacks=[pruning_callback])
#             preds = model.predict_proba(X_test)
#             cv_scores[i] = log_loss(y_test, preds)
#
#         return np.mean(cv_scores)
#
#     op_logging.set_verbosity(op_logging.ERROR)
#     # pruner = pruners.MedianPruner(n_warmup_steps=5)
#     pruner = pruners.SuccessiveHalvingPruner()
#     sampler = TPESampler(seed=11)
#     study = create_study(sampler=sampler, pruner=pruner, direction='minimize')
#     func = lambda trial: objective(trial, X, y)
#     study.optimize(func, n_trials=num)  #, show_progress_bar=True)
#     logger.debug(f"Optimisation Score: {study.best_trial.value:.1%}")
#
#     # test performance on unseen data
#     best_params = study.best_trial.params
#     val_model = lgbm.LGBMClassifier(objective='binary', boosting_type='gbdt', **best_params)
#     val_model.fit(X, y)
#
#     return val_model


def fit_xgb(X, y, num_trials):
    X = X.astype('float64')
    y = y.astype('int32')

    d_train = xgb.DMatrix(X, label=y)

    def objective(trial):
        params = {
            'verbosity': 0,
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'alpha': trial.suggest_float('alpha', 1e-8, 100.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 100.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 100.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'max_depth': trial.suggest_int('max_depth', 2, 30, step=1),
            'subsample': trial.suggest_float('subsample', 0.1, 0.5),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 100.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
            'n_thread': -1
        }


        pruning_callback = integration.XGBoostPruningCallback(trial, 'test-auc')
        history = xgb.cv(params, d_train,
                         num_boost_round=50000,
                         early_stopping_rounds=100,
                         metrics=['auc'],
                         nfold=5,
                         verbose_eval=False,
                         callbacks=[pruning_callback])

        return history['test-auc-mean'].values[-1]

    op_logging.set_verbosity(op_logging.ERROR)
    # pruner = pruners.MedianPruner(n_warmup_steps=5)
    pruner = pruners.SuccessiveHalvingPruner()
    sampler = TPESampler(seed=11)
    study = create_study(sampler=sampler, pruner=pruner, direction='maximize')
    study.optimize(objective, n_trials=num_trials)
    logger.debug(f"Optimisation Score: {study.best_trial.value:.1%}")

    best_params = study.best_trial.params
    best_params['eval_metric'] = 'auc'

    X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.333, random_state=55)
    d_fit = xgb.DMatrix(X, label=y)
    d_eval = xgb.DMatrix(X_eval, label=y_eval)
    best_model = xgb.train(params=best_params,
                           dtrain=d_fit,
                           evals=[(d_eval, 'eval')],
                           num_boost_round=50000,
                           early_stopping_rounds=100,
                           verbose_eval=False)

    return best_model


def features_labels_split(df):
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'num_trades',
                 'taker_buy_base_vol', 'taker_buy_quote_vol', 'vwma', 'r_pct', 'pnl_pct', 'pnl_r', 'pnl_cat',
                 'atr-25', 'atr-50', 'atr-100', 'atr-200', 'ema_12', 'ema_25', 'ema_50', 'ema_100', 'ema_200',
                 'hma_25', 'hma_50', 'hma_100', 'hma_200', 'lifespan', 'frac_high', 'frac_low', 'inval', 'daily_open',
                 'prev_daily_open', 'prev_daily_high', 'prev_daily_low', 'weekly_open', 'prev_weekly_open',
                 'prev_weekly_high', 'prev_weekly_low', 'bullish_doji', 'bearish_doji', 'entry_l', 'entry_s',
                 'entry_l_price', 'entry_s_price'],
                axis=1, errors='ignore')#.drop(index=df.index[-1], axis=0)
    y = df.pnl_cat#.shift(-1).drop(index=df.index[-1], axis=0)
    z = df.pnl_r#.shift(-1).drop(index=df.index[-1], axis=0)
    # the shift is better done at the point where the pnls are calculated, by this time, the data is no longer true timeseries

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

    quant_cols = ['ema_25_ratio', 'ema_50_ratio', 'ema_100_ratio', 'ema_200_ratio', 'hma_25_ratio', 'hma_50_ratio',
          'hma_100_ratio', 'hma_200_ratio', 'atr_5_pct', 'atr_10_pct', 'atr_25_pct', 'atr_50_pct']
    quant_cols = [qc for qc in quant_cols if qc in X_train.columns]

    transformers = [('minmax', MinMaxScaler(), min_max_cols), ('quantile', QuantileTransformer(n_quantiles=200), quant_cols)]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    feature_cols = [f.split('__')[1] for f in ct.get_feature_names_out()]

    return X_train, X_test, feature_cols


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



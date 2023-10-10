import numpy as np
import pandas as pd
import ml_funcs as mlf
import time
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
from binance import Client
import binance.exceptions as bx
import resources.keys as keys
from resources.loggers import create_logger

if not Path('/pi_2.txt').exists():
    from sklearnex import patch_sklearn

    patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split  # , cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif  # , chi2
from sklearn.preprocessing import MinMaxScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
import warnings
warnings.filterwarnings('ignore')

logger = create_logger('trail_fractals', 'trail_fractals')
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


def load_features(folder, side, tf):
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['features'])


def load_pairs(folder, side, tf):
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['pairs'])


def get_margin_pairs(method, num_pairs):
    start_pair = 0
    pairs = mlf.rank_pairs(method)

    with open('ohlc_lengths.json', 'r') as file:
        ohlc_lengths = json.load(file)

    client = init_client()
    exc_info = client.get_exchange_info()
    symbol_margin = {i['symbol']: i['isMarginTradingAllowed'] for i in exc_info['symbols'] if
                     i['quoteAsset'] == 'USDT'}
    pairs = [p for p in pairs if symbol_margin[p] and (ohlc_lengths[p] > 4032)]

    return pairs[start_pair:start_pair + num_pairs]


def fit_gbc(X, y, scorer):
    base_model = GradientBoostingClassifier(random_state=42, n_estimators=10000, validation_fraction=0.1,
                                            n_iter_no_change=50)
    params = dict(
        subsample=[0.25, 0.5, 1],
        min_samples_split=[2, 4, 8],
        max_depth=[5, 10, 15, 20],
        learning_rate=[0.05, 0.1]
    )
    gs = GridSearchCV(estimator=base_model, param_grid=params, scoring=scorer, n_jobs=-1, cv=5, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_


def fit_rfc(X_train, y_train):
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


def create_dataset(pairs, data_len, timeframe, width, atr_spacing, side):
    all_res = pd.DataFrame()
    for pair in pairs:
        df = mlf.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
        df = mlf.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
        res_list = mlf.trail_fractal(df, width, atr_spacing, side)
        res_df = pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)
        all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)
    all_res = all_res.sort_values('timestamp').reset_index(drop=True)

    # split features from labels
    X, y, _ = mlf.features_labels_split(all_res)

    # undersampling
    # us = RandomUnderSampler(random_state=0)
    us = ClusterCentroids(random_state=0)
    X, y = us.fit_resample(X, y)

    return X, y


def save_models(width, spacing, sel_method, num_pairs, side, tf, data_len, selected, pairs, cal_model, scaler, validity):
    folder = Path(f"/home/ross/coding/modular_trader/machine_learning/"
                  f"models/trail_fractals_{width}_{spacing}/{sel_method}_{num_pairs}")
    pi_folder = Path(f"/home/ross/coding/pi_2/modular_trader/machine_learning/"
                      f"models/trail_fractals_{width}_{spacing}/{sel_method}_{num_pairs}")
    model_file = f"{side}_{tf}_model_1a.sav"
    model_info = f"{side}_{tf}_info_1a.json"
    scaler_file = f"{side}_{tf}_scaler_1a.sav"

    info_dict = {'data_length': data_len, 'features': selected, 'pair_selection': sel_method,
                 'pairs': pairs, 'created': int(datetime.now(timezone.utc).timestamp()), 'validity': validity}

    # save local copy
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal_model, folder / model_file)
    info_path = folder / model_info
    info_path.touch(exist_ok=True)
    with open(info_path, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # save on pi
    pi_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal_model, pi_folder / model_file)
    info_path_pi = pi_folder / model_info
    info_path_pi.touch(exist_ok=True)
    with open(info_path_pi, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = pi_folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)


def trail_fractals_1a(side, tf, width, atr_spacing, num_pairs, selection_method):
    loop_start = time.perf_counter()
    print(f"\n- Running Trail_fractals_1a, {side}, {tf}, {width}, {atr_spacing}, {num_pairs}, {selection_method}")
    data_len = 500
    pairs = get_margin_pairs(selection_method, num_pairs)

    X, y = create_dataset(pairs, data_len, tf, width, atr_spacing, side)

    # split data for fitting and calibration
    X, X_cal, y, y_cal = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

    # TODO when i refactor this into a pipeline, i will need to remove the equivalent step from the agent definition
    # X, _, cols = mlf.transform_columns(X, X)
    cols = list(X.columns) # list of strings, names of all features
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X)

    # feature selection
    print(f"feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    selector = feature_selection_1a(X1, y, 1000, cols)
    selected = selector.k_feature_names_
    print(selected)

    # apply selector and scaler to X and X_cal
    X = selector.transform(X)
    X = pd.DataFrame(X, columns=selected)
    X = scaler.fit_transform(X)
    X_cal = selector.transform(X_cal)
    X_cal = pd.DataFrame(X_cal, columns=selected)
    X_cal = scaler.transform(X_cal)

    # fit model
    print(f"Training on {X.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X = pd.DataFrame(X, columns=selected)
    grid_model = fit_gbc(X, y, scorer)
    # grid_model = fit_rfc(X, y, scorer)
    train_score = grid_model.score(X, y)
    print(f"Model score on train set after grid search: {train_score:.1%}")
    test_score = grid_model.score(X_cal, y_cal)
    print(f"Model score on test set after grid search: {test_score:.1%}")

    # calibrate model
    print(f"Calibrating on {X_cal.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X_cal = pd.DataFrame(X_cal, columns=selected)
    cal_model = CalibratedClassifierCV(estimator=grid_model, cv='prefit', n_jobs=-1)
    cal_model.fit(X_cal, y_cal)
    cal_score = cal_model.score(X_cal, y_cal)
    print(f"Model score after calibration: {cal_score:.1%}")

    # save models and info
    save_models(width, atr_spacing, selection_method, num_pairs, side,
                tf, data_len, selected, pairs, cal_model, scaler, len(X_cal))

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"TF 1a test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")


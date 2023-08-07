"""This is the script that will be run once a day to retrain the model using a grid search on the latest data. once per
week it will also run a sequential floating backwards feature selection to update the feature list, and once a month it
will do a full search of williams fractal params as well"""

import pandas as pd
import ml_funcs as mlf
import time
import itertools
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
from binance import Client
import binance.exceptions as bx
import resources.keys as keys
from resources.loggers import create_logger
import requests

if not Path('/pi_2.txt').exists():
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler

def init_client(max_retries: int=360, delay: int=5):
    for i in range(max_retries):
        try:
            client = Client(keys.bPkey, keys.bSkey)
            print(f'initialising binance client worked on attempt number {i+1}')
            return client
        except bx.BinanceAPIException as e:
            if e.code != -3044:
                raise e
            print(f"System busy, retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception(f"Max retries exceeded. Request still failed after {max_retries} attempts.")

client = init_client()

logger = create_logger('trail_fractals')

all_start = time.perf_counter()
now = datetime.now().strftime('%Y/%m/%d %H:%M')
print(f"-:--:--:--:--:--:--:--:--:--:-  {now} UTC Running Trail Fractals Fitting  -:--:--:--:--:--:--:--:--:--:-")

def feature_selection(X, y, limit, quick=False):
    fs_start = time.perf_counter()

    if quick:
        selector = SelectKBest(mutual_info_classif, k=15)
    else:
        selector = SelectKBest(mutual_info_classif, k=36)
    X_selected = selector.fit_transform(X, y)
    selector_cols = selector.get_support(indices=True)
    selected_columns = [col for i, col in enumerate(cols) if i in selector_cols]
    X = pd.DataFrame(X_selected, columns=selected_columns)

    if X.shape[0] > limit:
        X_train, _, y_train, _ = train_test_split(X, y, train_size=limit, random_state=99)
    else:
        X_train, y_train = X, y

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
    X_transformed = selector.transform(X)
    selected = [cols[i] for i in selector.k_feature_idx_]
    print(f"Number of features selected: {len(selected)}")

    fs_end = time.perf_counter()
    fs_elapsed = fs_end - fs_start
    print(f"Feature selection time taken: {int(fs_elapsed // 60)}m {fs_elapsed % 60:.1f}s")

    return X_transformed, y, selected


def load_features(side, tf):
    folder = Path("machine_learning/models/trail_fractals_slow")
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['features'])


def load_pairs(side, tf):
    folder = Path("machine_learning/models/trail_fractals_slow")
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['pairs'])


def fit_gbc(X, y, scorer):
    base_model = GradientBoostingClassifier(random_state=42, n_estimators=10000, validation_fraction=0.1, n_iter_no_change=50)
    params = dict(
        subsample=[0.25, 0.5, 1],
        min_samples_split=[2, 4, 8],
        max_depth=[5, 10, 15, 20],
        learning_rate=[0.05, 0.1]
    )
    gs = GridSearchCV(estimator=base_model, param_grid=params, scoring=scorer, n_jobs=-1, cv=5, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_


running_on_pi = Path('/pi_2.txt').exists()
if not running_on_pi:
    import update_ohlc

timeframes = ['1h',
              '4h', '12h', '1d'
              ]
sides = ['long',
         'short'
         ]
data_len = 200
num_pairs = 30
start_pair = 0
width = 5
atr_spacing = 2
scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
configs = [(True, 'volumes'), (True, 'volatilities'), (False, 'volumes')]

for i in range(360):
    try:
        exc_info = client.get_exchange_info()
        logger.info(f"get exchange info worked on attempt number {i+1}")
        break
    except requests.exceptions.ConnectionError as e:
        time.sleep(5)

for side, timeframe, config in itertools.product(sides, timeframes, configs):
    speed = config[0]
    pair_selection = config[1]

    print(f"\nFitting {timeframe} {side}, {speed}, {pair_selection} model")
    loop_start = time.perf_counter()

    if running_on_pi:
        pairs = load_pairs(side, timeframe)
    else:
        pairs = mlf.rank_pairs(pair_selection)
        symbol_margin = {i['symbol']: i['isMarginTradingAllowed'] for i in exc_info['symbols'] if i['quoteAsset'] == 'USDT'}
        pairs = [p for p in pairs if symbol_margin[p]]
        pairs = pairs[start_pair:start_pair + num_pairs]

    # create dataset
    all_res = pd.DataFrame()
    for pair in pairs:
        df = mlf.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
        df = mlf.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
        res_list = mlf.trail_fractal(df, width, atr_spacing, side)
        res_df = pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)
        all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)
    all_res = all_res.sort_values('timestamp').reset_index(drop=True)

    # split features from labels
    X, y, z = mlf.features_labels_split(all_res)
    if running_on_pi:
        selected = load_features(side, timeframe)
        X = X.loc[:, selected]
    X, _, cols = mlf.transform_columns(X, X)


    # random undersampling
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)

    # feature selection
    if not running_on_pi:
        print(f"feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
        X, y, selected = feature_selection(X, y, 1000, quick=speed)

    # split data for fitting and calibration
    X, X_cal, y, y_cal = train_test_split(X, y, test_size=0.333, random_state=11)

    # fit model
    print(f"Training on {X.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X = pd.DataFrame(X, columns=selected)
    grid_model = fit_gbc(X, y, scorer)

    # calibrate model
    print(f"Calibrating on {X_cal.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X_cal = pd.DataFrame(X_cal, columns=selected)
    cal_model = CalibratedClassifierCV(estimator=grid_model, cv='prefit', n_jobs=-1)
    cal_model.fit(X_cal, y_cal)
    cal_score = cal_model.score(X_cal, y_cal)
    print(f"Model score after calibration: {cal_score:.1%}")

    # save to files
    quick_str = 'quick' if speed else 'slow'
    folder = Path(f"machine_learning/models/trail_fractals_{quick_str}_{pair_selection}")
    folder.mkdir(parents=True, exist_ok=True)
    pi2_folder = Path(f"/home/ross/coding/pi2/modular_trader/machine_learning/models/trail_fractals_{quick_str}_{pair_selection}")
    pi2_folder.mkdir(parents=True, exist_ok=True)

    # save ml model on laptop and pi
    model_file = Path(f"trail_fractal_{side}_{timeframe}_model.sav")
    joblib.dump(cal_model, folder / model_file)
    joblib.dump(cal_model, pi2_folder / model_file)

    # save info dict on laptop and pi
    model_info = Path(f"trail_fractal_{side}_{timeframe}_info.json")
    model_info.touch(exist_ok=True)
    if not running_on_pi:
        info_dict = {'features': selected,
                     'pairs': pairs,
                     'data_length': data_len,
                     'frac_width': width,
                     'atr_spacing': atr_spacing,
                     'created': int(datetime.now(timezone.utc).timestamp()),
                     'quick': speed,
                     'pair_selection': pair_selection}
        with open(folder / model_info, 'w') as info:
            json.dump(info_dict, info)
        with open(pi2_folder / model_info, 'w') as info:
            json.dump(info_dict, info)

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"This test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {all_elapsed % 60:.1f}s")

"""This is the script that will be run once a day to retrain the model using a grid search on the latest data. once per
week it will also run a sequential floating backwards feature selection to update the feature list, and once a month it
will do a full search of williams fractal params as well"""

import pandas as pd
import entry_modelling as em
import time
import itertools
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler

all_start = time.perf_counter()
import update_ohlc
now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M')
print(f"-:--:--:--:--:--:--:--:--:--:-  {now} Running Trail Fractals Fitting  -:--:--:--:--:--:--:--:--:--:-")


def feature_selection(X, y, limit, quick=False):
    fs_start = time.perf_counter()

    if X.shape[0] > limit:
        X_train, _, y_train, _ = train_test_split(X, y, train_size=limit, random_state=99)
    else:
        X_train, y_train = X, y

    selector_model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1,
                                                n_iter_no_change=5,
                                                subsample=0.5, min_samples_split=8, max_depth=12, learning_rate=0.3)
    if quick:
        selector = SFS(estimator=selector_model, k_features=10, forward=True, floating=False, verbose=0,
                       scoring=scorer, n_jobs=-1)
    else:
        selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=0,
                       scoring=scorer, n_jobs=-1)
    selector = selector.fit(X_train, y_train)
    X_transformed = selector.transform(X)
    selected = [cols[i] for i in selector.k_feature_idx_]

    fs_end = time.perf_counter()
    fs_elapsed = fs_end - fs_start
    print(f"\nFeature selection time taken: {int(fs_elapsed // 60)}m {fs_elapsed % 60:.1f}s")

    return X_transformed, y, selected

timeframes = ['1h', '4h', '12h', '1d']
sides = ['long', 'short']
data_len = 200
num_pairs = 20
start_pair = 0
width = 5
atr_spacing = 2
pairs = em.rank_pairs()[start_pair:start_pair + num_pairs]
scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

for side, timeframe in itertools.product(sides, timeframes):
    print(f"\nFitting {timeframe} {side} model")
    # create dataset
    all_res = pd.DataFrame()
    for pair in pairs:
        df = em.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
        df = em.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
        res_list = em.trail_fractal(df, width, atr_spacing, side)
        res_df = pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)
        all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)
    all_res = all_res.sort_values('timestamp').reset_index(drop=True)

    # split features from labels
    X, y, z = em.features_labels_split(all_res)
    X, _, cols = em.transform_columns(X, X)

    # random undersampling
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)

    # feature selection
    X, y, selected = feature_selection(X, y, 1000, quick=False)

    # split data for fitting and calibration
    X, X_cal, y, y_cal = train_test_split(X, y, test_size=0.333, random_state=11)

    # fit model
    print(f"Training on {X.shape[0]} observations")
    X = pd.DataFrame(X, columns=selected)
    base_model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5)
    params = dict(
        subsample=[0.25, 0.5, 1],
        min_samples_split=[2, 4, 8],
        max_depth=[5, 10, 15, 20],
        learning_rate=[0.05, 0.1]
    )
    gs = GridSearchCV(estimator=base_model, param_grid=params, scoring=scorer, n_jobs=-1, cv=5, verbose=0)
    gs.fit(X, y)
    grid_model = gs.best_estimator_

    # calibrate model
    print(f"Calibrating on {X_cal.shape[0]} observations")
    X_cal = pd.DataFrame(X_cal, columns=selected)
    cal_model = CalibratedClassifierCV(estimator=grid_model, cv='prefit', n_jobs=-1)
    cal_model.fit(X_cal, y_cal)
    cal_score = cal_model.score(X_cal, y_cal)
    print(f"\nModel score after calibration: {cal_score:.1%}")

    # save to files
    folder = Path("models/trail_fractals")
    folder.mkdir(parents=True, exist_ok=True)

    model_file = folder / f"trail_fractal_{side}_{timeframe}_model.sav"
    joblib.dump(cal_model, model_file)

    model_info = folder / f"trail_fractal_{side}_{timeframe}_info.json"
    model_info.touch(exist_ok=True)
    info_dict = {'features': selected, 'pairs': pairs, 'data_length': data_len, 'frac_width': width, 'atr_spacing': atr_spacing}
    with open(model_info, 'w') as info:
        json.dump(info_dict, info)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

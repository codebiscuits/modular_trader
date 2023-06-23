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
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler

all_start = time.perf_counter()

timeframes = ['1h']
sides = ['long', 'short']
data_len = 200
num_pairs = 10
start_pair = 0
width = 5
atr_spacing = 2
pairs = em.rank_pairs()[start_pair:start_pair + num_pairs]

for side, timeframe in itertools.product(sides, timeframes):
    print(f"Fitting {timeframe} {side} model")
    # create dataset
    all_res = pd.DataFrame()
    for pair in pairs:
        df = em.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
        df = em.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
        res_list = em.trail_fractal(df, width, atr_spacing, side)
        res_df = pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)
        all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)
    all_res = all_res.sort_values('timestamp').reset_index(drop=True)
    print(f"Training on {len(all_res)} observations")

    # split dataset
    X, y, z = em.features_labels_split(all_res)
    cols = X.columns
    X, _ = em.transform_columns(X, X)

    # feature selection
    fs_start = time.perf_counter()
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)
    scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
    selector_model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5,
                                       subsample=0.5, min_samples_split=8, max_depth=12, learning_rate=0.1)
    selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=2, scoring=scorer, n_jobs=-1)
    selector = selector.fit(X, y)
    X = selector.transform(X)
    selected = [cols[x] for x in selector.k_feature_idx_]
    fs_end = time.perf_counter()
    fs_elapsed = fs_end - fs_start
    print(f"\nFeature selection time taken: {int(fs_elapsed // 60)}m {fs_elapsed % 60:.1f}s")

    # fit model
    base_model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5)
    params = dict(
        subsample=[0.25, 0.5, 1],
        min_samples_split=[2, 4, 8],
        max_depth=[5, 10, 15, 20],
        learning_rate=[0.05, 0.1]
    )
    gs = GridSearchCV(estimator=base_model, param_grid=params, scoring=scorer, n_jobs=-1, cv=5, verbose=2)
    gs.fit(X, y)
    final_model = gs.best_estimator_

    # save to files
    folder = Path("/home/ross/Documents/backtester_2021/machine_learning/models/trail_fractals")
    folder.mkdir(parents=True, exist_ok=True)
    model_file = folder / f"trail_fractal_{side}_{timeframe}_model.sav"
    model_info = folder / f"trail_fractal_{side}_{timeframe}_info.json"
    model_info.touch(exist_ok=True)
    joblib.dump(final_model, model_file)
    info_dict = {'features': selected, 'pairs': pairs, 'data_length': data_len, 'frac_width': width, 'atr_spacing': atr_spacing}
    with open(model_info, 'w') as info:
        json.dump(info_dict, info)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

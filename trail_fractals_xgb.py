"""This is the script that will be run once a day to retrain the model using a grid search on the latest data. once per
week it will also run a sequential floating backwards feature selection to update the feature list, and once a month it
will do a full search of williams fractal params as well"""
import numpy as np
import pandas as pd
import ml_funcs as mlf
import time
import itertools
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
from pprint import pprint

if not Path('/pi_2.txt').exists():
    from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn
    patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from optuna import Trial, logging, visualization, integration, pruners, create_study
from optuna.samplers import TPESampler

all_start = time.perf_counter()
now = datetime.now().strftime('%Y/%m/%d %H:%M')
print(f"-:--:--:--:--:--:--:--:--:--:-  {now} UTC Running Trail Fractals Fitting  -:--:--:--:--:--:--:--:--:--:-")

def feature_selection(X, y, limit, scorer, quick=False):
    fs_start = time.perf_counter()
    print(f"feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

    if X.shape[0] > limit:
        X_train, _, y_train, _ = train_test_split(X, y, train_size=limit, random_state=99)
    else:
        X_train, y_train = X, y

    selector_model = GradientBoostingClassifier(random_state=42, n_estimators=10000, validation_fraction=0.1,
                                                n_iter_no_change=50,
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
    print(f"Number of features selected: {len(selected)}")

    fs_end = time.perf_counter()
    fs_elapsed = fs_end - fs_start
    print(f"feature selection ended: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    print(f"Feature selection time taken: {int(fs_elapsed // 60)}m {fs_elapsed % 60:.1f}s")

    return X_transformed, y, selected


def load_features(side, tf):
    folder = Path("machine_learning/models/trail_fractals")
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['features'])


def load_pairs(side, tf):
    folder = Path("machine_learning/models/trail_fractals")
    info_path = folder / f"trail_fractal_{side}_{tf}_info.json"
    with open(info_path, 'r') as ip:
        info = json.load(ip)

    return list(info['pairs'])


def fit_xgb(X, y):
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
                         early_stopping_rounds=50,
                         metrics=['auc'],
                         nfold=5,
                         verbose_eval=False,
                         callbacks=[pruning_callback])

        return history['test-auc-mean'].values[-1]

    logging.set_verbosity(logging.WARNING)
    # pruner = pruners.MedianPruner(n_warmup_steps=5)
    pruner = pruners.SuccessiveHalvingPruner()
    sampler = TPESampler(seed=11)
    study = create_study(sampler=sampler, pruner=pruner, direction='maximize')
    study.optimize(objective, n_trials=100)

    best = study.best_trial
    print(f"Best trial: {best.value:.1%}")
    print("Params:")
    pprint(best.params)

    best_params = best.params
    best_params['eval_metric'] = 'auc'

    X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.333, random_state=55)
    d_fit = xgb.DMatrix(X, label=y)
    d_eval = xgb.DMatrix(X_eval, label=y_eval)
    best_model = xgb.train(params=best_params,
                           dtrain=d_fit,
                           evals=[(d_eval, 'eval')],
                           num_boost_round=50000,
                           early_stopping_rounds=50)

    return best_model


running_on_pi = Path('/pi_2.txt').exists()
# if not running_on_pi:
#     import update_ohlc

# running_on_pi = True

timeframes = ['1h', '4h', '12h', '1d']
sides = ['long', 'short']
data_len = 200
num_pairs = 30
start_pair = 0
width = 5
atr_spacing = 2
fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

for side, timeframe in itertools.product(sides, timeframes):
    print(f"\nFitting {timeframe} {side} model")
    loop_start = time.perf_counter()

    if running_on_pi:
        pairs = load_pairs(side, timeframe)
    else:
        pairs = mlf.rank_pairs()[start_pair:start_pair + num_pairs]

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
        print(f"mutual info began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
        selector = SelectKBest(mutual_info_classif, k=50)
        X_selected = selector.fit_transform(X, y)
        selector_cols = selector.get_support(indices=True)
        selected_columns = [col for i, col in enumerate(cols) if i in selector_cols]
        X = pd.DataFrame(X_selected, columns=selected_columns)

        print(f"sequential feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
        X, y, selected = feature_selection(X, y, 1000, fb_scorer, quick=True)

    # split data for fitting and calibration
    X, X_cal, y, y_cal = train_test_split(X, y, test_size=0.333, random_state=11)

    # fit model
    print(f"Training on {X.shape[0]} observations")
    X = pd.DataFrame(X, columns=selected)
    final_model = fit_xgb(X, y)

    # # calibrate model
    # print(f"Calibrating on {X_cal.shape[0]} observations")
    # X_cal = pd.DataFrame(X_cal, columns=selected)
    # cal_model = CalibratedClassifierCV(estimator=final_model, cv='prefit', n_jobs=-1)
    # cal_model.fit(X_cal, y_cal)
    # cal_score = cal_model.score(X_cal, y_cal)
    # print(f"Model score after calibration: {cal_score:.1%}")

    # save to files
    folder = Path("machine_learning/models/trail_fractals")
    folder.mkdir(parents=True, exist_ok=True)

    model_file = folder / f"trail_fractal_{side}_{timeframe}_xgb_model.json"
    final_model.save_model(model_file)

    model_info = folder / f"trail_fractal_{side}_{timeframe}_xgb_info.json"
    model_info.touch(exist_ok=True)
    info_dict = {'features': selected,
                 'pairs': pairs,
                 'data_length': data_len,
                 'frac_width': width,
                 'atr_spacing': atr_spacing,
                 'created': int(datetime.now(timezone.utc).timestamp())}
    with open(model_info, 'w') as info:
        json.dump(info_dict, info)

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"This test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

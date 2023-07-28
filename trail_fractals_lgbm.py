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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import fbeta_score, make_scorer, log_loss
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgbm
from optuna import Trial, logging, visualization, integration, pruners, create_study
from optuna.samplers import TPESampler

all_start = time.perf_counter()
now = datetime.now().strftime('%Y/%m/%d %H:%M')
print(f"-:--:--:--:--:--:--:--:--:--:-  {now} UTC Running Trail Fractals Fitting  -:--:--:--:--:--:--:--:--:--:-")

def feature_selection(X, y, limit, scorer, quick=False):

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=0.9, random_state=99)

    selector_model = lgbm.LGBMClassifier(objective='binary',
                                         random_state=42,
                                         n_estimators=50,
                                         boosting='gbdt',
                                         verbosity=-1)
    if quick:
        selector = SFS(estimator=selector_model, k_features=10, forward=True, floating=False, verbose=0,
                       scoring=scorer, n_jobs=-1)
    else:
        selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=0,
                       scoring=scorer, n_jobs=-1)
    selector = selector.fit(X_train, y_train)
    X_transformed = selector.transform(X)
    selected = [cols[i] for i in selector.k_feature_idx_]
    print(f"Sequential FS selected {len(selected)} features.")

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


def fit_lgbm(X, y):
    X = X.astype('float64')
    y = y.astype('int32')

    def objective(trial, X, y):
        params = {
            'n_estimators': 50000,
            'lambda_l1': trial.suggest_int('lambda_l1', 0, 100, step=5),
            'lambda_l2': trial.suggest_int('lambda_l2', 0, 100, step=5),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 0.95, step=0.1),
            'bagging_freq': trial.suggest_categorical('bagging_freq', [1]),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.95, step=0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 3000, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50, log=True),
            'max_bin': trial.suggest_int('max_bin', 200, 300),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_thread': -1
        }

        pruning_callback = integration.XGBoostPruningCallback(trial, 'binary_logloss')

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        cv_scores = np.empty(5)
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = lgbm.LGBMClassifier(objective='binary', **params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric='binary_logloss',
                      early_stopping_rounds=50,
                      callbacks=[pruning_callback])
            preds = model.predict_proba(X_test)
            cv_scores[i] = log_loss(y_test, preds)

        return np.mean(cv_scores)

    logging.set_verbosity(logging.WARNING)
    # pruner = pruners.MedianPruner(n_warmup_steps=5)
    pruner = pruners.SuccessiveHalvingPruner()
    sampler = TPESampler(seed=11)
    study = create_study(sampler=sampler, pruner=pruner, direction='minimize')
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=100)

    best = study.best_trial
    print(f"Best trial: {best.value:.1%}")
    print("Params:")
    pprint(best.params)


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
        X, y, selected = feature_selection(X, y, 10000, fb_scorer, quick=True)

    # split data for fitting and calibration
    X, X_cal, y, y_cal = train_test_split(X, y, test_size=0.333, random_state=11)

    # fit model
    print(f"Training on {X.shape[0]} observations bbegan: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X = pd.DataFrame(X, columns=selected)
    final_model = fit_lgbm(X, y)

    # # calibrate model
    # print(f"Calibrating on {X_cal.shape[0]} observations")
    # X_cal = pd.DataFrame(X_cal, columns=selected)
    # cal_model = CalibratedClassifierCV(estimator=final_model, cv='prefit', n_jobs=-1)
    # cal_model.fit(X_cal, y_cal)
    # cal_score = cal_model.score(X_cal, y_cal)
    # print(f"Model score after calibration: {cal_score:.1%}")
    #
    # # save to files
    # folder = Path("machine_learning/models/trail_fractals")
    # folder.mkdir(parents=True, exist_ok=True)
    #
    # model_file = folder / f"trail_fractal_{side}_{timeframe}_xgb_model.json"
    # final_model.save_model(model_file)
    #
    # model_info = folder / f"trail_fractal_{side}_{timeframe}_xgb_info.json"
    # model_info.touch(exist_ok=True)
    # info_dict = {'features': selected,
    #              'pairs': pairs,
    #              'data_length': data_len,
    #              'frac_width': width,
    #              'atr_spacing': atr_spacing,
    #              'created': int(datetime.now(timezone.utc).timestamp())}
    # with open(model_info, 'w') as info:
    #     json.dump(info_dict, info)

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"This test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

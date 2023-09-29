import numpy as np
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
from pprint import pformat
import warnings

running_on_pi = Path('/pi_2.txt').exists()
if not running_on_pi:
    from sklearnex import patch_sklearn

    patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split  # , cross_val_score
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif  # , chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgbm
from optuna import Trial, logging as op_logging, visualization, integration, pruners, create_study
from optuna.samplers import TPESampler
from xgboost import XGBClassifier, DMatrix

################################################ - IMPORTANT - #####################################################

"""
I have trained the models on adjusted targets - ie i have classified positive PnL as being pnl > some threshold.
I have calculated the final validation scores for each test run by comparing the model's predictions against un-adjusted 
targets - ie PnL > 0.
So just remember that if the scores are really weird, then this may be a stupid thing to do.
"""

op_logging.set_verbosity(op_logging.ERROR)
warnings.filterwarnings('ignore')

all_start = time.perf_counter()

logger = create_logger('trail_fractals_2', 'trail_fractals_2')


def create_dataset(side, tf, frac_width, atr_spacing, thresh):
    records_path_1 = Path(f"/home/ross/coding/pi_2/modular_trader/records/trail_fractals_{tf}_"
                          f"None_{frac_width}_{atr_spacing}_{'1d_volumes'}_{30}")

    records_path_2 = Path(f"/home/ross/coding/pi_2/modular_trader/records/trail_fractals_{tf}_"
                          f"None_{frac_width}_{atr_spacing}_{'1w_volumes'}_{100}")

    with open(records_path_1 / "closed_trades.json", 'r') as real_file:
        real_records_1 = json.load(real_file)
    with open(records_path_1 / "closed_sim_trades.json", 'r') as sim_file:
        sim_records_1 = json.load(sim_file)

    with open(records_path_2 / "closed_trades.json", 'r') as real_file:
        real_records_2 = json.load(real_file)
    with open(records_path_2 / "closed_sim_trades.json", 'r') as sim_file:
        sim_records_2 = json.load(sim_file)

    all_records = real_records_1 | sim_records_1 | real_records_2 | sim_records_2
    observations = []

    for position in all_records.values():
        signal = position['signal']
        if ((signal['direction'] != side) or
                (signal.get('confidence_l') is None) or
                (signal.get('market_rank_1d') is None) or
                (signal.get('perf_ema4') is None)):
            continue

        trade = position['trade']
        pnl = 0.0
        for t in trade:
            if t.get('rpnl'):
                pnl += t['rpnl']

        try:
            observation = dict(
                # asset=signal['asset'],
                conf_l=signal['confidence_l'],
                conf_s=signal['confidence_s'],
                inval_ratio=signal['inval_ratio'],
                mkt_rank_1d=signal['market_rank_1d'],
                mkt_rank_1w=signal['market_rank_1w'],
                mkt_rank_1m=signal['market_rank_1m'],
                perf_ema_4=signal['perf_ema4'],
                perf_ema_8=signal['perf_ema8'],
                perf_ema_16=signal['perf_ema16'],
                perf_ema_32=signal['perf_ema32'],
                perf_ema_64=signal['perf_ema64'],
                pnl=pnl > 0,
                win=pnl > thresh
            )
        except KeyError:
            logger.debug(pformat(position))
        observations.append(observation)

    return pd.DataFrame(observations)


def feature_selection(X, y, X_val, scorer, arch):
    # feature selection on base model
    if arch == 'lgbm':
        selector_model = lgbm.LGBMClassifier(objective='binary',
                                             random_state=42,
                                             n_estimators=50,
                                             boosting='gbdt',
                                             verbosity=-1)
    elif arch == 'xgb':
        selector_model = GradientBoostingClassifier(random_state=42,
                                                    n_estimators=10000,
                                                    validation_fraction=0.1,
                                                    n_iter_no_change=50,
                                                    subsample=0.5,
                                                    min_samples_split=2,
                                                    max_depth=12,
                                                    learning_rate=0.3)

    selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=0,
                   scoring=scorer, n_jobs=-1)

    selector = selector.fit(X, y)
    X = selector.transform(X)
    X_val = selector.transform(X_val)
    selected = selector.k_feature_idx_

    sel_score = selector.k_score_
    print(f"Model score after feature selection: {sel_score:.1%}")

    return X, y, X_val, selected


def trail_fractals_2(side, tf, frac_width, atr_spacing, thresh):
    tf_start = time.perf_counter()

    results = create_dataset(side, tf, frac_width, atr_spacing, thresh)

    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

    # split features from labels
    X = results.drop('win', axis=1)
    y = results.win  # pnl > threshold

    # random undersampling
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)

    # split off validation set
    X, X_val, y, y_val = train_test_split(X, y, train_size=0.9, random_state=43875, stratify=y)
    if len(X_val) < 30:
        logger.warning(f"Only {len(X_val)} observations in validation set, results will be unreliable.")

    # split off validation labels
    X = X.drop('pnl', axis=1)
    z_val = X_val.pnl  # pnl > 0
    X_val = X_val.drop('pnl', axis=1)
    cols = X.columns

    print('')
    logger.debug(f"{len(y)} observations in {tf} {side} dataset")

    # feature scaling
    scaler = QuantileTransformer()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    # quick feature selection
    selector = SelectKBest(mutual_info_classif, k=7)
    selector.fit(X, y)
    cols_idx = list(selector.get_support(indices=True))
    lgb_feature_names = [col for i, col in enumerate(cols) if i in cols_idx]
    xgb_feature_names = [col for i, col in enumerate(cols) if i in cols_idx]
    lgb_X = X[:, cols_idx]
    lgb_X_val = X_val[:, cols_idx]
    xgb_X = X[:, cols_idx]
    xgb_X_val = X_val[:, cols_idx]

    # slow feature selection

    # hyperparameter optimisation
    start_lgb = time.perf_counter()
    # lgb_X, y, lgb_X_val, lgb_selected = feature_selection(X, y, X_val, fb_scorer, 'lgbm')
    # lgb_feature_names = [col for i, col in enumerate(cols) if i in lgb_selected]
    lgbm_model = mlf.fit_lgbm(lgb_X, y, 1000)

    y_pred = lgbm_model.predict(lgb_X_val)
    accuracy = accuracy_score(z_val, y_pred)
    f_beta = fbeta_score(z_val, y_pred, beta=0.333)
    logger.debug(f"LGBM Performance on validation set: accuracy: {accuracy:.1%}, f beta: {f_beta:.1%}")
    end_lgb = time.perf_counter()
    lgb_elapsed = end_lgb - start_lgb
    logger.debug(f"LGB time taken: {int(lgb_elapsed // 3600)}h {int(lgb_elapsed // 60) % 60}m {lgb_elapsed % 60:.1f}s")

    xgb_start = time.perf_counter()
    # xgb_X, y, xgb_X_val, xgb_selected = feature_selection(X, y, X_val, fb_scorer, 'xgb')
    # xgb_feature_names = [col for i, col in enumerate(cols) if i in xgb_selected]
    xgb_model = mlf.fit_xgb(xgb_X, y, 1000)

    d_val = DMatrix(xgb_X_val, label=z_val)
    y_pred = xgb_model.predict(d_val) > 0.5
    accuracy = accuracy_score(z_val, y_pred)
    f_beta = fbeta_score(z_val, y_pred, beta=0.333)
    logger.debug(f"XGB Performance on validation set: accuracy: {accuracy:.1%}, f beta: {f_beta:.1%}")
    xgb_end = time.perf_counter()
    xgb_elapsed = xgb_end - xgb_start
    logger.debug(f"XGB time taken: {int(xgb_elapsed // 3600)}h {int(xgb_elapsed // 60) % 60}m {xgb_elapsed % 60:.1f}s")


    tf_end = time.perf_counter()
    tf_elapsed = tf_end - tf_start
    logger.debug(f"Test time taken: {int(tf_elapsed // 3600)}h {int(tf_elapsed // 60) % 60}m {tf_elapsed % 60:.1f}s")

trail_fractals_2('long', '1h', 5, 2, 0.4)
trail_fractals_2('short', '1h', 5, 2, 0.4)
trail_fractals_2('long', '4h', 5, 2, 0.4)
trail_fractals_2('short', '4h', 5, 2, 0.4)
trail_fractals_2('long', '12h', 5, 2, 0.4)
trail_fractals_2('short', '12h', 5, 2, 0.4)
trail_fractals_2('long', '1d', 5, 2, 0.4)
trail_fractals_2('short', '1d', 5, 2, 0.4)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
logger.debug(f"Total time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {all_elapsed % 60:.1f}s")

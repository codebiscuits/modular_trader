import numpy as np
import pandas as pd
import ml_funcs as mlf
import time
from pathlib import Path
import joblib
import json
from resources.loggers import create_logger
from pprint import pformat
import warnings

running_on_pi = Path('/pi_2.txt').exists()
if not running_on_pi:
    from sklearnex import patch_sklearn

    patch_sklearn()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif  # , chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler
from optuna import logging as op_logging, visualization
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

logger = create_logger('trail_fractals_2', 'trail_fractals_2')

# TODO i want to try different undersamplers


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


def feature_selection(X, y, X_val, scorer):
    # feature selection on base model
    selector_model = XGBClassifier()

    selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=0,
                   scoring=scorer, n_jobs=-1)

    selector = selector.fit(X, y)
    X = selector.transform(X)
    X_val = selector.transform(X_val)
    selected = selector.k_feature_idx_

    sel_score = selector.k_score_
    print(f"Model score after feature selection: {sel_score:.1%}")

    return X, y, X_val, selected


def save_models(side, tf, width, atr_spacing, feature_names, thresh, X_val, model, scaler):
    folder = Path(f"/home/ross/coding/modular_trader/machine_learning/models/trail_fractals_{width}_{atr_spacing}")
    pi_folder = Path(f"/home/ross/coding/pi_2/modular_trader/machine_learning/"
                     f"models/trail_fractals_{width}_{atr_spacing}")
    model_file = f"{side}_{tf}_model_2.json"
    model_info = f"{side}_{tf}_info_2.json"
    scaler_file = f"{side}_{tf}_scaler_2.sav"

    info_dict = {'features': feature_names, 'pnl_threshold': thresh, 'valid': len(X_val) > 30}

    # save local copy
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(folder / model_file)
    info_path = folder / model_info
    info_path.touch(exist_ok=True)
    with open(info_path, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # save on pi
    pi_folder.mkdir(parents=True, exist_ok=True)
    model.save_model(pi_folder / model_file)
    info_path_pi = pi_folder / model_info
    info_path_pi.touch(exist_ok=True)
    with open(info_path_pi, 'w') as info:
        json.dump(info_dict, info)
    scaler_path = pi_folder / scaler_file
    scaler_path.touch(exist_ok=True)
    joblib.dump(scaler, scaler_path)


def trail_fractals_2(side, tf, width, atr_spacing, thresh):
    tf_start = time.perf_counter()
    print(f"\n- Running Trail_fractals_2, {side}, {tf}, {width}, {atr_spacing}, {thresh}")

    results = create_dataset(side, tf, width, atr_spacing, thresh)

    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

    # split features from labels
    X = results.drop('win', axis=1)
    y = results.win  # pnl > threshold

    # random undersampling
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)

    # split off validation set
    X, X_val, y, y_val = train_test_split(X, y, train_size=0.9, random_state=43875, stratify=y)

    # split off validation labels
    z = X.pnl  # pnl > 0
    X = X.drop('pnl', axis=1)
    z_val = X_val.pnl
    X_val = X_val.drop('pnl', axis=1)
    cols = X.columns

    warn = f"*** WARNING only {len(X_val)} observations in validation set. ***" if len(X_val) < 30 else ''
    logger.debug(f"{len(y)} observations in {tf} {side} dataset. {warn}")

    # feature scaling
    scaler = QuantileTransformer()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    # quick feature selection
    # selector = SelectKBest(mutual_info_classif, k=7)
    # selector.fit(X, y)
    # cols_idx = list(selector.get_support(indices=True))
    # feature_names = [col for i, col in enumerate(cols) if i in cols_idx]
    # X = X[:, cols_idx]
    # X_val = X_val[:, cols_idx]

    # slow feature selection
    X, y, X_val, selected = feature_selection(X, y, X_val, fb_scorer)
    feature_names = [col for i, col in enumerate(cols) if i in selected]

    # hyperparameter optimisation
    model = mlf.fit_xgb(X, y, 1000)

    # Test model on training set
    d_train = DMatrix(X, label=z)
    y_pred = model.predict(d_train) > 0.5
    training_accuracy = accuracy_score(z, y_pred)
    training_f_beta = fbeta_score(z, y_pred, beta=0.333)
    logger.debug(f"Performance on validation set: accuracy: {training_accuracy:.1%}, f beta: {training_f_beta:.1%}")

    # Test model on validation set
    d_val = DMatrix(X_val, label=z_val)
    y_pred = model.predict(d_val) > 0.5
    accuracy = accuracy_score(z_val, y_pred)
    f_beta = fbeta_score(z_val, y_pred, beta=0.333)
    logger.debug(f"Performance on validation set: accuracy: {accuracy:.1%}, f beta: {f_beta:.1%}")

    # save models and info
    save_models(side, tf, width, atr_spacing, feature_names, thresh, X_val, model, scaler)

    tf_end = time.perf_counter()
    tf_elapsed = tf_end - tf_start
    logger.debug(f"TF 2 time taken: {int(tf_elapsed // 3600)}h {int(tf_elapsed // 60) % 60}m {tf_elapsed % 60:.1f}s")

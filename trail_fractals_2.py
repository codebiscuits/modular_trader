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
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif  # , chi2
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgbm
from optuna import Trial, logging as op_logging, visualization, integration, pruners, create_study
from optuna.samplers import TPESampler

op_logging.set_verbosity(op_logging.ERROR)
warnings.filterwarnings('ignore')

all_start = time.perf_counter()

logger = create_logger('trail_fractals_2', 'trail_fractals_2')

def create_dataset(side, tf, selection_method, num_pairs, frac_width, atr_spacing, thresh):
    records_path = Path(f"/home/ross/coding/modular_trader/records/trail_fractals_{tf}_"
                        f"None_{frac_width}_{atr_spacing}_{selection_method}_{num_pairs}")

    with open(records_path / "closed_trades.json", 'r') as real_file:
        real_records = json.load(real_file)
    with open(records_path / "closed_sim_trades.json", 'r') as sim_file:
        sim_records = json.load(sim_file)

    all_records = real_records | sim_records
    observations = []

    for position in all_records.values():
        signal = position['signal']
        if signal['direction'] != side:
            continue

        trade = position['trade']
        pnl = 0.0
        for t in trade:
            if t.get('rpnl'):
                pnl += t['rpnl']

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
            pnl=pnl > thresh
        )
        observations.append(observation)

    return pd.DataFrame(observations)

def feature_selection(X, y, scorer):
    # feature selection on base model
    selector_model = lgbm.LGBMClassifier(objective='binary',
                                         random_state=42,
                                         n_estimators=50,
                                         boosting='gbdt',
                                         verbosity=-1)
    # lgbm long: 59.4, 59.7, 58.4, 62.2 1h 13m - short: 64.1, 66.3, 65.1, 69.5 1h 1m

    # selector_model = RandomForestClassifier(max_depth=4, min_samples_split=3, n_jobs=-1, random_state=54986)
    # rfc long: 59.3, 59.6, 58.6, 66.7, 1m 36s - short: 66.0, 65.7, 65.7, 69.4, 1m 52s

    selector = SFS(estimator=selector_model, k_features='best', forward=False, floating=True, verbose=0,
                   scoring=scorer, n_jobs=-1)

    selector = selector.fit(X, y)
    X = selector.transform(X)
    print(f"Sequential FS selected: {selector.k_feature_names_}")

    sel_score = selector.k_score_
    print(f"Model score after feature selection: {sel_score:.1%}")

    return X, y

def trail_fractals_2(side, tf, selection_method, num_pairs, frac_width, atr_spacing, thresh):
    tf_start = time.perf_counter()

    results = create_dataset(side, tf, selection_method, num_pairs, frac_width, atr_spacing, thresh)

    fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)

    # split features from labels
    X = results.drop('pnl', axis=1)
    y = results.pnl
    cols = X.columns

    # random undersampling
    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)
    print(f"Number of observations in dataset: {len(y)}")

    # split off validation set
    X, X_val, y, y_val = train_test_split(X, y, train_size=0.9, random_state=43875)

    # feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)

    # feature selection
    # X, y = feature_selection(X, y, fb_scorer)

    # hyperparameter optimisation
    model = mlf.fit_lgbm(X, y)

    # test performance on unseen data
    best_params = model.params
    val_model = lgbm.LGBMClassifier(objective='binary', **best_params)
    val_model.fit(X, y)
    y_pred = val_model.predict(X_val)
    y_prob = val_model.predict_proba(X_val)
    accuracy = val_model.score(X_val, y_val)
    f_beta = fbeta_score(y_val, y_pred, beta=0.333)
    logger.debug(f"Performance on validation set: accuracy: {accuracy:.1%}, f beta: {f_beta:.1%}")

    tf_end = time.perf_counter()
    tf_elapsed = tf_end - tf_start
    logger.debug(
    f"\nTest time taken: {int(tf_elapsed // 3600)}h {int(tf_elapsed // 60) % 60}m {tf_elapsed % 60:.1f}s")

for thresh in [0.0, 0.1, 0.2, 0.4]:
    print(f"\nTesting {thresh}")
    trail_fractals_2('short', '1h', '1d_volumes', 30, 5, 2, thresh)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
logger.debug(f"\nTotal time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {all_elapsed % 60:.1f}s")

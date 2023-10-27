import numpy as np
import pandas as pd
from mt.resources import ml_funcs as mlf
import time
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone
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
from imblearn.under_sampling import ClusterCentroids
import warnings

warnings.filterwarnings('ignore')

logger = create_logger('trail_fractals', 'trail_fractals')
scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)


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


def find_collinear(X_train, corr_thresh):
    corr_matrix = X_train.corr()
    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_thresh)]

    # Iterate through the columns to drop to record pairs of correlated features
    record_collinear = []
    for column in to_drop:
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > corr_thresh])

        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > corr_thresh])
        drop_features = [column for _ in range(len(corr_features))]

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                          'corr_feature': corr_features,
                                          'corr_value': corr_values})

        # Add to dataframe
        record_collinear.append(temp_df)

    return pd.concat(record_collinear, axis=0, ignore_index=True)


def trail_fractals_1a(side, tf, width, atr_spacing, num_pairs, selection_method):
    loop_start = time.perf_counter()
    print(f"\n- Running Trail_fractals_1a, {side}, {tf}, {width}, {atr_spacing}, {num_pairs}, {selection_method}")
    data_len = 500
    pairs = mlf.get_margin_pairs(selection_method, num_pairs)

    X, y = create_dataset(pairs, data_len, tf, width, atr_spacing, side)

    # split data for fitting and calibration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

    # feature scaling
    # TODO when i refactor this into a pipeline, i will need to remove the equivalent step from the agent definition
    # X_train, _, cols = mlf.transform_columns(X_train, X_train)
    original_cols = list(X_train.columns)  # list of strings, names of all features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=original_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=original_cols)

    # feature selection
    print(f"feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

    # remove features with too many missing values
    nan_condition = X_train.columns[X_train.isnull().mean(axis=0) < 0.1]
    X_train = X_train[nan_condition]
    X_test = X_test[nan_condition]

    # remove low variance features
    variance_condition = X_train.columns[X_train.var() > 0.001]
    X_train = X_train[variance_condition]
    X_test = X_test[variance_condition]

    # remove features that are highly correlated with other features
    collinear_features = find_collinear(X_train, 0.5)
    X_train = X_train.drop(list(collinear_features.corr_feature), axis=1)
    X_test = X_test.drop(list(collinear_features.corr_feature), axis=1)
    cols = X_train.columns

    selector = mlf.feature_selection_1a(np.array(X_train), y_train, 1000, cols)
    selected = selector.k_feature_names_
    print(selected)

    # apply selector and scaler to X_train and X_test
    X_train = selector.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=selected)
    X_train = scaler.fit_transform(X_train)
    X_test = selector.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=selected)
    X_test = scaler.transform(X_test)

    # fit model
    print(f"Training on {X_train.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X_train = pd.DataFrame(X_train, columns=selected)
    grid_model = fit_gbc(X_train, y_train, scorer)
    # grid_model = fit_rfc(X_train, y_train, scorer)
    train_score = grid_model.score(X_train, y_train)
    print(f"Model score on train set after grid search: {train_score:.1%}")
    test_score = grid_model.score(X_test, y_test)
    print(f"Model score on test set after grid search: {test_score:.1%}")

    # calibrate model
    print(f"Calibrating on {X_test.shape[0]} observations: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    X_test = pd.DataFrame(X_test, columns=selected)
    cal_model = CalibratedClassifierCV(estimator=grid_model, cv='prefit', n_jobs=-1)
    cal_model.fit(X_test, y_test)
    cal_score = cal_model.score(X_test, y_test)
    print(f"Model score after calibration: {cal_score:.1%}")

    # save models and info
    mlf.save_models(
        "trail_fractals",
        f"{width}_{atr_spacing}",
        selection_method,
        num_pairs,
        side,
        tf,
        data_len,
        selected,
        pairs,
        cal_model,
        scaler,
        len(X_test)
    )

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"TF 1a test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

import time
import pandas as pd
from pathlib import Path
import mt.resources.ml_funcs as mlf
import mt.resources.indicators as ind
import numpy as np
from datetime import datetime
import statistics as stats
from itertools import product
if not Path('/pi_2.txt').exists():
    from sklearnex import patch_sklearn
    patch_sklearn()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from xgboost import XGBClassifier, DMatrix
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

if not Path('pi_2.txt').exists():
    import mt.update_ohlc


def backtest_oco(df_0, side, lookback, trim_ohlc=2000):
    """i can either target the opposite side of the channel or the mid-point, or both"""

    df_0 = df_0.reset_index(drop=True)
    atr_lb = 10
    df_0 = ind.atr(df_0, atr_lb)

    # identify potential entries
    rows = list(df_0.loc[df_0[f"entry_{side[0]}"]].index)

    results = []
    for row in rows:
        if row == len(df_0) - 1:
            break
        df = df_0[row:row + trim_ohlc].copy().reset_index(drop=True)
        entry = df.close.iloc[0]
        atr = df[f"atr-{atr_lb}"].iloc[0]

        if side == 'long':
            highest = df.high.max()
            target = df[f"hh_{lookback}"].iloc[0]
            stop = df[f"ll_{lookback}"].iloc[0] - atr
            rr = abs((target / entry) - 1) / abs((stop / entry) - 1)
            target_hit_idx = df.high.clip(upper=target).idxmax()
            stop_hit_idx = df.low.clip(lower=stop).idxmin()
            if (target > highest) or (stop_hit_idx < target_hit_idx):
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = (stop - entry) / entry
            elif target_hit_idx < stop_hit_idx:
                exit_row = target_hit_idx
                pnl_cat = 1
                pnl = (target - entry) / entry
            else:
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = 0
        else:
            lowest = df.low.min()
            target = df[f"ll_{lookback}"].iloc[0]
            stop = df[f"hh_{lookback}"].iloc[0] + atr
            rr = abs((target / entry) - 1) / abs((stop / entry) - 1)
            target_hit_idx = df.low.clip(lower=target).idxmin()
            stop_hit_idx = df.high.clip(upper=stop).idxmax()
            if (target < lowest) or (stop_hit_idx < target_hit_idx):
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = (entry - stop) / entry
            elif target_hit_idx < stop_hit_idx:
                exit_row = target_hit_idx
                pnl_cat = 1
                pnl = (entry - target) / entry
            else:
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = 0

        row_data = df_0.iloc[row - 1].to_dict()
        pnl_pct = pnl - (2 * 0.0015)  # subtract trading fees and slippage estimate

        row_res = dict(
            # idx=row,
            r_pct=atr,
            rr=rr,
            lifespan=exit_row,
            pnl_pct=pnl_pct,
            pnl_r=pnl_pct / atr,
            pnl_cat=pnl_cat
        )

        results.append(row_data | row_res)

        msg = f"trade lifespans getting close to trimmed ohlc length ({exit_row / trim_ohlc:.1%}), increase trim ohlc"
        if exit_row / trim_ohlc > 0.9:
            print(msg)

    return results


def prepare_strat_data(df, lookback):
    df[f"ll_{lookback}"] = df.low.rolling(lookback).min()
    df[f"hh_{lookback}"] = df.high.rolling(lookback).max()

    # df['channel_mid'] = (df[f"hh_{lookback}"] + df[f"ll_{lookback}"]) / 2
    # df['channel_width'] = (df[f"hh_{lookback}"] - df[f"ll_{lookback}"]) / df.channel_mid

    # df['broke_support'] = df.low == df[f"ll_{lookback}"]
    # df['broke_resistance'] = df.high == df[f"hh_{lookback}"]

    # df['close_above_sup'] = df.close > df[f"ll_{lookback}"].shift()
    # df['close_below_res'] = df.close < df[f"hh_{lookback}"].shift()

    df['channel_position'] = (df.close - df[f"ll_{lookback}"]) / (df[f"hh_{lookback}"] - df[f"ll_{lookback}"])

    df['entry_l'] = df.channel_position < 0.05
    df['entry_s'] = df.channel_position > 0.95

    # df['entry_l_price'] = df.close.loc[df.entry_l]
    # df['entry_s_price'] = df.close.loc[df.entry_s]

    # df['support_diff_z'] = abs(ind.z_score(df[f"ll_{lookback}"].ffill().pct_change(), lookback) * df.broke_support)
    # df['resistance_diff_z'] = abs(
    #     ind.z_score(df[f"hh_{lookback}"].ffill().pct_change(), lookback) * df.broke_resistance)

    return df


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


def generate_dataset(pairs, side, timeframe, lookback, data_len):
    all_res = []
    for pair in pairs:
        df = mlf.get_data(pair, timeframe)
        df = mlf.add_features(df, timeframe)
        df = prepare_strat_data(df, lookback)
        df = df.tail(data_len).reset_index(drop=True)
        res = backtest_oco(df, side, lookback)
        all_res.extend(res)
    res_df = pd.DataFrame(all_res).sort_values('timestamp').reset_index(drop=True)

    return res_df.dropna(axis=1)


def channel_run_1(side, timeframe, lookback, num_pairs, selection_method, data_len, num_trials):
    loop_start = time.perf_counter()
    print(f"\n- Running Channel Run 1, {side}, {timeframe}, {lookback}, {num_pairs}, {selection_method}")

    # generate dataset
    pairs = mlf.get_margin_pairs(selection_method, num_pairs)
    res_df = generate_dataset(pairs, side, timeframe, lookback, data_len)

    # split features from labels
    X, y, _ = mlf.features_labels_split(res_df)

    # undersampling
    # us = RandomUnderSampler(random_state=0)
    us = ClusterCentroids(random_state=0)
    X, y = us.fit_resample(X, y)

    # split data for fitting and calibration
    X_final, y_final = X.copy(), y.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=11, stratify=y_test)
    print(f"Training dataset: {len(X_train)} observations, Test sets: {len(X_test)}, Final set: {len(X_final)}")

    # feature scaling
    original_cols = list(X_train.columns) # list of strings, names of all features
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=original_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=original_cols)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=original_cols)

    # remove features with too many missing values
    nan_condition = X_train.columns[X_train.isnull().mean(axis=0) < 0.1]
    X_train = X_train[nan_condition]
    X_test = X_test[nan_condition]
    X_val = X_val[nan_condition]

    # remove low variance features
    variance_condition = X_train.columns[X_train.var() > 0.001]
    X_train = X_train[variance_condition]
    X_test = X_test[variance_condition]
    X_val = X_val[variance_condition]

    # remove features that are highly correlated with other features
    collinear_features = find_collinear(X_train, 0.5)
    X_train = X_train.drop(list(collinear_features.corr_feature), axis=1)
    X_test = X_test.drop(list(collinear_features.corr_feature), axis=1)
    X_val = X_val.drop(list(collinear_features.corr_feature), axis=1)

    # mutual info feature selection
    print(f"feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    cols = list(X_train.columns) # list of strings, names of all features
    selector = SelectKBest(mutual_info_classif, k=15)
    selector.fit(X_train, y_train)
    mi_cols_idx = list(selector.get_support(indices=True))
    selected_columns = [col for i, col in enumerate(cols) if i in mi_cols_idx]
    X_train = pd.DataFrame(selector.transform(X_train), columns=selected_columns)
    X_test = pd.DataFrame(selector.transform((X_test)), columns=selected_columns)
    X_val = pd.DataFrame(selector.transform((X_val)), columns=selected_columns)
    print(selected_columns)

    # sequential feature selection
    sfs_selector_model = RandomForestClassifier()
    sfs_selector = SFS(estimator=sfs_selector_model, k_features='best', forward=False,
                         floating=True, verbose=0, scoring='accuracy', n_jobs=-1)
    sfs_selector = sfs_selector.fit(X_train, y_train)
    cols_idx = list(sfs_selector.k_feature_idx_)
    X_train = sfs_selector.transform(X_train)
    X_test = sfs_selector.transform(X_test)
    X_val = sfs_selector.transform(X_val)

    # hyperparameter optimisation
    def objective(trial):
        # Suggest values for hyperparameters
        # criterion = trial.suggest_categorical('criterion', 'gini', 'log_loss')
        # min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 12, 32)
        max_features = trial.suggest_float("max_features", 0.1, 1.0)
        max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
        ccp_alpha = trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True)

        # Create and fit random forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='log_loss',
            max_depth=max_depth,
            min_samples_split=2,
            max_features=max_features,
            max_samples=max_samples,
            ccp_alpha=ccp_alpha,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Score model
        scores = cross_val_score(model, X_test, y_test, verbose=0, n_jobs=-1)
        avg_score = stats.mean(scores)

        # Return score
        return avg_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials, n_jobs=-1)
    best_trials = [trial.params for trial in study.trials if trial.values[0] >= (study.best_value - 0.001)]
    best_df = pd.DataFrame(best_trials)
    best_df.describe()
    best_params = best_df.median(axis=0).to_dict()

    # remove features with low permutation importance
    imp_model = RandomForestClassifier()
    imp_model.fit(X_train, y_train)
    importances = permutation_importance(imp_model, X_test, y_test, n_repeats=1000, random_state=42, n_jobs=-1)
    imp_mean = pd.Series(importances.importances_mean, index=sfs_selector.k_feature_names_)
    imp_std = pd.Series(importances.importances_std, index=sfs_selector.k_feature_names_)
    final_features = list(imp_mean.index[imp_mean > 0.01])
    print(final_features)

    # final validation score before training production model
    X_train = pd.DataFrame(X_train, columns=sfs_selector.k_feature_names_)
    X_val = pd.DataFrame(X_val, columns=sfs_selector.k_feature_names_)
    X_train = X_train[final_features]
    X_val = X_val[final_features]
    val_model = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        criterion='log_loss',
        max_depth=int(best_params['max_depth']),
        min_samples_split=2,
        max_features=best_params['max_features'],
        max_samples=best_params['max_samples'],
        ccp_alpha=best_params['ccp_alpha'],
        random_state=42,
        n_jobs=-1
    )
    val_model.fit(X_train, y_train)
    score = val_model.score(X_val, y_val)
    print(f"Final model validation score: {score:.1%}")

    # train final model
    X_final = X_final[final_features]
    scaler = MinMaxScaler()
    X_final = scaler.fit_transform(X_final)
    X_final = pd.DataFrame(X_final, columns=final_features)

    final_model = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        criterion='log_loss',
        max_depth=int(best_params['max_depth']),
        min_samples_split=2,
        max_features=best_params['max_features'],
        max_samples=best_params['max_samples'],
        ccp_alpha=best_params['ccp_alpha'],
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_final, y_final)

    # save models and info
    mlf.save_models(
        "channel_run",
        f"{lookback}",
        selection_method,
        num_pairs,
        side,
        timeframe,
        data_len,
        final_features,
        pairs,
        final_model,
        scaler,
        len(X_final)
    )

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"TF 1a test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

all_start = time.perf_counter()

sides = ['long', 'short']
timeframes = ['15m', '30m', '1h', '4h']

for side, timeframe in product(sides, timeframes):
    channel_run_1(side, timeframe, 200, 150, '1w_volumes', 5000, 1000)

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"Total time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {int(all_elapsed % 60)}s")

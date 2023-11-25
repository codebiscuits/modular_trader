import time
import pandas as pd
from pathlib import Path
import mt.resources.ml_funcs as mlf
import mt.resources.indicators as ind
import numpy as np
from datetime import datetime, timezone
import statistics as stats
from itertools import product
from mt.resources.loggers import create_logger
import plotly.graph_objects as go
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
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
    # import mt.update_ohlc
    import mt.async_update_ohlc

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = create_logger('model_training')
use_local_data = True  # if false, uses trade records from the pi

fb_scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)


def backtest_oco(df_0, side, lookback, goal, trim_ohlc=2200):
    """i can either target the opposite side of the channel or the mid-point, or both"""

    df_0 = df_0.reset_index(drop=True)
    atr_lb = 10
    df_0 = ind.atr(df_0, atr_lb)

    # identify potential entries
    rows = list(df_0.loc[df_0[f"entry_{side[0]}"]].index)

    count = 0
    exit_idxs = []
    results = []
    for row in rows:
        if row == len(df_0) - 1:
            break
        df = df_0[row:row + trim_ohlc].copy().reset_index(drop=True)
        entry = df.close.iloc[0]
        atr = df[f"atr-{atr_lb}"].iloc[0]

        if side == 'long':
            highest = df.high.max()
            lowest = df.low.min()
            if goal == 'mid':
                target = df[f"channel_mid_{lookback}"].iloc[0]
            else:
                target = df[f"hh_{lookback}"].iloc[0]
            stop = df[f"ll_{lookback}"].iloc[0] - atr
            rr = abs((target / entry) - 1) / abs((stop / entry) - 1)
            target_hit_idx = df.high.clip(upper=target).idxmax()
            stop_hit_idx = df.low.clip(lower=stop).idxmin()
            if (stop > lowest) and ((target > highest) or (stop_hit_idx < target_hit_idx)):  # stop hit
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = (stop - entry) / entry
            elif (target < highest) and ((stop < lowest) or (target_hit_idx < stop_hit_idx)):  # target hit
                exit_row = target_hit_idx
                pnl_cat = 1
                pnl = (target - entry) / entry
            else:  # neither hit
                exit_row = df.index[-1]
                logger.debug(f'backtester running out of ohlc data at {exit_row} periods')
                continue
        else:  # if side == 'short'
            highest = df.high.max()
            lowest = df.low.min()
            if goal == 'mid':
                target = df[f"channel_mid_{lookback}"].iloc[0]
            else:
                target = df[f"ll_{lookback}"].iloc[0]
            stop = df[f"hh_{lookback}"].iloc[0] + atr
            rr = abs((target / entry) - 1) / abs((stop / entry) - 1)
            target_hit_idx = df.low.clip(lower=target).idxmin()  # returns index of earliest instance of lowest value
            stop_hit_idx = df.high.clip(upper=stop).idxmax()  # returns index of earliest instance of highest value
            if (highest > stop) and ((lowest > target) or (stop_hit_idx < target_hit_idx)):  # stop hit
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = (entry - stop) / entry
            elif (lowest < target) and ((highest < stop) or (target_hit_idx < stop_hit_idx)):  # target hit
                exit_row = target_hit_idx
                pnl_cat = 1
                pnl = (entry - target) / entry
            else:  # neither hit
                exit_row = df.index[-1]
                logger.debug(f'backtester running out of ohlc data at {exit_row} periods')
                continue

        row_data = df_0.iloc[row - 1].to_dict()
        pnl_pct = pnl - (3 * 0.0015)  # subtract trading fees and estimate for slippage + spread

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

        msg = (f"trade lifespans getting close to trimmed ohlc length ({trim_ohlc = }), {exit_row = }, {len(df) = } "
               f"({exit_row / len(df):.1%})")
        if exit_row / len(df) > 0.9:
            print(msg)
            count += 1

        exit_idxs.append(exit_row)  # this is to look at the distributions of how much trimmed ohlc data is being used

    return results, exit_idxs, count


def channel_run_entries(df, lookback):
    df[f"ll_{lookback}"] = df.low.rolling(lookback).min()
    df[f"hh_{lookback}"] = df.high.rolling(lookback).max()

    df[f'channel_mid_{lookback}'] = (df[f"hh_{lookback}"] + df[f"ll_{lookback}"]) / 2
    # df['channel_width'] = (df[f"hh_{lookback}"] - df[f"ll_{lookback}"]) / df[f"channel_mid_{lookback}"]

    # df['broke_support'] = df.low == df[f"ll_{lookback}"]
    # df['broke_resistance'] = df.high == df[f"hh_{lookback}"]

    # df['close_above_sup'] = df.close > df[f"ll_{lookback}"].shift()
    # df['close_below_res'] = df.close < df[f"hh_{lookback}"].shift()

    df[f'channel_position_{lookback}'] = (df.close - df[f"ll_{lookback}"]) / (df[f"hh_{lookback}"] - df[f"ll_{lookback}"])

    df['entry_l'] = df.channel_position < 0.1
    df['entry_s'] = df.channel_position > 0.9

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

    # Select the features with correlations with absolute value above the threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_thresh)]

    # Iterate through the columns to drop to record pairs of correlated features
    record_collinear = []
    for column in to_drop:
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > corr_thresh])
        drop_features = [column for _ in range(len(corr_features))]
        if drop_features:
            record_collinear.extend(drop_features)

    return list(set(record_collinear))


def plot_exit_row_dist(name, side, timeframe, all_exit_idx, data_len, param_1, param_2):
    trace = go.Histogram(x=all_exit_idx,
                         nbinsx=25,
                         autobinx=False,
                         xbins=dict(start=min(all_exit_idx),
                                    end=max(all_exit_idx)),
                         marker=dict(color='rgb(37, 150, 190)'),
                         histnorm='percent')
    layout = go.Layout(title=f"Distribution of exit row from backtest_trail_fractal. {data_len = }")
    plot_path = Path("/home/ross/coding/modular_trader/machine_learning/exit_idx_distrib")
    plot_path.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=go.Histogram(trace), layout=layout)
    fig.write_image(plot_path / f"{name}_{side}_{timeframe}_{param_1}_{param_2}.png")


def generate_channel_run_dataset(pairs: list, side: str, timeframe: str, strat_params: tuple, data_len: int):
    print(f"data generation began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

    lookback, goal = strat_params

    all_res = []
    all_exit_idx = []
    for n, pair in enumerate(pairs):
        df = mlf.get_data(pair, timeframe)
        df = mlf.add_features(df, timeframe)
        df = channel_run_entries(df, lookback)
        df = df.tail(data_len).reset_index(drop=True)
        res, exit_idxs, count = backtest_oco(df, side, lookback, goal)
        all_res.extend(res)
        all_exit_idx.extend(exit_idxs)

    res_df = pd.DataFrame(all_res).sort_values('timestamp').reset_index(drop=True)

    # save plot of exit_row distribution
    plot_exit_row_dist('channel_run', side, timeframe, all_exit_idx, data_len, lookback, goal)

    return res_df.dropna(axis=1)


def generate_trail_fractal_dataset(pairs: list, side: str, timeframe: str, strat_params: tuple, data_len: int):
    print(f"data generation began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

    width, atr_spacing = strat_params

    all_res = pd.DataFrame()
    all_exit_idx = []
    for pair in pairs:
        df = mlf.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
        df = mlf.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
        res_list, exit_idxs, count = mlf.trail_fractal(df, width, atr_spacing, side)
        res_df = pd.DataFrame(res_list).dropna(axis=0).reset_index(drop=True)
        all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)
        all_exit_idx.extend(exit_idxs)

    all_res = all_res.sort_values('timestamp').reset_index(drop=True)

    # save plot of exit_row distribution
    plot_exit_row_dist('trail_fractals', side, timeframe, all_exit_idx, data_len, width, atr_spacing)

    return all_res.dropna(axis=1)


def ttv_split(X, y):
    """first retains a copy of the full set to train the final model on, then performs stratified split into train and
    test sets, then further splits test set into test and validation sets"""
    split = 0.5 if len(X) < 80 else 0.6  # if there aren't many observations, it puts a bit more in the test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=11, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, random_state=11, stratify=y_test)
    print(f"Training dataset: {len(X_train)} observations, Test sets: {len(X_test)}, Final set: {len(X)}")

    return X_train, X_test, X_val, y_train, y_test, y_val


def scale_features(X_train, X_test, X_val, scaler):
    original_cols = list(X_train.columns)  # list of strings, names of all features
    scaler = scaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=original_cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=original_cols)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=original_cols)

    return X_train, X_test, X_val


def eliminate_features(X_train, X_test, X_val, y_train):
    print(f"feature elimination began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

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
    if collinear_features:
        X_train = X_train.drop(collinear_features, axis=1)
        X_test = X_test.drop(collinear_features, axis=1)
        X_val = X_val.drop(collinear_features, axis=1)

    # mutual info feature selection
    cols: list[str] = list(X_train.columns)  # names of all features
    mi_k = len(cols) if len(cols) in [1, 2] else min(int(len(cols) * 0.8), 15)  # 15 max, 1 or 2 min, otherwise 80%
    selector = SelectKBest(mutual_info_classif, k=mi_k)
    selector.fit(X_train, y_train)
    mi_cols_idx = list(selector.get_support(indices=True))
    selected_columns = [col for i, col in enumerate(cols) if i in mi_cols_idx]
    X_train = pd.DataFrame(selector.transform(X_train), columns=selected_columns)
    X_test = pd.DataFrame(selector.transform((X_test)), columns=selected_columns)
    X_val = pd.DataFrame(selector.transform((X_val)), columns=selected_columns)
    # print(selected_columns)

    return X_train, X_test, X_val, selected_columns


def rand_forest_sfs(X_train, X_test, X_val, y_train):
    print(f"sequential feature selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    sfs_selector_model = RandomForestClassifier()
    sfs_selector = SFS(estimator=sfs_selector_model, k_features='best', forward=False,
                       floating=True, verbose=0, scoring='accuracy', n_jobs=-1)
    sfs_selector = sfs_selector.fit(X_train, y_train)
    cols_idx = list(sfs_selector.k_feature_idx_)
    X_train = sfs_selector.transform(X_train)
    X_test = sfs_selector.transform(X_test)
    X_val = sfs_selector.transform(X_val)

    return X_train, X_test, X_val, sfs_selector


class RFObjective(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test

    def __call__(self, trial):
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
        model.fit(self.X_train, self.y_train)

        # Score model
        scores = cross_val_score(model, self.X_test, self.y_test, verbose=0, n_jobs=-1)
        avg_score = stats.mean(scores)

        return avg_score


def optimise(objective, num_trials):
    print(f"optimisation began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials, n_jobs=-1)
    best_trials = [trial.params for trial in study.trials if trial.values[0] >= (study.best_value - 0.001)]
    best_df = pd.DataFrame(best_trials)

    return best_df.median(axis=0).to_dict()


def rf_perm_importance(X_train, X_test, y_train, y_test, rf_sfs):
    print(f"permutation importance selection began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    imp_model = RandomForestClassifier()
    imp_model.fit(X_train, y_train)
    importances = permutation_importance(imp_model, X_test, y_test, n_repeats=1000, random_state=42, n_jobs=-1)
    imp_mean = pd.Series(importances.importances_mean, index=rf_sfs.k_feature_names_)
    imp_std = pd.Series(importances.importances_std, index=rf_sfs.k_feature_names_)
    final_features = list(imp_mean.index[imp_mean > 0.01])
    print(final_features)

    return final_features


def validate_findings(X_train, X_val, y_train, y_val, sfs_selector, final_features, best_params, fb_scorer):
    print(f"model validation began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
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
    train_score = val_model.score(X_train, y_train)
    val_score = val_model.score(X_val, y_val)

    fb_train = fb_scorer(val_model, X_train, y_train)
    fb_val = fb_scorer(val_model, X_val, y_val)

    print(f"Final model validation accuracy: {val_score:.1%}, f beta: {fb_val:.1%}")

    return train_score, val_score, fb_train, fb_val


def final_rf_train_and_save(mode, strat_name, X_final, y_final, final_features, best_params,
                            pairs, num_pairs, selection_method, strat_params, data_len):
    print(f"final model train and save began: {datetime.now().strftime('%Y/%m/%d %H:%M')}")

    X_final = X_final[final_features]
    final_scaler = MinMaxScaler()
    X_final = final_scaler.fit_transform(X_final)
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
    strat_params = [str(p) for p in strat_params]
    mlf.save_models(
        mode,
        strat_name,
        "_".join([str(sp) for sp in strat_params]),
        selection_method,
        num_pairs,
        side,
        timeframe,
        data_len,
        final_features,
        pairs,
        final_model,
        final_scaler,
        len(X_final)
    )


def load_secondary_data(strat_name, timeframe, strat_params, selection_method, num_pairs):
    root_dir = '/home/ross/coding' if use_local_data else '/home/ross/coding/pi_1'

    records_path = Path(f"{root_dir}/modular_trader/records/{strat_name}_{timeframe}_None_"
                        f"{'_'.join([str(sp) for sp in strat_params])}_{selection_method}_{num_pairs}")
    logger.debug(records_path)

    try:
        with open(records_path / "closed_trades.json", 'r') as real_file:
            real_records_1 = json.load(real_file)
    except FileNotFoundError:
        real_records_1 = {}
    try:
        with open(records_path / "closed_sim_trades.json", 'r') as sim_file:
            sim_records_1 = json.load(sim_file)
    except FileNotFoundError:
        sim_records_1 = {}

    return real_records_1 | sim_records_1


def create_risk_dataset(strat_name: str, side: str, timeframe: str, strat_params: tuple,
                        num_pairs: int, selection_method: str, thresh):
    """the two target columns (pnl and win) are both booleans and mean slightly different things. pnl is True if the
    final pnl of the trade was above zero, and win is True if the final pnl of the trade was above the threshold"""

    all_records = load_secondary_data(strat_name, timeframe, strat_params, selection_method, num_pairs)
    logger.debug(f"risk data length: {len(all_records)}")

    observations = []
    for position in all_records.values():
        signal = position['signal']

        # remove irrelevant/unusable signals
        if ((signal['direction'] != side) or
                (signal.get('confidence_l') in [None, 0]) or
                (signal.get('market_rank_1d') in [None, 0])):
            continue

        # sum up trade rpnl
        trade = position['trade']
        pnl = 0.0
        for t in trade:
            if t.get('rpnl'):
                pnl += t['rpnl']

        try:
            observation = dict(
                # asset=signal['asset'],
                conf_l=signal['conf_rf_usdt_l'],
                conf_s=signal['conf_rf_usdt_s'],
                inval_dist=signal['inval_dist'],
                mkt_rank_1d=signal['market_rank_1d'],
                mkt_rank_1w=signal['market_rank_1w'],
                mkt_rank_1m=signal['market_rank_1m'],
                pnl=pnl > 0,
                win=pnl > thresh
            )
            if strat_name == 'channel_run':
                observation['rr'] = signal['rr']

            observations.append(observation)
        except KeyError:
            # logger.debug("observation couldn't be added because the signal was missing a key")
            # logger.debug(pformat(position))
            continue

    return pd.DataFrame(observations)


def create_perf_dataset(strat_name: str, side: str, timeframe: str, strat_params: tuple,
                        num_pairs: int, selection_method: str, thresh):
    """the two target columns (pnl and win) are both booleans and mean slightly different things. pnl is True if the
    final pnl of the trade was above zero, and win is True if the final pnl of the trade was above the threshold"""

    all_records = load_secondary_data(strat_name, timeframe, strat_params, selection_method, num_pairs)
    logger.debug(f"perf data length: {len(all_records)}")

    observations = []
    for position in all_records.values():
        signal = position['signal']
        wanted = signal.get('wanted', True)
        if (signal['direction'] != side) or (signal.get('perf_ema4') is None) or (wanted is False):
            continue

        trade = position['trade']
        pnl = 0.0
        for t in trade:
            if t.get('rpnl'):
                pnl += t['rpnl']

        try:
            observation = dict(
                # asset=signal['asset'],
                perf_ema_4=signal['perf_ema4'],
                perf_ema_8=signal['perf_ema8'],
                perf_ema_16=signal['perf_ema16'],
                perf_ema_32=signal['perf_ema32'],
                perf_ema_64=signal['perf_ema64'],
                perf_ema_128=signal['perf_ema128'],
                perf_ema4_roc=signal['perf_ema4_roc'],
                perf_ema8_roc=signal['perf_ema8_roc'],
                perf_ema16_roc=signal['perf_ema16_roc'],
                perf_ema32_roc=signal['perf_ema32_roc'],
                perf_ema64_roc=signal['perf_ema64_roc'],
                perf_ema128_roc=signal['perf_ema128_roc'],
                perf_sum_4=signal['perf_sum4'],
                perf_sum_8=signal['perf_sum8'],
                perf_sum_16=signal['perf_sum16'],
                perf_sum_32=signal['perf_sum32'],
                perf_sum_64=signal['perf_sum64'],
                perf_sum_128=signal['perf_sum128'],
                pnl=pnl > 0,
                win=pnl > thresh
            )
            observations.append(observation)
        except KeyError:
            # logger.debug("observation couldn't be added because the signal was missing a key")
            # logger.debug(pformat(position))
            pass

    df = pd.DataFrame(observations)
    df = df.fillna(0.0)

    return df


def train_primary(strat_name: str, side: str, timeframe: str, strat_params: tuple,
                  num_pairs: int, selection_method: str, data_len: int, num_trials: int):
    loop_start = time.perf_counter()
    print(f"\n*1* {datetime.now().strftime('%H:%M:%S')} Running {strat_name} primary model, {side}, {timeframe}, "
          f"{', '.join([str(p) for p in strat_params])}, {num_pairs}, {selection_method} primary")

    # generate dataset
    pairs = mlf.get_margin_pairs(selection_method, num_pairs)
    if strat_name == 'channel_run':
        res_df = generate_channel_run_dataset(pairs, side, timeframe, strat_params, data_len)
    elif strat_name == 'trail_fractals':
        res_df = generate_trail_fractal_dataset(pairs, side, timeframe, strat_params, data_len)

    # split features from labels
    X, y, _ = mlf.features_labels_split(res_df)

    # balance classes
    if (len(y.unique()) < 2) or (y.value_counts().iloc[0] < 20) or (y.value_counts().iloc[1] < 20):
        logger.debug('stopped - not enough samples in both classes for cross-validation etc')
        return  # need enough samples in each class for cross-validation etc
    # us = RandomUnderSampler(random_state=0)
    us = ClusterCentroids(random_state=0)
    X, y = us.fit_resample(X, y)

    logger.debug(f"Length of dataset: {len(y)}")

    # split data for fitting and calibration
    X_train, X_test, X_val, y_train, y_test, y_val = ttv_split(X, y)

    if (y_test.value_counts().iloc[0] < 6) or (y_test.value_counts().iloc[1] < 6):
        logger.debug('stopped - not enough samples in both classes for cross-validation etc')
        return

    # feature scaling
    X_train, X_test, X_val = scale_features(X_train, X_test, X_val, QuantileTransformer)

    # feature selection
    try:
        X_train, X_test, X_val, selected_columns = eliminate_features(X_train, X_test, X_val, y_train)
        X_train, X_test, X_val, rf_sfs = rand_forest_sfs(X_train, X_test, X_val, y_train)
    except ValueError as e:
        logger.debug(e)
        return

    # hyperparameter optimisation
    best_params = optimise(RFObjective(X_train, X_test, y_train, y_test), num_trials)

    # remove features with low permutation importance
    final_features = rf_perm_importance(X_train, X_test, y_train, y_test, rf_sfs)
    if len(final_features) == 0:
        logger.debug('stopped - no features have any predictive power')
        return

    # final validation score before training production model
    train_score, val_score, fb_train, fb_val = validate_findings(X_train, X_val, y_train, y_val, rf_sfs, final_features,
                                                                 best_params, fb_scorer)

    # train final model
    final_rf_train_and_save('tech', strat_name, X, y, final_features, best_params,
                            pairs, num_pairs, selection_method, strat_params, data_len)

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"{strat_name} Technical test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

    return dict(
        strat_name=strat_name,
        side=side,
        timeframe=timeframe,
        strat_params=f"{strat_params[0]}_{strat_params[1]}",
        num_pairs=num_pairs,
        selection_method=selection_method,
        n_final_features=len(final_features),
        total_time=loop_elapsed,
        train_score=train_score,
        val_score=val_score,
        fb_train=fb_train,
        fb_val=fb_val
    )


def train_secondary(mode: str, strat_name: str, side: str, timeframe: str, strat_params: tuple,
                    num_pairs: int, selection_method: str, thresh: float, num_trials: int):
    """this function can be used to train the risk model or the performance model, selected by the mode parameter"""

    loop_start = time.perf_counter()
    print(f"\n*{2 if mode == 'risk' else 3}* {datetime.now().strftime('%H:%M:%S')} Running {strat_name} {mode} model "
          f"training, {side}, {timeframe}, {', '.join([str(p) for p in strat_params])}, {num_pairs}, {selection_method}")

    if mode == 'risk':
        results = create_risk_dataset(strat_name, side, timeframe, strat_params, num_pairs, selection_method, thresh)
    elif mode == 'perf':
        results = create_perf_dataset(strat_name, side, timeframe, strat_params, num_pairs, selection_method, thresh)

    if len(results) == 0:
        logger.debug('stopped - no samples in dataset')
        return

    # split features from labels
    X = results.drop('win', axis=1)
    y = results.win  # pnl > threshold

    if mode == 'risk':
        X['mkt_rank_1d'] = X.mkt_rank_1d.fillna(0.5)
        X['mkt_rank_1w'] = X.mkt_rank_1w.fillna(0.5)
        X['mkt_rank_1m'] = X.mkt_rank_1m.fillna(0.5)

    # balance classes
    if (len(y.unique()) < 2) or (y.value_counts().iloc[0] < 20) or (y.value_counts().iloc[1] < 20):
        logger.debug('stopped - not enough samples in both classes for cross-validation etc')
        return  # need enough samples in each class for cross-validation etc
    # us = RandomUnderSampler(random_state=0)
    us = ClusterCentroids(random_state=0)
    try:
        X, y = us.fit_resample(X, y)
    except ValueError as e:
        logger.debug(e)
        logger.debug(X)
        return

    logger.debug(f"Length of dataset: {len(y)}")

    # split off validation set
    X_train, X_test, X_val, y_train, y_test, y_val = ttv_split(X, y)
    if (y_test.value_counts().iloc[0] < 6) or (y_test.value_counts().iloc[1] < 6):
        logger.debug('stopped - not enough samples in both classes for cross-validation etc')
        return

    # split off validation labels
    z_train = X_train.pnl  # pnl > 0
    X_train = X_train.drop('pnl', axis=1)
    z_test = X_test.pnl
    X_test = X_test.drop('pnl', axis=1)
    z_val = X_val.pnl
    X_val = X_val.drop('pnl', axis=1)
    cols = X_train.columns

    warn = f"*** WARNING only {len(X_val)} observations in validation set. ***" if len(X_val) < 30 else ''
    logger.debug(f"{len(y)} observations in {timeframe} {side} dataset. {warn}")

    # feature scaling
    X_train, X_test, X_val = scale_features(X_train, X_test, X_val, QuantileTransformer)

    # feature selection
    try:
        X_train, X_test, X_val, selected_columns = eliminate_features(X_train, X_test, X_val, y_train)
        X_train, X_test, X_val, rf_sfs = rand_forest_sfs(X_train, X_test, X_val, y_train)
    except ValueError as e:
        logger.debug(e)
        return

    # hyperparameter optimisation
    best_params = optimise(RFObjective(X_train, X_test, y_train, y_test), num_trials)

    # remove features with low permutation importance
    final_features = rf_perm_importance(X_train, X_test, y_train, y_test, rf_sfs)
    if len(final_features) == 0:
        logger.debug('stopped - no features have any predictive power')
        return

    # final validation score before training production model
    train_score, val_score, fb_train, fb_val = validate_findings(X_train, X_val, z_train, z_val, rf_sfs, final_features,
                                                                 best_params, fb_scorer)

    # train final model
    final_rf_train_and_save(mode, strat_name, X, y, final_features, best_params,
                            [], num_pairs, selection_method, strat_params, 'na')

    loop_end = time.perf_counter()
    loop_elapsed = loop_end - loop_start
    print(f"{strat_name} {mode} test time taken: {int(loop_elapsed // 60)}m {loop_elapsed % 60:.1f}s")

    return dict(
        strat_name=strat_name,
        side=side,
        timeframe=timeframe,
        strat_params=f"{strat_params[0]}_{strat_params[1]}",
        num_pairs=num_pairs,
        selection_method=selection_method,
        n_final_features=len(final_features),
        total_time=loop_elapsed,
        train_score=train_score,
        val_score=val_score,
        fb_train=fb_train,
        fb_val=fb_val
    )


if __name__ == '__main__':
    all_start = time.perf_counter()

    sides = ['long', 'short']
    timeframes = ['15m', '30m', '1h', '4h', '12h', '1d']
    num_trials = 500

    # TODO i need to write a 'retrain_primary' function that just fits a model with preselected features and params to the
    #  new data, which i can run with the secondary training every day, and run the full train_primary once a week

    all_stats = []

    for side, timeframe in product(sides, timeframes):
        print(f"\n\n*******\nTesting {side} {timeframe}\n*******\n")
        if timeframe in ['15m', '30m', '1h', '4h']:
            all_stats.append(train_primary('channel_run', side, timeframe, (50, 'edge'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (50, 'edge'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (50, 'edge'), 50, '1w_volumes', 0.4, num_trials))

            all_stats.append(train_primary('channel_run', side, timeframe, (50, 'mid'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (50, 'mid'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (50, 'mid'), 50, '1w_volumes', 0.4, num_trials))

            all_stats.append(train_primary('channel_run', side, timeframe, (100, 'edge'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (100, 'edge'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (100, 'edge'), 50, '1w_volumes', 0.4, num_trials))

            all_stats.append(train_primary('channel_run', side, timeframe, (100, 'mid'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (100, 'mid'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (100, 'mid'), 50, '1w_volumes', 0.4, num_trials))

            all_stats.append(train_primary('channel_run', side, timeframe, (200, 'edge'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (200, 'edge'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (200, 'edge'), 50, '1w_volumes', 0.4, num_trials))

            all_stats.append(train_primary('channel_run', side, timeframe, (200, 'mid'), 50, '1w_volumes', 2500, num_trials))
            all_stats.append(train_secondary('risk', 'channel_run', side, timeframe, (200, 'mid'), 50, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'channel_run', side, timeframe, (200, 'mid'), 50, '1w_volumes', 0.4, num_trials))

        if timeframe in ['1h', '4h', '12h', '1d']:
            all_stats.append(train_primary('trail_fractals', side, timeframe, (5, 2), 30, '1d_volumes', 200, num_trials))
            all_stats.append(train_secondary('risk', 'trail_fractals', side, timeframe, (5, 2), 30, '1d_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'trail_fractals', side, timeframe, (5, 2), 30, '1d_volumes', 0.4, num_trials))

            all_stats.append(train_primary('trail_fractals', side, timeframe, (5, 2), 30, '1w_volumes', 200, num_trials))
            all_stats.append(train_secondary('risk', 'trail_fractals', side, timeframe, (5, 2), 30, '1w_volumes', 0.4, num_trials))
            all_stats.append(train_secondary('perf', 'trail_fractals', side, timeframe, (5, 2), 30, '1w_volumes', 0.4, num_trials))

    all_stats = [record for record in all_stats if record is not None]
    stats_df = pd.DataFrame().from_records(all_stats)
    now_date = datetime.now(tz=timezone.utc).strftime('%d-%m-%y')
    stats_df.to_parquet(f'/home/ross/coding/modular_trader/machine_learning/training_stats_{now_date}.parquet')

    all_end = time.perf_counter()
    all_elapsed = all_end - all_start
    print(f"Total time taken: {int(all_elapsed // 3600)}h {int(all_elapsed // 60) % 60}m {int(all_elapsed % 60)}s")

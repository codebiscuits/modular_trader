import time
import pandas as pd
from pathlib import Path
import mt.resources.ml_funcs as mlf
import mt.resources.indicators as ind
import numpy as np
from datetime import datetime, timezone
import statistics as stats
from itertools import product
import json
import joblib
# if not Path('/pi_2.txt').exists():
#     from sklearnex import patch_sklearn
#     patch_sklearn()
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


def backtest_oco(df_0, side, lookback, trim_ohlc=2200):
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
            lowest = df.low.min()
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
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = 0
        else:  # if side == 'short'
            highest = df.high.max()
            lowest = df.low.min()
            target = df[f"ll_{lookback}"].iloc[0]
            stop = df[f"hh_{lookback}"].iloc[0] + atr
            rr = abs((target / entry) - 1) / abs((stop / entry) - 1)
            target_hit_idx = df.low.clip(lower=target).idxmin()
            stop_hit_idx = df.high.clip(upper=stop).idxmax()
            if (highest > stop) and ((lowest > target) or (stop_hit_idx < target_hit_idx)):  # stop hit
                exit_row = stop_hit_idx
                pnl_cat = 0
                pnl = (entry - stop) / entry
            elif (lowest < target) and ((highest < stop) or (target_hit_idx < stop_hit_idx)):  # target hit
                exit_row = target_hit_idx
                pnl_cat = 1
                pnl = (entry - target) / entry
            else:  # neither hit
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


def channel_run_entries(df, lookback):
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


def generate_channel_run_dataset(pairs, side, timeframe, lookback, data_len):
    all_res = []
    for pair in pairs:
        df = mlf.get_data(pair, timeframe)
        df = mlf.add_features(df, timeframe)
        df = channel_run_entries(df, lookback)
        df = df.tail(data_len).reset_index(drop=True)
        res = backtest_oco(df, side, lookback)
        all_res.extend(res)
    res_df = pd.DataFrame(all_res).sort_values('timestamp').reset_index(drop=True)

    return res_df.dropna(axis=1)

pairs = ['BTCUSDT']
side = 'short'
tf = '15m'
lb = 200
data_len = 2000
sel_method = '1w_volumes'
num_pairs = 1

res_df = generate_channel_run_dataset(pairs, side, tf, lb, data_len)

# split features from labels
X, y, _ = mlf.features_labels_split(res_df)

features = ['chan_mid_width_200', 'day_of_week', 'weighted_50_bull_doji', 'weighted_50_bear_doji', 'unweighted_bull_doji', 'unweighted_bear_doji', 'fractal_trend_age_long', 'fractal_trend_age_short', 'spooky_num_prox', 'big_round_nums_close_20', 'rsi_50_above_70', 'ema_48_above_192', 'volume_climax_dn_50', 'vol_delta']

X_final = X[features]

scaler = MinMaxScaler()
X_final = scaler.fit_transform(X_final)
X_final = pd.DataFrame(X_final, columns=features)

best_params = dict(
    n_estimators=200,
    max_depth=15,
    max_features=0.8,
    max_samples=0.2,
    ccp_alpha=0.0001
)

model = RandomForestClassifier(
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

model.fit(X_final, y)

# # save local copy
# folder = Path(f"/home/ross/coding/modular_trader/channel_run_test")
# model_file = f"{side}_{tf}_model_1a.sav"
# model_info = f"{side}_{tf}_info_1a.json"
# scaler_file = f"{side}_{tf}_scaler_1a.sav"
#
# info_dict = {'data_length': data_len, 'features': features, 'pair_selection': sel_method,
#              'pairs': pairs, 'created': int(datetime.now(timezone.utc).timestamp()), 'validity': len(X_final)}
#
# folder.mkdir(parents=True, exist_ok=True)
# joblib.dump(model, folder / model_file)
# info_path = folder / model_info
# info_path.touch(exist_ok=True)
# with open(info_path, 'w') as info:
#     json.dump(info_dict, info)
# scaler_path = folder / scaler_file
# scaler_path.touch(exist_ok=True)
# joblib.dump(scaler, scaler_path)

model.predict(X_final)


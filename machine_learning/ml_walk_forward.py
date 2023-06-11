import pandas as pd
import numpy as np
from pathlib import Path
import indicators as ind
import features
import binance_funcs as funcs
import entry_modelling as em
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint
import time
from itertools import product
from xgboost import XGBClassifier

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# TODO lots of variations to test for better performance:
#  smaller groups of pairs,
#  larger groups but more stringent min_conf,
#  longer train period,
#  cross_val_score instead of just fitting the model (might need more data in each train set for this),
#  XGBoost,
#  adjust fr based on previous test set's scores

all_start = time.perf_counter()


def fit_rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(class_weight='balanced',
                                   bootstrap=True,
                                   n_estimators=300,
                                   min_samples_split=3,
                                   min_samples_leaf=3,
                                   max_features=10,
                                   max_depth=20,
                                   n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


def fit_gbc(X_train, X_test, y_train):
    model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


fig = px.line()

tf_settings = {'1h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '4h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '12h': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},
             '1d': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},}
timeframe = '1h'
side = 'long'
data_len = 3000
num_pairs = 10
start_pair = 0
exit_method = tf_settings[timeframe]
pairs = em.rank_pairs()[start_pair:start_pair+num_pairs]

all_res = pd.DataFrame()

for pair in pairs:
    df = em.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
    df = em.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
    res_df = em.project_pnl(df, side, exit_method)
    all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)

all_res = all_res.sort_values('timestamp').reset_index(drop=True)

# Walk-forward Tests
test_results = dict(
    y_test=[],
    y_pred=[],
    y_prob=[],
    z_test=[]
)

train_days = 30

day_idxs = [(g[1].index[0], g[1].index[-1]) for g in all_res.groupby(all_res.timestamp.dt.dayofyear)]
for i, g in enumerate(day_idxs):
    if i == len(day_idxs)-train_days:
        break
    print(f"Train/test period {i}")
    print(f"Train from {day_idxs[i][0]} to {day_idxs[i+train_days-1][1]} (days {i} - {i+train_days-1})")
    print(f"Test from {day_idxs[i+train_days][0]} to {day_idxs[i+train_days][1]} (day {i+train_days})")
    train_idxs = [day_idxs[i][0], day_idxs[i+train_days-1][1]]
    test_idxs = [day_idxs[i+train_days][0], day_idxs[i+train_days][1]]

    X, y, z = em.features_labels_split(all_res)
    X_train, X_test, y_train, y_test, z_test = em.tt_split_idx(X, y, z, train_idxs, test_idxs)
    X_train, X_test = em.transform_columns(X_train, X_test)

    y_pred, y_prob = fit_gbc(X_train, X_test, y_train)

    test_results['y_test'].extend(list(y_test))
    test_results['y_pred'].extend(list(y_pred))
    test_results['y_prob'].extend(list(y_prob))
    test_results['z_test'].extend(list(z_test))

for thresh in np.linspace(0.5, 0.9, num=20):
    thresh = round(thresh, 2)
    bt_results = em.backtest(test_results['y_test'],
                             test_results['y_pred'],
                             test_results['y_prob'],
                             test_results['z_test'],
                             # track_perf=True,
                             min_conf=thresh,
                             fr=0.01)
    trades_taken = bt_results.in_trade.sum()
    winners = len(bt_results.loc[bt_results.trades > 0])
    win_rate = winners / trades_taken
    final_pnl = bt_results.pnl_curve.iloc[-1]

    precision = precision_score(test_results['y_test'], test_results['y_pred'], zero_division=0)
    f_beta = fbeta_score(test_results['y_test'], test_results['y_pred'], beta=0.5, zero_division=0)

    print(f"{timeframe} {side} min_conf: {thresh} Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, from "
          f"{trades_taken} trades, {len(bt_results)} signals, Precision score: {precision:.1%}, F Beta score: {f_beta:.1%}")

    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results.pnl_curve, name=f'min_conf: {thresh}', showlegend=True))

fig.update_layout(title=f"{timeframe}, {pairs}, train: {train_days}")
fig.show()

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

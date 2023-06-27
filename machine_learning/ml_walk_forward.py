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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.calibration import CalibratedClassifierCV

# TODO lots of variations to test for better performance:
#  sequential backwards feature selection
#  smaller groups of pairs,
#  larger groups but more stringent min_conf,
#  longer train period,
#  cross_val_score instead of just fitting the model (might need more data in each train set for this),
#  XGBoost,
#  adjust fr based on previous test set's scores

all_start = time.perf_counter()


def fit_rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(class_weight='balanced',
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


def gbc_selector(model, X, y, z, cols):
    """splits off a small portion of the start of the data set (pre), to avoid look-ahead bias, and fits the column
    selector on that, then transforms the rest of the data set (main) and passes it back to the script to use for all
    other training and testing"""

    scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
    selector = SFS(estimator=model, k_features=15, forward=True, floating=False, verbose=2, scoring=scorer, n_jobs=-1)
    X_pre, X_main, y_pre, y_main, z_main = em.tt_split_bifurcate(X, y, z, 0.05)
    X_pre, _, cols = em.transform_columns(X_pre, X_main) # i don't want to apply feature scaling to the main data set here
    selector = selector.fit(X_pre, y_pre)
    X_main = selector.transform(X_main)

    selected = [cols[x] for x in selector.k_feature_idx_]
    print('Selected features:')
    print(selected)

    return X_main, y_main, z_main, selector.k_feature_idx_



def fit_gbc(model, X_train, X_test, y_train):
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_prob


fig = px.line()

tf_settings = {'1h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
               '4h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
               '12h': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},
               '1d': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2}, }
timeframe = '1h'
side = 'short'
data_len = 3000
num_pairs = 10
start_pair = 0
exit_method = tf_settings[timeframe]
pairs = em.rank_pairs()[start_pair:start_pair + num_pairs]

all_res = pd.DataFrame()

for pair in pairs:
    df = em.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
    df = em.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
    res_df = em.project_pnl(df, side, exit_method)
    all_res = pd.concat([all_res, res_df], axis=0, ignore_index=True)

all_res = all_res.sort_values('timestamp').reset_index(drop=True)

model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5,
                                   subsample=0.5, min_samples_split=8, max_depth=12, learning_rate=0.1)

X_0, y_0, z_0 = em.features_labels_split(all_res)
cols = X_0.columns

# Walk-forward Tests
test_results = dict(
    y_test=[],
    y_pred=[],
    y_prob=[],
    y_pred_cal=[],
    y_prob_cal=[],
    z_test=[]
)

train_days = 30

day_idxs = [(g[1].index[0], g[1].index[-1]) for g in all_res.groupby(all_res.timestamp.dt.dayofyear)]
for i, g in enumerate(day_idxs):
    if i == len(day_idxs) - train_days:
        break

    if i % 7 == 0:
        print(f"Day {i}, running feature selection")
        X, y, z, selected = gbc_selector(model, X_0, y_0, z_0, cols)

    print(f"Train/test period {i}")
    print(f"Train from {day_idxs[i][0]} to {day_idxs[i + train_days - 1][1]} (days {i} - {i + train_days - 1})")
    print(f"Test from {day_idxs[i + train_days][0]} to {day_idxs[i + train_days][1]} (day {i + train_days})")
    train_idxs = [day_idxs[i][0], day_idxs[i + train_days - 1][1]]
    test_idxs = [day_idxs[i + train_days][0], day_idxs[i + train_days][1]]

    X_train, X_test, y_train, y_test, z_test = em.tt_split_idx(X, y, z, train_idxs, test_idxs)

    if X_test.shape[0] < 1:
        break

    # split data for fitting and calibration
    X_test, X_cal, y_test, y_cal = train_test_split(X_test, y_test, test_size=0.25, random_state=11)

    model, y_pred, y_prob = fit_gbc(model, X_train, X_test, y_train)

    interim_precision = precision_score(test_results['y_test'], test_results['y_pred'], zero_division=0)
    interim_f_beta = fbeta_score(test_results['y_test'], test_results['y_pred'], beta=0.333, zero_division=0)
    print(f"Scores for current walk-forward period: precision = {interim_precision:.1%}, f beta = {interim_f_beta:.1%}")

    test_results['y_test'].extend(list(y_test))
    test_results['y_pred'].extend(list(y_pred))
    test_results['y_prob'].extend(list(y_prob))
    test_results['z_test'].extend(list(z_test))

    # calibrate model
    cal = CalibratedClassifierCV(estimator=model, cv='prefit', n_jobs=-1)
    cal, y_pred_cal, y_prob_cal = fit_gbc(cal, X_cal, X_test, y_cal)

    test_results['y_pred_cal'].extend(list(y_pred_cal))
    test_results['y_prob_cal'].extend(list(y_prob_cal))

    interim_precision_cal = precision_score(test_results['y_test'], test_results['y_pred_cal'], zero_division=0)
    interim_f_beta_cal = fbeta_score(test_results['y_test'], test_results['y_pred_cal'], beta=0.333, zero_division=0)
    print(f"Calibrated scores for current walk-forward period: "
          f"precision = {interim_precision_cal:.1%}, f beta = {interim_f_beta_cal:.1%}")

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
    f_beta = fbeta_score(test_results['y_test'], test_results['y_pred'], beta=0.333, zero_division=0)

    print(f"{timeframe} {side} min_conf: {thresh} Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, from "
          f"{trades_taken} trades, {len(bt_results)} signals, Precision score: {precision:.1%}, F Beta score: {f_beta:.1%}")

    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results.pnl_curve, name=f'min_conf: {thresh}', showlegend=True))

fig.update_layout(title=f"{timeframe}, {pairs}, train: {train_days}")
fig.show()

all_end = time.perf_counter()
all_elapsed = all_end - all_start
print(f"\nTotal time taken: {int(all_elapsed // 60)}m {all_elapsed % 60:.1f}s")

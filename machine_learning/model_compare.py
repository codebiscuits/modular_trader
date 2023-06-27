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
from itertools import product

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler

"""the idea of this script is to see how feature importances and predictive power change when swapping models but 
keeping all data the same"""

fig = px.line()

tf_settings = {'1h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '4h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '12h': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},
             '1d': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},}
timeframe = '4h'
side = 'short'
data_len = 150
num_pairs = 100

exit_method = tf_settings[timeframe]

all_res = pd.DataFrame()

pairs = em.rank_pairs()[:num_pairs]
for pair in pairs:
    df = em.get_data(pair, timeframe).tail(data_len + 200).reset_index(drop=True)
    df = em.add_features(df, timeframe).tail(data_len).reset_index(drop=True)
    res_df = em.project_pnl(df, side, exit_method)
    all_res = pd.concat([all_res, res_df], axis=0)

X, y, z = em.features_labels_split(all_res)
X_train, X_test, y_train, y_test, z_test = em.tt_split_rand(X, y, z, 0.9)
X_train, X_test, cols = em.transform_columns(X_train, X_test)

imp_df = pd.DataFrame()

knn = KNeighborsClassifier(n_neighbors=9, metric='manhattan', weights='uniform', n_jobs=-1)
rfc = RandomForestClassifier(class_weight='balanced', n_estimators=300, min_samples_split=4, min_samples_leaf=4,
                               max_features=10, max_depth=20, n_jobs=-1)
gbc = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5,
                                   subsample=0.25, min_samples_split=2, max_depth=4, learning_rate=0.1)
for model, rus in product([knn, rfc, gbc], [0, 1]):

    name = 'knn' if model == knn else 'rfc' if model == rfc else 'gbc'
    if rus:
        name += '_rus'
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    bt_results = em.backtest(y_test, y_pred, y_prob, z_test)
    trades_taken = bt_results.in_trade.sum()
    winners = len(bt_results.loc[bt_results.trades > 0])
    if trades_taken:
        win_rate = winners / trades_taken
    else:
        win_rate = 0
    mean_pnl = bt_results.trades.loc[bt_results.open_trade.astype(bool)].mean()
    med_pnl = bt_results.trades.loc[bt_results.open_trade.astype(bool)].median()
    final_pnl = bt_results.pnl_curve.iloc[-1]

    # print(bt_results.tail())

    fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results.pnl_curve, name=f'{timeframe} {side}'))
    # print(results.head())

    precision = precision_score(y_test, y_pred, zero_division=0)
    f_beta = fbeta_score(y_test, y_pred, beta=0.333, zero_division=0)

    print(f"{name} {timeframe} {side} Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, mean pnl: {mean_pnl:.3f}, "
          f"median pnl: {med_pnl:.3f}, from {trades_taken} trades, {len(bt_results)} signals, "
          f"Precision score: {precision:.1%}, F Beta score: {f_beta:.1%}")

    importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df[name] = pd.Series(importances.importances_mean, index=cols)#.sort_values(ascending=False)

imp_df['max'] = imp_df.max(axis=1) # max <= 0 will help me see which features are not useful to any model
print(imp_df.sort_values('avg', ascending=False))

# fig.show()


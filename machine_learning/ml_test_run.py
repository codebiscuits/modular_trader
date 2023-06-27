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

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn
patch_sklearn()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler

# TODO i want to loop through all results files, pull the best result from each one, and automatically set frac_width,
#  atr_spacing, min_samples_split, max_features and max_depth, then run the backtest, and finally save the plot to file

fig = px.line()

tf_settings = {'1h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '4h': {'type': 'trail_fractal', 'width': 5, 'atr_spacing': 3},
             '12h': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},
             '1d': {'type': 'trail_fractal', 'width': 3, 'atr_spacing': 2},}
timeframe = '12h'
side = 'long'
data_len = 150
num_pairs = 66

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

# rus = RandomUnderSampler(random_state=0)
# X_train, y_train = rus.fit_resample(X_train, y_train)

# model = RandomForestClassifier(class_weight='balanced', n_estimators=300, min_samples_split=4, min_samples_leaf=4,
#                                max_features=10, max_depth=20, n_jobs=-1)
model = GradientBoostingClassifier(random_state=42, n_estimators=1000, validation_fraction=0.1, n_iter_no_change=5,
                                   subsample=0.25, min_samples_split=2, max_depth=4, learning_rate=0.1)

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

print(bt_results.tail())

print(f"{timeframe} {side} Final PnL: {final_pnl:.1%}, win rate: {win_rate:.1%}, mean pnl: {mean_pnl:.3f}, "
      f"median pnl: {med_pnl:.3f}, from {trades_taken} trades, {len(bt_results)} signals")

fig.add_trace(go.Scatter(x=bt_results.index, y=bt_results.pnl_curve, name=f'{timeframe} {side}'))
# print(results.head())

precision = precision_score(y_test, y_pred, zero_division=0)
f_beta = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)

importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
imp_df = pd.Series(importances.importances_mean, index=cols).sort_values(ascending=False)
print(imp_df)

# final_pnl = results['']

print(f"Precision score: {precision:.1%}, F Beta score: {f_beta:.1%}")

fig.show()


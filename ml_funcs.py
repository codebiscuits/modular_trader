import time
import pandas as pd
from pathlib import Path
from resources import indicators as ind, binance_funcs as funcs, features as features
import numpy as np
import json
from itertools import product
from collections import Counter
from datetime import datetime
from pyarrow import ArrowInvalid

if not Path('/pi_2.txt').exists():
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

def rank_pairs(selection):
    with open(f'recent_1d_{selection}.json', 'r') as file:
        vols = json.load(file)

    return sorted(vols, key=lambda x: vols[x], reverse=True)


def get_data(pair, timeframe, vwma_periods=24):
    """loads the ohlc data from file or downloads it from binance if necessary, then calculates vwma at the correct
    scale before resampling to the desired timeframe.
    vwma_lengths just accounts for timeframe resampling, vwma_periods is a multiplier on that"""
    start = time.perf_counter()

    ohlc_folder = Path('bin_ohlc_5m')
    ohlc_path = ohlc_folder / f"{pair}.parquet"

    if ohlc_path.exists():
        try:
            df = pd.read_parquet(ohlc_path)
            # print("Loaded OHLC from file")
        except (ArrowInvalid, OSError) as e:
            print('Error:\n', e)
            print(f"Problem reading {pair} parquet file, downloading from scratch.")
            ohlc_path.unlink()
            df = funcs.get_ohlc(pair, '5m', '2 years ago UTC')
            df.to_parquet(ohlc_path)
    else:
        df = funcs.get_ohlc(pair, '5m', '2 years ago UTC')
        ohlc_folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ohlc_path)
        # print("Downloaded OHLC from internet")

    vwma_lengths = {'1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
    vwma = ind.vwma(df, vwma_lengths[timeframe] * vwma_periods)
    vwma = vwma[int(vwma_lengths[timeframe] / 2)::vwma_lengths[timeframe]].reset_index(drop=True)

    df = funcs.resample_ohlc(timeframe, None, df).tail(len(vwma)).reset_index(drop=True)
    df['vwma'] = vwma

    if timeframe == '1h':
        df = df.tail(8760).reset_index(drop=True)

    elapsed = time.perf_counter() - start
    # print(f"get_data took {int(elapsed // 60)}m {elapsed % 60:.1f}s")

    return df


def trail_fractal(df_0: pd.DataFrame, width: int, spacing: int, side: str, trim_ohlc: int=1000, r_threshold: float=0.5):
    """r_threshold is how much pnl a trade must make for the model to consider it a profitable trade.
    higher values will train the model to target only the trades which produce higher profits, but will also limit
    the number of true positives to train the model on """
    df_0 = ind.williams_fractals(df_0, width, spacing)
    df_0 = df_0.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1).dropna(
        axis=0).reset_index(drop=True)

    trend_condition = (df_0.open > df_0.frac_low) if side == 'long' else (df_0.open < df_0.frac_high)
    rows = list(df_0.loc[trend_condition].index)
    df_0[f'fractal_trend_age_{side}'] = ind.consec_condition(trend_condition)
    results = []

    # loop through potential entries
    for row in rows:
        df = df_0[row:row + trim_ohlc].copy().reset_index(drop=True)
        entry_price = df.open.iloc[0]

        if side == 'long':
            df['inval'] = df.frac_low.cummax()
            r_pct = abs(entry_price - df.frac_low.iloc[0]) / entry_price
            exit_row = df.loc[df.low < df.inval].index.min()
        else:
            df['inval'] = df.frac_high.cummin()
            r_pct = abs(entry_price - df.frac_high.iloc[0]) / entry_price
            exit_row = df.loc[df.high > df.inval].index.min()

        if not isinstance(exit_row, np.int64):
            continue

        lifespan = exit_row
        exit_price = df.inval.iloc[exit_row]
        trade_diff = exit_price / entry_price

        pnl_pct = (trade_diff - 1.003) if side == 'long' else (0.997 - trade_diff)  # accounting for 15bps fees and 15bps slippage
        pnl_r = pnl_pct / r_pct

        pnl_cat = 0 if (pnl_r <= r_threshold) else 1

        row_data = df_0.iloc[row-1].to_dict()

        row_res = dict(
            # idx=row,
            r_pct=r_pct,
            lifespan=lifespan,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r,
            pnl_cat=pnl_cat
        )

        results.append(row_data | row_res)

        msg = f"trade lifespans getting close to trimmed ohlc length ({lifespan / trim_ohlc:.1%}), increase trim ohlc"
        if lifespan / trim_ohlc > 0.5:
            print(msg)

    return results


def trail_atr(df, atr_len, atr_mult):
    pass


def oco(df, r_mult, inval_lb, side):
    """i want this function to either set the invalidation by looking at recent lows/highs or by using recent volatility
    as a multiplier"""

    # method 1
    # I want to ask how many rows from current_row till row.high / row.low exceeds current_row.stop / current_row.profit
    # then I can compare those umber to find which will be hit first

    # method 2
    # I want to find the index of the first high / low to exceed the profit value and the index of the first low / high
    # to exceed the stop value, then I can see which is first. I can use idxmax and idxmin for this, but I first need to
    # use clip to make sure that the first values to exceed my limits will be considered the first min/max value

    # calculate r by setting init stop based on last 2 bars ll/hh
    if side == 'long':
        df['r'] = abs((df.close - df.low.rolling(inval_lb).min()) / df.close).shift(1)
    else:
        df['r'] = abs((df.close - df.high.rolling(inval_lb).max()) / df.close).shift(1)
    pass


def add_features(df, tf):
    df = features.atr_zscore(df, 25)
    df = features.atr_zscore(df, 50)
    df = features.atr_zscore(df, 100)
    df = features.atr_zscore(df, 200)
    df = features.atr_pct(df, 5)
    df = features.atr_pct(df, 10)
    df = features.atr_pct(df, 25)
    df = features.atr_pct(df, 50)
    df = features.ats_z(df, 25)
    df = features.ats_z(df, 50)
    df = features.ats_z(df, 100)
    df = features.ats_z(df, 200)
    df = features.bull_bear_bar(df)
    df = features.channel_mid_ratio(df, 25)
    df = features.channel_mid_ratio(df, 50)
    df = features.channel_mid_ratio(df, 100)
    df = features.channel_mid_ratio(df, 200)
    df = features.channel_mid_width(df, 25)
    df = features.channel_mid_width(df, 50)
    df = features.channel_mid_width(df, 100)
    df = features.channel_mid_width(df, 200)
    df = features.daily_open_ratio(df)
    df = features.daily_roc(df, tf)
    df = features.day_of_week(df)
    df = features.day_of_week_180(df)
    df = features.dd_zscore(df, 12)
    df = features.dd_zscore(df, 25)
    df = features.dd_zscore(df, 50)
    df = features.dd_zscore(df, 100)
    df = features.dd_zscore(df, 200)
    df = features.doji(df, 0.5, 2, weighted=True)
    df = features.doji(df, 1, 2, weighted=True)
    df = features.doji(df, 1.5, 2, weighted=True)
    df = features.doji(df, 0.5, 2, weighted=False)
    df = features.engulfing(df, 1)
    df = features.engulfing(df, 2)
    df = features.engulfing(df, 3)
    df = features.ema_breakout(df, 12, 25)
    df = features.ema_breakout(df, 25, 50)
    df = features.ema_breakout(df, 50, 100)
    df = features.ema_breakout(df, 100, 200)
    df = features.ema_roc(df, 25)
    df = features.ema_roc(df, 50)
    df = features.ema_roc(df, 100)
    df = features.ema_roc(df, 200)
    df = features.ema_ratio(df, 25)
    df = features.ema_ratio(df, 50)
    df = features.ema_ratio(df, 100)
    df = features.ema_ratio(df, 200)
    df = features.fractal_trend_age(df)
    df = features.hma_roc(df, 25)
    df = features.hma_roc(df, 50)
    df = features.hma_roc(df, 100)
    df = features.hma_roc(df, 200)
    df = features.hma_ratio(df, 25)
    df = features.hma_ratio(df, 50)
    df = features.hma_ratio(df, 100)
    df = features.hma_ratio(df, 200)
    df = features.hour(df)
    df = features.hour_180(df)
    df = features.inside_bar(df)
    df = features.kurtosis(df, 6)
    df = features.kurtosis(df, 12)
    df = features.kurtosis(df, 25)
    df = features.kurtosis(df, 50)
    df = features.kurtosis(df, 100)
    df = features.kurtosis(df, 200)
    df = features.prev_daily_open_ratio(df)
    df = features.prev_daily_high_ratio(df)
    df = features.prev_daily_low_ratio(df)
    df = features.prev_weekly_open_ratio(df)
    df = features.prev_weekly_high_ratio(df)
    df = features.prev_weekly_low_ratio(df)
    df = features.round_numbers_proximity(df)
    df = features.big_round_nums_proximity(df)
    df = features.spooky_nums_proximity(df)
    df = features.round_numbers_close(df, 1)
    df = features.round_numbers_close(df, 2)
    df = features.round_numbers_close(df, 3)
    df = features.round_numbers_close(df, 4)
    df = features.round_numbers_close(df, 5)
    df = features.big_round_nums_close(df, 1)
    df = features.big_round_nums_close(df, 5)
    df = features.big_round_nums_close(df, 10)
    df = features.big_round_nums_close(df, 15)
    df = features.big_round_nums_close(df, 20)
    df = features.rsi(df, 14)
    df = features.rsi(df, 25)
    df = features.rsi(df, 50)
    df = features.rsi(df, 100)
    df = features.rsi(df, 200)
    df = features.rsi_above(df, 14, 30)
    df = features.rsi_above(df, 25, 30)
    df = features.rsi_above(df, 50, 30)
    df = features.rsi_above(df, 100, 30)
    df = features.rsi_above(df, 200, 30)
    df = features.rsi_above(df, 14, 50)
    df = features.rsi_above(df, 25, 50)
    df = features.rsi_above(df, 50, 50)
    df = features.rsi_above(df, 100, 50)
    df = features.rsi_above(df, 200, 50)
    df = features.rsi_above(df, 14, 70)
    df = features.rsi_above(df, 25, 70)
    df = features.rsi_above(df, 50, 70)
    df = features.rsi_above(df, 100, 70)
    df = features.rsi_above(df, 200, 70)
    df = features.rsi_timing_long(df, 3)
    df = features.rsi_timing_long(df, 5)
    df = features.rsi_timing_long(df, 7)
    df = features.rsi_timing_long(df, 9)
    df = features.rsi_timing_short(df, 3)
    df = features.rsi_timing_short(df, 5)
    df = features.rsi_timing_short(df, 7)
    df = features.rsi_timing_short(df, 9)
    df = features.skew(df, 6)
    df = features.skew(df, 12)
    df = features.skew(df, 25)
    df = features.skew(df, 50)
    df = features.skew(df, 100)
    df = features.skew(df, 200)
    df = features.stoch_base_vol(df, 25)
    df = features.stoch_base_vol(df, 50)
    df = features.stoch_base_vol(df, 100)
    df = features.stoch_base_vol(df, 200)
    df = features.stoch_num_trades(df, 25)
    df = features.stoch_num_trades(df, 50)
    df = features.stoch_num_trades(df, 100)
    df = features.stoch_num_trades(df, 200)
    df = features.stoch_vwma_ratio(df, 25)
    df = features.stoch_vwma_ratio(df, 50)
    df = features.stoch_vwma_ratio(df, 100)
    df = features.volume_climax_up(df, 12)
    df = features.volume_climax_up(df, 25)
    df = features.volume_climax_up(df, 50)
    df = features.volume_climax_down(df, 12)
    df = features.volume_climax_down(df, 25)
    df = features.volume_climax_down(df, 50)
    df = features.high_volume_churn(df, 12)
    df = features.high_volume_churn(df, 25)
    df = features.high_volume_churn(df, 50)
    df = features.low_volume_bar(df, 12)
    df = features.low_volume_bar(df, 25)
    df = features.low_volume_bar(df, 50)
    df = features.vol_delta(df)
    df = features.vol_delta_div(df, 1)
    df = features.vol_delta_div(df, 2)
    df = features.vol_delta_div(df, 3)
    df = features.vol_delta_div(df, 4)
    df = features.vol_delta_pct(df)
    df = features.vol_denom_roc(df, 2, 25)
    df = features.vol_denom_roc(df, 5, 50)
    # df = features.week_of_year(df)
    # df = features.week_of_year_180(df)
    df = features.weekly_roc(df, tf)
    df = features.weekly_open_ratio(df)

    return df


def features_labels_split(df):
    X = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol', 'num_trades',
                 'taker_buy_base_vol', 'taker_buy_quote_vol', 'vwma', 'r_pct', 'pnl_pct', 'pnl_r', 'pnl_cat',
                 'atr-25', 'atr-50', 'atr-100', 'atr-200', 'ema_12', 'ema_25', 'ema_50', 'ema_100', 'ema_200',
                 'hma_25', 'hma_50', 'hma_100', 'hma_200', 'lifespan', 'frac_high', 'frac_low', 'inval', 'daily_open',
                 'prev_daily_open', 'prev_daily_high', 'prev_daily_low', 'bullish_doji', 'bearish_doji'],
                axis=1, errors='ignore')
    y = df.pnl_cat.shift(-1)
    z = df.pnl_r.shift(-1)

    return X, y, z


def tt_split_rand(X: pd.DataFrame, y: pd.Series, z: pd.Series, split_pct: float) -> tuple:
    """split into train and test sets for hold-out validation"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_pct, random_state=11)
    _, _, _, z_test = train_test_split(X, z, train_size=split_pct, random_state=11)

    return X_train, X_test, y_train, y_test, z_test


def tt_split_bifurcate(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, z: pd.Series | np.ndarray, split_pct: float) -> tuple:
    """split into train and test sets for hold-out validation"""
    train_size = int(split_pct*len(X))
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
    else:
        X_train = X[:train_size, :]
        X_test = X[train_size:, :]
    if isinstance(y, pd.Series):
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]
    else:
        y_train = y[:train_size, :]
        y_test = y[train_size:, :]
    if isinstance(z, pd.Series):
        z_test = z.iloc[train_size:]
    else:
        z_test = z[train_size:, :]

    return X_train, X_test, y_train, y_test, z_test


def tt_split_idx(X, y, z, train_idxs, test_idxs):
    """split into train and test sets for hold-out validation"""

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idxs[0]:train_idxs[1]+1]
        X_test = X.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        X_train = X[train_idxs[0]:train_idxs[1] + 1, :]
        X_test = X[test_idxs[0]:test_idxs[1] + 1, :]
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idxs[0]:train_idxs[1]+1]
        y_test = y.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        y_train = y[train_idxs[0]:train_idxs[1] + 1]
        y_test = y[test_idxs[0]:test_idxs[1] + 1]
    if isinstance(z, pd.Series):
        z_test = z.iloc[test_idxs[0]:test_idxs[1]+1]
    else:
        z_test = z[test_idxs[0]:test_idxs[1] + 1]

    return X_train, X_test, y_train, y_test, z_test


def transform_columns(X_train, X_test):
    # column transformation
    min_max_cols = ['vol_delta_pct', 'ema_25_roc', 'ema_50_roc', 'ema_100_roc', 'ema_200_roc', 'hma_25_roc', 'hma_50_roc',
          'hma_100_roc', 'hma_200_roc', 'hour', 'hour_180', 'day_of_week', 'day_of_week_180', 'chan_mid_ratio_25',
          'chan_mid_ratio_50', 'chan_mid_ratio_100', 'chan_mid_ratio_200', 'chan_mid_width_25', 'chan_mid_width_50',
          'chan_mid_width_100', 'chan_mid_width_200']
    min_max_cols = [mmc for mmc in min_max_cols if mmc in X_train.columns]

    quant_cols = ['ema_25_ratio', 'ema_50_ratio', 'ema_100_ratio', 'ema_200_ratio', 'hma_25_ratio', 'hma_50_ratio',
          'hma_100_ratio', 'hma_200_ratio', 'atr_5_pct', 'atr_10_pct', 'atr_25_pct', 'atr_50_pct']
    quant_cols = [qc for qc in quant_cols if qc in X_train.columns]

    transformers = [('minmax', MinMaxScaler(), min_max_cols), ('quantile', QuantileTransformer(n_quantiles=200), quant_cols)]
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    feature_cols = [f.split('__')[1] for f in ct.get_feature_names_out()]

    return X_train, X_test, feature_cols


def backtest(y_test, y_pred, y_prob, z_test, track_perf = False, min_conf=0.75, fr=0.01):

    results = pd.DataFrame(
        {'labels': y_test,
         'predictions': y_pred,
         'confidence': y_prob,
         'pnl_r': z_test}
    ).reset_index(drop=True)

    start_cash = 1

    results['pnl_r'] = results.pnl_r.clip(lower=-1)
    results['confidence'] = results.confidence * results.predictions
    results['confidence'] = results.confidence.where(results.confidence >= min_conf, 0)
    results['in_trade'] = results.confidence > 0
    results['open_trade'] = results.in_trade & results.in_trade.diff()
    results['trades'] = results.confidence * results.pnl_r * results.open_trade
    if track_perf:
        results['perf_score'] = (results.pnl_r > 0).astype(int).rolling(100).mean().shift()
        results['trade_pnl_mult'] = ((results.trades * fr * results.perf_score) + 1).fillna(1)
    else:
        results['trade_pnl_mult'] = ((results.trades * fr) + 1).fillna(1)
    results['pnl_curve'] = (results.trade_pnl_mult.cumprod() * start_cash) - 1

    return results



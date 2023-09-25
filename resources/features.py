from resources import indicators as ind
import pandas as pd
import numpy as np

"""
Process for adding new features:
1 - Define the feature as a function which takes the dataframe as the first argument and returns the dataframe with any 
    new columns included.
2 - insert all variations of the feature into the add_feature function below, with the following format: 
    'column name created by the feature': {'call': name of feature function, 'params': (df, param_1, param_2, etc)},
    if there are no parameters other than the dataframe, just put (df, ) as the params value. This is how the modular 
    trading system accesses each individual feature it needs.
3 - add the same variations of the call to the 'add_features' function in ml_funcs so that all the training and analysis
    scripts have access to the feature when developing and training models
"""


def add_feature(df, name, timeframe):
    feature_lookup = {
        'atr_z_25': {'call': atr_zscore, 'params': (df, 25)},
        'atr_z_50': {'call': atr_zscore, 'params': (df, 50)},
        'atr_z_100': {'call': atr_zscore, 'params': (df, 100)},
        'atr_z_200': {'call': atr_zscore, 'params': (df, 200)},
        'atr_5_pct': {'call': atr_pct, 'params': (df, 5)},
        'atr_10_pct': {'call': atr_pct, 'params': (df, 10)},
        'atr_25_pct': {'call': atr_pct, 'params': (df, 25)},
        'atr_50_pct': {'call': atr_pct, 'params': (df, 50)},
        'ats_z_25': {'call': ats_z, 'params': (df, 25)},
        'ats_z_50': {'call': ats_z, 'params': (df, 50)},
        'ats_z_100': {'call': ats_z, 'params': (df, 100)},
        'ats_z_200': {'call': ats_z, 'params': (df, 200)},
        'bullish_bar': {'call': bull_bear_bar, 'params': (df,)},
        'bearish_bar': {'call': bull_bear_bar, 'params': (df,)},
        'weighted_50_bull_doji': {'call': doji, 'params': (df, 0.5, 2, True)},
        'weighted_50_bear_doji': {'call': doji, 'params': (df, 0.5, 2, True)},
        'weighted_100_bull_doji': {'call': doji, 'params': (df, 1, 2, True)},
        'weighted_100_bear_doji': {'call': doji, 'params': (df, 1, 2, True)},
        'weighted_150_bull_doji': {'call': doji, 'params': (df, 1.5, 2, True)},
        'weighted_150_bear_doji': {'call': doji, 'params': (df, 1.5, 2, True)},
        'unweighted_bull_doji': {'call': doji, 'params': (df, 0.5, 2, False)},
        'unweighted_bear_doji': {'call': doji, 'params': (df, 0.5, 2, False)},
        'bearish_engulf_1': {'call': engulfing, 'params': (df, 1)},
        'bearish_engulf_2': {'call': engulfing, 'params': (df, 2)},
        'bearish_engulf_3': {'call': engulfing, 'params': (df, 3)},
        'bullish_engulf_1': {'call': engulfing, 'params': (df, 1)},
        'bullish_engulf_2': {'call': engulfing, 'params': (df, 2)},
        'bullish_engulf_3': {'call': engulfing, 'params': (df, 3)},
        'chan_mid_ratio_25': {'call': channel_mid_ratio, 'params': (df, 25)},
        'chan_mid_ratio_50': {'call': channel_mid_ratio, 'params': (df, 50)},
        'chan_mid_ratio_100': {'call': channel_mid_ratio, 'params': (df, 100)},
        'chan_mid_ratio_200': {'call': channel_mid_ratio, 'params': (df, 200)},
        'chan_mid_width_25': {'call': channel_mid_width, 'params': (df, 25)},
        'chan_mid_width_50': {'call': channel_mid_width, 'params': (df, 50)},
        'chan_mid_width_100': {'call': channel_mid_width, 'params': (df, 100)},
        'chan_mid_width_200': {'call': channel_mid_width, 'params': (df, 200)},
        'daily_open_ratio': {'call': daily_open_ratio, 'params': (df,)},
        'roc_1d': {'call': daily_roc, 'params': (df, timeframe)},
        'day_of_week': {'call': day_of_week, 'params': (df,)},
        'day_of_week_180': {'call': day_of_week_180, 'params': (df,)},
        'dd_z_12': {'call': dd_zscore, 'params': (df, 12)},
        'dd_z_25': {'call': dd_zscore, 'params': (df, 25)},
        'dd_z_50': {'call': dd_zscore, 'params': (df, 50)},
        'dd_z_100': {'call': dd_zscore, 'params': (df, 100)},
        'dd_z_200': {'call': dd_zscore, 'params': (df, 200)},
        'ema_12_above_24': {'call': two_emas, 'params': (df, 12, 24)},
        'ema_12_above_48': {'call': two_emas, 'params': (df, 12, 48)},
        'ema_24_above_96': {'call': two_emas, 'params': (df, 24, 96)},
        'ema_48_above_192': {'call': two_emas, 'params': (df, 48, 192)},
        'ema_cross_up_12_24': {'call': two_emas, 'params': (df, 12, 24)},
        'ema_cross_up_12_48': {'call': two_emas, 'params': (df, 12, 48)},
        'ema_cross_up_24_96': {'call': two_emas, 'params': (df, 24, 96)},
        'ema_cross_up_48_192': {'call': two_emas, 'params': (df, 48, 192)},
        'ema_cross_down_12_24': {'call': two_emas, 'params': (df, 12, 24)},
        'ema_cross_down_12_48': {'call': two_emas, 'params': (df, 12, 48)},
        'ema_cross_down_24_96': {'call': two_emas, 'params': (df, 24, 96)},
        'ema_cross_down_48_192': {'call': two_emas, 'params': (df, 48, 192)},
        'ema_12_break_up': {'call': ema_breakout, 'params': (df, 12, 25)},
        'ema_12_break_down': {'call': ema_breakout, 'params': (df, 12, 25)},
        'ema_25_break_up': {'call': ema_breakout, 'params': (df, 25, 50)},
        'ema_25_break_down': {'call': ema_breakout, 'params': (df, 25, 50)},
        'ema_50_break_up': {'call': ema_breakout, 'params': (df, 50, 100)},
        'ema_50_break_down': {'call': ema_breakout, 'params': (df, 50, 100)},
        'ema_100_break_up': {'call': ema_breakout, 'params': (df, 100, 200)},
        'ema_100_break_down': {'call': ema_breakout, 'params': (df, 100, 200)},
        'ema_25_ratio': {'call': ema_ratio, 'params': (df, 25)},
        'ema_50_ratio': {'call': ema_ratio, 'params': (df, 50)},
        'ema_100_ratio': {'call': ema_ratio, 'params': (df, 100)},
        'ema_200_ratio': {'call': ema_ratio, 'params': (df, 200)},
        'ema_25_roc': {'call': ema_roc, 'params': (df, 25)},
        'ema_50_roc': {'call': ema_roc, 'params': (df, 50)},
        'ema_100_roc': {'call': ema_roc, 'params': (df, 100)},
        'ema_200_roc': {'call': ema_roc, 'params': (df, 200)},
        'hma_25_ratio': {'call': hma_ratio, 'params': (df, 25)},
        'hma_50_ratio': {'call': hma_ratio, 'params': (df, 50)},
        'hma_100_ratio': {'call': hma_ratio, 'params': (df, 100)},
        'hma_200_ratio': {'call': hma_ratio, 'params': (df, 200)},
        'hma_25_roc': {'call': hma_roc, 'params': (df, 25)},
        'hma_50_roc': {'call': hma_roc, 'params': (df, 50)},
        'hma_100_roc': {'call': hma_roc, 'params': (df, 100)},
        'hma_200_roc': {'call': hma_roc, 'params': (df, 200)},
        'hour': {'call': hour, 'params': (df,)},
        'hour_180': {'call': hour_180, 'params': (df,)},
        'inside_bar': {'call': inside_bar, 'params': (df,)},
        'kurtosis_6': {'call': kurtosis, 'params': (df, 6)},
        'kurtosis_12': {'call': kurtosis, 'params': (df, 12)},
        'kurtosis_25': {'call': kurtosis, 'params': (df, 25)},
        'kurtosis_50': {'call': kurtosis, 'params': (df, 50)},
        'kurtosis_100': {'call': kurtosis, 'params': (df, 100)},
        'kurtosis_200': {'call': kurtosis, 'params': (df, 200)},
        'prev_daily_open_ratio': {'call': prev_daily_open_ratio, 'params': (df,)},
        'prev_daily_high_ratio': {'call': prev_daily_high_ratio, 'params': (df,)},
        'prev_daily_low_ratio': {'call': prev_daily_low_ratio, 'params': (df,)},
        'prev_weekly_open_ratio': {'call': prev_weekly_open_ratio, 'params': (df,)},
        'prev_weekly_high_ratio': {'call': prev_weekly_high_ratio, 'params': (df,)},
        'prev_weekly_low_ratio': {'call': prev_weekly_low_ratio, 'params': (df,)},
        'recent_vd_div_1': {'call': vol_delta_div, 'params': (df, 1)},
        'recent_vd_div_2': {'call': vol_delta_div, 'params': (df, 2)},
        'recent_vd_div_3': {'call': vol_delta_div, 'params': (df, 3)},
        'recent_vd_div_4': {'call': vol_delta_div, 'params': (df, 4)},
        'round_num_prox': {'call': round_numbers_proximity, 'params': (df,)},
        'big_round_num_prox': {'call': big_round_nums_proximity, 'params': (df,)},
        'spooky_num_prox': {'call': spooky_nums_proximity, 'params': (df,)},
        'round_nums_close_1': {'call': round_numbers_close, 'params': (df, 1)},
        'round_nums_close_2': {'call': round_numbers_close, 'params': (df, 2)},
        'round_nums_close_3': {'call': round_numbers_close, 'params': (df, 3)},
        'round_nums_close_4': {'call': round_numbers_close, 'params': (df, 4)},
        'round_nums_close_5': {'call': round_numbers_close, 'params': (df, 5)},
        'big_round_nums_close_1': {'call': big_round_nums_close, 'params': (df, 1)},
        'big_round_nums_close_5': {'call': big_round_nums_close, 'params': (df, 5)},
        'big_round_nums_close_10': {'call': big_round_nums_close, 'params': (df, 10)},
        'big_round_nums_close_15': {'call': big_round_nums_close, 'params': (df, 15)},
        'big_round_nums_close_20': {'call': big_round_nums_close, 'params': (df, 20)},
        'rsi_14': {'call': rsi, 'params': (df, 14)},
        'rsi_25': {'call': rsi, 'params': (df, 25)},
        'rsi_50': {'call': rsi, 'params': (df, 50)},
        'rsi_100': {'call': rsi, 'params': (df, 100)},
        'rsi_200': {'call': rsi, 'params': (df, 200)},
        'rsi_14_above_30': {'call': rsi_above, 'params': (df, 14, 30)},
        'rsi_25_above_30': {'call': rsi_above, 'params': (df, 25, 30)},
        'rsi_50_above_30': {'call': rsi_above, 'params': (df, 50, 30)},
        'rsi_100_above_30': {'call': rsi_above, 'params': (df, 100, 30)},
        'rsi_200_above_30': {'call': rsi_above, 'params': (df, 200, 30)},
        'rsi_14_above_50': {'call': rsi_above, 'params': (df, 14, 50)},
        'rsi_25_above_50': {'call': rsi_above, 'params': (df, 25, 50)},
        'rsi_50_above_50': {'call': rsi_above, 'params': (df, 50, 50)},
        'rsi_100_above_50': {'call': rsi_above, 'params': (df, 100, 50)},
        'rsi_200_above_50': {'call': rsi_above, 'params': (df, 200, 50)},
        'rsi_14_above_70': {'call': rsi_above, 'params': (df, 14, 70)},
        'rsi_25_above_70': {'call': rsi_above, 'params': (df, 25, 70)},
        'rsi_50_above_70': {'call': rsi_above, 'params': (df, 50, 70)},
        'rsi_100_above_70': {'call': rsi_above, 'params': (df, 100, 70)},
        'rsi_200_above_70': {'call': rsi_above, 'params': (df, 200, 70)},
        'rsi_timing_l_3_14': {'call': rsi_timing_long, 'params': (df, 3)},
        'rsi_timing_l_5_14': {'call': rsi_timing_long, 'params': (df, 5)},
        'rsi_timing_l_7_14': {'call': rsi_timing_long, 'params': (df, 7)},
        'rsi_timing_l_9_14': {'call': rsi_timing_long, 'params': (df, 9)},
        'rsi_timing_s_3_14': {'call': rsi_timing_short, 'params': (df, 3)},
        'rsi_timing_s_5_14': {'call': rsi_timing_short, 'params': (df, 5)},
        'rsi_timing_s_7_14': {'call': rsi_timing_short, 'params': (df, 7)},
        'rsi_timing_s_9_14': {'call': rsi_timing_short, 'params': (df, 9)},
        'skew_6': {'call': skew, 'params': (df, 6)},
        'skew_12': {'call': skew, 'params': (df, 12)},
        'skew_25': {'call': skew, 'params': (df, 25)},
        'skew_50': {'call': skew, 'params': (df, 50)},
        'skew_100': {'call': skew, 'params': (df, 100)},
        'skew_200': {'call': skew, 'params': (df, 200)},
        'stoch_base_vol_25': {'call': stoch_base_vol, 'params': (df, 25)},
        'stoch_base_vol_50': {'call': stoch_base_vol, 'params': (df, 50)},
        'stoch_base_vol_100': {'call': stoch_base_vol, 'params': (df, 100)},
        'stoch_base_vol_200': {'call': stoch_base_vol, 'params': (df, 200)},
        'stoch_m_12': {'call': stoch_m, 'params': (df, 12)},
        'stoch_m_25': {'call': stoch_m, 'params': (df, 25)},
        'stoch_m_50': {'call': stoch_m, 'params': (df, 50)},
        'stoch_m_100': {'call': stoch_m, 'params': (df, 100)},
        'stoch_w_12': {'call': stoch_w, 'params': (df, 12)},
        'stoch_w_25': {'call': stoch_w, 'params': (df, 25)},
        'stoch_w_50': {'call': stoch_w, 'params': (df, 50)},
        'stoch_w_100': {'call': stoch_w, 'params': (df, 100)},
        'stoch_num_trades_25': {'call': stoch_num_trades, 'params': (df, 25)},
        'stoch_num_trades_50': {'call': stoch_num_trades, 'params': (df, 50)},
        'stoch_num_trades_100': {'call': stoch_num_trades, 'params': (df, 100)},
        'stoch_num_trades_200': {'call': stoch_num_trades, 'params': (df, 200)},
        'stoch_vwma_ratio_25': {'call': stoch_vwma_ratio, 'params': (df, 25)},
        'stoch_vwma_ratio_50': {'call': stoch_vwma_ratio, 'params': (df, 50)},
        'stoch_vwma_ratio_100': {'call': stoch_vwma_ratio, 'params': (df, 100)},
        'fractal_trend_age_long': {'call': fractal_trend_age, 'params': (df,)},
        'fractal_trend_age_short': {'call': fractal_trend_age, 'params': (df,)},
        'volume_climax_up_12': {'call': volume_climax_up, 'params': (df, 12)},
        'volume_climax_up_25': {'call': volume_climax_up, 'params': (df, 25)},
        'volume_climax_up_50': {'call': volume_climax_up, 'params': (df, 50)},
        'volume_climax_dn_12': {'call': volume_climax_down, 'params': (df, 12)},
        'volume_climax_dn_25': {'call': volume_climax_down, 'params': (df, 25)},
        'volume_climax_dn_50': {'call': volume_climax_down, 'params': (df, 50)},
        'high_volume_churn_12': {'call': high_volume_churn, 'params': (df, 12)},
        'high_volume_churn_25': {'call': high_volume_churn, 'params': (df, 25)},
        'high_volume_churn_50': {'call': high_volume_churn, 'params': (df, 50)},
        'low_volume_12': {'call': low_volume_bar, 'params': (df, 12)},
        'low_volume_25': {'call': low_volume_bar, 'params': (df, 25)},
        'low_volume_50': {'call': low_volume_bar, 'params': (df, 50)},
        'vol_delta': {'call': vol_delta, 'params': (df,)},
        'vol_delta_pct': {'call': vol_delta_pct, 'params': (df,)},
        'vol_denom_roc_2': {'call': vol_denom_roc, 'params': (df, 2, 25)},
        'vol_denom_roc_5': {'call': vol_denom_roc, 'params': (df, 5, 50)},
        'week_of_year': {'call': week_of_year, 'params': (df,)},
        'week_of_year_180': {'call': week_of_year_180, 'params': (df,)},
        'weekly_open_ratio': {'call': weekly_open_ratio, 'params': (df,)},
        'roc_1w': {'call': weekly_roc, 'params': (df, timeframe)}
    }
    feature = feature_lookup[name]
    df = feature['call'](*feature['params'])

    return df


def vol_doji(df: pd.DataFrame, thresh: float, lookback: int, weighted: bool) -> pd.DataFrame:
    """same as doji, but the wicks are measured in terms of recent atr volatility"""
    pass


def htf_fractals_proximity(df: pd.DataFrame, orig_tf: str, htf: str = 'W', frac_width: int = 3) -> pd.DataFrame:
    """resamples to weekly or monthly timeframe, calculates williams fractals on that, works out which is closest to the
    current price and returns the pct difference"""
    # TODO this isn't finished yet

    # only need to resample close column
    closes = df.close.resample(htf).shift(
        1)  # make sure the close price for each week is recorded on the following week
    # is this definitely giving the last close of each period?

    fractal_high = np.where(closes == closes.rolling(frac_width, center=True).max(), closes, np.nan)
    fractal_low = np.where(closes == closes.rolling(frac_width, center=True).min(), closes, np.nan)

    # make a list of all the levels with their timestamps (they are only valid levels after they have been established)

    # for each period in the original df, record which is the closest VALID level, then create a column that represents
    # how far those prices are from the open price or vwma

    return df


def daily_sfp(df: pd.DataFrame):
    # group by day (not day of week or day of year but every unique day) and calculate for each period whether the price
    # has started on one side of the open and then crossed to the other side.
    # maybe include a volume filter to ignore when a tiny percentage of normal volume was one one side to avoid falsly
    # classifying a retest as a cross
    pass


def two_emas(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    fast = df.close.ewm(a).mean()
    slow = df.close.ewm(b).mean()

    df[f"ema_cross_up_{a}_{b}"] = (fast > slow) & (fast.shift() < slow.shift())
    df[f"ema_cross_down_{a}_{b}"] = (fast < slow) & (fast.shift() > slow.shift())
    df[f"ema_{a}_above_{b}"] = fast > slow

    return df


def rsi_timing_long(df: pd.DataFrame, lookback: int, rsi_length: int = 14) -> pd.DataFrame:
    """returns True if all rsi values in the lookback window are less than the value 3 periods earlier, and the
    current value is less than 30"""
    if f"rsi_{rsi_length}" not in df.columns:
        df[f"rsi_{rsi_length}"] = ind.rsi(df.close, rsi_length)

    df[f"rsi_timing_l_{lookback}_{rsi_length}"] = (((df[f"rsi_{rsi_length}"].pct_change(3) < 0)
                                                    .rolling(lookback).sum() + (df[f"rsi_{rsi_length}"] < 30))
                                                   == lookback + 1)

    return df


def rsi_timing_short(df: pd.DataFrame, lookback: int, rsi_length: int = 14) -> pd.DataFrame:
    """returns True if all rsi values in the lookback window are greater than the value 3 periods earlier, and the
    current value is greater than 70"""
    if f"rsi_{rsi_length}" not in df.columns:
        df[f"rsi_{rsi_length}"] = ind.rsi(df.close, rsi_length)

    df[f"rsi_timing_s_{lookback}_{rsi_length}"] = (((df[f"rsi_{rsi_length}"].pct_change(3) > 0)
                                                    .rolling(lookback).sum() + (df[f"rsi_{rsi_length}"] > 70))
                                                   == lookback + 1)

    return df


def atr_zscore(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    atr = ind.atr(df, lookback)[f'atr_{lookback}_pct']
    atr_mean = atr.rolling(lookback).mean()
    atr_std = atr.rolling(lookback).std()
    df[f'atr_z_{lookback}'] = (atr - atr_mean) / atr_std

    return df


def dd_zscore(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    highest_close = df.close.rolling(lookback).max()
    pct_dd = (highest_close - df.close) / highest_close
    dd_mean = pct_dd.rolling(lookback).mean()
    dd_std = pct_dd.rolling(lookback).std()
    df[f'dd_z_{lookback}'] = (pct_dd - dd_mean) / dd_std

    return df


def volume_climax_up(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    bar_range = (df.close - df.open) / ((df.close + df.open) / 2)
    volume_up_bar = df.taker_buy_base_vol * bar_range
    df[f'volume_climax_up_{lookback}'] = volume_up_bar == volume_up_bar.rolling(lookback).max()

    return df


def volume_climax_down(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    bar_range = (df.close - df.open) / ((df.close + df.open) / 2)
    volume_dn_bar = (df.base_vol - df.taker_buy_base_vol) * (1 - bar_range)
    df[f'volume_climax_dn_{lookback}'] = volume_dn_bar == volume_dn_bar.rolling(lookback).max()

    return df


def high_volume_churn(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    wick_range = (df.high - df.low) / ((df.high + df.low) / 2)
    churn = df.base_vol / wick_range
    df[f'high_volume_churn_{lookback}'] = churn == churn.rolling(lookback).max()

    return df


def low_volume_bar(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f'low_volume_{lookback}'] = df.base_vol == df.base_vol.rolling(lookback).min()

    return df


def round_numbers_proximity(df: pd.DataFrame) -> pd.DataFrame:
    nums = [x * y for x in range(1, 10) for y in [10 ** z for z in range(-4, 6)]]
    df['round_num_prox'] = df.vwma.map(lambda x: min([abs(x - value) / ((x + value) / 2) for value in nums]))

    return df


def round_numbers_close(df: pd.DataFrame, threshold) -> pd.DataFrame:
    nums = [x * y for x in range(1, 10) for y in [10 ** z for z in range(-4, 6)]]
    round_num_prox = df.vwma.map(lambda x: min([abs(x - value) / ((x + value) / 2) for value in nums]))
    df[f"round_nums_close_{threshold}"] = (round_num_prox * 100) < threshold

    return df


def big_round_nums_proximity(df: pd.DataFrame) -> pd.DataFrame:
    big_nums = [10 ** z for z in range(-4, 6)]
    df['big_round_num_prox'] = df.vwma.map(lambda x: min([abs(x - value) / ((x + value) / 2) for value in big_nums]))

    return df


def big_round_nums_close(df: pd.DataFrame, threshold) -> pd.DataFrame:
    big_nums = [10 ** z for z in range(-4, 6)]
    big_round_num_prox = df.vwma.map(lambda x: min([abs(x - value) / ((x + value) / 2) for value in big_nums]))
    df[f"big_round_nums_close_{threshold}"] = (big_round_num_prox * 100) < threshold

    return df


def spooky_nums_proximity(df: pd.DataFrame) -> pd.DataFrame:
    spooky_nums = [8, 13, 39, 69, 88, 420, 666, 888]
    nums = [x * y for x in spooky_nums for y in [10 ** z for z in range(-4, 6)]]
    df['spooky_num_prox'] = df.vwma.map(lambda x: min([abs(x - value) / ((x + value) / 2) for value in nums]))

    return df


def daily_open_ratio(df: pd.DataFrame):
    # for each period in the df, find the daily open, then divide each period's close price by that daily open
    pass


def atr_pct(df, lb):
    df = ind.atr(df, lb)
    return df.drop(f'atr-{lb}', axis=1)


def stoch_vwma_ratio(df, lookback: int) -> pd.DataFrame:
    df[f"stoch_vwma_ratio_{lookback}"] = ind.stochastic(df.close / df.vwma, lookback)
    return df


def ema_roc(df, length) -> pd.DataFrame:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    df[f"ema_{length}_roc"] = (df[f"ema_{length}"].pct_change())

    return df


def ema_ratio(df, length) -> pd.DataFrame:
    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()

    df[f"ema_{length}_ratio"] = (df.close / df[f"ema_{length}"])

    return df


def ema_breakout(df: pd.DataFrame, length: int, lookback: int) -> pd.DataFrame:
    """creates two columns 'ema_break_up' and 'ema_break_down' which represent whether the ema of close prices is above or below the
    range it occupied over the lookback period. if both are false, it is within the range."""

    if f"ema_{length}" not in df.columns:
        df[f"ema_{length}"] = df.close.ewm(length).mean()
    ema_high = df[f"ema_{length}"].shift(1).rolling(lookback).max()
    ema_low = df[f"ema_{length}"].shift(1).rolling(lookback).min()

    if f'ema_{length}_break_up' not in df.columns:
        df[f'ema_{length}_break_up'] = (df[f"ema_{length}"] > ema_high)
        df[f'ema_{length}_break_down'] = (df[f"ema_{length}"] < ema_low)

    return df


def engulfing(df, lookback: int = 1) -> pd.DataFrame:
    if f'bullish_engulf_{lookback}' not in df.columns:
        df = ind.engulfing(df, lookback)

    return df


def doji(df: pd.DataFrame, thresh: float, lookback: int, weighted: bool) -> pd.DataFrame:
    """returns booleans which represent whether there was an upper or lower wick which met the threshold requirement in
    the lookback window. If weighted == True, a rolling 50 period z-score of base volume is calculated and the wick size
    is multiplied by the volume z-score, so higher than average volume will increase the chance of the wick being
    considered over the threshold"""

    w = f'weighted_{int(thresh * 100)}' if weighted else 'unweighted'

    if f'{w}_bull_doji' in df.columns:
        return df

    df = ind.doji(df)

    if weighted:
        z_volume = ind.z_score(df.base_vol, 50)
        bull_doji_bool = df.bullish_doji * z_volume >= thresh
        bear_doji_bool = df.bearish_doji * z_volume >= thresh
    else:
        bull_doji_bool = df.bullish_doji >= thresh
        bear_doji_bool = df.bearish_doji >= thresh

    # df = df.drop(['bullish_doji', 'bearish_doji'], axis=1)

    bull_bool_window = bull_doji_bool.rolling(lookback).sum() > 0
    bear_bool_window = bear_doji_bool.rolling(lookback).sum() > 0

    df[f'{w}_bull_doji'] = bull_bool_window
    df[f'{w}_bear_doji'] = bear_bool_window

    return df


def bull_bear_bar(df) -> pd.DataFrame:
    if 'bullish_bar' not in df.columns:
        df = ind.bull_bear_bar(df)

    return df


def hour(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df.timestamp.dt.hour
    return df


def hour_180(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_180'] = (df.timestamp.dt.hour + 12) % 24
    return df


def day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df.timestamp.dt.dayofweek
    return df


def day_of_week_180(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week_180'] = (df.timestamp.dt.dayofweek + 3) % 7
    return df


def week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    df['week_of_year'] = df.timestamp.dt.dayofyear // 7
    return df


def week_of_year_180(df: pd.DataFrame) -> pd.DataFrame:
    df['week_of_year_180'] = ((df.timestamp.dt.dayofyear // 7) + 26) % 52
    return df


def vol_denom_roc(df: pd.DataFrame, roc_lb: int, atr_lb: int) -> pd.DataFrame:
    """returns the roc of price over the specified lookback period divided by the percentage denominated atr"""
    if f'atr_{atr_lb}_pct' not in df.columns:
        df = atr_pct(df, atr_lb)

    df[f'vol_denom_roc_{roc_lb}'] = (df.close.pct_change(roc_lb) / df[f'atr_{atr_lb}_pct'])

    return df


def vol_delta_div(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """adds a boolean series to the dataframe representing whether there was a volume delta divergence at any time
    during the lookback period"""
    roc: pd.Series = df.close.pct_change(1)
    if 'vol_delta' not in df.columns:
        df['vol_delta'] = ind.vol_delta(df)

    vd_div_a = (roc > 0) & (df.vol_delta < 0)
    vd_div_b = (roc < 0) & (df.vol_delta > 0)
    vd_div = vd_div_a | vd_div_b

    df[f'recent_vd_div_{lookback}'] = vd_div.rolling(lookback).sum() > 0

    return df


def ats_z(df: pd.DataFrame, lookback: int):
    avg_trade_size_sm = (df.base_vol / df.num_trades).ewm(5).mean()
    ats_long_mean = avg_trade_size_sm.ewm(lookback).mean()
    ats_std = avg_trade_size_sm.ewm(lookback).std()
    df[f'ats_z_{lookback}'] = ((avg_trade_size_sm - ats_long_mean) / ats_std)

    return df


def hma_roc(df, length) -> pd.DataFrame:
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    df[f"hma_{length}_roc"] = (df[f"hma_{length}"].pct_change())

    return df


def hma_ratio(df, length) -> pd.DataFrame:
    if f"hma_{length}" not in df.columns:
        df[f"hma_{length}"] = ind.hma(df.close, length)

    df[f"hma_{length}_ratio"] = (df.close / df[f"hma_{length}"])

    return df


def channel_mid_ratio(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """ratio of closing price to the mid-point between the highest high and lowest low over the lookback window"""

    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    df[f'chan_mid_ratio_{lookback}'] = (df.close / (chan_hi + chan_lo) / 2)

    return df


def channel_mid_width(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """this is a kind of volatility metric, similar to bollinger bands width. the channel is the space between the
    highest high and the lowest low in the lookback  window, so the width of the channel (as a proportion of the
    mid-price) is an indication of how volatile price has been in that window."""

    chan_hi = df.high.rolling(lookback).max()
    chan_lo = df.low.rolling(lookback).min()

    chan_mid = (chan_hi + chan_lo) / 2

    df[f'chan_mid_width_{lookback}'] = ((chan_hi - chan_lo) / chan_mid)

    return df


def daily_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'daily_open' not in df.columns:
        df = ind.daily_open(df)

    df['daily_open_ratio'] = (df.close / df.daily_open)
    return df


def prev_daily_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_open' not in df.columns:
        df = ind.prev_daily_open(df)

    df['prev_daily_open_ratio'] = (df.close / df.prev_daily_open)
    return df


def prev_daily_high_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_high' not in df.columns:
        df = ind.prev_daily_high(df)

    df['prev_daily_high_ratio'] = (df.close / df.prev_daily_high)
    return df


def prev_daily_low_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_daily_low' not in df.columns:
        df = ind.prev_daily_low(df)

    df['prev_daily_low_ratio'] = (df.close / df.prev_daily_low)
    return df


def weekly_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'weekly_open' not in df.columns:
        df = ind.weekly_open(df)

    df['weekly_open_ratio'] = (df.close / df.weekly_open)
    return df


def prev_weekly_open_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_weekly_open' not in df.columns:
        df = ind.prev_weekly_open(df)

    df['prev_weekly_open_ratio'] = (df.close / df.prev_weekly_open)
    return df


def prev_weekly_high_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_weekly_high' not in df.columns:
        df = ind.prev_weekly_high(df)

    df['prev_weekly_high_ratio'] = (df.close / df.prev_weekly_high)
    return df


def prev_weekly_low_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'prev_weekly_low' not in df.columns:
        df = ind.prev_weekly_low(df)

    df['prev_weekly_low_ratio'] = (df.close / df.prev_weekly_low)
    return df


def vol_delta_pct(df: pd.DataFrame) -> pd.DataFrame:
    df['vol_delta_pct'] = ind.vol_delta_pct(df)
    return df


def stoch_base_vol(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f'stoch_base_vol_{lookback}'] = ind.stochastic(df.base_vol, lookback)
    return df


def stoch_num_trades(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f'stoch_num_trades_{lookback}'] = ind.stochastic(df.num_trades, lookback)
    return df


def inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    df['inside_bar'] = ind.inside_bars(df)
    return df


def rsi(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df[f"rsi_{lookback}"] = ind.rsi(df.close, lookback)
    return df


def rsi_above(df: pd.DataFrame, lookback: int, thresh: int) -> pd.DataFrame:
    df[f"rsi_{lookback}_above_{thresh}"] = ind.rsi(df.close, lookback) > thresh
    return df


def daily_roc(df: pd.DataFrame, timeframe) -> pd.DataFrame:
    periods_1d = {'1h': 24, '4h': 6, '12h': 2, '1d': 1}
    df['roc_1d'] = df.close.pct_change(periods_1d[timeframe])
    return df


def weekly_roc(df: pd.DataFrame, timeframe) -> pd.DataFrame:
    periods_1w = {'1h': 168, '4h': 42, '12h': 14, '1d': 7}
    df['roc_1w'] = df.close.pct_change(periods_1w[timeframe])
    return df


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['log_returns'] = np.log(df.close.pct_change() + 1)
    return df


def kurtosis(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if 'log_returns' not in df.columns:
        df = log_returns(df)

    df[f'kurtosis_{lookback}'] = df.log_returns.rolling(lookback).kurt()

    return df


def skew(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if 'log_returns' not in df.columns:
        df = log_returns(df)

    df[f'skew_{lookback}'] = df.log_returns.rolling(lookback).skew()

    return df


def vol_delta(df: pd.DataFrame) -> pd.DataFrame:
    df['vol_delta'] = ind.vol_delta(df)
    return df


def fractal_trend_age(df: pd.DataFrame, width: int = 5, spacing: int = 2) -> pd.DataFrame:
    if 'frac_low' not in df.columns:
        df = ind.williams_fractals(df, width, spacing)
        df = df.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1)

    long_trend_condition = df.open > df.frac_low
    df['fractal_trend_age_long'] = ind.consec_condition(long_trend_condition)

    short_trend_condition = df.open < df.frac_high
    df['fractal_trend_age_short'] = ind.consec_condition(short_trend_condition)

    return df


def stoch_w(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    stochastic W returns True on any row which meets the following conditions;
    a 'W' pattern (high, low, high, low, high) has formed on the stochastic oscillator,
    each point in the pattern is below the midline of the oscillator
    the close price at the end of the pattern is lower than the close price at the start of the pattern
    """

    stoch = ind.stochastic(df.close, lookback)

    cond_1 = stoch.iloc[-2] < stoch.iloc[-1] < 0.5
    cond_2 = stoch.iloc[-2] < stoch.iloc[-3] < 0.5
    cond_3 = stoch.iloc[-4] < stoch.iloc[-3]
    cond_4 = stoch.iloc[-4] < stoch.iloc[-5] < 0.5
    cond_5 = df.close.iloc[-1] < df.close.iloc[-5]

    df[f"stoch_w_{lookback}"] = cond_1 & cond_2 & cond_3 & cond_4 & cond_5

    return df


def stoch_m(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    stochastic M returns True on any row which meets the following conditions;
    an 'M' pattern (low, high, low, high, low) has formed on the stochastic oscillator,
    each point in the pattern is above the midline of the oscillator
    the close price at the end of the pattern is higher than the close price at the start of the pattern
    """

    stoch = ind.stochastic(df.close, lookback)

    cond_1 = stoch.iloc[-2] > stoch.iloc[-1] > 0.5
    cond_2 = stoch.iloc[-2] > stoch.iloc[-3] > 0.5
    cond_3 = stoch.iloc[-4] > stoch.iloc[-3]
    cond_4 = stoch.iloc[-4] > stoch.iloc[-5] > 0.5
    cond_5 = df.close.iloc[-1] > df.close.iloc[-5]

    df[f"stoch_m_{lookback}"] = cond_1 & cond_2 & cond_3 & cond_4 & cond_5

    return df

# def rolling_poc(df: pd.DataFrame, lookback: int=24) -> pd.DataFrame:
#     data = df.loc[:, ['close', 'base_vol']]
#     df[f"rolling_poc_{lookback}"] = (data.rolling(lookback, method='table')
#                                      .apply(ind.vol_profile_poc, raw=True, engine='numba'))
#
#     return df

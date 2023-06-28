import joblib
import json
import pandas as pd
import time
from pathlib import Path
from pprint import pprint
from machine_learning import entry_modelling as em
import indicators as ind
from machine_learning import features
from datetime import datetime
from pushbullet import Pushbullet
# import update_ohlc

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

timeframe = '1h'

# paths
folder = Path("/home/ross/coding/modular_trader/machine_learning/models/trail_fractals")
long_model_path = folder / f"trail_fractal_long_{timeframe}_model.sav"
short_model_path = folder / f"trail_fractal_short_{timeframe}_model.sav"
long_info_path = folder / f"trail_fractal_long_{timeframe}_info.json"
short_info_path = folder / f"trail_fractal_short_{timeframe}_info.json"

long_model = joblib.load(long_model_path)
short_model = joblib.load(short_model_path)
with open(long_info_path, 'r') as ip:
    long_info = json.load(ip)
with open(short_info_path, 'r') as ip:
    short_info = json.load(ip)

pairs = long_info['pairs']
feature_set = set(long_info['features'] + short_info['features'])
width = long_info['frac_width']
spacing = long_info['atr_spacing']
pprint(feature_set)

def add_feature(df, name, timeframe):
    feature_lookup = {
        'atr_5_pct': {'call': features.atr_pct, 'params': (df, 5)},
        'atr_10_pct': {'call': features.atr_pct, 'params': (df, 10)},
        'atr_25_pct': {'call': features.atr_pct, 'params': (df, 25)},
        'atr_50_pct': {'call': features.atr_pct, 'params': (df, 50)},
        'ats_z_25': {'call': features.ats_z, 'params': (df, 25)},
        'ats_z_50': {'call': features.ats_z, 'params': (df, 50)},
        'ats_z_100': {'call': features.ats_z, 'params': (df, 100)},
        'ats_z_200': {'call': features.ats_z, 'params': (df, 200)},
        'bullish_bar': {'call': features.bull_bear_bar, 'params': (df,)},
        'bearish_bar': {'call': features.bull_bear_bar, 'params': (df,)},
        'bullish_doji': {'call': features.doji, 'params': (df, 0.5, 2)},
        'bearish_doji': {'call': features.doji, 'params': (df, 0.5, 2)},
        'bearish_engulf_1': {'call': features.engulfing, 'params': (df, 1)},
        'bearish_engulf_2': {'call': features.engulfing, 'params': (df, 2)},
        'bearish_engulf_3': {'call': features.engulfing, 'params': (df, 3)},
        'bullish_engulf_1': {'call': features.engulfing, 'params': (df, 1)},
        'bullish_engulf_2': {'call': features.engulfing, 'params': (df, 2)},
        'bullish_engulf_3': {'call': features.engulfing, 'params': (df, 3)},
        'chan_mid_ratio_25': {'call': features.channel_mid_ratio, 'params': (df, 25)},
        'chan_mid_ratio_50': {'call': features.channel_mid_ratio, 'params': (df, 50)},
        'chan_mid_ratio_100': {'call': features.channel_mid_ratio, 'params': (df, 100)},
        'chan_mid_ratio_200': {'call': features.channel_mid_ratio, 'params': (df, 200)},
        'chan_mid_width_25': {'call': features.channel_mid_width, 'params': (df, 25)},
        'chan_mid_width_50': {'call': features.channel_mid_width, 'params': (df, 50)},
        'chan_mid_width_100': {'call': features.channel_mid_width, 'params': (df, 100)},
        'chan_mid_width_200': {'call': features.channel_mid_width, 'params': (df, 200)},
        'daily_open_ratio': {'call': features.daily_open_ratio, 'params': (df,)},
        'daily_roc': {'call': features.daily_roc, 'params': (df, timeframe)},
        'day_of_week': {'call': features.day_of_week, 'params': (df,)},
        'day_of_week_180': {'call': features.day_of_week_180, 'params': (df,)},
        'ema_12_break_up': {'call': features.ema_breakout, 'params': (df, 12, 25)},
        'ema_12_break_down': {'call': features.ema_breakout, 'params': (df, 12, 25)},
        'ema_25_break_up': {'call': features.ema_breakout, 'params': (df, 25, 50)},
        'ema_25_break_down': {'call': features.ema_breakout, 'params': (df, 25, 50)},
        'ema_50_break_up': {'call': features.ema_breakout, 'params': (df, 50, 100)},
        'ema_50_break_down': {'call': features.ema_breakout, 'params': (df, 50, 100)},
        'ema_100_break_up': {'call': features.ema_breakout, 'params': (df, 100, 200)},
        'ema_100_break_down': {'call': features.ema_breakout, 'params': (df, 100, 200)},
        'ema_25_ratio': {'call': features.ema_ratio, 'params': (df, 25)},
        'ema_50_ratio': {'call': features.ema_ratio, 'params': (df, 50)},
        'ema_100_ratio': {'call': features.ema_ratio, 'params': (df, 100)},
        'ema_200_ratio': {'call': features.ema_ratio, 'params': (df, 200)},
        'ema_25_roc': {'call': features.ema_roc, 'params': (df, 25)},
        'ema_50_roc': {'call': features.ema_roc, 'params': (df, 50)},
        'ema_100_roc': {'call': features.ema_roc, 'params': (df, 100)},
        'ema_200_roc': {'call': features.ema_roc, 'params': (df, 200)},
        'hma_25_ratio': {'call': features.hma_ratio, 'params': (df, 25)},
        'hma_50_ratio': {'call': features.hma_ratio, 'params': (df, 50)},
        'hma_100_ratio': {'call': features.hma_ratio, 'params': (df, 100)},
        'hma_200_ratio': {'call': features.hma_ratio, 'params': (df, 200)},
        'hma_25_roc': {'call': features.hma_roc, 'params': (df, 25)},
        'hma_50_roc': {'call': features.hma_roc, 'params': (df, 50)},
        'hma_100_roc': {'call': features.hma_roc, 'params': (df, 100)},
        'hma_200_roc': {'call': features.hma_roc, 'params': (df, 200)},
        'hour': {'call': features.hour, 'params': (df,)},
        'hour_180': {'call': features.hour_180, 'params': (df,)},
        'inside_bar': {'call': features.inside_bar, 'params': (df,)},
        'kurtosis_6': {'call': features.kurtosis, 'params': (df, 6)},
        'kurtosis_12': {'call': features.kurtosis, 'params': (df, 12)},
        'kurtosis_25': {'call': features.kurtosis, 'params': (df, 25)},
        'kurtosis_50': {'call': features.kurtosis, 'params': (df, 50)},
        'kurtosis_100': {'call': features.kurtosis, 'params': (df, 100)},
        'kurtosis_200': {'call': features.kurtosis, 'params': (df, 200)},
        'prev_daily_open_ratio': {'call': features.prev_daily_open_ratio, 'params': (df,)},
        'prev_daily_high_ratio': {'call': features.prev_daily_high_ratio, 'params': (df,)},
        'prev_daily_low_ratio': {'call': features.prev_daily_low_ratio, 'params': (df,)},
        'recent_bull_doji': {'call': features.doji, 'params': (df, 0.5, 2)},
        'recent_bear_doji': {'call': features.doji, 'params': (df, 0.5, 2)},
        'recent_vd_div_1': {'call': features.vol_delta_div, 'params': (df, 1)},
        'recent_vd_div_2': {'call': features.vol_delta_div, 'params': (df, 2)},
        'recent_vd_div_3': {'call': features.vol_delta_div, 'params': (df, 3)},
        'recent_vd_div_4': {'call': features.vol_delta_div, 'params': (df, 4)},
        'rsi_14': {'call': features.rsi, 'params': (df, 14)},
        'rsi_25': {'call': features.rsi, 'params': (df, 25)},
        'rsi_50': {'call': features.rsi, 'params': (df, 50)},
        'rsi_100': {'call': features.rsi, 'params': (df, 100)},
        'rsi_200': {'call': features.rsi, 'params': (df, 200)},
        'skew_6': {'call': features.skew, 'params': (df, 6)},
        'skew_12': {'call': features.skew, 'params': (df, 12)},
        'skew_25': {'call': features.skew, 'params': (df, 25)},
        'skew_50': {'call': features.skew, 'params': (df, 50)},
        'skew_100': {'call': features.skew, 'params': (df, 100)},
        'skew_200': {'call': features.skew, 'params': (df, 200)},
        'stoch_base_vol_25': {'call': features.stoch_base_vol, 'params': (df, 25)},
        'stoch_base_vol_50': {'call': features.stoch_base_vol, 'params': (df, 50)},
        'stoch_base_vol_100': {'call': features.stoch_base_vol, 'params': (df, 100)},
        'stoch_base_vol_200': {'call': features.stoch_base_vol, 'params': (df, 200)},
        'stoch_num_trades_25': {'call': features.stoch_num_trades, 'params': (df, 25)},
        'stoch_num_trades_50': {'call': features.stoch_num_trades, 'params': (df, 50)},
        'stoch_num_trades_100': {'call': features.stoch_num_trades, 'params': (df, 100)},
        'stoch_num_trades_200': {'call': features.stoch_num_trades, 'params': (df, 200)},
        'stoch_vwma_ratio_25': {'call': features.stoch_vwma_ratio, 'params': (df, 25)},
        'stoch_vwma_ratio_50': {'call': features.stoch_vwma_ratio, 'params': (df, 50)},
        'stoch_vwma_ratio_100': {'call': features.stoch_vwma_ratio, 'params': (df, 100)},
        'vol_delta': {'call': features.vol_delta, 'params': (df,)},
        'vol_delta_pct': {'call': features.vol_delta_pct, 'params': (df,)},
        'vol_denom_roc_2': {'call': features.vol_denom_roc, 'params': (df, 2, 25)},
        'vol_denom_roc_5': {'call': features.vol_denom_roc, 'params': (df, 2, 50)},
        'week_of_year': {'call': features.week_of_year, 'params': (df,)},
        'week_of_year_180': {'call': features.week_of_year_180, 'params': (df,)},
        'weekly_roc': {'call': features.weekly_roc, 'params': (df, timeframe)}
    }
    feature = feature_lookup[name]
    df = feature['call'](*feature['params'])

    return df

for pair in pairs:
    # print(f"\nTesting {pair}")
    df = em.get_data(pair, timeframe)
    df = ind.williams_fractals(df, width, spacing)
    df = (df.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1)
          .dropna(axis=0).reset_index(drop=True))

    for f in feature_set:
        # print(f"processing {f}")
        if f == 'r_pct':
            continue
        df = add_feature(df, f, timeframe)

    # print(df.columns)
    X, _, cols = em.transform_columns(df, df)
    df = pd.DataFrame(X, columns=cols)
    df['long_r_pct'] = abs(df.close - df.frac_low) / df.close
    df['short_r_pct'] = abs(df.close - df.frac_high) / df.close
    # print(df.tail())

    # Long model
    df['r_pct'] = df.long_r_pct
    long_features = df[long_info['features']].iloc[-1]
    long_X = pd.DataFrame(long_features).transpose()
    long_confidence = long_model.predict_proba(long_X)[0, 1]

    # Short model
    df = df.drop('long_r_pct', axis=1)
    df['r_pct'] = df.short_r_pct
    short_features = df[short_info['features']].iloc[-1]
    short_X = pd.DataFrame(short_features).transpose()
    short_confidence = short_model.predict_proba(short_X)[0, 1]

    combined_long = long_confidence - short_confidence
    combined_short = short_confidence - long_confidence

    now = datetime.now().strftime('%d/%m/%y %H:%M')

    if combined_long > 0.5:
        note = f"Buy {pair} @ {df.close.iloc[-1]} confidence: {combined_long:.1%}"
        print(note)
        pb.push_note(now, note)
    if combined_short > 0.5:
        note = f"Sell {pair} @ {df.close.iloc[-1]} confidence: {combined_short:.1%}"
        print(note)
        pb.push_note(now, note)

    # need to add r_pct to the features dict before i give it to the model
    # print(df.tail())

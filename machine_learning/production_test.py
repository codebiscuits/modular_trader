import joblib
import json
import pandas as pd
import time
from pathlib import Path
from pprint import pprint
import entry_modelling as em
import indicators as ind
import features
from sklearn.ensemble import GradientBoostingClassifier

timeframe = '1h'
side = 'short'

long_model_path = Path(f"models/trail_fractals/trail_fractal_long_{timeframe}_model.sav")
short_model_path = Path(f"models/trail_fractals/trail_fractal_short_{timeframe}_model.sav")
long_info_path = Path(f"models/trail_fractals/trail_fractal_long_{timeframe}_info.json")
short_info_path = Path(f"models/trail_fractals/trail_fractal_short_{timeframe}_info.json")
ohlc_path = Path("/home/ross/coding/modular_trader/bin_ohlc_5m")

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
print(feature_set)

def add_feature(df, name):
    feature_lookup = {
        'ema_25_ratio': features.ema_ratio(df, 25), 'ema_50_ratio': features.ema_ratio(df, 50),
        'ema_100_ratio': features.ema_ratio(df, 100), 'ema_200_ratio': features.ema_ratio(df, 200),
        'ema_25_roc': features.ema_roc(df, 25), 'ema_50_roc': features.ema_roc(df, 50),
        'ema_100_roc': features.ema_roc(df, 100), 'ema_200_roc': features.ema_roc(df, 200),
        'hma_25_ratio': features.hma_ratio(df, 25), 'hma_50_ratio': features.hma_ratio(df, 50),
        'hma_100_ratio': features.hma_ratio(df, 100), 'hma_200_ratio': features.hma_ratio(df, 200),
        'hma_25_roc': features.hma_roc(df, 25), 'hma_50_roc': features.hma_roc(df, 50),
        'hma_100_roc': features.hma_roc(df, 100), 'hma_200_roc': features.hma_roc(df, 200),
        'skew_6': features.skew(df, 6), 'skew_12': features.skew(df, 12),
        'skew_25': features.skew(df, 25), 'skew_50': features.skew(df, 50),
        'skew_100': features.skew(df, 100), 'skew_200': features.skew(df, 200),
        'kurtosis_6': features.kurtosis(df, 6), 'kurtosis_12': features.kurtosis(df, 12),
        'kurtosis_25': features.kurtosis(df, 25), 'kurtosis_50': features.kurtosis(df, 50),
        'kurtosis_100': features.kurtosis(df, 100), 'kurtosis_200': features.kurtosis(df, 200),
        'chan_mid_width_25': features.channel_mid_width(df, 25), 'chan_mid_width_50': features.channel_mid_width(df, 50),
        'chan_mid_width_100': features.channel_mid_width(df, 100), 'chan_mid_width_200': features.channel_mid_width(df, 200),
        'bearish_engulf_1': features.engulfing(df, 1), 'bearish_engulf_2': features.engulfing(df, 2),
        'bearish_engulf_3': features.engulfing(df, 3), 'bullish_engulf_1': features.engulfing(df, 1),
        'bullish_engulf_2': features.engulfing(df, 2), 'bullish_engulf_3': features.engulfing(df, 3),
        'bullish_doji': features.doji(df, 0.5, 2), 'bearish_doji': features.doji(df, 0.5, 2),
        'stoch_vwma_ratio_25': features.stoch_vwma_ratio(df, 25), 'stoch_vwma_ratio_50': features.stoch_vwma_ratio(df, 50),
        'stoch_vwma_ratio_100': features.stoch_vwma_ratio(df, 100),
        'vol_delta': features.vol_delta(df),
        'ema_12_break_up': features.ema_breakout(df, 12, 25), 'ema_12_break_down': features.ema_breakout(df, 12, 25),
        'ema_25_break_up': features.ema_breakout(df, 25, 50), 'ema_25_break_down': features.ema_breakout(df, 25, 50),
        'ema_50_break_up': features.ema_breakout(df, 50, 100), 'ema_50_break_down': features.ema_breakout(df, 50, 100),
        'ema_100_break_up': features.ema_breakout(df, 100, 200), 'ema_100_break_down': features.ema_breakout(df, 100, 200),
    }

    return feature_lookup[name]

for pair in pairs:
    print(f"Testing {pair}")
    df = em.get_data(pair, timeframe)
    df = ind.williams_fractals(df, width, spacing)
    df = (df.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1)
          .dropna(axis=0).reset_index(drop=True))

    print(df.head())

    for f in feature_set:
        print(f"processing {f}")
        if f == 'r_pct':
            continue
        df = add_feature(df, f)

    print(df.tail())

    # need to add r_pct to the features dict before i give it to the model
    # print(df.tail())

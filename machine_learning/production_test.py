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
ohlc_path = Path("/home/ross/Documents/backtester_2021/bin_ohlc_5m")

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
feature_lookup = {
    'hma_50_roc': features,
    'skew_200',
    'ema_200_ratio',
    'kurtosis_200',
    'skew_12',
    'skew_25',
    'chan_mid_width_100',
    'chan_mid_width_25',
    'hma_100_roc',
    'hma_200_ratio',
    'hma_100_ratio',
    'bearish_engulf_2',
    'ema_100_ratio',
    'stoch_vwma_ratio_100',
    'bearish_doji',
    'vol_delta',
    'hma_25_ratio',
    'hma_200_roc',
    'r_pct',
    'ema_100_break_down',
    'ema_12_break_up'
}

for pair in pairs:
    print(f"Testing {pair}")
    df = pd.read_parquet(ohlc_path / f"{pair}.parquet")
    df = ind.williams_fractals(df, width, spacing)
    df = (df.drop(['fractal_high', 'fractal_low', f"atr-{spacing}", f"atr_{spacing}_pct"], axis=1)
          .dropna(axis=0).reset_index(drop=True))
    # for f in features:
    #
    # print(df.tail())

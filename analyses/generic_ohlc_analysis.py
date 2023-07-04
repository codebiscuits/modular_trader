import pandas as pd
from pathlib import Path
from functions import indicators as ind, binance_funcs as funcs

all_pairs = funcs.get_pairs()
stables = ['EURUSDT', 'SUSDUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT', 'USTUSDT',
           'USDUSDT', 'GBPUSDT', 'MIMUSDT', 'FRAXUSDT', 'TUSDUSDT', 'USDPUSDT', 
           'AUDUSDT']
pairs = [p for p in all_pairs if p not in stables]

drops = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'st_loose_u', 'st_loose_d', 'st_u', 'st_d']

stats = []

for pair in pairs:
    filepath = Path(f'/mnt/pi_2/ohlc_binance_1h/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        if len(df.index) < 100:
            continue
        
        ind.supertrend_new(df, 1, 1)        
        
        # print(pair)
        # print(df.drop(drops, axis=1).tail())
        # print('-')


import pandas as pd
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

res_path = Path("/home/ross/Documents/backtester_2021/machine_learning/results/")

for p in res_path.glob('*'):
    pair = p.parts[7].split('_')[0]
    tf = p.parts[7].split('_')[2].split('.')[0]
    if tf == '12h':
        print('')
        df = pd.read_parquet(p).sort_values('precision', ascending=False)
        print(df.head(1))

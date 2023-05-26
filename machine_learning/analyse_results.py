import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

df = pd.read_parquet('/home/ross/coding/modular_trader/machine_learning/results/BTCUSDT_short_1h.parquet')

print(df.sort_values('precision', ascending=False).head(20))
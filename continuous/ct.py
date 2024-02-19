"""this is the continuous trader equivalent of setup_scanner, the central script that puts everything together and runs
it all"""

from wootrade import Client as Client_w
from binance.client import Client as Client_b
import binance.enums as be
import mt.resources.keys as keys
from datetime import datetime, timedelta, timezone
import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics as stats
from pathlib import Path
import components
from time import perf_counter

all_start = perf_counter()

pl.Config(tbl_cols=20, tbl_rows=50, tbl_width_chars=180)
# client = Client_w(keys.woo_key, keys.woo_secret, keys.woo_app_id, testnet=True)
# client = Client_b(keys.bPkey, keys.bSkey)

markets = [
    'BTCUSDT',
    # 'ETHUSDT',
#     'SOLUSDT',
#     'FXSUSDT',
#     'ROSEUSDT',
]

session = components.Session()
traders = {m: components.Trader(m) for m in markets}

for m in markets:
    print(m)

    traders[m].add_substrat(components.IchiTrend, {'market': m, 'timeframe': '8h', 'a': 10, 'b': 30})
    # traders[m].add_substrat(components.IchiTrend, {'market': m, 'timeframe': '8h', 'a': 20, 'b': 60})
    # traders[m].add_substrat(components.IchiTrend, {'market': m, 'timeframe': '1d', 'a': 10, 'b': 30})
    # traders[m].add_substrat(components.IchiTrend, {'market': m, 'timeframe': '1d', 'a': 20, 'b': 60})

    for ss in traders[m].strats:
        print(ss)
        print(ss.data.fetch())

all_end = perf_counter()
print(f"elapsed time: {all_end - all_start:.1f}s")
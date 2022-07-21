import keys
import math
import time
import json
import statistics as stats
import pandas as pd
import numpy as np
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
from pushbullet import Pushbullet
from decimal import Decimal
from pprint import pprint
from config import ohlc_data, not_pairs
from pathlib import Path
from datetime import datetime
import utility_funcs as uf
import indicators as ind
from timers import Timer
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import matplotlib.pyplot as plt

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

df = pd.read_pickle('/mnt/pi_2/ohlc_binance_1h/BTCUSDT.pkl')
ind.trend_score_new(df, 'close')
df = df.tail(1000)

fig, axs = plt.subplots(2)
fig.suptitle('Price at the top, Trend Score below')
axs[0].plot(df['close'])
axs[1].plot(df['trend_score'])
plt.show()


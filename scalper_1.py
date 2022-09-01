import keys
import math
import time
import json
import websocket
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

#########################################################################################

# historical test data
# df = pd.read_pickle('/mnt/pi_2/ohlc_binance_1h/BTCUSDT.pkl')
# ind.trend_score(df, 'close')
# df = df.tail(1000)

#########################################################################################

# live websocket data
socket ='wss://stream.binance.com:9443/ws/btcusdt@kline_1m'

data = []
wcloses = 0
vols = 0

def on_opn(ws):
    print('opened connection')

def on_cls(ws):
    print('closed connection')

def on_msg(ws, msg):
    global data
    global wcloses
    global vols
    json_msg = json.loads(msg)
    # print(json_msg)
    close = json_msg['k']['c']
    vol = json_msg['k']['v']
    # wcloses += close*vol
    # vols += vol
    if json_msg['k']['x']:
        pprint(json_msg)
    #     ts = json_msg['E'] # timestamp
    #     vwap = wcloses / vols
    #     tup = (ts, close, vols, vwap)
    #     data.append(tup)
    #     wcloses = 0
    #     vols = 0
    #
    # print(data[-5:])

print('reached websocket')
ws = websocket.WebSocketApp(socket, on_open=on_opn, on_close=on_cls, on_message=on_msg)
ws.run_forever()



##########################################################################################

# # plots
# fig, axs = plt.subplots(2)
# fig.suptitle('Price at the top, Trend Score below')
# axs[0].plot(df['close'])
# axs[1].plot(df['trend_score'])
# plt.show()
#

import json
from pprint import pprint
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from pushbullet import Pushbullet
import math
import statistics as stats

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

data_path = Path('/mnt/pi_2/market_data')
liq_data = data_path / 'binance_liquidity_history.txt'
slip_data = data_path / 'binance_slippage_history_2022.json'

with open(liq_data) as file:
    liquidity = [json.loads(line) for line in file.readlines()]

liq = {list(i.keys())[0]: list(i.values())[0] for i in liquidity}

multi_index = {(outerKey, innerKey): values
           for outerKey, innerDict in liq.items()
           for innerKey, values in innerDict.items()}


liq_df = pd.DataFrame(multi_index).transpose()
from pprint import pprint

import pandas as pd
from pushbullet import Pushbullet
from pathlib import Path
import json
import trade_records_funcs as trf
from binance.client import Client
import resources.keys as keys
from datetime import datetime, timezone

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

client = Client(keys.bPkey, keys.bSkey)

records_folder_1 = Path('/home/ross/coding/pi_down/modular_trader/records')
records_folder_2 = Path('/home/ross/coding/pi_2/modular_trader/records')

now = datetime.now().timestamp()

def print_positions(data, name):
    print(f'\n{name} open positions:')
    if data:
        for k, v in data.items():
            print(f"\n{'-'*20}\n{k}")
            for a, b in v.items():
                price = float(client.get_symbol_ticker(symbol=a)['price'])
                direction = b['position']['direction']
                duration = (now - (b['position']['open_time'] / 1000))
                duration_str = f"{int(duration // 3600)}h {int((duration / 60) % 60)}m"
                current_value = price * float(b['position']['base_size'])
                liability = float(b['position']['liability'])
                current_stop = float(b['position']['hard_stop'])
                open_risk_pct = (((price - current_stop) / price)
                                 if direction == 'long'
                                 else ((current_stop - price) / price))
                open_risk_usdt = current_value * open_risk_pct

                # check if it's a new trade (no adds or tps)
                new = (b['position']['base_size'] == b['position']['init_base_size']) and (len(b['trade']) == 1)

                if new:
                    pnl_ratio = ((price / float(b['trade'][0]['exe_price']))
                                 if direction == 'long' else
                                 (float(b['trade'][0]['exe_price']) / price))
                else:
                    costs = 0.0
                    returns = current_value
                    for i in b['trade']:
                        if i['action'] in ['open', 'add']:
                            costs += float(i['quote_size'])
                        elif i['action'] == 'tp':
                            returns += float(i['quote_size'])

                    pnl_ratio = (returns / costs) if direction == 'long' else (costs / returns)

                pnl_pct = f"{(pnl_ratio - 1):.1%}"
                pnl_usdt = float(b['trade'][0]['quote_size']) * (pnl_ratio - 1)

                print('')
                print(f"{a} {direction}, duration: {duration_str}")
                print(f"Value: {current_value:.2f} USDT, liability: {liability:.2f} USDT")
                print(f"PnL: {pnl_pct}, {pnl_usdt:.2f} USDT")
                print(f"Open risk: {open_risk_pct:.1%}, {open_risk_usdt:.2f} USDT")
    else:
        print('\nNone')

    # pprint(b)

data = trf.load_all(records_folder_2, ['open'])
print_positions(data, 'pi 2')

from pprint import pprint

import pandas as pd
from pushbullet import Pushbullet
from pathlib import Path
import json

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

records_folder_1 = Path('/home/ross/coding/pi_down/modular_trader/records')
records_folder_2 = Path('/home/ross/coding/pi_2/modular_trader/records')

def load_all(folder, state):
    all_data = {}
    for agent in folder.glob('*'):
        filepath = folder / agent.parts[7] / f'{state}_trades.json'
        print(filepath)

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            data = None

        if data:
            all_data[agent.parts[7]] = data

    return all_data

def print_positions(data, name):
    print(f'\n{name} open positions:')
    if data:
        for k, v in data.items():
            print(f"\n{k}")
            all_rpnls = []
            for a, b in v.items():
                if b['trade'][0]['wanted']:
                    print(f"\n{a}")
                    pprint(b)
                    rpnl = 0
                    for t in b['trade']:
                        if t.get('rpnl'):
                            rpnl += t['rpnl']
                    all_rpnls.append(rpnl)
            rpnl_s = pd.Series(all_rpnls)
            rpnl_cum = rpnl_s.cumsum()
            rpnl_df = pd.concat([rpnl_s, rpnl_cum], axis=1)
            rpnl_df.columns = ['rpnl', 'cum_rpnl']
            rpnl_df['ema_4'] = rpnl_df.rpnl.ewm(4).mean()
            rpnl_df['ema_12'] = rpnl_df.rpnl.ewm(12).mean()
            rpnl_df['ema_36'] = rpnl_df.rpnl.ewm(36).mean()
            print(rpnl_df)
    else:
        print('\nNone')

pi_2_data = load_all(records_folder_2, 'closed_sim')

print_positions(pi_2_data, 'pi 2')

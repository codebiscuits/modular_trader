import pandas as pd
from pushbullet import Pushbullet
from pathlib import Path
import trade_records_funcs as trf

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

records_folder_1 = Path('/home/ross/coding/pi_down/modular_trader/records')
records_folder_2 = Path('/home/ross/coding/pi_2/modular_trader/records')

def print_positions(data, name):
    print(f'\n{name} open positions:')
    if data:
        for k, v in data.items():
            print(f"\n{'-'*20}\n{k}")
            long_rpnls = []
            short_rpnls = []
            for a, b in v.items():
                a = float(a) / 1000 if float(a) > 20_000_000_000 else float(a)
                a = int(a) * 1_000_000_000
                if b['trade'][0].get('wanted', True):
                    # print(f"\n{a}")
                    # pprint(b)
                    state = b['trade'][0]['state']
                    dir = b['trade'][0]['direction']
                    rpnl = 0
                    for t in b['trade']:
                        if t.get('rpnl'):
                            rpnl += t['rpnl']
                    if dir == 'spot':
                        long_rpnls.append((a, state, rpnl))
                    elif dir == 'long':
                        long_rpnls.append((a, state, rpnl))
                    elif dir == 'short':
                        short_rpnls.append((a, state, rpnl))
            for k, v in {'long': long_rpnls, 'short': short_rpnls}.items():
                rpnl_df = pd.DataFrame(v, columns=['timestamp', 'state', 'rpnl'])
                rpnl_df = rpnl_df.sort_values('timestamp').reset_index(drop=True)

                rpnl_df['timestamp'] = pd.to_datetime(rpnl_df.timestamp, utc=True)
                rpnl_df['ema_4'] = rpnl_df.rpnl.ewm(4).mean()
                rpnl_df['ema_8'] = rpnl_df.rpnl.ewm(8).mean()
                rpnl_df['ema_16'] = rpnl_df.rpnl.ewm(16).mean()
                rpnl_df['ema_32'] = rpnl_df.rpnl.ewm(32).mean()
                rpnl_df['ema_64'] = rpnl_df.rpnl.ewm(64).mean()
                print('\n', k)
                print(rpnl_df.tail(1))
    else:
        print('\nNone')

data = trf.load_all(records_folder_2, ['closed', 'closed_sim'])
print_positions(data, 'pi 2')

from pprint import pprint
from pushbullet import Pushbullet
from pathlib import Path
import json

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

records_folder_1 = Path('/home/ross/coding/pi_down/modular_trader/records')
records_folder_2 = Path('/home/ross/coding/pi_2/modular_trader/records')

def load_all(folder):
    all_data = {}
    for agent in folder.glob('*'):
        filepath = folder / agent.parts[7] / 'open_trades.json'

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

            for a, b in v.items():
                print(f"\n{a}")
                pprint(b['position'])
    else:
        print('\nNone')

pi_1_data = load_all(records_folder_1)
pi_2_data = load_all(records_folder_2)

print_positions(pi_1_data, 'pi down')
print_positions(pi_2_data, 'pi 2')

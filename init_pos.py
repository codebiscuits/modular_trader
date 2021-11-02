import json
from rsi_optimising import get_pairs

pairs = get_pairs('usdt') + get_pairs('btc')

with open('positions.json', 'w') as write_file:
    positions = json.dump({x: 0 for x in pairs}, write_file)

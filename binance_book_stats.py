import pandas as pd
import matplotlib.pyplot as plt
import binance_funcs as funcs
import statistics as stats
from pprint import pprint
from datetime import datetime as dt
import json
import time
from binance.client import Client
import keys
from pushbullet import Pushbullet
from pathlib import Path

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

def get_book_stats(pair, quote, width=2):
    '''returns a dictionary containing the base asset, the quote asset, 
    the spread, and the bid and ask depth within the % range of price set 
    by the width param in base and quote denominations'''
    
    q_len = len(quote) * -1
    base = pair[:q_len]
    
    book = client.get_order_book(symbol=pair)
    
    best_bid = float(book.get('bids')[0][0])
    best_ask = float(book.get('asks')[0][0])
    mid_price = (best_bid + best_ask) / 2
    spread = (best_ask-best_bid) / mid_price
    
    max_price = mid_price * (1 + (width / 100)) # max_price is x% above price
    ask_depth = 0
    for i in book.get('asks'):
        if float(i[0]) <= max_price:
            ask_depth += float(i[1])
        else:
            break
    min_price = mid_price * (1 - (width / 100)) # max_price is x% above price
    bid_depth = 0
    for i in book.get('bids'):
        if float(i[0]) >= min_price:
            bid_depth += float(i[1])
        else:
            break
    
    q_bid_depth = bid_depth * mid_price
    q_ask_depth = ask_depth * mid_price
    
    stats = {'base': base, 'quote': quote, 'spread': spread, 
             'base_bids': bid_depth, 'base_asks': ask_depth, 
             'quote_bids': q_bid_depth, 'quote_asks': q_ask_depth}
    
    return stats 

start = time.perf_counter()

now = dt.now().strftime('%d/%m/%y %H:%M')

print(now, 'running binance book stats')

#######################################################

live = True
filepath1 = Path('/media/coding/market_data/binance_liquidity_history.txt')
filepath2 = Path('/mnt/pi_2/market_data/binance_liquidity_history.txt')
filepath3 = 'test.txt'

if filepath1.exists():
    fp = filepath1
elif filepath2.exists():
    fp = filepath2
    live = False
else:
    fp = filepath3
    live = False

if live: # only record a new observation if this is running on the correct machine
    quote = 'USDT'
    
    pairs = funcs.get_pairs(quote, 'SPOT')
    
    depth_dict = {}
    
    ba_ratios = []
    spreads = []
    
    for pair in pairs:
        pair_stats = get_book_stats(pair, quote, 2)
        depth_dict[pair] = pair_stats
        
        ba_ratio = pair_stats.get('quote_bids') / pair_stats.get('quote_asks')
        ba_ratios.append(ba_ratio)
        spreads.append(pair_stats.get('spread'))
        
    record = {now: depth_dict}

    with open(fp, 'a') as file:
        file.write(json.dumps(record))
        file.write('\n')

#######################################################

with open(fp, 'r') as file:
    records = file.readlines()

timestamps = []
avg_ratios = []
std_ratios = []
avg_spreads = []

for r in records:
    x = json.loads(r)
    
    timestamps.append(list(x.keys())[0])
    
    val = list(x.values())[0]
    ba_ratios = []
    spreads = []
    for k, v in val.items():
        ba_ratio = v.get('quote_bids') / v.get('quote_asks')
        ba_ratios.append(ba_ratio)
        spreads.append(v.get('spread'))
        
    avg_ratios.append(stats.median(ba_ratios))
    std_ratios.append(stats.stdev(ba_ratios))
    avg_spreads.append(stats.mean(spreads))
    

history = {'timestamp': timestamps, 'avg_ratio': avg_ratios, 
           'std_ratio': std_ratios, 'avg_spread': avg_spreads}

df = pd.DataFrame(history)

p1 = Path('/media/coding/scripts/backtester_2021')
p2 = Path('chart.png')
if p1.exists():
    chartpath = Path(p1 / p2)
else:
    chartpath = p2


plt.plot(df.timestamp, df.avg_ratio, label='avg_ratio')
plt.plot(df.timestamp, df.std_ratio, label='std_ratio')
plt.legend()
plt.xticks(rotation='vertical')
plt.savefig(chartpath, format='png')
plt.show()

# pb.push_note(now, )

end = time.perf_counter()

elapsed = round(end - start)

print(f'time taken: {elapsed//60}m {elapsed%60}s')



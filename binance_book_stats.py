import pandas as pd
import matplotlib.pyplot as plt
import binance_funcs as funcs
import statistics as stats
from pprint import pprint
from datetime import datetime as dt
import json
from json.decoder import JSONDecodeError
import time
from binance.client import Client
import keys
from pushbullet import Pushbullet
from pathlib import Path

#TODO if this works, i need to install it on the raspberry pi, wait for it to update the files a couple of times, then
# check the logs to see if it is updating 'the new way'. if so, delete all the code that chooses old files and old ways
# of updating. dont forget to check all mentions of the 'live' flag and put right anything ive done to them

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

def slippage(quote_size, book, side):
    best_bid = float(book.get('bids')[0][0])
    best_ask = float(book.get('asks')[0][0])

    book_side = 'asks' if side == 'buy' else 'bids'
    opposite = best_bid if side == 'buy' else best_ask
    cum_depth = 0
    for i in book[book_side]:
        cum_depth += (float(i[0])*float(i[1]))
        if cum_depth > quote_size:
            slip_price = float(i[0])
            break
    if cum_depth < quote_size:
        scalar = quote_size / cum_depth
        slip_price = book[book_side][-1][0]
        slippage = (abs(slip_price - opposite) / opposite) * scalar
        print('wasnt enough, estimating slippage')
        pb.push_note('book_stats', f"{pair} downloaded book didn't have enough depth, raise limit")
    else:
        slippage = abs(slip_price-opposite) / opposite

    return slippage


def get_book_stats(pair, book, quote, width=1):
    '''returns a dictionary containing the base asset, the quote asset, 
    the spread, and the bid and ask depth within the % range of price set 
    by the width param in base and quote denominations'''
    
    q_len = len(quote) * -1
    base = pair[:q_len]

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


def set_paths():
    curr_year = dt.now().year

    folder_1 = Path('/media/coding/market_data')
    folder_2 = Path('/mnt/pi_2/market_data')

    liq_path_a = Path("binance_liquidity_history.txt")
    liq_path_b = Path(f"binance_liquidity_history_{curr_year}.json")

    if folder_1.exists():
        fp = folder_1/liq_path_b if liq_path_b.exists() else folder_1/liq_path_a
        sf = folder_1 / f"binance_slippage_history_{curr_year}.json"
    elif folder_2.exists():
        fp = folder_2/liq_path_b if liq_path_b.exists() else folder_2/liq_path_a
        sf = folder_2 / f"binance_slippage_history_{curr_year}.json"
    else:
        fp = liq_path_b if liq_path_b.exists() else liq_path_a
        sf = Path(f'slip_test_{curr_year}.json')

    live = folder_1.exists()

    if live:
        fp.touch(exist_ok=True)
        sf.touch(exist_ok=True)

    return fp, sf, live


start = time.perf_counter()

now = dt.now().strftime('%d/%m/%y %H:%M')

print(now, 'running binance book stats')

#######################################################

fp, sf, live = set_paths()

print(fp)
print(sf)

quote = 'USDT'

pairs = funcs.get_pairs(quote, 'SPOT')

depth_dict = {}
slippage_dict = {}

for pair in pairs:
    slip_stats = {}
    book = client.get_order_book(symbol=pair, limit=500)
    slip_stats['buy_100'] = slippage(100, book, 'buy')
    slip_stats['sell_100'] = slippage(100, book, 'sell')
    slip_stats['buy_1000'] = slippage(1000, book, 'buy')
    slip_stats['sell_1000'] = slippage(1000, book, 'sell')
    slip_stats['buy_10000'] = slippage(10000, book, 'buy')
    slip_stats['sell_10000'] = slippage(10000, book, 'sell')

    slippage_dict[pair] = slip_stats

    pair_stats = get_book_stats(pair, book, quote, 2)
    depth_dict[pair] = pair_stats

if live:  # only record a new observation if this is running on the correct machine

    curr_year = dt.now().year

####################### depth data ####################################


    record = {now: depth_dict}

    try:
        with open(fp, 'r') as liq_file:
            try:
                liq_data = json.load(liq_file)
                print('loaded data the new way')
            except JSONDecodeError:
                print('loading data the old way')
                liquidity = [json.loads(line) for line in liq_file.readlines()]
                liq_data = {list(i.keys())[0]: list(i.values())[0] for i in liquidity}

        liq_data[now] = depth_dict
        with open(fp, 'w') as liq_file:
            json.dump(liq_data, liq_file)
        print('completed saving data the new way')

    except:
        print('saving data the old way')
        with open(fp, 'a') as file:
            file.write(json.dumps(record))
            file.write('\n')


################# slip data ############################

    with open(sf, 'r') as slip_file:
        try:
            all_data = json.load(slip_file)
        except JSONDecodeError as e:
            print(e)
            all_data = []
        except TypeError as e:
            print(e)
            all_data = []

    try:
        all_data[now] = slippage_dict
        print('updated slip data the new way')
    except TypeError:
        print('updating slip data the old way')
        all_data.append(slippage_dict)
        all_data = {list(i.keys())[0]: list(i.values())[0] for i in all_data}

    with open(sf, 'w') as slip_file:
        json.dump(all_data, slip_file)

#######################################################

# with open(fp, 'r') as file:
#     records = file.readlines()
#
# timestamps = []
# avg_ratios = []
# std_ratios = []
# avg_spreads = []
#
# for r in records:
#     x = json.loads(r)
#
#     timestamps.append(list(x.keys())[0])
#
#     val = list(x.values())[0]
#     ba_ratios = []
#     spreads = []
#     for k, v in val.items():
#         ba_ratio = v.get('quote_bids') / v.get('quote_asks')
#         ba_ratios.append(ba_ratio)
#         spreads.append(v.get('spread'))
#
#     avg_ratios.append(stats.median(ba_ratios))
#     std_ratios.append(stats.stdev(ba_ratios))
#     avg_spreads.append(stats.mean(spreads))
#
#
# history = {'timestamp': timestamps, 'avg_ratio': avg_ratios,
#            'std_ratio': std_ratios, 'avg_spread': avg_spreads}
#
# df = pd.DataFrame(history)
#
# p1 = Path('/media/coding/scripts/backtester_2021')
# p2 = Path('chart.png')
# if p1.exists():
#     chartpath = Path(p1 / p2)
# else:
#     chartpath = p2
#
# df['avg_ratio_ma'] = df.avg_ratio.ewm(4).mean()
#
#
# plt.plot(df.timestamp, df.avg_ratio, label='avg_ratio', linewidth=1)
# plt.plot(df.timestamp, df.avg_ratio_ma, label='avg_ratio_ma', linewidth=1)
# # plt.plot(df.timestamp, df.std_ratio, label='std_ratio')
# plt.legend()
# plt.xticks(rotation='vertical')
# plt.savefig(chartpath, format='png')
# plt.show()

end = time.perf_counter()

elapsed = round(end - start)

print(f'time taken: {elapsed//60}m {elapsed%60}s')



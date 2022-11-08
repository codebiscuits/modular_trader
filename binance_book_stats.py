import pandas as pd
import matplotlib.pyplot as plt
import binance_funcs as funcs
from datetime import datetime as dt
import json
from json.decoder import JSONDecodeError
import time
from binance.client import Client
import keys
from pushbullet import Pushbullet
from pathlib import Path
import ccxt
from pprint import pprint

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

def slippage(pair, quote_size, book, side, limit):
    try:
        best_bid = float(book.get('bids')[0][0])
        best_ask = float(book.get('asks')[0][0])
    except IndexError:
        return 1

    book_side = 'asks' if side == 'buy' else 'bids'
    opposite = best_bid if side == 'buy' else best_ask
    cum_depth = 0
    for i in book[book_side]:
        cum_depth += (float(i[0])*float(i[1]))
        if cum_depth > quote_size:
            slip_price = float(i[0])
            break
    if cum_depth < quote_size:
        slip_price = book[book_side][-1][0]
        slip_pct = abs(slip_price - opposite) / opposite
        if len(book[book_side]) < limit:
            print(f"{pair} book only has ${cum_depth} liquidity which would cause {slip_pct:.1%} slippage")
            return 1
        print(f"{book.get('symbol')} cum depth: {cum_depth}")
        scalar = quote_size / cum_depth
        print(f'wasnt enough for ${quote_size} size, estimating slippage')
        # pb.push_note('book_stats', f"{pair} downloaded book didn't have enough depth, raise limit")
        return slip_pct * scalar
    else:
        return abs(slip_price-opposite) / opposite


def get_book_stats(pair, book, quote, width=1):
    '''returns a dictionary containing the base asset, the quote asset, 
    the spread, and the bid and ask depth within the % range of price set 
    by the width param in base and quote denominations'''
    
    q_len = len(quote) * -1
    base = pair[:q_len]

    try:
        best_bid = float(book.get('bids')[0][0])
        best_ask = float(book.get('asks')[0][0])
    except IndexError:
        return {'base': base, 'quote': quote, 'spread': 1,
                'base_bids': 0, 'base_asks': 0,
                'quote_bids': 0, 'quote_asks': 0}

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
    
    return {'base': base, 'quote': quote, 'spread': spread,
            'base_bids': bid_depth, 'base_asks': ask_depth,
            'quote_bids': q_bid_depth, 'quote_asks': q_ask_depth}


def set_paths(curr_year, exchange):
    folder_1 = Path('/media/coding/market_data')
    folder_2 = Path('/mnt/pi_2/market_data')

    if folder_1.exists():
        fp = folder_1 / f"{exchange.id}_liquidity_history_{curr_year}.json"
        sf = folder_1 / f"{exchange.id}_slippage_history_{curr_year}.json"
    elif folder_2.exists():
        fp = folder_2 / f"{exchange.id}_liquidity_history_{curr_year}.json"
        sf = folder_2 / f"{exchange.id}_slippage_history_{curr_year}.json"
    else:
        fp = Path(f"liq_test_{curr_year}.json")
        sf = Path(f'slip_test_{curr_year}.json')

    live = folder_1.exists()

    if live:
        fp.touch(exist_ok=True)
        sf.touch(exist_ok=True)

    return fp, sf, live

start = time.perf_counter()

now = dt.now().strftime('%d/%m/%y %H:%M')
print(now, 'running book stats')
curr_year = dt.now().year
quote = 'USDT'
exchanges = [ccxt.ascendex, ccxt.binance, ccxt.ftx, ccxt.kucoin]

depth_dict = {}
slippage_dict = {}
for i in exchanges:
    ex = i()
    print(ex.id)
    pairs = ex.load_markets()
    for pair in pairs.keys():
        if '/' + quote in pair:
            slip_stats = {}
            base = pairs[pair]['base']
            quote = pairs[pair]['quote']
            lim = 100 if ex.id == 'kucoin' else 1000
            book = ex.fetch_order_book(pair, limit=lim)
            slip_stats['buy_100'] = slippage(pair, 100, book, 'buy', lim)
            slip_stats['sell_100'] = slippage(pair, 100, book, 'sell', lim)
            slip_stats['buy_1000'] = slippage(pair, 1000, book, 'buy', lim)
            slip_stats['sell_1000'] = slippage(pair, 1000, book, 'sell', lim)
            slip_stats['buy_10000'] = slippage(pair, 10000, book, 'buy', lim)
            slip_stats['sell_10000'] = slippage(pair, 10000, book, 'sell', lim)

            slippage_dict[pair] = slip_stats

            pair_stats = get_book_stats(pair, book, quote, 2)
            depth_dict[pair] = pair_stats

    fp, sf, live = set_paths(curr_year, ex)
    print(f"{fp}\n{sf}")

    # depth data
    try:
        with open(fp, 'r') as liq_file:
            liq_data = json.load(liq_file)
    except (FileNotFoundError, JSONDecodeError):
        liq_data = {}

    if not live:
        fp = Path(f"{ex.id}_liq_test_{curr_year}.json")
        fp.touch(exist_ok=True)
    liq_data[now] = depth_dict
    with open(fp, 'w') as liq_file:
        json.dump(liq_data, liq_file)

    # slip data
    try:
        with open(sf, 'r') as slip_file:
            all_data = json.load(slip_file)
    except (FileNotFoundError, JSONDecodeError):
        all_data = {}

    if not live:
        sf = Path(f'{ex.id}_slip_test_{curr_year}.json')
        sf.touch(exist_ok=True)
    all_data[now] = slippage_dict
    with open(sf, 'w') as slip_file:
        json.dump(all_data, slip_file)


end = time.perf_counter()

elapsed = round(end - start)

print(f'time taken: {elapsed//60}m {elapsed%60}s')

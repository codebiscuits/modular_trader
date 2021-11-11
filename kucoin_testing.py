import ccxt
import pandas as pd
from pprint import pprint

exchange = ccxt.kucoin ()
exchange.load_markets() # this caches the dict of markets in the exchange object
# so for the rest of the program i can call the cached version exchange.markets

# for any function which gets its data from tickers, if it's going to loop 
# through a lot of pairs, use fetch_tickers with a list of all the pairs, not 
# fetch_ticker for each pair one at a time. MUCH faster.

def kucoin_pairs(quote='USDT'):
    markets = exchange.markets
    pairs = []
    for k, v in markets.items():
        if v.get('quote') == quote and v.get('active') == True:
            pairs.append(k)
    return pairs

# print(kucoin_pairs('BTC'))

def kucoin_list_quotes():
    markets = exchange.markets
    
    quotes = []
    for k, v in markets.items():
        quotes.append(v.get('quote'))
    
    return set(quotes)

def compare_kucoin_markets():
    quotes = kucoin_list_quotes()
    
    for quote in quotes:
        pairs = kucoin_pairs(quote)
        vol_sum = 0
        tickers = exchange.fetch_tickers(pairs)
        for v in tickers.values():
            vol = float(v.get('quoteVolume'))
            vol_sum += vol
        print(f'{quote}: {len(pairs)} pairs, {int(vol_sum)} total volume')
        
# compare_kucoin_markets()

# pprint(exchange.fetch_ticker('BTC/USDT'))

# data = exchange.fetchOHLCV('BTC/USDT', '4h')
# cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
# df = pd.DataFrame(data, columns=cols)
# df.timestamp = df.timestamp * 1000000
# df.timestamp = pd.to_datetime(df.timestamp)
# print(df.head())

def kucoin_spreads(quote):
    pairs = kucoin_pairs(quote)
    tickers = exchange.fetch_tickers(pairs)
    spreads_dict = {}
    for v in tickers.values():
        pair = v.get('symbol')
        bid = v.get('bid')
        ask = v.get('ask')
        if bid and ask:
            mid = (bid + ask) / 2
            diff = ask - bid
            spread = diff / mid
        else:
            spread = None
        spreads_dict[pair] = spread
        if spread > 0.1:
            print(f'{pair}: {spread}')
    
    return spreads_dict
        
spreads = kucoin_spreads('BTC')

# pprint(spreads)
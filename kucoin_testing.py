import ccxt
import pandas as pd
from pprint import pprint

exchange = ccxt.kucoin ()
exchange.load_markets() # this caches the dict of markets in the exchange object
# so for the rest of the program i can call the cached version exchange.markets

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
        for pair in pairs:
            info = exchange.fetch_ticker(pair)
            vol = float(info.get('quoteVolume'))
            vol_sum += vol
        print(f'{quote}: {len(pairs)} pairs, {vol_sum} total volume')
        
# compare_kucoin_markets()

# pprint(exchange.fetch_ticker('BTC/USDT'))

data = exchange.fetchOHLCV('BTC/USDT', '4h')
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(data, columns=cols)
df.timestamp = df.timestamp * 1000000
df.timestamp = pd.to_datetime(df.timestamp)
print(df.head())
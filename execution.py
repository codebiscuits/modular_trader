import keys
import math
import time
import statistics as stats
from pprint import pprint
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size

client = Client(keys.bPkey, keys.bSkey)

def get_price(pair):
    '''returns the first ask price on the orderbook for the pair in question, 
    meant for buy orders'''
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol') == pair:
            usdt_price = float(t.get('askPrice'))
    return usdt_price

def get_spread(pair):
    '''returns the proportional distance between the first bid and ask for the 
    pair in question'''
    
    spreads = []
    mids = []
    for j in range(3):
        tickers = client.get_orderbook_tickers()    
        for t in tickers:
            if t.get('symbol') == pair:
                bid = float(t.get('bidPrice'))
                ask = float(t.get('askPrice'))
                spreads.append(ask - bid)
                mids.append((bid + ask) / 2)
        time.sleep(0.5)
        
    avg_abs_spread = stats.median(spreads)
    avg_mid = stats.median(mids)
    
    if avg_mid > 0:
        return avg_abs_spread / avg_mid
    else:
        return 'na'
    
def get_depth(pair, side):
    '''returns the quantities of the first bid and ask for the pair in question'''
    try:    
        bids = []
        asks = []
        for i in range(3):    
            tickers = client.get_orderbook_tickers()
            for t in tickers:
                if t.get('symbol') == pair:
                    bids.append(float(t.get('bidQty')))
                    asks.append(float(t.get('askQty')))
            time.sleep(2)
        
        avg_bid = stats.median(bids)
        avg_ask = stats.median(asks)
        if side == 'buy':
            return avg_ask
        elif side == 'sell':
            return avg_bid
    except TypeError as e:
        print(e)
        print('Skipping trade - binance returned book depth of None ')
        return 0.0
    
def binance_spreads(quote='USDT'):
    length = len(quote)
    avg_spreads = {}
    
    s_1 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1*length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_1[pair] = spread / mid
    
    time.sleep(1)
    
    s_2 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1*length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_2[pair] = spread / mid
    
    time.sleep(1)
    
    s_3 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1*length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_3[pair] = spread / mid
    
    for k in s_1:
        avg_spreads[k] = stats.median([s_1.get(k), s_2.get(k), s_3.get(k)])
    
    return avg_spreads

def binance_depths(quotes=['USDT', 'BTC']):
    avg_depths = {}

    for quote in quotes:  
        length = len(quote)
        bd_1 = {}
        sd_1 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1*length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_1[pair] = ask
                sd_1[pair] = bid
        
        time.sleep(1)
        
        bd_2 = {}
        sd_2 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1*length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_2[pair] = ask
                sd_2[pair] = bid
        
        time.sleep(1)
        
        bd_3 = {}
        sd_3 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1*length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_3[pair] = ask
                sd_3[pair] = bid
        
        for k in bd_1:
            avg_depths[k] = {'asks': stats.median([bd_1.get(k), bd_2.get(k), bd_3.get(k)]), 
                             'bids': stats.median([sd_1.get(k), sd_2.get(k), sd_3.get(k)])}
    
    return avg_depths

def to_precision(num, base):
    '''a rounding function which takes in the number to be rounded and the 
    step-size which the output must conform to.'''
    
    decimal_places = str(base)[::-1].find('.')
    precise = base * round(num / base)
    mult = 10 ** decimal_places
    return math.floor(precise * mult) / mult

def buy_asset(pair, usdt_size):
    print(f'buying {pair}')
    
    # calculate how much of the asset to buy
    usdt_price = get_price(pair)
    size = usdt_size / usdt_price
    
    # make sure order size has the right number of decimal places
    info = client.get_symbol_info(pair)
    step_size = float(info.get('filters')[2].get('stepSize'))
    order_size = round_step_size(size, step_size)
    print(f'{pair} Buy Order - raw size: {size}, step size: {step_size}, final size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_BUY, 
                                type=ORDER_TYPE_MARKET,
                                quantity=order_size)
    # print('-')
    return order

def sell_asset(pair):
    # print(f'selling {pair}')
    asset = pair[:-4]
    
    # request asset balance from binance
    info = client.get_account()
    bals = info.get('balances')
    for b in bals:
        if b.get('asset') == asset:
            if asset == 'BNB':
                asset_bal = float(b.get('free')) * 0.9 # always keep a bit of bnb
            else:
                asset_bal = float(b.get('free'))
    
    # make sure order size has the right number of decimal places
    info = client.get_symbol_info(pair)
    step_size = float(info.get('filters')[2].get('stepSize'))
    # TODO this rounding function needs to always round down here, i have 
    # subtracted step_size as a temporary fix
    order_size = round_step_size(asset_bal, step_size) - step_size
    print(f'{pair} Sell Order - raw size: {asset_bal}, step size: {step_size}, final size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_SELL, 
                                type=ORDER_TYPE_MARKET,
                                quantity=order_size)
    # print('-')
    return order

def set_stop(pair, price):
    # print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]
    
    info = client.get_symbol_info(pair)
    tick_size = float(info.get('filters')[0].get('tickSize'))
    step_size = float(info.get('filters')[2].get('stepSize'))
    
    info = client.get_account()
    bals = info.get('balances')
    for b in bals:
        if b.get('asset') == asset:
            if asset == 'BNB':
                asset_bal = float(b.get('free')) * 0.9 # always keep a bit of bnb
            else:
                asset_bal = float(b.get('free'))
    
    info = client.get_symbol_info(pair)
    # TODO this rounding function needs to always round down here, i have 
    # subtracted step_size as a temporary fix
    order_size = round_step_size(asset_bal, step_size) - step_size
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 10))
    trigger_price = round_step_size(price, tick_size)
    limit_price = round_step_size(lower_price, tick_size)
    print(f'{pair} Stop Order - trigger: {trigger_price}, limit: {limit_price}, size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_SELL, 
                                type= ORDER_TYPE_STOP_LOSS_LIMIT, 
                                timeInForce=TIME_IN_FORCE_GTC, 
                                stopPrice=trigger_price,
                                quantity=order_size, 
                                price=limit_price)
    # print('-')
    return order

def clear_stop(pair):
    print(f'cancelling {pair} stop')
    orders = client.get_open_orders(symbol=pair)
    ord_id = orders[0].get('orderId')
    result = client.cancel_order(symbol=pair, orderId=ord_id)
    print(result.get('status'))
    # print('-')

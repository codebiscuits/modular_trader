import keys
import math
import time
import statistics as stats
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
    for j in range(3):
        tickers = client.get_orderbook_tickers()    
        for t in tickers:
            if t.get('symbol') == pair:
                bid = float(t.get('bidPrice'))
                ask = float(t.get('askPrice'))
                spreads.append((ask - bid) / ((bid + ask) / 2))
        time.sleep(1)
        
    avg_spread = stats.mean(spreads)
    return avg_spread
    
def get_depth(pair):
    '''returns the quantities of the first bid and ask for the pair in question'''
    
    bids = []
    asks = []
    for i in range(3):    
        tickers = client.get_orderbook_tickers()        
        for t in tickers:
            if t.get('symbol') == pair:
                bids.append(float(t.get('bidQTY')))
                asks.append(float(t.get('askQTY')))
        time.sleep(1)
    
    avg_bid = stats.mean(bids)
    avg_ask = stats.mean(asks)
    return avg_bid, avg_ask
    
def to_precision(num, base):
    '''a rounding function which takes in the number to be rounded and the 
    step-size which the output must conform to.'''
    
    decimal_places = str(base)[::-1].find('.')
    precise = base * round(num / base)
    mult = 10 ** decimal_places
    print(f'to_precision - base: {base}, dec places: {decimal_places}, precise: {precise}, mult: {mult}')
    return math.floor(precise * mult) / mult

def buy_asset(pair, usdt_size):
    print(f'buying {pair}')
    
    # calculate how much of the asset to buy
    usdt_price = get_price(pair)
    size = usdt_size / usdt_price
    
    # make sure order size has the right number of decimal places
    info = client.get_symbol_info(pair)
    print('calculating buy order size precision')
    step_size = float(info.get('filters')[2].get('stepSize'))
    order_size = round_step_size(size, step_size)
    print(f'Buy Order - raw size: {size}, step size: {step_size}, final size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_BUY, 
                                type=ORDER_TYPE_MARKET,
                                quantity=order_size)
    print('-')
    return order

def sell_asset(pair):
    print(f'selling {pair}')
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
    print('calculating sell order size precision')
    order_size = round_step_size(asset_bal, step_size)
    print(f'Sell Order - raw size: {asset_bal}, step size: {step_size}, final size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_SELL, 
                                type=ORDER_TYPE_MARKET,
                                quantity=order_size)
    print('-')
    return order

def set_stop(pair, price):
    print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]
    
    info = client.get_symbol_info(pair)
    tick_size = float(info.get('filters')[0].get('tickSize'))
    step_size = float(info.get('filters')[2].get('stepSize'))
    print(f'from binance - tick size: {tick_size}, step size: {step_size}')
    
    info = client.get_account()
    bals = info.get('balances')
    for b in bals:
        if b.get('asset') == asset:
            if asset == 'BNB':
                asset_bal = float(b.get('free')) * 0.9 # always keep a bit of bnb
            else:
                asset_bal = float(b.get('free'))
    
    info = client.get_symbol_info(pair)
    print('calculating stop order size precision')
    order_size = round_step_size(asset_bal, step_size)
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 10))
    print('calculating stop order trigger price precision')
    trigger_price = round_step_size(price, tick_size)
    print('calculating stop order limit price precision')
    limit_price = round_step_size(lower_price, tick_size)
    print(f'Stop Order - trigger: {trigger_price}, limit: {limit_price}')
    print(f'Stop Order - raw size: {asset_bal}, step size: {step_size}, final size: {order_size}')
    
    order = client.create_order(symbol=pair, 
                                side=SIDE_SELL, 
                                type= ORDER_TYPE_STOP_LOSS_LIMIT, 
                                timeInForce=TIME_IN_FORCE_GTC, 
                                stopPrice=trigger_price,
                                quantity=order_size, 
                                price=limit_price)
    print('-')
    return order

def clear_stop(pair):
    print(f'cancelling {pair} stop')
    orders = client.get_open_orders(symbol=pair)
    ord_id = orders[0].get('orderId')
    result = client.cancel_order(symbol=pair, orderId=ord_id)
    print(result.get('status'))
    print('-')

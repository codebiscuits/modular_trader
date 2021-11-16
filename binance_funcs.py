import keys
from binance.client import Client
from binance.enums import *

client = Client(keys.bPkey, keys.bSkey)

def account_bal():
    info = client.get_account()
    bals = info.get('balances')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
    total = 0
    for b in bals:
        asset = b.get('asset')
        pair = asset + 'USDT'
        quant = float(b.get('free')) + float(b.get('locked'))
        price = price_dict.get(pair)
        if asset == 'USDT':
            total += quant
            continue
        if price == None:
            continue
        if quant > 0:
            value = price * quant
            total += value

    
    return total

def get_size(price, fr, balance, risk):
    trade_risk = risk / fr
    usdt_size = balance / trade_risk
    asset_quantity = usdt_size / price
    
    return asset_quantity, usdt_size


def current_positions(fr):
    total_bal = account_bal()
    threshold_bal = total_bal * fr
    
    info = client.get_account()
    bals = info.get('balances')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
    pos_dict = {}
    for b in bals:        
        asset = b.get('asset')
        if asset == 'USDT':
            continue
        pair = asset + 'USDT'
        price = price_dict.get(pair)
        if price == None:
            continue
        quant = float(b.get('free')) + float(b.get('locked'))
        value = price * quant
        if asset == 'BNB':
            if value >= threshold_bal and value > 15:
                pos_dict[pair] = 1
            else:
                pos_dict[pair] = 0
        else:
            if value >= threshold_bal and value > 10:
                pos_dict[pair] = 1
            else:
                pos_dict[pair] = 0
            
    return pos_dict

def current_sizing(fr):
    total_bal = account_bal()
    threshold_bal = total_bal * fr
    
    info = client.get_account()
    bals = info.get('balances')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
    size_dict = {}
    for b in bals:        
        asset = b.get('asset')
        if asset in ['USDT', 'USDC', 'BUSD']:
            value = float(b.get('free')) + float(b.get('locked'))
        else:
            pair = asset + 'USDT'
            price = price_dict.get(pair)
            if price == None:
                continue
            quant = float(b.get('free')) + float(b.get('locked'))
            value = price * quant
        if value >= threshold_bal:
            size_dict[asset] = round(value / total_bal, 5)
            
    return size_dict

def free_usdt():
    print('runnung free_usdt')
    usdt_bals = client.get_asset_balance(asset='USDT')
    return float(usdt_bals.get('free'))
        
def get_depth(pair):
    print('runnung funcs get_depth')
    usdt_depth = 0
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol') == pair:
            usdt_price = float(t.get('askPrice'))
            asset_qty = float(t.get('askQty'))
            usdt_depth = usdt_price * asset_qty
            
    return usdt_depth

def top_up_bnb(size):
    bnb_bal = client.get_asset_balance(asset='BNB')
    free_bnb = float(bnb_bal.get('free'))
    avg_price = client.get_avg_price(symbol='BNBUSDT')
    price = avg_price.get('price')
    bnb_value = free_bnb * price
    
    usdt_bal = client.get_asset_balance(asset='BNB')
    free_usdt = float(usdt_bal.get('free'))
    if bnb_value < 5 and free_usdt > size:
        print('Topping up BNB')
        order = client.create_order(symbol='BNBUSDT', 
                                    side=SIDE_BUY, 
                                    type=ORDER_TYPE_MARKET,
                                    quantity=size)
    return order
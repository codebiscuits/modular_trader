import keys
from binance.client import Client

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
        if value >= threshold_bal:
            pos_dict[pair] = 1
        else:
            pos_dict[pair] = 0
            
    return pos_dict

def free_usdt():
    usdt_bals = client.get_asset_balance(asset='USDT')
    return float(usdt_bals.get('free'))
        
def get_depth(pair):
    usdt_depth = 0
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol') == pair:
            usdt_price = float(t.get('askPrice'))
            asset_qty = float(t.get('askQty'))
            usdt_depth = usdt_price * asset_qty
            
    return usdt_depth
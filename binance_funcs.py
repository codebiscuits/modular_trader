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
    
    return asset_quantity

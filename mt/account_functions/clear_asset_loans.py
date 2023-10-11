from pprint import pprint
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
from functions import keys

client = Client(keys.bPkey, keys.bSkey)

#%%
info = client.get_margin_account()

for a in info['userAssets']:
    bor = float(a['borrowed'])
    free = float(a['free'])
    if bor and free:
        asset = a['asset']
        if asset == 'USDT':
            continue
        qty = a['free']
        client.repay_margin_loan(asset=asset, amount=qty)
        print(f"{qty} {asset} repaid")

#%%
info = client.get_margin_account()

for a in info['userAssets']:
    bor = float(a['borrowed'])
    free = float(a['free'])
    if bor and not free:
        pair = a['asset'] + 'USDT'
        qty = a['borrowed']
        try:
            buy_order = client.create_margin_order(symbol=pair,
                                               side=be.SIDE_BUY,
                                               type=be.ORDER_TYPE_MARKET,
                                               quantity=qty)
            pprint(buy_order)
            client.repay_margin_loan(asset=asset, amount=qty)
            print(f"{qty} {asset} repaid")
        except bx.BinanceAPIException:
            continue
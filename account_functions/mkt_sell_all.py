import binance_funcs as funcs
import utility_funcs as uf
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from decimal import Decimal
from pprint import pprint
from time import sleep

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)

info = client.get_margin_account()
orders = client.get_open_margin_orders()
# pprint(info)

# clear all stop-loss orders
print('Clearing Stops')
for order in orders:
    oid = order['orderId']
    # print(order)
    client.cancel_margin_order(symbol=order['symbol'], orderId=oid)

print('\nRunning through assets')
for asset in info['userAssets']:
    ast = asset['asset']
    free = Decimal(asset['free'])
    borrowed = Decimal(asset['borrowed'])
    interest = Decimal(asset['interest'])
    liability = borrowed + interest

    if liability > free:
        shortfall = str(liability - free)
        try:
            client.create_margin_order(symbol=ast + 'USDT',
                                       side=be.SIDE_BUY,
                                       type=be.ORDER_TYPE_MARKET,
                                       quantity=shortfall)
            print(f"bought {shortfall} {ast}")
        except BinanceAPIException as e:
            print(f"couldn't buy {ast}")
            print(e)
            continue
    if (ast not in ['BNB', 'USDT']) and free and not borrowed:
        try:
            client.create_margin_order(symbol=ast + 'USDT',
                                       side=be.SIDE_SELL,
                                       type=be.ORDER_TYPE_MARKET,
                                       quantity=asset['free'])
            print(f"sold {asset['free']} {ast}")
        except BinanceAPIException as e:
            print(f"couldn't sell {ast}")
            print(e)
            continue
    if free and liability:
        sleep(0.2)
        client.repay_margin_loan(asset=ast, amount=asset['free'])
        print(f"repaid {free} {ast}")


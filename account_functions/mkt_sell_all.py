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

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)

info = client.get_margin_account()

# pprint(info['userAssets'])

for asset in info['userAssets']:
    if float(asset['free']):
        try:
            client.create_margin_order(symbol=asset['asset']+'USDT', 
                                       side=be.SIDE_SELL, 
                                       type=be.ORDER_TYPE_MARKET, 
                                       quantity=asset['free'])
            print(f"sold {asset['free']} {asset['asset']}")
        except BinanceAPIException as e:
            print(f"couldn't sell {asset['asset']}")
            print(e)
            continue

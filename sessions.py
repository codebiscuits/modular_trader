import binance_funcs as funcs
from pathlib import Path
from datetime import datetime
from pushbullet import Pushbullet
from timers import Timer

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class MARGIN_SESSION:
    max_length = 201
    quote_asset = 'USDT'
    fr_max = 0.0005 # at 0.0025, one agent makes good use of total balance
    max_spread = 0.5
    above_200_ema = set()
    below_200_ema = set()
    
    
    def __init__(self):
        t = Timer('session init')
        t.start()
        self.name = 'agent names here'
        self.bal = funcs.account_bal_M()
        self.live = self.set_live()
        self.market_data = self.mkt_data_path()
        self.ohlc_data = self.ohlc_path()
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        t.stop()
        
        
    def set_live(self):
        y = Timer('set_live')
        y.start()
        live = Path('/home/ubuntu/rpi_2.txt').exists()
        
        if live:
            print('-:-' * 20)
        else:
            print('*** Warning: Not Live ***')
        y.stop()
        return live
    
    def mkt_data_path(self):
        u = Timer('mkt_data_path in session')
        u.start()
        market_data = None
        poss_paths = [Path('/media/coding/market_data'), 
                      Path('/mnt/pi_2/market_data')]
        
        for md_path in poss_paths:
            if md_path.exists():
                market_data = md_path
                break
        if not market_data:
            note = 'none of the paths for market_data are available'
            print(note)
        u.stop()
        return market_data
    
    def ohlc_path(self):
        v = Timer('ohlc_path in session')
        v.start()
        ohlc_data = None
        possible_paths = [Path('/media/coding/ohlc_binance_1h'), 
                          Path('/mnt/pi_2/ohlc_binance_1h')]

        for ohlc_path in possible_paths:
            if ohlc_path.exists():
                ohlc_data = ohlc_path
                break
        if not ohlc_data:
            note = 'none of the paths for ohlc_data are available'
            print(note)
        v.stop()
        return ohlc_data
    
    
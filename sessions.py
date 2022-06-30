import binance_funcs as funcs
from pathlib import Path
from datetime import datetime
from pushbullet import Pushbullet
from timers import Timer
from binance.client import Client
import keys
import time
from typing import Union, List, Tuple, Dict, Set, Optional, Any

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class MARGIN_SESSION:
    max_length = 201
    quote_asset = 'USDT'
    # fr_max = 0.0002
    max_spread = 0.5
    above_200_ema = set()
    below_200_ema = set()
    prices = {}
    symbol_info = {}
    
    
    def __init__(self, timeframe, offset, fr_max):
        t = Timer('session init')
        t.start()
        self.tf = timeframe
        self.offset = offset
        self.fr_max = fr_max # at 0.0025, one agent makes good use of total balance
        self.name = 'agent names here'
        self.bal = self.account_bal_M()
        self.usdt_bal = self.get_usdt_M()
        self.live = self.set_live()
        self.market_data = self.mkt_data_path()
        self.ohlc_data = self.ohlc_path()
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.last_price_update = 0
        self.margin_account_info()
        # self.get_asset_bals
        self.check_margin_lvl()
        t.stop()
        
        
    def account_bal_M(self) -> float:
        '''fetches the total value of the margin account holdings from binance 
        and returns it, denominated in USDT'''
        
        jh = Timer('account_bal_M')
        jh.start()
        info = client.get_margin_account()
        total_net = float(info.get('totalNetAssetOfBtc'))
        btc_price = funcs.get_price('BTCUSDT')
        usdt_total_net = total_net * btc_price
        jh.stop()
        return round(usdt_total_net, 2)
    
    def set_live(self) -> bool:
        '''checks whether the script is running on the raspberry pi or another 
        machine and sets the live flag to True or False accordingly'''
        
        y = Timer('set_live')
        y.start()
        live = Path('/home/ubuntu/rpi_2.txt').exists()
        
        if live:
            print('-:-' * 20)
        else:
            print('*** Warning: Not Live ***')
        y.stop()
        return live
    
    def mkt_data_path(self) -> Path:
        '''automatically sets the absolute path for the market_data folder'''
        
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
    
    def ohlc_path(self) -> Path:
        '''automatically sets the absolute path for the ohlc_data folder'''
        
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
    
    def get_usdt_M(self) -> Dict[str, float]:
        '''fetches the balance information for USDT from binance and returns it 
        as a dictionary of floats'''
        
        print('running get_usdt_M')
        '''checks current usdt balance and returns a dictionary for updating the sizing dict'''
        um = Timer('update_usdt_M')
        um.start()
        
        info = client.get_margin_account()
        bals = info.get('userAssets')
        
        balance = {}
        for bal in bals:
            if bal.get('asset') == 'USDT':
                net = float(bal.get('netAsset'))
                owed = float(bal.get('borrowed'))
                qty = float(bal.get('free'))
        
        value = round(net, 2)
        pct = round(100 * value / self.bal, 5)
        print(f'usdt stats: qty = {bal.get("free")}, owed = {bal.get("borrowed")}, {value = }, {pct = }, {self.bal = }')
        um.stop()
        return {'qty': qty, 'owed': owed, 'value': value, 'pf%': pct}
        
    def update_usdt_M(self, up: float=0.0, down: float=0.0, borrow: float=0.0, repay: float=0.0) -> None:
        '''calculates current usdt balance without fetching from binance, 
        called whenever usdt balance is changed'''
        
        hj = Timer('update_usdt_M')
        hj.start()
        
        qty = (self.usdt_bal.get('qty')) + up
        value = self.usdt_bal.get('value') + up
        qty = self.usdt_bal.get('qty') - down
        value = self.usdt_bal.get('value') - down
        
        owed = self.usdt_bal.get('owed') + borrow
        value = self.usdt_bal.get('value') - borrow
        owed = self.usdt_bal.get('owed') - repay
        value = self.usdt_bal.get('value') + repay
        
        pct = round(100 * value / self.bal, 5)
        
        # print(f'usdt stats: {qty = }, {owed = }, {value = }, {pct = }, {self.bal = }')
        self.usdt_bal = {'qty': float(qty), 'owed': float(owed), 'value': float(value), 'pf%': float(pct)}
        hj.stop()
    
    def margin_account_info(self) -> None:
        '''fetches account info from binance for use by other functions'''
        self.account_info = client.get_margin_account()
    
    def check_margin_lvl(self) -> None:
        '''checks how leveraged the account is and sends a warning push note if 
        leverage is getting too high'''
        
        margin_lvl = float(self.account_info.get('marginLevel'))
        print(f"Margin level: {margin_lvl:.2f}")
        
        if margin_lvl <= 2:
            pb.push_note('*** Warning ***', 'Margin level <= 2, reduce risk')
        elif margin_lvl <= 3:
            pb.push_note('Warning', 'Margin level <= 3, keep an eye on it')
            
    # def get_asset_bals(self) -> None:
    #     '''creates a dictionary of margin asset balances, stored as floats'''
        
    #     bals = self.account_info.get('userAssets')
        
    #     self.bals_dict = {}
        
    #     for bal in bals:
    #         asset = bal.get('asset')
    #         borrowed = float(bal.get('borrowed'))
    #         free = float(bal.get('free'))
    #         interest = float(bal.get('interest'))
    #         locked = float(bal.get('locked'))
    #         net_asset = float(bal.get('netAsset'))
    #         self.bals_dict[asset] = {'borrowed': borrowed, 
    #                                  'free': free, 
    #                                  'interest': interest, 
    #                                  'locked': locked, 
    #                                  'net_asset': net_asset}
    
    

    
    
import binance_funcs as funcs
from pathlib import Path
from datetime import datetime
from pushbullet import Pushbullet
from timers import Timer
from binance.client import Client
import keys
import indicators as ind
from typing import Union, List, Tuple, Dict, Set, Optional, Any
from collections import Counter
import sys

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
        self.fr_max = fr_max  # at 0.0025, one agent makes good use of total balance
        self.name = 'agent names here'
        self.margin_account_info()
        self.bal = self.account_bal_M()
        self.usdt_bal = self.get_usdt_M()
        self.live = self.set_live()
        self.market_data, self.test_mkt_data = self.mkt_data_path()
        self.read_records, self.write_records = self.records_path()
        self.ohlc_data = self.ohlc_path()
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.last_price_update = 0
        self.get_asset_bals()
        self.check_margin_lvl()
        self.algo_order_counts = self.count_algo_orders()
        self.max_loan_amounts = {}
        self.book_data = {}
        self.indicators = set()
        self.counts = []
        t.stop()

    def account_bal_M(self) -> float:
        '''fetches the total value of the margin account holdings from binance 
        and returns it, denominated in USDT'''

        jh = Timer('account_bal_M')
        jh.start()
        info = self.account_info
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

        if not live:
            print('*** Warning: Not Live ***')
        y.stop()
        return live

    def mkt_data_path(self) -> Path:
        '''automatically sets the absolute path for the market_data folder'''

        u = Timer('mkt_data_path in session')
        u.start()

        if self.live: # must be running on rpi
            market_data = Path('/media/coding/market_data')
        elif Path('/mnt/pi_2/market_data').exists(): # must be running on laptop and rpi is accessible
            market_data = Path('/mnt/pi_2/market_data')
        else: # running on laptop and rpi is not available
            market_data = Path('/home/ross/Documents/backtester_2021/market_data')

        test_mkt_data = Path('/home/ross/Documents/backtester_2021/market_data')

        u.stop()
        return market_data, test_mkt_data

    def records_path(self) -> tuple[Path, Path]:
        '''automatically sets the absolute path for the records folder'''

        func_name = sys._getframe().f_code.co_name
        x1 = Timer(f'{func_name}')
        x1.start()

        if self.live:
            read_records = Path(f'/media/coding/records/{self.tf}')
            write_records = Path(f'/media/coding/records/{self.tf}')
        elif Path(f'/mnt/pi_2/records/{self.tf}').exists():
            read_records = Path(f'/mnt/pi_2/records/{self.tf}')
            write_records = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.tf}')
        else:
            read_records = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.tf}')
            write_records = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.tf}')

        read_records.mkdir(parents=True, exist_ok=True)

        x1.stop()
        return read_records, write_records

    def ohlc_path(self) -> Path:
        '''automatically sets the absolute path for the ohlc_data folder'''

        v = Timer('ohlc_path in session')
        v.start()
        ohlc_data = None
        possible_paths = [Path('/media/coding/ohlc_binance_15m'),
                          Path('/mnt/pi_2/ohlc_binance_15m'),
                          Path('/home/ross/Documents/backtester_2021/bin_ohlc_15m'),
                          Path('/home/ross/PycharmProjects/backtester_2021/bin_ohlc_15m')]

        for ohlc_path in possible_paths:
            if ohlc_path.exists():
                ohlc_data = ohlc_path
                break
        if not ohlc_data:
            note = 'none of the paths for ohlc_data are available'
            print(note)
        v.stop()
        return ohlc_data

    def sync_mkt_data(self) -> None:
        func_name = sys._getframe().f_code.co_name
        x2 = Timer(f'{func_name}')
        x2.start()

        for data_file in ['binance_liquidity_history.txt',
                          'binance_depths_history.txt',
                          'binance_spreads_history.txt']:
            real_file = Path(self.market_data / data_file)
            test_file = Path(self.test_mkt_data / data_file)
            test_file.touch(exist_ok=True)

            if real_file.exists():
                with open(real_file, 'r') as file:
                    book_data = file.readlines()
                if book_data:
                    with open(test_file, 'w') as file:
                        file.writelines(book_data)

        x2.stop()

    def get_usdt_M(self) -> Dict[str, float]:
        '''fetches the balance information for USDT from binance and returns it 
        as a dictionary of floats'''

        print('running get_usdt_M')
        '''checks current usdt balance and returns a dictionary for updating the sizing dict'''
        um = Timer('update_usdt_M')
        um.start()

        info = self.account_info
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

    def update_usdt_M(self, up: float = 0.0, down: float = 0.0, borrow: float = 0.0, repay: float = 0.0) -> None:
        '''calculates current usdt balance without fetching from binance, 
        called whenever usdt balance is changed'''

        hj = Timer('update_usdt_M')
        hj.start()

        qty = (self.usdt_bal.get('qty')) + up - down
        value = self.usdt_bal.get('value') + up - down - borrow + repay
        owed = self.usdt_bal.get('owed') + borrow - repay

        pct = round(100 * value / self.bal, 5)

        # print(f'usdt stats: {qty = }, {owed = }, {value = }, {pct = }, {self.bal = }')
        self.usdt_bal = {'qty': float(qty), 'owed': float(owed), 'value': float(value), 'pf%': float(pct)}
        hj.stop()

    def margin_account_info(self) -> None:
        '''fetches account info from binance for use by other functions'''
        func_name = sys._getframe().f_code.co_name
        x3 = Timer(f'{func_name}')
        x3.start()

        self.account_info = client.get_margin_account()

        x3.stop()

    def check_margin_lvl(self) -> None:
        '''checks how leveraged the account is and sends a warning push note if 
        leverage is getting too high'''

        func_name = sys._getframe().f_code.co_name
        x4 = Timer(f'{func_name}')
        x4.start()

        margin_lvl = float(self.account_info.get('marginLevel'))
        print(f"Margin level: {margin_lvl:.2f}")

        if margin_lvl <= 2:
            pb.push_note('*** Warning ***', 'Margin level <= 2, reduce risk')
        elif margin_lvl <= 3:
            pb.push_note('Warning', 'Margin level <= 3, keep an eye on it')

        x4.stop()

    def get_asset_bals(self) -> None:
        '''creates a dictionary of margin asset balances, stored as floats'''

        func_name = sys._getframe().f_code.co_name
        x5 = Timer(f'{func_name}')
        x5.start()

        bals = self.account_info.get('userAssets')

        self.bals_dict = {}

        for bal in bals:
            asset = bal.get('asset')
            borrowed = float(bal.get('borrowed'))
            free = float(bal.get('free'))
            interest = float(bal.get('interest'))
            locked = float(bal.get('locked'))
            net_asset = float(bal.get('netAsset'))
            self.bals_dict[asset] = {'borrowed': borrowed,
                                     'free': free,
                                     'interest': interest,
                                     'locked': locked,
                                     'net_asset': net_asset}

        x5.stop()

    def get_pair_info(self, pair):
        func_name = sys._getframe().f_code.co_name
        x6 = Timer(f'{func_name}')
        x6.start()

        info = self.symbol_info.get(pair)
        if not info:
            info = client.get_symbol_info(pair)
            self.symbol_info[pair] = info

        x6.stop()
        self.counts.append('pair')

        return info

    def count_algo_orders(self):
        func_name = sys._getframe().f_code.co_name
        x7 = Timer(f'{func_name}')
        x7.start()

        orders = client.get_open_margin_orders()

        algo_types = ['STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']
        order_symbols = [d['symbol'] for d in orders if d['type'] in algo_types]

        x7.stop()

        return Counter(order_symbols)

    def algo_limit_reached(self, pair: str) -> bool:
        """compares the number of 'algo orders' (stop-loss and take-profit orders) currently open to the maximum allowed.
        returns True if more orders are allowed to be set, and False if the limit has been reached"""
        func_name = sys._getframe().f_code.co_name
        x8 = Timer(f'{func_name}')
        x8.start()

        count = self.algo_order_counts.get(pair, 0)

        symbol_info = self.get_pair_info(pair)
        symbol_filters = symbol_info['filters']
        for i in symbol_filters:
            if i['filterType'] == 'MAX_NUM_ALGO_ORDERS':
                limit = i['maxNumAlgoOrders']

        x8.stop()

        return limit == count

    def max_loan(self, asset):
        func_name = sys._getframe().f_code.co_name
        x9 = Timer(f'{func_name}')
        x9.start()

        if not self.max_loan_amounts.get(asset):
            self.max_loan_amounts[asset] = client.get_max_margin_loan(asset='SUSHI')

        x9.stop()

        return self.max_loan_amounts[asset]

    def get_book_data(self, pair):

        func_name = sys._getframe().f_code.co_name
        x10 = Timer(f'{func_name}')
        x10.start()

        if not self.book_data.get(pair):
            self.book_data[pair] = client.get_order_book(symbol=pair)

        x10.stop()
        self.counts.append('depth')

        return self.book_data[pair]

    def compute_indicators(self, df):
        '''takes the set of required indicators and the dataframe and applies the
        indicator functions as necessary'''

        ci = Timer('compute_indicators')
        ci.start()

        for i in self.indicators:
            vals = i.split('-')
            if vals[0] == 'ema':
                df[f"ema-{vals[1]}"] = df.close.ewm(int(vals[1])).mean()
            elif vals[0] == 'hma':
                df[f"hma-{vals[1]}"] = ind.hma(df.close, int(vals[1]))
            elif vals[0] == 'atr':
                ind.atr_bands(df, int(vals[1]), float(vals[2]))
            elif vals[0] == 'st':
                df = ind.supertrend(df, int(vals[1]), float(vals[2]))

        return df

        ci.stop()

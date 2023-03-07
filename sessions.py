import binance_funcs as funcs
from pathlib import Path
from datetime import datetime
from pushbullet import Pushbullet
from timers import Timer
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
import keys
import indicators as ind
from typing import Union, List, Tuple, Dict, Set, Optional, Any
from collections import Counter
import sys
import time
from decimal import Decimal
import statistics as stats
import pandas as pd
from pycoingecko import CoinGeckoAPI

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
cg = CoinGeckoAPI()


class TradingSession():
    min_length = 10000
    quote_asset = 'USDT'
    max_spread = 0.5
    ohlc_tf = '5m'
    above_200_ema = set()
    below_200_ema = set()
    prices = {}
    symbol_info = {}

    def __init__(self, fr_max):
        t = Timer('session init')
        t.start()

        # configure settings
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.client = Client(keys.bPkey, keys.bSkey, testnet=False)
        self.last_price_update = 0
        self.fr_max = fr_max  # at 0.0025, one agent makes good use of total balance
        self.name = 'agent names here'
        self.counts = []
        self.weights_count = []
        self.last_price_update = 0
        self.live = self.set_live()

        # get data from exchange
        self.get_cg_symbols()
        abc = Timer('all binance calls')
        abc.start()
        self.info = self.client.get_exchange_info()
        self.check_rate_limits()
        self.track_weights(10) # this should be before self.info, but would only work after check_rate_limits
        self.track_weights(10)
        self.acct = self.client.get_account()
        self.track_weights(10)
        self.m_acct = self.client.get_margin_account()
        self.track_weights(40)
        self.spot_orders = self.client.get_open_orders()
        self.track_weights(1)
        self.track_weights(len(self.client.get_margin_all_pairs()))  # weighting for this call = number of pairs on exchange
        self.margin_orders = self.client.get_open_margin_orders()
        self.track_weights(2)
        self.obt = self.client.get_orderbook_tickers()
        self.binance_spreads()
        abc.stop()

        # filter and organise data
        self.get_pairs_info()
        self.update_prices()
        self.get_asset_bals_s()
        self.get_asset_bals_m()
        self.check_open_spot_orders()
        self.check_open_margin_orders()
        self.check_fees()
        self.check_margin_lvl()
        self.top_up_bnb_s(15)
        self.top_up_bnb_m(15)
        self.spot_bal = self.account_bal_s()
        print(f"Spot balance: {self.spot_bal}")
        self.spot_usdt_bal = self.get_usdt_s()
        self.margin_bal = self.account_bal_m()
        print(f"Margin balance: {self.margin_bal}")
        self.margin_usdt_bal = self.get_usdt_m()

        # load local data and configure settings
        self.market_data_read, self.market_data_write = self.mkt_data_path()
        self.read_records, self.write_records = self.records_path()
        self.ohlc_data = self.ohlc_path()
        self.algo_order_counts = self.count_algo_orders()
        self.max_loan_amounts = {}
        self.book_data = {}
        self.indicators = {'ema-200', 'ema-100', 'ema-50', 'ema-20', 'ema_ratio-200', 'ema_ratio-20', 'vol_delta',
                           'vol_delta_div', 'roc_1d', 'roc_1w', 'roc_1m', 'vwma-24'}
        t.stop()

    def track_weights(self, weight):
        '''keeps track of total api request weight to make sure i don't go over the limit

        works by adding each new call weight with a timestamp to the end of a list, then counting back along the list up to the weight
        limit, and checking if enough time has passed over the last {weight limit} worth of requests. also discards list
        items beyond the time window to stop it getting too long'''

        tw = Timer('track_weights')
        tw.start()

        now = datetime.now().timestamp()
        new_weight = (now, weight)
        self.weights_count.append(new_weight)
        window = self.request_weight[0]
        weight_limit = self.request_weight[1]
        raw_window = self.raw_requests[0]
        raw_limit = self.raw_requests[1]
        # print(f"{window = } {weight_limit = } {raw_window = } {raw_limit = }")

        total = 0
        flag = 1
        rolling_weight = 0
        rolling_time = 0
        for n, w in list(enumerate(self.weights_count))[::-1]:
            total += w[1]
            timespan = now - w[0]

            request_limit_exceeded = total > weight_limit
            within_window = timespan < window

            if timespan >= 60 and not rolling_weight:
                rolling_weight = sum([w[1] for w in self.weights_count[n:]])
                rolling_time = round(now - self.weights_count[n][0])
            if request_limit_exceeded and within_window:
                flag = 0
                print(f"request weight limit: {weight_limit} per {window}s. currently: {total} in the last {timespan:.1f}s")
                print(f"track_weights needs {window - timespan:.1f}s of sleep")
                time.sleep(window - timespan)
            if timespan > max(window, raw_window):
                flag = 0
                # print(self.weights_count[0])
                # print(self.weights_count[n])
                self.weights_count = self.weights_count[n:]
                # print(self.weights_count[0])
                break

        raw_limit_exceeded = len(self.weights_count) > raw_limit
        within_raw_window = timespan < raw_window

        if raw_limit_exceeded and within_raw_window:
            flag = 0
            print(
                f"raw request limit: {raw_limit} per {raw_window}s. currently: {total} in the last {timespan:.1f}s")
            print(f"track_weights needs {raw_window - timespan:.1f}s of sleep")
            time.sleep(raw_window - timespan)

        # if flag and rolling_weight:
        #     print(f"Current request weight: {rolling_weight} over {rolling_time}s, raw count: {len(self.weights_count)}")
        # elif flag:
        #     pre_roll_w = sum([w[1] for w in self.weights_count[n:]])
        #     pre_roll_t = round(now - self.weights_count[n][0])
        #     print(f"Current request weight: {pre_roll_w} over {pre_roll_t}s, raw count: {len(self.weights_count)}")

        tw.stop()

    def check_rate_limits(self):
        """parses the rate limits from binance and warns me if these limits change

        if the limits do change, i need to update the value of 'old_limits' by calling client.get_exchange_info() in
        python console and copying the new value across from the variable explorer, and check that the way i'm
        calculating the session attributes still works with the new values"""

        limits = self.info['rateLimits']

        old_limits = [{'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 1200},
                      {'rateLimitType': 'ORDERS', 'interval': 'SECOND', 'intervalNum': 10, 'limit': 50},
                      {'rateLimitType': 'ORDERS', 'interval': 'DAY', 'intervalNum': 1, 'limit': 160000},
                      {'rateLimitType': 'RAW_REQUESTS', 'interval': 'MINUTE', 'intervalNum': 5, 'limit': 6100}]

        if limits != old_limits:
            note = 'binance rate limits have changed, check and adjust session definition'
            print('\n****************\n\n', note, '\n\n****************\n')
            pb.push_note('*** WARNING ***', note)

        for limit in limits:
            if limit['interval'] == 'MINUTE':
                seconds = 60 * limit['intervalNum']
            elif limit['interval'] == 'SECOND':
                seconds = limit['intervalNum']
            else:
                continue  # not interested in the daily order limit

            if limit['rateLimitType'] == 'REQUEST_WEIGHT':
                self.request_weight = (seconds, limit['limit'])
            elif limit['rateLimitType'] == 'RAW_REQUESTS':
                self.raw_requests = (seconds, limit['limit'])
            elif limit['rateLimitType'] == 'ORDERS':
                self.max_orders_sec = (seconds, limit['limit'])

    def get_cg_symbols(self):
        all_coins = cg.get_coins_list()
        self.cg_symbols = {x['symbol'].upper(): x['id'] for x in all_coins}

    def binance_spreads(self, quote: str = 'USDT') -> None:
        """returns a dictionary with pairs as keys and current average spread as values"""

        sx = Timer('binance_spreads')
        sx.start()

        def parse_ob_tickers(tickers):
            s = {}
            for t in tickers:
                if t.get('symbol')[-1 * length:] == quote:
                    pair = t.get('symbol')
                    bid = float(t.get('bidPrice'))
                    ask = float(t.get('askPrice'))
                    if bid and ask:
                        spread = ask - bid
                        mid = (ask + bid) / 2
                        s[pair] = spread / mid

            return s

        length = len(quote)
        avg_spreads = {}

        abc = Timer('all binance calls')
        abc.start()
        self.track_weights(2)
        s_1 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)

        self.track_weights(2)
        s_2 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)

        self.track_weights(2)
        s_3 = parse_ob_tickers(self.client.get_orderbook_tickers())
        time.sleep(1)
        abc.stop()

        for k in s_1:
            avg_spreads[k] = stats.median([s_1.get(k), s_2.get(k), s_3.get(k)])
        sx.stop()

        self.spreads = avg_spreads

    def update_prices(self) -> None:
        """fetches current prices for all pairs on binance. much faster than get_price"""
        up = Timer('update_prices')
        up.start()
        now = time.perf_counter()
        last = self.last_price_update
        if now - last > 60:
            print('get_all_tickers')
            self.track_weights(2)
            abc = Timer('all binance calls')
            abc.start()
            prices = {x['symbol']: float(x['price']) for x in self.client.get_all_tickers()}
            abc.stop()
            self.last_price_update = time.perf_counter()

            for p in self.pairs_data:
                self.pairs_data[p]['price'] = prices[p]
        up.stop()

    def get_pairs_info(self):

        not_pairs = ['GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT',
                     'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT',
                     'USTUSDT']

        self.pairs_data = {}

        symbols = self.info['symbols']
        for sym in symbols:
            pair = sym.get('symbol')
            right_quote = sym.get('quoteAsset') == self.quote_asset
            right_market = 'SPOT' in sym.get('permissions')
            allowed = pair not in not_pairs

            if right_quote and right_market and allowed:
                base_asset = sym.get('baseAsset')

                margin = 'MARGIN' in sym.get('permissions')
                oco_allowed = sym['ocoAllowed']
                quote_order_qty_allowed = sym['quoteOrderQtyMarketAllowed']

                if sym.get('status') != 'TRADING':
                    continue

                for filter in sym['filters']:
                    if filter['filterType'] == 'PRICE_FILTER':
                        tick_size = Decimal(filter['tickSize'])
                    elif filter['filterType'] == 'LOT_SIZE':
                        step_size = Decimal(filter['stepSize'])
                    elif filter['filterType'] == 'MIN_NOTIONAL':
                        min_size = filter['minNotional']
                    elif filter['filterType'] == 'MAX_NUM_ALGO_ORDERS':
                        max_algo = filter['maxNumAlgoOrders']

                spread = self.spreads[pair]

                self.pairs_data[sym.get('symbol')] = dict(
                    base_asset=base_asset,
                    # cg_symbol=cg_symbol,
                    spread=spread,
                    margin_allowed=margin,
                    price_tick_size=tick_size,
                    lot_step_size=step_size,
                    min_notional=min_size,
                    max_algo_orders=max_algo,
                    oco_allowed=oco_allowed,
                    qoq_allowed=quote_order_qty_allowed,
                    spot_orders=[],
                    margin_orders=[]
                )

    def check_fees(self):
        spot_bnb = self.spot_bals['BNB']
        margin_bnb = self.margin_bals['BNB']['free']
        maker = Decimal(self.acct['commissionRates']['maker'])
        taker = Decimal(self.acct['commissionRates']['taker'])

        self.fees = dict(
            spot_maker=maker * Decimal(0.75) if spot_bnb else maker,
            spot_taker=taker * Decimal(0.75) if spot_bnb else taker,
            margin_maker=maker * Decimal(0.75) if margin_bnb else maker,
            margin_taker=taker * Decimal(0.75) if margin_bnb else taker,
        )

    def set_live(self) -> bool:
        '''checks whether the script is running on the raspberry pi or another 
        machine and sets the live flag to True or False accordingly'''

        y = Timer('set_live')
        y.start()
        live = Path('/pi_downstairs.txt').exists()

        if not live:
            print('*** Warning: Not Live ***')
        y.stop()
        return live

    def mkt_data_path(self) -> Path:
        '''automatically sets the absolute path for the market_data folder'''

        u = Timer('mkt_data_path in session')
        u.start()

        if self.live:  # must be running on rpi
            market_data_read = Path('/home/pi/coding/modular_trader/market_data')
            market_data_write = Path('/home/pi/coding/modular_trader/market_data')
        elif Path('/mnt/pi_d/modular_trader/market_data').exists():  # must be running on laptop and rpi is accessible
            market_data_read = Path('/mnt/pi_d/modular_trader/market_data')
            market_data_write = Path('/home/ross/Documents/backtester_2021/market_data')
        else:  # running on laptop and rpi is not available
            market_data_read = Path('/home/ross/Documents/backtester_2021/market_data')
            market_data_write = Path('/home/ross/Documents/backtester_2021/market_data')

        market_data_write.mkdir(exist_ok=True)

        u.stop()
        return market_data_read, market_data_write

    def records_path(self) -> Tuple[Path, Path]:
        '''automatically sets the absolute path for the records folder'''

        func_name = sys._getframe().f_code.co_name
        x1 = Timer(f'{func_name}')
        x1.start()

        if self.live:
            read_records = Path(f'/home/pi/coding/modular_trader/records')
            write_records = Path(f'/home/pi/coding/modular_trader/records')
        elif Path(f'/mnt/pi_d/modular_trader/records').exists():
            read_records = Path(f'/mnt/pi_d/modular_trader/records')
            write_records = Path(f'/home/ross/Documents/backtester_2021/records')
        else:
            read_records = Path(f'/home/ross/Documents/backtester_2021/records')
            write_records = Path(f'/home/ross/Documents/backtester_2021/records')

        # print(f"{read_records = }")
        # print(f"{write_records = }")

        write_records.mkdir(parents=True, exist_ok=True)

        x1.stop()
        return read_records, write_records

    def set_ohlc_tf(self, tf):
        self.ohlc_tf = tf
        self.ohlc_data = self.ohlc_path()

    def ohlc_path(self) -> Path:
        '''automatically sets the absolute path for the ohlc_data folder'''

        v = Timer('ohlc_path in session')
        v.start()
        ohlc_data = None
        possible_paths = [Path(f'/home/pi/coding/modular_trader/bin_ohlc_{self.ohlc_tf}'),
                          Path(f'/home/ross/Documents/backtester_2021/bin_ohlc_{self.ohlc_tf}'),
                          Path(f'/home/ross/PycharmProjects/backtester_2021/bin_ohlc_{self.ohlc_tf}')]

        for ohlc_path in possible_paths:
            if ohlc_path.exists():
                ohlc_data = ohlc_path
                break
        if not ohlc_data:
            note = 'none of the paths for ohlc_data are available'
            print(note)
        v.stop()
        return ohlc_data

    def get_pair_info(self, pair):
        """tries to find details about the pair from local records. if no local data exists, fetches it from exchange"""

        func_name = sys._getframe().f_code.co_name
        x6 = Timer(f'{func_name}')
        x6.start()

        info = self.symbol_info.get(pair)
        if not info:
            print('get_exchange_info')
            self.track_weights(10)
            abc = Timer('all binance calls')
            abc.start()
            self.info = self.client.get_exchange_info()
            abc.stop()
            self.counts.append('get_exchange_info')
            info = self.info['symbols'][pair]

        x6.stop()

        return info

    def count_algo_orders(self):
        func_name = sys._getframe().f_code.co_name
        x7 = Timer(f'{func_name}')
        x7.start()

        algo_types = ['STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']
        order_symbols = [d['symbol'] for d in self.margin_orders if d['type'] in algo_types]

        x7.stop()

        return Counter(order_symbols)

    def algo_limit_reached(self, pair: str) -> bool:
        """compares the number of 'algo orders' (stop-loss and take-profit orders) currently open to the maximum allowed.
        returns True if more orders are allowed to be set, and False if the limit has been reached"""
        func_name = sys._getframe().f_code.co_name
        x8 = Timer(f'{func_name}')
        x8.start()

        count = self.algo_order_counts.get(pair, 0)

        limit = self.pairs_data[pair]['max_algo_orders']

        x8.stop()

        return limit == count

    def get_book_data(self, pair):

        func_name = sys._getframe().f_code.co_name
        x10 = Timer(f'{func_name}')
        x10.start()

        if not self.book_data.get(pair):
            self.track_weights(5)
            abc = Timer('all binance calls')
            abc.start()
            self.book_data[pair] = self.client.get_order_book(symbol=pair, limit=500)
            abc.stop()
            self.counts.append('get_order_book')

        x10.stop()

        return self.book_data[pair]

    def store_ohlc(self, df, pair, timeframes):
        """Calculates the minimum length of ohlc history needed to be stored in memory for the trading system to run.
        calc_min is the number of 5min periods needed to calculate all the necessary indicators at the longest timeframe
        being used in the current session
        bench refers to the number of periods needed to calculate the longest market benchmark statistics (1 mo roc)"""

        lengths = {'1w': 2016, '1d': 288, '12h': 144, '6h': 72, '4h': 48, '1h': 12}
        calc_min = (self.min_length + 1) * lengths[timeframes[-1][0]]
        bench = 8928 + 1
        enough = max(bench, calc_min)

        df = df.tail(enough).reset_index(drop=True)
        self.pairs_data[pair]['ohlc_5m'] = df
        # print(f"{pair} ohlc stored in session")

    def compute_indicators(self, df: pd.DataFrame, tf: str) -> dict:
        '''takes the set of required indicators and the dataframe and applies the
        indicator functions as necessary'''

        ci = Timer('compute_indicators')
        ci.start()

        for i in self.indicators:
            vals = i.split('-')
            if vals[0] == 'ema':
                df[f"ema_{vals[1]}"] = df.close.ewm(int(vals[1])).mean()
            elif vals[0] == 'hma':
                df[f"hma_{vals[1]}"] = ind.hma(df.close, int(vals[1]))
            elif vals[0] == 'ema_ratio':
                df[f"ema_{vals[1]}_ratio"] = ind.ema_ratio(df.close, vals[1])
            elif vals[0] == 'vol_delta':
                df['vol_delta'] = ind.vol_delta(df)
            elif vals[0] == 'vol_delta_div':
                df['vol_delta_div'] = ind.vol_delta_div(df)
            elif vals[0] == 'atr':
                df = ind.atr_bands(df, int(vals[1]), float(vals[2]))
            elif vals[0] == 'st':
                df = ind.supertrend(df, int(vals[1]), float(vals[2]))
            elif vals[0] == 'atsz':
                df = ind.ats_z(df, int(vals[1]))
            elif vals[0] == 'stoch_rsi':
                df['stoch_rsi'] = ind.stoch_rsi(df.close, int(vals[1]), int(vals[2]))
            elif vals[0] == 'inside':
                df = ind.inside_bars(df)
            elif vals[0] == 'doji':
                df = ind.doji(df)
            elif vals[0] == 'engulfing':
                df = ind.engulfing(df, int(vals[1]))
            elif vals[0] == 'bbb':
                df = ind.bull_bear_bar(df)
            elif vals[0] == 'roc_1d':
                df['roc_1d'] = ind.roc_1d(df.close, tf)
            elif vals[0] == 'roc_1w':
                df['roc_1d'] = ind.roc_1w(df.close, tf)
            elif vals[0] == 'roc_1m':
                df['roc_1d'] = ind.roc_1m(df.close, tf)
            elif vals[0] == 'vwma':
                df['vwma'] = ind.vwma(df, vals[1])

        return df

        ci.stop()

    # Spot Specific Methods
    def top_up_bnb_s(self, usdt_size: int) -> dict:
        """checks net BNB balance and interest owed, if net is below the threshold,
        buys BNB then repays any interest"""

        gh = Timer('top_up_bnb_M')
        gh.start()

        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # check balances
        free_bnb = self.spot_bals['BNB']['free']
        free_usdt = self.spot_bals['USDT']['free']

        # calculate value
        bnb_value = free_bnb * self.pairs_data['BNBUSDT']['price']

        # top up if needed
        if bnb_value < 10:
            if free_usdt > usdt_size:
                pb.push_note(now, 'Topping up spot BNB')
                # uid weight of 6
                order = self.client.create_order(
                    symbol='BNBUSDT',
                    side=be.SIDE_BUY,
                    type=be.ORDER_TYPE_MARKET,
                    quoteOrderQty=usdt_size)
                # pprint(order)
            else:
                pb.push_note(now, 'Warning - Spot BNB balance low and not enough USDT to top up')
        else:
            order = None

        gh.stop()
        return order

    def check_open_spot_orders(self):
        if self.spot_orders:
            for order in self.spot_orders:
                self.pairs_data[order['symbol']]['spot_orders'].append(order)

    def account_bal_s(self):

        total = 0.0
        for asset, bals in self.spot_bals.items():
            pair = f'{asset}USDT'
            qty = bals['free'] + bals['locked']
            if qty and (asset == 'USDT'):
                total += qty
            elif qty:
                price = self.pairs_data.get(pair, {'price': 0})['price']
                total += (qty * price)

        return round(total, 2)

    def get_usdt_s(self):
        """gets the usdt balance"""

        bal = self.spot_bals['USDT']
        free = bal['free']
        locked = bal['locked']

        value = round((free + locked), 2)
        pct = round((100 * value / self.spot_bal), 5)
        print(f'spot usdt stats: qty = {bal.get("free")}, {value = }, {pct = }, {self.spot_bal = }')

        return {'qty': free, 'value': value, 'pf%': pct}

    def update_usdt_s(self, up: float = 0.0, down: float = 0.0) -> None:
        """updates the local record of spot usdt holdings to avoid having to call the api"""

        uus = Timer('update_usdt_s')
        uus.start()

        qty = (self.spot_bals['USDT'].get('qty')) + up - down
        value = round(self.spot_bals['USDT'].get('value') + up - down, 2)

        pct = round(100 * value / self.bal, 5)

        self.spot_bals['USDT'] = {'qty': float(qty), 'value': float(value), 'pf%': float(pct)}

        uus.stop()
        return

    def get_asset_bals_s(self):
        '''creates a dictionary of spot asset balances, stored as floats'''

        func_name = sys._getframe().f_code.co_name
        gab = Timer(f'{func_name}')
        gab.start()

        bals = self.acct['balances']

        self.spot_bals = {}

        for bal in bals:
            asset = bal.get('asset')
            free = float(bal.get('free'))
            locked = float(bal.get('locked'))

            self.spot_bals[asset] = {'free': free, 'locked': locked}

        gab.stop()

    # Margin Specific Methods
    def top_up_bnb_m(self, usdt_size: int) -> dict:
        """checks net BNB balance and interest owed, if net is below the threshold,
        buys BNB then repays any interest"""

        gh = Timer('top_up_bnb_M')
        gh.start()

        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # check balances
        free_bnb = self.margin_bals['BNB']['free']
        interest = self.margin_bals['BNB']['interest']
        free_usdt = self.margin_bals['USDT']['free']
        net_bnb = free_bnb - interest

        if interest:
            print(f'BNB interest: {interest}')

        # calculate value
        bnb_value = net_bnb * self.pairs_data['BNBUSDT']['price']

        # top up if needed
        if bnb_value < 10:
            if free_usdt > usdt_size:
                pb.push_note(now, 'Topping up margin BNB')
                # uid weight of 6
                order = self.client.create_margin_order(
                    symbol='BNBUSDT',
                    side=be.SIDE_BUY,
                    type=be.ORDER_TYPE_MARKET,
                    quoteOrderQty=usdt_size)
                # pprint(order)
            else:
                pb.push_note(now, 'Warning - Margin BNB balance low and not enough USDT to top up')
        else:
            order = None

        # repay interest
        if float(interest):
            # uid weight of 3000. not sure how to keep track of this
            self.client.repay_margin_loan(asset='BNB', amount=interest)
        gh.stop()
        return order

    def check_open_margin_orders(self):
        # this may obviate the need for count_algo_orders if i can simply use the length of the lists in
        # pairs_data[pair]['spot_orders'] plus pairs_data[pair]['margin_orders'] as a count
        if self.margin_orders:
            for order in self.margin_orders:
                self.pairs_data[order['symbol']]['margin_orders'].append(order)

    def account_bal_m(self) -> float:
        '''fetches the total value of the margin account holdings from binance
        and returns it, denominated in USDT'''

        jh = Timer('account_bal_M')
        jh.start()
        info = self.m_acct
        total_net = float(info.get('totalNetAssetOfBtc'))
        btc_price = self.pairs_data['BTCUSDT']['price']
        usdt_total_net = total_net * btc_price
        jh.stop()
        return round(usdt_total_net, 2)

    def get_usdt_m(self) -> Dict[str, float]:
        '''fetches the balance information for USDT from binance and returns it
        as a dictionary of floats'''

        print('running get_usdt_M')
        '''checks current usdt balance and returns a dictionary for updating the sizing dict'''
        um = Timer('update_usdt_M')
        um.start()

        bal = self.margin_bals['USDT']
        net = float(bal.get('net_asset'))
        owed = float(bal.get('borrowed'))
        qty = float(bal.get('free'))

        value = round(net, 2)
        pct = round(100 * value / self.margin_bal, 5)
        print(f'margin usdt stats: qty = {bal.get("free")}, owed = {bal.get("borrowed")}, {value = }, {pct = }, {self.margin_bal = }')
        um.stop()
        return {'qty': qty, 'owed': owed, 'value': value, 'pf%': pct}

    def update_usdt_m(self, up: float = 0.0, down: float = 0.0, borrow: float = 0.0, repay: float = 0.0) -> None:
        '''calculates current usdt balance without fetching from binance,
        called whenever usdt balance is changed'''

        hj = Timer('update_usdt_M')
        hj.start()

        qty = (self.margin_bals['USDT'].get('qty')) + up - down
        value = self.margin_bals['USDT'].get('value') + up - down - borrow + repay
        owed = self.margin_bals['USDT'].get('owed') + borrow - repay

        pct = round(100 * value / self.bal, 5)

        # print(f'usdt stats: {qty = }, {owed = }, {value = }, {pct = }, {self.bal = }')
        self.margin_bals['USDT'] = {'qty': float(qty), 'owed': float(owed), 'value': float(value), 'pf%': float(pct)}
        hj.stop()

    def check_margin_lvl(self) -> None:
        '''checks how leveraged the account is and sends a warning push note if
        leverage is getting too high'''

        func_name = sys._getframe().f_code.co_name
        x4 = Timer(f'{func_name}')
        x4.start()

        margin_lvl = float(self.m_acct.get('marginLevel'))
        print(f"Margin level: {margin_lvl:.2f}")

        if margin_lvl <= 2:
            pb.push_note('*** Warning ***', 'Margin level <= 2, reduce risk')
        elif margin_lvl <= 3:
            pb.push_note('Warning', 'Margin level <= 3, keep an eye on it')

        x4.stop()

    def get_asset_bals_m(self) -> None:
        '''creates a dictionary of margin asset balances, stored as floats'''

        func_name = sys._getframe().f_code.co_name
        x5 = Timer(f'{func_name}')
        x5.start()

        bals = self.m_acct.get('userAssets')

        self.margin_bals = {}

        for bal in bals:
            asset = bal.get('asset')
            borrowed = float(bal.get('borrowed'))
            free = float(bal.get('free'))
            interest = float(bal.get('interest'))
            locked = float(bal.get('locked'))
            net_asset = float(bal.get('netAsset'))

            self.margin_bals[asset] = {'borrowed': borrowed,
                                       'free': free,
                                       'interest': interest,
                                       'locked': locked,
                                       'net_asset': net_asset}

        x5.stop()

    def max_loan(self, asset):
        func_name = sys._getframe().f_code.co_name
        x9 = Timer(f'{func_name}')
        x9.start()

        if not self.max_loan_amounts.get(asset):
            self.track_weights(50)
            abc = Timer('all binance calls')
            abc.start()
            self.max_loan_amounts[asset] = self.client.get_max_margin_loan(asset=asset)
            abc.stop()
            self.counts.append('get_max_margin_loan')

        x9.stop()

        return self.max_loan_amounts[asset]

    # Non-live Methods
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


class LightSession(TradingSession):
    def __init__(self):
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.client = Client(keys.bPkey, keys.bSkey)
        self.live = self.set_live()
        self.weights_count = []

        # get data from exchange
        self.get_cg_symbols()
        self.info = self.client.get_exchange_info()
        self.check_rate_limits()
        self.track_weights(10) # this should be before self.info, but would only work after check_rate_limits
        self.track_weights(2)
        self.obt = self.client.get_orderbook_tickers()
        self.binance_spreads()

        # filter and organise data
        self.get_pairs_info()

        # load local data and configure settings
        self.market_data_read, self.market_data_write = self.mkt_data_path()
        self.read_records, self.write_records = self.records_path()
        self.ohlc_data = self.ohlc_path()
        self.indicators = None # to stop any default indicators being calculated by inheritance


class CheckRecordsSession(TradingSession):
    def __init__(self):
        self.ohlc_length = 0
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.client = Client(keys.bPkey, keys.bSkey)
        self.live = self.set_live()
        self.weights_count = []

        # load local data and configure settings
        self.market_data_read, self.market_data_write = self.mkt_data_path()
        self.read_records, self.write_records = self.records_path()
        self.ohlc_data = self.ohlc_path()
        self.indicators = None
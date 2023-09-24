from pathlib import Path
from datetime import datetime, timezone
from resources.timers import Timer
from resources.loggers import create_logger
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
from resources import keys, features, utility_funcs as uf
import ml_funcs as mlf
from typing import Tuple, Dict
from collections import Counter
import sys
import time
from decimal import Decimal
import statistics as stats
import pandas as pd
import json

# pb = uf.init_pb()
logger = create_logger('  sessions   ')


def get_timeframes() -> list[tuple]:
    hour = datetime.now(timezone.utc).hour
    # hour = 0 # for testing all timeframes
    d = {1: ('1h', None), 4: ('4h', None), 12: ('12h', None), 24: ('1d', None)}

    timeframes = [d[tf] for tf in d if hour % tf == 0]
    # timeframes = [('1d', None), ('12h', None), ('4h', None), ('1h', None)]

    return timeframes


def set_live() -> bool:
    """checks whether the script is running on the raspberry pi or another
    machine and sets the live flag to True or False accordingly"""

    y = Timer('set_live')
    y.start()
    live = Path('/pi_downstairs.txt').exists() or Path('/pi_2.txt').exists()

    if live:
        logger.debug('*** Warning: Live ***')
    else:
        logger.info('*** Warning: Not Live ***')
    y.stop()
    return live


class TradingSession:
    min_length = 10000
    max_length = 0
    quote_asset = 'USDT'
    max_spread = 0.5
    ohlc_tf = '5m'
    above_200_ema = set()
    below_200_ema = set()
    prices = {}
    symbol_info = {}
    counts = []
    weights_count = []
    all_weights = []
    market_bias = {}
    request_weight = 0
    raw_requests = 0
    max_orders_sec = 0
    pairs_data = {}
    fees = {}
    spot_bals = {}
    margin_lvl = 0.0
    margin_bals = {}
    spot_orders = []
    margin_orders = []

    @uf.retry_on_busy()
    def __init__(self, fr_max):
        t = Timer('session init')
        t.start()

        # configure settings and constants
        self.now_start = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
        self.client = Client(keys.bPkey, keys.bSkey, testnet=False)
        self.last_price_update = 0
        self.fr_max = fr_max
        self.leverage = 3
        self.name = 'agent names here'
        self.last_price_update = 0
        self.live = set_live()
        self.min_size = 30
        self.timeframes = get_timeframes()

        # get data from exchange
        # self.get_cg_symbols()
        abc = Timer('all binance calls')
        abc.start()
        self.info = self.client.get_exchange_info()
        self.check_rate_limits()
        self.track_weights(10)  # this should be before self.info, but would only work after check_rate_limits
        self.track_weights(10)
        self.acct = self.client.get_account()
        self.track_weights(10)
        self.m_acct = self.client.get_margin_account()
        self.spreads = self.binance_spreads()
        abc.stop()

        # filter and organise data
        self.get_pairs_info()
        self.update_prices()
        self.get_asset_bals_s()
        self.get_asset_bals_m()
        self.spot_bal: float = self.account_bal_s()
        logger.info(f"Spot balance: {self.spot_bal}")
        self.spot_usdt_bal: dict = self.get_usdt_s()
        self.margin_bal: float = self.account_bal_m()
        logger.info(f"Margin balance: {self.margin_bal}")
        self.margin_usdt_bal: dict = self.get_usdt_m()
        self.check_fees()
        # self.check_margin_lvl()
        if self.spot_bal > 30:
            self.top_up_bnb_s(15)
        if self.margin_bal > 30:
            self.top_up_bnb_m(15)

        # load local data and configure settings
        self.mkt_data_r, self.mkt_data_w, self.records_r, self.records_w, self.ohlc_path = self.data_paths()
        self.save_spreads()
        self.load_mkt_ranks()
        self.max_loan_amounts = {}
        self.pairs_set = set()
        self.book_data = {}
        # self.indicators = {'ema-200', 'ema-100', 'ema-50', 'ema-25', 'vol_delta',
        #                    'vol_delta_div', 'roc_1d', 'roc_1w', 'roc_1m', 'vwma-24'}
        self.features = {tf[0]: set() for tf in self.timeframes}
        self.wrpnl_totals = {'spot': 0, 'long': 0, 'short': 0, 'count': 0}
        self.urpnl_totals = {'spot': 0, 'long': 0, 'short': 0, 'count': 0}
        t.stop()

    def track_weights(self, weight):
        """keeps track of total api request weight to make sure I don't go over the limit

        works by adding each new call weight with a timestamp to the end of a list, then counting back along the list up
        to the weight limit, and checking if enough time has passed over the last {weight limit} worth of requests. also
        discards list items beyond the time window to stop it getting too long"""

        tw = Timer('track_weights')
        tw.start()

        now = datetime.now(timezone.utc).timestamp()
        new_weight = (now, weight)
        self.weights_count.append(new_weight)
        self.all_weights.append(new_weight)
        window = self.request_weight[0]
        weight_limit = self.request_weight[1]
        raw_window = self.raw_requests[0]
        raw_limit = self.raw_requests[1]
        # logger.debug(f"{window = } {weight_limit = } {raw_window = } {raw_limit = }")

        total = 0
        # flag = 1
        # rolling_time = 0
        rolling_weight = 0
        timespan = 0
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
                logger.info(f"request weight limit: {weight_limit} per {window}s. currently: {total} in the last "
                            f"{timespan:.1f}s")
                logger.info(f"track_weights needs {window - timespan:.1f}s of sleep")
                logger.info(f"used-weight-1m: {self.client.response.headers.get('x-mbx-used-weight-1m')}")
                time.sleep(window - timespan)
            if timespan > max(window, raw_window):
                flag = 0
                self.weights_count = self.weights_count[n:]
                break

        raw_limit_exceeded = len(self.weights_count) > raw_limit
        within_raw_window = timespan < raw_window

        if raw_limit_exceeded and within_raw_window:
            flag = 0
            logger.debug(
                f"raw request limit: {raw_limit} per {raw_window}s. currently: {total} in the last {timespan:.1f}s")
            logger.debug(f"track_weights needs {raw_window - timespan:.1f}s of sleep")
            logger.debug(f"used-weight: {self.client.response.headers['x-mbx-used-weight']}")
            logger.debug(f"used-weight-1m: {self.client.response.headers['x-mbx-used-weight-1m']}")
            time.sleep(raw_window - timespan)

        # if flag and rolling_weight:
        #     logger.info(f"Current request weight: {rolling_weight} over {rolling_time}s, "
        #                 f"raw count: {len(self.weights_count)}")
        # elif flag:
        #     pre_roll_w = sum([w[1] for w in self.weights_count[n:]])
        #     pre_roll_t = round(now - self.weights_count[n][0])
        #     logger.info(f"Current request weight: {pre_roll_w} over {pre_roll_t}s, "
        #                 f"raw count: {len(self.weights_count)}")

        tw.stop()

    def check_rate_limits(self):
        """parses the rate limits from binance and warns me if these limits change

        if the limits do change, I need to update the value of 'old_limits' by calling client.get_exchange_info() in
        python console and copying the new value across from the variable explorer, and check that the way I'm
        calculating the session attributes still works with the new values"""

        limits = self.info['rateLimits']

        old_limits = [{'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 6000},
                      {'rateLimitType': 'ORDERS', 'interval': 'SECOND', 'intervalNum': 10, 'limit': 50},
                      {'rateLimitType': 'ORDERS', 'interval': 'DAY', 'intervalNum': 1, 'limit': 160000},
                      {'rateLimitType': 'RAW_REQUESTS', 'interval': 'MINUTE', 'intervalNum': 5, 'limit': 61000}]

        if limits != old_limits:
            note = 'binance rate limits have changed, check and adjust session definition'
            logger.debug('\n****************\n\n', note, '\n\n****************\n')
            logger.warning('\n****************\n\n', note, '\n\n****************\n')

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

    # def get_cg_symbols(self):
    #     all_coins = cg.get_coins_list()
    #     self.cg_symbols = {x['symbol'].upper(): x['id'] for x in all_coins}

    @uf.retry_on_busy()
    def binance_spreads(self, quote: str = 'USDT') -> dict[str: float]:
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

        return avg_spreads

    @uf.retry_on_busy()
    def update_prices(self) -> None:
        """fetches current prices for all pairs on binance. much faster than get_price"""
        up = Timer('update_prices')
        up.start()
        now = time.perf_counter()
        last = self.last_price_update
        if now - last > 30:
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

                for f in sym['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = Decimal(f['tickSize'])
                    elif f['filterType'] == 'LOT_SIZE':
                        step_size = Decimal(f['stepSize'])
                    elif f['filterType'] == 'NOTIONAL':
                        min_size = f['minNotional']
                    elif f['filterType'] == 'MAX_NUM_ALGO_ORDERS':
                        max_algo = f['maxNumAlgoOrders']

                spread = self.spreads.get(pair)

                self.pairs_data[pair] = dict(
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
                    margin_orders=[],
                )

    def check_fees(self):
        spot_bnb = self.spot_bals['BNB']
        margin_bnb = self.margin_bals['BNB']['free']
        maker = Decimal(self.acct['commissionRates']['maker'])
        taker = Decimal(self.acct['commissionRates']['taker'])

        self.fees['spot_maker'] = maker * Decimal(0.75) if spot_bnb else maker
        self.fees['spot_taker'] = taker * Decimal(0.75) if spot_bnb else taker
        self.fees['margin_maker'] = maker * Decimal(0.75) if margin_bnb else maker
        self.fees['margin_taker'] = taker * Decimal(0.75) if margin_bnb else taker

    def data_paths(self) -> tuple[Path, Path, Path, Path, Path]:
        """automatically sets the absolute paths for the market_data, records and ohlc folders"""

        u = Timer('mkt_data_path in session')
        u.start()

        if Path('/pi_2.txt').exists():
            mkt_data_r = Path('/home/ross/coding/modular_trader/market_data')
            records_r = Path(f'/home/ross/coding/modular_trader/records')
            ohlc_data = Path(f'/home/ross/coding/modular_trader/bin_ohlc_{self.ohlc_tf}')
        else:
            mkt_data_r = Path('/home/ross/coding/pi_2/modular_trader/market_data')
            records_r = Path(f'/home/ross/coding/pi_2/modular_trader/records')
            ohlc_data = Path(f'/home/ross/coding/pi_2/modular_trader/bin_ohlc_{self.ohlc_tf}')

        mkt_data_w = Path('/home/ross/coding/modular_trader/market_data')
        mkt_data_w.mkdir(parents=True, exist_ok=True)
        records_w = Path(f'/home/ross/coding/modular_trader/records')
        records_w.mkdir(exist_ok=True)

        return mkt_data_r, mkt_data_w, records_r, records_w, ohlc_data

    def set_ohlc_tf(self, tf):
        self.ohlc_tf = tf
        # self.ohlc_path = self.ohlc_path()

    @uf.retry_on_busy()
    def get_pair_info(self, pair):
        """tries to find details about the pair from local records. if no local data exists, fetches it from exchange"""

        func_name = sys._getframe().f_code.co_name
        x6 = Timer(f'{func_name}')
        x6.start()

        info = self.symbol_info.get(pair)
        if not info:
            logger.debug('get_exchange_info')
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
        spot_symbols = [d['symbol'] for d in self.spot_orders if d['type'] in algo_types]
        margin_symbols = [d['symbol'] for d in self.margin_orders if d['type'] in algo_types]
        order_symbols = spot_symbols + margin_symbols

        logger.info('algo order counts:')
        counted = Counter(order_symbols)
        for p, v in self.pairs_data.items():
            v['algo_orders'] = 0 if p not in counted else counted[p]

            if counted[p]:
                logger.info(f"{p}: {counted[p]}")

        x7.stop()

    @uf.retry_on_busy()
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
            self.counts.append('session.get_book_data')

        x10.stop()

        return self.book_data[pair]

    def store_ohlc(self, df, pair, timeframes):
        """Calculates the minimum length of ohlc history needed to be stored in memory for the trading system to run.
        calc_min is the number of 5min periods needed to calculate all the necessary indicators at the longest timeframe
        being used in the current session
        bench refers to the number of periods needed to calculate the longest market benchmark statistics (1 mo roc)"""

        lengths = {'1w': 2016, '3d': 864, '1d': 288, '12h': 144, '6h': 72, '4h': 48, '1h': 12}
        calc_min = (self.max_length + 1) * lengths[timeframes[-1][0]]
        bench = 8928 + 1
        enough = max(bench, calc_min)

        df = df.tail(enough).reset_index(drop=True)
        self.pairs_data[pair]['ohlc_5m'] = df
        # logger.debug(f"{pair} ohlc stored in session")

    def save_spreads(self):
        spreads_path = self.mkt_data_w / 'spreads.json'
        spreads_path.touch(exist_ok=True)

        with open(spreads_path, 'r') as file:
            try:
                spreads_data = json.load(file)
            except json.JSONDecodeError:
                spreads_data = {}

        spreads_data[self.now_start] = self.spreads

        with open(spreads_path, 'w') as file:
            json.dump(spreads_data, file)

        logger.info(f'\nsaved spreads to {spreads_path}\n')

    def load_mkt_ranks(self):
        filepath = self.mkt_data_r / 'market_ranks.parquet'

        market_ranks = pd.read_parquet(filepath)

        for pair in market_ranks.index:
            self.pairs_data[pair]['market_rank_1d'] = market_ranks.at[pair, 'rank_1d']
            self.pairs_data[pair]['market_rank_1w'] = market_ranks.at[pair, 'rank_1w']
            self.pairs_data[pair]['market_rank_1m'] = market_ranks.at[pair, 'rank_1m']

    # def compute_indicators(self, df: pd.DataFrame, tf: str) -> dict:
    #     '''takes the set of required indicators and the dataframe and applies the
    #     indicator functions as necessary'''
    #
    #     ci = Timer('compute_indicators')
    #     ci.start()
    #
    #     # indicator specs:
    #     # {'indicator': 'ema', 'length': lb, 'nans': 0}
    #     # {'indicator': 'hma', 'length': lb, 'nans': 0}
    #     # {'indicator': 'ema_ratio', 'length': lb, 'nans': 0}
    #     # {'indicator': 'vol_delta', 'length': 1, 'nans': 0}
    #     # {'indicator': 'vol_delta_div', 'length': 2, 'nans': 1}
    #     # {'indicator': 'atr', 'length': lb, 'lookback': lb, 'multiplier': 3, 'nans': 0}
    #     # {'indicator': 'supertrend', 'lookback': lb, 'multiplier': mult, 'length': lb*mult, 'nans': 1}
    #     # {'indicator': 'atsz', 'length': lb, 'nans': 1}
    #     # {'indicator': 'rsi', 'length': lb+1, 'nans': lb}
    #     # {'indicator': 'stoch_rsi', 'lookback': lb1, 'stoch_lookback': lb2, 'length': lb1+lb2+1, 'nans': lb1+lb2}
    #     # {'indicator': 'inside', 'length': 2, 'nans': 1}
    #     # {'indicator': 'doji', 'length': 1, 'nans': 0}
    #     # {'indicator': 'engulfing', 'length': lb+1, 'nans': lb}
    #     # {'indicator': 'bull_bear_bar', 'length': 1, 'nans': 0}
    #     # {'indicator': 'roc_1d', 'length': 2, 'nans': 1}
    #     # {'indicator': 'roc_1w', 'length': 2, 'nans': 1}
    #     # {'indicator': 'roc_1m', 'length': 2, 'nans': 1}
    #     # {'indicator': 'vwma', 'length': lb+1, 'nans': lb}
    #     # {'indicator': 'cross_age', 'series_1': s1, 'series_2': s2, 'length': lb, 'nans': 0}
    #
    #     for i in self.indicators:
    #         vals = i.split('-')
    #         if vals[0] == 'ema':
    #             df[f"ema_{vals[1]}"] = df.close.ewm(int(vals[1])).mean()
    #         elif vals[0] == 'hma':
    #             df[f"hma_{vals[1]}"] = ind.hma(df.close, int(vals[1]))
    #         elif vals[0] == 'ema_ratio':
    #             df[f"ema_{vals[1]}_ratio"] = ind.ema_ratio(df.close, int(vals[1]))
    #         elif vals[0] == 'vol_delta':
    #             df['vol_delta'] = ind.vol_delta(df)
    #         elif vals[0] == 'vol_delta_div':
    #             df['vol_delta_div'] = ind.vol_delta_div(df)
    #         elif vals[0] == 'atr':
    #             df = ind.atr_bands(df, int(vals[1]), float(vals[2]))
    #         elif vals[0] == 'supertrend':
    #             df = ind.supertrend(df, int(vals[1]), float(vals[2]))
    #         elif vals[0] == 'atsz':
    #             df = ind.ats_z(df, int(vals[1]))
    #         elif vals[0] == 'rsi':
    #             df['rsi'] = ind.rsi(df.close, int(vals[1]))
    #         elif vals[0] == 'stoch_rsi':
    #             df['stoch_rsi'] = ind.stoch_rsi(df.close, int(vals[1]), int(vals[2]))
    #         elif vals[0] == 'inside':
    #             df = ind.inside_bars(df)
    #         elif vals[0] == 'doji':
    #             df = ind.doji(df)
    #         elif vals[0] == 'engulfing':
    #             df = ind.engulfing(df, int(vals[1]))
    #         elif vals[0] == 'bbb':
    #             df = ind.bull_bear_bar(df)
    #         elif vals[0] == 'roc_1d':
    #             df['roc_1d'] = ind.roc_1d(df.close, tf)
    #         elif vals[0] == 'roc_1w':
    #             df['roc_1w'] = ind.roc_1w(df.close, tf)
    #         elif vals[0] == 'roc_1m':
    #             df['roc_1m'] = ind.roc_1m(df.close, tf)
    #         elif vals[0] == 'vwma':
    #             df[f'vwma_{vals[1]}'] = ind.vwma(df, int(vals[1]))
    #         elif vals[0] == 'cross_age':
    #             if vals[1] == 'st':
    #                 s1 = f"st-{int(vals[2])}-{float(vals[3]/10):.1f}"
    #                 if s1 not in df.columns:
    #                     df = ind.supertrend(df, int(vals[2]), float(vals[3]/10))
    #                 s2 = f"st-{int(vals[4])}-{float(vals[5]/10):.1f}"
    #                 if s2 not in df.columns:
    #                     df = ind.supertrend(df, int(vals[4]), float(vals[5]/10))
    #             elif vals[1] == 'ema':
    #                 s1 = f"ema_{vals[2]}"
    #                 if s1 not in df.columns:
    #                     df[f"ema_{vals[2]}"] = df.close.ewm(int(vals[2])).mean()
    #                 s2 = f"{vals[1]}_{vals[3]}"
    #                 if s2 not in df.columns:
    #                     df[f"ema_{vals[3]}"] = df.close.ewm(int(vals[3])).mean()
    #             df[i] = ind.consec_condition(df[s1] > df[s2])
    #
    #     ci.stop()
    #     return df

    def compute_features(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """takes the set of features that need to be calculated and applies them to the dataframe"""

        cf = Timer('compute features')
        cf.start()

        for f in self.features[tf]:
            if f == 'r_pct':
                continue
            df = features.add_feature(df, f, tf)
        X, _, cols = mlf.transform_columns(df, df)
        df = pd.DataFrame(X, columns=cols)

        cf.stop()
        return df

    # Spot Specific Methods

    @uf.retry_on_busy()
    def top_up_bnb_s(self, usdt_size: int) -> dict:
        """checks net BNB balance and interest owed, if net is below the threshold,
        buys BNB then repays any interest"""

        gh = Timer('top_up_bnb_M')
        gh.start()

        # check balances
        free_bnb = self.spot_bals['BNB']['free']
        free_usdt = self.spot_bals['USDT']['free']

        # calculate value
        bnb_value = free_bnb * self.pairs_data['BNBUSDT']['price']

        # top up if needed
        if bnb_value < 10:
            if free_usdt > usdt_size:
                # pb.push_note(now, 'Topping up spot BNB')
                # uid weight of 6
                order = self.client.create_order(
                    symbol='BNBUSDT',
                    side=be.SIDE_BUY,
                    type=be.ORDER_TYPE_MARKET,
                    quoteOrderQty=usdt_size)
                # logger.debug(pformat(order))
            else:
                logger.warning('Warning - Spot BNB balance low and not enough USDT to top up')
                order = None
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
        if self.spot_bal:
            pct = round((100 * value / self.spot_bal), 5)
        else:
            pct = 0
        # logger.debug(f'spot usdt stats: qty = {bal.get("free")}, {value = }, {pct = }, {self.spot_bal = }')

        return {'qty': free, 'value': value, 'pf%': pct}

    def update_usdt_s(self, up: float = 0.0, down: float = 0.0) -> None:
        """updates the local record of spot usdt holdings to avoid having to call the api"""

        uus = Timer('update_usdt_s')
        uus.start()

        qty = (self.spot_bals['USDT'].get('qty')) + up - down
        value = round(self.spot_bals['USDT'].get('value') + up - down, 2)

        if self.spot_bal:
            pct = round((100 * value / self.spot_bal), 5)
        else:
            pct = 0.0

        self.spot_bals['USDT'] = {'qty': qty, 'value': value, 'pf%': pct}

        uus.stop()
        return

    def get_asset_bals_s(self):
        """creates a dictionary of spot asset balances, stored as floats"""

        func_name = sys._getframe().f_code.co_name
        gab = Timer(f'{func_name}')
        gab.start()

        bals = self.acct['balances']

        for bal in bals:
            asset = bal.get('asset')
            free = float(bal.get('free'))
            locked = float(bal.get('locked'))

            self.spot_bals[asset] = {'free': free, 'locked': locked}

        gab.stop()

    # Margin Specific Methods
    @uf.retry_on_busy()
    def top_up_bnb_m(self, usdt_size: int) -> dict:
        """checks net BNB balance and interest owed, if net is below the threshold,
        buys BNB then repays any interest"""

        gh = Timer('top_up_bnb_M')
        gh.start()

        # check balances
        free_bnb = self.margin_bals['BNB']['free']
        interest = self.margin_bals['BNB']['interest']
        free_usdt = self.margin_bals['USDT']['free']
        net_bnb = free_bnb - interest

        if interest:
            logger.info(f'BNB interest: {interest}')

        # calculate value
        bnb_value = net_bnb * self.pairs_data['BNBUSDT']['price']

        # top up if needed
        if bnb_value < 10:
            if free_usdt > usdt_size:
                # pb.push_note(now, 'Topping up margin BNB')
                # uid weight of 6
                order = self.client.create_margin_order(
                    symbol='BNBUSDT',
                    side=be.SIDE_BUY,
                    type=be.ORDER_TYPE_MARKET,
                    quoteOrderQty=usdt_size)
                # logger.debug(pformat(order))
            else:
                logger.warning('Warning - Margin BNB balance low and not enough USDT to top up')
                order = None
        else:
            order = None

        # repay interest
        try:
            if float(interest) > 0.001:
                # uid weight of 3000. not sure how to keep track of this
                self.client.repay_margin_loan(asset='BNB', amount='0.001')
        except bx.BinanceAPIException as e:
            if e.code == -3015:
                logger.exception(" Top up BNB caused an exception trying to repay interest")
                return order
            else:
                raise e
        gh.stop()
        return order

    def check_open_margin_orders(self):
        # this may obviate the need for count_algo_orders if I can simply use the length of the lists in
        # pairs_data[pair]['spot_orders'] plus pairs_data[pair]['margin_orders'] as a count
        if self.margin_orders:
            for order in self.margin_orders:
                self.pairs_data[order['symbol']]['margin_orders'].append(order)

    def account_bal_m(self) -> float:
        """fetches the total value of the margin account holdings from binance
        and returns it, denominated in USDT"""

        jh = Timer('account_bal_M')
        jh.start()

        total_net = float(self.m_acct.get('totalNetAssetOfBtc'))
        btc_price = self.pairs_data['BTCUSDT']['price']
        usdt_total_net = total_net * btc_price

        jh.stop()
        return round(usdt_total_net, 2)

    def total_debt(self) -> float:
        td = Timer('account_bal_M')
        td.start()

        total_debt = float(self.m_acct.get('totalLiabilityOfBtc'))
        btc_price = self.pairs_data['BTCUSDT']['price']
        usdt_total_debt = total_debt * btc_price

        td.stop()
        return round(usdt_total_debt, 2)

    def get_usdt_m(self) -> Dict[str, float]:
        """checks current usdt balance and returns a dictionary for updating the sizing dict"""
        um = Timer('update_usdt_m')
        um.start()

        bal = self.margin_bals['USDT']
        net = float(bal.get('net_asset'))
        owed = float(bal.get('borrowed'))
        qty = float(bal.get('free'))

        value = round(net, 2)
        if self.spot_bal:
            pct = round((100 * value / self.margin_bal), 5)
        else:
            pct = 0
        # logger.debug(f'margin usdt stats: qty = {bal.get("free")}, owed = {bal.get("borrowed")}, {value = }, '
        #              f'{pct = }, {self.margin_bal = }')
        um.stop()
        return {'qty': qty, 'owed': owed, 'value': value, 'pf%': pct}

    def update_usdt_m(self, up: float = 0.0, down: float = 0.0, borrow: float = 0.0, repay: float = 0.0) -> None:
        """calculates current usdt balance without fetching from binance,
        called whenever usdt balance is changed"""

        hj = Timer('update_usdt_m')
        hj.start()

        qty = self.margin_usdt_bal.get('qty') + up - down
        value = self.margin_usdt_bal.get('value') + up - down - borrow + repay
        owed = self.margin_usdt_bal.get('owed') + borrow - repay

        if self.spot_bal:
            pct = round((100 * value / self.margin_bal), 5)
        else:
            pct = 0

        # logger.debug(f'usdt stats: {qty = }, {owed = }, {value = }, {pct = }, {self.margin_bal = }')
        self.margin_usdt_bal = {'qty': float(qty), 'owed': float(owed), 'value': float(value), 'pf%': float(pct)}
        hj.stop()

    def check_margin_lvl(self) -> float:
        """checks how leveraged the account is and sends a warning push note if
        leverage is getting too high"""

        func_name = sys._getframe().f_code.co_name
        x4 = Timer(f'{func_name}')
        x4.start()

        self.track_weights(10)
        abc = Timer('all binance calls')
        abc.start()
        self.m_acct = self.client.get_margin_account()
        abc.stop()
        self.margin_lvl = float(self.m_acct.get('marginLevel'))
        logger.info(f"Margin level: {self.margin_lvl:.2f}")

        net_asset = self.account_bal_m()
        max_debt = net_asset * (self.leverage - 1)  # 3x leverage = net_asset*2, 5x leverage = net_asset*4
        total_debt = self.total_debt()
        remaining = max_debt - total_debt

        x4.stop()
        return remaining

    def get_asset_bals_m(self) -> None:
        """creates a dictionary of margin asset balances, stored as floats"""

        func_name = sys._getframe().f_code.co_name
        x5 = Timer(f'{func_name}')
        x5.start()

        bals = self.m_acct.get('userAssets')

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

    @uf.retry_on_busy()
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
    # def sync_mkt_data(self) -> None:
    #     func_name = sys._getframe().f_code.co_name
    #     x2 = Timer(f'{func_name}')
    #     x2.start()
    #
    #     for data_file in ['binance_liquidity_history.txt',
    #                       'binance_depths_history.txt',
    #                       'binance_spreads_history.txt']:
    #         real_file = Path(self.market_data_read / data_file)
    #         test_file = Path(self.market_data_write / data_file)
    #         test_file.touch(exist_ok=True)
    #
    #         if real_file.exists():
    #             with open(real_file, 'r') as file:
    #                 book_data = file.readlines()
    #             if book_data:
    #                 with open(test_file, 'w') as file:
    #                     file.writelines(book_data)
    #
    #     x2.stop()

    @uf.retry_on_busy()
    def update_algo_orders(self):
        # TODO i need to implement a check so that spot orders can be looked for only if there is a spot agent in play
        # self.track_weights(40)
        # self.spot_orders = self.client.get_open_orders()

        if len(self.pairs_set) > 35:
            # logger.info("getting open margin orders for all pairs")
            self.track_weights(len(self.client.get_margin_all_pairs()))
            # weighting for this call = number of pairs on exchange
            self.margin_orders = self.client.get_open_margin_orders()
        else:
            for pair in self.pairs_set:
                # logger.info(f"getting open margin orders for {pair}")
                self.track_weights(10)
                self.margin_orders.extend(self.client.get_open_margin_orders(symbol=pair))
        self.check_open_spot_orders()
        self.check_open_margin_orders()
        self.count_algo_orders()  # kind of redundant since the above two methods create lists which could be counted


# class LightSession(TradingSession):
#     @uf.retry_on_busy()
#     def __init__(self):
#         TradingSession.__init__(self, 0.1)
#         self.now_start = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
#         self.client = Client(keys.bPkey, keys.bSkey)
#         self.live = set_live()
#         self.weights_count = []
#
#         # get data from exchange
#         # self.get_cg_symbols()
#         self.info = self.client.get_exchange_info()
#         self.check_rate_limits()
#         self.track_weights(10)  # this should be before self.info, but would only work after check_rate_limits
#         # self.track_weights(2)
#         # self.obt = self.client.get_orderbook_tickers()
#         self.spreads = self.binance_spreads()
#
#         # filter and organise data
#         self.get_pairs_info()
#
#         # load local data and configure settings
#         self.market_data_read, self.market_data_write = self.mkt_data_path()
#         self.read_records, self.write_records = self.records_path()
#         self.ohlc_data = self.ohlc_path()
#         self.save_spreads()
#         self.indicators = None  # to stop any default indicators being calculated by inheritance
#
#
# class CheckRecordsSession(TradingSession):
#     @uf.retry_on_busy()
#     def __init__(self):
#         TradingSession.__init__(self, 0.1)
#         # self.fr_max = 0.0005
#         # self.pairs_data = {}
#         # self.counts = []
#         # self.spot_bal = 1
#         # self.margin_bal = 1
#         # self.spot_usdt_bal = 1
#         # self.margin_usdt_bal = 1
#         #
#         # self.ohlc_length = 0
#         # self.now_start = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
#         # self.client = Client(keys.bPkey, keys.bSkey)
#         # self.live = self.set_live()
#         # self.weights_count = []
#
#         # load local data and configure settings
#         # self.market_data_read, self.market_data_write = self.mkt_data_path()
#         # self.read_records, self.write_records = self.records_path()
#         # self.ohlc_data = self.ohlc_path()
#         # self.indicators = None

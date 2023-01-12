import keys, math, time, json
import statistics as stats
import pandas as pd
import numpy as np
from binance.client import Client
import binance.enums as be
from pushbullet import Pushbullet
from decimal import Decimal, getcontext
from pprint import pprint
from config import ohlc_data, not_pairs
from pathlib import Path
from datetime import datetime
import utility_funcs as uf
from timers import Timer
from typing import Union, List, Tuple, Dict, Set, Optional, Any
from collections import Counter
from config import testing

client = Client(keys.bPkey, keys.bSkey, testnet=testing)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12


# -#-#- Utility Functions

def step_round(num: float, step: str) -> str:
    """rounds down to any step size"""
    gv = Timer('step_round')
    gv.start()
    if not float(step):
        return str(num)
    num = Decimal(num)
    step = Decimal(step)
    gv.stop()
    return str(math.floor(num / step) * step)


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """resamples a dataframe and resets the datetime index"""

    hb = Timer('resample')
    hb.start()
    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum'})
    df.reset_index(inplace=True)  # don't use drop=True because i want the
    # timestamp index back as a column
    hb.stop()
    return df


def order_by_volsm(pairs: list, lookback: int) -> List[Tuple[str, float]]:
    """measures recent volatility of all pairs, then sorts the list of pairs
    with the most volatile first"""
    tup_list = []

    for pair in pairs:
        # load data
        filepath = Path(f'{ohlc_data}/{pair}.pkl')
        if filepath.exists():
            df = pd.read_pickle(filepath)
        else:
            continue

        # trim data
        if len(df) > lookback:  # 8760 is 1 year's worth of 1h periods
            df = df.tail(lookback)
            df.reset_index(drop=True, inplace=True)

            # calc vol and add to list
            df['roc'] = df.close.pct_change(periods=4)
            df['roc_diff'] = abs(df['roc'] - df['roc'].shift(1))
            df['volatil'] = df.roc_diff.ewm(2).mean()
            try:
                high = df.high.max()
                low = df.low.min()
                mid = (high + low) / 2
                price_range = ((high - low) / mid)  # **2
                # volatility = df.roc.std()
                volatility = df.roc_diff.mean()
                smoothness = round(10 * price_range / volatility)
            except ValueError as e:
                print(pair, e)
            tup_list.append((pair, smoothness))

    # sort tup_list by vol
    tup_list = sorted(tup_list, key=lambda x: x[1], reverse=True)

    return tup_list


def get_size(agent, price: float, balance: float, risk: float) -> Tuple[float]:
    """calculates the desired position size in base or quote denominations
    using the total account balance, current fixed-risk setting, and the distance
    from current price to stop-loss"""

    jn = Timer('get_size')
    jn.start()
    usdt_size_l = balance * agent.fixed_risk_l / risk
    asset_size_l = float(usdt_size_l / price)

    usdt_size_s = balance * agent.fixed_risk_s / risk
    asset_size_s = float(usdt_size_s / price)
    jn.stop()
    return asset_size_l, usdt_size_l, asset_size_s, usdt_size_s


def calc_stop(inval: float, spread: float, price: float, min_risk: float = 0.01) -> float:
    """calculates what the stop-loss trigger price should be based on the current
    value of the supertrend line and the current spread (slippage proxy).
    if this is too close to the entry price, the stop will be set at the minimum
    allowable distance."""
    buffer = max(spread * 2, min_risk)

    if price > inval:
        stop_price = float(inval) * (1 - buffer)
    else:
        stop_price = float(inval) * (1 + buffer)

    # if (((price > inval) and (stop_price > price)) 
    #     or 
    #     ((price < inval) and (stop_price < price))):
    # print(f"{price = } {inval = } {buffer = } {stop_price = }")

    return stop_price


def calc_fee_bnb(usdt_size: str, fee_rate: float = 0.00075) -> float:
    """calculates the trade fee denominated in BNB"""

    jm = Timer('calc_fee_bnb')
    jm.start()
    bnb_price = get_price('BNBUSDT')
    fee_usdt = float(usdt_size) * fee_rate
    jm.stop()
    return fee_usdt / bnb_price


# -#-#- Market Data Functions


def get_price(pair: str) -> float:
    """fetches a single pairs price from binance. slow so only use when necessary"""

    hn = Timer('get_price')
    hn.start()
    price = float(client.get_ticker(symbol=pair).get('lastPrice'))
    hn.stop()
    return price


def update_prices(session) -> None:
    """fetches current prices for all pairs on binance. much faster than get_price"""
    up = Timer('update_prices')
    up.start()
    now = time.perf_counter()
    last = session.last_price_update
    if now - last > 60:
        prices = client.get_all_tickers()
        session.prices = {x.get('symbol', None): float(x.get('price', 0)) for x in prices}
        session.last_price_update = time.perf_counter()
    up.stop()


def get_mid_price(pair: str) -> float:
    """returns the midpoint between first bid and ask price on the orderbook
    for the pair in question"""

    gb = Timer('get_mid_price')
    gb.start()

    t = client.get_orderbook_ticker(symbol=pair)
    bid = float(t.get('bidPrice'))
    ask = float(t.get('askPrice'))
    gb.stop()
    return (bid + ask) / 2


def get_depth(session, pair: str) -> Tuple[float]:
    """returns the quantity (in the quote currency) that could be bought/sold
    within the % range of price set by the max_slip param"""

    max_slip = session.max_spread
    fv = Timer('get_depth')
    fv.start()

    price = session.prices[pair]
    book = session.get_book_data(pair)

    bid_price = float(book.get('bids')[0][0])
    # max_price is x% above price
    max_price = bid_price * (1 + (max_slip / 100))
    depth_l = 0
    for i in book.get('asks'):
        if float(i[0]) <= max_price:
            depth_l += float(i[1])
        else:
            break

    ask_price = float(book.get('asks')[0][0])
    # min_price is x% below price
    min_price = ask_price * (1 - (max_slip / 100))
    depth_s = 0
    for i in book.get('bids'):
        if float(i[0]) >= min_price:
            depth_s += float(i[1])
        else:
            break

    usdt_depth_l = depth_l * price
    usdt_depth_s = depth_s * price
    fv.stop()
    return usdt_depth_l, usdt_depth_s


def get_book_stats(session, pair: str, quote: str, width: Union[int, float] = 2) -> dict:
    """returns a dictionary containing the base asset, the quote asset,
    the spread, and the bid and ask depth within the % range of price set
    by the width param in base and quote denominations"""

    dc = Timer('get_book_stats')
    dc.start()

    q_len = len(quote) * -1
    base = pair[:q_len]

    book = session.get_book_data(pair)

    best_bid = float(book.get('bids')[0][0])
    best_ask = float(book.get('asks')[0][0])
    mid_price = (best_bid + best_ask) / 2
    spread = (best_ask - best_bid) / mid_price

    max_price = mid_price * (1 + (width / 100))  # max_price is x% above price
    ask_depth = 0
    for i in book.get('asks'):
        if float(i[0]) <= max_price:
            ask_depth += float(i[1])
        else:
            break
    min_price = mid_price * (1 - (width / 100))  # max_price is x% above price
    bid_depth = 0
    for i in book.get('bids'):
        if float(i[0]) >= min_price:
            bid_depth += float(i[1])
        else:
            break

    q_bid_depth = bid_depth * mid_price
    q_ask_depth = ask_depth * mid_price

    stats = {'base': base, 'quote': quote, 'spread': spread,
             'base_bids': bid_depth, 'base_asks': ask_depth,
             'quote_bids': q_bid_depth, 'quote_asks': q_ask_depth}
    dc.stop()
    return stats


def binance_spreads(quote: str = 'USDT') -> dict:
    """returns a dictionary with pairs as keys and current average spread as values"""

    sx = Timer('binance_spreads')
    sx.start()

    length = len(quote)
    avg_spreads = {}

    s_1 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1 * length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_1[pair] = spread / mid

    time.sleep(1)

    s_2 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1 * length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_2[pair] = spread / mid

    time.sleep(1)

    s_3 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1 * length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_3[pair] = spread / mid

    for k in s_1:
        avg_spreads[k] = stats.median([s_1.get(k), s_2.get(k), s_3.get(k)])
    sx.stop()
    return avg_spreads


def binance_depths(quotes: List[str] = None) -> dict:
    """calls get_orderbook_tickers 3 times, and loops through each pair averaging
    the quantities of the first bid and the first ask across those three. returns
    a dictionary with all pairs as keys, and a dictionary containing median bid
    depth and median ask depth as values"""

    if quotes is None:
        quotes = ['USDT', 'BTC']

    az = Timer('binance_depths')
    az.start()
    avg_depths = {}

    for quote in quotes:
        length = len(quote)
        bd_1 = {}
        sd_1 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1 * length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_1[pair] = ask
                sd_1[pair] = bid

        time.sleep(1)

        bd_2 = {}
        sd_2 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1 * length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_2[pair] = ask
                sd_2[pair] = bid

        time.sleep(1)

        bd_3 = {}
        sd_3 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1 * length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_3[pair] = ask
                sd_3[pair] = bid

        for k in bd_1:
            avg_depths[k] = {'asks': stats.median([bd_1.get(k), bd_2.get(k), bd_3.get(k)]),
                             'bids': stats.median([sd_1.get(k), sd_2.get(k), sd_3.get(k)])}
    az.stop()
    return avg_depths


def get_pairs(quote: str = 'USDT', market: str = 'SPOT') -> List[str]:
    """returns all active pairs for a given quote currency. possible values for
    quote are USDT, BTC, BNB etc. possible values for market are SPOT or CROSS"""

    sa = Timer('get_pairs')
    sa.start()

    if market == 'SPOT':
        info = client.get_exchange_info()
        symbols = info.get('symbols')
        pairs = []
        for sym in symbols:
            right_quote = sym.get('quoteAsset') == quote
            right_market = market in sym.get('permissions')
            trading = sym.get('status') == 'TRADING'
            allowed = sym.get('symbol') not in not_pairs
            if right_quote and right_market and trading and allowed:
                pairs.append(sym.get('symbol'))
    elif market == 'CROSS':
        pairs = []
        info = client.get_margin_all_pairs()
        for i in info:
            if i.get('quote') == quote:
                pairs.append(i.get('symbol'))
    sa.stop()
    return pairs


def get_ohlc(pair: str, timeframe: str, span: str = "1 year ago UTC") -> pd.DataFrame:
    """fetches kline data from binance for the stated pair and timeframe.
    span tells the function how far back to start the data, in plain english
    for timeframe, use strings like 5m or 1h or 1d"""

    # client = Client(keys.bPkey, keys.bSkey)
    tf = {'1m': Client.KLINE_INTERVAL_1MINUTE,
          '5m': Client.KLINE_INTERVAL_5MINUTE,
          '15m': Client.KLINE_INTERVAL_15MINUTE,
          '30m': Client.KLINE_INTERVAL_30MINUTE,
          '1h': Client.KLINE_INTERVAL_1HOUR,
          '4h': Client.KLINE_INTERVAL_4HOUR,
          '6h': Client.KLINE_INTERVAL_6HOUR,
          '8h': Client.KLINE_INTERVAL_8HOUR,
          '12h': Client.KLINE_INTERVAL_12HOUR,
          '1d': Client.KLINE_INTERVAL_1DAY,
          '3d': Client.KLINE_INTERVAL_3DAY,
          '1w': Client.KLINE_INTERVAL_1WEEK,
          }
    klines = client.get_historical_klines(pair, tf.get(timeframe), span)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(['close_time', 'ignore'], axis=1)

    return df


def update_ohlc(pair: str, timeframe: str, old_df: pd.DataFrame) -> pd.DataFrame:
    """takes an ohlc dataframe, works out when the data ends, then requests from
    binance all data from the end to the current moment. It then joins the new
    data onto the old data and returns the updated dataframe"""

    # client = Client(keys.bPkey, keys.bSkey)
    tf = {'1m': Client.KLINE_INTERVAL_1MINUTE,
          '5m': Client.KLINE_INTERVAL_5MINUTE,
          '15m': Client.KLINE_INTERVAL_15MINUTE,
          '30m': Client.KLINE_INTERVAL_30MINUTE,
          '1h': Client.KLINE_INTERVAL_1HOUR,
          '4h': Client.KLINE_INTERVAL_4HOUR,
          '6h': Client.KLINE_INTERVAL_6HOUR,
          '8h': Client.KLINE_INTERVAL_8HOUR,
          '12h': Client.KLINE_INTERVAL_12HOUR,
          '1d': Client.KLINE_INTERVAL_1DAY,
          '3d': Client.KLINE_INTERVAL_3DAY,
          '1w': Client.KLINE_INTERVAL_1WEEK,
          }
    old_end = int(old_df.at[len(old_df) - 1, 'timestamp'].timestamp()) * 1000
    klines = client.get_klines(symbol=pair, interval=tf.get(timeframe),
                               startTime=old_end)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(['close_time', 'ignore'], axis=1)

    return pd.concat([old_df[:-1], df], copy=True, ignore_index=True)

def resample_ohlc(tf, offset, df):
    """resamples ohlc data to the required timeframe and offset, then discards older rows if necessary to return a
    dataframe of the desired length"""

    tf_map = {'15m': '15T', '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H',
              '8h': '8H', '12h': '12H', '1d': '1D', '3d': '3D', '1w': '1W'}

    df = df.resample(tf_map[tf], on='timestamp',
                     offset=offset).agg({'open': 'first',
                                                 'high': 'max',
                                                 'low': 'min',
                                                 'close': 'last',
                                                 'base_vol': 'sum',
                                                 'quote_vol': 'sum',
                                                 'num_trades': 'sum',
                                                 'taker_buy_base_vol': 'sum',
                                                 'taker_buy_quote_vol': 'sum'})

    df = df.reset_index() # drop=False because we want to keep the timestamp column

    return df


def prepare_ohlc(session, pair: str) -> pd.DataFrame:
    """checks if there is old data already, if so it loads the old data and
    downloads an update, if not it downloads all data from scratch, then
    resamples all data to desired timeframe"""

    ds = Timer('prepare_ohlc')
    ds.start()

    if session.live:
        filepath = Path(f'{session.ohlc_data}/{pair}.pkl')
    else:
        filepath = Path(f'/home/ross/Documents/backtester_2021/bin_ohlc_15m/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        if len(df) > 2:
            df = df.iloc[:-1, :]
            df = update_ohlc(pair, '15m', df)

    else:
        df = get_ohlc(pair, '15m', '2 years ago UTC')
        print(f'downloaded {pair} from scratch')

    if len(df) > 70080:  # 70080 is 2 year's worth of 15m periods
        df = df.tail(70080)
        df.reset_index(drop=True, inplace=True)
    df.to_pickle(filepath)

    df = resample_ohlc(session.tf, session.offset, df)
    df = df.tail(session.max_length).reset_index(drop=True)

    ds.stop()
    return df


# -#-#- Trading Functions

def create_trade_dict(order: dict, price: float, live: bool) -> Dict[str, str]:
    """collects and returns the details of the order in a dictionary"""

    fd = Timer('create_trade_dict')
    fd.start()

    pair = order.get('symbol')
    if live:
        fills = order.get('fills')
        fee = sum([Decimal(fill.get('commission')) for fill in fills])
        qty = sum([Decimal(fill.get('qty')) for fill in fills])
        exe_prices = [Decimal(fill.get('price')) for fill in fills]
        avg_price = stats.mean(exe_prices)

        if float(order.get('cummulativeQuoteQty')):
            quote_qty = order.get('cummulativeQuoteQty')
        else:
            quote_qty = str(qty * avg_price)

        trade_dict = {'timestamp': order.get('transactTime'),
                      'trig_price': str(price),
                      'exe_price': str(avg_price),
                      'base_size': str(order.get('executedQty')),
                      'quote_size': quote_qty,
                      'fee': str(fee),
                      'fee_currency': fills[0].get('commissionAsset')
                      }
        if order.get('status') != 'FILLED':
            print(f'{pair} order not filled')
            pb.push_note('Warning', f'{pair} order not filled')

    else:
        trade_dict = {'timestamp': str(order.get('transactTime')),
                      "trig_price": str(price),
                      "exe_price": str(price),
                      "base_size": str(order.get('executedQty')),
                      "quote_size": str(order.get('cummulativeQuoteQty')),
                      "fee": '0',
                      "fee_currency": "BNB"
                      }
    fd.stop()
    return trade_dict


def valid_size(session, pair: str, size: float) -> str:
    """rounds the desired order size to the correct step size for *MARKET ORDERS* on binance"""

    gf = Timer('get_symbol_info valid')
    gf.start()

    info = session.get_pair_info(pair)

    for f in info.get('filters'):
        if f.get('filterType') == 'MARKET_LOT_SIZE':
            step_size = f.get('stepSize')

    gf.stop()
    return step_round(size, step_size)


def valid_price(session, pair: str, price: float) -> str:
    """rounds the desired order price to the correct step size for binance"""

    hg = Timer('get_symbol_info valid')
    hg.start()

    info = session.get_pair_info(pair)

    step_size = info.get('filters')[0].get('tickSize')
    hg.stop()
    return step_round(price, step_size)


# -#-#- Margin Account Functions


def free_usdt_M() -> float:
    """fetches the free balance of USDT in the margin account"""

    kj = Timer('free_usdt_M')
    kj.start()

    info = client.get_margin_account()
    assets = info.get('userAssets')
    bal = 0
    for a in assets:
        if a.get('asset') == 'USDT':
            bal = float(a.get('free'))
    kj.stop()
    return bal


# -#-#- Margin Trading Functions


def top_up_bnb_M(session, usdt_size: int) -> dict:
    """checks net BNB balance and interest owed, if net is below the threshold,
    buys BNB then repays any interest"""

    gh = Timer('top_up_bnb_M')
    gh.start()

    now = datetime.now().strftime('%d/%m/%y %H:%M')

    # check balances
    free_bnb = session.bals_dict['BNB']['free']
    interest = session.bals_dict['BNB']['interest']
    free_usdt = session.bals_dict['USDT']['free']
    net_bnb = free_bnb - interest

    if interest:
        print(f'BNB interest: {interest}')

    # calculate value
    avg_price = client.get_avg_price(symbol='BNBUSDT')
    price = float(avg_price.get('price'))
    bnb_value = net_bnb * price

    # top up if needed
    if bnb_value < 10:
        if free_usdt > usdt_size:
            pb.push_note(now, 'Topping up BNB')
            order = client.create_margin_order(
                symbol='BNBUSDT',
                side=be.SIDE_BUY,
                type=be.ORDER_TYPE_MARKET,
                quoteOrderQty=usdt_size)
            # pprint(order)
        else:
            pb.push_note(now, 'Warning - BNB balance low and not enough USDT to top up')
    else:
        order = None

    # repay interest
    if float(interest):
        client.repay_margin_loan(asset='BNB', amount=interest)
    gh.stop()
    return order


def buy_asset_M(session, pair: str, size: float, is_base: bool, price: float, live: bool) -> dict:
    """sends a market buy order to binance in the margin account and returns the
    order data"""

    fg = Timer('buy_asset_M')
    fg.start()

    if live and is_base:
        base_size = valid_size(session, pair, size)
        buy_order = client.create_margin_order(symbol=pair,
                                               side=be.SIDE_BUY,
                                               type=be.ORDER_TYPE_MARKET,
                                               quantity=base_size)
    elif live and not is_base:
        buy_order = client.create_margin_order(symbol=pair,
                                               side=be.SIDE_BUY,
                                               type=be.ORDER_TYPE_MARKET,
                                               quoteOrderQty=size)
    else:
        now = int(datetime.now().timestamp() * 1000)
        if is_base:
            base_size = size
            usdt_size = f"{float(size) * price:.2f}"
        else:
            base_size = valid_size(session, pair, size / price)
            print(f'buy_asset_M {size = }, {base_size = }')
            if not base_size:  # if size == 0, valid_size will output None
                print(f'*problem* non-live buy {pair}: {base_size = }')
                base_size = 0
            usdt_size = str(size)
        buy_order = {'clientOrderId': '111111',
                     'cummulativeQuoteQty': usdt_size,
                     'executedQty': str(base_size),
                     'fills': [{'commission': '0',
                                'commissionAsset': 'BNB',
                                'price': str(valid_price(session, pair, price)),
                                'qty': str(base_size)}],
                     'isIsolated': False,
                     'orderId': 123456,
                     'origQty': str(base_size),
                     'price': '0',
                     'side': 'BUY',
                     'status': 'FILLED',
                     'symbol': pair,
                     'timeInForce': 'GTC',
                     'transactTime': now,
                     'type': 'MARKET'}
    fg.stop()
    return buy_order


def sell_asset_M(session, pair: str, base_size: float, price: float, live: bool) -> dict:
    """sends a market sell order to binance in the margin account and returns the
    order data"""

    df = Timer('sell_asset_M')
    df.start()

    base_size = valid_size(session, pair, base_size)
    if not base_size:  # if size == 0, valid_size will output None
        base_size = 0
    if live:
        sell_order = client.create_margin_order(symbol=pair, side=be.SIDE_SELL, type=be.ORDER_TYPE_MARKET,
                                                quantity=base_size)
    else:
        now = int(datetime.now().timestamp() * 1000)
        usdt_size = valid_size(session, pair, float(base_size) * price)
        if not usdt_size:
            usdt_size = 0
        sell_order = {
            'clientOrderId': '111111',
            'cummulativeQuoteQty': str(usdt_size),
            'executedQty': str(base_size),
            'fills': [{'commission': '0',
                       'commissionAsset': 'BNB',
                       'price': str(valid_price(session, pair, price)),
                       'qty': str(base_size)}],
            'isIsolated': False,
            'orderId': 123456,
            'origQty': str(base_size),
            'price': '0',
            'side': 'SELL',
            'status': 'FILLED',
            'symbol': pair,
            'timeInForce': 'GTC',
            'transactTime': now,
            'type': 'MARKET'}
    df.stop()
    return sell_order


def borrow_asset_M(asset: str, qty: str, live: bool) -> None:
    """calls the binance api function to take out a margin loan"""

    if live:
        client.create_margin_loan(asset=asset, amount=qty)


def repay_asset_M(asset: str, qty: str, live: bool) -> None:
    """calls the binance api function to repay a margin loan"""

    if live:
        client.repay_margin_loan(asset=asset, amount=qty)


def set_stop_M(session, pair: str, size: float, side: str, trigger: float, limit: float) -> dict:
    """sends a margin stop-loss limit order to binance and returns the order data"""

    sd = Timer('set_stop_M')
    sd.start()

    trigger = valid_price(session, pair, trigger)
    limit = valid_price(session, pair, limit)
    stop_size = valid_size(session, pair, size)
    # print(f"setting {pair} stop: {stop_size = } {side = } {trigger = } {limit = }")
    if session.live:
        stop_sell_order = client.create_margin_order(symbol=pair,
                                                     side=side,
                                                     type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                     timeInForce=be.TIME_IN_FORCE_GTC,
                                                     stopPrice=trigger,
                                                     quantity=stop_size,
                                                     price=limit)
    else:
        stop_sell_order = {'orderId': 'not live'}

    session.algo_order_counts += Counter([pair])

    sd.stop()
    return stop_sell_order


def clear_stop_M(session, pair: str, position: dict) -> Tuple[Any, Decimal]:
    """finds the order id of the most recent stop-loss from the trade record
    and cancels that specific order. if no such id can be found, returns null values"""

    fc = Timer('clear_stop_M')
    fc.start()

    stop_id = position['stop_id']

    clear, base_size = None, None
    if session.live:
        if stop_id:
            clear = client.cancel_margin_order(symbol=pair, orderId=str(stop_id))
            base_size = clear.get('origQty')
        else:
            print(f'no recorded stop id for {pair}')
            clear = {}
            base_size = 0
    else:
        base_size = position['base_size']

    session.algo_order_counts -= Counter([pair])

    fc.stop()
    return clear, base_size

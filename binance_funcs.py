import keys
import math
import time
import statistics as stats
import pandas as pd
import numpy as np
from binance.client import Client
import binance.enums as enums
from pushbullet import Pushbullet
from decimal import Decimal
from pprint import pprint
from config import ohlc_data
from pathlib import Path
from datetime import datetime

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


# Utility Functions

def step_round(x, step):
    x = Decimal(x)
    step = Decimal(step)

    return math.floor(x / step) * step


def resample(df, timeframe):
    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum'})
    df.reset_index(inplace=True)  # don't use drop=True because i want the
    # timestamp index back as a column


def get_book_stats(pair, quote, width=2):
    '''returns a dictionary containing the base asset, the quote asset, 
    the spread, and the bid and ask depth within the % range of price set 
    by the width param in base and quote denominations'''

    q_len = len(quote) * -1
    base = pair[:q_len]

    book = client.get_order_book(symbol=pair)

    best_bid = float(book.get('bids')[0][0])
    best_ask = float(book.get('asks')[0][0])
    mid_price = (best_bid + best_ask) / 2
    spread = (best_ask-best_bid) / mid_price

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

    return stats

# Account Functions


def account_bal():
    info = client.get_account()
    bals = info.get('balances')

    # TODO need to find a better way of handling the readtimeout error that
    # get_all_tickers sometimes produces
    x = 0
    while x < 10:
        try:
            prices = client.get_all_tickers()
            x = 10
        except:
            x += 1
            continue
    price_dict = {x.get('symbol'): float(x.get('price')) for x in prices}

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
    usdt_size = float(balance / trade_risk)
    asset_quantity = float(usdt_size / price)

    return asset_quantity, usdt_size


def current_positions(fr):  # used to be current sizing
    '''returns a dict with assets as keys and various expressions of positioning as values'''
    total_bal = account_bal()
    # should be 1R, but also no less than min order size
    threshold_bal = max(total_bal * fr, 12)

    info = client.get_account()
    bals = info.get('balances')

    prices = client.get_all_tickers()
    price_dict = {x.get('symbol'): float(x.get('price')) for x in prices}

    size_dict = {}
    for b in bals:
        asset = b.get('asset')
        if asset in ['USDT', 'USDC', 'BUSD']:
            quant = float(b.get('free')) + float(b.get('locked'))
            value = quant
        else:
            pair = asset + 'USDT'
            price = price_dict.get(pair)
            if price == None:
                continue
            quant = float(b.get('free')) + float(b.get('locked'))
            value = price * quant
        if asset == 'BNB' and value < 20:
            continue
        elif asset == 'BNB' and value >= 20:
            value -= 10
        if value >= threshold_bal:
            pct = round(100 * value / total_bal, 5)
            size_dict[asset] = {'qty': quant,
                                'value': round(value, 2), 'pf%': pct}

    return size_dict


def free_usdt():
    usdt_bals = client.get_asset_balance(asset='USDT')
    return float(usdt_bals.get('free'))


def update_pos(asset, total_bal, inval, fixed_risk):
    '''checks for the current balance of a particular asset and returns it in 
    the correct format for the sizing dict. also calculates the open risk for 
    a given asset and returns it in R and $ denominations'''

    pair = asset + 'USDT'
    price = get_price(pair)
    bal = client.get_asset_balance(asset=asset)
    base_bal = float(bal.get('free')) + float(bal.get('locked'))
    value = price * base_bal
    pct = round(100 * value / total_bal, 5)
    open_risk = value - (value / inval)
    open_risk_r = (open_risk / total_bal) / fixed_risk

    return {'qty': base_bal, 'value': value, 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}


# Market Data Functions


def get_price(pair):
    '''returns the midpoint between first bid and ask price on the orderbook 
    for the pair in question'''
    t = client.get_orderbook_ticker(symbol=pair)
    bid = float(t.get('bidPrice'))
    ask = float(t.get('askPrice'))

    return (bid + ask) / 2


def get_spread(pair):  # possibly unused
    '''returns the proportional distance between the first bid and ask for the 
    pair in question'''

    spreads = []
    mids = []
    for j in range(3):
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol') == pair:
                bid = float(t.get('bidPrice'))
                ask = float(t.get('askPrice'))
                spreads.append(ask - bid)
                mids.append((bid + ask) / 2)
        time.sleep(0.5)

    avg_abs_spread = stats.median(spreads)
    avg_mid = stats.median(mids)

    if avg_mid > 0:
        return avg_abs_spread / avg_mid
    else:
        return 'na'


def get_depth(pair, side, max_slip=1):
    '''returns the quantity (in the quote currency) that could be bought/sold 
    within the % range of price set by the max_slip param'''

    price = get_price(pair)
    book = client.get_order_book(symbol=pair)

    if side == 'buy':
        price = float(book.get('bids')[0][0])
        # max_price is x% above price
        max_price = price * (1 + (max_slip / 100))
        depth = 0
        for i in book.get('asks'):
            if float(i[0]) <= max_price:
                depth += float(i[1])
            else:
                break
    elif side == 'sell':
        price = float(book.get('asks')[0][0])
        # max_price is x% above price
        min_price = price * (1 - (max_slip / 100))
        depth = 0
        for i in book.get('bids'):
            if float(i[0]) >= min_price:
                depth += float(i[1])
            else:
                break
    else:
        print('side param must be either buy or sell')

    usdt_depth = depth * price

    return usdt_depth


def get_depth_old(pair, side):
    '''returns the quantities (in quote denomination) of the first bid and ask 
    for the pair in question'''

    price = get_price(pair)
    try:
        bids = []
        asks = []
        for i in range(3):
            tickers = client.get_orderbook_tickers()
            for t in tickers:
                if t.get('symbol') == pair:
                    bids.append(float(t.get('bidQty')))
                    asks.append(float(t.get('askQty')))
            time.sleep(2)

        avg_bid = stats.median(bids)
        avg_ask = stats.median(asks)

        quote_bid = avg_bid * price
        quote_ask = avg_ask * price
        if side == 'buy':
            return quote_ask
        elif side == 'sell':
            return quote_bid
    except TypeError as e:
        print(e)
        print('Skipping trade - binance returned book depth of None ')
        return 0.0


def binance_spreads(quote='USDT'):
    '''returns a dictionary with pairs as keys and current average spread as values'''

    length = len(quote)
    avg_spreads = {}

    s_1 = {}
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol')[-1*length:] == quote:
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
        if t.get('symbol')[-1*length:] == quote:
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
        if t.get('symbol')[-1*length:] == quote:
            pair = t.get('symbol')
            bid = float(t.get('bidPrice'))
            ask = float(t.get('askPrice'))
            if bid and ask:
                spread = ask - bid
                mid = (ask + bid) / 2
                s_3[pair] = spread / mid

    for k in s_1:
        avg_spreads[k] = stats.median([s_1.get(k), s_2.get(k), s_3.get(k)])

    return avg_spreads


def binance_depths(quotes=['USDT', 'BTC']):
    avg_depths = {}

    for quote in quotes:
        length = len(quote)
        bd_1 = {}
        sd_1 = {}
        tickers = client.get_orderbook_tickers()
        for t in tickers:
            if t.get('symbol')[-1*length:] == quote:
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
            if t.get('symbol')[-1*length:] == quote:
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
            if t.get('symbol')[-1*length:] == quote:
                pair = t.get('symbol')
                bid = float(t.get('bidQty'))
                ask = float(t.get('askQty'))
                bd_3[pair] = ask
                sd_3[pair] = bid

        for k in bd_1:
            avg_depths[k] = {'asks': stats.median([bd_1.get(k), bd_2.get(k), bd_3.get(k)]),
                             'bids': stats.median([sd_1.get(k), sd_2.get(k), sd_3.get(k)])}

    return avg_depths


def get_pairs(quote='USDT', market='SPOT'):
    '''possible values for quote are USDT, BTC, BNB etc. possible values for 
    market are SPOT or CROSS'''
    if market == 'SPOT':
        info = client.get_exchange_info()
        symbols = info.get('symbols')
        pairs = []
        for sym in symbols:
            right_quote = sym.get('quoteAsset') == quote
            right_market = market in sym.get('permissions')
            trading = sym.get('status') == 'TRADING'
            if right_quote and right_market and trading:
                pairs.append(sym.get('symbol'))
    elif market == 'CROSS':
        pairs = []
        info = client.get_margin_all_pairs()
        for i in info:
            if i.get('quote') == quote:
                pairs.append(i.get('symbol'))

    return pairs


def get_ohlc(pair, timeframe, span="1 year ago UTC"):
    client = Client(keys.bPkey, keys.bSkey)
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
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
            'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    return df


def update_ohlc(pair, timeframe, old_df):
    client = Client(keys.bPkey, keys.bSkey)
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

    old_end = int(old_df.at[len(old_df)-1, 'timestamp'].timestamp()) * 1000
    klines = client.get_klines(symbol=pair, interval=tf.get(timeframe),
                               startTime=old_end)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
            'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    df_new = pd.concat([old_df[:-1], df], copy=True, ignore_index=True)
    return df_new


def prepare_ohlc(pair, is_live, timeframe='4H', bars=2190):
    '''checks if there is old data already, if so it loads the old data and 
    downloads an update, if not it downloads all data from scratch, then 
    resamples all data to desired timeframe'''

    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    if filepath.exists():
        try:
            df = pd.read_pickle(filepath)
            if len(df) > 2:
                df = df.iloc[:-1, ]
                df = update_ohlc(pair, '1h', df)
        except:
            print('-')
            print(f'read_pickle went wrong with {pair}, downloading old data')
            df = get_ohlc(pair, '1h', '1 year ago UTC')

    else:
        df = get_ohlc(pair, '1h', '1 year ago UTC')
        print(f'downloaded {pair} from scratch')

    if len(df) > 8760:  # 8760 is 1 year's worth of 1h periods
        df = df.tail(8760)
        df.reset_index(drop=True, inplace=True)
    if is_live:
        try:
            df.to_pickle(filepath)
        except:
            print('-')
            print(f'to_pickle went wrong with {pair}')
            print('-')

    # print(df.tail())
    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum'})
    if len(df) > bars:
        df = df.tail(bars)
    # drop=False because we want to keep the timestamp column
    df.reset_index(inplace=True)

    return df


def get_avg_price(pair):
    ticker = client.get_ticker(symbol=pair)
    price = float(ticker.get('weightedAvgPrice'))
    vol = float(ticker.get('quoteVolume'))
    # print(f'{pair} {price} - {vol}')

    return price, vol


def get_avg_prices(quote='USDT'):
    tickers_24h = client.get_ticker()  # no symbol specified so all symbols returned
    qlen = len(quote) * -1
    waps = {}
    for i in tickers_24h:
        pair = i.get('symbol')
        if pair[qlen:] == quote:
            price = float(i.get('weightedAvgPrice'))
            vol = float(i.get('quoteVolume'))
            waps['pair'] = [price, vol]

    return waps


# Trading Functions


def top_up_bnb(usdt_size):
    bnb_bal = client.get_asset_balance(asset='BNB')
    free_bnb = float(bnb_bal.get('free'))
    avg_price = client.get_avg_price(symbol='BNBUSDT')
    price = float(avg_price.get('price'))
    bnb_value = free_bnb * price

    info = client.get_symbol_info('BNBUSDT')
    step_size = info.get('filters')[2].get('stepSize')

    usdt_bal = client.get_asset_balance(asset='USDT')
    free_usdt = float(usdt_bal.get('free'))
    bnb_size = step_round(usdt_size / price, step_size)
    if bnb_value < 5 and free_usdt > usdt_size:
        print('Topping up BNB')
        order = client.create_order(symbol='BNBUSDT',
                                    side=enums.SIDE_BUY,
                                    type=enums.ORDER_TYPE_MARKET,
                                    quantity=bnb_size)

    else:
        # print(f'Didnt top up BNB, current val: {bnb_value:.3} USDT, free usdt: {free_usdt:.2f} USDT')
        order = None
    return order


def buy_asset(pair, usdt_size, live):
    # print(f'buying {pair}')

    # calculate how much of the asset to buy
    usdt_price = get_price(pair)
    size = usdt_size / usdt_price

    # make sure order size has the right number of decimal places
    info = client.get_symbol_info(pair)
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    order_size = step_round(size, step_size)
    # print(f'{pair} Buy Order - raw size: {size:.5}, step size: {step_size:.2}, final size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=enums.SIDE_BUY,
                                    type=enums.ORDER_TYPE_MARKET,
                                    quantity=order_size)
        fills = order.get('fills')
        fee = 0
        exe_prices = []
        for fill in fills:
            fee += float(fill.get('commission'))
            exe_prices.append(float(fill.get('price')))
        avg_price = stats.mean(exe_prices)

        trade_dict = {'timestamp': order.get('transactTime'),
                      'pair': order.get('symbol'),
                      'trig_price': usdt_price,
                      'exe_price': avg_price,
                      'base_size': float(order.get('executedQty')),
                      'quote_size': float(order.get('cummulativeQuoteQty')),
                      'fee': fee,
                      'fee_currency': fills[0].get('commissionAsset')
                      }
        if order.get('status') != 'FILLED':
            print(f'{pair} order not filled')
            pb.push_note('Warning', f'{pair} order not filled')

    else:
        trade_dict = {"pair": pair, 
                     "trig_price": usdt_price,
                     "base_size": float(order_size),
                     "quote_size": usdt_size,
                     "fee": 0,
                     "fee_currency": "BNB"
                     }
    
    # print('-')
    return trade_dict


def sell_asset(pair, live, pct=100):
    # print(f'selling {pair}')
    asset = pair[:-4]
    usdt_price = get_price(pair)

    # request asset balance from binance
    bal = client.get_asset_balance(asset=asset)
    if asset == 'BNB':
        reserve = 10 / usdt_price  # amount of bnb to reserve ($10 worth)
        asset_bal = float(bal.get('free')) - reserve  # always keep $10 of bnb
    else:
        asset_bal = float(bal.get('free'))

    # make sure order size has the right number of decimal places
    trade_size = asset_bal * (pct / 100)
    info = client.get_symbol_info(pair)
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    order_size = step_round(trade_size, step_size)  # - step_size
    # print(f'{pair} Sell Order - raw size: {asset_bal:.5}, step size: {step_size:.2}, final size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=enums.SIDE_SELL,
                                    type=enums.ORDER_TYPE_MARKET,
                                    quantity=order_size)
        fills = order.get('fills')
        fee = 0
        exe_prices = []
        for fill in fills:
            fee += float(fill.get('commission'))
            exe_prices.append(float(fill.get('price')))
        avg_price = stats.mean(exe_prices)

        trade_dict = {'timestamp': order.get('transactTime'),
                      'pair': order.get('symbol'),
                      'trig_price': usdt_price,
                      'exe_price': avg_price,
                      'base_size': float(order.get('executedQty')),
                      'quote_size': float(order.get('cummulativeQuoteQty')),
                      'fee': fee,
                      'fee_currency': fills[0].get('commissionAsset')
                      }
        if order.get('status') != 'FILLED':
            print(f'{pair} order not filled')
            pb.push_note('Warning', f'{pair} order not filled')

    else:
        trade_dict = {"pair": pair, 
                    "trig_price": usdt_price,
                    "base_size": float(order_size),
                    "quote_size": float(order_size) * usdt_price,
                    "fee": 0,
                    "fee_currency": "BNB"
                    }

    # print('-')
    return trade_dict


def set_stop(pair, price, live):
    # print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]

    info = client.get_symbol_info(pair)
    tick_size = info.get('filters')[0].get('tickSize')
    step_size = Decimal(info.get('filters')[2].get('stepSize'))

    reserve = 10 / price  # amount of asset that would be worth $10 at stop price

    bal = client.get_asset_balance(asset=asset)
    if asset == 'BNB':
        asset_bal = float(bal.get('free')) - reserve  # always keep $10 of bnb
    else:
        asset_bal = float(bal.get('free'))

    info = client.get_symbol_info(pair)
    order_size = step_round(asset_bal, step_size)  # - step_size
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 30))
    trigger_price = step_round(price, tick_size)
    limit_price = step_round(lower_price, tick_size)
    # print(f'{pair} Stop Order - trigger: {trigger_price:.5}, limit: {limit_price:.5}, size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=enums.SIDE_SELL,
                                    type=enums.ORDER_TYPE_STOP_LOSS_LIMIT,
                                    timeInForce=enums.TIME_IN_FORCE_GTC,
                                    stopPrice=trigger_price,
                                    quantity=order_size,
                                    price=limit_price)
    else:
        order = {"pair": pair, 
                "trig_price": float(trigger_price),
                "base_size": float(order_size),
                "quote_size": float(order_size) * float(trigger_price),
                "fee": 0,
                "fee_currency": "BNB"
                }

    # print('-')
    return order


def clear_stop(pair, live):
    '''blindly cancels the first resting order relating to the pair in question.
    works as a "clear stop" function only when the strategy sets one 
    stop-loss per position and uses no other resting orders'''

    # sanity check
    bal = client.get_asset_balance(asset=pair[:-4])
    if float(bal.get('locked')) == 0:
        print('no stop to cancel')
    else:
        # print(f'cancelling {pair} stop')
        orders = client.get_open_orders(symbol=pair)
        if live:
            if orders:
                ord_id = orders[0].get('orderId')
                result = client.cancel_order(symbol=pair, orderId=ord_id)
                # print(result.get('status'))
            else:
                print('no stop to cancel')
            # print('-')
        else:
            print('simulated canceling stop')


def reduce_risk_old(pos_open_risk, r_limit, live):
    positions = []
    trade_notes = []

    # create a list of open positions in profit and their open risk value
    for p, r in pos_open_risk.items():
        if r.get('R') > 1:
            positions.append((p, r.get('R')))

    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)

        # # create a new list with just the R values
        r_list = [x.get('R') for x in pos_open_risk.values()]
        total_r = sum(r_list)

        for pos in sorted_pos:
            if total_r > r_limit:
                pair = pos[0] + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = get_price(pair)
                note = f"*** sell {pair} @ {price}"
                print(now, note)
                if live:
                    push = pb.push_note(now, note)
                    clear_stop(pair)
                    sell_order = sell_asset(pair)
                    sell_order['type'] = 'close_long'
                    sell_order['reason'] = 'portfolio risk limiting'
                    trade_notes.append(sell_order)
                    total_r -= pos[1]
                    

    return trade_notes

def reduce_risk(sizing, signals, params, live):
    r_limit = params.get('total_r_limit')
    fixed_risk = params.get('fixed_risk')
    
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R')) for p, r in sizing.items() if r.get('or_R') and r.get('or_R') > 0]
    
    # for p, r in sizing.items():
    #     positions.append((p, r.get('or_R')))

    trade_notes = []
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in sizing.values() if x.get('or_R')]
        total_r = sum(r_list)
        num_pos = len(r_list)

        for pos in sorted_pos:
            if (total_r > r_limit or num_pos >= r_limit) and pos[1] > 1.5:
                pair = pos[0] + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = get_price(pair)
                note = f"reduce risk {pair} @ {price}"
                print(now, note)
                if live:
                    push = pb.push_note(now, note)
                    clear_stop(pair, live)
                    sell_order = sell_asset(pair, live)
                    sell_order['type'] = 'close_long'
                    sell_order['reason'] = 'portfolio risk limiting'
                    trade_notes.append(sell_order)
                    total_r -= pos[1]
                    del sizing[pos[0]]
                else:
                    push = pb.push_note(now, f'sim reduce risk {pair}')
                    clear_stop(pair, live)
                    sell_order = sell_asset(pair, live)
                    sell_order['type'] = 'close_long'
                    sell_order['reason'] = 'portfolio risk limiting'
                    trade_notes.append(sell_order)
                    total_r -= pos[1]
                    del sizing[pos[0]]
            else:
                break

    return sizing, trade_notes
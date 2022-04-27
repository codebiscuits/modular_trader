import keys, math, time, json
import statistics as stats
import pandas as pd
import numpy as np
from binance.client import Client
import binance.enums as be
from pushbullet import Pushbullet
from decimal import Decimal
from pprint import pprint
from config import ohlc_data#, market_data
from pathlib import Path
from datetime import datetime
import utility_funcs as uf

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


#-#-#- Utility Functions

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
    return df


def order_by_volsm(pairs, lookback):
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
                price_range = ((high - low) / mid)#**2
                # volatility = df.roc.std()
                volatility = df.roc_diff.mean()
                smoothness = round(10 * price_range / volatility)
            except ValueError as e:
                print(pair, e)
            tup_list.append((pair, smoothness))
    
    # sort tup_list by vol
    tup_list = sorted(tup_list, key=lambda x: x[1], reverse=True)
    
    return tup_list


def get_size(price, fr, balance, risk):
    if fr:
        trade_risk = risk / fr
        usdt_size = float(balance / trade_risk)
    else:
        trade_risk = 0
        usdt_size = 0
    
    asset_quantity = float(usdt_size / price)

    return asset_quantity, usdt_size


def calc_stop(st, spread,  price, min_risk=0.01):
    '''calculates what the stop-loss trigger price should be based on the current 
    value of the supertrend line and the current spread (for slippage). 
    if this is too close to the entry price, the stop will be set at the minimum 
    allowable distance. 
    I would like to make min_risk part of the adaptive settings, so if too 
    many positions are getting stopped out too early, the system can increase 
    it automatically'''
    buffer = max(spread * 2, min_risk) 
    if price > st:
        return float(st) * (1 - buffer)
    else:
        return float(st) * (1 + buffer)


def calc_fee_bnb(usdt_size, fee_rate=0.00075):
    bnb_price = get_price('BNBUSDT')
    fee_usdt = float(usdt_size) * fee_rate
    return fee_usdt / bnb_price


#-#-#- Account Functions


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


def current_positions(strat, switch:str):
    '''creates a dictionary of open positions by checking either 
    open_trades.json, sim_trades.json or tracked_trades.json'''
        
    filepath = Path(f'{strat.market_data}/{strat.name}_{switch}_trades.json')
    with open(filepath, 'r') as file:
        try:
            data = json.load(file)
        except:
            data = {}
    
    size_dict = {}
    now = datetime.now()
    total_bal = account_bal()
    
    for k, v in data.items():
        if switch == 'open':
            asset = k[:-4]
            size_dict[asset] = uf.open_trade_stats(now, total_bal, v)
        elif switch == 'sim':
            asset = v[0].get('pair')[:-4]
            size_dict[asset] = uf.open_trade_stats(now, total_bal, v)
        elif switch == 'tracked':
            asset = v[0].get('pair')[:-4]
            size_dict[asset] = {}
    
    return size_dict
        

def current_positions_old(market_data, strat_name, fr):  # used to be current sizing
    '''returns a dict with assets as keys and various expressions of positioning as values'''
    
    o_path = Path(f'{market_data}/{strat_name}_open_trades.json')
    with open(o_path, 'r') as file:
        try:
            o_data = json.load(file)
        except:
            o_data = {}
        
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
        if asset not in ['USDT', 'USDC', 'BUSD']:
        #     quant = float(b.get('free')) + float(b.get('locked'))
        #     value = quant
        # else:
            pair = asset + 'USDT'
            price = price_dict.get(pair)
            if price == None:
                continue
            if pair in o_data.keys():
                now = datetime.now()
                ots = uf.open_trade_stats(now, total_bal, pair, o_data.get(pair))
                quant = float(b.get('free')) + float(b.get('locked'))
                value = price * quant
                if asset == 'BNB' and value < 20:
                    continue
                elif asset == 'BNB' and value >= 20:
                    value -= 10
                if value >= threshold_bal:
                    pct = round(100 * value / total_bal, 5)
                    size_dict[asset] = {'qty': quant,
                                        'value': round(value, 2), 'pf%': pct,
                                        'pnl_R': ots.get('pnl_R'),
                                        'pnl_%': ots.get('pnl_%')}

    return size_dict


def free_usdt():
    usdt_bals = client.get_asset_balance(asset='USDT')
    return float(usdt_bals.get('free'))


def update_pos(strat, asset, base_bal, inval, pos_fr_dol):
    '''checks for the current balance of a particular asset and returns it in 
    the correct format for the sizing dict. also calculates the open risk for 
    a given asset and returns it in R and $ denominations'''

    pair = asset + 'USDT'
    price = get_price(pair)
    value = price * float(base_bal)
    pct = round(100 * value / strat.bal, 5)
    open_risk = value - (value / inval)
    if pos_fr_dol:
        open_risk_r = open_risk / pos_fr_dol
    else:
        open_risk_r = 0

    return {'qty': base_bal, 'value': value, 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}


def update_pos_old(asset, total_bal, inval, pos_fr_dol):
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
    if pos_fr_dol:
        open_risk_r = open_risk / pos_fr_dol
    else:
        open_risk_r = 0

    return {'qty': base_bal, 'value': value, 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}


def update_usdt(total_bal):
    '''checks current usdt balance and returns a dictionary for updating the sizing dict'''
    bal = client.get_asset_balance(asset='USDT')
    base_bal = float(bal.get('free')) + float(bal.get('locked'))
    value = round(base_bal, 2)
    pct = round(100 * value / total_bal, 5)
    
    return {'qty': base_bal, 'value': value, 'pf%': pct}


#-#-#- Market Data Functions


def get_price(pair):
    return float(client.get_ticker(symbol=pair).get('lastPrice'))


def get_mid_price(pair):
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

    if is_live:
        filepath = Path(f'{ohlc_data}/{pair}.pkl')
    else:
        filepath = Path(f'bin_ohlc/{pair}.pkl')
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

    if len(df) > 17520:  # 17520 is 2 year's worth of 1h periods
        df = df.tail(17520)
        df.reset_index(drop=True, inplace=True)
    try:
        df.to_pickle(filepath)
    except:
        print('-')
        print(f'to_pickle went wrong with {pair}')
        print('-')

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


#-#-#- Trading Functions


def top_up_bnb(usdt_size):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    # calculate bnb balance in usdt
    bnb_bal = client.get_asset_balance(asset='BNB')
    free_bnb = float(bnb_bal.get('free'))
    avg_price = client.get_avg_price(symbol='BNBUSDT')
    price = float(avg_price.get('price'))
    bnb_value = free_bnb * price

    # get free usdt balance
    usdt_bal = client.get_asset_balance(asset='USDT')
    free_usdt = float(usdt_bal.get('free'))
    
    if bnb_value < 5:
        if free_usdt > usdt_size:
            pb.push_note(now, 'Topping up BNB')
            # round size to proper step size
            info = client.get_symbol_info('BNBUSDT')
            step_size = info.get('filters')[2].get('stepSize')
            bnb_size = step_round(usdt_size / price, step_size)
            # send order
            order = client.create_order(symbol='BNBUSDT',
                                        side=be.SIDE_BUY,
                                        type=be.ORDER_TYPE_MARKET,
                                        quantity=bnb_size)
        else:
            pb.push_note(now, 'Warning - BNB balance low and not enough USDT to top up')

    else:
        # print(f'Didnt top up BNB, current val: {bnb_value:.3} USDT, free usdt: {free_usdt:.2f} USDT')
        order = None
    return order


def create_trade_dict(order, price, live):
    '''collects and returns the details of the order in a dictionary'''
    pair = order.get('symbol')
    if live:
        fills = order.get('fills')
        fee = sum([Decimal(fill.get('commission')) for fill in fills])
        qty = sum([Decimal(fill.get('qty')) for fill in fills])
        exe_prices = [Decimal(fill.get('price')) for fill in fills]
        avg_price = stats.mean(exe_prices)

        trade_dict = {'timestamp': order.get('transactTime'),
                      'pair': pair,
                      'trig_price': str(price),
                      'exe_price': str(avg_price),
                      'base_size': order.get('executedQty'),
                      'quote_size': order.get('cummulativeQuoteQty'),
                      'fee': str(fee),
                      'fee_currency': fills[0].get('commissionAsset')
                      }
        if order.get('status') != 'FILLED':
            print(f'{pair} order not filled')
            pb.push_note('Warning', f'{pair} order not filled')

    else:
        trade_dict = {"pair": pair, 
                     "trig_price": str(price),
                     "exe_price": str(price),
                     "base_size": str(order.get('base_size')),
                     "quote_size": str(order.get('quote_size')),
                     "fee": '0',
                     "fee_currency": "BNB"
                     }
    
    return trade_dict


def valid_size(pair, size):
    '''rounds the desired order size to the correct step size for binance'''
    info = client.get_symbol_info(pair)
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    
    return step_round(size, step_size)


def buy_asset(pair, usdt_size, live):
    '''takes the pair and the dollar value of the desired position, and 
    calculates the exact amount to order. then executes a market buy.'''

    # calculate the exact size of the order
    usdt_price = get_price(pair)
    order_size = valid_size(pair, usdt_size/usdt_price)
    print(f'order size: {order_size}')
    
    # send the order
    if live:
        order = client.create_order(symbol=pair,
                                    side=be.SIDE_BUY,
                                    type=be.ORDER_TYPE_MARKET,
                                    quantity=order_size)
        
    else:
        order = {'symbol': pair, 'price': get_price(pair), 
                 'base_size': order_size, 'quote_size': usdt_size}
    
    return order


def sell_asset(pair, asset_bal, live, pct=100):
    # print(f'selling {pair}')
    asset = pair[:-4]
    usdt_price = get_price(pair)

    # make sure order size has the right number of decimal places
    trade_size = Decimal(asset_bal) * Decimal(pct / 100)
    order_size = valid_size(pair, trade_size)
    # print(f'{pair} Sell Order - raw size: {asset_bal:.5}, step size: {step_size:.2}, final size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=be.SIDE_SELL,
                                    type=be.ORDER_TYPE_MARKET,
                                    quantity=order_size)
        
    else:
        order = {'symbol': pair, 'price': usdt_price, 
                 'base_size': order_size, 'quote_size': (order_size*Decimal(usdt_price))}
        
    return order


def sell_asset_old(pair, live, pct=100):
    # print(f'selling {pair}')
    asset = pair[:-4]
    usdt_price = get_price(pair)

    # request asset balance from binance
    bal = client.get_asset_balance(asset=asset)
    if asset == 'BNB':
        reserve = 10 / usdt_price  # amount of bnb to reserve ($10 worth)
        asset_bal = Decimal(bal.get('free')) - reserve  # always keep $10 of bnb
    else:
        asset_bal = Decimal(bal.get('free'))

    # make sure order size has the right number of decimal places
    trade_size = asset_bal * Decimal(pct / 100)
    order_size = valid_size(pair, trade_size)
    # print(f'{pair} Sell Order - raw size: {asset_bal:.5}, step size: {step_size:.2}, final size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=be.SIDE_SELL,
                                    type=be.ORDER_TYPE_MARKET,
                                    quantity=order_size)
        
    else:
        order = {'symbol': pair, 'price': usdt_price, 
                 'base_size': order_size, 'quote_size': (order_size*Decimal(usdt_price))}
        
    return order


def set_stop(pair, base_size, price, live):
    # print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]

    info = client.get_symbol_info(pair)
    tick_size = info.get('filters')[0].get('tickSize')
    step_size = Decimal(info.get('filters')[2].get('stepSize'))

    reserve = 10 / price  # amount of BNB that would be worth $10 at stop price

    order_size = step_round(base_size, step_size)  # - step_size
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 30))
    trigger_price = step_round(price, tick_size)
    limit_price = step_round(lower_price, tick_size)
    # print(f'{pair} Stop Order - trigger: {trigger_price:.5}, limit: {limit_price:.5}, size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=be.SIDE_SELL,
                                    type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                    timeInForce=be.TIME_IN_FORCE_GTC,
                                    stopPrice=trigger_price,
                                    quantity=order_size,
                                    price=limit_price)
    else:
        order = {"pair": pair, 
                "trig_price": Decimal(trigger_price),
                "base_size": Decimal(order_size),
                "quote_size": Decimal(order_size) * Decimal(trigger_price),
                "fee": 0,
                "fee_currency": "BNB"
                }

    # print('-')
    return order


def set_stop_old(pair, price, live):
    # print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]

    info = client.get_symbol_info(pair)
    tick_size = info.get('filters')[0].get('tickSize')
    step_size = Decimal(info.get('filters')[2].get('stepSize'))

    reserve = 10 / price  # amount of BNB that would be worth $10 at stop price

    bal = client.get_asset_balance(asset=asset)
    if asset == 'BNB':
        asset_bal = Decimal(bal.get('free')) - reserve  # always keep $10 of bnb
    else:
        asset_bal = Decimal(bal.get('free'))

    order_size = step_round(asset_bal, step_size)  # - step_size
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 30))
    trigger_price = step_round(price, tick_size)
    limit_price = step_round(lower_price, tick_size)
    # print(f'{pair} Stop Order - trigger: {trigger_price:.5}, limit: {limit_price:.5}, size: {order_size:.5}')

    if live:
        order = client.create_order(symbol=pair,
                                    side=be.SIDE_SELL,
                                    type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                    timeInForce=be.TIME_IN_FORCE_GTC,
                                    stopPrice=trigger_price,
                                    quantity=order_size,
                                    price=limit_price)
    else:
        order = {"pair": pair, 
                "trig_price": Decimal(trigger_price),
                "base_size": Decimal(order_size),
                "quote_size": Decimal(order_size) * Decimal(trigger_price),
                "fee": 0,
                "fee_currency": "BNB"
                }

    # print('-')
    return order


def clear_stop(pair, trade_record, live):
    '''finds the order id of the most recent stop-loss from the trade record
    and cancels that specific order. if no such id can be found, blindly cancels 
    the most recent stop-limit order relating to the pair'''

    # sanity check
    bal = client.get_asset_balance(asset=pair[:-4])
    if Decimal(bal.get('locked')) == 0:
        print('no stop to cancel')
    else:
        print(f'{pair} locked balance = {bal.get("locked")}')
        _, stop_id, _ = uf.latest_stop_id(trade_record)
        
        if not stop_id: # if the function above didn't find anything
            orders = client.get_open_orders(symbol=pair)
            if orders and orders[-1].get('type') == 'STOP_LOSS_LIMIT':
                stop_id = orders[-1].get('orderId')
        
        if live and stop_id:
            result = client.cancel_order(symbol=pair, orderId=stop_id)


def clear_stop_old(pair, live):
    '''blindly cancels the first resting order relating to the pair in question.
    works as a "clear stop" function only when the strategy sets one 
    stop-loss per position and uses no other resting orders'''

    # sanity check
    bal = client.get_asset_balance(asset=pair[:-4])
    if Decimal(bal.get('locked')) == 0:
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


#-#-#- Margin Account Functions


def account_bal_M():
    info = client.get_margin_account()
    total_net = float(info.get('totalNetAssetOfBtc'))
    btc_price = get_price('BTCUSDT')
    usdt_total_net = total_net * btc_price
    
    return round(usdt_total_net, 2)


def current_positions_M(strat, fr):  # used to be current sizing
    '''returns a dict with assets as keys and various expressions of positioning as values'''
    
    o_path = Path(f'{strat.market_data}/{strat}_open_trades.json')
    with open(o_path, 'r') as file:
        try:
            o_data = json.load(file)
        except:
            o_data = {}
        
    total_bal = account_bal_M()
    # should be 1R, but also no less than min order size
    threshold_bal = max(total_bal * fr, 12)

    info = client.get_margin_account()
    assets = info.get('userAssets')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol'): float(x.get('price')) for x in prices}

    size_dict = {}
    for a in assets:        
        asset = a.get('asset')
        if asset not in ['USDT', 'USDC', 'BUSD']:
            pair = asset + 'USDT'
            price = price_dict.get(pair)
            if price == None:
                continue
            if pair in o_data.keys():
                now = datetime.now()
                ots = uf.open_trade_stats(now, pair, o_data.get(pair))
                quant = float(a.get('netAsset'))
                value = price * quant
                if asset == 'BNB' and value < 20:
                    continue
                elif asset == 'BNB' and value >= 20:
                    value -= 10
                if value >= threshold_bal:
                    pct = round(100 * value / total_bal, 5)
                    size_dict[asset] = {'qty': quant,
                                        'value': round(value, 2), 'pf%': pct,
                                        'pnl': ots.get('pnl_R')}

    return size_dict


def free_usdt_M():
    info = client.get_margin_account()
    assets = info.get('userAssets')
    bal = 0
    for a in assets:
        if a.get('asset') == 'USDT':
            bal = float(a.get('free'))
    return bal


def asset_bal_M(asset):
    info = client.get_margin_account()
    bals = info.get('userAssets')
    
    balance = 0
    for bal in bals:
        if bal.get('asset') == asset:
            balance = bal.get('netAsset')
    
    return balance


def free_bal_M(asset):
    info = client.get_margin_account()
    bal = 0
    for i in info.get('userAssets'):
        if i.get('asset') == asset:
            bal = i.get('free')
            
    return bal


def update_pos_M(asset, total_bal, inval, pos_fr_dol):
    '''checks for the current balance of a particular asset and returns it in 
    the correct format for the sizing dict. also calculates the open risk for 
    a given asset and returns it in R and $ denominations'''

    pair = asset + 'USDT'
    price = get_price(pair)
    bal = asset_bal_M(asset)
    value = price * bal
    pct = round(100 * value / total_bal, 5)
    open_risk = value - (value / inval)
    open_risk_r = open_risk / pos_fr_dol

    return {'qty': bal, 'value': value, 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}


def update_usdt_M(total_bal):
    '''checks current usdt balance and returns a dictionary for updating the sizing dict'''
    bal = asset_bal_M('USDT')
    value = round(bal, 2)
    pct = round(100 * value / total_bal, 5)
    
    return {'qty': bal, 'value': value, 'pf%': pct}


#-#-#- Margin Trading Functions


def top_up_bnb_M(usdt_size):
    '''checks net BNB balance and interest owed, if net is below the threshold,
    buys BNB then repays any interest'''
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    # check balances
    info = client.get_margin_account()
    assets = info.get('userAssets')
    free_bnb = 0
    interest = 0
    free_usdt = 0
    for a in assets:
        if a.get('asset') == 'BNB':
            free_bnb = float(a.get('free'))
            interest = float(a.get('interest'))
            print(interest)
        if a.get('asset') == 'USDT':
            free_usdt = float(a.get('free'))
    net_bnb = free_bnb - interest
    
    # calculate value
    avg_price = client.get_avg_price(symbol='BNBUSDT')
    price = float(avg_price.get('price'))
    bnb_value = net_bnb * price
    
    # top up if needed
    info = client.get_symbol_info('BNBUSDT')
    # step_size = info.get('filters')[2].get('stepSize') # might not be needed
    # bnb_size = step_round(usdt_size / price, step_size)
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
    if interest > 0:
        repay = client.repay_margin_loan(asset='BNB', amount=str(interest))
    else:
        repay = None
    
    return order, repay


def open_long(trade_pair, usdt_size):
    # borrow usdt
    print(f'amount: {usdt_size}')
    usdt_borrow = client.create_margin_loan(asset='USDT', amount=usdt_size)
    # pprint(usdt_borrow)
    
    # execute trade
    long_order = client.create_margin_order(
        symbol=trade_pair,
        side=be.SIDE_BUY,
        type=be.ORDER_TYPE_MARKET,
        quoteOrderQty=usdt_size)
    # pprint(long_order)
    
    return long_order


def close_long(trade_pair, pct=1):
    asset = trade_pair[:-4]
    
    # calculate size
    bal = free_bal_M(asset)
    order_size = valid_size(trade_pair, float(bal)*pct)
    
    # execute trade
    order = client.create_margin_order(
        symbol=trade_pair,
        side=be.SIDE_SELL,
        type=be.ORDER_TYPE_MARKET,
        quantity=order_size)
    pprint(order)
    
    # repay loan
    info = client.get_margin_account()
    for i in info.get('userAssets'):
        if i.get('asset') == 'USDT':
            fre = i.get('free')
            bor = i.get('borrowed')
    max_repay = min(float(fre), float(bor))
    usdt_repay = client.repay_margin_loan(asset='USDT', amount=max_repay)
    
    return order


def set_stop_M(pair, order, side, trigger, limit):
    info = client.get_symbol_info(pair)
    tick_size = info.get('filters')[0].get('tickSize')
    stop_size = Decimal(order.get('executedQty'))
    stop_sell_order = client.create_margin_order(
        symbol=pair,
        side=side,
        type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
        timeInForce=be.TIME_IN_FORCE_GTC,
        stopPrice=str(step_round(trigger, tick_size)),
        quantity=stop_size,
        price=str(step_round(limit, tick_size)))
    pprint(stop_sell_order)
    
    return stop_sell_order


def clear_stop_M(pair):
    orders = client.get_all_margin_orders(symbol=pair)
    if orders[-1].get('type') == 'STOP_LOSS_LIMIT':
        stop_id = orders[-1].get('orderId')
        client.cancel_margin_order(symbol=pair, orderId=stop_id)
    else:
        print('no stop to clear')



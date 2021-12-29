import keys
import math
import time
import statistics as stats
import pandas as pd
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


### Utility Functions

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
    df.reset_index(inplace=True) # don't use drop=True because i want the 
    # timestamp index back as a column

### Account Functions


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
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
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
    usdt_size = balance / trade_risk
    asset_quantity = usdt_size / price
    
    return asset_quantity, usdt_size

def current_positions(fr):
    total_bal = account_bal()
    threshold_bal = total_bal * fr # asset balances below this value are considered dust
    # also, this is 1R, ie the amount of money lost from entry to stop
    
    info = client.get_account()
    bals = info.get('balances')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
    pos_dict = {}
    for b in bals:        
        asset = b.get('asset')
        if asset == 'USDT':
            continue
        pair = asset + 'USDT'
        price = price_dict.get(pair)
        if price == None:
            continue
        quant = float(b.get('free')) + float(b.get('locked'))
        value = price * quant # dollar value of the position
        if asset == 'BNB':
            if value >= threshold_bal and value > 15:
                pos_dict[pair] = value / total_bal
            else:
                pos_dict[pair] = 0
        else:
            if value >= threshold_bal and value > 10:
                pos_dict[pair] = value / total_bal
            else:
                pos_dict[pair] = 0
            
    return pos_dict

def current_sizing(fr):
    '''returns a dict with assets as keys and asset value as a proportion of total as values'''
    total_bal = account_bal()
    threshold_bal = total_bal * fr
    
    info = client.get_account()
    bals = info.get('balances')
    
    prices = client.get_all_tickers()
    price_dict = {x.get('symbol') : float(x.get('price')) for x in prices}
    
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
        if value >= threshold_bal:
            pct = round(value / total_bal, 5)
            size_dict[asset] = {'qty': quant, 'allocation': pct}
            
    return size_dict

def free_usdt():
    usdt_bals = client.get_asset_balance(asset='USDT')
    return float(usdt_bals.get('free'))


### Market Data Functions


def get_price(pair):
    '''returns the first ask price on the orderbook for the pair in question, 
    meant for buy orders'''
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol') == pair:
            usdt_price = float(t.get('askPrice'))
    return usdt_price

def get_spread(pair): # possibly unused
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
    
def get_depth(pair, side):
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

def get_depth_unused(pair):
    usdt_depth = 0
    tickers = client.get_orderbook_tickers()
    for t in tickers:
        if t.get('symbol') == pair:
            usdt_price = float(t.get('askPrice'))
            asset_qty = float(t.get('askQty'))
            usdt_depth = usdt_price * asset_qty
            
    return usdt_depth
    
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
    market are SPOT or MARGIN'''
    info = client.get_exchange_info()
    symbols = info.get('symbols')
    pairs = []
    for sym in symbols:
        right_quote = sym.get('quoteAsset') == quote
        right_market = market in sym.get('permissions')
        trading = sym.get('status') == 'TRADING'
        if right_quote and right_market and trading:
            pairs.append(sym.get('symbol'))
    
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

def prepare_ohlc(pair, timeframe='4H', bars=2190):
    '''checks if there is old data already, if so it loads the old data and 
    downloads an update, if not it downloads all data from scratch, then 
    resamples all data to desired timeframe'''
    
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    if filepath.exists():
        try:
            df = pd.read_pickle(filepath)
            if len(df) > 2:
                df = df.iloc[:-1,]
                df = update_ohlc(pair, '1h', df)
        except:
            print('-')
            print(f'read_pickle went wrong with {pair}, downloading old data')
            df = get_ohlc(pair, '1h', '1 year ago UTC')
        
    else:
        df = get_ohlc(pair, '1h', '1 year ago UTC')
        print(f'downloaded {pair} from scratch')

    if len(df) > 8760: # 8760 is 1 year's worth of 1h periods
        df = df.tail(8760)
        df.reset_index(drop=True, inplace=True)
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
    df.reset_index(inplace=True) # drop=False because we want to keep the timestamp column
    
    return df

def get_avg_price(pair):
    ticker = client.get_ticker(symbol=pair)
    price = float(ticker.get('weightedAvgPrice'))
    vol = float(ticker.get('quoteVolume'))
    # print(f'{pair} {price} - {vol}')
    
    return price, vol

def get_avg_prices(quote='USDT'):
    tickers_24h = client.get_ticker() # no symbol specified so all symbols returned
    qlen = len(quote) * -1
    waps = {}
    for i in tickers_24h:
        pair = i.get('symbol')
        if pair[qlen:] == quote:
            price = float(i.get('weightedAvgPrice'))
            vol = float(i.get('quoteVolume'))
            waps['pair'] = [price, vol]
    
    return waps


### Trading Functions


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
        
def buy_asset(pair, usdt_size):
    # print(f'buying {pair}')
    
    # calculate how much of the asset to buy
    usdt_price = get_price(pair)
    size = usdt_size / usdt_price
    
    # make sure order size has the right number of decimal places
    info = client.get_symbol_info(pair)
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    order_size = step_round(size, step_size)
    # print(f'{pair} Buy Order - raw size: {size:.5}, step size: {step_size:.2}, final size: {order_size:.5}')
    
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
                  'side': order.get('side'), 
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
    # print('-')
    return trade_dict

def sell_asset(pair, pct=100):
    # print(f'selling {pair}')
    asset = pair[:-4]
    usdt_price = get_price(pair)
    
    # request asset balance from binance
    info = client.get_account()
    bals = info.get('balances')
    for b in bals:
        if b.get('asset') == asset:
            if asset == 'BNB':
                reserve = 10 / usdt_price # amount of bnb to reserve ($10 worth)
                asset_bal = float(b.get('free')) - reserve
            else:
                asset_bal = float(b.get('free'))
    
    # make sure order size has the right number of decimal places
    trade_size = asset_bal * (pct / 100)
    info = client.get_symbol_info(pair)
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    order_size = step_round(trade_size, step_size)# - step_size
    # print(f'{pair} Sell Order - raw size: {asset_bal:.5}, step size: {step_size:.2}, final size: {order_size:.5}')
    
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
                  'side': order.get('side'), 
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
    # print('-')
    return trade_dict

def set_stop(pair, price):
    # print(f'setting {pair} stop @ {price}')
    asset = pair[:-4]
    
    info = client.get_symbol_info(pair)
    tick_size = info.get('filters')[0].get('tickSize')
    step_size = Decimal(info.get('filters')[2].get('stepSize'))
    
    info = client.get_account()
    bals = info.get('balances')
    for b in bals:
        if b.get('asset') == asset:
            if asset == 'BNB':
                asset_bal = float(b.get('free')) * 0.9 # always keep a bit of bnb
            else:
                asset_bal = float(b.get('free'))
    
    info = client.get_symbol_info(pair)
    order_size = step_round(asset_bal, step_size)# - step_size
    spread = get_spread(pair)
    lower_price = price * (1 - (spread * 10))
    trigger_price = step_round(price, tick_size)
    limit_price = step_round(lower_price, tick_size)
    # print(f'{pair} Stop Order - trigger: {trigger_price:.5}, limit: {limit_price:.5}, size: {order_size:.5}')
    
    order = client.create_order(symbol=pair, 
                                side=enums.SIDE_SELL, 
                                type=enums.ORDER_TYPE_STOP_LOSS_LIMIT, 
                                timeInForce=enums.TIME_IN_FORCE_GTC, 
                                stopPrice=trigger_price,
                                quantity=order_size, 
                                price=limit_price)
    
    
    # print('-')
    return order

def clear_stop(pair):
    '''blindly cancels the first resting order relating to the pair in question.
    works as a "clear stop" function only when the strategy sets one 
    stop-loss per position and uses no other resting orders'''
    
    # print(f'cancelling {pair} stop')
    orders = client.get_open_orders(symbol=pair)
    if orders:
        ord_id = orders[0].get('orderId')
        result = client.cancel_order(symbol=pair, orderId=ord_id)
        # print(result.get('status'))
    else:
        print('no stop to cancel')
    # print('-')

def reduce_risk(pos_open_risk, r_limit, live):
    positions = []
    trade_notes = []
    for p, r in pos_open_risk.items():
        if r >1:
            positions.append((p, r))
    sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
    # print(sorted_pos)
    
    total_r = sum(pos_open_risk.values())
    
    for pos in sorted_pos:
        if total_r > r_limit:
            pair = pos[0]
            now = datetime.now().strftime('%d/%m/%y %H:%M')
            price = get_price(pair)
            note = f"*** sell {pair} @ {price}"
            print(now, note)
            if live:
                push = pb.push_note(now, note)
                clear_stop(pair)
                sell_order = sell_asset(pair)
                sell_order['reason'] = 'portfolio risk limiting'
                trade_notes.append(sell_order)
                total_r -= pos[1]
    
    return trade_notes






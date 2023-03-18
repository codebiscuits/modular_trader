import keys, math, time, json
import statistics as stats
import pandas as pd
import polars as pl
import numpy as np
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
from pushbullet import Pushbullet
from decimal import Decimal, getcontext
from pprint import pprint
from pathlib import Path
from datetime import datetime, timedelta
import utility_funcs as uf
from timers import Timer
from typing import Union, List, Tuple, Dict, Set, Optional, Any
from collections import Counter
import sys
from pycoingecko import CoinGeckoAPI
from pyarrow import ArrowInvalid

client = Client(keys.bPkey, keys.bSkey)
cg = CoinGeckoAPI()
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12

tf_dict = {'1m': Client.KLINE_INTERVAL_1MINUTE,
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


# -#-#- Utility Functions


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


def calc_stop(inval: float, spread: float, price: float, min_risk: float = 0.002) -> float:
    """calculates what the stop-loss trigger price should be based on the current
    value of the supertrend line and the current spread (slippage proxy).
    if this is too close to the entry price, the stop will be set at the minimum
    allowable distance."""
    buffer = max(spread * 2, min_risk)

    if price > inval:
        stop_price = float(inval) * (1 - buffer)
    else:
        stop_price = float(inval) * (1 + buffer)

    return stop_price


# -#-#- Market Data Functions


def get_depth(session, pair: str) -> Tuple[float, float]:
    """returns the quantity (in the quote currency) that could be bought/sold
    within the % range of price set by the max_slip param"""

    max_slip = session.max_spread

    price = session.pairs_data[pair]['price']
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

    usdt_depth_l = float(depth_l * price)
    usdt_depth_s = float(depth_s * price)

    return usdt_depth_l, usdt_depth_s


# def get_pairs(quote: str = 'USDT', market: str = 'SPOT', session=None) -> List[str]:
#     """returns all active pairs for a given quote currency. possible values for
#     quote are USDT, BTC, BNB etc. possible values for market are SPOT or CROSS"""
#
#     sa = Timer('get_pairs')
#     sa.start()
#
#     if market == 'SPOT':
#         print('get_exchange_info')
#         if session:
#             session.track_weights(10)
#         abc = Timer('all binance calls')
#         abc.start()
#         info = client.get_exchange_info()
#         abc.stop()
#         symbols = info.get('symbols')
#         pairs = []
#         for sym in symbols:
#             right_quote = sym.get('quoteAsset') == quote
#             right_market = market in sym.get('permissions')
#             trading = sym.get('status') == 'TRADING'
#             allowed = sym.get('symbol') not in not_pairs
#             if right_quote and right_market and trading and allowed:
#                 pairs.append(sym.get('symbol'))
#     elif market == 'MARGIN':
#         pairs = []
#         print('get_margin_all_pairs')
#         if session:
#             session.track_weights(1)
#         abc = Timer('all binance calls')
#         abc.start()
#         info = client.get_margin_all_pairs()
#         abc.stop()
#         for i in info:
#             if i.get('quote') == quote:
#                 pairs.append(i.get('symbol'))
#     sa.stop()
#     return pairs





def get_current_cg_data(cg_symbol):
    data = cg.get_coins_markets(vs_currency='usd', ids=cg_symbol)

    return dict(
        mcap=data['market_cap'],
        mcap_rank=data['markt_cap_rank'],
        total_volume=data['total_volume']
    )


def get_cg_data(cg_symbol, days):
    data = cg.get_coin_market_chart_by_id(id=cg_symbol, vs_currency='usd', days=days)
    data_dict = {
        'timestamp': [d[0] for d in data['prices']],
        'mcap': [d[1] for d in data['market_caps']],
        'tot_vol': [d[1] for d in data['total_volumes']]
    }
    df = pd.DataFrame(data_dict)
    df['date'] = pd.to_datetime(df.timestamp, unit='ms')

    df = df[['date', 'tot_vol', 'mcap']].set_index('date', drop=True)

    if days == 1:
        df = df.resample('15T').agg('mean')
    elif 1 < days <=90:
        df = df.resample('4H').agg('mean')
    else:
        df = df.resample('3D').agg('mean')

    df = df.resample('5T').interpolate()

    return df


def get_bin_ohlc(pair: str, timeframe: str, span: str = "2 years ago UTC", session=None) -> pd.DataFrame:
    """fetches kline data from binance for the stated pair and timeframe.
    span tells the function how far back to start the data, in plain english
    for timeframe, use strings like 5m or 1h or 1d"""

    if session:
        session.track_weights(1)
    abc = Timer('all binance calls')
    abc.start()
    klines = client.get_historical_klines(pair, tf_dict.get(timeframe), span)
    abc.stop()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(['close_time', 'ignore'], axis=1)

    return df


def get_all_cg_data(session, pair, tf, df_first, df_last):
    timespan = df_last - df_first

    cg_data = get_cg_data(session.pairs_data[pair]['cg_symbol'], 1)
    print(cg_data)

    if timespan.days > 1:
        short_start = cg_data.indexc[0]
        med_cg_data = get_cg_data(session.pairs_data[pair]['cg_symbol'], min(timespan.days + 1, 90))
        med_cg_data = med_cg_data.loc[med_cg_data.index < short_start]
        cg_data = pd.concat([cg_data, med_cg_data]).sort_index()

    if timespan.days > 90:
        med_start = cg_data.index[0]
        long_cg_data = get_cg_data(session.pairs_data[pair]['cg_symbol'], timespan.days + 1)
        long_cg_data = long_cg_data.loc[long_cg_data.index < med_start]
        cg_data = pd.concat([cg_data, long_cg_data]).sort_index()

    if tf == '5m':
        pass
    elif tf == '1m':
        cg_data = cg_data.resample('1T').interpolate()
    else:
        cg_data = resample(cg_data, tf)

    if cg_data.index[0] < df_first:
        cg_data = cg_data.loc[cg_data.index >= df_first]

    if cg_data.index[-1] > df_last:
        print('cg data was actually longer than binance data')
        cg_data = cg_data.loc[cg_data.index <= df_last]

    return cg_data.reset_index(drop=True)


def get_ohlc(pair: str, timeframe: str, span: str = "2 years ago UTC", session=None) -> pd.DataFrame:
    """calls get_bin_ohlc and get_5min_cg_data or get_daily_cg_data and stitches their outputs together"""

    bin_data = get_bin_ohlc(pair, timeframe, span, session)

    # cg_data = get_all_cg_data(session, pair, timeframe, bin_data.timestamp.iloc[0], bin_data.timestamp.iloc[-1])

    # df = pd.concat([bin_data, cg_data['mcap', 'tot_vol']], axis=1).fillna(method='ffill')

    return bin_data


def update_ohlc(pair: str, timeframe: str, old_df: pd.DataFrame, session=None) -> pd.DataFrame:
    """takes an ohlc dataframe, works out when the data ends, then requests from
    binance all data from the end to the current moment. It then joins the new
    data onto the old data and returns the updated dataframe"""
    old_end = int(old_df.timestamp.iloc[-1].timestamp()) * 1000

    # print('get_klines')
    if session:
        session.track_weights(1)
    abc = Timer('all binance calls')
    abc.start()
    klines = client.get_klines(symbol=pair, interval=tf_dict.get(timeframe), startTime=old_end)
    abc.stop()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(['close_time', 'ignore'], axis=1)

    return pd.concat([old_df.drop(old_df.index[-1]), df], copy=True, ignore_index=True)


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


def prepare_ohlc(session, timeframes: list, pair: str) -> dict:
    """checks if there is old data already, if so it loads the old data and
    downloads an update, if not it downloads all data from scratch, then
    resamples all data to desired timeframe"""

    ds = Timer('prepare_ohlc')
    ds.start()

    if session.pairs_data[pair].get('ohlc_5m', None) is not None:
        df = session.pairs_data[pair]['ohlc_5m']
        # print('got df from session.pairs_data')

    else:
        filepath = Path(f'{session.ohlc_data}/{pair}.parquet')

        if filepath.exists():
            # df = pd.read_parquet(filepath)
            try:
                pldf = pl.read_parquet(source=filepath, use_pyarrow=True)
                df = pldf.to_pandas()
            except ArrowInvalid as e:
                print(f"Problem reading {pair} parquet file, downloading from scratch.")
                print(e)
                filepath.unlink()
                df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)

            last_timestamp = df.timestamp.iloc[-1].timestamp()
            now = datetime.now().timestamp()
            data_age_mins = (now - last_timestamp) / 60
            # print(f"\n{pair} ohlc data ends: {(now - last_timestamp) / 60:.1f} minutes ago")
            if (data_age_mins < 15) and (len(df) > 2):
                # update last close price with current price
                # print(f"{pair} ohlc data less than 15 mins old, adjusting last close")
                last_idx = df.index[-1]
                df.at[last_idx, 'close'] = session.pairs_data[pair]['price']
            elif len(df) > 2:
                df = update_ohlc(pair, session.ohlc_tf, df, session)
                # print('updated ohlc')
            else:
                df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
                # print(f'{pair} ohlc too short to update, downloaded from scratch')

        else:
            df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
            print(f'downloaded {pair} from scratch')

        # max_len = 210240 # 210240 is 2 years' worth of 5m periods
        # if len(df) > max_len:
        #     df = df.tail(max_len).reset_index(drop=True)
        # pldf = pl.from_pandas(df)
        # pldf.write_parquet(filepath, use_pyarrow=True)

        session.store_ohlc(df, pair, timeframes)

    df_dict = {}
    for tf, offset in timeframes:
        df_dict[tf] = resample_ohlc(tf, offset, df.copy()).tail(session.min_length).reset_index(drop=True)

    ds.stop()
    return df_dict


# -#-#- Trading Functions

def create_stop_dict(session, order: dict) -> dict:
    '''collects and returns the details of filled stop-loss order in a dictionary'''

    yu = Timer('create_stop-dict')
    yu.start()

    pprint(order)

    pair = order.get('symbol')
    quote_qty = order.get('cummulativeQuoteQty')
    base_qty = order.get('executedQty')
    avg_price = round(float(quote_qty) / float(base_qty), 8)

    bnb_fee = float(quote_qty) * float(session.fees['margin_taker']) / session.pairs_data['BNBUSDT']['price']

    trade_dict = {'timestamp': int(order.get('updateTime') / 1000),
                  'pair': pair,
                  'trig_price': order.get('stopPrice'),
                  'limit_price': order.get('price'),
                  'exe_price': str(avg_price),
                  'base_size': base_qty,
                  'quote_size': quote_qty,
                  'fee': f"{bnb_fee:.8f}",
                  'fee_currency': 'BNB'
                  }
    if order.get('status') != 'FILLED':
        print(f'{pair} order not filled')
        pb.push_note('Warning', f'{pair} stop-loss hit but not filled')
    yu.stop()
    return trade_dict


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

        trade_dict = {'timestamp': int(float(order.get('transactTime')) / 1000),
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


# -#-#- Margin Trading Functions


def buy_asset_M(session, pair: str, size: float, is_base: bool, price: float, live: bool) -> dict:
    """sends a market buy order to binance in the margin account and returns the
    order data"""

    fg = Timer('buy_asset_M')
    fg.start()

    if live and is_base:
        base_size = uf.valid_size(session, pair, size)
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
            base_size = uf.valid_size(session, pair, size / price)
            print(f'buy_asset_M {size = }, {base_size = }')
            if not float(base_size):  # if size == 0, valid_size will output None
                print(f'*problem* non-live buy {pair}: {base_size = }')
                base_size = 0
            usdt_size = str(size)
        buy_order = {'clientOrderId': '111111',
                     'cummulativeQuoteQty': usdt_size,
                     'executedQty': str(base_size),
                     'fills': [{'commission': '0',
                                'commissionAsset': 'BNB',
                                'price': str(uf.valid_price(session, pair, price)),
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

    base_size = uf.valid_size(session, pair, base_size)
    if not base_size:  # if size == 0, valid_size will output None
        base_size = 0
    if live:
        sell_order = client.create_margin_order(symbol=pair, side=be.SIDE_SELL, type=be.ORDER_TYPE_MARKET,
                                                quantity=base_size)
    else:
        now = int(datetime.now().timestamp() * 1000)
        usdt_size = uf.valid_size(session, pair, float(base_size) * price)
        if not usdt_size:
            usdt_size = 0
        sell_order = {
            'clientOrderId': '111111',
            'cummulativeQuoteQty': str(usdt_size),
            'executedQty': str(base_size),
            'fills': [{'commission': '0',
                       'commissionAsset': 'BNB',
                       'price': str(uf.valid_price(session, pair, price)),
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

    if live and float(qty):
        try:
            client.repay_margin_loan(asset=asset, amount=qty)
        except bx.BinanceAPIException as e:
            print(f"*** Exception whilst trying to repay {qty} {asset}. If it says 'repay amount larger than loan "
                  f"amount, it's most likely no loan to be repayed.")
            print(e.status_code)
            print(e.message)


def set_stop_M(session, pair: str, size: float, side: str, trigger: float, limit: float) -> dict:
    """sends a margin stop-loss limit order to binance and returns the order data"""

    sd = Timer('set_stop_M')
    sd.start()

    now = datetime.now().timestamp()

    trigger = uf.valid_price(session, pair, trigger)
    limit = uf.valid_price(session, pair, limit)
    stop_size = uf.valid_size(session, pair, size)
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
        stop_sell_order = {'orderId': 'not live',
                           'transactTime': now * 1000}

    session.algo_order_counts += Counter([pair])

    sd.stop()
    return stop_sell_order


def clear_stop_M(session, pair: str, position: dict) -> Tuple[Any, Decimal]:
    """finds the order id of the most recent stop-loss from the trade record
    and cancels that specific order. if no such id can be found, returns null values"""

    fc = Timer('clear_stop_M')
    fc.start()

    stop_id = position['stop_id']

    clear, base_size = {}, 0
    if session.live:
        if stop_id:
            try:
                clear = client.cancel_margin_order(symbol=pair, orderId=str(stop_id))
                base_size = clear.get('origQty')
            except bx.BinanceAPIException as e:
                print(f"Exception during clear_stop_M on {pair}. If it's 'unknown order sent' then it was probably "
                      f"trying to cancel a stop-loss that had already been cancelled")
                print(e.status_code)
                print(e.message)
        else:
            print(f'no recorded stop id for {pair}')
    else:
        base_size = position['base_size']

    session.algo_order_counts -= Counter([pair])

    fc.stop()
    return clear, base_size

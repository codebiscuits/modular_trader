import statistics as stats
import pandas as pd
import polars as pl
from binance.client import Client
import binance.enums as be
import binance.exceptions as bx
from decimal import Decimal, getcontext
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Any
import sys
from pycoingecko import CoinGeckoAPI
from pyarrow import ArrowInvalid
from mt.resources import keys, indicators as ind, utility_funcs as uf
from mt.resources.loggers import create_logger
from mt.resources.timers import Timer

cg = CoinGeckoAPI()
ctx = getcontext()
ctx.prec = 12
logger = create_logger('binance_funcs')

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


# -#-#- Market Data Functions


def get_max_borrow(session, asset: str) -> float:
    abc = Timer('all binance calls')
    abc.start()
    logger.debug('running get_max_borrow')
    session.track_weights(50)
    try:
        limits = session.client.get_max_margin_loan(asset=asset)
        borrow = min(float(limits['amount']), float(limits['borrowLimit']))
    except bx.BinanceAPIException:
        borrow = 0
        logger.info(f"No borrow available for {asset}")
        # logger.exception(e)
    abc.stop()

    return borrow


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


def get_current_cg_data(cg_symbol):
    data = cg.get_coins_markets(vs_currency='usd', ids=cg_symbol)

    return dict(
        mcap=data['market_cap'],
        mcap_rank=data['markt_cap_rank'],
        total_volume=data['total_volume']
    )


# def get_cg_data(cg_symbol, days):
#     data = cg.get_coin_market_chart_by_id(id=cg_symbol, vs_currency='usd', days=days)
#     data_dict = {
#         'timestamp': [d[0] for d in data['prices']],
#         'mcap': [d[1] for d in data['market_caps']],
#         'tot_vol': [d[1] for d in data['total_volumes']]
#     }
#     df = pd.DataFrame(data_dict)
#     df['date'] = pd.to_datetime(df.timestamp, unit='ms')
#
#     df = df[['date', 'tot_vol', 'mcap']].set_index('date', drop=True)
#
#     if days == 1:
#         df = df.resample('15T').agg('mean')
#     elif 1 < days <= 90:
#         df = df.resample('4H').agg('mean')
#     else:
#         df = df.resample('3D').agg('mean')
#
#     df = df.resample('5T').interpolate()
#
#     return df


@uf.retry_on_busy()
def get_bin_ohlc(pair: str, timeframe: str, span: str = "2 years ago UTC", session=None) -> pd.DataFrame:
    """fetches kline data from binance for the stated pair and timeframe.
    span tells the function how far back to start the data, in plain english
    for timeframe, use strings like 5m or 1h or 1d"""

    abc = Timer('all binance calls')
    abc.start()

    if session:
        session.track_weights(1)
        klines = session.client.get_historical_klines(pair, tf_dict.get(timeframe), span)
    else:
        client = Client(keys.bPkey, keys.bSkey)
        klines = client.get_historical_klines(pair, tf_dict.get(timeframe), span)
    abc.stop()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # check df is localised to UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    try:
        df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
    except TypeError:
        pass

    df = df.drop(['close_time', 'ignore'], axis=1)

    return df


def get_ohlc(pair: str, timeframe: str, span: str = "2 years ago UTC", session=None) -> pd.DataFrame:
    """calls get_bin_ohlc and get_5min_cg_data or get_daily_cg_data and stitches their outputs together"""

    bin_data = get_bin_ohlc(pair, timeframe, span, session=session)

    # cg_data = get_all_cg_data(session, pair, timeframe, bin_data.timestamp.iloc[0], bin_data.timestamp.iloc[-1])

    # df = pd.concat([bin_data, cg_data['mcap', 'tot_vol']], axis=1).fillna(method='ffill')

    return bin_data


@uf.retry_on_busy()
def update_ohlc(pair: str, timeframe: str, old_df: pd.DataFrame, session=None) -> pd.DataFrame:
    """takes an ohlc dataframe, works out when the data ends, then requests from
    binance all data from the end to the current moment. It then joins the new
    data onto the old data and returns the updated dataframe"""

    abc = Timer('all binance calls')
    abc.start()

    if session:
        session.track_weights(1)
        client = session.client
    else:
        client = Client(keys.bPkey, keys.bSkey)

    # check old_df is localised to UTC
    old_df['timestamp'] = pd.to_datetime(old_df['timestamp'])
    try:
        old_df['timestamp'] = old_df.timestamp.dt.tz_localize('UTC')
    except TypeError:
        pass

    # get_klines is quicker than get_historical_klines but will only download 500 periods, calculate which to use
    deltas = {'1m': timedelta(minutes=1), '5m': timedelta(minutes=5), '15m': timedelta(minutes=15)}
    span_periods = (datetime.now(timezone.utc) - old_df.timestamp.iloc[-1]) / deltas[timeframe]
    if span_periods >= 500:
        old_end = str(old_df.timestamp.iloc[-1])
        klines = client.get_historical_klines(symbol=pair, interval=tf_dict.get(timeframe), start_str=old_end)
    else:
        old_end = int(old_df.timestamp.iloc[-1].timestamp()) * 1000
        klines = client.get_klines(symbol=pair, interval=tf_dict.get(timeframe), startTime=old_end)

    abc.stop()
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'close_time',
            'quote_vol', 'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # check df is localised to UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    try:
        df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
    except TypeError:
        pass

    df = df.drop(['close_time', 'ignore'], axis=1)

    return pd.concat([old_df.drop(old_df.index[-1]), df], copy=True, ignore_index=True)


def resample_ohlc(tf, offset, df):
    """resamples ohlc data to the required timeframe and offset, then discards older rows if necessary to return a
    dataframe of the desired length"""

    tf_map = {'15m': '15T', '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H',
              '8h': '8H', '12h': '12H', '1d': '1D', '3d': '3D', '1w': '1W'}

    df = df.resample(tf_map[tf], on='timestamp',
                     offset=offset).agg(
        {'open': 'first',
         'high': 'max',
         'low': 'min',
         'close': 'last',
         'base_vol': 'sum',
         'quote_vol': 'sum',
         'num_trades': 'sum',
         'taker_buy_base_vol': 'sum',
         'taker_buy_quote_vol': 'sum',
         }
    )

    df = df.reset_index()  # drop=False because we want to keep the timestamp column

    return df


def prepare_ohlc(session, timeframes: list, pair: str) -> dict:
    """checks if there is old data already, if so it loads the old data and
    downloads an update, if not it downloads all data from scratch, then
    resamples all data to desired timeframe"""

    ds = Timer('prepare_ohlc')
    ds.start()

    if session.pairs_data[pair].get('ohlc_5m', None) is not None:
        df = session.pairs_data[pair]['ohlc_5m']

    else:
        read_path = Path(f'{session.ohlc_r}/{pair}.parquet')

        if read_path.exists():
            try:
                pldf = pl.read_parquet(source=read_path, use_pyarrow=True)
                df = pldf.to_pandas()
            except (ArrowInvalid, OSError):
                logger.exception(f"Problem reading {pair} parquet file, downloading from scratch.")
                read_path.unlink()
                df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC')

            # check df is localised to UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            try:
                df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
            except TypeError:
                pass

            last_timestamp = df.timestamp.iloc[-1].timestamp()
            now = datetime.now(timezone.utc).timestamp()
            data_age_mins = (now - last_timestamp) / 60
            if (data_age_mins < 15) and (len(df) > 2):
                # update last close price with current price
                last_idx = df.index[-1]
                df.at[last_idx, 'close'] = session.pairs_data[pair]['price']
            elif len(df) > 2:
                logger.debug(f"{pair} ohlc data ends: {(now - last_timestamp) / 60:.1f} minutes ago, updating")
                df = update_ohlc(pair, session.ohlc_tf, df, session)
            else:
                df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)

        else:
            df = get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)

        # check df is localised to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        try:
            df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
        except TypeError:
            pass

        session.store_ohlc(df, pair, timeframes)

    df_dict = {}
    for tf, offset, _ in timeframes:
        vwma_lengths = {'15m': 3, '30m': 6, '1h': 12, '4h': 48, '6h': 70, '8h': 96, '12h': 140, '1d': 280}
        vwma = ind.vwma(df, vwma_lengths[tf] * 24)
        vwma = vwma[int(vwma_lengths[tf] / 2)::vwma_lengths[tf]].reset_index(drop=True)

        res_df = resample_ohlc(tf, offset, df.copy()).tail(session.max_length).reset_index(drop=True)
        res_df['vwma'] = vwma

        if (tf in ['12h', '1d']) or (len(res_df) >= session.min_length):
            df_dict[tf] = res_df

    ds.stop()
    return df_dict


# -#-#- Trading Functions

def create_stop_dict(session, order: dict) -> dict:
    """collects and returns the details of filled stop-loss order in a dictionary"""

    yu = Timer('create_stop-dict')
    yu.start()

    pair = order.get('symbol')
    quote_qty = order.get('cummulativeQuoteQty')
    base_qty = order.get('executedQty')
    avg_price = round(float(quote_qty) / float(base_qty), 8)

    bnb_fee = float(quote_qty) * float(session.fees['margin_taker']) / session.pairs_data['BNBUSDT']['price']

    # TODO when i'm getting real responses from binance, i must check if this / 1000 is still appropriate
    trade_dict = {'timestamp': int(order.get('updateTime')),
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
        logger.warning(f'{pair} order not filled')
        # pb.push_note('Warning', f'{pair} stop-loss hit but not filled')
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

        trade_dict = {'timestamp': int(order.get('transactTime')),
                      'trig_price': str(price),
                      'exe_price': str(avg_price),
                      'base_size': str(order.get('executedQty')),
                      'quote_size': quote_qty,
                      'fee': str(fee),
                      'fee_currency': fills[0].get('commissionAsset')
                      }
        if order.get('status') != 'FILLED':
            logger.warning(f'{pair} order not filled')
            # pb.push_note('Warning', f'{pair} order not filled')

    else:
        trade_dict = {'timestamp': int(order.get('transactTime')),
                      "trig_price": str(price),
                      "exe_price": str(price),
                      "base_size": str(order.get('executedQty')),
                      "quote_size": str(order.get('cummulativeQuoteQty')),
                      "fee": '0',
                      "fee_currency": "BNB"
                      }
    fd.stop()
    return trade_dict


# -#-#- Spot Trading Functions


@uf.retry_on_busy()
def buy_asset_s(session, pair: str, size: float, live: bool) -> dict:
    """sends a market buy order to binance in the spot account and returns the order data"""

    bas = Timer('buy_asset_s')
    bas.start()

    now = int(datetime.now(timezone.utc).timestamp())
    price = session.pairs_data[pair]['price']
    base_size: str = uf.valid_size(session, pair, size)
    if live:
        buy_order = session.client.order_market_buy(symbol=pair, quantity=base_size)

    else:
        usdt_size: str = f"{base_size} * {price}:.2f"
        buy_order = {'clientOrderId': '111111',
                     'cummulativeQuoteQty': usdt_size,
                     'executedQty': base_size,
                     'fills': [{'commission': '0',
                                'commissionAsset': 'BNB',
                                'price': str(uf.valid_price(session, pair, price)),
                                'qty': base_size}],
                     'orderId': 123456,
                     'origQty': base_size,
                     'price': '0',
                     'side': 'BUY',
                     'status': 'FILLED',
                     'symbol': pair,
                     'timeInForce': 'GTC',
                     'transactTime': now,
                     'type': 'MARKET'}

    bas.stop()
    return buy_order


@uf.retry_on_busy()
def sell_asset_s(session, pair: str, size: float, live: bool) -> dict:
    """sends a market sell order to binance in the spot account and returns the order data"""

    sas = Timer('sell_asset_s')
    sas.start()

    now = int(datetime.now(timezone.utc).timestamp())
    price = session.pairs_data[pair]['price']
    base_size = uf.valid_size(session, pair, size)

    if live:
        sell_order = session.client.order_market_sell(symbol=pair, quantity=base_size)
    else:
        usdt_size: str = f"{base_size} * {price}:.2f"
        sell_order = {'clientOrderId': '111111',
                      'cummulativeQuoteQty': usdt_size,
                      'executedQty': base_size,
                      'fills': [{'commission': '0',
                                 'commissionAsset': 'BNB',
                                 'price': str(uf.valid_price(session, pair, price)),
                                 'qty': base_size}],
                      'orderId': 123456,
                      'origQty': base_size,
                      'price': '0',
                      'side': 'SELL',
                      'status': 'FILLED',
                      'symbol': pair,
                      'timeInForce': 'GTC',
                      'transactTime': now,
                      'type': 'MARKET'}

    sas.stop()
    return sell_order


@uf.retry_on_busy()
def set_stop_s(session, pair, trigger, limit, size):
    """sends a stop-loss limit order to binance spot account and returns the order data"""

    func_name = sys._getframe().f_code.co_name
    t = Timer(f'{func_name}')
    t.start()

    now = datetime.now(timezone.utc).timestamp()

    trigger = uf.valid_price(session, pair, trigger)
    limit = uf.valid_price(session, pair, limit)
    stop_size = uf.valid_size(session, pair, size)
    logger.info(f"setting {pair} stop: {stop_size = } {trigger = } {limit = }")
    if session.live:
        stop_sell_order = session.client.create_order(symbol=pair,
                                                      side=be.SIDE_SELL,
                                                      type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                      timeInForce=be.TIME_IN_FORCE_GTC,
                                                      stopPrice=trigger,
                                                      quantity=stop_size,
                                                      price=limit)
    else:
        stop_sell_order = {'orderId': 'not live',
                           'transactTime': now}

    session.pairs_data[pair]['algo_orders'] += 1

    t.stop()
    return stop_sell_order


@uf.retry_on_busy()
def clear_stop_s(session, pair: str, position: dict) -> Tuple[Any, Decimal]:
    """finds the order id of the most recent stop-loss from the trade record
    and cancels that specific order. if no such id can be found, returns null values"""

    func_name = sys._getframe().f_code.co_name
    t = Timer(f'{func_name}')
    t.start()

    stop_id = position['stop_id']

    clear, base_size = {}, 0
    if session.live:
        if stop_id:
            try:
                clear = session.client.cancel_order(symbol=pair, orderId=str(stop_id))
                base_size = clear.get('origQty')
            except bx.BinanceAPIException as e:
                logger.exception(
                    f"Exception during clear_stop_s on {pair}. If it's 'unknown order sent' then it was probably "
                    f"trying to cancel a stop-loss that had already been cancelled")
                logger.error(e.status_code)
                logger.error(e.message)
        else:
            logger.warning(f'no recorded stop id for {pair}')
    else:
        base_size = position['base_size']

    session.pairs_data[pair]['algo_orders'] -= 1

    t.stop()
    return clear, base_size


# -#-#- Margin Trading Functions


@uf.retry_on_busy()
def buy_asset_M(session, pair: str, size: float, is_base: bool, live: bool) -> dict:
    """sends a market buy order to binance in the margin account and returns the order data"""

    fg = Timer('buy_asset_M')
    fg.start()

    if live and is_base:
        base_size = uf.valid_size(session, pair, size)
        buy_order = session.client.create_margin_order(symbol=pair,
                                                       side=be.SIDE_BUY,
                                                       type=be.ORDER_TYPE_MARKET,
                                                       quantity=base_size)
    elif live and not is_base:
        buy_order = session.client.create_margin_order(symbol=pair,
                                                       side=be.SIDE_BUY,
                                                       type=be.ORDER_TYPE_MARKET,
                                                       quoteOrderQty=size)
    else:
        now = int(datetime.now(timezone.utc).timestamp())
        price = session.pairs_data[pair]['price']
        if is_base:
            base_size = size
            usdt_size = f"{float(size) * price:.2f}"
        else:
            base_size = uf.valid_size(session, pair, size / price)
            logger.info(f'buy_asset_M {size = }, {base_size = }')
            if not float(base_size):  # if size == 0, valid_size will output None
                logger.warning(f'*problem* non-live buy {pair}: {base_size = }')
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


@uf.retry_on_busy()
def sell_asset_M(session, pair: str, base_size: float, live: bool) -> dict:
    """sends a market sell order to binance in the margin account and returns the order data"""

    df = Timer('sell_asset_M')
    df.start()

    base_size = uf.valid_size(session, pair, base_size)
    if not base_size:  # if size == 0, valid_size will output None
        base_size = 0
    if live:
        sell_order = session.client.create_margin_order(symbol=pair, side=be.SIDE_SELL, type=be.ORDER_TYPE_MARKET,
                                                        quantity=base_size)
    else:
        now = int(datetime.now(timezone.utc).timestamp())
        price = session.pairs_data[pair]['price']
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


@uf.retry_on_busy()
def borrow_asset_M(session, asset: str, qty: str, live: bool) -> str:
    """calls the binance api function to take out a margin loan"""

    if live:
        try:
            session.client.create_margin_loan(asset=asset, amount=qty)
        except bx.BinanceAPIException as e:
            if e.code == -3045:  # the system does not have enough asset now
                logger.error(f"Problem borrowing {qty} {asset}, not enough to borrow.")
            logger.exception(e)
            return 0

        return qty

    else:
        return qty


@uf.retry_on_busy()
def repay_asset_M(session, asset: str, qty: str, live: bool) -> bool:
    """calls the binance api function to repay a margin loan"""

    if live and float(qty):
        try:
            session.client.repay_margin_loan(asset=asset, amount=qty)
            return True
        except bx.BinanceAPIException as e:
            logger.exception(
                f"*** Exception whilst trying to repay {qty} {asset}. If it says 'repay amount larger than loan "
                f"amount, it's most likely no loan to be repayed.")
            logger.error(e.code)
            logger.error(e.message)
            return False


@uf.retry_on_busy()
def set_stop_M(session, pair: str, size: float, side: str, trigger: float) -> dict:
    """sends a margin stop-loss limit order to binance and returns the order data"""

    sd = Timer('set_stop_M')
    sd.start()

    now = datetime.now(timezone.utc).timestamp()
    curr_price = session.pairs_data[pair]['price']
    limit = curr_price * 0.81 if side == be.SIDE_SELL else curr_price * 1.19

    trigger = uf.valid_price(session, pair, trigger)
    limit = uf.valid_price(session, pair, limit)
    stop_size = uf.valid_size(session, pair, size)

    logger.info(f"setting {pair} stop: {stop_size = } {side = } {trigger = } {limit = }")
    if session.live:
        try:
            stop_sell_order = session.client.create_margin_order(symbol=pair,
                                                                 side=side,
                                                                 type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                                 timeInForce=be.TIME_IN_FORCE_GTC,
                                                                 stopPrice=trigger,
                                                                 quantity=stop_size,
                                                                 price=limit)
        except bx.BinanceAPIException as e:
            if e.code == -2010:
                logger.exception(e)
                now = datetime.now(timezone=timezone.utc).strftime('%d/%m/%y %H:%M')
                logger.error(f"current time: {now}, failed to set stop: {trigger} on {pair}")
                if side == be.SIDE_SELL:
                    real_curr_price = float(session.client.get_orderbook_ticker(symbol=pair)['bidPrice'])
                    trigger = uf.valid_price(session, pair, real_curr_price * 0.99)
                    limit = uf.valid_price(session, pair, real_curr_price * 0.81)
                else:
                    real_curr_price = float(session.client.get_orderbook_ticker(symbol=pair)['askPrice'])
                    trigger = uf.valid_price(session, pair, real_curr_price * 1.01)
                    limit = uf.valid_price(session, pair, real_curr_price * 1.19)

                stop_sell_order = session.client.create_margin_order(symbol=pair,
                                                                     side=side,
                                                                     type=be.ORDER_TYPE_STOP_LOSS_LIMIT,
                                                                     timeInForce=be.TIME_IN_FORCE_GTC,
                                                                     stopPrice=trigger,
                                                                     quantity=stop_size,
                                                                     price=limit)
                logger.error(f"New {pair} {side} stop successfuly set at {trigger}")
        session.pairs_data[pair]['algo_orders'] += 1
    else:
        stop_sell_order = {'orderId': 'not live',
                           'transactTime': now}

    sd.stop()
    return stop_sell_order


@uf.retry_on_busy()
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
                clear = session.client.cancel_margin_order(symbol=pair, orderId=str(stop_id))
                base_size = clear.get('origQty')
                session.pairs_data[pair]['algo_orders'] -= 1
            except bx.BinanceAPIException as e:
                logger.exception(
                    f"Exception during clear_stop_M on {pair}. If it's 'unknown order sent' then it was probably "
                    f"trying to cancel a stop-loss that had already been cancelled")
                logger.error(e.status_code)
                logger.error(e.message)
        else:
            logger.warning(f'no recorded stop id for {pair}')
    else:
        base_size = position['base_size']

    fc.stop()
    return clear, base_size


def set_oco_stop_m(session, pair: str, size: float, side: str, target: float, stop: float) -> dict:
    """sends a margin oco order to binance and returns the order data"""

    sd = Timer('set_margin_oco_m')
    sd.start()

    now = datetime.now(timezone.utc).timestamp()

    target = uf.valid_price(session, pair, target)

    curr_price = session.pairs_data[pair]['price']
    stop_limit = curr_price * 0.81 if side == be.SIDE_SELL else curr_price * 1.19
    stop_limit = uf.valid_price(session, pair, stop_limit)
    stop_trigger = uf.valid_price(session, pair, stop)
    size = uf.valid_size(session, pair, size)

    logger.info(f"setting {pair} stop: {size = } {side = } {stop_trigger = } {target = }")
    if session.live:
        try:
            oco_order = session.client.create_margin_oco_order(
                symbol=pair,
                side=side,
                quantity=size,
                price=target,
                stopPrice=stop_trigger,
                stopLimitPrice=stop_limit,
                stopLimitTimeInForce=be.TIME_IN_FORCE_GTC
            )
            session.pairs_data[pair]['algo_orders'] += 2
            logger.error(f"New {pair} {side} oco successfuly set at {target}, {stop_trigger}")

        except bx.BinanceAPIException as e:
            if e.code == -2010:
                logger.exception(e)
                now = datetime.now(timezone=timezone.utc).strftime('%d/%m/%y %H:%M')
                logger.error(f"current time: {now}, failed to set oco: {target}, {stop_trigger} on {pair}")
                if side == be.SIDE_SELL:
                    real_curr_price = float(session.client.get_orderbook_ticker(symbol=pair)['bidPrice'])
                    trigger = uf.valid_price(session, pair, real_curr_price * 0.99)
                    stop_trigger = uf.valid_price(session, pair, real_curr_price * 0.81)
                else:
                    real_curr_price = float(session.client.get_orderbook_ticker(symbol=pair)['askPrice'])
                    trigger = uf.valid_price(session, pair, real_curr_price * 1.01)
                    stop_trigger = uf.valid_price(session, pair, real_curr_price * 1.19)

                oco_order = session.client.create_margin_oco_order(
                    symbol=pair,
                    side=side,
                    quantity=size,
                    price=target,
                    stopPrice=stop,
                    stopLimitPrice=stop_trigger,
                    stopLimitTimeInForce=be.TIME_IN_FORCE_GTC
                )
                session.pairs_data[pair]['algo_orders'] += 2
                logger.error(f"New {pair} {side} oco successfuly set at {target}, {stop_trigger}")

    else:
        oco_order = {'orderId': 'not live',
                           'transactTime': now}

    return oco_order

import numpy as np
import pandas as pd
from pathlib import Path
import json
from json.decoder import JSONDecodeError
import statistics as stats
from mt.resources.timers import Timer
from mt.resources.loggers import create_logger
from typing import Dict
from mt.resources import indicators as ind, utility_funcs as uf, binance_funcs as funcs
from datetime import datetime, timezone
import binance.exceptions as bx
import binance.enums as be
from decimal import Decimal, getcontext
from pprint import pformat
import sys
from pyarrow import ArrowInvalid
import joblib
from xgboost import XGBClassifier

ctx = getcontext()
ctx.prec = 12
timestring = '%d/%m/%y %H:%M'
logger = create_logger('   agents    ')


class Agent:
    """generic agent class for each strategy to inherit from"""

    # name = None
    # id = None
    # ohlc_length = 0
    # quote_asset = 'USDT'
    # mode = None
    # tf = None
    # signal_age = None
    # perf_log = {}
    # or_list = []
    # total_open_risk = 0.0
    # num_open_positions = 0

    def __init__(self, session):
        t = Timer('agent init')
        t.start()
        self.live = session.live
        self.fr_max = session.fr_max
        self.realised_pnls = dict(
            real_spot=0,
            real_long=0,
            real_short=0,
            sim_spot=0,
            sim_long=0,
            sim_short=0,
            wanted_spot=0,
            wanted_long=0,
            wanted_short=0,
            unwanted_spot=0,
            unwanted_long=0,
            unwanted_short=0)
        self.target_risk = 0.1
        self.counts_dict = {
            'real_stop_spot': 0, 'real_open_spot': 0, 'real_add_spot': 0, 'real_tp_spot': 0, 'real_close_spot': 0,
            'sim_stop_spot': 0, 'sim_open_spot': 0, 'sim_add_spot': 0, 'sim_tp_spot': 0, 'sim_close_spot': 0,
            'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0, 'real_close_long': 0,
            'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0, 'sim_close_long': 0,
            'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0, 'real_close_short': 0,
            'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0, 'sim_close_short': 0,
            'too_small': 0, 'low_score': 0, 'too_many_pos': 0, 'too_much_or': 0, 'algo_order_limit': 0,
            'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0, 'reduce_risk': 0}
        if self.live:
            self.load_perf_log(session)
        else:
            self.sync_test_records(session)
        self.open_trades = self.read_open_trade_records(session, 'open')
        self.sim_trades = self.read_open_trade_records(session, 'sim')
        self.tracked_trades = self.read_open_trade_records(session, 'tracked')
        self.closed_trades = self.read_closed_trade_records(session)
        self.closed_sim_trades = self.read_closed_sim_trade_records(session)
        self.real_pos = self.current_positions(session, 'open')
        self.sim_pos = self.current_positions(session, 'sim')
        self.tracked = self.current_positions(session, 'tracked')
        self.backup_trade_records(session)
        self.repair_trade_records(session)
        self.check_valid_open(session)
        # self.calc_init_opnl(session)
        # self.open_pnl_changes = {}
        self.indiv_r_limit = 1.8
        self.fr_div = 10
        self.next_id = int(datetime.now(timezone.utc).timestamp())
        session.min_length = min(session.min_length, self.ohlc_length)
        session.max_length = max(session.min_length, self.ohlc_length)
        t.stop()

    def __str__(self):
        return self.id

    def load_perf_log(self, session):
        folder = Path(f"{session.records_r}/{self.id}")
        if not folder.exists():
            folder.mkdir(parents=True)
        bal_path = Path(folder / 'perf_log.json')
        bal_path.touch(exist_ok=True)
        try:
            with open(bal_path, "r") as file:
                self.perf_log = json.load(file)
        except JSONDecodeError:
            logger.error(f"{bal_path} was an empty file.")
            self.perf_log = None

    def sync_test_records(self, session) -> None:
        """takes the trade records from the raspberry pi and saves them over
        the local trade records. only runs when not live"""

        q = Timer('sync_test_records')
        q.start()
        real_folder = Path(f"{session.records_r}/{self.id}")
        test_folder = Path(f'{session.records_w}/{self.id}')
        if not test_folder.exists():
            test_folder.mkdir(parents=True)
        bal_path = Path(real_folder / 'perf_log.json')
        test_bal = Path(test_folder / 'perf_log.json')
        test_bal.touch(exist_ok=True)

        if bal_path.exists():
            try:
                with open(bal_path, "r") as file:
                    self.perf_log = json.load(file)
                with open(test_bal, "w") as file:
                    json.dump(self.perf_log, file)
            except JSONDecodeError:
                # logger.info(f"{bal_path} was an empty file.")
                self.perf_log = None

        def sync_trades_records(switch):
            w = Timer(f'sync_trades_records-{switch}')
            w.start()
            trades_path = Path(f'{session.records_r}/{self.id}/{switch}_trades.json')
            test_trades = Path(f'{session.records_w}/{self.id}/{switch}_trades.json')
            test_trades.touch(exist_ok=True)

            if trades_path.exists():
                try:
                    with open(trades_path, 'r') as tr_file:
                        data = json.load(tr_file)
                    if data:
                        with open(test_trades, 'w') as tr_file:
                            json.dump(data, tr_file)
                except JSONDecodeError:
                    # logger.info(f'{switch}_trades file empty')
                    pass
            w.stop()

        sync_trades_records('open')
        sync_trades_records('sim')
        sync_trades_records('tracked')
        sync_trades_records('closed')
        sync_trades_records('closed_sim')

        q.stop()

    def read_open_trade_records(self, session, state: str) -> dict:
        """loads records from open_trades/sim_trades/tracked_trades and returns
        them in a dictionary"""

        w = Timer(f'read_open_trade_records-{state}')
        w.start()
        ot_path = Path(f"{session.records_r}/{self.id}")
        ot_path.mkdir(parents=True, exist_ok=True)
        ot_path = ot_path / f'{state}_trades.json'

        if ot_path.exists():
            with open(ot_path, "r") as ot_file:
                try:
                    open_trades = json.load(ot_file)
                except JSONDecodeError:
                    open_trades = {}
        else:
            open_trades = {}
            ot_path.touch()

        w.stop()
        return open_trades

    def read_closed_trade_records(self, session) -> dict:
        """loads trade records from closed_trades and returns them as a dictionary"""

        e = Timer('read_closed_trade_records')
        e.start()
        ct_path = Path(f"{session.records_r}/{self.id}/closed_trades.json")
        if Path(ct_path).exists():
            with open(ct_path, "r") as ct_file:
                try:
                    closed_trades = json.load(ct_file)
                    if closed_trades.keys():
                        key_ints = [int(x) for x in closed_trades.keys()]
                        self.next_id = len(key_ints) + 1
                    else:
                        self.next_id = 0
                except JSONDecodeError:
                    closed_trades = {}
                    self.next_id = 0
        else:
            closed_trades = {}
            # logger.info(f'{ct_path} not found')
        e.stop()
        return closed_trades

    def read_closed_sim_trade_records(self, session) -> dict:
        """loads closed_sim_trades and returns them as a dictionary"""

        r = Timer('read_closed_sim_trade_records')
        r.start()
        cs_path = Path(f"{session.records_r}/{self.id}/closed_sim_trades.json")
        if Path(cs_path).exists():
            with open(cs_path, "r") as cs_file:
                try:
                    cs_trades = json.load(cs_file)
                except JSONDecodeError:
                    cs_trades = {}

        else:
            cs_trades = {}
            # logger.info(f'{cs_path} not found')

        limit = 5000
        if len(cs_trades.keys()) > limit:
            logger.info(f"{self.id} closed sim trades on record: {len(cs_trades.keys())}")
            closed_sim_tups = sorted(zip(cs_trades.keys(), cs_trades.values()), key=lambda x: int(x[0]))
            closed_sim_trades = dict(closed_sim_tups[-limit:])
        else:
            closed_sim_trades = cs_trades

        r.stop()
        return closed_sim_trades

    def backup_trade_records(self, session) -> None:
        """updates the backup file for each trades dictionary, on the condition
        that they are not empty"""

        y = Timer('backup_trade_records')
        y.start()

        if self.open_trades:
            with open(f"{session.records_w}/{self.id}/ot_backup.json", "w") as ot_file:
                json.dump(self.open_trades, ot_file)

        if self.sim_trades:
            with open(f"{session.records_w}/{self.id}/st_backup.json", "w") as st_file:
                json.dump(self.sim_trades, st_file)

        if self.tracked_trades:
            with open(f"{session.records_w}/{self.id}/tr_backup.json", "w") as tr_file:
                json.dump(self.tracked_trades, tr_file)

        if self.closed_trades:
            with open(f"{session.records_w}/{self.id}/ct_backup.json", "w") as ct_file:
                json.dump(self.closed_trades, ct_file)

        if self.closed_sim_trades:
            with open(f"{session.records_w}/{self.id}/cs_backup.json", "w") as cs_file:
                json.dump(self.closed_sim_trades, cs_file)

        y.stop()

    def check_valid_open(self, session) -> None:
        """checks all currently open real positions to make sure binance hasn't messed up the stop-loss execution. if a
        position is the wrong side of its stop-loss order and hasn't been automatically closed, this function will
        manually close it and record it as a stop"""

        for pair, pos in self.open_trades.items():
            position = pos['position']
            if position['direction'] in ['long', 'spot']:
                valid = float(position['hard_stop']) < session.pairs_data[pair]['price']
            else:
                valid = float(position['hard_stop']) > session.pairs_data[pair]['price']

            if not valid:
                logger.warning(
                    f"{self.id} {pair} {position['direction']} position somehow passed its stop-loss without "
                    f"closing")
                # TODO close position and record as stopped

    # record stopped trades ------------------------------------------------

    @uf.retry_on_busy()
    def find_order(self, session, pair, sid):
        if sid == 'not live':
            return dict()
        # logger.info('get_margin_order')
        session.track_weights(10)
        abc = Timer('all binance calls')
        abc.start()
        order = session.client.get_margin_order(symbol=pair, orderId=sid)
        abc.stop()
        session.counts.append('get_margin_order')

        if not order:
            logger.info(f'No orders on binance for {pair}')

        # insert placeholder record
        placeholder = {'action': "stop",
                       'direction': self.open_trades[pair]['position']['direction'],
                       'state': 'real',
                       'pair': pair,
                       'order': order,
                       'completed': 'find_order'
                       }
        self.open_trades[pair]['placeholder'] = placeholder

        return order

    def repay_stop(self, session, pair, order):
        if order.get('side') == 'BUY':
            asset = pair[:-4]
            stop_size = order.get('executedQty')
            funcs.repay_asset_M(session, asset, stop_size, session.live)
        else:
            stop_size = order.get('cummulativeQuoteQty')
            funcs.repay_asset_M(session, 'USDT', stop_size, session.live)

        self.open_trades[pair]['placeholder']['completed'] = 'repay'

        return stop_size

    def create_stop_dict(self, session, pair, order, stop_size):
        stop_dict = funcs.create_stop_dict(session, order)
        stop_dict['action'] = "stop"
        stop_dict['direction'] = self.open_trades[pair]['position']['direction']
        stop_dict['state'] = 'real'
        stop_dict['reason'] = 'hit hard stop'
        stop_dict['liability'] = uf.update_liability(self.open_trades[pair], stop_size, 'reduce')

        if float(stop_dict['liability']) > (float(stop_size) * 0.01):
            logger.warning(
                f"+++ WARNING {self.id} {pair} stop hit, liability record doesn't add up. Recorded value: "
                f"{stop_dict['liability']} +++")

        return stop_dict

    def save_records(self, session, pair, stop_dict, order):
        self.open_trades[pair]['trade'].append(stop_dict)
        self.open_trades[pair]['trade'][-1]['liability'] = str(Decimal(0) - Decimal(order['executedQty']))
        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        ts_id = int(self.open_trades[pair]['position']['open_time'])
        del self.open_trades[pair]['position']
        self.closed_trades[ts_id] = self.open_trades[pair]
        self.record_trades(session, 'closed')
        del self.open_trades[pair]
        self.record_trades(session, 'open')
        asset = pair[:-len(session.quote_asset)]
        del self.real_pos[asset]

    def error_print(self, session, pair, stage, e):
        self.record_trades(session, 'all')
        logger.exception(f'{self.id} problem with record_stopped_trades during {pair} {stage}')
        logger.error(f"code: {e.code}")
        logger.error(f"message: {e.message}")

    def rst_iteration_m(self, session, pair, order):
        """21st sept 2023 - the reason there is a load of stuff commented out here is that i have realised it is
        completely unnecessary in this function (execution will only get inside this function if there is a valid order
        and all the commented code is for when there isn't a valid order). I don't think i need that code at all any
        more because i have other things in place like repair_trade_records, however, just in case i break something by
        removing it, i have just commented rather than deleting"""

        direction = self.open_trades[pair]['position']['direction']

        # if order:
        # if an order can be found on binance, complete the records
        stop_size = self.repay_stop(session, pair, order)
        stop_dict = self.create_stop_dict(session, pair, order, stop_size)
        self.save_records(session, pair, stop_dict, order)

        self.counts_dict[f'real_stop_{direction}'] += 1

        # else:  # if no order can be found, try to close the position
        #     if session.live:
        #         logger.warning(f"record_stopped_trades found no stop for {pair} {self.id}")
        #     if direction == 'long':
        #         free_bal = session.margin_bals[pair[:-4]].get('free')
        #         pos_size = self.open_trades[pair]['position']['base_size']
        #
        #         if Decimal(free_bal) >= Decimal(pos_size):
        #             # if the free balance covers the position, close it
        #             logger.info(f'{self.id} record_stopped_trades will close {pair} now')
        #             try:
        #                 self.close_real_full_M(session, pair, direction, action)
        #             except bx.BinanceAPIException as e:
        #                 self.error_print(session, pair, 'close', e)
        #         else:
        #             # if the free balance doesn't cover the position, notify me
        #             price = session.pairs_data[pair]['price']
        #             value = free_bal * price
        #             if (pair == 'BNBUSDT' and value > 15) or (pair != 'BNBUSDT' and value > 10):
        #                 note = f'{pair} in position with no stop-loss'
        #                 # pb.push_note(session.now_start, note)
        #     else:  # if direction == 'short'
        #         owed = float(session.margin_bals[pair[:-4]].get('borrowed'))
        #         if session.live and not owed:
        #             logger.info(f'{pair[:-4]} loan already repaid, no action needed')
        #             del self.open_trades[pair]
        #         elif not owed:
        #             return
        #         else:  # if live and owed
        #             logger.info(f'{self.id} record_stopped_trades will close {pair} now')
        #             try:
        #                 size = self.open_trades[pair]['position']['base_size']
        #                 self.close_real_full_M(session, pair, direction, action, size=size, stage=2)
        #             except bx.BinanceAPIException as e:
        #                 self.error_print(session, pair, 'close', e)

    @uf.retry_on_busy()
    def record_stopped_trades(self, session, timeframes) -> None:
        m = Timer('record_stopped_trades')
        m.start()

        # get a list of (pair, stop_id, stop_time) for all open_trades records
        old_ids = list(self.open_trades.items())

        for pair, v in old_ids:
            sid = v['position']['stop_id']
            direction = v['position']['direction']
            stop_time = v['position']['stop_time']
            if sid == 'not live':
                # logger.info(f"{pair} record non-live")
                df = self.get_data(session, pair, timeframes, stop_time)
                stop = float(v['position']['hard_stop'])
                stopped, overshoot_pct, stop_hit_time = self.check_stop_hit(pair, df, direction, stop)
                if stopped:
                    logger.info(f"{self.id} {pair} real {direction} stopped out")
                    base_size = float(v['position']['base_size'])
                    stop_dict = {
                        'timestamp': int(stop_time),
                        'pair': pair,
                        'trig_price': stop,
                        'limit_price': stop,
                        'exe_price': stop,
                        'base_size': base_size,
                        'executedQty': base_size,
                        'quote_size': str(base_size * stop),
                        'fee': "0",
                        'fee_currency': 'BNB',
                        'action': 'stop',
                        'direction': direction,
                        'state': 'real',
                        'live': False,
                        'reason': 'hit hard stop',
                        'liability': v['position']['liability'] * -1,
                    }

                    order = {'executedQty': base_size}
                    self.save_records(session, pair, stop_dict, order)
                    self.counts_dict[f'real_stop_{direction}'] += 1
                else:
                    # logger.info(f"{self.id} {pair} {direction} still open")
                    pass
                continue

            if sid is None:
                # TODO repair trade records should be fixing positions without a stop so they don't get this far
                continue

            session.track_weights(10)
            abc = Timer('all binance calls')
            abc.start()
            order = self.find_order(session, pair, sid)
            abc.stop()
            session.counts.append('get_margin_order')

            if order.get('status') == 'FILLED':
                logger.info(f"{self.id} {pair} stop order filled")
                try:
                    self.rst_iteration_m(session, pair, order)
                except bx.BinanceAPIException as e:
                    self.record_trades(session, 'all')
                    logger.exception(f'{self.id} problem with record_stopped_trades during {pair}')
                    logger.error(e)
            elif order.get('status') == 'CANCELED':
                logger.warning(f'Problem with {self.id} {pair} trade record')

                # TODO it should be possible to check the placeholder in the trade record and piece together what to do
                ph = self.open_trades[pair].get('placeholder')

                if ph:
                    logger.warning(f"{self.id} {pair} stop canceled, placeholder found")
                    logger.warning('use this to modify record_stopped_trades with useful behaviour in this scenario:')
                    logger.warning(pformat(self.open_trades[pair]))
                    logger.warning('')
                    logger.warning(pformat(order))
                else:
                    logger.warning(f"{self.id} {pair} stop canceled, no placeholder found, deleting record")
                    logger.warning(pformat(self.open_trades[pair]))
                    del self.open_trades[pair]
                    del self.real_pos[pair[:-len(self.quote_asset)]]
                    self.record_trades(session, 'open')
            elif order.get('status') == 'PARTIALLY_FILLED':
                logger.info(f"{self.id} {pair} stop hit and partially filled, recording trade.")

                old_size = Decimal(order['origQty'])
                exe_size = Decimal(order['executedQty'])
                stop_price = Decimal(order['stopPrice'])

                # update trade record
                self.open_trades[pair]['trade'].append(
                    {'base_size': order['executedQty'],
                     'exe_price': order['stopPrice'],
                     'quote_size': str(exe_size * stop_price),
                     'state': 'real',
                     'stop_id': order['orderId'],
                     'timestamp': order['updateTime'],
                     'trig_price': order['stopPrice'],
                     'action': 'partstop',
                     'direction': direction}
                )

                # update position record
                self.open_trades[pair]['position']['base_size'] = str(old_size - exe_size)

                # TODO need to repay loan that was partially freed up

            else:
                # logger.info(f"{self.id} {pair} stop order (id {sid}) not filled, status: {order['status']}")
                del self.open_trades[pair]['placeholder']

        self.record_trades(session, 'closed')
        self.record_trades(session, 'open')

        m.stop()

    # record stopped sim trades ----------------------------------------------

    def get_data(self, session, pair, timeframes: list, stop_time):

        rsst_gd = Timer('rsst - get_data')
        rsst_gd.start()

        # logger.debug(f"rsst {self.id} {pair}")

        filepath = Path(f'{session.ohlc_r}/{pair}.parquet')
        check_recent = False  # flag to decide whether the ohlc needs updating or not

        if session.pairs_data[pair].get('ohlc_5m', None) is not None:
            df = session.pairs_data[pair]['ohlc_5m']
            source = 'mem'
            check_recent = True

        else:
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    source = 'file'
                    check_recent = True
                except (ArrowInvalid, OSError) as e:
                    logger.exception(f"problem loading {pair} ohlc")
                    logger.error(e)
                    filepath.unlink()
                    df = funcs.get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
                    source = 'exchange'
                    logger.info(f'downloaded {pair} from scratch')
            else:
                df = funcs.get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
                source = 'exchange'

            session.store_ohlc(df, pair, timeframes)

        # check df is localised to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        try:
            df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
        except TypeError:
            pass

        if check_recent:
            last = df.timestamp.iloc[-1]
            timespan = datetime.now(timezone.utc).timestamp() - (last.timestamp())
            if timespan > 900:
                df = funcs.update_ohlc(pair, session.ohlc_tf, df, session)
                source += ' and exchange'
                session.store_ohlc(df, pair, timeframes)

        # try:  # this try/except block can be removed when the problem is solved
        stop_dt = datetime.fromtimestamp(stop_time / 1000).astimezone(timezone.utc)
        df = df.loc[df.timestamp > stop_dt].reset_index(drop=True)
        # except ValueError as e:
        #     logger.exception(f"ValueError for {pair} sim stop time: {e.args}")
        #     logger.error(pformat(self.sim_trades[pair]))

        rsst_gd.stop()

        return df

    # def plot_trade(self, pair, df, stop, exit_dt=None):
    #     fig = go.Figure(data=go.Ohlc(x=df['timestamp'],
    #                                  open=df['open'],
    #                                  high=df['high'],
    #                                  low=df['low'],
    #                                  close=df['close']))
    #
    #     # fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f"st-{lb}-{mult}-up"], mode='lines', name='Supertrend Up'))
    #     # fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f"st-{lb}-{mult}-dn"], mode='lines', name='Supertrend Down'))
    #
    #     if exit_dt:
    #         fig.add_trace(go.Scatter(x=[exit_dt], y=[stop], mode='markers', marker=dict(color='red', size=10), name='Pnl Realised'))
    #
    #     fig.add_shape(type='line', x0=df.timestamp.iloc[0], x1=df.timestamp.iloc[-1],
    #                   y0=stop, y1=stop, line=dict(color='blue', width=2), name='Hard Stop')
    #
    #     fig.update(layout_xaxis_rangeslider_visible=False)
    #     fig.update_layout(
    #         width=1920, height=1080,
    #         title=f"{pair}",
    #         autotypenumbers='convert types',
    #         # xaxis=dict(
    #         #   title='Date and Time',
    #         # ),
    #         # yaxis=dict(
    #         #   title='Price',
    #         # ),
    #     )
    #     plot_folder = Path(f"/home/ross/Documents/backtester_2021/trade_plots/{self.id}")
    #     plot_folder.mkdir(parents=True, exist_ok=True)
    #     fig.write_image(f"{plot_folder}/{pair}.png")

    def check_stop_hit(self, pair, df, direction, stop):
        func_name = sys._getframe().f_code.co_name
        k16 = Timer(f'{func_name}')
        k16.start()

        stop_hit_time = None
        if direction == 'long':
            ll = df.low.min()
            stopped = ll < stop
            overshoot_pct = round((100 * (stop - ll) / stop), 3)  # % distance that price broke through the stop
            if stopped:
                stop_hit_dt = df.loc[df.low <= stop].timestamp.iloc[0]
                if isinstance(stop_hit_dt, pd.Timestamp):
                    stop_hit_time = stop_hit_dt.timestamp()

        else:
            hh = df.high.max()
            stopped = hh > stop
            overshoot_pct = round((100 * (hh - stop) / stop), 3)  # % distance that price broke through the stop
            if stopped:
                stop_hit_dt = df.loc[df.high >= stop].timestamp.iloc[0]
                if isinstance(stop_hit_dt, pd.Timestamp):
                    stop_hit_time = stop_hit_dt.timestamp()

        if stop_hit_time:
            stop_hit_time = int(stop_hit_time)

        k16.stop()

        return stopped, overshoot_pct, stop_hit_time

    def create_trade_dict(self, pair, direction, stop, base_size, stop_hit_time, overshoot_pct, state):
        liability = f"{0 - (base_size * stop):.2f}" if direction == 'long' else f"{0 - base_size:.2f}"
        trade_dict = {'timestamp': stop_hit_time,
                      'pair': pair,
                      'direction': direction,
                      'action': 'stop',
                      'exe_price': str(stop),
                      'base_size': str(base_size),
                      'quote_size': str(round(base_size * stop, 2)),
                      'fee': 0,
                      'fee_currency': 'BNB',
                      'reason': 'hit hard stop',
                      'state': state,
                      'overshoot': overshoot_pct,
                      'liability': liability
                      }

        return trade_dict

    def record_stopped_sim_trades(self, session, timeframes: list) -> None:
        """goes through all trades in the sim_trades file and checks their recent price action
        against their most recent hard_stop to see if any of them would have got stopped out"""

        n = Timer('record_stopped_sim_trades')
        n.start()
        session.counts.append('rsst')

        check_pairs = list(self.sim_trades.items())
        for pair, v in check_pairs:  # can't loop through the dictionary directly because i delete items as i go
            direction = v['position']['direction']
            base_size = float(v['position']['base_size'])
            stop = float(v['position']['hard_stop'])
            stop_time = v['position']['stop_time']

            df = self.get_data(session, pair, timeframes, stop_time)
            if df.empty:
                logger.warning(f"RSST couldn't find a valid stop time for {pair} {direction}")
                logger.warning(f"Stop time on record: {stop_time}")
                continue

            stopped, overshoot_pct, stop_hit_time = self.check_stop_hit(pair, df, direction, stop)
            if stopped:
                trade_dict = self.create_trade_dict(pair, direction, stop, base_size, stop_hit_time, overshoot_pct,
                                                    'sim')
                self.sim_to_closed_sim(session, pair, trade_dict, save_file=False)
                self.counts_dict[f'sim_stop_{direction}'] += 1

        self.record_trades(session, 'closed_sim')
        self.record_trades(session, 'sim')

        n.stop()

    # risk ----------------------------------------------------------------------

    def realised_pnl(self, trade_record: dict) -> float:
        """calculates realised pnl of a tp or close denominated in the trade's
        own R value"""

        # TODO this function sometimes gets called during repair_trade_records. i need to make sure that it would still
        #  work even if the record was broken

        func_name = sys._getframe().f_code.co_name
        k15 = Timer(f'{func_name}')
        k15.start()

        position = trade_record['position']

        trades = trade_record['trade']
        entry = float(position['entry_price'])
        init_stop = float(position['init_hard_stop'])
        final_exit = float(trades[-1].get('exe_price'))

        r_val = (entry - init_stop) / entry
        trade_pnl = (final_exit - entry) / entry
        trade_r = round(trade_pnl / r_val, 3)
        scalar = position['pct_of_full_pos']
        realised_r = trade_r * scalar

        logger.debug(f"{position['pair']} realised_r: {realised_r:.2f}")
        k15.stop()

        return realised_r

    def record_trades(self, session, state: str) -> None:
        """saves any trades dictionary to its respective json file"""

        b = Timer(f'record_trades {state}')
        b.start()
        session.counts.append(f'record_trades {state}')

        if state in {'open', 'all'}:
            filepath = Path(f"{session.records_w}/{self.id}/open_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.open_trades, file)
        if state in {'sim', 'all'}:
            filepath = Path(f"{session.records_w}/{self.id}/sim_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.sim_trades, file)
        if state in {'tracked', 'all'}:
            filepath = Path(f"{session.records_w}/{self.id}/tracked_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.tracked_trades, file)
        if state in {'closed', 'all'}:
            filepath = Path(f"{session.records_w}/{self.id}/closed_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.closed_trades, file)
        if state in {'closed_sim', 'all'}:
            filepath = Path(f"{session.records_w}/{self.id}/closed_sim_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.closed_sim_trades, file)

        b.stop()

    def check_open_risk(self, session):
        """goes through all trades in the open_trades and sim_trades dictionaries and checks their open risk, then adds
        them to the list of tp/close signals if necessary"""

        n = Timer('check_open_risk')
        n.start()

        signals = []

        positions = list(self.open_trades.items()) + list(self.sim_trades.items())

        for pair, pos in positions:

            # TODO from here down to the open risk condition might be the exact functionality of update_pos. maybe i
            #  could redefine update_pos as these lines and then just call update_pos here

            direction = pos['position']['direction']
            price = session.pairs_data[pair]['price']
            current_size = float(pos['position']['base_size'])
            current_value = price * current_size
            bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
            pct = round(100 * float(current_value) / bal, 2)

            ep = float(pos['position']['entry_price'])
            price_delta = (price - ep) / ep

            open_risk = uf.open_risk_calc(session, pos, 'all')

            asset = pair[:-len(session.quote_asset)]
            state = pos['position']['state']
            if state == 'real':
                self.real_pos[asset]['value'] = current_value
                self.real_pos[asset]['pf%'] = pct
                self.real_pos[asset]['or_$'] = open_risk['usdt']
                self.real_pos[asset]['or_R'] = open_risk['r']
                self.real_pos[asset]['price_delta'] = price_delta
            elif state == 'sim':
                self.sim_pos[asset]['value'] = current_value
                self.sim_pos[asset]['pf%'] = pct
                self.sim_pos[asset]['or_$'] = open_risk['usdt']
                self.sim_pos[asset]['or_R'] = open_risk['r']
                self.sim_pos[asset]['price_delta'] = price_delta

            # identify wanted positions by checking sim reasons (subbed with empty list for real positions)
            if 'low_score' not in pos['signal'].get('sim_reasons', []):
                if not session.open_risk_records.get(f"{self.tf}_{self.name}"):
                    session.open_risk_records[f"{self.tf}_{self.name}"] = []
                session.open_risk_records[f"{self.tf}_{self.name}"].append(open_risk['r'])

            if 0 <= open_risk['r'] < self.indiv_r_limit:
                # logger.debug(f"{self.id} {pair} {direction} open risk: {open_risk['r']:.3f}R, within limits "
                #              f"(0-{self.indiv_r_limit})")
                continue

            # TODO i could check for failed stops here too by simply looking for positions with negative open risk.
            #  in these cases, perhaps i could create a signal with 'stop' as the action, then arrange for those signals
            #  to be sent to the close_position omf

            signal = {
                'agent': self.id,
                'mode': self.mode,
                'tf': self.tf,
                'pair': pair,
                'asset': asset,
                'action': 'tp',
                'direction': direction,
                'state': state,
                'inval': float(pos['position']['hard_stop'])
            }

            if current_value < session.min_size:
                signal['action'] = 'close'
            if open_risk['r'] < 0:
                signal['action'] = 'stop'

            signals.append(signal)

            logger.debug(f"{state} {pair} open risk over threshold, or: {open_risk['r']:.1f}R. "
                         f"Size: {current_value:.2f}USDT so action is {signal['action']}")
            logger.info(f"{state} {pair} open risk over threshold, or: {open_risk['r']:.1f}R. "
                        f"Size: {current_value:.2f}USDT so action is {signal['action']}")

        return signals

    def reduce_fr(self, factor: float, fr_prev: float, fr_inc: float):
        """reduces fixed_risk by factor (with the floor value being 0)"""
        ideal = fr_prev * factor
        reduce = max(ideal, fr_inc)
        return max((fr_prev - reduce), 0)

    def get_pnls(self, direction: str) -> dict:
        """ retrieves the pnls of all closed (real and wanted sim) trades for the agent, then collates the pnls into a
        dataframe, linearly transforms them (with clipping) to a scale of 0-1 where zero is any pnl <= -1R and 1 is any
        pnl >= 1R, and calculates several moving averages on them. it then returns a dictionary of the latest row of
        those moving averages"""

        all_rpnls = []
        for a, b in self.closed_trades.items():
            wanted = (b['trade'][0]['state'] == 'real') or (b['trade'][0]['wanted'])
            right_direction = b['trade'][0]['direction'] == direction
            if wanted and right_direction:
                rpnl = 0
                for t in b['trade']:
                    if t.get('rpnl'):
                        rpnl += float(t['rpnl'])
                all_rpnls.append((int(a), rpnl))
        for a, b in self.closed_sim_trades.items():
            wanted = b['trade'][0]['wanted']
            right_direction = b['trade'][0]['direction'] == direction
            if wanted and right_direction:
                rpnl = 0
                for t in b['trade']:
                    if t.get('rpnl'):
                        rpnl += float(t['rpnl'])
                all_rpnls.append((int(a), rpnl))
        rpnl_df = pd.DataFrame(all_rpnls, columns=['timestamp', 'rpnl'])
        rpnl_df['scaled_rpnl'] = ((rpnl_df.rpnl + 1) / 2).clip(lower=0.0, upper=1.0)
        rpnl_df = rpnl_df.sort_values('timestamp').reset_index(drop=True)
        rpnl_df['ema_4'] = rpnl_df.scaled_rpnl.ewm(4).mean()
        rpnl_df['ema_8'] = rpnl_df.scaled_rpnl.ewm(8).mean()
        rpnl_df['ema_16'] = rpnl_df.scaled_rpnl.ewm(16).mean()
        rpnl_df['ema_32'] = rpnl_df.scaled_rpnl.ewm(32).mean()
        rpnl_df['ema_64'] = rpnl_df.scaled_rpnl.ewm(64).mean()

        if len(rpnl_df) >= 4:
            return rpnl_df.to_dict(orient='records')[-1]
        else:
            return {'ema_4': 0, 'ema_8': 0, 'ema_16': 0, 'ema_32': 0, 'ema_64': 0}

    def set_max_pos(self) -> int:
        """sets the maximum number of open positions for the agent. if the median
        pnl of current open positions is greater than 0, max pos will be set to 50,
        otherwise max_pos will be set to 20"""

        p = Timer('set_max_pos')
        p.start()
        avg_open_pnl = 0
        opnls = [v.get('pnl_R') for k, v in self.real_pos.items() if k != 'USDT']
        if opnls:
            avg_open_pnl = stats.median(opnls)
        logger.debug(f"set_max_pos calculates {avg_open_pnl = }")
        max_pos = 6 if avg_open_pnl <= 0 else 12
        p.stop()
        return max_pos

    def calc_tor(self) -> None:
        """collects all the open risk values from real_pos into a list and
        calculates the sum total of all the open risk for the agent in question"""

        u = Timer('calc_tor')
        u.start()
        self.or_list = [float(v.get('or_R')) for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
        u.stop()

    # signal scores -------------------------------------------------------------

    def calc_inval_risk_score(self, inval: float) -> float:
        """i want to analyse the probability of any given value of inval_risk at trade entry producing a positive pnl.
        Once i have a set of scores (1 for each band of inval_risk range) i can normalise them to a 0-1 range and output
        that as the inval_risk_score"""

        # TODO ulitmately i want this function to work as described in the docstring but for a quick fix it can just use
        #  a scalar on the inval distance for each timeframe.

        score = 0
        if self.tf == '1h':
            score = max(1 - (inval * 10), 0)  # inval of 0% returns max score of 1, 10% or more returns 0
        elif self.tf == '4h':
            score = max(1 - (inval * 5), 0)  # inval of 20% or more returns 0
        elif self.tf == '12h':
            score = max(1 - (inval * 3), 0)  # inval of 33% or more returns 0
        elif self.tf == '1d':
            score = max(1 - (inval * 2), 0)  # inval of 50% or more returns 0

        return score

    # positions -----------------------------------------------------------------

    def open_trade_stats(self, session, total_bal: float, v: dict) -> dict:
        """takes an entry from the open_trades dictionary, returns information about
        that position including current profit, size and direction"""

        we = Timer('open_trade_stats')
        we.start()

        pair = v['position']['pair']

        current_base_size = v['position']['base_size']
        entry_price = float(v['position']['entry_price'])

        open_time = int(v['position']['open_time'])
        now = datetime.now(timezone.utc)
        duration = round((now.timestamp() * 1000 - open_time) / 3600, 1)

        direction = v['position']['direction']

        curr_price = session.pairs_data[pair]['price']
        long = v['position']['direction'] == 'long'
        pos_scale = v['position']['pct_of_full_pos']
        trig = float(v['position']['entry_price'])
        init_sl = float(v['position']['init_hard_stop'])
        r = 100 * abs(trig - init_sl) / init_sl
        if long:
            pnl = 100 * (curr_price - entry_price) / entry_price
            pnl_r = round((pnl / r) * pos_scale, 5)
        else:
            pnl = 100 * (entry_price - curr_price) / entry_price
            pnl_r = round((pnl / r) * pos_scale, 5)
        value = round(float(current_base_size) * curr_price, 2)
        pf_pct = round(100 * value / total_bal, 2)

        wanted = v['trade'][0].get('wanted', True)

        stats_dict = {'value': str(value), 'pf%': pf_pct, 'duration (h)': duration,
                      'pnl_R': pnl_r, 'pnl_%': round(pnl, 5),
                      'direction': direction, 'wanted': wanted
                      }

        we.stop()
        return stats_dict

    def current_positions(self, session, state: str) -> dict:
        """creates a dictionary of open positions by checking either
        open_trades.json, sim_trades.json or tracked_trades.json"""

        a = Timer(f'current_positions-{state}')
        a.start()

        if state == 'open':
            data = self.open_trades
        elif state == 'sim':
            data = self.sim_trades
        else:
            data = self.tracked_trades

        size_dict = {}
        drop_items = []
        for k, v in data.items():

            # deal with open trades on delisted pairs
            if k not in session.pairs_data.keys():
                if state == 'open':
                    # TODO need to work out what to do with open trades on coins which have been delisted
                    continue
                elif state == 'sim':
                    drop_items.append(k)
                    continue
                elif state == 'tracked':
                    # TODO need to work out what to do with tracked trades on coins which have been delisted
                    continue

            asset = k[:-4]
            if v['position']['direction'] == 'spot':
                total_bal = session.spot_bal
            else:
                total_bal = session.margin_bal

            if state != 'tracked':
                try:
                    size_dict[asset] = self.open_trade_stats(session, total_bal, v)
                except KeyError as e:
                    logger.exception(f"Problem calling open_trade_stats on {self.id}.{state}_trades, {asset}")
                    logger.error('KeyError:', e)
                    logger.error('')
                    logger.error(pformat(v))
                    logger.error('')
            else:
                direction = v['position']['direction']
                wanted = v['trade'][0].get('wanted', True)
                size_dict[asset] = {'value': '0', 'pf%': 0, 'duration (h)': 0, 'pnl_R': 0,
                                    'pnl_%': 0, 'direction': direction, 'wanted': wanted}

        for i in drop_items:
            if state == 'open':
                del self.open_trades[i]
            if state == 'sim':
                del self.sim_trades[i]
            if state == 'tracked':
                del self.tracked_trades[i]
        self.record_trades(session, state)

        if self.mode == 'spot':
            size_dict['USDT'] = session.spot_usdt_bal
        elif self.mode == 'margin':
            size_dict['USDT'] = session.margin_usdt_bal

        a.stop()
        return size_dict

    def update_pos(self, session, pair: str, new_bal: str, state: str) -> Dict[str, float]:
        """checks for the current balance of a particular asset and returns it in
        the correct format for the sizing dict. also calculates the open risk for
        a given asset and returns it in R and $ denominations.
        The dictionary returned by this method should be used to update the existing position dictionary using the
        '.update()' dictionary method."""

        jk = Timer('update_pos')
        jk.start()

        price = session.pairs_data[pair]['price']

        if state == 'real':
            trade_record = self.open_trades[pair]
        elif state == 'sim':
            trade_record = self.sim_trades[pair]
        else:
            trade_record = self.tracked_trades[pair]

        value = f"{price * float(new_bal):.2f}"
        bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
        pct = round(100 * float(value) / bal, 2)

        or_dict = uf.open_risk_calc(session, trade_record, 'all')

        jk.stop()

        return {'value': value, 'pf%': pct,
                'or_R': or_dict['r'],
                'or_$': or_dict['usdt']}

    def update_non_live_tp(self, session, asset: str, state: str) -> dict:  # dict[str, float | str | Any]:
        """updates sizing dictionaries (real/sim) with new open trade stats when
        state is sim or real but not live and a take-profit is triggered"""
        qw = Timer('update_non_live_tp')
        qw.start()

        pair = asset + session.quote_asset
        if state == 'real':
            trades = self.open_trades[pair]
        elif state == 'sim':
            trades = self.sim_trades[pair]
        else:
            trades = self.tracked_trades[pair]

        bal = session.margin_bal if self.mode == 'margin' else session.spot_bal
        base_size = float(trades['position']['base_size'])
        stop = float(trades['position']['hard_stop'])
        pfrd = float(trades['position']['pfrd'])
        pos_scale = trades['position']['pct_of_full_pos']
        price = session.pairs_data[pair]['price']
        val = base_size * price
        dist_to_inval = abs(price - stop) / price

        pf = val / bal
        or_dol = dist_to_inval * val
        or_r = (or_dol / pfrd) * pos_scale

        qw.stop()

        return {'value': f"{val:.2f}", 'pf%': pf, 'or_R': or_r, 'or_$': or_dol}

    # move_stop -----------------------------------------------------------------
    def create_placeholder(self, pair, direction, atr):
        now = datetime.now(timezone.utc).strftime(timestring)
        placeholder = {'action': 'move_stop',
                       'direction': direction,
                       'state': 'real',
                       'pair': pair,
                       'stop_price': atr,
                       'utc_datetime': now,
                       'completed': None
                       }
        self.open_trades[pair]['placeholder'] = placeholder

    def clear_stop(self, session, pair, pos_record):
        _, base_size = funcs.clear_stop_M(session, pair, pos_record)
        if not base_size:
            logger.info('--- running move_stop and needed to use position base size')
            base_size = pos_record['base_size']
        # else:
        #     logger.info("--- running move_stop and DIDN'T need to use position base size")

        return base_size

    def update_records_1(self, session, pair, base_size):
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = base_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'
        self.record_trades(session, 'open')

    def reset_stop(self, session, pair, base_size, direction, atr):
        trade_side = be.SIDE_SELL if direction == 'long' else be.SIDE_BUY
        stop_order = funcs.set_stop_M(session, pair, base_size, trade_side, atr)

        return stop_order

    def update_records_2(self, session, pair, atr, stop_order):
        self.open_trades[pair]['position']['hard_stop'] = atr
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime'))
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def move_api_stop(self, session, pair, direction, atr, pos_record, stage=0):
        if stage == 0:
            self.create_placeholder(pair, direction, atr)
        if stage <= 1:
            base_size = self.clear_stop(session, pair, pos_record)

            self.update_records_1(session, pair, base_size)

        if stage <= 2:
            if stage == 2:
                base_size = pos_record['base_size']
            stop_order = self.reset_stop(session, pair, base_size, direction, atr)

            self.update_records_2(session, pair, atr, stop_order)

    def move_real_stop(self, session, signal):
        func_name = sys._getframe().f_code.co_name
        k14 = Timer(f'{func_name}')
        k14.start()

        pair = signal['pair']
        direction = 'long' if (signal['bias'] == 'bullish') else 'short'
        inval = signal['inval']

        move_condition = False
        try:
            current_stop = float(self.open_trades[pair]['position']['hard_stop'])
            move_condition = (((direction in ['long', 'spot']) and (inval > (current_stop * 1.001)))
                              or ((direction == 'short') and (inval < (current_stop / 1.001))))
        except TypeError as e:
            logger.error(f"move_real_stop encountered an error on {self.id} {pair}")
            logger.error(pformat(self.open_trades[pair]['position']))
            logger.exception(e)

        if move_condition:
            logger.debug(f"*** {self.id} {pair} move real {direction} stop from {current_stop:.5} to {inval:.5}")
            try:
                self.move_api_stop(session, pair, direction, inval, self.open_trades[pair]['position'])
            except bx.BinanceAPIException as e:
                self.record_trades(session, 'all')
                logger.exception(f'{self.id} problem with move_stop order for {pair}')
                logger.error(e)

        asset = signal['pair'][:-len(session.quote_asset)]
        size = self.open_trades[pair]['position']['base_size']
        self.real_pos[asset].update(self.update_pos(session, pair, size, 'real'))

        k14.stop()

    def move_non_real_stop(self, session, signal, state):
        func_name = sys._getframe().f_code.co_name
        k13 = Timer(f'{func_name}')
        k13.start()

        pair = signal['pair']
        direction = 'long' if (signal['bias'] == 'bullish') else 'short'
        inval = signal['inval']

        if state == 'sim':
            trade_record = self.sim_trades[pair]
        else:
            trade_record = self.tracked_trades[pair]

        move_condition = False
        try:
            current_stop = float(trade_record['position']['hard_stop'])
            move_condition = (((direction in ['long', 'spot']) and (inval > current_stop))
                              or ((direction == 'short') and (inval < current_stop)))
        except TypeError as e:
            logger.error(f"Error while trying to move {state} stop on {self.id} {pair}")
            logger.error(pformat(trade_record['position']))
            logger.exception(e)

        if move_condition:
            if state == 'sim':
                self.sim_trades[pair]['position']['hard_stop'] = inval
            elif state == 'tracked':
                self.tracked_trades[pair]['position']['hard_stop'] = inval

        asset = signal['pair'][:-len(session.quote_asset)]
        size = trade_record['position']['base_size']
        if state == 'sim':
            self.sim_pos[asset].update(self.update_pos(session, pair, size, state))
        elif state == 'tracked':
            self.tracked[asset].update(self.update_pos(session, pair, size, state))
        self.record_trades(session, state)

        k13.stop()

    # dispatch ---------------------------------------------------------------------------------------------------------

    def tp_pos(self, session, signal):
        if signal['state'] == 'real' and signal['mode'] == 'margin':
            self.tp_real_full_M(session, signal['pair'], signal['inval'], signal['direction'])

        elif signal['state'] == 'real' and signal['mode'] == 'spot':
            self.tp_real_full_s(session, signal['pair'], signal['inval'])

        elif signal['state'] == 'sim':
            self.tp_sim(session, signal['pair'], signal['inval'], signal['direction'])

        elif signal['state'] == 'tracked':
            self.tp_tracked(session, signal['pair'], signal['inval'], signal['direction'])

    def close_pos(self, session, signal):
        if signal['state'] == 'real' and signal['mode'] == 'margin':
            self.close_real_full_M(session, signal['pair'], signal['direction'], signal['action'])

        elif signal['state'] == 'real' and signal['mode'] == 'spot':
            self.close_real_full_s(session, signal['pair'], signal['action'])

        elif signal['state'] == 'sim':
            self.close_sim(session, signal['pair'], signal['direction'])

        elif signal['state'] == 'tracked':
            self.close_tracked(session, signal['pair'], signal['direction'])

    # real open margin -------------------------------------------------------------------------------------------------

    def create_record(self, signal):
        pair = signal['pair']
        direction = signal['direction']
        now = datetime.now(timezone.utc).strftime(timestring)

        placeholder = {'utc_datetime': now,
                       'completed': None
                       }
        self.open_trades[pair] = {}
        self.open_trades[pair]['placeholder'] = placeholder
        self.open_trades[pair]['signal'] = signal
        self.open_trades[pair]['position'] = {'pair': pair, 'direction': direction, 'state': 'real', 'agent': self.id}

        size = signal['base_size']
        usdt_size = signal['quote_size']
        price = signal['trig_price']
        stp = signal['inval']
        score = float(signal['score'])
        note = (f"{self.id} real open {direction} {size:.5} {pair} ({usdt_size:.2f} usdt) @ {price}, "
                f"stop @ {stp:.5}, score: {score:.1%}")
        logger.info(note)

    def omf_borrow(self, session, pair, size, direction):
        if direction == 'long':
            price = session.pairs_data[pair]['price']
            borrow_size = f"{size * price:.2f}"
            borrow_size = funcs.borrow_asset_M(session, 'USDT', borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = 'USDT'
        else:
            asset = pair[:-4]
            borrow_size = uf.valid_size(session, pair, size)
            borrow_size = funcs.borrow_asset_M(session, asset, borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = asset

        if borrow_size:
            self.open_trades[pair]['position']['liability'] = borrow_size
            self.open_trades[pair]['placeholder']['liability'] = borrow_size
            self.open_trades[pair]['placeholder']['completed'] = 'borrow'
            return True
        else:
            return False

    def increase_position(self, session, pair, size, direction):
        price = session.pairs_data[pair]['price']
        usdt_size = f"{size * price:.2f}"

        if direction == 'long' and session.pairs_data[pair]['qoq_allowed']:
            api_order = funcs.buy_asset_M(session, pair, float(usdt_size), False, session.live)
        elif direction == 'long':
            api_order = funcs.buy_asset_M(session, pair, size, True, session.live)
        elif direction == 'short':
            api_order = funcs.sell_asset_M(session, pair, size, session.live)

        self.open_trades[pair]['position']['base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['init_base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['open_time'] = int(api_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return api_order

    def open_trade_dict(self, session, signal, api_order):
        pair = signal['pair']
        price = session.pairs_data[pair]['price']

        open_order = funcs.create_trade_dict(api_order, price, session.live)
        open_order['pair'] = pair
        open_order['action'] = "open"
        open_order['direction'] = signal['direction']
        open_order['state'] = 'real'
        open_order['score'] = 'signal score'
        open_order['hard_stop'] = str(signal['inval'])

        pfrd = signal['quote_size'] * abs(1 - signal['inval_ratio'])
        self.open_trades[pair]['position']['pfrd'] = str(pfrd)
        self.open_trades[pair]['position']['entry_price'] = open_order['exe_price']
        self.open_trades[pair]['position']['pct_of_full_pos'] = signal['pct_of_full_pos']
        self.open_trades[pair]['placeholder']['open_order'] = open_order  # this is for repair_trade_records
        self.open_trades[pair]['placeholder'].update(open_order)  # this is for normal operation
        self.open_trades[pair]['placeholder']['completed'] = 'trade_dict'

        return open_order

    def open_set_stop(self, session, pair, stp, open_order, direction):
        # stop_size = float(open_order.get('base_size'))
        stop_size = open_order.get('base_size')  # this is a string, go back to using above line if this causes bugs

        if direction == 'long':
            stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_SELL, stp)
        elif direction == 'short':
            stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_BUY, stp)

        open_order['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['hard_stop'] = str(stp)
        self.open_trades[pair]['position']['init_hard_stop'] = str(stp)
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

    def open_save_records(self, session, pair):
        del self.open_trades[pair]['placeholder']['completed']
        del self.open_trades[pair]['placeholder']['api_order']
        del self.open_trades[pair]['placeholder']['open_order']
        self.open_trades[pair]['trade'] = [self.open_trades[pair]['placeholder']]
        del self.open_trades[pair]['placeholder']
        try:
            self.record_trades(session, 'open')
        except TypeError as e:
            logger.exception(e)
            logger.error(pformat(self.open_trades))

    def open_update_real_pos_usdtM_counts(self, session, pair, size, inval_ratio, direction):
        price = session.pairs_data[pair]['price']
        usdt_size = f"{size * price:.2f}"
        asset = pair[:-4]

        if session.live:
            # self.real_pos[asset] = self.update_pos(session, pair, size, 'real')
            self.real_pos[asset] = self.open_trade_stats(session, session.margin_bal, self.open_trades[pair])
            self.real_pos[asset]['pnl_R'] = 0
            if direction == 'long':
                session.update_usdt_m(borrow=float(usdt_size))
            elif direction == 'short':
                session.update_usdt_m(up=float(usdt_size))
        else:  # TODO does this really need to be done differently from live trades?
            bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
            pf = f"{100 * float(usdt_size) / bal:.2f}"
            or_dol = f"{float(usdt_size) * abs(1 - inval_ratio):.2f}"
            self.real_pos[asset] = {'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol), 'pnl_R': 0}

        self.counts_dict[f'real_open_{direction}'] += 1
        self.num_open_positions += 1

    def open_real_M(self, session, signal, stage):

        func_name = sys._getframe().f_code.co_name
        k11 = Timer(f'{func_name}')
        k11.start()

        pair = signal['pair']
        direction = signal['direction']
        size = signal['base_size']
        stp = signal['inval']
        inval_ratio = signal['inval_ratio']

        placeholder = self.open_trades.get(pair, dict()).get(
            'placeholder')  # if placeholder is needed in stage 1 or 2, this
        # will contain the relevant data. if it is empty, it won't be needed anyway.

        if stage == 0:
            self.create_record(signal)
            successful = self.omf_borrow(session, pair, size, direction)
            if not successful:
                del self.open_trades[pair]
                return False
            try:
                api_order = self.increase_position(session, pair, size, direction)
            except bx.BinanceAPIException as e:
                logger.exception(e)
                if e.code == -2010:
                    del self.open_trades[pair]
                    return False

        if stage <= 1:
            if stage == 1:
                api_order = placeholder['api_order']
            open_order = self.open_trade_dict(session, signal, api_order)
        if stage <= 2:
            if stage == 2:
                open_order = placeholder['open_order']
            self.open_set_stop(session, pair, stp, open_order, direction)
        if stage <= 3:
            self.open_save_records(session, pair)
            self.open_update_real_pos_usdtM_counts(session, pair, size, inval_ratio, direction)
        k11.stop()

        return True

    # real open spot ---------------------------------------------------------------------------------------------------

    def open_real_s(self, session, signal, stage):
        func_name = sys._getframe().f_code.co_name
        ros = Timer(f'{func_name}')
        ros.start()

        if stage == 0:
            self.create_record(signal)

        ros.stop()

    # real tp ----------------------------------------------------------------------------------------------------------

    def create_tp_placeholder(self, session, pair, stp, direction):
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        # insert placeholder record
        placeholder = {'action': 'tp',
                       'direction': direction,
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'stop_price': stp,
                       'inval': price / float(stp),
                       'utc_datetime': now,
                       'completed': None
                       }
        self.open_trades[pair]['placeholder'] = placeholder

    def tp_set_pct(self, pair):
        asset = pair[:-4]

        real_val = abs(Decimal(self.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100

        return pct

    def tp_clear_stop(self, session, pair):
        clear, cleared_size = funcs.clear_stop_M(session, pair, self.open_trades[pair]['position'])
        cleared_size = self.check_size_against_records(pair, cleared_size)

        # update position and placeholder
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

        return cleared_size

    def tp_reduce_position(self, session, pair, base_size, pct, direction):
        price = session.pairs_data[pair]['price']
        order_size = float(base_size) * (pct / 100)
        if direction == 'long':
            api_order = funcs.sell_asset_M(session, pair, order_size, session.live)
        elif direction == 'short':
            api_order = funcs.buy_asset_M(session, pair, order_size, True, session.live)

        # update records
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        self.open_trades[pair]['position']['pct_of_full_pos'] *= (pct / 100)
        # logger.debug(f"+++ {self.id} {pair} tp {direction} resulted in base qty: {new_base_size}")
        tp_order = funcs.create_trade_dict(api_order, price, session.live)

        self.open_trades[pair]['placeholder']['tp_order'] = tp_order
        self.open_trades[pair]['placeholder']['pct'] = pct
        self.open_trades[pair]['placeholder']['order_size'] = order_size
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return tp_order

    def tp_repay_100(self, session, pair, tp_order, direction):
        price = session.pairs_data[pair]['price']
        asset = pair[:-4]
        liability = self.open_trades[pair]['position']['liability']
        if direction == 'long':
            repay_size = str(max(Decimal(tp_order.get('quote_size', 0)), Decimal(liability)))
            funcs.repay_asset_M(session, 'USDT', repay_size, session.live)
        else:
            repay_size = str(max(Decimal(liability), Decimal(tp_order.get('base_size', 0))))
            funcs.repay_asset_M(session, asset, repay_size, session.live)

        # update trade dict
        tp_order['action'] = 'close'
        tp_order['direction'] = direction
        tp_order['state'] = 'real'
        tp_order['reason'] = 'trade over-extended'

        liability = Decimal(self.open_trades[pair]['position']['liability'])
        self.open_trades[pair]['position']['liability'] = str(liability - Decimal(repay_size))
        tp_order['liability'] = str(Decimal(0) - Decimal(repay_size))

        self.open_trades[pair]['placeholder'].update(tp_order)
        self.open_trades[pair]['placeholder']['tp_order'] = tp_order
        self.open_trades[pair]['placeholder']['completed'] = 'repay_100'

        return tp_order

    def open_to_tracked(self, session, pair, stp, close_order, direction):
        now = int(datetime.now().timestamp() * 1000)

        self.open_trades[pair]['trade'].append(close_order)
        self.open_trades[pair]['trade'][-1]['utc_datetime'] = self.open_trades[pair]['placeholder']['utc_datetime']

        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        del self.open_trades[pair]['placeholder']
        self.open_trades[pair]['position']['state'] = 'tracked'
        self.open_trades[pair]['position']['hard_stop'] = str(stp)
        self.open_trades[pair]['position']['stop_time'] = now

        self.tracked_trades[pair] = self.open_trades[pair]
        self.record_trades(session, 'tracked')

        del self.open_trades[pair]
        self.record_trades(session, 'open')
        asset = pair[:-len(session.quote_asset)]
        del self.real_pos[asset]

        self.tracked[asset] = {'qty': '0', 'value': '0', 'pf%': '0', 'or_R': '0', 'or_$': '0'}

    def tp_update_records_100(self, session, pair, order_size, direction):
        asset = pair[:-4]
        price = session.pairs_data[pair]['price']
        usdt_size = f"{float(order_size) * price:.2f}"

        if session.live and direction == 'long':
            session.update_usdt_m(repay=float(usdt_size))
        elif session.live and direction == 'short':
            session.update_usdt_m(down=float(usdt_size))
        elif (not session.live) and direction == 'long':
            self.real_pos['USDT']['qty'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] += float(self.real_pos[asset].get('pf%'))
        elif (not session.live) and direction == 'short':
            self.real_pos['USDT']['qty'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] -= float(self.real_pos[asset].get('pf%'))

        self.counts_dict[f'real_close_{direction}'] += 1

    def tp_repay_partial(self, session, pair, stp, tp_order, direction):
        asset = pair[:-4]

        if direction == 'long':
            repay_size = tp_order.get('quote_size')
            funcs.repay_asset_M(session, 'USDT', repay_size, session.live)
        elif direction == 'short':
            repay_size = tp_order.get('base_size')
            funcs.repay_asset_M(session, asset, repay_size, session.live)

        # create trade dict
        tp_order['action'] = 'tp'
        tp_order['direction'] = direction
        tp_order['state'] = 'real'
        tp_order['hard_stop'] = str(stp)
        tp_order['reason'] = 'trade over-extended'

        liability = Decimal(self.open_trades[pair]['position']['liability'])
        self.open_trades[pair]['position']['liability'] = str(liability - Decimal(repay_size))
        tp_order['liability'] = str(Decimal(0) - Decimal(repay_size))

        self.open_trades[pair]['placeholder'].update(tp_order)
        self.open_trades[pair]['placeholder']['tp_order'] = tp_order
        self.open_trades[pair]['placeholder']['completed'] = 'repay_part'

        return tp_order

    def tp_reset_stop(self, session, pair, stp, tp_order, direction):
        new_size = self.open_trades[pair]['position']['base_size']

        if direction == 'long':
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_SELL, stp)
        elif direction == 'short':
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_BUY, stp)

        tp_order['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['hard_stop'] = str(stp)
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['hard_stop'] = str(stp)
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return tp_order

    def open_to_open(self, session, pair, tp_order):
        self.open_trades[pair]['trade'].append(tp_order)
        self.open_trades[pair]['trade'][-1]['utc_datetime'] = self.open_trades[pair]['placeholder']['utc_datetime']

        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl
        logger.info(f"{pair} real tp recorded {rpnl} in real_{direction} and wanted_{direction}")

        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def tp_update_records_partial(self, session, pair, pct, stp, order_size, tp_order, direction):
        asset = pair[:-len(session.quote_asset)]
        price = session.pairs_data[pair]['price']
        new_size = self.open_trades[pair]['position']['base_size']

        pfrd = float(self.open_trades[pair]['position']['pfrd'])
        inval_ratio = price / float(stp)
        if session.live:
            self.real_pos[asset].update(
                self.update_pos(session, pair, new_size, 'real'))
            if direction == 'long':
                repay_size = tp_order.get('base_size')
                session.update_usdt_m(repay=float(repay_size))
            elif direction == 'short':
                usdt_size = round(float(order_size) * price, 5)
                session.update_usdt_m(down=usdt_size)
        else:
            self.real_pos[asset].update(self.update_non_live_tp(session, asset, 'real'))

        self.counts_dict[f'real_tp_{direction}'] += 1

    def tp_real_full_M(self, session, pair, stp, direction):
        k10 = Timer(f'tp_real_full')
        k10.start()

        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        # check stop is within 20% of current price
        if direction == 'long':
            min_stp = price * 0.81
            stp = max(stp, min_stp)
        else:
            max_stp = price * 1.19
            stp = min(stp, max_stp)

        self.create_tp_placeholder(session, pair, stp, direction)
        pct = self.tp_set_pct(pair)
        # clear stop
        cleared_size = self.tp_clear_stop(session, pair)

        if not cleared_size:
            logger.warning(
                f'{self.id} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
            cleared_size = self.set_size_from_free(session, pair)

        # execute trade
        tp_order = self.tp_reduce_position(session, pair, cleared_size, pct, direction)

        note = f"{self.id} real take-profit {pair} {direction} {pct}% @ {price}"
        logger.info(note)

        if pct == 100:
            # repay assets
            tp_order = self.tp_repay_100(session, pair, tp_order, direction)
            # update records
            self.open_to_tracked(session, pair, stp, tp_order, direction)
            self.tp_update_records_100(session, pair, cleared_size, direction)

        else:  # if pct < 100%
            # repay assets
            tp_order = self.tp_repay_partial(session, pair, stp, tp_order, direction)
            # set new stop
            tp_order = self.tp_reset_stop(session, pair, stp, tp_order, direction)
            # update records
            self.open_to_open(session, pair, tp_order)
            self.tp_update_records_partial(session, pair, pct, stp, cleared_size, tp_order, direction)

        k10.stop()

    # real close -------------------------------------------------------------------------------------------------------

    def create_close_placeholder(self, session, pair, direction, action):
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        # temporary check to catch a possible bug, can delete after ive had a few reduce_risk calls with no bugs
        if direction not in ['long', 'short']:
            logger.warning(
                f'*** WARNING, string "{direction}" being passed to create_close_placeholder, either from close omf or reduce_risk')

        # insert placeholder record
        placeholder = {'action': action,
                       'direction': direction,
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'utc_datetime': now,
                       'completed': None
                       }
        self.open_trades[pair]['placeholder'] = placeholder

    def close_clear_stop(self, session, pair):
        clear, cleared_size = funcs.clear_stop_M(session, pair, self.open_trades[pair]['position'])
        cleared_size = self.check_size_against_records(pair, cleared_size)

        # update position and placeholder
        # self.open_trades[pair]['position']['hard_stop'] = None # don't want this to be changed to none in case it is
        # becoming a tracked trade, in which case i will still need a stop price in order to move non-real stop
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

        return cleared_size

    def close_position(self, session, pair, close_size, reason, direction, action):
        price = session.pairs_data[pair]['price']

        if direction == 'long':
            api_order = funcs.sell_asset_M(session, pair, close_size, session.live)
        else:
            api_order = funcs.buy_asset_M(session, pair, close_size, True, session.live)

        # update position and placeholder
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        if (float(new_base_size) * price) < 0.01:
            new_base_size = Decimal(0)
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        if new_base_size != 0:
            logger.debug(f"+++ {self.id} {pair} close {direction} resulted in base qty: {new_base_size}")
        close_order = funcs.create_trade_dict(api_order, price, session.live)

        close_order['pair'] = pair
        close_order['action'] = action
        close_order['direction'] = direction
        close_order['state'] = 'real'
        close_order['reason'] = reason
        close_order['utc_datetime'] = datetime.now(timezone.utc).strftime(timestring)
        self.open_trades[pair]['placeholder']['close_order'] = close_order
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return close_order

    def close_repay(self, session, pair, close_order, direction):
        asset = pair[:-4]
        liability = self.open_trades[pair]['position']['liability']

        if direction == 'long':
            repay_size = str(max(Decimal(close_order.get('quote_size', 0)), Decimal(liability)))
            funcs.repay_asset_M(session, 'USDT', repay_size, session.live)
        elif direction == 'short':
            repay_size = str(max(Decimal(liability), Decimal(close_order.get('base_size', 0))))
            funcs.repay_asset_M(session, asset, repay_size, session.live)

        # update records
        self.open_trades[pair]['position']['liability'] = '0'
        self.open_trades[pair]['placeholder']['completed'] = 'repay'
        self.open_trades[pair]['placeholder']['repay_size'] = repay_size

        return repay_size

    def open_to_closed(self, session, pair, close_order, repay_size):

        self.open_trades[pair]['trade'].append(close_order)
        self.open_trades[pair]['trade'][-1]['liability'] = str(Decimal(0) - Decimal(repay_size))
        self.open_trades[pair]['trade'][-1]['utc_datetime'] = self.open_trades[pair]['placeholder']['utc_datetime']

        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        if rpnl <= -1:
            logger.warning(f"*.*.* problem with real trade rpnl ({rpnl:.2f})")
            logger.warning(self.id)
            logger.warning(pformat(self.open_trades[pair]))

        trade_id = int(datetime.now().timestamp() * 1000)
        del self.open_trades[pair]['position']
        del self.open_trades[pair]['placeholder']
        self.closed_trades[trade_id] = self.open_trades[pair]
        self.record_trades(session, 'closed')

        del self.open_trades[pair]
        self.record_trades(session, 'open')
        asset = pair[:-len(session.quote_asset)]
        del self.real_pos[asset]

    def close_real_7(self, session, pair, close_size, direction):
        price = session.pairs_data[pair]['price']

        if direction == 'long' and session.live:
            session.update_usdt_m(repay=float(close_size))
        elif direction == 'short' and session.live:
            usdt_size = round(float(close_size) * price, 5)
            session.update_usdt_m(down=usdt_size)
        elif direction == 'long' and not session.live:
            self.real_pos['USDT']['value'] += float(close_size)
            self.real_pos['USDT']['owed'] -= float(close_size)
        elif direction == 'short' and not session.live:
            self.real_pos['USDT']['value'] += (float(close_size) * price)
            self.real_pos['USDT']['owed'] -= (float(close_size) * price)

        # save records and update counts
        self.counts_dict[f'real_close_{direction}'] += 1

    def close_real_full_M(self, session, pair, direction, action, size=0, stage=0):
        k9 = Timer(f'close_real_full')
        k9.start()

        price = session.pairs_data[pair]['price']
        logger.info(f"{self.id} real {action} {direction} {pair} @ {price}")

        placeholder = self.open_trades[pair].get('placeholder')

        if stage == 0:
            if self.open_trades[pair].get('placeholder'):
                del self.open_trades[pair]['placeholder']
            self.create_close_placeholder(session, pair, direction, action)
        if stage <= 1:
            # cancel stop
            cleared_size = self.close_clear_stop(session, pair)
            # if close_clear_stop returns 0, assume that the stop was already canceled and check for free balance.
            if not cleared_size:
                logger.warning(
                    f'{self.id} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
                cleared_size = self.set_size_from_free(session, pair)
        if stage <= 2:
            # execute trade
            if stage == 2:
                cleared_size = size

            if cleared_size:
                reason = 'close_signal' if action == 'close' else 'hit_hard_stop'
                close_order = self.close_position(session, pair, cleared_size, reason, direction, action)
            else:
                # in this case, the trade is closed and the record is ruined, so just delete the record and move on
                logger.warning(f"{self.id} {pair} {direction} position no longer exists, deleting trade record")
                del self.open_trades[pair]
                return
        if stage <= 3:
            # repay loan
            if stage == 3:
                close_order = placeholder['close_order']
            repay_size = self.close_repay(session, pair, close_order, direction)
        if stage <= 4:
            # update records
            if stage == 4:
                repay_size = placeholder['repay_size']
            self.open_to_closed(session, pair, close_order, repay_size)
            self.close_real_7(session, pair, repay_size, direction)

        k9.stop()

    # sim --------------------------------------------------------------------------------------------------------------

    def open_sim(self, session, signal):
        k8 = Timer(f'open_sim')
        k8.start()

        """in order for all the statistics to make sense, a simulated fixed_risk setting of 1/2 of session.fr_max is 
        used for every sim trade"""

        pair = signal['pair']
        asset = pair[:-4]
        direction = signal['direction']
        price = signal['trig_price']

        spot_usdt_size = session.fr_max * session.spot_bal * float(signal['score'])
        margin_usdt_size = session.fr_max * session.margin_bal * float(signal['score'])
        usdt_size = spot_usdt_size if self.mode == 'spot' else margin_usdt_size
        pfrd = usdt_size * signal['inval_dist']
        size = f"{usdt_size / price:.8f}"

        wanted = 'low_score' not in signal['sim_reasons']

        now = datetime.now(timezone.utc)

        sim_order = {
            'pair': pair,
            'direction': direction,
            'action': 'open',
            'trig_price': str(price),
            'exe_price': signal['trig_price'],
            'base_size': size,
            'quote_size': usdt_size,
            'hard_stop': str(signal['inval']),
            'stop_time': int(now.timestamp() * 1000),
            'timestamp': int(now.timestamp() * 1000),
            'utc_datetime': now.strftime(timestring),
            'state': 'sim',
            'fee': '0',
            'fee_currency': 'BNB',
            'wanted': wanted}

        pos_record = {
            'base_size': size,
            'init_base_size': size,
            'quote_size': usdt_size,
            'init_quote_size': usdt_size,
            'direction': direction,
            'entry_price': str(price),
            'hard_stop': str(signal['inval']),
            'init_hard_stop': str(signal['inval']),
            'open_time': int(now.timestamp() * 1000),
            'pair': pair,
            'liability': '0',
            'stop_id': 'not live',
            'stop_time': int(now.timestamp() * 1000),
            'state': 'sim',
            'pfrd': pfrd,
            'pct_of_full_pos': signal['pct_of_full_pos']}

        self.sim_trades[pair] = {'trade': [sim_order], 'position': pos_record, 'signal': signal}

        # self.sim_pos[asset].update(self.update_pos(session, pair, size, 'sim'))
        bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
        self.sim_pos[asset] = self.open_trade_stats(session, bal, self.sim_trades[pair])
        self.sim_pos[asset]['pnl_R'] = 0
        self.counts_dict[f'sim_open_{direction}'] += 1

        k8.stop()

    def tp_sim(self, session, pair, stp, direction):
        k7 = Timer(f'tp_sim')
        k7.start()

        price = session.pairs_data[pair]['price']
        asset = pair[:-4]
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])
        order_size = sim_bal / 2
        usdt_size = f"{order_size * price:.2f}"

        logger.info(f"{self.id} sim take-profit {pair} {direction} @ {price}")

        # execute order
        now = datetime.now(timezone.utc)
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': str(order_size),
                    'quote_size': usdt_size,
                    'hard_stop': str(stp),
                    'stop_time': int(now.timestamp() * 1000),
                    'reason': 'trade over-extended',
                    'timestamp': int(now.timestamp() * 1000),
                    'utc_datetime': now.strftime(timestring),
                    'action': 'tp',
                    'direction': direction,
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'sim'}
        self.sim_trades[pair]['trade'].append(tp_order)

        self.sim_trades[pair]['position']['base_size'] = str(order_size)
        self.sim_trades[pair]['position']['hard_stop'] = str(stp)
        self.sim_trades[pair]['position']['stop_time'] = int(now.timestamp() * 1000)
        self.sim_trades[pair]['position']['pct_of_full_pos'] /= 2

        rpnl = self.realised_pnl(self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"sim_{direction}"] += rpnl
        if 'low_score' in self.sim_trades[pair]['signal']['sim_reasons']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl

        # save records
        self.record_trades(session, 'sim')

        # update sim_pos
        self.sim_pos[asset].update(self.update_non_live_tp(session, asset, 'sim'))
        self.counts_dict[f'sim_tp_{direction}'] += 1

        k7.stop()

    def sim_to_closed_sim(self, session, pair, close_order, save_file):

        self.sim_trades[pair]['trade'].append(close_order)

        rpnl = self.realised_pnl(self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.sim_trades[pair]['position']['direction']
        self.realised_pnls[f"sim_{direction}"] += rpnl

        if rpnl < -1:
            logger.warning(f"*.*.* problem with sim trade rpnl ({rpnl:.2f})")
            logger.warning(self.id)
            logger.warning(pformat(self.sim_trades[pair]))

        if 'low_score' in self.sim_trades[pair]['signal']['sim_reasons']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl
        trade_id = int(datetime.now().timestamp() * 1000)
        del self.sim_trades[pair]['position']
        self.closed_sim_trades[trade_id] = self.sim_trades[pair]

        del self.sim_trades[pair]
        asset = pair[:-len(session.quote_asset)]
        del self.sim_pos[asset]

        if save_file:
            self.record_trades(session, 'closed_sim')
            self.record_trades(session, 'sim')

    def close_sim(self, session, pair, direction):
        k6 = Timer(f'close_sim')
        k6.start()

        price = session.pairs_data[pair]['price']
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])

        logger.info(f"{self.id} sim close {pair} {direction} @ {price}")

        # execute order
        now = datetime.now(timezone.utc)
        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': str(sim_bal),
                       'quote_size': f"{sim_bal * price:.2f}",
                       'reason': 'close_signal',
                       'timestamp': int(now.timestamp() * 1000),
                       'utc_datetime': now.strftime(timestring),
                       'action': 'close',
                       'direction': direction,
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'sim'}

        self.sim_to_closed_sim(session, pair, close_order, save_file=True)

        self.counts_dict[f'sim_close_{direction}'] += 1

        k6.stop()

    # tracked ----------------------------------------------------------------------------------------------------------

    def tp_tracked(self, session, pair, stp, direction):
        k5 = Timer(f'tp_tracked')
        k5.start()
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        logger.info(f"{self.id} tracked take-profit {pair} {direction} 50% @ {price}")

        trade_record = self.tracked_trades[pair]['trade']
        timestamp = round(datetime.now(timezone.utc).timestamp())

        # execute order
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': '0',
                    'quote_size': '0',
                    'reason': 'trade over-extended',
                    'timestamp': timestamp * 1000,
                    'action': 'tp',
                    'direction': direction,
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'tracked'}
        trade_record.append(tp_order)

        self.tracked_trades[pair]['position']['hard_stop'] = str(stp)
        self.tracked_trades[pair]['position']['stop_time'] = timestamp

        # update records
        self.tracked_trades[pair]['trade'] = trade_record
        self.record_trades(session, 'tracked')

        k5.stop()

    def tracked_to_closed(self, session, pair, close_order):
        asset = pair[:-4]

        self.tracked_trades[pair]['trade'].append(close_order)

        trade_id = int(datetime.now().timestamp() * 1000)
        del self.tracked_trades[pair]['position']
        self.closed_trades[trade_id] = self.tracked_trades[pair]
        self.record_trades(session, 'closed')
        del self.tracked_trades[pair]
        self.record_trades(session, 'tracked')
        del self.tracked[asset]

    def close_tracked(self, session, pair, direction):
        k4 = Timer(f'close_tracked')
        k4.start()

        price = session.pairs_data[pair]['price']
        logger.info(f"{self.id} tracked close {direction} {pair} @ {price}")

        now = datetime.now(timezone.utc)
        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': '0',
                       'quote_size': '0',
                       'reason': 'close_signal',
                       'timestamp': int(now.timestamp() * 1000),
                       'utc_datetime': now.strftime(timestring),
                       'action': 'close',
                       'direction': direction,
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'tracked'}

        self.tracked_to_closed(session, pair, close_order)

        k4.stop()

    # other ------------------------------------------------------------------------------------------------------------

    def print_rpnls(self):
        self.pnls = dict(
            spot=self.get_pnls('spot'),
            long=self.get_pnls('long'),
            short=self.get_pnls('short'),
        )
        logger.info(f"\n{self.id} scaled pnls")
        if self.mode == 'margin':
            logger.info("Long:")
            logger.info(f"EMA4: {self.pnls['long']['ema_4']:.2f}, EMA8: {self.pnls['long']['ema_8']:.2f}, "
                        f"EMA16: {self.pnls['long']['ema_16']:.2f}, EMA32: {self.pnls['long']['ema_32']:.2f}, "
                        f"EMA64: {self.pnls['long']['ema_64']:.2f}")
            logger.info("Short:")
            logger.info(f"EMA4: {self.pnls['short']['ema_4']:.2f}, EMA8: {self.pnls['short']['ema_8']:.2f}, "
                        f"EMA16: {self.pnls['short']['ema_16']:.2f}, EMA32: {self.pnls['short']['ema_32']:.2f}, "
                        f"EMA64: {self.pnls['short']['ema_64']:.2f}")
        else:
            logger.info("Spot:")
            logger.info(f"EMA4: {self.pnls['spot']['ema_4']:.2f}, EMA8: {self.pnls['spot']['ema_8']:.2f}, "
                        f"EMA16: {self.pnls['spot']['ema_16']:.2f}, EMA32: {self.pnls['spot']['ema_32']:.2f}, "
                        f"EMA64: {self.pnls['spot']['ema_64']:.2f}")

    def calc_stop(self, inval: float, spread: float, price: float, min_risk: float = 0.0015) -> float:
        """calculates what the stop-loss trigger price should be based on the current
        invalidation price and the current spread (slippage proxy).
        if this is too close to the entry price, the stop will be set at the minimum
        allowable distance."""
        buffer = max(spread * 2, min_risk)

        if price > inval:
            stop_price = float(inval) * (1 - buffer)
        else:
            stop_price = float(inval) * (1 + buffer)

        return stop_price

    def get_size(self, session, signal) -> tuple[float, float]:
        """calculates the desired position size in base or quote denominations
        using the total account balance, current fixed-risk setting, and the distance
        from current price to stop-loss"""

        jn = Timer('get_size')
        jn.start()

        balance = session.spot_bal if self.mode == 'spot' else session.margin_bal

        price = session.pairs_data[signal['pair']]['price']

        usdt_size = balance * session.fr_max * float(signal['score'])

        base_size = float(usdt_size / price)

        jn.stop()
        return base_size, usdt_size

    def aged_condition(self, signal_age, series_1, series_2):
        """returns True if series_1 has been above series_2 for {signal_age} consecutive periods"""
        tot = 0

        for x in range(-1, -(signal_age + 1), -1):
            tot += int(series_1.iloc[x] > series_2.iloc[x])

        # if tot adds up to the same number as signal age that shows that all loops returned True
        return tot == self.signal_age

    def check_size_against_records(self, pair, base_size):
        k3 = Timer(f'check_size_against_records')
        k3.start()

        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])
        base_size = Decimal(base_size)
        if base_size and (real_bal != base_size):  # check records match reality
            logger.warning(f"{self.id} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            mismatch = 100 * abs(base_size - real_bal) / base_size
            logger.warning(f"{mismatch = }%")
        elif base_size == 0:  # in this case the stop was probably cleared previously
            base_size = real_bal

        k3.stop()
        return str(base_size)

    @uf.retry_on_busy()
    def set_size_from_free(self, session, pair):
        """if clear_stop returns a base size of 0, this can be called to check for free balance,
        in case the position was there but just not in a stop order"""
        k2 = Timer(f'set_size_from_fee')
        k2.start()

        asset = pair[:-4]
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])

        if self.mode == 'spot':
            session.acct = session.client.get_account()
            session.get_asset_bals_s()
            free_bal = session.spot_bals[asset]['free']
        else:
            session.m_acct = session.client.get_margin_account()
            session.get_asset_bals_m()
            free_bal = session.margin_bals[asset]['free']

        k2.stop()

        return min(free_bal, real_bal)

    # repair trades ----------------------------------------------------------------------------------------------------

    def check_invalidation(self, session, ph):
        """returns true if trade is still valid, false otherwise.
        trade is still valid if direction is long and price is above invalidation, OR if dir is short and price is below"""
        pair = ph['pair']
        stp = ph['stop_price']

        dir_up = ph['direction'] == 'long'
        price = session.pairs_data[pair]['price']
        price_up = price > stp

        return dir_up == price_up

    def check_close_sig(self, session, ph):
        """returns true if trade is still above the previous close signal, false otherwise.
        trade is still valid if direction is long and price is above trig_price, OR if dir is short and price is below"""
        pair = ph['pair']
        trig = ph['trig_price']

        dir_up = ph.get('direction') or ph['direction'] == 'long'
        price = session.pairs_data[pair]['price']
        price_up = price > trig

        return dir_up == price_up

    def repair_open(self, session, ph, signal):
        pair = ph['pair']
        size = ph['base_size']
        stp = ph['stop_price']
        price = session.pairs_data[pair]['price']
        valid = self.check_invalidation(session, ph)
        direction = ph.get('direction')

        if ph['completed'] is None:
            del self.open_trades[pair]

        elif ph['completed'] == 'borrow':
            try:
                funcs.repay_asset_M(session, ph['loan_asset'], ph['liability'], session.live)
            except bx.BinanceAPIException as e:
                logger.exception('Problem during repair_open')
                logger.error(pformat(ph))
                logger.error(e.status_code)
                logger.error(e.message)
            finally:
                del self.open_trades[pair]

        elif ph['completed'] == 'execute':
            if valid:
                self.open_real_M(session, signal, 1)
            else:
                close_size = ph['api_order']['executedQty']
                if ph['direction'] == 'long':
                    funcs.sell_asset_M(session, pair, close_size, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, session.live)
                funcs.repay_asset_M(session, ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'trade_dict':
            if valid:
                self.open_real_M(session, signal, 2)
            else:
                close_size = ph['api_order']['executedQty']
                if direction == 'long':
                    funcs.sell_asset_M(session, pair, close_size, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, session.live)
                funcs.repay_asset_M(session, ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'set_stop':
            self.open_real_M(session, signal, 3)

    def repair_tp(self, session, ph):
        pair = ph['pair']
        cleared_size = ph.get('cleared_size')
        pct = ph.get('pct')
        stp = ph['stop_price']
        valid = self.check_invalidation(session, ph)
        direction = ph.get('direction')

        if ph['completed'] is None:
            del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if valid:
                # the function call below sets new stop and updates position dict
                self.open_set_stop(session, pair, stp, ph, direction)
            else:
                self.close_real_full_M(session, pair, direction, 'close')

        elif ph['completed'] == 'execute':
            if pct == 100:
                tp_order, usdt_size = self.tp_repay_100(session, pair, ph['tp_order'], direction)
                self.open_to_tracked(session, pair, stp, tp_order, direction)
                self.tp_update_records_100(session, pair, usdt_size, direction)
            else:
                if valid:
                    tp_order = self.tp_repay_partial(session, pair, stp, ph['tp_order'], direction)
                    tp_order = self.tp_reset_stop(session, pair, stp, tp_order, direction)
                    self.open_to_open(session, pair, tp_order)
                    self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order, direction)
                else:
                    remaining = self.open_trades[pair]['position']['base_size']
                    close_order = self.close_position(session, pair, remaining, 'close_signal', direction, 'close')
                    repay_size = self.close_repay(session, pair, close_order, direction)
                    self.open_to_closed(session, pair, close_order, direction)
                    self.close_real_7(session, pair, repay_size, direction)

        elif ph['completed'] == 'repay_100':
            self.open_to_tracked(session, pair, stp, ph['tp_order'], direction)
            self.tp_update_records_100(session, pair, ph['repay_usdt'], direction)

        elif ph['completed'] == 'repay_part':
            if valid:
                tp_order = self.tp_reset_stop(session, pair, stp, ph['tp_order'], direction)
                self.open_to_open(session, pair, tp_order)
                self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order, direction)
            else:
                remaining = self.open_trades[pair]['position']['base_size']
                close_order = self.close_position(session, pair, remaining, 'close_signal', direction, 'close')
                repay_size = self.close_repay(session, pair, close_order, direction)
                self.open_to_closed(session, pair, close_order, direction)
                self.close_real_7(session, pair, repay_size, direction)

        elif ph['completed'] == 'set_stop':
            self.open_to_open(session, pair, ph['tp_order'])
            self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, ph['tp_order'], direction)

    def repair_close(self, session, ph):
        pair = ph['pair']
        direction = ph.get('direction')
        action = ph['action']

        if ph['completed'] is None:
            self.close_real_full_M(session, pair, direction, action, stage=1)

        elif ph['completed'] == 'clear_stop':
            self.close_real_full_M(session, pair, direction, action, stage=2)

        elif ph['completed'] == 'execute':
            self.close_real_full_M(session, pair, direction, action, stage=3)

        elif ph['completed'] == 'repay':
            self.close_real_full_M(session, pair, direction, action, stage=4)

    def repair_move_stop(self, session, ph):
        # when it failed to move the stop up, price was above the current and new stop levels. since then, price could
        # have stayed above, or it could have moved below one or both stop levels. if price is above both, i simply reset
        # the stop as planned. if price is below the new stop, i should close the position. if price is below the original
        # stop, the position should also be closed but will already have been if the old stop was still in place

        pair = ph['pair']
        price = session.pairs_data[pair]['price']
        pos_record = self.open_trades[pair]['position']
        direction = ph.get('direction')
        action = ph.get('action')

        if ph['completed'] is None:
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, direction, ph['stop_price'], pos_record, stage=1)
            elif price > self.open_trades[pair]['position']['hard_stop']:
                self.close_real_full_M(session, pair, direction, action)
            else:
                del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, direction, ph['stop_price'], pos_record, stage=2)
            else:
                self.close_real_full_M(session, pair, direction, action)

    def repair_trade_records(self, session):
        k1 = Timer(f'repair trade records')
        k1.start()
        ph_list = []
        for pair in self.open_trades.values():
            if pair.get('placeholder'):
                placeholder = pair['placeholder']
                signal = pair['signal']
                ph_list.append((placeholder, signal))

        for ph, sig in ph_list:
            try:
                if ph['action'] == 'open':
                    self.repair_open(session, ph, sig)
                elif ph['action'] == 'tp':
                    self.repair_tp(session, ph)
                elif ph['action'] == 'close':
                    self.repair_close(session, ph)
                elif ph['action'] == 'move_stop':
                    self.repair_move_stop(session, ph)
            except bx.BinanceAPIException as e:
                logger.exception("problem during repair_trade_records")
                logger.error(pformat(ph))
                self.record_trades(session, 'all')
                logger.error(e.status_code)
                logger.error(e.message)
        k1.stop()


class TrailFractals(Agent):
    """Machine learning strategy based around williams fractals trailing stops"""

    def __init__(self, session, tf: str, offset: int, width: int, spacing: int, training_pair_selection: str,
                 num_pairs: int) -> None:
        t = Timer('TrailFractals init')
        t.start()
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.width = width
        self.spacing = spacing
        self.training_pair_selection = training_pair_selection
        self.training_pairs_n = num_pairs
        self.load_primary_model_data(session, tf)
        self.load_secondary_model_data()
        session.pairs_set.update(self.pairs)
        self.name = (f'trail_fractals_{self.width}_{self.spacing}')
        self.id = (f"trail_fractals_{self.tf}_{self.offset}_{self.width}_{self.spacing}_"
                   f"{self.training_pair_selection}_{self.training_pairs_n}")
        self.ohlc_length = 4035
        self.trail_stop = True
        self.notes = ''
        Agent.__init__(self, session)
        session.features[tf].update(self.features)
        t.stop()

    def load_primary_model_data(self, session, tf):
        # paths
        primary_folder = Path(f"/home/ross/coding/modular_trader/machine_learning/models/trail_fractals_{self.width}_"
                              f"{self.spacing}/{self.training_pair_selection}_{self.training_pairs_n}")
        self.long_model_path = primary_folder / f"long_{self.tf}_model_1a.sav"
        self.short_model_path = primary_folder / f"short_{self.tf}_model_1a.sav"
        long_scaler_path = primary_folder / f"long_{self.tf}_scaler_1a.sav"
        short_scaler_path = primary_folder / f"short_{self.tf}_scaler_1a.sav"
        long_info_path = primary_folder / f"long_{self.tf}_info_1a.json"
        short_info_path = primary_folder / f"short_{self.tf}_info_1a.json"

        self.long_model = joblib.load(self.long_model_path)
        self.short_model = joblib.load(self.short_model_path)
        self.long_scaler = joblib.load(long_scaler_path)
        self.short_scaler = joblib.load(short_scaler_path)

        with open(long_info_path, 'r') as ip:
            self.long_info = json.load(ip)
        with open(short_info_path, 'r') as ip:
            self.short_info = json.load(ip)

        self.pairs = self.long_info['pairs']
        self.features = set(self.long_info['features'] + self.short_info['features'])
        session.features[tf].update(self.features)

    def load_secondary_model_data(self):
        # paths
        secondary_folder = Path(f"/home/ross/coding/modular_trader/machine_learning/models/"
                                f"trail_fractals_{self.width}_{self.spacing}")
        long_model_file = secondary_folder / f"long_{self.tf}_model_2.json"
        short_model_file = secondary_folder / f"short_{self.tf}_model_2.json"
        long_scaler_file = secondary_folder / f"long_{self.tf}_scaler_2.sav"
        short_scaler_file = secondary_folder / f"short_{self.tf}_scaler_2.sav"
        long_model_info = secondary_folder / f"long_{self.tf}_info_2.json"
        short_model_info = secondary_folder / f"short_{self.tf}_info_2.json"

        self.long_model_2 = XGBClassifier()
        self.long_model_2.load_model(long_model_file)
        self.short_model_2 = XGBClassifier()
        self.short_model_2.load_model(short_model_file)

        self.long_scaler_2 = joblib.load(long_scaler_file)
        self.short_scaler_2 = joblib.load(short_scaler_file)

        with open(long_model_info, 'r') as ip:
            self.long_info_2 = json.load(ip)
        with open(short_model_info, 'r') as ip:
            self.short_info_2 = json.load(ip)

        self.features_2 = set(self.long_info_2['features'] + self.short_info_2['features'])

    def secondary_prediction(self, signal):
        direction = signal['direction']

        conf_l = signal['confidence_l']
        conf_s = signal['confidence_s']

        inval_ratio = signal['inval_ratio']

        perf_ema_4 = self.pnls[direction]['ema_4']
        perf_ema_8 = self.pnls[direction]['ema_8']
        perf_ema_16 = self.pnls[direction]['ema_16']
        perf_ema_32 = self.pnls[direction]['ema_32']
        perf_ema_64 = self.pnls[direction]['ema_64']

        mkt_rank_1d = signal.get('market_rank_1d', 1)
        mkt_rank_1w = signal.get('market_rank_1w', 1)
        mkt_rank_1m = signal.get('market_rank_1m', 1)

        features = [conf_l, conf_s, inval_ratio, mkt_rank_1d, mkt_rank_1w, mkt_rank_1m,
                    perf_ema_4, perf_ema_8, perf_ema_16, perf_ema_32, perf_ema_64]
        names = ['conf_l', 'conf_s', 'inval_ratio', 'mkt_rank_1d', 'mkt_rank_1w', 'mkt_rank_1m',
                 'perf_ema_4', 'perf_ema_8', 'perf_ema_16', 'perf_ema_32', 'perf_ema_64']
        data = np.array(features).reshape(1, -1)
        self.long_features_2_idx = [i for i, f in enumerate(names) if f in self.long_info_2['features']]
        self.short_features_2_idx = [i for i, f in enumerate(names) if f in self.short_info_2['features']]

        if direction == 'long':
            long_data = self.long_scaler_2.transform(data)
            long_data = long_data[:, self.long_features_2_idx]
            score = float(self.long_model_2.predict(long_data))
            score = min(1, max(0.01, score))
            validity = self.long_info_2['validity']
            print(f"secondary long prediction: {score:.1%}, validity: {validity}")
        else:
            short_data = self.short_scaler_2.transform(data)
            short_data = short_data[:, self.short_features_2_idx]
            score = float(self.short_model_2.predict_proba(short_data)[-1, 0])
            score = min(1, max(0.01, score))
            validity = self.short_info_2['validity']
            print(f"secondary short prediction: {score:.1%}, validity: {validity}")

        return score, validity

    def secondary_manual_prediction(self, session, signal):
        signal['perf_ema4'] = self.pnls[signal['direction']]['ema_4']
        signal['perf_ema8'] = self.pnls[signal['direction']]['ema_8']
        signal['perf_ema16'] = self.pnls[signal['direction']]['ema_16']
        signal['perf_ema32'] = self.pnls[signal['direction']]['ema_32']
        signal['perf_ema64'] = self.pnls[signal['direction']]['ema_64']

        sig_bias = signal['bias']

        perf_score, rank_score = 0, 0
        if signal['tf'] == '1h':
            perf_score = ((signal['perf_ema64'] > 0.5) + (signal['perf_ema32'] > 0.5) + (
                    signal['perf_ema16'] > 0.5)) / 3
            rank_score = signal.get('market_rank_1d', 1) if sig_bias == 'bullish' else (
                    1 - signal.get('market_rank_1d', 1))
        elif signal['tf'] in {'4h', '12h'}:
            perf_score = ((signal['perf_ema32'] > 0.5) + (signal['perf_ema16'] > 0.5) + (signal['perf_ema8'] > 0.5)) / 3
            rank_score = signal.get('market_rank_1w', 1) if sig_bias == 'bullish' else (
                    1 - signal.get('market_rank_1w', 1))
        elif signal['tf'] == '1d':
            perf_score = ((signal['perf_ema16'] > 0.5) + (signal['perf_ema8'] > 0.5) + (signal['perf_ema4'] > 0.5)) / 3
            rank_score = signal.get('market_rank_1m', 1) if sig_bias == 'bullish' else (
                    1 - signal.get('market_rank_1m', 1))

        if not session.live:
            perf_score = 1.0

        inval_scalar = 1 + abs(1 - signal['inval_ratio'])
        sig_score = signal['confidence'] * rank_score
        risk_scalar = (sig_score * perf_score) / inval_scalar
        score = round(risk_scalar, 5)

        print(f"secondary manual prediction: {score:.1%}")

        return score

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates spot buy signals based on the ats_z indicator. does not account for currently open positions,
        just generates signals as the strategy dictates"""

        sig = Timer('ats_spot_signals')
        sig.start()

        # check if it's a valid margin pair first
        if not session.pairs_data[pair]['margin_allowed']:
            logger.info(f"{pair} not a margin pair, but was trying to get margin signals")
            return None

        signal_dict = {'agent': self.id, 'mode': self.mode, 'pair': pair}

        df = ind.williams_fractals(df, self.width, self.spacing)

        # calculate % from invalidation
        df['long_r_pct'] = abs(df.close - df.frac_low) / df.close
        df['short_r_pct'] = abs(df.close - df.frac_high) / df.close

        price = session.pairs_data[pair]['price']
        long_stop = self.calc_stop(df.frac_low.iloc[-1], session.pairs_data[pair]['spread'], price)
        short_stop = self.calc_stop(df.frac_high.iloc[-1], session.pairs_data[pair]['spread'], price)

        if pair not in self.pairs:  # might need inval ratio for pairs that used to be in the list
            return {'long_ratio': long_stop / price,
                    'short_ratio': short_stop / price}

        # Long model
        df['r_pct'] = df.long_r_pct
        long_features = df[self.long_info['features']]
        # long_features, _, cols = mlf.transform_columns(long_features, long_features)
        # long_features = pd.DataFrame(long_features, columns=cols)
        long_features_scaled = self.long_scaler.transform(long_features)
        # long_features = long_features[-1, :]

        long_X = pd.DataFrame(long_features_scaled, columns=self.long_info['features']).iloc[-2:, :]
        # long_X = np.array(pd.Series(long_X.iloc[-1])).reshape(1, -1)
        try:
            long_confidence = self.long_model.predict_proba(long_X, )[-1, 1]
        except ValueError as e:
            # logger.exception('NaN in prediction set')
            print(f'\n{pair}\n')
            logger.error(e)
            print(long_features.tail())
            print(long_features_scaled[-3:, :])
            long_confidence = 0

        # Short model
        df = df.drop('long_r_pct', axis=1)
        df['r_pct'] = df.short_r_pct
        short_features = df[self.short_info['features']]
        # short_features, _, cols = mlf.transform_columns(short_features, short_features)
        # short_features = pd.DataFrame(short_features, columns=cols)
        short_features_scaled = self.short_scaler.transform(short_features)
        # short_features = short_features[-1, :]

        short_X = pd.DataFrame(short_features_scaled, columns=self.short_info['features']).iloc[-2:, :]
        # short_X = np.array(pd.Series(short_X.iloc[-1])).reshape(1, -1)
        try:
            short_confidence = self.short_model.predict_proba(short_X)[-1, 1]
        except ValueError as e:
            # logger.exception('NaN in prediction set')
            print(f'\n{pair}\n')
            logger.error(e)
            print(short_features.tail())
            print(short_features_scaled[-3:, :])
            short_confidence = 0

        # logger.debug(f"{self.id} {pair} {self.tf} long conf: {long_confidence:.1%} short conf: {short_confidence:.1%}")

        # these are deliberately back-to-front because my analysis showed they were actually inversely correlated to pnl
        combined_long = short_confidence - long_confidence
        combined_short = long_confidence - short_confidence
        # combined confidence will not be needed at all when the secondary ml model is doing signal scores

        if (price > df.frac_low.iloc[-1]) and (combined_long > 0):
            signal_dict['confidence'] = combined_long
            signal_dict['bias'] = 'bullish'
            inval = df.frac_low.iloc[-1]
            model_created = self.long_info.get('created')
        elif (price < df.frac_high.iloc[-1]) and (combined_short > 0):
            signal_dict['confidence'] = combined_short
            signal_dict['bias'] = 'bearish'
            inval = df.frac_high.iloc[-1]
            model_created = self.short_info.get('created')
        else:  # if there is no signal, long or short ratio might still be needed for moving stops
            return {'long_ratio': long_stop / price,
                    'short_ratio': short_stop / price}

        created_dt = datetime.fromtimestamp(model_created).astimezone(timezone.utc)
        model_age = datetime.now(timezone.utc) - created_dt

        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        signal_dict['inval'] = stp
        signal_dict['inval_ratio'] = stp / price
        signal_dict['inval_dist'] = abs(stp - price) / price
        # inval score will not be needed after the secondary ml model takes over scoring
        signal_dict['inval_score'] = self.calc_inval_risk_score(abs((price - stp) / price))
        signal_dict['trig_price'] = price
        signal_dict['pct_of_full_pos'] = 1
        signal_dict['tf'] = self.tf
        signal_dict['asset'] = pair[:-len(session.quote_asset)]
        signal_dict['model_age'] = model_age.seconds
        signal_dict['confidence_l'] = long_confidence
        signal_dict['confidence_s'] = short_confidence
        signal_dict['market_rank_1d'] = session.pairs_data[pair]['market_rank_1d']
        signal_dict['market_rank_1w'] = session.pairs_data[pair]['market_rank_1w']
        signal_dict['market_rank_1m'] = session.pairs_data[pair]['market_rank_1m']
        signal_dict['width'] = self.width
        signal_dict['spacing'] = self.spacing
        signal_dict['training_pair_selection'] = self.training_pair_selection
        signal_dict['training_pairs_n'] = len(self.pairs)

        sig.stop()

        return signal_dict

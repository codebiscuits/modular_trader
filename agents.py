import binance.exceptions as bx
import pandas as pd
import polars as pl
from pathlib import Path
import json
from json.decoder import JSONDecodeError
import statistics as stats
from timers import Timer
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import binance_funcs as funcs
import utility_funcs as uf
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from decimal import Decimal, getcontext
from pprint import pprint
import sys
import math

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12


# TODO i need to check that every agent is working out inval and inval ratio in the same way, because i have a feeling
#  that one or more of them may be inverting something by accident on the short trades

class Agent():
    '''generic agent class for each strategy to inherit from'''

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
        self.counts_dict = {'real_stop_spot': 0, 'real_open_spot': 0, 'real_add_spot': 0, 'real_tp_spot': 0,
                            'real_close_spot': 0,
                            'sim_stop_spot': 0, 'sim_open_spot': 0, 'sim_add_spot': 0, 'sim_tp_spot': 0,
                            'sim_close_spot': 0,
                            'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0,
                            'real_close_long': 0,
                            'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0,
                            'sim_close_long': 0,
                            'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0,
                            'real_close_short': 0,
                            'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0,
                            'sim_close_short': 0,
                            'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 'too_much_or': 0, 'algo_order_limit': 0,
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
        self.backup_trade_records(session)
        self.repair_trade_records(session)
        self.real_pos = self.current_positions(session, 'open')
        self.sim_pos = self.current_positions(session, 'sim')
        self.tracked = self.current_positions(session, 'tracked')
        self.record_stopped_trades(session)
        self.calc_init_opnl(session)
        self.open_pnl_changes = {}
        self.fixed_risk_spot = self.set_fixed_risk(session, 'spot')
        self.fixed_risk_l = self.set_fixed_risk(session, 'long')
        self.fixed_risk_s = self.set_fixed_risk(session, 'short')
        self.test_fixed_risk(0.0002, 0.0002)
        self.max_positions = self.set_max_pos()
        self.total_r_limit = self.max_positions * 1.75
        self.indiv_r_limit = 2
        self.fr_dol_spot = self.fixed_risk_spot * session.spot_bal
        self.fixed_risk_dol_l = self.fixed_risk_l * session.margin_bal
        self.fixed_risk_dol_s = self.fixed_risk_s * session.margin_bal
        self.calc_tor()
        self.next_id = int(datetime.now().timestamp())
        session.min_length = min(session.min_length, self.ohlc_length)
        t.stop()

    def __str__(self):
        return self.id

    def load_perf_log(self, session):
        folder = Path(f"{session.read_records}/{self.id}")
        if not folder.exists():
            folder.mkdir(parents=True)
        bal_path = Path(folder / 'perf_log.json')
        bal_path.touch(exist_ok=True)
        try:
            with open(bal_path, "r") as file:
                self.perf_log = json.load(file)
        except JSONDecodeError:
            print(f"{bal_path} was an empty file.")
            self.perf_log = None

    def sync_test_records(self, session) -> None:
        '''takes the trade records from the raspberry pi and saves them over 
        the local trade records. only runs when not live'''

        q = Timer('sync_test_records')
        q.start()
        real_folder = Path(f"{session.read_records}/{self.id}")
        test_folder = Path(f'{session.write_records}/{self.id}')
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
                print(f"{bal_path} was an empty file.")
                self.perf_log = None

        def sync_trades_records(switch):
            w = Timer(f'sync_trades_records-{switch}')
            w.start()
            trades_path = Path(f'{session.read_records}/{self.id}/{switch}_trades.json')
            test_trades = Path(f'{session.write_records}/{self.id}/{switch}_trades.json')
            test_trades.touch(exist_ok=True)

            if trades_path.exists():
                try:
                    with open(trades_path, 'r') as file:
                        data = json.load(file)
                    if data:
                        with open(test_trades, 'w') as file:
                            json.dump(data, file)
                except JSONDecodeError:
                    # print(f'{switch}_trades file empty')
                    pass
            w.stop()

        sync_trades_records('open')
        sync_trades_records('sim')
        sync_trades_records('tracked')
        sync_trades_records('closed')
        sync_trades_records('closed_sim')

        q.stop()

    def read_open_trade_records(self, session, state: str) -> dict:
        '''loads records from open_trades/sim_trades/tracked_trades and returns
        them in a dictionary'''

        w = Timer(f'read_open_trade_records-{state}')
        w.start()
        ot_path = Path(f"{session.read_records}/{self.id}")
        ot_path.mkdir(parents=True, exist_ok=True)
        ot_path = ot_path / f'{state}_trades.json'

        if ot_path.exists():
            with open(ot_path, "r") as ot_file:
                try:
                    open_trades = json.load(ot_file)
                except JSONDecodeError as e:
                    open_trades = {}
        else:
            # print("ot_path doesn't exist")
            open_trades = {}
            ot_path.touch()
            # print(f'{ot_path} not found')

        # print(f"in read_open_trade_records - {self.name} {state} trades: {open_trades.keys()}")

        w.stop()
        return open_trades

    def read_closed_trade_records(self, session) -> dict:
        '''loads trade records from closed_trades and returns them as a dictionary'''

        e = Timer('read_closed_trade_records')
        e.start()
        ct_path = Path(f"{session.read_records}/{self.id}/closed_trades.json")
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
            # print(f'{ct_path} not found')
        e.stop()
        return closed_trades

    def read_closed_sim_trade_records(self, session) -> dict:
        '''loads closed_sim_trades and returns them as a dictionary'''

        r = Timer('read_closed_sim_trade_records')
        r.start()
        cs_path = Path(f"{session.read_records}/{self.id}/closed_sim_trades.json")
        if Path(cs_path).exists():
            with open(cs_path, "r") as cs_file:
                try:
                    cs_trades = json.load(cs_file)
                except JSONDecodeError:
                    cs_trades = {}

        else:
            cs_trades = {}
            print(f'{cs_path} not found')

        limit = 2000
        if len(cs_trades.keys()) > limit:
            # print(f"{self.name} closed sim trades on record: {len(cs_trades.keys())}")
            closed_sim_tups = sorted(zip(cs_trades.keys(), cs_trades.values()), key=lambda x: int(x[0]))
            closed_sim_trades = dict(closed_sim_tups[-limit:])
        else:
            closed_sim_trades = cs_trades

        r.stop()
        return closed_sim_trades

    def backup_trade_records(self, session) -> None:
        '''updates the backup file for each trades dictionary, on the condition 
        that they are not empty'''

        y = Timer('backup_trade_records')
        y.start()
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        if self.open_trades:
            with open(f"{session.write_records}/{self.id}/ot_backup.json", "w") as ot_file:
                json.dump(self.open_trades, ot_file)

        if self.sim_trades:
            with open(f"{session.write_records}/{self.id}/st_backup.json", "w") as st_file:
                json.dump(self.sim_trades, st_file)

        if self.tracked_trades:
            with open(f"{session.write_records}/{self.id}/tr_backup.json", "w") as tr_file:
                json.dump(self.tracked_trades, tr_file)

        if self.closed_trades:
            with open(f"{session.write_records}/{self.id}/ct_backup.json", "w") as ct_file:
                json.dump(self.closed_trades, ct_file)

        if self.closed_sim_trades:
            with open(f"{session.write_records}/{self.id}/cs_backup.json", "w") as cs_file:
                json.dump(self.closed_sim_trades, cs_file)

        y.stop()

    # record stopped trades ------------------------------------------------

    def find_order_old(self, session, pair, sid):
        if sid == 'not live':
            return None
        print('get_all_margin_orders')
        session.track_weights(200)
        abc = Timer('all binance calls')
        abc.start()
        order_list = client.get_all_margin_orders(symbol=pair, orderId=sid)
        abc.stop()
        session.counts.append('get_all_margin_orders')

        order = order_list[0] if order_list[0]['orderId'] == sid else None

        if not order:
            print(f'No orders on binance for {pair}')

        # insert placeholder record
        placeholder = {'type': f"stop_{self.open_trades[pair]['position']['direction']}",
                       'state': 'real',
                       'pair': pair,
                       'order': order,
                       'completed': 'order'
                       }
        self.open_trades[pair]['placeholder'] = placeholder

        return order

    def find_order(self, session, pair, sid):
        if sid == 'not live':
            return None
        print('get_margin_order')
        session.track_weights(10)
        abc = Timer('all binance calls')
        abc.start()
        order = client.get_margin_order(symbol=pair, orderId=sid)
        abc.stop()
        session.counts.append('get_margin_order')

        if not order:
            print(f'No orders on binance for {pair}')

        # insert placeholder record
        placeholder = {'type': f"stop_{self.open_trades[pair]['position']['direction']}",
                       'state': 'real',
                       'pair': pair,
                       'order': order,
                       'completed': 'order'
                       }
        self.open_trades[pair]['placeholder'] = placeholder

        return order

    def repay_stop(self, session, pair, order):
        if (order.get('side') == 'BUY'):
            trade_type = 'stop_short'
            asset = pair[:-4]
            stop_size = Decimal(order.get('executedQty'))
            repayed = funcs.repay_asset_M(asset, stop_size, session.live)
        else:
            trade_type = 'stop_long'
            stop_size = Decimal(order.get('cummulativeQuoteQty'))
            repayed = funcs.repay_asset_M('USDT', stop_size, session.live)

        self.open_trades[pair]['placeholder']['completed'] = 'repay'

        return stop_size

    def create_stop_dict(self, session, pair, order, stop_size):
        stop_dict = funcs.create_stop_dict(session, order)
        stop_dict['type'] = f"stop_{self.open_trades[pair]['position']['direction']}"
        stop_dict['state'] = 'real'
        stop_dict['reason'] = 'hit hard stop'
        stop_dict['liability'] = uf.update_liability(self.open_trades[pair], stop_size, 'reduce')
        if stop_dict['liability'] not in ['0', '0.0']:
            print(
                f"+++ WARNING {self.name} {pair} stop hit, liability record doesn't add up. Recorded value: {stop_dict['liability']} +++")

        return stop_dict

    def save_records(self, session, pair, stop_dict):
        self.open_trades[pair]['trade'].append(stop_dict)
        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        ts_id = int(self.open_trades[pair]['position']['open_time'])
        self.closed_trades[ts_id] = self.open_trades[pair]['trade']
        self.record_trades(session, 'closed')
        del self.open_trades[pair]
        self.record_trades(session, 'open')

    def error_print(self, session, pair, stage, e):
        self.record_trades(session, 'all')
        print(f'{self.name} problem with record_stopped_trades during {pair} {stage}')
        print(f"code: {e.code}")
        print(f"message: {e.message}")

    def rst_iteration_m(self, session, pair, sid):
        direction = self.open_trades[pair]['position']['direction']
        try:
            order = self.find_order(session, pair, sid)
        except bx.BinanceAPIException as e:
            self.error_print(session, pair, 'find_order', e)

        if order:
            # if an order can be found on binance, complete the records
            stop_size = self.repay_stop(session, pair, order)
            stop_dict = self.create_stop_dict(session, pair, order, stop_size)
            self.save_records(session, pair, stop_dict)

            self.counts_dict[f'real_stop_{direction}'] += 1

        else:  # if no order can be found, try to close the position
            if session.live:
                print(f"record_stopped_trades found no stop for {pair} {self.name}")
            if direction == 'long':
                free_bal = session.margin_bals[pair[:-4]].get('free')
                pos_size = self.open_trades[pair]['position']['base_size']

                if Decimal(free_bal) >= Decimal(pos_size):
                    # if the free balance covers the position, close it
                    print(f'{self.name} record_stopped_trades will close {pair} now')
                    try:
                        self.close_real_full(session, pair, direction)
                    except bx.BinanceAPIException as e:
                        self.error_print(session, pair, 'close', e)
                else:
                    # if the free balance doesn't cover the position, notify me
                    price = session.pairs_data[pair]['price']
                    value = free_bal * price
                    if (pair == 'BNBUSDT' and value > 15) or (pair != 'BNBUSDT' and value > 10):
                        note = f'{pair} in position with no stop-loss'
                        pb.push_note(session.now_start, note)
            else:  # if direction == 'short'
                owed = float(session.margin_bals[pair[:-4]].get('borrowed'))
                if session.live and not owed:
                    print(f'{pair[:-4]} loan already repaid, no action needed')
                    del self.open_trades[pair]
                elif not owed:
                    return
                else:  # if live and owed
                    print(f'{self.name} record_stopped_trades will close {pair} now')
                    try:
                        size = self.open_trades[pair]['position']['base_size']
                        self.close_real_full(session, pair, direction, size=size, stage=2)
                    except bx.BinanceAPIException as e:
                        self.error_print(session, pair, 'close', e)

    def record_stopped_trades(self, session) -> None:
        m = Timer('record_stopped_trades')
        m.start()

        # get a list of (pair, stop_id, stop_time) for all open_trades records
        old_ids = [(pair, v['position']['stop_id'], v['position']['stop_time'])
                   for pair, v in self.open_trades.items()]

        for pair, sid, time in old_ids:
            if sid == 'not live':
                print(f"{pair} record non-live")
                continue

            print('get_margin_order')
            session.track_weights(10)
            abc = Timer('all binance calls')
            abc.start()
            order = client.get_margin_order(symbol=pair, orderId=sid)
            abc.stop()
            session.counts.append('get_margin_order')

            if order['status'] == 'FILLED':
                print(f"{self.name} {pair} stop order filled")
                try:
                    self.rst_iteration_m(session, pair, sid)
                except bx.BinanceAPIException as e:
                    self.record_trades(session, 'all')
                    print(f'{self.name} problem with record_stopped_trades during {pair}')
                    print(e)

            elif order['status'] == 'CANCELED':
                print(f'\nProblem with {self.name} {pair} trade record\n')

                # it should be possible to check the placeholder in the trade record and piece together what to do
                ph = self.open_trades[pair].get('placeholder')

                if ph:
                    print(f"{self.name} {pair} stop canceled, placeholder found")
                    print('use this to modify record_stopped_trades with useful behaviour in this scenario')
                    pprint(self.open_trades[pair])
                    print('')
                    pprint(order)
                else:
                    print('\n\n*******************************************\n\n')
                    print(f"{self.name} {pair} stop canceled, no placeholder found, deleting record")
                    pprint(self.open_trades[pair])
                    del self.open_trades[pair]
                    self.record_trades(session, 'open')
                    print('\n\n*******************************************\n\n')

            elif order['status'] == 'PARTIALLY_FILLED':
                print(f"{self.name} {pair} stop hit and partially filled, recording trade.")

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
                     'type': 'partstop_long'}
                )

                # update position record
                self.open_trades[pair]['position']['base_size'] = str(old_size - exe_size)

            else:
                # print(f"{self.name} {pair} stop order (id {sid}) not filled, status: {order['status']}")
                pass

        m.stop()

    # record stopped sim trades ----------------------------------------------

    def get_data(self, session, pair, timeframes: list, stop_time):

        rsst_gd = Timer('rsst - get_data')
        rsst_gd.start()

        # print(f"rsst {self.name} {pair}")

        filepath = Path(f'{session.ohlc_data}/{pair}.parquet')
        check_recent = False

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
                except OSError as e:
                    print(f"problem loading {pair} ohlc")
                    print(e)
                    filepath.unlink()
                    df = funcs.get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
                    source = 'exchange'
                    print(f'downloaded {pair} from scratch')
            else:
                print(f"{filepath} doesn't exist")
                df = funcs.get_ohlc(pair, session.ohlc_tf, '2 years ago UTC', session)
                source = 'exchange'
                print(f'downloaded {pair} from scratch')
                # pldf = pl.from_pandas(df)
                # pldf.write_parquet(filepath, use_pyarrow=True)

            session.store_ohlc(df, pair, timeframes)

        if check_recent:
            last = df.timestamp.iloc[-1]
            timespan = datetime.now().timestamp() - (last.timestamp())
            if timespan > 900:
                df = funcs.update_ohlc(pair, session.ohlc_tf, df, session)
                source += ' and exchange'
                # pldf = pl.from_pandas(df)
                # pldf.write_parquet(filepath, use_pyarrow=True)
                session.store_ohlc(df, pair, timeframes)

        stop_dt = datetime.fromtimestamp(stop_time)  # / 1000)
        df = df.loc[df.timestamp > stop_dt].reset_index(drop=True)
        # print(f'::: rsst {self.name} get_data {pair} from {source} :::')

        rsst_gd.stop()

        return df

    def check_stop_hit(self, df, direction, stop):
        func_name = sys._getframe().f_code.co_name
        k16 = Timer(f'{func_name}')
        k16.start()

        stop_hit_time = None
        trade_type = f"stop_{direction}"
        if direction == 'long':
            ll = df.low.min()
            stopped = ll < stop
            overshoot_pct = round((100 * (stop - ll) / stop), 3)  # % distance that price broke through the stop
            if stopped:

                stop_hit_time = df.loc[df.low <= stop].timestamp.iloc[0]
                if isinstance(stop_hit_time, pd.Timestamp):
                    stop_hit_time = stop_hit_time.timestamp()

        else:
            hh = df.high.max()
            stopped = hh > stop
            overshoot_pct = round((100 * (hh - stop) / stop), 3)  # % distance that price broke through the stop
            if stopped:

                stop_hit_time = df.loc[df.high >= stop].timestamp.iloc[0]
                if isinstance(stop_hit_time, pd.Timestamp):
                    stop_hit_time = stop_hit_time.timestamp()

        if stop_hit_time:
            stop_hit_time = int(stop_hit_time)

        k16.stop()

        return stopped, trade_type, overshoot_pct, stop_hit_time

    def create_trade_dict(self, pair, trade_type, stop, base_size, stop_hit_time, overshoot_pct):
        trade_dict = {'timestamp': stop_hit_time,
                      'pair': pair,
                      'type': trade_type,
                      'exe_price': str(stop),
                      'base_size': str(base_size),
                      'quote_size': str(round(base_size * stop, 2)),
                      'fee': 0,
                      'fee_currency': 'BNB',
                      'reason': 'hit hard stop',
                      'state': 'sim',
                      'overshoot': overshoot_pct
                      }

        return trade_dict

    def record_stopped_sim_trades(self, session, timeframes: list) -> None:
        """goes through all trades in the sim_trades file and checks their recent price action
        against their most recent hard_stop to see if any of them would have got stopped out"""

        n = Timer('record_stopped_sim_trades')
        n.start()
        session.counts.append('rsst')

        check_pairs = list(self.sim_trades.items())
        print(f"{self.name} rsst, checking {len(check_pairs)} pairs")
        for pair, v in check_pairs:  # can't loop through the dictionary directly because i delete items as i go
            direction = v['position']['direction']
            base_size = float(v['position']['base_size'])
            stop = float(v['position']['hard_stop'])
            stop_time = v['position']['stop_time']

            df = self.get_data(session, pair, timeframes, stop_time)
            stopped, trade_type, overshoot_pct, stop_hit_time = self.check_stop_hit(df, direction, stop)
            if stopped:
                # print(f"{v['position']['open_time']}, {stop_hit_time}, {v['position']['entry_price']}")
                trade_dict = self.create_trade_dict(pair, trade_type, stop, base_size, stop_hit_time, overshoot_pct)
                self.sim_to_closed_sim(session, pair, trade_dict, save_file=False)
                self.counts_dict[f'sim_stop_{direction}'] += 1
            # else:
            #     print(f"{pair} still open")

        self.record_trades(session, 'closed_sim')
        self.record_trades(session, 'sim')

        n.stop()

    # risk ----------------------------------------------------------------------

    def realised_pnl(self, trade_record: dict) -> float:
        '''calculates realised pnl of a tp or close denominated in the trade's
        own R value'''

        # TODO this function sometimes gets called during repair_trade_records. i need to make sure that it would still
        #  work even if the record was broken

        func_name = sys._getframe().f_code.co_name
        k15 = Timer(f'{func_name}')
        k15.start()

        position = trade_record['position']
        side = position['direction']

        trades = trade_record['trade']
        entry = float(position['entry_price'])
        init_stop = float(position['init_hard_stop'])
        init_size = float(position['init_base_size'])
        final_exit = float(trades[-1].get('exe_price'))
        final_size = float(trades[-1].get('base_size'))
        r_val = abs((entry - init_stop) / entry)
        if side in ['spot', 'long']:
            trade_pnl = (final_exit - entry) / entry
        else:
            trade_pnl = (entry - final_exit) / entry
        trade_r = round(trade_pnl / r_val, 3)
        if init_size:
            scalar = final_size / init_size
        else:
            scalar = 0
        realised_r = trade_r * scalar

        # if position.get('state') == 'real':
        #     if side == 'spot':
        #         self.realised_pnls['real_spot'] += realised_r
        #     elif side == 'long':
        #         self.realised_pnls['real_long'] += realised_r
        #     else:
        #         self.realised_pnls['real_short'] += realised_r
        # elif position.get('state') == 'sim':
        #     if side == 'spot':
        #         self.realised_pnls['sim_spot'] += realised_r
        #     elif side == 'long':
        #         self.realised_pnls['sim_long'] += realised_r
        #     else:
        #         self.realised_pnls['sim_pshort'] += realised_r
        # else:
        #     print(f"state in record: {position.get('state')}")
        #     print(f'{trade_r = }')

        # print(f"{position['pair']} realised {position['state']} {side} pnl: {realised_r}")
        k15.stop()

        return realised_r

    def record_trades(self, session, state: str) -> None:
        '''saves any trades dictionary to its respective json file'''

        b = Timer(f'record_trades {state}')
        b.start()
        session.counts.append(f'record_trades {state}')

        if state in {'open', 'all'}:
            filepath = Path(f"{session.write_records}/{self.id}/open_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.open_trades, file)
        if state in {'sim', 'all'}:
            filepath = Path(f"{session.write_records}/{self.id}/sim_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.sim_trades, file)
        if state in {'tracked', 'all'}:
            filepath = Path(f"{session.write_records}/{self.id}/tracked_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.tracked_trades, file)
        if state in {'closed', 'all'}:
            filepath = Path(f"{session.write_records}/{self.id}/closed_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.closed_trades, file)
        if state in {'closed_sim', 'all'}:
            filepath = Path(f"{session.write_records}/{self.id}/closed_sim_trades.json")
            if not filepath.exists():
                filepath.touch()
            with open(filepath, "w") as file:
                json.dump(self.closed_sim_trades, file)
        b.stop()

    def reduce_fr(self, factor: float, fr_prev: float, fr_inc: float):
        """reduces fixed_risk by factor (with the floor value being 0)"""
        ideal = fr_prev * factor
        reduce = max(ideal, fr_inc)
        return max((fr_prev - reduce), 0)

    def score_accum(self, session, direction: str):
        '''calculates perf score from recent performance. also saves the
        instance property open_pnl_changes dictionary'''

        if self.perf_log:
            last = self.perf_log[-1]
        if self.perf_log and last.get(f'wanted_open_pnl_{direction[0]}'):
            prev_open_pnl = last.get(f'wanted_open_pnl_{direction[0]}')
            if direction == 'spot':
                curr_open_pnl = self.starting_wopnl_spot
            elif direction == 'long':
                curr_open_pnl = self.starting_wopnl_l
            elif direction == 'short':
                curr_open_pnl = self.starting_wopnl_s

            opnl_change_pct = 100 * (curr_open_pnl - prev_open_pnl) / prev_open_pnl
            self.open_pnl_changes['wanted'] = opnl_change_pct
        elif self.perf_log:
            total_bal = session.spot_bal if direction == 'spot' else session.margin_bal
            prev_bal = last.get('spot_balance', total_bal) if direction == 'spot' else last.get('margin_balance',
                                                                                                total_bal)
            opnl_change_pct = 100 * (total_bal - prev_bal) / prev_bal
        else:
            opnl_change_pct = 0

        lookup = f'wanted_rpnl_{direction}'
        pnls = {}
        for i in range(1, 5):
            if self.perf_log and len(self.perf_log) > 5:
                pnls[i] = self.perf_log[-1 * i].get(lookup, -1)
            else:
                pnls[i] = -1  # if there's no data yet, return -1 instead

        score_1 = 0
        score_2 = 0
        if opnl_change_pct > 0.1:
            score_1 += 5
        elif opnl_change_pct < -0.1:
            score_1 -= 5
        if pnls.get(1) > 0:
            score_2 += 4
        elif pnls.get(1) < 0:
            score_2 -= 4
        if pnls.get(2) > 0:
            score_2 += 3
        elif pnls.get(2) < 0:
            score_2 -= 3
        if pnls.get(3) > 0:
            score_2 += 2
        elif pnls.get(3) < 0:
            score_2 -= 2
        if pnls.get(4) > 0:
            score_2 += 1
        elif pnls.get(4) < 0:
            score_2 -= 1

        return score_1, score_2, pnls

    def set_fixed_risk(self, session, direction: str) -> float:
        """calculates fixed risk setting for new trades based on recent performance
        and previous setting. if recent performance is very good, fr is increased slightly.
        if not, fr is decreased by thirds"""

        o = Timer(f'set_fixed_risk-{direction}')
        o.start()

        if (self.mode == 'spot' and direction in ['long', 'short']) or (self.mode == 'margin' and direction == 'spot'):
            return 0

        now = datetime.now().strftime('%d/%m/%y %H:%M')

        if self.perf_log:
            fr_prev = self.perf_log[-1].get(f'fr_{direction}', 0)
        else:
            fr_prev = 0
        fr_inc = self.fr_max / 10  # increment fr in 10% steps of the range

        score_1, score_2, pnls = self.score_accum(session, direction)
        if score_1 + score_2 >= 11:
            print(f"set_fixed_risk {direction}: score {score_1 + score_2}")
        score = score_1 + score_2

        if score == 15:
            fr = min(fr_prev + (2 * fr_inc), self.fr_max)
        elif score >= 11:
            fr = min(fr_prev + fr_inc, self.fr_max)
        elif score >= 3:
            fr = fr_prev
        elif score >= -3:
            fr = self.reduce_fr(0.333, fr_prev, fr_inc)
        elif score >= -7:
            fr = self.reduce_fr(0.5, fr_prev, fr_inc)
        else:
            fr = 0

        if fr != fr_prev:
            title = f'{now}'
            note = f'{self.name} {direction} fixed risk adjusted from {round(fr_prev * 10000, 1)}bps to {round(fr * 10000, 1)}bps'
            # pb.push_note(title, note)
            print(note)
            print(f"{self.name} calculated {direction} score: {score}, pnls: {pnls}")

        o.stop()
        return round(fr, 5)

    def test_fixed_risk(self, fr_l: float, fr_s: float) -> None:
        """manually overrides fixed risk settings for testing purposes"""
        if not self.live:
            print(f'*** WARNING: FIXED RISK MANUALLY SET to {fr_l} / {fr_s} ***')
            self.fixed_risk_l = fr_l
            self.fixed_risk_s = fr_s

    def print_fixed_risk(self):
        if self.mode == 'spot' and self.fixed_risk_spot:
            print(f"{self.name} fixed risk: {(self.fixed_risk_spot * 10000):.2f}bps")
        elif self.mode == 'margin' and self.fixed_risk_l or self.fixed_risk_s:
            frl_bps = self.fixed_risk_l * 10000
            frs_bps = self.fixed_risk_s * 10000
            print(f"{self.name} fr long: {(frl_bps):.2f}bps, fr short: {(frs_bps):.2f}bps")

    def set_max_pos(self) -> int:
        """sets the maximum number of open positions for the agent. if the median
        pnl of current open positions is greater than 0, max pos will be set to 50,
        otherwise max_pos will be set to 20"""

        p = Timer('set_max_pos')
        p.start()
        max_pos = 12
        if self.real_pos:
            open_pnls = [v.get('pnl') for v in self.real_pos.values() if v.get('pnl')]
            if open_pnls:
                avg_open_pnl = stats.median(open_pnls)
            else:
                avg_open_pnl = 0
            max_pos = 6 if avg_open_pnl <= 0 else 12
        p.stop()
        return max_pos

    def calc_init_opnl(self, session):
        if self.mode == 'spot':
            self.real_pos['USDT'] = session.spot_usdt_bal

            ropnl_spot = self.open_pnl('spot', 'real')
            wanted_spot, unwanted_spot = self.open_pnl('spot', 'sim')

            self.starting_ropnl_spot = ropnl_spot
            self.starting_sopnl_spot = wanted_spot + unwanted_spot
            self.starting_wopnl_spot = ropnl_spot + wanted_spot
            self.starting_uopnl_spot = unwanted_spot

        elif self.mode == 'margin':
            self.real_pos['USDT'] = session.margin_usdt_bal

            ropnl_long = self.open_pnl('long', 'real')
            wanted_long, unwanted_long = self.open_pnl('long', 'sim')

            self.starting_ropnl_l = ropnl_long
            self.starting_sopnl_l = wanted_long + unwanted_long
            self.starting_wopnl_l = ropnl_long + wanted_long
            self.starting_uopnl_l = unwanted_long

            ropnl_short = self.open_pnl('short', 'real')
            wanted_short, unwanted_short = self.open_pnl('short', 'sim')

            self.starting_ropnl_s = ropnl_short
            self.starting_sopnl_s = wanted_short + unwanted_short
            self.starting_wopnl_s = ropnl_short + wanted_short
            self.starting_uopnl_s = unwanted_short

    def calc_tor(self) -> None:
        '''collects all the open risk values from real_pos into a list and
        calculates the sum total of all the open risk for the agent in question'''

        u = Timer('calc_tor')
        u.start()
        self.or_list = [float(v.get('or_R')) for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
        u.stop()

    def filter_signals(self, session, pair, signals, inval_risk_score, usdt_size, usdt_depth):
        fs = Timer('filter_signals')
        fs.start()

        now = datetime.now()
        filters = []

        if not session.pairs_data[pair]['margin_allowed']:
            filters.append('not_a_margin_pair')

        if ((('spot' in signals.get('signal')) and (inval_risk_score < 0.5))
                or
                (('long' in signals.get('signal')) and (inval_risk_score < 0.5))
                or
                (('short' in signals.get('signal')) and (inval_risk_score < 0.5))):
            if not self.in_pos['sim']:
                self.counts_dict['too_risky'] += 1
            # print(f"{self.name} {pair} open signal, risk score: {inval_risk_score}")
            filters.append('too_risky')

        if usdt_depth == 0:
            if not self.in_pos['sim']:
                self.counts_dict['too_much_spread'] += 1
            filters.append('too_much_spread')

        if usdt_size > usdt_depth > (usdt_size / 2):  # only trim size if books are a bit too thin
            self.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth

        if usdt_depth < usdt_size:
            if not self.in_pos['sim']:
                self.counts_dict['books_too_thin'] += 1
            filters.append('books_too_thin')

        elif usdt_size < 30:
            if not self.in_pos['sim']:
                self.counts_dict['too_small'] += 1
            filters.append('too_small')

        if float(self.real_pos['USDT']['qty']) < usdt_size:
            if not self.in_pos['sim']:
                self.counts_dict['not_enough_usdt'] += 1
            filters.append('not_enough_usdt')

        if session.algo_limit_reached(pair):
            if not self.in_pos['sim']:
                self.counts_dict['algo_order_limit'] += 1
            filters.append('algo_order_limit')

        # only continue if a real trade is still possible
        if filters:
            fs.stop()
            return filters

        # check total open risk and close profitable positions if necessary -----------
        self.reduce_risk_M(session)
        usdt_bal = session.spot_usdt_bal if self.mode == 'spot' else session.margin_usdt_bal
        self.real_pos['USDT'] = usdt_bal

        # make sure there aren't too many open positions now --------------------------
        self.calc_tor()
        if self.num_open_positions >= self.max_positions:
            if not self.in_pos['sim']:
                self.counts_dict['too_many_pos'] += 1
            # print(f"{self.name} {pair} positions: {self.num_open_positions} max: {self.max_positions}")
            filters.append('too_many_pos')
        if self.total_open_risk > self.total_r_limit:
            if not self.in_pos['sim']:
                self.counts_dict['too_much_or'] += 1
            filters.append('too_much_or')

        fs.stop()
        return filters

    # signal scores -------------------------------------------------------------

    def calc_inval_risk_score(self, inval: float, mean: float=0.0, std: float=0.03) -> float:
        """i want to analyse the probability of any given value of inval_risk at trade entry producing a positive pnl.
        Once i have a set of scores (1 for each band of inval_risk range) i can normalise them to a 0-1 range and output
        that as the inval_risk_score"""

        # TODO ulitmately i want this function to work as described in the docstring but for a quick fix it can just use
        #  a normal probability density function to achieve a similar result

        # def normpdf(x, mean, std):
        #     var = float(std) ** 2
        #     denom = (2 * math.pi * var) ** .5
        #     num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        #     return num / denom
        #
        # score =  normpdf(inval, mean, std)

        # this is a temporary way to score inval, a linear function which scores 0% inval at 1, >=10% inval at 0
        score = max(1 - (inval * 10), 0)

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
        now = datetime.now()
        duration = round((now.timestamp() - open_time) / 3600, 1)

        direction = v['position']['direction']

        curr_price = session.pairs_data[pair]['price']
        long = v['position']['direction'] == 'long'
        if long:
            pnl = 100 * (curr_price - entry_price) / entry_price
        else:
            pnl = 100 * (entry_price - curr_price) / entry_price
        trig = float(v['position']['entry_price'])
        sl = float(v['position']['init_hard_stop'])
        r = 100 * abs(trig - sl) / sl
        value = round(float(current_base_size) * curr_price, 2)
        pf_pct = round(100 * value / total_bal, 5)

        wanted = v['trade'][0].get('wanted', True)

        stats_dict = {'value': str(value), 'pf%': pf_pct, 'duration (h)': duration,
                      'pnl_R': round(pnl / r, 5), 'pnl_%': round(pnl, 5), 'direction': direction,
                      'wanted': wanted
                      }

        we.stop()
        return stats_dict

    def current_positions(self, session, state: str) -> dict:
        '''creates a dictionary of open positions by checking either 
        open_trades.json, sim_trades.json or tracked_trades.json'''

        a = Timer(f'current_positions-{state}')
        a.start()

        if state == 'open':
            data = self.open_trades
        elif state == 'sim':
            data = self.sim_trades
        elif state == 'tracked':
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
            elif v['position']['direction'] in ['long', 'short']:
                total_bal = session.margin_bal
            if state == 'tracked':
                size_dict[asset] = {}
            else:
                try:
                    size_dict[asset] = self.open_trade_stats(session, total_bal, v)
                except KeyError as e:
                    print(f"Problem calling open_trade_stats on {self.name}.{state}_trades, {asset}")
                    print('KeyError:', e)
                    print('')
                    pprint(v)
                    print('')

        for i in drop_items:
            if state == 'open':
                del self.open_trades[i]
            if state == 'sim':
                del self.sim_trades[i]
            if state == 'tracked':
                del self.tracked_trades[i]
        self.record_trades(session, state)

        a.stop()
        return size_dict

    def update_pos(self, session, asset: str, new_bal: str, inval_ratio: float, state: str) -> Dict[str, float]:
        """checks for the current balance of a particular asset and returns it in
        the correct format for the sizing dict. also calculates the open risk for
        a given asset and returns it in R and $ denominations"""

        jk = Timer('update_pos')
        jk.start()

        pair = f'{asset}USDT'
        price = session.pairs_data[pair]['price']

        if state == 'real':
            direction = self.open_trades[pair]['position']['direction']
            pfrd = self.open_trades[pair]['position']['pfrd']
        elif state == 'sim':
            direction = self.sim_trades[pair]['position']['direction']
            pfrd = self.sim_trades[pair]['position']['pfrd']

        value = price * float(new_bal)
        bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
        pct = round(100 * value / bal, 5)

        open_risk = value * (1 - inval_ratio) if direction in ['long', 'spot'] else value * (inval_ratio - 1)
        if float(pfrd):
            open_risk_r = (open_risk / float(pfrd))
        else:
            open_risk_r = 0.0

        jk.stop()

        return {'value': f"{value:.2f}", 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}

    def update_non_live_tp(self, asset: str, tp_pct: int, switch: str) -> dict:  # dict[str, float | str | Any]:
        """updates sizing dictionaries (real/sim) with new open trade stats when
        state is sim or real but not live and a take-profit is triggered"""
        qw = Timer('update_non_live_tp')
        qw.start()
        tp_scalar = 1 - (tp_pct / 100)
        if switch == 'real':
            pos_dict = self.real_pos
        elif switch == 'sim':
            pos_dict = self.sim_pos

        val = float(pos_dict.get(asset).get('value')) * tp_scalar
        pf = pos_dict.get(asset).get('pf%') * tp_scalar
        or_R = pos_dict.get(asset).get('or_R') * tp_scalar
        or_dol = pos_dict.get(asset).get('or_$') * tp_scalar

        qw.stop

        return {'value': f"{val:.2f}", 'pf%': pf, 'or_R': or_R, 'or_$': or_dol}

    def init_in_pos(self, pair: str) -> None:
        """initialises the in_pos dictionary for the current pair and fills it with values"""

        f = Timer(f'init_in_pos')
        f.start()

        self.in_pos = {'pair': pair, 'real': None, 'sim': None, 'tracked': None}

        if pair in self.open_trades.keys():
            self.in_pos['real'] = self.open_trades[pair]['position']['direction']
        if pair in self.sim_trades.keys():
            self.in_pos['sim'] = self.sim_trades[pair]['position']['direction']
        if pair in self.tracked_trades.keys():
            self.in_pos['tracked'] = self.tracked_trades[pair]['position']['direction']
        f.stop()

    def too_new(self, df: pd.DataFrame) -> bool:
        """returns True if there is less than 200 periods of history AND if
        there are no current positions in the asset"""

        g = Timer('too_new')
        g.start()

        if self.in_pos['real'] or self.in_pos['sim'] or self.in_pos['tracked']:
            no_pos = False
        else:
            no_pos = True

        g.stop()

        return len(df) < self.ohlc_length and no_pos

    def open_pnl(self, direction: str, state: str) -> Union[int, float]:
        '''adds up the pnls of all open positions for a given state'''

        h = Timer(f'open_pnl {state}')
        h.start()
        real_total = 0
        sim_total = [0, 0]
        if state == 'real':
            for pos in self.real_pos.values():
                if pos.get('pnl_R') and (pos['direction'] == direction):
                    real_total += pos['pnl_R']
            total = real_total

        elif state == 'sim':
            for pos in self.sim_pos.values():
                wanted = pos.get('wanted') or 1
                if pos.get('pnl_R') and (pos['direction'] == direction) and wanted:
                    sim_total[0] += pos['pnl_R']
                elif pos.get('pnl_R') and (pos['direction'] == direction) and not wanted:
                    sim_total[1] += pos['pnl_R']
            total = sim_total

        else:
            print('open_pnl requires argument real or sim')

        h.stop()
        return total

    # move_stop
    def create_placeholder(self, pair, direction, atr):
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        placeholder = {'type': f'move_stop_{direction}',
                       'state': 'real',
                       'pair': pair,
                       'stop_price': atr,
                       'timestamp': now,
                       'completed': None
                       }
        self.open_trades[pair]['placeholder'] = placeholder

    def clear_stop(self, session, pair, pos_record):
        _, base_size = funcs.clear_stop_M(session, pair, pos_record)
        if not base_size:
            print('--- running move_stop and needed to use position base size')
            base_size = pos_record['base_size']
        # else:
        #     print("--- running move_stop and DIDN'T need to use position base size")

        return base_size

    def update_records_1(self, pair, base_size):
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = base_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

    def reset_stop(self, session, pair, base_size, direction, atr):
        trade_side = be.SIDE_SELL if direction == 'long' else be.SIDE_BUY
        lim = atr * 0.8 if direction == 'long' else atr * 1.2
        stop_order = funcs.set_stop_M(session, pair, base_size, trade_side, atr, lim)

        return stop_order

    def update_records_2(self, session, pair, atr, stop_order):
        self.open_trades[pair]['position']['hard_stop'] = atr
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime') / 1000)
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def move_api_stop(self, session, pair, direction, atr, pos_record, stage=0):
        if stage == 0:
            self.create_placeholder(pair, direction, atr)
        if stage <= 1:
            base_size = self.clear_stop(session, pair, pos_record)

            self.update_records_1(pair, base_size)

        if stage <= 2:
            stop_order = self.reset_stop(session, pair, base_size, direction, atr)

            self.update_records_2(session, pair, atr, stop_order)

    def move_real_stop(self, session, pair, df, direction):
        func_name = sys._getframe().f_code.co_name
        k14 = Timer(f'{func_name}')
        k14.start()

        low_atr = df[f'atr-10-{float(self.mult)}-lower'].iloc[-1]
        high_atr = df[f'atr-10-{float(self.mult)}-upper'].iloc[-1]
        atr = low_atr if direction == 'long' else high_atr

        current_stop = float(self.open_trades[pair]['position']['hard_stop'])

        move_condition = (((direction in ['long', 'spot']) and (low_atr > (current_stop * 1.001)))
                          or ((direction == 'short') and (high_atr < (current_stop / 1.001))))

        if move_condition:
            print(f"*** {self.name} {pair} move real {direction} stop from {current_stop:.5} to {atr:.5}")
            try:
                self.move_api_stop(session, pair, direction, atr, self.open_trades[pair]['position'])
            except bx.BinanceAPIException as e:
                self.record_trades(session, 'all')
                print(f'{self.name} problem with move_stop order for {pair}')
                print(e)

        k14.stop()

    def move_non_real_stop(self, session, pair, df, state, direction):
        func_name = sys._getframe().f_code.co_name
        k13 = Timer(f'{func_name}')
        k13.start()

        if state == 'sim':
            trade_record = self.sim_trades[pair]
        elif state == 'tracked':
            trade_record = self.tracked_trades[pair]

        current_stop = float(trade_record['position']['hard_stop'])

        low_atr = df[f'atr-10-{float(self.mult)}-lower'].iloc[-1]
        high_atr = df[f'atr-10-{float(self.mult)}-upper'].iloc[-1]
        atr = low_atr if direction == 'long' else high_atr

        move_condition = (((direction in ['long', 'spot']) and (low_atr > current_stop))
                          or ((direction == 'short') and (high_atr < current_stop)))

        if move_condition:
            if state == 'sim':
                self.sim_trades[pair]['position']['hard_stop'] = atr
            elif state == 'tracked':
                self.tracked_trades[pair]['position']['hard_stop'] = atr
            self.record_trades(session, state)

        k13.stop()

    def tp_signals(self, asset: str) -> None:
        '''calculates whether the current position needs to take profit and stores 
        the result in the in_pos dictionary. this cant be done in the main signals
        function because those signals are based solely on the indicators and are 
        therefore state-agnostic. these take-profit signals are based on the indicators 
        and the risk management parameters and so produce unique signals for each 
        position, therefore they cannot be sent to the omfs to be blindly split 
        into state signals in the same way as open and close signals can'''

        j = Timer('tp_signals')
        j.start()
        if self.real_pos.get(asset):
            real_or = self.real_pos.get(asset).get('or_R', 0)
            self.in_pos['real_tp_sig'] = ((float(real_or) > self.indiv_r_limit) and
                                          (abs(self.in_pos.get('real_price_delta', 0)) > 0.002))
        if self.sim_pos.get(asset):
            sim_or = self.sim_pos.get(asset).get('or_R', 0)
            self.in_pos['sim_tp_sig'] = ((float(sim_or) > self.indiv_r_limit) and
                                         (abs(self.in_pos.get('sim_price_delta', 0)) > 0.002))
        j.stop()

    # dispatch

    def open_pos(self, session, pair, size, stp, inval_ratio, mkt_state, sim_reason, direction):
        # real
        if (direction in ['long', 'short']) and (self.in_pos['real'] is None) and (not sim_reason):
            self.open_real_M(session, pair, size, stp, inval_ratio, mkt_state, direction, 0)

        if (direction == 'spot') and (self.in_pos['real'] is None) and (not sim_reason):
            self.open_real_s(session, pair, size, stp, inval_ratio, mkt_state, 0)

        # sim
        if (direction in ['spot', 'long', 'short']) and (self.in_pos['sim'] is None) and sim_reason:
            self.open_sim(session, pair, stp, inval_ratio, mkt_state, sim_reason, direction)

    def tp_pos(self, session, pair, stp, inval_ratio, direction):
        if self.in_pos.get('real_tp_sig'):
            self.tp_real_full_M(session, pair, stp, inval_ratio, direction)

        if self.in_pos.get('sim_tp_sig'):
            self.tp_sim(session, pair, stp, direction)

        if self.in_pos.get('tracked_tp_sig'):
            self.tp_tracked(session, pair, direction)

    def close_pos(self, session, pair, direction):
        if self.in_pos['real'] == direction:
            print('')
            self.close_real_full_M(session, pair, direction)

        if self.in_pos['sim'] == direction:
            self.close_sim(session, pair, direction)

        if self.in_pos['tracked'] == direction:
            self.close_tracked(session, pair, direction)

    def reduce_risk_M(self, session):
        func_name = sys._getframe().f_code.co_name
        k12 = Timer(f'{func_name}')
        k12.start()

        if not (positions := [(p, float(r.get('or_R')), float(r.get('pnl_%'))) for p, r in self.real_pos.items()
                              if r.get('or_R') and (float(r.get('or_R')) > 0) and r.get('pnl_%')]):
            return

        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # find the sum of all R values
        total_r = sum(float(x.get('or_R', 0)) for x in self.real_pos.values())

        for pos in sorted_pos:
            asset, or_R, pnl_pct = pos
            if not (total_r > self.total_r_limit and or_R > self.indiv_r_limit and pnl_pct > 0.3):
                continue

            print(f'\n*** {self.name} tor: {total_r:.1f}, reducing risk ***')
            pair = f"{asset}USDT"
            now = datetime.now().strftime('%d/%m/%y %H:%M')
            note = f"{self.name} reduce risk {pair}, or: {or_R}R, pnl: {pnl_pct}%"
            print(now, note)
            try:
                direction = self.open_trades[pair]['position']['direction']
                # insert placeholder record
                self.create_close_placeholder(session, pair, direction)
                # clear stop
                cleared_size = self.close_clear_stop(session, pair)

                if not cleared_size:
                    print(
                        f'{self.name} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
                    cleared_size = self.set_size_from_free(session, pair)

                # execute trade
                close_order = self.close_position(session, pair, cleared_size, 'reduce_risk', direction)
                # repay loan
                repay_size = self.close_repay(session, pair, close_order, direction)
                # update records
                self.open_to_tracked(session, pair, close_order, direction)
                # update in-pos, real_pos, counts etc
                self.close_real_7(session, pair, repay_size, direction)

                total_r -= or_R

            except BinanceAPIException as e:
                self.record_trades(session, 'all')
                print(f'problem with reduce_risk order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} reduce_risk order')
                continue

        k12.stop()

    # real open margin

    def create_record(self, session, pair, size, stp, inval_ratio, mkt_state, direction):
        price = session.pairs_data[pair]['price']
        usdt_size: str = f"{size * price:.2f}"
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        placeholder = {'type': f'open_{direction}',
                       'state': 'real',
                       'pair': pair,
                       'signal_score': 'signal_score',
                       'base_size': size,
                       'quote_size': usdt_size,
                       'trig_price': price,
                       'stop_price': stp,
                       'inval': inval_ratio,
                       'timestamp': now,
                       'completed': None
                       }
        self.open_trades[pair] = {}
        self.open_trades[pair]['placeholder'] = placeholder | mkt_state
        self.open_trades[pair]['position'] = {'pair': pair, 'direction': direction, 'state': 'real'}

        note = f"{self.name} real open {direction} {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)

    def omf_borrow(self, session, pair, size, direction):
        if direction == 'long':
            price = session.pairs_data[pair]['price']
            borrow_size = f"{size * price:.2f}"
            funcs.borrow_asset_M('USDT', borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = 'USDT'
        elif direction == 'short':
            asset = pair[:-4]
            borrow_size = uf.valid_size(session, pair, size)
            funcs.borrow_asset_M(asset, borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = asset
        else:
            print('*** WARNING open_real_2 given wrong direction argument ***')

        self.open_trades[pair]['position']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['completed'] = 'borrow'

    def increase_position(self, session, pair, size, direction):
        price = session.pairs_data[pair]['price']
        usdt_size = f"{size * price:.2f}"

        if direction == 'long':
            api_order = funcs.buy_asset_M(session, pair, float(usdt_size), False, price, session.live)
        elif direction == 'short':
            api_order = funcs.sell_asset_M(session, pair, size, price, session.live)

        self.open_trades[pair]['position']['base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['init_base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['open_time'] = int(api_order.get('transactTime') / 1000)
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return api_order

    def open_trade_dict(self, session, pair, api_order, stp, direction):
        price = session.pairs_data[pair]['price']

        open_order = funcs.create_trade_dict(api_order, price, session.live)
        open_order['pair'] = pair
        open_order['type'] = f"open_{direction}"
        open_order['state'] = 'real'
        open_order['score'] = 'signal score'
        open_order['hard_stop'] = str(stp)

        self.open_trades[pair]['position']['entry_price'] = open_order['exe_price']
        if direction == 'long':
            self.open_trades[pair]['position']['pfrd'] = str(self.fixed_risk_dol_l)
        else:
            self.open_trades[pair]['position']['pfrd'] = str(self.fixed_risk_dol_s)
        self.open_trades[pair]['placeholder'].update(open_order)
        self.open_trades[pair]['placeholder']['completed'] = 'trade_dict'

        return open_order

    def open_set_stop(self, session, pair, stp, open_order, direction):
        # stop_size = float(open_order.get('base_size'))
        stop_size = open_order.get('base_size')  # this is a string, go back to using above line if this causes bugs

        if direction == 'long':
            stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_SELL, stp, stp * 0.8)
        elif direction == 'short':
            stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_BUY, stp, stp * 1.2)

        open_order['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['hard_stop'] = str(stp)
        self.open_trades[pair]['position']['init_hard_stop'] = str(stp)
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime') / 1000)
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = int(stop_order.get('transactTime') / 1000)
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return open_order

    def open_save_records(self, session, pair, open_order):
        self.open_trades[pair]['trade'] = [open_order]
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def open_update_real_pos_usdtM_counts(self, session, pair, size, inval_ratio, direction):
        price = session.pairs_data[pair]['price']
        usdt_size = f"{size * price:.2f}"
        asset = pair[:-4]

        self.in_pos['real'] = direction

        if session.live:
            self.real_pos[asset] = self.update_pos(session, asset, size, inval_ratio, 'real')
            self.real_pos[asset]['pnl_R'] = 0
            if direction == 'long':
                session.update_usdt_m(borrow=float(usdt_size))
            elif direction == 'short':
                session.update_usdt_m(up=float(usdt_size))
        else:
            bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
            pf = f"{float(usdt_size) / bal:.2f}"
            if direction == 'long':
                or_dol = f"{bal * self.fixed_risk_l:.2f}"
            elif direction == 'short':
                or_dol = f"{bal * self.fixed_risk_s:.2f}"
            self.real_pos[asset] = {'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol)}

        self.counts_dict[f'real_open_{direction}'] += 1
        self.num_open_positions += 1

    def open_real_M(self, session, pair, size, stp, inval_ratio, mkt_state, direction, stage):
        func_name = sys._getframe().f_code.co_name
        k11 = Timer(f'{func_name}')
        k11.start()

        if stage == 0:
            print('')
            self.create_record(session, pair, size, stp, inval_ratio, mkt_state, direction)
            self.omf_borrow(session, pair, size, direction)
            api_order = self.increase_position(session, pair, size, direction)
        if stage <= 1:
            open_order = self.open_trade_dict(session, pair, api_order, stp, direction)
        if stage <= 2:
            open_order = self.open_set_stop(session, pair, stp, open_order, direction)
        if stage <= 3:
            self.open_save_records(session, pair, open_order)
            self.open_update_real_pos_usdtM_counts(session, pair, size, inval_ratio, direction)
        k11.stop()

    # real open spot

    def open_real_s(self, session, pair, size, stp, inval_ratio, mkt_state, stage):
        func_name = sys._getframe().f_code.co_name
        ros = Timer(f'{func_name}')
        ros.start()

        if stage == 0:
            print('')
            self.create_record(session, pair, size, stp, inval_ratio, mkt_state, 'spot')

        ros.stop()

    # real tp

    def create_tp_placeholder(self, session, pair, stp, inval_ratio, direction):
        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # insert placeholder record
        placeholder = {'type': f'tp_{direction}',
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'stop_price': stp,
                       'inval': inval_ratio,
                       'timestamp': now,
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
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])
        self.check_size_against_records(pair, real_bal, cleared_size)

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
            api_order = funcs.sell_asset_M(session, pair, order_size, price, session.live)
        elif direction == 'short':
            api_order = funcs.buy_asset_M(session, pair, order_size, True, price, session.live)

        # update records
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        print(f"+++ {self.name} {pair} tp {direction} resulted in base qty: {new_base_size}")
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
            repay_usdt = str(max(Decimal(tp_order.get('quote_size', 0)), Decimal(liability)))
            funcs.repay_asset_M('USDT', repay_usdt, session.live)
        else:
            repay_asset = str(max(Decimal(liability), Decimal(tp_order.get('base_size', 0))))
            repay_usdt = f"{float(repay_asset) * price:.2f}"
            funcs.repay_asset_M(asset, repay_asset, session.live)

        # update trade dict
        tp_order['type'] = f'close_{direction}'
        tp_order['state'] = 'real'
        tp_order['reason'] = 'trade over-extended'
        self.open_trades[pair]['position']['liability'] = '0'
        self.open_trades[pair]['placeholder'].update(tp_order)
        self.open_trades[pair]['placeholder']['repay_usdt']
        self.open_trades[pair]['placeholder']['completed'] = 'repay_100'

        return tp_order, repay_usdt

    def open_to_tracked(self, session, pair, close_order, direction):
        asset = pair[:-4]
        self.open_trades[pair]['trade'].append(close_order)

        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        del self.open_trades[pair]['placeholder']
        self.tracked_trades[pair] = self.open_trades[pair]
        self.open_trades[pair]['position']['state'] = 'tracked'
        self.record_trades(session, 'tracked')

        del self.open_trades[pair]
        self.record_trades(session, 'open')

        self.in_pos['real'] = None
        self.in_pos['tracked'] = direction

        self.tracked[asset] = {'qty': '0', 'value': '0', 'pf%': '0', 'or_R': '0', 'or_$': '0'}

    def tp_update_records_100(self, session, pair, order_size, usdt_size, direction):
        asset = pair[:-4]
        price = session.pairs_data[pair]['price']

        if session.live and direction == 'long':
            session.update_usdt_m(repay=float(usdt_size))
        elif session.live and direction == 'short':
            usdt_size = round(order_size * price, 5)
            session.update_usdt_m(down=usdt_size)
        elif (not session.live) and direction == 'long':
            self.real_pos['USDT']['qty'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] += float(self.real_pos[asset].get('pf%'))
        elif (not session.live) and direction == 'short':
            self.real_pos['USDT']['qty'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] -= float(self.real_pos[asset].get('pf%'))

        del self.real_pos[asset]

        del self.real_pos[asset]
        self.counts_dict[f'real_close_{direction}'] += 1

    def tp_repay_partial(self, session, pair, stp, tp_order, direction):
        asset = pair[:-4]

        if direction == 'long':
            repay_size = tp_order.get('quote_size')
            funcs.repay_asset_M('USDT', repay_size, session.live)
        elif direction == 'short':
            repay_size = tp_order.get('base_size')
            funcs.repay_asset_M(asset, repay_size, session.live)

        # create trade dict
        tp_order['type'] = f'tp_{direction}'
        tp_order['state'] = 'real'
        tp_order['hard_stop'] = str(stp)
        tp_order['reason'] = 'trade over-extended'

        self.open_trades[pair]['position']['liability'] = uf.update_liability(self.open_trades[pair], repay_size,
                                                                              'reduce')
        self.open_trades[pair]['placeholder'].update(tp_order)
        self.open_trades[pair]['placeholder']['tp_order'] = tp_order
        self.open_trades[pair]['placeholder']['completed'] = 'repay_part'

        return tp_order

    def tp_reset_stop(self, session, pair, stp, tp_order, direction):
        new_size = self.open_trades[pair]['position']['base_size']

        if direction == 'long':
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_SELL, stp, stp * 0.8)
        elif direction == 'short':
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_BUY, stp, stp * 1.2)

        tp_order['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['hard_stop'] = str(stp)
        self.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime') / 1000)
        self.open_trades[pair]['placeholder']['hard_stop'] = str(stp)
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = int(stop_order.get('transactTime') / 1000)
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return tp_order

    def open_to_open(self, session, pair, tp_order):
        self.open_trades[pair]['trade'].append(tp_order)
        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def tp_update_records_partial(self, session, pair, pct, inval_ratio, order_size, tp_order, direction):
        asset = pair[:-4]
        price = session.pairs_data[pair]['price']
        new_size = self.open_trades[pair]['position']['base_size']

        self.open_trades['position']['pfrd'] = self.open_trades['position']['pfrd'] * (pct / 100)
        if session.live:
            self.real_pos[asset].update(
                self.update_pos(session, asset, new_size, inval_ratio, 'real'))
            if direction == 'long':
                repay_size = tp_order.get('base_size')
                session.update_usdt_m(repay=float(repay_size))
            elif direction == 'short':
                usdt_size = round(order_size * price, 5)
                session.update_usdt_m(down=usdt_size)
        else:
            self.real_pos[asset].update(self.update_non_live_tp(asset, pct, 'real'))

        self.counts_dict[f'real_tp_{direction}'] += 1

    def tp_real_full_M(self, session, pair, stp, inval_ratio, direction):
        k10 = Timer(f'tp_real_full')
        k10.start()

        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        print('')

        self.create_tp_placeholder(session, pair, stp, inval_ratio, direction)
        pct = self.tp_set_pct(pair)
        # clear stop
        cleared_size = self.tp_clear_stop(session, pair)

        if not cleared_size:
            print(
                f'{self.name} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
            cleared_size = self.set_size_from_free(session, pair)

        # execute trade
        tp_order = self.tp_reduce_position(session, pair, cleared_size, pct, direction)

        note = f"{self.name} real take-profit {pair} {direction} {pct}% @ {price}"
        print(now, note)

        if pct == 100:
            # repay assets
            tp_order, usdt_size = self.tp_repay_100(session, pair, tp_order, direction)
            # update records
            self.open_to_tracked(session, pair, tp_order, direction)
            self.tp_update_records_100(session, pair, cleared_size, usdt_size, direction)

        else:  # if pct < 100%
            # repay assets
            tp_order = self.tp_repay_partial(session, pair, stp, tp_order, direction)
            # set new stop
            tp_order = self.tp_reset_stop(session, pair, stp, tp_order, direction)
            # update records
            self.open_to_open(session, pair, tp_order)
            self.tp_update_records_partial(session, pair, pct, inval_ratio, cleared_size, tp_order, direction)

        k10.stop()

    # real close

    def create_close_placeholder(self, session, pair, direction):
        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # temporary check to catch a possible bug, can delete after ive had a few reduce_risk calls with no bugs
        if direction not in ['long', 'short']:
            print(
                f'*** WARNING, string "{direction}" being passed to create_close_placeholder, either from close omf or reduce_risk')

        # insert placeholder record
        placeholder = {'type': f'close_{direction}',
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'timestamp': now,
                       'completed': None
                       }
        self.open_trades[pair]['placeholder'] = placeholder

    def close_clear_stop(self, session, pair):
        clear, cleared_size = funcs.clear_stop_M(session, pair, self.open_trades[pair]['position'])
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])
        self.check_size_against_records(pair, real_bal, cleared_size)

        # update position and placeholder
        # self.open_trades[pair]['position']['hard_stop'] = None # don't want this to be changed to none in case it is
        # becoming a tracked trade, in which case i will still need a stop price in order to move non-real stop
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

        return cleared_size

    def close_position(self, session, pair, close_size, reason, direction):
        price = session.pairs_data[pair]['price']

        if direction == 'long':
            api_order = funcs.sell_asset_M(session, pair, close_size, price, session.live)
        elif direction == 'short':
            api_order = funcs.buy_asset_M(session, pair, close_size, True, price, session.live)

        # update position and placeholder
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        print(f"+++ {self.name} {pair} close {direction} resulted in base qty: {new_base_size}")
        close_order = funcs.create_trade_dict(api_order, price, session.live)

        close_order['pair'] = pair
        close_order['type'] = f'close_{direction}'
        close_order['state'] = 'real'
        close_order['reason'] = reason
        self.open_trades[pair]['placeholder'].update(close_order)
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return close_order

    def close_repay(self, session, pair, close_order, direction):
        asset = pair[:-4]
        liability = self.open_trades[pair]['position']['liability']

        if direction == 'long':
            repay_size = str(max(Decimal(close_order.get('quote_size', 0)), Decimal(liability)))
            funcs.repay_asset_M('USDT', repay_size, session.live)
        elif direction == 'short':
            repay_size = str(max(Decimal(liability), Decimal(close_order.get('base_size', 0))))
            funcs.repay_asset_M(asset, repay_size, session.live)

        # update records
        self.open_trades[pair]['position']['liability'] = '0'
        self.open_trades[pair]['placeholder']['completed'] = 'repay'

        return repay_size

    def open_to_closed(self, session, pair, close_order):
        self.open_trades[pair]['trade'].append(close_order)
        rpnl = self.realised_pnl(self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        trade_id = int(self.open_trades[pair]['position']['open_time'])
        self.closed_trades[trade_id] = self.open_trades[pair]['trade']
        self.record_trades(session, 'closed')

        del self.open_trades[pair]
        self.record_trades(session, 'open')

        if hasattr(self,
                   'in_pos'):  # if open_to_closed is being called inside record_stopped_trades, in_pos will not exist
            self.in_pos['real'] = None

    def close_real_7(self, session, pair, close_size, direction):
        asset = pair[:-4]
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
        del self.real_pos[asset]
        self.counts_dict[f'real_close_{direction}'] += 1

    def close_real_full_M(self, session, pair, direction, size=0, stage=0):
        k9 = Timer(f'close_real_full')
        k9.start()

        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        note = f"{self.name} real close {direction} {pair} @ {price}"
        print(now, note)

        if stage == 0:
            if self.open_trades.get('placeholder'):
                del self.open_trades[pair]['placeholder']
            self.create_close_placeholder(session, pair, direction)
        if stage <= 1:
            # cancel stop
            cleared_size = self.close_clear_stop(session, pair)

            if not cleared_size:
                print(
                    f'{self.name} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
                cleared_size = self.set_size_from_free(session, pair)
        if stage <= 2:
            # execute trade
            if size:  # this condition is for when this function is called from this specific stage
                cleared_size = size

            if cleared_size:
                close_order = self.close_position(session, pair, cleared_size, 'close_signal', direction)
            else:
                # in this case, the trade is closed and the record is ruined, so just delete the record and move on
                print(f"{self.name} {pair} {direction} position no longer exists, deleting trade record")
                del self.open_trades[pair]
                return
        if stage <= 3:
            # repay loan
            repay_size = self.close_repay(session, pair, close_order, direction)
        if stage <= 4:
            # update records
            self.open_to_closed(session, pair, close_order)
            # update in_pos, real_pos, counts etc
            self.close_real_7(session, pair, repay_size, direction)

        k9.stop()

    # sim
    def open_sim(self, session, pair, stp, inval_ratio, mkt_state, sim_reason, direction):
        k8 = Timer(f'open_sim')
        k8.start()

        asset = pair[:-4]
        price = session.pairs_data[pair]['price']
        usdt_size = 128.0
        size = f"{usdt_size / price:.8f}"
        pfrd = str(self.fixed_risk_dol_l) if direction in ['spot', 'long'] else str(self.fixed_risk_dol_s)

        wanted = False if 'too_risky' in sim_reason else True

        sim_order = {'pair': pair,
                     'exe_price': str(price),
                     'trig_price': str(price),
                     'base_size': size,
                     'quote_size': '128.0',
                     'hard_stop': str(stp),
                     'reason': sim_reason,
                     'timestamp': int(datetime.utcnow().timestamp()),
                     'type': f'open_{direction}',
                     'fee': '0',
                     'fee_currency': 'BNB',
                     'state': 'sim',
                     'wanted': wanted}

        pos_record = {'base_size': size,
                      'init_base_size': size,
                      'direction': direction,
                      'entry_price': str(price),
                      'hard_stop': str(stp),
                      'init_hard_stop': str(stp),
                      'open_time': int(datetime.utcnow().timestamp()),
                      'pair': pair,
                      'liability': '0',
                      'stop_id': 'not live',
                      'stop_time': int(datetime.utcnow().timestamp()),
                      'state': 'sim',
                      'pfrd': pfrd}

        sim_order = sim_order | mkt_state

        self.sim_trades[pair] = {'trade': [sim_order], 'position': pos_record}

        # self.record_trades(session, 'sim') # might not be necessary to do this on every trade

        self.in_pos['sim'] = direction
        self.sim_pos[asset] = self.update_pos(session, asset, float(size), inval_ratio, 'sim')
        self.sim_pos[asset]['pnl_R'] = 0
        self.counts_dict[f'sim_open_{direction}'] += 1

        k8.stop()

    def tp_sim(self, session, pair, stp, direction):
        k7 = Timer(f'tp_sim')
        k7.start()

        price = session.pairs_data[pair]['price']
        asset = pair[:-4]
        bin_ts = round(datetime.utcnow().timestamp() * 1000)
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])
        order_size = sim_bal / 2
        usdt_size = f"{order_size * price:.2f}"

        # execute order
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': str(order_size),
                    'quote_size': usdt_size,
                    'hard_stop': str(stp),
                    'reason': 'trade over-extended',
                    'timestamp': bin_ts,
                    'type': f'tp_{direction}',
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'sim'}
        self.sim_trades[pair]['trade'].append(tp_order)

        self.sim_trades[pair]['position']['base_size'] = str(order_size)
        self.sim_trades[pair]['position']['hard_stop'] = str(stp)
        self.sim_trades[pair]['position']['stop_time'] = bin_ts
        self.sim_trades[pair]['position']['pfrd'] /= 2

        rpnl = self.realised_pnl(self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"sim_{direction}"] += rpnl
        if 'too_risky' in self.sim_trades[pair]['trade']['reason']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl

        # save records
        # self.record_trades(session, 'sim')

        # update sim_pos
        self.sim_pos[asset].update(self.update_non_live_tp(asset, 50, 'sim'))
        self.counts_dict[f'sim_tp_{direction}'] += 1

        k7.stop()

    def sim_to_closed_sim(self, session, pair, close_order, save_file):
        self.sim_trades[pair]['trade'].append(close_order)
        rpnl = self.realised_pnl(self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.sim_trades[pair]['position']['direction']
        self.realised_pnls[f"sim_{direction}"] += rpnl
        if 'too_risky' in self.sim_trades[pair]['trade'][0]['reason']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl

        trade_id = int(self.sim_trades[pair]['position']['open_time'])
        self.closed_sim_trades[trade_id] = self.sim_trades[pair]['trade']

        del self.sim_trades[pair]

        if save_file:
            self.record_trades(session, 'closed_sim')
            self.record_trades(session, 'sim')

    def close_sim(self, session, pair, direction):
        k6 = Timer(f'close_sim')
        k6.start()

        price = session.pairs_data[pair]['price']
        asset = pair[:-4]
        bin_ts = round(datetime.utcnow().timestamp() * 1000)
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])

        # execute order
        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': str(sim_bal),
                       'quote_size': f"{sim_bal * price:.2f}",
                       'reason': 'close_signal',
                       'timestamp': bin_ts,
                       'type': f'close_{direction}',
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'sim'}

        self.sim_to_closed_sim(session, pair, close_order, save_file=True)

        self.in_pos['sim'] = None
        del self.sim_pos[asset]

        self.counts_dict[f'sim_close_{direction}'] += 1

        k6.stop()

    # tracked

    def tp_tracked(self, session, pair, stp, direction):
        k5 = Timer(f'tp_tracked')
        k5.start()
        print('')
        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        note = f"{self.name} tracked take-profit {pair} {direction} 50% @ {price}"
        print(now, note)

        trade_record = self.tracked_trades[pair]['trade']
        timestamp = round(datetime.utcnow().timestamp() * 1000)

        # execute order
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': '0',
                    'quote_size': '0',
                    'reason': 'trade over-extended',
                    'timestamp': timestamp,
                    'type': f'tp_{direction}',
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'tracked'}
        trade_record.append(tp_order)

        self.tracked_trades[pair]['position']['hard_stop'] = str(stp)
        self.tracked_trades[pair]['position']['stop_time'] = timestamp

        # update records
        self.tracked_trades[pair]['trade'] = trade_record
        self.record_trades(session, 'tracked')

        self.in_pos['tracked_pfrd'] = self.in_pos['tracked_pfrd'] / 2

        k5.stop()

    def tracked_to_closed(self, session, pair, close_order):
        asset = pair[:-4]

        self.tracked_trades[pair]['trade'].append(close_order)

        trade_id = int(self.tracked_trades[pair]['position']['open_time'])
        self.closed_trades[trade_id] = self.tracked_trades[pair]['trade']
        self.record_trades(session, 'closed')

        del self.tracked_trades[pair]
        self.record_trades(session, 'tracked')

        del self.tracked[asset]
        self.in_pos['tracked'] = None

    def close_tracked(self, session, pair, direction):
        k4 = Timer(f'close_tracked')
        k4.start()

        print('')
        price = session.pairs_data[pair]['price']
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        note = f"{self.name} tracked close {direction} {pair} @ {price}"
        print(now, note)

        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': '0',
                       'quote_size': '0',
                       'reason': 'close_signal',
                       'timestamp': timestamp,
                       'type': f'close_{direction}',
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'tracked'}

        self.tracked_to_closed(session, pair, close_order)

        k4.stop()

    # other

    def aged_condition(self, signal_age, series_1, series_2):
        tot = 0

        for x in range(-1, -(signal_age + 1), -1):
            tot += int(series_1.iloc[x] > series_2.iloc[x])

        # if tot adds up to the same number as signal age that shows that all loops returned True
        return tot == self.signal_age

    def check_size_against_records(self, pair, real_bal, base_size):
        k3 = Timer(f'check_size_against_records')
        k3.start()

        base_size = Decimal(base_size)
        if base_size and (real_bal != base_size):  # check records match reality
            print(f"{self.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            mismatch = 100 * abs(base_size - real_bal) / base_size
            print(f"{mismatch = }%")

        k3.stop()

    def set_size_from_free(self, session, pair):
        """if clear_stop returns a base size of 0, this can be called to check for free balance,
        in case the position was there but just not in a stop order"""
        k2 = Timer(f'set_size_from_fee')
        k2.start()

        asset = pair[:-4]
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])

        session.margin_account_info()
        session.get_asset_bals()
        free_bal = session.margin_bals[asset]['free']

        k2.stop()

        return min(free_bal, real_bal)

    # repair trades
    def check_invalidation(self, session, ph):
        """returns true if trade is still valid, false otherwise.
        trade is still valid if direction is long and price is above invalidation, OR if dir is short and price is below"""
        pair = ph['pair']
        stp = ph['stop_price']

        dir_up = ph['type'].split('_')[1] == 'long'
        price = session.pairs_data[pair]['price']
        price_up = price > stp

        return dir_up == price_up

    def check_close_sig(self, session, ph):
        """returns true if trade is still above the previous close signal, false otherwise.
        trade is still valid if direction is long and price is above trig_price, OR if dir is short and price is below"""
        pair = ph['pair']
        trig = ph['trig_price']

        dir_up = ph.get('direction') or ph['type'].split('_')[1] == 'long'
        price = session.pairs_data[pair]['price']
        price_up = price > trig

        return dir_up == price_up

    def repair_open(self, session, ph):
        pair = ph['pair']
        size = ph['base_size']
        stp = ph['stop_price']
        price = session.pairs_data[pair]['price']
        valid = self.check_invalidation(session, ph)
        direction = ph.get('direction') or ph['type'].split('_')[1]

        if ph['completed'] is None:
            del self.open_trades[pair]

        elif ph['completed'] == 'borrow':
            try:
                funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
            except bx.BinanceAPIException as e:
                print('Problem during repair_open')
                pprint(ph)
                print(e.status_code)
                print(e.message)
            finally:
                del self.open_trades[pair]

        elif ph['completed'] == 'execute':
            if valid:
                self.open_real(session, pair, size, stp, ph['inval'], direction, 1)
            else:
                close_size = ph['api_order']['executedQty']
                if 'long' in ph['type']:
                    funcs.sell_asset_M(session, pair, close_size, price, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, price, session.live)
                funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'trade_dict':
            if valid:
                self.open_real(session, pair, size, stp, ph['inval'], direction, 2)
            else:
                close_size = ph['api_order']['executedQty']
                if direction == 'long':
                    funcs.sell_asset_M(session, pair, close_size, price, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, price, session.live)
                funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'set_stop':
            self.open_real(session, pair, size, stp, ph['inval'], direction, 3)

    def repair_tp(self, session, ph):
        pair = ph['pair']
        cleared_size = ph.get('cleared_size')
        pct = ph.get('pct')
        stp = ph['stop_price']
        valid = self.check_invalidation(session, ph)
        direction = ph.get('direction') or ph['type'].split('_')[1]

        if ph['completed'] is None:
            del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if valid:
                # the function call below sets new stop and updates position dict
                self.open_set_stop(session, pair, stp, ph, direction)
            else:
                self.close_real_full(session, pair, direction)

        elif ph['completed'] == 'execute':
            if pct == 100:
                tp_order, usdt_size = self.tp_repay_100(session, pair, ph['tp_order'], direction)
                self.open_to_tracked(session, pair, tp_order, direction)
                self.tp_update_records_100(session, pair, cleared_size, usdt_size, direction)
            else:
                if valid:
                    tp_order = self.tp_repay_partial(session, pair, stp, ph['tp_order'], direction)
                    tp_order = self.tp_reset_stop(session, pair, stp, tp_order, direction)
                    self.open_to_open(session, pair, tp_order)
                    self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order,
                                                   direction)
                else:
                    remaining = self.open_trades[pair]['position']['base_size']
                    close_order = self.close_position(session, pair, remaining, 'close_signal', direction)
                    repay_size = self.close_repay(session, pair, close_order, direction)
                    self.open_to_closed(session, pair, close_order, direction)
                    self.close_real_7(session, pair, repay_size, direction)

        elif ph['completed'] == 'repay_100':
            self.open_to_tracked(session, pair, ph['tp_order'], direction)
            self.tp_update_records_100(session, pair, cleared_size, ph['repay_usdt'], direction)

        elif ph['completed'] == 'repay_part':
            if valid:
                tp_order = self.tp_reset_stop(session, pair, stp, ph['tp_order'], direction)
                self.open_to_open(session, pair, tp_order)
                self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order,
                                               direction)
            else:
                remaining = self.open_trades[pair]['position']['base_size']
                close_order = self.close_position(session, pair, remaining, 'close_signal', direction)
                repay_size = self.close_repay(session, pair, close_order, direction)
                self.open_to_closed(session, pair, close_order, direction)
                self.close_real_7(session, pair, repay_size, direction)

        elif ph['completed'] == 'set_stop':
            self.open_to_open(session, pair, ph['tp_order'])
            self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, ph['tp_order'],
                                           direction)

    def repair_close(self, session, ph):
        pair = ph['pair']
        direction = ph.get('direction') or ph['type'].split('_')[1]

        if ph['completed'] is None:
            self.close_real_full(session, pair, direction, stage=1)

        elif ph['completed'] == 'clear_stop':
            self.close_real_full(session, pair, direction, stage=2)

        elif ph['completed'] == 'execute':
            self.close_real_full(session, pair, direction, stage=3)

        elif ph['completed'] == 'repay':
            self.close_real_full(session, pair, direction, stage=4)

    def repair_move_stop(self, session, ph):
        # when it failed to move the stop up, price was above the current and new stop levels. since then, price could
        # have stayed above, or it could have moved below one or both stop levels. if price is above both, i simply reset
        # the stop as planned. if price is below the new stop, i should close the position. if price is below the original
        # stop, the position should also be closed but will already have been if the old stop was still in place

        pair = ph['pair']
        price = session.pairs_data[pair]['price']
        pos_record = self.open_trades[pair]['position']
        direction = ph.get('direction') or ph['type'].split('_')[1]

        if ph['completed'] == None:
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, direction, ph['stop_price'], pos_record, stage=1)
            elif price > self.open_trades[pair]['position']['hard_stop']:
                self.close_real_full(session, pair, direction)
            else:
                del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, direction, ph['stop_price'], pos_record, stage=2)
            else:
                self.close_real_full(session, pair, direction)

    def repair_trade_records(self, session):
        k1 = Timer(f'repair trade records')
        k1.start()
        ph_list = []
        for pair in self.open_trades.values():
            if pair.get('placeholder'):
                ph_list.append(pair['placeholder'])

        for ph in ph_list:
            try:
                if ph['type'] in ['open_long', 'open_short']:
                    self.repair_open(session, ph)
                elif ph['type'] in ['tp_long', 'tp_short']:
                    self.repair_tp(session, ph)
                elif ph['type'] in ['close_long', 'close_short']:
                    self.repair_close(session, ph)
                elif ph['type'] in ['move_stop_long', 'move_stop_short']:
                    self.repair_move_stop(session, ph)
            except bx.BinanceAPIException as e:
                print("problem during repair_trade_records")
                pprint(ph)
                self.record_trades(session, all)
                print(e.status_code)
                print(e.message)
        k1.stop()


class DoubleST(Agent):
    '''200EMA and regular supertrend for bias with tight supertrend for entries/exits'''

    def __init__(self, session, tf, offset, mult1: int, mult2: float):
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.mult1 = int(mult1)
        self.mult2 = float(mult2)
        self.signal_age = 1
        self.name = f'{self.tf} dst {self.mult1}-{self.mult2}'
        self.id = f"double_st_{self.tf}_{self.offset}_{self.mult1}_{self.mult2}"
        self.ohlc_length = 200 + self.signal_age
        self.cross_age_name = f"cross_age-st-10-{self.mult1}-10-{self.mult2}"
        Agent.__init__(self, session)
        session.indicators.update(['ema-200',
                                   f"st-10-{self.mult1}",
                                   f"st-10-{self.mult2}",
                                   self.cross_age_name])

    def spot_signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates spot buy and sell signals based on 2 supertrend indicators
        and a 200 period EMA"""

        bullish_ema = df.close.iloc[-1] > df['ema_200'].iloc[-1]
        bullish_loose = df.close.iloc[-1] > df[f'st-10-{float(self.mult1)}'].iloc[-1]
        bullish_tight = df.close.iloc[-1] > df[f'st-10-{self.mult2}'].iloc[-1]
        bearish_tight = df.close.iloc[-1] < df[f'st-10-{self.mult2}'].iloc[-1]

        if bullish_ema:
            session.above_200_ema.add(pair)
        else:
            session.below_200_ema.add(pair)

        if bullish_ema and bullish_loose and bullish_tight:
            signal = 'spot_open'
        elif bearish_tight:
            signal = 'spot_close'
        else:
            signal = None

        if inval := df[f'st-10-{self.mult2}'].iloc[-1]:
            inval_ratio = inval / df.close.iloc[-1]
        else:
            inval = 0
            inval_ratio = 100000

        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA"""

        k = Timer(f'dst_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return {'signal': None, 'inval': 0, 'inval_ratio': 100000}

        bullish_ema = df.close.iloc[-1] > df.ema_200.iloc[-1]
        bearish_ema = df.close.iloc[-1] < df.ema_200.iloc[-1]
        bullish_loose = self.aged_condition(self.signal_age, df.close, df[f'st-10-{float(self.mult1)}'])
        bearish_loose = self.aged_condition(self.signal_age, df[f'st-10-{float(self.mult1)}'], df.close)
        bullish_tight = self.aged_condition(self.signal_age, df.close, df[f'st-10-{self.mult2}'])
        bearish_tight = self.aged_condition(self.signal_age, df[f'st-10-{self.mult2}'], df.close)

        if bullish_ema:
            session.above_200_ema.add(pair)
        else:
            session.below_200_ema.add(pair)

        # bullish_book = bid_ask_ratio > 1
        # bearish_book = bid_ask_ratio < 1
        # bullish_volume = price rising on low volume or price falling on high volume
        # bearish_volume = price rising on high volume or price falling on low volume

        if bullish_ema and bullish_loose and bullish_tight:  # and bullish_book
            signal = 'open_long'
        elif bearish_ema and bearish_loose and bearish_tight:  # and bearish_book
            signal = 'open_short'
        elif bearish_tight:
            signal = 'close_long'
        elif bullish_tight:
            signal = 'close_short'
        else:
            signal = None

        if inval := df[f'st-10-{self.mult2}'].iloc[-1]:
            inval_ratio = inval / df.close.iloc[-1]
        else:
            inval = 0
            inval_ratio = 100000
        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}


class EMACross(Agent):
    '''Simple EMA cross strategy with a longer-term EMA to set bias and a 
    trailing stop based on ATR bands'''

    def __init__(self, session, tf, offset, lookback_1, lookback_2, mult):
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.lb1 = lookback_1
        self.lb2 = lookback_2
        self.mult = mult
        self.name = f'{self.tf} emacross {self.lb1}-{self.lb2}-{self.mult}'
        self.id = f"ema_cross_{self.tf}_{self.offset}_{self.lb1}_{self.lb2}_{self.mult}"
        self.ohlc_length = max(200 + 2, self.lb1, self.lb2)
        self.signal_age = 1
        self.cross_age_name = f"cross_age-ema-{self.lb1}-{self.lb2}"
        Agent.__init__(self, session)
        session.indicators.update(['ema-200',
                                   f"ema-{self.lb1}",
                                   f"ema-{self.lb2}",
                                   self.cross_age_name,
                                   f"atr-10-{self.mult}"])

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''

        k = Timer('emax_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return {'signal': None, 'inval': 0, 'inval_ratio': 100000}

        fast_ema_str = f"ema_{self.lb1}"
        slow_ema_str = f"ema_{self.lb2}"
        bias_ema_str = "ema_200"

        bullish_bias = df.close.iloc[-1] > df[bias_ema_str].iloc[-1]
        bearish_bias = df.close.iloc[-1] < df[bias_ema_str].iloc[-1]

        bullish_emas = df[fast_ema_str].iloc[-1] > df[slow_ema_str].iloc[-1]
        bearish_emas = df[fast_ema_str].iloc[-1] < df[slow_ema_str].iloc[-1]
        bullish_emas = self.aged_condition(self.signal_age, df[fast_ema_str], df[slow_ema_str])
        bearish_emas = self.aged_condition(self.signal_age, df[slow_ema_str], df[fast_ema_str])

        atr_lower_below = df.close.iloc[-1] > df[f'atr-10-{self.mult}-lower'].iloc[-1]
        atr_upper_above = df.close.iloc[-1] < df[f'atr-10-{self.mult}-upper'].iloc[-1]

        x_age = 0 - (self.signal_age + 1)
        bullish_cross = bullish_emas and (df[fast_ema_str].iloc[x_age] < df[slow_ema_str].iloc[x_age])
        bearish_cross = bearish_emas and (df[fast_ema_str].iloc[x_age] > df[slow_ema_str].iloc[x_age])

        in_long = (self.in_pos['real'] == 'long'
                   or self.in_pos['sim'] == 'long'
                   or self.in_pos['tracked'] == 'long')
        in_short = (self.in_pos['real'] == 'short'
                    or self.in_pos['sim'] == 'short'
                    or self.in_pos['tracked'] == 'short')

        if bullish_bias and bullish_emas and atr_lower_below and not in_long:
            signal = 'open_long'
        elif bearish_bias and bearish_emas and atr_upper_above and not in_short:
            signal = 'open_short'
        elif bearish_emas and in_long:
            signal = 'close_long'
        elif bullish_emas and in_short:
            signal = 'close_short'
        else:
            signal = None

        if bullish_bias:
            session.above_200_ema.add(pair)
        else:
            session.below_200_ema.add(pair)

        if self.in_pos['real']:
            self.move_real_stop(session, pair, df, self.in_pos['real'])
        for state in ['sim', 'tracked']:
            if self.in_pos[state]:
                self.move_non_real_stop(session, pair, df, state, self.in_pos[state])

        if ((signal == 'open_long') or in_long) and df[f'atr-10-{self.mult}-lower'].iloc[-1]:
            inval = df[f'atr-10-{self.mult}-lower'].iloc[-1]
            inval_ratio = inval / df.close.iloc[-1]

        elif ((signal == 'open_short') or in_short) and df[f'atr-10-{self.mult}-upper'].iloc[-1]:
            inval = df[f'atr-10-{self.mult}-upper'].iloc[-1]
            inval_ratio = inval / df.close.iloc[-1]

        else:
            inval = None
            inval_ratio = None

        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}


class EMACrossHMA(Agent):
    """Simple EMA cross strategy with a longer-term HMA to set bias more
    responsively and a trailing stop based on ATR bands"""

    def __init__(self, session, tf, offset, lookback_1, lookback_2, mult):
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.lb1 = lookback_1
        self.lb2 = lookback_2
        self.mult = mult
        self.name = f'{self.tf} emaxhma {self.lb1}-{self.lb2}-{self.mult}'
        self.id = f"ema_cross_hma_{self.tf}_{self.offset}_{self.lb1}_{self.lb2}_{self.mult}"
        self.ohlc_length = max(200 + 2, self.lb1, self.lb2)
        self.signal_age = 1
        self.cross_age_name = f"cross_age-ema-{self.lb1}-{self.lb2}"
        Agent.__init__(self, session)
        session.indicators.update(['hma-200',
                                   f"ema-{self.lb1}",
                                   f"ema-{self.lb2}",
                                   self.cross_age_name,
                                   f"atr-10-{self.mult}"])

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''

        k = Timer('emaxhma_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return {'signal': None, 'inval': 0, 'inval_ratio': 100000}

        fast_ema_str = f"ema_{self.lb1}"
        slow_ema_str = f"ema_{self.lb2}"
        bias_hma_str = "hma_200"

        bullish_bias = df.close.iloc[-1] > df[bias_hma_str].iloc[-1]
        bearish_bias = df.close.iloc[-1] < df[bias_hma_str].iloc[-1]

        # bullish_emas = df[fast_ema_str].iloc[-1] > df[slow_ema_str].iloc[-1]
        # bearish_emas = df[fast_ema_str].iloc[-1] < df[slow_ema_str].iloc[-1]
        bullish_emas = self.aged_condition(self.signal_age, df[fast_ema_str], df[slow_ema_str])
        bearish_emas = self.aged_condition(self.signal_age, df[slow_ema_str], df[fast_ema_str])

        x_age = 0 - (self.signal_age + 1)
        bullish_cross = bullish_emas and (df[fast_ema_str].iloc[x_age] < df[slow_ema_str].iloc[x_age])
        bearish_cross = bearish_emas and (df[fast_ema_str].iloc[x_age] > df[slow_ema_str].iloc[x_age])

        lower = f'atr-10-{self.mult}-lower'
        upper = f'atr-10-{self.mult}-upper'
        atr_lower_below = df.close.iloc[-1] > df[lower].iloc[-1]
        atr_upper_above = df.close.iloc[-1] < df[upper].iloc[-1]

        in_long = (self.in_pos['real'] == 'long'
                   or self.in_pos['sim'] == 'long'
                   or self.in_pos['tracked'] == 'long')
        in_short = (self.in_pos['real'] == 'short'
                    or self.in_pos['sim'] == 'short'
                    or self.in_pos['tracked'] == 'short')

        if bullish_bias and bullish_emas and atr_lower_below:
            signal = 'open_long'
        elif bearish_bias and bearish_emas and atr_upper_above:
            signal = 'open_short'
        elif bearish_emas and in_long:
            signal = 'close_long'
        elif bullish_emas and in_short:
            signal = 'close_short'
        else:
            signal = None

        if bullish_bias:
            session.above_200_ema.add(pair)
        else:
            session.below_200_ema.add(pair)

        if self.in_pos['real']:
            self.move_real_stop(session, pair, df, self.in_pos['real'])
        for state in ['sim', 'tracked']:
            if self.in_pos[state]:
                self.move_non_real_stop(session, pair, df, state, self.in_pos[state])

        if ((signal == 'open_long') or in_long) and df[lower].iloc[-1]:
            inval = df[lower].iloc[-1]
            inval_ratio = inval / df.close.iloc[-1]

        elif ((signal == 'open_short') or in_short) and df[upper].iloc[-1]:
            inval = df[upper].iloc[-1]
            inval_ratio = inval / df.close.iloc[-1]

        else:
            inval = None
            inval_ratio = None

        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}


class AvgTradeSize(Agent):
    """Long-only strategy that looks for unusual spikes in average trade size to detect bullish reversals"""

    def __init__(self, session, tf, offset, min_z: int, lookback: int, mult: float, exit_strat: str) -> None:
        self.mode = 'spot'
        self.tf = tf
        self.offset = offset
        self.mult = mult  # atr multiplier for trailing stop / oco targets
        self.atr_lb = 5
        self.min_z = min_z
        self.lookback = lookback
        self.exit = exit_strat
        self.rr_ratio = 2
        self.name = f'{self.tf} ats {self.min_z}-{self.lookback}-{self.mult}-{self.exit}'
        self.id = f"avg_trade_size_{self.tf}_{self.offset}_{self.min_z}_{self.lookback}_{self.mult}_{self.exit}"
        self.ohlc_length = max(200, self.lookback)
        Agent.__init__(self, session)
        session.indicators.update(['ema-200',
                                   f'atsz-{lookback}',
                                   'stoch_rsi-14-14',
                                   'inside',
                                   'doji',
                                   'engulfing-1',
                                   f"atr-{self.atr_lb}-{self.mult}"])

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates spot buy signals based on the ats_z indicator. does not account for currently open positions,
        just generates signals as the strategy dictates"""

        sig = Timer('ats_spot_signals')
        sig.start()

        signal_dict = {'direction': 'spot'}

        atsz = df.ats_z.iloc[-10:].max()
        stoch_rsi = df.stoch_rsi.iloc[-1]
        doji = 'doji' if df.bullish_doji.iloc[-1] else ''
        engulf = 'engulf' if df.bullish_engulf.iloc[-1] else ''
        ib = 'inside bar' if df.inside_bar.iloc[-1] else ''
        candle = ' '.join([doji, engulf, ib])

        bullish_atsz = atsz > self.min_z
        bullish_candles = df.inside_bar.iloc[-1] or df.bullish_doji.iloc[-1] or df.bullish_engulf.iloc[-1]

        lower = f'atr-{self.atr_lb}-{self.mult}-lower'
        signal_dict['inval'] = df[lower].iloc[-1]
        signal_dict['inval_ratio'] = df[lower].iloc[-1] / df.close.iloc[-1]

        if bullish_atsz and bullish_candles and (signal_dict['inval_ratio'] < 1):
            if self.exit == 'trail':
                signal_dict['signal'] = 'open'
            elif self.exit == 'oco':
                signal_dict['signal'] = 'oco'
                signal_dict['target'] = df.close.iloc[-1] * (signal_dict['inval_ratio'] ** self.rr_ratio)
        else:
            signal_dict['signal'] = None

        if self.exit == 'trail':
            if self.in_pos['real']:
                self.move_real_stop(session, pair, df, self.in_pos['real'])
            if self.in_pos['sim']:
                self.move_non_real_stop(session, pair, df, 'sim', self.in_pos['sim'])
            if self.in_pos['tracked']:
                self.move_non_real_stop(session, pair, df, 'tracked', self.in_pos['tracked'])

        # record signals in json file for analysis
        if signal_dict['signal']:
            record_dict = signal_dict
            record_dict['time'] = df.timestamp.iloc[-1].timestamp()
            record_dict['pair'] = pair
            record_dict['timeframe'] = self.tf
            record_dict['entry'] = df.close.iloc[-1]
            record_dict['stoch_rsi'] = stoch_rsi
            record_dict['ats_z'] = atsz
            record_dict['pattern'] = candle
            record_dict['exit'] = self.exit
            record_dict['min_z'] = self.min_z
            record_dict['lookback'] = self.lookback
            record_dict['atr_mult'] = self.mult
            record_dict['bullish_ema'] = int(df['ema_200'].iloc[-1] > df['ema_200'].iloc[-5])

            fp = Path('/home/ross/Documents/backtester_2021/trades.json')
            fp.touch(exist_ok=True)
            with open(fp, 'r') as file:
                try:
                    data = json.load(file)
                except JSONDecodeError:
                    data = None
            if data:
                data.append(record_dict)
            else:
                data = [record_dict]
            with open(fp, 'w') as file:
                json.dump(data, file)

        sig.stop()

        ############################# THIS MUST BE CHANGED BACK WHEN I WANT TRAIL SIGNALS ##############################
        if self.exit == 'oco':
            return signal_dict
        else:
            signal_dict['signal'] = None
            return signal_dict

    def signals_old(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates spot buy signals based on the ats_z indicator"""

        sig = Timer('ats_spot_signals')
        sig.start()

        signal_dict = {}

        bullish_atsz = df.atsz.iloc[-10:].max() > self.min_z
        bullish_candles = df.inside_bar.iloc[-1] | df.bullish_doji.iloc[-1] | df.bullish_engulf.iloc[-1]

        if bullish_atsz and bullish_candles and not self.in_pos['real']:
            signal_dict['signal'] = 'real_open_long'
        elif bullish_atsz and bullish_candles and not self.in_pos['sim']:
            signal_dict['signal'] = 'sim_open_long'
        else:
            signal_dict['signal'] = None

        upper = f'atr-10-{self.mult}-upper'
        lower = f'atr-10-{self.mult}-lower'
        signal_dict['inval'] = df[lower].iloc[-1]
        signal_dict['inval_ratio'] = float(
            df.close.iloc[-1] / df[lower].iloc[-1])  # current price proportional to invalidation price

        if self.exit == 'trail':
            if self.in_pos['real']:
                self.move_real_stop(session, pair, df, self.in_pos['real'])
            for state in ['sim', 'tracked']:
                if self.in_pos[state]:
                    self.move_non_real_stop(session, pair, df, state, self.in_pos[state])

        elif self.exit == 'oco':
            if signal_dict['signal']:
                signal_dict['target'] = df[upper].iloc[-1]

        if signal_dict['signal']:
            print(self.name)
            pprint(signal_dict)

        sig.stop()
        return signal_dict

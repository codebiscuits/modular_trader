import binance.exceptions as bx
import pandas as pd
from pathlib import Path
import json
from json.decoder import JSONDecodeError
import statistics as stats
from resources.timers import Timer
from typing import Dict
from resources import indicators as ind, keys, utility_funcs as uf, binance_funcs as funcs
from datetime import datetime, timezone
from pushbullet import Pushbullet
from binance.client import Client
import binance.enums as be
from decimal import Decimal, getcontext
from pprint import pprint
import sys
from pyarrow import ArrowInvalid
import traceback
import joblib

# client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12
timestring = '%d/%m/%y %H:%M'

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
        self.backup_trade_records(session)
        self.repair_trade_records(session)
        self.real_pos = self.current_positions(session, 'open')
        self.sim_pos = self.current_positions(session, 'sim')
        self.tracked = self.current_positions(session, 'tracked')
        # self.calc_init_opnl(session)
        # self.open_pnl_changes = {}
        self.max_positions = self.set_max_pos()
        self.total_r_limit = self.max_positions * 1.7 # TODO need to update reduce_risk and run it before/after set_fixed_risk
        self.indiv_r_limit = 1.8
        self.fr_div = 10
        self.next_id = int(datetime.now(timezone.utc).timestamp())
        session.min_length = min(session.min_length, self.ohlc_length)
        session.max_length = max(session.min_length, self.ohlc_length)
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
        else:
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
        now = datetime.now(timezone.utc).strftime(timestring)
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

    @uf.retry_on_busy()
    def find_order(self, session, pair, sid):
        if sid == 'not live':
            return None
        print('get_margin_order')
        session.track_weights(10)
        abc = Timer('all binance calls')
        abc.start()
        order = session.client.get_margin_order(symbol=pair, orderId=sid)
        abc.stop()
        session.counts.append('get_margin_order')

        if not order:
            print(f'No orders on binance for {pair}')

        # insert placeholder record
        placeholder = {'action': "stop",
                       'direction': self.open_trades[pair]['position']['direction'],
                       'state': 'real',
                       'pair': pair,
                       'order': order,
                       'completed': 'order'
                       }
        self.open_trades[pair]['placeholder'] = placeholder

        return order

    def repay_stop(self, session, pair, order):
        if (order.get('side') == 'BUY'):
            asset = pair[:-4]
            stop_size = Decimal(order.get('executedQty'))
            repayed = funcs.repay_asset_M(session, asset, stop_size, session.live)
        else:
            stop_size = Decimal(order.get('cummulativeQuoteQty'))
            repayed = funcs.repay_asset_M(session, 'USDT', stop_size, session.live)

        self.open_trades[pair]['placeholder']['completed'] = 'repay'

        return stop_size

    def create_stop_dict(self, session, pair, order, stop_size):
        stop_dict = funcs.create_stop_dict(session, order)
        stop_dict['action'] = "stop"
        stop_dict['direction'] = self.open_trades[pair]['position']['direction']
        stop_dict['state'] = 'real'
        stop_dict['reason'] = 'hit hard stop'
        stop_dict['liability'] = uf.update_liability(self.open_trades[pair], stop_size, 'reduce')
        if stop_dict['liability'] not in ['0', '0.0']:
            print(
                f"+++ WARNING {self.name} {pair} stop hit, liability record doesn't add up. Recorded value: {stop_dict['liability']} +++")

        return stop_dict

    def save_records(self, session, pair, stop_dict):
        self.open_trades[pair]['trade'].append(stop_dict)
        self.open_trades[pair]['trade'][-1]['liability'] = str(Decimal(0) - Decimal(stop_dict['executedQty']))
        rpnl = self.realised_pnl(session, self.open_trades[pair])
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

    @uf.retry_on_busy()
    def retrieve_margin_order(self, session, pair, sid):
        return session.client.get_margin_order(symbol=pair, orderId=sid)

    def record_stopped_trades(self, session, timeframes) -> None:
        m = Timer('record_stopped_trades')
        m.start()

        # get a list of (pair, stop_id, stop_time) for all open_trades records
        old_ids = list(self.open_trades.items())

        # print(f"{self.name} rst, checking {len(old_ids)} pairs")

        for pair, v in old_ids:
            sid = v['position']['stop_id']
            direction = v['position']['direction']
            stop_time = v['position']['stop_time']
            if sid == 'not live':
                # print(f"{pair} record non-live")
                df = self.get_data(session, pair, timeframes, stop_time)
                stop = float(v['position']['hard_stop'])
                stopped, overshoot_pct, stop_hit_time = self.check_stop_hit(pair, df, direction, stop)
                if stopped:
                    open_dt = datetime.fromtimestamp(v['position']['open_time']).astimezone(timezone.utc)
                    stop_dt = datetime.fromtimestamp(stop_time).astimezone(timezone.utc)
                    hit_dt = datetime.fromtimestamp(stop_hit_time).astimezone(timezone.utc)
                    entry_price = v['position']['entry_price']
                    print(f"{self.name} {pair} {direction} {open_dt = }, {stop_dt = } {hit_dt = }, {entry_price = } {stop = }")
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

                    self.save_records(session, pair, stop_dict)
                    self.counts_dict[f'real_stop_{direction}'] += 1
                else:
                    # print(f"{self.name} {pair} {direction} still open")
                    pass
                continue

            # print('get_margin_order')
            session.track_weights(10)
            abc = Timer('all binance calls')
            abc.start()
            order = self.retrieve_margin_order(session, pair, sid)
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

                # TODO it should be possible to check the placeholder in the trade record and piece together what to do
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
                    del self.real_pos[pair[:-len(self.quote_asset)]]
                    now = datetime.now().strftime("%y/%m/%d %H:%M")
                    pb.push_note(now, f"{self.name} {pair} records deleted, check exchange for remaining position")
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
                     'action': 'partstop',
                     'direction': direction}
                )

                # update position record
                self.open_trades[pair]['position']['base_size'] = str(old_size - exe_size)

                # TODO need to repay loan that was partially freed up

            else:
                # print(f"{self.name} {pair} stop order (id {sid}) not filled, status: {order['status']}")
                pass

        self.record_trades(session, 'closed')
        self.record_trades(session, 'open')

        m.stop()

    # record stopped sim trades ----------------------------------------------

    def get_data(self, session, pair, timeframes: list, stop_time):

        rsst_gd = Timer('rsst - get_data')
        rsst_gd.start()

        # print(f"rsst {self.name} {pair}")

        filepath = Path(f'{session.ohlc_data}/{pair}.parquet')
        check_recent = False # flag to decide whether the ohlc needs updating or not

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
                    print(f"problem loading {pair} ohlc")
                    print(e)
                    filepath.unlink()
                    df = funcs.get_ohlc(session, pair, session.ohlc_tf, '2 years ago UTC')
                    source = 'exchange'
                    print(f'downloaded {pair} from scratch')
            else:
                print(f"{filepath} doesn't exist")
                df = funcs.get_ohlc(session, pair, session.ohlc_tf, '2 years ago UTC')
                source = 'exchange'
                print(f'downloaded {pair} from scratch')

            session.store_ohlc(df, pair, timeframes)

        # check df is localised to UTC
        try:
            df['timestamp'] = df.timestamp.dt.tz_localize('UTC')
            print(f"rsst get_data - {pair} ohlc data wasn't timezone aware, fixing now.")
        except TypeError:
            pass

        if check_recent:
            last = df.timestamp.iloc[-1]
            timespan = datetime.now(timezone.utc).timestamp() - (last.timestamp())
            if timespan > 900:
                df = funcs.update_ohlc(pair, session.ohlc_tf, df, session)
                source += ' and exchange'
                session.store_ohlc(df, pair, timeframes)

        try: # this try/except block can be removed when the problem is solved
            stop_dt = datetime.fromtimestamp(stop_time).astimezone(timezone.utc)
        except ValueError as e:
            print(f"ValueError for {pair} sim stop time: {e.args}")
            traceback.print_stack()
            pprint(self.sim_trades[pair])
        df = df.loc[df.timestamp > stop_dt].reset_index(drop=True)
        # print(f'::: rsst {self.name} get_data {pair} from {source} :::')

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
    #     plot_folder = Path(f"/home/ross/Documents/backtester_2021/trade_plots/{self.name}")
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
        # print(f"\n{self.name} rsst, checking {len(check_pairs)} pairs")
        for pair, v in check_pairs:  # can't loop through the dictionary directly because i delete items as i go
            direction = v['position']['direction']
            base_size = float(v['position']['base_size'])
            stop = float(v['position']['hard_stop'])
            stop_time = v['position']['stop_time']

            df = self.get_data(session, pair, timeframes, stop_time)
            stopped, overshoot_pct, stop_hit_time = self.check_stop_hit(pair, df, direction, stop)
            if stopped:
                print(f"{self.name} {pair} {v['position']['open_time'] = }, {stop_hit_time = }, "
                      f"{v['position']['entry_price'] = }")
                trade_dict = self.create_trade_dict(pair, direction, stop, base_size, stop_hit_time, overshoot_pct, 'sim')
                self.sim_to_closed_sim(session, pair, trade_dict, save_file=False)
                self.counts_dict[f'sim_stop_{direction}'] += 1
            # else:
            #     print(f"{pair} still open")

        self.record_trades(session, 'closed_sim')
        self.record_trades(session, 'sim')

        n.stop()

    # risk ----------------------------------------------------------------------

    def realised_pnl(self, session, trade_record: dict) -> float:
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
        scalar = position['pct_of_full_pos']
        realised_r = trade_r * scalar

        print(f"{position['pair']} rpnl calc: r_val: {r_val:.1%} trade_pnl: {trade_pnl:.1%} trade_r: {trade_r:.2f} "
              f"{scalar = } realised_r: {realised_r:.2f}")
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


    def score_accum(self, direction: str):
        '''calculates perf score from recent performance. also saves the
        instance property open_pnl_changes dictionary'''

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
        rpnl_df = rpnl_df.sort_values('timestamp').reset_index(drop=True)
        rpnl_df['cum_rpnl'] = rpnl_df.rpnl.cumsum()
        rpnl_df['ema_3'] = rpnl_df.rpnl.ewm(3).mean()
        rpnl_df['ema_9'] = rpnl_df.rpnl.ewm(9).mean()
        rpnl_df['ema_27'] = rpnl_df.rpnl.ewm(27).mean()
        rpnl_df['ema_81'] = rpnl_df.rpnl.ewm(81).mean()
        # rpnl_df['timestamp'] = rpnl_df.timestamp.astype(int)
        # print(direction)
        # print(rpnl_df.tail())

        if len(rpnl_df) >= 4:
            pnls = rpnl_df.to_dict(orient='records')[-1]
            print(f'{direction} pnls:', pnls)

            score = 0
            if  rpnl_df.rpnl.iloc[-1] > 0.1:
                score += 5
            elif rpnl_df.rpnl.iloc[-1] < -0.1:
                score -= 5
            if rpnl_df.ema_3.iloc[-1] > 0:
                score += 4
            elif rpnl_df.ema_3.iloc[-1] < 0:
                score -= 4
            if rpnl_df.ema_9.iloc[-1] > 0:
                score += 3
            elif rpnl_df.ema_9.iloc[-1] < 0:
                score -= 3
            if rpnl_df.ema_27.iloc[-1] > 0:
                score += 2
            elif rpnl_df.ema_27.iloc[-1] < 0:
                score -= 2
            if rpnl_df.ema_81.iloc[-1] > 0:
                score += 1
            elif rpnl_df.ema_81.iloc[-1] < 0:
                score -= 1

        else:
            score = 0
            pnls = {'rpnl': 0, 'ema_3': 0, 'ema_9': 0, 'ema_27': 0, 'ema_81': 0}

        return score, pnls

    def score_accum_old(self, direction: str):
        '''calculates perf score from recent performance. also saves the
        instance property open_pnl_changes dictionary'''

        lookup = f'wanted_rpnl_{direction}'
        pnls = {0: self.realised_pnls[f"wanted_{direction}"]}
        for i in range(1, 5):
            if self.perf_log and len(self.perf_log) > 5:
                pnls[i] = self.perf_log[-1 * i].get(lookup, -1)
            else:
                pnls[i] = -1  # if there's no data yet, return -1 instead

        score = 0
        if  pnls.get(0) > 0.1:
            score += 5
        elif pnls.get(0) < -0.1:
            score -= 5
        if pnls.get(1) > 0:
            score += 4
        elif pnls.get(1) < 0:
            score -= 4
        if pnls.get(2) > 0:
            score += 3
        elif pnls.get(2) < 0:
            score -= 3
        if pnls.get(3) > 0:
            score += 2
        elif pnls.get(3) < 0:
            score -= 2
        if pnls.get(4) > 0:
            score += 1
        elif pnls.get(4) < 0:
            score -= 1

        return score, pnls

    def fixed_risk_score(self, direction: str) -> float:
        """calculates fixed risk setting for new trades based on recent performance and previous setting. if recent
        performance is very good, fr is increased slightly. if not, fr is decreased by thirds"""

        o = Timer(f'set_fixed_risk-{direction}')
        o.start()

        if (
            self.mode == 'spot'
            and direction in {'long', 'short'}
            or (self.mode == 'margin' and direction == 'spot')
        ):
            return 0

        fr_prev = self.perf_log[-1].get(f'fr_{direction}', 0) if self.perf_log else 0
        score, pnls = self.score_accum(direction)
        score_str = f"rpnl: {pnls['rpnl']:.2f}, ema_3: {pnls['ema_3']:.2f}, ema_9: {pnls['ema_9']:.2f}, " \
                    f"ema_27: {pnls['ema_27']:.2f}, ema_81: {pnls['ema_81']:.2f}"
        print(f"{direction} score accum returned score: {score}, pnls: {score_str}")

        if score == 15:
            fr = min(fr_prev + 2, self.fr_div)
        elif score >= 11:
            fr = min(fr_prev + 1, self.fr_div)
        elif score >= 3:
            fr = fr_prev
        elif score >= -3:
            fr = self.reduce_fr(0.333, fr_prev, 1)
        elif score >= -7:
            fr = self.reduce_fr(0.5, fr_prev, 1)
        else:
            fr = 0

        now = datetime.now(timezone.utc).strftime(timestring)
        if fr != fr_prev:
            title = f'{now}'
            note = f'{self.name} {direction} fixed risk score adjusted from {fr_prev} to {fr}'
            # pb.push_note(title, note)
            print(note)
            print(f"{self.name} calculated {direction} score: {score}, pnls: {pnls}")

        o.stop()
        return fr


    def set_fixed_risk(self, session):
        """takes the fr_score calculated by fixed_risk_score and uses it to scale fr_max into the current
        fixed risk setting"""
        if self.mode == 'spot':
            self.fr_score_spot = self.fixed_risk_score('spot')
            self.fixed_risk_spot = round(self.fr_score_spot * self.fr_max / self.fr_div, 5)
            self.fr_dol_spot = self.fixed_risk_spot * session.spot_bal
            print(f"{self.name} spot fixed risk score: {self.fr_score_spot} fixed risk: {self.fixed_risk_spot}")

        elif self.mode == 'margin':
            self.fr_score_l = self.fixed_risk_score('long')
            self.fixed_risk_l = round(self.fr_score_l * self.fr_max / self.fr_div, 5)
            self.fixed_risk_dol_l = self.fixed_risk_l * session.margin_bal
            if self.fr_score_l:
                print(f"{self.name} long fixed risk score: {self.fr_score_l} fixed risk: {self.fixed_risk_l}")

            self.fr_score_s = self.fixed_risk_score('short')
            self.fixed_risk_s = round(self.fr_score_s * self.fr_max / self.fr_div, 5)
            self.fixed_risk_dol_s = self.fixed_risk_s * session.margin_bal
            if self.fr_score_s:
                print(f"{self.name} short fixed risk score: {self.fr_score_s} fixed risk: {self.fixed_risk_s}")

    def test_fixed_risk(self, fr_l: float, fr_s: float) -> None:
        """manually overrides fixed risk settings for testing purposes"""
        if not self.live:
            print(f'*** WARNING: FIXED RISK MANUALLY SET to {fr_l} / {fr_s} ***')
            self.fixed_risk_l = fr_l
            self.fixed_risk_s = fr_s

    def print_fixed_risk(self):
        if self.mode == 'spot':# and self.fixed_risk_spot:
            print(f"{self.name} fixed risk: {(self.fixed_risk_spot * 10000):.2f}bps")
        elif self.mode == 'margin':# and self.fixed_risk_l or self.fixed_risk_s:
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

    # def calc_init_opnl(self, session):
    #     if self.mode == 'spot':
    #         self.real_pos['USDT'] = session.spot_usdt_bal
    #
    #         ropnl_spot = self.open_pnl('spot', 'real')
    #         wanted_spot, unwanted_spot = self.open_pnl('spot', 'sim')
    #
    #         self.starting_ropnl_spot = ropnl_spot
    #         self.starting_sopnl_spot = wanted_spot + unwanted_spot
    #         self.starting_wopnl_spot = ropnl_spot + wanted_spot
    #         self.starting_uopnl_spot = unwanted_spot
    #
    #     elif self.mode == 'margin':
    #         self.real_pos['USDT'] = session.margin_usdt_bal
    #
    #         ropnl_long = self.open_pnl('long', 'real')
    #         wanted_long, unwanted_long = self.open_pnl('long', 'sim')
    #
    #         self.starting_ropnl_l = ropnl_long
    #         self.starting_sopnl_l = wanted_long + unwanted_long
    #         self.starting_wopnl_l = ropnl_long + wanted_long
    #         self.starting_uopnl_l = unwanted_long
    #
    #         ropnl_short = self.open_pnl('short', 'real')
    #         wanted_short, unwanted_short = self.open_pnl('short', 'sim')
    #
    #         self.starting_ropnl_s = ropnl_short
    #         self.starting_sopnl_s = wanted_short + unwanted_short
    #         self.starting_wopnl_s = ropnl_short + wanted_short
    #         self.starting_uopnl_s = unwanted_short

    def calc_tor(self) -> None:
        '''collects all the open risk values from real_pos into a list and
        calculates the sum total of all the open risk for the agent in question'''

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

        if self.tf == '1h':
            score = max(1 - (inval * 10), 0) # inval of 0% returns max score of 1, 10% or more returns 0
        elif self.tf == '4h':
            score = max(1-(inval * 5), 0) # inval of 20% or more returns 0
        elif self.tf == '12h':
            score = max(1-(inval * 3), 0) # inval of 33% or more returns 0
        elif self.tf == '1d':
            score = max(1-(inval * 2), 0) # inval of 50% or more returns 0

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
        duration = round((now.timestamp() - open_time) / 3600, 1)

        direction = v['position']['direction']

        curr_price = session.pairs_data[pair]['price']
        long = v['position']['direction'] == 'long'
        pos_scale = v['position']['pct_of_full_pos']
        trig = float(v['position']['entry_price'])
        sl = float(v['position']['init_hard_stop'])
        r = 100 * abs(trig - sl) / sl
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

            if state != 'tracked':
                try:
                    size_dict[asset] = self.open_trade_stats(session, total_bal, v)
                except KeyError as e:
                    print(f"Problem calling open_trade_stats on {self.name}.{state}_trades, {asset}")
                    print('KeyError:', e)
                    print('')
                    pprint(v)
                    print('')
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

    def update_pos(self, session, pair: str, new_bal: str, inval_ratio: float, state: str) -> Dict[str, float]:
        """checks for the current balance of a particular asset and returns it in
        the correct format for the sizing dict. also calculates the open risk for
        a given asset and returns it in R and $ denominations"""

        jk = Timer('update_pos')
        jk.start()

        price = session.pairs_data[pair]['price']

        if state == 'real':
            trade_record = self.open_trades[pair]
        elif state == 'sim':
            trade_record = self.sim_trades[pair]

        pfrd = trade_record['position']['pfrd']
        pos_scale = trade_record['position']['pct_of_full_pos']
        value = f"{price * float(new_bal):.2f}"
        bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
        pct = round(100 * float(value) / bal, 2)

        open_risk = float(value) * abs(1 - inval_ratio)
        open_risk_r = (open_risk / float(pfrd)) * pos_scale

        # if open_risk_r > self.indiv_r_limit:
        #     print(f"{state} {pair} update_pos - {value = } inval_ratio: {inval_ratio:.4f} open_risk: ${open_risk:.2f}, "
        #           f"open_risk_r: {open_risk_r:.2f}R")
        # if open_risk_r > 5:
        #     print(f"excessive open risk on {pair}. current price: ${price}, current inval: ${price / inval_ratio}")
        #     pprint(trade_record)

        jk.stop()

        return {'value': value, 'pf%': pct, 'or_R': open_risk_r, 'or_$': open_risk}

    def update_non_live_tp(self, session, asset: str, tp_pct: int, state: str) -> dict:  # dict[str, float | str | Any]:
        """updates sizing dictionaries (real/sim) with new open trade stats when
        state is sim or real but not live and a take-profit is triggered"""
        qw = Timer('update_non_live_tp')
        qw.start()
        tp_scalar = 1 - (tp_pct / 100)
        pair = asset + session.quote_asset
        if state == 'real':
            trades = self.open_trades[pair]
        elif state == 'sim':
            trades = self.sim_trades[pair]

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
        or_R = (or_dol / pfrd) * pos_scale

        qw.stop

        return {'value': f"{val:.2f}", 'pf%': pf, 'or_R': or_R, 'or_$': or_dol}

    # def open_pnl(self, direction: str, state: str) -> Union[float, list[float]]:
    #     '''adds up the pnls of all open positions for a given state. if the state is real, the real total opnl is
    #     returned, if the state is sim, the wanted sim total and the unwanted sim total are returned in a list'''
    #
    #     h = Timer(f'open_pnl {state}')
    #     h.start()
    #     real_total = 0.0
    #     sim_total = [0, 0] # [wanted, unwanted]
    #     if state == 'real':
    #         for pair, pos in self.real_pos.items():
    #             if pos.get('opnl_R') and (pos['direction'] == direction):
    #                 real_total += pos['opnl_R']
    #                 # print(f"open_pnl - {pair} {pos['opnl_R']}R")
    #         total = real_total
    #
    #     elif state == 'sim':
    #         for pos in self.sim_pos.values():
    #             wanted = pos.get('wanted') or 1
    #             if pos.get('opnl_R') and (pos['direction'] == direction) and wanted:
    #                 sim_total[0] += pos['pnl_R']
    #             elif pos.get('opnl_R') and (pos['direction'] == direction) and not wanted:
    #                 sim_total[1] += pos['opnl_R']
    #         total = sim_total
    #
    #     else:
    #         print('open_pnl requires argument real or sim')
    #
    #     h.stop()
    #     return total

    # move_stop
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
            print('--- running move_stop and needed to use position base size')
            base_size = pos_record['base_size']
        # else:
        #     print("--- running move_stop and DIDN'T need to use position base size")

        return base_size

    def update_records_1(self, session, pair, base_size):
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = base_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'
        self.record_trades(session, 'open')

    def reset_stop(self, session, pair, base_size, direction, atr):
        trade_side = be.SIDE_SELL if direction == 'long' else be.SIDE_BUY
        lim = atr * 0.8 if direction == 'long' else atr * 1.2
        stop_order = funcs.set_stop_M(session, pair, base_size, trade_side, atr, lim)

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
            stop_order = self.reset_stop(session, pair, base_size, direction, atr)

            self.update_records_2(session, pair, atr, stop_order)

    def move_real_stop(self, session, signal):
        func_name = sys._getframe().f_code.co_name
        k14 = Timer(f'{func_name}')
        k14.start()

        pair = signal['pair']
        direction = 'long' if (signal['bias'] == 'bullish') else 'short'
        inval = signal['inval']

        current_stop = float(self.open_trades[pair]['position']['hard_stop'])

        move_condition = (((direction in ['long', 'spot']) and (inval > (current_stop * 1.001)))
                          or ((direction == 'short') and (inval < (current_stop / 1.001))))

        if move_condition:
            print(f"*** {self.name} {pair} move real {direction} stop from {current_stop:.5} to {inval:.5}")
            try:
                self.move_api_stop(session, pair, direction, inval, self.open_trades[pair]['position'])
            except bx.BinanceAPIException as e:
                self.record_trades(session, 'all')
                print(f'{self.name} problem with move_stop order for {pair}')
                print(e)

        asset = signal['pair'][:-len(session.quote_asset)]
        size = self.open_trades[pair]['position']['base_size']
        inval_ratio = signal['inval_ratio']
        self.real_pos[asset].update(self.update_pos(session, pair, size, inval_ratio, 'real'))

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
        elif state == 'tracked':
            trade_record = self.tracked_trades[pair]

        current_stop = float(trade_record['position']['hard_stop'])

        move_condition = (((direction in ['long', 'spot']) and (inval > current_stop))
                          or ((direction == 'short') and (inval < current_stop)))

        if move_condition:
            if state == 'sim':
                self.sim_trades[pair]['position']['hard_stop'] = inval
            elif state == 'tracked':
                self.tracked_trades[pair]['position']['hard_stop'] = inval

        asset = signal['pair'][:-len(session.quote_asset)]
        size = trade_record['position']['base_size']
        inval_ratio = signal['inval_ratio']
        if state == 'sim':
            self.sim_pos[asset] = self.update_pos(session, pair, size, inval_ratio, state)
        elif state == 'tracked':
            self.tracked[asset] = self.update_pos(session, pair, size, inval_ratio, state)
        self.record_trades(session, state)

        k13.stop()

    # dispatch

    def tp_pos(self, session, signal):
        if signal['state'] == 'real' and signal['mode'] == 'margin':
            self.tp_real_full_M(session, signal['pair'], signal['inval'], signal['inval_ratio'], signal['direction'])

        elif signal['state'] == 'real' and signal['mode'] == 'spot':
            self.tp_real_full_s(session, signal['pair'], signal['inval'], signal['inval_ratio'])

        elif signal['state'] == 'sim':
            self.tp_sim(session, signal['pair'], signal['inval'], signal['direction'])

        elif signal['state'] == 'tracked':
            self.tp_tracked(session, signal['pair'], signal['direction'])

    def close_pos(self, session, signal):
        if signal['state'] == 'real' and signal['mode'] == 'margin':
            self.close_real_full_M(session, signal['pair'], signal['direction'])

        elif signal['state'] == 'real' and signal['mode'] == 'spot':
            self.close_real_full_s(session, signal['pair'])

        elif signal['state'] == 'sim':
            self.close_sim(session, signal['pair'], signal['direction'])

        elif signal['state'] == 'tracked':
            self.close_tracked(session, signal['pair'], signal['direction'])

    # real open margin

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
        self.open_trades[pair]['position'] = {'pair': pair, 'direction': direction, 'state': 'real'}

        size = signal['base_size']
        usdt_size = signal['quote_size']
        price = signal['trig_price']
        stp = signal['inval']
        score = signal['inval_score']
        note = f"{self.name} real open {direction} {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}, " \
               f"score: {score:.1%}"
        print(now, note)

    def omf_borrow(self, session, pair, size, direction):
        if direction == 'long':
            price = session.pairs_data[pair]['price']
            borrow_size = f"{size * price:.2f}"
            funcs.borrow_asset_M(session, 'USDT', borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = 'USDT'
        elif direction == 'short':
            asset = pair[:-4]
            borrow_size = uf.valid_size(session, pair, size)
            funcs.borrow_asset_M(session, asset, borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = asset
        else:
            print('*** WARNING open_real_2 given wrong direction argument ***')

        self.open_trades[pair]['position']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['completed'] = 'borrow'

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
        self.open_trades[pair]['position']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = int(stop_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return open_order

    def open_save_records(self, session, pair):
        del self.open_trades[pair]['placeholder']['completed']
        del self.open_trades[pair]['placeholder']['api_order']
        self.open_trades[pair]['trade'] = [self.open_trades[pair]['placeholder']]
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def open_update_real_pos_usdtM_counts(self, session, pair, size, inval_ratio, direction):
        price = session.pairs_data[pair]['price']
        usdt_size = f"{size * price:.2f}"
        asset = pair[:-4]

        if session.live:
            self.real_pos[asset] = self.update_pos(session, pair, size, inval_ratio, 'real')
            self.real_pos[asset]['pnl_R'] = 0
            if direction == 'long':
                session.update_usdt_m(borrow=float(usdt_size))
            elif direction == 'short':
                session.update_usdt_m(up=float(usdt_size))
        else: # TODO does this really need to be done differently from live trades?
            bal = session.spot_bal if self.mode == 'spot' else session.margin_bal
            pf = f"{100 * float(usdt_size) / bal:.2f}"
            or_dol = f"{float(usdt_size) * abs(1 - inval_ratio):.2f}"
            self.real_pos[asset] = {'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol), 'pnl_R': 0}

        self.counts_dict[f'real_open_{direction}'] += 1
        self.num_open_positions += 1

    def open_real_M(self, session, signal, stage, api_order=None, open_order=None):

        # TODO stages 1 - 3 need api_order and/or open_order which they might not get from the previous stage. in
        #  repair_trade_records, i need to be passing the original api_order/open_order from the stage it got to in to
        #  the stage it's triggering so that things go right

        func_name = sys._getframe().f_code.co_name
        k11 = Timer(f'{func_name}')
        k11.start()

        pair = signal['pair']
        direction = signal['direction']
        size = signal['base_size']
        stp = signal['inval']
        inval_ratio = signal['inval_ratio']

        if stage == 0:
            # print('')
            self.create_record(signal)
            self.omf_borrow(session, pair, size, direction)
            api_order = self.increase_position(session, pair, size, direction)
        if stage <= 1:
            open_order = self.open_trade_dict(session, signal, api_order)
        if stage <= 2:
            open_order = self.open_set_stop(session, pair, stp, open_order, direction)
        if stage <= 3:
            self.open_save_records(session, pair)
            self.open_update_real_pos_usdtM_counts(session, pair, size, inval_ratio, direction)
        k11.stop()

    # real open spot

    def open_real_s(self, session, signal, stage):
        func_name = sys._getframe().f_code.co_name
        ros = Timer(f'{func_name}')
        ros.start()

        if stage == 0:
            print('')
            self.create_record(signal)

        ros.stop()

    # real tp

    def create_tp_placeholder(self, session, pair, stp, inval_ratio, direction):
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        # insert placeholder record
        placeholder = {'action': 'tp',
                       'direction': direction,
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'stop_price': stp,
                       'inval': inval_ratio,
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
            api_order = funcs.sell_asset_M(session, pair, order_size, session.live)
        elif direction == 'short':
            api_order = funcs.buy_asset_M(session, pair, order_size, True, session.live)

        # update records
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        self.open_trades[pair]['position']['pct_of_full_pos'] *= (pct / 100)
        # print(f"+++ {self.name} {pair} tp {direction} resulted in base qty: {new_base_size}")
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

    def open_to_tracked(self, session, pair, close_order, direction):
        asset = pair[:-4]
        self.open_trades[pair]['trade'].append(close_order)
        self.open_trades[pair]['trade'][-1]['utc_datetime'] = self.open_trades[pair]['placeholder']['utc_datetime']

        rpnl = self.realised_pnl(session, self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        del self.open_trades[pair]['placeholder']
        self.tracked_trades[pair] = self.open_trades[pair]
        self.open_trades[pair]['position']['state'] = 'tracked'
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
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_SELL, stp, stp * 0.8)
        elif direction == 'short':
            stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_BUY, stp, stp * 1.2)

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

        rpnl = self.realised_pnl(session, self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        print(f"{pair} real tp recorded {rpnl} in real_{direction}")
        self.realised_pnls[f"wanted_{direction}"] += rpnl
        print(f"{pair} real tp recorded {rpnl} in wanted_{direction}")

        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def tp_update_records_partial(self, session, pair, pct, inval_ratio, order_size, tp_order, direction):
        asset = pair[:-len(session.quote_asset)]
        price = session.pairs_data[pair]['price']
        new_size = self.open_trades[pair]['position']['base_size']

        pfrd = float(self.open_trades[pair]['position']['pfrd'])
        if session.live:
            self.real_pos[asset].update(
                self.update_pos(session, pair, new_size, inval_ratio, 'real'))
            if direction == 'long':
                repay_size = tp_order.get('base_size')
                session.update_usdt_m(repay=float(repay_size))
            elif direction == 'short':
                usdt_size = round(order_size * price, 5)
                session.update_usdt_m(down=usdt_size)
        else:
            self.real_pos[asset].update(self.update_non_live_tp(session, asset, pct, 'real'))

        self.counts_dict[f'real_tp_{direction}'] += 1

    def tp_real_full_M(self, session, pair, stp, inval_ratio, direction):
        k10 = Timer(f'tp_real_full')
        k10.start()

        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

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
            tp_order = self.tp_repay_100(session, pair, tp_order, direction)
            # update records
            self.open_to_tracked(session, pair, tp_order, direction)
            self.tp_update_records_100(session, pair, cleared_size, direction)

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
        now = datetime.now(timezone.utc).strftime(timestring)

        # temporary check to catch a possible bug, can delete after ive had a few reduce_risk calls with no bugs
        if direction not in ['long', 'short']:
            print(
                f'*** WARNING, string "{direction}" being passed to create_close_placeholder, either from close omf or reduce_risk')

        # insert placeholder record
        placeholder = {'action': 'close',
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
            api_order = funcs.sell_asset_M(session, pair, close_size, session.live)
        elif direction == 'short':
            api_order = funcs.buy_asset_M(session, pair, close_size, True, session.live)

        # update position and placeholder
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        curr_base_size = self.open_trades[pair]['position']['base_size']
        new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
        if (float(new_base_size)*price) < 0.01:
            new_base_size = Decimal(0)
        self.open_trades[pair]['position']['base_size'] = str(new_base_size)
        if new_base_size != 0:
            (f"+++ {self.name} {pair} close {direction} resulted in base qty: {new_base_size}")
        close_order = funcs.create_trade_dict(api_order, price, session.live)

        close_order['pair'] = pair
        close_order['action'] = 'close'
        close_order['direction'] = direction
        close_order['state'] = 'real'
        close_order['reason'] = reason
        close_order['utc_datetime'] = datetime.now(timezone.utc).strftime(timestring)
        self.open_trades[pair]['placeholder'].update(close_order)
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

        return repay_size

    def open_to_closed(self, session, pair, close_order, repay_size):

        self.open_trades[pair]['trade'].append(close_order)
        self.open_trades[pair]['trade'][-1]['liability'] = str(Decimal(0) - Decimal(repay_size))
        self.open_trades[pair]['trade'][-1]['utc_datetime'] = self.open_trades[pair]['placeholder']['utc_datetime']

        rpnl = self.realised_pnl(session, self.open_trades[pair])
        self.open_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.open_trades[pair]['position']['direction']
        self.realised_pnls[f"real_{direction}"] += rpnl
        self.realised_pnls[f"wanted_{direction}"] += rpnl

        if rpnl <= -1:
            print(f"*.*.* problem with real trade rpnl ({rpnl:.2f})")
            print(self.name)
            pprint(self.open_trades[pair])

        trade_id = int(datetime.now().timestamp()*1000)
        del self.open_trades[pair]['position']
        self.closed_trades[trade_id] = self.open_trades[pair]
        self.record_trades(session, 'closed')

        del self.open_trades[pair]
        self.record_trades(session, 'open')
        asset = pair[:-len(session.quote_asset)]
        del self.real_pos[asset]

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
        self.counts_dict[f'real_close_{direction}'] += 1

    def close_real_full_M(self, session, pair, direction, size=0, stage=0):
        k9 = Timer(f'close_real_full')
        k9.start()

        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

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
            self.open_to_closed(session, pair, close_order, repay_size)
            # TODO if this function was ever called from stage 4, repay_size would not be defined.
            self.close_real_7(session, pair, repay_size, direction)

        k9.stop()

    # sim
    def open_sim(self, session, signal):
        k8 = Timer(f'open_sim')
        k8.start()

        """in order for all the statistics to make sense, a simulated fixed_risk setting of 1/2 of session.fr_max is 
        used for every sim trade"""

        pair = signal['pair']
        asset = pair[:-4]
        direction = signal['direction']
        price = signal['trig_price']

        spot_pfrd = (session.fr_max / 2) * session.spot_bal
        margin_pfrd = (session.fr_max / 2) * session.margin_bal
        pfrd = spot_pfrd if self.mode == 'spot' else margin_pfrd
        usdt_size = pfrd / abs(1 - signal['inval_ratio'])
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
            'stop_time': int(now.timestamp()),
            'timestamp': int(now.timestamp()),
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
            'open_time': int(now.timestamp()),
            'pair': pair,
            'liability': '0',
            'stop_id': 'not live',
            'stop_time': int(now.timestamp()),
            'state': 'sim',
            'pfrd': pfrd,
            'pct_of_full_pos': signal['pct_of_full_pos']}

        self.sim_trades[pair] = {'trade': [sim_order], 'position': pos_record, 'signal': signal}

        self.sim_pos[asset] = self.update_pos(session, pair, float(size), signal['inval_ratio'], 'sim')
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

        now = datetime.now(timezone.utc)
        note = f"{self.name} sim take-profit {pair} {direction} @ {price}"
        print(now.strftime(timestring), note)

        # execute order
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': str(order_size),
                    'quote_size': usdt_size,
                    'hard_stop': str(stp),
                    'stop_time': int(now.timestamp()),
                    'reason': 'trade over-extended',
                    'timestamp': int(now.timestamp()),
                    'utc_datetime': now.strftime(timestring),
                    'action': 'tp',
                    'direction': direction,
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'sim'}
        self.sim_trades[pair]['trade'].append(tp_order)

        self.sim_trades[pair]['position']['base_size'] = str(order_size)
        self.sim_trades[pair]['position']['hard_stop'] = str(stp)
        self.sim_trades[pair]['position']['stop_time'] = now.timestamp()
        self.sim_trades[pair]['position']['pct_of_full_pos'] /= 2

        rpnl = self.realised_pnl(session, self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        self.realised_pnls[f"sim_{direction}"] += rpnl
        if 'low_score' in self.sim_trades[pair]['signal']['sim_reasons']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl

        # save records
        self.record_trades(session, 'sim')

        # update sim_pos
        self.sim_pos[asset].update(self.update_non_live_tp(session, asset, 50, 'sim'))
        self.counts_dict[f'sim_tp_{direction}'] += 1

        k7.stop()

    def sim_to_closed_sim(self, session, pair, close_order, save_file):

        self.sim_trades[pair]['trade'].append(close_order)

        rpnl = self.realised_pnl(session, self.sim_trades[pair])
        self.sim_trades[pair]['trade'][-1]['rpnl'] = rpnl
        direction = self.sim_trades[pair]['position']['direction']
        self.realised_pnls[f"sim_{direction}"] += rpnl

        # TODO use logging to create a dedicated file for each agent's printouts of trades closed at init inval. that
        #  will make them much much easier to see and compare

        if rpnl < -1:
            print(f"*.*.* problem with sim trade rpnl ({rpnl:.2f})")
            print(self.name)
            pprint(self.sim_trades[pair])

        if 'low_score' in self.sim_trades[pair]['signal']['sim_reasons']:
            self.realised_pnls[f"unwanted_{direction}"] += rpnl
        else:
            self.realised_pnls[f"wanted_{direction}"] += rpnl
        trade_id = int(datetime.now().timestamp()*1000)
        del self.sim_trades[pair]['position']
        self.closed_sim_trades[trade_id] = self.sim_trades[pair]

        print(f"added {pair} record to close_sim_trades, id: {trade_id}")

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

        now = datetime.now(timezone.utc)
        note = f"{self.name} sim close {pair} {direction} @ {price}"
        print(now.strftime(timestring), note)

        # execute order
        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': str(sim_bal),
                       'quote_size': f"{sim_bal * price:.2f}",
                       'reason': 'close_signal',
                       'timestamp': int(now.timestamp()),
                       'utc_datetime': now.strftime(timestring),
                       'action': 'close',
                       'direction': direction,
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'sim'}

        self.sim_to_closed_sim(session, pair, close_order, save_file=True)

        self.counts_dict[f'sim_close_{direction}'] += 1

        k6.stop()

    # tracked

    def tp_tracked(self, session, pair, stp, direction):
        k5 = Timer(f'tp_tracked')
        k5.start()
        print('')
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc).strftime(timestring)

        note = f"{self.name} tracked take-profit {pair} {direction} 50% @ {price}"
        print(now, note)

        trade_record = self.tracked_trades[pair]['trade']
        timestamp = round(datetime.now(timezone.utc).timestamp())

        # execute order
        tp_order = {'pair': pair,
                    'exe_price': str(price),
                    'trig_price': str(price),
                    'base_size': '0',
                    'quote_size': '0',
                    'reason': 'trade over-extended',
                    'timestamp': timestamp,
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

        trade_id = int(datetime.now().timestamp()*1000)
        del self.tracked_trades[pair]['position']
        self.closed_trades[trade_id] = self.tracked_trades[pair]
        self.record_trades(session, 'closed')
        del self.tracked_trades[pair]
        self.record_trades(session, 'tracked')
        del self.tracked[asset]

    def close_tracked(self, session, pair, direction):
        k4 = Timer(f'close_tracked')
        k4.start()

        print('')
        price = session.pairs_data[pair]['price']
        now = datetime.now(timezone.utc)
        note = f"{self.name} tracked close {direction} {pair} @ {price}"
        print(now, note)

        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': '0',
                       'quote_size': '0',
                       'reason': 'close_signal',
                       'timestamp': int(now.timestamp()),
                       'utc_datetime': now.strftime(timestring),
                       'action': 'close',
                       'direction': direction,
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'tracked'}

        self.tracked_to_closed(session, pair, close_order)

        k4.stop()

    # other

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
        risk = abs(1 - signal['inval_ratio'])

        direction = signal['direction']
        price = session.pairs_data[signal['pair']]['price']
        if direction == 'spot':
            usdt_size = balance * self.fixed_risk_spot / risk
        elif direction == 'long':
            usdt_size = balance * self.fixed_risk_l / risk
        elif direction == 'short':
            usdt_size = balance * self.fixed_risk_s / risk

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

    def check_size_against_records(self, pair, real_bal, base_size):
        k3 = Timer(f'check_size_against_records')
        k3.start()

        base_size = Decimal(base_size)
        if base_size and (real_bal != base_size):  # check records match reality
            print(f"{self.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            mismatch = 100 * abs(base_size - real_bal) / base_size
            print(f"{mismatch = }%")

        k3.stop()

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
        elif self.mode == 'margin':
            session.m_acct = session.client.get_margin_account()
            session.get_asset_bals_m()
            free_bal = session.margin_bals[asset]['free']

        k2.stop()

        return min(free_bal, real_bal)

    # repair trades
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

    def repair_open(self, session, ph):
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
                if ph['direction'] == 'long':
                    funcs.sell_asset_M(session, pair, close_size, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, session.live)
                funcs.repay_asset_M(session, ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'trade_dict':
            if valid:
                self.open_real(session, pair, size, stp, ph['inval'], direction, 2)
            else:
                close_size = ph['api_order']['executedQty']
                if direction == 'long':
                    funcs.sell_asset_M(session, pair, close_size, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, session.live)
                funcs.repay_asset_M(session, ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'set_stop':
            self.open_real(session, pair, size, stp, ph['inval'], direction, 3)

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
        direction = ph.get('direction')

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
        direction = ph.get('direction')

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
                if ph['action'] == 'open':
                    self.repair_open(session, ph)
                elif ph['action'] == 'tp':
                    self.repair_tp(session, ph)
                elif ph['action'] == 'close':
                    self.repair_close(session, ph)
                elif ph['action'] == 'move_stop':
                    self.repair_move_stop(session, ph)
            except bx.BinanceAPIException as e:
                print("problem during repair_trade_records")
                pprint(ph)
                self.record_trades(session, all)
                print(e.status_code)
                print(e.message)
        k1.stop()


# indicator specs:
        # {'indicator': 'ema', 'length': lb, 'nans': 0}
        # {'indicator': 'hma', 'length': lb, 'nans': lb-4}
        # {'indicator': 'ema_ratio', 'length': lb, 'nans': 0}
        # {'indicator': 'vol_delta', 'length': 1, 'nans': 0}
        # {'indicator': 'vol_delta_div', 'length': 2, 'nans': 1}
        # {'indicator': 'atr', 'length': lb, 'lookback': lb, 'multiplier': 3, 'nans': 0}
        # {'indicator': 'supertrend', 'lookback': lb, 'multiplier': mult, 'length': lb*mult, 'nans': 1}
        # {'indicator': 'atsz', 'length': lb, 'nans': 1}
        # {'indicator': 'rsi', 'length': lb+1, 'nans': lb}
        # {'indicator': 'stoch_rsi', 'lookback': lb1, 'stoch_lookback': lb2, 'length': lb1+lb2+1, 'nans': lb1+lb2}
        # {'indicator': 'inside', 'length': 2, 'nans': 1}
        # {'indicator': 'doji', 'length': 1, 'nans': 0}
        # {'indicator': 'engulfing', 'length': lb+1, 'nans': lb}
        # {'indicator': 'bull_bear_bar', 'length': 1, 'nans': 0}
        # {'indicator': 'roc_1d', 'length': 2, 'nans': 1}
        # {'indicator': 'roc_1w', 'length': 2, 'nans': 1}
        # {'indicator': 'roc_1m', 'length': 2, 'nans': 1}
        # {'indicator': 'vwma', 'length': lb+1, 'nans': lb}
        # {'indicator': 'cross_age', 'series_1': s1, 'series_2': s2, 'length': lb, 'nans': 0}


class DoubleST(Agent):
    '''200EMA and regular supertrend for bias with tight supertrend for entries/exits'''

    def __init__(self, session, tf, offset, mult1: int, mult2: float):
        t = Timer('DoubleST init')
        t.start()
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.mult1 = int(mult1)
        self.mult2 = float(mult2)
        self.signal_age = 1
        self.name = f'{self.tf} dst {self.mult1}-{self.mult2}'
        self.id = f"double_st_{self.tf}_{self.offset}_{self.mult1}_{self.mult2}"
        self.ohlc_length = 200 + self.signal_age
        self.cross_age_name = f"cross_age-st-10-{int(self.mult1 * 10)}-10-{int(self.mult2 * 10)}"
        self.trail_stop = False
        Agent.__init__(self, session)
        session.indicators.update(['ema-200', f"st-10-{self.mult1}", f"st-10-{self.mult2}", self.cross_age_name])
        # st_lb = 10
        # session.indicators.update([
        #     {'indicator': 'ema', 'length': 200, 'nans': 0},
        #     {'indicator': 'supertrend',
        #      'lookback': st_lb,
        #      'multiplier': self.mult1,
        #      'length': math.ceil(st_lb * self.mult1),
        #      'nans': 1},
        #     {'indicator': 'supertrend',
        #      'lookback': st_lb,
        #      'multiplier': self.mult2,
        #      'length': math.ceil(st_lb * self.mult2),
        #      'nans': 1},
        #     {'indicator': 'cross_age',
        #      'series_1': f"st-{st_lb}-{self.mult1}",
        #      'series_2': f"st-{st_lb}-{self.mult2}",
        #      'length': math.ceil(st_lb * self.mult2),
        #      'nans': 0}
        # ])
        t.stop()

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
            return None

        price = df.close.iloc[-1]
        trend_established = df[self.cross_age_name].iloc[-1] >= self.signal_age

        bullish_ema = price > df.ema_200.iloc[-1]
        bearish_ema = price < df.ema_200.iloc[-1]
        bullish_loose = (price > df[f'st-10-{float(self.mult1)}'].iloc[-1])
        bearish_loose = (price < df[f'st-10-{float(self.mult1)}'].iloc[-1])
        bullish_tight = (price > df[f'st-10-{float(self.mult2)}'].iloc[-1])
        bearish_tight = (price < df[f'st-10-{float(self.mult2)}'].iloc[-1])

        if bullish_ema and bullish_loose and bullish_tight and trend_established:  # and bullish_book
            bias = 'bullish'
        elif bearish_ema and bearish_loose and bearish_tight and trend_established:  # and bearish_book
            bias = 'bearish'
        elif bearish_tight:
            bias = 'neutral'
        elif bullish_tight:
            bias = 'neutral'
        else:
            bias = None

        inval = df[f'st-10-{self.mult2}'].iloc[-1]
        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        inval_ratio = stp / price

        # scores
        # TODO need to look again at calc_inval_risk_score
        inval_score = self.calc_inval_risk_score(abs(1 - inval_ratio))

        k.stop()

        return {
            'pair': pair,
            'agent': self.id,
            'mode': self.mode,
            'tf': self.tf,
            'bias': bias,
            'inval': stp,
            'inval_ratio': inval_ratio,  # might not actually be needed now i have inval_score
            'inval_score': inval_score,
            'trig_price': df.close.iloc[-1],
            'pct_of_full_pos': 1,
        }


class DoubleSTnoEMA(Agent):
    '''regular supertrend for bias with tight supertrend for entries/exits'''

    def __init__(self, session, tf, offset, mult1: int, mult2: float):
        t = Timer('DoubleSTnoEMA init')
        t.start()
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.mult1 = int(mult1)
        self.mult2 = float(mult2)
        self.signal_age = 1
        self.name = f'{self.tf} dst no ema {self.mult1}-{self.mult2}'
        self.id = f"double_st_no_ema_{self.tf}_{self.offset}_{self.mult1}_{self.mult2}"
        self.ohlc_length = 10 + self.signal_age
        self.cross_age_name = f"cross_age-st-10-{int(self.mult1 * 10)}-10-{int(self.mult2 * 10)}"
        self.trail_stop = False
        Agent.__init__(self, session)
        session.indicators.update([f"st-10-{self.mult1}", f"st-10-{self.mult2}", self.cross_age_name])
        # st_lb = 10
        # session.indicators.update([
        #     {'indicator': 'supertrend',
        #      'lookback': st_lb,
        #      'multiplier': self.mult1,
        #      'length': math.ceil(st_lb * self.mult1),
        #      'nans': 1},
        #     {'indicator': 'supertrend',
        #      'lookback': st_lb,
        #      'multiplier': self.mult2,
        #      'length': math.ceil(st_lb * self.mult2),
        #      'nans': 1},
        #     {'indicator': 'cross_age',
        #      'series_1': f"st-{st_lb}-{self.mult1}",
        #      'series_2': f"st-{st_lb}-{self.mult2}",
        #      'length': math.ceil(st_lb * self.mult2),
        #      'nans': 0}
        # ])
        t.stop()

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

    def signals_old(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA"""

        k = Timer(f'dst_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return {'signal': None, 'inval': 0, 'inval_ratio': 100000}

        bullish_loose = self.aged_condition(self.signal_age, df.close, df[f'st-10-{float(self.mult1)}'])
        bearish_loose = self.aged_condition(self.signal_age, df[f'st-10-{float(self.mult1)}'], df.close)
        bullish_tight = self.aged_condition(self.signal_age, df.close, df[f'st-10-{self.mult2}'])
        bearish_tight = self.aged_condition(self.signal_age, df[f'st-10-{self.mult2}'], df.close)


        # bullish_book = bid_ask_ratio > 1
        # bearish_book = bid_ask_ratio < 1
        # bullish_volume = price rising on low volume or price falling on high volume
        # bearish_volume = price rising on high volume or price falling on low volume

        if bullish_loose and bullish_tight:  # and bullish_book
            signal = 'open_long'
        elif bearish_loose and bearish_tight:  # and bearish_book
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

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA"""

        k = Timer(f'dst_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return None

        price = df.close.iloc[-1]
        trend_established = df[self.cross_age_name].iloc[-1] >= self.signal_age

        bullish_loose = price > df[f'st-10-{float(self.mult1)}'].iloc[-1]
        bearish_loose = price < df[f'st-10-{float(self.mult1)}'].iloc[-1]
        bullish_tight = price > df[f'st-10-{float(self.mult2)}'].iloc[-1]
        bearish_tight = price < df[f'st-10-{float(self.mult2)}'].iloc[-1]

        if bullish_loose and bullish_tight and trend_established:  # and bullish_book
            bias = 'bullish'
        elif bearish_loose and bearish_tight and trend_established:  # and bearish_book
            bias = 'bearish'
        elif bearish_tight:
            bias = 'neutral'
        elif bullish_tight:
            bias = 'neutral'
        else:
            bias = None

        inval = df[f'st-10-{self.mult2}'].iloc[-1]
        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        inval_ratio = stp / df.close.iloc[-1]

        # scores
        inval_dist = abs((price - stp) / price)
        # TODO need to look again at calc_inval_risk_score
        inval_score = self.calc_inval_risk_score(inval_dist)

        k.stop()

        return {
            'pair': pair,
            'agent': self.id,
            'mode': self.mode,
            'tf': self.tf,
            'bias': bias,
            'inval': stp,
            'inval_ratio': inval_ratio,  # might not actually be needed now i have inval_score
            'inval_score': inval_score,
            'trig_price': df.close.iloc[-1],
            'pct_of_full_pos': 1,
        }


class EMACross(Agent):
    '''Simple EMA cross strategy with a longer-term EMA to set bias and a 
    trailing stop based on ATR bands'''

    def __init__(self, session, tf, offset, lookback_1, lookback_2, mult):
        t = Timer('EMACross init')
        t.start()
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
        self.trail_stop = True
        Agent.__init__(self, session)
        session.indicators.update(['ema-200',
                                   f"ema-{self.lb1}",
                                   f"ema-{self.lb2}",
                                   self.cross_age_name,
                                   f"atr-10-{self.mult}"])
        # session.indicators.update([
        #     {'indicator': 'ema', 'length': 200, 'nans': 0},
        #     {'indicator': 'ema', 'length': self.lb1, 'nans': 0},
        #     {'indicator': 'ema', 'length': self.lb2, 'nans': 0},
        #     {'indicator': 'atr', 'length': 10, 'multiplier': self.mult},
        #     {'indicator': 'cross_age',
        #      'series_1': f"ema_{self.lb1}",
        #      'series_2': f"ema_{self.lb2}",
        #      'length': 100,
        #      'nans': 0}
        # ])
        t.stop()

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''

        k = Timer('emax_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return None

        price = df.close.iloc[-1]
        trend_established = df[self.cross_age_name].iloc[-1] >= self.signal_age

        fast_ema_str = f"ema_{self.lb1}"
        slow_ema_str = f"ema_{self.lb2}"
        bias_ema_str = "ema_200"

        bullish_bias = price > df[bias_ema_str].iloc[-1]
        bearish_bias = price < df[bias_ema_str].iloc[-1]

        bullish_emas = (df[fast_ema_str].iloc[-1] > df[slow_ema_str].iloc[-1])
        bearish_emas = (df[fast_ema_str].iloc[-1] < df[slow_ema_str].iloc[-1])

        atr_lower_below = price > df[f'atr-10-{self.mult}-lower'].iloc[-1]
        atr_upper_above = price < df[f'atr-10-{self.mult}-upper'].iloc[-1]

        if bullish_bias and bullish_emas and atr_lower_below and trend_established:
            bias = 'bullish'
            inval = df[f'atr-10-{self.mult}-lower'].iloc[-1]
        elif bearish_bias and bearish_emas and atr_upper_above and trend_established:
            bias = 'bearish'
            inval = df[f'atr-10-{self.mult}-upper'].iloc[-1]
        elif bearish_emas:
            bias = 'neutral'
            inval = df[f'atr-10-{self.mult}-lower'].iloc[-1]
        elif bullish_emas:
            bias = 'neutral'
            inval = df[f'atr-10-{self.mult}-upper'].iloc[-1]
        else:
            bias = None

        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        inval_ratio = stp / price

        # scores
        inval_dist = abs((price - stp) / price)
        # TODO need to look again at calc_inval_risk_score
        inval_score = self.calc_inval_risk_score(inval_dist)

        k.stop()

        return {
            'pair': pair,
            'agent': self.id,
            'mode': self.mode,
            'tf': self.tf,
            'bias': bias,
            'inval': stp,
            'inval_ratio': inval_ratio, # might not actually be needed now i have inval_score
            'inval_score': inval_score,
            'trig_price': df.close.iloc[-1],
            'pct_of_full_pos': 1,
        }


class EMACrossHMA(Agent):
    """Simple EMA cross strategy with a longer-term HMA to set bias more
    responsively and a trailing stop based on ATR bands"""

    def __init__(self, session, tf, offset, lookback_1, lookback_2, mult):
        t = Timer('EMACrossHMA init')
        t.start()
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
        self.trail_stop = True
        Agent.__init__(self, session)
        session.indicators.update(['hma-200',
                                   f"ema-{self.lb1}",
                                   f"ema-{self.lb2}",
                                   self.cross_age_name,
                                   f"atr-10-{self.mult}"])
        # session.indicators.update([
        #     {'indicator': 'hma', 'length': 200, 'nans': 196},
        #     {'indicator': 'ema', 'length': self.lb1, 'nans': 0},
        #     {'indicator': 'ema', 'length': self.lb2, 'nans': 0},
        #     {'indicator': 'atr', 'length': 10, 'multiplier': self.mult},
        #     {'indicator': 'cross_age',
        #      'series_1': f"ema_{self.lb1}",
        #      'series_2': f"ema_{self.lb2}",
        #      'length': 100,
        #      'nans': 0}
        # ])
        t.stop()

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''

        k = Timer('emaxhma_margin_signals')
        k.start()

        if not session.pairs_data[pair]['margin_allowed']:
            k.stop()
            return None

        price = df.close.iloc[-1]
        trend_established = df[self.cross_age_name].iloc[-1] >= self.signal_age

        fast_ema_str = f"ema_{self.lb1}"
        slow_ema_str = f"ema_{self.lb2}"
        bias_hma_str = "hma_200"

        bullish_bias = price > df[bias_hma_str].iloc[-1]
        bearish_bias = price < df[bias_hma_str].iloc[-1]

        bullish_emas = df[fast_ema_str].iloc[-1] > df[slow_ema_str].iloc[-1]
        bearish_emas = df[fast_ema_str].iloc[-1] < df[slow_ema_str].iloc[-1]

        lower = f'atr-10-{self.mult}-lower'
        upper = f'atr-10-{self.mult}-upper'
        atr_lower_below = price > df[lower].iloc[-1]
        atr_upper_above = price < df[upper].iloc[-1]

        if bullish_bias and bullish_emas and atr_lower_below and trend_established:
            bias = 'bullish'
            inval = df[f'atr-10-{self.mult}-lower'].iloc[-1]
        elif bearish_bias and bearish_emas and atr_upper_above and trend_established:
            bias = 'bearish'
            inval = df[f'atr-10-{self.mult}-upper'].iloc[-1]
        elif bearish_emas:
            bias = 'neutral'
            inval = df[f'atr-10-{self.mult}-lower'].iloc[-1]
        elif bullish_emas:
            bias = 'neutral'
            inval = df[f'atr-10-{self.mult}-upper'].iloc[-1]
        else:
            bias = None

        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        inval_ratio = stp / price

        # scores
        inval_dist = abs((price - stp) / price)
        # TODO need to look again at calc_inval_risk_score
        inval_score = self.calc_inval_risk_score(inval_dist)

        k.stop()

        return {
            'pair': pair,
            'agent': self.id,
            'mode': self.mode,
            'tf': self.tf,
            'bias': bias,
            'inval': stp,
            'inval_ratio': inval_ratio, # might not actually be needed now i have inval_score
            'inval_score': inval_score,
            'trig_price': df.close.iloc[-1],
            'pct_of_full_pos': 1,
        }


class TrailFractals(Agent):
    """Machine learning strategy based around williams fractals trailing stops"""

    # TODO think about how to use the confidence score (and other scores)
    # TODO if i'm only going to use machine learning strats from this point forward, it would be worth  making a
    #  session.valid_pairs set just like the session.features set so i'm not going through hundreds of pairs for no
    #  reason, but make sure things like spreads still get recorded for every pair (maybe it's time to move that to
    #  update_ohlc or something)
    # TODO need to make sure that model info gets recorded in the logs at the end of the session

    def __init__(self, session, tf: str, offset: int, min_conf: float=0.75) -> None:
        t = Timer('TrailFractals init')
        t.start()
        self.mode = 'margin'
        self.tf = tf
        self.offset = offset
        self.min_confidence = min_conf
        self.load_data(session, tf)
        self.name = f'{self.tf} trail_fractals {self.width}-{self.spacing}'
        self.id = f"trail_fractals_{self.tf}_{self.offset}_{self.width}_{self.spacing}"
        self.ohlc_length = 201
        self.trail_stop = True
        self.notes = ''
        Agent.__init__(self, session)
        session.features[tf].update(self.features)
        t.stop()

    def load_data(self, session, tf):
        # paths
        folder = Path("/home/ross/coding/modular_trader/machine_learning/models/trail_fractals")
        long_model_path = folder / f"trail_fractal_long_{self.tf}_model.sav"
        short_model_path = folder / f"trail_fractal_short_{self.tf}_model.sav"
        long_info_path = folder / f"trail_fractal_long_{self.tf}_info.json"
        short_info_path = folder / f"trail_fractal_short_{self.tf}_info.json"

        self.long_model = joblib.load(long_model_path)
        self.short_model = joblib.load(short_model_path)
        with open(long_info_path, 'r') as ip:
            self.long_info = json.load(ip)
        with open(short_info_path, 'r') as ip:
            self.short_info = json.load(ip)

        self.pairs = self.long_info['pairs']
        self.features = set(self.long_info['features'] + self.short_info['features'])
        session.features[tf].update(self.features)
        self.width = self.long_info['frac_width']
        self.spacing = self.long_info['atr_spacing']

    def signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        """generates spot buy signals based on the ats_z indicator. does not account for currently open positions,
        just generates signals as the strategy dictates"""

        sig = Timer('ats_spot_signals')
        sig.start()

        if pair not in self.pairs:
            return None

        signal_dict = {'agent': self.id, 'mode': self.mode, 'pair': pair}

        df = ind.williams_fractals(df, self.width, self.spacing)

        # calculate % from invalidation
        df['long_r_pct'] = abs(df.close - df.frac_low) / df.close
        df['short_r_pct'] = abs(df.close - df.frac_high) / df.close

        # Long model
        df['r_pct'] = df.long_r_pct
        long_features = df[self.long_info['features']].iloc[-1]
        # print(f"{self.name} {self.tf} long inputs:")
        # print(long_features)
        long_X = pd.DataFrame(long_features).transpose()
        long_confidence = self.long_model.predict_proba(long_X)[0, 1]

        # Short model
        df = df.drop('long_r_pct', axis=1)
        df['r_pct'] = df.short_r_pct
        short_features = df[self.short_info['features']].iloc[-1]
        # print(f"{self.name} {self.tf} short inputs:")
        # print(short_features)
        short_X = pd.DataFrame(short_features).transpose()
        short_confidence = self.short_model.predict_proba(short_X)[0, 1]

        # print(f"{self.name} {pair} {self.tf} long conf: {long_confidence:.1%} short conf: {short_confidence:.1%}")

        price = df.close.iloc[-1]

        combined_long = long_confidence - short_confidence
        combined_short = short_confidence - long_confidence

        if (price > df.frac_low.iloc[-1]) and (combined_long > 0):
            signal_dict['confidence'] = combined_long
            signal_dict['bias'] = 'bullish'
            inval = df.frac_low.iloc[-1]
            note = f"{self.name} Long {self.tf} {pair} @ {df.close.iloc[-1]} confidence: {long_confidence - short_confidence:.1%}\n"
            # print(note)
            self.notes += note
        elif (price < df.frac_high.iloc[-1]) and (combined_short > 0):
            signal_dict['confidence'] = combined_short
            signal_dict['bias'] = 'bearish'
            inval = df.frac_high.iloc[-1]
            note = f"{self.name} Short {self.tf} {pair} @ {df.close.iloc[-1]} confidence: {short_confidence - long_confidence:.1%}\n"
            # print(note)
            self.notes += note
        else:
            return None


        stp = self.calc_stop(inval, session.pairs_data[pair]['spread'], price)
        signal_dict['inval'] = stp
        signal_dict['inval_ratio'] = stp / price
        signal_dict['inval_score'] = self.calc_inval_risk_score(abs((price - stp) / price))
        signal_dict['trig_price'] = price
        signal_dict['pct_of_full_pos'] = 1
        signal_dict['tf'] = self.tf
        signal_dict['asset'] = pair[:-len(session.quote_asset)]

        sig.stop()

        return signal_dict


import binance.exceptions as bx
import pandas as pd
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

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12

class Agent():
    '''generic agent class for each strategy to inherit from'''
    
    realised_pnl_long = 0
    realised_pnl_short = 0
    sim_pnl_long = 0
    sim_pnl_short = 0
    indiv_r_limit = 1.2
    total_r_limit = 20
    target_risk = 0.1
    max_pos = 20
    
    # presets = {1: {'timeframe': '4h', 'tf_offset': None}}
    
    def __init__(self, session):
        t = Timer('agent init')
        t.start()
        self.live = session.live
        self.bal = session.bal
        self.fr_max = session.fr_max
        self.prices = session.prices
        self.lookback_limit = session.max_length - 1
        self.market_data = self.mkt_data_path()
        self.counts_dict = {'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0, 'real_close_long': 0, 
                           'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0, 'sim_close_long': 0, 
                           'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0, 'real_close_short': 0, 
                           'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0, 'sim_close_short': 0, 
                           'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 'too_much_or': 0, 'asset_pos_limit': 0,
                           'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0, 'reduce_risk': 0}
        if not self.live:
            self.sync_test_records()
        self.open_trades = self.read_open_trade_records('open')
        self.sim_trades = self.read_open_trade_records('sim')
        self.tracked_trades = self.read_open_trade_records('tracked')
        self.closed_trades = self.read_closed_trade_records()
        self.closed_sim_trades = self.read_closed_sim_trade_records()
        self.backup_trade_records()
        if not self.live:
            session.market_data = Path('/home/ross/Documents/backtester_2021/test_records')
        self.repair_trade_records(session, self)
        self.record_stopped_trades(session)
        self.record_stopped_sim_trades(session)
        self.real_pos = self.current_positions('open')
        self.sim_pos = self.current_positions('sim')
        self.tracked = self.current_positions('tracked')
        self.open_pnl_changes = {}
        self.fixed_risk_l = self.set_fixed_risk('long')
        self.fixed_risk_s = self.set_fixed_risk('short')
        self.test_fixed_risk(0.0002, 0.0002)
        self.max_positions = self.set_max_pos()
        self.max_init_r_l = self.fixed_risk_l * self.total_r_limit
        self.max_init_r_s = self.fixed_risk_s * self.total_r_limit
        self.fixed_risk_dol_l = self.fixed_risk_l * self.bal
        self.fixed_risk_dol_s = self.fixed_risk_s * self.bal
        self.next_id = int(datetime.now().timestamp())
        t.stop()
        
    def __str__(self):
        return self.id
    
    def sync_test_records(self) -> None:
        '''takes the trade records from the raspberry pi and saves them over 
        the local trade records. only runs when not live'''
        
        q = Timer('sync_test_records')
        q.start()
        folder = Path(f"{self.market_data}/{self.id}")
        test_folder = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.id}')
        if not test_folder.exists():
            test_folder.mkdir(parents=True)
        bal_path = Path(folder / 'bal_history.txt')
        test_bal = Path(test_folder / 'bal_history.txt')
        liq_path = Path(f'{self.market_data}/binance_liquidity_history.txt')
        if bal_path.exists():
            with open(bal_path, "r") as file:
                bal_data = file.readlines()
        else:
            bal_data = []
        if not test_bal.exists():
            test_bal.touch()
        if bal_data:
            with open(Path(test_folder / "bal_history.txt"), "w") as file:
                file.writelines(bal_data)
        
        with open(liq_path, 'r') as file:
            book_data = file.readlines()
        if book_data:
            with open('/home/ross/Documents/backtester_2021/test_records/binance_liquidity_history.txt', 'w') as file:
                file.writelines(book_data)
        
        def sync_trades_records(switch):
            w = Timer(f'sync_trades_records-{switch}')
            w.start()
            trades_path = Path(f'{self.market_data}/{self.id}/{switch}_trades.json')
            test_trades = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.id}/{switch}_trades.json')
            if not test_trades.exists():
                print(f'creating {test_trades}')
                test_trades.touch()
            
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
            else:
                if self.live:
                    trades_path.touch()
            w.stop()
        
        sync_trades_records('open')
        sync_trades_records('sim')
        sync_trades_records('tracked')
        sync_trades_records('closed')
        sync_trades_records('closed_sim')
        
        # now that trade records have been loaded, path can be changed
        self.market_data = Path('/home/ross/Documents/backtester_2021/test_records')
        q.stop()

    def mkt_data_path(self) -> Path:
        '''works out what the absolute path should be to access the local market 
        data folder'''
        
        u = Timer('mkt_data_path in agent')
        u.start()
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        market_data = None
        poss_paths = [Path('/media/coding/market_data'), 
                      Path('/mnt/pi_2/market_data')]
        
        for md_path in poss_paths:
            if md_path.exists():
                market_data = md_path
                break
        if not market_data:
            note = 'none of the paths for market_data are available'
            print(note)
        u.stop()
        return market_data
    
    def read_open_trade_records(self, state: str) -> dict:
        '''loads records from open_trades/sim_trades/tracked_trades and returns
        them in a dictionary'''
        
        w = Timer(f'read_open_trade_records-{state}')
        w.start()
        ot_path = Path(f"{self.market_data}/{self.id}")
        if not ot_path.exists():
            ot_path.mkdir(parents=True)
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
    
            
        w.stop()
        return open_trades

    def read_closed_trade_records(self) -> dict:
        '''loads trade records from closed_trades and returns them as a dictionary'''
        
        e = Timer('read_closed_trade_records')
        e.start()
        ct_path = Path(f"{self.market_data}/{self.id}/closed_trades.json")
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

    def read_closed_sim_trade_records(self) -> dict:
        '''loads closed_sim_trades and returns them as a dictionary'''
        
        r = Timer('read_closed_sim_trade_records')
        r.start()
        cs_path = Path(f"{self.market_data}/{self.id}/closed_sim_trades.json")
        if Path(cs_path).exists():
            with open(cs_path, "r") as cs_file:
                try:
                    closed_sim_trades = json.load(cs_file)
                except JSONDecodeError:
                    closed_sim_trades = {}
        
        else:
            closed_sim_trades = {}
            print(f'{cs_path} not found')
        r.stop()
        return closed_sim_trades

    def backup_trade_records(self) -> None:
        '''updates the backup file for each trades dictionary, on the condition 
        that they are not empty'''
        
        y = Timer('backup_trade_records')
        y.start()
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        if self.open_trades:
            with open(f"{self.market_data}/{self.id}/ot_backup.json", "w") as ot_file:
                json.dump(self.open_trades, ot_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'open trades file empty')
        
        if self.sim_trades:
            with open(f"{self.market_data}/{self.id}/st_backup.json", "w") as st_file:
                json.dump(self.sim_trades, st_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'sim trades file empty')
        
        if self.tracked_trades:
            with open(f"{self.market_data}/{self.id}/tr_backup.json", "w") as tr_file:
                json.dump(self.tracked_trades, tr_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'tracked trades file empty')
        
        if self.closed_trades:
            with open(f"{self.market_data}/{self.id}/ct_backup.json", "w") as ct_file:
                json.dump(self.closed_trades, ct_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'closed trades file empty')
        
        if self.closed_sim_trades:
            with open(f"{self.market_data}/{self.id}/cs_backup.json", "w") as cs_file:
                json.dump(self.closed_sim_trades, cs_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'closed sim trades file empty')
        y.stop()
    
    def calc_tor(self) -> None:
        '''collects all the open risk values from real_pos into a list and 
        calculates the sum total of all the open risk for the agent in question'''
        
        u = Timer('calc_tor')
        u.start()
        self.or_list = [float(v.get('or_R')) for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
        u.stop()



    # record stopped trades ------------------------------------------------

    def create_stopped_list(self):
        old_ids = [(pair, v['position']['stop_id'], v['position']['stop_time'])
                   for pair, v in self.open_trades.items()]
        open_orders = client.get_open_margin_orders()
        remaining_ids = [i.get('orderId') for i in open_orders]

        stopped = []
        for pair, sid, time in old_ids:
            if sid not in remaining_ids:
                stopped.append((pair, sid, time))

        return stopped

    def find_order(self, pair, sid):
        if sid == 'not live':
            return None
        order_list = client.get_all_margin_orders(symbol=pair, orderId=sid)

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

    def repay_stop(self, session, pair, order):
        if (order.get('side') == 'BUY'):
            trade_type = 'stop_short'
            asset = pair[:-4]
            stop_size = Decimal(order.get('executedQty'))
            funcs.repay_asset_M(asset, stop_size, session.live)
        else:
            trade_type = 'stop_long'
            stop_size = Decimal(order.get('cummulativeQuoteQty'))
            funcs.repay_asset_M('USDT', stop_size, session.live)

        self.open_trades[pair]['placeholder']['completed'] = 'repay'

        return stop_size

    def create_stop_dict(self, pair, order, stop_size):
        stop_dict = uf.create_stop_dict(order)
        stop_dict['type'] = f"stop_{self.open_trades[pair]['position']['direction']}"
        stop_dict['state'] = 'real'
        stop_dict['reason'] = 'hit hard stop'
        stop_dict['liability'] = uf.update_liability(self.open_trades[pair], stop_size, 'reduce')
        if stop_dict['liability'] not in ['0', '0.0']:
            print(f"+++ WARNING {self.name} {pair} stop hit, liability record doesn't add up. Recorded value: {stop_dict['liability']} +++")

        return stop_dict

    def save_records(self, session, pair, stop_dict):
        self.open_trades[pair]['trade'].append(stop_dict)
        ts_id = int(self.open_trades[pair]['position']['open_time'])
        self.closed_trades[ts_id] = {}
        self.closed_trades[ts_id]['trade'] = self.open_trades[pair]['trade']
        self.record_trades(session, 'closed')
        del self.open_trades[pair]
        self.record_trades(session, 'open')

        return ts_id

    def error_print(self, session, pair, stage, e):
        self.record_trades(session, 'all')
        print(f'{self.name} problem with record_stopped_trades during {pair} {stage}')
        print(f"status code: {e.status_code}")
        print(f"message: {e.message}")

    def rst_iteration(self, session, pair, sid):
        direction = self.open_trades[pair]['position']['direction']
        try:
            order = self.find_order(pair, sid)
        except bx.BinanceAPIException as e:
            self.error_print(session, pair, 'find_order', e)

        if order:
            stop_size = self.repay_stop(pair, order)
            stop_dict = self.create_stop_dict(pair, order, stop_size)
            ts_id = self.save_records(session, pair, stop_dict)

            self.realised_pnl(self.closed_trades[ts_id])
            self.counts_dict[f'real_stop_{direction}'] += 1
        else:
            print(f"record_stopped_trades found no stop for {pair} {self.name}")
            free_bal = session.bals_dict[pair[:-4]].get('free')
            pos_size = self.open_trades[pair]['position']['base_size']
            if Decimal(free_bal) >= Decimal(pos_size):
                print(f'{self.name} record_stopped_trades will close {pair} now')
                try:
                    self.close_real_full(session, self, pair, direction)
                except bx.BinanceAPIException as e:
                    self.error_print(session, pair, 'close', e)
            else:
                price = session.prices[pair]
                value = free_bal * price
                if value > 10:
                    note = f'{pair} in position with no stop-loss'
                    pb.push_note(session.now_start, note)

    def record_stopped_trades(self, session) -> None:
        m = Timer('record_stopped_trades')
        m.start()

        stopped = self.create_stopped_list()

        for pair, sid, time in stopped:
            try:
                self.rst_iteration(session, pair, sid)
            except bx.BinanceAPIException as e:
                self.record_trades(session, 'all')
                print(f'{self.name} problem with record_stopped_trades during {pair}')
                print(e)
        m.stop()

    # record stopped sim trades ----------------------------------------------

    def calc_size_and_stop(self, v):
        x = Timer('calc base size and stop')
        x.start()
        # calculate current base size
        base_size = 0

        for i in v['trade']:
            if i.get('type') in ['open_long', 'open_short', 'add_long', 'add_short']:
                base_size += float(i.get('base_size'))
            else:
                base_size -= float(i.get('base_size'))

        # find most recent hard stop
        for i in v['trade'][-1::-1]:
            if i.get('hard_stop'):
                stop = float(i.get('hard_stop'))
                stop_time = int(i.get('timestamp'))
                break

        return base_size, stop, stop_time
        x.stop()

    def get_data(self, session, pair, stop_time):
        timespan = datetime.now().timestamp() - (stop_time / 1000)

        if timespan > 36000:
            z = Timer('read ohlc pickles')
            z.start()
            filepath = Path(f'{session.ohlc_data}/{pair}.pkl')
            if filepath.exists():
                df = pd.read_pickle(filepath)
                if df['timestamp'].dtype == 'Timestamp':
                    df['timestamp'] = df['timestamp'].timestamp()
            else:
                klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, stop_time)
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                        'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
                df = pd.DataFrame(klines, columns=cols)
                df['timestamp'] = df['timestamp'] * 1000000
                df = df.astype(float)
            z.stop()

            # trim df down to just the rows since the last stop was set
            stop_dt = datetime.fromtimestamp(stop_time / 1000)
            df = df.loc[df.timestamp > stop_dt]


        else:
            z = Timer('get_historical_klines')
            z.start()
            klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_5MINUTE, stop_time)
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                    'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
            df = pd.DataFrame(klines, columns=cols)
            df['timestamp'] = df['timestamp'] * 1000000
            df = df.astype(float)
            z.stop()

        df.reset_index(inplace=True)

        return df

    def check_stop_hit(self, df, long_trade, stop):
        stop_hit_time = None
        if long_trade:
            trade_type = 'stop_long'
            ll = df.low.min()
            stopped = ll < stop
            overshoot_pct = round((100 * (stop - ll) / stop), 3)  # % distance that price broke through the stop
            if stopped:
                for i in range(len(df)):
                    if df.at[i, 'low'] <= stop:
                        stop_hit_time = df.at[i, 'timestamp']
                        if isinstance(stop_hit_time, pd.Timestamp):
                            stop_hit_time = stop_hit_time.timestamp()
        else:
            trade_type = 'stop_short'
            hh = df.high.max()
            stopped = hh > stop
            overshoot_pct = round((100 * (hh - stop) / stop), 3)  # % distance that price broke through the stop
            if stopped:
                for i in range(len(df)):
                    if df.at[i, 'high'] >= stop:
                        stop_hit_time = df.at[i, 'timestamp']
                        if isinstance(stop_hit_time, pd.Timestamp):
                            stop_hit_time = stop_hit_time.timestamp()

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

    def sim_to_closed_sim(self, session, pair, v, trade_dict):
        v['trade'].append(trade_dict)

        ts_id = int(v['trade'][0].get('timestamp'))
        self.closed_sim_trades[ts_id] = {}
        self.closed_sim_trades[ts_id]['trade'] = v['trade']
        self.record_trades(session, 'closed_sim')

    def record_stopped_sim_trades(self, session) -> None:
        """goes through all trades in the sim_trades file and checks their recent
                price action against their most recent hard_stop to see if any of them would have
                got stopped out"""

        n = Timer('record_stopped_sim_trades')
        n.start()

        del_pairs = []
        for pair, v in self.sim_trades.items():

            print(pair)
            print(v)
            print('')

            base_size, stop, stop_time = self.calc_size_and_stop(v)
            df = self.get_data(session, pair, stop_time)
            long_trade = 'long' in v['trade'][0].get('type')
            stopped, trade_type, overshoot_pct, stop_hit_time = self.check_stop_hit(df, long_trade, stop)
            if stopped:
                trade_dict = self.create_trade_dict(pair, trade_type, stop, base_size, stop_hit_time, overshoot_pct)
                self.sim_to_closed_sim(session, pair, v, trade_dict)

                direction = v['trade'][0].get('type')[5:]
                self.realised_pnl(v)
                self.counts_dict[f'sim_stop_{direction}'] += 1

                del_pairs.append(pair)

        for p in del_pairs:
            del self.sim_trades[p]
        n.stop()

    # ---------------------------------------------------------------------------

    def realised_pnl(self, trade_record: List[dict]) -> None:
        '''calculates realised pnl of a tp or close denominated in the trades 
        own R value'''

        side = trade_record['position']['direction']
        
        i = Timer(f'realised_pnl {side}')
        i.start()
        trades = trade_record['trade']
        entry = float(trade_record['position']['entry_price'])
        init_stop = float(trade_record['position']['init_hard_stop'])
        init_size = float(trade_record['position']['init_base_size'])
        final_exit = float(trades[-1].get('exe_price'))
        final_size = float(trades[-1].get('base_size'))
        r_val = abs((entry - init_stop) / entry)
        if side == 'long':
            trade_pnl = (final_exit - entry) / entry
        else:
            trade_pnl = (entry - final_exit) / entry
        trade_r = round(trade_pnl / r_val, 3)
        scalar = final_size / init_size
        realised_r = trade_r * scalar
        
        if trade_record['position'].get('state') == 'real':
            if side == 'long':
                self.realised_pnl_long += realised_r
            else:
                self.realised_pnl_short += realised_r
        elif trade_record['position'].get('state') == 'sim':
            if side == 'long':
                self.sim_pnl_long += realised_r
            else:
                self.sim_pnl_short += realised_r
        else:
            print(f"state in record: {trade_record['position'].get('state')}")
            print(f'{trade_r = }')
        i.stop()
    
    def record_trades(self, session, state: str) -> None:
        '''saves any trades dictionary to its respective json file'''
        
        b = Timer(f'record_trades {state}')
        b.start()
        filepath = Path(f"{session.market_data}/{self.id}/{state}_trades.json")
        if not filepath.exists():
            filepath.touch()
        with open(filepath, "w") as file:
            if state in ['open', 'all']:
                json.dump(self.open_trades, file)
            if state in ['sim', 'all']:
                json.dump(self.sim_trades, file)
            if state in ['tracked', 'all']:
                json.dump(self.tracked_trades, file)
            if state in ['closed', 'all']:
                json.dump(self.closed_trades, file)
            if state in ['closed_sim', 'all']:
                json.dump(self.closed_sim_trades, file)
        b.stop()
    
    def set_fixed_risk(self, direction:str) -> None:
        '''calculates fixed risk setting for new trades based on recent performance 
        and previous setting. if recent performance is very good, fr is increased slightly.
        if not, fr is decreased by thirds'''
        
        o = Timer(f'set_fixed_risk-{direction}')
        o.start()
        
        def reduce_fr(factor: float, fr_prev: float, fr_inc: float):
            '''reduces fixed_risk by factor (with the floor value being 0)'''
            ideal = fr_prev * factor
            reduce = max(ideal, fr_inc)
            return max((fr_prev-reduce), 0)
        
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        
        filepath = Path(f"{self.market_data}/{self.id}/bal_history.txt")
        if self.live:
            filepath.touch(exist_ok=True)
        with open(filepath, "r") as file:
            bal_data = file.readlines()
        
        if bal_data:
            fr_prev = json.loads(bal_data[-1]).get(f'fr_{direction}', 0)
        else:
            fr_prev = 0
        fr_inc = self.fr_max / 10 # increment fr in 10% steps of the range
        
        def score_accum(direction:str, switch:str):
            '''calculates perf score from recent performance. also saves the
            instance property open_pnl_changes dictionary'''
            
            with open(f"{self.market_data}/{self.id}/bal_history.txt", "r") as file:
                bal_data = file.readlines()
            
            if bal_data:
                last = json.loads(bal_data[-1])
            if bal_data and last.get(f'{switch}_open_pnl_{direction[0]}'):
                prev_open_pnl = last.get(f'{switch}_open_pnl_{direction[0]}')
                curr_open_pnl = self.open_pnl(direction, switch)
                pnl_change_pct = 100 * (curr_open_pnl - prev_open_pnl) / prev_open_pnl
                self.open_pnl_changes[switch] = pnl_change_pct
            elif bal_data:
                prev_bal = last.get('balance')
                pnl_change_pct = 100 * (self.bal - prev_bal) / prev_bal
            else:
                pnl_change_pct = 0
            
            lookup = f'realised_pnl_{direction}' if switch == 'real' else f'sim_r_pnl_{direction}'
            pnls = {}
            for i in range(1, 5):
                if bal_data and len(bal_data) > 5:
                    pnls[i] = json.loads(bal_data[-1*i]).get(lookup, -1)
                else:
                    pnls[i] = -1 # if there's no data yet, return -1 instead
            
            score_1 = 0
            score_2 = 0
            if pnl_change_pct > 0.1:
                score_1 += 5
            elif pnl_change_pct < -0.1:
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
        
        # real_score_1, real_score_2, real_pnls = score_accum(direction, 'real')
        sim_score_1, sim_score_2, sim_pnls = score_accum(direction, 'sim')
        if sim_score_1 + sim_score_2 >= 11:
            print(f"set_fixed_risk {direction}: sim_score {sim_score_1 + sim_score_2}")
        score = sim_score_1 + sim_score_2
        pnls = sim_pnls
        
        if score == 15:
            fr = min(fr_prev + (2*fr_inc), self.fr_max)
        elif score >= 11:
            fr = min(fr_prev + fr_inc, self.fr_max)
        elif score >= 3:
            fr = fr_prev
        elif score >= -3:
            fr = reduce_fr(0.333, fr_prev, fr_inc)
        elif score >= -7:
            fr = reduce_fr(0.5, fr_prev, fr_inc)
        else:
            fr = 0
            
        if fr != fr_prev:
            title = f'{now} {self.name}'
            note = f'{direction} fixed risk adjusted from {round(fr_prev*10000, 1)}bps to {round(fr*10000, 1)}bps'
            pb.push_note(title, note)
            print(note)
            print(f"calculated {direction} score: {score}, pnls: {pnls}")
        o.stop()
        return round(fr, 5)
    
    def test_fixed_risk(self, fr_l: float, fr_s: float) -> None:
        '''manually overrides fixed risk settings for testing purposes'''
        if not self.live:
            print(f'*** WARNING: FIXED RISK MANUALLY SET to {fr_l} / {fr_s} ***')
            self.fixed_risk_l = fr_l
            self.fixed_risk_s = fr_s
    
    def set_max_pos(self) -> int:
        '''sets the maximum number of open positions for the agent. if the median 
        pnl of current open positions is greater than 0, max pos will be set to 50, 
        otherwise max_pos will be set to 20'''
        
        p = Timer('set_max_pos')
        p.start()
        max_pos = 10
        if self.real_pos:
            open_pnls = [v.get('pnl') for v in self.real_pos.values() if v.get('pnl')]
            if open_pnls:
                avg_open_pnl = stats.median(open_pnls)
            else:
                avg_open_pnl = 0
            max_pos = 10 if avg_open_pnl <= 0 else 20
        p.stop()
        return max_pos
    
    def current_positions(self, state: str) -> dict:
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
        now = datetime.now()
        total_bal = self.bal
        
        for k, v in data.items():
            asset = k[:-4]
            if state == 'tracked':
                size_dict[asset] = {}
            else:
                price = self.prices[k]
                size_dict[asset] = uf.open_trade_stats(now, total_bal, v, price)
        a.stop()
        return size_dict

    def set_in_pos(self, pair: str, state: str) -> None:
        '''fills in in_pos with a trade direction if applicable'''
        
        d = Timer(f'set_in_pos {state}')
        d.start()
        asset = pair[:-4]
        if state == 'real':
            pos_dict = self.real_pos.keys()
            trade_record = self.open_trades.get(pair)
        elif state == 'sim':
            pos_dict = self.sim_pos.keys()
            trade_record = self.sim_trades.get(pair)
        elif state == 'tracked':
            pos_dict = self.tracked.keys()
            trade_record = self.tracked_trades.get(pair)
        
        if asset in pos_dict:
            if trade_record['position']['direction'] == 'long':
                self.in_pos[state] = 'long'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record['trade'], 'long', state)
            else:
                self.in_pos[state] = 'short'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record['trade'], 'short', state)
        d.stop()

    def init_in_pos(self, pair: str) -> None:
        '''initialises the in_pos dictionary and fills it with None values'''
        
        f = Timer(f'init_in_pos')
        f.start()
        self.in_pos = {'real':None, 'sim':None, 'tracked':None, 
                  'real_ep': None, 'sim_ep': None, 'tracked_ep': None, 
                  'real_hs': None, 'sim_hs': None, 'tracked_hs': None, 
                  'real_pfrd': None, 'sim_pfrd': None, 'tracked_pfrd': None}
        
        self.set_in_pos(pair, 'real')
        self.set_in_pos(pair, 'sim')
        self.set_in_pos(pair, 'tracked')
        f.stop()
    
    def too_new(self, df: pd.DataFrame) -> bool:
        '''returns True if there is less than 200 periods of history AND if
        there are no current positions in the asset'''
        
        g = Timer('too_new')
        g.start()
        
        if self.in_pos['real'] or self.in_pos['sim'] or self.in_pos['tracked']:
            no_pos = False
        else:
            no_pos = True    
        
        return len(df) <= self.lookback_limit and no_pos

    def open_pnl(self, direction: str, state: str) -> Union[int, float]:
        '''adds up the pnls of all open positions for a given state'''
        
        h = Timer(f'open_pnl {state}')
        h.start()
        total = 0
        if state == 'real':
            pos_dict = self.real_pos.values()
        elif state == 'sim':
            pos_dict = self.sim_pos.values()
        else:
            print('open_pnl requires argument real or sim')
        
        for pos in pos_dict:
            if pos.get('pnl_R'):
                if (direction == 'long') and pos['long']:
                    total += pos['pnl_R']
                elif (direction == 'short') and not pos['long']:
                    total += pos['pnl_R']
                        
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
        _, base_size = funcs.clear_stop_M(pair, pos_record, session.live)
        if not base_size:
            print('--- running move_stop and needed to use position base size')
            base_size = pos_record['base_size']
        else:
            print("--- running move_stop and DIDN'T need to use position base size")

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
        self.open_trades[pair]['position']['stop_time'] = stop_order.get('transactTime')
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
        low_atr = df.at[len(df) - 1, f'atr-10-{float(self.mult)}-lower']
        high_atr = df.at[len(df) - 1, f'atr-10-{float(self.mult)}-upper']
        atr = low_atr if direction == 'long' else high_atr

        current_stop = float(self.open_trades[pair]['position']['hard_stop'])

        if (direction == 'long' and low_atr > current_stop) or (direction == 'short' and high_atr < current_stop):
            print(f"*** {self.name} {pair} move real {direction} stop from {current_stop:.3} to {atr:.3}")
            try:
                self.move_api_stop(session, pair, direction, atr, self.open_trades[pair]['position'])
            except bx.BinanceAPIException as e:
                self.record_trades(session, 'all')
                print(f'{self.name} problem with move_stop order for {pair}')
                print(e)

    def move_non_real_stop(self, session, pair, df, state, direction):
        if state == 'sim':
            trade_record = self.sim_trades[pair]
        elif state == 'tracked':
            trade_record = self.tracked_trades[pair]

        current_stop = float(trade_record['position']['hard_stop'])

        low_atr = df.at[len(df) - 1, f'atr-10-{float(self.mult)}-lower']
        high_atr = df.at[len(df) - 1, f'atr-10-{float(self.mult)}-upper']
        atr = low_atr if direction == 'long' else high_atr

        move_condition = (direction == 'long' and low_atr > current_stop) or (direction == 'short' and high_atr < current_stop)

        if move_condition:
            print(
                f"{self.name} {pair} {state} move {direction} stop from {trade_record['position']['hard_stop']} to {atr}")
            if state == 'sim':
                self.sim_trades[pair]['position']['hard_stop'] = atr
            elif state == 'tracked':
                self.tracked_trades[pair]['position']['hard_stop'] = atr
            self.record_trades(session, state)

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
                                          (abs(self.in_pos.get('real_price_delta', 0)) > 0.001))
        if self.sim_pos.get(asset):
            sim_or = self.sim_pos.get(asset).get('or_R', 0)
            self.in_pos['sim_tp_sig'] = ((float(sim_or) > self.indiv_r_limit) and 
                                         (abs(self.in_pos.get('sim_price_delta', 0)) > 0.001))
        j.stop()

    # dispatch

    def open_pos(self, session, pair, size, stp, inval, sim_reason, direction):
        if self.in_pos['real'] is None and not sim_reason:
            self.open_real(session, pair, size, stp, inval, direction, 0)

        if self.in_pos['sim'] is None and sim_reason:
            self.open_sim(session, pair, stp, inval, sim_reason, direction)

    def tp_pos(self, session, pair, stp, inval, direction):
        if self.in_pos.get('real_tp_sig'):
            self.tp_real_full(session, pair, stp, inval, direction)

        if self.in_pos.get('sim_tp_sig'):
            self.tp_sim(session, pair, stp, direction)

        if self.in_pos.get('tracked_tp_sig'):
            self.tp_tracked(session, pair, direction)

    def close_pos(self, session, pair, direction):
        if self.in_pos['real'] == direction:
            print('')
            self.close_real_full(session, pair, direction)

        if self.in_pos['sim'] == direction:
            self.close_sim(session, pair, direction)

        if self.in_pos['tracked'] == direction:
            self.close_tracked(session, pair, direction)

    def reduce_risk_M(self, session):
        # create a list of open positions in profit and their open risk value
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

            print(f'\n*** tor: {total_r:.1f}, reducing risk ***')
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

    # real open

    def create_record(self, session, pair, size, stp, inval, direction):
        price = session.prices[pair]
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
                       'inval': inval,
                       'timestamp': now,
                       'completed': None
                       }
        self.open_trades[pair] = {}
        self.open_trades[pair]['placeholder'] = placeholder
        self.open_trades[pair]['position'] = {}
        self.open_trades[pair]['position']['pair'] = pair
        self.open_trades[pair]['position']['direction'] = direction
        self.open_trades[pair]['position']['state'] = 'real'

        note = f"{self.name} real open {direction} {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)

    def omf_borrow(self, session, pair, size, direction):
        if direction == 'long':
            price = session.prices[pair]
            borrow_size = f"{size * price:.2f}"
            funcs.borrow_asset_M('USDT', borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = 'USDT'
        elif direction == 'short':
            asset = pair[:-4]
            borrow_size = funcs.valid_size(session, pair, size)
            funcs.borrow_asset_M(asset, borrow_size, session.live)
            self.open_trades[pair]['placeholder']['loan_asset'] = asset
        else:
            print('*** WARNING open_real_2 given wrong direction argument ***')

        self.open_trades[pair]['position']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['liability'] = borrow_size
        self.open_trades[pair]['placeholder']['completed'] = 'borrow'

    def increase_position(self, session, pair, size, direction):
        price = session.prices[pair]
        usdt_size = f"{size * price:.2f}"

        if direction == 'long':
            api_order = funcs.buy_asset_M(session, pair, float(usdt_size), False, price, session.live)
        elif direction == 'short':
            api_order = funcs.sell_asset_M(session, pair, size, price, session.live)

        self.open_trades[pair]['position']['base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['init_base_size'] = str(api_order.get('executedQty'))
        self.open_trades[pair]['position']['open_time'] = str(api_order.get('transactTime'))
        self.open_trades[pair]['placeholder']['api_order'] = api_order
        self.open_trades[pair]['placeholder']['completed'] = 'execute'

        return api_order

    def open_trade_dict(self, session, pair, api_order, stp, direction):
        price = session.prices[pair]

        open_order = funcs.create_trade_dict(api_order, price, session.live)
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
        self.open_trades[pair]['position']['stop_time'] = stop_order.get('transactTime')
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = stop_order.get('transactTime')
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return open_order

    def open_save_records(self, session, pair, open_order):
        self.open_trades[pair]['trade'] = [open_order]
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def open_update_in_pos(self, session, pair, stp, direction):
        price = session.prices[pair]

        self.in_pos['real'] = direction
        if direction == 'long':
            self.in_pos['real_pfrd'] = self.fixed_risk_dol_l
        elif direction == 'short':
            self.in_pos['real_pfrd'] = self.fixed_risk_dol_s
        self.in_pos['real_ep'] = price
        self.in_pos['real_hs'] = stp

    def open_update_real_pos_usdtM_counts(self, session, pair, size, inval, direction):
        price = session.prices[pair]
        usdt_size = f"{size * price:.2f}"
        asset = pair[:-4]

        if session.live:
            self.real_pos[asset] = funcs.update_pos_M(session, asset, size, inval, self.in_pos['real'],
                                                       self.in_pos['real_pfrd'])
            self.real_pos[asset]['pnl_R'] = 0
            if direction == 'long':
                session.update_usdt_M(borrow=float(usdt_size))
            elif direction == 'short':
                session.update_usdt_M(up=float(usdt_size))
        else:
            pf = f"{float(usdt_size) / session.bal:.2f}"
            if direction == 'long':
                or_dol = f"{session.bal * self.fixed_risk_l:.2f}"
            elif direction == 'short':
                or_dol = f"{session.bal * self.fixed_risk_s:.2f}"
            self.real_pos[asset] = {'qty': str(size), 'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol)}

        self.counts_dict[f'real_open_{direction}'] += 1

    def open_real(self, session, pair, size, stp, inval, direction, stage):
        if stage == 0:
            print('')
            self.create_record(session, pair, size, stp, inval, direction)
            self.omf_borrow(session, pair, size, direction)
            api_order = self.increase_position(session, pair, size, direction)
        if stage <= 1:
            open_order = self.open_trade_dict(session, pair, api_order, stp, direction)
        if stage <= 2:
            open_order = self.open_set_stop(session, pair, stp, open_order, direction)
        if stage <= 3:
            self.open_save_records(session, pair, open_order)
            self.open_update_in_pos(session, pair, stp, direction)
            self.open_update_real_pos_usdtM_counts(session, pair, size, inval, direction)

    # real tp

    def create_tp_placeholder(self, session, pair, stp, inval, direction):
        price = session.prices[pair]
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # insert placeholder record
        placeholder = {'type': f'tp_{direction}',
                       'state': 'real',
                       'pair': pair,
                       'trig_price': price,
                       'stop_price': stp,
                       'inval': inval,
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
        print(f'{self.name} clearing {pair} stop')
        clear, cleared_size = funcs.clear_stop_M(pair, self.open_trades[pair]['position'], session.live)
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])
        self.check_size_against_records(pair, real_bal, cleared_size)

        # update position and placeholder
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

        return cleared_size

    def tp_reduce_position(self, session, pair, base_size, pct, direction):
        price = session.prices[pair]
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
        price = session.prices[pair]
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
        self.open_trades[pair]['trade'].append(close_order)

        self.realised_pnl(self.open_trades[pair])
        del self.open_trades[pair]['placeholder']
        self.tracked_trades[pair] = self.open_trades[pair]
        self.open_trades[pair]['position']['state'] = 'tracked'
        self.record_trades(session, 'tracked')

        del self.open_trades[pair]
        self.record_trades(session, 'open')

    def tp_update_records_100(self, session, pair, order_size, usdt_size, direction):
        asset = pair[:-4]
        price = session.prices[pair]

        self.in_pos['real'] = None
        self.in_pos['tracked'] = direction

        self.tracked[asset] = {'qty': '0', 'value': '0', 'pf%': '0', 'or_R': '0', 'or_$': '0'}

        if session.live and direction == 'long':
            session.update_usdt_M(repay=float(usdt_size))
        elif session.live and direction == 'short':
            usdt_size = round(order_size * price, 5)
            session.update_usdt_M(down=usdt_size)
        elif (not session.live) and direction == 'long':
            self.real_pos['USDT']['qty'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] += float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] += float(self.real_pos[asset].get('pf%'))
        elif (not session.live) and direction == 'short':
            self.real_pos['USDT']['qty'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['value'] -= float(self.real_pos[asset].get('value'))
            self.real_pos['USDT']['pf%'] -= float(self.real_pos[asset].get('pf%'))

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
        self.open_trades[pair]['position']['stop_time'] = stop_order.get('transactTime')
        self.open_trades[pair]['placeholder']['hard_stop'] = str(stp)
        self.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
        self.open_trades[pair]['placeholder']['stop_time'] = stop_order.get('transactTime')
        self.open_trades[pair]['placeholder']['completed'] = 'set_stop'

        return tp_order

    def open_to_open(self, session, pair, tp_order, direction):
        self.open_trades[pair]['trade'].append(tp_order)
        self.realised_pnl(self.open_trades[pair])
        print('\ntp partial placeholder')
        pprint(self.open_trades[pair]['placeholder'])
        del self.open_trades[pair]['placeholder']
        self.record_trades(session, 'open')

    def tp_update_records_partial(self, session, pair, pct, inval, order_size, tp_order, direction):
        asset = pair[:-4]
        price = session.prices[pair]
        new_size = self.open_trades[pair]['position']['base_size']

        self.in_pos['real_pfrd'] = self.in_pos['real_pfrd'] * (pct / 100)
        if session.live:
            self.real_pos[asset].update(
                funcs.update_pos_M(session, asset, new_size, inval, self.in_pos['real'],
                                   self.in_pos['real_pfrd']))
            if direction == 'long':
                repay_size = tp_order.get('base_size')
                session.update_usdt_M(repay=float(repay_size))
            elif direction == 'short':
                usdt_size = round(order_size * price, 5)
                session.update_usdt_M(down=usdt_size)
        else:
            self.calc_sizing_non_live_tp(session, asset, pct, 'real')

        self.counts_dict[f'real_tp_{direction}'] += 1

    def tp_real_full(self, session, pair, stp, inval, direction):
        price = session.prices[pair]
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        print('')

        self.create_tp_placeholder(session, pair, stp, inval, direction)
        pct = self.tp_set_pct(pair)
        # clear stop
        cleared_size = self.tp_clear_stop(session, pair)

        if not cleared_size:
            print(f'{self.name} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
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
            self.open_to_open(session, pair, tp_order, direction)
            self.tp_update_records_partial(session, pair, pct, inval, cleared_size, tp_order, direction)

    # real close

    def create_close_placeholder(self, session, pair, direction):
        price = session.prices[pair]
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
        print(f'{self.name} clearing {pair} stop')
        clear, cleared_size = funcs.clear_stop_M(pair, self.open_trades[pair]['position'], session.live)
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])
        self.check_size_against_records(pair, real_bal, cleared_size)

        # update position and placeholder
        self.open_trades[pair]['position']['hard_stop'] = None
        self.open_trades[pair]['position']['stop_id'] = None
        self.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
        self.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

        return cleared_size

    def close_position(self, session, pair, close_size, reason, direction):
        price = session.prices[pair]

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

    def open_to_closed(self, session, pair, close_order, direction):
        self.open_trades[pair]['trade'].append(close_order)
        self.realised_pnl(self.open_trades[pair])

        trade_id = int(self.open_trades[pair]['position']['open_time'])
        self.closed_trades[trade_id] = {}
        self.closed_trades[trade_id]['trade'] = self.open_trades[pair]['trade']
        self.record_trades(session, 'closed')

        del self.open_trades[pair]
        self.record_trades(session, 'open')

    def close_real_7(self, session, pair, close_size, direction):
        asset = pair[:-4]
        price = session.prices[pair]

        self.in_pos['real'] = None
        self.in_pos['real_pfrd'] = 0

        if direction == 'long' and session.live:
            session.update_usdt_M(repay=float(close_size))
        elif direction == 'short' and session.live:
            usdt_size = round(float(close_size) * price, 5)
            session.update_usdt_M(down=usdt_size)
        elif direction == 'long' and not session.live:
            self.real_pos['USDT']['value'] += float(close_size)
            self.real_pos['USDT']['owed'] -= float(close_size)
        elif direction == 'short' and not session.live:
            self.real_pos['USDT']['value'] += (float(close_size) * price)
            self.real_pos['USDT']['owed'] -= (float(close_size) * price)

        # save records and update counts
        del self.real_pos[asset]
        self.counts_dict[f'real_close_{direction}'] += 1

    def close_real_full(self, session, pair, direction, stage=0):
        price = session.prices[pair]
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        note = f"{self.name} real close {direction} {pair} @ {price}"
        print(now, note)

        if stage == 0:
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
            close_order = self.close_position(session, pair, cleared_size, 'close_signal', direction)
        if stage <= 3:
            # repay loan
            repay_size = self.close_repay(session, pair, close_order, direction)
        if stage <= 4:
            # update records
            self.open_to_closed(session, pair, close_order, direction)
            # update in-pos, real_pos, counts etc
            self.close_real_7(session, pair, repay_size, direction)

    # sim
    def open_sim(self, session, pair, stp, inval, sim_reason, direction):
        asset = pair[:-4]
        price = session.prices[pair]

        usdt_size = 128.0
        size = f"{usdt_size / price:.8f}"
        # if not session.live:
        #     now = datetime.now().strftime('%d/%m/%y %H:%M')
        #     print('')
        #     note = f"{self.name} sim open {direction} {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        #     print(now, note)

        timestamp = round(datetime.utcnow().timestamp() * 1000)
        sim_order = {'pair': pair,
                     'exe_price': str(price),
                     'trig_price': str(price),
                     'base_size': size,
                     'quote_size': '128.0',
                     'hard_stop': str(stp),
                     'reason': sim_reason,
                     'timestamp': timestamp,
                     'type': f'open_{direction}',
                     'fee': '0',
                     'fee_currency': 'BNB',
                     'state': 'sim'}
        trade_record = [sim_order]
        self.sim_trades[pair] = {}
        self.sim_trades[pair]['trade'] = trade_record

        pos_record = {'base_size': size,
                      'direction': direction,
                      'entry_price': str(price),
                      'hard_stop': str(stp),
                      'open_time': timestamp,
                      'pair': pair,
                      'liability': '0',
                      'stop_id': 'not live',
                      'stop_time': None}
        self.sim_trades[pair]['position'] = pos_record
        self.sim_trades[pair]['position']['state'] = 'sim'

        self.record_trades(session, 'sim')

        self.in_pos['sim'] = direction
        if direction == 'long':
            self.in_pos['sim_pfrd'] = self.fixed_risk_dol_l
        else:
            self.in_pos['sim_pfrd'] = self.fixed_risk_dol_s
        self.sim_pos[asset] = funcs.update_pos_M(session, asset, float(size), inval,
                                                  self.in_pos['sim'], self.in_pos['sim_pfrd'])
        self.sim_pos[asset]['pnl_R'] = 0

        self.counts_dict[f'sim_open_{direction}'] += 1

    def tp_sim(self, session, pair, stp, direction):
        # if not session.live:
        #     print('')
        #     note = f"{self.name} sim take-profit {pair} {direction} 50% @ {price}"
        #     print(now, note)

        price = session.prices[pair]
        asset = pair[:-4]

        trade_record = self.sim_trades[pair]['trade']
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)
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
                    'timestamp': timestamp,
                    'type': f'tp_{direction}',
                    'fee': '0',
                    'fee_currency': 'BNB',
                    'state': 'sim'}
        trade_record.append(tp_order)
        self.sim_trades[pair]['trade'] = trade_record

        self.sim_trades[pair]['position']['base_size'] = str(order_size)
        self.sim_trades[pair]['position']['hard_stop'] = str(stp)
        self.sim_trades[pair]['position']['stop_time'] = timestamp

        # save records
        self.record_trades(session, 'sim')

        # update sim_pos
        self.calc_sizing_non_live_tp(session, asset, 50, 'sim')

        self.counts_dict[f'sim_tp_{direction}'] += 1
        self.in_pos['sim_pfrd'] = self.in_pos['sim_pfrd'] / 2
        self.realised_pnl(self.sim_trades[pair])

    def sim_to_sim_closed(self, session, pair, close_order, direction):
        self.sim_trades[pair]['trade'].append(close_order)
        self.realised_pnl(self.sim_trades[pair])

        trade_id = int(self.sim_trades[pair]['position']['open_time'])
        self.closed_sim_trades[trade_id] = {}
        self.closed_sim_trades[trade_id]['trade'] = self.sim_trades[pair]['trade']
        self.record_trades(session, 'closed_sim')

        del self.sim_trades[pair]
        self.record_trades(session, 'sim')

    def close_sim(self, session, pair, direction):
        # if not session.live:
        #     print('')
        #     note = f"{self.name} sim close {direction} {pair} @ {price}"
        #     print(now, note)

        price = session.prices[pair]
        asset = pair[:-4]

        # initialise stuff
        sim_bal = float(self.sim_trades[pair]['position']['base_size'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)

        # execute order
        close_order = {'pair': pair,
                       'exe_price': str(price),
                       'trig_price': str(price),
                       'base_size': str(sim_bal),
                       'quote_size': f"{sim_bal * price:.2f}",
                       'reason': 'close_signal',
                       'timestamp': timestamp,
                       'type': f'close_{direction}',
                       'fee': '0',
                       'fee_currency': 'BNB',
                       'state': 'sim'}

        self.sim_trades[pair]['position']['base_size'] = '0'
        self.sim_trades[pair]['position']['hard_stop'] = None

        # update records
        self.sim_to_sim_closed(session, pair, close_order, direction)

        # update counts and live variables
        self.in_pos['sim'] = None
        self.in_pos['sim_pfrd'] = 0
        del self.sim_pos[asset]

        self.counts_dict[f'sim_close_{direction}'] += 1

    # tracked

    def tp_tracked(self, session, pair, stp, direction):
        print('')
        price = session.prices[pair]
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

    def tracked_to_closed(self, session, pair, close_order):
        self.tracked_trades[pair]['trade'].append(close_order)

        trade_id = int(self.tracked_trades[pair]['position']['open_time'])
        self.closed_trades[trade_id]['trade'] = self.tracked_trades[pair]['trade']
        self.record_trades(session, 'closed')

        del self.tracked_trades[pair]
        self.record_trades(session, 'tracked')

    def close_tracked(self, session, pair, direction):
        print('')
        asset = pair[:-4]
        price = session.prices[pair]
        now = datetime.now().strftime('%d/%m/%y %H:%M')

        note = f"{self.name} tracked close {direction} {pair} @ {price}"
        print(now, note)

        # initialise stuff
        timestamp = round(datetime.utcnow().timestamp() * 1000)

        # execute order
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

        # update records
        self.tracked_to_closed(session, pair, close_order)

        # update counts and live variables
        del self.tracked[asset]
        self.in_pos['tracked'] = None
        self.in_pos['tracked_pfrd'] = 0

    # other

    def check_size_against_records(self, pair, real_bal, base_size):
        base_size = Decimal(base_size)
        if base_size and (real_bal != base_size):  # check records match reality
            print(f"{self.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            mismatch = 100 * abs(base_size - real_bal) / base_size
            print(f"{mismatch = }%")

    def set_size_from_free(self, session, pair):
        """if clear_stop returns a base size of 0, this can be called to check for free balance,
        in case the position was there but just not in a stop order"""
        asset = pair[:-4]
        real_bal = Decimal(self.open_trades[pair]['position']['base_size'])

        session.margin_account_info()
        session.get_asset_bals()
        free_bal = session.bals_dict[asset]['free']

        return min(free_bal, real_bal)

    def calc_sizing_non_live_tp(self, session, asset: str, tp_pct: int, switch: str) -> None:
        '''updates sizing dictionaries (real/sim) with new open trade stats when
        state is sim or real but not live and a take-profit is triggered'''
        qw = Timer('calc_sizing_non_live_tp')
        qw.start()
        tp_scalar = 1 - (tp_pct / 100)
        if switch == 'real':
            pos_dict = self.real_pos
            entry = self.in_pos['real_ep']
            stop = self.in_pos['real_hs']
        elif switch == 'sim':
            pos_dict = self.sim_pos
            entry = self.in_pos['sim_ep']
            stop = self.in_pos['sim_hs']

        qty = float(pos_dict.get(asset).get('qty')) * tp_scalar
        val = float(pos_dict.get(asset).get('value')) * tp_scalar
        pf = pos_dict.get(asset).get('pf%') * tp_scalar
        or_R = pos_dict.get(asset).get('or_R') * tp_scalar
        or_dol = pos_dict.get(asset).get('or_$') * tp_scalar

        curr_price = session.prices[asset + 'USDT']
        r = (entry - stop) / entry
        pnl = (curr_price - entry) / entry
        pnl_r = pnl / r

        pos_dict[asset].update(
            {'qty': qty, 'value': f"{val:.2f}", 'pf%': pf, 'or_R': or_R, 'or_$': or_dol, 'pnl_R': pnl_r})

        qw.stop

    # repair trades
    def check_invalidation(self, session, ph):
        """returns true if trade is still valid, false otherwise.
        trade is still valid if direction is long and price is above invalidation, OR if dir is short and price is below"""
        pair = ph['pair']
        stp = ph['stop_price']

        dir_up = ph['direction'] == 'long'
        price = session.prices[pair]
        price_up = price > stp

        return dir_up == price_up

    def check_close_sig(self, session, ph):
        """returns true if trade is still above the previous close signal, false otherwise.
        trade is still valid if direction is long and price is above trig_price, OR if dir is short and price is below"""
        pair = ph['pair']
        trig = ph['trig_price']

        dir_up = ph['direction'] == 'long'
        price = session.prices[pair]
        price_up = price > trig

        return dir_up == price_up

    def repair_open(self, session, ph):
        pair = ph['pair']
        size = ph['base_size']
        stp = ph['stop_price']
        price = session.prices[pair]
        valid = self.check_invalidation(session, ph)

        if ph['completed'] is None:
            del self.open_trades[pair]

        elif ph['completed'] == 'borrow':
            funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
            del self.open_trades[pair]

        elif ph['completed'] == 'execute':
            if valid:
                self.open_real(session, pair, size, stp, ph['inval'], ph['direction'], 1)
            else:
                close_size = ph['api_order']['executedQty']
                if ph['direction'] == 'long':
                    funcs.sell_asset_M(session, pair, close_size, price, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, price, session.live)
                funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'trade_dict':
            if valid:
                self.open_real(session, pair, size, stp, ph['inval'], ph['direction'], 2)
            else:
                close_size = ph['api_order']['executedQty']
                if ph['direction'] == 'long':
                    funcs.sell_asset_M(session, pair, close_size, price, session.live)
                else:
                    funcs.buy_asset_M(session, pair, close_size, True, price, session.live)
                funcs.repay_asset_M(ph['loan_asset'], ph['liability'], session.live)
                del self.open_trades[pair]

        elif ph['completed'] == 'set_stop':
            self.open_real(session, pair, size, stp, ph['inval'], ph['direction'], 3)

    def repair_tp(self, session, ph):
        pair = ph['pair']
        cleared_size = ph.get('cleared_size')
        pct = ph.get('pct')
        stp = ph['stop_price']
        valid = self.check_invalidation(session, ph)

        if ph['completed'] is None:
            del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if valid:
                # the function call below sets new stop and updates position dict
                self.open_set_stop(session, pair, stp, ph, ph['direction'])
            else:
                self.close_real_full(session, pair, ph['direction'])

        elif ph['completed'] == 'execute':
            if pct == 100:
                tp_order, usdt_size = self.tp_repay_100(session, pair, ph['tp_order'], ph['direction'])
                self.open_to_tracked(session, pair, tp_order, ph['direction'])
                self.tp_update_records_100(session, pair, cleared_size, usdt_size, ph['direction'])
            else:
                if valid:
                    tp_order = self.tp_repay_partial(session, pair, stp, ph['tp_order'], ph['direction'])
                    tp_order = self.tp_reset_stop(session, pair, stp, tp_order, ph['direction'])
                    self.open_to_open(session, pair, tp_order, ph['direction'])
                    self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order,
                                                  ph['direction'])
                else:
                    remaining = self.open_trades[pair]['position']['base_size']
                    close_order = self.close_position(session, pair, remaining, 'close_signal', ph['direction'])
                    repay_size = self.close_repay(session, pair, close_order, ph['direction'])
                    self.open_to_closed(session, pair, close_order, ph['direction'])
                    self.close_real_7(session, pair, repay_size, ph['direction'])

        elif ph['completed'] == 'repay_100':
            self.open_to_tracked(session, pair, ph['tp_order'], ph['direction'])
            self.tp_update_records_100(session, pair, cleared_size, ph['repay_usdt'], ph['direction'])

        elif ph['completed'] == 'repay_part':
            if valid:
                tp_order = self.tp_reset_stop(session, pair, stp, ph['tp_order'], ph['direction'])
                self.open_to_open(session, pair, tp_order, ph['direction'])
                self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, tp_order,
                                              ph['direction'])
            else:
                remaining = self.open_trades[pair]['position']['base_size']
                close_order = self.close_position(session, pair, remaining, 'close_signal', ph['direction'])
                repay_size = self.close_repay(session, pair, close_order, ph['direction'])
                self.open_to_closed(session, pair, close_order, ph['direction'])
                self.close_real_7(session, pair, repay_size, ph['direction'])

        elif ph['completed'] == 'set_stop':
            self.open_to_open(session, pair, ph['tp_order'], ph['direction'])
            self.tp_update_records_partial(session, pair, pct, ph['inval'], cleared_size, ph['tp_order'],
                                          ph['direction'])

    def repair_close(self, session, ph):
        pair = ph['pair']

        if ph['completed'] is None:
            self.close_real_full(session, pair, ph['direction'], 1)

        elif ph['completed'] == 'clear_stop':
            self.close_real_full(session, pair, ph['direction'], 2)

        elif ph['completed'] == 'execute':
            self.close_real_full(session, pair, ph['direction'], 3)

        elif ph['completed'] == 'repay':
            self.close_real_full(session, pair, ph['direction'], 4)

    def repair_move_stop(self, session, ph):
        # when it failed to move the stop up, price was above the current and new stop levels. since then, price could
        # have stayed above, or it could have moved below one or both stop levels. if price is above both, i simply reset
        # the stop as planned. if price is below the new stop, i should close the position. if price is below the original
        # stop, the position should also be closed but will already have been if the old stop was still in place

        pair = ph['pair']
        price = session.prices[pair]
        pos_record = self.open_trades[pair]['position']

        if ph['completed'] == None:
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, ph['direction'], ph['stop_price'], pos_record, stage=1)
            elif price > self.open_trades[pair]['position']['hard_stop']:
                self.close_real_full(session, pair, ph['direction'])
            else:
                del self.open_trades[pair]['placeholder']

        elif ph['completed'] == 'clear_stop':
            if price > ph['stop_price']:
                self.move_api_stop(session, pair, ph['direction'], ph['stop_price'], pos_record, stage=2)
            else:
                self.close_real_full(session, pair, ph['direction'])

    def repair_trade_records(self, session, agent):
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
                elif ph['type'] in ['stop_long', 'stop_short']:
                    self.repair_record_stop(session, ph)
                elif ph['type'] in ['move_stop_long', 'move_stop_short']:
                    self.repair_move_stop(session, ph)
            except bx.BinanceAPIException as e:
                print("problem during repair_trade_records")
                pprint(ph)
                self.record_trades(session, all)
                print(e.status_code)
                print(e.message)


class DoubleST(Agent):
    '''200EMA and regular supertrend for bias with tight supertrend for entries/exits'''
    
    def __init__(self, session, mult1: float, mult2: float):
        # self.tf = session.tf
        # self.offset = session.offset
        self.mult1 = mult1
        self.mult2 = mult2
        self.name = f'{session.tf} dst {self.mult1}-{self.mult2}'
        self.id = f"double_st_{session.tf}_{session.offset}_{self.mult1}_{self.mult2}"
        Agent.__init__(self, session)
        session.indicators.update(['ema-200', 
                                   f"st-10-{self.mult1}", 
                                   f"st-10-{self.mult2}"])
        
    def spot_signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates spot buy and sell signals based on 2 supertrend indicators
        and a 200 period EMA'''
        
        # df['ema-200'] = df.close.ewm(200).mean()
        # ind.supertrend_new(df, 10, self.mult1)
        # # df.rename(columns={'st': 'st_loose', 'st_u': 'st_loose_u', 'st_d': 'st_loose_d'}, inplace=True)
        # ind.supertrend_new(df, 10, self.mult2)
        
        bullish_ema = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'ema-200']
        bullish_loose = df.at[len(df)-1, 'close'] > df.at[len(df)-1, f'st-10-{self.mult1}']
        bullish_tight = df.at[len(df)-1, 'close'] > df.at[len(df)-1, f'st-10-{self.mult2}']
        bearish_tight = df.at[len(df)-1, 'close'] < df.at[len(df)-1, f'st-10-{self.mult2}']
        
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
        
        if df.at[len(df)-1, f'st-10-{self.mult2}']:
            inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'st-10-{self.mult2}']) # current price proportional to invalidation price
        else:
            inval = 100000
            
        return {'signal': signal, 'inval': inval}
    
    def margin_signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''
        
        k = Timer(f'margin_signals')
        k.start()
        
        bullish_ema = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'ema-200']
        bearish_ema = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'ema-200']
        bullish_loose = df.at[len(df)-1, 'close'] > df.at[len(df)-1, f'st-10-{float(self.mult1)}']
        bearish_loose = df.at[len(df)-1, 'close'] < df.at[len(df)-1, f'st-10-{float(self.mult1)}']
        bullish_tight = df.at[len(df)-1, 'close'] > df.at[len(df)-1, f'st-10-{float(self.mult2)}']
        bearish_tight = df.at[len(df)-1, 'close'] < df.at[len(df)-1, f'st-10-{float(self.mult2)}']
        
        if bullish_ema:
            session.above_200_ema.add(pair)
        else:
            session.below_200_ema.add(pair)
        
        # bullish_book = bid_ask_ratio > 1
        # bearish_book = bid_ask_ratio < 1
        # bullish_volume = price rising on low volume or price falling on high volume
        # bearish_volume = price rising on high volume or price falling on low volume
        
        if bullish_ema and bullish_loose and bullish_tight: # and bullish_book
            signal = 'open_long'
        elif bearish_ema and bearish_loose and bearish_tight: # and bearish_book
            signal = 'open_short'
        elif bearish_tight:
            signal = 'close_long'
        elif bullish_tight:
            signal = 'close_short'
        else:
            signal = None
        
        if df.at[len(df)-1, f'st-10-{self.mult2}']:
            inval = df.at[len(df)-1, f'st-10-{self.mult2}']
            inval_ratio = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'st-10-{self.mult2}']) # current price proportional to invalidation price
        else:
            inval = 0
            inval_ratio = 100000
        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}
    

class EMACross(Agent):
    '''Simple EMA cross strategy with a longer-term EMA to set bias and a 
    trailing stop based on ATR bands'''
    
    def __init__(self, session, lookback_1, lookback_2, mult):
        # self.tf = session.tf
        # self.offset = session.offset
        self.lb1 = lookback_1
        self.lb2 = lookback_2
        self.mult = mult
        self.name = f'{session.tf} emacross {self.lb1}-{self.lb2}-{self.mult}'
        self.id = f"ema_cross_{session.tf}_{session.offset}_{self.lb1}_{self.lb2}_{self.mult}"
        Agent.__init__(self, session)
        session.indicators.update(['ema-200', 
                                   f"ema-{self.lb1}", 
                                   f"ema-{self.lb2}", 
                                   f"atr-10-{self.mult}"])
        
    
    def margin_signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''
        
        k = Timer('margin_signals')
        k.start()
        
        fast_ema_str = f"ema-{self.lb1}"
        slow_ema_str = f"ema-{self.lb2}"
        bias_ema_str = "ema-200"
        
        # if bias_ema_str not in df.columns:
        #     df[bias_ema_str] = df.close.ewm(self.lookback_limit).mean()
        # df[fast_ema_str] = df.close.ewm(self.lb1).mean()
        # df[slow_ema_str] = df.close.ewm(self.lb2).mean()
        # ind.atr_bands(df, 10, self.mult)

        bullish_bias = df.at[len(df)-1, 'close'] > df.at[len(df)-1, bias_ema_str]
        bearish_bias = df.at[len(df)-1, 'close'] < df.at[len(df)-1, bias_ema_str]
        bullish_cross = (df.at[len(df)-1, fast_ema_str] > df.at[len(df)-1, slow_ema_str]
                         and
                         df.at[len(df)-2, fast_ema_str] < df.at[len(df)-2, slow_ema_str])
        bearish_cross = (df.at[len(df)-1, fast_ema_str] < df.at[len(df)-1, slow_ema_str]
                         and
                         df.at[len(df)-2, fast_ema_str] > df.at[len(df)-2, slow_ema_str])
        bullish_emas = df.at[len(df)-1, fast_ema_str] > df.at[len(df)-1, slow_ema_str]
        bearish_emas = df.at[len(df)-1, fast_ema_str] < df.at[len(df)-1, slow_ema_str]
        
        in_long = (self.in_pos['real'] == 'long' 
                   or self.in_pos['sim'] == 'long'
                   or self.in_pos['tracked'] == 'long')
        in_short = (self.in_pos['real'] == 'short' 
                   or self.in_pos['sim'] == 'short'
                   or self.in_pos['tracked'] == 'short')
        
        if bullish_bias and bullish_emas and not in_long:
            signal = 'open_long'
        elif bearish_bias and bearish_emas and not in_short:
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
        
        if ((signal == 'open_long') or in_long) and df.at[len(df)-1, f'atr-10-{self.mult}-lower']:
            inval = df.at[len(df)-1, f'atr-10-{self.mult}-lower']
            inval_ratio = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'atr-10-{self.mult}-lower']) # current price proportional to invalidation price
        elif ((signal == 'open_short') or in_short) and df.at[len(df)-1, f'atr-10-{self.mult}-upper']:
            inval = df.at[len(df)-1, f'atr-10-{self.mult}-upper']
            inval_ratio = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'atr-10-{self.mult}-upper']) # current price proportional to invalidation price
        else:
            inval = None
            inval_ratio = None
        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}


class EMACrossHMA(Agent):
    '''Simple EMA cross strategy with a longer-term HMA to set bias more 
    responsively and a trailing stop based on ATR bands'''
    
    def __init__(self, session, lookback_1, lookback_2, mult):
        # self.tf = session.tf
        # self.offset = session.offset
        self.lb1 = lookback_1
        self.lb2 = lookback_2
        self.mult = mult
        self.name = f'{session.tf} emaxhma {self.lb1}-{self.lb2}-{self.mult}'
        self.id = f"ema_cross_hma_{session.tf}_{session.offset}_{self.lb1}_{self.lb2}_{self.mult}"
        Agent.__init__(self, session)
        session.indicators.update(['hma-200', 
                                   f"ema-{self.lb1}", 
                                   f"ema-{self.lb2}", 
                                   f"atr-10-{self.mult}"])
        
    
    def margin_signals(self, session, df: pd.DataFrame, pair: str) -> dict:
        '''generates open and close signals for long and short trades based on
        two supertrend indicators and a 200 period EMA'''
        
        k = Timer(f'margin_signals')
        k.start()
        
        fast_ema_str = f"ema-{self.lb1}"
        slow_ema_str = f"ema-{self.lb2}"
        bias_hma_str = f"hma-200"
        
        # if bias_hma_str not in df.columns:
        #     df[bias_hma_str] = ind.hma(df.close, self.lookback_limit)
        # df[fast_ema_str] = df.close.ewm(self.lb1).mean()
        # df[slow_ema_str] = df.close.ewm(self.lb2).mean()
        # ind.atr_bands(df, 10, self.mult)
        
        bullish_bias = df.at[len(df)-1, 'close'] > df.at[len(df)-1, bias_hma_str]
        bearish_bias = df.at[len(df)-1, 'close'] < df.at[len(df)-1, bias_hma_str]
        bullish_cross = (df.at[len(df)-1, fast_ema_str] > df.at[len(df)-1, slow_ema_str]
                         and
                         df.at[len(df)-2, fast_ema_str] < df.at[len(df)-2, slow_ema_str])
        bearish_cross = (df.at[len(df)-1, fast_ema_str] < df.at[len(df)-1, slow_ema_str]
                         and
                         df.at[len(df)-2, fast_ema_str] > df.at[len(df)-2, slow_ema_str])
        bullish_emas = df.at[len(df)-1, fast_ema_str] > df.at[len(df)-1, slow_ema_str]
        bearish_emas = df.at[len(df)-1, fast_ema_str] < df.at[len(df)-1, slow_ema_str]
        
        in_long = (self.in_pos['real'] == 'long'
                   or self.in_pos['sim'] == 'long'
                   or self.in_pos['tracked'] == 'long')
        in_short = (self.in_pos['real'] == 'short' 
                   or self.in_pos['sim'] == 'short'
                   or self.in_pos['tracked'] == 'short')
        
        if bullish_bias and bullish_cross:
            signal = 'open_long'
        elif bearish_bias and bearish_cross:
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
                
        if ((signal == 'open_long') or in_long) and df.at[len(df)-1, f'atr-10-{self.mult}-lower']:
            inval = df.at[len(df)-1, f'atr-10-{self.mult}-lower']
            inval_ratio = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'atr-10-{self.mult}-lower']) # current price proportional to invalidation price
        elif ((signal == 'open_short') or in_short) and df.at[len(df)-1, f'atr-10-{self.mult}-upper']:
            inval = df.at[len(df)-1, f'atr-10-{self.mult}-upper']
            inval_ratio = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, f'atr-10-{self.mult}-upper']) # current price proportional to invalidation price
        else:
            inval = None
            inval_ratio = None
        
        k.stop()
        return {'signal': signal, 'inval': inval, 'inval_ratio': inval_ratio}


from binance.client import Client
import binance.enums as be
import pandas as pd
import indicators as ind
import utility_funcs as uf
from pathlib import Path
import json
import keys
from json.decoder import JSONDecodeError
from datetime import datetime
from pushbullet import Pushbullet
import statistics as stats
from pprint import pprint
from timers import Timer
from decimal import Decimal, getcontext
import binance_funcs as funcs
from typing import Union, List, Tuple, Dict, Set, Optional, Any

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
    
    def record_stopped_trades(self, session) -> None:
        '''compiles a list of stop-loss order ids from the open_trades dict, 
        checks that list against open stop-loss orders on binance, and works 
        out which stops have been hit since the last session, then updates all
        relevant trade records and performance metrics'''
        
        m = Timer('record_stopped_trades')
        m.start()
        # loop through agent.open_trades and call latest_stop_id(trade_record) to
        # compile a list of order ids for each open trade's stop loss orders, then 
        # check binance to find which don't have an active stop-loss
        stop_ids = [uf.latest_stop_id(v) for v in self.open_trades.values()]
        
        open_orders = client.get_open_margin_orders()
        ids_remaining = [i.get('orderId') for i in open_orders]
        symbols_remaining = [i.get('symbol') for i in open_orders]
        
        stopped = []
        for pair, sid, time in stop_ids:
            if sid:
                if sid not in ids_remaining:
                    stopped.append((pair, sid, time))
                else:
                    continue
            elif pair not in symbols_remaining:
                stopped.append((pair, sid, time))
        
        # print(f'number of stopped trades: {len(stopped)}')
        
        # for any that don't, assume that the stop was hit and check for exchange records
        for pair, sid, time in stopped:
            trade_record = self.open_trades.get(pair)
            if sid == 'not live':
                continue
            order_list = client.get_all_margin_orders(symbol=pair, orderId=sid, startTime=time-10000)
            
            # print(sid)
            # pprint(order_list)
            
            order = None
            if order_list and sid:
                for o in order_list[::-1]:
                    if o.get('orderId') == sid and o.get('status') == 'FILLED':
                        order = o
                        break
            elif order_list and not sid:
                for o in order_list[::-1]:
                    if o.get('type') == 'STOP_LOSS_LIMIT' and o.get('status') == 'FILLED':
                        order = o
                        break
            else:
                print(f'No orders on binance for {pair}')
                
            if order:
                if (order.get('side') == 'BUY'):
                    trade_type = 'stop_short'
                    asset = pair[:-4]
                    stop_size = Decimal(order.get('executedQty'))
                    funcs.repay_asset_M(asset, stop_size, session.live)
                else:
                    trade_type = 'stop_long'
                    stop_size = Decimal(order.get('cummulativeQuoteQty'))
                    funcs.repay_asset_M('USDT', stop_size, session.live)
                
                
                stop_dict = uf.create_stop_dict(order)
                stop_dict['type'] = trade_type                
                stop_dict['state'] = 'real'
                stop_dict['reason'] = 'hit hard stop'
                stop_dict['liability'] = uf.update_liability(trade_record, stop_size, 'reduce')
                
                trade_record.append(stop_dict)
                
                ts_id = int(trade_record[0].get('timestamp'))
                self.closed_trades[ts_id] = trade_record
                self.record_trades(session, 'closed')
                del self.open_trades[pair]
                self.record_trades(session, 'open')
                
                if trade_type == 'stop_long':
                    self.realised_pnl(trade_record, 'long')
                    self.counts_dict['real_stop_long'] += 1
                else:
                    self.realised_pnl(trade_record, 'short')
                    self.counts_dict['real_stop_short'] += 1
                
            else:
                print(f"no stop for {pair} {self.name}")
                # check for a free balance matching the size. if there is, that means
                # the stop was never set in the first place and needs to be set
                free_bal = session.bals_dict[pair[:-4]].get('free')
                # # need to finish writing this bit - check size fits and then set stop if it does or sell if it doesn't
                # trade_size = trade_record[-1]['base_size']
                price = session.prices[pair]
                value = free_bal * price
                if value > 10:
                    note = f'{pair} in position with no stop-loss'
                    pb.push_note(session.now_start, note)
        m.stop()
    
    def record_stopped_sim_trades(self, session) -> None:
        '''goes through all trades in the sim_trades file and checks their recent 
        price action against their most recent hard_stop to see if any of them would have 
        got stopped out'''
        
        n = Timer('record_stopped_sim_trades')
        n.start()
        
        del_pairs = []
        for pair, v in self.sim_trades.items():
            # first filter out all trades which started out real
            if v[0].get('real'):
                continue
            
            long_trade = ('long' in v[0].get('type'))
            
            x = Timer('calc base size and stop')
            x.start()
            # calculate current base size
            base_size = 0
            for i in v:
                if i.get('type') in ['open_long', 'open_short', 'add_long', 'add_short']:
                    base_size += float(i.get('base_size'))
                else:
                    base_size -= float(i.get('base_size'))
            
            # find most recent hard stop
            for i in v[-1::-1]:
                if i.get('hard_stop'):
                    stop = float(i.get('hard_stop'))
                    stop_time = int(i.get('timestamp'))
                    break
            x.stop()
            
            # check lowest low since stop was set
            
            timespan = datetime.now().timestamp() - (stop_time/1000)
            
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
                stop_dt = datetime.fromtimestamp(stop_time/1000)
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
            
            if long_trade:
                trade_type = 'stop_long'
                ll = df.low.min()
                stopped = ll < stop
                overshoot_pct = round((100 * (stop - ll) / stop), 3) # % distance that price broke through the stop
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
                overshoot_pct = round((100 * (hh - stop) / stop), 3) # % distance that price broke through the stop
                if stopped:
                    for i in range(len(df)):
                        if df.at[i, 'high'] >= stop:
                            stop_hit_time = df.at[i, 'timestamp']
                            if isinstance(stop_hit_time, pd.Timestamp):
                                stop_hit_time = stop_hit_time.timestamp()
                    
            
            if stopped:
                # create trade dict
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
                
                v.append(trade_dict)
                
                ts_id = int(v[0].get('timestamp'))
                self.closed_sim_trades[ts_id] = v
                self.record_trades(session, 'closed_sim')
                
                if long_trade:
                    self.realised_pnl(v, 'long')
                    self.counts_dict['sim_stop_long'] += 1
                else:
                    self.realised_pnl(v, 'short')
                    self.counts_dict['sim_stop_short'] += 1
                del_pairs.append(pair)
            
        for p in del_pairs:
            del self.sim_trades[p]
        n.stop()
    
    def realised_pnl(self, trade_record: dict, side: str) -> None:
        '''calculates realised pnl of a tp or close denominated in the trades 
        own R value'''
        
        i = Timer(f'realised_pnl {side}')
        i.start()
        entry = float(trade_record[0].get('exe_price'))
        init_stop = float(trade_record[0].get('hard_stop'))
        init_size = float(trade_record[0].get('base_size'))
        final_exit = float(trade_record[-1].get('exe_price'))
        final_size = float(trade_record[-1].get('base_size'))
        r_val = abs((entry - init_stop) / entry)
        if side == 'long':
            trade_pnl = (final_exit - entry) / entry
        else:
            trade_pnl = (entry - final_exit) / entry
        trade_r = round(trade_pnl / r_val, 3)
        scalar = final_size / init_size
        realised_r = trade_r * scalar
        
        if trade_record[-1].get('state') == 'real':
            if side == 'long':
                self.realised_pnl_long += realised_r
            else:
                self.realised_pnl_short += realised_r
        elif trade_record[-1].get('state') == 'sim':
            if side == 'long':
                self.sim_pnl_long += realised_r
            else:
                self.sim_pnl_short += realised_r
        else:
            print(f'state in record: {trade_record[-1].get("state")}')
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
            if state == 'open':
                json.dump(self.open_trades, file)
            if state == 'sim':
                json.dump(self.sim_trades, file)
            if state == 'tracked':
                json.dump(self.tracked_trades, file)
            if state == 'closed':
                json.dump(self.closed_trades, file)
            if state == 'closed_sim':
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
            if state == 'tracked':
                asset = k[:-4]
                size_dict[asset] = {}
            else:
                asset = k[:-4]
                price = self.prices[k]
                size_dict[asset] = uf.open_trade_stats(now, total_bal, v, price)
        a.stop()
        return size_dict

    def calc_pos_fr_dol(self, trade_record: list, direction: str, state: str) -> None:
        '''populates in_pos dictionary for the current position with pfrd, ep and hs
        values. pfrd is the amount in usdt that 1R represents for this position.
        ep and hs are the original entry price and the current hard stop'''
        
        s = Timer(f'calc_pos_fr_dol {direction} {state}')
        s.start()
        if self.in_pos[state] and trade_record and trade_record[0].get('type')[0] == 'o':
            qs = float(trade_record[0].get('quote_size'))
            ep = float(trade_record[0].get('exe_price'))
            hs = float(trade_record[0].get('hard_stop'))
            pos_fr_dol = qs * ((ep - hs) / ep) if direction == 'long' else qs * ((hs - ep) / ep)
        else:
            ep = None # i refer to this later and need it to exist even if it has no value
            hs = None
            pos_fr_dol = self.fixed_risk_dol_l if direction == 'long' else self.fixed_risk_dol_s
        
        self.in_pos[f'{state}_pfrd'] = pos_fr_dol
        self.in_pos[f'{state}_ep'] = ep
        self.in_pos[f'{state}_hs'] = hs
        s.stop()

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
            if trade_record[0].get('type')[-4:] == 'long':
                self.in_pos[state] = 'long'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record, 'long', state)
            else:
                self.in_pos[state] = 'short'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record, 'short', state)
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
    
    def move_stop(self, session, pair, df, state, direction):
        if state == 'real':
            trade_record = self.open_trades[pair]
        elif state == 'sim':
            trade_record = self.sim_trades[pair]
        elif state == 'tracked':
            trade_record = self.tracked_trades[pair]
        
        current_stop = None
        for record in trade_record[::-1]:
           if record.get('hard_stop'):
               current_stop = float(record.get('hard_stop'))
               break
        
        low_atr = df.at[len(df)-1, f'atr-10-{float(self.mult)}-lower']
        high_atr = df.at[len(df)-1, f'atr-10-{float(self.mult)}-upper']
        
        if state == 'real' and direction == 'long' and low_atr > current_stop:
            print(f"*** {self.name} {pair} {state} {direction} move stop from {current_stop:.3} to {low_atr:.3}")
            _, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
            if trade_record[-1].get('curr_base_size') and not base_size:
                base_size = trade_record[-1].get('curr_base_size')
            stop_order = funcs.set_stop_M(session, pair, base_size, be.SIDE_SELL, low_atr, low_atr*0.8)
            self.open_trades[pair][-1]['stop_id'] = stop_order.get('orderId')
            self.open_trades[pair][-1]['hard_stop'] = low_atr
            self.record_trades(session, 'open')
        
        elif state in ['sim', 'tracked'] and direction == 'long' and low_atr > current_stop:
            if state == 'sim':
                self.sim_trades[pair][-1]['hard_stop'] = low_atr
            else:
                self.tracked_trades[pair][-1]['hard_stop'] = low_atr
            self.record_trades(session, state)
            
        elif state == 'real' and direction == 'short' and high_atr < current_stop:
            print(f"*** {self.name} {pair} {state} {direction} move stop from {current_stop:.3} to {high_atr:.3}")
            _, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
            stop_order = funcs.set_stop_M(session, pair, base_size, be.SIDE_BUY, high_atr, high_atr*1.2)
            self.open_trades[pair][-1]['stop_id'] = stop_order.get('orderId')
            self.open_trades[pair][-1]['hard_stop'] = high_atr
            self.record_trades(session, 'open')
            
        elif state in ['sim', 'tracked'] and direction == 'short' and high_atr < current_stop:
            if state == 'sim':
                self.sim_trades[pair][-1]['hard_stop'] = high_atr
            else:
                self.tracked_trades[pair][-1]['hard_stop'] = high_atr
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
        
        for state in ['real', 'sim', 'tracked']:
            if self.in_pos[state]:
                self.move_stop(session, pair, df, state, self.in_pos[state])
        
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
        
        for state in ['real', 'sim', 'tracked']:
            if self.in_pos[state]:
                self.move_stop(session, pair, df, state, self.in_pos[state])
                
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
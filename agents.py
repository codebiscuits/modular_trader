from binance.client import Client
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
from decimal import Decimal
import binance_funcs as funcs

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class DoubleST():
    '''regular supertrend for bias with tight supertrend for entries/exits'''
    
    realised_pnl_long = 0
    realised_pnl_short = 0
    sim_pnl_long = 0
    sim_pnl_short = 0
    indiv_r_limit = 1.2
    total_r_limit = 20
    target_risk = 0.1
    max_pos = 20
    
    presets = {1: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 1, 'mult': 1.0}, 
               2: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 3, 'mult': 1.2}, 
               3: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 4, 'mult': 1.4}, 
               4: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 5, 'mult': 1.6}, 
               5: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 5, 'mult': 1.8}, 
               6: {'timeframe': '4h', 'tf_offset': 0, 'lookback': 6, 'mult': 2.0}, 
               }
    
    def __init__(self, session, preset):
        t = Timer('agent init')
        t.start()
        self.live = session.live
        self.bal = session.bal
        self.fr_max = session.fr_max
        self.prices = session.prices
        self.lb = self.presets[preset]['lookback']
        self.mult = self.presets[preset]['mult']
        self.tf = self.presets[preset]['timeframe']
        self.offset = self.presets[preset]['tf_offset']
        self.name = f'{self.tf} dst {self.lb}-{self.mult}'
        self.id = f'double_st_{self.tf}_{self.offset}_{self.lb}_{self.mult}'
        print(f'\nInitialising {self.name}')
        self.market_data = self.mkt_data_path()
        self.counts_dict = {'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0, 'real_close_long': 0, 
                           'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0, 'sim_close_long': 0, 
                           'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0, 'real_close_short': 0, 
                           'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0, 'sim_close_short': 0, 
                           'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 'too_much_or': 0, 
                           'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0}
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
        self.fixed_risk_l = self.set_fixed_risk('long')
        self.fixed_risk_s = self.set_fixed_risk('short')
        # self.test_fixed_risk(0.0001, 0.0001)
        self.max_positions = self.set_max_pos()
        self.max_init_r_l = self.fixed_risk_l * self.total_r_limit
        self.max_init_r_s = self.fixed_risk_s * self.total_r_limit
        self.fixed_risk_dol_l = self.fixed_risk_l * self.bal
        self.fixed_risk_dol_s = self.fixed_risk_s * self.bal
        t.stop()
        
    def __str__(self):
        return self.id
    
    def sync_test_records(self):
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
                    print(f'{switch}_trades file empty')
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

    def mkt_data_path(self):
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
    
    def read_open_trade_records(self, switch):
        w = Timer(f'read_open_trade_records-{switch}')
        w.start()
        ot_path = Path(f"{self.market_data}/{self.id}")
        if not ot_path.exists():
            ot_path.mkdir(parents=True)
        ot_path = ot_path / f'{switch}_trades.json'
        
        if ot_path.exists():
            with open(ot_path, "r") as ot_file:
                try:
                    open_trades = json.load(ot_file)
                except JSONDecodeError as e:
                    open_trades = {}
        else:
            print("ot_path doesn't exist")
            open_trades = {}
            ot_path.touch()
            print(f'{ot_path} not found')
    
            
        w.stop()
        return open_trades

    def read_closed_trade_records(self):
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
            print(f'{ct_path} not found')
        e.stop()
        return closed_trades

    def read_closed_sim_trade_records(self):
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

    def backup_trade_records(self):
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
    
    def calc_tor(self):
        u = Timer('calc_tor')
        u.start()
        self.or_list = [v.get('or_R') for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
        u.stop()
    
    def record_stopped_trades(self, session):
        m = Timer('record_stopped_trades')
        m.start()
        # loop through agent.open_trades and call latest_stop_id(trade_record) to
        # compile a list of order ids for each open trade's stop loss orders, then 
        # check binance to find which don't have an active stop-loss
        stop_ids = [uf.latest_stop_id(v) for v in self.open_trades.values()]
        
        open_orders = client.get_open_orders()
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
            # order_list = client.get_all_orders(symbol=pair, orderId=sid, startTime=time-10000)
            if sid == 'not live':
                continue
            order_list = client.get_all_margin_orders(symbol=pair, orderId=sid, startTime=time-10000)
            
            order = None
            if order_list and sid:
                for o in order_list[::-1]:
                    if o.get('order_id') == sid and o.get('status') == 'FILLED':
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
                
                ts_id = trade_record[0].get('timestamp')
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
                # check for a free balance matching the size. if there is, that means 
                # the stop was never set in the first place and needs to be set
                print(f'getting {pair[:-4]} free balance')
                free_bal = float(client.get_asset_balance(pair[:-4]).get('free'))
                print(f'getting {pair} price')
                price = session.prices[pair]
                value = free_bal * price
                if value > 10:
                    note = f'{pair} in position with no stop-loss'
                    pb.push_note(session.now_start, note)   
        m.stop()
    
    def record_stopped_sim_trades(self, session):
        n = Timer('record_stopped_sim_trades')
        n.start()
        '''goes through all trades in the sim_trades file and checks their recent 
        price action against their most recent hard_stop to see if any of them would have 
        got stopped out'''
        
        del_pairs = []
        for pair, v in self.sim_trades.items():
            # first filter out all trades which started out real
            if v[0].get('real'):
                continue
            
            long_trade = True if 'long' in v[0].get('type') else False
            
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
                    stop_time = i.get('timestamp')
                    break
            x.stop()    
            # check lowest low since stop was set
            z = Timer('read ohlc pickles')
            z.start()
            df = pd.read_pickle(f'{session.ohlc_data}/{pair}.pkl')
            z.stop()
            
            # trim df down to just the rows since the last stop was set
            if (datetime.now().timestamp() - stop_time/1000) < 13000:
                df = df.tail(2)
            else:
                stop_dt = datetime.fromtimestamp(stop_time/1000)
                df = df.loc[df.timestamp > stop_dt]
            
            if df.empty:
                z = Timer('get_historical_klines')
                z.start()
                klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, stop_time)
                cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                        'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
                df = pd.DataFrame(klines, columns=cols)
                df['timestamp'] = df['timestamp'] * 1000000
                df = df.astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                z.stop()
            
            if long_trade:
                trade_type = 'stop_long'
                ll = df.low.min()
                # stop_time = find timestamp of that lowest low candle
                stopped = ll < stop
                overshoot_pct = round((100 * (stop - ll) / stop), 3) # % distance that price broke through the stop
            else:
                trade_type = 'stop_short'
                hh = df.high.max()
                # stop_time = find timestamp of that highest high candle
                stopped = hh > stop
                overshoot_pct = round((100 * (hh - stop) / stop), 3) # % distance that price broke through the stop
            
            if stopped:
                # create trade dict
                trade_dict = {'timestamp': stop_time, 
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
                note = f"*sim* stopped out {pair} @ {stop}"
                print(session.now_start, note)
                
                v.append(trade_dict)
                
                ts_id = v[0].get('timestamp')
                self.closed_sim_trades[ts_id] = v
                self.record_trades(session, 'closed_sim')
                
                if long_trade:
                    self.realised_pnl(v, 'long')
                    self.counts_dict['sim_stop_long'] += 1
                else:
                    self.realised_pnl(v, 'short')
                    self.counts_dict['sim_stop_short'] += 1
                del_pairs.append(pair)
            
        # print(f"number of stopped sim trades: {self.counts_dict['sim_stop_long'] +  self.counts_dict['sim_stop_short']}")        
        
        for p in del_pairs:
            del self.sim_trades[p]
        n.stop()
    
    def realised_pnl(self, trade_record, side):
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
        # print(f'\nrealised pnl: {realised_r:.1f}R')
        # print(f'{entry = }, {init_stop = }, {final_exit = }')
        # print(f'{r_val = }, {trade_r = }, {scalar = }')
        # print(f'{init_size = }, {final_size = }\n')
        
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
    
    def record_trades(self, session, switch):
        b = Timer(f'record_trades {switch}')
        b.start()
        filepath = Path(f"{session.market_data}/{self.id}/{switch}_trades.json")
        if not filepath.exists():
            print('filepath doesnt exist')
            filepath.touch()
        with open(filepath, "w") as file:
            if switch == 'open':
                json.dump(self.open_trades, file)
            if switch == 'sim':
                json.dump(self.sim_trades, file)
            if switch == 'tracked':
                json.dump(self.tracked_trades, file)
            if switch == 'closed':
                json.dump(self.closed_trades, file)
            if switch == 'closed_sim':
                json.dump(self.closed_sim_trades, file)
        b.stop()
    
    def set_fixed_risk(self, direction:str):
        o = Timer(f'set_fixed_risk-{direction}')
        o.start()
        '''calculates fixed risk setting for new trades based on recent performance 
        and previous setting. if recent performance is very good, fr is increased slightly.
        if not, fr is decreased by thirds'''
        
        def reduce_fr(factor, fr_prev, fr_inc):
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
            with open(f"{self.market_data}/{self.id}/bal_history.txt", "r") as file:
                bal_data = file.readlines()
            
            if bal_data:
                prev_bal = json.loads(bal_data[-1]).get('balance')
            else:
                prev_bal = self.bal
            bal_change_pct = 100 * (self.bal - prev_bal) / prev_bal
            
            lookup = f'realised_pnl_{direction}' if switch == 'real' else f'sim_r_pnl_{direction}'
            pnls = {}
            for i in range(1, 6):
                if bal_data and len(bal_data) > 5:
                    pnls[i] = json.loads(bal_data[-1*i]).get(lookup, -1)
                else:
                    pnls[i] = -1 # if there's no data yet, return -1 instead
            
            score = 0
            if pnls.get(1) > 0:
                score += 5
            elif pnls.get(1) < 0:
                score -= 5
            if pnls.get(2) > 0:
                score += 4
            elif pnls.get(2) < 0:
                score -= 4
            if pnls.get(3) > 0:
                score += 3
            elif pnls.get(3) < 0:
                score -= 3
            if pnls.get(4) > 0:
                score += 2
            elif pnls.get(4) < 0:
                score -= 2
            if pnls.get(5) > 0:
                score += 1
            elif pnls.get(5) < 0:
                score -= 1
            
            return score
        
        real_score = score_accum(direction, 'real')
        sim_score = score_accum(direction, 'sim')
        
        if self.open_trades and real_score:
            score = real_score
        else:
            score = sim_score
        
        if bal_data:
            prev_bal = json.loads(bal_data[-1]).get('balance')
        else:
            prev_bal = self.bal
        bal_change_pct = round(100 * (self.bal - prev_bal) / prev_bal, 3)
        if bal_change_pct < -0.1:
            score -= 1
        
        # print('-')
        # print(f'{direction} - {real_score = }, {sim_score = }, {bal_change_pct = }, {score = }')
        
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
            note = f'fixed risk adjusted from {round(fr_prev*10000, 1)}bps to {round(fr*10000, 1)}bps'
            pb.push_note(title, note)
        o.stop()
        return round(fr, 5)
    
    def test_fixed_risk(self, fr_l, fr_s):
        print(f'*** WARNING: FIXED RISK MANUALLY SET to {fr_l} / {fr_s} ***')
        self.fixed_risk_l = fr_l
        self.fixed_risk_s = fr_s
    
    def set_max_pos(self):
        p = Timer('set_max_pos')
        p.start()
        max_pos = 20
        if self.real_pos:
            open_pnls = [v.get('pnl') for v in self.real_pos.values() if v.get('pnl')]
            if open_pnls:
                avg_open_pnl = stats.median(open_pnls)
            else:
                avg_open_pnl = 0
            max_pos = 20 if avg_open_pnl <= 0 else 50
        p.stop()
        return max_pos
    
    def current_positions(self, switch:str):
        a = Timer(f'current_positions-{switch}')
        a.start()
        '''creates a dictionary of open positions by checking either 
        open_trades.json, sim_trades.json or tracked_trades.json'''
            
        # filepath = Path(f'{self.market_data}/{self.id}/{switch}_trades.json')
        # with open(filepath, 'r') as file:
        #     try:
        #         data = json.load(file)
        #     except:
        #         data = {}
        
        if switch == 'open':
            data = self.open_trades
        elif switch == 'sim':
            data = self.sim_trades
        elif switch == 'tracked':
            data = self.tracked_trades
        
        size_dict = {}
        now = datetime.now()
        total_bal = self.bal
        
        for k, v in data.items():
            if switch == 'tracked':
                asset = k[:-4]
                size_dict[asset] = {}
            else:
                asset = k[:-4]
                price = self.prices[k]
                size_dict[asset] = uf.open_trade_stats(now, total_bal, v, price)
        a.stop()
        return size_dict# -*- coding: utf-8 -*-

    def calc_pos_fr_dol(self, trade_record, direction, switch):   
        s = Timer(f'calc_pos_fr_dol {direction} {switch}')
        s.start()
        if self.in_pos[switch] and trade_record and trade_record[0].get('type')[0] == 'o':
            qs = float(trade_record[0].get('quote_size'))
            ep = float(trade_record[0].get('exe_price'))
            hs = float(trade_record[0].get('hard_stop'))
            pos_fr_dol = qs * ((ep - hs) / ep) if direction == 'long' else qs * ((hs - ep) / ep)
        else:
            ep = None # i refer to this later and need it to exist even if it has no value
            hs = None
            pos_fr_dol = self.fixed_risk_dol_l if direction == 'long' else self.fixed_risk_dol_s
        
        self.in_pos[f'{switch}_pfrd'] = pos_fr_dol
        self.in_pos[f'{switch}_ep'] = ep
        self.in_pos[f'{switch}_hs'] = hs
        s.stop()

    def set_in_pos(self, pair, switch):
        d = Timer(f'set_in_pos {switch}')
        d.start()
        asset = pair[:-4]
        if switch == 'real':
            pos_dict = self.real_pos.keys()
            trade_record = self.open_trades.get(pair)
        elif switch == 'sim':
            pos_dict = self.sim_pos.keys()
            trade_record = self.sim_trades.get(pair)
        elif switch == 'tracked':
            pos_dict = self.tracked.keys()
            trade_record = self.tracked_trades.get(pair)
        
        if asset in pos_dict:
            if trade_record[0].get('type')[-4:] == 'long':
                self.in_pos[switch] = 'long'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record, 'long', switch)
            else:
                self.in_pos[switch] = 'short'
                # calculate dollar denominated fixed-risk per position
                self.calc_pos_fr_dol(trade_record, 'short', switch)
        d.stop()

    def init_in_pos(self, pair):
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
    
    def too_new(self, df):
        g = Timer('too_new')
        g.start()
        '''returns True if there is less than 200 hours of history AND if
        there are no current positions in the asset'''
        if self.in_pos['real'] or self.in_pos['sim'] or self.in_pos['tracked']:
            no_pos = False
        else:
            no_pos = True    
        
        return len(df) <= 200 and no_pos

    def open_pnl(self, switch):
        h = Timer(f'open_pnl {switch}')
        h.start()
        total = 0
        if switch == 'real':
            pos_dict = self.real_pos.values()
        elif switch == 'sim':
            pos_dict = self.sim_pos.values()
            for pos in pos_dict:
                if pos.get('pnl_R'):
                    total += pos['pnl_R']
        else:
            print('open_pnl requires argument real or sim')
        h.stop()
        return total
    
    def tp_signals(self, asset):
        j = Timer('tp_signals')
        j.start()
        if self.real_pos.get(asset):
            real_or = self.real_pos.get(asset).get('or_R')
            self.in_pos['real_tp_sig'] = ((real_or > self.indiv_r_limit) and 
                                          (abs(self.in_pos.get('real_price_delta', 0)) > 0.001))
        if self.sim_pos.get(asset):
            sim_or = self.sim_pos.get(asset).get('or_R')
            self.in_pos['sim_tp_sig'] = ((sim_or > self.indiv_r_limit) and 
                                         (abs(self.in_pos.get('sim_price_delta', 0)) > 0.001))
        j.stop()
    
    def spot_signals(self, session, df):
        
        df['ema200'] = df.close.ewm(200).mean()
        ind.supertrend_new(df, 10, 3)
        df.rename(columns={'st': 'st_loose', 'st_u': 'st_loose_u', 'st_d': 'st_loose_d'}, inplace=True)
        ind.supertrend_new(df, self.lb, self.mult)
        
        bullish_ema = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'ema200']
        bullish_loose = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st_loose']
        bullish_tight = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st']
        bearish_tight = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st']
        
        if bullish_ema:
            session.above_200_ema[0] += 1
            session.above_200_ema[1] += 1
        else:
            session.above_200_ema[1] += 1
        
        if bullish_ema and bullish_loose and bullish_tight:
            signal = 'spot_open'
        elif bearish_tight:
            signal = 'spot_close'
        else:
            signal = None
        
        if df.at[len(df)-1, 'st']:
            inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, 'st']) # current price proportional to invalidation price
        else:
            inval = 100000
            
        return {'signal': signal, 'inval': inval}
    
    def margin_signals(self, session, df, pair):
        k = Timer(f'margin_signals')
        k.start()
        
        if 'ema200' not in df.columns:
            df['ema200'] = df.close.ewm(200).mean()
        ind.supertrend_new(df, 10, 3)
        df.rename(columns={'st': 'st_loose', 'st_u': 'st_loose_u', 'st_d': 'st_loose_d'}, inplace=True)
        ind.supertrend_new(df, self.lb, self.mult)
    
        bullish_ema = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'ema200']
        bearish_ema = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'ema200']
        bullish_loose = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st_loose']
        bearish_loose = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st_loose']
        bullish_tight = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st']
        bearish_tight = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st']
        
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
        
        if df.at[len(df)-1, 'st']:
            inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, 'st']) # current price proportional to invalidation price
        else:
            inval = 100000
        k.stop()
        return {'signal': signal, 'inval': inval}




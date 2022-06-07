import indicators as ind
import utility_funcs as uf
from pathlib import Path
import json
from json.decoder import JSONDecodeError
from datetime import datetime
from pushbullet import Pushbullet
import statistics as stats
from pprint import pprint

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class DoubleST():
    '''regular supertrend for bias with tight supertrend for entries/exits'''
    
    realised_pnl_long = 0
    realised_pnl_short = 0
    sim_pnl_long = 0
    sim_pnl_short = 0
    indiv_r_limit = 1.4
    total_r_limit = 20
    target_risk = 0.1
    max_pos = 20
    
    
    def __init__(self, session, timeframe, tf_offset, lookback, mult):
        self.live = session.live
        self.bal = session.bal
        self.fr_max = session.fr_max
        self.name = f'double_st_{timeframe}_{tf_offset}_{lookback}_{mult}'
        self.lb = lookback
        self.mult = mult
        self.market_data = self.mkt_data_path()
        if not self.live:
            self.sync_test_records()
        self.open_trades = self.read_open_trade_records('open')
        self.sim_trades = self.read_open_trade_records('sim')
        self.tracked_trades = self.read_open_trade_records('tracked')
        self.closed_trades = self.read_closed_trade_records()
        self.closed_sim_trades = self.read_closed_sim_trade_records()
        self.backup_trade_records()
        self.real_pos = self.current_positions('open')
        self.sim_pos = self.current_positions('sim')
        self.tracked = self.current_positions('tracked')
        self.fixed_risk_l = self.set_fixed_risk('long')
        self.fixed_risk_s = self.set_fixed_risk('short')
        self.max_positions = self.set_max_pos()
        self.max_init_r_l = self.fixed_risk_l * self.total_r_limit
        self.max_init_r_s = self.fixed_risk_s * self.total_r_limit
        self.fixed_risk_dol_l = self.fixed_risk_l * self.bal
        self.fixed_risk_dol_s = self.fixed_risk_s * self.bal
        self.counts_dict = {'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0, 'real_close_long': 0, 
                           'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0, 'sim_close_long': 0, 
                           'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0, 'real_close_short': 0, 
                           'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0, 'sim_close_short': 0, 
                           'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 'too_much_or': 0, 
                           'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0}
        
    def __str__(self):
        return self.name
    
    def sync_test_records(self):
        folder = Path(f"{self.market_data}/{self.name}")
        test_folder = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.name}')
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
            trades_path = Path(f'{self.market_data}/{self.name}/{switch}_trades.json')
            test_trades = Path(f'/home/ross/Documents/backtester_2021/test_records/{self.name}/{switch}_trades.json')
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
        
        sync_trades_records('open')
        sync_trades_records('sim')
        sync_trades_records('tracked')
        sync_trades_records('closed')
        sync_trades_records('closed_sim')
        
        # now that trade records have been loaded, path can be changed
        self.market_data = Path('/home/ross/Documents/backtester_2021/test_records')

    def mkt_data_path(self):
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
        
        return market_data
    
    def read_open_trade_records(self, switch):
        ot_path = Path(f"{self.market_data}/{self.name}")
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
    
            
        
        return open_trades

    def read_closed_trade_records(self):
        ct_path = Path(f"{self.market_data}/{self.name}/closed_trades.json")
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
        
        return closed_trades

    def read_closed_sim_trade_records(self):
        cs_path = Path(f"{self.market_data}/{self.name}/closed_sim_trades.json")
        if Path(cs_path).exists():
            with open(cs_path, "r") as cs_file:
                try:
                    closed_sim_trades = json.load(cs_file)
                except JSONDecodeError:
                    closed_sim_trades = {}
        
        else:
            closed_sim_trades = {}
            print(f'{cs_path} not found')
        
        return closed_sim_trades

    def backup_trade_records(self):
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        if self.open_trades:
            with open(f"{self.market_data}/{self.name}/ot_backup.json", "w") as ot_file:
                json.dump(self.open_trades, ot_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'open trades file empty')
        
        if self.sim_trades:
            with open(f"{self.market_data}/{self.name}/st_backup.json", "w") as st_file:
                json.dump(self.sim_trades, st_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'sim trades file empty')
        
        if self.tracked_trades:
            with open(f"{self.market_data}/{self.name}/tr_backup.json", "w") as tr_file:
                json.dump(self.tracked_trades, tr_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'tracked trades file empty')
        
        if self.closed_trades:
            with open(f"{self.market_data}/{self.name}/ct_backup.json", "w") as ct_file:
                json.dump(self.closed_trades, ct_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'closed trades file empty')
        
        if self.closed_sim_trades:
            with open(f"{self.market_data}/{self.name}/cs_backup.json", "w") as cs_file:
                json.dump(self.closed_sim_trades, cs_file)
        # else:
        #     if self.live:
        #         pb.push_note(now, 'closed sim trades file empty')
    
    def calc_tor(self):
        self.or_list = [v.get('or_R') for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
    
    def set_fixed_risk(self, direction:str):
        '''calculates fixed risk setting for new trades based on recent performance 
        and previous setting. if recent performance is very good, fr is increased slightly.
        if not, fr is decreased by thirds'''
        
        def reduce_fr(factor, fr_prev, fr_inc):
            '''reduces fixed_risk by factor (with the floor value being 0)'''
            ideal = fr_prev * factor
            reduce = max(ideal, fr_inc)
            return max((fr_prev-reduce), 0)
        
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        
        filepath = Path(f"{self.market_data}/{self.name}/bal_history.txt")
        if self.live:
            filepath.touch(exist_ok=True)
        with open(filepath, "r") as file:
            bal_data = file.readlines()
        
        if bal_data:
            fr_prev = json.loads(bal_data[-1]).get(f'fr_{direction}', 0) # default to 0 if no history
        else:
            fr_prev = 0
        fr_inc = self.fr_max / 10 # increment fr in 10% steps of the range
        
        def score_accum(direction:str, switch:str):
            with open(f"{self.market_data}/{self.name}/bal_history.txt", "r") as file:
                bal_data = file.readlines()
            
            if bal_data:
                prev_bal = json.loads(bal_data[-1]).get('balance')
            else:
                prev_bal = self.bal
            bal_change_pct = 100 * (self.bal - prev_bal) / prev_bal
            
            lookup = 'realised_pnl' if switch == 'real' else 'sim_r_pnl'
            if direction == 'long':
                lookup += '_long'
            else:
                lookup += '_short'
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
        
        if real_score > 0:
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
            note = f'fixed risk adjusted from {round(fr_prev*10000, 1)}bps to {round(fr*10000, 1)}bps'
            pb.push_note(now, note)
        
        return round(fr, 5)
    
    def set_max_pos(self):
        max_pos = 20
        if self.real_pos:
            open_pnls = [v.get('pnl') for v in self.real_pos.values() if v.get('pnl')]
            if open_pnls:
                avg_open_pnl = stats.median(open_pnls)
            else:
                avg_open_pnl = 0
            max_pos = 20 if avg_open_pnl <= 0 else 50
        
        return max_pos
    
    def current_positions(self, switch:str):
        '''creates a dictionary of open positions by checking either 
        open_trades.json, sim_trades.json or tracked_trades.json'''
            
        # filepath = Path(f'{self.market_data}/{self.name}/{switch}_trades.json')
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
                size_dict[asset] = uf.open_trade_stats(now, total_bal, v)
        
        return size_dict# -*- coding: utf-8 -*-

    def calc_pos_fr_dol(self, trade_record, direction, switch):   
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

    def set_in_pos(self, pair, switch):
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

    def init_in_pos(self, pair):
        self.in_pos = {'real':None, 'sim':None, 'tracked':None, 
                  'real_ep': None, 'sim_ep': None, 'tracked_ep': None, 
                  'real_hs': None, 'sim_hs': None, 'tracked_hs': None, 
                  'real_pfrd': None, 'sim_pfrd': None, 'tracked_pfrd': None}
        
        self.set_in_pos(pair, 'real')
        self.set_in_pos(pair, 'sim')
        self.set_in_pos(pair, 'tracked')
    
    def too_new(self, df):
        '''returns True if there is less than 200 hours of history AND if
        there are no current positions in the asset'''
        if self.in_pos['real'] or self.in_pos['sim'] or self.in_pos['tracked']:
            no_pos = False
        else:
            no_pos = True    
        return len(df) <= 200 and no_pos

    def open_pnl(self, switch):
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
        
        return total
    
    def tp_signals(self, asset):
        if self.real_pos.get(asset):
            real_or = self.real_pos.get(asset).get('or_R')
            self.in_pos['real_tp_sig'] = ((real_or > self.indiv_r_limit) and 
                                          (abs(self.in_pos.get('real_price_delta', 0)) > 0.001))
        if self.sim_pos.get(asset):
            sim_or = self.sim_pos.get(asset).get('or_R')
            self.in_pos['sim_tp_sig'] = ((sim_or > self.indiv_r_limit) and 
                                         (abs(self.in_pos.get('sim_price_delta', 0)) > 0.001))
    
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
        
        return {'signal': signal, 'inval': inval}




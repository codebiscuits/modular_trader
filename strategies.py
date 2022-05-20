import pandas as pd
import numpy as np
# import talib
from ta.momentum import RSIIndicator
import indicators as ind
import binance_funcs as funcs
import utility_funcs as uf
from pathlib import Path
import json
from json.decoder import JSONDecodeError
from datetime import datetime
from pushbullet import Pushbullet
import statistics as stats

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

### Backtesting Strategies

def ha_st_lo_bt(df):
    '''buy 1/3 position when supertrend turns green, increase to full position 
    if price bounces close to st line, take profit if price gets too far above 
    st line, sell when supertrend turns red. supertrend is based on heikin ashi 
    candles'''
    
    signals = [np.nan]
    s_buy = [np.nan]
    s_add = [np.nan]
    s_sell = [np.nan]
    s_tp = [np.nan]
    s_stop = [np.nan]
    # ratio_list = [1]
    sl_list = [np.nan]
    stop_price = 0 # initial stop, trails loosly behind supertrend
    exe_price = [df.at[0, 'close']] # for claculating pnl evolution
    pos_list = [0, 0] # create column which tracks when the strat is in a position
    in_pos = 0
    tp_ready = False
    buys = 0
    adds = 0
    sells = 0
    tps = 0
    stops = 0
    
    max_lev = 3
    leverage = max_lev / 3
    init_size = leverage# * 0.333
    tp_thresh = 1.05
    tp_amt = 0.5
    
    # loop through ohlc data and generate signals
    for i in range(1, len(df)):
        
        # ratio = df.ha_close[i] / df.st[i]
        # ratio_list.append(ratio)
        
        size_denom = (df.ratio[i] - 1) * 100
        
        stop_ratio = stop_price / df.st[i]
        if stop_ratio < 0.8:
            stop_price = df.st[i] * 0.8
            # print(f'hard stop raised to {stop_price:.3}')
            
        
        ### trade conditions
        init_entry = init_size / df.ratio[i] # adjust init_entry sizing for risk
        rising = df.ha_close[i] > df.ha_close[i-1]
        trend_up = df.ema200[i] > df.ema200[i-1]
        cross_up = (df.ha_close[i] > df.st[i])# and (df.ha_close[i-1] <= df.st[i-1])
        cross_down = (df.ha_close[i] < df.st[i])# and (df.ha_close[i-1] >= df.st[i-1])
        cross_down_tight = df.close[i] < df.st2[i]
        stop_loss = df.low[i] < stop_price
        low_risk = 1 <= df.ratio[i] < 1.1
        high_risk = df.ratio[i] > tp_thresh
        add_size = (0.5 * leverage / size_denom)
        
        buy_sig = trend_up and cross_up and not in_pos
        add_sig = trend_up and low_risk and rising and 0 < in_pos < (leverage / size_denom) # make sure adding would actually increase pos size
        sell_sig = cross_down and in_pos
        tp_sig = high_risk and in_pos
        tp_exe = tp_ready and cross_down_tight
        stop_sig = stop_loss and in_pos
        
        if buy_sig:
            stop_price = df.st[i] # init stop, trails loosly behind supertrend
            # print(f'init stop set at {stop_price:.3}')
            # signals.append(f'buy, init stop @ {stop_price}')
            s_buy.append(df.close[i])
            s_add.append(np.nan)
            s_sell.append(np.nan)
            s_tp.append(np.nan)
            s_stop.append(np.nan)
            signals.append('buy')
            exe_price.append(df.close[i])
            in_pos = init_entry
            pos_list.append(in_pos)
            buys += 1
            tp_ready = False
            # print(f'init entry, pos size: {in_pos:.3} (ratio: {ratio:.3})')
        elif add_sig: # add to position when r/r is good
            signals.append('add')
            s_buy.append(np.nan)
            s_add.append(df.close[i])
            s_sell.append(np.nan)
            s_tp.append(np.nan)
            s_stop.append(np.nan)
            exe_price.append(df.close[i])
            in_pos = min((pos_list[-1]+add_size), max_lev)
            pos_list.append(in_pos)
            adds += 1
            tp_ready = False
            # print(f'add, pos size: {in_pos:.3}')
        elif sell_sig: # sell in profit at trailing stop
            signals.append('sell')
            s_buy.append(np.nan)
            s_add.append(np.nan)
            s_sell.append(max(df.close[i], stop_price))
            s_tp.append(np.nan)
            s_stop.append(np.nan)
            exe_price.append(max(df.close[i], stop_price))
            in_pos = 0
            pos_list.append(in_pos)
            sells += 1
            tp_ready = False
            # print(f'sell, pos size: {in_pos}')
        elif tp_sig: # tighten trailing stop when r/r is bad
            signals.append('tp_sig')
            s_buy.append(np.nan)
            s_add.append(np.nan)
            s_sell.append(np.nan)
            s_tp.append(np.nan)
            s_stop.append(np.nan)
            exe_price.append(df.close[i])
            pos_list.append(pos_list[-1])
            tp_ready = True
        elif tp_exe: # take partial profit when tight trailing stop is hit
            signals.append('tp_exe')
            s_buy.append(np.nan)
            s_add.append(np.nan)
            s_sell.append(np.nan)
            s_tp.append(df.close[i])
            s_stop.append(np.nan)
            exe_price.append(df.close[i])
            in_pos = pos_list[-1] * (1 - tp_amt)
            pos_list.append(in_pos) # if tp_amt is 0.25 
            # this will reduce position size by 25%
            tps += 1
            tp_ready = False
            # print(f'take profit, pos size: {in_pos:.3}')
        elif stop_sig: # sell at a loss at initial stop
            signals.append('stop')
            s_buy.append(np.nan)
            s_add.append(np.nan)
            s_sell.append(np.nan)
            s_tp.append(np.nan)
            s_stop.append(stop_price)
            exe_price.append(stop_price)
            in_pos = 0
            pos_list.append(in_pos)
            stops += 1
            tp_ready = False
            # print(f'stopped out, pos size: {in_pos}')
        else:
            signals.append(np.nan)
            s_buy.append(np.nan)
            s_add.append(np.nan)
            s_sell.append(np.nan)
            s_tp.append(np.nan)
            s_stop.append(np.nan)
            exe_price.append(df.close[i])
            pos_list.append(pos_list[-1])

        if in_pos:
            sl_list.append(stop_price)
        else:
            sl_list.append(np.nan)

    # df['ratio'] = ratio_list
    df['signals'] = signals
    df['in_pos'] = pos_list[:-1]
    df['stop_loss'] = sl_list
    df['exe_price'] = exe_price
    df['exe_roc'] = df['exe_price'].pct_change()
    df['roc'] = df['close'].pct_change()
    
    # evo is calculated by keeping a running total of profit, and for each period,
    # proportionally adjusting the running total by the 'exe_roc' * pos value
    # so for any period where pos is 0, the running total will not change
    # for any period where pos is non-zero, the running total will change by an
    # amount proportional to the change in exe_price since the previous period
    # if the position was opened or closed in the previous period, the price change
    # since then should reflect the change between the current close and the 
    # price at which the trade was executed, hence why exe_price == close price
    # whenever signals == NaN
    
    evo = [1]
    hodl_evo = [1]
    for e in range(1, len(df.index)):
        evo.append(evo[-1] * (1 + (df.at[e, 'exe_roc'] * df.at[e, 'in_pos'])))
        hodl_evo.append(hodl_evo[-1] * (1 + (df.at[e, 'roc'])))
    df['pnl_evo'] = evo
    df['hodl_evo'] = hodl_evo
    
    # sb = buy signals, sse = sell signals, sst = stop signals
    # these are used for plotting buys/sells/stops on charts of single backtests
    sb, sad, sse, stp, sst = pd.Series(s_buy), pd.Series(s_add), pd.Series(s_sell), pd.Series(s_tp), pd.Series(s_stop)
    sb.index, sad.index, sse.index, stp.index, sst.index = df.index, df.index, df.index, df.index, df.index
    # buys, sells, and stops are counters
    return (buys, adds, sells, tps, stops), (sb, sad, sse, stp, sst)
    

def rsi_st_ema_lo_bt(df, buy_thresh, sell_thresh):
    '''during an uptrend as defined by 20ema being above 200ema and price 
    being above st line, set 'trade ready'. if rsi subsequently drops below and 
    then crosses back above x, trigger a buy.
    if rsi goes above y and then crosses back below OR if price closes below 
    st line, trigger a sell
    * only conditions that actually trigger a trade need to be defined as a 
    specific moment in time (eg a cross up or down), other conditions which set 
    the stage should be more diffuse (eg price is above supertrend line). if 
    there are several conditions which are only true on individual moments, 
    the chances of them lining up are much lower, ie good trades will be missed.
    im not looking for price crossing above the st line because that is not what 
    triggers a buy, i only care whether it is above or below, but a cross BELOW
    the st line will trigger a stop, so that condition must be a cross, not 
    just a simple less-than'''
    signals = [np.nan]
    s_buy = [np.nan]
    s_sell = [np.nan]
    s_stop = [np.nan]
    stop_price = 0
    trade_ready = 0
    in_pos = 0
    buys = 0
    sells = 0
    stops = 0
    for i in range(1, len(df)):
        ### trade conditions
        trend_up = (df.loc[i, '20ema'] > df.loc[i, '200ema'])
        st_up = (df.close[i] > df.st[i]) # not a trigger so doesnt need to be a cross
        cross_down = (df.close[i] < df.st[i]) and (df.close[i-1] >= df.st[i-1]) # this is a trigger so does need to be a cross
        rsi_buy = (df.rsi[i] >= buy_thresh) and (df.rsi[i-1] < buy_thresh)
        rsi_sell = (df.rsi[i] <= sell_thresh) and (df.rsi[i-1] > sell_thresh)
            
        if trend_up and st_up and in_pos == 0:
            trade_ready = 1
        else:
            trade_ready = 0
        
        if trade_ready == 1 and rsi_buy:
            sl = df.st[i] * 0.99 # init stop, doesn't change while in_pos
            signals.append(f'buy, init stop @ {sl}')
            stop_price = sl # trailing stop, changes any time st line moves while in_pos
            s_buy.append(df.close[i])
            s_sell.append(np.nan)
            s_stop.append(np.nan)
            in_pos = 1
            buys += 1
        elif in_pos and cross_down:
            signals.append('stop')
            s_buy.append(np.nan)
            s_sell.append(np.nan)
            s_stop.append(df.close[i-1])
            in_pos = 0
            stops += 1
        elif in_pos and rsi_sell:
            signals.append('sell')
            s_buy.append(np.nan)
            s_sell.append(df.close[i])
            s_stop.append(np.nan)
            in_pos = 0
            sells += 1
        else:
            signals.append(np.nan)
            s_buy.append(np.nan)
            s_sell.append(np.nan)
            s_stop.append(np.nan)

        if in_pos == 1 and ((df.st[i] * 0.99) > stop_price):
            stop_price = df.st[i] * 0.99
            
    
    
    df['signals'] = signals
    
    pos_list = [0, 0]
    for p in range(1, len(df.index)):
        if pd.isnull(df.at[p, 'signals']):
            pos_list.append(pos_list[-1])
        elif df.at[p, 'signals'][:3] == 'buy':
            pos_list.append(1)
        elif df.at[p, 'signals'] == 'sell':
            pos_list.append(0)
        elif df.at[p, 'signals'] == 'stop':
            pos_list.append(0)
        else:
            pos_list.append(pos_list[-1])
    
    df['in_pos'] = pos_list[:-1]
    
    exe_price = []
    for e in range(len(df.index)):
        if pd.isnull(df.at[e, 'signals']):
            exe_price.append(df.at[e, 'close'])
        elif df.at[e, 'signals'] == 'stop':
            exe_price.append(df.at[e-1, 'st'])
        else:
            exe_price.append(df.at[e, 'close'])
    df['exe_price'] = exe_price
    df['exe_roc'] = df['exe_price'].pct_change()
    df['roc'] = df['close'].pct_change()
    
    evo = [1]
    hodl_evo = [1]
    for e in range(1, len(df.index)):
        evo.append(evo[-1] * (1 + (df.at[e, 'exe_roc'] * df.at[e, 'in_pos'])))
        hodl_evo.append(hodl_evo[-1] * (1 + (df.at[e, 'roc'])))
    df['pnl_evo'] = evo
    df['hodl_evo'] = hodl_evo
    
    # print(df.drop(['open', 'high', 'low', 'volume', 'st_u', 'st_d', 
    #                '20ema', '200ema', 'rsi'], axis=1).tail())
        
    
    sb, sse, sst = pd.Series(s_buy), pd.Series(s_sell), pd.Series(s_stop)
    sb.index, sse.index, sst.index = df.index, df.index, df.index
    return buys, sells, stops, sb, sse, sst


### Live Trading Strategies

def ha_st_lo(df, in_pos, mult):
    '''buy when supertrend turns green, sell when supertrend turns red. 
    supertrend is based on heikin ashi candles, supertrend multiplier is set
    according to backtest results'''
    
    df = ind.heikin_ashi(df)
    
    df['st'], df['st_u'], df['st_d'] = ind.supertrend(df.ha_high, df.ha_low, df.ha_close, 10, mult)
    
    st_up = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st']
    st_down = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st']
    
    tp_long = None
    close_long = in_pos and st_down
    open_long = st_up and not in_pos
    
    inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, 'st']) # current price proportional to invalidation price
    
    return {'open_long': open_long, 'close_long': close_long, 
            'tp_long': tp_long, 'inval': inval}

class RSI_ST_EMA:
    
    description = '20/200ema cross and supertrend with rsi triggers'
    max_length = 250
    
    def __init__(self, len, os, ob):
        self.name = 'rsi_st_ema'
        self.len = len
        self.os = os
        self.ob = ob
        
    def __str__(self):
        return f'{self.name} rsi: {self.len}-{self.os}-{self.ob}'
    
    def live_signals(self, df, in_pos):
        df['20ema'] = df.close.ewm(20).mean()
        df['200ema'] = df.close.ewm(200).mean()
        ema_ratio = df.at[len(df)-1, '20ema'] / df.at[len(df)-1, '200ema']
        
        if ema_ratio < 1 and in_pos == 0: # this condition is just to save time
            return {'open_long': False, 'close_long': False, 
                    'tp_long': False, 'add_long': False, 
                    'open_short': False, 'close_short': False, 
                    'tp_short': False, 'add_short': False, 
                    'open_spot': False, 'close_spot': False, 
                    'tp_spot': False, 'add_spot': False, 
                    'inval': 1}
        else:    
            df['st'], df['st_u'], df['st_d'] = ind.supertrend(df.high, df.low, df.close, 10, 3)
            df['rsi'] = RSIIndicator(df.close, self.len).rsi()
            
            trend_up = df.at[len(df)-1, '20ema'] > df.at[len(df)-1, '200ema']
            st_up = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st'] # not a trigger so doesnt need to be a cross
            st_down = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st'] # this is a trigger so does need to be a cross
            rsi_buy = (df.at[len(df)-1, 'rsi'] >= self.os) and (df.at[len(df)-2, 'rsi'] < self.os)
            rsi_sell = (df.at[len(df)-1, 'rsi'] <= self.ob) and (df.at[len(df)-2, 'rsi'] > self.ob)
            
            tp_spot = in_pos and rsi_sell
            close_spot = in_pos and st_down
            open_spot = trend_up and st_up and rsi_buy and not in_pos
            # if open_long:
            #     ema20 = df.at[len(df)-1, '20ema']
            #     ema200 = df.at[len(df)-1, '200ema']
            #     print(f"20ema: {ema20:.3}, 200ema: {ema200:.3}, ratio: {ema20/ema200:.3}")
            
            inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, 'st']) # current price proportional to invalidation price
            
            return {'open_long': False, 'close_long': False, 
                    'tp_long': False, 'add_long': False, 
                    'open_short': False, 'close_short': False, 
                    'tp_short': False, 'add_short': False, 
                    'open_spot': open_spot, 'close_spot': close_spot, 
                    'tp_spot': tp_spot, 'add_spot': False, 
                    'inval': inval}

class DoubleST:
    name = 'double_st'
    description = 'regular supertrend for bias with tight supertrend for entries/exits'
    max_length = 201
    realised_pnl_long = 0
    realised_pnl_short = 0
    sim_pnl_long = 0
    sim_pnl_short = 0
    quote_asset = 'USDT'
    fr_range = (0, 0.0005) # 0.0025 makes good use of total balance
    max_spread = 0.5
    indiv_r_limit = 1.4
    total_r_limit = 20
    target_risk = 0.1
    max_pos = 20
    counts_dict = {'real_stop': 0, 'real_open': 0, 'real_add': 0, 'real_tp': 0, 'real_close': 0, 
                   'sim_stop': 0, 'sim_open': 0, 'sim_add': 0, 'sim_tp': 0, 'sim_close': 0, 
                   'real_stop_long': 0, 'real_open_long': 0, 'real_add_long': 0, 'real_tp_long': 0, 'real_close_long': 0, 
                   'sim_stop_long': 0, 'sim_open_long': 0, 'sim_add_long': 0, 'sim_tp_long': 0, 'sim_close_long': 0, 
                   'real_stop_short': 0, 'real_open_short': 0, 'real_add_short': 0, 'real_tp_short': 0, 'real_close_short': 0, 
                   'sim_stop_short': 0, 'sim_open_short': 0, 'sim_add_short': 0, 'sim_tp_short': 0, 'sim_close_short': 0, 
                   'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 'too_much_or': 0, 
                   'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0}
    
    def __init__(self, lb, mult):
        self.lb = lb
        self.mult = mult
        self.bal = funcs.account_bal_M()
        self.market_data = self.mkt_data_path()
        self.live = self.set_live()
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
        self.now_start = datetime.now().strftime('%d/%m/%y %H:%M')
        self.max_init_r_l = self.fixed_risk_l * self.total_r_limit
        self.max_init_r_s = self.fixed_risk_s * self.total_r_limit
        self.fixed_risk_dol_l = self.fixed_risk_l * self.bal
        self.fixed_risk_dol_s = self.fixed_risk_s * self.bal
        
        
    def __str__(self):
        return f'{self.name} st2: {self.lb}-{self.mult}'
    
    def spot_signals(self, df):
        
        df['ema200'] = df.close.ewm(200).mean()
        ind.supertrend_new(df, 10, 3)
        df.rename(columns={'st': 'st_loose', 'st_u': 'st_loose_u', 'st_d': 'st_loose_d'}, inplace=True)
        ind.supertrend_new(df, self.lb, self.mult)
        
        bullish_ema = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'ema200']
        bullish_loose = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st_loose']
        bullish_tight = df.at[len(df)-1, 'close'] > df.at[len(df)-1, 'st']
        bearish_tight = df.at[len(df)-1, 'close'] < df.at[len(df)-1, 'st']
        
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
    
    def margin_signals(self, df):
        
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
        
        # bullish_book = bid_ask_ratio > 1
        # bearish_book = bid_ask_ratio < 1
        # bullish_volume = price rising on low volume or price falling on high volume
        # bearish_volume = price rising on high volume or price falling on low volume
        
        if bullish_ema and bullish_loose and bullish_tight: # and bullish_book
            signal = 'open_long'
        if bearish_tight:
            signal = 'close_long'
        if bearish_ema and bearish_loose and bearish_tight: # and bearish_book
            signal = 'open_short'
        if bullish_tight:
            signal = 'close_short'
        
        if df.at[len(df)-1, 'st']:
            inval = float(df.at[len(df)-1, 'close'] / df.at[len(df)-1, 'st']) # current price proportional to invalidation price
        else:
            inval = 100000
            
        return {'signal': signal, 'inval': inval}
    
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
    
    def set_live(self):
        live = Path('/home/ubuntu/rpi_2.txt').exists()
        
        if live:
            print('-:-' * 20)
        else:
            print('*** Warning: Not Live ***')
        
        return live
    
    def sync_test_records(self):
        with open(f"{self.market_data}/{self.name}/bal_history.txt", "r") as file:
            bal_data = file.readlines()
        if bal_data:
            with open(f"test_records/{self.name}/bal_history.txt", "w") as file:
                file.writelines(bal_data)
        
        with open(f'{self.market_data}/binance_liquidity_history.txt', 'r') as file:
            book_data = file.readlines()
        if book_data:
            with open('test_records/binance_liquidity_history.txt', 'w') as file:
                file.writelines(book_data)
        
        def sync_trades_records(switch):        
            try:
                with open(f'{self.market_data}/{self.name}/{switch}_trades.json', 'r') as file:
                    data = json.load(file)
                if data:
                    with open(f'test_records/{self.name}/{switch}_trades.json', 'w') as file:
                        json.dump(data, file)
            except JSONDecodeError:
                print(f'{switch}_trades file empty')
        
        sync_trades_records('open')
        sync_trades_records('sim')
        sync_trades_records('tracked')
        sync_trades_records('closed')
        sync_trades_records('closed_sim')
        
        # now that trade records have been loaded, path can be changed
        self.market_data = Path('test_records')

    def read_open_trade_records(self, switch):
        ot_path = f"{self.market_data}/{self.name}/{switch}_trades.json"
        if Path(ot_path).exists():
            with open(ot_path, "r") as ot_file:
                try:
                    open_trades = json.load(ot_file)
                except JSONDecodeError:
                    open_trades = {}
        else:
            open_trades = {}
            print(f'{ot_path} not found')
        
        return open_trades

    def read_closed_trade_records(self):
        ct_path = f"{self.market_data}/{self.name}/closed_trades.json"
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
        
        return closed_trades

    def read_closed_sim_trade_records(self):
        cs_path = f"{self.market_data}/{self.name}/closed_sim_trades.json"
        if Path(cs_path).exists():
            with open(cs_path, "r") as cs_file:
                try:
                    closed_sim_trades = json.load(cs_file)
                except JSONDecodeError:
                    closed_sim_trades = {}
        
        return closed_sim_trades

    def backup_trade_records(self):
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        if self.open_trades:
            with open(f"{self.market_data}/{self.name}/ot_backup.json", "w") as ot_file:
                json.dump(self.open_trades, ot_file)
        else:
            if self.live:
                pb.push_note(now, 'open trades file empty')
        
        if self.sim_trades:
            with open(f"{self.market_data}/{self.name}/st_backup.json", "w") as st_file:
                json.dump(self.sim_trades, st_file)
        else:
            if self.live:
                pb.push_note(now, 'sim trades file empty')
        
        if self.tracked_trades:
            with open(f"{self.market_data}/{self.name}/tr_backup.json", "w") as tr_file:
                json.dump(self.tracked_trades, tr_file)
        else:
            if self.live:
                pb.push_note(now, 'tracked trades file empty')
        
        if self.closed_trades:
            with open(f"{self.market_data}/{self.name}/ct_backup.json", "w") as ct_file:
                json.dump(self.closed_trades, ct_file)
        else:
            if self.live:
                pb.push_note(now, 'closed trades file empty')
        
        if self.closed_sim_trades:
            with open(f"{self.market_data}/{self.name}/cs_backup.json", "w") as cs_file:
                json.dump(self.closed_sim_trades, cs_file)
        else:
            if self.live:
                pb.push_note(now, 'closed sim trades file empty')
    
    def calc_tor(self):
        self.or_list = [v.get('or_R') for v in self.real_pos.values() if v.get('or_R')]
        self.total_open_risk = sum(self.or_list)
        self.num_open_positions = len(self.or_list)
    
    def set_fixed_risk(self, direction:str):
        '''calculates fixed risk setting for new trades based on recent performance 
        and previous setting. if recent performance is very good, fr is increased slightly.
        if not, fr is decreased by thirds'''
        
        def reduce_fr(factor, fr_prev, fr_min, fr_inc):
            '''reduces fixed_risk by factor (with the floor value being fr_min)'''
            ideal = (fr_prev - fr_min) * factor
            reduce = max(ideal, fr_inc)
            return max((fr_prev-reduce), fr_min)
        
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        
        with open(f"{self.market_data}/{self.name}/bal_history.txt", "r") as file:
            bal_data = file.readlines()
        
        if bal_data:
            fr_prev = json.loads(bal_data[-1]).get(f'fr_{direction}', 0) # default to 0 if no history
        else:
            fr_prev = 0
        fr_min = self.fr_range[0]
        fr_max = self.fr_range[1]
        fr_inc = (fr_max - fr_min) / 10 # increment fr in 10% steps of the range
        
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
            
            score = 15
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
        
        if real_score:
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
        
        print('-')
        print(f'{direction} - {real_score = }, {sim_score = }, {bal_change_pct = }, {score = }')
        
        if score == 30:
            fr = min(fr_prev + (2*fr_inc), fr_max)
        elif score >= 26:
            fr = min(fr_prev + fr_inc, fr_max)
        elif score >= 18:
            fr = fr_prev
        elif score >= 12:
            fr = reduce_fr(0.333, fr_prev, fr_min, fr_inc)
        elif score >= 8:
            fr = reduce_fr(0.5, fr_prev, fr_min, fr_inc)
        else:
            fr = fr_min
            
        if fr != fr_prev:
            note = f'fixed risk adjusted from {round(fr_prev*10000, 1)}bps to {round(fr*10000, 1)}bps'
            pb.push_note(now, note)
        
        print(f'fixed risk perf score: {score}')
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
            
        filepath = Path(f'{self.market_data}/{self.name}/{switch}_trades.json')
        with open(filepath, 'r') as file:
            try:
                data = json.load(file)
            except:
                data = {}
        
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
        
        return size_dict




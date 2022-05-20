'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, time
from datetime import datetime
import binance_funcs as funcs
import strategies as strats
import order_management_funcs as omf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs
from pprint import pprint
import utility_funcs as uf
from random import shuffle

# setup
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
all_start = time.perf_counter()

# strat = strats.RSI_ST_EMA(4, 45, 96)
strat = strats.DoubleST(3, 1.2)

# update trade records --------------------------------------------------------
uf.record_stopped_trades(strat)
uf.record_stopped_sim_trades(strat)

# compile and sort list of pairs to loop through ------------------------------
all_pairs = funcs.get_pairs(market='CROSS')
shuffle(all_pairs)
positions = list(strat.real_pos.keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if (not p in pairs_in_pos) and (not p in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first
print(f'{len(pairs) = }')

print(f"Current time: {strat.now_start}, {strat}, fixed risk l: {strat.fixed_risk_l}, fixed risk s: {strat.fixed_risk_s}")

funcs.top_up_bnb_M(15)
spreads = funcs.binance_spreads('USDT') # dict

strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
if strat.max_positions > 20:
    print(f'max positions: {strat.max_positions}')
strat.calc_tor()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

for pair in pairs:
    asset = pair[:-1*len(strat.quote_asset)]
    
# set in_pos and look up or calculate $ fixed risk ----------------------------
    in_pos = {'real':None, 'sim':None, 'tracked':None, 
              'real_ep': None, 'sim_ep': None, 'tracked_ep': None, 
              'real_hs': None, 'sim_hs': None, 'tracked_hs': None, 
              'real_pfrd': None, 'sim_pfrd': None, 'tracked_pfrd': None}
    if asset in strat.real_pos.keys():
        real_trade_record = strat.open_trades.get(pair)
        if real_trade_record[0].get('type')[-4:] == 'long':
            in_pos['real'] = 'long'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(real_trade_record, strat.fixed_risk_dol_l, in_pos, 'real')
        else:
            in_pos['real'] = 'short'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(real_trade_record, strat.fixed_risk_dol_s, in_pos, 'real')
    
    if asset in strat.sim_pos.keys():
        sim_trade_record = strat.sim_trades.get(pair)
        if sim_trade_record[0].get('type')[-4:] == 'long':
            in_pos['sim'] = 'long'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(sim_trade_record, strat.fixed_risk_dol_l, in_pos, 'sim')
        else:
            in_pos['sim'] = 'short'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(sim_trade_record, strat.fixed_risk_dol_s, in_pos, 'sim')
    
    if asset in strat.tracked.keys():
        tracked_trade_record = strat.tracked_trades.get(pair)
        if tracked_trade_record[0].get('type')[-4:] == 'long':
            in_pos['tracked'] = 'long'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(tracked_trade_record, strat.fixed_risk_dol_l, in_pos, 'tracked')
        else:
            in_pos['tracked'] = 'short'
            # calculate dollar denominated fixed-risk per position
            in_pos = uf.calc_pos_fr_dol(tracked_trade_record, strat.fixed_risk_dol_s, in_pos, 'tracked')
        
    
# get data --------------------------------------------------------------------
    df = funcs.prepare_ohlc(pair, strat.live)
    
    if uf.too_new(df, in_pos):
        continue
    
    if len(df) > strat.max_length:
        df = df.tail(strat.max_length)
        df.reset_index(drop=True, inplace=True)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
# generate signals ------------------------------------------------------------
    # signals = strat.spot_signals(df)
    signals = strat.margin_signals(df)
    
    price = df.at[len(df)-1, 'close']
    st = df.at[len(df)-1, 'st']
    inval_dist = signals.get('inval')
    stp = funcs.calc_stop(st, spreads.get(pair), price)
    
    if st == 0:
        note = f'{pair} supertrend 0 error, skipping pair'
        print(note)
        push = pb.push_note(now, note)
        continue
    
# update positions dictionary with current pair's open_risk values ------------
    if in_pos['real']:
        real_qty = float(strat.real_pos[asset]['qty'])
        strat.real_pos[asset].update(funcs.update_pos_M(strat, asset, real_qty, inval_dist, in_pos['real'], in_pos['real_pfrd']))
        if in_pos['real_ep']:
            price_delta = (price - in_pos['real_ep']) / in_pos['real_ep'] # how much has price moved since entry
            
    if in_pos['sim']:
        sim_qty = float(strat.sim_pos[asset]['qty'])
        strat.sim_pos[asset].update(funcs.update_pos_M(strat, asset, sim_qty, inval_dist, in_pos['sim'], in_pos['sim_pfrd']))
        if in_pos['sim_ep'] and not in_pos['real_ep']:
            price_delta = (price - in_pos['sim_ep']) / in_pos['sim_ep']
    
# execute orders --------------------------------------------------------------
    if signals.get('signal') == 'spot_tp':
        try:
            in_pos = omf.spot_tp(strat, pair, in_pos, price, price_delta, 
                                 stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp order')
            continue
        
    elif signals.get('signal') == 'spot_close':
        try:
            in_pos = omf.spot_sell(strat, pair, in_pos, price)
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} sell order')
            continue
    
    elif signals.get('signal') == 'spot_open':        
        risk = (price - stp) / price
        mir = uf.max_init_risk(strat.num_open_positions, strat.target_risk)
        size, usdt_size = funcs.get_size(price, strat.fixed_risk, strat.bal, risk)
        usdt_depth = funcs.get_depth(pair, 'buy', strat.max_spread)
        
        if usdt_size > usdt_depth > (usdt_size / 2): # only trim size if books are a bit too thin
            strat.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
        
        if usdt_size < 30:
            strat.counts_dict['too_small'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'too_small', in_pos)
            continue
        elif risk > mir:
            strat.counts_dict['too_risky'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'too_risky', in_pos)
            continue
        elif usdt_depth == 0:
            strat.counts_dict['too_much_spread'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'too_much_spread', in_pos)
            continue
        elif usdt_depth < usdt_size:
            strat.counts_dict['books_too_thin'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'books_too_thin', in_pos)
            continue
                    
# check total open risk and close profitable positions if necessary -----------
        omf.reduce_risk(strat)
        strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
            
# make sure there aren't too many open positions now --------------------------
        strat.calc_tor()
        if strat.num_open_positions >= strat.max_positions:
            strat.counts_dict['too_many_pos'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'too_many_pos', in_pos)
            continue
        elif strat.total_open_risk > strat.total_r_limit:
            strat.counts_dict['too_much_or'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'too_much_or', in_pos)
            continue
        elif strat.real_pos['USDT']['qty'] < usdt_size:
            strat.counts_dict['not_enough_usdt'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'not_enough_usdt', in_pos)
            continue
            
# open new position -----------------------------------------------------------
        if not in_pos['real']:
            try:
                in_pos = omf.spot_buy(strat, pair, in_pos, size, usdt_size, 
                                  price, stp, inval_dist) 
            except BinanceAPIException as e:
                print(f'problem with buy order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} buy order')
                continue
            
# margin execution ------------------------------------------------------------
    elif signals.get('signal') == 'open_long':
        risk = (price - stp) / price
        mir = uf.max_init_risk(strat.num_open_positions, strat.target_risk)
        size, usdt_size = funcs.get_size(price, strat.fixed_risk_l, strat.bal, risk)
        usdt_depth = funcs.get_depth(pair, 'buy', strat.max_spread)
        
        if usdt_size > usdt_depth > (usdt_size / 2): # only trim size if books are a bit too thin
            strat.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
        sim_reason = None
        if usdt_size < 30:
            if not in_pos['sim']:
                strat.counts_dict['too_small'] += 1
            sim_reason = 'too_small'
        elif risk > mir:
            if not in_pos['sim']:
                strat.counts_dict['too_risky'] += 1
            sim_reason = 'too_risky'
        elif usdt_depth == 0:
            if not in_pos['sim']:
                strat.counts_dict['too_much_spread'] += 1
            sim_reason = 'too_much_spread'
        elif usdt_depth < usdt_size:
            if not in_pos['sim']:
                strat.counts_dict['books_too_thin'] += 1
            sim_reason = 'books_too_thin'
        
# check total open risk and close profitable positions if necessary -----------
        omf.reduce_risk_M(strat)
        strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
            
# make sure there aren't too many open positions now --------------------------
        strat.calc_tor()
        if strat.num_open_positions >= strat.max_positions:
            if not in_pos['sim']:
                strat.counts_dict['too_many_pos'] += 1
            sim_reason = 'too_many_pos'
        elif strat.total_open_risk > strat.total_r_limit:
            if not in_pos['sim']:
                strat.counts_dict['too_much_or'] += 1
            sim_reason = 'too_much_or'
        elif float(strat.real_pos['USDT']['qty']) < usdt_size:
            if not in_pos['sim']:
                strat.counts_dict['not_enough_usdt'] += 1
            sim_reason = 'not_enough_usdt'
        
        try:
            in_pos = omf.open_long(strat, in_pos, pair, size, stp, inval_dist, sim_reason)
        except BinanceAPIException as e:
            print(f'problem with open_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} open_long order')
            continue
    
    elif signals.get('signal') == 'tp_long':
        try:
            in_pos = omf.tp_long(strat, in_pos, pair, stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_long order')
            continue
    
    elif signals.get('signal') == 'close_long':
        try:
            in_pos = omf.close_long(strat, in_pos, pair)
        except BinanceAPIException as e:
            print(f'problem with close_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} close_long order')
            continue
    
    elif signals.get('signal') == 'open_short':
        risk = (stp - price) / price
        mir = uf.max_init_risk(strat.num_open_positions, strat.target_risk)
        size, usdt_size = funcs.get_size(price, strat.fixed_risk_s, strat.bal, risk)
        usdt_depth = funcs.get_depth(pair, 'sell', strat.max_spread)
        
        if usdt_size > usdt_depth > (usdt_size / 2): # only trim size if books are a bit too thin
            strat.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
        sim_reason = None
        if usdt_size < 30:
            if not in_pos['sim']:
                strat.counts_dict['too_small'] += 1
            sim_reason = 'too_small'
        elif risk > mir:
            if not in_pos['sim']:
                strat.counts_dict['too_risky'] += 1
            sim_reason = 'too_risky'
        elif usdt_depth == 0:
            if not in_pos['sim']:
                strat.counts_dict['too_much_spread'] += 1
            sim_reason = 'too_much_spread'
        elif usdt_depth < usdt_size:
            if not in_pos['sim']:
                strat.counts_dict['books_too_thin'] += 1
            sim_reason = 'books_too_thin'
        
# check total open risk and close profitable positions if necessary -----------
        omf.reduce_risk_M(strat)
        strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
            
# make sure there aren't too many open positions now --------------------------
        strat.calc_tor()
        if strat.num_open_positions >= strat.max_positions:
            if not in_pos['sim']:
                strat.counts_dict['too_many_pos'] += 1
            sim_reason = 'too_many_pos'
        elif strat.total_open_risk > strat.total_r_limit:
            if not in_pos['sim']:
                strat.counts_dict['too_much_or'] += 1
            sim_reason = 'too_much_or'
        elif float(strat.real_pos['USDT']['qty']) < usdt_size:
            if not in_pos['sim']:
                strat.counts_dict['not_enough_usdt'] += 1
            sim_reason = 'not_enough_usdt'
        
        try:
            in_pos = omf.open_short(strat, in_pos, pair, size, stp, inval_dist, sim_reason)
        except BinanceAPIException as e:
            print(f'problem with open_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} open_short order')
            continue
    
    elif signals.get('signal') == 'tp_short':
        try:
            in_pos = omf.tp_short()
        except BinanceAPIException as e:
            print(f'problem with tp_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_short order')
            continue
    
    elif signals.get('signal') == 'close_short':
        try:
            in_pos = omf.close_short(strat, in_pos, pair)
        except BinanceAPIException as e:
            print(f'problem with close_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} close_short order')
            continue
    
    

# calculate open risk and take profit if necessary ----------------------------
    if in_pos['real']:
        pos_bal = strat.real_pos.get(asset)['value']
        open_risk = strat.real_pos.get(asset)['or_$']
        open_risk_r = strat.real_pos.get(asset)['or_R']
        
        if open_risk_r > strat.indiv_r_limit and price_delta and (price_delta > 0.001):
            in_pos = omf.spot_tp(strat, pair, in_pos, price, price_delta, 
                                 stp, inval_dist)
        strat.calc_tor()
        
        
# log all data from the session and print/push summary-------------------------
strat.calc_tor()
print(f'realised real long pnl: {strat.realised_pnl_long:.1f}R, realised sim long pnl: {strat.sim_pnl_long:.1f}R')
print(f'realised real short pnl: {strat.realised_pnl_short:.1f}R, realised sim short pnl: {strat.sim_pnl_short:.1f}R')
print(f'tor: {strat.total_open_risk}')
print(f'or list: {[round(x, 2) for x in sorted(strat.or_list, reverse=True)]}')
strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)

benchmark = uf.log(strat, spreads)
if not strat.live:
    print('\n*** real_pos ***')
    pprint(strat.real_pos)
    # print('\n*** sim_pos ***')
    # pprint(strat.sim_pos)
    uf.interpret_benchmark(benchmark)
    print('warning: logging directed to test_records')

uf.scanner_summary(strat, all_start, benchmark)

print('Counts:')
for k, v in strat.counts_dict.items():
    if v:
        print(k, v)
print('-:-' * 20)
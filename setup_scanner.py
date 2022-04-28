'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, time
from datetime import datetime
import binance_funcs as funcs
import strategies as strats
import statistics as stats
import order_management_funcs as omf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs
from pprint import pprint
import utility_funcs as uf
import adaptive_funcs as af
from pathlib import Path
from random import shuffle

# setup
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
now_start = datetime.now().strftime('%d/%m/%y %H:%M')
all_start = time.perf_counter()

pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()
if live:
    print('-:-' * 20)
else:
    print('*** Warning: Not Live ***')

# strat = strats.RSI_ST_EMA(4, 45, 96)
strat = strats.DoubleSTLO(3, 1.2)

# update trade records --------------------------------------------------------
if not live:
    uf.sync_test_records(strat)
    # now that trade records have been loaded, path can be changed
    strat.market_data = Path('test_records')
uf.backup_trade_records(strat)

uf.record_stopped_trades(strat, now_start, live)
uf.record_stopped_sim_trades(strat, now_start)

# compile and sort list of pairs to loop through ------------------------------
all_pairs = funcs.get_pairs()
shuffle(all_pairs)
# strat.sizing = funcs.current_positions(strat, 'open')
# strat.sim_pos = funcs.current_positions(strat, 'sim')
# strat.tracked = funcs.current_positions(strat, 'tracked')
positions = list(strat.sizing.keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if (not p in pairs_in_pos) and (not p in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first


# set fixed risk, max_init_r and fixed_risk_dol
# fixed_risk = af.set_fixed_risk(strat, strat.bal)

# TODO these can be put in strat too
max_init_r = strat.fixed_risk * strat.total_r_limit
strat.fixed_risk_dol = strat.fixed_risk * strat.bal

print(f"Current time: {now_start}, {strat}, fixed risk: {strat.fixed_risk}")

funcs.top_up_bnb(15)
spreads = funcs.binance_spreads('USDT') # dict

strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
# max_positions = uf.set_max_pos(strat)
if strat.max_positions > 20:
    print(f'max positions: {strat.max_positions}')
# num_open_positions = len(pairs_in_pos)
strat.calc_tor()

for pair in pairs:
    asset = pair[:-1*len(strat.quote_asset)]
    
# set in_pos and look up or calculate $ fixed risk ----------------------------
    in_pos = {'real':False, 'sim':False, 'tracked':False, 
              'real_ep': None, 'sim_ep': None, 'tracked_ep': None, 
              'real_pfrd': None, 'sim_pfrd': None, 'tracked_pfrd': None}
    if asset in strat.sizing.keys():
        in_pos['real'] = True
        real_trade_record = strat.open_trades.get(pair)
        # calculate dollar denominated fixed-risk per position
        in_pos = uf.calc_pos_fr_dol(real_trade_record, strat.fixed_risk_dol, in_pos, 'real')
    if asset in strat.sim_pos.keys():
        in_pos['sim'] = True
        sim_trade_record = strat.sim_trades.get(pair)
        # calculate dollar denominated fixed-risk per position
        in_pos = uf.calc_pos_fr_dol(sim_trade_record, strat.fixed_risk_dol, in_pos, 'sim')
    if asset in strat.tracked.keys():
        in_pos['tracked'] = True
        tracked_trade_record = strat.tracked_trades.get(pair)
        # calculate dollar denominated fixed-risk per position
        in_pos = uf.calc_pos_fr_dol(tracked_trade_record, strat.fixed_risk_dol, in_pos, 'tracked')
        
    
# get data --------------------------------------------------------------------
    df = funcs.prepare_ohlc(pair, live)
    
    if uf.too_new(df, in_pos):
        continue
    
    if len(df) > strat.max_length:
        df = df.tail(strat.max_length)
        df.reset_index(drop=True, inplace=True)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
# generate signals ------------------------------------------------------------
    signals = strat.live_signals(df, in_pos)
    
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
        real_qty = strat.sizing[asset]['qty']
        strat.sizing[asset].update(funcs.update_pos(strat, asset, real_qty, inval_dist, in_pos['real_pfrd']))
        if in_pos['real_ep']:
            price_delta = (price - in_pos['real_ep']) / in_pos['real_ep'] # how much has price moved since entry
            
    if in_pos['sim']:
        sim_qty = strat.sim_pos[asset]['qty']
        strat.sim_pos[asset].update(funcs.update_pos(strat, asset, sim_qty, inval_dist, in_pos['sim_pfrd']))
        if in_pos['sim_ep'] and not in_pos['real_ep']:
            price_delta = (price - in_pos['sim_ep']) / in_pos['sim_ep']
    
# execute orders --------------------------------------------------------------
    if signals.get('signal') == 'tp':
        try:
            in_pos = omf.spot_tp(strat, pair, in_pos, price, price_delta, 
                                 stp, inval_dist, live)
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp order')
            continue
        
    elif signals.get('signal') == 'close':
        try:
            in_pos = omf.spot_sell(strat, pair, in_pos, price, live)
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} sell order')
            continue
    
    elif signals.get('signal') == 'open':        
        risk = (price - stp) / price
        mir = uf.max_init_risk(strat.num_open_positions, strat.target_risk)
        size, usdt_size = funcs.get_size(price, strat.fixed_risk, strat.bal, risk)
        usdt_depth = funcs.get_depth(pair, 'buy', strat.max_spread)
        
        if usdt_size > usdt_depth > (usdt_size / 2): # only trim size if books are a bit too thin
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
        omf.reduce_risk(strat, live)
        strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            
# make sure there aren't too many open positions now --------------------------
        # or_list = [v.get('or_R') for v in strat.sizing.values() if v.get('or_R')]
        # total_open_risk = sum(or_list)
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
        elif strat.sizing['USDT']['qty'] < usdt_size:
            strat.counts_dict['not_enough_usdt'] += 1
            if not in_pos['sim']:
                in_pos = omf.sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, 'not_enough_usdt', in_pos)
            continue
            
# open new position -----------------------------------------------------------
        if not in_pos['real']:
            try:
                in_pos = omf.spot_buy(strat, pair, in_pos, size, usdt_size, 
                                  price, stp, inval_dist, live) 
            except BinanceAPIException as e:
                print(f'problem with buy order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} buy order')
                continue
            
# margin execution ------------------------------------------------------------
    elif signals.get('open_long'):
        pass
    
    elif signals.get('tp_long'):
        pass
    
    elif signals.get('close_long'):
        pass
    
    elif signals.get('open_short'):
        pass
    
    elif signals.get('tp_short'):
        pass
    
    elif signals.get('close_short'):
        pass
    
    

# calculate open risk and take profit if necessary ----------------------------
    if in_pos['real']:
        pos_bal = strat.sizing.get(asset)['value']
        open_risk = strat.sizing.get(asset)['or_$']
        open_risk_r = strat.sizing.get(asset)['or_R']
        
        if open_risk_r > strat.indiv_r_limit and price_delta and (price_delta > 0.001):
            in_pos = omf.spot_tp(strat, pair, in_pos, price, price_delta, 
                                 stp, inval_dist, live)
        # or_list = [v.get('or_R') for v in strat.sizing.values() if v.get('or_R')]
        # total_open_risk = round(sum(or_list), 2)
        strat.calc_tor()
        
        
# log all data from the session and print/push summary-------------------------
strat.calc_tor()
print(f'realised real pnl: {strat.realised_pnl:.1f}R, realised sim pnl: {strat.sim_pnl:.1f}R')
print(f'tor: {strat.total_open_risk}')
print(f'or list: {[round(x, 2) for x in sorted(strat.or_list, reverse=True)]}')
strat.sizing['USDT'] = funcs.update_usdt(strat.bal)

benchmark = uf.log(live, strat, strat.fixed_risk, spreads, now_start)
if not live:
    print('*** sizing ***')
    pprint(strat.sizing)
    # print('*** sim pos ***')
    # pprint(strat.sim_pos)
    uf.interpret_benchmark(benchmark)
    print('warning: logging directed to test_records')

uf.scanner_summary(strat, all_start, benchmark, live)

pprint(strat.counts_dict)
print('-:-' * 20)
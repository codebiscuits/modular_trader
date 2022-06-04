'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, time
from datetime import datetime
import binance_funcs as funcs
from agents import DoubleST
import order_management_funcs as omf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs
from pprint import pprint
import utility_funcs as uf
from random import shuffle
import sessions
from pathlib import Path
# import argparse

# setup
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
all_start = time.perf_counter()


# parser = argparse.ArgumentParser()
# parser.add_argument('strat', metavar='strategy', type=str, 
#                     help='choose the strategy to run', 
#                     required = True)
# parser.add_argument('tf', metavar='timeframe', type=str, 
#                     help='choose from 1h, 4h, 6h, 12h, 1d', 
#                     required = True)
# parser.add_argument('param_1', metavar='parameter 1', type=int, 
#                     help='input the value for the first parameter', 
#                     required = True)
# parser.add_argument('param_2', metavar='parameter 2', type=float, 
#                     help='input the value for the second parameter', 
#                     required = False)
# parser.add_argument('param_3', metavar='parameter 3', type=float, 
#                     help='input the value for the third parameter', 
#                     required = False)
# args = parser.parse_args()


dst_presets = {1: ('4h', 0, 1, 1.0), 
           2: ('4h', 0, 2, 1.1), 
           3: ('4h', 0, 3, 1.2), 
           4: ('4h', 0, 3, 1.4), 
           5: ('4h', 0, 4, 1.6), 
           6: ('4h', 0, 4, 1.8), 
           7: ('4h', 0, 5, 2.0), 
           8: ('4h', 0, 5, 2.2), 
           9: ('4h', 0, 5, 2.4), 
           10: ('4h', 0, 6, 2.6), 
           11: ('4h', 0, 6, 2.8), 
           12: ('4h', 0, 6, 3.0), 
           }




# session = strats.RSI_ST_EMA(4, 45, 96)
session = sessions.MARGIN_SESSION()
agent_1 = DoubleST(session, *dst_presets[3])
agents = [agent_1]
session.name = '-'.join([n.name for n in agents])

# now that trade records have been loaded, path can be changed
if not session.live:
    session.market_data = Path('/home/ross/Documents/backtester_2021/test_records')

# update trade records --------------------------------------------------------
for agent in agents:
    print('-')
    uf.record_stopped_trades(session, agent)
    uf.record_stopped_sim_trades(session, agent)
    agent.real_pos = agent.current_positions('open')
    agent.sim_pos = agent.current_positions('sim')
    agent.tracked = agent.current_positions('tracked')
    
    print(f'{agent.realised_pnl_long = }')
    print(f'{agent.realised_pnl_short = }')
    print(f'{agent.sim_pnl_long = }')
    print(f'{agent.sim_pnl_short = }')
    print('-')

# compile and sort list of pairs to loop through ------------------------------
all_pairs = funcs.get_pairs(market='CROSS')
shuffle(all_pairs)
positions = list(agent_1.real_pos.keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if (not p in pairs_in_pos) and (not p in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first

print(f"Current time: {session.now_start}, {session.name}")
for agent in agents:
    print(f"{agent.name} fixed risk l: {agent.fixed_risk_l}, fixed risk s: {agent.fixed_risk_s}")

funcs.top_up_bnb_M(15)
session.spreads = funcs.binance_spreads('USDT') # dict

for agent in agents:
    agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
    if agent.max_positions > 20:
        print(f'max positions: {agent.max_positions}')
    agent.calc_tor()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

for n, pair in enumerate(pairs):
    print('\n-\n')
    asset = pair[:-1*len(session.quote_asset)]
    for agent in agents:
        agent.init_in_pos(pair)    
    df = funcs.prepare_ohlc(pair, session.live)
    
    if len(df) < 10:
        continue
    
    if len(df) > session.max_length:
        df = df.tail(session.max_length)
        df.reset_index(drop=True, inplace=True)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
# generate signals ------------------------------------------------------------
    signals = agent_1.margin_signals(session, df, pair)
    
    price = df.at[len(df)-1, 'close']
    st = df.at[len(df)-1, 'st']
    inval_dist = signals.get('inval')
    stp = funcs.calc_stop(st, session.spreads.get(pair), price)
    risk = (price - stp) / price
    mir = uf.max_init_risk(agent_1.num_open_positions, agent_1.target_risk)
    size_l, usdt_size_l, size_s, usdt_size_s = funcs.get_size(agent_1, price, session.bal, risk)
    usdt_depth_l, usdt_depth_s = funcs.get_depth(pair, session.max_spread)
    
    if st == 0:
        note = f'{pair} supertrend 0 error, skipping pair'
        print(note)
        push = pb.push_note(now, note)
        continue
    
# update positions dictionary with current pair's open_risk values ------------
    if agent_1.in_pos['real']:
        real_qty = float(agent_1.real_pos[asset]['qty'])
        agent_1.real_pos[asset].update(funcs.update_pos_M(session, asset, real_qty, inval_dist, agent_1.in_pos['real'], agent_1.in_pos['real_pfrd']))
        if agent_1.in_pos['real_ep']:
            agent_1.in_pos['real_price_delta'] = (price - agent_1.in_pos['real_ep']) / agent_1.in_pos['real_ep'] # how much has price moved since entry
            
    if agent_1.in_pos['sim']:
        sim_qty = float(agent_1.sim_pos[asset]['qty'])
        agent_1.sim_pos[asset].update(funcs.update_pos_M(session, asset, sim_qty, inval_dist, agent_1.in_pos['sim'], agent_1.in_pos['sim_pfrd']))
        if agent_1.in_pos['sim_ep']:
            agent_1.in_pos['sim_price_delta'] = (price - agent_1.in_pos['sim_ep']) / agent_1.in_pos['sim_ep']
        
            
# margin order execution ------------------------------------------------------
    if signals.get('signal') == 'open_long':
        # risk = (price - stp) / price
        # mir = uf.max_init_risk(agent_1.num_open_positions, agent_1.target_risk)
        # size, usdt_size = funcs.get_size(price, agent_1.fixed_risk_l, session.bal, risk)
        # usdt_depth = funcs.get_depth(pair, 'buy', session.max_spread)
        
        if usdt_size_l > usdt_depth_l > (usdt_size_l / 2): # only trim size if books are a bit too thin
            agent_1.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size_l:.3} to {usdt_depth_l:.3}'
            print(trim_size)
            usdt_size_l = usdt_depth_l
        sim_reason = None
        if usdt_size_l < 30:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_small'] += 1
            sim_reason = 'too_small'
        elif risk > mir:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_risky'] += 1
            sim_reason = 'too_risky'
        elif usdt_depth_l == 0:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_much_spread'] += 1
            sim_reason = 'too_much_spread'
        elif usdt_depth_l < usdt_size_l:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['books_too_thin'] += 1
            sim_reason = 'books_too_thin'
        
# check total open risk and close profitable positions if necessary -----------
        omf.reduce_risk_M(session, agent_1)
        agent_1.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
            
# make sure there aren't too many open positions now --------------------------
        agent_1.calc_tor()
        if agent_1.num_open_positions >= agent_1.max_positions:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_many_pos'] += 1
            sim_reason = 'too_many_pos'
        elif agent_1.total_open_risk > agent_1.total_r_limit:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_much_or'] += 1
            sim_reason = 'too_much_or'
        elif float(agent_1.real_pos['USDT']['qty']) < usdt_size_l:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['not_enough_usdt'] += 1
            sim_reason = 'not_enough_usdt'
        
        try:
            omf.open_long(session, agent_1, pair, size_l, stp, inval_dist, sim_reason)
        except BinanceAPIException as e:
            print(f'problem with open_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} open_long order')
            continue
    
    elif signals.get('signal') == 'tp_long':
        try:
            omf.tp_long(session, agent_1, pair, stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_long order')
            continue
    
    elif signals.get('signal') == 'close_long':
        try:
            omf.close_long(session, agent_1, pair)
        except BinanceAPIException as e:
            print(f'problem with close_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} close_long order')
            continue
    
    elif signals.get('signal') == 'open_short':
        # risk = (stp - price) / price
        # mir = uf.max_init_risk(agent_1.num_open_positions, agent_1.target_risk)
        # size, usdt_size = funcs.get_size(price, agent_1.fixed_risk_s, session.bal, risk)
        # usdt_depth = funcs.get_depth(pair, 'sell', session.max_spread)
        
        if usdt_size_s > usdt_depth_s > (usdt_size_s / 2): # only trim size if books are a bit too thin
            agent_1.counts_dict['books_too_thin'] += 1
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size_s:.3} to {usdt_depth_s:.3}'
            print(trim_size)
            usdt_size_s = usdt_depth_s
        sim_reason = None
        if usdt_size_s < 30:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_small'] += 1
            sim_reason = 'too_small'
        elif risk > mir:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_risky'] += 1
            sim_reason = 'too_risky'
        elif usdt_depth_s == 0:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_much_spread'] += 1
            sim_reason = 'too_much_spread'
        elif usdt_depth_s < usdt_size_s:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['books_too_thin'] += 1
            sim_reason = 'books_too_thin'
        
# check total open risk and close profitable positions if necessary -----------
        omf.reduce_risk_M(session, agent_1)
        agent_1.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
            
# make sure there aren't too many open positions now --------------------------
        agent_1.calc_tor()
        if agent_1.num_open_positions >= agent_1.max_positions:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_many_pos'] += 1
            sim_reason = 'too_many_pos'
        elif agent_1.total_open_risk > agent_1.total_r_limit:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['too_much_or'] += 1
            sim_reason = 'too_much_or'
        elif float(agent_1.real_pos['USDT']['qty']) < usdt_size_s:
            if not agent_1.in_pos['sim']:
                agent_1.counts_dict['not_enough_usdt'] += 1
            sim_reason = 'not_enough_usdt'
        
        try:
            omf.open_short(session, agent_1, pair, size_s, stp, inval_dist, sim_reason)
        except BinanceAPIException as e:
            print(f'problem with open_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} open_short order')
            continue
    
    elif signals.get('signal') == 'tp_short':
        try:
            omf.tp_short(session, agent_1, pair, stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_short order')
            continue
    
    elif signals.get('signal') == 'close_short':
        try:
            omf.close_short(session, agent_1, pair)
        except BinanceAPIException as e:
            print(f'problem with close_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} close_short order')
            continue
    
    

# calculate open risk and take profit if necessary ----------------------------
    agent_1.tp_signals(asset)
    if agent_1.in_pos.get('real_tp_sig') or agent_1.in_pos.get('sim_tp_sig'):
        pprint(agent_1.in_pos)
    if agent_1.in_pos['real'] == 'long' or agent_1.in_pos['sim'] == 'long':
        try:
            omf.tp_long(session, agent_1, pair, stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp_long order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_long order')
            continue
    elif agent_1.in_pos['real'] == 'short' or agent_1.in_pos['sim'] == 'short':
        try:
            omf.tp_short(session, agent_1, pair, stp, inval_dist)
        except BinanceAPIException as e:
            print(f'problem with tp_short order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp_short order')
            continue
    
    agent_1.calc_tor()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#   
        
# log all data from the session and print/push summary-------------------------
print(f'realised real long pnl: {agent_1.realised_pnl_long:.1f}R, realised sim long pnl: {agent_1.sim_pnl_long:.1f}R')
print(f'realised real short pnl: {agent_1.realised_pnl_short:.1f}R, realised sim short pnl: {agent_1.sim_pnl_short:.1f}R')
print(f'tor: {agent_1.total_open_risk}')
print(f'or list: {[round(x, 2) for x in sorted(agent_1.or_list, reverse=True)]}')
print(f"real open pnl: {agent_1.open_pnl('real'):.1f}R")
print(f"sim open pnl: {agent_1.open_pnl('sim'):.1f}R")
agent_1.real_pos['USDT'] = funcs.update_usdt_M(session.bal)

benchmark = uf.log(session, [agent_1])
if not session.live:
    print('\n*** real_pos ***')
    pprint(agent_1.real_pos)
    print('\n*** sim_pos ***')
    pprint(agent_1.sim_pos.keys())
    uf.interpret_benchmark(session, [agent_1])
    print('warning: logging directed to test_records')

uf.scanner_summary(session, [agent_1])

print('Counts:')
for agent in [agent_1]:
    for k, v in agent.counts_dict.items():
        if v:
            print(k, v)
    print('-')
print('-:-' * 20)
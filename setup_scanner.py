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
from timers import Timer
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
           4: ('4h', 0, 3, 1.3), 
           5: ('4h', 0, 4, 1.4), 
           6: ('4h', 0, 4, 1.5), 
           7: ('4h', 0, 5, 1.6), 
           8: ('4h', 0, 5, 1.7), 
           9: ('4h', 0, 5, 1.8), 
           10: ('4h', 0, 6, 1.9), 
           11: ('4h', 0, 6, 2.0), 
           }


# session = strats.RSI_ST_EMA(4, 45, 96)
session = sessions.MARGIN_SESSION()
agent_1 = DoubleST(session, *dst_presets[3])
agent_2 = DoubleST(session, *dst_presets[7])
agent_3 = DoubleST(session, *dst_presets[11])
agents = [agent_1, agent_2, agent_3]
session.name = '-'.join([n.name for n in agents])

# now that trade records have been loaded, path can be changed
# if not session.live:
#     session.market_data = Path('/home/ross/Documents/backtester_2021/test_records')

# update trade records --------------------------------------------------------
# for agent in agents:
#     uf.record_stopped_trades(session, agent)
#     uf.record_stopped_sim_trades(session, agent)
#     agent.real_pos = agent.current_positions('open')
#     agent.sim_pos = agent.current_positions('sim')
#     agent.tracked = agent.current_positions('tracked')

# compile and sort list of pairs to loop through ------------------------------
all_pairs = funcs.get_pairs(market='CROSS')
shuffle(all_pairs)
positions = []
for agent in agents:
    posis = list(agent.real_pos.keys())
    positions.extend(posis)
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if (not p in pairs_in_pos) and (not p in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first

# pairs = pairs[:10] ############# delete when testing is finished #############

print(f"Current time: {session.now_start}, {session.name}")
for agent in agents:
    print(f"{agent.name} fr long: {agent.fixed_risk_l}, fr short: {agent.fixed_risk_s}")

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
    # print(pair, 'main loop')
    session.prices[pair] = funcs.get_price(pair)
    asset = pair[:-1*len(session.quote_asset)]
    for agent in agents:
        agent.init_in_pos(pair)    
    df = funcs.prepare_ohlc(pair, session.live)
    
    too_new = 0
    for agent in agents:
        if agent.too_new(df):
            too_new += 1
    if too_new == len(agents):
        continue
    
    if len(df) > session.max_length:
        df = df.tail(session.max_length)
        df.reset_index(drop=True, inplace=True)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
# generate signals ------------------------------------------------------------
    signals = {}
    for agent in agents:
        # print('*****', agent.name)
        signals = agent.margin_signals(session, df, pair)
    
        price = df.at[len(df)-1, 'close']
        st = df.at[len(df)-1, 'st']
        inval_dist = signals.get('inval')
        stp = funcs.calc_stop(st, session.spreads.get(pair), price)
        risk = (price - stp) / price
        mir = uf.max_init_risk(agent.num_open_positions, agent.target_risk)
        size_l, usdt_size_l, size_s, usdt_size_s = funcs.get_size(agent, price, session.bal, risk)
        usdt_depth_l, usdt_depth_s = funcs.get_depth(session, pair)
        
        df.drop(columns=['st_loose', 'st_loose_u', 'st_loose_d', 'st', 'st_u', 'st_d'], inplace=True)
        
        if st == 0:
            note = f'{pair} supertrend 0 error, skipping pair'
            print(note)
            push = pb.push_note(now, note)
            continue
    
# update positions dictionary with current pair's open_risk values ------------
        if agent.in_pos['real']:
            real_qty = float(agent.real_pos[asset]['qty'])
            agent.real_pos[asset].update(funcs.update_pos_M(session, asset, real_qty, inval_dist, agent.in_pos['real'], agent.in_pos['real_pfrd']))
            if agent.in_pos['real_ep']:
                agent.in_pos['real_price_delta'] = (price - agent.in_pos['real_ep']) / agent.in_pos['real_ep'] # how much has price moved since entry
                
        if agent.in_pos['sim']:
            sim_qty = float(agent.sim_pos[asset]['qty'])
            agent.sim_pos[asset].update(funcs.update_pos_M(session, asset, sim_qty, inval_dist, agent.in_pos['sim'], agent.in_pos['sim_pfrd']))
            if agent.in_pos['sim_ep']:
                agent.in_pos['sim_price_delta'] = (price - agent.in_pos['sim_ep']) / agent.in_pos['sim_ep']
        
            
# margin order execution ------------------------------------------------------
        if signals.get('signal') == 'open_long':
            # risk = (price - stp) / price
            # mir = uf.max_init_risk(agent.num_open_positions, agent.target_risk)
            # size, usdt_size = funcs.get_size(price, agent.fixed_risk_l, session.bal, risk)
            # usdt_depth = funcs.get_depth(pair, 'buy', session.max_spread)
            
            if usdt_size_l > usdt_depth_l > (usdt_size_l / 2): # only trim size if books are a bit too thin
                agent.counts_dict['books_too_thin'] += 1
                trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size_l:.3} to {usdt_depth_l:.3}'
                print(trim_size)
                usdt_size_l = usdt_depth_l
            sim_reason = None
            if usdt_size_l < 30:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_small'] += 1
                sim_reason = 'too_small'
            elif risk > mir:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_risky'] += 1
                sim_reason = 'too_risky'
            elif usdt_depth_l == 0:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_much_spread'] += 1
                sim_reason = 'too_much_spread'
            elif usdt_depth_l < usdt_size_l:
                if not agent.in_pos['sim']:
                    agent.counts_dict['books_too_thin'] += 1
                sim_reason = 'books_too_thin'
            
    # check total open risk and close profitable positions if necessary -----------
            omf.reduce_risk_M(session, agent)
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
                
    # make sure there aren't too many open positions now --------------------------
            agent.calc_tor()
            if agent.num_open_positions >= agent.max_positions:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_many_pos'] += 1
                sim_reason = 'too_many_pos'
            elif agent.total_open_risk > agent.total_r_limit:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_much_or'] += 1
                sim_reason = 'too_much_or'
            elif float(agent.real_pos['USDT']['qty']) < usdt_size_l:
                if not agent.in_pos['sim']:
                    agent.counts_dict['not_enough_usdt'] += 1
                sim_reason = 'not_enough_usdt'
            
            try:
                omf.open_long(session, agent, pair, size_l, stp, inval_dist, sim_reason)
            except BinanceAPIException as e:
                print(f'problem with open_long order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} open_long order')
                continue
        
        elif signals.get('signal') == 'tp_long':
            try:
                omf.tp_long(session, agent, pair, stp, inval_dist)
            except BinanceAPIException as e:
                print(f'problem with tp_long order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} tp_long order')
                continue
        
        elif signals.get('signal') == 'close_long':
            try:
                omf.close_long(session, agent, pair)
            except BinanceAPIException as e:
                print(f'problem with close_long order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} close_long order')
                continue
        
        elif signals.get('signal') == 'open_short':
            # risk = (stp - price) / price
            # mir = uf.max_init_risk(agent.num_open_positions, agent.target_risk)
            # size, usdt_size = funcs.get_size(price, agent.fixed_risk_s, session.bal, risk)
            # usdt_depth = funcs.get_depth(pair, 'sell', session.max_spread)
            
            if usdt_size_s > usdt_depth_s > (usdt_size_s / 2): # only trim size if books are a bit too thin
                agent.counts_dict['books_too_thin'] += 1
                trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size_s:.3} to {usdt_depth_s:.3}'
                print(trim_size)
                usdt_size_s = usdt_depth_s
            sim_reason = None
            if usdt_size_s < 30:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_small'] += 1
                sim_reason = 'too_small'
            elif risk > mir:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_risky'] += 1
                sim_reason = 'too_risky'
            elif usdt_depth_s == 0:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_much_spread'] += 1
                sim_reason = 'too_much_spread'
            elif usdt_depth_s < usdt_size_s:
                if not agent.in_pos['sim']:
                    agent.counts_dict['books_too_thin'] += 1
                sim_reason = 'books_too_thin'
            
    # check total open risk and close profitable positions if necessary -----------
            omf.reduce_risk_M(session, agent)
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
                
    # make sure there aren't too many open positions now --------------------------
            agent.calc_tor()
            if agent.num_open_positions >= agent.max_positions:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_many_pos'] += 1
                sim_reason = 'too_many_pos'
            elif agent.total_open_risk > agent.total_r_limit:
                if not agent.in_pos['sim']:
                    agent.counts_dict['too_much_or'] += 1
                sim_reason = 'too_much_or'
            elif float(agent.real_pos['USDT']['qty']) < usdt_size_s:
                if not agent.in_pos['sim']:
                    agent.counts_dict['not_enough_usdt'] += 1
                sim_reason = 'not_enough_usdt'
            
            try:
                omf.open_short(session, agent, pair, size_s, stp, inval_dist, sim_reason)
            except BinanceAPIException as e:
                print(f'problem with open_short order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} open_short order')
                continue
        
        elif signals.get('signal') == 'tp_short':
            try:
                omf.tp_short(session, agent, pair, stp, inval_dist)
            except BinanceAPIException as e:
                print(f'problem with tp_short order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} tp_short order')
                continue
        
        elif signals.get('signal') == 'close_short':
            try:
                omf.close_short(session, agent, pair)
            except BinanceAPIException as e:
                print(f'problem with close_short order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} close_short order')
                continue
        
    

# calculate open risk and take profit if necessary ----------------------------
        agent.tp_signals(asset)
        if agent.in_pos['real'] == 'long' or agent.in_pos['sim'] == 'long':
            try:
                omf.tp_long(session, agent, pair, stp, inval_dist)
            except BinanceAPIException as e:
                print(f'problem with tp_long order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} tp_long order')
                continue
        elif agent.in_pos['real'] == 'short' or agent.in_pos['sim'] == 'short':
            try:
                omf.tp_short(session, agent, pair, stp, inval_dist)
            except BinanceAPIException as e:
                print(f'problem with tp_short order for {pair}')
                print(e)
                pb.push_note(now, f'exeption during {pair} tp_short order')
                continue
        
        agent.calc_tor()

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#   
        
# log all data from the session and print/push summary-------------------------
print('-:-' * 20)
for agent in agents:
    print(agent.name, 'summary')
    print(f'realised real long pnl: {agent.realised_pnl_long:.1f}R, realised sim long pnl: {agent.sim_pnl_long:.1f}R')
    print(f'realised real short pnl: {agent.realised_pnl_short:.1f}R, realised sim short pnl: {agent.sim_pnl_short:.1f}R')
    print(f'tor: {agent.total_open_risk}')
    print(f'or list: {[round(x, 2) for x in sorted(agent.or_list, reverse=True)]}')
    print(f"real open pnl: {agent.open_pnl('real'):.1f}R")
    print(f"sim open pnl: {agent.open_pnl('sim'):.1f}R")
    agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
    
    benchmark = uf.log(session, [agent])
    if not session.live:
        print('\n*** real_pos ***')
        pprint(agent.real_pos)
        print('\n*** sim_pos ***')
        pprint(agent.sim_pos.keys())
        uf.interpret_benchmark(session, [agent])
        print('warning: logging directed to test_records')
    
    uf.scanner_summary(session, [agent])
    
    print('Counts:')
    for agent in [agent]:
        for k, v in agent.counts_dict.items():
            if v:
                print(k, v)
        print('-')
    print('-:-' * 20)

for k, v in Timer.timers.items():
    if v > 30:
        print(k, round(v))

end = time.perf_counter()
elapsed = round(end - all_start)
print(f'Total time taken: {elapsed//60}m, {elapsed%60}s')


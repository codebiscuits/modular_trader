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
from config import not_pairs, params, market_data
from pprint import pprint
import utility_funcs as uf
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
# strat = strats.RSI_ST_EMA(4, 45, 96)
strat = strats.DoubleSTLO(3, 1.2)

pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()
if live:
    print('-:-' * 20)
else:
    print('*** Warning: Not Live ***')


# compile and sort list of pairs to loop through ------------------------------
all_pairs = funcs.get_pairs()
shuffle(all_pairs)
positions = list(funcs.current_positions(strat.name, params.get('fr_range')[0]).keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if (not p in pairs_in_pos) and (not p in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first


# counts_dict = {'stop_count': 0, 'open_count': 0, 'add_count': 0, 'tp_count': 0, 'close_count': 0, 
#                'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 
#                'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0}

# update trade records --------------------------------------------------------
open_trades, closed_trades, next_id = uf.read_trade_records(market_data, strat.name)
if not live:
    uf.sync_test_records(strat, market_data)
    # now that trade records have been loaded, path can be changed
    market_data = Path('test_records')
uf.backup_trade_records(strat.name, market_data, open_trades, closed_trades)
next_id = uf.record_stopped_trades(open_trades, closed_trades, 
                                                pairs_in_pos, now_start, 
                                                next_id, strat, 
                                                market_data)

# set fixed risk
total_bal = funcs.account_bal()
fixed_risk = uf.set_fixed_risk(strat, market_data, total_bal)
max_init_r = fixed_risk * params.get('total_r_limit')
fixed_risk_dol = fixed_risk * strat.bal

print(f"Current time: {now_start}, {strat}, fixed risk: {fixed_risk}")

funcs.top_up_bnb(15)
spreads = funcs.binance_spreads('USDT') # dict

strat.sizing = funcs.current_positions(strat.name, fixed_risk)
strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
max_positions = uf.set_max_pos(strat.sizing, params)
if max_positions > 20:
    print(f'{max_positions = }')
num_open_positions = len(pairs_in_pos)

for pair in pairs:
    asset = pair[:-1*len(params.get('quote_asset'))]
    in_pos = asset in strat.sizing.keys()
    
# get data --------------------------------------------------------------------
    df = funcs.prepare_ohlc(pair, live)
    
    if len(df) <= 200 and not in_pos:
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
    
# look up or calculate $ fixed risk -------------------------------------------

    if open_trades.get(pair):
        trade_record = open_trades.get(pair)
    else:
        trade_record = []

    pos_fr_dol, ep = uf.calc_pos_fr_dol(trade_record, fixed_risk_dol, in_pos)
    
# update positions dictionary with current pair's open_risk values ------------
    if in_pos:
        strat.sizing[asset].update(funcs.update_pos(asset, strat.bal, inval_dist, pos_fr_dol))
    
# execute orders --------------------------------------------------------------
    tp_trades = []
    
    if signals.get('tp_spot'):
        trade_record = omf.spot_tp(strat, pair, price, stp, inval_dist, pos_fr_dol, trade_record, 
                                                open_trades, market_data, live)
        
    elif signals.get('close_spot'):
        open_trades, closed_trades, in_pos = omf.spot_sell(strat, pair, price, next_id, trade_record, open_trades, 
                                                                                closed_trades, market_data, live)
    
    elif signals.get('open_spot'):        
        risk = (price - stp) / price
        mir = uf.max_init_risk(num_open_positions, params.get('target_risk'))
        # print(f'risk: {risk}, max_init_risk: {mir}, open positions: {num_open_positions}')
        # TODO max init risk should be based on average inval dist of signals, not fixed risk setting
        if risk > mir:
            strat.counts_dict['too_risky'] += 1
            # print('too risky')
            continue
        size, usdt_size = funcs.get_size(price, fixed_risk, strat.bal, risk)
        usdt_depth = funcs.get_depth(pair, 'buy', params.get('max_spread'))
        if usdt_depth < usdt_size and usdt_depth > (usdt_size/2): # only trim size if books are a bit too thin
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
            strat.counts_dict['books_too_thin'] += 1
        
        enough_depth = usdt_depth >= usdt_size
        enough_size = usdt_size > (21 * (1 + risk)) # this ensures size will be
        # big enough for init stop to be set on half the position
        
        if not enough_depth:
            if usdt_depth == 0:
                strat.counts_dict['too_much_spread'] += 1
            else:
                strat.counts_dict['books_too_thin'] += 1
        if not enough_size:
            strat.counts_dict['too_small'] += 1
            # print(f'too small, size: ${usdt_size}')
        
        if enough_size and enough_depth:            
# check total open risk and close profitable positions if necessary -----------
            # tp_trades = funcs.reduce_risk_old(strat.sizing, signals, params, fixed_risk, live)
            open_trades, closed_trades, next_id = omf.reduce_risk(strat, params, open_trades, closed_trades, market_data, next_id, live)
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            
# transfer trade records from reduce_risk into json records -------------------
            # for t in tp_trades:
            #     sym = t.get('pair')
            #     if open_trades.get(sym):
            #         trade_record = open_trades.get(sym)
            #     else:
            #         trade_record = []
            #     trade_record.append(t)
            #     if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            #         trade_id = trade_record[0].get('timestamp')
            #         closed_trades[trade_id] = trade_record
            #     else:
            #         closed_trades[next_id] = trade_record
            #     uf.record_closed_trades(strat.name, market_data, closed_trades)
            #     next_id += 1
            #     if open_trades[sym]:
            #         del open_trades[sym]
            #         uf.record_open_trades(strat.name, market_data, open_trades)
            #     strat.counts_dict['close_count'] += 1
            
# make sure there aren't too many open positions now --------------------------
            or_list = [v.get('or_R') for v in strat.sizing.values() if v.get('or_R')]
            total_open_risk = sum(or_list)
            num_open_positions = len(or_list)
            if num_open_positions >= max_positions or total_open_risk > params.get('total_r_limit'):
                strat.counts_dict['too_many_pos'] += 1
                # print(f'max exposure reached: {total_open_risk = }, {num_open_positions = }')
                continue
            
            usdt_bal = funcs.free_usdt()
            enough_usdt = usdt_bal > usdt_size
            if not enough_usdt:
                strat.counts_dict['not_enough_usdt'] += 1
                continue
            
# open new position -----------------------------------------------------------
            open_trades, in_pos = omf.spot_buy(strat, pair, fixed_risk, size, usdt_size, price, stp, 
                                                                    inval_dist, pos_fr_dol, 
                                                                    params, market_data, open_trades, live)            
            
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
    if in_pos:
        pos_bal = strat.sizing.get(asset)['value']
        open_risk = strat.sizing.get(asset)['or_$']
        open_risk_r = strat.sizing.get(asset)['or_R']
        
# calculate how much price has moved since entry ------------------------------
        if ep:
            price_delta = (price - ep) / ep
    
# take profit on risky positions ----------------------------------------------
            if open_risk_r > params.get('indiv_r_limit') and price_delta > 0.001:
                tp_pct = 50 if pos_bal > 30 else 100
                open_trades, closed_trades, in_pos = omf.spot_risk_limit_tp(strat, pair, tp_pct, price, 
                                                                            price_delta, trade_record, open_trades, 
                                                                            closed_trades, next_id, market_data, stp, 
                                                                            inval_dist, pos_fr_dol, in_pos, live)
            
            or_list = [v.get('or_R') for v in strat.sizing.values() if v.get('or_R')]
            total_open_risk = round(sum(or_list), 2)
            num_open_positions = len(or_list)
        
        
# log all data from the session and print/push summary-------------------------
strat.sizing['USDT'] = funcs.update_usdt(strat.bal)

benchmark = uf.log(live, strat, fixed_risk, market_data, spreads, now_start, 
                   tp_trades, open_trades, closed_trades)
if not live:
    pprint(strat.sizing)
    uf.interpret_benchmark(benchmark)
    print('warning: logging directed to test_records')

uf.scanner_summary(strat, market_data, all_start, benchmark, live)

pprint(strat.counts_dict)
print('-:-' * 20)
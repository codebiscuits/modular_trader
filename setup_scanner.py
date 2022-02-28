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

# setup
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)
client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
now_start = datetime.now().strftime('%d/%m/%y %H:%M')
all_start = time.perf_counter()
total_bal = funcs.account_bal()
# strat = strats.RSI_ST_EMA(4, 45, 96)
strat = strats.DoubleSTLO(3, 1.4)

pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()
if live:
    print('-:-' * 20)
else:
    print('*** Warning: Not Live ***')


# compile and sort list of pairs to loop through ------------------------------
spreads = funcs.binance_spreads('USDT') # dict
all_pairs = sorted(spreads.items(), key=lambda x:x[1])
positions = list(funcs.current_positions(strat.name, 0.00025).keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p[0] for p in all_pairs if (not p[0] in pairs_in_pos) and (not p[0] in not_pairs)]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first


counts_dict = {'stop_count': 0, 'open_count': 0, 'add_count': 0, 'tp_count': 0, 'close_count': 0, 
               'too_small': 0, 'too_risky': 0, 'too_many_pos': 0, 
               'books_too_thin': 0, 'too_much_spread': 0, 'not_enough_usdt': 0}

# update trade records --------------------------------------------------------
open_trades, closed_trades, next_id = uf.read_trade_records(market_data, strat.name)
if not live:
    uf.sync_test_records(strat, market_data)
    # now that trade records have been loaded, path can be changed
    market_data = Path('test_records')
uf.backup_trade_records(strat.name, market_data, open_trades, closed_trades)
next_id, counts_dict = uf.record_stopped_trades(open_trades, closed_trades, 
                                                pairs_in_pos, now_start, 
                                                next_id, strat, 
                                                market_data, counts_dict)

# set fixed risk
fixed_risk = uf.set_fixed_risk(strat, market_data)
max_init_r = fixed_risk * params.get('total_r_limit')
fixed_risk_dol = fixed_risk * total_bal

print(f"Current time: {now_start}, {strat}, fixed risk: {fixed_risk}")

funcs.top_up_bnb(15)

sizing = funcs.current_positions(strat.name, fixed_risk)
if sizing:
    open_pnls = [v.get('pnl') for v in sizing.values() if v.get('pnl')]
    avg_open_pnl = stats.median(open_pnls)
    max_positions = params.get('max_pos') if avg_open_pnl <= 0 else 50
else:
    max_positions = 20
    

for pair in pairs:
    asset = pair[:-1*len(params.get('quote_asset'))]
    in_pos = pair in pairs_in_pos
    
# look up or calculate $ fixed risk -------------------------------------------
    if open_trades.get(pair):
        trade_record = open_trades.get(pair)
    else:
        trade_record = []
    
    if in_pos and trade_record and trade_record[0].get('type')[0] == 'o':
        qs = float(trade_record[0].get('quote_size'))
        ep = float(trade_record[0].get('exe_price'))
        hs = trade_record[0].get('hard_stop')
        pos_fr_dol = qs * ((ep - hs) / ep)
    else:
        ep = None # i refer to this later and need it to exist even if it has no value
        pos_fr_dol = fixed_risk_dol
    
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
    inval_dist = signals.get('inval')
    
# calculate where stop_loss should be set if needed ---------------------------
    buffer = spreads.get(pair) * 2 # stop-market order will not get perfect execution, so
    stp = float(df.at[len(df)-1, 'st']) * (1-buffer) # expect some slippage in risk calc
    
    if df.at[len(df)-1, 'st'] == 0:
        note = f'{pair} supertrend 0 error, skipping pair'
        print(note)
        push = pb.push_note(now, note)
        continue
    
# calculate how much price has moved since entry ------------------------------
    price = df.at[len(df)-1, 'close']
    if ep:
        price_delta = (price - ep) / ep
    
# update positions dictionary with open_risk values ---------------------------
    if in_pos:
        sizing[asset] = funcs.update_pos(asset, total_bal, inval_dist, pos_fr_dol)
    
# execute orders --------------------------------------------------------------
    tp_trades = []
    
    if signals.get('tp_long'):
        counts_dict, trade_record = omf.spot_tp(strat, pair, price, stp, sizing, total_bal, 
                                                inval_dist, pos_fr_dol, trade_record, 
                                                open_trades, market_data, counts_dict, live)
        
            
    elif signals.get('close_long'):
        sizing, counts_dict, open_trades, closed_trades, in_pos = omf.spot_sell(strat, pair, price, next_id, sizing, 
                                                                                counts_dict, trade_record, open_trades, 
                                                                                closed_trades, total_bal, market_data, live)
    
    elif signals.get('open_long'):        
        risk = (price - stp) / price
        mir = uf.max_init_risk(len(pairs_in_pos), params.get('target_risk'))
        # TODO max init risk should be based on average inval dist of signals, not fixed risk setting
        if risk > mir:
            counts_dict['too_risky'] += 1
            continue
        size, usdt_size = funcs.get_size(price, fixed_risk, total_bal, risk)
        usdt_depth = funcs.get_depth(pair, 'buy', params.get('max_spread'))
        if usdt_depth < usdt_size and usdt_depth > (usdt_size/2): # only trim size if books are a bit too thin
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
            counts_dict['books_too_thin'] += 1
        
        enough_depth = usdt_depth >= usdt_size
        enough_size = usdt_size > (24 * (1 + risk)) # this ensures size will be
        # big enough for init stop to be set on half the position
        
        if not enough_depth:
            if usdt_depth == 0:
                counts_dict['too_much_spread'] += 1
            else:
                counts_dict['books_too_thin'] += 1
        if not enough_size:
            counts_dict['too_small'] += 1
        
        if enough_size and enough_depth:            
# check total risk and close profitable positions if necessary ----------------
            sizing, tp_trades = funcs.reduce_risk(sizing, signals, params, fixed_risk, live)
            sizing['USDT'] = funcs.update_usdt(total_bal)
            
# transfer trade records from reduce_risk into json records -------------------
            for t in tp_trades:
                sym = t.get('pair')
                if open_trades.get(sym):
                    trade_record = open_trades.get(sym)
                else:
                    trade_record = []
                trade_record.append(t)
                if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                    trade_id = trade_record[0].get('timestamp')
                    closed_trades[trade_id] = trade_record
                else:
                    closed_trades[next_id] = trade_record
                uf.record_closed_trades(strat.name, market_data, closed_trades)
                next_id += 1
                if open_trades[sym]:
                    del open_trades[sym]
                    uf.record_open_trades(strat.name, market_data, open_trades)
                counts_dict['close_count'] += 1
            
# make sure there aren't too many open positions now --------------------------
            or_list = [v.get('or_R') for v in sizing.values() if v.get('or_R')]
            total_open_risk = sum(or_list)
            num_open_positions = len(or_list)
            if num_open_positions >= max_positions or total_open_risk > params.get('total_r_limit'):
                counts_dict['too_many_pos'] += 1
                continue
            
            usdt_bal = funcs.free_usdt()
            enough_usdt = usdt_bal > usdt_size
            if not enough_usdt:
                counts_dict['not_enough_usdt'] += 1
                continue
            
# open new position -----------------------------------------------------------
            sizing, counts_dict, open_trades, in_pos = omf.spot_buy(strat, pair, fixed_risk, size, usdt_size, price, stp, 
                                                                    sizing, total_bal, inval_dist, pos_fr_dol, 
                                                                    params, market_data, counts_dict, open_trades, live)            
            
# calculate open risk and take profit if necessary ----------------------------
    if in_pos:
        pos_bal = sizing.get(asset)['value']
        open_risk = sizing.get(asset)['or_$']
        open_risk_r = sizing.get(asset)['or_R']
        
# take profit on risky positions ----------------------------------------------
        if open_risk_r > params.get('indiv_r_limit') and price_delta > 0.001:
            tp_pct = 50 if pos_bal > 30 else 100
            sizing, counts_dict, open_trades, closed_trades, in_pos = omf.spot_risk_limit_tp(strat, pair, tp_pct, price, 
                                                                                             price_delta, sizing, trade_record, 
                                                                                             open_trades, closed_trades, next_id, 
                                                                                             market_data, counts_dict, stp, 
                                                                                             total_bal, inval_dist, pos_fr_dol, in_pos, live)
        
        or_list = [v.get('or_R') for v in sizing.values() if v.get('or_R')]
        total_open_risk = round(sum(or_list), 2)
        num_open_positions = len(or_list)
        
        
# log all data from the session and print/push summary-------------------------
sizing['USDT'] = funcs.update_usdt(total_bal)
if not live:
    pprint(sizing)
    print('warning: logging directed to test_records')

benchmark = uf.log(live, strat, fixed_risk, 
                   market_data, spreads, now_start, 
                   sizing, tp_trades, counts_dict, 
                   open_trades, closed_trades)
uf.interpret_benchmark(benchmark)    

uf.scanner_summary(all_start, sizing, counts_dict, benchmark, live)


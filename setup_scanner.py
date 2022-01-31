'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, time, json
from json.decoder import JSONDecodeError
from datetime import datetime
import binance_funcs as funcs
import strategies as strats
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs, market_data
from pprint import pprint
import utility_funcs as uf
from pathlib import Path


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

# if the path below doesn't exist, the script is running on the wrong computer
pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()

if live:
    print('-:-' * 20)
else:
    print('*** Warning: Not Live ***')

all_start = time.perf_counter()

# constants
params = {'quote_asset': 'USDT', 
          'fixed_risk': 0.001, 
          'max_spread': 0.5, 
          'indiv_r_limit': 3, 
          'total_r_limit': 20}
max_positions = params.get('total_r_limit') # if all pos are below b/e i don't want to open more
max_init_r = params.get('fixed_risk') * params.get('total_r_limit')

# strat = strats.RSI_ST_EMA(4, 45, 96)
strat = strats.DoubleSTLO(3, 1.4)

# create pairs list
all_pairs = funcs.get_pairs('USDT', 'SPOT') # list
spreads = funcs.binance_spreads('USDT') # dict
positions = list(funcs.current_positions(params.get('fixed_risk')).keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']
other_pairs = [p for p in all_pairs if p in spreads and 
                                       spreads.get(p) < 0.01 and 
                                       not p in pairs_in_pos]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first

now_start = datetime.now().strftime('%d/%m/%y %H:%M')

# update trade records
# if live:
# read trade records
ot_path = f"{market_data}/{strat.name}_open_trades.json"
with open(ot_path, "r") as ot_file:
    try:
        ot = json.load(ot_file)
    except JSONDecodeError:
        ot = {}
ct_path = f"{market_data}/{strat.name}_closed_trades.json"
with open(ct_path, "r") as ct_file:
    try:
        closed_trades = json.load(ct_file)
        if closed_trades.keys():
            key_ints = [int(x) for x in closed_trades.keys()]
            next_id = sorted(key_ints)[-1] + 1
        else:
            next_id = 0
    except JSONDecodeError:
        closed_trades = {}
        next_id = 0

if not live: # now that trade records have been loaded, path can be changed
    market_data = Path('test_records')

# create list of trade records which don't match current positions
open_trades = list(ot.keys())
stopped_trades = [st for st in open_trades if st not in pairs_in_pos] # these positions must have been stopped out

# look for stopped out positions and complete trade records
for i in stopped_trades:
    trade_record = ot.get(i)
    close_is_buy = trade_record[0].get('type') == 'open_short'
    trades = client.get_my_trades(symbol=i)
    
    last_time = trades[-1].get('time')
    
    agg_price = []
    agg_base = []
    agg_quote = []
    agg_fee = []
    for t in trades[::-1]:
        if t.get('isBuyer') == close_is_buy:
            agg_price.append(float(t.get('price')))
            agg_base.append(float(t.get('qty')))
            agg_quote.append(float(t.get('quoteQty')))
            agg_fee.append(float(t.get('commission')))
        else:
            break
    # aggregate trade stats
    avg_exe_price = sum([p*b for p,b in zip(agg_price, agg_base)]) / sum(agg_base)
    tot_base = sum(agg_base)
    tot_quote = sum(agg_quote)
    tot_fee = sum(agg_fee)
    trade_type = 'stop_short' if close_is_buy else 'stop_long'
    # create dict
    trade_dict = {'timestamp': last_time, 
                  'pair': t.get('symbol'), 
                  'type': trade_type, 
                  'exe_price': avg_exe_price, 
                  'base_size': tot_base, 
                  'quote_size': tot_quote, 
                  'fee': tot_fee, 
                  'fee_currency': t.get('commissionAsset'), 
                  'reason': 'hit hard stop', 
                  }
    note = f"*** stopped out {t.get('symbol')} @ {t.get('price')}"
    print(now_start, note)
    push = pb.push_note(now_start, note)
    trade_record.append(trade_dict)
    if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
        trade_id = trade_record[0].get('timestamp')
        closed_trades[trade_id] = trade_record
    else:
        closed_trades[next_id] = trade_record
    uf.record_closed_trades(strat.name, market_data, closed_trades)
    next_id += 1
    if ot[i]:
        del ot[i]
        uf.record_open_trades(strat.name, market_data, ot)

print(f"Current time: {now_start}, {strat}, fixed risk: {params.get('fixed_risk')}")

total_bal = funcs.account_bal()
avg_prices = funcs.get_avg_prices()

funcs.top_up_bnb(15)

non_trade_notes = []
total_open_risk = 0 # expressed in terms of R
pos_open_risk = {} # dict of dicts {asset: {R:val, $:val}, }

for pair in pairs:
    asset = pair[:-1*len(params.get('quote_asset'))]
    in_pos = pair in pairs_in_pos
    if pair in not_pairs and not in_pos:
        continue
    # get data
    df = funcs.prepare_ohlc(pair, live)
    
    if len(df) <= 200 and not in_pos:
        continue
    
    if len(df) > strat.max_length:
        df = df.tail(strat.max_length)
        df.reset_index(drop=True, inplace=True)
    
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    # generate signals
    signals = strat.live_signals(df, in_pos)
    
    if df.at[len(df)-1, 'st'] == 0:
        print(pair, 'supertrend 0 error, skipping pair')
        note = f'{pair} supertrend 0 error, skipping pair'
        push = pb.push_note(now, note)
        continue
    
    # execute orders
    price = df.at[len(df)-1, 'close']
    tp_trades = []
    
    if signals.get('tp_long'):
        note = f"*** sell {pair} @ {price}"
        print(now, note)
        if live:
            push = pb.push_note(now, note)
            try:
                funcs.clear_stop(pair)
                # TODO isn't this supposed to be a take-profit order?!?!?!
                tp_order = funcs.sell_asset(pair)
                tp_order['type'] = 'tp_long'
                tp_order['reason'] = 'trade over-extended'
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(tp_order)
                ot['pair'] = trade_record
                uf.record_open_trades(strat.name, market_data, ot)
            except BinanceAPIException as e:
                print(f'problem with tp order for {pair}')
                print(e)
                push = pb.push_note(now, f'exeption during {pair} tp order')
    elif signals.get('close_long'):
        note = f"*** sell (stop) {pair} @ {price}"
        print(now, note)
        if live:
            push = pb.push_note(now, note)
            try:
                funcs.clear_stop(pair)
                sell_order = funcs.sell_asset(pair)
                sell_order['type'] = 'close_long'
                sell_order['reason'] = 'hit trailing stop'
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(sell_order)
                if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                    trade_id = trade_record[0].get('timestamp')
                    closed_trades[trade_id] = trade_record
                else:
                    closed_trades[next_id] = trade_record
                uf.record_closed_trades(strat.name, market_data, closed_trades)
                next_id += 1
                if ot[pair]:
                    del ot[pair]
                    uf.record_open_trades(strat.name, market_data, ot)
                in_pos = False
            except BinanceAPIException as e:
                print(f'problem with sell order for {pair}')
                print(e)
                push = pb.push_note(now, f'exeption during {pair} sell order')
    elif signals.get('open_long'):
        
        buffer = spreads.get(pair) * 2 # stop-market order will not get perfect execution, so
        stp = float(df.at[len(df)-1, 'st']) * (1-buffer) # expect some slippage in risk calc
        risk = (price - stp) / price
        mir = uf.max_init_risk(len(pairs_in_pos), max_init_r, max_positions)
        # TODO max init risk should be based on average inval dist of signals, not fixed risk setting
        if risk > mir:
            non_trade = f'{now} {pair} signal, too far from invalidation ({risk * 100:.1f}%)'
            print(non_trade)
            non_trade_notes.append(non_trade)
            print('-')
            continue
        size, usdt_size = funcs.get_size(price, params.get('fixed_risk'), total_bal, risk)
        usdt_bal = funcs.free_usdt()
        usdt_depth = funcs.get_depth(pair, 'buy', params.get('max_spread'))
        if usdt_depth < usdt_size and usdt_depth > (usdt_size/2): # only trim size if books are a bit too thin
            trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
            print(trim_size)
            usdt_size = usdt_depth
            non_trade_notes.append(trim_size)
        
        enough_depth = usdt_depth >= usdt_size
        enough_usdt = usdt_bal > usdt_size
        enough_size = usdt_size > (24 * (1 + risk)) # this ensures size will be
        # big enough for init stop to be set on half the position
        
        if not enough_depth:
            if usdt_depth == 0:
                non_trade = f"{now} {pair} signal, spread wider than {params.get('max_spread')} limit"
                print(non_trade)
                non_trade_notes.append(non_trade)
            else:
                non_trade = f'{now} {pair} signal, books too thin for {usdt_size:.3}USDT buy'
                print(non_trade)
                non_trade_notes.append(non_trade)
        if not enough_usdt:
            non_trade = f'{now} {pair} signal, not enough free usdt for {usdt_size:.3}USDT buy'
            print(non_trade)
            non_trade_notes.append(non_trade)
        if not enough_size:
            non_trade = f'{now} {pair} signal, size too small to trade ({usdt_size:.3}USDT)'
            print(non_trade)
            non_trade_notes.append(non_trade)
        
        if enough_usdt and enough_size and enough_depth:            
            # check total risk and close profitable positions if necessary
            tp_trades = funcs.reduce_risk(pos_open_risk, params.get('total_r_limit'), live)
            for t in tp_trades:
                sym = t.get('pair')
                if ot.get(sym):
                    trade_record = ot.get(sym)
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
                if ot[sym]:
                    del ot[sym]
                    uf.record_open_trades(strat.name, market_data, ot)
            
            # make sure there aren't too many open positions now
            if len(pairs_in_pos) >= max_positions:
                print(f'{now} {pair} signal, too many open positions already')
                continue
            
            # open new position
            note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
            print(now, note)
            if live:
                push = pb.push_note(now, note)
                try:
                    buy_order = funcs.buy_asset(pair, usdt_size)
                    buy_order['type'] = 'open_long'
                    buy_order['reason'] = 'buy conditions met'
                    buy_order['hard_stop'] = stp
                    ot[pair] = [buy_order]
                    stop_order = funcs.set_stop(pair, stp)
                    in_pos = True
                    uf.record_open_trades(strat.name, market_data, ot)
                except BinanceAPIException as e:
                    print(f'problem with buy order for {pair}')
                    print(e)
                    push = pb.push_note(now, f'exeption during {pair} buy order')
        print('-')
            
    sizing = funcs.current_positions(params.get('fixed_risk'))

    inval_dist = signals.get('inval')
    
    # calculate open risk and take profit if necessary
    if in_pos:
        pos_bal = sizing.get(asset)['value']
        # calculate open risk
        open_risk = pos_bal - (pos_bal / inval_dist) # dollar amount i would lose 
        # from current value if this position ended up getting stopped out
        open_risk_r = (open_risk / total_bal) / params.get('fixed_risk')
        
        # take profit on risky positions
        if open_risk_r > params.get('indiv_r_limit'):
            tp_pct = 50
            note = f"*** {pair} take profit {tp_pct}% @ {price}"
            print(now, note)
            print(f'pos_bal: ${pos_bal}, inval_dist: {inval_dist}')
            print(f'open_risk: ${open_risk:.2f}, open_risk_r: {open_risk_r:.3}R')
            print('-')
            if live:
                push = pb.push_note(now, note)
                funcs.clear_stop(pair)
                tp_order = funcs.sell_asset(pair, pct=50)
                tp_order['type'] = 'tp_long'
                buffer = spreads.get(pair) * 2 # stop-market order will not get perfect execution, so
                stp = float(df.at[len(df)-1, 'st']) * (1-buffer) # expect some slippage in risk calc
                stop_order = funcs.set_stop(pair, stp)
                tp_order['hard_stop'] = stp
                tp_order['reason'] = 'position R limit exceeded'
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(tp_order)
                ot[pair] = trade_record
                uf.record_open_trades(strat.name, market_data, ot)
            open_risk = pos_bal - (pos_bal / inval_dist) # update with new position
            open_risk_r = (open_risk / total_bal) / params.get('fixed_risk')
        
        total_open_risk += open_risk_r
        pos_open_risk[asset] = {'R': round(open_risk_r, 3), '$': round(open_risk, 2)}
        
    # TODO maybe have a plot rendered and saved every time a trade is closed

# incorporate pos_open_risk into sizing
for asset, v in sizing.items():
    if not asset in pos_open_risk:
        v['or_R'] = 0
        v['or_$'] = 0
    else:
        R = pos_open_risk[asset].get('R')
        dollar = pos_open_risk[asset].get('$')
        v['or_R'] = R
        v['or_$'] = dollar
    
if not live:
    print('---------------- pos_open_risk ----------------')
    pprint(pos_open_risk)
    

num_open_positions = len(pos_open_risk)
dollar_tor = total_bal * params.get('fixed_risk') * total_open_risk

if not live:
    print(f'{num_open_positions = }, {total_open_risk = }R, ie ${dollar_tor:.2f}')
    
    print('---------------- sizing ----------------')
    pprint(sizing)


# log all data from the session
uf.log(live, params, strat, market_data, spreads, now_start, sizing, pos_open_risk, tp_trades, 
    non_trade_notes, ot, closed_trades)

if not live:
    print('warning: logging switched off')
    
all_end = time.perf_counter()
all_time = all_end - all_start
elapsed_str = f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s'
rfb = round(total_bal-dollar_tor, 2)
final_msg = f'{elapsed_str}, total bal: ${total_bal:.2f} (${rfb} + ${dollar_tor:.2f}) \
positions {num_open_positions}'
print(final_msg)
push = pb.push_note(now, final_msg)
if live:
    print('-:-' * 20)

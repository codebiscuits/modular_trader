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
# params = {'strat': '20/200ema cross and supertrend with rsi triggers', 
#           'current_strat': 'rsi_st_ema', 
#           'quote_asset': 'USDT', 
#           'rsi_length': 4, 
#           'oversold': 45, 
#           'overbought': 96, 
#           'fixed_risk': 0.003, 
#           'max_spread': 0.5, 
#           'total_r_limit': 30, 
#           'max_length': 250}
params = {'strat': 'regular supertrend for bias with tight supertrend for entries/exits', 
          'current_strat': 'double_st_lo', 
          'quote_asset': 'USDT', 
          'st2_periods': 2, 
          'st2_mult': 1.5, 
          'fixed_risk': 0.001, 
          'max_spread': 0.5, 
          'total_r_limit': 30, 
          'max_length': 20}
max_positions = params.get('total_r_limit') # if all pos are below b/e i don't want to open more
max_init_r = params.get('fixed_risk') * params.get('total_r_limit')


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

# read trade records
if live:
    ot_path = f"{market_data}/{params.get('current_strat')}_open_trades.json"
else:
    ot_path = f"/mnt/pi_2/market_data/{params.get('current_strat')}_open_trades.json"
with open(ot_path, "r") as ot_file:
    try:
        ot = json.load(ot_file)
    except JSONDecodeError:
        ot = {}
if live:
    ct_path = f"{market_data}/{params.get('current_strat')}_closed_trades.json"
else:
    ct_path = f"/mnt/pi_2/market_data/{params.get('current_strat')}_closed_trades.json"
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

# create list of trade records which don't match current positions
open_trades = list(ot.keys())
stopped_trades = [st for st in open_trades if st not in pairs_in_pos] # these positions must have been stopped out

# look for stopped out positions and complete trade records
for i in stopped_trades:
    if ot.get(i): # this condition means that the exit will be recorded even if there is no entry in  the records
        trade_record = ot.get(i)
    else:
        trade_record = []
    close_is_buy = trade_record[0].get('type') == 'open_short'
    trades = client.get_my_trades(symbol=i)
    for t in trades[::-1]:
        if t.get('isBuyer') == close_is_buy:
            trade_dict = {'timestamp': t.get('time'), 
                          'pair': t.get('symbol'), 
                          'type': 'stop_short' if t.get('isBuyer') else 'stop_long', 
                          'exe_price': float(t.get('price')), 
                          'base_size': float(t.get('qty')), 
                          'quote_size': float(t.get('quoteQty')), 
                          'fee': t.get('commission'), 
                          'fee_currency': t.get('commissionAsset'), 
                          'reason': 'hit hard stop', 
                          }
            note = f"*** stopped out {t.get('symbol')} @ {t.get('price')}"
            print(now_start, note)
            push = pb.push_note(now_start, note)
            break
    trade_record.append(trade_dict)
    closed_trades[next_id] = trade_record
    uf.record_closed_trades(params, market_data, closed_trades)
    next_id += 1
    if ot[i]:
        del ot[i]
        uf.record_open_trades(params, market_data, ot)

# print(f"Current time: {now_start}, {params.get('current_strat')} \
# rsi: {params.get('rsi_length')}-{params.get('oversold')}-{params.get('overbought')}, \
# fixed risk: {params.get('fixed_risk')}")
print(f"Current time: {now_start}, {params.get('current_strat')} \
st2: {params.get('st2_periods')}-{params.get('st2_mult')}, \
fixed risk: {params.get('fixed_risk')}")

total_bal = funcs.account_bal()
avg_prices = funcs.get_avg_prices()

funcs.top_up_bnb(15)

trade_notes = [] # can be removed when i know the new system is working
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
    
    if len(df) > params.get('max_length'):
        df = df.tail(params.get('max_length'))
        df.reset_index(drop=True, inplace=True)
    
    # generate signals (tp_long, close_long, open_long)
    # signals = strats.rsi_st_ema_lo(df, in_pos, params.get('rsi_length'), params.get('overbought'), params.get('oversold'))
    signals = strats.double_st_lo(df, in_pos, params.get('st2_periods'), params.get('st2_mult'))
    
    if df.at[len(df)-1, 'st'] == 0:
        print(pair, 'supertrend 0')
    
    # execute orders
    # TODO need to integrate ALL binance filters into order calculations
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    price = df.at[len(df)-1, 'close']
    tp_trades = []
    
    if signals.get('tp_long'):
        note = f"*** sell {pair} @ {price}"
        print(now, note)
        if live:
            push = pb.push_note(now, note)
            try:
                funcs.clear_stop(pair)
                tp_order = funcs.sell_asset(pair)
                tp_order['type'] = 'tp_long'
                tp_order['reason'] = 'trade over-extended'
                trade_notes.append(tp_order) # hopefuly obsolete now
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(tp_order)
                ot['pair'] = trade_record
                uf.record_open_trades(params, market_data, ot)
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
                trade_notes.append(sell_order)
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(sell_order)
                closed_trades[next_id] = trade_record
                uf.record_closed_trades(params, market_data, closed_trades)
                next_id += 1
                if ot[pair]:
                    del ot[pair]
                    uf.record_open_trades(params, market_data, ot)
                in_pos = False
            except BinanceAPIException as e:
                print(f'problem with sell order for {pair}')
                print(e)
                push = pb.push_note(now, f'exeption during {pair} sell order')
    elif signals.get('open_long'):
        # # calc and record volume trend
        # df['vol_ema20'] = df.volume.ewm(20).mean()
        # df['vol_ema200'] = df.volume.ewm(200).mean()
        # df['vol_trend'] = df.vol_ema20 / df.vol_ema200
        # vol_trend = df.at[len(df)-1, 'vol_trend']
        # record = {'timestamp': now, 'pair': pair, 'side': 'long', 
        #           'price': price, 'vol_trend': vol_trend}
        
        # if len(pairs_in_pos) >= max_positions:
        #     print(f'{now} {pair} signal, too many open positions already')
        #     continue
        stp = df.at[len(df)-1, 'st'] # TODO incorporate spread into this
        risk = (price - stp) / price
        # print(f'risk: {risk:.4}, stp: {stp:.4}, spread: {sprd:.4}')
        mir = uf.max_init_risk(len(pairs_in_pos), max_init_r, max_positions)
        # TODO max init risk should be based on average inval dist of signals, not fixed risk setting
        if risk > mir:
            print(f'{now} {pair} signal, too far from invalidation ({risk * 100:.1f}%)')
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
        enough_size = usdt_size > (12 * (1 + risk)) # this ensures size will be big enough for init stop to be set
        
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
                    rec = ot.get(sym)
                else:
                    rec = []
                rec.append(t)
                closed_trades[next_id] = rec
                uf.record_closed_trades(params, market_data, closed_trades)
                next_id += 1
                if ot[sym]:
                    del ot[sym]
                    uf.record_open_trades(params, market_data, ot)
            
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
                    trade_notes.append(buy_order)
                    ot[pair] = [buy_order]
                    stop_order = funcs.set_stop(pair, stp)
                    in_pos = True
                    uf.record_open_trades(params, market_data, ot)
                except BinanceAPIException as e:
                    print(f'problem with buy order for {pair}')
                    print(e)
                    push = pb.push_note(now, f'exeption during {pair} buy order')
        print('-')
            
    sizing = funcs.current_positions(params.get('fixed_risk'))

    inval_dist = signals.get('inval')
    
    if in_pos:
        pos_bal = sizing.get(asset)['value']
        # calculate open risk
        open_risk = pos_bal - (pos_bal / inval_dist) # dollar amount i would lose 
        # from current value if this position ended up getting stopped out
        open_risk_r = (open_risk / total_bal) / params.get('fixed_risk')
        
        # take profit on risky positions
        if open_risk_r > 10:
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
                tp_order['reason'] = 'reducing portfolio risk'
                stp = df.at[len(df)-1, 'st']
                stop_order = funcs.set_stop(pair, stp)
                tp_order['hard_stop'] = stp
                trade_notes.append(tp_order)
                if ot.get(pair):
                    trade_record = ot.get(pair)
                else:
                    trade_record = []
                trade_record.append(tp_order)
                ot[pair] = trade_record
                uf.record_open_trades(params, market_data, ot)
            open_risk = pos_bal - (pos_bal / inval_dist) # update with new position
            open_risk_r = (open_risk / total_bal) / params.get('fixed_risk')
        
        total_open_risk += open_risk_r
        pos_open_risk[asset] = {'R': round(open_risk_r, 3), '$': round(open_risk, 2)}
        
    # TODO maybe have a plot rendered and saved every time a trade is triggered

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

print(f'{num_open_positions = }, {total_open_risk = }R, ie ${dollar_tor:.2f}')

print('---------------- sizing ----------------')
pprint(sizing)

if live:
    
    def log(params, market_data, spreads, pos_open_risk, non_trade_notes, ot, closed_trades):    
        
        # check total balance and record it in a file for analysis
        total_bal = funcs.account_bal()
        bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'positions': sizing, 'params': params}
        new_line = json.dumps(bal_record)
        with open(f"{market_data}/{params.get('current_strat')}_bal_history.txt", "a") as file:
            file.write(new_line)
            file.write('\n')
        
        # save a json of any trades that have happened with relevant data
        if tp_trades: # if the reduce_risk function closed any positions, they will be in here
            trade_notes.extend(tp_trades)
        with open(f"{market_data}/{params.get('current_strat')}_trades.txt", "a") as file:
            for trade in trade_notes:
                file.write(json.dumps(trade))
                file.write('\n')        
        with open(f"{market_data}/{params.get('current_strat')}_open_trades.json", "w") as ot_file:
            json.dump(ot, ot_file)            
        with open(f"{market_data}/{params.get('current_strat')}_closed_trades.json", "w") as ct_file:
            json.dump(closed_trades, ct_file)
        
        # record open_risk statistics
        risk_record = {'timestamp': now_start, 'open_risk': pos_open_risk}
        with open(f"{market_data}/{params.get('current_strat')}_open_risk.txt", "a") as file:
            file.write(json.dumps(risk_record))
            file.write('\n')
        
        # record all skipped or reduced trades
        non_trade_record = {'timestamp': now_start, 'non_trades': non_trade_notes}
        with open(f"{market_data}/{params.get('current_strat')}_non_trades.txt", "a") as file:
            file.write(json.dumps(non_trade_record))
            file.write('\n')
    
    log(params, market_data, spreads, pos_open_risk, non_trade_notes, ot, closed_trades)

else:
    print('warning: logging switched off')
    
all_end = time.perf_counter()
all_time = all_end - all_start
elapsed_str = f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s'
rfb = round(total_bal-dollar_tor, 2)
final_msg = f'Setup Scanner Finished. {elapsed_str}, total open risk: ${dollar_tor:.2f}, risk-free bal: ${rfb}'
print(final_msg)
push = pb.push_note(now, final_msg)
if live:
    print('-:-' * 20)

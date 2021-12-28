'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, time, json
from datetime import datetime
import binance_funcs as funcs
import strategies as strats
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs, market_data
from pprint import pprint
import utility_funcs as uf


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

live = True
if live:
    print('-:-' * 20)
else:
    print('*** Warning: Not Live ***')

all_start = time.perf_counter()

# constants
rsi_length = 4
oversold = 45
overbought = 96
fixed_risk = 0.003
total_r_limit = 30
max_positions = total_r_limit # if all pos are below b/e i don't want to open more
max_init_r = fixed_risk * total_r_limit
max_length = 250
current_strat = 'rsi_st_ema'
quote_asset = 'USDT'

# create pairs list
all_pairs = funcs.get_pairs('USDT', 'SPOT') # list
spreads = funcs.binance_spreads('USDT') # dict
positions = funcs.current_positions(fixed_risk)
pairs_in_pos = [pip for pip in all_pairs if positions.get(pip) != 0]
other_pairs = [p for p in all_pairs if p in spreads and 
                                       spreads.get(p) < 0.01 and 
                                       positions.get(p) == 0]
pairs = pairs_in_pos + other_pairs # this ensures open positions will be checked first

now_start = datetime.now().strftime('%d/%m/%y %H:%M')

print(f'Current time: {now_start}, rsi: {rsi_length}-{oversold}-{overbought}, fixed risk: {fixed_risk}')

total_bal = funcs.account_bal()
avg_prices = funcs.get_avg_prices()

funcs.top_up_bnb(15)

trade_notes = []
non_trade_notes = []
total_open_risk = 0 # expressed in terms of R
pos_open_risk = {} # expressed in terms of R

for pair in pairs:
    asset = pair[:-1*len(quote_asset)]
    in_pos = bool(positions.get(pair))
    if pair in not_pairs and not in_pos:
        continue
    # get data
    df = funcs.prepare_ohlc(pair)
    
    if len(df) <= 200 and not in_pos:
        continue
    
    if len(df) > max_length:
        df = df.iloc[-1*max_length:,]
        df.reset_index(drop=True, inplace=True)
    
    # generate signals (tp_long, close_long, open_long)
    signals, inval_dist = strats.rsi_st_ema_lo(df, in_pos, rsi_length, overbought, oversold)
    
    # execute orders
    # TODO need to integrate ALL binance filters into order calculations
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    price = df.at[len(df)-1, 'close']
    tp_trades = []
    
    if signals[0]: # tp_long
        note = f"*** sell {pair} @ {price}"
        print(now, note)
        if live:
            push = pb.push_note(now, note)
            try:
                funcs.clear_stop(pair)
                sell_order = funcs.sell_asset(pair)
                sell_order['reason'] = 'trade over-extended'
                trade_notes.append(sell_order)
                in_pos = False
            except BinanceAPIException as e:
                print(f'problem with tp order for {pair}')
                print(e)
                push = pb.push_note(now, f'exeption during {pair} tp order')
    elif signals[1]: # close_long
        note = f"*** sell (stop) {pair} @ {price}"
        print(now, note)
        if live:
            push = pb.push_note(now, note)
            try:
                funcs.clear_stop(pair)
                sell_order = funcs.sell_asset(pair)
                sell_order['reason'] = 'hit trailing stop'
                trade_notes.append(sell_order)
                in_pos = False
            except BinanceAPIException as e:
                print(f'problem with sell order for {pair}')
                print(e)
                push = pb.push_note(now, f'exeption during {pair} sell order')
    elif signals[2]: # open_long
        # # calc and record volume trend
        # df['vol_ema20'] = df.volume.ewm(20).mean()
        # df['vol_ema200'] = df.volume.ewm(200).mean()
        # df['vol_trend'] = df.vol_ema20 / df.vol_ema200
        # vol_trend = df.at[len(df)-1, 'vol_trend']
        # record = {'timestamp': now, 'pair': pair, 'side': 'long', 
        #           'price': price, 'vol_trend': vol_trend}
        
        if len(pairs_in_pos) >= max_positions:
            print(f'{now} {pair} signal, too many open positions already')
            continue
        sprd = spreads.get(pair)
        stp = df.at[len(df)-1, 'st'] # TODO incorporate spread into this
        risk = (price - stp) / price
        # print(f'risk: {risk:.4}, stp: {stp:.4}, spread: {sprd:.4}')
        mir = uf.max_init_risk(len(pairs_in_pos), max_init_r, max_positions)
        if risk > mir:
            print(f'{now} {pair} signal, too far from invalidation ({risk * 100:.1f}%)')
            continue
        size, usdt_size = funcs.get_size(price, fixed_risk, total_bal, risk)
        usdt_bal = funcs.free_usdt()
        usdt_depth = funcs.get_depth(pair, 'buy')
        enough_depth = usdt_depth >= usdt_size
        enough_usdt = usdt_bal > usdt_size
        enough_size = usdt_size > (12 * (1 + risk)) # this ensures size will be big enough for init stop to be set
        if not enough_depth:
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
            tp_trades = funcs.reduce_risk(pos_open_risk, total_r_limit, live)
            # open new position
            note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
            print(now, note)
            if live:
                push = pb.push_note(now, note)
                try:
                    buy_order = funcs.buy_asset(pair, usdt_size)
                    buy_order['reason'] = 'buy conditions met'
                    trade_notes.append(buy_order)
                    stop_order = funcs.set_stop(pair, stp)
                    stop_order['reason'] = 'safety first'
                    trade_notes.append(stop_order)
                    in_pos = True
                except BinanceAPIException as e:
                    print(f'problem with buy order for {pair}')
                    print(e)
                    push = pb.push_note(now, f'exeption during {pair} buy order')
        print('-')
            
    sizing = funcs.current_sizing(fixed_risk)
    
    if in_pos:
        pos_bal = sizing.get(asset) * total_bal
        # calculate open risk
        open_risk = pos_bal - (pos_bal / inval_dist) # dollar amount i would lose 
        # from current value if this position ended up getting stopped out
        open_risk_r = (open_risk / total_bal) / fixed_risk
        
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
                sell_order = funcs.sell_asset(pair, pct=50)
                sell_order['reason'] = 'taking partial profit'
                stp = df.at[len(df)-1, 'st']
                stop_order = funcs.set_stop(pair, stp)
                trade_notes.append(sell_order)
            open_risk = pos_bal - (pos_bal / inval_dist) # update with new position
            open_risk_r = (open_risk / total_bal) / fixed_risk
        
        total_open_risk += open_risk_r
        pos_open_risk[pair] = round(open_risk_r, 3)
        
    # TODO maybe have a plot rendered and saved every time a trade is triggered

if not live:
    print('pos_open_risk')
    pprint(pos_open_risk)                  


num_open_positions = len(pos_open_risk)
dollar_tor = total_bal * fixed_risk * total_open_risk

print(f'{num_open_positions = }, {total_open_risk = }R, ie ${dollar_tor:.2f}')

if live:
    # check total balance and record it in a file for analysis
    params = {'strat': '20/200ema cross and supertrend with rsi triggers', 
              'rsi length': rsi_length, 'oversold': oversold, 
              'overbought': overbought, 'fixed risk': fixed_risk}
    sizing = funcs.current_sizing(fixed_risk)
    total_bal = funcs.account_bal()
    bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'positions': sizing, 'params': params}
    new_line = json.dumps(bal_record)
    with open(f"{market_data}/rsi-st-ema_bal_history.txt", "a") as file:
        file.write(new_line)
        file.write('\n')
    
    # save a json of any trades that have happened with relevant data
    if tp_trades: # if the reduce_risk function closed any positions, they will be in here
        trade_notes.extend(tp_trades)
    with open(f"{market_data}/{current_strat}_trades.txt", "a") as file:
        for trade in trade_notes:
            file.write(json.dumps(trade))
            file.write('\n')
    
    # record spreads and depths for other analysis
    stamped_spreads = {'timestamp': now_start, 'spreads': spreads}
    with open(f"{market_data}/binance_spreads_history.txt", "a") as file:
        file.write(json.dumps(stamped_spreads))
        file.write('\n')
    
    depths = funcs.binance_depths()
    stamped_depths = {'timestamp': now_start, 'depths': depths}
    with open(f"{market_data}/binance_depths_history.txt", "a") as file:
        file.write(json.dumps(stamped_depths))
        file.write('\n')
    
    # record open_risk statistics
    risk_record = {'timestamp': now_start, 'positions': pos_open_risk}
    with open(f"{market_data}/{current_strat}_open_risk.txt", "a") as file:
        file.write(json.dumps(risk_record))
        file.write('\n')
else:
    print('warning: logging switched off')
    
all_end = time.perf_counter()
all_time = all_end - all_start
elapsed_str = f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s'
rfb = round(total_bal-dollar_tor, 2)
final_msg = f'Setup Scanner Finished. {elapsed_str}, total open risk: ${dollar_tor:.2f}, risk-free bal: ${rfb}'
print(final_msg)
push = pb.push_note(now, final_msg)

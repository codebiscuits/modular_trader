'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

import pandas as pd
import matplotlib.pyplot as plt
import keys, talib, time, json
from datetime import datetime
from pathlib import Path
import binance_funcs as funcs
import indicators as ind
import strategies as strats
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from config import not_pairs, ohlc_data, market_data

# TODO need to sort out error handling

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


all_start = time.perf_counter()

# constants
rsi_length = 4
oversold = 45
overbought = 96
fixed_risk = 0.003
max_length = 250
current_strat = 'rsi_st_ema'

# create pairs list
all_pairs = funcs.get_pairs('USDT', 'SPOT') # list
spreads = funcs.binance_spreads('USDT') # dict
positions = funcs.current_positions(fixed_risk)
pairs = [p for p in all_pairs if spreads.get(p) < 0.01 or positions.get(p) == 1]

now_start = datetime.now().strftime('%d/%m/%y %H:%M')

print('-*-' * 20)
print(f'Current time: {now_start}, rsi: {rsi_length}-{oversold}-{overbought}, fixed risk: {fixed_risk}')

funcs.top_up_bnb(15)

trade_notes = []

for pair in pairs:
    in_pos = positions.get(pair)
    if pair in not_pairs and in_pos == 0:
        continue
    # get data
    filepath = Path(f'{ohlc_data}/{pair}.pkl')
    if filepath.exists():
        try:
            df = pd.read_pickle(filepath)
        except:
            print(f'read_pickle went wrong with {pair}')
            df = funcs.get_ohlc(pair, '4h', '1 year ago UTC')
        
        if len(df) > 2:
            df = df.iloc[:-1,]
            df = funcs.update_ohlc(pair, '4h', df)
    else:
        df = funcs.get_ohlc(pair, '4h', '1 year ago UTC')
    if len(df) > 2190: # 2190 is 1 year's worth of 4h periods
        df = df.iloc[-2190:,]
        df.reset_index(drop=True, inplace=True)
    try:
        df.to_pickle(filepath)
    except:
        print(f'to_pickle went wrong with {pair}')
    
    if len(df) <= 200 and in_pos == 0:
        continue
    
    if len(df) > max_length:
        df = df.iloc[-1*max_length:,]
        df.reset_index(drop=True, inplace=True)
    
    # generate signals (tp_long, close_long, open_long)
    signals = strats.rsi_st_ema_lo(df, in_pos, rsi_length, overbought, oversold)
    
    # execute orders
    # TODO need to integrate ALL binance filters into order calculations
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    price = df.at[len(df)-1, 'close']
    balance = funcs.account_bal()
    
    if signals[0]: # tp_long
        note = f"*** {now} sell {pair} @ {price}"
        print(now, note)
        push = pb.push_note(now, note)
        funcs.clear_stop(pair)
        sell_order = funcs.sell_asset(pair)
        trade_notes.append(sell_order)
    elif signals[1]: # close_long
        note = f"*** {now} sell (stop) {pair} @ {price}"
        print(now, note)
        push = pb.push_note(now, note)
        funcs.clear_stop(pair)
        sell_order = funcs.sell_asset(pair)
        trade_notes.append(sell_order)
    elif signals[2]: # open_long
        stp = df.at[len(df)-1, 'st'] # TODO incorporate spread into this
        risk = (price - stp) / price
        # if risk > 0.1:
        #     print(f'{now} {pair} signal, too far from invalidation ({risk * 100:.1f}%)')
        #     continue
        size, usdt_size = funcs.get_size(price, fixed_risk, balance, risk)
        usdt_bal = funcs.free_usdt()
        usdt_depth = funcs.get_depth(pair, 'buy')
        enough_depth = usdt_depth >= usdt_size
        enough_usdt = usdt_bal > usdt_size
        enough_size = usdt_size > (12 * (1 + risk)) # this ensures size will be big enough for init stop to be set
        if not enough_depth:
            print(f'{now} {pair} signal, books too thin for {usdt_size:.3}USDT buy')
        if not enough_usdt:
            print(f'{now} {pair} signal, not enough free usdt for {usdt_size:.3}USDT buy')
        if not enough_size:
            print(f'{now} {pair} signal, size too small to trade ({usdt_size:.3}USDT)')
        if enough_usdt and enough_size and enough_depth:
            note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
            push = pb.push_note(now, note)
            print(now, note)
            # TODO these try/except blocks could maybe go inside the functions
            try:
                buy_order = funcs.buy_asset(pair, usdt_size)
                trade_notes.append(buy_order)
                stop_order = funcs.set_stop(pair, stp)
                trade_notes.append(stop_order)
            except 'BinanceAPIException' as e:
                print(f'problem with order for {pair}')
                print(e)
            
                
    # TODO maybe have a plot rendered and saved every time a trade is triggered
    
    # if in_pos and signals == (0, 0, 0):
        # open_risk = (price - stp) / price

logging = True
if logging:
    # check total balance and record it in a file for analysis
    params = {'strat': '20/200ema cross and supertrend with rsi triggers', 
              'rsi length': rsi_length, 'oversold': oversold, 
              'overbought': overbought, 'fixed risk': fixed_risk}
    total_bal = funcs.account_bal()
    sizing = funcs.current_sizing(fixed_risk)
    bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'positions': sizing, 'params': params}
    new_line = json.dumps(bal_record)
    with open(f"{market_data}/rsi-st-ema_bal_history.txt", "a") as file:
        file.write(new_line)
        file.write('\n')
    
    # save a json of any trades that have happened with relevant data
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
else:
    print('warning: logging switched off')
    
all_end = time.perf_counter()
all_time = all_end - all_start
print(f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s')
push = pb.push_note(now, 'Setup Scanner Finished')

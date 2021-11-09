'''this script is an analysis where i can input a single set of settings 
and then plot the backtest for one pair at those settings, with buys and sells 
on the chart and pnl evolution underneath, so i can cycle through all pairs 
and see whats happening pair by pair, trade by trade. that might give me some 
insight into why so few pairs are producing 30+ trades in a year'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
import keys
import talib
import time
import json
from datetime import datetime
from pathlib import Path
from rsi_optimising import get_pairs, get_ohlc, update_ohlc, get_supertrend, get_signals, get_results
from binance_funcs import account_bal, get_size, current_positions, free_usdt, get_depth
from execution import buy_asset, sell_asset, set_stop, clear_stop
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet

# TODO need to sort out error handling

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


pairs = get_pairs('usdt')
not_pairs = ['GPBUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 'USDCUSDT', 'PAXUSDT', 'COCOSUSDT',
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

rsi_length = 4
oversold = 45
overbought = 96
fixed_risk = 0.004

max_length = 250

positions = current_positions(fixed_risk)

# pairs = ['BTCUSDT']


while True:
    # quicker check every 4 hours just for ema uptrend
    curr = datetime.now().strftime('%d/%m/%y %H:%M')
    print(f'Current time: {curr}, rsi: {rsi_length}-{oversold}-{overbought}, fixed risk: {fixed_risk}')
    short_pairs = []
    quick_start = time.perf_counter()
    dropped = 0
    for pair in pairs:
        if pair in not_pairs:
            continue
        # get data
        filepath = Path(f'ohlc_data/{pair}.pkl')
        if filepath.exists():
            df = pd.read_pickle(filepath)
        else:
            df = get_ohlc(pair, '4h', '35 days ago UTC')
        if len(df) <= 200:
            continue
        df['20ema'] = talib.EMA(df.close, 20)
        df['200ema'] = talib.EMA(df.close, 200)
        ema_ratio = df.at[len(df)-1, '20ema'] / df.at[len(df)-1, '200ema']
        #TODO might be worth adding in a check for spread here, so any pairs with 
        # unusually large spread get dropped from the list
        if ema_ratio < 0.9:
            dropped += 1
            continue
        else:
            short_pairs.append(pair)
    for p in positions:
        if positions.get(p) == 1 and p not in short_pairs:
            # TODO instead of appending to short_pairs, perhaps i should be market selling
            short_pairs.append(p)
            print(f'added {p} to shortlist')
    
    quick_end = time.perf_counter()
    quick_time = quick_end - quick_start
    print(f'Shortlist time taken: {round((quick_time) // 60)}m {(quick_time) % 60:.3}s')
    print(f'pairs in shortlist: {len(short_pairs)}, pairs dropped: {dropped}')
    print('\n----------------------------------\n')
    
    
    # check total balance and record it in a file for analysis
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    total_bal = account_bal()
    bal_record = {'timestamp': now, 'balance': total_bal}
    new_line = json.dumps(bal_record)
    with open("total_bal_history.txt", "a") as file:
        file.write(new_line)
        file.write('\n')
    
    
    # full check on shortlisted pairs that loops
    for i in range(50):
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        if i%10 == 0:
            print(now)
        
        #TODO if there's less than $10 free usdt, there's no point constantly checking 
        # for new setups, only monitoring open positions, so in this case, reassign 
        # the shortlist to a new list with just the open positions in it, until
        # such time that some more usdt has been freed up, then go back to the 
        # main shortlist
        
        all_start = time.perf_counter()
        # try:
        positions = current_positions(fixed_risk)
        #TODO might be worth having a check here to see if any open positions 
        # should not still be open and need to be closed immediately
        for pair in short_pairs:
            in_pos = positions.get(pair)
            if pair in not_pairs:
                continue
            # get data
            filepath = Path(f'ohlc_data/{pair}.pkl')
            if filepath.exists():
                df = pd.read_pickle(filepath)
                df = update_ohlc(pair, '4h', df)
            else:
                df = get_ohlc(pair, '4h', '35 days ago UTC')
            if len(df) <= 200:
                continue
            if len(df) > max_length:
                df = df.iloc[-1*max_length:,]
                df.reset_index(drop=True, inplace=True)
            
            # compute indicators
            df['20ema'] = talib.EMA(df.close, 20)
            df['200ema'] = talib.EMA(df.close, 200)
            if df.at[len(df)-1, '20ema'] < df.at[len(df)-1, '200ema']:
                continue
            # print(df)
            df['st'], df['st_u'], df['st_d'] = get_supertrend(df.high, df.low, df.close, 10, 3)
            df['rsi'] = talib.RSI(df.close, rsi_length)
            hodl = df['close'].iloc[-1] / df['close'].iloc[0]
            
            #TODO run a function here which checks recorded position against
            # whether the asset's price is below st line. if position is 1 and 
            
            # generate signals
            buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = get_signals(df, oversold, overbought)
            
            # calculate results
            # TODO these two lines might be fine to delete
            pnl, pnl_list = get_results(df)
            pnl_bth = pnl / hodl
            
            
            # TODO need to integrate ALL binance filters into order calculations
            if not pd.isna(df.at[len(df)-1, 'signals']):
                buy_sig = df.at[len(df)-1, 'signals'][:3] == 'buy'
                sell_sig = df.at[len(df)-1, 'signals'] == 'sell'
                stop_sig = df.at[len(df)-1, 'signals'] == 'stop'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = df.at[len(df)-1, 'close']
                balance = account_bal()
                if in_pos:
                    orders = client.get_open_orders(symbol=pair)
                    if sell_sig:
                        note = f"*** {now} sell {pair} @ {price}"
                        print(note)
                        # push = pb.push_note(now, note)
                        clear_stop(pair)
                        sell_asset(pair)
                    elif stop_sig:
                        note = f"*** {now} sell (stop) {pair} @ {price}"
                        print(note)
                        # push = pb.push_note(now, note)
                        clear_stop(pair)
                        sell_asset(pair)
                else:
                    if buy_sig:
                        print(f'{now} potential {pair} buy signal')
                        stp = float(df.at[len(df)-1, 'signals'][17:24])
                        risk = (price - stp) / price
                        if risk > 0.2:
                            print(f'{now} {pair} signal, too far from invalidation ({risk * 100}%)')
                        size, usdt_size = get_size(price, fixed_risk, balance, risk)
                        usdt_bal = free_usdt()
                        usdt_depth = get_depth(pair)
                        enough_depth = usdt_depth >= usdt_size
                        enough_usdt = usdt_bal > usdt_size
                        enough_size = usdt_size > (10 * (1 + risk)) # this ensures size will be big enough for init stop to be set
                        if not enough_depth:
                            print(f'{now} {pair} signal, books too thin for {usdt_size:.3}USDT buy')
                        if not enough_usdt:
                            print(f'{now} {pair} signal, not enough free usdt for {usdt_size:.3}USDT buy')
                        if not enough_size:
                            print(f'{now} {pair} signal, size too small to trade ({usdt_size:.3}USDT)')
                        if enough_usdt and enough_size and enough_depth:
                            note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, init stop @ {stp}"
                            # push = pb.push_note(now, note)
                            print(note)
                            # TODO these try/except blocks could maybe go inside the functions
                            try:
                                buy_asset(pair, usdt_size)
                            except 'BinanceAPIException' as e:
                                print(f'problem with buy order for {pair}')
                                print(e)
                            try:
                                set_stop(pair, stp)
                            except 'BinanceAPIException' as e:
                                print(f'problem with stop order for {pair}')
                                print(e)
                                
                            # TODO maybe have a plot rendered and saved every time a trade is triggered
                            
                
                    
            
            df = df.iloc[:-1,]
            df.to_pickle(filepath)
        
        
        all_end = time.perf_counter()
        all_time = all_end - all_start
        # print(f'Round {i+1} time taken: {round((all_time) // 60)}m {(all_time) % 60:.3}s')
        # except:
        #     print('*-' * 30)
        #     exc = f'{sys.exc_info()} exception occured'
        #     # push = pb.push_note(now, exc)
        #     print(exc)
        #     print('*-' * 30)
        #     continue          

    
    print('\n----------------------------------\n')

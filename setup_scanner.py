'''this script scans all pairs for a particular quote asset looking for setups 
which match the criteria for a trade, then executes the trade if possible'''

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
from binance_funcs import account_bal, get_size, current_positions, free_usdt
from execution import buy_asset, sell_asset, set_stop, clear_stop, get_depth, get_spread
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet

# TODO need to sort out error handling

# TODO the best way to run this script would be once every 4 hours to make 
# entries only if rsi closes above the threshold. if either of the exit triggers 
# need to be made between closes, i can put open positions on a websocket 
# connection to be watched constantly.

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False)

client = Client(keys.bPkey, keys.bSkey)

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


pairs = get_pairs('USDT', 'SPOT')
not_pairs = ['GPBUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT', 
             'USDCUSDT', 'PAXUSDT', 'COCOSUSDT',
             'ADADOWNUSDT', 'LINKDOWNUSDT', 'BNBDOWNUSDT', 'ETHDOWNUSDT']

rsi_length = 4
oversold = 45
overbought = 96
fixed_risk = 0.004

max_length = 250

positions = current_positions(fixed_risk)

# check total balance and record it in a file for analysis
now = datetime.now().strftime('%d/%m/%y %H:%M')
total_bal = account_bal()
bal_record = {'timestamp': now, 'balance': total_bal}
new_line = json.dumps(bal_record)
with open("total_bal_history.txt", "a") as file:
    file.write(new_line)
    file.write('\n')

print(f'Current time: {now}, rsi: {rsi_length}-{oversold}-{overbought}, fixed risk: {fixed_risk}')

all_start = time.perf_counter()
# try:
positions = current_positions(fixed_risk)
for pair in pairs:
    in_pos = positions.get(pair)
    if pair in not_pairs and positions.get(pair) == 0:
        continue
    if get_spread(pair) > 1 and positions.get(pair) == 0:
        continue
    # get data
    filepath = Path(f'ohlc_data/{pair}.pkl')
    if filepath.exists():
        df = pd.read_pickle(filepath)
        df = update_ohlc(pair, '4h', df)
    else:
        df = get_ohlc(pair, '4h', '35 days ago UTC')
    if len(df) <= 200 and positions.get(pair) == 0:
        continue
    if len(df) > max_length:
        df = df.iloc[-1*max_length:,]
        df.reset_index(drop=True, inplace=True)
    
    # compute indicators
    df['20ema'] = talib.EMA(df.close, 20)
    df['200ema'] = talib.EMA(df.close, 200)
    ema_ratio = df.at[len(df)-1, '20ema'] / df.at[len(df)-1, '200ema']
    if ema_ratio < 1 and positions.get(pair) == 0:
        continue
    # print(df)
    df['st'], df['st_u'], df['st_d'] = get_supertrend(df.high, df.low, df.close, 10, 3)
    df['rsi'] = talib.RSI(df.close, rsi_length)
    hodl = df['close'].iloc[-1] / df['close'].iloc[0]
    
    # generate signals
    buys, sells, stops, df['s_buy'], df['s_sell'], df['s_stop'] = get_signals(df, oversold, overbought)
    
    
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
                if risk > 0.1:
                    print(f'{now} {pair} signal, too far from invalidation ({risk * 100}%)')
                    continue
                size, usdt_size = get_size(price, fixed_risk, balance, risk)
                usdt_bal = free_usdt()
                usdt_depth = get_depth(pair, 'buy')
                print(f'usdt depth: {usdt_depth}')
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
print(f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s')
# except:
#     print('*-' * 30)
#     exc = f'{sys.exc_info()} exception occured'
#     # push = pb.push_note(now, exc)
#     print(exc)
#     print('*-' * 30)
#     continue

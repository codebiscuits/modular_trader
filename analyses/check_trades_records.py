from pathlib import Path
import keys
import json
from json.decoder import JSONDecodeError
import utility_funcs as uf
from pprint import pprint
import binance_funcs as funcs
import statistics as stats
from datetime import datetime as dt
from binance.client import Client
from pushbullet import Pushbullet
import strategies as strats

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

strat = strats.DoubleST(3, 1.2)
print_all = 1 # 0 = minimal, 1 = print open trades, 2 = print all trades
window = 50 # lookback window for how many closed trades to include in statistics

pi2path = Path('/home/ubuntu/rpi_2.txt')
live = pi2path.exists()
now = dt.now()
print(now.strftime('%d/%m/%y %H:%M'))

market_data = Path('/mnt/pi_2/market_data')

############################################################################

o_path = Path(f'{market_data}/{strat.name}/sim_trades.json')
c_path = Path(f'{market_data}/{strat.name}/closed_sim_trades.json')

with open(o_path, 'r') as file:
    o_data = json.load(file)
    
with open(c_path, 'r') as file:
    try:
        c_data = json.load(file)
        if c_data.keys():
            key_ints = [int(x) for x in c_data.keys()]
            next_id = sorted(key_ints)[-1] + 1
        else:
            next_id = 0
    except JSONDecodeError:
        c_data = []
        next_id = 0
        print('no closed trades yet')
    
############################################################################

positions = list(funcs.current_positions_old(strat, 'sim').keys())
pairs_in_pos = [p + 'USDT' for p in positions if p != 'USDT']

# create list of trade records which don't match current positions
open_trades = list(o_data.keys())
stopped_trades = [st for st in open_trades if st not in pairs_in_pos] # these positions must have been stopped out

# look for stopped out positions and complete trade records
if stopped_trades:
    for i in stopped_trades:
        trade_record = o_data.get(i)
        
        # work out how much base size has been bought vs sold
        init_base = 0
        add_base = 0
        tp_base = 0
        close_base = 0
        for x in trade_record:
            if x.get('type')[:4] == 'open':
                init_base = float(x.get('base_size'))
            elif x.get('type')[:3] == 'add':
                add_base += float(x.get('base_size'))
            elif x.get('type')[:2] == 'tp':
                tp_base += float(x.get('base_size'))
            elif x.get('type')[:5] in ['close', 'stop_']:
                close_base = float(x.get('base_size'))
        
        diff = (init_base + add_base) - (tp_base + close_base)        
        
        close_is_buy = trade_record[0].get('type') == 'open_short'
        trades = client.get_my_trades(symbol=i)
        
        last_time = trades[-1].get('time')
        
        agg_price = []
        agg_base = []
        agg_quote = []
        agg_fee = []
        base_size_count = 0
        for t in trades[::-1]:
            if abs((base_size_count / diff) - 1) > 0.03:
                agg_price.append(float(t.get('price')))
                agg_base.append(float(t.get('qty')))
                agg_quote.append(float(t.get('quoteQty')))
                agg_fee.append(float(t.get('commission')))
                base_size_count += float(t.get('qty'))
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
        print(now, note)
        if live:
            push = pb.push_note(now, note)
        trade_record.append(trade_dict)
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = int(trade_record[0].get('timestamp'))
            c_data[trade_id] = trade_record
        else:
            c_data[next_id] = trade_record
        next_id += 1
    
        if not live:
            market_data = 'test_records'
        uf.record_closed_trades(strat.name, c_data)
        if o_data[i]:
            del o_data[i]
            uf.record_open_trades(strat.name, o_data)


############################################################################

if print_all:
    print('\n---- Open Trades ----\n')
# pprint(o_data)

total_r = 0
winning = 0
losing = 0

total_bal = funcs.account_bal()

for pair, v in o_data.items():
    ots = uf.open_trade_stats(now, total_bal, v)
    # print(ots)
    
    total_r += ots.get('pnl_R')
    if ots.get('pnl_R') >= 0:
        winning += 1
    else:
        losing += 1
    
    if print_all:
        print(f"{pair} :: {'long' if ots.get('long') else 'short'} :: \
entry: {ots.get('entry_price'):.3} :: current pnl: \
{ots.get('pnl_R'):.2}R :: duration: {ots.get('duration (h)')} hours\n")

print(f'Open Total PnL: {float(total_r):.1f}R, {winning} in profit, {losing} in loss')

############################################################################

if print_all:
    print('\n---- Closed Trades ----\n')

all_r = []
all_open_dates = []

if c_data:
    
    print(f'num closed trades {len(c_data.keys())}')
    
    # bad_keys = uf.find_bad_keys(c_data)
    # bad_keys = [k.get('key') for k in bad_keys]
    
    add_counts = []
    tp_counts = []
    
    wins_count = []
    losses_count = []
    
    tp_trade_wins = []
    add_trade_wins = []
    tp_trade_losses = []
    add_trade_losses = []
    
    num_closed = len(c_data.keys()) + 1
    
    for x, y in enumerate(c_data.items()):
        k = y[0]
        # if k in bad_keys:
        #     print(f'ignoring {k}, bad key')
        #     continue
        trade = y[1]
        adds = 0 # total value of adds
        n_adds = 0 # total number of adds
        tps = 0 # total value of take-profits
        n_tps = 0 # total number of take-profits
        for n, i in enumerate(trade):
            if i.get('type')[:4] == 'open':
                entry = trade[n]
            elif i.get('type')[:3] == 'add':
                adds += float(trade[n].get('quote_size'))
                n_adds += 1
            elif i.get('type')[:2] == 'tp':
                tps += float(trade[n].get('quote_size'))
                n_tps += 1
            elif i.get('type')[:5] in ['close', 'stop_']:
                close = trade[n]
                # pprint(close)
                exit_type = i.get('type')
            
        pair = entry.get('pair')
        
        if n_adds:
            add_counts.append(n_adds)
        else:
            add_counts.append(0)
        if n_tps:
            tp_counts.append(n_tps)
        else:
            tp_counts.append(0)
        
        # if pair == 'MATICUSDT':
        #     pprint(trade)
        
        # entry_price = entry.get('exe_price')
        # exit_price = close.get('exe_price')
        # TODO this calc needs a lot of work, i'm not getting all the data about each trade yet
        entry_net = float(entry.get('quote_size')) + adds
        exit_net = float(close.get('quote_size')) + tps
        if entry_net == 0:
            continue        
        
        pnl = 100 * (exit_net - entry_net) / entry_net
        # print(f'{entry_net = }, {exit_net = }, pnl = {pnl}%')
        
        trig = float(entry.get('exe_price'))
        sl = float(entry.get('hard_stop'))
        r = 100 * (trig-sl) / sl
        final_r = pnl / r
        
        all_r.append(final_r)
        
        trade_start = int(entry.get('timestamp')) / 1000
        
        if int(close.get('timestamp')):
            trade_end = int(close.get('timestamp')) / 1000
        else:
            trade_end = trade_start + 3600
        
        duration = round((trade_end - trade_start) / 3600, 1)
        
        all_open_dates.append(trade_start)
        
        win = 'win' if final_r > 0 else ''
        
        if win:
            wins_count.append(final_r)
        else:
            losses_count.append(final_r)
        if tps and win:
            tp_trade_wins.append(final_r)
        if adds and win:
            add_trade_wins.append(final_r)
        if tps and not win:
            tp_trade_losses.append(final_r)
        if adds and not win:
            add_trade_losses.append(final_r)
        
        if x > num_closed-window and print_all == 2:
            exit_str = 'tp + ' if tps else ''
            if exit_type[0] == 'c':
                exit_str += 'trailing stop'
            elif exit_type[0] == 's':
                exit_str += 'invalidated'
            print(f'{pair} {exit_str}, final pnl: {final_r:.2f}R, duration: {duration} hours {win}\n')
    
    # all trades
    q = len(tp_trade_wins)
    s = len(wins_count)
    u = len(tp_trade_losses)
    v = len(losses_count)
    if s:
        tps_in_wins = q / s
    else:
        tps_in_wins = 0
    if v:
        tps_in_losses = u / v
    else:
        tps_in_losses = 0
    tp_ws_over_tp_ls = q / (u + q)
    print(f'\n-- All Trades ({s+v}) --')
    print(f'{tps_in_wins*100:.2f}% of wins took profit ({q}/{s})')
    print(f'{tps_in_losses*100:.2f}% of losses took profit ({u}/{v})')
    print(f'{tp_ws_over_tp_ls*100:.2f}% of trades with TPs were wins ({q}/{u + q})')
    print(f'Total closed trades: {len(all_r)}\n')
    
    # last 50 trades
    print(f'-- Last {window} Trades --')
    all_recent_r = all_r[-1*window:]
    recent_r_total = sum(all_recent_r)
    recent_r_avg = stats.mean(all_recent_r)
    
    recent_wins = [x for x in all_recent_r if x>0]
    recent_losses = [x for x in all_recent_r if x<=0]
    
    avg_win = stats.mean(recent_wins)
    avg_loss = stats.mean(recent_losses)
    
    wins = 0
    for i in all_recent_r:
        if i > 0:
            wins += 1
    winrate = round(100*wins/len(all_recent_r))
    
    add_tp_str = ''
    add_counts = add_counts[-1*window:]
    add_counts = [a for a in add_counts if a]
    tp_counts = tp_counts[-1*window:]
    tp_counts = [t for t in tp_counts if t]
    if add_counts:
        trade_adds = len(add_counts)
        add_tp_str = add_tp_str + str(trade_adds) + ' trades added '
    if tp_counts:
        trade_tps = len(tp_counts)
        add_tp_str = add_tp_str + str(trade_tps) + ' trades TPed'
    
    
    window_trades_start = all_open_dates[-1*window] if window < len(all_open_dates) else all_open_dates[0]
    window_duration = round((now.timestamp() - window_trades_start) / 3600, 1)
    print(f'avg recent win: {avg_win:.2f}R, avg recent loss: {avg_loss:.2f}R')
    print(f'Total {round(recent_r_total, 1)}R, (avg recent ev: {stats.mean(all_recent_r):.2f}R)')
    print(f'{winrate}% winrate spanning {window_duration/24:.1f} days {add_tp_str}')
    
else:
    print('no closed trades yet')
            
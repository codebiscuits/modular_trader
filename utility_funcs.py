import json, keys
from json.decoder import JSONDecodeError
import binance_funcs as funcs
from binance.client import Client
from pathlib import Path
from config import ohlc_data
import pandas as pd
from datetime import datetime, timedelta
import statistics as stats
from pushbullet import Pushbullet
import time
from pprint import pprint
from decimal import Decimal

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
# now = datetime.now().strftime('%d/%m/%y %H:%M')


def open_trade_stats(now, total_bal, v):
    '''inputs are key and value from the open trades dictionary
    returns current profit denominated in R and in %'''
    
    pair = v[0].get('pair')
    
    if v[0].get('type')[:4] != 'open':
        print('Warning - {pair} record missing open trade')
    
    current_base_size = 0
    for i in v:
        if i.get('type') in ['open_long', 'add_long', 'tp_short', 'close_short']:
            current_base_size += Decimal(i.get('base_size'))
        elif i.get('type') in ['open_short', 'add_short', 'tp_long', 'close_long']:
            current_base_size -= Decimal(i.get('base_size'))
    
    entry_price = float(v[0].get('exe_price'))
    curr_price = funcs.get_price(pair)
    pnl = 100 * (curr_price - entry_price) / entry_price
    
    open_time = v[0].get('timestamp') / 1000
    duration = round((now.timestamp() - open_time) / 3600, 1)
    
    trig = float(v[0].get('trig_price'))
    sl = float(v[0].get('hard_stop'))
    r = 100 * (trig-sl) / sl
    
    value = round(float(current_base_size) * curr_price, 2)
    pf_pct = round(100 * value / total_bal, 5)
    
    return {'qty': str(current_base_size), 'value': str(value), 'pf%': pf_pct, 
            'pnl_R': round(pnl / r, 5), 'pnl_%': round(pnl, 5), 
            'entry_price': entry_price, 'duration (h)': duration}

def adjust_max_positions(max_pos, sizing):
    '''the max_pos input tells the function what the strategy has as a default
    the sizing input is the dictionary of currently open positions with their 
    associated open risk
    
    this function decides if there should currently be a limit on how many positions
    can be open at once, based on current performance of currently open positions'''
    
def max_init_risk(n, target_risk):
    '''n = number of open positions, target_risk is the percentage distance 
    from invalidation this function should converge on, max_pos is the maximum
    number of open positions as set in the main script
    
    this function takes a target max risk and adjusts that up to 2x target depending
    on how many positions are currently open.
    whatever the output is, the system will ignore entry signals further away 
    from invalidation than that. if there are a lot of open positions, i want
    to be more picky about what new trades i open, but if there are few trades
    available then i don't want to be so picky
    
    the formula is set so that when there are no trades currently open, the 
    upper limit on initial risk will be twice as high as the main script has
    set, and as more trades are opened, that upper limit comes down relatively
    quickly, then gradually settles on the target limit'''
    
    if n > 20:
        n = 20
    
    exp = 4
    scale = (20-n)**exp
    scale_limit = 20 ** exp
    
    # when n is 0, scale and scale_limit cancel out
    # and the whole thing becomes (2 * target) + target
    output = (2 * target_risk  * scale / scale_limit) + target_risk
    # print(f'mir output: {round(output, 2) * 100}%')
        
    return round(output, 2)

def record_open_trades(strat):
    with open(f"{strat.market_data}/{strat.name}_open_trades.json", "w") as ot_file:
        json.dump(strat.open_trades, ot_file)

def record_sim_trades(strat):
    with open(f"{strat.market_data}/{strat.name}_sim_trades.json", "w") as st_file:
        json.dump(strat.sim_trades, st_file)

def record_tracked_trades(strat):
    with open(f"{strat.market_data}/{strat.name}_tracked_trades.json", "w") as tr_file:
        json.dump(strat.tracked_trades, tr_file)

def record_closed_trades(strat):
    with open(f"{strat.market_data}/{strat.name}_closed_trades.json", "w") as ct_file:
        json.dump(strat.closed_trades, ct_file)

def record_closed_sim_trades(strat):
    with open(f"{strat.market_data}/{strat.name}_closed_sim_trades.json", "w") as cs_file:
        json.dump(strat.closed_sim_trades, cs_file)

def backup_trade_records(strat):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    if strat.open_trades:
        with open(f"{strat.market_data}/{strat.name}_ot_backup.json", "w") as ot_file:
            json.dump(strat.open_trades, ot_file)
    else:
        pb.push_note(now, 'open trades file empty')
    
    if strat.sim_trades:
        with open(f"{strat.market_data}/{strat.name}_st_backup.json", "w") as st_file:
            json.dump(strat.sim_trades, st_file)
    else:
        pb.push_note(now, 'sim trades file empty')
    
    if strat.tracked_trades:
        with open(f"{strat.market_data}/{strat.name}_tr_backup.json", "w") as tr_file:
            json.dump(strat.tracked_trades, tr_file)
    else:
        pb.push_note(now, 'tracked trades file empty')
    
    if strat.closed_trades:
        with open(f"{strat.market_data}/{strat.name}_ct_backup.json", "w") as ct_file:
            json.dump(strat.closed_trades, ct_file)
    else:
        pb.push_note(now, 'closed trades file empty')
    
    if strat.closed_sim_trades:
        with open(f"{strat.market_data}/{strat.name}_cs_backup.json", "w") as cs_file:
            json.dump(strat.closed_sim_trades, cs_file)
    else:
        pb.push_note(now, 'closed sim trades file empty')

def market_benchmark(live):
    all_1d = []
    all_1w = []
    all_1m = []
    btc_1d = None
    btc_1w = None
    btc_1m = None
    eth_1d = None
    eth_1w = None
    eth_1m = None
        
    for x in ohlc_data.glob('*.*'):
        df = pd.read_pickle(x)
        if len(df) > 721:
            df = df.tail(721)
            df.reset_index(inplace=True)
        last_idx = len(df) - 1
        last_stamp = df.at[last_idx, 'timestamp']
        now = datetime.now()
        window = timedelta(hours=4)
        if last_stamp > now - window: # if there is data up to the last 4 hours
            if len(df) >= 25:
                df['roc_1d'] = df.close.pct_change(24)
                all_1d.append(df.at[last_idx, 'roc_1d'])
            if len(df) >= 169:
                df['roc_1w'] = df.close.pct_change(168)
                all_1w.append(df.at[last_idx, 'roc_1w'])
            if len(df) >= 721:
                df['roc_1m'] = df.close.pct_change(720)
                all_1m.append(df.at[last_idx, 'roc_1m'])
            if x.stem == 'BTCUSDT':
                btc_1d = df.at[last_idx, 'roc_1d']
                btc_1w = df.at[last_idx, 'roc_1w']
                btc_1m = df.at[last_idx, 'roc_1m']
            elif x.stem == 'ETHUSDT':
                eth_1d = df.at[last_idx, 'roc_1d']
                eth_1w = df.at[last_idx, 'roc_1w']
                eth_1m = df.at[last_idx, 'roc_1m']
    market_1d = stats.median(all_1d) if len(all_1d)>3 else 0
    market_1w = stats.median(all_1w) if len(all_1w)>3 else 0
    market_1m = stats.median(all_1m) if len(all_1m)>3 else 0
    print(f'1d median based on {len(all_1d)} data points')
    print(f'1w median based on {len(all_1w)} data points')
    print(f'1m median based on {len(all_1m)} data points')
    
    all_pairs = len(list(ohlc_data.glob('*.*')))
    valid_pairs = len(all_1d)
    if valid_pairs:
        valid = True
        if all_pairs / valid_pairs > 1.5:
            print('warning (strat benchmark): lots of pairs ohlc data not up to date')
    else:
        valid = False
    
    if live:
        print(f'pairs with recent data: {len(all_1d)} / {len(list(ohlc_data.glob("*.*")))}')
    
    return {'btc_1d': btc_1d, 'btc_1w': btc_1w, 'btc_1m': btc_1m, 
            'eth_1d': eth_1d, 'eth_1w': eth_1w, 'eth_1m': eth_1m, 
            'market_1d': market_1d, 'market_1w': market_1w, 'market_1m': market_1m, 
            'valid': valid}

def strat_benchmark(strat, benchmark):
    now = datetime.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    bal_now, bal_1d, bal_1w, bal_1m = None, None, None, None
    
    with open(f"{strat.market_data}/{strat.name}_bal_history.txt", "r") as file:
        bal_data = file.readlines()
    
    bal_now = json.loads(bal_data[-1]).get('balance')
    
    for row in bal_data[-1:0:-1]:
        row = json.loads(row)
        row_dt = datetime.strptime(row.get('timestamp'), '%d/%m/%y %H:%M')
        if row_dt < month_ago and not bal_1m:
            try:
                bal_1m = row.get('balance')
            except AttributeError:
                continue 
        if row_dt < week_ago and not bal_1w:
            try:
                bal_1w = row.get('balance')
            except AttributeError:
                continue
        if row_dt < day_ago and not bal_1d:
            try:
                bal_1d = row.get('balance')
            except AttributeError:
                continue
        
    strat_1d = (bal_now - bal_1d) / bal_1d
    strat_1w = (bal_now - bal_1w) / bal_1w
    strat_1m = (bal_now - bal_1m) / bal_1m
    
    benchmark['strat_1d'] = strat_1d
    benchmark['strat_1w'] = strat_1w
    benchmark['strat_1m'] = strat_1m
    
    
    return benchmark 

def log(live, strat, fixed_risk, spreads, now_start):    
    
    # check total balance and record it in a file for analysis
    total_bal = funcs.account_bal()
    
    params = {'quote_asset': 'USDT', 
              'fr_range': strat.fr_range,
              'max_spread': strat.max_spread, 
              'indiv_r_limit': strat.indiv_r_limit, 
              'total_r_limit': strat.total_r_limit, 
              'target_risk': strat.target_risk, 
              'max_pos': strat.max_positions}
    
    bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'fr': 
                  fixed_risk, 'positions': strat.sizing, 'params': params, 
                  'trade_counts': strat.counts_dict, 
                  'realised_pnl': strat.realised_pnl, 'sim_r_pnl': strat.sim_pnl}
    new_line = json.dumps(bal_record)
    if live:
        with open(f"{strat.market_data}/{strat.name}_bal_history.txt", "a") as file:
            file.write(new_line)
            file.write('\n')
    
    # if live:
    benchmark = market_benchmark(live)
    benchmark = strat_benchmark(strat, benchmark)

    # save a json of any trades that have happened with relevant data
    if live:
        # should be able to remove these two function calls
        with open(f"{strat.market_data}/{strat.name}_open_trades.json", "w") as ot_file:
            json.dump(strat.open_trades, ot_file)            
        with open(f"{strat.market_data}/{strat.name}_closed_trades.json", "w") as ct_file:
            json.dump(strat.closed_trades, ct_file)
    
    return benchmark

def interpret_benchmark(benchmark):
    if benchmark['valid']:
        d_ranking = [
            ('btc', round(benchmark['btc_1d']*100, 3)), 
            ('eth', round(benchmark['eth_1d']*100, 3)), 
            ('mkt', round(benchmark['market_1d']*100, 3)), 
            ('strat', round(benchmark['strat_1d']*100, 3))
            ]
        d_ranking = sorted(d_ranking, key=lambda x: x[1], reverse=True)
        print('1 day stats')
        for e, r in enumerate(d_ranking):
            print(f'rank {e+1}: {r[0]} {r[1]}%')
        w_ranking = [
            ('btc', round(benchmark['btc_1w']*100, 2)), 
            ('eth', round(benchmark['eth_1w']*100, 2)), 
            ('mkt', round(benchmark['market_1w']*100, 2)), 
            ('strat', round(benchmark['strat_1w']*100, 2))
            ]
        w_ranking = sorted(w_ranking, key=lambda x: x[1], reverse=True)
        print('1 week stats')
        for e, r in enumerate(w_ranking):
            print(f'rank {e+1}: {r[0]} {r[1]}%')
        m_ranking = [
            ('btc', round(benchmark['btc_1m']*100, 1)), 
            ('eth', round(benchmark['eth_1m']*100, 1)), 
            ('mkt', round(benchmark['market_1m']*100, 1)), 
            ('strat', round(benchmark['strat_1m']*100, 1))
            ]
        m_ranking = sorted(m_ranking, key=lambda x: x[1], reverse=True)
        print('1 month stats')
        for e, r in enumerate(m_ranking):
            print(f'rank {e+1}: {r[0]} {r[1]}%')
    else:
        print('no benchmarking data available')

def count_trades(counts):
    count_list = []
    if counts.get("stop_count"):
        count_list.append(f'stopped: {counts.get("stop_count")}') 
    if counts.get("open_count"):
        count_list.append(f'opened: {counts.get("open_count")}') 
    if counts.get("add_count"):
        count_list.append(f'added: {counts.get("add_count")}') 
    if counts.get("tp_count"):
        count_list.append(f'tped: {counts.get("tp_count")}') 
    if counts.get("close_count"):
        count_list.append(f'closed: {counts.get("close_count")}') 
    if count_list:
        counts_str = '\n' + ', '.join(count_list)
    else:
        counts_str = ''
    
    return counts_str

def find_bad_keys(c_data):
    bad_keys = []
    for k, v in c_data.items():
        try:
            init_base = 0
            add_base = 0
            tp_base = 0
            close_base = 0
            for x in v:
                if x.get('type')[:4] == 'open':
                    init_base = float(x.get('base_size'))
                elif x.get('type')[:3] == 'add':
                    add_base += float(x.get('base_size'))
                elif x.get('type')[:2] == 'tp':
                    tp_base += float(x.get('base_size'))
                elif x.get('type')[:5] in ['close', 'stop_']:
                    close_base = float(x.get('base_size'))
            
            gross_buy = init_base + add_base
            gross_sell = tp_base + close_base
            diff = (gross_buy - gross_sell) / gross_buy
            pair = x.get('pair')
            if abs(diff) > 0.03:
                # print(f'{k} {pair} - bought: {gross_buy} sold: {gross_sell}')
                bad_keys.append({'key': k, 'pair': pair, 'buys': gross_buy, 'sells': gross_sell})
        except:
            bad_keys.append({'key': k, 'pair': pair, 'buys': gross_buy, 'sells': gross_sell})
            # print('bad key:', k)
            continue
    
    return bad_keys

def realised_pnl(strat, trade_record):
    entry = float(trade_record[0].get('exe_price'))
    init_stop = float(trade_record[0].get('hard_stop'))
    init_size = float(trade_record[0].get('base_size'))
    final_exit = float(trade_record[-1].get('exe_price'))
    final_size = float(trade_record[-1].get('base_size'))
    r_val = (entry - init_stop) / entry
    trade_pnl = (final_exit - entry) / entry
    trade_r = round(trade_pnl / r_val, 3)
    
    if trade_record[-1].get('state') == 'real':
        scalar = final_size / init_size
        realised_r = trade_r * scalar
        strat.realised_pnl += realised_r
    elif trade_record[-1].get('state') == 'sim':
        strat.sim_pnl += trade_r # realised sim pnl ignores trade size because it's often 0
    else:
        print(f'state in record: {trade_record[-1].get("state")}')
        print(f'{trade_r = }')

def latest_stop_id(trade_record):
    '''looks through trade_record for a stop id and retrieves the pair, id and 
    timestamp for when the stop was set. if nothing is found, just retreives the 
    pair and timestamp from the start of the trade_record'''
    pair = None
    stop_id = None
    stop_time = None
    for i in trade_record[::-1]:
        if i.get('stop_id'):
            pair = i.get('pair')
            stop_id = i.get('stop_id')
            stop_time = i.get('timestamp')
            break
    if not pair:
        pair = trade_record[0].get('pair')
        stop_time = trade_record[0].get('timestamp')
            
    
    return pair, stop_id, stop_time

def create_stop_dict(order):
    '''collects and returns the details of filled stop-loss order in a dictionary'''
    
    pair = order.get('symbol')
    quote_qty = order.get('cummulativeQuoteQty')
    base_qty = order.get('executedQty')
    avg_price = round(float(quote_qty) / float(base_qty), 8)
    
    bnb_fee = funcs.calc_fee_bnb(quote_qty)

    trade_dict = {'timestamp': order.get('updateTime'),
                  'pair': pair,
                  'trig_price': order.get('stopPrice'), 
                  'limit_price': order.get('price'), 
                  'exe_price': str(avg_price),
                  'base_size': base_qty,
                  'quote_size': quote_qty,
                  'fee': str(bnb_fee),
                  'fee_currency': 'BNB'
                  }
    if order.get('status') != 'FILLED':
        print(f'{pair} order not filled')
        pb.push_note('Warning', f'{pair} stop-loss hit but not filled')
    
    return trade_dict

def record_stopped_trades(strat, now, live):
    # loop through strat.open_trades and call latest_stop_id(trade_record) to
    # compile a list of order ids for each open trade's stop loss orders, then 
    # check binance to find which don't have an active stop-loss
    stop_ids = [latest_stop_id(v) for v in strat.open_trades.values()]
    
    open_orders = client.get_open_orders()
    ids_remaining = [i.get('orderId') for i in open_orders]
    symbols_remaining = [i.get('symbol') for i in open_orders]
    
    stopped = []
    for pair, sid, time in stop_ids:
        if sid:
            if sid not in ids_remaining:
                stopped.append((pair, sid, time))
            else:
                continue
        elif pair not in symbols_remaining:
            stopped.append((pair, sid, time))
    
    # for any that don't, assume that the stop was hit and check for exchange records
    for pair, sid, time in stopped:
        order_list = client.get_all_orders(symbol=pair, orderId=sid, startTime=time-10000)
        
        if order_list and sid:
            order = None
            for o in order_list:
                if o.get('order_id'):
                    order = o
                    break
        elif order_list and not sid:
            order = None
            for o in order_list[::-1]:
                if o.get('type') == 'STOP_LOSS_LIMIT':
                    order = o
                    break
        else:
            print(f'No orders on binance for {pair}')
            
        if order:
            trade_type = 'stop_short' if (order.get('side') == 'BUY') else 'stop_long'
            
            stop_dict = create_stop_dict(order)
            stop_dict['type'] = trade_type                
            stop_dict['state'] = 'real'
            stop_dict['reason'] = 'hit hard stop'
            
            trade_record = strat.open_trades.get(pair)
            trade_record.append(stop_dict)
            
            ts_id = trade_record[0].get('timestamp')
            strat.closed_trades[ts_id] = trade_record
            record_closed_trades(strat)
            del strat.open_trades[pair]
            record_open_trades(strat)
            
            strat.counts_dict['real_stop'] += 1
            realised_pnl(strat, trade_record)
            
        else:
            # check for a free balance matching the size. if there is, that means 
            # the stop was never set in the first place and needs to be set
            print(f'getting {pair[:-4]} free balance')
            free_bal = float(client.get_asset_balance(pair[:-4]).get('free'))
            print(f'getting {pair} price')
            price = funcs.get_price(pair)
            value = free_bal * price
            if value > 10:
                note = f'{pair} in position with no stop-loss'
                pb.push_note(now, note)

def record_stopped_trades_old(pairs_in_pos, now_start, strat):
    print('running record_stopped_trades')
    # create list of trade records which don't match current positions
    open_trades_list = list(strat.open_trades.keys())
    stopped_trades = [st for st in open_trades_list if st not in pairs_in_pos] # these positions must have been stopped out
    
    print(open_trades_list)
    print(stopped_trades)
    
    # look for stopped out positions and complete trade records
    for i in stopped_trades:
        print(i)
        trade_record = strat.open_trades.get(i)
        
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
        
        # aggregate trade stats
        base_size_count = 0
        agg_price = []
        agg_base = []
        agg_quote = []
        agg_fee = []
        for t in trades[::-1]:
            # if t.get('isBuyer') == close_is_buy:
            if abs((base_size_count / diff) - 1) > 0.03:
                agg_price.append(float(t.get('price')))
                agg_base.append(float(t.get('qty')))
                agg_quote.append(float(t.get('quoteQty')))
                agg_fee.append(float(t.get('commission')))
                base_size_count += float(t.get('qty'))
            else:
                # print(f'trade record completed, {round(abs((base_size_count / diff) - 1), 3)} of diff unaccounted for')
                break
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
                      'state': 'real', 
                      }
        note = f"stopped out {t.get('symbol')} @ {t.get('price')}"
        print(now_start, note)
        # push = pb.push_note(now_start, note)
        trade_record.append(trade_dict)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
            print(f'warning, trade record for {t.get("symbol")} missing trade open')
        record_closed_trades(strat)
        strat.counts_dict['real_stop'] += 1
        strat.next_id += 1
        if strat.open_trades[i]:
            del strat.open_trades[i]
            record_open_trades(strat)
        realised_pnl(strat, trade_record)

def record_stopped_sim_trades(strat, now_start):
    '''goes through all trades in the sim_trades file and checks their recent 
    price action against their most recent hard_stop to see if any of them would have 
    got stopped out'''
    
    del_pairs = []
    for pair, v in strat.sim_trades.items():
        # first filter out all trades which started out real
        if v[0].get('real'):
            continue
        
        long_trade = True if v[0].get('type')[-4:] == 'long' else False
        
        # calculate current base size
        base_size = 0
        for i in v:
            if i.get('type') in ['open_long', 'add_long', 'tp_short']:
                base_size += float(i.get('base_size'))
            else:
                base_size -= float(i.get('base_size'))
        
        # find most recent hard stop
        for i in v[-1::-1]:
            if i.get('hard_stop'):
                stop = float(i.get('hard_stop'))
                stop_time = i.get('timestamp')
                break
            
        # check lowest low since stop was set
        klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, stop_time)
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = df['timestamp'] * 1000000
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if long_trade:
            trade_type = 'stop_long'
            ll = df.low.min()
            # stop_time = find timestamp of that lowest low candle
            stopped = ll < stop
            overshoot_pct = round((100 * (stop - ll) / stop), 3) # % distance that price broke through the stop
        else:
            trade_type = 'stop_short'
            hh = df.high.max()
            # stop_time = find timestamp of that highest high candle
            stopped = hh > stop
            overshoot_pct = round((100 * (hh - stop) / stop), 3) # % distance that price broke through the stop
        
        if stopped:
            # create trade dict
            trade_dict = {'timestamp': stop_time, 
                          'pair': pair, 
                          'type': trade_type, 
                          'exe_price': str(stop), 
                          'base_size': str(base_size), 
                          'quote_size': str(round(base_size * stop, 2)), 
                          'fee': 0, 
                          'fee_currency': 'BNB', 
                          'reason': 'hit hard stop', 
                          'state': 'sim', 
                          }
            note = f"*sim* stopped out {pair} @ {stop}"
            # print(now_start, note)
            
            v.append(trade_dict)
            
            ts_id = v[0].get('timestamp')
            strat.closed_sim_trades[ts_id] = v
            record_closed_sim_trades(strat)
            
            realised_pnl(strat, v)            
            strat.counts_dict['sim_stop'] += 1
            
    for p in del_pairs:
        del strat.sim_trades[pair]
    record_sim_trades(strat)

def recent_perf_str(strat):
    '''generates a string of + and - to represent recent strat performance'''
    
    with open(f"{strat.market_data}/{strat.name}_bal_history.txt", "r") as file:
        bal_data = file.readlines()
    
    bal_0 = json.loads(bal_data[-1]).get('balance')
    bal_1 = json.loads(bal_data[-2]).get('balance')
    bal_2 = json.loads(bal_data[-3]).get('balance')
    bal_3 = json.loads(bal_data[-4]).get('balance')
    bal_4 = json.loads(bal_data[-5]).get('balance')
    
    a = '+' if bal_0 > bal_1 else '-'
    b = '+' if bal_1 > bal_2 else '-'
    c = '+' if bal_2 > bal_3 else '-'
    d = '+' if bal_3 > bal_4 else '-'
    
    return f'  {a} | {b} {c} {d}'

def scanner_summary(strat, all_start, benchmark, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    # all_end = time.perf_counter()
    # all_time = all_end - all_start
    
    total_bal = funcs.account_bal()
    
    or_list = [v.get('or_$') for v in strat.sizing.values() if v.get('or_$')]
    # dollar_tor = round(sum(or_list), 2)
    num_open_positions = len(or_list)
    # rfb = round(total_bal-dollar_tor, 2)
    vol_exp = round(100 - strat.sizing.get('USDT').get('pf%'))
    
    live_str = '' if live else '*not live* '
    elapsed_str = '' # f'Time taken: {round((all_time) // 60)}m {round((all_time) % 60)}s, '
    count_str = count_trades(strat.counts_dict)
    perf_str = recent_perf_str(strat)
    bench_str = f"1m perf: strat {round(benchmark.get('strat_1m')*100, 2)}%, mkt {round(benchmark.get('market_1m')*100, 2)}%"
    final_msg = f'{live_str}{elapsed_str}total bal: ${total_bal:.2f} {perf_str}\npositions {num_open_positions}, exposure {vol_exp}% {count_str}\n{bench_str}'
    print(final_msg)
    
    if live:
        pb.push_note(now, final_msg)

def sync_test_records(strat):
    with open(f"{strat.market_data}/{strat.name}_bal_history.txt", "r") as file:
        bal_data = file.readlines()
    with open(f"test_records/{strat.name}_bal_history.txt", "w") as file:
        file.writelines(bal_data)
    
    
    try:
        with open(f'{strat.market_data}/{strat.name}_open_trades.json', 'r') as file:
            o_data = json.load(file)
        with open(f'test_records/{strat.name}_open_trades.json', 'w') as file:
            json.dump(o_data, file)
    except JSONDecodeError:
        print('open_trades file empty')
    
    
    try:
        with open(f'{strat.market_data}/{strat.name}_sim_trades.json', 'r') as file:
            s_data = json.load(file)
        with open(f'test_records/{strat.name}_sim_trades.json', 'w') as file:
            json.dump(s_data, file)
    except JSONDecodeError:
        print('sim_trades file empty')
    
    
    try:
        with open(f'{strat.market_data}/{strat.name}_tracked_trades.json', 'r') as file:
            tr_data = json.load(file)
        with open(f'test_records/{strat.name}_tracked_trades.json', 'w') as file:
            json.dump(tr_data, file)
    except JSONDecodeError:
        print('tracked_trades file empty')
    
    
    try:
        with open(f'{strat.market_data}/{strat.name}_closed_trades.json', 'r') as file:
            c_data = json.load(file)
        with open(f'test_records/{strat.name}_closed_trades.json', 'w') as file:
            json.dump(c_data, file)
    except JSONDecodeError:
        print('closed_trades file empty')


    try:
        with open(f'{strat.market_data}/{strat.name}_closed_sim_trades.json', 'r') as file:
            cs_data = json.load(file)
        with open(f'test_records/{strat.name}_closed_sim_trades.json', 'w') as file:
            json.dump(cs_data, file)
    except JSONDecodeError:
        print('closed_sim_trades file empty')


    with open(f'{strat.market_data}/binance_liquidity_history.txt', 'r') as file:
        book_data = file.readlines()
    with open('test_records/binance_liquidity_history.txt', 'w') as file:
        file.writelines(book_data)

def set_max_pos_old(strat):
    if strat.sizing:
        open_pnls = [v.get('pnl') for v in strat.sizing.values() if v.get('pnl')]
        if open_pnls:
            avg_open_pnl = stats.median(open_pnls)
        else:
            avg_open_pnl = 0
        strat.max_pos = 20 if avg_open_pnl <= 0 else 50
    else:
        strat.max_pos = 20

def calc_pos_fr_dol(trade_record, fixed_risk_dol, in_pos, switch):   
    if in_pos[switch] and trade_record and trade_record[0].get('type')[0] == 'o':
        qs = float(trade_record[0].get('quote_size'))
        ep = float(trade_record[0].get('exe_price'))
        hs = float(trade_record[0].get('hard_stop'))
        pos_fr_dol = qs * ((ep - hs) / ep)
    else:
        ep = None # i refer to this later and need it to exist even if it has no value
        pos_fr_dol = fixed_risk_dol
    
    in_pos[f'{switch}_pfrd'] = pos_fr_dol
    in_pos[f'{switch}_ep'] = ep

    return in_pos

def calc_sizing_non_live_tp(strat, asset, tp_pct, switch):
    tp_scalar = 1 - (100 / tp_pct)
    qty = strat.sizing.get(asset).get('qty') * tp_scalar
    val = strat.sizing.get(asset).get('value') * tp_scalar
    pf = strat.sizing.get(asset).get('pf%') * tp_scalar
    or_R = strat.sizing.get(asset).get('or_R') * tp_scalar
    or_dol = strat.sizing.get(asset).get('or_$') * tp_scalar
    if switch == 'real':
        strat.sizing[asset].update({'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol})
    elif switch == 'sim':
        strat.sim_pos[asset].update({'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol})

def too_new(df, in_pos):
    if in_pos['real'] or in_pos['sim'] or in_pos['tracked']:
        no_pos = False
    else:
        no_pos = True    
    return len(df) <= 200 and no_pos


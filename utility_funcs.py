import json, keys
from json.decoder import JSONDecodeError
import binance_funcs as funcs
from pprint import pprint
from binance.client import Client
from binance.exceptions import BinanceAPIException

client = Client(keys.bPkey, keys.bSkey)

def max_init_risk(n, target_risk, max_pos):
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
    
    exp = 4
    scale = (max_pos-n)**exp
    scale_limit = max_pos ** exp
    
    # when n is 0, scale and scale_limit cancel out
    # and the whole thing becomes (2 * target) + target
    output = (2 * target_risk  * scale / scale_limit) + target_risk
    # print(f'mir output: {round(output, 2) * 100}%')
        
    return round(output, 2)

def record_open_trades(strat_name, market_data, ot):
    with open(f"{market_data}/{strat_name}_open_trades.json", "w") as ot_file:
        json.dump(ot, ot_file)

def record_closed_trades(strat_name, market_data, ct):
    with open(f"{market_data}/{strat_name}_closed_trades.json", "w") as ct_file:
        json.dump(ct, ct_file)

def log(live, params, strat, market_data, spreads, 
        now_start, sizing, tp_trades, 
        non_trade_notes, counts_dict, ot, closed_trades):    
    
    # check total balance and record it in a file for analysis
    total_bal = funcs.account_bal()
    bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'positions': sizing, 'params': params, 'trade_counts': counts_dict}
    new_line = json.dumps(bal_record)
    if live:
        with open(f"{market_data}/{strat.name}_bal_history.txt", "a") as file:
            file.write(new_line)
            file.write('\n')
    # else:
    #     pprint(new_line)
    
    # save a json of any trades that have happened with relevant data
    if live:
        # should be able to remove these two function calls
        with open(f"{market_data}/{strat.name}_open_trades.json", "w") as ot_file:
            json.dump(ot, ot_file)            
        with open(f"{market_data}/{strat.name}_closed_trades.json", "w") as ct_file:
            json.dump(closed_trades, ct_file)
    
    # record all skipped or reduced trades
    non_trade_record = {'timestamp': now_start, 'non_trades': non_trade_notes}
    if live:
        with open(f"{market_data}/{strat.name}_non_trades.txt", "a") as file:
            file.write(json.dumps(non_trade_record))
            file.write('\n')
    # else:
    #     pprint(non_trade_record)

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
    counts_str = ', '.join(count_list)
    
    return counts_str

def read_trade_records(market_data, strat_name):
    ot_path = f"{market_data}/{strat_name}_open_trades.json"
    with open(ot_path, "r") as ot_file:
        try:
            open_trades = json.load(ot_file)
        except JSONDecodeError:
            open_trades = {}
    ct_path = f"{market_data}/{strat_name}_closed_trades.json"
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
    
    return open_trades, closed_trades, next_id

def record_stopped_trades(open_trades, closed_trades, pairs_in_pos, now_start, 
                          next_id, strat, market_data, counts_dict):
    # create list of trade records which don't match current positions
    open_trades_list = list(open_trades.keys())
    stopped_trades = [st for st in open_trades_list if st not in pairs_in_pos] # these positions must have been stopped out

    # look for stopped out positions and complete trade records
    for i in stopped_trades:
        trade_record = open_trades.get(i)
        
        # work out how much base size has been bought vs sold
        init_base = 0
        add_base = 0
        tp_base = 0
        close_base = 0
        for x in trade_record:
            if x.get('type')[:4] == 'open':
                init_base = x.get('base_size')
            elif x.get('type')[:3] == 'add':
                add_base += x.get('base_size')
            elif x.get('type')[:2] == 'tp':
                tp_base += x.get('base_size')
            elif x.get('type')[:5] == 'close':
                close_base = x.get('base_size')
        
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
            if abs((base_size_count / diff) - 1) > 0.1:
                agg_price.append(float(t.get('price')))
                agg_base.append(float(t.get('qty')))
                agg_quote.append(float(t.get('quoteQty')))
                agg_fee.append(float(t.get('commission')))
                base_size_count += float(t.get('qty'))
            else:
                print(f'trade record completed, {round(abs((base_size_count / diff) - 1), 3)} of diff unaccounted for')
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
                      }
        note = f"stopped out {t.get('symbol')} @ {t.get('price')}"
        print(now_start, note)
        # push = pb.push_note(now_start, note)
        trade_record.append(trade_dict)
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            closed_trades[trade_id] = trade_record
        else:
            closed_trades[next_id] = trade_record
            print(f'warning, trade record for {t.get("symbol")} missing trade open')
        record_closed_trades(strat.name, market_data, closed_trades)
        counts_dict['stop_count'] += 1
        next_id += 1
        if open_trades[i]:
            del open_trades[i]
            record_open_trades(strat.name, market_data, open_trades)
    
    return next_id, counts_dict


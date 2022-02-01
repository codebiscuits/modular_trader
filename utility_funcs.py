import json
import binance_funcs as funcs
from pprint import pprint

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
        non_trade_notes, ot, closed_trades):    
    
    # check total balance and record it in a file for analysis
    total_bal = funcs.account_bal()
    bal_record = {'timestamp': now_start, 'balance': round(total_bal, 2), 'positions': sizing, 'params': params}
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

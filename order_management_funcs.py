import binance_funcs as funcs
import utility_funcs as uf
import strategies
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from pprint import pprint

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)


def spot_buy(strat, pair, fixed_risk, size, usdt_size, price, stp, inval_dist, pos_fr_dol, params, market_data, open_trades, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
    in_pos = False
    print(now, note)
    if live:
        try:
            api_order = funcs.buy_asset(pair, usdt_size, live)
            buy_order = funcs.create_trade_dict(api_order, price, live)
            buy_order['type'] = 'open_long'
            buy_order['reason'] = 'buy conditions met'
            buy_order['hard_stop'] = stp
            open_trades[pair] = [buy_order]
            uf.record_open_trades(strat.name, market_data, open_trades)
            stop_order = funcs.set_stop(pair, stp, live)
            in_pos = True
            strat.sizing[asset] = funcs.update_pos(asset, strat.bal, inval_dist, pos_fr_dol)
            strat.sizing[asset]['pnl_R'] = 0
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['open_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with buy order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} buy order')
    else:
        try:
            api_order = funcs.buy_asset(pair, usdt_size, live)
            buy_order = funcs.create_trade_dict(api_order, price, live)
            buy_order['type'] = 'open_long'
            buy_order['reason'] = 'buy conditions met'
            buy_order['hard_stop'] = stp
            open_trades[pair] = [buy_order]
            uf.record_open_trades(strat.name, 'test_records', open_trades)
            stop_order = funcs.set_stop(pair, stp, live)
            in_pos = True
            pf = usdt_size / strat.bal
            or_dol = strat.bal * fixed_risk
            strat.sizing[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
            strat.counts_dict['open_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sim buy order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during sim {pair} buy order')
    
    return open_trades, in_pos

def spot_strat_tp(strat, pair, price, stp, inval_dist, pos_fr_dol, trade_record, open_trades, market_data, live):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"{pair} take-profit @ {price}"
    print(now, note)
    if live:
        try:
            funcs.clear_stop(pair, live)
            api_order = funcs.sell_asset(pair, live, 50)
            tp_order = funcs.create_trade_dict(api_order, price, live)
            tp_order['type'] = 'tp_long'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            strat.sizing[asset].update(funcs.update_pos(asset, strat.bal, inval_dist, pos_fr_dol))
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['tp_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} tp order')
    else:
        try:
            funcs.clear_stop(pair, live)
            api_order = funcs.sell_asset(pair, live, 50)
            tp_order = funcs.create_trade_dict(api_order, price, live)
            tp_order['type'] = 'tp_long'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, 'test_records', open_trades)
            qty = strat.sizing.get(asset).get('qty') / 2
            val = strat.izing.get(asset).get('value') / 2
            pf = strat.sizing.get(asset).get('pf%') / 2
            or_R = strat.sizing.get(asset).get('or_R') / 2
            or_dol = strat.sizing.get(asset).get('or_$') / 2
            strat.sizing[asset].update({'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol})
            strat.counts_dict['tp_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during sim {pair} tp order')
            
    return trade_record

def spot_sell(strat, pair, price, next_id, trade_record, open_trades, closed_trades, market_data, live):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"{pair} hit trailing stop @ {price}"
    print(now, note)
    if live:
        try:
            funcs.clear_stop(pair, live)
            api_order = funcs.sell_asset(pair, live)
            sell_order = funcs.create_trade_dict(api_order, price, live)
            sell_order['type'] = 'close_long'
            sell_order['reason'] = 'hit trailing stop'
            trade_record.append(sell_order)
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = trade_record[0].get('timestamp')
                closed_trades[trade_id] = trade_record
            else:
                closed_trades[next_id] = trade_record
            uf.record_closed_trades(strat.name, market_data, closed_trades)
            next_id += 1
            if open_trades[pair]:
                del open_trades[pair]
                uf.record_open_trades(strat.name, market_data, open_trades)
            in_pos = False
            del strat.sizing[asset]
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['close_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} sell order')
    else:
        try:
            funcs.clear_stop(pair, live)
            api_order = funcs.sell_asset(pair, live)
            sell_order = funcs.create_trade_dict(api_order, price, live)
            sell_order['type'] = 'close_long'
            sell_order['reason'] = 'hit trailing stop'
            trade_record.append(sell_order)
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = trade_record[0].get('timestamp')
                closed_trades[trade_id] = trade_record
            else:
                closed_trades[next_id] = trade_record
                next_id += 1
            uf.record_closed_trades(strat.name, 'test_records', closed_trades)
            if open_trades[pair]:
                del open_trades[pair]
                uf.record_open_trades(strat.name, 'test_records', open_trades)
            in_pos = False
            strat.sizing['USDT']['value'] += strat.sizing.get(asset).get('value')
            del strat.sizing[asset]
            strat.counts_dict['close_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during sim {pair} sell order')
            
    return open_trades, closed_trades, in_pos

def spot_risk_limit_tp(strat, pair, tp_pct, price, price_delta, trade_record, open_trades, closed_trades, next_id, market_data, stp, inval_dist, pos_fr_dol, in_pos, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    if live:
        note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        print(now, note)
        funcs.clear_stop(pair, live)
        api_order = funcs.sell_asset(pair, live, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, live)
        if tp_pct == 100:
            tp_order['type'] = 'close_long'
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)  
            
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = trade_record[0].get('timestamp')
                closed_trades[trade_id] = trade_record
            else:
                closed_trades[next_id] = trade_record
            uf.record_closed_trades(strat.name, market_data, closed_trades)
            next_id += 1
            if open_trades[pair]:
                del open_trades[pair]
                uf.record_open_trades(strat.name, market_data, open_trades)
            in_pos = False
            del strat.sizing[asset]
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['close_count'] += 1
        else:
            tp_order['type'] = 'tp_long'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades[pair] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            strat.sizing[asset].update(funcs.update_pos(asset, strat.bal, inval_dist, pos_fr_dol))
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['tp_count'] += 1
        
    else:
        note = f"sim {pair} take profit"
        print(now, note)
        funcs.clear_stop(pair, live)
        api_order = funcs.sell_asset(pair, live, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, live)
        tp_order['type'] = 'tp_long'
        stop_order = funcs.set_stop(pair, stp, live)
        tp_order['hard_stop'] = stp
        tp_order['reason'] = 'position R limit exceeded'
        trade_record.append(tp_order)
        open_trades[pair] = trade_record
        uf.record_open_trades(strat.name, 'test_records', open_trades)
        qty = strat.sizing.get(asset).get('qty') / 2
        val = strat.sizing.get(asset).get('value') / 2
        pf = strat.sizing.get(asset).get('pf%') / 2
        or_R = strat.sizing.get(asset).get('or_R') / 2
        or_dol = strat.sizing.get(asset).get('or_$') / 2
        print(strat.sizing[asset])
        strat.sizing[asset].update({'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol})
        strat.counts_dict['tp_count'] += 1
    
    return open_trades, closed_trades, in_pos

def reduce_risk(strat, params, open_trades, closed_trades, market_data, next_id, live):
    r_limit = params.get('total_r_limit')
    
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R')) 
                 for p, r in strat.sizing.items() 
                 if r.get('or_R') and r.get('or_R') > 0]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in strat.sizing.values() if x.get('or_R')]
        total_r = sum(r_list)
        counted = len(r_list)        
        
        for pos in sorted_pos:
            if total_r > r_limit and pos[1] > 1:
                print(f'*** tor: {total_r:.1f}, reducing risk ***')
                pair = pos[0] + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"reduce risk {pair} @ {price}"
                
                print(now, note)
                if live:
                    try:
                        # push = pb.push_note(now, note)
                        funcs.clear_stop(pair, live)
                        api_order = funcs.sell_asset(pair, live)
                        sell_order = funcs.create_trade_dict(api_order, price, live)
                        sell_order['type'] = 'close_long'
                        sell_order['reason'] = 'portfolio risk limiting'
                        if open_trades.get(pair):
                            trade_record = open_trades.get(pair)
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
                        if open_trades[pair]:
                            del open_trades[pair]
                            uf.record_open_trades(strat.name, market_data, open_trades)
                        strat.counts_dict['close_count'] += 1
                        total_r -= pos[1]
                        del strat.sizing[pos[0]]
                    except BinanceAPIException as e:
                        print(f'problem with sell order for {pair}')
                        print(e)
                        pb.push_note(now, f'exeption during {pair} sell order')
                else:
                    funcs.clear_stop(pair, live)
                    api_order = funcs.sell_asset(pair, live)
                    sell_order = funcs.create_trade_dict(api_order, price, live)
                    sell_order['type'] = 'close_long'
                    sell_order['reason'] = 'portfolio risk limiting'
                    if open_trades.get(pair):
                        trade_record = open_trades.get(pair)
                    else:
                        trade_record = []
                    trade_record.append(sell_order)
                    if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                        trade_id = trade_record[0].get('timestamp')
                        closed_trades[trade_id] = trade_record
                    else:
                        closed_trades[next_id] = trade_record
                    uf.record_closed_trades(strat.name, 'test_records', closed_trades)
                    next_id += 1
                    if open_trades[pair]:
                        del open_trades[pair]
                        uf.record_open_trades(strat.name, 'test_records', open_trades)
                    strat.counts_dict['close_count'] += 1
                    total_r -= pos[1]
                    del strat.sizing[pos[0]]
    
    return open_trades, closed_trades, next_id



def margin_open_long(strat, pair, size, stp, inval_dist, pos_fr_dol, open_trades, market_data, in_pos, live):
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"open long {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
    print(now, note)
    asset = pair[:-4]
    strat.bal = funcs.account_bal_M()
    
    if live:
        api_order = funcs.open_long(pair, usdt_size)
        long_order = funcs.create_trade_dict(api_order, price, live)
        long_order['type'] = 'open_long'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = stp
        open_trades[pair] = [long_order]
        uf.record_open_trades(strat.name, market_data, open_trades)
        funcs.set_stop_M(pair, long_order, be.SIDE_SELL, stp, stp*0.9)
        in_pos = True
        strat.sizing[asset] = funcs.update_pos_M(asset, strat.bal, inval_dist, pos_fr_dol)
        strat.sizing['USDT'] = funcs.update_usdt_M(strat.bal)
        strat.counts_dict['open_count'] += 1
    else:
        api_order = {'symbol': pair, 'price': funcs.get_price(pair), 'quote_size': usdt_size}
        long_order = funcs.create_trade_dict(api_order, price, live)
        long_order['type'] = 'open_long'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = stp
        open_trades[pair] = [long_order]
        uf.record_open_trades(strat.name, market_data, open_trades)
        
    return in_pos

def margin_tp_long(pair, pct, stp, live):
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"tp long {pct} {pair} @ {price}, new stop @ {stp:.5}"
    print(now, note)
    
    if live:
        api_order = funcs.close_long(pair, pct)
    else:
        pass

def margin_close_long(pair, live):
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"long {pair} hit trailing stop @ {price}"
    print(now, note)
    
    if live:
        pass
    else:
        pass

def margin_open_short(pair, size, stp, live):
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"open short {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
    print(now, note)
    
    if live:
        pass
    else:
        pass

def margin_tp_short(pair, pct, stp, live):
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"tp short {pct} {pair} @ {price}, new stop @ {stp:.5}"
    print(now, note)
    
    if live:
        pass
    else:
        pass

def margin_close_short(pair, live):
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"short {pair} hit trailing stop @ {price}"
    print(now, note)
    
    if live:
        pass
    else:
        pass


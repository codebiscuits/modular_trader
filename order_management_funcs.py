import binance_funcs as funcs
import utility_funcs as uf
import strategies
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


def spot_buy(strat, pair, fixed_risk, size, usdt_size, price, stp, sizing, total_bal, inval_dist, pos_fr_dol, params, market_data, counts_dict, open_trades, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    note = f"buy {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
    print(now, note)
    if live:
        try:
            buy_order = funcs.buy_asset(pair, usdt_size, live)
            buy_order['type'] = 'open_long'
            buy_order['reason'] = 'buy conditions met'
            buy_order['hard_stop'] = stp
            open_trades[pair] = [buy_order]
            stop_order = funcs.set_stop(pair, stp, live)
            in_pos = True
            uf.record_open_trades(strat.name, market_data, open_trades)
            sizing[asset] = funcs.update_pos(asset, total_bal, inval_dist, pos_fr_dol)
            sizing['USDT'] = funcs.update_usdt(total_bal)
            counts_dict['open_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with buy order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during {pair} buy order')
    else:
        try:
            buy_order = funcs.buy_asset(pair, usdt_size, live)
            buy_order['type'] = 'open_long'
            buy_order['reason'] = 'buy conditions met'
            buy_order['hard_stop'] = stp
            open_trades[pair] = [buy_order]
            stop_order = funcs.set_stop(pair, stp, live)
            in_pos = True
            uf.record_open_trades(strat.name, 'test_records', open_trades)
            pf = usdt_size / total_bal
            or_dol = total_bal * fixed_risk
            sizing[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
            counts_dict['open_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sim buy order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during sim {pair} buy order')
    
    return sizing, counts_dict, open_trades, in_pos

def spot_strat_tp(strat, pair, price, stp, sizing, total_bal, inval_dist, pos_fr_dol, trade_record, open_trades, market_data, counts_dict, live):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"{pair} take-profit @ {price}"
    print(now, note)
    if live:
        try:
            funcs.clear_stop(pair, live)
            tp_order = funcs.sell_asset(pair, live, 50)
            tp_order['type'] = 'tp_long'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            sizing[asset] = funcs.update_pos(asset, total_bal, inval_dist, pos_fr_dol)
            sizing['USDT'] = funcs.update_usdt(total_bal)
            counts_dict['tp_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during {pair} tp order')
    else:
        try:
            funcs.clear_stop(pair, live)
            tp_order = funcs.sell_asset(pair, live, 50)
            tp_order['type'] = 'tp_long'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, 'test_records', open_trades)
            qty = sizing.get(asset).get('qty') / 2
            val = sizing.get(asset).get('value') / 2
            pf = sizing.get(asset).get('pf%') / 2
            or_R = sizing.get(asset).get('or_R') / 2
            or_dol = sizing.get(asset).get('or_$') / 2
            sizing[asset] = {'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol}
            counts_dict['tp_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with tp order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during sim {pair} tp order')
            
    return counts_dict, trade_record

def spot_sell(strat, pair, price, next_id, sizing, counts_dict, trade_record, open_trades, closed_trades, total_bal, market_data, live):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"{pair} hit trailing stop @ {price}"
    print(now, note)
    if live:
        try:
            funcs.clear_stop(pair, live)
            sell_order = funcs.sell_asset(pair, live)
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
            del sizing[asset]
            sizing['USDT'] = funcs.update_usdt(total_bal)
            counts_dict['close_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during {pair} sell order')
    else:
        try:
            funcs.clear_stop(pair, live)
            sell_order = funcs.sell_asset(pair, live)
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
            sizing['USDT'] += sizing.get('asset').get('value')
            del sizing[asset]
            counts_dict['close_count'] += 1
        except BinanceAPIException as e:
            print(f'problem with sell order for {pair}')
            print(e)
            push = pb.push_note(now, f'exeption during sim {pair} sell order')
            
    return sizing, counts_dict, open_trades, closed_trades, in_pos

def spot_risk_limit_tp(strat, pair, tp_pct, price, price_delta, sizing, trade_record, open_trades, closed_trades, 
                       next_id, market_data, counts_dict, stp, total_bal, inval_dist, pos_fr_dol, in_pos, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    if live:
        note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        print(now, note)
        funcs.clear_stop(pair, live)
        tp_order = funcs.sell_asset(pair, live, pct=tp_pct)
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
            del sizing[asset]
            sizing['USDT'] = funcs.update_usdt(total_bal)
            counts_dict['close_count'] += 1
        else:
            tp_order['type'] = 'tp_long'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'position R limit exceeded'
            trade_record.append(tp_order)
            open_trades[pair] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            sizing[asset] = funcs.update_pos(asset, total_bal, inval_dist, pos_fr_dol)
            sizing['USDT'] = funcs.update_usdt(total_bal)
            counts_dict['tp_count'] += 1
        
    else:
        note = f"sim {pair} take profit"
        print(now, note)
        funcs.clear_stop(pair, live)
        tp_order = funcs.sell_asset(pair, live, pct=tp_pct)
        tp_order['type'] = 'tp_long'
        stop_order = funcs.set_stop(pair, stp, live)
        tp_order['hard_stop'] = stp
        tp_order['reason'] = 'position R limit exceeded'
        trade_record.append(tp_order)
        open_trades[pair] = trade_record
        uf.record_open_trades(strat.name, 'test_records', open_trades)
        qty = sizing.get(asset).get('qty') / 2
        val = sizing.get(asset).get('value') / 2
        pf = sizing.get(asset).get('pf%') / 2
        or_R = sizing.get(asset).get('or_R') / 2
        or_dol = sizing.get(asset).get('or_$') / 2
        sizing[asset] = {'qty': qty, 'value': val, 'pf%': pf, 'or_R': or_R, 'or_$': or_dol}
        counts_dict['tp_count'] += 1
    
    return sizing, counts_dict, open_trades, closed_trades, in_pos



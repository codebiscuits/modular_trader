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
from decimal import Decimal
import math

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)


def sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, reason, in_pos):
    # note = f'*sim* buy {pair} because {reason}'
    # print(note)
    asset = pair[:-4]
    timestamp = round(datetime.utcnow().timestamp() * 1000)
    in_pos['sim'] = True
    in_pos['sim_pfrd'] = strat.fixed_risk_dol
    strat.sim_pos[asset] = funcs.update_pos(strat, asset, size, inval_dist, in_pos['sim_pfrd'])
    strat.sim_pos[asset]['pnl_R'] = 0
    buy_order = {'pair': pair, 
                 'exe_price': str(price), 
                 'trig_price': str(price), 
                 'base_size': str(size), 
                 'quote_size': str(round(usdt_size, 2)), 
                 'hard_stop': str(stp), 
                 'reason': reason, 
                 'timestamp': timestamp, 
                 'type': 'open_long', 
                 'fee': '0', 
                 'fee_currency': 'BNB', 
                 'state': 'sim'}
    trade_record = [buy_order]
    strat.sim_trades[pair] = trade_record
    
    uf.record_sim_trades(strat)
    strat.counts_dict['sim_open'] += 1
    
    return in_pos

def spot_buy(strat, pair, in_pos, size, usdt_size, price, stp, inval_dist):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    note = f"buy {float(size):.5} {pair} ({float(usdt_size):.5} usdt) @ {price}, stop @ {float(stp):.5}"
    print(now, note)
    
    api_order = funcs.buy_asset(pair, usdt_size, strat.live)
    buy_order = funcs.create_trade_dict(api_order, price, strat.live)
    buy_order['type'] = 'open_long'
    buy_order['state'] = 'real'
    buy_order['reason'] = 'buy conditions met'
    buy_order['hard_stop'] = str(stp)
    
    stop_size = Decimal(buy_order['base_size'])
    stop_order = funcs.set_stop(pair, stop_size, stp, strat.live)
    buy_order['stop_id'] = stop_order.get('orderId')
    
    strat.open_trades[pair] = [buy_order]
    uf.record_open_trades(strat)
    
    in_pos['real'] = True
    
    if strat.live:
        strat.real_pos[asset] = funcs.update_pos(strat, asset, size, inval_dist, in_pos['real_pfrd'])
        strat.real_pos[asset]['pnl_R'] = 0
        strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
    else:
        pf = usdt_size / strat.bal
        or_dol = strat.bal * strat.fixed_risk
        strat.real_pos[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
    
    strat.counts_dict['real_open'] += 1
    
    
    return in_pos

def spot_tp(strat, pair, in_pos, price, price_delta, stp, inval_dist):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    if in_pos['real']:
        trade_record = strat.open_trades.get(pair)
        real_bal = Decimal(strat.real_pos[asset]['qty'])
        real_val = Decimal(strat.real_pos[asset]['value'])
        funcs.clear_stop(pair, trade_record, strat.live)
        tp_pct = 50 if real_val > 24 else 100
        note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        print(now, note)
        api_order = funcs.sell_asset(pair, real_bal, strat.live, tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, strat.live)
        
        if tp_pct == 100:
            # coming from open_trades, moving to tracked_trades
            tp_order['type'] = 'close_long'                
            tp_order['state'] = 'real'
            tp_order['reason'] = 'trade over-extended'
            trade_record.append(tp_order)
            
            strat.tracked_trades[pair] = trade_record
            uf.record_tracked_trades(strat)
            
            if strat.open_trades[pair]:
                del strat.open_trades[pair]
                uf.record_open_trades(strat)
            
            in_pos['real'] = False
            in_pos['tracked'] = True
            
            strat.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            del strat.real_pos[asset]
            if strat.live:
                strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
            else:
                strat.real_pos['USDT']['qty'] += strat.real_pos[asset].get('value')
                strat.real_pos['USDT']['value'] += strat.real_pos[asset].get('value')
                strat.real_pos['USDT']['pf%'] += strat.real_pos[asset].get('pf%')
            
            strat.counts_dict['real_close'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
        else: # if tp_pct < 100
            # coming from open_trades, staying in open_trades
            tp_order['type'] = 'tp_long'
            tp_order['state'] = 'real'
            tp_order['reason'] = 'trade over-extended'
            
            new_size = real_bal - Decimal(tp_order['base_size'])
            stop_order = funcs.set_stop(pair, new_size, stp, strat.live)
            tp_order['hard_stop'] = str(stp)
            
            trade_record.append(tp_order)
            
            strat.open_trades['pair'] = trade_record
            uf.record_open_trades(strat)
           
            if strat.live:
                strat.real_pos[asset].update(funcs.update_pos(strat, asset, new_size, inval_dist, in_pos['real_pfrd']))
                strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
            else:
                uf.calc_sizing_non_live_tp(strat, asset, tp_pct, 'real')
            
            strat.counts_dict['real_tp'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
            
    if in_pos['sim']:
        trade_record = strat.sim_trades.get(pair)
        sim_bal = Decimal(strat.sim_pos[asset]['qty'])
        api_order = funcs.sell_asset(pair, sim_bal, False, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, False)
        tp_pct = 50 if sim_bal > 24 else 100
        # note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        # print(note)
        if tp_pct == 100:
            # coming from sim_trades, staying in sim_trades
            tp_order['type'] = 'close_long'
            tp_order['state'] = 'sim'
            tp_order['reason'] = 'trade over extended'
            
            trade_record.append(tp_order)
            
            strat.sim_trades[pair] = trade_record
            uf.record_sim_trades(strat)
            
            strat.sim_pos[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}
            strat.sim_pos['USDT']['qty'] += strat.real_pos[asset].get('value')
            strat.sim_pos['USDT']['value'] += strat.real_pos[asset].get('value')
            strat.sim_pos['USDT']['pf%'] += strat.real_pos[asset].get('pf%')
            
            strat.counts_dict['sim_close'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
        else: # if tp_pct < 100
            # coming from sim_trades, staying in sim_trades
            tp_order['type'] = 'tp_long'
            tp_order['state'] = 'sim'
            tp_order['hard_stop'] = str(stp)
            tp_order['reason'] = 'trade over extended'
            
            trade_record.append(tp_order)
            
            strat.sim_trades[pair] = trade_record
            uf.record_sim_trades(strat)
            
            # update sim_pos
            uf.calc_sizing_non_live_tp(strat, asset, tp_pct, 'sim')
            
            strat.counts_dict['sim_tp'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
            
    if in_pos['tracked']:
        trade_record = strat.tracked_trades.get(pair)
        api_order = funcs.sell_asset(pair, 0, False, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, False)
        tp_order['type'] = 'tp_long'
        tp_order['state'] = 'tracked'
        tp_order['hard_stop'] = str(stp)
        tp_order['reason'] = 'trade over extended'
        
        trade_record.append(tp_order)
        
        strat.tracked_trades[pair] = trade_record
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['sim_tp'] += 1
    
    return in_pos

def spot_sell(strat, pair, in_pos, price):
    asset = pair[:-4]
    
    if in_pos['real']:
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        note = f"{pair} closed on signal @ {price}"
        print(now, note)
        trade_record = strat.open_trades.get(pair)
        real_bal = Decimal(strat.real_pos[asset]['qty'])
        
        funcs.clear_stop(pair, trade_record, strat.live)
        api_order = funcs.sell_asset(pair, real_bal, strat.live)
        
        sell_order = funcs.create_trade_dict(api_order, price, strat.live)
        sell_order['type'] = 'close_long'
        sell_order['state'] = 'real'
        sell_order['reason'] = 'strategy close signal'
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
        uf.record_closed_trades(strat)
        strat.next_id += 1
        
        if strat.open_trades[pair]:
            del strat.open_trades[pair]
            uf.record_open_trades(strat)
        
        in_pos['real'] = False
        
        if strat.live:
            del strat.real_pos[asset]
            strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
        else:
            strat.real_pos['USDT']['value'] += strat.real_pos.get(asset).get('value')
            del strat.real_pos[asset]
        
        strat.counts_dict['real_close'] += 1
        uf.realised_pnl(strat, trade_record, 'long')

    if in_pos['sim']:
        trade_record = strat.sim_trades.get(pair)
        # note = f"*sim* {pair} closed on signal @ {price}"
        # print(note)
        sim_bal = Decimal(strat.sim_pos[asset]['qty'])
        api_order = funcs.sell_asset(pair, sim_bal, False)
        sell_order = funcs.create_trade_dict(api_order, price, False)
        sell_order['type'] = 'close_long'
        sell_order['state'] = 'sim'
        sell_order['reason'] = 'strategy close signal'
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_sim_trades[trade_id] = trade_record
        else:
            strat.closed_sim_trades[strat.next_id] = trade_record
            strat.next_id += 1
        uf.record_closed_sim_trades(strat)
        
        if strat.sim_trades[pair]:
            del strat.sim_trades[pair]
            uf.record_sim_trades(strat)
        
        in_pos['sim'] = False
        in_pos['sim_pfrd'] = 0
        
        del strat.sim_pos[asset]
        
        strat.counts_dict['sim_close'] += 1
        uf.realised_pnl(strat, trade_record, 'long')
    
    if in_pos['tracked']:
        trade_record = strat.tracked_trades.get(pair)
        api_order = funcs.sell_asset(pair, 0, False)
        sell_order = funcs.create_trade_dict(api_order, price, False)
        sell_order['type'] = 'close_long'
        sell_order['state'] = 'tracked'
        sell_order['reason'] = 'strategy close signal'
        
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
        uf.record_closed_trades(strat)
        strat.next_id += 1
        
        del strat.tracked[asset]
        
        del strat.tracked_trades[pair]
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['sim_close'] += 1
        
        in_pos['tracked'] = False
            
    return in_pos

def reduce_risk(strat):
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R'), r.get('pnl_%')) 
                 for p, r in strat.real_pos.items() 
                 if r.get('or_R') and (r.get('or_R') > 0)]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in strat.real_pos.values() if x.get('or_R')]
        total_r = sum(r_list)
        counted = len(r_list)        
        
        for pos in sorted_pos:
            if total_r > strat.total_r_limit and pos[1] > 1.1 and pos[2] > 0.3:
                print(f'*** tor: {total_r:.1f}, reducing risk ***')
                pair = pos[0] + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"reduce risk {pair}, or: {pos[1]}R, pnl: {pos[2]}%"
                print(now, note)
                try:
                    # push = pb.push_note(now, note)
                    if strat.open_trades.get(pair):
                        trade_record = strat.open_trades.get(pair)
                    else:
                        trade_record = []
                    
                    funcs.clear_stop(pair, trade_record, strat.live)
                    api_order = funcs.sell_asset(pair, strat.live)
                    
                    sell_order = funcs.create_trade_dict(api_order, price, strat.live)
                    sell_order['type'] = 'close_long'
                    sell_order['state'] = 'real'
                    sell_order['reason'] = 'portfolio risk limiting'
                    
                    trade_record.append(sell_order)
                    
                    strat.sim_trades[pair] = trade_record
                    uf.record_sim_trades(strat)
                    
                    if strat.open_trades[pair]:
                        del strat.open_trades[pair]
                        uf.record_open_trades(strat)
                    
                    strat.counts_dict['real_close'] += 1
                    total_r -= pos[1]
                    
                    if not strat.live:
                        strat.real_pos['USDT']['qty'] += strat.real_pos[pos[0]].get('value')
                        strat.real_pos['USDT']['value'] += strat.real_pos[pos[0]].get('value')
                        strat.real_pos['USDT']['pf%'] += strat.real_pos[pos[0]].get('pf%')
                    
                    del strat.real_pos[pos[0]]
                    uf.realised_pnl(strat, trade_record, 'long')
                except BinanceAPIException as e:
                    print(f'problem with sell order for {pair}')
                    print(e)
                    pb.push_note(now, f'exeption during {pair} sell order')
                    continue



def open_long(strat, in_pos, pair, size, stp, inval, sim_reason):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    strat.bal = funcs.account_bal_M()
    now = datetime.now().strftime('%d/%m/%y %H:%M')
        
    if in_pos['real'] == None and not sim_reason: # if state = real
        note = f"open real long {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow usdt
        funcs.borrow_asset_M('USDT', usdt_size, strat.live)
        
        # execute
        api_order = funcs.buy_asset_M(pair, usdt_size, False, strat.live)
        
        # create trade record
        long_order = funcs.create_trade_dict(api_order, price, strat.live)
        long_order['type'] = 'open_long'
        long_order['state'] = 'real'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = str(stp)
        long_order['liability'] = uf.update_liability(None, usdt_size, 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(pair, stop_size, be.SIDE_SELL, stp, stp*0.8)
        long_order['stop_id'] = stop_order.get('orderId')
        
        strat.open_trades[pair] = [long_order]
        uf.record_open_trades(strat)
        
        # update positions dictionaries
        in_pos['real'] = 'long'
        in_pos['real_pfrd'] = strat.fixed_risk_dol_l
        in_pos['real_ep'] = price
        in_pos['real_hs'] = stp
        if strat.live:
            strat.real_pos[asset] = funcs.update_pos_M(strat, asset, size, inval, in_pos['real'], in_pos['real_pfrd'])
            strat.real_pos[asset]['pnl_R'] = 0
            strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
        else:
            pf = usdt_size / strat.bal
            or_dol = strat.bal * strat.fixed_risk_l
            strat.real_pos[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
        
        # save records and update counts
        strat.counts_dict['real_open_long'] +=1
        
    if in_pos['sim'] == None and sim_reason:
        note = f"open sim long {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        long_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(size), 
                     'quote_size': str(round(usdt_size, 2)), 
                     'hard_stop': str(stp), 
                     'reason': sim_reason, 
                     'timestamp': timestamp, 
                     'type': 'open_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record = [long_order]
        strat.sim_trades[pair] = trade_record
        uf.record_sim_trades(strat)
        
        in_pos['sim'] = 'long'
        in_pos['sim_pfrd'] = strat.fixed_risk_dol_l
        strat.sim_pos[asset] = funcs.update_pos_M(strat, asset, size, inval, in_pos['sim'], in_pos['sim_pfrd'])
        strat.sim_pos[asset]['pnl_R'] = 0
        
        strat.counts_dict['sim_open_long'] += 1
        
    return in_pos

def tp_long(strat, in_pos, pair, stp, inval):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    strat.bal = funcs.account_bal_M()
    
    if in_pos['real'] == 'long':        
        trade_record = strat.open_trades.get(pair)
        real_bal = abs(float(strat.real_pos[asset]['qty']))
        real_val = abs(float(strat.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, strat.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        order_size = base_size * (pct/100)
        api_order = funcs.sell_asset_M(pair, order_size, strat.live)
        sell_order = funcs.create_trade_dict(api_order, price, strat.live)
        usdt_size = api_order.get('cummulativeQuoteQty')
        funcs.repay_asset_M('USDT', usdt_size, strat.live)
        
        note = f"real take-profit {pair} long {pct}% @ {price}"
        print(now, note)        
        
        if pct == 100:
            # create trade dict
            sell_order['type'] = 'close_long'
            sell_order['state'] = 'real'
            sell_order['reason'] = 'trade over-extended'
            sell_order['liability'] = uf.update_liability(trade_record, usdt_size, 'reduce')
            trade_record.append(sell_order)
            
            # update records            
            strat.tracked_trades[pair] = trade_record
            uf.record_tracked_trades(strat)
            
            if strat.open_trades[pair]:
                del strat.open_trades[pair]
                uf.record_open_trades(strat)
            
            in_pos['real'] = None
            in_pos['tracked'] = 'long'
            
            strat.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            del strat.real_pos[asset]
            if strat.live:
                strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
            else:
                qty = float(strat.real_pos['USDT']['qty'])
                strat.real_pos['USDT']['qty'] = qty + float(strat.real_pos[asset].get('value'))
                value = float(strat.real_pos['USDT']['value'])
                strat.real_pos['USDT']['value'] = value + float(strat.real_pos[asset].get('value'))
                pf_pct = float(strat.real_pos['USDT']['pf%'])
                strat.real_pos['USDT']['pf%'] = pf_pct + float(strat.real_pos[asset].get('pf%'))
            
            strat.counts_dict['real_close_long'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
        
        
        else: # if pct < 100%
            # create trade dict
            sell_order['type'] = 'tp_long'
            sell_order['state'] = 'real'
            sell_order['hard_stop'] = str(stp)
            sell_order['reason'] = 'trade over-extended'
            sell_order['liability'] = uf.update_liability(trade_record, usdt_size, 'reduce')
            
            # set new stop
            new_size = real_bal - float(sell_order['base_size'])
            stop_order = funcs.set_stop_M(pair, new_size, be.SIDE_SELL, stp, stp*0.8)
            sell_order['stop_id'] = stop_order.get('orderId')
            
            trade_record.append(sell_order)
            
            # update records
            strat.open_trades['pair'] = trade_record
            uf.record_open_trades(strat)
           
            in_pos['real_pfrd'] = in_pos['real_pfrd'] * (pct / 100)
            if strat.live:
                strat.real_pos[asset].update(funcs.update_pos_M(strat, asset, new_size, inval, in_pos['real'], in_pos['real_pfrd']))
                strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
            else:
                uf.calc_sizing_non_live_tp(strat, asset, pct, 'real')
            
            strat.counts_dict['real_tp_long'] += 1
            uf.realised_pnl(strat, trade_record, 'long')
        
    if in_pos['sim'] == 'long':
        note = f"sim take-profit {pair} long 50% @ {price}"
        print(now, note)
        
        trade_record = strat.open_trades.get(pair)
        sim_bal = abs(float(strat.sim_pos[asset]['qty']))
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        order_size = sim_bal / 2
        quote_order_size = round(order_size * price, 2)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(order_size), 
                     'quote_size': str(quote_order_size), 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record.append(tp_order)
        
        # update records
        strat.sim_trades[pair] = trade_record
        uf.record_sim_trades(strat)
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(strat, in_pos, asset, 50, 'sim')
    
        strat.counts_dict['sim_tp_long'] += 1
        in_pos['sim_pfrd'] = in_pos['sim_pfrd'] / 2
        uf.realised_pnl(strat, trade_record, 'long')
        
    if in_pos['tracked'] == 'long':
        note = f"tracked take-profit {pair} long 50% @ {price}"
        print(now, note)
        
        trade_record = strat.tracked_trades.get(pair)
        tracked_bal = abs(float(strat.tracked[asset]['qty']))
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        order_size = tracked_bal / 2
        quote_order_size = round(order_size * price, 2)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(order_size), 
                     'quote_size': str(quote_order_size), 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(tp_order)
        
        # update records
        strat.tracked_trades[pair] = trade_record
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['tracked_tp_long'] += 1
        in_pos['tracked_pfrd'] = in_pos['tracked_pfrd'] / 2
        
    
    return in_pos

def close_long(strat, in_pos, pair):
    # initialise stuff
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    strat.bal = funcs.account_bal_M()
    
    if in_pos['real'] == 'long':
        note = f"real close long {pair} @ {price}"
        print(now, note)
        
        trade_record = strat.open_trades.get(pair)
        real_bal = abs(float(strat.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, strat.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        api_order = funcs.sell_asset_M(pair, base_size, strat.live)
        usdt_size = api_order.get('cummulativeQuoteQty')
        funcs.repay_asset_M('USDT', usdt_size, strat.live)
        
        sell_order = funcs.create_trade_dict(api_order, price, strat.live)
        sell_order['type'] = 'close_long'
        sell_order['state'] = 'real'
        sell_order['reason'] = 'strategy close long signal'
        sell_order['liability'] = uf.update_liability(trade_record, usdt_size, 'reduce')
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
        uf.record_closed_trades(strat)
        strat.next_id += 1
        
        if strat.open_trades[pair]:
            del strat.open_trades[pair]
            uf.record_open_trades(strat)
        
        in_pos['real'] = None
        in_pos['real_pfrd'] = 0
        del strat.real_pos[asset]
        if strat.live:
            strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
        else:
            value = float(strat.real_pos['USDT']['value'])
            strat.real_pos['USDT']['value'] = value + float(usdt_size)
            owed = float(strat.real_pos['USDT']['owed'])
            strat.real_pos['USDT']['owed'] = owed - float(usdt_size)
        
        # save records and update counts
        strat.counts_dict['real_close_long'] +=1
        uf.realised_pnl(strat, trade_record, 'long')
    
    if in_pos['sim'] == 'long':
        note = f"sim close long {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = strat.sim_trades[pair]
        sim_bal = float(strat.sim_pos[asset]['qty'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        long_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(sim_bal), 
                     'quote_size': str(round(sim_bal*price, 2)), 
                     'reason': 'strategy close long signal', 
                     'timestamp': timestamp, 
                     'type': 'close_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record.append(long_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_sim_trades[trade_id] = trade_record
        else:
            strat.closed_sim_trades[strat.next_id] = trade_record
            strat.next_id += 1
        uf.record_closed_sim_trades(strat)
        
        if strat.sim_trades[pair]:
            del strat.sim_trades[pair]
            uf.record_sim_trades(strat)
        
        # update counts and live variables
        in_pos['sim'] = None
        in_pos['sim_pfrd'] = 0        
        del strat.sim_pos[asset]
        
        strat.counts_dict['sim_close_long'] += 1
        uf.realised_pnl(strat, trade_record, 'long')
        
    if in_pos['tracked'] == 'long':
        note = f"tracked close long {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = strat.tracked_trades[pair]
        tracked_bal = Decimal(strat.tracked[asset]['qty'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        long_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(tracked_bal), 
                     'quote_size': str(round(sim_bal*price, 2)), 
                     'reason': 'strategy close long signal', 
                     'timestamp': timestamp, 
                     'type': 'close_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(long_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
            strat.next_id += 1
        uf.record_closed_trades(strat)
        
        if strat.tracked_trades[pair]:
            del strat.tracked_trades[pair]
            uf.record_tracked_trades(strat)
        
        # update counts and live variables
        del strat.tracked[asset]
        
        in_pos['tracked'] = None
        in_pos['tracked_pfrd'] = 0        
        
        strat.counts_dict['tracked_close_long'] += 1
    
    return in_pos

def open_short(strat, in_pos, pair, size, stp, inval, sim_reason):
    # initialise stuff
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    strat.bal = funcs.account_bal_M()
    
    if in_pos['real'] == None and not sim_reason:
        note = f"real open short {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow
        funcs.borrow_asset_M(asset, size, strat.live)
        
        # execute
        api_order = funcs.sell_asset_M(pair, size, strat.live)
        
        # create trade record
        short_order = funcs.create_trade_dict(api_order, price, strat.live)
        short_order['type'] = 'open_short'
        short_order['state'] = 'real'
        short_order['score'] = 'signal score'
        short_order['hard_stop'] = str(stp)
        short_order['liability'] = uf.update_liability(None, size, 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(pair, stop_size, be.SIDE_BUY, stp, stp*1.2, strat.live)
        short_order['stop_id'] = stop_order.get('orderId')
        strat.open_trades[pair] = [short_order]
        uf.record_open_trades(strat)
        
        # update positions dictionaries
        in_pos['real'] = 'short'
        in_pos['real_pfrd'] = strat.fixed_risk_dol_s
        in_pos['real_ep'] = price
        in_pos['real_hs'] = stp
        if strat.live:
            strat.real_pos[asset] = funcs.update_pos_M(strat, asset, size, inval, in_pos['real'], in_pos['real_pfrd'])
            strat.real_pos[asset]['pnl_R'] = 0
            strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
        else:
            pf = usdt_size / strat.bal
            or_dol = strat.bal * strat.fixed_risk_s
            strat.real_pos[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
        
        # save records and update counts
        strat.counts_dict['real_open_short'] +=1
        
    if in_pos['sim'] == None and sim_reason:
        note = f"sim open short {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        short_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(size), 
                     'quote_size': str(round(usdt_size, 2)), 
                     'hard_stop': str(stp), 
                     'reason': sim_reason, 
                     'timestamp': timestamp, 
                     'type': 'open_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record = [short_order]
        strat.sim_trades[pair] = trade_record
        uf.record_sim_trades(strat)
        
        in_pos['sim'] = 'short'
        in_pos['sim_pfrd'] = strat.fixed_risk_dol_s
        strat.sim_pos[asset] = funcs.update_pos_M(strat, asset, size, inval, in_pos['sim'], in_pos['sim_pfrd'])
        strat.sim_pos[asset]['pnl_R'] = 0
        
        strat.counts_dict['sim_open_short'] += 1
        
    return in_pos

def tp_short(strat, in_pos, pair, stp, inval):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    strat.bal = funcs.account_bal_M()
    
    if in_pos['real'] == 'short':        
        trade_record = strat.open_trades.get(pair)
        real_bal = abs(float(strat.real_pos[asset]['qty']))
        real_val = abs(float(strat.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, strat.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        order_size = base_size * (pct/100)
        api_order = funcs.buy_asset_M(pair, order_size, True, strat.live)
        buy_order = funcs.create_trade_dict(api_order, price, strat.live)
        repay_size = buy_order.get('base_size')
        funcs.repay_asset_M(asset, repay_size, strat.live)
        
        note = f"real take-profit {pair} short {pct}% @ {price}"
        print(now, note)        
        
        if pct == 100:
            # create trade dict
            buy_order['type'] = 'close_short'
            buy_order['state'] = 'real'
            buy_order['reason'] = 'trade over-extended'
            buy_order['liability'] = uf.update_liability(trade_record, repay_size, 'reduce')
            trade_record.append(buy_order)
            
            # update records            
            strat.tracked_trades[pair] = trade_record
            uf.record_tracked_trades(strat)
            
            if strat.open_trades[pair]:
                del strat.open_trades[pair]
                uf.record_open_trades(strat)
            
            in_pos['real'] = None
            in_pos['tracked'] = 'short'
            del strat.real_pos[asset]
            if strat.live:
                strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
            else:
                strat.real_pos['USDT']['qty'] -= strat.real_pos[asset].get('value')
                strat.real_pos['USDT']['value'] -= strat.real_pos[asset].get('value')
                strat.real_pos['USDT']['pf%'] -= strat.real_pos[asset].get('pf%')
            strat.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            
            strat.counts_dict['real_close_short'] += 1
            uf.realised_pnl(strat, trade_record, 'short')
        
        else: # if pct < 100%
            # create trade dict
            buy_order['type'] = 'tp_short'
            buy_order['state'] = 'real'
            buy_order['hard_stop'] = str(stp)
            buy_order['reason'] = 'trade over-extended'
            buy_order['liability'] = uf.update_liability(trade_record, repay_size, 'reduce')
            
            # set new stop
            new_size = real_bal - float(buy_order['base_size'])
            stop_order = funcs.set_stop_M(pair, new_size, be.SIDE_SELL, stp, stp*1.2)
            buy_order['stop_id'] = stop_order.get('orderId')
            
            trade_record.append(buy_order)
            
            # update records
            strat.open_trades['pair'] = trade_record
            uf.record_open_trades(strat)
           
            in_pos['real_pfrd'] = in_pos['real_pfrd'] * (pct / 100)
            if strat.live:
                strat.real_pos[asset].update(funcs.update_pos_M(strat, asset, new_size, inval, in_pos['real'], in_pos['real_pfrd']))
                strat.real_pos['USDT'] = funcs.update_usdt(strat.bal)
            else:
                uf.calc_sizing_non_live_tp(strat, asset, pct, 'real')
            
            strat.counts_dict['real_tp_short'] += 1
            uf.realised_pnl(strat, trade_record, 'short')
        
    if in_pos['sim'] == 'short':
        note = f"sim take-profit {pair} short 50% @ {price}"
        print(now, note) 
        
        trade_record = strat.open_trades.get(pair)
        sim_bal = abs(float(strat.sim_pos[asset]['qty']))
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        order_size = sim_bal / 2
        quote_order_size = round(order_size * price, 2)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(order_size), 
                     'quote_size': str(quote_order_size), 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record.append(tp_order)
        
        # update records
        in_pos['sim_pfrd'] = in_pos['sim_pfrd'] * (pct / 100)
        strat.sim_trades[pair] = trade_record
        uf.record_sim_trades(strat)
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(strat, in_pos, asset, 50, 'sim')
    
        strat.counts_dict['sim_tp_short'] += 1
        
    if in_pos['tracked'] == 'short':
        note = f"tracked take-profit {pair} short 50% @ {price}"
        print(now, note) 
        
        trade_record = strat.tracked_trades.get(pair)
        tracked_bal = abs(float(strat.tracked[asset]['qty']))
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        order_size = tracked_bal / 2
        quote_order_size = round(order_size * price, 2)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(order_size), 
                     'quote_size': str(quote_order_size), 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(tp_order)
        
        # update records
        strat.tracked_trades[pair] = trade_record
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['tracked_tp_short'] += 1
        in_pos['tracked_pfrd'] = in_pos['tracked_pfrd'] / 2

def close_short(strat, in_pos, pair):
    # initialise stuff
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    strat.bal = funcs.account_bal_M()
    
    if in_pos['real'] == 'short':
        note = f"real close short {pair} @ {price}"
        print(now, note)
        
        trade_record = strat.open_trades.get(pair)
        real_bal = abs(float(strat.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, strat.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        api_order = funcs.buy_asset_M(pair, base_size, True, strat.live)
        funcs.repay_asset_M(asset, base_size, strat.live)
        
        sell_order = funcs.create_trade_dict(api_order, price, strat.live)
        sell_order['type'] = 'close_short'
        sell_order['state'] = 'real'
        sell_order['reason'] = 'strategy close short signal'
        sell_order['liability'] = uf.update_liability(trade_record, base_size, 'reduce')
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
        uf.record_closed_trades(strat)
        strat.next_id += 1
        
        if strat.open_trades[pair]:
            del strat.open_trades[pair]
            uf.record_open_trades(strat)
        
        in_pos['real'] = None
        in_pos['real_pfrd'] = 0
        if strat.live:
            del strat.real_pos[asset]
            strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
        else:
            del strat.real_pos[asset]
            value = float(strat.real_pos['USDT']['value'])
            strat.real_pos['USDT']['value'] = value + float(base_size * price)
            owed = float(strat.real_pos['USDT']['owed'])
            strat.real_pos['USDT']['owed'] = owed - float(base_size * price)
        
        # save records and update counts
        strat.counts_dict['real_close_long'] +=1
        uf.realised_pnl(strat, trade_record, 'short')
    
    if in_pos['sim'] == 'short':
        note = f"sim close short {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = strat.sim_trades[pair]
        sim_bal = float(strat.sim_pos[asset]['qty'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        short_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(sim_bal), 
                     'quote_size': str(round(sim_bal*price, 2)), 
                     'reason': 'strategy close short signal', 
                     'timestamp': timestamp, 
                     'type': 'close_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record.append(short_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_sim_trades[trade_id] = trade_record
        else:
            strat.closed_sim_trades[strat.next_id] = trade_record
            strat.next_id += 1
        uf.record_closed_sim_trades(strat)
        
        if strat.sim_trades[pair]:
            del strat.sim_trades[pair]
            uf.record_sim_trades(strat)
        
        # update live variables
        del strat.sim_pos[asset]
        in_pos['sim'] = None
        in_pos['sim_pfrd'] = 0        
        
        strat.counts_dict['sim_close_long'] += 1
        uf.realised_pnl(strat, trade_record, 'short')
        
    if in_pos['tracked'] == 'short':
        note = f"tracked close short {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = strat.tracked_trades[pair]
        tracked_bal = Decimal(strat.tracked[asset]['qty'])
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        short_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(tracked_bal), 
                     'quote_size': str(round(tracked_bal*price, 2)), 
                     'reason': 'strategy close short signal', 
                     'timestamp': timestamp, 
                     'type': 'close_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(short_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            strat.closed_trades[trade_id] = trade_record
        else:
            strat.closed_trades[strat.next_id] = trade_record
            strat.next_id += 1
        uf.record_closed_trades(strat)
        
        if strat.tracked_trades[pair]:
            del strat.tracked_trades[pair]
            uf.record_tracked_trades(strat)
        
        # update live variables
        del strat.tracked[asset]
        in_pos['tracked'] = None
        in_pos['tracked_pfrd'] = 0        
        
        strat.counts_dict['tracked_close_long'] += 1
    
    return in_pos

def reduce_risk_M(strat):
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R'), r.get('pnl_%')) 
                 for p, r in strat.real_pos.items() 
                 if r.get('or_R') and (r.get('or_R') > 0)]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in strat.real_pos.values() if x.get('or_R')]
        total_r = sum(r_list)
        counted = len(r_list)        
        
        for pos in sorted_pos:
            asset = pos[0]
            or_R = pos[1]
            pnl_pct = pos[2]
            if total_r > strat.total_r_limit and or_R > 1.1 and pnl_pct > 0.3:
                print(f'*** tor: {total_r:.1f}, reducing risk ***')
                pair = asset + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"reduce risk {pair}, or: {or_R}R, pnl: {pnl_pct}%"
                print(now, note)
                try:
                    # push = pb.push_note(now, note)
                    if strat.open_trades.get(pair):
                        trade_record = strat.open_trades.get(pair)
                    else:
                        trade_record = []
                    
                    real_bal = abs(float(strat.real_pos[asset]['qty']))
                    
                    # clear stop
                    clear, base_size = funcs.clear_stop_M(pair, trade_record, strat.live)
                    if base_size and (real_bal != base_size): # check records match reality
                        print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
                    if not base_size:
                        base_size = real_bal
                    
                    long = trade_record[0].get('type')[-4:] == 'long'
                    if long:
                        api_order = funcs.sell_asset_M(pair, base_size, strat.live)
                        usdt_size = api_order.get('cummulativeQuoteQty')
                        repay_size = usdt_size
                        funcs.repay_asset_M('USDT', repay_size, strat.live)
                    else:
                        api_order = funcs.buy_asset_M(pair, base_size, True, strat.live)
                        repay_size = base_size
                        funcs.repay_asset_M(asset, repay_size, strat.live)
                    
                    reduce_order = funcs.create_trade_dict(api_order, price, strat.live)
                    reduce_order['type'] = 'close_long' if long else 'close_short'
                    reduce_order['state'] = 'real'
                    reduce_order['reason'] = 'portfolio risk limiting'
                    reduce_order['liability'] = uf.update_liability(trade_record, repay_size, 'reduce')
                    
                    trade_record.append(reduce_order)
                    
                    strat.sim_trades[pair] = trade_record
                    uf.record_sim_trades(strat)
                    
                    if strat.open_trades[pair]:
                        del strat.open_trades[pair]
                        uf.record_open_trades(strat)
                    
                    strat.counts_dict['reduce_risk'] += 1
                    total_r -= or_R
                    
                    if strat.live:
                        del strat.real_pos[asset]
                        strat.real_pos['USDT'] = funcs.update_usdt_M(strat.bal)
                    elif long and not strat.live:
                        del strat.real_pos[asset]
                        strat.real_pos['USDT']['value'] += float(usdt_size)
                        strat.real_pos['USDT']['owed'] -= float(usdt_size)
                    else:
                        del strat.real_pos[asset]
                        strat.real_pos['USDT']['value'] += float(base_size * price)
                        strat.real_pos['USDT']['owed'] -= float(base_size * price)
                    
                    del strat.real_pos[asset]
                    uf.realised_pnl(strat, trade_record, 'long')
                except BinanceAPIException as e:
                    print(f'problem with sell order for {pair}')
                    print(e)
                    pb.push_note(now, f'exeption during {pair} sell order')
                    continue





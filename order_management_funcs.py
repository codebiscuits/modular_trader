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

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)


def sim_spot_buy(strat, pair, size, usdt_size, price, stp, inval_dist, reason, in_pos):
    print(f'sim buy {pair} because {reason}')
    asset = pair[:-4]
    timestamp = round(datetime.utcnow().timestamp() * 1000)
    in_pos['sim'] = True
    in_pos['sim_pfrd'] = strat.fixed_risk_dol
    strat.sim_pos[asset] = funcs.update_pos(asset, size, strat.bal, inval_dist, in_pos['sim_pfrd'])
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
    strat.sim_trades[asset] = trade_record
    
    uf.record_sim_trades(strat)
    
    return in_pos

def spot_buy(strat, pair, in_pos, fixed_risk, size, usdt_size, price, stp, inval_dist, params, live):
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    note = f"buy {float(size):.5} {pair} ({float(usdt_size):.5} usdt) @ {price}, stop @ {float(stp):.5}"
    print(now, note)
    
    api_order = funcs.buy_asset(pair, usdt_size, live)
    buy_order = funcs.create_trade_dict(api_order, price, live)
    buy_order['type'] = 'open_long'
    buy_order['state'] = 'real'
    buy_order['reason'] = 'buy conditions met'
    buy_order['hard_stop'] = stp
    
    strat.open_trades[pair] = [buy_order]
    uf.record_open_trades(strat)
    
    stop_order = funcs.set_stop(pair, stp, live)
    
    in_pos['real'] = True
    
    if live:
        strat.sizing[asset] = funcs.update_pos(asset, size, strat.bal, inval_dist, in_pos['real_pfrd'])
        strat.sizing[asset]['pnl_R'] = 0
        strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
    else:
        pf = usdt_size / strat.bal
        or_dol = strat.bal * fixed_risk
        strat.sizing[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
    
    strat.counts_dict['real_open'] += 1
    
    
    return in_pos

def spot_tp(strat, pair, in_pos, price, price_delta, stp, inval_dist, live):
    asset = pair[:-4]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    
    if in_pos['real']:
        trade_record = strat.open_trades.get(pair)
        real_bal = Decimal(strat.sizing[asset]['qty'])
        funcs.clear_stop(pair, live)
        tp_pct = 50 if real_bal > 24 else 100
        note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        print(now, note)
        api_order = funcs.sell_asset(pair, real_bal, live, tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, live)
        
        if tp_pct == 100:
            # coming from open_trades, moving to sim_trades
            tp_order['type'] = 'close_long'                
            tp_order['state'] = 'real'
            tp_order['reason'] = 'trade over-extended'
            trade_record.append(tp_order)
            
            strat.sim_trades[pair] = trade_record
            uf.record_sim_trades(strat)
            
            if strat.open_trades[pair]:
                del strat.open_trades[pair]
                uf.record_open_trades(strat)
            
            in_pos['real'] = False
            in_pos['sim'] = True
            
            strat.sim_pos[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            del strat.sizing[asset]
            if live:
                strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            else:
                strat.sizing['USDT']['qty'] += strat.sizing[asset].get('value')
                strat.sizing['USDT']['value'] += strat.sizing[asset].get('value')
                strat.sizing['USDT']['pf%'] += strat.sizing[asset].get('pf%')
            
            strat.counts_dict['real_close'] += 1
            uf.realised_pnl(strat, trade_record)
        else: # if tp_pct < 100
            # coming from open_trades, staying in open_trades
            tp_order['type'] = 'tp_long'
            tp_order['state'] = 'real'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            
            trade_record.append(tp_order)
            
            strat.open_trades['pair'] = trade_record
            uf.record_open_trades(strat)
           
            if live:
                new_size = real_bal - tp_order['base_size']
                strat.sizing[asset].update(funcs.update_pos(asset, new_size, strat.bal, inval_dist, in_pos['real_pfrd']))
                strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            else:
                uf.calc_sizing_non_live_tp(strat, asset, tp_pct, 'real')
            
            strat.counts_dict['real_tp'] += 1
            uf.realised_pnl(strat, trade_record)
            
    if in_pos['sim']:
        trade_record = strat.sim_trades.get(pair)
        sim_bal = Decimal(strat.sim_pos[asset]['qty'])
        funcs.clear_stop(pair, False)
        api_order = funcs.sell_asset(pair, sim_bal, False, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, False)
        tp_pct = 50 if sim_bal > 24 else 100
        note = f"{pair} take profit {tp_pct}% @ {price}, {round(price_delta*100, 2)}% from entry"
        print(now, note)
        if tp_pct == 100:
            # coming from sim_trades, staying in sim_trades
            tp_order['type'] = 'close_long'
            tp_order['state'] = 'sim'
            tp_order['reason'] = 'trade over extended'
            
            trade_record.append(tp_order)
            
            strat.sim_trades[asset] = trade_record
            uf.record_sim_trades(strat)
            
            strat.sim_pos[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}
            strat.sim_pos['USDT']['qty'] += strat.sizing[asset].get('value')
            strat.sim_pos['USDT']['value'] += strat.sizing[asset].get('value')
            strat.sim_pos['USDT']['pf%'] += strat.sizing[asset].get('pf%')
            
            strat.counts_dict['sim_close'] += 1
            uf.realised_pnl(strat, trade_record)
        else: # if tp_pct < 100
            # coming from sim_trades, staying in sim_trades
            tp_order['type'] = 'tp_long'
            tp_order['state'] = 'sim'
            tp_order['hard_stop'] = stp
            tp_order['reason'] = 'trade over extended'
            
            trade_record.append(tp_order)
            
            strat.sim_trades[asset] = trade_record
            uf.record_sim_trades(strat)
            
            uf.calc_sizing_non_live_tp(strat, asset, tp_pct, 'sim')
            
            strat.counts_dict['sim_tp'] += 1
            uf.realised_pnl(strat, trade_record)
            
    if in_pos['tracked']:
        trade_record = strat.tracked_trades.get(pair)
        api_order = funcs.sell_asset(pair, 0, False, pct=tp_pct)
        tp_order = funcs.create_trade_dict(api_order, price, False)
        tp_order['type'] = 'tp_long'
        tp_order['state'] = 'tracked'
        tp_order['hard_stop'] = stp
        tp_order['reason'] = 'trade over extended'
        
        trade_record.append(tp_order)
        
        strat.tracked_trades[pair] = trade_record
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['sim_tp'] += 1
    
    return in_pos

def spot_sell(strat, pair, in_pos, price, live):
    asset = pair[:-4]
    
    if in_pos['real']:
        trade_record = strat.open_trades.get(pair)
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        note = f"{pair} closed on signal @ {price}"
        print(now, note)
        # coming from open_trades, moving to closed_trades
        real_bal = Decimal(strat.sizing[asset]['qty'])
        funcs.clear_stop(pair, live)
        api_order = funcs.sell_asset(pair, real_bal, live)
        sell_order = funcs.create_trade_dict(api_order, price, live)
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
        
        if live:
            del strat.sizing[asset]
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
        else:
            strat.sizing['USDT']['value'] += strat.sizing.get(asset).get('value')
            del strat.sizing[asset]
        
        strat.counts_dict['real_close'] += 1
        uf.realised_pnl(strat, trade_record)

    if in_pos['sim']:
        trade_record = strat.sim_trades.get(asset)
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        note = f"*sim* {pair} closed on signal @ {price}"
        print(now, note)
        sim_bal = Decimal(strat.sim_pos[asset]['qty'])
        funcs.clear_stop(pair, False)
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
        
        if strat.sim_trades[asset]:
            del strat.sim_trades[asset]
            uf.record_sim_trades(strat)
        
        in_pos['sim'] = False
        
        del strat.sim_pos[asset]
        
        strat.counts_dict['sim_close'] += 1
        uf.realised_pnl(strat, trade_record)
    
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
        
        del strat.tracked_pos[asset]
        
        del strat.tracked_trades[pair]
        uf.record_tracked_trades(strat)
        
        strat.counts_dict['sim_close'] += 1
        
        in_pos['tracked'] = False
            
    return in_pos

def reduce_risk(strat, params, live):
    r_limit = params.get('total_r_limit')
    
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R'), r.get('pnl_%')) 
                 for p, r in strat.sizing.items() 
                 if r.get('or_R') and (r.get('or_R') > 0)]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in strat.sizing.values() if x.get('or_R')]
        total_r = sum(r_list)
        counted = len(r_list)        
        
        for pos in sorted_pos:
            if total_r > r_limit and pos[1] > 1.1 and pos[2] > 0.3:
                print(f'*** tor: {total_r:.1f}, reducing risk ***')
                pair = pos[0] + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"reduce risk {pair}, or: {pos[1]}R, pnl: {pos[2]}%"
                print(now, note)
                try:
                    # push = pb.push_note(now, note)
                    funcs.clear_stop(pair, live)
                    api_order = funcs.sell_asset(pair, live)
                    
                    sell_order = funcs.create_trade_dict(api_order, price, live)
                    sell_order['type'] = 'close_long'
                    sell_order['state'] = 'real'
                    sell_order['reason'] = 'portfolio risk limiting'
                    
                    if strat.open_trades.get(pair):
                        trade_record = strat.open_trades.get(pair)
                    else:
                        trade_record = []
                    trade_record.append(sell_order)
                    
                    strat.sim_trades[pair] = trade_record
                    uf.record_sim_trades(strat)
                    
                    if strat.open_trades[pair]:
                        del strat.open_trades[pair]
                        uf.record_open_trades(strat)
                    
                    strat.counts_dict['real_close'] += 1
                    total_r -= pos[1]
                    
                    if not live:
                        strat.sizing['USDT']['qty'] += strat.sizing[pos[0]].get('value')
                        strat.sizing['USDT']['value'] += strat.sizing[pos[0]].get('value')
                        strat.sizing['USDT']['pf%'] += strat.sizing[pos[0]].get('pf%')
                    
                    del strat.sizing[pos[0]]
                    uf.realised_pnl(strat, trade_record)
                except BinanceAPIException as e:
                    print(f'problem with sell order for {pair}')
                    print(e)
                    pb.push_note(now, f'exeption during {pair} sell order')
                    continue



def margin_open_long(strat, pair, size, stp, inval_dist, open_trades, market_data, in_pos, live):
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
        strat.sizing[asset] = funcs.update_pos_M(asset, strat.bal, inval_dist, in_pos['real_pfrd'])
        strat.sizing['USDT'] = funcs.update_usdt_M(strat.bal)
        strat.counts_dict['open_count'] += 1
    else:
        api_order = {'symbol': pair, 'price': price, 'quote_size': usdt_size}
        long_order = funcs.create_trade_dict(api_order, price, live)
        long_order['type'] = 'open_long'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = stp
        open_trades[pair] = [long_order]
        uf.record_open_trades(strat.name, market_data, open_trades)
        
    return in_pos

def margin_tp_long(strat, pair, pct, stp, inval_dist, trade_record, open_trades, closed_trades, market_data, live):
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    note = f"tp long {pct} {pair} @ {price}, new stop @ {stp:.5}"
    print(now, note)
    asset = pair[:-4]
    
    if live:
        api_order = funcs.close_long(pair, pct)
        tp_order = funcs.create_trade_dict(api_order, price, live)
        if pct == 100:
            tp_order['type'] = 'close_long'
            tp_order['reason'] = 'trade over-extended'
            trade_record.append(tp_order)            
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = trade_record[0].get('timestamp')
                closed_trades[trade_id] = trade_record
            else:
                closed_trades[strat.next_id] = trade_record
            uf.record_closed_trades(strat.name, market_data, closed_trades)
            strat.next_id += 1
            if open_trades[pair]:
                del open_trades[pair]
                uf.record_open_trades(strat.name, market_data, open_trades)
            in_pos = False
            del strat.sizing[asset]
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['close_count'] += 1
            
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            strat.sizing[asset].update(funcs.update_pos(asset, strat.bal, inval_dist, in_pos['real_pfrd']))
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['close_count'] += 1
            uf.realised_pnl(strat, trade_record)
        else:
            tp_order['type'] = 'tp_long'
            tp_order['reason'] = 'trade over-extended'
            stop_order = funcs.set_stop(pair, stp, live)
            tp_order['hard_stop'] = stp
            trade_record.append(tp_order)
            open_trades['pair'] = trade_record
            uf.record_open_trades(strat.name, market_data, open_trades)
            strat.sizing[asset].update(funcs.update_pos(asset, strat.bal, inval_dist, in_pos['real_pfrd']))
            strat.sizing['USDT'] = funcs.update_usdt(strat.bal)
            strat.counts_dict['tp_count'] += 1
            uf.realised_pnl(strat, trade_record)
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


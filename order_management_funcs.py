import binance_funcs as funcs
import utility_funcs as uf
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from decimal import Decimal

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)


def open_long(session, agent, pair, size, stp, inval, sim_reason):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    session.bal = funcs.account_bal_M()
    now = datetime.now().strftime('%d/%m/%y %H:%M')
        
    if agent.in_pos['real'] == None and not sim_reason: # if state = real
        note = f"open real long {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow usdt
        funcs.borrow_asset_M('USDT', usdt_size, session.live)
        
        # execute
        api_order = funcs.buy_asset_M(pair, usdt_size, False, session.live)
        
        # create trade record
        long_order = funcs.create_trade_dict(api_order, price, session.live)
        long_order['type'] = 'open_long'
        long_order['state'] = 'real'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = str(stp)
        long_order['liability'] = uf.update_liability(None, usdt_size, 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(pair, stop_size, be.SIDE_SELL, stp, stp*0.8)
        long_order['stop_id'] = stop_order.get('orderId')
        
        agent.open_trades[pair] = [long_order]
        uf.record_trades(session, agent, 'open')
        
        # update positions dictionaries
        agent.in_pos['real'] = 'long'
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_l
        agent.in_pos['real_ep'] = price
        agent.in_pos['real_hs'] = stp
        if session.live:
            agent.real_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd'])
            agent.real_pos[asset]['pnl_R'] = 0
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
        else:
            pf = usdt_size / session.bal
            or_dol = session.bal * agent.fixed_risk_l
            agent.real_pos[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
        
        # save records and update counts
        agent.counts_dict['real_open_long'] +=1
        
    if agent.in_pos['sim'] == None and sim_reason:
        usdt_size = 100.0
        size = round(usdt_size / price, 8)
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
        agent.sim_trades[pair] = trade_record
        uf.record_trades(session, agent, 'sim')
        
        agent.in_pos['sim'] = 'long'
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_l
        agent.sim_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['sim'], agent.in_pos['sim_pfrd'])
        agent.sim_pos[asset]['pnl_R'] = 0
        
        agent.counts_dict['sim_open_long'] += 1

def tp_long(session, agent, pair, stp, inval):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    session.bal = funcs.account_bal_M()
    
    if agent.in_pos.get('real_tp_sig'):        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(float(agent.real_pos[asset]['qty']))
        real_val = abs(float(agent.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        order_size = base_size * (pct/100)
        api_order = funcs.sell_asset_M(pair, order_size, session.live)
        sell_order = funcs.create_trade_dict(api_order, price, session.live)
        usdt_size = api_order.get('cummulativeQuoteQty')
        funcs.repay_asset_M('USDT', usdt_size, session.live)
        
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
            agent.tracked_trades[pair] = trade_record
            uf.record_trades(session, agent, 'tracked')
            
            if agent.open_trades[pair]:
                del agent.open_trades[pair]
                uf.record_trades(session, agent, 'open')
            
            agent.in_pos['real'] = None
            agent.in_pos['tracked'] = 'long'
            
            agent.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            del agent.real_pos[asset]
            if session.live:
                agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
            else:
                qty = float(agent.real_pos['USDT']['qty'])
                agent.real_pos['USDT']['qty'] = qty + float(agent.real_pos[asset].get('value'))
                value = float(agent.real_pos['USDT']['value'])
                agent.real_pos['USDT']['value'] = value + float(agent.real_pos[asset].get('value'))
                pf_pct = float(agent.real_pos['USDT']['pf%'])
                agent.real_pos['USDT']['pf%'] = pf_pct + float(agent.real_pos[asset].get('pf%'))
            
            agent.counts_dict['real_close_long'] += 1
            uf.realised_pnl(agent, trade_record, 'long')
        
        
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
            agent.open_trades['pair'] = trade_record
            uf.record_trades(session, agent, 'open')
           
            agent.in_pos['real_pfrd'] = agent.in_pos['real_pfrd'] * (pct / 100)
            if session.live:
                agent.real_pos[asset].update(funcs.update_pos_M(session, asset, new_size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd']))
                agent.real_pos['USDT'] = funcs.update_usdt(session.bal)
            else:
                uf.calc_sizing_non_live_tp(session, asset, pct, 'real')
            
            agent.counts_dict['real_tp_long'] += 1
            uf.realised_pnl(agent, trade_record, 'long')
        
    if agent.in_pos.get('sim_tp_sig'):
        note = f"sim take-profit {pair} long 50% @ {price}"
        print(now, note)
        
        trade_record = agent.sim_trades.get(pair)
        sim_bal = abs(float(agent.sim_pos[asset]['qty']))
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
        agent.sim_trades[pair] = trade_record
        uf.record_trades(session, agent, 'sim')
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(session, agent.in_pos, asset, 50, 'sim')
    
        agent.counts_dict['sim_tp_long'] += 1
        agent.in_pos['sim_pfrd'] = agent.in_pos['sim_pfrd'] / 2
        uf.realised_pnl(agent, trade_record, 'long')
        
    if agent.in_pos.get('tracked_tp_sig'):
        note = f"tracked take-profit {pair} long 50% @ {price}"
        print(now, note)
        
        trade_record = agent.tracked_trades.get(pair)
        tracked_bal = abs(float(agent.tracked[asset]['qty']))
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
        agent.tracked_trades[pair] = trade_record
        uf.record_trades(session, agent, 'tracked')
        
        agent.counts_dict['tracked_tp_long'] += 1
        agent.in_pos['tracked_pfrd'] = agent.in_pos['tracked_pfrd'] / 2

def close_long(session, agent, pair):
    # initialise stuff
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == 'long':
        note = f"real close long {pair} @ {price}"
        print(now, note)
        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(float(agent.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        api_order = funcs.sell_asset_M(pair, base_size, session.live)
        usdt_size = api_order.get('cummulativeQuoteQty')
        funcs.repay_asset_M('USDT', usdt_size, session.live)
        
        sell_order = funcs.create_trade_dict(api_order, price, session.live)
        sell_order['type'] = 'close_long'
        sell_order['state'] = 'real'
        sell_order['reason'] = 'strategy close long signal'
        sell_order['liability'] = uf.update_liability(trade_record, usdt_size, 'reduce')
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
        uf.record_trades(session, agent, 'closed')
        agent.next_id += 1
        
        if agent.open_trades[pair]:
            del agent.open_trades[pair]
            uf.record_trades(session, agent, 'open')
        
        agent.in_pos['real'] = None
        agent.in_pos['real_pfrd'] = 0
        del agent.real_pos[asset]
        if session.live:
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
        else:
            value = float(agent.real_pos['USDT']['value'])
            agent.real_pos['USDT']['value'] = value + float(usdt_size)
            owed = float(agent.real_pos['USDT']['owed'])
            agent.real_pos['USDT']['owed'] = owed - float(usdt_size)
        
        # save records and update counts
        agent.counts_dict['real_close_long'] +=1
        uf.realised_pnl(agent, trade_record, 'long')
    
    if agent.in_pos['sim'] == 'long':
        note = f"sim close long {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.sim_trades[pair]
        sim_bal = float(agent.sim_pos[asset]['qty'])
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
            agent.closed_sim_trades[trade_id] = trade_record
        else:
            agent.closed_sim_trades[agent.next_id] = trade_record
            agent.next_id += 1
        uf.record_trades(session, agent, 'closed_sim')
        
        if agent.sim_trades[pair]:
            del agent.sim_trades[pair]
            uf.record_trades(session, agent, 'sim')
        
        # update counts and live variables
        agent.in_pos['sim'] = None
        agent.in_pos['sim_pfrd'] = 0        
        del agent.sim_pos[asset]
        
        agent.counts_dict['sim_close_long'] += 1
        uf.realised_pnl(agent, trade_record, 'long')
        
    if agent.in_pos['tracked'] == 'long':
        note = f"tracked close long {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.tracked_trades[pair]
        tracked_bal = Decimal(agent.tracked[asset]['qty'])
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
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
            agent.next_id += 1
        uf.record_trades(session, agent, 'closed')
        
        if agent.tracked_trades[pair]:
            del agent.tracked_trades[pair]
            uf.record_trades(session, agent, 'tracked')
        
        # update counts and live variables
        del agent.tracked[asset]
        
        agent.in_pos['tracked'] = None
        agent.in_pos['tracked_pfrd'] = 0        
        
        agent.counts_dict['tracked_close_long'] += 1

def open_short(session, agent, pair, size, stp, inval, sim_reason):
    # initialise stuff
    price = funcs.get_price(pair)
    usdt_size = round(size*price, 2)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == None and not sim_reason:
        note = f"real open short {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow
        funcs.borrow_asset_M(asset, size, session.live)
        
        # execute
        api_order = funcs.sell_asset_M(pair, size, session.live)
        
        # create trade record
        short_order = funcs.create_trade_dict(api_order, price, session.live)
        short_order['type'] = 'open_short'
        short_order['state'] = 'real'
        short_order['score'] = 'signal score'
        short_order['hard_stop'] = str(stp)
        short_order['liability'] = uf.update_liability(None, size, 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(pair, stop_size, be.SIDE_BUY, stp, stp*1.2, session.live)
        short_order['stop_id'] = stop_order.get('orderId')
        agent.open_trades[pair] = [short_order]
        uf.record_trades(session, agent, 'open')
        
        # update positions dictionaries
        agent.in_pos['real'] = 'short'
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_s
        agent.in_pos['real_ep'] = price
        agent.in_pos['real_hs'] = stp
        if session.live:
            agent.real_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd'])
            agent.real_pos[asset]['pnl_R'] = 0
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
        else:
            pf = usdt_size / session.bal
            or_dol = session.bal * agent.fixed_risk_s
            agent.real_pos[asset] = {'qty': size, 'value': usdt_size, 'pf%': pf, 'or_R': 1, 'or_$': or_dol}
        
        # save records and update counts
        agent.counts_dict['real_open_short'] +=1
        
    if agent.in_pos['sim'] == None and sim_reason:
        usdt_size = 100.0
        size = round(usdt_size / price, 8)
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
        agent.sim_trades[pair] = trade_record
        uf.record_trades(session, agent, 'sim')
        
        agent.in_pos['sim'] = 'short'
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_s
        agent.sim_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['sim'], agent.in_pos['sim_pfrd'])
        agent.sim_pos[asset]['pnl_R'] = 0
        
        agent.counts_dict['sim_open_short'] += 1

def tp_short(session, agent, pair, stp, inval):
    asset = pair[:-4]
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    session.bal = funcs.account_bal_M()
    
    if agent.in_pos.get('real_tp_sig'):        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(float(agent.real_pos[asset]['qty']))
        real_val = abs(float(agent.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        order_size = base_size * (pct/100)
        api_order = funcs.buy_asset_M(pair, order_size, True, session.live)
        buy_order = funcs.create_trade_dict(api_order, price, session.live)
        repay_size = buy_order.get('base_size')
        funcs.repay_asset_M(asset, repay_size, session.live)
        
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
            agent.tracked_trades[pair] = trade_record
            uf.record_trades(session, agent, 'tracked')
            
            if agent.open_trades[pair]:
                del agent.open_trades[pair]
                uf.record_trades(session, agent, 'open')
            
            agent.in_pos['real'] = None
            agent.in_pos['tracked'] = 'short'
            del agent.real_pos[asset]
            if session.live:
                agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
            else:
                agent.real_pos['USDT']['qty'] -= agent.real_pos[asset].get('value')
                agent.real_pos['USDT']['value'] -= agent.real_pos[asset].get('value')
                agent.real_pos['USDT']['pf%'] -= agent.real_pos[asset].get('pf%')
            agent.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
            
            agent.counts_dict['real_close_short'] += 1
            uf.realised_pnl(agent, trade_record, 'short')
        
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
            agent.open_trades['pair'] = trade_record
            uf.record_trades(session, agent, 'open')
           
            agent.in_pos['real_pfrd'] = agent.in_pos['real_pfrd'] * (pct / 100)
            if session.live:
                agent.real_pos[asset].update(funcs.update_pos_M(session, asset, new_size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd']))
                agent.real_pos['USDT'] = funcs.update_usdt(session.bal)
            else:
                uf.calc_sizing_non_live_tp(session, asset, pct, 'real')
            
            agent.counts_dict['real_tp_short'] += 1
            uf.realised_pnl(agent, trade_record, 'short')
        
    if agent.in_pos.get('sim_tp_sig'):
        note = f"sim take-profit {pair} short 50% @ {price}"
        print(now, note) 
        
        trade_record = agent.sim_trades.get(pair)
        sim_bal = abs(float(agent.sim_pos[asset]['qty']))
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
        agent.in_pos['sim_pfrd'] = agent.in_pos['sim_pfrd'] * (pct / 100)
        agent.sim_trades[pair] = trade_record
        uf.record_trades(session, agent, 'sim')
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(session, agent.in_pos, asset, 50, 'sim')
    
        agent.counts_dict['sim_tp_short'] += 1
        uf.realised_pnl(agent, trade_record, 'short')
        
    if agent.in_pos.get('tracked_tp_sig'):
        note = f"tracked take-profit {pair} short 50% @ {price}"
        print(now, note) 
        
        trade_record = agent.tracked_trades.get(pair)
        tracked_bal = abs(float(agent.tracked[asset]['qty']))
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
        agent.tracked_trades[pair] = trade_record
        uf.record_trades(session, agent, 'tracked')
        
        agent.counts_dict['tracked_tp_short'] += 1
        agent.in_pos['tracked_pfrd'] = agent.in_pos['tracked_pfrd'] / 2

def close_short(session, agent, pair):
    # initialise stuff
    price = funcs.get_price(pair)
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == 'short':
        note = f"real close short {pair} @ {price}"
        print(now, note)
        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(float(agent.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if base_size and (real_bal != base_size): # check records match reality
            print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
        if not base_size:
            base_size = real_bal
        
        # execute trade
        api_order = funcs.buy_asset_M(pair, base_size, True, session.live)
        funcs.repay_asset_M(asset, base_size, session.live)
        
        sell_order = funcs.create_trade_dict(api_order, price, session.live)
        sell_order['type'] = 'close_short'
        sell_order['state'] = 'real'
        sell_order['reason'] = 'strategy close short signal'
        sell_order['liability'] = uf.update_liability(trade_record, base_size, 'reduce')
        trade_record.append(sell_order)
        
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = trade_record[0].get('timestamp')
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
        uf.record_trades(session, agent, 'closed')
        agent.next_id += 1
        
        if agent.open_trades[pair]:
            del agent.open_trades[pair]
            uf.record_trades(session, agent, 'open')
        
        agent.in_pos['real'] = None
        agent.in_pos['real_pfrd'] = 0
        if session.live:
            del agent.real_pos[asset]
            agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
        else:
            del agent.real_pos[asset]
            value = float(agent.real_pos['USDT']['value'])
            agent.real_pos['USDT']['value'] = value + float(base_size * price)
            owed = float(agent.real_pos['USDT']['owed'])
            agent.real_pos['USDT']['owed'] = owed - float(base_size * price)
        
        # save records and update counts
        agent.counts_dict['real_close_short'] +=1
        uf.realised_pnl(agent, trade_record, 'short')
    
    if agent.in_pos['sim'] == 'short':
        note = f"sim close short {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.sim_trades[pair]
        sim_bal = float(agent.sim_pos[asset]['qty'])
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
            agent.closed_sim_trades[trade_id] = trade_record
        else:
            agent.closed_sim_trades[agent.next_id] = trade_record
            agent.next_id += 1
        uf.record_trades(session, agent, 'closed_sim')
        
        if agent.sim_trades[pair]:
            del agent.sim_trades[pair]
            uf.record_trades(session, agent, 'sim')
        
        # update live variables
        del agent.sim_pos[asset]
        agent.in_pos['sim'] = None
        agent.in_pos['sim_pfrd'] = 0        
        
        agent.counts_dict['sim_close_short'] += 1
        uf.realised_pnl(agent, trade_record, 'short')
        
    if agent.in_pos['tracked'] == 'short':
        note = f"tracked close short {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.tracked_trades[pair]
        tracked_bal = Decimal(agent.tracked[asset]['qty'])
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
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
            agent.next_id += 1
        uf.record_trades(session, agent, 'closed')
        
        if agent.tracked_trades[pair]:
            del agent.tracked_trades[pair]
            uf.record_trades(session, agent, 'tracked')
        
        # update live variables
        del agent.tracked[asset]
        agent.in_pos['tracked'] = None
        agent.in_pos['tracked_pfrd'] = 0        
        
        agent.counts_dict['tracked_close_short'] += 1

def reduce_risk_M(session, agent):
    # create a list of open positions in profit and their open risk value
    positions = [(p, r.get('or_R'), r.get('pnl_%')) 
                 for p, r in agent.real_pos.items() 
                 if r.get('or_R') and (r.get('or_R') > 0)]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [x.get('or_R') for x in agent.real_pos.values() if x.get('or_R')]
        total_r = sum(r_list)       
        
        for pos in sorted_pos:
            asset = pos[0]
            or_R = pos[1]
            pnl_pct = pos[2]
            if total_r > agent.total_r_limit and or_R > 1.1 and pnl_pct > 0.3:
                print(f'*** tor: {total_r:.1f}, reducing risk ***')
                pair = asset + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"reduce risk {pair}, or: {or_R}R, pnl: {pnl_pct}%"
                print(now, note)
                try:
                    # push = pb.push_note(now, note)
                    if agent.open_trades.get(pair):
                        trade_record = agent.open_trades.get(pair)
                    else:
                        trade_record = []
                    
                    real_bal = abs(float(agent.real_pos[asset]['qty']))
                    
                    # clear stop
                    clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
                    if base_size and (real_bal != base_size): # check records match reality
                        print(f"{pair} records don't match real balance. {real_bal = }, {base_size = }")
                    if not base_size:
                        base_size = real_bal
                    
                    long = trade_record[0].get('type')[-4:] == 'long'
                    if long:
                        api_order = funcs.sell_asset_M(pair, base_size, session.live)
                        usdt_size = api_order.get('cummulativeQuoteQty')
                        repay_size = usdt_size
                        funcs.repay_asset_M('USDT', repay_size, session.live)
                    else:
                        api_order = funcs.buy_asset_M(pair, base_size, True, session.live)
                        repay_size = base_size
                        funcs.repay_asset_M(asset, repay_size, session.live)
                    
                    reduce_order = funcs.create_trade_dict(api_order, price, session.live)
                    reduce_order['type'] = 'close_long' if long else 'close_short'
                    reduce_order['state'] = 'real'
                    reduce_order['reason'] = 'portfolio risk limiting'
                    reduce_order['liability'] = uf.update_liability(trade_record, repay_size, 'reduce')
                    
                    trade_record.append(reduce_order)
                    
                    agent.sim_trades[pair] = trade_record
                    uf.record_trades(session, agent, 'sim')
                    
                    if agent.open_trades[pair]:
                        del agent.open_trades[pair]
                        uf.record_trades(session, agent, 'open')
                    
                    agent.counts_dict['reduce_risk'] += 1
                    total_r -= or_R
                    
                    if session.live:
                        del agent.real_pos[asset]
                        agent.real_pos['USDT'] = funcs.update_usdt_M(session.bal)
                    elif long and not session.live:
                        del agent.real_pos[asset]
                        agent.real_pos['USDT']['value'] += float(usdt_size)
                        agent.real_pos['USDT']['owed'] -= float(usdt_size)
                    else:
                        del agent.real_pos[asset]
                        agent.real_pos['USDT']['value'] += float(base_size * price)
                        agent.real_pos['USDT']['owed'] -= float(base_size * price)
                    
                    del agent.real_pos[asset]
                    uf.realised_pnl(agent, trade_record, 'long')
                except BinanceAPIException as e:
                    print(f'problem with sell order for {pair}')
                    print(e)
                    pb.push_note(now, f'exeption during {pair} sell order')
                    continue





import binance_funcs as funcs
import utility_funcs as uf
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from decimal import Decimal, getcontext

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)
ctx = getcontext()
ctx.prec = 12


def open_long(session, agent, pair, size, stp, inval, sim_reason):
    
    asset = pair[:-4]
    price = session.prices[pair]
    usdt_size: str = f"{size*price:.2f}"
    # session.bal = funcs.account_bal_M()
    now = datetime.now().strftime('%d/%m/%y %H:%M')
        
    if agent.in_pos['real'] == None and not sim_reason: # if state = real
        print('')
        
        # # insert placeholder record
        # placeholder = {'order': 'open_long', 
        #                 'state': 'real', 
        #                 'agent': agent.id, 
        #                 'pair': pair, 
        #                 'base_size': size, 
        #                 'stop_price': stp, 
        #                 'inval': inval, 
        #                 'timestamp': now, 
        #                 'completed': None
        #                 }
        # agent.open_trades[pair] = [placeholder]
        # agent.record_trades(session, 'open')
    
        note = f"{agent.name} real open long {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow usdt
        funcs.borrow_asset_M('USDT', usdt_size, session.live)
        
        # execute
        api_order = funcs.buy_asset_M(session, pair, float(usdt_size), False, price, session.live)
        
        # create trade record
        long_order = funcs.create_trade_dict(api_order, price, session.live)
        long_order['type'] = 'open_long'
        long_order['state'] = 'real'
        long_order['score'] = 'signal score'
        long_order['hard_stop'] = str(stp)
        long_order['init_hs'] = str(stp)
        long_order['liability'] = uf.update_liability(None, usdt_size, 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_SELL, stp, stp*0.8)
        long_order['stop_id'] = stop_order.get('orderId')
        
        agent.open_trades[pair] = [long_order]
        agent.record_trades(session, 'open')
        
        # update positions dictionaries
        agent.in_pos['real'] = 'long'
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_l
        agent.in_pos['real_ep'] = price
        agent.in_pos['real_hs'] = stp
        if session.live:
            agent.real_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd'])
            agent.real_pos[asset]['pnl_R'] = 0
            session.update_usdt_M(borrow=float(usdt_size))
        else:
            pf = f"{float(usdt_size)/session.bal:.2f}"
            or_dol = f"{session.bal*agent.fixed_risk_l:.2f}"
            agent.real_pos[asset] = {'qty': str(size), 'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol)}
        
        # save records and update counts
        agent.counts_dict['real_open_long'] +=1
        
    if agent.in_pos['sim'] == None and sim_reason:
        
        # # insert placeholder record
        # placeholder = {'order': 'open_long', 
        #                 'state': 'sim', 
        #                 'agent': agent.id, 
        #                 'pair': pair, 
        #                 'base_size': size, 
        #                 'stop_price': stp, 
        #                 'inval': inval, 
        #                 'sim_reason': sim_reason, 
        #                 'timestamp': now, 
        #                 'upto': None
        #                 }
        # agent.sim_trades[pair] = [placeholder]
        # agent.record_trades(session, 'sim')
    
        usdt_size = 128.0
        size = f"{usdt_size/price:.8f}"
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim open long {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        #     print(now, note)
        
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        long_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(size), 
                     'quote_size': f"{usdt_size:.2f}", 
                     'hard_stop': str(stp), 
                     'reason': sim_reason, 
                     'timestamp': timestamp, 
                     'type': 'open_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record = [long_order]
        agent.sim_trades[pair] = trade_record
        agent.record_trades(session, 'sim')
        
        agent.in_pos['sim'] = 'long'
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_l
        agent.sim_pos[asset] = funcs.update_pos_M(session, asset, float(size), inval, agent.in_pos['sim'], agent.in_pos['sim_pfrd'])
        agent.sim_pos[asset]['pnl_R'] = 0
        
        agent.counts_dict['sim_open_long'] += 1

def tp_long(session, agent, pair, stp, inval):
    asset = pair[:-4]
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    # session.bal = funcs.account_bal_M()
    
    if agent.in_pos.get('real_tp_sig'):
        print('')
        trade_record = agent.open_trades.get(pair)
        
        # # insert placeholder record
        # placeholder = {'order': 'tp_long', 
        #                 'state': 'real', 
        #                 'agent': agent.id, 
        #                 'pair': pair, 
        #                 'stop_price': stp, 
        #                 'inval': inval, 
        #                 'timestamp': now, 
        #                 'completed': None
        #                 }
        # agent.open_trades[pair].append(placeholder)
        # agent.record_trades(session, 'open')
    
        real_bal = abs(Decimal(agent.real_pos[asset]['qty']))
        real_val = abs(Decimal(agent.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        
        if clear == 'error':
            print(f"{agent.name} Can't be sure which {pair} stop to clear, tp_long aborted")
            pb.push_note(pair, "Can't be sure which stop to clear, tp_long aborted")
        else:
            if base_size and (float(real_bal) != float(base_size)): # check records match reality
                print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            if not base_size:
                base_size = real_bal
            
            order_size = float(base_size) * (pct/100)
            
            # # update placeholder
            # placeholder['base_size'] = order_size
            # placeholder['completed'] = 'clear_stop'
            # agent.open_trades[pair].append(placeholder)
            # agent.record_trades(session, 'open')
            
            # execute trade
            api_order = funcs.sell_asset_M(session, pair, order_size, price, session.live)
            sell_order = funcs.create_trade_dict(api_order, price, session.live)
            
            # # update placeholder
            # placeholder['completed'] = 'sell_asset'
            # agent.open_trades[pair] = [placeholder]
            # agent.record_trades(session, 'open')
            
            note = f"{agent.name} real take-profit {pair} long {pct}% @ {price}"
            print(now, note)        
            
            if pct == 100:
                # repay assets
                usdt_size = str(max(Decimal(api_order.get('cummulativeQuoteQty', 0)), Decimal(trade_record[-1].get('liability', 0))))
                funcs.repay_asset_M('USDT', usdt_size, session.live)
                
                # create trade dict
                sell_order['type'] = 'close_long'
                sell_order['state'] = 'real'
                sell_order['reason'] = 'trade over-extended'
                sell_order['liability'] = '0'
                trade_record.append(sell_order)
                
                # update records            
                agent.tracked_trades[pair] = trade_record
                agent.record_trades(session, 'tracked')
                
                if agent.open_trades[pair]:
                    del agent.open_trades[pair]
                    agent.record_trades(session, 'open')
                
                agent.in_pos['real'] = None
                agent.in_pos['tracked'] = 'long'
                
                agent.tracked[asset] = {'qty': '0', 'value': '0', 'pf%': '0', 'or_R': '0', 'or_$': '0'}                
                if session.live:
                    session.update_usdt_M(repay=float(usdt_size))
                else:
                    qty = float(agent.real_pos['USDT']['qty']) + float(agent.real_pos[asset].get('value'))
                    agent.real_pos['USDT']['qty'] = f"{qty:.2f}"
                    value = float(agent.real_pos['USDT']['value']) + float(agent.real_pos[asset].get('value'))
                    agent.real_pos['USDT']['value'] = f"{value:.2f}"
                    pf_pct = float(agent.real_pos['USDT']['pf%']) + float(agent.real_pos[asset].get('pf%'))
                    agent.real_pos['USDT']['pf%'] = f"{pf_pct:.2f}"
                
                del agent.real_pos[asset]
                agent.counts_dict['real_close_long'] += 1
                agent.realised_pnl(trade_record, 'long')
            
            
            else: # if pct < 100%
                # repay assets
                usdt_size = api_order.get('cummulativeQuoteQty')
                funcs.repay_asset_M('USDT', usdt_size, session.live)
                
                # create trade dict
                sell_order['type'] = 'tp_long'
                sell_order['state'] = 'real'
                sell_order['hard_stop'] = str(stp)
                sell_order['reason'] = 'trade over-extended'
                sell_order['liability'] = uf.update_liability(trade_record, usdt_size, 'reduce')
                
                # set new stop
                new_size = real_bal - Decimal(sell_order['base_size'])
                stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_SELL, stp, stp*0.8)
                sell_order['stop_id'] = stop_order.get('orderId')
                
                trade_record.append(sell_order)
                
                # update records
                agent.open_trades[pair] = trade_record
                agent.record_trades(session, 'open')
               
                agent.in_pos['real_pfrd'] = agent.in_pos['real_pfrd'] * (pct / 100)
                if session.live:
                    agent.real_pos[asset].update(funcs.update_pos_M(session, asset, new_size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd']))
                    session.update_usdt_M(repay=float(usdt_size))
                else:
                    uf.calc_sizing_non_live_tp(session, agent, asset, pct, 'real')
                
                agent.counts_dict['real_tp_long'] += 1
                agent.realised_pnl(trade_record, 'long')
            
    if agent.in_pos.get('sim_tp_sig'):
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim take-profit {pair} long 50% @ {price}"
        #     print(now, note)
        
        trade_record = agent.sim_trades.get(pair)
        sim_bal = abs(float(agent.sim_pos[asset]['qty']))
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        order_size = sim_bal / 2
        usdt_size = f"{order_size * price:.2f}"
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(order_size), 
                     'quote_size': usdt_size, 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record.append(tp_order)
        
        # update records
        agent.sim_trades[pair] = trade_record
        agent.record_trades(session, 'sim')
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(session, agent, asset, 50, 'sim')
    
        agent.counts_dict['sim_tp_long'] += 1
        agent.in_pos['sim_pfrd'] = agent.in_pos['sim_pfrd'] / 2
        agent.realised_pnl(trade_record, 'long')
        
    if agent.in_pos.get('tracked_tp_sig'):
        print('')
        note = f"{agent.name} tracked take-profit {pair} long 50% @ {price}"
        print(now, note)
        
        trade_record = agent.tracked_trades.get(pair)
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': '0', 
                     'quote_size': '0', 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(tp_order)
        
        # update records
        agent.tracked_trades[pair] = trade_record
        agent.record_trades(session, 'tracked')
        
        agent.in_pos['tracked_pfrd'] = agent.in_pos['tracked_pfrd'] / 2

def close_long(session, agent, pair):
    # initialise stuff
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    # session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == 'long':
        print('')
        note = f"{agent.name} real close long {pair} @ {price}"
        print(now, note)
        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(Decimal(agent.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if clear == 'error':
            print(f"{agent.name} Can't be sure which {pair} stop to clear, close_long aborted")
            pb.push_note(pair, "Can't be sure which stop to clear, close_long aborted")
        else:
            if base_size and (float(real_bal) != float(base_size)): # check records match reality
                print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            if not base_size:
                base_size = real_bal
            
            # execute trade
            api_order = funcs.sell_asset_M(session, pair, real_bal, price, session.live)
            usdt_size = str(max(Decimal(api_order.get('cummulativeQuoteQty', 0)), Decimal(trade_record[-1].get('liability', 0))))
            funcs.repay_asset_M('USDT', usdt_size, session.live)
            
            sell_order = funcs.create_trade_dict(api_order, price, session.live)
            sell_order['type'] = 'close_long'
            sell_order['state'] = 'real'
            sell_order['reason'] = 'strategy close long signal'
            sell_order['liability'] = '0'
            trade_record.append(sell_order)
            
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = int(trade_record[0].get('timestamp'))
                agent.closed_trades[trade_id] = trade_record
            else:
                agent.closed_trades[agent.next_id] = trade_record
            agent.record_trades(session, 'closed')
            agent.next_id += 1
            
            if agent.open_trades[pair]:
                del agent.open_trades[pair]
                agent.record_trades(session, 'open')
            
            agent.in_pos['real'] = None
            agent.in_pos['real_pfrd'] = 0
            if session.live:
                session.update_usdt_M(repay=float(usdt_size))
            else:
                value = float(agent.real_pos['USDT']['value'])
                agent.real_pos['USDT']['value'] = f"{value + float(usdt_size):.2f}"
                owed = float(agent.real_pos['USDT']['owed'])
                agent.real_pos['USDT']['owed'] = f"{owed - float(usdt_size):.2f}"
            
            # save records and update counts
            del agent.real_pos[asset]
            agent.counts_dict['real_close_long'] +=1
            agent.realised_pnl(trade_record, 'long')
    
    if agent.in_pos['sim'] == 'long':
        
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim close long {pair} @ {price}"
        #     print(now, note)
        
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
            trade_id = int(trade_record[0].get('timestamp'))
            agent.closed_sim_trades[trade_id] = trade_record
        else:
            agent.closed_sim_trades[agent.next_id] = trade_record
            agent.next_id += 1
        agent.record_trades(session, 'closed_sim')
        
        if agent.sim_trades[pair]:
            del agent.sim_trades[pair]
            agent.record_trades(session, 'sim')
        
        # update counts and live variables
        agent.in_pos['sim'] = None
        agent.in_pos['sim_pfrd'] = 0        
        del agent.sim_pos[asset]
        
        agent.counts_dict['sim_close_long'] += 1
        agent.realised_pnl(trade_record, 'long')
        
    if agent.in_pos['tracked'] == 'long':
        print('')
        note = f"{agent.name} tracked close long {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.tracked_trades[pair]
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        long_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': '0', 
                     'quote_size': '0', 
                     'reason': 'strategy close long signal', 
                     'timestamp': timestamp, 
                     'type': 'close_long', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(long_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = int(trade_record[0].get('timestamp'))
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
            agent.next_id += 1
        agent.record_trades(session, 'closed')
        
        if agent.tracked_trades[pair]:
            del agent.tracked_trades[pair]
            agent.record_trades(session, 'tracked')
        
        # update counts and live variables
        del agent.tracked[asset]
        
        agent.in_pos['tracked'] = None
        agent.in_pos['tracked_pfrd'] = 0

def open_short(session, agent, pair, size, stp, inval, sim_reason):
    # initialise stuff
    price = session.prices[pair]
    usdt_size = f"{size*price:.2f}"
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    # session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == None and not sim_reason:
        print('')
        
        # # insert placeholder record
        # placeholder = {'order': 'open_short', 
        #                 'state': 'real', 
        #                 'agent': agent.id, 
        #                 'pair': pair, 
        #                 'base_size': size, 
        #                 'stop_price': stp, 
        #                 'inval': inval, 
        #                 'timestamp': now, 
        #                 'completed': None
        #                 }
        # agent.open_trades[pair] = [placeholder]
        # agent.record_trades(session, 'open')
        
        note = f"{agent.name} real open short {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}"
        print(now, note)
        
        # borrow
        size = funcs.valid_size(session, pair, size)
        funcs.borrow_asset_M(asset, size, session.live)
        
        # execute
        api_order = funcs.sell_asset_M(session, pair, size, price, session.live)
        
        # create trade record
        short_order = funcs.create_trade_dict(api_order, price, session.live)
        short_order['type'] = 'open_short'
        short_order['state'] = 'real'
        short_order['score'] = 'signal score'
        short_order['hard_stop'] = str(stp)
        short_order['init_hs'] = str(stp)
        short_order['liability'] = uf.update_liability(None, str(size), 'increase')
        
        # set stop and add to trade record
        stop_size = float(api_order.get('executedQty'))
        stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_BUY, stp, stp*1.2)
        short_order['stop_id'] = stop_order.get('orderId')
        agent.open_trades[pair] = [short_order]
        agent.record_trades(session, 'open')
        
        # update positions dictionaries
        agent.in_pos['real'] = 'short'
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_s
        agent.in_pos['real_ep'] = price
        agent.in_pos['real_hs'] = stp
        if session.live:
            agent.real_pos[asset] = funcs.update_pos_M(session, asset, float(size), inval, agent.in_pos['real'], agent.in_pos['real_pfrd'])
            agent.real_pos[asset]['pnl_R'] = 0
            session.update_usdt_M(up=float(usdt_size))
        else:
            pf = float(usdt_size) / session.bal
            or_dol = session.bal * agent.fixed_risk_s
            agent.real_pos[asset] = {'qty': str(size), 'value': usdt_size, 'pf%': str(pf), 'or_R': '1', 'or_$': str(or_dol)}
        
        # save records and update counts
        agent.counts_dict['real_open_short'] +=1
        
    if agent.in_pos['sim'] == None and sim_reason:
        
        # # insert placeholder record
        # placeholder = {'order': 'open_short', 
        #                 'state': 'sim', 
        #                 'agent': agent.id, 
        #                 'pair': pair, 
        #                 'base_size': size, 
        #                 'stop_price': stp, 
        #                 'inval': inval, 
        #                 'sim_reason': sim_reason, 
        #                 'timestamp': now, 
        #                 'upto': None
        #                 }
        # agent.sim_trades[pair] = [placeholder]
        # agent.record_trades(session, 'sim')
    
        usdt_size = 128.0
        size = round(usdt_size / price, 8)
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim open short {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
        #     print(now, note)
        
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        short_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': str(size), 
                     'quote_size': f"{usdt_size:.2f}", 
                     'hard_stop': str(stp), 
                     'reason': sim_reason, 
                     'timestamp': timestamp, 
                     'type': 'open_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'sim'}
        trade_record = [short_order]
        agent.sim_trades[pair] = trade_record
        agent.record_trades(session, 'sim')
        
        agent.in_pos['sim'] = 'short'
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_s
        agent.sim_pos[asset] = funcs.update_pos_M(session, asset, float(size), inval, agent.in_pos['sim'], agent.in_pos['sim_pfrd'])
        agent.sim_pos[asset]['pnl_R'] = 0
        
        agent.counts_dict['sim_open_short'] += 1

def tp_short(session, agent, pair, stp, inval):
    asset = pair[:-4]
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    # session.bal = funcs.account_bal_M()
    
    if agent.in_pos.get('real_tp_sig'):
        print('')
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(Decimal(agent.real_pos[asset]['qty']))
        real_val = abs(Decimal(agent.real_pos[asset]['value']))
        pct = 50 if real_val > 24 else 100
        
        # clear stop
        
        print('clearing stop')
        
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if clear == 'error':
            print(f"{agent.name} Can't be sure which {pair} stop to clear, tp_short aborted")
            pb.push_note(pair, "Can't be sure which stop to clear, tp_short aborted")
        else:
            if base_size and (float(real_bal) != float(base_size)): # check records match reality
                print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            if not base_size:
                base_size = real_bal
            
            # execute trade
            order_size = float(base_size) * (pct/100)
            
            api_order = funcs.buy_asset_M(session, pair, order_size, True, price, session.live)
            buy_order = funcs.create_trade_dict(api_order, price, session.live)
            
            note = f"{agent.name} real take-profit {pair} short {pct}% @ {price}"
            print(now, note)        
            
            if pct == 100:
                repay_size = str(max(Decimal(trade_record[-1].get('liability', 0)), Decimal(buy_order.get('base_size', 0))))
                funcs.repay_asset_M(asset, repay_size, session.live)
                
                # create trade dict
                buy_order['type'] = 'close_short'
                buy_order['state'] = 'real'
                buy_order['reason'] = 'trade over-extended'
                buy_order['liability'] = '0'
                trade_record.append(buy_order)
                
                # update records            
                agent.tracked_trades[pair] = trade_record
                agent.record_trades(session, 'tracked')
                
                if agent.open_trades[pair]:
                    del agent.open_trades[pair]
                    agent.record_trades(session, 'open')
                
                agent.in_pos['real'] = None
                agent.in_pos['tracked'] = 'short'
                if session.live:
                    usdt_size = round(order_size * price, 5)
                    session.update_usdt_M(down=usdt_size)
                else:
                    agent.real_pos['USDT']['qty'] -= float(agent.real_pos[asset].get('value'))
                    agent.real_pos['USDT']['value'] -= float(agent.real_pos[asset].get('value'))
                    agent.real_pos['USDT']['pf%'] -= float(agent.real_pos[asset].get('pf%'))
                agent.tracked[asset] = {'qty': 0, 'value': 0, 'pf%': 0, 'or_R': 0, 'or_$': 0}                
                
                del agent.real_pos[asset]
                agent.counts_dict['real_close_short'] += 1
                agent.realised_pnl(trade_record, 'short')
            
            else: # if pct < 100%
                repay_size = buy_order.get('base_size')
                funcs.repay_asset_M(asset, repay_size, session.live)
                
                # create trade dict
                buy_order['type'] = 'tp_short'
                buy_order['state'] = 'real'
                buy_order['hard_stop'] = str(stp)
                buy_order['reason'] = 'trade over-extended'
                buy_order['liability'] = uf.update_liability(trade_record, repay_size, 'reduce')
                
                # set new stop
                new_size = real_bal - Decimal(buy_order['base_size'])
                
                stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_BUY, stp, stp*1.2)
                buy_order['stop_id'] = stop_order.get('orderId')
                
                trade_record.append(buy_order)
                
                # update records
                agent.open_trades[pair] = trade_record
                agent.record_trades(session, 'open')
               
                agent.in_pos['real_pfrd'] = agent.in_pos['real_pfrd'] * (pct / 100)
                if session.live:
                    agent.real_pos[asset].update(funcs.update_pos_M(session, asset, new_size, inval, agent.in_pos['real'], agent.in_pos['real_pfrd']))
                    usdt_size = round(order_size * price, 5)
                    session.update_usdt_M(down=usdt_size)
                else:
                    uf.calc_sizing_non_live_tp(session, agent, asset, pct, 'real')
                
                agent.counts_dict['real_tp_short'] += 1
                agent.realised_pnl(trade_record, 'short')
            
    if agent.in_pos.get('sim_tp_sig'):
        
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim take-profit {pair} short 50% @ {price}"
        #     print(now, note) 
        
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
        agent.in_pos['sim_pfrd'] = agent.in_pos['sim_pfrd'] / 2
        agent.sim_trades[pair] = trade_record
        agent.record_trades(session, 'sim')
        
        # update sim_pos
        uf.calc_sizing_non_live_tp(session, agent, asset, 50, 'sim')
    
        agent.counts_dict['sim_tp_short'] += 1
        agent.realised_pnl(trade_record, 'short')
        
    if agent.in_pos.get('tracked_tp_sig'):
        print('')
        note = f"{agent.name} tracked take-profit {pair} short 50% @ {price}"
        print(now, note) 
        
        trade_record = agent.tracked_trades.get(pair)
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        tp_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': '0', 
                     'quote_size': '0', 
                     'reason': 'trade over-extended', 
                     'timestamp': timestamp, 
                     'type': 'tp_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(tp_order)
        
        # update records
        agent.tracked_trades[pair] = trade_record
        agent.record_trades(session, 'tracked')
        
        agent.in_pos['tracked_pfrd'] = agent.in_pos['tracked_pfrd'] / 2

def close_short(session, agent, pair):
    # initialise stuff
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    asset = pair[:-4]
    # session.bal = funcs.account_bal_M()
    
    if agent.in_pos['real'] == 'short':
        print('')
        note = f"{agent.name} real close short {pair} @ {price}"
        print(now, note)
        
        trade_record = agent.open_trades.get(pair)
        real_bal = abs(Decimal(agent.real_pos[asset]['qty']))
        
        # cancel stop
        clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
        if clear == 'error':
            print(f"{agent.name} Can't be sure which {pair} stop to clear, close_short aborted")
            pb.push_note(pair, "Can't be sure which stop to clear, close_short aborted")
        else:
            if base_size and (float(real_bal) != float(base_size)): # check records match reality
                print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
            if not base_size:
                base_size = real_bal
            
            # execute trade
            api_order = funcs.buy_asset_M(session, pair, base_size, True, price, session.live)
            sell_order = funcs.create_trade_dict(api_order, price, session.live)
            repay_size = str(max(Decimal(trade_record[-1].get('liability', 0)), Decimal(sell_order.get('base_size', 0))))
            funcs.repay_asset_M(asset, repay_size, session.live)
            
            sell_order['type'] = 'close_short'
            sell_order['state'] = 'real'
            sell_order['reason'] = 'strategy close short signal'
            sell_order['liability'] = '0'
            trade_record.append(sell_order)
            
            if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
                trade_id = int(trade_record[0].get('timestamp'))
                agent.closed_trades[trade_id] = trade_record
            else:
                agent.closed_trades[agent.next_id] = trade_record
            agent.record_trades(session, 'closed')
            agent.next_id += 1
            
            if agent.open_trades[pair]:
                del agent.open_trades[pair]
                agent.record_trades(session, 'open')
            
            agent.in_pos['real'] = None
            agent.in_pos['real_pfrd'] = 0
            if session.live:
                usdt_size = round(base_size * price, 5)
                session.update_usdt_M(down=usdt_size)
            else:
                value = float(agent.real_pos['USDT']['value'])
                agent.real_pos['USDT']['value'] = value + (float(base_size) * price)
                owed = float(agent.real_pos['USDT']['owed'])
                agent.real_pos['USDT']['owed'] = owed - (float(base_size) * price)
            
            # save records and update counts
            del agent.real_pos[asset]
            agent.counts_dict['real_close_short'] +=1
            agent.realised_pnl(trade_record, 'short')
    
    if agent.in_pos['sim'] == 'short':
        
        # if not session.live:
        #     print('')
        #     note = f"{agent.name} sim close short {pair} @ {price}"
        #     print(now, note)
        
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
            trade_id = int(trade_record[0].get('timestamp'))
            agent.closed_sim_trades[trade_id] = trade_record
        else:
            agent.closed_sim_trades[agent.next_id] = trade_record
            agent.next_id += 1
        agent.record_trades(session, 'closed_sim')
        
        if agent.sim_trades[pair]:
            del agent.sim_trades[pair]
            agent.record_trades(session, 'sim')
        
        # update live variables
        del agent.sim_pos[asset]
        agent.in_pos['sim'] = None
        agent.in_pos['sim_pfrd'] = 0        
        
        agent.counts_dict['sim_close_short'] += 1
        agent.realised_pnl(trade_record, 'short')
        
    if agent.in_pos['tracked'] == 'short':
        print('')
        note = f"{agent.name} tracked close short {pair} @ {price}"
        print(now, note)
        
        # initialise stuff
        trade_record = agent.tracked_trades[pair]
        timestamp = round(datetime.utcnow().timestamp() * 1000)
        
        # execute order
        short_order = {'pair': pair, 
                     'exe_price': str(price), 
                     'trig_price': str(price), 
                     'base_size': '0', 
                     'quote_size': '0', 
                     'reason': 'strategy close short signal', 
                     'timestamp': timestamp, 
                     'type': 'close_short', 
                     'fee': '0', 
                     'fee_currency': 'BNB', 
                     'state': 'tracked'}
        trade_record.append(short_order)
        
        # update records
        if trade_record[0].get('type')[0] == 'o': # if the trade record includes the trade open
            trade_id = int(trade_record[0].get('timestamp'))
            agent.closed_trades[trade_id] = trade_record
        else:
            agent.closed_trades[agent.next_id] = trade_record
            agent.next_id += 1
        agent.record_trades(session, 'closed')
        
        if agent.tracked_trades[pair]:
            del agent.tracked_trades[pair]
            agent.record_trades(session, 'tracked')
        
        # update live variables
        del agent.tracked[asset]
        agent.in_pos['tracked'] = None
        agent.in_pos['tracked_pfrd'] = 0

def reduce_risk_M(session, agent):
    # create a list of open positions in profit and their open risk value
    positions = [(p, float(r.get('or_R')), float(r.get('pnl_%'))) 
                 for p, r in agent.real_pos.items() 
                 if r.get('or_R') and (float(r.get('or_R')) > 0) 
                 and r.get('pnl_%')]
    
    if positions:
        # sort the list so biggest open risk is first
        sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
        # pprint(sorted_pos)

        # # create a new list with just the R values
        r_list = [float(x.get('or_R')) for x in agent.real_pos.values() if x.get('or_R')]
        total_r = sum(r_list)       
        
        for pos in sorted_pos:
            asset = pos[0]
            or_R = pos[1]
            pnl_pct = pos[2]
            if total_r > agent.total_r_limit and or_R > agent.indiv_r_limit and pnl_pct > 0.3:
                print(f'\n*** tor: {total_r:.1f}, reducing risk ***')
                pair = asset + 'USDT'
                now = datetime.now().strftime('%d/%m/%y %H:%M')
                price = funcs.get_price(pair)
                note = f"{agent.name} reduce risk {pair}, or: {or_R}R, pnl: {pnl_pct}%"
                print(now, note)
                try:
                    # push = pb.push_note(now, note)
                    if agent.open_trades.get(pair):
                        trade_record = agent.open_trades.get(pair)
                    else:
                        trade_record = []
                    
                    real_bal = abs(Decimal(agent.real_pos[asset]['qty']))
                    
                    # clear stop
                    clear, base_size = funcs.clear_stop_M(pair, trade_record, session.live)
                    if clear == 'error':
                        note = f"{agent.name} Can't be sure which {pair} stop to clear, close_short aborted"
                        print(note)
                        pb.push_note(pair, note)
                    else:
                        if base_size and (float(real_bal) != float(base_size)): # check records match reality
                            print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
                            mismatch = 100 * abs(base_size - real_bal) / base_size
                            print(f"{mismatch = }%")
                        if not base_size:
                            base_size = real_bal
                        
                        long = trade_record[0].get('type')[-4:] == 'long'
                        if long:
                            api_order = funcs.sell_asset_M(session, pair, base_size, price, session.live)
                            usdt_size = api_order.get('cummulativeQuoteQty')
                            repay_size = max(usdt_size, trade_record[-1].get('liability', 0))
                            funcs.repay_asset_M('USDT', repay_size, session.live)
                        else:
                            api_order = funcs.buy_asset_M(session, pair, base_size, True, price, session.live)
                            usdt_size = base_size * price
                            repay_size = str(max(base_size, trade_record[-1].get('liability', 0)))
                            funcs.repay_asset_M(asset, repay_size, session.live)
                        
                        reduce_order = funcs.create_trade_dict(api_order, price, session.live)
                        reduce_order['type'] = 'close_long' if long else 'close_short'
                        reduce_order['state'] = 'real'
                        reduce_order['reason'] = 'portfolio risk limiting'
                        reduce_order['liability'] = '0'
                        
                        trade_record.append(reduce_order)
                        
                        agent.sim_trades[pair] = trade_record
                        agent.record_trades(session, 'sim')
                        
                        if agent.open_trades[pair]:
                            del agent.open_trades[pair]
                            agent.record_trades(session, 'open')
                        
                        agent.counts_dict['reduce_risk'] += 1
                        total_r -= or_R
                        
                        if session.live:
                            if long:
                                session.update_usdt_M(repay=float(usdt_size))
                            else:
                                session.update_usdt_M(down=float(usdt_size))
                        elif long and not session.live:
                            agent.real_pos['USDT']['value'] += float(usdt_size)
                            agent.real_pos['USDT']['owed'] -= float(usdt_size)
                        else:
                            agent.real_pos['USDT']['value'] += float(base_size * price)
                            agent.real_pos['USDT']['owed'] -= float(base_size * price)
                        
                        del agent.real_pos[asset]
                        if long:
                            agent.realised_pnl(trade_record, 'long')
                        else:
                            agent.realised_pnl(trade_record, 'short')
                except BinanceAPIException as e:
                    print(f'problem with reduce_risk order for {pair}')
                    print(e)
                    pb.push_note(now, f'exeption during {pair} sell order')
                    continue





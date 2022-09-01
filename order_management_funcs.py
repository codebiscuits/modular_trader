import binance_funcs as funcs
import utility_funcs as uf
from datetime import datetime
from binance.exceptions import BinanceAPIException
from pushbullet import Pushbullet
from binance.client import Client
import keys
import binance.enums as be
from decimal import Decimal, getcontext
from pprint import pprint

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
client = Client(keys.bPkey, keys.bSkey)
ctx = getcontext()
ctx.prec = 12


# dispatch

def open_pos(session, agent, pair, size, stp, inval, sim_reason, direction):
    if agent.in_pos['real'] is None and not sim_reason:
        open_real(session, agent, pair, size, stp, inval, direction, 0)

    if agent.in_pos['sim'] is None and sim_reason:
        open_sim(session, agent, pair, stp, inval, sim_reason, direction)


def tp_pos(session, agent, pair, stp, inval, direction):
    if agent.in_pos.get('real_tp_sig'):
        tp_real_full(session, agent, pair, stp, inval, direction)

    if agent.in_pos.get('sim_tp_sig'):
        tp_sim(session, agent, pair, stp, direction)

    if agent.in_pos.get('tracked_tp_sig'):
        tp_tracked(session, agent, pair, direction)


def close_pos(session, agent, pair, direction):
    if agent.in_pos['real'] == direction:
        print('')
        close_real_full(session, agent, pair, direction)

    if agent.in_pos['sim'] == direction:
        close_sim(session, agent, pair, direction)

    if agent.in_pos['tracked'] == direction:
        close_tracked(session, agent, pair, direction)


def reduce_risk_M(session, agent):
    # create a list of open positions in profit and their open risk value
    if not (positions := [(p, float(r.get('or_R')), float(r.get('pnl_%'))) for p, r in agent.real_pos.items()
                          if r.get('or_R') and (float(r.get('or_R')) > 0) and r.get('pnl_%')]):
        return

    # sort the list so biggest open risk is first
    sorted_pos = sorted(positions, key=lambda x: x[1], reverse=True)
    # find the sum of all R values
    total_r = sum(float(x.get('or_R', 0)) for x in agent.real_pos.values())

    for pos in sorted_pos:
        asset, or_R, pnl_pct = pos
        if not (total_r > agent.total_r_limit and or_R > agent.indiv_r_limit and pnl_pct > 0.3):
            continue

        print(f'\n*** tor: {total_r:.1f}, reducing risk ***')
        pair = f"{asset}USDT"
        now = datetime.now().strftime('%d/%m/%y %H:%M')
        note = f"{agent.name} reduce risk {pair}, or: {or_R}R, pnl: {pnl_pct}%"
        print(now, note)
        try:
            direction = agent.open_trades[pair]['position']['direction']
            # insert placeholder record
            create_close_placeholder(session, agent, pair, direction)
            # clear stop
            cleared_size = close_clear_stop(session, agent, pair)

            if not cleared_size:
                print(
                    f'{agent} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
                cleared_size = set_size_from_free(session, agent, pair)

            # execute trade
            close_order = close_position(session, agent, pair, cleared_size, 'reduce_risk', direction)
            # repay loan
            repay_size = close_repay(session, agent, pair, close_order, direction)
            # update records
            open_to_tracked(session, agent, pair, close_order, direction)
            # update in-pos, real_pos, counts etc
            close_real_7(session, agent, pair, repay_size, direction)

            total_r -= or_R

        except BinanceAPIException as e:
            agent.record_trades(session, 'all')
            print(f'problem with reduce_risk order for {pair}')
            print(e)
            pb.push_note(now, f'exeption during {pair} reduce_risk order')
            continue


# real open

def create_record(session, agent, pair, size, stp, inval, direction):
    price = session.prices[pair]
    usdt_size: str = f"{size * price:.2f}"
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    placeholder = {'type': f'open_{direction}',
                   'state': 'real',
                   'pair': pair,
                   'signal_score': 'signal_score',
                   'base_size': size,
                   'quote_size': usdt_size,
                   'trig_price': price,
                   'stop_price': stp,
                   'inval': inval,
                   'timestamp': now,
                   'completed': None
                   }
    agent.open_trades[pair] = {}
    agent.open_trades[pair]['placeholder'] = placeholder
    agent.open_trades[pair]['position'] = {}
    agent.open_trades[pair]['position']['pair'] = pair
    agent.open_trades[pair]['position']['direction'] = direction

    note = f"{agent.name} real open {direction} {size:.5} {pair} ({usdt_size} usdt) @ {price}, stop @ {stp:.5}"
    print(now, note)


def omf_borrow(session, agent, pair, size, direction):
    if direction == 'long':
        price = session.prices[pair]
        borrow_size = f"{size * price:.2f}"
        funcs.borrow_asset_M('USDT', borrow_size, session.live)
        agent.open_trades[pair]['placeholder']['loan_asset'] = 'USDT'
    elif direction == 'short':
        asset = pair[:-4]
        borrow_size = funcs.valid_size(session, pair, size)
        funcs.borrow_asset_M(asset, borrow_size, session.live)
        agent.open_trades[pair]['placeholder']['loan_asset'] = asset
    else:
        print('*** WARNING open_real_2 given wrong direction argument ***')

    agent.open_trades[pair]['position']['liability'] = borrow_size
    agent.open_trades[pair]['placeholder']['liability'] = borrow_size
    agent.open_trades[pair]['placeholder']['completed'] = 'borrow'


def increase_position(session, agent, pair, size, direction):
    price = session.prices[pair]
    usdt_size = f"{size * price:.2f}"

    if direction == 'long':
        api_order = funcs.buy_asset_M(session, pair, float(usdt_size), False, price, session.live)
    elif direction == 'short':
        api_order = funcs.sell_asset_M(session, pair, size, price, session.live)

    agent.open_trades[pair]['position']['base_size'] = str(api_order.get('executedQty'))
    agent.open_trades[pair]['position']['open_time'] = api_order.get('transactTime')
    agent.open_trades[pair]['placeholder']['api_order'] = api_order
    agent.open_trades[pair]['placeholder']['completed'] = 'execute'

    return api_order


def open_trade_dict(session, agent, pair, api_order, stp, direction):
    price = session.prices[pair]

    open_order = funcs.create_trade_dict(api_order, price, session.live)
    open_order['type'] = f"open_{direction}"
    open_order['state'] = 'real'
    open_order['score'] = 'signal score'
    open_order['hard_stop'] = str(stp)

    agent.open_trades[pair]['position']['entry_price'] = open_order['exe_price']
    agent.open_trades[pair]['placeholder'].update(open_order)
    agent.open_trades[pair]['placeholder']['completed'] = 'trade_dict'

    return open_order


def open_set_stop(session, agent, pair, stp, open_order, direction):
    # stop_size = float(open_order.get('base_size'))
    stop_size = open_order.get('base_size')  # this is a string, go back to using above line if this causes bugs

    if direction == 'long':
        stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_SELL, stp, stp * 0.8)
    elif direction == 'short':
        stop_order = funcs.set_stop_M(session, pair, stop_size, be.SIDE_BUY, stp, stp * 1.2)

    open_order['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['position']['hard_stop'] = str(stp)
    agent.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['position']['stop_time'] = stop_order.get('transactTime')
    agent.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['placeholder']['stop_time'] = stop_order.get('transactTime')
    agent.open_trades[pair]['placeholder']['completed'] = 'set_stop'

    return open_order


def open_save_records(session, agent, pair, open_order):
    agent.open_trades[pair]['trade'] = [open_order]
    del agent.open_trades[pair]['placeholder']
    agent.record_trades(session, 'open')


def open_update_in_pos(session, agent, pair, stp, direction):
    price = session.prices[pair]

    agent.in_pos['real'] = direction
    if direction == 'long':
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_l
    elif direction == 'short':
        agent.in_pos['real_pfrd'] = agent.fixed_risk_dol_s
    agent.in_pos['real_ep'] = price
    agent.in_pos['real_hs'] = stp


def open_update_real_pos_usdtM_counts(session, agent, pair, size, inval, direction):
    price = session.prices[pair]
    usdt_size = f"{size * price:.2f}"
    asset = pair[:-4]

    if session.live:
        agent.real_pos[asset] = funcs.update_pos_M(session, asset, size, inval, agent.in_pos['real'],
                                                   agent.in_pos['real_pfrd'])
        agent.real_pos[asset]['pnl_R'] = 0
        if direction == 'long':
            session.update_usdt_M(borrow=float(usdt_size))
        elif direction == 'short':
            session.update_usdt_M(up=float(usdt_size))
    else:
        pf = f"{float(usdt_size) / session.bal:.2f}"
        if direction == 'long':
            or_dol = f"{session.bal * agent.fixed_risk_l:.2f}"
        elif direction == 'short':
            or_dol = f"{session.bal * agent.fixed_risk_s:.2f}"
        agent.real_pos[asset] = {'qty': str(size), 'value': usdt_size, 'pf%': pf, 'or_R': '1', 'or_$': str(or_dol)}

    agent.counts_dict[f'real_open_{direction}'] += 1


def open_real(session, agent, pair, size, stp, inval, direction, stage):
    if stage == 0:
        print('')
        create_record(session, agent, pair, size, stp, inval, direction)
        omf_borrow(session, agent, pair, size, direction)
        api_order = increase_position(session, agent, pair, size, direction)
    if stage <= 1:
        open_order = open_trade_dict(session, agent, pair, api_order, stp, direction)
    if stage <= 2:
        open_order = open_set_stop(session, agent, pair, stp, open_order, direction)
    if stage <= 3:
        open_save_records(session, agent, pair, open_order)
        open_update_in_pos(session, agent, pair, stp, direction)
        open_update_real_pos_usdtM_counts(session, agent, pair, size, inval, direction)


# real tp

def create_tp_placeholder(session, agent, pair, stp, inval, direction):
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    # insert placeholder record
    placeholder = {'type': f'tp_{direction}',
                   'state': 'real',
                   'pair': pair,
                   'trig_price': price,
                   'stop_price': stp,
                   'inval': inval,
                   'timestamp': now,
                   'completed': None
                   }
    agent.open_trades[pair]['placeholder'] = placeholder


def tp_set_pct(agent, pair):
    asset = pair[:-4]

    real_val = abs(Decimal(agent.real_pos[asset]['value']))
    pct = 50 if real_val > 24 else 100

    return pct


def tp_clear_stop(session, agent, pair):
    print(f'{agent.name} clearing {pair} stop')
    clear, cleared_size = funcs.clear_stop_M(pair, agent.open_trades[pair]['position'], session.live)
    real_bal = Decimal(agent.open_trades[pair]['position']['base_size'])
    check_size_against_records(agent, pair, real_bal, cleared_size)

    # update position and placeholder
    agent.open_trades[pair]['position']['hard_stop'] = None
    agent.open_trades[pair]['position']['stop_id'] = None
    agent.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
    agent.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

    return cleared_size


def tp_reduce_position(session, agent, pair, base_size, pct, direction):
    price = session.prices[pair]
    order_size = float(base_size) * (pct / 100)
    if direction == 'long':
        api_order = funcs.sell_asset_M(session, pair, order_size, price, session.live)
    elif direction == 'short':
        api_order = funcs.buy_asset_M(session, pair, order_size, True, price, session.live)

    # update records
    agent.open_trades[pair]['placeholder']['api_order'] = api_order
    curr_base_size = agent.open_trades[pair]['position']['base_size']
    new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
    agent.open_trades[pair]['position']['base_size'] = str(new_base_size)
    print(f"+++ {agent} {pair} tp {direction} resulted in base qty: {new_base_size}")
    tp_order = funcs.create_trade_dict(api_order, price, session.live)

    agent.open_trades[pair]['placeholder']['tp_order'] = tp_order
    agent.open_trades[pair]['placeholder']['pct'] = pct
    agent.open_trades[pair]['placeholder']['order_size'] = order_size
    agent.open_trades[pair]['placeholder']['completed'] = 'execute'

    return tp_order


def tp_repay_100(session, agent, pair, tp_order, direction):
    price = session.prices[pair]
    asset = pair[:-4]
    liability = agent.open_trades[pair]['position']['liability']
    if direction == 'long':
        repay_usdt = str(max(Decimal(tp_order.get('quote_size', 0)), Decimal(liability)))
        funcs.repay_asset_M('USDT', repay_usdt, session.live)
    else:
        repay_asset = str(max(Decimal(liability), Decimal(tp_order.get('base_size', 0))))
        repay_usdt = f"{float(repay_asset) * price:.2f}"
        funcs.repay_asset_M(asset, repay_asset, session.live)

    # update trade dict
    tp_order['type'] = f'close_{direction}'
    tp_order['state'] = 'real'
    tp_order['reason'] = 'trade over-extended'
    agent.open_trades[pair]['position']['liability'] = '0'
    agent.open_trades[pair]['placeholder'].update(tp_order)
    agent.open_trades[pair]['placeholder']['repay_usdt']
    agent.open_trades[pair]['placeholder']['completed'] = 'repay_100'

    return tp_order, repay_usdt


def open_to_tracked(session, agent, pair, close_order, direction):
    agent.open_trades[pair]['trade'].append(close_order)

    agent.realised_pnl(agent.open_trades[pair], direction)
    del agent.open_trades[pair]['placeholder']
    agent.tracked_trades[pair] = agent.open_trades[pair]
    agent.record_trades(session, 'tracked')

    del agent.open_trades[pair]
    agent.record_trades(session, 'open')


def tp_update_records_100(session, agent, pair, order_size, usdt_size, direction):
    asset = pair[:-4]
    price = session.prices[pair]

    agent.in_pos['real'] = None
    agent.in_pos['tracked'] = direction

    agent.tracked[asset] = {'qty': '0', 'value': '0', 'pf%': '0', 'or_R': '0', 'or_$': '0'}

    if session.live and direction == 'long':
        session.update_usdt_M(repay=float(usdt_size))
    elif session.live and direction == 'short':
        usdt_size = round(order_size * price, 5)
        session.update_usdt_M(down=usdt_size)
    elif (not session.live) and direction == 'long':
        agent.real_pos['USDT']['qty'] += float(agent.real_pos[asset].get('value'))
        agent.real_pos['USDT']['value'] += float(agent.real_pos[asset].get('value'))
        agent.real_pos['USDT']['pf%'] += float(agent.real_pos[asset].get('pf%'))
    elif (not session.live) and direction == 'short':
        agent.real_pos['USDT']['qty'] -= float(agent.real_pos[asset].get('value'))
        agent.real_pos['USDT']['value'] -= float(agent.real_pos[asset].get('value'))
        agent.real_pos['USDT']['pf%'] -= float(agent.real_pos[asset].get('pf%'))

    del agent.real_pos[asset]
    agent.counts_dict[f'real_close_{direction}'] += 1


def tp_repay_partial(session, agent, pair, stp, tp_order, direction):
    asset = pair[:-4]

    if direction == 'long':
        repay_size = tp_order.get('quote_size')
        funcs.repay_asset_M('USDT', repay_size, session.live)
    elif direction == 'short':
        repay_size = tp_order.get('base_size')
        funcs.repay_asset_M(asset, repay_size, session.live)

    # create trade dict
    tp_order['type'] = f'tp_{direction}'
    tp_order['state'] = 'real'
    tp_order['hard_stop'] = str(stp)
    tp_order['reason'] = 'trade over-extended'

    agent.open_trades[pair]['position']['liability'] = uf.update_liability(agent.open_trades[pair], repay_size, 'reduce')
    agent.open_trades[pair]['placeholder'].update(tp_order)
    agent.open_trades[pair]['placeholder']['tp_order'] = tp_order
    agent.open_trades[pair]['placeholder']['completed'] = 'repay_part'

    return tp_order


def tp_reset_stop(session, agent, pair, stp, tp_order, direction):
    new_size = agent.open_trades[pair]['position']['base_size']

    if direction == 'long':
        stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_SELL, stp, stp * 0.8)
    elif direction == 'short':
        stop_order = funcs.set_stop_M(session, pair, new_size, be.SIDE_BUY, stp, stp * 1.2)

    tp_order['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['position']['hard_stop'] = str(stp)
    agent.open_trades[pair]['position']['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['position']['stop_time'] = stop_order.get('transactTime')
    agent.open_trades[pair]['placeholder']['hard_stop'] = str(stp)
    agent.open_trades[pair]['placeholder']['stop_id'] = stop_order.get('orderId')
    agent.open_trades[pair]['placeholder']['stop_time'] = stop_order.get('transactTime')
    agent.open_trades[pair]['placeholder']['completed'] = 'set_stop'

    return tp_order


def open_to_open(session, agent, pair, tp_order, direction):
    agent.open_trades[pair]['trade'].append(tp_order)
    agent.realised_pnl(agent.open_trades[pair], direction)
    print('\ntp partial placeholder')
    pprint(agent.open_trades[pair]['placeholder'])
    del agent.open_trades[pair]['placeholder']
    agent.record_trades(session, 'open')


def tp_update_records_partial(session, agent, pair, pct, inval, order_size, tp_order, direction):
    asset = pair[:-4]
    price = session.prices[pair]
    new_size = agent.open_trades[pair]['position']['base_size']

    agent.in_pos['real_pfrd'] = agent.in_pos['real_pfrd'] * (pct / 100)
    if session.live:
        agent.real_pos[asset].update(
            funcs.update_pos_M(session, asset, new_size, inval, agent.in_pos['real'],
                               agent.in_pos['real_pfrd']))
        if direction == 'long':
            repay_size = tp_order.get('base_size')
            session.update_usdt_M(repay=float(repay_size))
        elif direction == 'short':
            usdt_size = round(order_size * price, 5)
            session.update_usdt_M(down=usdt_size)
    else:
        uf.calc_sizing_non_live_tp(session, agent, asset, pct, 'real')

    agent.counts_dict[f'real_tp_{direction}'] += 1


def tp_real_full(session, agent, pair, stp, inval, direction):
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')
    print('')

    create_tp_placeholder(session, agent, pair, stp, inval, direction)
    pct = tp_set_pct(agent, pair)
    # clear stop
    cleared_size = tp_clear_stop(session, agent, pair)

    if not cleared_size:
        print(f'{agent} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
        cleared_size = set_size_from_free(session, agent, pair)

    # execute trade
    tp_order = tp_reduce_position(session, agent, pair, cleared_size, pct, direction)

    note = f"{agent.name} real take-profit {pair} {direction} {pct}% @ {price}"
    print(now, note)

    if pct == 100:
        # repay assets
        tp_order, usdt_size = tp_repay_100(session, agent, pair, tp_order, direction)
        # update records
        open_to_tracked(session, agent, pair, tp_order, direction)
        tp_update_records_100(session, agent, pair, cleared_size, usdt_size, direction)

    else:  # if pct < 100%
        # repay assets
        tp_order = tp_repay_partial(session, agent, pair, stp, tp_order, direction)
        # set new stop
        tp_order = tp_reset_stop(session, agent, pair, stp, tp_order, direction)
        # update records
        open_to_open(session, agent, pair, tp_order, direction)
        tp_update_records_partial(session, agent, pair, pct, inval, cleared_size, tp_order, direction)


# real close

def create_close_placeholder(session, agent, pair, direction):
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    # temporary check to catch a possible bug, can delete after ive had a few reduce_risk calls with no bugs
    if direction not in ['long', 'short']:
        print(
            f'*** WARNING, string "{direction}" being passed to create_close_placeholder, either from close omf or reduce_risk')

    # insert placeholder record
    placeholder = {'type': f'close_{direction}',
                   'state': 'real',
                   'pair': pair,
                   'trig_price': price,
                   'timestamp': now,
                   'completed': None
                   }
    agent.open_trades[pair]['placeholder'] = placeholder


def close_clear_stop(session, agent, pair):
    print(f'{agent.name} clearing {pair} stop')
    clear, cleared_size = funcs.clear_stop_M(pair, agent.open_trades[pair]['position'], session.live)
    real_bal = Decimal(agent.open_trades[pair]['position']['base_size'])
    check_size_against_records(agent, pair, real_bal, cleared_size)

    # update position and placeholder
    agent.open_trades[pair]['position']['hard_stop'] = None
    agent.open_trades[pair]['position']['stop_id'] = None
    agent.open_trades[pair]['placeholder']['cleared_size'] = cleared_size
    agent.open_trades[pair]['placeholder']['completed'] = 'clear_stop'

    return cleared_size


def close_position(session, agent, pair, close_size, reason, direction):
    price = session.prices[pair]

    if direction == 'long':
        api_order = funcs.sell_asset_M(session, pair, close_size, price, session.live)
    elif direction == 'short':
        api_order = funcs.buy_asset_M(session, pair, close_size, True, price, session.live)

    # update position and placeholder
    agent.open_trades[pair]['placeholder']['api_order'] = api_order
    curr_base_size = agent.open_trades[pair]['position']['base_size']
    new_base_size = Decimal(curr_base_size) - Decimal(api_order.get('executedQty'))
    agent.open_trades[pair]['position']['base_size'] = str(new_base_size)
    print(f"+++ {agent} {pair} close {direction} resulted in base qty: {new_base_size}")
    close_order = funcs.create_trade_dict(api_order, price, session.live)

    close_order['type'] = f'close_{direction}'
    close_order['state'] = 'real'
    close_order['reason'] = reason
    agent.open_trades[pair]['placeholder'].update(close_order)
    agent.open_trades[pair]['placeholder']['completed'] = 'execute'

    return close_order


def close_repay(session, agent, pair, close_order, direction):
    asset = pair[:-4]
    liability = agent.open_trades[pair]['position']['liability']

    if direction == 'long':
        repay_size = str(max(Decimal(close_order.get('quote_size', 0)), Decimal(liability)))
        funcs.repay_asset_M('USDT', repay_size, session.live)
    elif direction == 'short':
        repay_size = str(max(Decimal(liability), Decimal(close_order.get('base_size', 0))))
        funcs.repay_asset_M(asset, repay_size, session.live)

    # update records
    agent.open_trades[pair]['position']['liability'] = '0'
    agent.open_trades[pair]['placeholder']['completed'] = 'repay'

    return repay_size


def open_to_closed(session, agent, pair, close_order, direction):
    agent.open_trades[pair]['trade'].append(close_order)
    agent.realised_pnl(agent.open_trades[pair], direction)

    trade_id = int(agent.open_trades[pair]['position']['open_time'])
    agent.closed_trades[trade_id] = {}
    agent.closed_trades[trade_id]['trade'] = agent.open_trades[pair]['trade']
    agent.record_trades(session, 'closed')

    del agent.open_trades[pair]
    agent.record_trades(session, 'open')


def close_real_7(session, agent, pair, close_size, direction):
    asset = pair[:-4]
    price = session.prices[pair]

    agent.in_pos['real'] = None
    agent.in_pos['real_pfrd'] = 0

    if direction == 'long' and session.live:
        session.update_usdt_M(repay=float(close_size))
    elif direction == 'short' and session.live:
        usdt_size = round(float(close_size) * price, 5)
        session.update_usdt_M(down=usdt_size)
    elif direction == 'long' and not session.live:
        agent.real_pos['USDT']['value'] += float(close_size)
        agent.real_pos['USDT']['owed'] -= float(close_size)
    elif direction == 'short' and not session.live:
        agent.real_pos['USDT']['value'] += (float(close_size) * price)
        agent.real_pos['USDT']['owed'] -= (float(close_size) * price)

    # save records and update counts
    del agent.real_pos[asset]
    agent.counts_dict[f'real_close_{direction}'] += 1


def close_real_full(session, agent, pair, direction, stage=0):
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    note = f"{agent.name} real close {direction} {pair} @ {price}"
    print(now, note)

    if stage == 0:
        del agent.open_trades[pair]['placeholder']
        create_close_placeholder(session, agent, pair, direction)
    if stage <= 1:
        # cancel stop
        cleared_size = close_clear_stop(session, agent, pair)

        if not cleared_size:
            print(f'{agent} {pair} clear_stop returned base_size 0, checking exchange bals before closing {direction}')
            cleared_size = set_size_from_free(session, agent, pair)
    if stage <= 2:
        # execute trade
        close_order = close_position(session, agent, pair, cleared_size, 'close_signal', direction)
    if stage <= 3:
        # repay loan
        repay_size = close_repay(session, agent, pair, close_order, direction)
    if stage <= 4:
        # update records
        open_to_closed(session, agent, pair, close_order, direction)
        # update in-pos, real_pos, counts etc
        close_real_7(session, agent, pair, repay_size, direction)


# sim
def open_sim(session, agent, pair, stp, inval, sim_reason, direction):
    asset = pair[:-4]
    price = session.prices[pair]

    usdt_size = 128.0
    size = f"{usdt_size / price:.8f}"
    # if not session.live:
    #     now = datetime.now().strftime('%d/%m/%y %H:%M')
    #     print('')
    #     note = f"{agent.name} sim open {direction} {size:.5} {pair} ({usdt_size:.5} usdt) @ {price}, stop @ {stp:.5}"
    #     print(now, note)

    timestamp = round(datetime.utcnow().timestamp() * 1000)
    sim_order = {'pair': pair,
                 'exe_price': str(price),
                 'trig_price': str(price),
                 'base_size': size,
                 'quote_size': '128.0',
                 'hard_stop': str(stp),
                 'reason': sim_reason,
                 'timestamp': timestamp,
                 'type': f'open_{direction}',
                 'fee': '0',
                 'fee_currency': 'BNB',
                 'state': 'sim'}
    trade_record = [sim_order]
    agent.sim_trades[pair] = {}
    agent.sim_trades[pair]['trade'] = trade_record

    pos_record = {'base_size': size,
                  'direction': direction,
                  'entry_price': str(price),
                  'hard_stop': str(stp),
                  'open_time': timestamp,
                  'pair': pair,
                  'liability': '0',
                  'stop_id': 'not live',
                  'stop_time': None}
    agent.sim_trades[pair]['position'] = pos_record

    agent.record_trades(session, 'sim')

    agent.in_pos['sim'] = direction
    if direction == 'long':
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_l
    else:
        agent.in_pos['sim_pfrd'] = agent.fixed_risk_dol_s
    agent.sim_pos[asset] = funcs.update_pos_M(session, asset, float(size), inval,
                                              agent.in_pos['sim'], agent.in_pos['sim_pfrd'])
    agent.sim_pos[asset]['pnl_R'] = 0

    agent.counts_dict[f'sim_open_{direction}'] += 1


def tp_sim(session, agent, pair, stp, direction):
    # if not session.live:
    #     print('')
    #     note = f"{agent.name} sim take-profit {pair} {direction} 50% @ {price}"
    #     print(now, note)

    price = session.prices[pair]
    asset = pair[:-4]

    trade_record = agent.sim_trades[pair]['trade']
    sim_bal = float(agent.sim_trades[pair]['position']['base_size'])
    timestamp = round(datetime.utcnow().timestamp() * 1000)
    order_size = sim_bal / 2
    usdt_size = f"{order_size * price:.2f}"

    # execute order
    tp_order = {'pair': pair,
                'exe_price': str(price),
                'trig_price': str(price),
                'base_size': str(order_size),
                'quote_size': usdt_size,
                'hard_stop': str(stp),
                'reason': 'trade over-extended',
                'timestamp': timestamp,
                'type': f'tp_{direction}',
                'fee': '0',
                'fee_currency': 'BNB',
                'state': 'sim'}
    trade_record.append(tp_order)
    agent.sim_trades[pair]['trade'] = trade_record

    agent.sim_trades[pair]['position']['base_size'] = str(order_size)
    agent.sim_trades[pair]['position']['hard_stop'] = str(stp)
    agent.sim_trades[pair]['position']['stop_time'] = timestamp

    # save records
    agent.record_trades(session, 'sim')

    # update sim_pos
    uf.calc_sizing_non_live_tp(session, agent, asset, 50, 'sim')

    agent.counts_dict[f'sim_tp_{direction}'] += 1
    agent.in_pos['sim_pfrd'] = agent.in_pos['sim_pfrd'] / 2
    agent.realised_pnl(agent.sim_trades[pair], direction)


def sim_to_sim_closed(session, agent, pair, close_order, direction):
    agent.sim_trades[pair]['trade'].append(close_order)
    agent.realised_pnl(agent.sim_trades[pair], direction)

    trade_id = int(agent.sim_trades[pair]['position']['open_time'])
    agent.closed_sim_trades[trade_id] = {}
    agent.closed_sim_trades[trade_id]['trade'] = agent.sim_trades[pair]['trade']
    agent.record_trades(session, 'closed_sim')

    del agent.sim_trades[pair]
    agent.record_trades(session, 'sim')


def close_sim(session, agent, pair, direction):
    # if not session.live:
    #     print('')
    #     note = f"{agent.name} sim close {direction} {pair} @ {price}"
    #     print(now, note)

    price = session.prices[pair]
    asset = pair[:-4]

    # initialise stuff
    sim_bal = float(agent.sim_trades[pair]['position']['base_size'])
    timestamp = round(datetime.utcnow().timestamp() * 1000)

    # execute order
    close_order = {'pair': pair,
                   'exe_price': str(price),
                   'trig_price': str(price),
                   'base_size': str(sim_bal),
                   'quote_size': f"{sim_bal * price:.2f}",
                   'reason': 'close_signal',
                   'timestamp': timestamp,
                   'type': f'close_{direction}',
                   'fee': '0',
                   'fee_currency': 'BNB',
                   'state': 'sim'}

    agent.sim_trades[pair]['position']['base_size'] = '0'
    agent.sim_trades[pair]['position']['hard_stop'] = None

    # update records
    sim_to_sim_closed(session, agent, pair, close_order, direction)

    # update counts and live variables
    agent.in_pos['sim'] = None
    agent.in_pos['sim_pfrd'] = 0
    del agent.sim_pos[asset]

    agent.counts_dict[f'sim_close_{direction}'] += 1


# tracked

def tp_tracked(session, agent, pair, stp, direction):
    print('')
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    note = f"{agent.name} tracked take-profit {pair} {direction} 50% @ {price}"
    print(now, note)

    trade_record = agent.tracked_trades[pair]['trade']
    timestamp = round(datetime.utcnow().timestamp() * 1000)

    # execute order
    tp_order = {'pair': pair,
                'exe_price': str(price),
                'trig_price': str(price),
                'base_size': '0',
                'quote_size': '0',
                'reason': 'trade over-extended',
                'timestamp': timestamp,
                'type': f'tp_{direction}',
                'fee': '0',
                'fee_currency': 'BNB',
                'state': 'tracked'}
    trade_record.append(tp_order)

    agent.tracked_trades[pair]['position']['hard_stop'] = str(stp)
    agent.tracked_trades[pair]['position']['stop_time'] = timestamp

    # update records
    agent.tracked_trades[pair]['trade'] = trade_record
    agent.record_trades(session, 'tracked')

    agent.in_pos['tracked_pfrd'] = agent.in_pos['tracked_pfrd'] / 2


def tracked_to_closed(session, agent, pair, close_order):
    agent.tracked_trades[pair]['trade'].append(close_order)

    trade_id = int(agent.tracked_trades[pair]['position']['open_time'])
    agent.closed_trades[trade_id]['trade'] = agent.tracked_trades[pair]['trade']
    agent.record_trades(session, 'closed')

    del agent.tracked_trades[pair]
    agent.record_trades(session, 'tracked')


def close_tracked(session, agent, pair, direction):
    print('')
    asset = pair[:-4]
    price = session.prices[pair]
    now = datetime.now().strftime('%d/%m/%y %H:%M')

    note = f"{agent.name} tracked close {direction} {pair} @ {price}"
    print(now, note)

    # initialise stuff
    timestamp = round(datetime.utcnow().timestamp() * 1000)

    # execute order
    close_order = {'pair': pair,
                  'exe_price': str(price),
                  'trig_price': str(price),
                  'base_size': '0',
                  'quote_size': '0',
                  'reason': 'close_signal',
                  'timestamp': timestamp,
                  'type': f'close_{direction}',
                  'fee': '0',
                  'fee_currency': 'BNB',
                  'state': 'tracked'}

    # update records
    tracked_to_closed(session, agent, pair, close_order)

    # update counts and live variables
    del agent.tracked[asset]
    agent.in_pos['tracked'] = None
    agent.in_pos['tracked_pfrd'] = 0


# other

def check_size_against_records(agent, pair, real_bal, base_size):
    base_size = Decimal(base_size)
    if base_size and (real_bal != base_size):  # check records match reality
        print(f"{agent.name} {pair} records don't match real balance. {real_bal = }, {base_size = }")
        mismatch = 100 * abs(base_size - real_bal) / base_size
        print(f"{mismatch = }%")


def set_size_from_free(session, agent, pair):
    """if clear_stop returns a base size of 0, this can be called to check for free balance,
    in case the position was there but just not in a stop order"""
    asset = pair[:-4]
    real_bal = Decimal(agent.open_trades[pair]['position']['base_size'])

    session.margin_account_info()
    session.get_asset_bals()
    free_bal = session.bals_dict[asset]['free']

    return min(free_bal, real_bal)

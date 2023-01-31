import time
from datetime import datetime
import binance_funcs as funcs
from agents import DoubleST, EMACross, EMACrossHMA, AvgTradeSize
import binance.exceptions as bx
from config import not_pairs
from pprint import pprint
import utility_funcs as uf
from random import shuffle
import sessions
from timers import Timer
from pushbullet import Pushbullet
from collections import Counter

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')


def setup_scan(timeframe: str, offset: str) -> None:
    """this function scans all pairs for a particular quote asset looking for setups
    which match the criteria for a trade, then executes the trade if possible"""

    print(f"Running setup_scan({timeframe}, {offset})")
    all_start = time.perf_counter()
    session = sessions.MARGIN_SESSION(timeframe, offset, 0.0005)
    print(f"\nCurrent time: {session.now_start}, {session.name}\n")
    funcs.update_prices(session)

    agents = [
        DoubleST(session, 3, 1.0),
        DoubleST(session, 3, 1.4),
        DoubleST(session, 3, 1.8),
        DoubleST(session, 5, 2.2),
        DoubleST(session, 5, 2.8),
        DoubleST(session, 5, 3.4),
        EMACross(session, 12, 21, 1.2),
        EMACross(session, 12, 21, 1.8),
        EMACross(session, 12, 21, 2.4),
        EMACrossHMA(session, 12, 21, 1.2),
        EMACrossHMA(session, 12, 21, 1.8),
        EMACrossHMA(session, 12, 21, 2.4),
        # AvgTradeSize(session, 2, 1000, 1.1, 'oco'),
        # AvgTradeSize(session, 2, 1000, 1.1, 'trail'),
        # AvgTradeSize(session, 2, 1000, 2.0, 'oco'),
        # AvgTradeSize(session, 2, 1000, 2.0, 'trail'),
        # AvgTradeSize(session, 2, 1000, 3.0, 'oco'),
        # AvgTradeSize(session, 2, 1000, 3.0, 'trail'),
        # AvgTradeSize(session, 2, 1000, 4.0, 'oco'),
        # AvgTradeSize(session, 2, 1000, 4.0, 'trail'),
    ]

    session.name = ' | '.join([n.name for n in agents])

    # compile and sort list of pairs to loop through ------------------------------
    # TODO need to work out what to do when some strats want the spot pairs list and others just want the cross pairs
    all_pairs = funcs.get_pairs(market='CROSS')
    shuffle(all_pairs)
    positions = []
    for agent in agents:
        posis = list(agent.real_pos.keys())
        positions.extend(posis)
    pairs_in_pos = [p + 'USDT' for p in set(positions) if p != 'USDT']
    print(f"Total {pairs_in_pos = }")
    other_pairs = [p for p in all_pairs if (p not in pairs_in_pos) and (p not in not_pairs)]
    pairs = pairs_in_pos + other_pairs  # this ensures open positions will be checked first

    # pairs = pairs[:10] # for testing the loop quickly

    funcs.top_up_bnb_M(session, 15)
    session.spreads = funcs.binance_spreads('USDT')  # dict

    for agent in agents:
        if agent.fixed_risk_l or agent.fixed_risk_s:
            print(f"{agent.name} fr long: {(agent.fixed_risk_l * 10000):.2f}bps, \
fr short: {(agent.fixed_risk_s * 10000):.2f}bps")
        agent.real_pos['USDT'] = session.usdt_bal
        agent.starting_ropnl_l = agent.open_pnl('long', 'real')
        agent.starting_sopnl_l = agent.open_pnl('long', 'sim')
        agent.starting_ropnl_s = agent.open_pnl('short', 'real')
        agent.starting_sopnl_s = agent.open_pnl('short', 'sim')

        if agent.max_positions > 10:
            print(f'max positions: {agent.max_positions}')
        agent.calc_tor()

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    for n, pair in enumerate(pairs):
        # print(pair)
        funcs.update_prices(session)
        asset = pair[:-1 * len(session.quote_asset)]
        for agent in agents:
            agent.init_in_pos(pair)

        df = funcs.prepare_ohlc(session, pair)

        # if pair is too new for all agents, skip it
        too_new = 0
        for agent in agents:
            if agent.too_new(df):
                too_new += 1
        if too_new == len(agents):
            print(f"{pair} too new: {len(df)}")
            continue

        if len(df) > session.max_length:
            df = df.tail(session.max_length)
            df.reset_index(drop=True, inplace=True)

        now = datetime.now().strftime('%d/%m/%y %H:%M')

        # generate signals ------------------------------------------------------------
        df = session.compute_indicators(df)

        signals = {}
        usdt_depth_l, usdt_depth_s = funcs.get_depth(session, pair)
        price = session.prices[pair]
        for agent in agents:
            # print('*****', agent.name)
            df_2 = df.copy()
            signals = agent.signals(session, df_2, pair)

            inval = signals.get('inval')
            inval_ratio = signals.get('inval_ratio')
            if inval:
                stp = funcs.calc_stop(inval, session.spreads.get(pair), price)
                risk = abs((price - stp) / price)
                size_l, usdt_size_l, size_s, usdt_size_s = funcs.get_size(agent, price, session.bal, risk)

            # remove indicators to avoid errors
            df_2 = df_2[['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol',
                         'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol']]

            if inval == 0:
                note = f'{pair} supertrend 0 error, skipping pair'
                print(note)
                pb.push_note(now, note)
                continue

            # update positions dictionary with current pair's open_risk values ------------
            if agent.in_pos['real']:
                real_qty = float(agent.open_trades[pair]['position']['base_size'])
                agent.real_pos[asset].update(
                    agent.update_pos(session, asset, real_qty, inval_ratio, 'real'))

                real_ep = float(agent.open_trades[pair]['position']['entry_price'])
                if real_ep:
                    agent.real_pos[asset]['price_delta'] = (price - real_ep) / real_ep  # how much has price moved since entry

                # check if price has moved beyond reach of normal close signal
                if agent.real_pos[asset]['or_R'] < 0:
                    signals['signal'] = f"close_{agent.in_pos['real']}"

            if agent.in_pos['sim']:
                sim_qty = float(agent.sim_trades[pair]['position']['base_size'])
                agent.sim_pos[asset].update(
                    agent.update_pos(session, asset, sim_qty, inval_ratio, 'sim'))
                sim_ep = float(agent.sim_trades[pair]['position']['entry_price'])
                if sim_ep:
                    agent.sim_pos[asset]['price_delta'] = (price - sim_ep) / sim_ep

                # check if price has moved beyond reach of normal close signal
                if agent.sim_pos[asset]['or_R'] < 0:
                    signals['signal'] = f"close_{agent.in_pos['sim']}"

            # margin order execution ------------------------------------------------------
            agent.max_init_r_l = agent.max_init_risk(agent.fixed_risk_l)
            agent.max_init_r_s = agent.max_init_risk(agent.fixed_risk_s)

            if signals.get('signal') in ['open_spot', 'tp_spot', 'close_spot']:
                usdt_size, usdt_depth, size = usdt_size_l, usdt_depth_l, size_l
                direction = 'spot'
            elif signals.get('signal') in ['open_long', 'tp_long', 'close_long']:
                usdt_size, usdt_depth, size = usdt_size_l, usdt_depth_l, size_l
                direction = 'long'
            elif signals.get('signal') in ['open_short', 'tp_short', 'close_short']:
                usdt_size, usdt_depth, size = usdt_size_s, usdt_depth_s, size_s
                direction = 'short'

            if signals.get('signal') in ['close_spot', 'close_long', 'close_short']:
                try:
                    agent.close_pos(session, pair, direction)
                except bx.BinanceAPIException as e:
                    agent.record_trades(session, 'all')
                    print(f'{agent.name} problem with close_{direction} order for {pair}')
                    print(e.code)
                    print(e.message)
                    pb.push_note(now, f'{agent.name} exeption during {pair} close_{direction} order')
                    continue

            elif signals.get('signal') in ['open_spot', 'open_long', 'open_short']:
                if usdt_size > usdt_depth > (usdt_size / 2):  # only trim size if books are a bit too thin
                    agent.counts_dict['books_too_thin'] += 1
                    trim_size = f'{now} {pair} books too thin, reducing size from {usdt_size:.3} to {usdt_depth:.3}'
                    print(trim_size)
                    usdt_size = usdt_depth
                sim_reason = None

                if usdt_size < 30:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['too_small'] += 1
                    sim_reason = 'too_small'

                elif ((('long' in signals.get('signal')) and (risk > agent.max_init_r_l))
                      or
                      (('short' in signals.get('signal')) and (risk > agent.max_init_r_s))):
                    if not agent.in_pos['sim']:
                        agent.counts_dict['too_risky'] += 1
                    sim_reason = 'too_risky'

                elif usdt_depth == 0:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['too_much_spread'] += 1
                    sim_reason = 'too_much_spread'

                elif usdt_depth < usdt_size:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['books_too_thin'] += 1
                    sim_reason = 'books_too_thin'

                # check total open risk and close profitable positions if necessary -----------
                agent.reduce_risk_M(session)
                agent.real_pos['USDT'] = session.usdt_bal

                # make sure there aren't too many open positions now --------------------------
                agent.calc_tor()
                if agent.num_open_positions >= agent.max_positions:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['too_many_pos'] += 1
                    sim_reason = 'too_many_pos'
                elif agent.total_open_risk > agent.total_r_limit:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['too_much_or'] += 1
                    sim_reason = 'too_much_or'
                elif float(agent.real_pos['USDT']['qty']) < usdt_size:
                    if not agent.in_pos['sim']:
                        agent.counts_dict['not_enough_usdt'] += 1
                    sim_reason = 'not_enough_usdt'
                elif session.algo_limit_reached(pair):
                    agent.counts_dict['algo_order_limit'] += 1
                    sim_reason = 'algo_order_limit'

                try:
                    agent.open_pos(session, pair, size, stp, inval_ratio, sim_reason, direction)
                except bx.BinanceAPIException as e:
                    if e.code == -3045: # borrow failed because there weren't enough assets to borrow
                        del agent.open_trades[pair]
                        agent.open_pos(session, pair, size, stp, inval_ratio, 'not_enough_borrow', direction)
                    else:
                        agent.record_trades(session, 'all')
                        print(f'{agent.name} problem with open_{direction} order for {pair}')
                        print(e.code)
                        print(e.message)
                        print(f"{size = } {stp = } {inval_ratio = }")
                        pb.push_note(now, f'{agent.name} exeption during {pair} open_{direction} order')
                    continue

            elif signals.get('signal') in ['tp_long', 'tp_short']:
                try:
                    agent.tp_pos(session, pair, stp, inval_ratio, direction)
                except bx.BinanceAPIException as e:
                    agent.record_trades(session, 'all')
                    print(f'{agent.name} problem with tp_{direction} order for {pair}')
                    print(e.code)
                    print(e.message)
                    print(f"{stp = } {inval_ratio = }")
                    pb.push_note(now, f'{agent.name} exception during {pair} tp_{direction} order')
                    continue

            # calculate open risk and take profit if necessary ----------------------------
            agent.tp_signals(asset)
            if agent.in_pos['real']:
                direction = agent.in_pos['real']
                try:
                    agent.tp_pos(session, pair, stp, inval_ratio, direction)
                except bx.BinanceAPIException as e:
                    agent.record_trades(session, 'all')
                    print(f'{agent.name} problem with tp_{direction} order for {pair}')
                    print(e.code)
                    print(e.message)
                    pb.push_note(now, f'{agent.name} exeption during {pair} tp_{direction} order')
                    continue
            if agent.in_pos['sim']:
                direction = agent.in_pos['sim']
                try:
                    agent.tp_pos(session, pair, stp, inval_ratio, direction)
                except bx.BinanceAPIException as e:
                    agent.record_trades(session, 'all')
                    print(f'problem with tp_{direction} order for {pair}')
                    print(e.code)
                    print(e.message)
                    pb.push_note(now, f'{agent.name} exeption during {pair} tp_{direction} order')
                    continue

            agent.calc_tor()

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    # log all data from the session and print/push summary-------------------------
    before = session.usdt_bal
    session.get_usdt_M()
    after = session.usdt_bal
    if before != after:
        print('\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
        print(f'USDT balance wrong\nbefore: {before}\nafter: {after}')
        print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

    print('-:-' * 20)

    for agent in agents:
        if not session.live:
            print('')
            print(agent.name.upper(), 'SUMMARY')
            if agent.realised_pnl_long or agent.sim_pnl_long:
                print(
                    f'realised real long pnl: {agent.realised_pnl_long:.1f}R, realised sim long pnl: {agent.sim_pnl_long:.1f}R')
            if agent.realised_pnl_short or agent.sim_pnl_short:
                print(
                    f'realised real short pnl: {agent.realised_pnl_short:.1f}R, realised sim short pnl: {agent.sim_pnl_short:.1f}R')
            print(f'tor: {agent.total_open_risk:.1f}')
            # print(f'or list: {[round(x, 2) for x in sorted(agent.or_list, reverse=True)]}')
            lropnl = agent.open_pnl('long', 'real')
            if lropnl:
                print(f"real open pnl long: {lropnl:.1f}R")
            sropnl = agent.open_pnl('short', 'real')
            if sropnl:
                print(f"real open pnl short: {sropnl:.1f}R")
            lsopnl = agent.open_pnl('long', 'sim')
            if lsopnl:
                print(f"sim open pnl long: {lsopnl:.1f}R")
            ssopnl = agent.open_pnl('short', 'sim')
            if ssopnl:
                print(f"sim open pnl short: {ssopnl:.1f}R")

        agent.real_pos['USDT'] = session.usdt_bal

        print(f'{agent.name} Counts:')
        for k, v in agent.counts_dict.items():
            if v:
                print(k, v)
        print('-:-' * 20)

    if not session.live:
        # print('\n*** real_pos ***')
        # pprint(agent.real_pos)
        print('warning: logging directed to test_records')

    uf.log(session, agents)
    uf.scanner_summary(session, agents)

    # uf.interpret_benchmark(session, agents)

    print('\n---- Timers ----')
    for k, v in Timer.timers.items():
        if v > 30:
            print(k, round(v))

    print('-------------------- Counts --------------------')
    print(f"pairs tested: {len(pairs)}")
    pprint(Counter(session.counts))

    end = time.perf_counter()
    elapsed = round(end - all_start)
    print(f'Total time taken: {elapsed // 60}m, {elapsed % 60}s')
    print('\n-------------------------------------------------------------------------------\n')

    # for agent in agents:
    #     pprint(agent.open_trades)

"""this script runs the entire setup scanning process for each timeframe, with 
the appropriate timedelta offset"""

import keys
from binance.client import Client
import time
from datetime import datetime, timezone
import binance_funcs as funcs
from agents import DoubleST, EMACross, EMACrossHMA, AvgTradeSize
import binance.exceptions as bx
from pprint import pprint
import utility_funcs as uf
from random import shuffle
import sessions
from timers import Timer
from pushbullet import Pushbullet
from collections import Counter

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

script_start = time.perf_counter()

print('\n-+-+-+-+-+-+-+-+-+-+-+- Running Setup Scanner -+-+-+-+-+-+-+-+-+-+-+-\n')


def get_timeframes():
    hour = datetime.now(timezone.utc).hour
    # hour = 0 # for testing all timeframes
    d = {1: ('1h', None), 4: ('4h', None), 12: ('12h', None), 24: ('1d', None)}

    return [d[tf] for tf in d if hour % tf == 0]


########################################################################################################################

timeframes = get_timeframes()

print(f"Running setup_scan({timeframes})")
all_start = time.perf_counter()
session = sessions.TradingSession(0.0005)
print(f"\nCurrent time: {session.now_start}, {session.name}\n")

agents = []

for timeframe, offset in timeframes:
    agents.extend(
        [
            DoubleST(session, timeframe, offset, 3, 1.0),
            DoubleST(session, timeframe, offset, 3, 1.4),
            DoubleST(session, timeframe, offset, 3, 1.8),
            DoubleST(session, timeframe, offset, 5, 2.2),
            DoubleST(session, timeframe, offset, 5, 2.8),
            DoubleST(session, timeframe, offset, 5, 3.4),
            EMACross(session, timeframe, offset, 12, 21, 1.2),
            EMACross(session, timeframe, offset, 12, 21, 1.8),
            EMACross(session, timeframe, offset, 12, 21, 2.4),
            EMACrossHMA(session, timeframe, offset, 12, 21, 1.2),
            EMACrossHMA(session, timeframe, offset, 12, 21, 1.8),
            EMACrossHMA(session, timeframe, offset, 12, 21, 2.4),
            # AvgTradeSize(session, timeframe, offset, 2, 1000, 1.1, 'oco'),
            # AvgTradeSize(session, timeframe, offset, 2, 1000, 2.0, 'oco'),
            # AvgTradeSize(session, timeframe, offset, 2, 1000, 3.0, 'oco'),
            # AvgTradeSize(session, timeframe, offset, 2, 1000, 4.0, 'oco'),
        ]
    )

# session.name = ' | '.join([n.name for n in agents])

print("\n-*-*-*- Running record_stopped_sim_trades for all agents -*-*-*-\n")
for agent in agents:
    agent.record_stopped_sim_trades(session, timeframes)
print("\n-*-*-*- record_stopped_sim_trades finished for all agents -*-*-*-\n")

# compile and sort list of pairs to loop through ------------------------------
all_pairs = [k for k in session.pairs_data.keys()
             # if (session.pairs_data[k]['margin_allowed'])
             ]
shuffle(all_pairs)

positions = []
for agent in agents:
    posis = list(agent.real_pos.keys())
    positions.extend(posis)
pairs_in_pos = [p + 'USDT' for p in set(positions) if p != 'USDT']
print(f"Total {pairs_in_pos = }")
other_pairs = [p for p in all_pairs if p not in pairs_in_pos]
pairs = pairs_in_pos + other_pairs  # this ensures open positions will be checked first
# pairs = pairs[:10] # for testing the loop quickly

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

for n, pair in enumerate(pairs):
    # print('\n', n, pair, '\n')
    session.update_prices()
    asset = pair[:-1 * len(session.quote_asset)]
    for agent in agents:
        agent.init_in_pos(pair)

    now = datetime.now().strftime('%d/%m/%y %H:%M')

    # generate signals ------------------------------------------------------------

    # df_dict contains ohlc dataframes for each active timeframe for the current pair
    df_dict = funcs.prepare_ohlc(session, timeframes, pair)

    # if there is not enough history at a given timeframe, this function will return None instead of the df
    # TODO this would be a good function to start the migration to polars
    for tf, df in df_dict.items():
        if len(df) >= session.min_length:
            df_dict[tf] = session.compute_indicators(df, tf)
        else:
            print(f"length of {pair} {tf} data: {len(df)}")
            df_dict[tf] = None

    price = session.pairs_data[pair]['price']

    for agent in agents:
        # print('*****', agent.name)
        if df_dict[agent.tf] is not None:
            df_2 = df_dict[agent.tf].copy()
        else:
            print(f"{pair} too new for {agent.name}")
            continue

        market_state = uf.get_market_state(session, pair, df_2)

        signals = agent.signals(session, df_2, pair)

        if signals.get('inval'):
            stp = funcs.calc_stop(signals.get('inval'), session.pairs_data[pair]['spread'], price)
            inval_risk = abs((price - stp) / price)
            inval_risk_score = agent.calc_inval_risk_score(inval_risk)
            bal = session.spot_bal if agent.mode == 'spot' else session.margin_bal
            size_l, usdt_size_l, size_s, usdt_size_s = funcs.get_size(agent, price, bal, inval_risk)

        # remove indicators to avoid errors
        df_2 = df_2[['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol',
                     'num_trades', 'taker_buy_base_vol', 'taker_buy_quote_vol']]

        # update positions dictionary with current pair's open_risk values ------------
        if agent.in_pos['real']:
            real_qty = float(agent.open_trades[pair]['position']['base_size'])
            agent.real_pos[asset].update(
                agent.update_pos(session, asset, real_qty, signals['inval_ratio'], 'real'))

            real_ep = float(agent.open_trades[pair]['position']['entry_price'])
            if real_ep:
                agent.real_pos[asset]['price_delta'] = (price - real_ep) / real_ep  # how much has price moved since entry

            # check if price has moved beyond reach of normal close signal
            if agent.real_pos[asset]['or_R'] < 0:
                signals['signal'] = f"close_{agent.in_pos['real']}"

        if agent.in_pos['sim']:
            sim_qty = float(agent.sim_trades[pair]['position']['base_size'])
            agent.sim_pos[asset].update(
                agent.update_pos(session, asset, sim_qty, signals['inval_ratio'], 'sim'))
            sim_ep = float(agent.sim_trades[pair]['position']['entry_price'])
            if sim_ep:
                agent.sim_pos[asset]['price_delta'] = (price - sim_ep) / sim_ep

            # check if price has moved beyond reach of normal close signal
            if agent.sim_pos[asset]['or_R'] < 0:
                signals['signal'] = f"close_{agent.in_pos['sim']}"

        # margin order execution ------------------------------------------------------

        if signals.get('signal') in ['open_spot', 'tp_spot', 'close_spot']:
            usdt_depth_l, _ = funcs.get_depth(session, pair)
            usdt_size, usdt_depth, size = usdt_size_l, usdt_depth_l, size_l
            direction = 'spot'
        elif signals.get('signal') in ['open_long', 'tp_long', 'close_long']:
            usdt_depth_l, _ = funcs.get_depth(session, pair)
            usdt_size, usdt_depth, size = usdt_size_l, usdt_depth_l, size_l
            direction = 'long'
        elif signals.get('signal') in ['open_short', 'tp_short', 'close_short']:
            _, usdt_depth_s = funcs.get_depth(session, pair)
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

            elif ((('spot' in signals.get('signal')) and (inval_risk_score < 0.5))
                  or
                  (('long' in signals.get('signal')) and (inval_risk_score < 0.5))
                  or
                  (('short' in signals.get('signal')) and (inval_risk_score < 0.5))):
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
            usdt_bal = session.spot_usdt_bal if agent.mode == 'spot' else session.margin_usdt_bal
            agent.real_pos['USDT'] = usdt_bal

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
                agent.open_pos(session, pair, size, stp, signals['inval_ratio'], market_state, sim_reason, direction)
            except bx.BinanceAPIException as e:
                if e.code == -3045:  # borrow failed because there weren't enough assets to borrow
                    del agent.open_trades[pair]
                    agent.open_pos(session, pair, size, stp, signals['inval_ratio'], market_state, 'not_enough_borrow', direction)
                else:
                    agent.record_trades(session, 'all')
                    print(f'{agent.name} problem with open_{direction} order for {pair}')
                    print(e.code)
                    print(e.message)
                    print(f"{size = } {stp = } {signals['inval_ratio'] = }")
                    pb.push_note(now, f'{agent.name} exeption during {pair} open_{direction} order')
                continue

        elif signals.get('signal') in ['tp_spot', 'tp_long', 'tp_short']:
            try:
                agent.tp_pos(session, pair, stp, signals['inval_ratio'], direction)
            except bx.BinanceAPIException as e:
                agent.record_trades(session, 'all')
                print(f'{agent.name} problem with tp_{direction} order for {pair}')
                print(e.code)
                print(e.message)
                print(f"{stp = } {signals['inval_ratio'] = }")
                pb.push_note(now, f'{agent.name} exception during {pair} tp_{direction} order')
                continue

        # calculate open risk and take profit if necessary ----------------------------
        agent.tp_signals(asset)
        if agent.in_pos['real']:
            direction = agent.in_pos['real']
            try:
                agent.tp_pos(session, pair, stp, signals['inval_ratio'], direction)
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
                agent.tp_pos(session, pair, stp, signals['inval_ratio'], direction)
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
before = session.margin_usdt_bal
session.get_usdt_m()
after = session.margin_usdt_bal
if before != after:
    print('\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    print(f'USDT margin balance wrong\nbefore: {before}\nafter: {after}')
    print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

before = session.spot_usdt_bal
session.get_usdt_s()
after = session.spot_usdt_bal
if before != after:
    print('\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    print(f'USDT spot balance wrong\nbefore: {before}\nafter: {after}')
    print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')

print('-:-' * 20)

for agent in agents:
    agent.record_trades(session, 'all')
    if not session.live:
        print('')
        print(agent.name.upper(), 'SUMMARY')
        if agent.realised_pnl_spot or agent.sim_pnl_spot:
            print(
                f'realised real spot pnl: {agent.realised_pnl_spot:.1f}R, realised sim spot pnl: {agent.sim_pnl_spot:.1f}R')
        if agent.realised_pnl_long or agent.sim_pnl_long:
            print(
                f'realised real long pnl: {agent.realised_pnl_long:.1f}R, realised sim long pnl: {agent.sim_pnl_long:.1f}R')
        if agent.realised_pnl_short or agent.sim_pnl_short:
            print(
                f'realised real short pnl: {agent.realised_pnl_short:.1f}R, realised sim short pnl: {agent.sim_pnl_short:.1f}R')
        print(f'tor: {agent.total_open_risk:.1f}')
        # print(f'or list: {[round(x, 2) for x in sorted(agent.or_list, reverse=True)]}')

        propnl = agent.open_pnl('spot', 'real')
        if propnl:
            print(f"real open pnl spot: {propnl:.1f}R")

        lropnl = agent.open_pnl('long', 'real')
        if lropnl:
            print(f"real open pnl long: {lropnl:.1f}R")

        sropnl = agent.open_pnl('short', 'real')
        if sropnl:
            print(f"real open pnl short: {sropnl:.1f}R")

        psopnl = agent.open_pnl('spot', 'sim')
        if psopnl:
            print(f"sim open pnl spot: {psopnl:.1f}R")

        lsopnl = agent.open_pnl('long', 'sim')
        if lsopnl:
            print(f"sim open pnl long: {lsopnl:.1f}R")

        ssopnl = agent.open_pnl('short', 'sim')
        if ssopnl:
            print(f"sim open pnl short: {ssopnl:.1f}R")

        print(f'{agent.name} Counts:')
        for k, v in agent.counts_dict.items():
            if v:
                print(k, v)
        print('-:-' * 20)

    usdt_bal = session.spot_usdt_bal if agent.mode == 'spot' else session.margin_usdt_bal
    agent.real_pos['USDT'] = usdt_bal

if not session.live:
    # print('\n*** real_pos ***')
    # pprint(agent.real_pos)
    print('warning: logging directed to test_records')

uf.market_benchmark(session)
for agent in agents:
    uf.log(session, agent)
    agent.benchmark = uf.strat_benchmark(session, agent)
uf.scanner_summary(session, agents)

uf.interpret_benchmark(session, agents)

print('\n---- Timers ----')
for k, v in Timer.timers.items():
    if v > 10:
        elapsed = f"{int(v // 60)}m {v % 60:.1f}s"
        print(k, elapsed)

print('-------------------- Counts --------------------')
print(f"pairs tested: {len(pairs)}")
pprint(Counter(session.counts))

end = time.perf_counter()
elapsed = round(end - all_start)
print(f'Total time taken: {elapsed // 60}m, {elapsed % 60}s')
print('\n-------------------------------------------------------------------------------\n')

# for agent in agents:
#     pprint(agent.open_trades)

########################################################################################################################


script_end = time.perf_counter()

total_time = script_end - script_start
print(f"Scanner finished, total time taken: {int(total_time // 60)}m {int(total_time % 60)}s")

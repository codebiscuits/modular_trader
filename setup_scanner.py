import time
script_start = time.perf_counter()

from resources import utility_funcs as uf, binance_funcs as funcs
from datetime import datetime, timezone
from agents import TrailFractals
from pprint import pprint
import sessions
from resources.timers import Timer
from pushbullet import Pushbullet
from collections import Counter

# TODO current (02/04/23) roadmap should be:
#  * get detailed push notes in all exception handling code so i always know whats going wrong, and change the ss_log
#  so it creates a new file for each session, named by the date and time they took place
#  * start integrating polars and doing anything else i can to speed things up enough to run 3day and 1week timeframes
#  in the same session as everything else
#  * get spot trading and oco entries and trade adds working so i can use other strats

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

print('\n-+-+-+-+-+-+-+-+-+-+-+- Running Setup Scanner -+-+-+-+-+-+-+-+-+-+-+-\n')

########################################################################################################################

session = sessions.TradingSession(0.002)

print(f"Running setup_scan({session.timeframes})")
print(f"\nCurrent time: {session.now_start}, {session.name}\n")

# initialise agents
agents = []
for timeframe, offset in session.timeframes:
    agents.extend(
        [
            TrailFractals(session, timeframe, offset, min_conf=0.5),
            # DoubleST(session, timeframe, offset, 3, 1.0),
            # DoubleST(session, timeframe, offset, 3, 1.4),
            # DoubleST(session, timeframe, offset, 3, 1.8),
            # DoubleST(session, timeframe, offset, 5, 2.2),
            # DoubleST(session, timeframe, offset, 5, 2.8),
            # DoubleST(session, timeframe, offset, 5, 3.4),
            # DoubleSTnoEMA(session, timeframe, offset, 3, 1.0),
            # DoubleSTnoEMA(session, timeframe, offset, 3, 1.4),
            # DoubleSTnoEMA(session, timeframe, offset, 3, 1.8),
            # DoubleSTnoEMA(session, timeframe, offset, 5, 2.2),
            # DoubleSTnoEMA(session, timeframe, offset, 5, 2.8),
            # DoubleSTnoEMA(session, timeframe, offset, 5, 3.4),
            # EMACross(session, timeframe, offset, 12, 21, 1.2),
            # EMACross(session, timeframe, offset, 12, 21, 1.8),
            # EMACross(session, timeframe, offset, 12, 21, 2.4),
            # EMACrossHMA(session, timeframe, offset, 12, 21, 1.2),
            # EMACrossHMA(session, timeframe, offset, 12, 21, 1.8),
            # EMACrossHMA(session, timeframe, offset, 12, 21, 2.4),
        ]
    )

agents = {a.id: a for a in agents}

pprint(session.features)

# session.name = ' | '.join([n.name for n in agents.values()])

print("\n-*-*-*- Running rst and rsst for all agents -*-*-*-\n")
for agent in agents.values():
    agent.record_stopped_trades(session, session.timeframes)
    agent.record_stopped_sim_trades(session, session.timeframes)

    # pprint(agent.real_pos)
print("\n-*-*-*- rst and rsst finished for all agents -*-*-*-\n")

# compile and sort list of pairs to loop through ------------------------------
pairs = [k for k in session.pairs_data.keys()]
# pairs = pairs[:10] # for testing the loop quickly

init_end = time.perf_counter()
init_elapsed = init_end - script_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Generate Technical Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
print("\n-*-*-*- Generating Technical Signals for all agents -*-*-*-\n")
tech_start = time.perf_counter()

raw_signals = []

for n, pair in enumerate(pairs):
    # print('-' * 100)
    # print('\n', n, pair, '\n')
    # print(n, pair)
    session.update_prices()

    now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')

    # df_dict contains ohlc dataframes for each active timeframe for the current pair
    df_dict = funcs.prepare_ohlc(session, session.timeframes, pair)

    # if there is not enough history at a given timeframe, this function will return None instead of the df
    # TODO this would be a good function to start the migration to polars
    if not df_dict:
        continue

    for tf, df in df_dict.items():
        df_dict[tf] = session.compute_indicators(df, tf)
        df_dict[tf] = session.compute_features(df, tf)

    for agent in agents.values():
        if agent.tf not in df_dict: # some pairs will not have all required timeframes for the session
            continue
        # print(f"{agent.name}")
        df_2 = df_dict[agent.tf].copy()
        # print(f"\n{pair} {tf} {len(df_2)}")

        # market_state = uf.get_market_state(session, agent, pair, df_2)

        signal = agent.signals(session, df_2, pair)
        if signal.get('bias'):
            # signal.update(market_state)
            raw_signals.append(signal)

        # if this agent is in position with this pair, calculate open risk and related metrics here
        # update positions dictionary with current pair's open_risk values ------------
        price = df_2.close.iloc[-1]
        asset = pair[:-1 * len(session.quote_asset)]

        real_match = (asset in agent.real_pos.keys()) == (pair in agent.open_trades.keys())
        sim_match = (asset in agent.sim_pos.keys()) == (pair in agent.sim_trades.keys())
        if not real_match:
            print(f"{pair} real pos doesn't match real trades")
        if not sim_match:
            print(f"{pair} sim pos doesn't match sim trades")

        if pair in agent.open_trades:
            if signal.get('inval_ratio'):
                inval_ratio = signal['inval_ratio']
            elif agent.open_trades[pair]['position']['direction'] in ['spot', 'long']:
                inval_ratio = signal['long_ratio']
            elif agent.open_trades[pair]['position']['direction'] == 'short':
                inval_ratio = signal['short_ratio']
            real_qty = float(agent.open_trades[pair]['position']['base_size'])
            new_stats = agent.update_pos(session, pair, real_qty, inval_ratio, 'real')
            agent.real_pos[asset].update(new_stats)
            real_ep = float(agent.open_trades[pair]['position']['entry_price'])
            agent.real_pos[asset]['price_delta'] = (price - real_ep) / real_ep
        if pair in agent.sim_trades:
            if signal.get('inval_ratio'):
                inval_ratio = signal['inval_ratio']
            elif agent.sim_trades[pair]['position']['direction'] in ['spot', 'long']:
                inval_ratio = signal['long_ratio']
            elif agent.sim_trades[pair]['position']['direction'] == 'short':
                inval_ratio = signal['short_ratio']
            sim_qty = float(agent.sim_trades[pair]['position']['base_size'])
            new_stats = agent.update_pos(session, pair, sim_qty, inval_ratio, 'sim')
            agent.sim_pos[asset].update(new_stats)
            sim_ep = float(agent.sim_trades[pair]['position']['entry_price'])
            agent.sim_pos[asset]['price_delta'] = (price - sim_ep) / sim_ep

signal_counts = Counter([f"{signal['tf']}_{signal['bias']}" for signal in raw_signals])
for tf in session.timeframes:
    tf_signals = [sig for sig in raw_signals if sig['tf'] == tf[0]]
    session.market_bias[tf[0]] = (
            (
                    signal_counts.get(f'{tf[0]}_bullish', 0)
                    - signal_counts.get(f'{tf[0]}_bearish', 0)
            )
            / max(len(raw_signals), 1)
    )

tech_end = time.perf_counter()
tech_took = tech_end - tech_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Process Raw Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
process_start = time.perf_counter()
print(f"\n-*-*-*- Processing {len(raw_signals)} Raw Signals for all agents -*-*-*-\n")

# gather data on current algo orders
session.update_algo_orders()

processed_signals = dict(
    real_sim_tp_close=[],  # all real and sim tps and closes go in here for immediate execution
    unassigned=[],  # all real open signals go in here for further selection
    sim_open=[],  # nothing will be put in here at first but many real open signals will end up in here
    real_open=[],  # nothing will be put in here at first, real open signals will end up here if they pass all tests
    tracked_close=[],  # can be left until last
)

while raw_signals:
    signal = raw_signals.pop(0)

    sig_bias = signal['bias']
    sig_agent = agents[signal['agent']]
    sig_pair = signal['pair']

    # TODO i need to add other scores here
    inval_score = signal['inval_score']
    confidence_score = signal['confidence']
    if signal['tf'] == '1h':
        rank_score = signal.get('market_rank_1d', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1d', 1))
    elif signal['tf'] in {'4h', '12h'}:
        rank_score = signal.get('market_rank_1w', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1w', 1))
    elif signal['tf'] == '1d':
        rank_score = signal.get('market_rank_1m', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1m', 1))
    sig_score = ((2 * confidence_score) + (1 * inval_score) + (1 * rank_score)) / 4
    signal['score'] = sig_score
    min_score = 0.5

    # find whether i am currently long, short or flat on the agent and pair in this signal
    try:
        real_position = sig_agent.real_pos.get(signal['asset'], {'direction': 'flat'})['direction']
        sim_position = sig_agent.sim_pos.get(signal['asset'], {'direction': 'flat'})['direction']  # returns 'flat' if no position
        tracked_position = sig_agent.tracked.get(signal['asset'], {'direction': 'flat'})['direction']
    except KeyError as e:
        print('KeyError')
        print(e)
        pprint(signal)
        pprint(sig_agent.tracked)

    bullish_pos = 'spot' if (sig_agent.mode == 'spot') else 'long'

    # TODO 'oco' signals are an alternative to 'open', so they should be treated as such in this section, maybe as a
    #  condition which requires that position is flat and signal['exit'] is oco and bias is either bullish or bearish

    # TODO i need to add conditions so that strategies which trail stops don't have their positions closed by a bias
    #  flip, only new positions will be opened on  signals. so i might end up with long and short positions open
    #  simultaneously, but that's ok because each position will be managed and the price will decide which should stay open

    # TODO perhaps there should be a flag in each signal that says whether that agent's positions should be closed with
    #  signals or not, because oco orders should never be managed by signals but trailing stop strategies sometimes have
    #  close/tp conditions and sometimes don't

    if sig_bias == 'bullish':
        if real_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)
            # check if tp is necessary
            if sig_agent.real_pos[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Real {sig_pair} or_R: {sig_agent.real_pos[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'real', 'long'))

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'short'))
            if sig_score >= min_score:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'long'))
            elif sim_position == 'flat':
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', 'long'))

        elif real_position == 'flat':
            if sig_score >= min_score:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', bullish_pos))
            elif sim_position == 'flat':
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', bullish_pos))
        else:
            print("bullish bias didn't produce a tracked outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'sim')
            # check if tp is necessary
            if sig_agent.sim_pos[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Sim {sig_pair} or_R: {sig_agent.sim_pos[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'sim', 'long'))

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif sim_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', 'short'))
        elif sim_position == 'flat':
            pass
        else:
            print("bullish bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'tracked')
            # check if tp is necessary
            if sig_agent.tracked[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Tracked {sig_pair} or_R: {sig_agent.tracked[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'tracked', 'long'))

        elif tracked_position == 'short':
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', 'short'))
        elif tracked_position == 'flat':
            pass
        else:
            print("bullish bias didn't produce a tracked outcome, logic needs more work")

    # -------------------------------------------------------------------------------------------------------------------

    elif sig_bias == 'bearish':
        if real_position == 'spot':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'spot'))
        elif real_position == 'long':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'long'))
            if sig_score >= min_score:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
            elif sim_position == 'flat':
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', 'short'))
        elif real_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)
            # check if tp is necessary
            if sig_agent.real_pos[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Real {sig_pair} or_R: {sig_agent.real_pos[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'real', 'short'))

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'flat':
            if sig_score >= min_score:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
            elif sim_position == 'flat':
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', 'short'))
        else:
            print("bearish bias didn't produce a real outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', bullish_pos))
        elif sim_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'sim')
            # check if tp is necessary
            if sig_agent.sim_pos[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Sim {sig_pair} or_R: {sig_agent.sim_pos[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'sim', 'short'))

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif sim_position == 'flat':
            pass
        else:
            print("bearish bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', bullish_pos))
        elif tracked_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'tracked')
            # check if tp is necessary
            if sig_agent.tracked[signal['asset']]['or_R'] > sig_agent.indiv_r_limit:
                print(f"{sig_agent} Tracked {sig_pair} or_R: {sig_agent.tracked[signal['asset']]['or_R']}")
                processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'tracked', 'short'))


        elif tracked_position == 'flat':
            pass
        else:
            print("bearish bias didn't produce a tracked outcome, logic needs more work")

    # -------------------------------------------------------------------------------------------------------------------

    elif sig_bias == 'neutral':
        if real_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', bullish_pos))
        elif real_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'short'))
        elif real_position == 'flat':
            pass
        else:
            print("neutral bias didn't produce a real outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', bullish_pos))
        elif sim_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', 'short'))
        elif sim_position == 'flat':
            pass
        else:
            print("neutral bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', bullish_pos))
        elif tracked_position == 'short':
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', 'short'))
        elif tracked_position == 'flat':
            pass
        else:
            print("neutral bias didn't produce a tracked outcome, logic needs more work")

    else:
        print("signal had no bias")
        pprint(signal)

process_end = time.perf_counter()
process_took = process_end - process_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute TP / Close Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
tp_close_start = time.perf_counter()
# execute any real and sim technical close and tp signals
print(f"\n-+-+-+-+-+-+-+- Executing {len(processed_signals['real_sim_tp_close'])} Real/Sim TPs/Closes -+-+-+-+-+-+-+-")

checked_signals = uf.remove_duplicates(processed_signals['real_sim_tp_close'])
print(f"{len(checked_signals) = }")

for signal in checked_signals:
    print(f"\nProcessing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} {signal['direction']}")
    if signal['action'] == 'close':
        agents[signal['agent']].close_pos(session, signal)
    elif signal['action'] == 'tp':
        agents[signal['agent']].tp_pos(session, signal)
    else:
        print("problem with signal in tps/closes")
        pprint(signal)

tp_close_end = time.perf_counter()
tp_close_took = tp_close_end - tp_close_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Calculate Fixed Risk
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# calculate fixed risk for each agent using wanted rpnl
print(f"\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+- Calculating Fixed Risk -+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
for agent in agents.values():
    agent.set_fixed_risk(session)
    agent.test_fixed_risk(0.0002, 0.0002)
    agent.print_fixed_risk()
    agent.calc_tor()

    wanted_spot = round(agent.realised_pnls['wanted_spot'], 1)
    wanted_long = round(agent.realised_pnls['wanted_long'], 1)
    wanted_short = round(agent.realised_pnls['wanted_short'], 1)
    # print(f"\n{agent.name} Realised PnLs: {wanted_spot = }R, {wanted_long = }R, {wanted_short = }R\n")
    session.wrpnl_totals['spot'] += wanted_spot
    session.wrpnl_totals['long'] += wanted_long
    session.wrpnl_totals['short'] += wanted_short

    unwanted_spot = round(agent.realised_pnls['unwanted_spot'], 1)
    unwanted_long = round(agent.realised_pnls['unwanted_long'], 1)
    unwanted_short = round(agent.realised_pnls['unwanted_short'], 1)
    # print(f"\n{agent.name} Realised PnLs: {unwanted_spot = }R, {unwanted_long = }R, {unwanted_short = }R\n")
    session.urpnl_totals['spot'] += unwanted_spot
    session.urpnl_totals['long'] += unwanted_long
    session.urpnl_totals['short'] += unwanted_short

    # TODO make sure algo_order counts are up to date after tps and closes
    # print(f"{agent.name} {agent.fixed_risk_spot} {agent.fixed_risk_l} {agent.fixed_risk_s}")

print('\n\nwanted rpnl totals:')
pprint(session.wrpnl_totals)
print('\n\nunwanted rpnl totals:')
pprint(session.urpnl_totals)

for signal in processed_signals['unassigned']:
    signal['base_size'], signal['quote_size'] = agents[signal['agent']].get_size(session, signal)

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Sort and Filter Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
sort_start = time.perf_counter()

# gather data on current algo orders
session.update_algo_orders()

# next sort the unassigned list by scores. items are popped from the end of the list so i want the best signals to be
# last so that they get processed first, so i don't use the 'reverse=True' option
unassigned = sorted(processed_signals['unassigned'], key=lambda x: x['inval_score'])
print(f"\n-*-*-*- Sorting and Filtering {len(unassigned)} Processed Signals for all agents -*-*-*-\n")

# work through the list and check each filter for each signal
or_limits = {agent.name: agent.total_open_risk for agent in agents.values()}
pos_limits = {agent.name: agent.num_open_positions for agent in agents.values()}
algo_limits = {pair: (v['max_algo_orders'] - v['algo_orders']) for pair, v in session.pairs_data.items()}
usdt_bal_s = session.spot_usdt_bal

while unassigned:
    s = unassigned.pop()

    # set variables
    agent = agents[s['agent']]
    sim_reasons = []
    balance = session.spot_bal if s['mode'] == 'spot' else session.margin_bal
    quote_size = s['quote_size']
    r = 1

    if s['direction'] in {'spot', 'long'}:
        usdt_depth, _ = funcs.get_depth(session, s['pair'])
    elif s['direction'] == 'short':
        _, usdt_depth = funcs.get_depth(session, s['pair'])

    if s['mode'] == 'margin' and not session.pairs_data[s['pair']]['margin_allowed']:
        sim_reasons.append('not_a_margin_pair')

    if usdt_depth == 0:
        sim_reasons.append('too_much_spread')
    elif usdt_depth < (quote_size / 2):
        sim_reasons.append('books_too_thin')
    elif quote_size > usdt_depth >= (quote_size / 2):  # only trim size if books are a bit too thin
        r = usdt_depth / quote_size
        quote_size = usdt_depth
        print(f"{now} {s['pair']} books too thin, reducing size from {quote_size:.3} to {usdt_depth:.3}")

    if quote_size > (balance * 0.1):
        r = (balance * 0.1) / quote_size
        quote_size = balance * 0.1

    if or_limits[agent.name] >= agent.total_r_limit:
        sim_reasons.append('too_much_or')
    if pos_limits[agent.name] >= agent.max_positions:
        sim_reasons.append('too_many_pos')
    if (algo_limits[s['pair']] < 2 and s['action'] == 'oco') or (algo_limits[s['pair']] < 1 and s['action'] == 'open'):
        sim_reasons.append('algo_order_limit')

    if s['mode'] == 'spot' and quote_size > usdt_bal_s > (quote_size / 2):
        r = (usdt_bal_s - 1) / quote_size
        quote_size = usdt_bal_s - 1
    elif s['mode'] == 'spot' and quote_size > usdt_bal_s:
        sim_reasons.append('not_enough_usdt')

    if s['mode'] == 'margin' and session.margin_lvl < 3:
        r = session.margin_lvl / 3
        quote_size *= r

    if s['mode'] == 'margin' and session.margin_lvl < 2:
        sim_reasons.append('margin_acct_too_levered')

    if quote_size < session.min_size: # this condition must come after all the conditions which could reduce size
        sim_reasons.append('too_small')

    # TODO need to work out how to do 'not_enough_borrow' in this section, or if that fails, a way to switch a real
    #  trade to sim at the borrowing stage

    if sim_reasons:
        s['sim_reasons'] = sim_reasons
        s['state'] = 'sim'
        s['pct_of_full_pos'] *= r
        processed_signals['sim_open'].append(s)
    else:
        or_limits[agent.name] += r
        pos_limits[agent.name] += 1
        algo_limits[s['pair']] -= 2 if s['action'] == 'oco' else 1

        if r != 1:
            print(f"{s['agent']} {s['pair']} {r = }, size adjusted:")
            print(f"orig: {s['quote_size']:.2f}, {s['base_size']}. new: {quote_size:.2f}, {s['base_size'] * r}")
        s['quote_size'] = quote_size
        s['base_size'] = s['base_size'] * r # the value of r is equivalent to the change in size, if any.
        s['pct_of_full_pos'] *= r

        processed_signals['real_open'].append(s)

sort_end = time.perf_counter()
sort_took = sort_end - sort_start
# pprint(processed_signals['real_open'])

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Real Open Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
real_open_start = time.perf_counter()

print(f"\n-+-+-+-+-+-+-+-+-+-+-+- Executing {len(processed_signals['real_open'])} Real Opens -+-+-+-+-+-+-+-+-+-+-+-")

# TODO i have used margin_lvl to decide whether the account is over-levered, but it would be more precise to use total
#  assets and total liabilities and actualy calculate how much more i can still borrow, then i can still do some real
#  opens until the leverage really runs out
if session.check_margin_lvl():
    for signal in processed_signals['real_open']:
        signal['state'] == 'sim'
        signal['sim_reasons'] = ['too_much_leverage']
        processed_signals['sim_open'].append(signal)

else:
    for signal in processed_signals['real_open']:
        print(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} {signal['direction']}")
        if signal['mode'] == 'margin':
            agents[signal['agent']].open_real_M(session, signal, 0)
        elif signal['mode'] == 'spot':
            agents[signal['agent']].open_real_s(session, signal, 0)

# when they are all finished, update records once

real_open_end = time.perf_counter()
real_open_took = real_open_end - real_open_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Sim Open Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
sim_open_start = time.perf_counter()

sim_opens = [sig for sig in processed_signals['sim_open'] # discard signals for existing sim positions
             if sig['asset'] not in agents[sig['agent']].sim_pos.keys()]

print(f"\n-+-+-+-+-+-+-+-+-+-+-+- Executing {len(sim_opens)} Sim Opens -+-+-+-+-+-+-+-+-+-+-+-")

for signal in sim_opens:

    # print(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} {signal['direction']}")
    # print(f"Sim reason: {signal['sim_reasons']}, score: {signal['score']}")
    agents[signal['agent']].open_sim(session, signal)

# when they are all finished, update records once

sim_open_end = time.perf_counter()
sim_open_took = sim_open_end - sim_open_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Tracked Close Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# probably won't need anything too clever since i don't think these come up very often

# when they are all finished, update records once

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Logs and Summaries
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
log_start = time.perf_counter()

session.get_usdt_m()
session.get_usdt_s()

print('-:-' * 20)

for agent in agents.values():
    agent.record_trades(session, 'all')

    #################################

    if not session.live:
        print('')
        print(agent.name.upper(), 'SUMMARY')

        print(f"{len(agent.real_pos.keys())} real positions, {len(agent.sim_pos.keys())} sim positions")

        if agent.realised_pnls['real_spot'] or agent.realised_pnls['sim_spot']:
            print(f"realised real spot pnl: {agent.realised_pnls['real_spot']:.1f}R, "
                  f"realised sim spot pnl: {agent.agent.realised_pnls['sim_spot']:.1f}R")

        if agent.realised_pnls['real_long'] or agent.realised_pnls['sim_long']:
            print(f"realised real long pnl: {agent.realised_pnls['real_long']:.1f}R, "
                  f"realised sim long pnl: {agent.realised_pnls['sim_long']:.1f}R")

        if agent.realised_pnls['real_short'] or agent.realised_pnls['sim_short']:
            print(f"realised real short pnl: {agent.realised_pnls['real_short']:.1f}R, "
                  f"realised sim short pnl: {agent.realised_pnls['sim_short']:.1f}R")
        print(f'tor: {agent.total_open_risk:.1f}')

    print(f'{agent.name} Counts:')
    for k, v in agent.counts_dict.items():
        if v:
            print(k, v)
    print('-:-' * 20)

    ################################

    usdt_bal = session.spot_usdt_bal if agent.mode == 'spot' else session.margin_usdt_bal
    agent.real_pos['USDT'] = usdt_bal

if not session.live:
    print('warning: logging directed to test_records')

for tf in session.timeframes:
    print(f"\n\n{tf[0]} market bias: {session.market_bias[tf[0]]:.2f} range from 1 (bullish) to -1 (bearish)\n\n")
pprint(signal_counts)

uf.market_benchmark(session)
for agent in agents.values():
    uf.log(session, agent)
    agent.benchmark = uf.strat_benchmark(session, agent)
uf.scanner_summary(session, agents.values())

log_end = time.perf_counter()
log_elapsed = log_end - log_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# End
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

print('\n---- Timers ----')
for k, v in Timer.timers.items():
    if v > 10:
        elapsed = f"{int(v // 60)}m {v % 60:.1f}s"
        print(k, elapsed)

print('-------------------- Counts --------------------')
print(f"pairs tested: {len(pairs)}")
pprint(Counter(session.counts))
print('\n-------------------------------------------------------------------------------\n')

for agent in agents.values():
    print(agent.name)
    print('real_pos')
    pprint(agent.real_pos)
    print('open_trades')
    pprint(agent.open_trades)
    print('<->' * 15)

script_end = time.perf_counter()
total_time = script_end - script_start

def section_times():
    print('Scanner finished')
    print(f"Initialisation took: {int(init_elapsed // 60)}m {int(init_elapsed % 60)}s")
    print(f"Generating Technical Signals took: {int(tech_took // 60)}m {int(tech_took % 60)}s")
    print(f"Processing Technical Signals took: {int(process_took // 60)}m {int(process_took % 60)}s")
    print(f"Executing TP/Close Signals took: {int(tp_close_took // 60)}m {int(tp_close_took % 60)}s")
    print(f"Sorting Technical Signals took: {int(sort_took // 60)}m {int(sort_took % 60)}s")
    print(f"Executing Real Open Signals took: {int(real_open_took // 60)}m {int(real_open_took % 60)}s")
    print(f"Executing Sim Open Signals took: {int(sim_open_took // 60)}m {int(sim_open_took % 60)}s")
    print(f"Logging took: {int(log_elapsed // 60)}m {int(log_elapsed % 60)}s")
    print(f"Total time taken: {int(total_time // 60)}m {int(total_time % 60)}s")
section_times()

print(f"used-weight: {session.client.response.headers.get('x-mbx-used-weight')}")
print(f"used-weight-1m: {session.client.response.headers.get('x-mbx-used-weight-1m')}")

# uf.plot_call_weights(session)

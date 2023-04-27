import time
script_start = time.perf_counter()

import keys
from binance.client import Client
from datetime import datetime, timezone
import binance_funcs as funcs
from agents import DoubleST, DoubleSTnoEMA, EMACross, EMACrossHMA, AvgTradeSize
import binance.exceptions as bx
from pprint import pprint
import utility_funcs as uf
import sessions
from timers import Timer
from pushbullet import Pushbullet
from collections import Counter
import plotly.express as px
import traceback

# TODO current (02/04/23) roadmap should be:
#  1: get setup scanner back up and running with the new architecture in place
#  2: get detailed push notes in all exception handling code so i always know whats going wrong, and change the ss_log
#  so it creates a new file for each session, named by the date and time they took place
#  3: start integrating polars and doing anything else i can to speed things up enough to run 3day and 1week timeframes
#  in the same session as everything else
#  4: get spot trading and oco entries working so i can use other strats
#  5: ML analysis for better performing strats

client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

print('\n-+-+-+-+-+-+-+-+-+-+-+- Running Setup Scanner -+-+-+-+-+-+-+-+-+-+-+-\n')


def get_timeframes():
    hour = datetime.now(timezone.utc).hour
    # hour = 0 # for testing all timeframes
    d = {1: ('1h', None), 4: ('4h', None), 12: ('12h', None), 24: ('1d', None)}

    return [d[tf] for tf in d if hour % tf == 0]


########################################################################################################################

# timeframes = get_timeframes()
timeframes = [('1h', None)]

print(f"Running setup_scan({timeframes})")
session = sessions.TradingSession(0.0003)
print(f"\nCurrent time: {session.now_start}, {session.name}\n")

# initialise agents
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
            DoubleSTnoEMA(session, timeframe, offset, 3, 1.0),
            DoubleSTnoEMA(session, timeframe, offset, 3, 1.4),
            DoubleSTnoEMA(session, timeframe, offset, 3, 1.8),
            DoubleSTnoEMA(session, timeframe, offset, 5, 2.2),
            DoubleSTnoEMA(session, timeframe, offset, 5, 2.8),
            DoubleSTnoEMA(session, timeframe, offset, 5, 3.4),
            EMACross(session, timeframe, offset, 12, 21, 1.2),
            EMACross(session, timeframe, offset, 12, 21, 1.8),
            EMACross(session, timeframe, offset, 12, 21, 2.4),
            EMACrossHMA(session, timeframe, offset, 12, 21, 1.2),
            EMACrossHMA(session, timeframe, offset, 12, 21, 1.8),
            EMACrossHMA(session, timeframe, offset, 12, 21, 2.4),
            # AvgTradeSize(session, timeframe, offset, 2, 200, 1.1, 'trail'),
            # AvgTradeSize(session, timeframe, offset, 2, 200, 2.0, 'oco'),
            # AvgTradeSize(session, timeframe, offset, 2, 200, 3.0, 'oco'),
            # AvgTradeSize(session, timeframe, offset, 2, 200, 4.0, 'oco'),
        ]
    )

agents = {a.id: a for a in agents}

# session.name = ' | '.join([n.name for n in agents.values()])

print("\n-*-*-*- Running rst and rsst for all agents -*-*-*-\n")
for agent in agents.values():
    agent.record_stopped_trades(session, timeframes)
    agent.record_stopped_sim_trades(session, timeframes)

    # pprint(agent.real_pos)
print("\n-*-*-*- rst and rsst finished for all agents -*-*-*-\n")

# compile and sort list of pairs to loop through ------------------------------
pairs = [k for k in session.pairs_data.keys()]
# pairs = pairs[:10] # for testing the loop quickly

session.min_length += 15 # TODO this is a temp fix to compensate for all the NaN rows which will be dropped when
session.max_length += 15 #  indicators are calculated. I really need a way for each indicator to add the correct number
# of rows to session.min_length as it is added to the central indicator set so i can be sure there will always be enough
# rows once indicators have been calculated

init_end = time.perf_counter()
init_elapsed = init_end - script_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Generate Technical Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
print("\n-*-*-*- Generating Technical Signals for all agents -*-*-*-\n")
tech_start = time.perf_counter()

raw_signals = []

for n, pair in enumerate(pairs):
    # print('\n', n, pair, '\n')
    session.update_prices()

    now = datetime.now().strftime('%d/%m/%y %H:%M')

    # df_dict contains ohlc dataframes for each active timeframe for the current pair
    df_dict = funcs.prepare_ohlc(session, timeframes, pair)

    # if there is not enough history at a given timeframe, this function will return None instead of the df
    # TODO this would be a good function to start the migration to polars
    for tf, df in df_dict.items():
        # print(f"\n{pair} {tf} {len(df)}")
        if len(df) >= session.min_length:
            df_dict[tf] = session.compute_indicators(df, tf)
        else:
            # print(f"length of {pair} {tf} data: {len(df)}")
            df_dict[tf] = None

    for agent in agents.values():
        if df_dict[agent.tf] is not None:
            df_2 = df_dict[agent.tf].copy()
        else:
            continue

        market_state = uf.get_market_state(session, agent, pair, df_2)

        signal = agent.signals(session, df_2, pair)
        if signal and signal['bias']:
            signal.update(market_state)
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

            if pair in agent.open_trades.keys():
                real_qty = float(agent.open_trades[pair]['position']['base_size'])
                agent.real_pos[asset].update(
                    agent.update_pos(session, asset, real_qty, signal['inval_ratio'], 'real'))
                real_ep = float(agent.open_trades[pair]['position']['entry_price'])
                agent.real_pos[asset]['price_delta'] = (price - real_ep) / real_ep

            if pair in agent.sim_trades.keys():
                sim_qty = float(agent.sim_trades[pair]['position']['base_size'])
                new_stats = agent.update_pos(session, asset, sim_qty, signal['inval_ratio'], 'sim')
                agent.sim_pos[asset].update(new_stats)
                sim_ep = float(agent.sim_trades[pair]['position']['entry_price'])
                agent.sim_pos[asset]['price_delta'] = (price - sim_ep) / sim_ep

signal_counts = Counter([signal['bias'] for signal in raw_signals])
session.market_bias = (signal_counts.get('bullish', 0) - signal_counts.get('bearish', 0)) / len(raw_signals)

tech_end = time.perf_counter()
tech_took = tech_end - tech_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Process Raw Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
process_start = time.perf_counter()
print(f"\n-*-*-*- Processing {len(raw_signals)} Raw Signals for all agents -*-*-*-\n")

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
    sig_score = signal['inval_score']  # TODO i need to add other scores here and create a way of combining them

    # find whether i am currently long, short or flat on the agent and pair in this signal
    asset = sig_pair[:-len(session.quote_asset)]
    real_position = sig_agent.real_pos.get(asset, {'direction': 'flat'})['direction']
    sim_position = sig_agent.sim_pos.get(asset, {'direction': 'flat'})['direction']  # returns 'flat' if no position
    tracked_position = sig_agent.tracked.get(asset, {'direction': 'flat'})['direction']

    bullish_pos = 'spot' if (sig_agent.mode == 'spot') else 'long'

    # TODO every raw signal that goes through here must end up as at least 1 processed signal. i need to check this
    #  logic to make sure that there is a concrete outcome for every possible input.
    # TODO 'oco' signals are an alternative to 'open', so they should be treated as such in this section, maybe as a
    #  condition which requires that position is flat and signal['exit'] is oco and bias is either bullish or bearish

    if sig_bias == 'bullish':
        if real_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)
            # check if tp is necessary
            try: # this try/except bloack can be removed when the problem is solved
                if sig_agent.real_pos[asset]['or_R'] > sig_agent.indiv_r_limit:
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'real', 'long'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                traceback.print_stack()
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.real_pos[asset])

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'short':
            print(f"{sig_agent.real_pos[asset]['or_R'] = } {sig_agent.indiv_r_limit = }")
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'short'))
            if sig_score >= 0.5:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'long'))
            else:
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', 'long'))

        elif real_position == 'flat':
            if sig_score >= 0.5:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', bullish_pos))
            else:
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', bullish_pos))
        else:
            print("bullish bias didn't produce a tracked outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'sim')
            # check if tp is necessary
            try: # this try/except block can be removed when the problem is solved
                if sig_agent.sim_pos[asset]['or_R'] > sig_agent.indiv_r_limit:
                    print(f"{sig_agent.sim_pos[asset]['or_R'] = }")
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'sim', 'long'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                traceback.print_stack()
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.sim_pos[asset])

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
            try: # this try/except bloack can be removed when the problem is solved
                if sig_agent.tracked[asset]['or_R'] > sig_agent.indiv_r_limit:
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'tracked', 'long'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.tracked[asset])

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
            if sig_score >= 0.5:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
            else:
                signal['sim_reasons'] = ['low_score']
                processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', 'short'))
        elif real_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)
            # check if tp is necessary
            try: # this try/except bloack can be removed when the problem is solved
                print(f"{sig_agent.real_pos[asset]['or_R'] = } {sig_agent.indiv_r_limit = }")
                if sig_agent.real_pos[asset]['or_R'] > sig_agent.indiv_r_limit:
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'real', 'short'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                traceback.print_stack()
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.real_pos[asset])

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'flat':
            if sig_score >= 0.5:
                processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
            else:
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
            try: # this try/except bloack can be removed when the problem is solved
                if sig_agent.sim_pos[asset]['or_R'] > sig_agent.indiv_r_limit:
                    print(f"{sig_agent.sim_pos[asset]['or_R'] = }")
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'sim', 'short'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                traceback.print_stack()
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.sim_pos[asset])

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
            try: # this try/except bloack can be removed when the problem is solved
                if sig_agent.tracked[asset]['or_R'] > sig_agent.indiv_r_limit:
                    processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'tp', 'tracked', 'short'))
            except KeyError as e:
                print(f"KeyError for {asset}: {e.args}")
                traceback.print_stack()
                print('signal:')
                pprint(signal)
                print('pos record:')
                pprint(sig_agent.tracked[asset])


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
print(f"\n-+-+-+-+-+-+-+- Processing {len(processed_signals['real_sim_tp_close'])} Real/Sim TPs/Closes -+-+-+-+-+-+-+-")

# TODO start by checking the tp/close list for any tp signals with the same agent, pair and direction and state as close
#  signals, and discard the tp if found

wanted_spot = round(agent.realised_pnls['wanted_spot'], 1)
wanted_long = round(agent.realised_pnls['wanted_long'], 1)
wanted_short = round(agent.realised_pnls['wanted_short'], 1)
print(f"\n{agent.name} Realised PnLs: {wanted_spot = }R, {wanted_long = }R, {wanted_short = }R\n")

for signal in processed_signals['real_sim_tp_close']:
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
    # TODO i need to make sure the fr values that get recorded in the log are the fr score (integers from 0 to 10) not
    #  the actual fixed risk amount
    agent.set_fixed_risk(session)
    # agent.test_fixed_risk(0.0002, 0.0002) # TODO i will need to do some tests with this disabled to make sure fr works
    agent.print_fixed_risk()
    agent.calc_tor()

    wanted_spot = round(agent.realised_pnls['wanted_spot'], 1)
    wanted_long = round(agent.realised_pnls['wanted_long'], 1)
    wanted_short = round(agent.realised_pnls['wanted_short'], 1)
    print(f"\n{agent.name} Realised PnLs: {wanted_spot = }R, {wanted_long = }R, {wanted_short = }R\n")

    # TODO make sure algo_order counts are up to date after tps and closes
    # print(f"{agent.name} {agent.fixed_risk_spot} {agent.fixed_risk_l} {agent.fixed_risk_s}")

for signal in processed_signals['unassigned']:
    signal['base_size'], signal['quote_size'] = agents[signal['agent']].get_size(session, signal)

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Sort and Filter Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
sort_start = time.perf_counter()
# next sort the unassigned list by scores. items are popped from the end of the list so i want the best signals to be
# last so that they get processed first, so i don't use the 'reverse=True' option
unassigned = sorted(processed_signals['unassigned'], key=lambda x: x['inval_score'])
print(f"\n-*-*-*- Sorting and Filtering {len(unassigned)} Processed Signals for all agents -*-*-*-\n")

# work through the list and check each filter for each signal
or_limits = {agent.name: agent.total_open_risk for agent in agents.values()}
pos_limits = {agent.name: agent.num_open_positions for agent in agents.values()}
algo_limits = {pair: (v['max_algo_orders'] - v['algo_orders']) for pair, v in session.pairs_data.items()}
usdt_bal_s = session.spot_usdt_bal
session.check_margin_lvl()

while unassigned:
    s = unassigned.pop()

    # set variables
    agent = agents[s['agent']]
    sim_reasons = []
    balance = session.spot_bal if s['mode'] == 'spot' else session.margin_bal
    quote_size = s['quote_size']
    r = 1

    if s['direction'] in {'spot', 'long'}:
        usdt_depth, _ = funcs.get_depth(session, pair)
    elif s['direction'] == 'short':
        _, usdt_depth = funcs.get_depth(session, pair)

    if s['mode'] == 'margin' and not session.pairs_data[pair]['margin_allowed']:
        sim_reasons.append('not_a_margin_pair')

    if usdt_depth == 0:
        sim_reasons.append('too_much_spread')
    elif usdt_depth < (quote_size / 2):
        sim_reasons.append('books_too_thin')
    elif quote_size > usdt_depth >= (quote_size / 2):  # only trim size if books are a bit too thin
        r = quote_size / usdt_depth
        quote_size = usdt_depth
        print(f'{now} {pair} books too thin, reducing size from {quote_size:.3} to {usdt_depth:.3}')

    if quote_size > (balance * 0.1):
        r = quote_size / (balance * 0.1)
        quote_size = balance * 0.1

    if or_limits[agent.name] >= agent.total_r_limit:
        sim_reasons.append('too_much_or')
    if pos_limits[agent.name] >= agent.max_positions:
        sim_reasons.append('too_many_pos')
    if (algo_limits[pair] < 2 and s['action'] == 'oco') or (algo_limits[pair] < 1 and s['action'] == 'open'):
        sim_reasons.append('algo_order_limit')

    if s['mode'] == 'spot' and quote_size > usdt_bal_s > (quote_size / 2):
        r = quote_size / (usdt_bal_s - 1)
        quote_size = usdt_bal_s - 1
    elif s['mode'] == 'spot' and quote_size > usdt_bal_s:
        sim_reasons.append('not_enough_usdt')

    if s['mode'] == 'margin' and 3 <= session.margin_lvl < 4:
        r = 0.5
        quote_size = quote_size / 2
    elif s['mode'] == 'margin' and session.margin_lvl < 3:
        sim_reasons.append('margin_acct_too_levered')

    if quote_size < session.min_size: # this condition must come after all the conditions which could reduce size
        sim_reasons.append('too_small')

    # TODO need to work out how to do 'not_enough_borrow' in this section, or if that fails, a way to switch a real
    #  trade to sim at the borrowing stage

    if sim_reasons:
        s['sim_reasons'] = sim_reasons
        s['state'] = 'sim'
        s['pct_of_full_pos'] = r
        processed_signals['sim_open'].append(s)
    else:
        or_limits[agent.name] += r
        pos_limits[agent.name] += 1
        algo_limits[s['pair']] -= 2 if s['action'] == 'oco' else 1

        print(f"orig sizes: {s['quote_size']:.2f}, {s['base_size']}. new sizes: {quote_size:.2f}, {s['base_size'] * r}")
        s['quote_size'] = quote_size
        s['base_size'] = s['base_size'] * r # the value of r is equivalent to the change in size, if any.
        s['pct_of_full_pos'] = r

        processed_signals['real_open'].append(s)

sort_end = time.perf_counter()
sort_took = sort_end - sort_start
# pprint(processed_signals['real_open'])

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Real Open Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
real_open_start = time.perf_counter()

print(f"\n-+-+-+-+-+-+-+-+-+-+-+- Processing {len(processed_signals['real_open'])} Real Opens -+-+-+-+-+-+-+-+-+-+-+-")
for signal in processed_signals['real_open']:
    # print(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} {signal['direction']}")
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

# unlike the real open signals, i should use every trick i can to make this quick. either pass the list of signals and
# the sim_open omf to apply() or try to use parallelism in some way.

sim_opens = [sig for sig in processed_signals['sim_open'] # discard signals for existing sim positions
             if signal['pair'] not in agents[signal['agent']].sim_pos.keys()]
print(f"\n-+-+-+-+-+-+-+-+-+-+-+- Processing {len(sim_opens)} Sim Opens -+-+-+-+-+-+-+-+-+-+-+-")
for signal in sim_opens:
        # print(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} {signal['direction']}")
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

print(f"\n\nbias_indicator: {session.market_bias:.2f} range from 1 (bullish) to -1 (bearish) {signal_counts = }\n\n")

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

# for agent in agents.values():
#     print(f'{agent.name} real_pos')
#     pprint(agent.real_pos)
#     print('open_trades')
#     pprint(agent.open_trades)

session.save_spreads()

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

print(f"used-weight-1m: {client.response.headers['x-mbx-used-weight-1m']}")

# start_dt = datetime.fromtimestamp(session.all_weights[0][0])
# plot_df = pd.DataFrame({
#     'times': [time for time, weight in session.all_weights],
#     'weights': [weight for time, weight in session.all_weights]})
# plot_df['seconds'] = plot_df.times - plot_df.times.iloc[0]
# plot_df['cum_weight'] = plot_df.weights.cumsum()
# plot_df = plot_df.drop('weights', axis=1)
# fig = px.scatter(plot_df, x='seconds', y='cum_weight')
# fig.show()

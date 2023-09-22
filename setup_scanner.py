import time
from resources import utility_funcs as uf, binance_funcs as funcs
from datetime import datetime, timezone
from agents import TrailFractals
from pprint import pformat
import sessions
from resources.timers import Timer
from collections import Counter
from resources.loggers import create_logger

script_start = time.perf_counter()

# import update_ohlc

# TODO current (02/04/23) roadmap should be:
#  * start integrating polars and doing anything else i can to speed things up enough to run 3day and 1week timeframes
#  in the same session as everything else
#  * get oco entries and trade adds working so i can use other strats

logger = create_logger('setup_scanner')

########################################################################################################################

session = sessions.TradingSession(0.1)  # this argument is now max position size rather than max fixed risk

logger.debug(f'-+-+-+-+-+-+-+-+ {session.now_start} Running Setup Scanner ({session.timeframes}) +-+-+-+-+-+-+-+-')

# initialise agents
agents = []
for timeframe, offset in session.timeframes:
    agents.extend(
        [
            TrailFractals(session, timeframe, offset, '1d_volumes', 30),
            TrailFractals(session, timeframe, offset, '1w_volumes', 100),
        ]
    )

agents = {a.id: a for a in agents}

# session.name = ' | '.join([n.name for n in agents.values()])

logger.debug("-*-*-*- Checking all positions for stops and open-risk -*-*-*-")
logger.info("-*-*-*- Checking all positions for stops and open-risk -*-*-*-")

real_sim_tps_closes = []
for agent in agents.values():
    agent.record_stopped_trades(session, session.timeframes)
    agent.record_stopped_sim_trades(session, session.timeframes)
    real_sim_tps_closes.extend(agent.check_open_risk(session))
    agent.max_positions = agent.set_max_pos()
    agent.total_r_limit = agent.max_positions * 1.7  # TODO need to update reduce_risk and run it before/after set_fixed_risk

init_end = time.perf_counter()
init_elapsed = init_end - script_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Generate Technical Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
logger.debug("-*-*-*- Generating Technical Signals for all agents -*-*-*-")
logger.info("-*-*-*- Generating Technical Signals for all agents -*-*-*-")
tech_start = time.perf_counter()

raw_signals = []

for n, pair in enumerate(session.pairs_set):
    # logger.debug('-' * 100)
    # logger.debug(n, pair)
    session.update_prices()

    now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')

    # df_dict contains ohlc dataframes for each active timeframe for the current pair
    df_dict = funcs.prepare_ohlc(session, session.timeframes, pair)

    # if there is not enough history at a given timeframe, this function will return None instead of the df
    # TODO this would be a good function to start the migration to polars
    if not df_dict:
        continue

    for tf, df in df_dict.items():
        # df_dict[tf] = session.compute_indicators(df, tf)
        df_dict[tf] = session.compute_features(df, tf)

    for agent in agents.values():
        if agent.tf not in df_dict:  # some pairs will not have all required timeframes for the session
            continue
        # logger.debug(f"{agent.name}")
        df_2 = df_dict[agent.tf].copy()
        # logger.debug(f"{pair} {tf} {len(df_2)}")

        signal = agent.signals(session, df_2, pair)
        if signal.get('bias'):
            raw_signals.append(signal)

        # if this agent is in position with this pair, calculate open risk and related metrics here
        # update positions dictionary with current pair's open_risk values ------------
        price = df_2.close.iloc[-1]
        asset = pair[:-1 * len(session.quote_asset)]

        real_match = (asset in agent.real_pos.keys()) == (pair in agent.open_trades.keys())
        sim_match = (asset in agent.sim_pos.keys()) == (pair in agent.sim_trades.keys())
        if not real_match:
            logger.error(f"{pair} real pos doesn't match real trades")
        if not sim_match:
            logger.error(f"{pair} sim pos doesn't match sim trades")

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
logger.debug(f"-*-*-*- Processing {len(raw_signals)} Raw Signals for all agents -*-*-*-")
logger.info(f"-*-*-*- Processing {len(raw_signals)} Raw Signals for all agents -*-*-*-")

# gather data on current algo orders
session.update_algo_orders()

processed_signals = dict(
    real_sim_tp_close=real_sim_tps_closes,  # all real and sim tps and closes go in here for immediate execution
    unassigned=[],  # all real open signals go in here for further selection
    scored=[],  # once a score has been calculated for the unassigned signals, any with a good score go in here
    sim_open=[],  # nothing will be put in here at first but many real open signals will end up in here
    real_open=[],  # nothing will be put in here at first, real open signals will end up here if they pass all tests
    tracked_close=[],  # can be left until last
)

while raw_signals:
    signal = raw_signals.pop(0)

    sig_bias = signal['bias']
    sig_agent = agents[signal['agent']]
    sig_pair = signal['pair']

    # find whether I am currently long, short or flat on the agent and pair in this signal
    real_position = sig_agent.real_pos.get(signal['asset'], {'direction': 'flat'})['direction']
    sim_position = sig_agent.sim_pos.get(signal['asset'], {'direction': 'flat'})['direction']  # returns 'flat' if none
    tracked_position = sig_agent.tracked.get(signal['asset'], {'direction': 'flat'})['direction']

    bullish_pos = 'spot' if (sig_agent.mode == 'spot') else 'long'

    # TODO 'oco' signals are an alternative to 'open', so they should be treated as such in this section, maybe as a
    #  condition which requires that position is flat and signal['exit'] is oco and bias is either bullish or bearish

    # TODO i need to add conditions so that strategies which trail stops don't have their positions closed by a bias
    #  flip, only new positions will be opened on  signals. so i might end up with long and short positions open
    #  simultaneously, but that's ok because each position will be managed and the price will decide which should stay
    #  open

    # TODO perhaps there should be a flag in each signal that says whether that agent's positions should be closed with
    #  signals or not, because oco orders should never be managed by signals but trailing stop strategies sometimes have
    #  close/tp conditions and sometimes don't

    if sig_bias == 'bullish':
        if real_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'short'))
            processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'long'))

        elif real_position == 'flat':
            processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', bullish_pos))
        else:
            logger.error("bullish bias didn't produce a tracked outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'sim')

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif sim_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', 'short'))
        elif sim_position == 'flat':
            pass
        else:
            logger.error("bullish bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'tracked')

        elif tracked_position == 'short':
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', 'short'))
        elif tracked_position == 'flat':
            pass
        else:
            logger.error("bullish bias didn't produce a tracked outcome, logic needs more work")

    # -------------------------------------------------------------------------------------------------------------------

    elif sig_bias == 'bearish':
        if real_position == 'spot':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'spot'))
        elif real_position == 'long':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'long'))
            processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
        elif real_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_real_stop(session, signal)

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif real_position == 'flat':
            processed_signals['unassigned'].append(uf.transform_signal(signal, 'open', 'real', 'short'))
        else:
            logger.error("bearish bias didn't produce a real outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', bullish_pos))
        elif sim_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'sim')

            # check if add is necessary
            # TODO check for low or and make add signals if so

        elif sim_position == 'flat':
            pass
        else:
            logger.error("bearish bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', bullish_pos))
        elif tracked_position == 'short':
            if sig_agent.trail_stop:
                sig_agent.move_non_real_stop(session, signal, 'tracked')

        elif tracked_position == 'flat':
            pass
        else:
            logger.error("bearish bias didn't produce a tracked outcome, logic needs more work")

    # -------------------------------------------------------------------------------------------------------------------

    elif sig_bias == 'neutral':
        if real_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', bullish_pos))
        elif real_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'real', 'short'))
        elif real_position == 'flat':
            pass
        else:
            logger.error("neutral bias didn't produce a real outcome, logic needs more work")

        if sim_position in ['long', 'spot']:
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', bullish_pos))
        elif sim_position == 'short':
            processed_signals['real_sim_tp_close'].append(uf.transform_signal(signal, 'close', 'sim', 'short'))
        elif sim_position == 'flat':
            pass
        else:
            logger.error("neutral bias didn't produce a sim outcome, logic needs more work")

        if tracked_position in ['long', 'spot']:
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', bullish_pos))
        elif tracked_position == 'short':
            processed_signals['tracked_close'].append(uf.transform_signal(signal, 'close', 'tracked', 'short'))
        elif tracked_position == 'flat':
            pass
        else:
            logger.error("neutral bias didn't produce a tracked outcome, logic needs more work")

    else:
        logger.error(f"{sig_agent} signal had no bias")
        logger.error(pformat(signal))

process_end = time.perf_counter()
process_took = process_end - process_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute TP / Close Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
tp_close_start = time.perf_counter()
# execute any real and sim technical close and tp signals
logger.debug(f"-+-+-+-+-+-+- Executing {len(processed_signals['real_sim_tp_close'])} Real/Sim TPs/Closes -+-+-+-+-+-+-")
logger.info(f"-+-+-+-+-+-+- Executing {len(processed_signals['real_sim_tp_close'])} Real/Sim TPs/Closes -+-+-+-+-+-+-")

checked_signals = uf.remove_duplicates(processed_signals['real_sim_tp_close'])
logger.debug(f"{len(checked_signals) = }")

for signal in checked_signals:
    logger.debug('')
    logger.debug(f"Executing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} "
                 f"{signal['direction']}")
    logger.info(f"\nExecuting {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} "
                f"{signal['direction']}")
    if signal['action'] in ['close', 'stop']:  # TODO stop signals could be added in here
        agents[signal['agent']].close_pos(session, signal)
    elif signal['action'] == 'tp':
        agents[signal['agent']].tp_pos(session, signal)
    else:
        logger.error("problem with signal in tps/closes")
        logger.error(pformat(signal))

tp_close_end = time.perf_counter()
tp_close_took = tp_close_end - tp_close_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Calculate Fixed Risk/Signal Scores
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# calculate fixed risk for each agent using wanted rpnl

logger.info(f"\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+- Calculating Risk Scalars -+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")
logger.info("these values have been clipped and normalised, so 0.5 is equivalent to break-even, 0 is the lowest "
            "possible value and 1 is the highest possible value")

for agent in agents.values():
    agent.pnls = dict(
        spot=agent.get_pnls('spot'),
        long=agent.get_pnls('long'),
        short=agent.get_pnls('short'),
    )
    logger.info(f"\n{agent.id} scaled pnls")
    if agent.mode == 'margin':
        logger.info("Long:")
        logger.info(f"EMA4: {agent.pnls['long']['ema_4']:.2f}, EMA8: {agent.pnls['long']['ema_8']:.2f}, "
                    f"EMA16: {agent.pnls['long']['ema_16']:.2f}, EMA32: {agent.pnls['long']['ema_32']:.2f}, "
                    f"EMA64: {agent.pnls['long']['ema_64']:.2f}")
        logger.info("Short:")
        logger.info(f"EMA4: {agent.pnls['short']['ema_4']:.2f}, EMA8: {agent.pnls['short']['ema_8']:.2f}, "
                    f"EMA16: {agent.pnls['short']['ema_16']:.2f}, EMA32: {agent.pnls['short']['ema_32']:.2f}, "
                    f"EMA64: {agent.pnls['short']['ema_64']:.2f}")
    else:
        logger.info("Spot:")
        logger.info(f"EMA4: {agent.pnls['spot']['ema_4']:.2f}, EMA8: {agent.pnls['spot']['ema_8']:.2f}, "
                    f"EMA16: {agent.pnls['spot']['ema_16']:.2f}, EMA32: {agent.pnls['spot']['ema_32']:.2f}, "
                    f"EMA64: {agent.pnls['spot']['ema_64']:.2f}")

while processed_signals['unassigned']:
    signal = processed_signals['unassigned'].pop()

    # when the secondary ml model is ready, it will replace the contents of this while-loop down to sig_score. I will
    # simply pass the inval distance (0-1, 1 being 100% between entry and init stop), 5 perf_emas, the 3 market_ranks,
    # and confidence numbers from however many ml models have made a prediction (2 currently), and the model will return
    # a signal score. I record them all in the signal record, calculate size, and if the score is too low, move the
    # signal over to sim_opens

    signal['perf_ema4'] = agents[signal['agent']].pnls[signal['direction']]['ema_4']
    signal['perf_ema8'] = agents[signal['agent']].pnls[signal['direction']]['ema_8']
    signal['perf_ema16'] = agents[signal['agent']].pnls[signal['direction']]['ema_16']
    signal['perf_ema32'] = agents[signal['agent']].pnls[signal['direction']]['ema_32']
    signal['perf_ema64'] = agents[signal['agent']].pnls[signal['direction']]['ema_64']

    sig_bias = signal['bias']

    perf_score, rank_score = 0, 0
    if signal['tf'] == '1h':
        perf_score = ((signal['perf_ema64'] > 0.5) + (signal['perf_ema32'] > 0.5) + (signal['perf_ema16'] > 0.5)) / 3
        rank_score = signal.get('market_rank_1d', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1d', 1))
    elif signal['tf'] in {'4h', '12h'}:
        perf_score = ((signal['perf_ema32'] > 0.5) + (signal['perf_ema16'] > 0.5) + (signal['perf_ema8'] > 0.5)) / 3
        rank_score = signal.get('market_rank_1w', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1w', 1))
    elif signal['tf'] == '1d':
        perf_score = ((signal['perf_ema16'] > 0.5) + (signal['perf_ema8'] > 0.5) + (signal['perf_ema4'] > 0.5)) / 3
        rank_score = signal.get('market_rank_1m', 1) if sig_bias == 'bullish' else (1 - signal.get('market_rank_1m', 1))

    if not session.live:
        perf_score = 1.0

    # TODO don't forget inval_score
    inval_scalar = 1 + abs(1 - signal['inval_ratio'])
    sig_score = signal['confidence'] * rank_score
    risk_scalar = (sig_score * perf_score) / inval_scalar
    signal['score'] = sig_score
    signal['risk_scalar'] = risk_scalar

    score_threshold = 0.3
    if sig_score > score_threshold:
        logger.info('')
        logger.info(f"{signal['pair']}, {signal['tf']}, {signal['direction']}")
        logger.info(f"risk scalar: {risk_scalar:.1%}, signal score: {sig_score:.1%}, "
                    f"conf: {signal['confidence']:.1%}, rank: {rank_score:.1%}, perf: {perf_score:.1%}")

    signal['base_size'], signal['quote_size'] = agents[signal['agent']].get_size(session, signal)

    sim_position = agents[signal['agent']].sim_pos.get(signal['asset'], {'direction': 'flat'})['direction']
    if signal['score'] >= score_threshold:
        processed_signals['scored'].append(signal)
    # separate unwanted signals
    elif signal['score'] < score_threshold and sim_position == 'flat':
        signal['sim_reasons'] = ['low_score']
        # sig_direction = 'long' if signal['bias'] == 'bullish' else 'short'
        processed_signals['sim_open'].append(uf.transform_signal(signal, 'open', 'sim', signal['direction']))

logger.info(f"\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+- Calculating Fixed Risk -+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")
logger.debug(f"-+-+-+-+-+-+-+-+-+-+-+-+-+-+- Calculating Fixed Risk -+-+-+-+-+-+-+-+-+-+-+-+-+-+-")

for agent in agents.values():
    agent.calc_tor()

    wanted_spot = round(agent.realised_pnls['wanted_spot'], 1)
    wanted_long = round(agent.realised_pnls['wanted_long'], 1)
    wanted_short = round(agent.realised_pnls['wanted_short'], 1)
    session.wrpnl_totals['spot'] += wanted_spot
    session.wrpnl_totals['long'] += wanted_long
    session.wrpnl_totals['short'] += wanted_short

    unwanted_spot = round(agent.realised_pnls['unwanted_spot'], 1)
    unwanted_long = round(agent.realised_pnls['unwanted_long'], 1)
    unwanted_short = round(agent.realised_pnls['unwanted_short'], 1)
    session.urpnl_totals['spot'] += unwanted_spot
    session.urpnl_totals['long'] += unwanted_long
    session.urpnl_totals['short'] += unwanted_short

logger.info('wanted rpnl totals:')
logger.info(pformat(session.wrpnl_totals))
logger.info('unwanted rpnl totals:')
logger.info(pformat(session.urpnl_totals))

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Sort and Filter Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
sort_start = time.perf_counter()

# gather data on current algo orders
session.update_algo_orders()

# next sort the unassigned list by scores. items are popped from the end of the list, so I want the best signals to be
# last so that they get processed first, so I don't use the 'reverse=True' option
unassigned = sorted(processed_signals['scored'], key=lambda x: x['score'])
logger.info(f"\n-*-*-*- Sorting and Filtering {len(unassigned)} Processed Signals for all agents -*-*-*-\n")
logger.debug(f"-*-*-*- Sorting and Filtering {len(unassigned)} Processed Signals for all agents -*-*-*-")

# work through the list and check each filter for each signal
or_limits = {agent.name: agent.total_open_risk for agent in agents.values()}
pos_limits = {agent.name: agent.num_open_positions for agent in agents.values()}
algo_limits = {pair: (v['max_algo_orders'] - v['algo_orders']) for pair, v in session.pairs_data.items()}
usdt_bal_s = session.spot_usdt_bal['qty']

while unassigned:
    logger.debug(f"{len(unassigned)} unassigned signals left to process")
    s = unassigned.pop()

    # set variables
    agent = agents[s['agent']]
    sim_reasons = []
    balance = session.spot_bal if s['mode'] == 'spot' else session.margin_bal
    quote_size = s['quote_size']
    r = 1

    if s['mode'] == 'margin' and not session.pairs_data[s['pair']]['margin_allowed']:
        sim_reasons.append('not_a_margin_pair')

    if s['direction'] in {'spot', 'long'}:
        usdt_depth, _ = funcs.get_depth(session, s['pair'])
    else:
        _, usdt_depth = funcs.get_depth(session, s['pair'])

    if usdt_depth == 0:
        sim_reasons.append('too_much_spread')
    elif usdt_depth < (quote_size / 2):
        sim_reasons.append('books_too_thin')
    elif quote_size > usdt_depth >= (quote_size / 2):  # only trim size if books are a bit too thin
        r = usdt_depth / quote_size
        quote_size = usdt_depth
        logger.info(f"{s['pair']} books too thin, reducing size from {quote_size:.3} to {usdt_depth:.3}")

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

    if quote_size < session.min_size:  # this condition must come after all the conditions which could reduce size
        sim_reasons.append('too_small')

    if s['direction'] == 'short' and not sim_reasons:
        max_borrow = funcs.get_max_borrow(session, s['asset'])
        if float(s['base_size']) > max_borrow * 0.9:
            sim_reasons.append('not_enough_borrow')

    if sim_reasons:
        s['sim_reasons'] = sim_reasons
        s['state'] = 'sim'
        s['pct_of_full_pos'] *= r
        processed_signals['sim_open'].append(s)
        logger.info(f"{s['pair']} {sim_reasons}, {s['quote_size']:.2f}USDT")
    else:
        # TODO since i'm moving away from fixed risk, i should add a check here which makes sure pfrd isn't too much,
        #  maybe scale the position down if it is
        or_limits[agent.name] += r
        pos_limits[agent.name] += 1
        algo_limits[s['pair']] -= 2 if s['action'] == 'oco' else 1

        if r != 1:
            logger.info(f"{s['agent']} {s['pair']} {r = }, size adjusted:")
            logger.info(f"orig: {s['quote_size']:.2f}, {s['base_size']}. new: {quote_size:.2f}, {s['base_size'] * r}")
        s['quote_size'] = quote_size
        s['base_size'] = s['base_size'] * r  # the value of r is equivalent to the change in size, if any.
        s['pct_of_full_pos'] *= r

        processed_signals['real_open'].append(s)
        logger.info(f"{s['pair']} {s['tf']} real open signal, {s['quote_size']:.2f}USDT")

sort_end = time.perf_counter()
sort_took = sort_end - sort_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Real Open Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
real_open_start = time.perf_counter()

logger.info(f"\n-+-+-+-+-+-+-+-+-+- Executing {len(processed_signals['real_open'])} Real Opens -+-+-+-+-+-+-+-+-+-\n")
logger.debug(f"-+-+-+-+-+-+-+-+-+- Executing {len(processed_signals['real_open'])} Real Opens -+-+-+-+-+-+-+-+-+-")

remaining_borrow = session.check_margin_lvl()

for signal in processed_signals['real_open']:
    if signal['mode'] == 'spot':
        agents[signal['agent']].open_real_s(session, signal, 0)

    elif signal['quote_size'] > (remaining_borrow - 100):
        signal['state'] = 'sim'
        signal['sim_reasons'] = ['too_much_leverage']
        processed_signals['sim_open'].append(signal)
        logger.info("changed real open signal to sim, borrow limit reached")
        # now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
        # pb.push_note(now, 'Margin limit reached, maybe add collateral')

    else:
        logger.info(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} "
                    f"{signal['direction']}")
        successful = agents[signal['agent']].open_real_M(session, signal, 0)
        if successful:
            remaining_borrow -= signal['quote_size']
            logger.info(f"remaining_borrow: {remaining_borrow:.2f}")

# when they are all finished, update records once

real_open_end = time.perf_counter()
real_open_took = real_open_end - real_open_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Sim Open Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
sim_open_start = time.perf_counter()

sim_opens = [sig for sig in processed_signals['sim_open']  # discard signals for existing sim positions
             if sig['asset'] not in agents[sig['agent']].sim_pos.keys()]

logger.info(f"\n-+-+-+-+-+-+-+-+-+-+-+- Executing {len(sim_opens)} Sim Opens -+-+-+-+-+-+-+-+-+-+-+-\n")
logger.debug(f"-+-+-+-+-+-+-+-+-+-+-+- Executing {len(sim_opens)} Sim Opens -+-+-+-+-+-+-+-+-+-+-+-")

for signal in sim_opens:
    # logger.debug(f"Processing {signal['agent']} {signal['pair']} {signal['action']} {signal['state']} "
    #              f"{signal['direction']}")
    # logger.debug(f"Sim reason: {signal['sim_reasons']}, score: {signal['score']}")
    agents[signal['agent']].open_sim(session, signal)

# when they are all finished, update records once

sim_open_end = time.perf_counter()
sim_open_took = sim_open_end - sim_open_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Execute Tracked Close Signals
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# probably won't need anything too clever since I don't think these come up very often

# when they are all finished, update records once

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Logs and Summaries
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
log_start = time.perf_counter()

session.get_usdt_m()
session.get_usdt_s()

logger.debug('-:-' * 20)
logger.info('-:-' * 20)

for agent in agents.values():
    agent.record_trades(session, 'all')

    #################################

    logger.info(f"\n{agent.name.upper()} SUMMARY")

    logger.info(f"\n{len(agent.real_pos.keys())} real positions, {len(agent.sim_pos.keys())} sim positions\n")

    if agent.realised_pnls['real_spot'] or agent.realised_pnls['sim_spot']:
        logger.info(f"realised real spot pnl: {agent.realised_pnls['real_spot']:.1f}R, "
                    f"realised sim spot pnl: {agent.agent.realised_pnls['sim_spot']:.1f}R")

    if agent.realised_pnls['real_long'] or agent.realised_pnls['sim_long']:
        logger.info(f"realised real long pnl: {agent.realised_pnls['real_long']:.1f}R, "
                    f"realised sim long pnl: {agent.realised_pnls['sim_long']:.1f}R")

    if agent.realised_pnls['real_short'] or agent.realised_pnls['sim_short']:
        logger.info(f"realised real short pnl: {agent.realised_pnls['real_short']:.1f}R, "
                    f"realised sim short pnl: {agent.realised_pnls['sim_short']:.1f}R")
    logger.info(f'tor: {agent.total_open_risk:.1f}')

    logger.info(f'\n{agent.name} Counts:')
    logger.info(pformat({f"{k}: {v}" for k, v in agent.counts_dict.items() if v}))
    logger.info('-:-' * 20)

    ################################

    usdt_bal = session.spot_usdt_bal if agent.mode == 'spot' else session.margin_usdt_bal
    agent.real_pos['USDT'] = usdt_bal

if not session.live:
    logger.warning('warning: logging directed to test_records')

for tf in session.timeframes:
    logger.info(f"\n{tf[0]} market bias: {session.market_bias[tf[0]]:.2f} range from 1 (bullish) to -1 (bearish)")

uf.market_benchmark(session)
for agent in agents.values():
    uf.log(session, agent)
    agent.benchmark = uf.strat_benchmark(session, agent)
uf.scanner_summary(session, list(agents.values()))

log_end = time.perf_counter()
log_elapsed = log_end - log_start

# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# End
# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

logger.info('\n-------------------- Timers --------------------\n')
for k, v in Timer.timers.items():
    if v > 10:
        elapsed = f"{k}: {int(v // 60)}m {v % 60:.1f}s"
        logger.info(elapsed)

logger.info('\n-------------------- Counts --------------------\n')
logger.info(f"pairs tested: {len(session.pairs_set)}")
logger.info(pformat(Counter(session.counts)))
logger.info(f"used-weight: {session.client.response.headers.get('x-mbx-used-weight')}")
logger.info(f"used-weight-1m: {session.client.response.headers.get('x-mbx-used-weight-1m')}")
logger.debug('-----------------------------------------------\n')
logger.info('\n-----------------------------------------------\n')

script_end = time.perf_counter()
total_time = script_end - script_start


def section_times():
    logger.info('Scanner finished')
    logger.info(f"Initialisation took: {int(init_elapsed // 60)}m {int(init_elapsed % 60)}s")
    logger.info(f"Generating Technical Signals took: {int(tech_took // 60)}m {int(tech_took % 60)}s")
    logger.info(f"Processing Technical Signals took: {int(process_took // 60)}m {int(process_took % 60)}s")
    logger.info(f"Executing TP/Close Signals took: {int(tp_close_took // 60)}m {int(tp_close_took % 60)}s")
    logger.info(f"Sorting Technical Signals took: {int(sort_took // 60)}m {int(sort_took % 60)}s")
    logger.info(f"Executing Real Open Signals took: {int(real_open_took // 60)}m {int(real_open_took % 60)}s")
    logger.info(f"Executing Sim Open Signals took: {int(sim_open_took // 60)}m {int(sim_open_took % 60)}s")
    logger.info(f"Logging took: {int(log_elapsed // 60)}m {int(log_elapsed % 60)}s")
    logger.info(f"Total time taken: {int(total_time // 60)}m {int(total_time % 60)}s")


section_times()

# uf.plot_call_weights(session)

logger.info(
    '\n<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>\n\n')
logger.debug(
    '<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>-<=>\n\n')

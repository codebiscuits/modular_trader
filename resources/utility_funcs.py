import json
from json.decoder import JSONDecodeError
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
import statistics as stats
from pushbullet import Pushbullet
import requests
from pprint import pprint
from decimal import Decimal, getcontext
from resources.timers import Timer
from typing import Tuple, Dict
import sys
import math
import pytz
import time
from functools import wraps
from binance.exceptions import BinanceAPIException
import plotly.express as px

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12


def transform_signal(signal: dict, type: str, state: str, direction: str) -> dict:
    """takes a raw signal as input, returns a 'processed' signal ready to be scored and passed to an omf"""

    if type == 'close':
        return {'agent': signal['agent'],
                'pair': signal['pair'],
                'action': 'close',
                'direction': direction,
                'state': state,
                'mode': signal['mode']}

    elif type == 'open':
        signal['action'] = 'open'
        signal['direction'] = direction
        signal['state'] = state
        return signal

    elif type == 'oco':
        signal['action'] = 'oco'
        signal['direction'] = direction
        signal['state'] = state
        return signal

    elif type == 'tp':
        signal['action'] = 'tp'
        signal['direction'] = direction
        signal['state'] = state
        return signal

    elif type == 'add':
        signal['action'] = 'add'
        signal['direction'] = direction
        signal['state'] = state
        return signal


def step_round(num: float, step: str) -> str:
    """rounds down to any step size"""
    gv = Timer('step_round')
    gv.start()
    if not float(step):
        return str(num)
    num = Decimal(num)
    step = Decimal(step)
    gv.stop()
    return str(math.floor(num / step) * step)


def valid_size(session, pair: str, size: float) -> str:
    """rounds the desired order size to the correct step size for *MARKET ORDERS* on binance"""

    gf = Timer('get_symbol_info valid')
    gf.start()

    step_size = session.pairs_data[pair]['lot_step_size']

    gf.stop()
    return step_round(size, step_size)


def valid_price(session, pair: str, price: float) -> str:
    """rounds the desired order price to the correct step size for binance"""

    hg = Timer('get_symbol_info valid')
    hg.start()

    tick_size = session.pairs_data[pair]['price_tick_size']

    hg.stop()
    return step_round(price, tick_size)


def adjust_max_positions(max_pos: int, sizing: dict) -> int:
    '''the max_pos input tells the function what the strategy has as a default
    the sizing input is the dictionary of currently open positions with their 
    associated open risk
    
    this function decides if there should currently be a limit on how many positions
    can be open at once, based on current performance of currently open positions'''

    pass


def get_market_state(session, agent, pair, data: pd.DataFrame) -> dict[str, float]:

    ema_200_ratio = data.close.iloc[-1] / data.ema_200.iloc[-1]
    ema_100_ratio = data.close.iloc[-1] / data.ema_100.iloc[-1]
    ema_50_ratio = data.close.iloc[-1] / data.ema_50.iloc[-1]
    ema_25_ratio = data.close.iloc[-1] / data.ema_25.iloc[-1]

    try:
        market_rank_1d = session.market_ranks.at[pair, 'rank_1d']
        market_rank_1w = session.market_ranks.at[pair, 'rank_1w']
        market_rank_1m = session.market_ranks.at[pair, 'rank_1m']
    except KeyError:
        market_rank_1d = 1
        market_rank_1w = 1
        market_rank_1m = 1

    if hasattr(agent, 'cross_age_name'):
        cross_age = int(data[agent.cross_age_name].iloc[-1])
    else:
        cross_age = None


    return dict(
        ema_200=data.ema_200.iloc[-1],
        ema_100=data.ema_100.iloc[-1],
        ema_50=data.ema_50.iloc[-1],
        ema_25=data.ema_25.iloc[-1],
        ema_200_ratio=ema_200_ratio,
        ema_100_ratio=ema_100_ratio,
        ema_50_ratio=ema_50_ratio,
        ema_25_ratio=ema_25_ratio,
        vol_delta=data.vol_delta.iloc[-1],
        vol_delta_div=bool(data.vol_delta_div.iloc[-1]),
        vwma_24=data.vwma_24.iloc[-1],
        roc_1d=data.roc_1d.iloc[-1],
        roc_1w=data.roc_1w.iloc[-1],
        roc_1m=data.roc_1m.iloc[-1],
        market_rank_1d=market_rank_1d,
        market_rank_1w=market_rank_1w,
        market_rank_1m=market_rank_1m,
        cross_age=cross_age
    )


def market_benchmark(session) -> None:
    '''calculates daily, weekly and monthly returns for btc, eth and the median 
    altcoin on binance'''

    func_name = sys._getframe().f_code.co_name
    x20 = Timer(f'{func_name}')
    x20.start()

    all_1d = []
    all_1w = []
    all_1m = []
    btc_1d = None
    btc_1w = None
    btc_1m = None
    eth_1d = None
    eth_1w = None
    eth_1m = None

    # data = session.ohlc_data.glob('*.*')

    for x in session.pairs_data.keys():
        # if x.suffix != '.pkl':
        #     continue
        # df = pd.read_pickle(x)

        try:
            df = session.pairs_data[x]['ohlc_5m']
        except KeyError as e:
            print(f"market benchmark: couldn't get any ohlc data for {x}")
            continue

        if len(df) > 8929:  # 1 month of 5min periods is 31 * 24 * 12 = 8928
            df = df.tail(8929).reset_index()
        last_stamp = df.timestamp.iloc[-1]
        now = datetime.now(timezone.utc)
        window = timedelta(hours=4)
        if last_stamp > (now - window):  # if there is data up to the last 4 hours
            if len(df) > 288:
                df['roc_1d'] = df.close.pct_change(288)
                all_1d.append(df.at[df.index[-1], 'roc_1d'])
            else:
                print(f"market_benchmark: {x} ohlc data not long enough to be included")
            if len(df) > 2016:
                df['roc_1w'] = df.close.pct_change(2016)
                all_1w.append(df.at[df.index[-1], 'roc_1w'])
            if len(df) > 8928:
                df['roc_1m'] = df.close.pct_change(8928)
                all_1m.append(df.at[df.index[-1], 'roc_1m'])


            if x == 'BTCUSDT':
                btc_1d = df.roc_1d.iloc[-1]
                btc_1w = df.roc_1w.iloc[-1]
                btc_1m = df.roc_1m.iloc[-1]
                print(f"btc benchmark scores: {btc_1d:.1%}, {btc_1w:.1%}, {btc_1m:.1%}")
            elif x == 'ETHUSDT':
                eth_1d = df.roc_1d.iloc[-1]
                eth_1w = df.roc_1w.iloc[-1]
                eth_1m = df.roc_1m.iloc[-1]
                print(f"eth benchmark scores: {eth_1d:.1%}, {eth_1w:.1%}, {eth_1m:.1%}")
    market_1d = stats.median(all_1d) if len(all_1d) > 3 else 0
    market_1w = stats.median(all_1w) if len(all_1w) > 3 else 0
    market_1m = stats.median(all_1m) if len(all_1m) > 3 else 0
    print(f'1d median based on {len(all_1d)} data points')
    print(f'1w median based on {len(all_1w)} data points')
    print(f'1m median based on {len(all_1m)} data points')

    all_pairs = len(set(session.pairs_data.keys()))
    valid_pairs = len(all_1d)
    if valid_pairs > 3:
        valid = True
        if (all_pairs / valid_pairs) > 1.5:
            print(f'warning (market benchmark): lots of pairs ohlc data not up to date: {all_pairs = }, {len(all_1d) = }')
    else:
        valid = False

    # if session.live:
    #     print(f'pairs with recent data: {len(all_1d)} / {all_pairs}')

    x20.stop()

    session.benchmark = {'btc_1d': btc_1d, 'btc_1w': btc_1w, 'btc_1m': btc_1m,
                         'eth_1d': eth_1d, 'eth_1w': eth_1w, 'eth_1m': eth_1m,
                         'market_1d': market_1d, 'market_1w': market_1w, 'market_1m': market_1m,
                         'valid': valid}


def strat_benchmark(session, agent) -> None:
    '''calculates daily, weekly and monthly returns for the agent in question'''

    bal_now = session.spot_bal if agent.mode == 'spot' else session.margin_bal
    bal_1d, bal_1w, bal_1m = None, None, None

    filepath = Path(f"{session.read_records}/{agent.id}/perf_log.json")
    try:
        with open(filepath, 'r') as rec_file:
            bal_data = json.load(rec_file)
    except (FileNotFoundError, JSONDecodeError):
        bal_data = {}

    benchmark = dict(strat_1d=0, strat_1w=0, strat_1m=0)
    if not bal_data:
        return benchmark

    now = datetime.now(timezone.utc)
    for row in bal_data[::-1]:
        row_dt = pytz.timezone('UTC').localize(datetime.strptime(row.get('timestamp'), '%d/%m/%y %H:%M'))
        if row_dt > (now - timedelta(days=30)):  # and not bal_1m:
            try:
                bal_1m = row.get('balance')
            except AttributeError:
                print("strat_benchmark bal_1m produced attribute error")
                continue
        if row_dt > (now - timedelta(days=7)):  # and not bal_1w:
            try:
                bal_1w = row.get('balance')
            except AttributeError:
                print("strat_benchmark bal_1w produced attribute error")
                continue
        if row_dt > (now - timedelta(days=1)):  # and not bal_1d:
            try:
                bal_1d = row.get('balance')
            except AttributeError:
                print("strat_benchmark bal_1d produced attribute error")
                continue

    if bal_1d:
        benchmark['strat_1d'] = (bal_now - bal_1d) / bal_1d
    if bal_1w:
        benchmark['strat_1w'] = (bal_now - bal_1w) / bal_1w
    if bal_1m:
        benchmark['strat_1m'] = (bal_now - bal_1m) / bal_1m

    return benchmark


def log(session, agent) -> None:
    '''records all data from the session as a line in the perf_log.json file'''

    new_record = {'timestamp': session.now_start,
                  'positions': agent.real_pos, 'trade_counts': agent.counts_dict,
                  'median_spread': stats.median(session.spreads.values()),
                  'quote_asset': session.quote_asset, 'fr_max': session.fr_max,
                  'max_spread': session.max_spread, 'indiv_r_limit': agent.indiv_r_limit,
                  'total_r_limit': agent.total_r_limit, 'target_risk': agent.target_risk,
                  'max_pos': agent.max_positions, 'market_bias': session.market_bias
    }

    if agent.mode == 'spot':
        new_record['balance'] = round(session.spot_bal, 2)
        new_record['fr_spot'] = agent.fr_score_spot
        new_record['model_info'] = agent.long_info

        new_record['real_rpnl_spot'] = agent.realised_pnls['real_spot']
        new_record['sim_rpnl_spot'] = agent.realised_pnls['sim_spot']
        new_record['wanted_rpnl_spot'] = agent.realised_pnls['wanted_spot']
        new_record['unwanted_rpnl_spot'] = agent.realised_pnls['unwanted_spot']

    elif agent.mode == 'margin':
        new_record['balance'] = round(session.margin_bal, 2)
        new_record['fr_long'] = agent.fr_score_l
        new_record['fr_short'] = agent.fr_score_s
        new_record['long_model_info'] = agent.long_info
        new_record['short_model_info'] = agent.short_info

        new_record['real_rpnl_long'] = agent.realised_pnls['real_long']
        new_record['sim_rpnl_long'] = agent.realised_pnls['sim_long']
        new_record['wanted_rpnl_long'] = agent.realised_pnls['wanted_long']
        new_record['unwanted_rpnl_long'] = agent.realised_pnls['unwanted_long']

        new_record['real_rpnl_short'] = agent.realised_pnls['real_short']
        new_record['sim_rpnl_short'] = agent.realised_pnls['sim_short']
        new_record['wanted_rpnl_short'] = agent.realised_pnls['wanted_short']
        new_record['unwanted_rpnl_short'] = agent.realised_pnls['unwanted_short']
    else:
        print(f'*** warning log function not working for {agent.name} ***')

    read_folder = Path(f"{session.read_records}/{agent.id}")
    read_path = read_folder / "perf_log.json"

    write_folder = Path(f"{session.write_records}/{agent.id}")
    write_folder.mkdir(parents=True, exist_ok=True)
    write_path = write_folder / "perf_log.json"
    write_path.touch(exist_ok=True)

    try:
        with open(read_path, 'r') as rec_file:
            old_records = json.load(rec_file)
    except (FileNotFoundError, JSONDecodeError):
        print(f'{agent} log file not found')
        old_records = []

    if isinstance(old_records, list):
        old_records.append(new_record)
        all_records = old_records
    else:
        all_records = [new_record]

    try:
        with open(write_path, 'w') as rec_file:
            json.dump(all_records, rec_file)
    except TypeError:
        pprint(new_record)


def interpret_benchmark(session, agents: list) -> None:
    '''takes the benchmark results, ranks them by performance, and prints them 
    in a table'''

    func_name = sys._getframe().f_code.co_name
    x11 = Timer(f'{func_name}')
    x11.start()

    mkt_bench = session.benchmark
    d_ranking = []
    w_ranking = []
    m_ranking = []
    if mkt_bench['valid']:
        d_ranking = [
            ('btc', round(mkt_bench['btc_1d'] * 100, 3)),
            ('eth', round(mkt_bench['eth_1d'] * 100, 3)),
            ('mkt', round(mkt_bench['market_1d'] * 100, 3))
        ]
        w_ranking = [
            ('btc', round(mkt_bench['btc_1w'] * 100, 2)),
            ('eth', round(mkt_bench['eth_1w'] * 100, 2)),
            ('mkt', round(mkt_bench['market_1w'] * 100, 2))
        ]
        m_ranking = [
            ('btc', round(mkt_bench['btc_1m'] * 100, 1)),
            ('eth', round(mkt_bench['eth_1m'] * 100, 1)),
            ('mkt', round(mkt_bench['market_1m'] * 100, 1))
        ]

    for agent in agents:
        agent_bench = agent.benchmark
        if mkt_bench['valid']:
            d_ranking.append((agent.name, round(agent_bench['strat_1d'] * 100, 3)))
            w_ranking.append((agent.name, round(agent_bench['strat_1w'] * 100, 2)))
            m_ranking.append((agent.name, round(agent_bench['strat_1m'] * 100, 1)))

        else:
            print(f'no benchmarking data available for {agent.name}')

    if d_ranking:
        d_ranking = sorted(d_ranking, key=lambda x: x[1], reverse=True)
        print('1 day stats')
        for e, r in enumerate(d_ranking):
            print(f'rank {e + 1}: {r[1]}% {r[0]}')
    if w_ranking:
        w_ranking = sorted(w_ranking, key=lambda x: x[1], reverse=True)
        print('1 week stats')
        for e, r in enumerate(w_ranking):
            print(f'rank {e + 1}: {r[1]}% {r[0]}')
    if m_ranking:
        m_ranking = sorted(m_ranking, key=lambda x: x[1], reverse=True)
        print('1 month stats')
        for e, r in enumerate(m_ranking):
            print(f'rank {e + 1}: {r[1]}% {r[0]}')

    x11.stop()


def count_trades(counts: dict) -> str:
    '''returns a summary of the counts dict as a human-readable string for use
    in the scanner_summary function'''

    er = Timer('count_trades')
    er.start()

    count_list = []

    for t in ['stop', 'open', 'add', 'tp', 'close']:
        if cnt := counts[f"real_{t}_spot"] + counts[f"real_{t}_long"] + counts[f"real_{t}_short"]:
            count_list.append(f'{t}s: {cnt}')

    er.stop()
    return '\n' + ', '.join(count_list) if count_list else ''


def update_liability(trade_record: Dict[str, dict], size: str, operation: str) -> str:
    """this function finds the previous value for liability and returns the new value as a string. the size argument
    should be denominated in the asset being borrowed/repayed"""

    ty = Timer('update_liability')
    ty.start()

    prev_liability = Decimal(trade_record['position']['liability'])
    adjustment = Decimal(size)

    if operation == 'increase':
        new_liability = prev_liability + adjustment
    else:
        new_liability = prev_liability - adjustment

    ty.stop()
    return str(new_liability)


def score_accum(log_path, direction: str) -> Tuple[int, str]:
    """goes through recent perf logs and uses the wanted pnl"""
    func_name = sys._getframe().f_code.co_name
    x12 = Timer(f'{func_name}')
    x12.start()

    read_path = Path(f"{log_path}/perf_log.json")
    try:
        with open(read_path, 'r') as rec_file:
            bal_data = json.load(rec_file)
    except (FileNotFoundError, JSONDecodeError):
        bal_data = {}

    d = -1  # default value
    pnls = {1: d, 2: d, 3: d, 4: d, 5: d}
    if bal_data:
        lookup = f'wanted_pnl_{direction}'
        max_i = min(6, len(bal_data))
        for i in range(1, max_i):
            pnls[i] = json.load(bal_data[-1 * i]).get(lookup, -1)

    score = 0
    if pnls.get(1) > 0:
        score += 5
    elif pnls.get(1) < 0:
        score -= 5
    if pnls.get(2) > 0:
        score += 4
    elif pnls.get(2) < 0:
        score -= 4
    if pnls.get(3) > 0:
        score += 3
    elif pnls.get(3) < 0:
        score -= 3
    if pnls.get(4) > 0:
        score += 2
    elif pnls.get(4) < 0:
        score -= 2
    if pnls.get(5) > 0:
        score += 1
    elif pnls.get(5) < 0:
        score -= 1

    if pnls.get(1) > 0:
        perf_str = '+ |'
    elif pnls.get(1) < 0:
        perf_str = '- |'
    else:
        perf_str = '0 |'

    for j in range(2, 6):
        if pnls.get(j, -1) > 0:
            perf_str += ' +'
        elif pnls.get(j, -1) < 0:
            perf_str += ' -'
        else:
            perf_str += ' 0'

    x12.stop()

    return score, perf_str


def recent_perf_str(session, agent) -> Tuple[str, int, int, int]:
    '''generates a string of + and - to represent recent strat performance
    returns the perf string and the relevant long and short perf scores'''

    func_name = sys._getframe().f_code.co_name
    x13 = Timer(f'{func_name}')
    x13.start()

    log_path = Path(f"{session.read_records}/{agent.id}")
    if agent.mode == 'spot':
        score_spot, perf_str_spot = score_accum(log_path, 'spot')
    else:
        score_l, perf_str_l = score_accum(log_path, 'long')
        score_s, perf_str_s = score_accum(log_path, 'short')

    if agent.open_trades and score_spot:
        perf_str_spot = perf_str_spot
        perf_summ_spot = f"real: score {score_spot} rpnl {agent.realised_pnls['wanted_spot']:.1f}"

    if agent.open_trades and score_l:
        perf_str_l = perf_str_l
        perf_summ_l = f"real: score {score_l} rpnl {agent.realised_pnls['wanted_long']:.1f}"

    if (agent.open_trades and score_s):
        perf_str_s = perf_str_s
        perf_summ_s = f"real: score {score_s} rpnl {agent.realised_pnls['wanted_short']:.1f}"

    full_perf_str = (f'spot: {perf_str_spot}\n{perf_summ_spot}\n'
                     f'long: {perf_str_l}\n{perf_summ_l}\n'
                     f'short: {perf_str_s}\n{perf_summ_s}')

    x13.stop()

    return full_perf_str, score_spot, score_l, score_s


def tot_rpnl(agents: list) -> str:
    total = 0

    for agent in agents:
        total += agent.realised_pnls.get('wanted_spot', 0)
        total += agent.realised_pnls.get('wanted_long', 0)
        total += agent.realised_pnls.get('wanted_short', 0)
        total += agent.realised_pnls.get('wanted_neutral', 0)

    return f"{total:.1f}"


def scanner_summary(session, agents: list) -> None:
    '''prints a summary of the agents recent performance, current exposure, 
    benchmarks, trade counts etc'''

    func_name = sys._getframe().f_code.co_name
    x14 = Timer(f'{func_name}')
    x14.start()

    now = datetime.now(timezone.utc).strftime('%d/%m/%y %H:%M')
    title = f'{now} Spot: ${session.spot_bal:.2f}, Margin: ${session.margin_bal:.2f}'
    live_str = '' if session.live else '*not live* '
    above_ema = len(session.above_200_ema)
    below_ema = len(session.below_200_ema)
    ema_str = f'above ema: {above_ema}/{above_ema + below_ema}'
    mkt_bench = session.benchmark

    final_msg = f"\n{live_str} {ema_str}\nmkt perf 1w {mkt_bench.get('market_1w'):.2%}" \
                f"\nTotal wanted rpnl: {tot_rpnl(agents)}\n-"

    for agent in agents:
        print_msg = False
        agent_msg = f'\n{agent.name}'

        # spot
        if (agent.mode == 'spot') and agent.fixed_risk_spot:
            print_msg = True
            agent_msg += f"\nfixed risk spot: {agent.fixed_risk_spot * 10000:.1f}Bps"

        if (agent.mode == 'spot') and agent.realised_pnls['real_spot']:
            print_msg = True
            agent_msg += f"\nrealised real spot pnl: {agent.realised_pnls['real_spot']:.1f}R"
        elif (agent.mode == 'spot') and (agent.realised_pnls['sim_spot'] > 0):
            print_msg = True
            agent_msg += f"\nrealised wanted spot pnl: {agent.realised_pnls['wanted_spot']:.1f}R"

        # margin
        if (agent.mode == 'margin') and agent.fixed_risk_l:
            print_msg = True
            agent_msg += f"\nfixed risk long: {agent.fixed_risk_l * 10000:.1f}Bps"
        elif (agent.mode == 'margin') and agent.fixed_risk_s:
            print_msg = True
            agent_msg += f"\nfixed risk short: {agent.fixed_risk_s * 10000:.1f}Bps"

        if (agent.mode == 'margin') and agent.realised_pnls['real_long']:
            print_msg = True
            agent_msg += f"\nrealised real long pnl: {agent.realised_pnls['real_long']:.1f}R"
        elif (agent.mode == 'margin') and (agent.realised_pnls['sim_long'] > 0):
            print_msg = True
            agent_msg += f"\nrealised wanted long pnl: {agent.realised_pnls['wanted_long']:.1f}R"

        if (agent.mode == 'margin') and agent.realised_pnls['real_short']:
            print_msg = True
            agent_msg += f"\nrealised real short pnl: {agent.realised_pnls['real_short']:.1f}R"
        elif (agent.mode == 'margin') and (agent.realised_pnls['sim_short'] > 0):
            print_msg = True
            agent_msg += f"\nrealised wanted short pnl: {agent.realised_pnls['wanted_short']:.1f}R"

        or_list = [v.get('or_$') for v in agent.real_pos.values() if v.get('or_$')]
        num_open_positions = len(or_list)
        vol_exp = 0
        for k, v in agent.real_pos.items():
            if k != 'USDT':
                vol_exp += float(v.get('pf%'))
        if num_open_positions or (vol_exp > 1):
            print_msg = True
            agent_msg += f"\npositions {num_open_positions}, exposure {vol_exp:.2f}%"

        agent_msg += count_trades(agent.counts_dict)

        if print_msg:
            final_msg += agent_msg

    if session.live:
        pb.push_note(title, final_msg)
    else:
        print(f'-\n{title}\n{final_msg}')

    x14.stop()

def remove_duplicates(signals: list[dict]) -> list[dict]:
    close_sigs = [sig for sig in signals if sig['action'] == 'close']
    tp_sigs = [sig for sig in signals if sig['action'] == 'tp']
    checked_signals = []

    # check if any two dictionaries share the same salient values
    key1, key2, key3, key4 = 'agent', 'pair', 'direction', 'state'
    seen = {}
    while close_sigs:
        d = close_sigs.pop()
        k = (d[key1], d[key2], d[key3], d[key4])
        if k in seen:
            print(f"duplicate close signal found: {key1} {key2} {key3} {key4}")
            continue
        seen[k] = d
        checked_signals.append(d)

    while tp_sigs:
        d = tp_sigs.pop()
        k = (d[key1], d[key2], d[key3], d[key4])
        if k in seen:
            print(f"duplicate tp signal found: {key1} {key2} {key3} {key4}")
            continue
        seen[k] = d
        checked_signals.append(d)

    return checked_signals


def plot_call_weights(session):
    plot_df = pd.DataFrame({
        'times': [time for time, weight in session.all_weights],
        'weights': [weight for time, weight in session.all_weights]})
    plot_df['seconds'] = plot_df.times - plot_df.times.iloc[0]
    plot_df['cum_weight'] = plot_df.weights.cumsum()
    plot_df = plot_df.drop('weights', axis=1)
    fig = px.scatter(plot_df, x='seconds', y='cum_weight')
    fig.show()


def retry_on_busy(max_retries=360, delay=5):
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            for _ in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                except BinanceAPIException as e:
                    if e.code != -3044:
                        raise e
                    print("System busy, retrying in {} seconds...".format(delay))
                    time.sleep(delay)
                except requests.exceptions.ConnectionError:
                    print("System busy, retrying in {} seconds...".format(delay))
                    time.sleep(delay)
            raise Exception("Max retries exceeded. Request still failed after {} attempts.".format(max_retries))
        return wrapper_retry
    return decorator_retry
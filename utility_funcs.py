import json, keys
from json.decoder import JSONDecodeError
import binance_funcs as funcs
from binance.client import Client
from pathlib import Path
from config import testing
import pandas as pd
from datetime import datetime, timedelta
import statistics as stats
from pushbullet import Pushbullet
import time
from pprint import pprint
from decimal import Decimal, getcontext
from timers import Timer
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import sys

client = Client(keys.bPkey, keys.bSkey, testnet=testing)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12


# now = datetime.now().strftime('%d/%m/%y %H:%M')


def adjust_max_positions(max_pos: int, sizing: dict) -> int:
    '''the max_pos input tells the function what the strategy has as a default
    the sizing input is the dictionary of currently open positions with their 
    associated open risk
    
    this function decides if there should currently be a limit on how many positions
    can be open at once, based on current performance of currently open positions'''

    pass


# def max_init_risk(n: int, target_risk: float) -> float:
#     '''n = number of open positions, target_risk is the percentage distance
#     from invalidation this function should converge on, max_pos is the maximum
#     number of open positions as set in the main script
#
#     this function takes a target max risk and adjusts that up to 2x target depending
#     on how many positions are currently open.
#     whatever the output is, the system will ignore entry signals further away
#     from invalidation than that. if there are a lot of open positions, i want
#     to be more picky about what new trades i open, but if there are few trades
#     available then i don't want to be so picky
#
#     the formula is set so that when there are no trades currently open, the
#     upper limit on initial risk will be twice as high as the main script has
#     set, and as more trades are opened, that upper limit comes down relatively
#     quickly, then gradually settles on the target limit'''
#
#     if n > 20:
#         n = 20
#
#     exp = 4
#     scale = (20 - n) ** exp
#     scale_limit = 20 ** exp
#
#     # when n is 0, scale and scale_limit cancel out
#     # and the whole thing becomes (2 * target) + target
#     output = (2 * target_risk * scale / scale_limit) + target_risk
#     # print(f'mir output: {round(output, 2) * 100}%')
#
#     return round(output, 2)


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

    data = session.ohlc_data.glob('*.*')

    for x in data:
        df = pd.read_pickle(x)
        if len(df) > 2977: # 1 month of 15min periods is 31 * 24 * 4 = 2976
            df = df.tail(2977).reset_index()
        last_stamp = df.at[df.index[-1], 'timestamp']
        now = datetime.now()
        window = timedelta(hours=4)
        if last_stamp > now - window:  # if there is data up to the last 4 hours
            if len(df) >= 97:
                df['roc_1d'] = df.close.pct_change(96)
                all_1d.append(df.at[df.index[-1], 'roc_1d'])
            if len(df) >= 673:
                df['roc_1w'] = df.close.pct_change(672)
                all_1w.append(df.at[df.index[-1], 'roc_1w'])
            if len(df) >= 2977:
                df['roc_1m'] = df.close.pct_change(2976)
                all_1m.append(df.at[df.index[-1], 'roc_1m'])
            if x.stem == 'BTCUSDT':
                btc_1d = df.at[df.index[-1], 'roc_1d']
                btc_1w = df.at[df.index[-1], 'roc_1w']
                btc_1m = df.at[df.index[-1], 'roc_1m']
            elif x.stem == 'ETHUSDT':
                eth_1d = df.at[df.index[-1], 'roc_1d']
                eth_1w = df.at[df.index[-1], 'roc_1w']
                eth_1m = df.at[df.index[-1], 'roc_1m']
    market_1d = stats.median(all_1d) if len(all_1d) > 3 else 0
    market_1w = stats.median(all_1w) if len(all_1w) > 3 else 0
    market_1m = stats.median(all_1m) if len(all_1m) > 3 else 0
    # print(f'1d median based on {len(all_1d)} data points')
    # print(f'1w median based on {len(all_1w)} data points')
    # print(f'1m median based on {len(all_1m)} data points')

    all_pairs = len(list(data))
    valid_pairs = len(all_1d) > 3
    if valid_pairs:
        valid = True
        if all_pairs / valid_pairs > 1.5:
            print('warning (market benchmark): lots of pairs ohlc data not up to date')
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

    now = datetime.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    bal_now, bal_1d, bal_1w, bal_1m = session.bal, None, None, None

    filepath = Path(f"{session.read_records}/{agent.id}/perf_log.json")
    try:
        with open(filepath, 'r') as rec_file:
            bal_data = json.load(rec_file)
    except (FileNotFoundError, JSONDecodeError):
        bal_data = {}

    benchmark = {}
    if bal_data:
        for row in bal_data[-1:0:-1]:
            row = json.loads(row)
            row_dt = datetime.strptime(row.get('timestamp'), '%d/%m/%y %H:%M')
            if row_dt > month_ago:  # and not bal_1m:
                try:
                    bal_1m = row.get('balance')
                except AttributeError:
                    continue
            if row_dt > week_ago:  # and not bal_1w:
                try:
                    bal_1w = row.get('balance')
                except AttributeError:
                    continue
            if row_dt > day_ago:  # and not bal_1d:
                try:
                    bal_1d = row.get('balance')
                except AttributeError:
                    continue

        if bal_1d:
            benchmark['strat_1d'] = (bal_now - bal_1d) / bal_1d
        else:
            benchmark['strat_1d'] = 0

        if bal_1w:
            benchmark['strat_1w'] = (bal_now - bal_1w) / bal_1w
        else:
            benchmark['strat_1w'] = 0

        if bal_1m:
            benchmark['strat_1m'] = (bal_now - bal_1m) / bal_1m
        else:
            benchmark['strat_1m'] = 0

    else:
        bal_now = session.bal
        benchmark['strat_1d'] = 0
        benchmark['strat_1w'] = 0
        benchmark['strat_1m'] = 0

    agent.benchmark = benchmark


def log(session, agents: list) -> None:
    '''records all data from the session as a line in the perf_log.json file'''

    market_benchmark(session)
    for agent in agents:
        params = {}

        new_record = {'timestamp': session.now_start, 'balance': round(session.bal, 2),
                      'fr_long': agent.fixed_risk_l, 'fr_short': agent.fixed_risk_s,
                      'positions': agent.real_pos, 'trade_counts': agent.counts_dict,
                      'realised_pnl_long': agent.realised_pnl_long, 'sim_r_pnl_long': agent.sim_pnl_long,
                      'realised_pnl_short': agent.realised_pnl_short, 'sim_r_pnl_short': agent.sim_pnl_short,
                      'median_spread': stats.median(session.spreads.values()),
                      'real_open_pnl_l': agent.open_pnl('long', 'real'),
                      'real_open_pnl_s': agent.open_pnl('short', 'real'),
                      'sim_open_pnl_l': agent.open_pnl('long', 'sim'),
                      'sim_open_pnl_s': agent.open_pnl('short', 'sim'),
                      'quote_asset': session.quote_asset, 'fr_max': session.fr_max,
                      'max_spread': session.max_spread, 'indiv_r_limit': agent.indiv_r_limit,
                      'total_r_limit': agent.total_r_limit, 'target_risk': agent.target_risk,
                      'max_pos': agent.max_positions}

        read_folder = Path(f"{session.read_records}/{agent.id}")
        read_folder.mkdir(parents=True, exist_ok=True)
        read_path = read_folder / "perf_log.json"
        read_path.touch(exist_ok=True)

        write_folder = Path(f"{session.write_records}/{agent.id}")
        write_folder.mkdir(parents=True, exist_ok=True)
        write_path = write_folder / "perf_log.json"
        write_path.touch(exist_ok=True)

        try:
            with open(read_path, 'r') as rec_file:
                old_records = json.load(rec_file)
        except (FileNotFoundError, JSONDecodeError):
            print(f'{agent} log file not found')
            old_records = None

        if old_records:
            all_records = old_records.append(new_record)
        else:
            all_records = [new_record]

        #temporary debugging
        print(f"{agent.name} {len(all_records) = }")

        with open(write_path, 'w') as rec_file:
            json.dump(all_records, rec_file)

        strat_benchmark(session, agent)


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
    stops = counts["real_stop_long"] + counts["real_stop_short"]
    if stops:
        count_list.append(f'stopped: {stops}')
    opens = counts["real_open_long"] + counts["real_open_short"]
    if opens:
        count_list.append(f'opened: {opens}')
    adds = counts["real_add_long"] + counts["real_add_short"]
    if adds:
        count_list.append(f'added: {adds}')
    tps = counts["real_tp_long"] + counts["real_tp_short"]
    if tps:
        count_list.append(f'tped: {tps}')
    closes = counts["real_close_long"] + counts["real_close_short"]
    if closes:
        count_list.append(f'closed: {closes}')
    if count_list:
        counts_str = '\n' + ', '.join(count_list)
    else:
        counts_str = ''
    er.stop()
    return counts_str


def find_bad_keys(c_data: dict) -> list:
    '''returns the ids of any trade records whos trade sizes don't add up'''

    bad_keys = []
    for k, v in c_data.items():
        try:
            init_base = 0
            add_base = 0
            tp_base = 0
            close_base = 0
            for x in v:
                if x.get('type')[:4] == 'open':
                    init_base = float(x.get('base_size'))
                elif x.get('type')[:3] == 'add':
                    add_base += float(x.get('base_size'))
                elif x.get('type')[:2] == 'tp':
                    tp_base += float(x.get('base_size'))
                elif x.get('type')[:5] in ['close', 'stop_']:
                    close_base = float(x.get('base_size'))

            gross_buy = init_base + add_base
            gross_sell = tp_base + close_base
            diff = (gross_buy - gross_sell) / gross_buy
            pair = x.get('pair')
            if abs(diff) > 0.03:
                # print(f'{k} {pair} - bought: {gross_buy} sold: {gross_sell}')
                bad_keys.append({'key': k, 'pair': pair, 'buys': gross_buy, 'sells': gross_sell})
        except:
            bad_keys.append({'key': k, 'pair': pair, 'buys': gross_buy, 'sells': gross_sell})
            # print('bad key:', k)
            continue

    return bad_keys


# def update_liability(trade_record: Dict[str, dict], size: str, operation: str) -> str:
#     """this function finds the previous value for liability and returns the new value as a string"""
#
#     ty = Timer('update_liability')
#     ty.start()
#
#     prev_liability = Decimal(trade_record['position']['liability'])
#     adjustment = Decimal(size)
#
#     if operation == 'increase':
#         new_liability = prev_liability + adjustment
#     else:
#         new_liability = prev_liability - adjustment
#
#     ty.stop()
#     return str(new_liability)


def create_stop_dict(order: dict) -> dict:
    '''collects and returns the details of filled stop-loss order in a dictionary'''

    yu = Timer('create_stop-dict')
    yu.start()

    pair = order.get('symbol')
    quote_qty = order.get('cummulativeQuoteQty')
    base_qty = order.get('executedQty')
    avg_price = round(float(quote_qty) / float(base_qty), 8)

    bnb_fee = funcs.calc_fee_bnb(quote_qty)

    trade_dict = {'timestamp': order.get('updateTime'),
                  'pair': pair,
                  'trig_price': order.get('stopPrice'),
                  'limit_price': order.get('price'),
                  'exe_price': str(avg_price),
                  'base_size': base_qty,
                  'quote_size': quote_qty,
                  'fee': str(bnb_fee),
                  'fee_currency': 'BNB'
                  }
    if order.get('status') != 'FILLED':
        print(f'{pair} order not filled')
        pb.push_note('Warning', f'{pair} stop-loss hit but not filled')
    yu.stop()
    return trade_dict


def recent_perf_str(session, agent) -> Tuple[str, int, int]:
    '''generates a string of + and - to represent recent strat performance
    returns the perf string and the relevant long and short perf scores'''

    def score_accum(state: str, direction: str) -> tuple[int, str]:
        func_name = sys._getframe().f_code.co_name
        x12 = Timer(f'{func_name}')
        x12.start()

        read_path = Path(f"{session.read_records}/{agent.id}/perf_log.json")
        try:
            with open(read_path, 'r') as rec_file:
                bal_data = json.load(rec_file)
        except (FileNotFoundError, JSONDecodeError):
            bal_data = {}

        # if bal_data:
        #     prev_bal = json.loads(bal_data[-1]).get('balance')
        # else:
        #     prev_bal = session.bal
        # bal_change_pct = 100 * (session.bal - prev_bal) / prev_bal

        # other_last = json.loads(bal_data[-2])
        # if agent.open_pnl_changes.get(state):
        #     bal_change_pct = agent.open_pnl_changes.get(state)
        #     print(f"{state} open pnl change: {bal_change_pct:.2f}")
        # elif bal_data and len(bal_data) > 1:
        #     # last = json.loads(bal_data[-2])
        #     # prev_bal = last.get('balance')
        #     # bal_change_pct = 100 * (agent.bal - prev_bal) / prev_bal
        #     bal_change_pct = agent.open_pnl_changes.get(state, 0)
        #     print(f"bal change: {bal_change_pct:.2f}")
        # else:
        #     bal_change_pct = 0
        #     print(f"real open pnl change: {bal_change_pct}")

        d = -1  # default value
        pnls = {1: d, 2: d, 3: d, 4: d, 5: d}
        if bal_data:
            lookup = f'realised_pnl_{direction}' if state == 'real' else f'sim_r_pnl_{direction}'
            max_i = min(6, len(bal_data))
            for i in range(1, max_i):
                pnls[i] = json.loads(bal_data[-1 * i]).get(lookup)

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

    func_name = sys._getframe().f_code.co_name
    x13 = Timer(f'{func_name}')
    x13.start()

    real_score_l, real_perf_str_l = score_accum('real', 'long')
    real_score_s, real_perf_str_s = score_accum('real', 'short')
    sim_score_l, sim_perf_str_l = score_accum('sim', 'long')
    sim_score_s, sim_perf_str_s = score_accum('sim', 'short')

    if (agent.open_trades and real_score_l):
        perf_str_l = real_perf_str_l
        perf_summ_l = f"real: score {real_score_l} rpnl {agent.realised_pnl_long:.1f}"
        score_l = real_score_l
    else:
        perf_str_l = sim_perf_str_l
        perf_summ_l = f"sim: score {sim_score_l} rpnl {agent.sim_pnl_long:.1f}"
        score_l = sim_score_l

    if (agent.open_trades and real_score_s):
        perf_str_s = real_perf_str_s
        perf_summ_s = f"real: score {real_score_s} rpnl {agent.realised_pnl_short:.1f}"
        score_s = real_score_s
    else:
        perf_str_s = sim_perf_str_s
        perf_summ_s = f"sim: score {sim_score_s} rpnl {agent.sim_pnl_short:.1f}"
        score_s = sim_score_s

    full_perf_str = f'long: {perf_str_l}\n{perf_summ_l}\nshort: {perf_str_s}\n{perf_summ_s}'

    x13.stop()

    return full_perf_str, score_l, score_s


def scanner_summary(session, agents: list) -> None:
    '''prints a summary of the agents recent performance, current exposure, 
    benchmarks, trade counts etc'''

    func_name = sys._getframe().f_code.co_name
    x14 = Timer(f'{func_name}')
    x14.start()

    now = datetime.now().strftime('%d/%m/%y %H:%M')
    title = f'{now} ${session.bal:.2f}'
    live_str = '' if session.live else '*not live* '
    above_ema = len(session.above_200_ema)
    below_ema = len(session.below_200_ema)
    ema_str = f'above ema: {above_ema}/{above_ema + below_ema}'
    mkt_bench = session.benchmark
    final_msg = f"{live_str} {ema_str}\nmkt perf 1w {mkt_bench.get('market_1w'):.2%}\n-"

    for agent in agents:
        agent_msg = f'\n{agent.name}'

        if agent.realised_pnl_long:
            agent_msg += f"\nrealised real long pnl: {agent.realised_pnl_long:.1f}R"
        if agent.realised_pnl_short:
            agent_msg += f"\nrealised real short pnl: {agent.realised_pnl_short:.1f}R"

        or_list = [v.get('or_$') for v in agent.real_pos.values() if v.get('or_$')]
        num_open_positions = len(or_list)
        vol_exp = 0
        for k, v in agent.real_pos.items():
            if k != 'USDT':
                vol_exp += float(v.get('pf%'))
        if num_open_positions or (vol_exp > 1):
            agent_msg += f"\npositions {num_open_positions}, exposure {vol_exp:.2f}%"

        agent_msg += count_trades(agent.counts_dict)

        final_msg += agent_msg

    if session.live:
        pb.push_note(title, final_msg)
    else:
        print(f'-\n{title}\n{final_msg}')

    x14.stop()






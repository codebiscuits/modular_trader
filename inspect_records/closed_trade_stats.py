import pandas as pd
# from pushbullet import Pushbullet
from pathlib import Path
import trade_records_funcs as trf
from itertools import product
from pprint import pprint
from collections import Counter
from resources.loggers import create_logger

# pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

logger = create_logger('closed_trade_stats')

records_folder = Path('/home/ross/coding/pi_2/modular_trader/records')

# TODO i want to see a counts dict representing how many times each sim reason has come up in *wanted* sim trades over a
#  rolling window. maybe when this is a dashboard, i could have a pie chart showing the percentages of each sim reason

def duration(trade):
    d = trade[-1]['timestamp'] - trade[0]['timestamp']
    return f"{int(d // 3600)}h {int(d // 60 % 60)}m {int(d % 60)}s"


data = trf.load_all(records_folder, ['closed', 'closed_sim'])
# pprint(data)

all_trades = []
sim_reasons = {'1h': [], '4h': [], '12h': [], '1d': []}

for v in data.values():
    for k, d in v.items():
        k = float(k) / 1000 if float(k) > 20_000_000_000 else float(k)
        k = int(k) * 1_000_000_000
        inval_dist = abs(d['signal']['inval_ratio'] - 1)

        agent = d['signal']['agent']
        if len(agent.split('_')) == 6:
            agent += '_slow_volumes_30'
        elif len(agent.split('_')) == 7:
            agent += '_volumes_30'
        elif len(agent.split('_')) == 8:
            agent += '_30'
        feature_selection = agent.split('_')[6]
        pair_selection = agent.split('_')[7]

        age = d['signal'].get('model_age', 3_600_000) / 3_600

        stats = dict(
            timestamp=k,
            agent=agent,
            pair=d['signal']['pair'],
            state=d['trade'][0]['state'],
            direction=d['signal']['direction'],
            timeframe=d['signal']['tf'],
            feat_select=feature_selection,
            pair_select=pair_selection,
            wanted=d['trade'][0].get('wanted', True),
            confidence=round(d['signal']['confidence'], 5),
            confidence_l=round(d['signal'].get('confidence_l', 0), 5),
            confidence_s=round(d['signal'].get('confidence_s', 0), 5),
            score=round(d['signal']['score'], 5),
            inval_distance=round(inval_dist, 5),
            perf_ema4=d['signal'].get('perf_ema4'),
            perf_ema8=d['signal'].get('perf_ema8'),
            perf_ema16=d['signal'].get('perf_ema16'),
            perf_ema32=d['signal'].get('perf_ema32'),
            perf_ema64=d['signal'].get('perf_ema64'),
            rank_1d=d['signal'].get('market_rank_1d'),
            rank_1w=d['signal'].get('market_rank_1w'),
            rank_1m=d['signal'].get('market_rank_1m'),
            model_age=round(age),
            duration=duration(d['trade']), # look-ahead bias
            closed_by=d['trade'][-1]['action'], # look-ahead bias
            rpnl=sum([t.get('rpnl', 0) for t in d['trade']]), # target
        )
        if stats.get('rank_1d'): # only include records with full set of stats in results and sim reasons counts
            all_trades.append(stats)
            sim_reasons[d['signal']['tf']].extend(d['signal'].get('sim_reasons', []))

# pprint(all_trades)
df = pd.DataFrame(all_trades).sort_values('timestamp').reset_index(drop=True)
df['timestamp'] = pd.to_datetime(df.timestamp)

timeframes = [
    '1h',
    '4h', '12h', '1d'
]
directions = [
    'long',
    'short'
]
all_dfs = {}
for tf, dir in product(timeframes, directions):
    print(f"\ndf_{tf}_{dir}")
    all_dfs[f"df_{tf}_{dir}"] = (df.loc[(df.timeframe == tf) & (df.direction == dir)].reset_index(drop=True))

    conf_corr = all_dfs[f"df_{tf}_{dir}"].confidence.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    conf_l_corr = all_dfs[f"df_{tf}_{dir}"].confidence_l.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    conf_s_corr = all_dfs[f"df_{tf}_{dir}"].confidence_s.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    wanted_corr = all_dfs[f"df_{tf}_{dir}"].wanted.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    inval_corr = all_dfs[f"df_{tf}_{dir}"].inval_distance.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    signal_corr = all_dfs[f"df_{tf}_{dir}"].score.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    perf_ema4_corr = all_dfs[f"df_{tf}_{dir}"].perf_ema4.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    perf_ema8_corr = all_dfs[f"df_{tf}_{dir}"].perf_ema8.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    perf_ema16_corr = all_dfs[f"df_{tf}_{dir}"].perf_ema16.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    perf_ema32_corr = all_dfs[f"df_{tf}_{dir}"].perf_ema32.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    perf_ema64_corr = all_dfs[f"df_{tf}_{dir}"].perf_ema64.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    print(f"confidence correlation: {conf_corr:.1%}, \nlong confidence correlation: {conf_l_corr:.1%}, "
          f"\nshort confidence correlation: {conf_s_corr:.1%}, \nwanted correlation: {wanted_corr:.1%}, "
          f"\ninval correlation: {inval_corr:.1%}, \nsignal score correlation: {signal_corr:.1%}"
          f"\nperf_ema4 correlation: {perf_ema4_corr:.1%}, \nperf_ema8 correlation: {perf_ema8_corr:.1%}, "
          f"\nperf_ema16 correlation: {perf_ema16_corr:.1%}, \nperf_ema32 correlation: {perf_ema32_corr:.1%}, "
          f"\nperf_ema64 correlation: {perf_ema64_corr:.1%}")

    print(all_dfs[f"df_{tf}_{dir}"].tail())

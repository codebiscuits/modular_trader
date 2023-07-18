import pandas as pd
from pushbullet import Pushbullet
from pathlib import Path
import trade_records_funcs as trf
from itertools import product
from pprint import pprint

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

records_folder_1 = Path('/home/ross/coding/pi_down/modular_trader/records')
records_folder_2 = Path('/home/ross/coding/pi_2/modular_trader/records')

def duration(trade):
    d = trade[-1]['timestamp'] - trade[0]['timestamp']

    return f"{int(d // 3600)}h {int(d / 3600 % 60)}m {int(d % 60)}s"


data = trf.load_all(records_folder_2, ['closed', 'closed_sim'])
# pprint(data)

all_trades = []

for v in data.values():
    for k, d in v.items():
        k = float(k) / 1000 if float(k) > 20_000_000_000 else float(k)
        k = int(k) * 1_000_000_000
        inval_dist = abs(d['signal']['inval_ratio'] - 1)
        stats = dict(
            timestamp=k,
            agent=d['signal']['agent'],
            pair=d['signal']['pair'],
            state=d['trade'][0]['state'],
            direction=d['signal']['direction'],
            timeframe=d['signal']['tf'],
            wanted=d['trade'][0].get('wanted', True),
            confidence=round(d['signal']['confidence'], 5),
            inval_distance=round(inval_dist, 5),
            # perf_ema4=d['signal'].get('perf_ema4'),
            # perf_ema8=d['signal'].get('perf_ema8'),
            # perf_ema16=d['signal'].get('perf_ema16'),
            # perf_ema32=d['signal'].get('perf_ema32'),
            # perf_ema64=d['signal'].get('perf_ema64'),
            # duration=duration(d['trade']), # look-ahead bias
            # closed_by=d['trade'][-1]['action'], # look-ahead bias
            rpnl=sum([t.get('rpnl', 0) for t in d['trade']]), # target
        )
        all_trades.append(stats)

df = pd.DataFrame(all_trades).sort_values('timestamp').reset_index(drop=True)
df['timestamp'] = pd.to_datetime(df.timestamp)

all_dfs = {}
for tf, dir in product(['1h', '4h', '12h', '1d'], ['long', 'short']):
    print(f"\ndf_{tf}_{dir}")
    all_dfs[f"df_{tf}_{dir}"] = (df.loc[(df.timeframe == tf) & (df.direction == dir)].reset_index(drop=True))
    # all_dfs[f"df_{tf}_{dir}"]['cum_rpnl'] = all_dfs[f"df_{tf}_{dir}"].rpnl.cumsum().fillna('ffill')

    # WARNING these columns must not bbe in the ml model because they have look-ahead bias
    all_dfs[f"df_{tf}_{dir}"]['rpnl_ema4'] = all_dfs[f"df_{tf}_{dir}"].loc[all_dfs[f"df_{tf}_{dir}"].wanted].rpnl.ewm(4).mean().interpolate('ffill')
    all_dfs[f"df_{tf}_{dir}"]['rpnl_ema8'] = all_dfs[f"df_{tf}_{dir}"].loc[all_dfs[f"df_{tf}_{dir}"].wanted].rpnl.ewm(8).mean().interpolate('ffill')
    all_dfs[f"df_{tf}_{dir}"]['rpnl_ema16'] = all_dfs[f"df_{tf}_{dir}"].loc[all_dfs[f"df_{tf}_{dir}"].wanted].rpnl.ewm(16).mean().interpolate('ffill')
    all_dfs[f"df_{tf}_{dir}"]['rpnl_ema32'] = all_dfs[f"df_{tf}_{dir}"].loc[all_dfs[f"df_{tf}_{dir}"].wanted].rpnl.ewm(32).mean().interpolate('ffill')
    all_dfs[f"df_{tf}_{dir}"]['rpnl_ema64'] = all_dfs[f"df_{tf}_{dir}"].loc[all_dfs[f"df_{tf}_{dir}"].wanted].rpnl.ewm(64).mean().interpolate('ffill')

    conf_corr = all_dfs[f"df_{tf}_{dir}"].confidence.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    inval_corr = all_dfs[f"df_{tf}_{dir}"].inval_distance.corr(all_dfs[f"df_{tf}_{dir}"].rpnl)
    print(f"confidence correlation: {conf_corr:.1%}, inval correlation: {inval_corr:.1%}")

    print(all_dfs[f"df_{tf}_{dir}"].tail())
